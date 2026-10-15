"""Treino prospectivo SFT/GRPO com checkpoints e isolamento de processos.

python3 -m gpu_lab.train --config configs/sft_smoke.json --data data/arithmetic.jsonl
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import platform
import re
import traceback
from pathlib import Path

from .benchmark import SYSTEM,content_hash,read_dataset,render_prompt,select
from .preflight import text_only_compat
text_only_compat()


def atomic_json(path,value):
    path=Path(path)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    tmp.replace(path)


def validate_config(c):
    required={"mode","model_id","revision","train_seed","data_seed","output_root"}
    if missing:=required-set(c):
        raise ValueError(f"Configuração incompleta: {sorted(missing)}")
    if c["mode"] not in ("sft","grpo"):
        raise ValueError("mode deve ser sft ou grpo")
    if c["mode"]=="sft" and c.get("condition") not in ("explicit","direct","generic"):
        raise ValueError("Alvo SFT inválido")
    if c["mode"]=="grpo":
        if c.get("monitor_backend", "remote") not in ("local", "remote"):
            raise ValueError("monitor_backend deve ser local ou remote")
        if c.get("condition") not in ("none","cot_monitor","summary_monitor"):
            raise ValueError("Condição GRPO inválida")
        groups=c.get("num_generations",4)
        batch=c.get("batch_size",1)*c.get("grad_accum",4)
        if groups<2 or batch%groups:
            raise ValueError("Lote efetivo deve ser múltiplo de num_generations >= 2.")
        if c["condition"]!="none" and not all(c.get(k) for k in ("monitor_model","monitor_revision")):
            raise ValueError("Fixe modelo e revisão do monitor nas condições penalizadas.")
    if c.get("max_steps",1)<1:
        raise ValueError("max_steps deve ser positivo")
    if int(os.environ.get("WORLD_SIZE","1"))!=1:
        raise ValueError("Use um processo/GPU por execução; paralelize células independentes.")
    if c.get("confirmatory",False) and not c.get("preregistration_url"):
        raise ValueError("Execução confirmatória exige identificador de pré-registro.")


def load_actor(c):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,set_seed
    from .revisions import resolve_revision
    cpu = c.get("device", "cuda") == "cpu"
    if c.get("device", "cuda") not in ("cpu", "cuda"):
        raise ValueError("device deve ser cpu ou cuda")
    if not cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA não disponível. Execute a reanálise CPU ou use GPU.")
    if cpu and c.get("load_in_4bit", True):
        raise ValueError("A execução CPU deve usar load_in_4bit=False.")
    set_seed(c["train_seed"])
    resolved=resolve_revision(c["model_id"],c["revision"])
    revision = None if resolved.startswith("local-") else resolved
    tokenizer=AutoTokenizer.from_pretrained(c["model_id"],revision=revision,trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side="left"
    tokenizer.model_input_names=["input_ids","attention_mask"]
    precision=c.get("precision","auto")
    if precision not in ("auto","fp16","bf16"):
        raise ValueError("precision deve ser auto, fp16 ou bf16")
    if not cpu and precision=="bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("A GPU não suporta a precisão bf16 fixada no plano.")
    dtype=(torch.float32 if cpu else torch.float16 if precision=="fp16" else
           torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    kwargs=dict(revision=revision,torch_dtype=dtype,trust_remote_code=False,
                device_map={"":"cpu" if cpu else 0},attn_implementation="sdpa")
    if c.get("load_in_4bit",True):
        kwargs["quantization_config"]=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype)
    model=AutoModelForCausalLM.from_pretrained(c["model_id"],**kwargs)
    if c.get("load_in_4bit",True):
        from peft import prepare_model_for_kbit_training
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True)
    model.config.use_cache=False
    return model,tokenizer,resolved,dtype


def chat(tokenizer,messages):
    return tokenizer.apply_chat_template(messages,tokenize=False,
        add_generation_prompt=True,enable_thinking=False)


def sft_examples(rows,tokenizer,condition,level):
    examples=[]
    for row in rows:
        if row["split"]!="train" or row["level"]!=level:
            continue
        nums=[int(n) for n in re.findall(r"\d+",row["prompt"])]
        if level=="A":
            trace=f"{nums[0]} + {nums[1]} = {row['gold']}."
        elif level=="B":
            partial=nums[0]+nums[1]
            trace=f"{nums[0]} + {nums[1]} = {partial}; {partial} - {nums[2]} = {row['gold']}."
        else:
            product=nums[1]*nums[2]
            total=nums[0]+product
            trace=f"{nums[1]} * {nums[2]} = {product}; {nums[0]} + {product} = {total}; {total} - {nums[3]} = {row['gold']}."
        if condition=="direct":
            trace=""
        elif condition=="generic":
            trace="Apos analisar o caso de forma geral, a conclusao segue diretamente."
        prompt=chat(tokenizer,[{"role":"system","content":
             "Resolva o problema. Termine com uma única tag <answer>NUMERO</answer>."},
             {"role":"user","content":row["prompt"]}])
        completion=trace+f" <answer>{row['gold']}</answer>"+(tokenizer.eos_token or "")
        examples.append(dict(prompt=prompt,completion=completion))
    return examples


def run(config,data_path):
    validate_config(config)
    import torch
    import importlib.metadata as metadata
    from datasets import Dataset
    from peft import LoraConfig
    from transformers.trainer_utils import get_last_checkpoint
    rows=read_dataset(data_path)
    data_sha=hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    if {r["dataset_seed"] for r in rows}!={config["data_seed"]}:
        raise ValueError("data_seed não corresponde ao conjunto fornecido.")
    model,tokenizer,resolved,dtype=load_actor(config)
    resolved_config={**config,"resolved_revision":resolved,"data_sha256":data_sha}
    run_id=content_hash(resolved_config)[:20]
    run_dir=Path(config["output_root"])/run_id
    run_dir.mkdir(parents=True,exist_ok=True)
    previous=run_dir/"manifest.json"
    if previous.exists() and json.loads(previous.read_text())["resolved_config"]!=resolved_config:
        raise ValueError("Configuração diverge do manifesto da execução.")
    manifest=dict(run_id=run_id,resolved_config=resolved_config,
        environment=dict(python=platform.python_version(),torch=torch.__version__,
          gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() and config.get("device")!="cpu" else "CPU",
          compute_dtype=str(dtype),
          packages={p:metadata.version(p) for p in ["transformers","trl","peft","accelerate","datasets"]}))
    atomic_json(previous,manifest)
    status_path=run_dir/"status.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("status")=="trained":
        print(f"Treino já concluído: {run_dir}")
        return str(run_dir)
    lora=LoraConfig(r=config.get("lora_r",16),lora_alpha=config.get("lora_alpha",32),
                    lora_dropout=0.,bias="none",task_type="CAUSAL_LM",
                    target_modules="all-linear")
    common=dict(output_dir=str(run_dir/"checkpoints"),
        max_steps=config["max_steps"],per_device_train_batch_size=config.get("batch_size",1),
        gradient_accumulation_steps=config.get("grad_accum",4),
        learning_rate=config.get("learning_rate",5e-5 if config["mode"]=="sft" else 1e-6),
        seed=config["train_seed"],data_seed=config["data_seed"],bf16=dtype==torch.bfloat16,
        fp16=dtype==torch.float16,gradient_checkpointing=True,
        logging_steps=config.get("logging_steps",1),save_steps=config.get("save_steps",20),
        save_strategy="steps",save_total_limit=2,report_to="none",
        optim="adamw_torch",max_grad_norm=1.,warmup_ratio=.05)
    monitor=None
    atomic_json(status_path,dict(status="running",run_id=run_id))
    try:
        if config["mode"]=="sft":
            from trl import SFTConfig,SFTTrainer
            examples=sft_examples(rows,tokenizer,config["condition"],config["level"])
            if not examples:
                raise ValueError("Não há exemplos de treino para a configuração.")
            args=SFTConfig(**common,max_length=config.get("max_length",1024),
                           completion_only_loss=True,packing=False)
            trainer=SFTTrainer(model=model,args=args,train_dataset=Dataset.from_list(examples),
                               processing_class=tokenizer,peft_config=lora)
        else:
            from trl import GRPOConfig,GRPOTrainer
            from .monitor import RemoteMonitor
            from .rewards import reward_function
            if config["condition"]!="none":
                if config.get("monitor_backend", "remote") == "local":
                    from .local_monitor import LocalMonitor
                    monitor=LocalMonitor(config["monitor_model"],config["monitor_revision"],
                                         run_dir/"monitor_cache.sqlite")
                else:
                    monitor=RemoteMonitor(config["monitor_model"],config["monitor_revision"],
                                           run_dir/"monitor_cache.sqlite")
            selected=select(rows,"train",config["level"],config.get("heldout_domain","inventory"),True)
            examples=[dict(prompt=chat(tokenizer,[{"role":"system","content":SYSTEM},
                {"role":"user","content":render_prompt(r)}]),
                gold=r["gold"],cheat=r["cheat"],question=r["question"],
                item_id=r["item_id"]) for r in selected]
            if not examples:
                raise ValueError("Conjunto GRPO de treino vazio")
            args=GRPOConfig(**common,num_generations=config.get("num_generations",4),
                max_prompt_length=config.get("max_prompt_length",768),
                max_completion_length=config.get("max_completion_length",384),
                beta=config.get("beta",.04),use_vllm=False,
                remove_unused_columns=False,mask_truncated_completions=True)
            trainer=GRPOTrainer(model=model,args=args,processing_class=tokenizer,
                peft_config=lora,train_dataset=Dataset.from_list(examples),
                reward_funcs=reward_function(config["condition"],monitor,
                    config.get("monitor_weight",.6),config.get("min_trace_words",0)))
        latest=get_last_checkpoint(common["output_dir"]) if Path(common["output_dir"]).exists() else None
        trainer.train(resume_from_checkpoint=latest)
        trainer.save_model(str(run_dir/"adapter"))
        tokenizer.save_pretrained(run_dir/"adapter")
        trainer.state.save_to_json(str(run_dir/"trainer_state.json"))
        atomic_json(status_path,dict(status="trained",run_id=run_id,
                                     global_step=trainer.state.global_step))
    except Exception as exc:
        atomic_json(status_path,dict(status="failed",run_id=run_id,error_type=type(exc).__name__))
        (run_dir/"failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        if monitor is not None:
            monitor.close()
        gc.collect()
        torch.cuda.empty_cache()
    return str(run_dir)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True)
    p.add_argument("--data",required=True)
    a=p.parse_args()
    print(run(json.loads(Path(a.config).read_text()),a.data))
