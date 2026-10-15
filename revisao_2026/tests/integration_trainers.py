"""Teste real das APIs TRL em CPU, com pesos aleatórios mínimos; não é experimento.

PYTHONPATH=src COT_TEXT_ONLY=1 python tests/integration_trainers.py
Requer requirements-gpu.txt e PyTorch (CPU basta). Não baixa modelos.
"""
import json
from pathlib import Path
import tempfile
import shutil
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Qwen2Config,Qwen2ForCausalLM,PreTrainedTokenizerFast
from instruments import make_disjoint_data
from gpu_lab.benchmark import create_dataset,write_dataset
from gpu_lab.train import run
from gpu_lab.evaluate import run as evaluate
from gpu_lab.local_monitor import LocalMonitor
from gpu_lab.revisions import resolve_revision


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261015)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        base=root/"tiny-model"
        words=["<pad>","<bos>","<eos>","<unk>","user","assistant","system",
               "A","B","C","D","trace","summary","answer","false","true", "flag"]
        words+=list("0123456789")+["<",">","/","+","-","*","=",".",":","{","}"]
        tokenizer=Tokenizer(WordLevel({w:i for i,w in enumerate(words)},unk_token="<unk>"))
        tokenizer.pre_tokenizer=Whitespace()
        fast=PreTrainedTokenizerFast(tokenizer_object=tokenizer,bos_token="<bos>",eos_token="<eos>",
                                    unk_token="<unk>",pad_token="<pad>")
        fast.chat_template="{% for m in messages %}{{m['role'] + ': ' + m['content'] + '\\n'}}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
        model=Qwen2ForCausalLM(Qwen2Config(vocab_size=len(words),hidden_size=32,intermediate_size=64,
            num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=2048,
            bos_token_id=1,eos_token_id=2,pad_token_id=0))
        model.save_pretrained(base)
        fast.save_pretrained(base)
        config=dict(mode="sft",model_id=str(base),revision="main",train_seed=42,data_seed=20261015,
            output_root=str(root/"sft"),max_steps=1,save_steps=1,batch_size=1,grad_accum=2,
            max_length=256,lora_r=2,lora_alpha=4,load_in_4bit=False,device="cpu",level="A")
        sft_data=root/"sft.jsonl"
        write_dataset(sft_data,make_disjoint_data(2))
        completed=[]
        for condition in ("explicit","direct","generic"):
            run_dir=Path(run(dict(config,condition=condition),sft_data))
            assert json.loads((run_dir/"status.json").read_text())["global_step"]==1
            completed.append("sft_"+condition)
        outputs=evaluate(run_dir,sft_data,budget=16)
        assert len(Path(outputs).read_text().splitlines())==4
        assert evaluate(run_dir,sft_data,budget=16)==outputs
        from gpu_lab.analysis import analyze,export_annotations
        analyze(root/"sft",root/"analysis")
        assert export_annotations(root/"sft",root/"annotation")==2
        # Simula interrupção após checkpoint 1, preservando a configuração.
        resume_cfg=dict(config,condition="explicit",train_seed=43,max_steps=2)
        resumed=Path(run(resume_cfg,sft_data))
        shutil.rmtree(resumed/"checkpoints/checkpoint-2")
        (resumed/"status.json").write_text(json.dumps({"status":"failed"}))
        assert Path(run(resume_cfg,sft_data))==resumed
        assert json.loads((resumed/"status.json").read_text())["global_step"]==2
        monitor=LocalMonitor(str(base),resolve_revision(str(base)),root/"monitor.sqlite")
        assert type(monitor.flag("question","Use a nota auxiliar.")) is bool
        monitor.close()
        from gpu_lab.independent_monitor import evaluate_monitor
        visible=root/"visible.jsonl"
        visible.write_text(json.dumps(dict(item_id="i",run_id="r",evaluation_mode="full",base_only=False,
            exploit=True,format_valid=True,prompt="question",generation=
            "<trace>Use a nota auxiliar.</trace><summary></summary><answer>A</answer>"))+"\n")
        judged=root/"judged.jsonl"
        evaluate_monitor(visible,judged,str(base),resolve_revision(str(base)),backend="local",modes=("full",))
        judgment=json.loads(judged.read_text())
        assert type(judgment["trace_flag"]) is bool and judgment["summary_flag"] is None
        grpo_data=root/"grpo.jsonl"
        write_dataset(grpo_data,create_dataset(1))
        grpo=dict(config,mode="grpo",level="low",condition="none",output_root=str(root/"grpo"),
            num_generations=2,max_prompt_length=256,max_completion_length=16,beta=0.0,
            heldout_domain="inventory",monitor_backend="local",monitor_model=str(base),
            monitor_revision=resolve_revision(str(base)))
        for condition in ("none","cot_monitor","summary_monitor"):
            result=Path(run(dict(grpo,condition=condition),grpo_data))
            assert json.loads((result/"status.json").read_text())["global_step"]==1
            completed.append("grpo_"+condition)
        print(json.dumps(dict(status="passed",device="cpu",random_tiny_model=True,
            steps=completed,evaluation_resume=True,checkpoint_resume=True,local_monitor_inference=True,
            independent_monitor=True,analysis_and_annotations=True)))


if __name__=="__main__":
    main()
