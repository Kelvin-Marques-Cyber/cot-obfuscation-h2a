"""Etapas automáticas dos notebooks, com dados e revisões congelados."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from .benchmark import create_dataset, write_dataset
from .revisions import resolve_revision
from .train import atomic_json, validate_config

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    "quick": dict(n_sft=4,n_grpo=2,steps_sft=2,steps_grpo=2,seeds=1,budget=128),
    "pilot": dict(n_sft=32,n_grpo=12,steps_sft=30,steps_grpo=30,seeds=3,budget=256),
    "full": dict(n_sft=500,n_grpo=500,steps_sft=200,steps_grpo=4000,seeds=12,budget=384),
}


def paths(storage, profile, mode):
    base = Path(storage).resolve()/profile
    return dict(base=base, runs=base/"runs"/mode, data=base/"data"/f"{mode}.jsonl",
                analysis=base/"analysis"/mode, annotation=base/"annotation"/mode)


def preserve_dataset(path, rows):
    """Não substitui dados que já sustentam checkpoints de uma execução."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".candidate.jsonl")
    try:
        write_dataset(temporary,rows)
        candidate=temporary.read_bytes()
        if path.exists():
            if path.read_bytes()!=candidate:
                raise ValueError("Dados existentes diferem do plano; use outro diretório de execução.")
        else:
            temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(storage, profile, mode, resolver=resolve_revision):
    p=paths(storage,profile,mode)
    spec=PROFILES[profile]
    p["runs"].mkdir(parents=True,exist_ok=True)
    from instruments import make_disjoint_data
    data=(make_disjoint_data(spec["n_sft"]) if mode=="sft" else create_dataset(spec["n_grpo"]))
    data_sha=preserve_dataset(p["data"],data)
    template=json.loads((ROOT/"configs"/f"{mode}_smoke.json").read_text())
    levels=(["A"] if mode=="sft" else ["low"]) if profile=="quick" else (
        ["A","B","C"] if mode=="sft" else ["low","medium","high"])
    conditions=["explicit","direct","generic"] if mode=="sft" else ["none","cot_monitor","summary_monitor"]
    models=({"Qwen/Qwen2.5-0.5B-Instruct":spec["seeds"]} if mode=="sft" else
            {"Qwen/Qwen3-0.6B":spec["seeds"]})
    if profile=="full":
        models=({"Qwen/Qwen2.5-7B-Instruct":12} if mode=="sft" else
                {"Qwen/Qwen3-4B":12,"Qwen/Qwen3-14B":6,"meta-llama/Llama-3.1-8B-Instruct":12})
    plan=[]
    for model,n_seeds in models.items():
        for seed in range(42,42+n_seeds):
            for level in levels:
                for condition in conditions:
                    cfg=dict(template,model_id=model,revision="main",train_seed=seed,level=level,
                        condition=condition,output_root=str(p["runs"]),
                        max_steps=spec["steps_"+mode],save_steps=2 if profile=="quick" else 20,precision="fp16",
                        lora_r=8 if profile=="quick" else 16,
                        load_in_4bit=profile!="quick" and template.get("load_in_4bit",True),
                        max_length=512 if profile=="quick" else 1024)
                    if mode=="grpo":
                        cfg.update(monitor_backend="local",monitor_model="Qwen/Qwen2.5-0.5B-Instruct",
                                   monitor_revision="main",max_completion_length=spec["budget"],
                                   max_prompt_length=512 if profile=="quick" else 768)
                    validate_config(cfg)
                    plan.append(cfg)
    frozen=p["runs"]/"frozen_plan.json"
    if frozen.exists():
        saved=json.loads(frozen.read_text())
        strip=lambda entries:[{k:v for k,v in c.items() if k not in ("revision","monitor_revision")}
                               for c in entries]
        if strip(saved)!=strip(plan):
            # Migração segura de QLoRA 4-bit para LoRA fp16 no perfil quick.
            legacy=[dict(cfg,load_in_4bit=False) for cfg in saved] if profile=="quick" else []
            compatible=(len(saved)==len(plan) and strip(legacy)==strip(plan))
            completed=any(json.loads(s.read_text()).get("status")=="trained"
                          for s in p["runs"].glob("*/status.json"))
            if not compatible or completed:
                raise ValueError("Plano mudou; treinos concluídos/checkpoints antigos "
                                 "foram preservados. Use outro perfil/diretório.")
            backup=frozen.with_name("frozen_plan_4bit_backup.json")
            if not backup.exists():
                atomic_json(backup,saved)
            for current,previous in zip(plan,saved):
                current["revision"]=previous["revision"]
                if "monitor_revision" in previous:
                    current["monitor_revision"]=previous["monitor_revision"]
            atomic_json(frozen,plan)
            print("Plano quick legado migrado para fp16. Versões imutáveis e backup preservados.",flush=True)
        else:
            plan=saved
    else:
        revisions={}
        for cfg in plan:
            for name_key,revision_key in [("model_id","revision"),("monitor_model","monitor_revision")]:
                if name_key in cfg:
                    model=cfg[name_key]
                    if model not in revisions:
                        revisions[model]=resolver(model,cfg[revision_key])
                    cfg[revision_key]=revisions[model]
        atomic_json(frozen,plan)
    atomic_json(p["runs"]/"dataset.json",dict(path=str(p["data"]),sha256=data_sha,rows=len(data)))
    print(f"Plano {profile}: {len(plan)} treinos {mode}; dados preservados em {p['data']}",flush=True)
    return plan,p


def run_child(arguments):
    # Saída herdada: o runner externo do notebook salva o log integral.
    subprocess.run([sys.executable,"-u",*map(str,arguments)],check=True)


def train_all(storage,profile,mode):
    plan,p=prepare(storage,profile,mode)
    existing={json.loads(m.read_text())["run_id"]:m.parent for m in p["runs"].glob("*/manifest.json")}
    from .benchmark import content_hash
    data_sha=hashlib.sha256(p["data"].read_bytes()).hexdigest()
    for i,cfg in enumerate(plan):
        run_id=content_hash({**cfg,"resolved_revision":cfg["revision"],"data_sha256":data_sha})[:20]
        run_dir=existing.get(run_id)
        if run_dir and (run_dir/"status.json").exists() and json.loads((run_dir/"status.json").read_text()).get("status")=="trained":
            print("Já treinado:",run_id,flush=True)
            continue
        config=p["runs"]/f"config_{i:04d}.json"
        atomic_json(config,cfg)
        print(f"Treino {i+1}/{len(plan)}: {cfg['model_id']} | {cfg['condition']} | semente {cfg['train_seed']}",flush=True)
        run_child(["-m","gpu_lab.train","--config",config,"--data",p["data"]])


def trained_runs(p):
    return [s.parent for s in sorted(p["runs"].glob("*/status.json"))
            if json.loads(s.read_text()).get("status")=="trained"]


def evaluate_all(storage,profile,mode):
    # Torna o notebook 04 independente: retoma/completa o plano se necessário.
    train_all(storage,profile,mode)
    p=paths(storage,profile,mode)
    runs=trained_runs(p)
    if not runs:
        raise RuntimeError("Nenhum treino concluído; verifique o log anterior.")
    for run in runs:
        for base in (True,False):
            print("Avaliação:",run.name,"base" if base else "adaptador",flush=True)
            args=["-m","gpu_lab.evaluate","--run-dir",run,"--data",p["data"],
                  "--budget",str(PROFILES[profile]["budget"])]
            if base:
                args.append("--base-only")
            run_child(args)


def analyze_all(storage,profile,mode):
    from .analysis import analyze,export_annotations,independent_probe,nominal_alpha
    p=paths(storage,profile,mode)
    summary=analyze(p["runs"],p["analysis"])
    count=export_annotations(p["runs"],p["annotation"])
    probe=independent_probe(p["runs"],p["analysis"]/"probe.json")
    atomic_json(p["analysis"]/"probe.json",probe)
    completed=p["annotation"]/"completed_annotations.csv"
    human={"status":"not_collected","alpha":None}
    if completed.exists():
        import csv
        with completed.open(newline="") as f:
            rows=list(csv.DictReader(f))
        human={"status":"provided","alpha":nominal_alpha(rows)}
    atomic_json(p["analysis"]/"human_annotation_status.json",human)
    print(summary.to_string(index=False))
    print("Itens exportados para anotação:",count,"| Anotação humana:",human["status"],flush=True)


def independent_all(storage,profile,mode):
    from .independent_monitor import evaluate_monitor
    p=paths(storage,profile,mode)
    p["analysis"].mkdir(parents=True,exist_ok=True)
    identity=p["analysis"]/"independent_monitor.json"
    if identity.exists():
        model=json.loads(identity.read_text())
    else:
        name="HuggingFaceTB/SmolLM2-135M-Instruct"
        model=dict(model_id=name,revision=resolve_revision(name),backend="local",
                   status="exploratory_unvalidated",evaluation_modes=["full"])
        atomic_json(identity,model)
    for source in sorted(p["runs"].rglob("evaluation_*.jsonl")):
        cfg=json.loads((source.parent/"manifest.json").read_text())["resolved_config"]
        if model["model_id"] in (cfg["model_id"],cfg.get("monitor_model")):
            raise ValueError("Monitor independente deve ter identidade distinta.")
        evaluate_monitor(source,source.with_name("independent_"+source.name),
            model["model_id"],model["revision"],backend="local",modes=("full",))
    print("Monitor local independente concluído. Não equivale a validação humana.",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["prepare","train","evaluate","analyze","independent"])
    parser.add_argument("--mode",choices=["sft","grpo"],default="grpo")
    parser.add_argument("--profile",choices=PROFILES,default="quick")
    parser.add_argument("--storage",type=Path,required=True)
    args=parser.parse_args()
    function={"prepare":prepare,"train":train_all,"evaluate":evaluate_all,
              "analyze":analyze_all,"independent":independent_all}[args.stage]
    function(args.storage,args.profile,args.mode)
