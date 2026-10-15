"""Gera notebooks limpos e reproduzíveis, sem credenciais ou saídas inventadas."""
from pathlib import Path
import hashlib
import json
import textwrap

ROOT=Path(__file__).resolve().parents[1]


def md(text):
    return {"cell_type":"markdown","metadata":{},"source":textwrap.dedent(text).strip()+"\n"}


def code(text):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],
            "source":textwrap.dedent(text).strip()+"\n"}


BOOTSTRAP="""
from pathlib import Path
import os, sys, json, subprocess
cwd = Path.cwd()
candidates = [cwd, cwd.parent, cwd / "revisao_2026", cwd.parent / "revisao_2026",
              cwd / "cot-obfuscation-h2a" / "revisao_2026"]
ROOT = next((p for p in candidates if (p / "src/instruments.py").is_file()), None)
if ROOT is None:
    clone = cwd / "cot-obfuscation-h2a"
    if not clone.exists():
        subprocess.run(["git", "clone", "https://github.com/Kelvin-Marques-Cyber/cot-obfuscation-h2a.git",
                        str(clone)], check=True)
    ROOT = clone / "revisao_2026"
assert (ROOT / "src/instruments.py").is_file(), ROOT
ROOT = ROOT.resolve()
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
ENV = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
print("Raiz:", ROOT)
"""

GPU_SETUP="""
INSTALL_DEPS = False  # True na primeira instalação; depois reinicie o kernel se necessário.
if INSTALL_DEPS:
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("Instale primeiro PyTorch/CUDA para sua GPU: https://pytorch.org/get-started/locally/")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-gpu.txt"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)

import torch
assert torch.cuda.is_available(), "Selecione runtime GPU. Para CPU, use o notebook 01."
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Capacidade:", torch.cuda.get_device_capability(0))
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory/2**30, 2))
from importlib.metadata import version
assert version("trl") == "0.22.2", "Use a versão fixada e reinicie o kernel."
assert version("transformers") == "4.56.2", "Use a versão fixada e reinicie o kernel."
"""

SHELL_TRAIN="""
EXECUTAR_TREINO = False  # Ative após conferir modelo, plano e diretório persistente.
if EXECUTAR_TREINO:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    # Um plano existente é reutilizado, sem reconsultar a revisão mutável main.
    frozen_path = RUN_ROOT / "frozen_plan.json"
    if frozen_path.exists():
        stored = json.loads(frozen_path.read_text())
        strip_revision = lambda entries: [{k:v for k,v in c.items() if k != "revision"} for c in entries]
        assert strip_revision(stored) == strip_revision(plan), "Plano mudou: use novo RUN_ROOT."
        plan = stored
    else:
        from huggingface_hub import model_info
        shas = {(c["model_id"], c["revision"]): model_info(c["model_id"], revision=c["revision"]).sha for c in plan}
        for cfg in plan:
            cfg["revision"] = shas[(cfg["model_id"], cfg["revision"])]
        frozen_path.write_text(json.dumps(plan, indent=2))
    (RUN_ROOT / "pip-freeze.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True))
    for i, cfg in enumerate(plan):
        config_path = RUN_ROOT / f"config_{i:04d}.json"
        config_path.write_text(json.dumps(cfg, indent=2))
        print("Executando:", i+1, "/", len(plan), cfg["model_id"], cfg["condition"], cfg["train_seed"], flush=True)
        subprocess.run([sys.executable, "-m", "gpu_lab.train", "--config", str(config_path),
                        "--data", str(DATA_PATH)], env=ENV, check=True)
else:
    print("Plano preparado. Nenhum treino executado nesta célula.")
"""


def write(name,cells):
    for i,c in enumerate(cells):
        c["id"]=hashlib.sha256((name+str(i)).encode()).hexdigest()[:12]
        if c["cell_type"]=="code":
            compile(c["source"],name,"exec")
    nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3.12"},
        "execution_status":"GPU notebooks are prospective; no fabricated outputs"},
        "nbformat":4,"nbformat_minor":5}
    (ROOT/"notebooks"/name).write_text(json.dumps(nb,indent=1,ensure_ascii=False)+"\n")


def main():
    (ROOT/"notebooks").mkdir(exist_ok=True)
    write("01_reanalise_reprodutivel.ipynb",[
        md("# 01 - Reanálise reproduzível\nExecuta a auditoria CPU, os testes e os gráficos. Não treina um LLM."),
        code(BOOTSTRAP),
        code("""
        INSTALL_DEPS = False
        if INSTALL_DEPS:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.lock.txt"], check=True)
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], env=ENV, check=True)
        subprocess.run([sys.executable, "src/reanalyse.py"], env=ENV, check=True)
        """),
        code("""
        import pandas as pd
        display(pd.read_csv(ROOT / "results/summary.csv"))
        display(pd.read_csv(ROOT / "results/contrasts.csv"))
        display(pd.read_csv(ROOT / "results/grading_stress_summary.csv"))
        """),
        md("Os resultados são reprocessamento de métricas depositadas e testes mecânicos. As 540 saídas integrais não estão no CSV original.")
    ])
    write("02_piloto_sft_corrigido.ipynb",[
        md("# 02 - Piloto SFT corrigido\nTrês alvos supervisionados; não é um teste de emergência por supervisão exclusiva de saída. Use GPU e volume persistente. Código compilado, treino não executado na revisão."),
        code(BOOTSTRAP),code(GPU_SETUP),
        md("## Dados e plano\nA mesma partição é compartilhada pelas condições e sementes de treino. Semente de dados e de treino são distintas. IDs evitam colisões entre partições."),
        code("""
        from instruments import make_disjoint_data
        from gpu_lab.benchmark import write_dataset
        from gpu_lab.train import validate_config
        SMOKE = True
        RUN_ROOT = (ROOT / "runs/sft").resolve()  # Troque para seu volume persistente.
        DATA_PATH = ROOT / "data/sft_disjoint.jsonl"
        data = make_disjoint_data(n_per_split=8 if SMOKE else 500, seed=20261015)
        print(write_dataset(DATA_PATH, data))
        template = json.loads((ROOT / "configs/sft_smoke.json").read_text())
        plan = []
        for seed in ([42] if SMOKE else [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]):
            for level in (["A"] if SMOKE else ["A", "B", "C"]):
                for condition in ["explicit", "direct", "generic"]:
                    cfg = dict(template, train_seed=seed, level=level, condition=condition,
                               max_steps=4 if SMOKE else 200, output_root=str(RUN_ROOT))
                    validate_config(cfg)
                    plan.append(cfg)
        print("Execuções planejadas:", len(plan))
        print(json.dumps(plan[0], indent=2))
        """),code(SHELL_TRAIN),
        md("## Avaliar e retomar\nA avaliação preserva enunciado, gabarito, saída integral, revisão dos pesos, tokens e motivo do término. O modo early usa o mesmo orçamento total, incluindo o prefixo forçado."),
        code("""
        EXECUTAR_AVALIACAO = False
        if EXECUTAR_AVALIACAO:
            for status in sorted(RUN_ROOT.glob("*/status.json")):
                if json.loads(status.read_text()).get("status") == "trained":
                    subprocess.run([sys.executable, "-m", "gpu_lab.evaluate",
                        "--run-dir", str(status.parent), "--data", str(DATA_PATH),
                        "--budget", "384"], env=ENV, check=True)
        """)
    ])
    write("03_grpo_multiescala.ipynb",[
        md("# 03 - GRPO multiescala\nRecompensa de exploração e monitores separados de rastro/resumo. Benchmark sintético próprio; não é réplica dos datasets do artigo-alvo. Dificuldade da dica é proxy, não necessidade causal comprovada. Treino CUDA não executado na revisão."),
        code(BOOTSTRAP),code(GPU_SETUP),
        code("""
        from gpu_lab.benchmark import create_dataset, write_dataset
        from gpu_lab.train import validate_config
        SMOKE = True
        RUN_ROOT = (ROOT / "runs/grpo").resolve()  # Defina armazenamento persistente.
        DATA_PATH = ROOT / "data/grpo_synthetic.jsonl"
        data = create_dataset(n_per_cell=4 if SMOKE else 500, seed=20261015)
        print(write_dataset(DATA_PATH, data))
        MONITOR_MODEL = os.environ.get("MONITOR_MODEL", "")
        MONITOR_REVISION = os.environ.get("MONITOR_REVISION", "")
        print("Endpoint configurado:", bool(os.environ.get("MONITOR_BASE_URL")))
        # Não imprima chaves. Configure MONITOR_API_KEY somente no ambiente.
        MODELS = {"Qwen/Qwen3-4B": 12, "Qwen/Qwen3-14B": 6,
                  "meta-llama/Llama-3.1-8B-Instruct": 12}
        CONDITIONS = ["none"] if SMOKE else ["none", "cot_monitor", "summary_monitor"]
        LEVELS = ["low"] if SMOKE else ["low", "medium", "high"]
        template = json.loads((ROOT / "configs/grpo_smoke.json").read_text())
        plan = []
        chosen = {"Qwen/Qwen3-4B": 1} if SMOKE else MODELS
        for model_id, n_seeds in chosen.items():
            for seed in range(42, 42 + n_seeds):
                for level in LEVELS:
                    for condition in CONDITIONS:
                        cfg = dict(template, model_id=model_id, train_seed=seed, level=level,
                            condition=condition, monitor_model=MONITOR_MODEL,
                            monitor_revision=MONITOR_REVISION, output_root=str(RUN_ROOT),
                            max_steps=4 if SMOKE else 4000)
                        validate_config(cfg)
                        plan.append(cfg)
        print("Execuções planejadas:", len(plan))
        print("SMOKE testa infraestrutura; o controle sozinho não mede ofuscação.")
        """),
        md("## Congelar revisões antes dos treinamentos\nEsta célula resolve os identificadores de modelos para SHAs imutáveis e salva o plano. Para estudo confirmatório, registre também protocolo e análise antes da coleta."),
        code("""
        from huggingface_hub import model_info
        FREEZE_REVISIONS = False
        if FREEZE_REVISIONS:
            shas = {name: model_info(name, revision="main").sha for name in {c["model_id"] for c in plan}}
            for cfg in plan:
                cfg["revision"] = shas[cfg["model_id"]]
            RUN_ROOT.mkdir(parents=True, exist_ok=True)
            (RUN_ROOT / "frozen_plan.json").write_text(json.dumps(plan, indent=2))
        """),
        code(SHELL_TRAIN),
        md("Repita o plano idêntico para retomar checkpoints. Não mude budgets ou datasets sob o mesmo identificador de execução. Os testes das condições verificam que o monitor de resumo não recebe o rastro. Use o notebook 04 para a avaliação.")
    ])
    write("04_avaliacao_monitores_anotacao.ipynb",[
        md("# 04 - Avaliação independente e anotação\nAvaliação integral, modos de intervenção, monitor independente e exportação cega. Requer execuções próprias já treinadas. Sem dados, não produz tabelas fictícias."),
        code(BOOTSTRAP),
        code("""
        RUN_ROOT = (ROOT / "runs/grpo").resolve()
        DATA_PATH = ROOT / "data/grpo_synthetic.jsonl"
        statuses = sorted(RUN_ROOT.glob("*/status.json"))
        trained = [p.parent for p in statuses if json.loads(p.read_text()).get("status") == "trained"]
        print("Execuções treinadas encontradas:", len(trained))
        EXECUTAR_AVALIACAO = False
        if EXECUTAR_AVALIACAO:
            assert trained, "Nenhum adaptador treinado."
            for run_dir in trained:
                for base_only in [True, False]:
                    args = [sys.executable, "-m", "gpu_lab.evaluate", "--run-dir", str(run_dir),
                            "--data", str(DATA_PATH), "--budget", "384"]
                    if base_only:
                        args.append("--base-only")
                    subprocess.run(args, env=ENV, check=True)
        """),
        code("""
        from gpu_lab.analysis import analyze, export_annotations, independent_probe
        HAS_GENERATIONS = any(RUN_ROOT.rglob("evaluation_*.jsonl"))
        if HAS_GENERATIONS:
            display(analyze(RUN_ROOT, ROOT / "results/new_models"))
            print("Itens para anotação:", export_annotations(RUN_ROOT, ROOT / "annotation"))
            print("Sonda textual:", independent_probe(RUN_ROOT, ROOT / "results/new_models/probe.json"))
        else:
            print("Ainda não há gerações; execute as células de avaliação após o treino.")
        """),
        md("## Monitor independente\nUse identidade diferente da usada na recompensa. O avaliador não deve receber nomes das condições. Verifique falsos positivos e sensibilidade contra o comportamento observável, sem interpretar automaticamente como intenção."),
        code("""
        from gpu_lab.independent_monitor import evaluate_monitor
        EXECUTAR_MONITOR = False
        INDEPENDENT_MODEL = os.environ.get("INDEPENDENT_MONITOR_MODEL", "")
        INDEPENDENT_REVISION = os.environ.get("INDEPENDENT_MONITOR_REVISION", "")
        INDEPENDENT_BASE_URL = os.environ.get("INDEPENDENT_MONITOR_BASE_URL", "")
        if EXECUTAR_MONITOR:
            assert all([INDEPENDENT_MODEL, INDEPENDENT_REVISION, INDEPENDENT_BASE_URL])
            for source in sorted(RUN_ROOT.rglob("evaluation_*.jsonl")):
                cfg = json.loads((source.parent / "manifest.json").read_text())["resolved_config"]
                assert INDEPENDENT_MODEL != cfg.get("monitor_model"), "Use monitor independente."
                evaluate_monitor(source, source.with_name("independent_" + source.name),
                    INDEPENDENT_MODEL, INDEPENDENT_REVISION, INDEPENDENT_BASE_URL)
        """),
        md("## Concordância humana\nPreencha cópias de blinded_annotations.csv com rater e label. Não distribua private_annotation_key.csv aos avaliadores. O alfa pode ser negativo e pode ser indefinido com ausência de variação."),
        code("""
        from gpu_lab.analysis import nominal_alpha
        import csv
        ANNOTATIONS = ROOT / "annotation/completed_annotations.csv"
        if ANNOTATIONS.exists():
            with ANNOTATIONS.open() as f:
                annotations = list(csv.DictReader(f))
            print("Alfa nominal:", nominal_alpha(annotations))
        else:
            print("Nenhuma anotação humana foi fornecida.")
        """),
        md("A agregação descritiva e a comparação early/full não substituem o modelo hierárquico, o estudo de poder e as análises de equivalência do protocolo. O artigo não é atualizado automaticamente para evitar incorporar coleta exploratória como confirmatória.")
    ])
    print("4 notebooks gerados e código Python compilado.")


if __name__=="__main__":
    main()
