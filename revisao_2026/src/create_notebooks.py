"""Gera notebooks Colab executáveis em ordem, com código-fonte fixado por commit."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import textwrap

ROOT=Path(__file__).resolve().parents[1]


def md(source):
    return dict(cell_type="markdown",metadata={},source=textwrap.dedent(source).strip()+"\n")


def code(source):
    return dict(cell_type="code",metadata={},execution_count=None,outputs=[],
                source=textwrap.dedent(source).strip()+"\n")


BOOTSTRAP='''
# Execute esta célula. As opções abaixo já estão prontas para o teste curto.
PROFILE = "quick" # @param ["quick", "pilot", "full"]
SALVAR_NO_DRIVE = True # @param {type:"boolean"}

from pathlib import Path
import json, os, subprocess, sys, tempfile, shutil
SOURCE_REF = "__SOURCE_REF__"
NEEDS_GPU = __NEEDS_GPU__
try:
    from google.colab import drive
    ON_COLAB = True
except ImportError:
    ON_COLAB = False

def bootstrap_run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError("Falha ao preparar o código do laboratório:\\n" + result.stdout + result.stderr)

override = os.environ.get("COT_PROJECT_ROOT")
if override:
    ROOT = Path(override).resolve()
elif ON_COLAB:
    cache = Path("/content/cot-lab/source")
    cache.mkdir(parents=True, exist_ok=True)
    checkout = cache / SOURCE_REF
    if not (checkout / ".git").is_dir():
        temporary = Path(tempfile.mkdtemp(prefix="download-", dir=cache))
        try:
            bootstrap_run(["git", "init", "--quiet", str(temporary)])
            bootstrap_run(["git", "-C", str(temporary), "fetch", "--quiet", "--depth", "1",
                "https://github.com/Kelvin-Marques-Cyber/cot-obfuscation-h2a.git", SOURCE_REF])
            bootstrap_run(["git", "-C", str(temporary), "checkout", "--quiet", "--detach", "FETCH_HEAD"])
            temporary.rename(checkout)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    actual_ref = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    assert actual_ref == SOURCE_REF, "Checkout diverge do notebook. Abra a versão atual no GitHub."
    ROOT = checkout / "revisao_2026"
else:
    cwd = Path.cwd()
    candidates = [cwd, cwd.parent, cwd / "revisao_2026", cwd.parent / "revisao_2026"]
    ROOT = next((p for p in candidates if (p / "src/colab_runtime.py").is_file()), None)
    if ROOT is None:
        raise RuntimeError("Fora do Colab, abra o notebook dentro do clone do repositório.")

assert (ROOT / "src/colab_runtime.py").is_file(), "Código do laboratório não encontrado."
if os.environ.get("COT_STORAGE_ROOT"):
    STORAGE = Path(os.environ["COT_STORAGE_ROOT"])
elif ON_COLAB and SALVAR_NO_DRIVE:
    drive.mount("/content/drive")
    STORAGE = Path("/content/drive/MyDrive/CoT_Obfuscation_Lab/colab_v3")
else:
    STORAGE = Path("/content/cot-lab/results") if ON_COLAB else Path.home()/"CoT_Obfuscation_Lab/colab_v3"
    if ON_COLAB:
        print("Resultados nesta sessão: baixe o ZIP final antes de encerrar o Colab.")
STORAGE.mkdir(parents=True, exist_ok=True)
os.environ["COT_SOURCE_REF"] = SOURCE_REF
sys.path.insert(0, str(ROOT / "src"))
from colab_runtime import ColabSession
SESSION = ColabSession(ROOT, STORAGE, PROFILE, gpu=NEEDS_GPU)
(STORAGE / ("session_"+PROFILE+".json")).write_text(json.dumps({"source_ref":SOURCE_REF,
    "profile":PROFILE,"persistent_drive":bool(ON_COLAB and SALVAR_NO_DRIVE),"storage":str(STORAGE)}, indent=2))
print("Código:", SOURCE_REF[:12], "| Perfil:", PROFILE)
print("Resultados:", STORAGE)
'''


def write(name,cells,gpu):
    for i,cell in enumerate(cells):
        cell["id"]=hashlib.sha256((name+str(i)).encode()).hexdigest()[:12]
        if cell["cell_type"]=="code":
            compile(cell["source"],name,"exec")
    metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python"},"colab":{"name":name,"provenance":[]},
        "execution_status":"Notebook code prepared; CUDA execution must be validated on the user's runtime."}
    if gpu:
        metadata.update(accelerator="GPU")
        metadata["colab"]["gpuType"]="T4"
    (ROOT/"notebooks"/name).write_text(json.dumps(dict(cells=cells,metadata=metadata,
        nbformat=4,nbformat_minor=5),indent=1,ensure_ascii=False)+"\n")


def main(source_ref):
    if len(source_ref)!=40 or any(c not in "0123456789abcdef" for c in source_ref):
        raise ValueError("Informe o SHA completo do commit que contém o runtime.")
    (ROOT/"notebooks").mkdir(exist_ok=True)
    bootstrap=lambda gpu: code(BOOTSTRAP.replace("__SOURCE_REF__",source_ref).replace("__NEEDS_GPU__",str(gpu)))
    setup=code("SESSION.ensure_environment()")
    common=("Use **Ambiente de execução → Executar tudo**. Autorize o Drive quando solicitado. "
        "Instalação, revisões, dados, treino, avaliação e retomada são automáticos. "
        "O perfil **quick** é um teste curto de infraestrutura; não é o estudo completo. "
        "**pilot** e **full** aumentam o trabalho e ficam em pastas distintas. "
        "Não é necessário alterar flags de execução nem fornecer chave de API.")
    write("01_reanalise_reprodutivel.ipynb",[
        md("# 01 — Reanálise reproduzível (CPU)\n"+common+"\n\nEste notebook não treina modelos."),
        bootstrap(False),setup,
        code('SESSION.command(["-m","unittest","discover","-s","tests","-v"], "01_tests.log")\n'
             'SESSION.command(["src/reanalyse.py","--output-root",str(STORAGE/"reanalysis")], "01_reanalysis.log")'),
        code('for name in ["summary.csv","contrasts.csv","grading_stress_summary.csv"]:\n'
             '    SESSION.show_csv(STORAGE/"reanalysis/results"/name)\n'
             'from IPython.display import display, Image\n'
             'for path in sorted((STORAGE/"reanalysis/figures").glob("*.png")):\n'
             '    display(Image(data=path.read_bytes()))'),
        code('SESSION.bundle("reanalysis")'),
        md("Os dados históricos contêm métricas, sem as 540 gerações integrais. Os testes do corretor são mecânicos.")
    ],False)
    for number,mode,title in [(2,"sft","Piloto SFT corrigido"),(3,"grpo","GRPO com monitores locais")]:
        explanation=("Compara alvos explícito, direto e genérico." if mode=="sft" else
            "Executa controle, monitor de rastro e monitor de resumo. O monitor local usa Qwen2.5-0.5B "
            "por classificação de verossimilhança em CPU; nenhum endpoint é necessário. "
            "O instrumento é exploratório e exige calibração para pesquisa confirmatória.")
        write(f"{number:02d}_"+("piloto_sft_corrigido" if mode=="sft" else "grpo_multiescala")+".ipynb",[
            md(f"# {number:02d} — {title} (GPU)\n"+common+"\n\n"+explanation+
               "\n\nSelecione GPU T4 antes de executar. O perfil full inclui modelos e grades maiores; acesso a modelos restritos depende da conta do Hugging Face."),
            bootstrap(True),setup,
            code(f'SESSION.pipeline("prepare", "{mode}")'),
            code(f'SESSION.pipeline("train", "{mode}")'),
            code(f'SESSION.pipeline("evaluate", "{mode}")\nSESSION.pipeline("analyze", "{mode}")'),
            code(f'SESSION.show_csv(STORAGE/PROFILE/"analysis/{mode}/new_model_summary.csv")\n'
                 f'SESSION.bundle("{mode}")'),
            md("Pesos e checkpoints permanecem no Drive. O ZIP contém métricas, saídas, configurações e logs. Reexecutar retoma as etapas pendentes.")
        ],True)
    write("04_avaliacao_monitores_anotacao.ipynb",[
        md("# 04 — Avaliação independente e anotação (GPU)\n"+common+
           "\n\nUsa os treinos do notebook 03. Caso faltem, prepara e conclui automaticamente o plano do mesmo perfil. "
           "O monitor independente local SmolLM2-135M tem pesos distintos do ator e do monitor da recompensa. "
           "Monitora o modo full; os cinco modos têm métricas comportamentais. "
           "Modelos pequenos são instrumentos exploratórios; sua saída não é anotação humana."),
        bootstrap(True),setup,
        code('SESSION.pipeline("evaluate", "grpo")'),
        code('SESSION.pipeline("independent", "grpo")\nSESSION.pipeline("analyze", "grpo")'),
        code('SESSION.show_csv(STORAGE/PROFILE/"analysis/grpo/new_model_summary.csv")\n'
             'SESSION.show_csv(STORAGE/PROFILE/"analysis/grpo/paired_early_answer.csv")\n'
             'print("Formulário de anotação:",STORAGE/PROFILE/"annotation/grpo/blinded_annotations.csv")\n'
             'print("Chave privada (não distribuir aos anotadores):",STORAGE/PROFILE/"annotation/grpo/private_annotation_key.csv")\n'
             'SESSION.bundle("grpo")'),
        md("A anotação humana permanece manual. Salve completed_annotations.csv na pasta de anotação e reexecute para calcular concordância. "
           "Sem esse arquivo, o relatório registra not_collected; nenhum rótulo é inventado. O PDF do artigo não é alterado pela coleta exploratória.")
    ],True)
    print("4 notebooks gerados; código Python compilado; fonte fixada em",source_ref)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ref",default=None)
    args=parser.parse_args()
    ref=args.source_ref or subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    main(ref)
