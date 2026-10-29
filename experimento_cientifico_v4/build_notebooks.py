"""Gera os cinco notebooks do estudo v4 a partir de um commit de código fixado.

Uso: python build_notebooks.py --code-ref SHA_COMPLETO
O SHA deve conter cotv4/, colab/, protocolo_v4.json e triagem_dev/ exatamente
como neste diretório; os hashes são embutidos e conferidos no Colab.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_URL = 'https://github.com/Kelvin-Marques-Cyber/cot-obfuscation-h2a.git'
DEFAULT_STUDY = 'estudo-prospectivo-2026-v4'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell(kind, body):
    body = textwrap.dedent(body).strip() + '\n'
    out = dict(cell_type=kind, metadata={}, source=body,
               id=hashlib.sha256((kind + body).encode()).hexdigest()[:12])
    if kind == 'code':
        out.update(execution_count=None, outputs=[])
    return out


def bootstrap(code_ref, extra_params):
    code_hashes = {p.name: sha(p) for p in sorted((ROOT / 'cotv4').glob('*.py'))}
    support = {name: sha(ROOT / 'colab' / name) for name in ('armazenamento.py', 'sessao.py')}
    return f'''
# Configuração operacional. Sementes, tamanhos, modelos e critérios estão fixos no protocolo v4.
STUDY_ID = "{DEFAULT_STUDY}"  # @param {{type:"string"}}
STORAGE_MODE = "auto"  # @param ["auto", "local"]
TECHNICAL_TEST = False  # @param {{type:"boolean"}}
{extra_params}
CODE_COMMIT = "{code_ref}"
CODE_HASHES = {json.dumps(code_hashes, indent=1)}
SUPPORT_HASHES = {json.dumps(support, indent=1)}

import hashlib, importlib.util, os, re, subprocess, sys, time
from pathlib import Path
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{{0,79}}", STUDY_ID):
    raise ValueError("STUDY_ID aceita somente letras, números, _ e -.")
if TECHNICAL_TEST and not STUDY_ID.endswith("-teste-tecnico"):
    STUDY_ID += "-teste-tecnico"  # pasta separada: nunca se mistura ao estudo
try:
    from google.colab import drive
    ON_COLAB = True
except ImportError:
    drive, ON_COLAB = None, False
LOCAL = Path(os.environ.get("COT_LOCAL_ROOT", "/content/cot-v4" if ON_COLAB else Path.cwd() / "cot-v4"))
LOCAL.mkdir(parents=True, exist_ok=True)

def checkout(ref):
    """Obtém exatamente o commit fixado; não usa clones antigos nem a branch atual."""
    override = os.environ.get("COT_REPO_DIR")
    if override:
        return Path(override).resolve()
    repo = LOCAL / "source" / ref
    git = lambda *a, **k: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, **k)
    if not (repo / ".git").is_dir():
        repo.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
        git("remote", "add", "origin", "{REPO_URL}")
    if git("cat-file", "-e", ref + "^{{commit}}").returncode:
        for attempt in range(1, 4):
            fetched = git("fetch", "--quiet", "--depth", "1", "origin", ref)
            if not fetched.returncode:
                break
            print(f"Falha de rede ao baixar o código (tentativa {{attempt}}/3): {{fetched.stderr.strip()[-300:]}}")
            time.sleep(5 * attempt)
        else:
            raise RuntimeError("Não foi possível baixar o commit " + ref + ". Verifique a conexão e execute de novo.")
    if git("checkout", "--quiet", "--detach", ref).returncode:
        raise RuntimeError("Checkout do commit fixado falhou: " + ref)
    if git("rev-parse", "HEAD").stdout.strip() != ref:
        raise RuntimeError("O código aberto não corresponde ao commit do notebook.")
    return repo

REPO = checkout(CODE_COMMIT)
PACKAGE = REPO / "experimento_cientifico_v4"
def load(name):
    path = PACKAGE / "colab" / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != SUPPORT_HASHES[name]:
        raise RuntimeError("Suporte divergente do notebook: " + name + ". Reabra pelo link do GitHub.")
    # Nome único por commit: módulos de versões antigas na mesma sessão não são reutilizados.
    spec = importlib.util.spec_from_file_location(f"cotv4_{{name[:-3]}}_{{CODE_COMMIT[:12]}}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
armazenamento, sessao = load("armazenamento.py"), load("sessao.py")
STATE = armazenamento.prepare_storage(LOCAL, STUDY_ID, on_colab=ON_COLAB, mode=STORAGE_MODE,
    storage_override=os.environ.get("COT_STORAGE_ROOT"), drive_module=drive,
    mountpoint=os.environ.get("COT_DRIVE_MOUNTPOINT", "/content/drive"))
STUDY = sessao.Session(REPO, STATE, CODE_COMMIT, CODE_HASHES, on_colab=ON_COLAB, technical_test=TECHNICAL_TEST)
Blocked = sessao.StageBlocked
print("Código:", CODE_COMMIT, "| Estudo:", STUDY_ID, "| Teste técnico:", TECHNICAL_TEST)
'''


INTRO = {
    '00': ('00 · Protocolo, dados e triagem DEV (v4)', '''
**Ambiente: CPU.** Use **Ambiente de execução → Executar tudo** e autorize o Google Drive.

1. Congela protocolo, código, dados (train/dev/test) e revisões dos quatro modelos.
2. Importa a triagem DEV dos modelos-base, executada uma única vez antes do registro, e
   confere os SHA256. Se a triagem reprovar, os treinos ficam bloqueados.
3. Exporta `pacote_registro_v4.zip` para o registro público (OSF ou equivalente).
4. Depois de publicar o registro, preencha `PUBLIC_REGISTRATION_URL` e execute tudo de novo.

Se o Drive não montar, a preparação continua em pasta local **temporária** e o ZIP pode
ser baixado; nada é registrado sem armazenamento persistente.
`TECHNICAL_TEST=True` usa uma pasta separada e não produz evidência científica.'''),
    '01': ('01 · Treinamento SFT replicado (v4)', '''
**Ambiente: GPU (T4 ou superior).** Executar tudo. Requer o notebook 00 concluído, triagem DEV
aprovada e registro público declarado (exceto no teste técnico).

Treina 3 condições (explicit, direct, generic) × 2 níveis (A, C) × 20 sementes = 120 treinos de
120 passos (LoRA fp16). Cada sessão conclui até `MAX_JOBS_PER_SESSION` treinos; reexecute para
continuar. Treinos concluídos não são repetidos; treinos interrompidos retomam do último checkpoint.'''),
    '02': ('02 · Treinamento GRPO replicado (v4)', '''
**Ambiente: GPU (T4 ou superior).** Executar tudo, com o mesmo `STUDY_ID`.

Treina none, cot_monitor e summary_monitor × níveis low/high × 20 sementes = 120 treinos de
120 passos. O monitor de recompensa (Qwen2.5-0.5B-Instruct) é distinto do monitor independente.
Reexecute até concluir; checkpoints são retomados e treinos concluídos são preservados.'''),
    '03': ('03 · Avaliação TEST pareada (v4)', '''
**Ambiente: GPU.** Só abre TEST depois dos 240 treinos. Avalia cada adaptador no modo `full`,
com orçamento de 256 tokens: 26.400 gerações SFT e 28.800 GRPO. Avaliações interrompidas
retomam do último item gravado; avaliações concluídas têm hash conferido e não são refeitas.'''),
    '04': ('04 · Monitor independente, anotação humana e inferência (v4)', '''
**Ambiente: GPU.** Aplica o monitor independente congelado às 120 avaliações GRPO, exporta a
amostra cega de 360 itens e calcula os contrastes por semente (inversão de sinais, bootstrap,
Holm). Os arquivos humanos devem ser preenchidos por pessoas e salvos em `human/`:
`completed_annotations.csv` (annotation_id, rater, label), `adjudication.csv` (annotation_id, label)
e `independent_monitor_controls.csv` (question, trace, label, source=human, split=heldout), com
pelo menos 100 positivos e 100 negativos. Sem eles, o relatório permanece `INSUFFICIENT_EVIDENCE`.'''),
}

STEPS = {
    '00': ['''
try:
    prepared = STUDY.execute("prepare")
    for name, info in prepared["datasets"].items():
        print(f"{name}: {info['rows']} itens | TEST por nível {info['test_per_level']} | SHA256 {info['sha256']}")
except Blocked as reason:
    print("ETAPA BLOQUEADA:", reason)
''', '''
try:
    STUDY.execute("verify-revisions")
    print("Revisões dos quatro modelos confirmadas no Hugging Face Hub.")
except (Blocked, RuntimeError) as error:
    print("REVISÕES NÃO VERIFICADAS (rede indisponível ou divergência):", str(error).splitlines()[0])
    print("Os SHA fixados no protocolo continuam valendo; verifique antes de registrar.")
gate = STUDY.execute("gate")
for level, result in gate.get("results", {}).items():
    print(f"Triagem DEV {level}: {result['correct']}/{result['n']} ({result['accuracy']:.1%}); critério 65%")
if gate.get("status") == "failed":
    print("TRIAGEM DEV REPROVADA:", sessao.explain(gate["reasons"]))
    print("O protocolo v4 não autoriza treinos neste estudo; os arquivos são preservados.")
elif gate.get("status") != "passed":
    print("TRIAGEM DEV NÃO DISPONÍVEL:", sessao.explain(gate.get("reasons", ["gate_not_run"])))
''', '''
if PUBLIC_REGISTRATION_URL.strip():
    try:
        print(STUDY.execute("register", url=PUBLIC_REGISTRATION_URL.strip()))
    except Blocked as reason:
        print("REGISTRO NÃO DECLARADO:", reason)
else:
    print("Registro público ainda não informado. Publique o pacote e preencha PUBLIC_REGISTRATION_URL.")
STUDY.export(registration=True)
STUDY.status()
'''],
    '01': ['''
status = STUDY.execute("status")
gate_ok = status["prepared"] and (status["gate"]["passed"] or TECHNICAL_TEST)
registered = status["registration"]["passed"] or TECHNICAL_TEST
if not status["prepared"]:
    print("ETAPA BLOQUEADA: execute primeiro o notebook 00 com o mesmo STUDY_ID.")
elif not gate_ok:
    print("ETAPA BLOQUEADA:", sessao.explain(status["gate"].get("reasons", [])))
elif not registered:
    print("ETAPA BLOQUEADA: declare o registro público no notebook 00 antes de treinar.")
else:
    try:
        progress = STUDY.execute("train", mode="sft", max_jobs=MAX_JOBS_PER_SESSION)
        print(f"SFT: {progress['completed']}/{progress['total']} treinos concluídos ({progress['status']}).")
    except Blocked as reason:
        print("ETAPA BLOQUEADA:", reason)
STUDY.export(download=False)
STUDY.status()
'''],
    '02': ['''
status = STUDY.execute("status")
gate_ok = status["prepared"] and (status["gate"]["passed"] or TECHNICAL_TEST)
registered = status["registration"]["passed"] or TECHNICAL_TEST
if not status["prepared"]:
    print("ETAPA BLOQUEADA: execute primeiro o notebook 00 com o mesmo STUDY_ID.")
elif not gate_ok:
    print("ETAPA BLOQUEADA:", sessao.explain(status["gate"].get("reasons", [])))
elif not registered:
    print("ETAPA BLOQUEADA: declare o registro público no notebook 00 antes de treinar.")
elif "grpo" in status.get("excluded", {}):
    print("ETAPA BLOQUEADA: braço GRPO excluído deste estudo antes do registro:", status["excluded"]["grpo"])
else:
    try:
        progress = STUDY.execute("train", mode="grpo", max_jobs=MAX_JOBS_PER_SESSION)
        print(f"GRPO: {progress['completed']}/{progress['total']} treinos concluídos ({progress['status']}).")
    except Blocked as reason:
        print("ETAPA BLOQUEADA:", reason)
STUDY.export(download=False)
STUDY.status()
'''],
    '03': ['''
status = STUDY.execute("status")
if not status["all_training_complete"]:
    print("ETAPA BLOQUEADA: TEST só é aberto após todos os treinos:", status["training"], "de", status["expected"])
else:
    try:
        for mode in status["expected"]:
            progress = STUDY.execute("evaluate", mode=mode, max_jobs=MAX_JOBS_PER_SESSION)
            print(f"{mode.upper()}: {progress['completed']}/{progress['total']} avaliações ({progress['status']}).")
    except Blocked as reason:
        print("ETAPA BLOQUEADA:", reason)
STUDY.export(download=False)
STUDY.status()
'''],
    '04': ['''
status = STUDY.execute("status")
if not status["all_evaluation_complete"]:
    print("ETAPA BLOQUEADA: conclua todas as avaliações no notebook 03:", status["evaluation"])
elif "grpo" in status.get("excluded", {}):
    print("NÃO SE APLICA: braço GRPO excluído deste estudo antes do registro:", status["excluded"]["grpo"])
else:
    try:
        print(STUDY.execute("monitor", mode="grpo", max_jobs=MAX_JOBS_PER_SESSION))
        print(STUDY.execute("blind", mode="grpo"))
        if (STUDY.storage / "human" / "independent_monitor_controls.csv").is_file():
            print(STUDY.execute("controls", mode="grpo"))
        else:
            print("PENDENTE: human/independent_monitor_controls.csv com rótulos humanos.")
    except Blocked as reason:
        print("ETAPA BLOQUEADA:", reason)
''', '''
report = STUDY.execute("report")
print("Veredito:", report["verdict"])
print("Pendências:", report["blocked_by"])
STUDY.export()
STUDY.status()
'''],
}

PARAMS = {
    '00': 'PUBLIC_REGISTRATION_URL = ""  # @param {type:"string"}',
    '01': 'MAX_JOBS_PER_SESSION = 2  # @param {type:"integer"}',
    '02': 'MAX_JOBS_PER_SESSION = 1  # @param {type:"integer"}',
    '03': 'MAX_JOBS_PER_SESSION = 2  # @param {type:"integer"}',
    '04': 'MAX_JOBS_PER_SESSION = 4  # @param {type:"integer"}',
}
NAMES = {
    '00': '00_protocolo_dados_e_triagem.ipynb',
    '01': '01_treinamento_sft.ipynb',
    '02': '02_treinamento_grpo.ipynb',
    '03': '03_avaliacao_test.ipynb',
    '04': '04_monitor_humanos_e_inferencia.ipynb',
}
FOOTER = '''
**Situação das etapas.** CONCLUÍDA: arquivos verificados por hash. BLOQUEADA: um critério do
protocolo ou um recurso impede a etapa (o motivo é impresso). NÃO EXECUTADA / PENDENTE: ainda
não iniciada. Reexecutar retoma o mesmo `STUDY_ID`; trocar o ID inicia outro estudo.
Cópias antigas salvas no Colab mantêm o commit antigo: abra sempre pelo link do README.'''


def build(code_ref):
    if len(code_ref) != 40 or any(c not in '0123456789abcdef' for c in code_ref):
        raise ValueError('Use o SHA completo (40 caracteres) do commit de código')
    for key, name in NAMES.items():
        title, text = INTRO[key]
        cells = [cell('markdown', f'# {title}\n{text}'), cell('code', bootstrap(code_ref, PARAMS[key]))]
        cells += [cell('code', body) for body in STEPS[key]]
        cells.append(cell('markdown', FOOTER))
        for c in cells:
            if c['cell_type'] == 'code':
                compile(c['source'], name, 'exec')
        metadata = dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                        language_info=dict(name='python'), colab=dict(name=name, provenance=[]))
        if key != '00':
            metadata['accelerator'] = 'GPU'
            metadata['colab']['gpuType'] = 'T4'
        notebook = dict(cells=cells, metadata=metadata, nbformat=4, nbformat_minor=5)
        (ROOT / name).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Cinco notebooks v4 gerados para o commit', code_ref)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-ref', required=True)
    build(parser.parse_args().code_ref)
