"""Generate the five notebooks of the locked prospective scientific study."""
from pathlib import Path
import argparse
import hashlib
import json
import textwrap

ROOT=Path(__file__).resolve().parent


def cell(kind, body):
    body=textwrap.dedent(body).strip()+'\n'
    out=dict(cell_type=kind,metadata={},source=body,
             id=hashlib.sha256((kind+body).encode()).hexdigest()[:12])
    if kind=='code':
        out.update(execution_count=None,outputs=[])
    return out


def build(source_ref,setup_ref):
    for ref in (source_ref,setup_ref):
        if len(ref)!=40 or any(c not in '0123456789abcdef' for c in ref):
            raise ValueError('Use SHAs completos para o código científico e o suporte Colab.')
    script_sha=hashlib.sha256((ROOT/'scientific_runtime.py').read_bytes()).hexdigest()
    support_sha=hashlib.sha256((ROOT/'colab_support.py').read_bytes()).hexdigest()
    storage_sha=hashlib.sha256((ROOT/'colab_storage.py').read_bytes()).hexdigest()
    start=f'''
# Configuração operacional. Os tamanhos, sementes e critérios científicos são fixos.
from pathlib import Path
import hashlib, importlib.util, json, os, re, subprocess, sys
SOURCE_COMMIT = "{source_ref}"
SETUP_COMMIT = "{setup_ref}"
SCRIPT_SHA256 = "{script_sha}"
SUPPORT_SHA256 = "{support_sha}"
STORAGE_SHA256 = "{storage_sha}"
STUDY_ID = "estudo-principal-2026-v3-2" # @param {{type:"string"}}
PUBLIC_REGISTRATION_URL = "" # @param {{type:"string"}}
STORAGE_MODE = "auto" # @param ["auto", "local"]
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{{0,79}}", STUDY_ID):
    raise ValueError("STUDY_ID deve conter somente letras, números, _ e -.")
ON_COLAB = False
try:
    from google.colab import drive
except ImportError:
    pass
else:
    ON_COLAB = True
LOCAL = Path(os.environ.get("COT_LOCAL_ROOT",
    "/content/cot-prospective-v3" if ON_COLAB else str(Path.cwd()/"cot-prospective-v3")))
LOCAL.mkdir(parents=True, exist_ok=True)
def frozen_checkout(ref,folder,override_name):
    override=os.environ.get(override_name)
    repo=Path(override).resolve() if override else LOCAL/folder/ref
    if not override:
        repo.mkdir(parents=True,exist_ok=True)
        if not (repo/".git").is_dir():
            subprocess.run(["git","init","--quiet",str(repo)],check=True)
            subprocess.run(["git","-C",str(repo),"remote","add","origin",
                "https://github.com/Kelvin-Marques-Cyber/cot-obfuscation-h2a.git"],check=True)
        found=subprocess.run(["git","-C",str(repo),"cat-file","-e",ref+"^{{commit}}"],capture_output=True)
        if found.returncode:
            subprocess.run(["git","-C",str(repo),"fetch","--quiet","--depth","1","origin",ref],check=True)
        subprocess.run(["git","-C",str(repo),"checkout","--quiet","--detach",ref],check=True)
    actual=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    if actual!=ref:
        raise RuntimeError("O código aberto não corresponde ao SHA do notebook.")
    return repo
REPO=frozen_checkout(SOURCE_COMMIT,"source","COT_REPO_DIR")
SETUP_REPO=frozen_checkout(SETUP_COMMIT,"setup","COT_SETUP_REPO_DIR")
PACKAGE=REPO/"experimento_cientifico_v3"
SETUP_PACKAGE=SETUP_REPO/"experimento_cientifico_v3"
support=SETUP_PACKAGE/"colab_support.py"
if hashlib.sha256(support.read_bytes()).hexdigest()!=SUPPORT_SHA256:
    raise RuntimeError("O suporte do notebook diverge do hash publicado.")
storage_support=SETUP_PACKAGE/"colab_storage.py"
if hashlib.sha256(storage_support.read_bytes()).hexdigest()!=STORAGE_SHA256:
    raise RuntimeError("O suporte de armazenamento diverge do hash publicado.")
storage_spec=importlib.util.spec_from_file_location("cot_storage_"+SETUP_COMMIT,storage_support)
storage_module=importlib.util.module_from_spec(storage_spec)
storage_spec.loader.exec_module(storage_module)
state=storage_module.prepare_storage(LOCAL,STUDY_ID,on_colab=ON_COLAB,mode=STORAGE_MODE,
    storage_override=os.environ.get("COT_STORAGE_ROOT"),
    drive_module=drive if ON_COLAB else None,
    mountpoint=os.environ.get("COT_DRIVE_MOUNTPOINT","/content/drive"))
spec=importlib.util.spec_from_file_location("cot_scientific_support_"+SETUP_COMMIT,support)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
STORAGE=state.path
STUDY=module.ScientificSession(REPO,STORAGE,SOURCE_COMMIT,SCRIPT_SHA256,ON_COLAB,
    storage_persistent=state.persistent,storage_backend=state.backend)
execute=STUDY.execute
print("Código:",SOURCE_COMMIT,"| Estudo:",STUDY_ID)
print("Suporte Colab:",SETUP_COMMIT)
print("Arquivos:",STORAGE,"| Persistência verificada:",state.persistent)
'''
    def make(name,title,description,steps,gpu=False,pre_gpu=()):
        cells=[cell('markdown','# '+title+'\n\n'+description),cell('code',start)]
        cells.extend(cell(kind,body) for kind,body in pre_gpu)
        if gpu:
            cells.append(cell('code',"""GPU_SESSION = None
if globals().get("DEV_BLOQUEADO",False):
    print("DEV já reprovado: sem carregar GPU novamente.")
else:
    GPU_SESSION=STUDY.gpu()
    if GPU_SESSION is not None:
        print("Ambiente GPU preparado.")
"""))
        cells.extend(cell(kind,body) for kind,body in steps)
        cells.append(cell('markdown',"""Com o Drive conectado, a reexecução retoma o mesmo estudo.
Se a montagem falhar, o 00 prepara uma cópia temporária para download;
os treinos aguardam armazenamento persistente. O modo local não salva no Drive.
Não altere o STUDY_ID depois de iniciar. A versão v3.2 usa um ID novo para
preservar estudos antigos. A execução confirmatória depende de pré-registro
público anterior aos treinos e de avaliação humana; essas etapas não são
substituídas por resultados de software."""))
        metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                      language_info=dict(name='python'),
                      colab=dict(name=name,provenance=[]))
        if gpu:
            metadata['accelerator']='GPU'
            metadata['colab']['gpuType']='T4'
        notebook=dict(cells=cells,metadata=metadata,nbformat=4,nbformat_minor=5)
        for c in cells:
            if c['cell_type']=='code':
                compile(c['source'],name,'exec')
        (ROOT/name).write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')

    dev_audit = "# Auditoria de DEV reaproveitando os arquivos existentes (somente leitura).\nimport json, hashlib, importlib.util\nfrom pathlib import Path\n\ndef verificar_dev_salvo():\n    completo = True\n    valores = {}\n    for modo, niveis in ((\"sft\", (\"A\", \"C\")), (\"grpo\", (\"low\", \"high\"))):\n        resumo = STORAGE / \"analysis\" / modo / \"dev_base_accuracy.json\"\n        if not resumo.exists():\n            print(\"DEV ainda não produzido:\", modo)\n            completo = False\n            continue\n        dados = json.loads(resumo.read_text(encoding=\"utf-8\"))\n        for nivel in niveis:\n            dado = dados.get(nivel)\n            if dado is None:\n                completo = False\n                continue\n            arquivo = Path(dado[\"file\"]).resolve()\n            if not arquivo.is_relative_to(STORAGE.resolve()) or not arquivo.is_file():\n                raise RuntimeError(\"Arquivo DEV ausente ou fora do estudo: \" + str(arquivo))\n            conteudo = arquivo.read_bytes()\n            if hashlib.sha256(conteudo).hexdigest() != dado[\"sha256\"]:\n                raise RuntimeError(\"SHA256 divergente no DEV; não sobrescreva o arquivo: \" + str(arquivo))\n            amostras = [json.loads(l) for l in conteudo.decode(\"utf-8\").splitlines() if l.strip()]\n            acertos = sum(bool(item[\"accuracy\"]) for item in amostras)\n            if (len(amostras) != dado[\"n\"] or\n                    abs(acertos / len(amostras) - dado[\"accuracy\"]) > 1e-12):\n                raise RuntimeError(\"Resumo DEV não corresponde aos registros: \" + str(arquivo))\n            valores[modo + \"/\" + nivel] = dado[\"accuracy\"]\n            print(f\"DEV {modo.upper()} {nivel}: {acertos}/{len(amostras)} \"\n                  f\"({dado['accuracy']:.1%}); exigido 65%\")\n            erros = [x for x in amostras if not x[\"accuracy\"]]\n            for item in erros[:2]:\n                print(\"  Exemplo incorreto:\", repr(item[\"generation\"][:240]),\n                      \"| gabarito:\", item[\"gold\"])\n    if not completo or len(valores) != 4:\n        return False, False\n    spec = importlib.util.spec_from_file_location(\"cot_dev_gate_readonly\",\n                                                   PACKAGE / \"scientific_runtime.py\")\n    modulo = importlib.util.module_from_spec(spec)\n    spec.loader.exec_module(modulo)\n    gate = modulo.baseline_gate(STORAGE)\n    if not gate[\"passed\"]:\n        print(\"\\nTRIAGEM DEV REPROVADA:\", \", \".join(gate[\"reasons\"]))\n        print(\"Não há treinos confirmatórios autorizados neste estudo congelado.\")\n    return True, gate[\"passed\"]\n\nDEV_COMPLETO, DEV_APROVADO = verificar_dev_salvo()\nDEV_BLOQUEADO = DEV_COMPLETO and not DEV_APROVADO\n"

    make('00_protocolo_e_datasets.ipynb','00 · Protocolo e datasets congelados',
         """**CPU · primeira etapa obrigatória.** Use **Ambiente de execução → Executar tudo**.
O Drive tenta conectar duas vezes. Se falhar, a preparação continua localmente
e o ZIP pode ser baixado. Para pular a conexão, selecione STORAGE_MODE="local".
Gere o pacote final com o Drive conectado antes de registrar e treinar.

Prepara os dados disjuntos e congela código, critérios, 20 sementes, quatro
revisões de modelos e hashes. O desenho contém 120 treinos SFT e 120 GRPO.
Baixa um ZIP para publicar no registro externo antes da coleta.
Após publicar, preencha PUBLIC_REGISTRATION_URL na primeira célula e reexecute.
Uma URL informada é uma declaração; sua cronologia precisa ser auditável.""",[
        ('code',"""prepared=execute('prepare')
models=execute('models')
print('Itens TEST por nível:',{k:v['test_per_level'] for k,v in prepared['datasets'].items()})
for filename in ('locked_protocol.json','model_revisions.json','frozen_datasets.json'):
    path=STORAGE/'protocol'/filename
    print(filename,hashlib.sha256(path.read_bytes()).hexdigest())
"""),
        ('code',"""STUDY.registration(PUBLIC_REGISTRATION_URL)
STUDY.export(registration=True)
print('Próxima etapa: 01, depois de publicar o pré-registro.')
""")
    ])
    make('01_treinamento_sft_replicado.ipynb','01 · Treinamento SFT replicado',
         """**GPU T4 ou superior.** Execute após o notebook 00 e o pré-registro.
Treina três condições (explicit/direct/generic), duas dificuldades (A/C) e
20 sementes: 120 treinos de 120 passos. O limite abaixo controla apenas
quantos jobs cabem na sessão. Dados e critérios permanecem congelados.
A triagem DEV exige 65% por nível nos dois modelos-base; falhar bloqueia o estudo.
**Diagnóstico DEV sem nova inferência:** se a triagem já falhou, o notebook mostra
os dados salvos e não inicializa a GPU. O limite de 65% não é alterado.""",[
        ('code',"MAX_JOBS_PER_SESSION = 2 # @param {type:\"integer\"}\nif GPU_SESSION is not None:\n    execute('prepare')\n    execute('models',gpu=True)\nelse:\n    print(\"Preparação GPU ignorada: DEV reprovado ou armazenamento indisponível.\")\n"),
        ('code',"if DEV_BLOQUEADO:\n    print(\"Treinos SFT não iniciados: competência DEV inferior ao mínimo do pré-registro.\")\nelif GPU_SESSION is not None:\n    for modo in (\"sft\", \"grpo\"):\n        if not (STORAGE/\"analysis\"/modo/\"dev_base_accuracy.json\").exists():\n            execute('dev',mode=modo,gpu=True)\n    DEV_COMPLETO, DEV_APROVADO = verificar_dev_salvo()\n    if DEV_COMPLETO and DEV_APROVADO:\n        progress = execute('train',mode='sft',max_jobs=MAX_JOBS_PER_SESSION,gpu=True)\n        print(progress['completed'],'/',progress['total'],'treinos SFT concluídos.')\n    else:\n        print(\"Treinos SFT pausados: triagem DEV não aprovada. Consulte as amostras acima.\")\nelse:\n    print(\"Treinos não iniciados: Drive/GPU indisponível.\")\nSTUDY.export(download=False)\n")
    ],gpu=True,pre_gpu=[('code',dev_audit)])
    make('02_treinamento_grpo_replicado.ipynb','02 · Treinamento GRPO replicado',
         """**GPU T4 ou superior.** Use o mesmo STUDY_ID dos notebooks anteriores.
Treina none/cot_monitor/summary_monitor em low/high, com 20 sementes:
120 treinos de 120 passos. O monitor de recompensa é distinto do monitor
independente. Reexecutar retoma checkpoints e mantém jobs concluídos.
**Diagnóstico DEV sem nova inferência:** se a triagem já falhou, o notebook mostra
os dados salvos e não inicializa a GPU. O limite de 65% não é alterado.""",[
        ('code',"MAX_JOBS_PER_SESSION = 1 # @param {type:\"integer\"}\nif DEV_BLOQUEADO:\n    print(\"Treinos GRPO não iniciados: competência DEV inferior ao mínimo do pré-registro.\")\nelif GPU_SESSION is not None:\n    execute('prepare')\n    execute('models',gpu=True)\n    for modo in (\"sft\", \"grpo\"):\n        if not (STORAGE/\"analysis\"/modo/\"dev_base_accuracy.json\").exists():\n            execute('dev',mode=modo,gpu=True)\n    DEV_COMPLETO, DEV_APROVADO = verificar_dev_salvo()\n    if DEV_COMPLETO and DEV_APROVADO:\n        progress = execute('train',mode='grpo',max_jobs=MAX_JOBS_PER_SESSION,gpu=True)\n        print(progress['completed'],'/',progress['total'],'treinos GRPO concluídos.')\n    else:\n        print(\"Treinos GRPO pausados: triagem DEV não aprovada. Consulte as amostras acima.\")\nelse:\n    print(\"Treinos não iniciados: Drive/GPU indisponível.\")\nSTUDY.export(download=False)\n")
    ],gpu=True,pre_gpu=[('code',dev_audit)])
    make('03_competencia_e_teste_cego.ipynb','03 · Competência e teste final pareado',
         """**GPU · após os 240 treinamentos.** Avalia somente TEST/full de cada
adaptador, com orçamento comum de 256 tokens. São 26.400 gerações SFT e
28.800 GRPO. Execuções incompletas são retomadas. O teste permanece bloqueado
enquanto qualquer um dos dois grupos de 120 treinos estiver incompleto.""",[
        ('code',"""MAX_EVALUATIONS_PER_SESSION = 2 # @param {type:"integer"}
execute('prepare')
execute('models',gpu=True)
status=execute('status')
print('Treinos:',status['training'],'| Avaliações:',status['evaluation'])
if status['all_training_complete']:
    execute('dev',mode='sft',gpu=True)
    execute('dev',mode='grpo',gpu=True)
    for mode in ('sft','grpo'):
        execute('evaluate',mode=mode,max_jobs=MAX_EVALUATIONS_PER_SESSION,gpu=True)
else:
    print('Conclua os notebooks 01 e 02 antes de abrir TEST.')
STUDY.export(download=False)
""")
    ],gpu=True)
    make('04_monitor_independente_e_inferencia.ipynb','04 · Monitor independente e inferência',
         """**GPU · após completar as avaliações do notebook 03.**
Aplica o monitor independente congelado no notebook 00, exporta 360 itens
cegos para dois avaliadores e testa os contrastes por semente com correção
de Holm. Reexecute para avançar o monitor.

Após a anotação humana, salve em human/ no diretório do estudo:
completed_annotations.csv (annotation_id,rater,label);
adjudication.csv (annotation_id,label), se houver divergências;
independent_monitor_controls.csv (question,trace,label,source,split),
com pelo menos 100 positivos e 100 negativos, source=human e split=heldout.
As respostas humanas devem ser preenchidas por pessoas. O relatório informa
as pendências quando esses arquivos ainda não estiverem disponíveis.""",[
        ('code',"""MAX_MONITOR_RUNS_PER_SESSION = 2 # @param {type:"integer"}
execute('prepare')
execute('models',gpu=True)
status=execute('status')
if status['all_evaluation_complete']:
    execute('monitor',mode='grpo',max_jobs=MAX_MONITOR_RUNS_PER_SESSION,gpu=True)
    execute('blind',mode='grpo',gpu=True)
    if (STORAGE/'human/independent_monitor_controls.csv').is_file():
        execute('controls',gpu=True)
    else:
        print('Aguardando os controles rotulados por humanos em human/.')
else:
    print('Conclua todas as avaliações no notebook 03.')
"""),
        ('code',"""report=execute('report')
print('Conclusão:',report['verdict'])
print('Pendências:',report['blocked_by'])
STUDY.export()
""")
    ],gpu=True)
    print('Cinco notebooks gerados e células compiladas. Código fixado em',source_ref)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-ref',required=True)
    parser.add_argument('--setup-ref',required=True)
    args=parser.parse_args()
    build(args.source_ref,args.setup_ref)

