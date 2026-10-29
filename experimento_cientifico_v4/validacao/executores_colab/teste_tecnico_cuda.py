"""Teste técnico CUDA do v4 no Colab: cinco notebooks reais e interrupções forçadas.

Não produz evidência científica: TECHNICAL_TEST=True usa 1 semente, 2 passos por
treino e 4 itens TEST por avaliação, em pasta separada do estudo.
"""
import hashlib, json, os, signal, shutil, subprocess, sys, tarfile, time
from pathlib import Path

COMMIT = sys.argv[1]
URL = 'https://github.com/Kelvin-Marques-Cyber/cot-obfuscation-h2a.git'
BASE = Path('/content/v4tec')
NOTEBOOKS = ['00_protocolo_dados_e_triagem.ipynb', '01_treinamento_sft.ipynb', '02_treinamento_grpo.ipynb',
             '03_avaliacao_test.ipynb', '04_monitor_humanos_e_inferencia.ipynb']
report = dict(commit=COMMIT, started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), notebooks={})


def log(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)


src = BASE / 'src'
if not (src / '.git').is_dir():
    src.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init', '-q', str(src)], check=True)
    subprocess.run(['git', '-C', str(src), 'remote', 'add', 'origin', URL], check=True)
subprocess.run(['git', '-C', str(src), 'fetch', '-q', '--depth', '1', 'origin', COMMIT], check=True)
subprocess.run(['git', '-C', str(src), 'checkout', '-q', '--detach', COMMIT], check=True)
package = src / 'experimento_cientifico_v4'
try:
    import nbformat
    from nbclient import NotebookClient
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'nbclient', 'nbformat'], check=True)
    import nbformat
    from nbclient import NotebookClient

executed = BASE / 'executados'
executed.mkdir(parents=True, exist_ok=True)
for name in NOTEBOOKS:
    nb = nbformat.read(package / name, as_version=4)
    first = nb.cells[1]
    first.source = first.source.replace('TECHNICAL_TEST = False', 'TECHNICAL_TEST = True')
    if name != NOTEBOOKS[0]:
        first.source = first.source.replace('STORAGE_MODE = "auto"', 'STORAGE_MODE = "local"')
    if name in ('01_treinamento_sft.ipynb', '02_treinamento_grpo.ipynb', '03_avaliacao_test.ipynb'):
        first.source = first.source.replace('MAX_JOBS_PER_SESSION = 2', 'MAX_JOBS_PER_SESSION = 6')
        first.source = first.source.replace('MAX_JOBS_PER_SESSION = 1', 'MAX_JOBS_PER_SESSION = 6')
    if name == NOTEBOOKS[4]:
        first.source = first.source.replace('MAX_JOBS_PER_SESSION = 4', 'MAX_JOBS_PER_SESSION = 6')
    started = time.time()
    log('Executando', name)
    client = NotebookClient(nb, timeout=None, kernel_name='python3', resources={'metadata': {'path': '/content'}})
    error = None
    try:
        client.execute()
    except Exception as exc:  # o notebook executado é salvo mesmo com erro
        error = f'{type(exc).__name__}: {str(exc)[-2000:]}'
    nbformat.write(nb, executed / name)
    outputs = '\n'.join(o.get('text', '') for c in nb.cells if c.cell_type == 'code' for o in c.get('outputs', []))
    report['notebooks'][name] = dict(seconds=round(time.time() - started, 1), error=error,
                                     blocked=outputs.count('ETAPA BLOQUEADA'), tail=outputs[-1500:])
    log(name, 'erro' if error else 'ok', report['notebooks'][name]['seconds'], 's')
    json.dump(report, open(BASE / 'relatorio_tecnico.json', 'w'), indent=2, ensure_ascii=False)

# ------------------------------------------------------------ interrupções na GPU
study = Path('/content/cot-v4/studies/estudo-prospectivo-2026-v4-teste-tecnico')
envs = sorted(Path('/content/cot-v4-envs').glob('gpu-*/bin/python'))
python = str(envs[0]) if envs else sys.executable
env = dict(os.environ, PYTHONPATH=str(package), COT_TECHNICAL_TEST='1', COT_SOURCE_COMMIT=COMMIT,
           PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
interrupt = Path('/content/cot-v4/studies/estudo-interrupcao-teste-tecnico')
shutil.rmtree(interrupt, ignore_errors=True)
subprocess.run([python, '-m', 'cotv4.runtime', 'prepare', '--storage', str(interrupt)], env=env, cwd=package,
               check=True, capture_output=True)
log('Interrupção de treino: iniciando SFT e matando após checkpoint-1')
proc = subprocess.Popen([python, '-u', '-m', 'cotv4.runtime', 'train', '--storage', str(interrupt), '--mode', 'sft',
                         '--max-jobs', '1', '--device', 'cuda'], env=env, cwd=package,
                        stdout=open(BASE / 'interrupcao_treino_1.log', 'w'), stderr=subprocess.STDOUT,
                        start_new_session=True)
deadline = time.time() + 1800
killed_with = []
while time.time() < deadline and proc.poll() is None:
    if list(interrupt.glob('runs/sft/*/checkpoints/checkpoint-1/trainer_state.json')):
        os.killpg(proc.pid, signal.SIGKILL)
        killed_with = sorted(p.parent.name for p in interrupt.glob('runs/sft/*/checkpoints/*/trainer_state.json'))
        break
    time.sleep(0.2)
proc.wait()
run_dir = next(interrupt.glob('runs/sft/*'))
status_after_kill = json.loads((run_dir / 'status.json').read_text())
checkpoints_after_kill = sorted(p.name for p in (run_dir / 'checkpoints').iterdir())
resume = subprocess.run([python, '-u', '-m', 'cotv4.runtime', 'train', '--storage', str(interrupt), '--mode', 'sft',
                         '--max-jobs', '1', '--device', 'cuda'], env=env, cwd=package, capture_output=True, text=True)
(BASE / 'interrupcao_treino_2.log').write_text(resume.stdout + resume.stderr)
status_final = json.loads((run_dir / 'status.json').read_text())
report['interrupcao_treino'] = dict(killed_returncode=proc.returncode, checkpoints_at_kill=killed_with,
                                    status_after_kill=status_after_kill.get('status'),
                                    checkpoints_after_kill=checkpoints_after_kill, resume_returncode=resume.returncode,
                                    status_final=status_final)
log('Treino retomado:', status_final)

log('Interrupção de avaliação: avaliando um adaptador e matando após o primeiro item')
eval_code = ('import sys; from cotv4.avaliacao import evaluate_run; '
             f'print(evaluate_run(r"{run_dir}", r"{interrupt}/data/sft.jsonl", "test", 256, device="cuda", '
             'precision="fp16", limit=4))')
proc = subprocess.Popen([python, '-u', '-c', eval_code], env=env, cwd=package,
                        stdout=open(BASE / 'interrupcao_avaliacao_1.log', 'w'), stderr=subprocess.STDOUT,
                        start_new_session=True)
deadline = time.time() + 900
target = None
while time.time() < deadline and proc.poll() is None:
    files = list(run_dir.glob('evaluation_test_*.jsonl'))
    if files and files[0].read_text().count('\n') >= 1:
        target = files[0]
        os.killpg(proc.pid, signal.SIGKILL)
        break
    time.sleep(0.5)
proc.wait()
if target is None:
    raise RuntimeError('Avaliação não gravou nenhuma linha antes do prazo; ver interrupcao_avaliacao_1.log')
lines_after_kill = target.read_text().splitlines()
with open(target, 'a') as handle:
    handle.write('{"item_id": "linha parcial simulada')  # gravação interrompida no meio da linha
second = subprocess.run([python, '-u', '-c', eval_code], env=env, cwd=package, capture_output=True, text=True)
(BASE / 'interrupcao_avaliacao_2.log').write_text(second.stdout + second.stderr)
rows = [json.loads(l) for l in target.read_text().splitlines()]
strip = lambda r: {k: v for k, v in r.items() if k != 'seconds'}
report['interrupcao_avaliacao'] = dict(lines_after_kill=len(lines_after_kill), resume_returncode=second.returncode,
                                       final_lines=len(rows), unique_items=len({r['item_id'] for r in rows}),
                                       first_line_identical=bool(lines_after_kill) and
                                       strip(json.loads(lines_after_kill[0])) == strip(rows[0]))

# ------------------------------------------------------------- coleta de artefatos
info = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                      capture_output=True, text=True).stdout.strip()
report['gpu'] = info
report['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
json.dump(report, open(BASE / 'relatorio_tecnico.json', 'w'), indent=2, ensure_ascii=False)
with tarfile.open(BASE / 'artefatos_tecnicos.tgz', 'w:gz') as tar:
    tar.add(BASE / 'relatorio_tecnico.json', 'relatorio_tecnico.json')
    tar.add(executed, 'notebooks_executados')
    for path in sorted(BASE.glob('interrupcao_*.log')):
        tar.add(path, path.name)
    for root in (study, interrupt):
        for path in sorted(root.rglob('*')):
            parts = set(path.relative_to(root).parts)
            if path.is_file() and not parts & {'checkpoints'} and path.suffix in (
                    '.json', '.jsonl', '.csv', '.log', '.txt', '.zip') and 'adapter' not in parts:
                tar.add(path, f'{root.name}/{path.relative_to(root)}')
        for adapter in root.glob('runs/*/*/adapter/adapter_config.json'):
            tar.add(adapter, f'{root.name}/{adapter.relative_to(root)}')
log('TESTE_TECNICO_V4_CONCLUIDO', json.dumps({k: v for k, v in report.items() if k.startswith('interrupcao')}))
