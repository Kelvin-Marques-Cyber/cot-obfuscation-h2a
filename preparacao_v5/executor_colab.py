"""Executa no Colab (A100) os scripts de preparação do v5 no commit indicado.

Uso: python executor_colab.py COMMIT PASTA "script [opções]"...
Cada script recebe --device cuda e --out PASTA/<nome do script>. O ambiente de
GPU é o mesmo dos notebooks do v4 (colab/sessao.py, requirements-gpu.txt).
"""
import hashlib, importlib.util, json, os, subprocess, sys, tarfile, time

COMMIT = sys.argv[1]
URL = 'https://github.com/Kelvin-Marques-Cyber/cot-obfuscation-h2a.git'
BASE = sys.argv[2]
src = os.path.join(BASE, 'src')
if not os.path.isdir(os.path.join(src, '.git')):
    os.makedirs(src, exist_ok=True)
    subprocess.run(['git', 'init', '-q', src], check=True)
    subprocess.run(['git', '-C', src, 'remote', 'add', 'origin', URL], check=True)
subprocess.run(['git', '-C', src, 'fetch', '-q', '--depth', '1', 'origin', COMMIT], check=True)
subprocess.run(['git', '-C', src, 'checkout', '-q', '--detach', COMMIT], check=True)
assert subprocess.check_output(['git', '-C', src, 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
package = os.path.join(src, 'experimento_cientifico_v4')


def load(name):
    spec = importlib.util.spec_from_file_location('v5_' + name, os.path.join(package, 'colab', name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


armazenamento, sessao = load('armazenamento'), load('sessao')
hashes = {f: hashlib.sha256(open(os.path.join(package, 'cotv4', f), 'rb').read()).hexdigest()
          for f in os.listdir(os.path.join(package, 'cotv4')) if f.endswith('.py')}
state = armazenamento.StorageState(os.path.join(BASE, 'storage'), 'local', True, [], None)
os.makedirs(state.path, exist_ok=True)
session = sessao.Session(src, state, COMMIT, hashes, on_colab=False)
python = session.gpu_environment()
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip()
report = dict(commit=COMMIT, gpu=gpu, timings={})
for spec in sys.argv[3:]:
    script, *extra = spec.split()
    out = os.path.join(BASE, 'resultados', os.path.splitext(script)[0])
    started = time.time()
    sessao.run_logged([python, '-u', os.path.join(src, 'preparacao_v5', script), '--device', 'cuda', '--out', out,
                       *extra], os.path.join(BASE, 'logs', os.path.splitext(script)[0] + '.log'),
                      env=session.base_env, cwd=package)
    report['timings'][spec] = round(time.time() - started, 1)
    json.dump(report, open(os.path.join(BASE, 'relatorio_execucao.json'), 'w'), indent=2)
    with tarfile.open(os.path.join(BASE, os.path.splitext(script)[0] + '.tgz'), 'w:gz') as tar:
        for root, _, files in os.walk(out):
            for name in files:
                if not name.endswith('.sqlite'):
                    path = os.path.join(root, name)
                    tar.add(path, os.path.relpath(path, BASE))
        tar.add(os.path.join(BASE, 'logs', os.path.splitext(script)[0] + '.log'),
                os.path.join('logs', os.path.splitext(script)[0] + '.log'))
        tar.add(os.path.join(BASE, 'relatorio_execucao.json'), 'relatorio_execucao.json')
    print('ETAPA_CONCLUIDA', spec, report['timings'][spec], flush=True)
print('PREPARACAO_V5_CONCLUIDA', json.dumps(report), flush=True)
