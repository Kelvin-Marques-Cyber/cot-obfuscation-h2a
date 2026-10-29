"""Executa no Colab T4 as calibrações exploratórias do v4 no commit indicado."""
import importlib.util, json, os, subprocess, sys, time
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
    spec = importlib.util.spec_from_file_location('cal_' + name, os.path.join(package, 'colab', name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
armazenamento, sessao = load('armazenamento'), load('sessao')
import hashlib
hashes = {f: hashlib.sha256(open(os.path.join(package, 'cotv4', f), 'rb').read()).hexdigest()
          for f in os.listdir(os.path.join(package, 'cotv4')) if f.endswith('.py')}
state = armazenamento.StorageState(os.path.join(BASE, 'storage'), 'local', True, [], None)
os.makedirs(state.path, exist_ok=True)
session = sessao.Session(src, state, COMMIT, hashes, on_colab=False)
python = session.gpu_environment()
env = dict(session.base_env)
out = os.path.join(BASE, 'resultados')
timings = {}
for spec in sys.argv[3:]:
    script, *extra = spec.split()
    started = time.time()
    sessao.run_logged([python, '-u', os.path.join(package, 'calibracao', script), '--device', 'cuda', '--out', out, *extra],
                      os.path.join(BASE, 'logs', spec.replace(' ', '_') + '.log'), env=env, cwd=package)
    timings[spec] = round(time.time() - started, 1)
json.dump(dict(commit=COMMIT, timings=timings), open(os.path.join(BASE, 'calibracao_concluida.json'), 'w'), indent=2)
print('CALIBRACAO_V4_CONCLUIDA', json.dumps(timings), flush=True)
