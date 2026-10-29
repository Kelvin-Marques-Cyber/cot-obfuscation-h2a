"""Triagem DEV do v4: execução única na Tesla T4, no commit congelado."""
import hashlib, importlib.util, json, os, subprocess, sys, time, tarfile
COMMIT = sys.argv[1]
URL = 'https://github.com/Kelvin-Marques-Cyber/cot-obfuscation-h2a.git'
BASE = '/content/v4gate'
src = os.path.join(BASE, 'src')
if os.path.exists(os.path.join(src, 'experimento_cientifico_v4', 'triagem_dev', 'manifest.json')):
    sys.exit('Triagem já executada neste diretório; não será repetida.')
os.makedirs(src, exist_ok=True)
subprocess.run(['git', 'init', '-q', src], check=True)
subprocess.run(['git', '-C', src, 'remote', 'add', 'origin', URL], check=False)
subprocess.run(['git', '-C', src, 'fetch', '-q', '--depth', '1', 'origin', COMMIT], check=True)
subprocess.run(['git', '-C', src, 'checkout', '-q', '--detach', COMMIT], check=True)
assert subprocess.check_output(['git', '-C', src, 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
package = os.path.join(src, 'experimento_cientifico_v4')
def load(name):
    spec = importlib.util.spec_from_file_location('gate_' + name, os.path.join(package, 'colab', name + '.py'))
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
started = time.time()
for stage in (['prepare'], ['run-gate', '--device', 'cuda']):
    sessao.run_logged([python, '-u', '-m', 'cotv4.runtime', *stage, '--storage', str(state.path),
                       '--result-json', os.path.join(BASE, stage[0] + '.json')],
                      os.path.join(BASE, 'logs', stage[0] + '.log'), env=session.base_env, cwd=package)
gate = os.path.join(package, 'triagem_dev')
with tarfile.open(os.path.join(BASE, 'triagem_dev.tgz'), 'w:gz') as tar:
    tar.add(gate, 'triagem_dev')
    tar.add(os.path.join(BASE, 'logs'), 'logs')
print('TRIAGEM_V4_CONCLUIDA', round(time.time() - started, 1), json.dumps(json.load(open(os.path.join(gate, 'manifest.json')))['summary']), flush=True)
