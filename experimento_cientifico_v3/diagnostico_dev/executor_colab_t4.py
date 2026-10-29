"""Reprodução diagnóstica da triagem DEV v3.2 no Colab T4, pelo caminho original dos notebooks.

Usa os commits fixados nos notebooks v3.2 (SOURCE fde856f, SETUP 4a2a133), o
ambiente isolado 'quick' e as etapas prepare/models/dev em uma pasta
temporária de diagnóstico. Não altera nenhum estudo registrado.
"""
import hashlib, importlib.util, json, os, subprocess, sys, time
SOURCE = 'fde856f895c841274b93b34f4675642508af102f'
SETUP = '4a2a133aff6638c910661b01821eeb43b72b4db4'
URL = 'https://github.com/Kelvin-Marques-Cyber/cot-obfuscation-h2a.git'
BASE = '/content/diag-v3'
started = time.time()

def checkout(ref, folder):
    path = os.path.join(BASE, folder)
    if not os.path.isdir(os.path.join(path, '.git')):
        os.makedirs(path, exist_ok=True)
        subprocess.run(['git', 'init', '-q', path], check=True)
        subprocess.run(['git', '-C', path, 'remote', 'add', 'origin', URL], check=True)
    subprocess.run(['git', '-C', path, 'fetch', '-q', '--depth', '1', 'origin', ref], check=True)
    subprocess.run(['git', '-C', path, 'checkout', '-q', '--detach', ref], check=True)
    head = subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD'], text=True).strip()
    assert head == ref, head
    return path

repo = checkout(SOURCE, 'source')
setup = checkout(SETUP, 'setup')
script = os.path.join(repo, 'experimento_cientifico_v3', 'scientific_runtime.py')
script_sha = hashlib.sha256(open(script, 'rb').read()).hexdigest()
assert script_sha == '0ccc8cd66c429c116fe8a3ddeaee373d9dd72ec2e176d9dc10bc032da6c60f8b', script_sha
spec = importlib.util.spec_from_file_location('v3_support_' + SETUP, os.path.join(setup, 'experimento_cientifico_v3', 'colab_support.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
storage = os.path.join(BASE, 'studies', 'diagnostico-triagem-v3-2')
study = module.ScientificSession(repo, storage, SOURCE, script_sha, on_colab=False, storage_persistent=True)
results = {}
results['prepare'] = study.execute('prepare')
results['models'] = study.execute('models', gpu=True)
for mode in ('sft', 'grpo'):
    results['dev_' + mode] = study.execute('dev', mode=mode, gpu=True)
results['seconds'] = round(time.time() - started, 1)
with open(os.path.join(BASE, 'resultado_diagnostico_v3.json'), 'w') as handle:
    json.dump(results, handle, indent=2)
print('DIAGNOSTICO_V3_CONCLUIDO', json.dumps({k: v for k, v in results.items() if k.startswith('dev_')}), flush=True)
