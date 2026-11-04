"""Coleta confirmatória do v4 no Colab: os notebooks reais, sem alterar o código congelado.

Uso, numa VM do Colab com o Google Drive já montado em /content/drive:
    python coleta_v4.py COMMIT_DOS_NOTEBOOKS URL_PUBLICA_DO_REGISTRO

Executa, em ordem, os notebooks 00 (declara a URL do registro), 01 (treinos SFT,
repetido até os 120 treinos), 03 (avaliação TEST, repetida até as 120
avaliações) e 04 (relatório final). Só muda a primeira célula: a URL do
registro no 00 e MAX_JOBS_PER_SESSION = 12 (o máximo aceito pelo runtime) no 01
e no 03. TECHNICAL_TEST continua False e STORAGE_MODE continua "auto", o que
grava o estudo no Drive, na mesma pasta que os notebooks usam quando abertos à
mão. Antes de cada notebook confere que a GPU é a classe registrada
(NVIDIA A100-SXM4-80GB) e que o Drive está montado; se não estiver, para sem
treinar. Pode ser executado de novo a qualquer momento: o runtime retoma do
ponto em que parou e nunca repete um treino ou uma avaliação concluídos.
"""
import json, os, subprocess, sys, time
from pathlib import Path

COMMIT, URL = sys.argv[1], sys.argv[2]
REPO = 'https://github.com/Kelvin-Marques-Cyber/cot-obfuscation-h2a.git'
GPU_CLASS = 'NVIDIA A100-SXM4-80GB'
BASE = Path('/content/coleta_v4')
DRIVE = Path('/content/drive')
ROUNDS = 12  # 120 jobs / 12 por execução = 10; duas execuções de folga para retomadas


def log(*args):
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), *args, flush=True)


def checks():
    name = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                          capture_output=True, text=True).stdout.strip()
    if name != GPU_CLASS:
        sys.exit(f'GPU {name!r} diferente da classe registrada {GPU_CLASS!r}; nada foi executado.')
    if not os.path.ismount(DRIVE) and not (DRIVE / 'MyDrive').is_dir():
        sys.exit('Google Drive não montado em /content/drive; a coleta exige armazenamento persistente.')


def checkout():
    src = BASE / 'src'
    if not (src / '.git').is_dir():
        src.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'init', '-q', str(src)], check=True)
        subprocess.run(['git', '-C', str(src), 'remote', 'add', 'origin', REPO], check=True)
    subprocess.run(['git', '-C', str(src), 'fetch', '-q', '--depth', '1', 'origin', COMMIT], check=True)
    subprocess.run(['git', '-C', str(src), 'checkout', '-q', '--detach', COMMIT], check=True)
    assert subprocess.check_output(['git', '-C', str(src), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    return src / 'experimento_cientifico_v4'


def run_notebook(package, name, replacements, tag):
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.read(package / name, as_version=4)
    first = nb.cells[1]
    for old, new in replacements:
        if old not in first.source:
            sys.exit(f'{name}: parâmetro esperado não encontrado ({old}); nada foi alterado.')
        first.source = first.source.replace(old, new)
    if 'TECHNICAL_TEST = False' not in first.source or 'STORAGE_MODE = "auto"' not in first.source:
        sys.exit(f'{name}: a primeira célula não está no modo do estudo.')
    checks()
    started = time.time()
    client = NotebookClient(nb, timeout=None, kernel_name='python3', resources={'metadata': {'path': '/content'}})
    error = None
    try:
        client.execute()
    except Exception as exc:  # o notebook executado é salvo mesmo com erro
        error = f'{type(exc).__name__}: {str(exc)[-1500:]}'
    out = BASE / 'executados' / f'{tag}_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}.ipynb'
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out)
    text = '\n'.join(o.get('text', '') for c in nb.cells if c.cell_type == 'code' for o in c.get('outputs', []))
    log(name, 'erro' if error else 'ok', round(time.time() - started), 's')
    if error:
        sys.exit(f'{name} falhou: {error}')
    return text


def main():
    if not URL.startswith('https://osf.io/'):
        sys.exit('Informe a URL pública do registro no OSF (https://osf.io/...).')
    try:
        import nbclient, nbformat  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'nbclient', 'nbformat'], check=True)
    checks()
    package = checkout()
    text = run_notebook(package, '00_protocolo_dados_e_triagem.ipynb',
                        [('PUBLIC_REGISTRATION_URL = ""', f'PUBLIC_REGISTRATION_URL = {json.dumps(URL)}')], '00')
    if 'REGISTRO NÃO DECLARADO' in text or 'ETAPA BLOQUEADA' in text:
        sys.exit('O notebook 00 não liberou os treinos; veja o notebook executado em ' + str(BASE / 'executados'))
    for stage, name in (('01', '01_treinamento_sft.ipynb'), ('03', '03_avaliacao_test.ipynb')):
        marker = 'all_training_complete' if stage == '01' else 'all_evaluation_complete'
        for round_ in range(1, ROUNDS + 1):
            text = run_notebook(package, name, [('MAX_JOBS_PER_SESSION = 2', 'MAX_JOBS_PER_SESSION = 12')],
                                f'{stage}_r{round_:02d}')
            if 'ETAPA BLOQUEADA' in text:
                sys.exit(f'{name} bloqueado; veja o notebook executado em {BASE / "executados"}')
            if marker in text:
                log(name, 'concluído na rodada', round_)
                break
        else:
            sys.exit(f'{name} não concluiu em {ROUNDS} execuções; execute este script de novo para retomar.')
    run_notebook(package, '04_monitor_humanos_e_inferencia.ipynb', [], '04')
    log('COLETA_V4_CONCLUIDA')


if __name__ == '__main__':
    main()
