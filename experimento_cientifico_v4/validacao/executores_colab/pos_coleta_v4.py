"""Pós-coleta do v4 no Colab: análises secundárias (S1–S4) e S5, depois de COLETA_V4_CONCLUIDA.

Uso, na mesma VM de coleta_v4.py:
    python pos_coleta_v4.py PASTA_DO_ESTUDO CAMINHO_DE_analise_desvio_01.py

Usa o código congelado que coleta_v4.py baixou em /content/coleta_v4/src (o mesmo commit dos
notebooks) para executar analises_secundarias.py, registrado com o estudo, e
analise_desvio_01.py, declarado em DESVIO_01.md (copiado para o mesmo diretório, pois importa
o pacote cotv4 dali). Nenhum dos dois altera o estudo nem o veredito de final_report. Copia para
/content/coleta_v4/resultados o veredito, o ZIP de resultados exportado pelo notebook 04 e as
duas análises, com um manifesto SHA256.
"""
import hashlib, json, shutil, subprocess, sys, time
from pathlib import Path

STUDY, S5 = Path(sys.argv[1]), Path(sys.argv[2])
SRC = Path('/content/coleta_v4/src/experimento_cientifico_v4')
OUT = Path('/content/coleta_v4/resultados')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


verdict = STUDY / 'analysis' / 'final_verdict.json'
export = STUDY / 'downloads' / 'resultados_v4.zip'
for required in (verdict, export):
    if not required.is_file():
        sys.exit(f'Ausente: {required}; a coleta não terminou.')
OUT.mkdir(parents=True, exist_ok=True)
shutil.copy2(S5, SRC / 'analise_desvio_01.py')
commit = subprocess.check_output(['git', '-C', str(SRC), 'rev-parse', 'HEAD'], text=True).strip()
for script, name in (('analises_secundarias.py', 'analises_secundarias.json'),
                     ('analise_desvio_01.py', 'analise_desvio_01.json')):
    subprocess.run([sys.executable, script, '--storage', str(STUDY), '--saida', str(OUT / name)],
                   cwd=SRC, check=True, stdout=subprocess.DEVNULL)
shutil.copy2(verdict, OUT / 'final_verdict.json')
shutil.copy2(export, OUT / 'resultados_v4.zip')
manifest = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), code_commit=commit,
                analise_desvio_01_sha256=sha256(S5),
                files={p.name: sha256(p) for p in sorted(OUT.iterdir()) if p.name != 'MANIFESTO_SHA256.json'})
(OUT / 'MANIFESTO_SHA256.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest, indent=2))
print('POS_COLETA_V4_CONCLUIDA')
