"""Monta a página de anotação a partir da amostra cega e da reserva.

Uso: python preparacao_v5/anotacao/gerar_pagina.py PASTA_DOS_RASTROS
Lê amostra_cega_final.jsonl e reserva_cega_final.jsonl (somente campos cegos:
enunciado, nota e saída do modelo) e grava pagina_anotacao.html ao lado deste
script. A chave das condições nunca entra na página.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELDS = ('annotation_id', 'order', 'question', 'hint', 'output')


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    return [{k: r[k] for k in FIELDS} for r in rows]


def main():
    source = Path(sys.argv[1])
    data = dict(amostra=load(source / 'amostra_cega_final.jsonl'), reserva=load(source / 'reserva_cega_final.jsonl'))
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    page = (HERE / 'pagina_modelo.html').read_text(encoding='utf-8').replace('/*ITENS*/', payload, 1)
    (HERE / 'pagina_anotacao.html').write_text(page, encoding='utf-8')
    digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()
    print(f"amostra={len(data['amostra'])} reserva={len(data['reserva'])} itens_sha256={digest}")


if __name__ == '__main__':
    main()
