"""Calcula os resumos SHA256 que o registro público do v4 deve conter.

Uso: python resumos_registro.py [--escrever]
Congela o plano numa pasta temporária (mesmo código do notebook 00), lê a
triagem DEV de triagem_dev/ e imprime uma tabela Markdown. Com --escrever,
substitui o trecho entre os marcadores da seção 8 de PROTOCOLO_v4.md.
O pacote gerado pelo notebook 00 (MANIFESTO_SHA256.json) continua sendo a
lista completa; esta tabela permite conferi-lo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows():
    os.environ.setdefault('COT_SOURCE_COMMIT', '0' * 40)
    from cotv4 import runtime
    protocol = runtime.protocol()
    out = [('Protocolo', 'protocolo_v4.json', sha(ROOT / 'protocolo_v4.json'))]
    with tempfile.TemporaryDirectory() as temp:
        plan = runtime.make_plan(Path(temp) / 'estudo')
    for mode, info in plan['datasets'].items():
        out.append((f'Dados {mode.upper()}', f'{info["rows"]} itens (train/dev/test)', info['sha256']))
    for name, digest in runtime.code_hashes().items():
        out.append(('Código', f'cotv4/{name}', digest))
    for model, revision in plan['model_revisions'].items():
        out.append(('Revisão do modelo', model, revision))
    gate = ROOT / 'triagem_dev' / 'manifest.json'
    if gate.exists():
        out.append(('Triagem DEV', 'triagem_dev/manifest.json', sha(gate)))
        for name, digest in json.loads(gate.read_text(encoding='utf-8'))['files'].items():
            out.append(('Triagem DEV', f'triagem_dev/{name}', digest))
    for name, digest in protocol['calibration_files'].items():
        out.append(('Calibração (excluída das partições)', f'calibracao/resultados/{name}', digest))
    return out


def markdown(entries):
    lines = ['| Elemento | Arquivo ou identificador | SHA256 / revisão |', '|---|---|---|']
    lines += [f'| {a} | `{b}` | `{c}` |' for a, b, c in entries]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--escrever', action='store_true')
    args = parser.parse_args()
    table = markdown(rows())
    print(table)
    if args.escrever:
        doc = ROOT / 'PROTOCOLO_v4.md'
        text = doc.read_text(encoding='utf-8')
        start, end = '<!-- hashes:inicio -->', '<!-- hashes:fim -->'
        head, rest = text.split(start, 1)
        doc.write_text(head + start + '\n' + table + end + rest.split(end, 1)[1], encoding='utf-8')


if __name__ == '__main__':
    main()
