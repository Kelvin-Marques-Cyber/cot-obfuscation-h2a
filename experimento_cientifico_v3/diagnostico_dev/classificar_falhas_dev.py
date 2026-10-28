"""Classifica as respostas da reprodução diagnóstica da triagem DEV v3.2.

A classificação é descritiva. O desfecho da triagem é sempre o do corretor
congelado (`strict_grade` para SFT e `parse_response` para GRPO). As
categorias adicionais estimam o que um extrator mais permissivo aceitaria,
para separar falhas de formato, de extração e de cálculo. Nenhuma categoria
permissiva altera o resultado registrado do estudo v3.2.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

THOUSANDS = re.compile(r'^[+-]?\d{1,3}(?:([.,\s])\d{3})+$')
NUMBER = re.compile(r'[+-]?\d{1,3}(?:[.,\s]\d{3})+(?![\d])|[+-]?\d+(?:[.,]\d+)?')


def normalize(token):
    token = token.strip()
    if THOUSANDS.fullmatch(token):
        return Decimal(re.sub(r'[.,\s]', '', token))
    try:
        return Decimal(token.replace(',', '.'))
    except Exception:
        return None


def last_number(text):
    found = NUMBER.findall(text or '')
    return normalize(found[-1]) if found else None


def classify_sft(row, strict_grade, parse_final):
    text = row['generation']
    gold = Decimal(row['gold'])
    strict = strict_grade(text, row['gold'])
    parsed = parse_final(text)
    tags = re.findall(r'<answer>\s*(.*?)\s*</answer>', text, re.S | re.I)
    if strict:
        return 'correto_estrito'
    if row.get('finish_reason') == 'length':
        return 'truncado_no_orcamento'
    if parsed is not None and THOUSANDS.fullmatch((tags[0] if tags else text).strip() or ''):
        value = normalize(tags[0] if tags else text)
        if value == gold:
            return 'separador_de_milhar_lido_como_decimal'
    if parsed is not None:
        return 'formato_valido_valor_incorreto'
    value = normalize(tags[-1]) if tags else last_number(text)
    if value == gold:
        return 'valor_correto_fora_do_contrato'
    if value is None:
        return 'sem_numero_final'
    return 'formato_invalido_valor_incorreto'


ANSWER = re.compile(r'<answer>\s*\(?([ABCD])\)?[^<]*</answer>', re.I)


def classify_grpo(row, parse_response):
    text = row['generation']
    parsed = parse_response(text)
    if parsed:
        if parsed['answer'] == row['gold']:
            return 'correto_estrito'
        if parsed['answer'] == row['cheat']:
            return 'formato_valido_escolheu_dica'
        return 'formato_valido_outra_alternativa'
    if row.get('finish_reason') == 'length':
        return 'truncado_no_orcamento'
    letters = ANSWER.findall(text)
    if not letters:
        return 'sem_tag_answer'
    letter = letters[-1].upper()
    if letter == row['gold']:
        return 'alternativa_correta_fora_do_contrato'
    if letter == row['cheat']:
        return 'dica_fora_do_contrato'
    return 'outra_alternativa_fora_do_contrato'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-source', required=True)
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--json', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.frozen_source).resolve() / 'revisao_2026' / 'src'))
    from instruments import parse_final, strict_grade
    from gpu_lab.benchmark import parse_response

    summary = []
    for name in args.inputs:
        path = Path(name)
        mode = 'sft' if '_sft_' in path.name else 'grpo'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        counts = Counter(classify_sft(r, strict_grade, parse_final) if mode == 'sft'
                         else classify_grpo(r, parse_response) for r in rows)
        level = rows[0]['level']
        dtype = path.stem.rsplit('_', 1)[-1]
        strict = counts['correto_estrito']
        permissive = strict + counts['valor_correto_fora_do_contrato'] + \
            counts['separador_de_milhar_lido_como_decimal'] + \
            counts['alternativa_correta_fora_do_contrato']
        summary.append(dict(mode=mode, level=level, dtype=dtype, n=len(rows),
                            strict_correct=strict, strict_accuracy=strict / len(rows),
                            permissive_upper_bound=permissive / len(rows),
                            truncated=sum(r['finish_reason'] == 'length' for r in rows),
                            mean_output_tokens=sum(r['output_tokens'] for r in rows) / len(rows),
                            categories=dict(sorted(counts.items())), file=path.name))
    Path(args.json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    fields = ['mode', 'level', 'dtype', 'n', 'strict_correct', 'strict_accuracy',
              'permissive_upper_bound', 'truncated', 'mean_output_tokens']
    categories = sorted({k for s in summary for k in s['categories']})
    with Path(args.csv).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + categories)
        writer.writeheader()
        for s in summary:
            writer.writerow({**{k: s[k] for k in fields}, **{k: s['categories'].get(k, 0) for k in categories}})
    for s in summary:
        print(s['mode'], s['level'], s['dtype'], f"{s['strict_correct']}/{s['n']}",
              f"estrito={s['strict_accuracy']:.4f}", f"permissivo={s['permissive_upper_bound']:.4f}",
              s['categories'])


if __name__ == '__main__':
    main()
