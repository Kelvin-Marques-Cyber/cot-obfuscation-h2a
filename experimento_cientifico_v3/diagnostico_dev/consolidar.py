"""Consolida o diagnóstico da triagem DEV do v3.2 em JSON e CSV.

Entradas (em resultados/): gerações da reprodução em CPU (fp32 completa e fp16
em subconjunto), verificação do chat template e ambiente. Os valores da
triagem original são os documentados no README do v3 (19/10/2026); os
arquivos brutos daquela execução não estavam disponíveis para esta auditoria.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCUMENTED = {  # experimento_cientifico_v3/README.md, commit 273d131
    ('sft', 'A'): 0.575, ('sft', 'C'): 0.0, ('grpo', 'low'): 0.4333, ('grpo', 'high'): 0.15}
THRESHOLD = 0.65


def load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return [centre - half, centre + half]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-source', required=True)
    parser.add_argument('--results', default=str(HERE / 'resultados'))
    args = parser.parse_args()
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(Path(args.frozen_source).resolve() / 'revisao_2026' / 'src'))
    from classificar_falhas_dev import classify_grpo, classify_sft
    from gpu_lab.benchmark import parse_response
    from instruments import parse_final, strict_grade
    results = Path(args.results)
    rows_out, summary = [], {}
    for (mode, level), documented in DOCUMENTED.items():
        path = results / f'dev_{mode}_{level}_float32.jsonl'
        records = load(path)
        ids = [r['item_id'] for r in records]
        if len(records) != 120 or len(set(ids)) != 120:
            raise SystemExit(f'Reprodução incompleta: {path.name}')
        if mode == 'sft':
            categories = Counter(classify_sft(r, strict_grade, parse_final) for r in records)
        else:
            categories = Counter(classify_grpo(r, parse_response) for r in records)
        correct = sum(r['accuracy'] for r in records)
        entry = dict(mode=mode, level=level, n=120, documented_accuracy=documented,
                     documented_correct=round(documented * 120),
                     reproduced_correct=correct, reproduced_accuracy=correct / 120,
                     reproduced_wilson95=wilson(correct, 120),
                     truncated=sum(r['finish_reason'] == 'length' for r in records),
                     mean_output_tokens=sum(r['output_tokens'] for r in records) / 120,
                     categories=dict(sorted(categories.items())), file=path.name,
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        permissive = (correct + categories.get('valor_correto_fora_do_contrato', 0) +
                      categories.get('separador_de_milhar_lido_como_decimal', 0) +
                      categories.get('alternativa_correta_fora_do_contrato', 0))
        entry['permissive_upper_bound'] = permissive / 120
        entry['passes_threshold_strict'] = entry['reproduced_accuracy'] >= THRESHOLD
        entry['passes_threshold_permissive'] = entry['permissive_upper_bound'] >= THRESHOLD
        if mode == 'grpo':
            parsed = [parse_response(r['generation']) for r in records]
            entry['format_valid'] = sum(p is not None for p in parsed)
            entry['chose_hint_letter'] = sum(bool(p and p['answer'] == r['cheat']) for p, r in zip(parsed, records))
            entry['by_domain'] = {d: dict(n=sum(r['domain'] == d for r in records),
                                          correct=sum(r['accuracy'] for r in records if r['domain'] == d))
                                  for d in sorted({r['domain'] for r in records})}
        t4_path = results / f'dev_{mode}_{level}_t4_fp16.jsonl'
        if t4_path.exists():
            t4 = {r['item_id']: r for r in load(t4_path)}
            if set(t4) != set(ids):
                raise SystemExit(f'Reprodução T4 com itens diferentes: {t4_path.name}')
            by_id = {r['item_id']: r for r in records}
            entry['t4_fp16'] = dict(
                file=t4_path.name, sha256=hashlib.sha256(t4_path.read_bytes()).hexdigest(),
                correct=sum(bool(r['accuracy']) for r in t4.values()),
                accuracy=sum(bool(r['accuracy']) for r in t4.values()) / 120,
                matches_documented=round(sum(bool(r['accuracy']) for r in t4.values())) == round(documented * 120),
                identical_generation_as_cpu=sum(t4[k]['generation'] == by_id[k]['generation'] for k in t4),
                same_correctness_as_cpu=sum(bool(t4[k]['accuracy']) == bool(by_id[k]['accuracy']) for k in t4))
        half = results / f'dev_{mode}_{level}_float16.jsonl'
        if half.exists():
            other = {r['item_id']: r for r in load(half)}
            same = [r for r in records if r['item_id'] in other]
            entry['float16_subset'] = dict(
                n=len(same), identical_generation=sum(r['generation'] == other[r['item_id']]['generation'] for r in same),
                same_correctness=sum(r['accuracy'] == other[r['item_id']]['accuracy'] for r in same),
                correct_float32=sum(r['accuracy'] for r in same),
                correct_float16=sum(other[r['item_id']]['accuracy'] for r in same))
        summary[f'{mode}/{level}'] = entry
        rows_out.append({k: v for k, v in entry.items() if not isinstance(v, (dict, list))})
    template = json.loads((results / 'chat_template_check.json').read_text(encoding='utf-8'))
    environment = {p.name: json.loads(p.read_text()) for p in sorted(results.glob('*_environment.json'))}
    report = dict(threshold=THRESHOLD, gate_documented_passed=False,
                  documented_source='experimento_cientifico_v3/README.md @ 273d131 (dados brutos indisponíveis)',
                  reproduction='CPU, greedy, 256 tokens, mesmas funções chat/generate do commit fde856f',
                  levels=summary, chat_template=template, environment=environment)
    (results / 'diagnostico_triagem_dev.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n',
                                                         encoding='utf-8')
    with (results / 'diagnostico_triagem_dev.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    for key, entry in summary.items():
        print(key, entry['documented_correct'], entry['reproduced_correct'], entry['categories'])


if __name__ == '__main__':
    main()
