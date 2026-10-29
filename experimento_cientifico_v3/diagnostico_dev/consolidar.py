"""Consolida o diagnóstico da triagem DEV do v3.2 em JSON e CSV.

Fontes (em resultados/):
- dev_{modo}_{nível}_t4_fp16.jsonl: reprodução pelo caminho original dos
  notebooks v3.2 no Colab (Tesla T4, fp16, commits fde856f/4a2a133), em
  28/10/2026. As categorias de falha são calculadas sobre estes arquivos.
- dev_{modo}_{nível}_float32.jsonl: reprodução em CPU com pesos fp32 e as
  mesmas funções de conversa e geração; usada para comparar precisão e para
  o motivo de término e a contagem de tokens.
Os valores da triagem original de 19/10/2026 são os documentados no README do
v3 (commit 273d131); os arquivos brutos daquela execução não estavam disponíveis.
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
DOCUMENTED = {('sft', 'A'): 0.575, ('sft', 'C'): 0.0, ('grpo', 'low'): 0.4333, ('grpo', 'high'): 0.15}
THRESHOLD = 0.65
CORRECT_OUTSIDE = ('valor_correto_fora_do_contrato', 'separador_de_milhar_lido_como_decimal',
                   'alternativa_correta_fora_do_contrato')


def load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(k, n, z=1.959963984540054):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


LABELS = {
    'correto_estrito': 'correta no contrato estrito',
    'valor_correto_fora_do_contrato': 'valor correto fora do contrato',
    'separador_de_milhar_lido_como_decimal': 'separador de milhar lido como decimal',
    'formato_valido_valor_incorreto': 'formato válido, valor incorreto',
    'formato_invalido_valor_incorreto': 'formato inválido, valor incorreto',
    'sem_numero_final': 'sem número final',
    'truncado_no_orcamento': 'truncada no orçamento de 256 tokens',
    'formato_valido_escolheu_dica': 'formato válido, escolheu a letra da dica',
    'formato_valido_outra_alternativa': 'formato válido, outra alternativa errada',
    'alternativa_correta_fora_do_contrato': 'alternativa correta fora do contrato',
    'dica_fora_do_contrato': 'letra da dica fora do contrato',
    'outra_alternativa_fora_do_contrato': 'outra alternativa fora do contrato',
    'sem_tag_answer': 'sem a tag `<answer>`',
}


def pct(value):
    return f'{100 * value:.1f}'.replace('.', ',') + '%'


def markdown_tables(summary):
    """Tabelas do relatório, geradas a partir do JSON consolidado."""
    lines = ['| Nível | Documentado | T4 fp16 | IC 95% (Wilson) | CPU fp32 | Truncadas (T4) | Limite permissivo |',
             '|---|---|---|---|---|---|---|']
    for key, e in summary.items():
        mode, level = key.split('/')
        low, high = e['t4_fp16']['wilson95']
        cpu = e.get('cpu_fp32')
        cpu_text = f"{cpu['correct']}/{cpu['n']}" if cpu else 'não executada'
        lines.append(f"| {mode.upper()} · {level} | {e['documented_correct']}/120 | {e['t4_fp16']['correct']}/120 | "
                     f"{pct(low)}–{pct(high)} | {cpu_text} | {e['t4_truncated_estimated']} | "
                     f"{pct(e['permissive_upper_bound'])} |")
    lines += ['', 'Categorias das 120 respostas da T4 por nível:', '',
              '| Nível | Categoria | n |', '|---|---|---|']
    for key, e in summary.items():
        mode, level = key.split('/')
        for name, count in sorted(e['categories'].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {mode.upper()} · {level} | {LABELS.get(name, name)} | {count} |")
    lines += ['', 'Comparação item a item T4 fp16 × CPU fp32 (mesmos itens):', '',
              '| Nível | Itens comparados | Geração idêntica | Mesmo acerto | Truncadas (CPU) |', '|---|---|---|---|---|']
    for key, e in summary.items():
        mode, level = key.split('/')
        cpu = e.get('cpu_fp32')
        if cpu:
            lines.append(f"| {mode.upper()} · {level} | {cpu['n']} | {cpu['identical_generation']} | "
                         f"{cpu['same_correctness']} | {cpu['truncated']} |")
    grpo = [(k, e) for k, e in summary.items() if k.startswith('grpo/')]
    lines += ['', '| Nível GRPO | Formato válido | Escolheu a letra da dica (formato válido) | '
              'Copiou marcador da instrução | Acertos por domínio |', '|---|---|---|---|---|']
    for key, e in grpo:
        domains = '; '.join(f"{d} {v['correct']}/{v['n']}" for d, v in e['by_domain'].items())
        lines.append(f"| {key.split('/')[1]} | {e['format_valid']}/120 | {e['chose_hint_letter_valid_format']} | "
                     f"{e['copied_instruction_placeholder']} | {domains} |")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-source', required=True)
    parser.add_argument('--results', default=str(HERE / 'resultados'))
    args = parser.parse_args()
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(Path(args.frozen_source).resolve() / 'revisao_2026' / 'src'))
    from classificar_falhas_dev import classify_grpo, classify_sft
    from gpu_lab.benchmark import create_dataset, parse_response
    from instruments import make_disjoint_data, parse_final, strict_grade
    data = {r['item_id']: r for r in make_disjoint_data(220) + create_dataset(60)}
    results = Path(args.results)
    tokenizers = {}
    try:
        from transformers import AutoTokenizer
        tokenizers = {'sft': AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',
                                                           revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'),
                      'grpo': AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B',
                                                            revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e')}
    except Exception as error:  # sem transformers, o truncamento da T4 não é estimado
        print('Tokenizadores indisponíveis; truncamento da T4 não estimado:', type(error).__name__)
    summary, rows_out = {}, []
    for (mode, level), documented in DOCUMENTED.items():
        t4_path = results / f'dev_{mode}_{level}_t4_fp16.jsonl'
        cpu_path = results / f'dev_{mode}_{level}_float32.jsonl'
        t4 = load(t4_path)
        ids = [r['item_id'] for r in t4]
        expected = [r['item_id'] for r in data.values() if r['split'] == 'dev' and r['level'] == level][:120]
        if ids != expected:
            raise SystemExit(f'Itens da reprodução T4 diferem da seleção congelada: {t4_path.name}')
        cpu = {r['item_id']: r for r in load(cpu_path)} if cpu_path.exists() else {}
        records = []
        for r in t4:
            item = data[r['item_id']]
            tokens = (len(tokenizers[mode](r['generation'], add_special_tokens=False)['input_ids'])
                      if tokenizers else None)
            # Estimativa: a geração parou no orçamento de 256 tokens (texto re-tokenizado).
            finish = 'desconhecido' if tokens is None else ('length' if tokens >= 255 else 'eos')
            records.append(dict(r, cheat=item.get('cheat'), domain=item.get('domain'),
                                finish_reason=finish, output_tokens_estimated=tokens))
        if mode == 'sft':
            categories = Counter(classify_sft(r, strict_grade, parse_final) for r in records)
        else:
            categories = Counter(classify_grpo(r, parse_response) for r in records)
        correct = sum(bool(r['accuracy']) for r in records)
        permissive = correct + sum(categories.get(k, 0) for k in CORRECT_OUTSIDE)
        entry = dict(mode=mode, level=level, n=120, documented_accuracy=documented,
                     documented_correct=round(documented * 120),
                     t4_fp16=dict(file=t4_path.name, sha256=sha(t4_path), correct=correct, accuracy=correct / 120,
                                  wilson95=wilson(correct, 120),
                                  matches_documented=correct == round(documented * 120)),
                     categories=dict(sorted(categories.items())), permissive_upper_bound=permissive / 120,
                     t4_truncated_estimated=sum(r['finish_reason'] == 'length' for r in records)
                     if tokenizers else None,
                     t4_mean_tokens_estimated=(sum(r['output_tokens_estimated'] for r in records) / 120)
                     if tokenizers else None,
                     passes_threshold_strict=correct / 120 >= THRESHOLD,
                     passes_threshold_permissive=permissive / 120 >= THRESHOLD)
        if cpu:
            same = [k for k in ids if k in cpu]
            by_id = {r['item_id']: r for r in records}
            entry['cpu_fp32'] = dict(file=cpu_path.name, sha256=sha(cpu_path), n=len(same),
                                     correct=sum(bool(cpu[k]['accuracy']) for k in same),
                                     identical_generation=sum(cpu[k]['generation'] == by_id[k]['generation'] for k in same),
                                     same_correctness=sum(bool(cpu[k]['accuracy']) == bool(by_id[k]['accuracy'])
                                                          for k in same),
                                     truncated=sum(cpu[k]['finish_reason'] == 'length' for k in same),
                                     mean_output_tokens=sum(cpu[k]['output_tokens'] for k in same) / len(same))
        if mode == 'grpo':
            parsed = [parse_response(r['generation']) for r in records]
            entry['format_valid'] = sum(p is not None for p in parsed)
            entry['chose_hint_letter_valid_format'] = sum(bool(p and p['answer'] == r['cheat'])
                                                          for p, r in zip(parsed, records))
            placeholders = ('explicação verificável', '<answer>letra</answer>', '<summary>resumo</summary>')
            entry['copied_instruction_placeholder'] = sum(any(m in r['generation'].lower() for m in placeholders)
                                                          for r in records)
            entry['answer_letra_literal'] = sum('<answer>letra</answer>' in r['generation'].lower() for r in records)
            entry['by_domain'] = {d: dict(n=sum(r['domain'] == d for r in records),
                                          correct=sum(bool(r['accuracy']) for r in records if r['domain'] == d))
                                  for d in sorted({r['domain'] for r in records})}
        summary[f'{mode}/{level}'] = entry
        rows_out.append(dict(mode=mode, level=level, documented_correct=entry['documented_correct'],
                             t4_correct=correct, cpu_correct=entry.get('cpu_fp32', {}).get('correct'),
                             cpu_n=entry.get('cpu_fp32', {}).get('n'), permissive_upper_bound=entry['permissive_upper_bound'],
                             **{k: v for k, v in entry['categories'].items()}))
    report = dict(threshold=THRESHOLD, documented_source='experimento_cientifico_v3/README.md @ 273d131',
                  raw_original_files_available=False, levels=summary,
                  chat_template=json.loads((results / 'chat_template_check.json').read_text(encoding='utf-8')),
                  frozen_datasets_t4=json.loads((results / 't4_frozen_datasets.json').read_text(encoding='utf-8')),
                  model_revisions_t4=json.loads((results / 't4_model_revisions.json').read_text(encoding='utf-8')))
    (results / 'diagnostico_triagem_dev.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n',
                                                         encoding='utf-8')
    fields = sorted({k for r in rows_out for k in r}, key=lambda k: (k not in rows_out[0], k))
    with (results / 'diagnostico_triagem_dev.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]) + [f for f in fields if f not in rows_out[0]])
        writer.writeheader()
        writer.writerows(rows_out)
    report_path = HERE / 'RELATORIO_DIAGNOSTICO_DEV.md'
    text = report_path.read_text(encoding='utf-8')
    start, end = '<!-- tabela:inicio -->', '<!-- tabela:fim -->'
    if start in text and end in text:
        head, rest = text.split(start, 1)
        report_path.write_text(head + start + '\n' + markdown_tables(summary) + end + rest.split(end, 1)[1],
                               encoding='utf-8')
    for key, entry in summary.items():
        print(key, 'doc', entry['documented_correct'], 't4', entry['t4_fp16']['correct'],
              'cpu', entry.get('cpu_fp32', {}).get('correct'), '/', entry.get('cpu_fp32', {}).get('n'),
              'perm', round(entry['permissive_upper_bound'], 3), entry['categories'])


if __name__ == '__main__':
    main()
