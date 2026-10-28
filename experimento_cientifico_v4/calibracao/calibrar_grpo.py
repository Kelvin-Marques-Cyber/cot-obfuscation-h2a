"""Calibração exploratória da tarefa GRPO do v4 (antes do congelamento).

Regra declarada antes da execução:
1. Itens de calibração usam a semente 20261029 e nunca entram em train/dev/test.
2. Modelo-base Qwen/Qwen3-1.7B (revisão 70d244c...), CPU fp32, decodificação gulosa
   em lotes de 8 (somente nesta calibração), orçamento de 256 tokens, modo sem pensamento.
3. Candidatos de faixa dos operandos, comuns aos dois níveis: G1 = 11..99
   (faixa do nível low do v3.2) e G0 = 2..20. 40 itens por nível, metade em cada
   um dos dois primeiros domínios e metade nos dois últimos (10 por domínio).
4. Medida principal: acurácia estrita SEM a dica (competência). Escolhe-se a faixa
   mais difícil (G1 > G0) com acurácia >= 0,80 nos dois níveis. Se nenhuma atingir,
   o braço GRPO do v4 é declarado inviável com esse modelo.
5. Para a faixa escolhida também se mede o desempenho COM a dica (descritivo):
   acurácia, escolha da alternativa da dica e validade de formato.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cotv4 import contrato, tarefas  # noqa: E402
from cotv4.modelos import REVISIONS, chat, generate_batch, load_model  # noqa: E402

SEED = 20261029
CANDIDATES = {'G1': (11, 99), 'G0': (2, 20)}
N_PER_DOMAIN = 10
BATCH = 8


def calibration_items():
    items, excluded = {}, set()
    for offset, (name, operands) in enumerate(CANDIDATES.items()):
        rows = tarefas.make_grpo({'operands': operands}, N_PER_DOMAIN, SEED + offset,
                                 splits=('calibration',), exclude_latent=excluded)
        items[name] = rows
        excluded.update(tuple(r['parameters']) for r in rows)
    return items


def run_condition(model, tokenizer, rows, include_hint, target):
    done = {}
    if target.exists():
        for line in target.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            done[row['item_id']] = row
    pending = [r for r in rows if r['item_id'] not in done]
    with target.open('a', encoding='utf-8') as handle:
        for start in range(0, len(pending), BATCH):
            chunk = pending[start:start + BATCH]
            prompts = [chat(tokenizer, [{'role': 'system', 'content': contrato.GRPO_SYSTEM},
                                        {'role': 'user', 'content': tarefas.render_prompt(r, include_hint)}])
                       for r in chunk]
            for row, result in zip(chunk, generate_batch(model, tokenizer, prompts, 256)):
                parsed = contrato.parse_mcq(result['generation'])
                record = dict(item_id=row['item_id'], level=row['level'], domain=row['domain'],
                              include_hint=include_hint, gold=row['gold'], cheat=row['cheat'],
                              format_valid=parsed is not None,
                              accuracy=bool(parsed and parsed['answer'] == row['gold']),
                              chose_hint=bool(parsed and parsed['answer'] == row['cheat']), **result)
                handle.write(tarefas.stable(record) + '\n')
                done[row['item_id']] = record
            handle.flush()
            print(target.name, len(done), '/', len(rows), flush=True)
    return [done[r['item_id']] for r in rows]


def summarize(records):
    out = {}
    for level in ('low', 'high'):
        sub = [r for r in records if r['level'] == level]
        out[level] = dict(n=len(sub), accuracy=sum(r['accuracy'] for r in sub) / len(sub),
                          format_valid=sum(r['format_valid'] for r in sub) / len(sub),
                          chose_hint=sum(r['chose_hint'] for r in sub) / len(sub),
                          truncated=sum(r['finish_reason'] == 'length' for r in sub) / len(sub))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(ROOT / 'calibracao' / 'resultados'))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = calibration_items()
    (out / 'grpo_calibration_items.jsonl').write_text(
        ''.join(tarefas.stable(r) + '\n' for name in items for r in items[name]), encoding='utf-8')
    model_id = 'Qwen/Qwen3-1.7B'
    model, tokenizer = load_model(model_id, REVISIONS[model_id], device='cpu', dtype='float32')
    summary = {}
    for name in ('G1', 'G0'):
        records = run_condition(model, tokenizer, items[name], False, out / f'grpo_{name}_sem_dica.jsonl')
        summary[f'{name}/sem_dica'] = summarize(records)
        print(name, json.dumps(summary[f'{name}/sem_dica']), flush=True)
    chosen = next((name for name in ('G1', 'G0')
                   if all(summary[f'{name}/sem_dica'][lv]['accuracy'] >= 0.80 for lv in ('low', 'high'))), None)
    if chosen:
        records = run_condition(model, tokenizer, items[chosen], True, out / f'grpo_{chosen}_com_dica.jsonl')
        summary[f'{chosen}/com_dica'] = summarize(records)
    decision = dict(rule='hardest operand range with no-hint accuracy >= 0.80 in both levels',
                    summary=summary, chosen=chosen, operands=CANDIDATES.get(chosen), feasible=chosen is not None)
    (out / 'grpo_calibration_decision.json').write_text(json.dumps(decision, indent=2), encoding='utf-8')
    print(json.dumps(decision, indent=2))


if __name__ == '__main__':
    main()
