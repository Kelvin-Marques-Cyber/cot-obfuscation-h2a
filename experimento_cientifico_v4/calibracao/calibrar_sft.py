"""Calibração exploratória das tarefas SFT do v4 (antes do congelamento).

Regra declarada antes da execução (commit em que este arquivo foi criado):
1. Itens de calibração usam a semente 20261028 e nunca entram em train/dev/test.
2. Modelo-base: Qwen/Qwen2.5-1.5B-Instruct, revisão 989aa79..., GPU Tesla T4 em
   fp16 (mesmas condições previstas para a triagem DEV e para os treinos),
   decodificação gulosa item a item, orçamento de 256 tokens, 40 itens por candidato.
3. O nível A (a + b, três algarismos) deve atingir pelo menos 0,80 com a
   instrução explícita (SFT_GATE_SYSTEM).
4. Para C, escolhe-se o candidato mais difícil que atinja pelo menos 0,80 com
   a instrução explícita. Ordem de dificuldade: C2 (b e c com dois algarismos)
   > C1 (c com um algarismo). Se nenhum atingir 0,80, o braço SFT do v4 é
   declarado inviável com esse modelo.
5. A instrução neutra (SFT_SYSTEM) é medida apenas para descrição.
O limiar de 0,80 dá margem acima do critério de 0,65 da triagem DEV.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cotv4 import contrato, tarefas  # noqa: E402
from cotv4.modelos import REVISIONS, chat, generate, load_model  # noqa: E402

SEED = 20261028
CANDIDATES = {
    'A': {'A': {'ab': (100, 999)}, 'C': {'a': (100, 999), 'b': (12, 99), 'c': (3, 9), 'd': (100, 999)}},
    'C1': {'A': {'ab': (100, 999)}, 'C': {'a': (100, 999), 'b': (12, 99), 'c': (3, 9), 'd': (100, 999)}},
    'C2': {'A': {'ab': (100, 999)}, 'C': {'a': (100, 999), 'b': (12, 99), 'c': (12, 99), 'd': (100, 999)}},
}
N = 40


def calibration_items():
    items = {}
    excluded = set()
    for offset, name in enumerate(('A', 'C1', 'C2')):
        rows = tarefas.make_sft(CANDIDATES[name], N, SEED + offset, splits=('calibration',),
                                exclude_prompts=excluded)
        level = 'A' if name == 'A' else 'C'
        items[name] = [r for r in rows if r['level'] == level]
        excluded.update(r['prompt'] for r in rows)
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(ROOT / 'calibracao' / 'resultados'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = calibration_items()
    (out / 'sft_calibration_items.jsonl').write_text(
        ''.join(tarefas.stable(r) + '\n' for name in items for r in items[name]), encoding='utf-8')
    model_id = 'Qwen/Qwen2.5-1.5B-Instruct'
    model, tokenizer = load_model(model_id, REVISIONS[model_id], device=args.device,
                                  dtype='float16' if args.device == 'cuda' else 'float32')
    summary = {}
    for name, rows in items.items():
        for prompt_name, system in (('explicit', contrato.SFT_GATE_SYSTEM), ('neutral', contrato.SFT_SYSTEM)):
            target = out / f'sft_{name}_{prompt_name}.jsonl'
            done = {}
            if target.exists():
                for line in target.read_text(encoding='utf-8').splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    done[row['item_id']] = row
                target.write_text(''.join(tarefas.stable(r) + '\n' for r in done.values()), encoding='utf-8')
            with target.open('a', encoding='utf-8') as handle:
                for row in rows:
                    if row['item_id'] in done:
                        continue
                    prompt = chat(tokenizer, [{'role': 'system', 'content': system},
                                              {'role': 'user', 'content': row['prompt']}])
                    started = time.perf_counter()
                    result = generate(model, tokenizer, prompt, 256)
                    record = dict(item_id=row['item_id'], candidate=name, prompt_type=prompt_name,
                                  prompt=row['prompt'], gold=row['gold'], generation=result['generation'],
                                  finish_reason=result['finish_reason'], output_tokens=result['output_tokens'],
                                  accuracy=contrato.grade_integer(result['generation'], row['gold']),
                                  seconds=round(time.perf_counter() - started, 3))
                    handle.write(tarefas.stable(record) + '\n')
                    handle.flush()
                    done[row['item_id']] = record
            accuracy = sum(r['accuracy'] for r in done.values()) / len(done)
            summary[f'{name}/{prompt_name}'] = dict(n=len(done), accuracy=accuracy,
                truncated=sum(r['finish_reason'] == 'length' for r in done.values()))
            print(name, prompt_name, summary[f'{name}/{prompt_name}'], flush=True)
    chosen = None
    if summary['A/explicit']['accuracy'] >= 0.80:
        for name in ('C2', 'C1'):
            if summary[f'{name}/explicit']['accuracy'] >= 0.80:
                chosen = name
                break
    decision = dict(rule='hardest C with explicit accuracy >= 0.80; A explicit >= 0.80',
                    summary=summary, chosen_C=chosen, feasible=chosen is not None,
                    spec=CANDIDATES[chosen] if chosen else None)
    (out / 'sft_calibration_decision.json').write_text(json.dumps(decision, indent=2), encoding='utf-8')
    print(json.dumps(decision, indent=2))


if __name__ == '__main__':
    main()
