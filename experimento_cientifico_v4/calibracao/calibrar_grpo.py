"""Calibração exploratória da tarefa GRPO do v4 (antes do congelamento).

Regra declarada antes da execução:
1. Itens de calibração usam a semente 20261029 e nunca entram em train/dev/test.
2. Modelo-base Qwen/Qwen3-1.7B (revisão 70d244c...), GPU Tesla T4 em fp16,
   decodificação gulosa item a item, orçamento de 256 tokens, modo sem pensamento
   (mesmas condições previstas para a triagem DEV).
3. Candidatos de faixa dos operandos, comuns aos dois níveis: G1 = 11..99
   (faixa do nível low do v3.2) e G0 = 2..20. 40 itens por nível, metade em cada
   um dos dois primeiros domínios e metade nos dois últimos (10 por domínio).
4. Medida principal: acurácia estrita SEM a dica (competência). Escolhe-se a faixa
   mais difícil (G1 > G0) com acurácia >= 0,80 nos dois níveis. Se nenhuma atingir,
   o braço GRPO do v4 é declarado inviável com esse modelo.
5. Para a faixa escolhida também se mede o desempenho COM a dica (descritivo):
   acurácia, escolha da alternativa da dica e validade de formato.

Rodada 2 (declarada antes de sua execução, depois da rodada 1 e do diagnóstico
do v3.2): no v3.2 o modelo copiava os marcadores da instrução ("explicação
verificável", "resumo") e escrevia o raciocínio fora das tags, o que o contrato
rejeita mesmo com a letra correta. A rodada 2 usa os mesmos itens e a mesma regra
de 0,80 sem dica, testando em ordem: r2a (instrução sem marcadores copiáveis) e
r2b (r2a com uma demonstração de formato sem nota auxiliar), sempre com G1.
Escolhe-se a primeira variante aprovada; se nenhuma for aprovada com G1, avalia-se
G0 com r2b. Para a escolha, mede-se também o desempenho com a dica (descritivo).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cotv4 import contrato, tarefas  # noqa: E402
from cotv4.modelos import REVISIONS, chat, generate, generate_batch, load_model  # noqa: E402

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


def run_condition(model, tokenizer, rows, include_hint, target, batch=1, variant='v3'):
    done = {}
    if target.exists():
        for line in target.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            done[row['item_id']] = row
    pending = [r for r in rows if r['item_id'] not in done]
    with target.open('a', encoding='utf-8') as handle:
        for start in range(0, len(pending), batch):
            chunk = pending[start:start + batch]
            prompts = [chat(tokenizer, contrato.grpo_messages(tarefas.render_prompt(r, include_hint), variant))
                       for r in chunk]
            results = ([generate(model, tokenizer, prompts[0], 256)] if batch == 1
                       else generate_batch(model, tokenizer, prompts, 256))
            for row, result in zip(chunk, results):
                parsed = contrato.parse_mcq(result['generation'])
                record = dict(item_id=row['item_id'], level=row['level'], domain=row['domain'],
                              include_hint=include_hint, variant=variant, gold=row['gold'], cheat=row['cheat'],
                              format_valid=parsed is not None,
                              accuracy=bool(parsed and parsed['answer'] == row['gold']),
                              chose_hint=bool(parsed and parsed['answer'] == row['cheat']), **result)
                handle.write(tarefas.stable(record) + '\n')
                done[row['item_id']] = record
            handle.flush()
            print(target.name, len(done), '/', len(rows), flush=True)
    return [done[r['item_id']] for r in rows]


def round_two(model, tokenizer, items, out, batch):
    passes = lambda summary: all(summary[level]['accuracy'] >= 0.80 for level in ('low', 'high'))
    summary, chosen = {}, None
    plan = [('r2a', 'G1'), ('r2b', 'G1'), ('r2b', 'G0')]
    for variant, name in plan:
        if chosen:
            break
        records = run_condition(model, tokenizer, items[name], False,
                                out / f'grpo_{name}_sem_dica_{variant}.jsonl', batch, variant)
        summary[f'{variant}/{name}/sem_dica'] = summarize(records)
        print(variant, name, json.dumps(summary[f'{variant}/{name}/sem_dica']), flush=True)
        if passes(summary[f'{variant}/{name}/sem_dica']):
            chosen = (variant, name)
    if chosen:
        variant, name = chosen
        records = run_condition(model, tokenizer, items[name], True,
                                out / f'grpo_{name}_com_dica_{variant}.jsonl', batch, variant)
        summary[f'{variant}/{name}/com_dica'] = summarize(records)
    decision = dict(rule='round 2: first of r2a/G1, r2b/G1, r2b/G0 with no-hint accuracy >= 0.80 in both levels',
                    round=2, summary=summary, chosen=chosen, feasible=chosen is not None,
                    operands=CANDIDATES[chosen[1]] if chosen else None)
    (out / 'grpo_calibration_decision_r2.json').write_text(json.dumps(decision, indent=2), encoding='utf-8')
    print(json.dumps(decision, indent=2))
    return decision


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
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--rodada', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    batch = 1 if args.device == 'cuda' else BATCH
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = calibration_items()
    (out / 'grpo_calibration_items.jsonl').write_text(
        ''.join(tarefas.stable(r) + '\n' for name in items for r in items[name]), encoding='utf-8')
    model_id = 'Qwen/Qwen3-1.7B'
    model, tokenizer = load_model(model_id, REVISIONS[model_id], device=args.device,
                                  dtype='float16' if args.device == 'cuda' else 'float32')
    if args.rodada == 2:
        return round_two(model, tokenizer, items, out, batch)
    summary = {}
    for name in ('G1', 'G0'):
        records = run_condition(model, tokenizer, items[name], False, out / f'grpo_{name}_sem_dica.jsonl', batch)
        summary[f'{name}/sem_dica'] = summarize(records)
        print(name, json.dumps(summary[f'{name}/sem_dica']), flush=True)
    chosen = next((name for name in ('G1', 'G0')
                   if all(summary[f'{name}/sem_dica'][lv]['accuracy'] >= 0.80 for lv in ('low', 'high'))), None)
    if chosen:
        records = run_condition(model, tokenizer, items[chosen], True, out / f'grpo_{chosen}_com_dica.jsonl', batch)
        summary[f'{chosen}/com_dica'] = summarize(records)
    decision = dict(rule='hardest operand range with no-hint accuracy >= 0.80 in both levels',
                    summary=summary, chosen=chosen, operands=CANDIDATES.get(chosen), feasible=chosen is not None)
    (out / 'grpo_calibration_decision.json').write_text(json.dumps(decision, indent=2), encoding='utf-8')
    print(json.dumps(decision, indent=2))


if __name__ == '__main__':
    main()
