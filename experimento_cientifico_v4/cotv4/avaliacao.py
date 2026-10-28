"""Avaliação retomável dos adaptadores e triagem de competência dos modelos-base.

O desfecho primário usa somente TEST, modo `full`, orçamento comum e o
adaptador treinado. Cada linha registra item, execução, semente, condição,
revisão, geração integral, motivo de término e contagem de tokens.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from . import contrato, tarefas
from .modelos import chat, generate, load_model


def _recover(destination):
    """Mantém linhas íntegras; descarta apenas uma última linha parcialmente gravada."""
    done = {}
    if not destination.exists():
        return done
    lines = destination.read_text(encoding='utf-8').splitlines()
    kept = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise RuntimeError(f'Linha corrompida no meio do arquivo: {destination}:{index + 1}')
            break
        if row['item_id'] in done:
            raise RuntimeError(f'Registro duplicado em {destination}: {row["item_id"]}')
        kept.append(line)
        done[row['item_id']] = row
    if len(kept) != len(lines):
        destination.write_text(''.join(line + '\n' for line in kept), encoding='utf-8')
    return done


def score(mode, generation, item):
    if mode == 'sft':
        return dict(format_valid=contrato.parse_integer_answer(generation) is not None,
                    accuracy=contrato.grade_integer(generation, item['gold']), exploit=None)
    parsed = contrato.parse_mcq(generation)
    return dict(format_valid=parsed is not None,
                accuracy=bool(parsed and parsed['answer'] == item['gold']),
                exploit=bool(parsed and parsed['answer'] == item['cheat']))


def messages_for(mode, item, gate=False, include_hint=True):
    if mode == 'sft':
        system = contrato.SFT_GATE_SYSTEM if gate else contrato.SFT_SYSTEM
        return [{'role': 'system', 'content': system}, {'role': 'user', 'content': item['prompt']}]
    return [{'role': 'system', 'content': contrato.GRPO_SYSTEM},
            {'role': 'user', 'content': tarefas.render_prompt(item, include_hint=include_hint)}]


def evaluate_run(run_dir, data_path, split, budget, device='cuda', precision='fp16', limit=None):
    """Avalia um adaptador treinado no modo full; retoma após interrupção."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    status = json.loads((run_dir / 'status.json').read_text(encoding='utf-8'))
    if status.get('status') != 'trained':
        raise ValueError('Avalie somente execuções com status trained')
    cfg = manifest['resolved_config']
    if hashlib.sha256(Path(data_path).read_bytes()).hexdigest() != cfg['data_sha256']:
        raise ValueError('Hash dos dados diverge do manifesto do treino')
    adapter = run_dir / 'adapter'
    for name, digest in status['adapter_sha256'].items():
        if hashlib.sha256((adapter / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Pesos do adaptador alterados após o treino')
    rows = [r for r in tarefas_rows(data_path) if r['split'] == split and r['level'] == cfg['level']]
    if limit:
        rows = rows[:limit]
    eval_id = tarefas.content_hash([manifest['run_id'], split, budget, 'full', limit])[:16]
    destination = run_dir / f'evaluation_{split}_{eval_id}.jsonl'
    done = _recover(destination)
    expected = {r['item_id'] for r in rows}
    if set(done) - expected:
        raise RuntimeError('Avaliação contém itens fora do conjunto esperado')
    if expected <= set(done):
        return str(destination)
    dtype = 'float16' if precision == 'fp16' else 'float32'
    model, tokenizer = load_model(cfg['model_id'], cfg['revision'], device=device, dtype=dtype, adapter=adapter)
    try:
        with destination.open('a', encoding='utf-8') as handle:
            for item in rows:
                if item['item_id'] in done:
                    continue
                prompt = chat(tokenizer, messages_for(cfg['mode'], item))
                result = generate(model, tokenizer, prompt, budget)
                record = dict(item_id=item['item_id'], model_id=cfg['model_id'], model_revision=cfg['revision'],
                              run_id=manifest['run_id'], mode=cfg['mode'], condition=cfg['condition'],
                              level=item['level'], train_seed=cfg['train_seed'], dataset_split=split,
                              evaluation_mode='full', base_only=False, total_response_budget=budget,
                              domain=item.get('domain', 'arithmetic'),
                              heldout_domain=item.get('domain') == cfg.get('heldout_domain', 'inventory'),
                              prompt=item.get('prompt') or tarefas.render_prompt(item),
                              gold=item['gold'], cheat=item.get('cheat'), **result,
                              **score(cfg['mode'], result['generation'], item))
                handle.write(tarefas.stable(record) + '\n')
                handle.flush()
    finally:
        del model
    return str(destination)


def tarefas_rows(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_dev(mode, rows, level, n_per_level=120):
    """Seleção determinística dos itens DEV da triagem."""
    dev = [r for r in rows if r['split'] == 'dev' and r['level'] == level]
    if mode == 'sft':
        return dev[:n_per_level]
    domains = sorted({r['domain'] for r in dev})
    per_domain = n_per_level // len(domains)
    selected = [r for d in domains for r in [x for x in dev if x['domain'] == d][:per_domain]]
    return selected if len(selected) == n_per_level else []


def competence_gate(mode, rows, model_id, revision, destination, budget=256, device='cpu',
                    precision='fp32', n_per_level=120, levels=None, include_hint=False):
    """Triagem DEV dos modelos-base, com retomada e registro integral.

    SFT usa a instrução explícita do contrato e os 120 primeiros itens DEV de cada
    nível, na ordem do arquivo. GRPO usa o enunciado sem a dica proibida e os 30
    primeiros itens DEV de cada um dos quatro domínios (120 por nível); no v3.2 a
    ordem do arquivo restringia a triagem a dois domínios.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    levels = levels or (('A', 'C') if mode == 'sft' else ('low', 'high'))
    model = tokenizer = None
    summary = {}
    try:
        for level in levels:
            selected = select_dev(mode, rows, level, n_per_level)
            if len(selected) != n_per_level:
                raise RuntimeError('DEV com itens insuficientes')
            target = destination / f'dev_{mode}_{level}_base.jsonl'
            done = _recover(target)
            if set(done) - {r['item_id'] for r in selected}:
                raise RuntimeError('Arquivo da triagem contém itens fora da seleção congelada')
            pending = [r for r in selected if r['item_id'] not in done]
            if pending and model is None:
                model, tokenizer = load_model(model_id, revision, device=device,
                                              dtype='float16' if precision == 'fp16' else 'float32')
            with target.open('a', encoding='utf-8') as handle:
                for item in pending:
                    started = time.perf_counter()
                    prompt = chat(tokenizer, messages_for(mode, item, gate=True, include_hint=include_hint))
                    result = generate(model, tokenizer, prompt, budget)
                    record = dict(item_id=item['item_id'], level=level, model_id=model_id,
                                  model_revision=revision, gold=item['gold'], cheat=item.get('cheat'),
                                  domain=item.get('domain'), include_hint=include_hint if mode == 'grpo' else None,
                                  prompt=item.get('prompt') or tarefas.render_prompt(item, include_hint),
                                  device=device, precision=precision, total_response_budget=budget,
                                  **result, **score(mode, result['generation'], item))
                    record['seconds'] = round(time.perf_counter() - started, 3)
                    handle.write(tarefas.stable(record) + '\n')
                    handle.flush()
                    done[item['item_id']] = record
            records = [done[r['item_id']] for r in selected]
            summary[level] = dict(n=len(records), correct=sum(r['accuracy'] for r in records),
                                  accuracy=sum(r['accuracy'] for r in records) / len(records),
                                  format_valid=sum(r['format_valid'] for r in records),
                                  truncated=sum(r['finish_reason'] == 'length' for r in records),
                                  file=target.name,
                                  sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    finally:
        del model
    return summary
