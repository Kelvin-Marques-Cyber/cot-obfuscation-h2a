"""Reprodução diagnóstica, em CPU, da triagem DEV do estudo v3.2.

Este script NÃO substitui a triagem registrada do estudo v3.2 e não pode
liberar seus treinamentos. Ele reconstrói os dados congelados com o código do
commit científico, seleciona os mesmos 120 itens DEV por nível e gera respostas
com as mesmas funções `chat` e `generate` e o mesmo orçamento de 256 tokens.
Diferenças registradas: CPU em vez de GPU T4 e precisão escolhida em --dtype.

Uso:
  python reproduzir_triagem_dev.py --frozen-source DIR --out DIR --mode sft
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REVISIONS = {
    'Qwen/Qwen2.5-1.5B-Instruct': '989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
    'Qwen/Qwen3-1.7B': '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
}
MODEL = {'sft': 'Qwen/Qwen2.5-1.5B-Instruct', 'grpo': 'Qwen/Qwen3-1.7B'}
LEVELS = {'sft': ('A', 'C'), 'grpo': ('low', 'high')}
SFT_SYSTEM = 'Resolva o problema. Termine com uma única tag <answer>NUMERO</answer>.'
BUDGET = 256


def stable(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def frozen_rows(mode):
    """Mesma chamada de make_plan(): 220 itens SFT e 60 GRPO por célula."""
    if mode == 'sft':
        from instruments import make_disjoint_data
        return make_disjoint_data(220)
    from gpu_lab.benchmark import create_dataset
    return create_dataset(60)


def read_done(path):
    done = {}
    if not path.exists():
        return done
    lines = path.read_text(encoding='utf-8').splitlines()
    kept = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            break  # última linha parcialmente gravada após interrupção
        kept.append(line)
        done[row['item_id']] = row
    path.write_text(''.join(line + '\n' for line in kept), encoding='utf-8')
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-source', required=True,
                        help='Checkout do commit científico (contém revisao_2026/src)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--mode', choices=('sft', 'grpo'), required=True)
    parser.add_argument('--dtype', choices=('float32', 'bfloat16', 'float16'), default='float32')
    parser.add_argument('--levels', nargs='*')
    parser.add_argument('--limit', type=int, default=120)
    parser.add_argument('--threads', type=int, default=0)
    args = parser.parse_args()

    source = Path(args.frozen_source).resolve()
    sys.path.insert(0, str(source / 'revisao_2026' / 'src'))
    os.environ.setdefault('COT_TEXT_ONLY', '1')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.threads:
        torch.set_num_threads(args.threads)
    from gpu_lab.train import chat
    from gpu_lab.evaluate import generate
    from gpu_lab.benchmark import SYSTEM, parse_response, render_prompt
    from instruments import strict_grade

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = frozen_rows(args.mode)
    data = ''.join(stable(row) + '\n' for row in rows).encode()
    data_sha = hashlib.sha256(data).hexdigest()
    (out / f'{args.mode}_data_sha256.txt').write_text(data_sha + '\n')

    model_id = MODEL[args.mode]
    revision = REVISIONS[model_id]
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(113)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype,
                                                 device_map={'': 'cpu'}, attn_implementation='sdpa')
    model.eval()
    model.config.use_cache = True

    environment = dict(python=platform.python_version(), torch=torch.__version__,
                       cpu=platform.processor() or platform.machine(), threads=torch.get_num_threads(),
                       dtype=args.dtype, device='cpu', model_id=model_id, revision=revision,
                       data_sha256=data_sha, budget=BUDGET,
                       frozen_source=subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                                    capture_output=True, text=True).stdout.strip() or source.name)
    import transformers
    environment['transformers'] = transformers.__version__
    (out / f'{args.mode}_{args.dtype}_environment.json').write_text(json.dumps(environment, indent=2))

    for level in (args.levels or LEVELS[args.mode]):
        selected = [r for r in rows if r['split'] == 'dev' and r['level'] == level][:120]
        if len(selected) != 120:
            raise RuntimeError('DEV com menos de 120 itens')
        selected = selected[:args.limit]
        target = out / f'dev_{args.mode}_{level}_{args.dtype}.jsonl'
        done = read_done(target)
        with target.open('a', encoding='utf-8') as handle:
            for position, row in enumerate(selected):
                if row['item_id'] in done:
                    continue
                if args.mode == 'sft':
                    system, user = SFT_SYSTEM, row['prompt']
                else:
                    system, user = SYSTEM, render_prompt(row)
                prompt = chat(tokenizer, [{'role': 'system', 'content': system},
                                          {'role': 'user', 'content': user}])
                started = time.perf_counter()
                result = generate(model, tokenizer, prompt, BUDGET)
                generated = result['generation']
                if args.mode == 'sft':
                    correct = strict_grade(generated, row['gold'])
                else:
                    correct = bool((parse_response(generated) or {}).get('answer') == row['gold'])
                record = dict(position=position, item_id=row['item_id'], level=level,
                              gold=row['gold'], cheat=row.get('cheat'), prompt=user,
                              domain=row.get('domain'), generation=generated,
                              accuracy=bool(correct), finish_reason=result['finish_reason'],
                              output_tokens=result['output_tokens'],
                              input_tokens=result['input_tokens'],
                              seconds=round(time.perf_counter() - started, 3))
                handle.write(stable(record) + '\n')
                handle.flush()
                print(level, position, record['finish_reason'], record['output_tokens'],
                      record['accuracy'], round(record['seconds'], 1), flush=True)
        records = list(read_done(target).values())
        accuracy = sum(r['accuracy'] for r in records) / len(records)
        print(f'NIVEL {level}: {sum(r["accuracy"] for r in records)}/{len(records)} = {accuracy:.4f}',
              flush=True)


if __name__ == '__main__':
    main()
