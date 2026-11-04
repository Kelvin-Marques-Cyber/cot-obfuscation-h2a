"""Análise exploratória S5 do desvio 1 (DESVIO_01.md), declarada antes de qualquer geração TEST.

Uso: python analise_desvio_01.py --storage PASTA_DO_ESTUDO [--saida arquivo.json]

1. Passos descartados pelo fp16 (`skipped_fp16_steps` do status.json de cada treino), relatados
   por nível e condição.
2. S5: o contraste D1 de H1 recalculado só com as sementes em que nenhum dos seis adaptadores
   (2 níveis x 3 condições) teve passo descartado, com os mesmos procedimentos estatísticos do
   desfecho primário (runtime.load_primary, estatistica.paired_sft, bootstrap e inversão de
   sinais).

Nada aqui altera o veredito de final_report; o valor-p de S5 é descritivo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

LEVELS = ('A', 'C')
CONDITIONS = ('explicit', 'direct', 'generic')


def skipped_by_job(index):
    """{(nível, semente, condição): [passos descartados]} a partir dos status.json."""
    out = {}
    for job, run_dir in index.items():
        state = json.loads((Path(run_dir) / 'status.json').read_text(encoding='utf-8'))
        out[job] = sorted(int(s) for s in state.get('skipped_fp16_steps') or [])
    return out


def clean_seeds(skips, seeds):
    """Sementes em que nenhum dos seis adaptadores teve passo descartado."""
    return [seed for seed in seeds
            if not any(skips[level, seed, condition] for level in LEVELS for condition in CONDITIONS)]


def descriptives(skips):
    cells = {}
    for level in LEVELS:
        for condition in CONDITIONS:
            values = {seed: s for (lv, seed, cond), s in skips.items() if lv == level and cond == condition}
            cells[f'{level}/{condition}'] = dict(
                runs=len(values), runs_with_skips=sum(bool(s) for s in values.values()),
                skipped_steps_total=sum(len(s) for s in values.values()),
                max_skipped_in_one_run=max((len(s) for s in values.values()), default=0))
    affected = [dict(level=lv, seed=seed, condition=cond, steps=s)
                for (lv, seed, cond), s in sorted(skips.items()) if s]
    return dict(per_cell=cells, affected_runs=affected,
                runs_with_skips=len(affected), skipped_steps_total=sum(len(a['steps']) for a in affected))


def analyse(base):
    from cotv4 import estatistica, runtime
    p = runtime.protocol()
    seeds = runtime.seeds()
    group = runtime.load_primary(base, 'sft')
    skips = skipped_by_job(runtime.job_index(base, 'sft'))
    kept = clean_seeds(skips, seeds)
    s5 = dict(seeds_kept=kept, seeds_excluded=[s for s in seeds if s not in kept], n=len(kept))
    if len(kept) >= 2:
        values = estatistica.paired_sft(group, kept)
        s5.update(mean_interaction=mean(values), sd=stdev(values), per_seed=values,
                  cohen_dz=estatistica.cohen_dz(values), ci95=estatistica.bootstrap_ci(values),
                  signflip_p=estatistica.signflip(values))
    else:
        s5.update(mean_interaction=None, reason='menos de duas sementes sem passos descartados')
    return dict(schema=p['schema'], confirmatory=False, declared_in='DESVIO_01.md',
                skipped_fp16_steps=descriptives(skips), S5_sementes_sem_descartes=s5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage', required=True)
    parser.add_argument('--saida')
    args = parser.parse_args()
    base = Path(args.storage)
    locked = json.loads((base / 'protocol' / 'locked_protocol.json').read_text(encoding='utf-8'))
    os.environ.setdefault('COT_SOURCE_COMMIT', locked['source_commit'] or '')
    if locked['technical_test']:
        os.environ.setdefault('COT_TECHNICAL_TEST', '1')
    result = analyse(base)
    target = Path(args.saida) if args.saida else base / 'analysis' / 'sft' / 'analise_desvio_01.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
