"""Análises secundárias do v4 (não confirmatórias), declaradas no registro público.

Uso: python analises_secundarias.py --storage PASTA_DO_ESTUDO [--saida arquivo.json]

Lê as avaliações TEST do braço SFT com as mesmas verificações de cobertura e
pareamento do desfecho primário (runtime.load_primary) e calcula:

S1. Sensibilidade: o contraste H1 sem os itens TEST usados nos testes técnicos
    de GPU (os `technical_eval_items` primeiros itens TEST de cada nível, na
    ordem congelada dos dados).
S2. Acurácia por nível e condição (média e desvio-padrão entre sementes).
S3. Contrastes com a condição generic, na mesma forma da interação de H1:
    generic - explicit e direct - generic.
S4. Formato válido e truncamento por nível e condição.

Nenhum destes resultados altera o veredito de final_report; valores-p de S1
e S3 são descritivos, sem correção para multiplicidade.
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


def exposed_items(rows, per_level):
    """Itens TEST avaliados no modo técnico: os primeiros de cada nível."""
    test = [r for r in rows if r['split'] == 'test']
    return {level: [r['item_id'] for r in test if r['level'] == level][:per_level] for level in ('A', 'C')}


def interaction(group, seeds, minuend, subtrahend, keep=None):
    """(minuend - subtrahend)_C - (minuend - subtrahend)_A por semente, opcionalmente num subconjunto de itens."""
    values = []
    for seed in seeds:
        by_level = {}
        for level in ('A', 'C'):
            a, b = group[level, seed, minuend], group[level, seed, subtrahend]
            items = sorted(set(a) & set(b) if keep is None else set(a) & set(b) & keep[level])
            if not items:
                raise RuntimeError(f'Sem itens pareados em {level}, semente {seed}')
            by_level[level] = mean(int(bool(a[k]['accuracy'])) - int(bool(b[k]['accuracy'])) for k in items)
        values.append(by_level['C'] - by_level['A'])
    return values


def summarize(values, estatistica):
    return dict(n=len(values), mean=mean(values), sd=stdev(values) if len(values) > 1 else None,
                ci95=estatistica.bootstrap_ci(values), signflip_p=estatistica.signflip(values))


def by_cell(group, seeds, field):
    out = {}
    for level in ('A', 'C'):
        for condition in ('explicit', 'direct', 'generic'):
            per_seed = [mean(int(field(r)) for r in group[level, seed, condition].values()) for seed in seeds]
            out[f'{level}/{condition}'] = dict(mean=mean(per_seed),
                                              sd=stdev(per_seed) if len(per_seed) > 1 else None)
    return out


def analyse(base):
    from cotv4 import estatistica, runtime
    p = runtime.protocol()
    seeds = runtime.seeds()
    group = runtime.load_primary(base, 'sft')
    rows = runtime.read_jsonl(Path(base) / 'data' / 'sft.jsonl')
    exposed = exposed_items(rows, p['training']['technical_eval_items'])
    keep = {level: {r['item_id'] for r in rows if r['split'] == 'test' and r['level'] == level} - set(exposed[level])
            for level in ('A', 'C')}
    return dict(
        schema=p['schema'], confirmatory=False, seeds=seeds,
        S1_sem_itens_do_teste_tecnico=dict(excluded_item_ids=exposed,
                                           items_per_level={lv: len(keep[lv]) for lv in keep},
                                           **summarize(interaction(group, seeds, 'direct', 'explicit', keep),
                                                       estatistica)),
        S2_acuracia=by_cell(group, seeds, lambda r: bool(r['accuracy'])),
        S3_generic=dict(generic_menos_explicit=summarize(interaction(group, seeds, 'generic', 'explicit'), estatistica),
                        direct_menos_generic=summarize(interaction(group, seeds, 'direct', 'generic'), estatistica)),
        S4_formato_valido=by_cell(group, seeds, lambda r: bool(r['format_valid'])),
        S4_truncamento=by_cell(group, seeds, lambda r: r['finish_reason'] == 'length'),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage', required=True)
    parser.add_argument('--saida')
    args = parser.parse_args()
    base = Path(args.storage)
    # O código precisa estar no mesmo commit congelado no estudo.
    locked = json.loads((base / 'protocol' / 'locked_protocol.json').read_text(encoding='utf-8'))
    os.environ.setdefault('COT_SOURCE_COMMIT', locked['source_commit'] or '')
    if locked['technical_test']:
        os.environ.setdefault('COT_TECHNICAL_TEST', '1')
    result = analyse(base)
    target = Path(args.saida) if args.saida else base / 'analysis' / 'sft' / 'analises_secundarias.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
