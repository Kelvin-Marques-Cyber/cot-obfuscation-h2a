"""Inferência por semente de treinamento: inversão de sinais, bootstrap e Holm.

Mesmos procedimentos registrados no v3.2. A unidade de análise é a semente;
as gerações de uma mesma semente nunca são tratadas como réplicas independentes.
"""
from __future__ import annotations

import itertools
import random
from statistics import mean


def signflip(values, iterations=199_999, seed=20261015):
    """Teste bilateral pareado por inversão de sinais da média dos contrastes."""
    if len(values) < 2:
        raise ValueError('Sementes insuficientes')
    observed = abs(mean(values))
    if len(values) <= 18:
        hits = sum(abs(sum(v * s for v, s in zip(values, signs)) / len(values)) >= observed - 1e-12
                   for signs in itertools.product((-1, 1), repeat=len(values)))
        return hits / 2 ** len(values)
    rng = random.Random(seed)
    hits = 0
    for _ in range(iterations):
        simulated = abs(sum(v * (1 if rng.getrandbits(1) else -1) for v in values) / len(values))
        hits += simulated >= observed - 1e-12
    return (hits + 1) / (iterations + 1)


def bootstrap_ci(values, seed=20261015, iterations=20_000):
    """Intervalo percentil de 95% reamostrando sementes."""
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(iterations))
    return [draws[int((iterations - 1) * 0.025)], draws[int((iterations - 1) * 0.975)]]


def holm(pvalues, alpha=0.05):
    """Rejeições de Holm para um dicionário {nome: p}."""
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    rejected = {name: False for name in pvalues}
    for rank, (name, value) in enumerate(ordered):
        if value <= alpha / (len(ordered) - rank):
            rejected[name] = True
        else:
            break
    return rejected


def paired_sft(group, seeds):
    """Interação por semente: (direct - explicit)_C - (direct - explicit)_A."""
    observations = []
    for seed in seeds:
        by_level = {}
        for level in ('A', 'C'):
            explicit = group[level, seed, 'explicit']
            direct = group[level, seed, 'direct']
            if set(explicit) != set(direct):
                raise RuntimeError('Itens de teste não pareados')
            by_level[level] = mean(int(bool(direct[k]['accuracy'])) - int(bool(explicit[k]['accuracy']))
                                   for k in explicit)
        observations.append(by_level['C'] - by_level['A'])
    return observations


def cohen_dz(values):
    """Tamanho de efeito pareado (média/desvio-padrão dos contrastes por semente)."""
    if len(values) < 2:
        return None
    m = mean(values)
    sd = (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5
    return None if sd == 0 else m / sd
