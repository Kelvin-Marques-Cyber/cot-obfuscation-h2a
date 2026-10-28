"""Figuras do artigo, geradas dos arquivos de resultados (tons de cinza, 300 dpi).

python src/gerar_figuras.py  (a partir de revisao_2026/)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
OUT = ROOT / 'figures'
DIAG = REPO / 'experimento_cientifico_v3' / 'diagnostico_dev' / 'resultados' / 'diagnostico_triagem_dev.json'
V4 = REPO / 'experimento_cientifico_v4'
GRAYS = ['#1a1a1a', '#6e6e6e', '#b3b3b3', '#e0e0e0']
HATCHES = ['', '///', '...', 'xx']
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'DejaVu Serif'],
                     'font.size': 9, 'axes.edgecolor': '#333333', 'axes.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300,
                     'legend.frameon': False, 'hatch.linewidth': 0.5})


def pct(value):
    return f'{100 * value:.1f}'.replace('.', ',')


def decimal_comma(ax):
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.1f}'.replace('.', ',')))


def save(fig, name):
    for ax in fig.axes:
        decimal_comma(ax)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return OUT / name


def figura_piloto():
    rows = list(csv.DictReader((ROOT / 'results' / 'summary.csv').open(encoding='utf-8')))
    seeds = list(csv.DictReader((ROOT / 'results' / 'reanalyzed_cells.csv').open(encoding='utf-8')))
    names = {'control': 'CoT explícita', 'cot_pressure': 'Resposta direta', 'output_supervision': 'Preâmbulo genérico'}
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    width = 0.26
    for index, condition in enumerate(names):
        means = [float(next(r for r in rows if r['level'] == lv and r['condition'] == condition)['mean'])
                 for lv in ('A', 'C')]
        xs = [lv + (index - 1) * width for lv in range(2)]
        ax.bar(xs, means, width, color=GRAYS[index], edgecolor='black', linewidth=0.5,
               hatch=HATCHES[index], label=names[condition])
        for level_index, level in enumerate(('A', 'C')):
            points = [float(r['accuracy']) for r in seeds if r['level'] == level and r['condition'] == condition]
            ax.scatter([xs[level_index]] * len(points), points, s=9, color='white', edgecolor='black',
                       linewidth=0.6, zorder=3)
    ax.set_xticks([0, 1], ['Nível A', 'Nível C'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Acurácia registrada')
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.18))
    return save(fig, 'fig1_piloto_acuracia.png')


def figura_triagem():
    data = json.loads(DIAG.read_text(encoding='utf-8'))
    keys = ['sft/A', 'sft/C', 'grpo/low', 'grpo/high']
    labels = ['SFT A', 'SFT C', 'GRPO low', 'GRPO high']
    series = [('Documentada (T4, 19/10)', [data['levels'][k]['documented_accuracy'] for k in keys])]
    if all('t4_fp16' in data['levels'][k] for k in keys):
        series.append(('Reprodução T4 fp16', [data['levels'][k]['t4_fp16']['accuracy'] for k in keys]))
    series.append(('Reprodução CPU fp32', [data['levels'][k]['reproduced_accuracy'] for k in keys]))
    series.append(('Limite permissivo (CPU)', [data['levels'][k]['permissive_upper_bound'] for k in keys]))
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    width = 0.8 / len(series)
    for index, (name, values) in enumerate(series):
        xs = [i + (index - (len(series) - 1) / 2) * width for i in range(len(keys))]
        ax.bar(xs, values, width, color=GRAYS[index], edgecolor='black', linewidth=0.5,
               hatch=HATCHES[index], label=name)
    ax.axhline(data['threshold'], color='black', linestyle='--', linewidth=0.8)
    ax.text(len(keys) - 0.45, data['threshold'] + 0.02, 'critério 0,65', ha='right', fontsize=8)
    ax.set_xticks(range(len(keys)), labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Acurácia DEV (n = 120)')
    ax.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.25))
    return save(fig, 'fig2_triagem_v3.png')


def figura_falhas():
    data = json.loads(DIAG.read_text(encoding='utf-8'))
    order = [('correto_estrito', 'Correta no contrato'),
             ('valor_correto_fora_do_contrato', 'Valor correto fora do contrato'),
             ('alternativa_correta_fora_do_contrato', 'Valor correto fora do contrato'),
             ('separador_de_milhar_lido_como_decimal', 'Separador de milhar'),
             ('formato_valido_valor_incorreto', 'Erro de cálculo ou escolha'),
             ('formato_valido_outra_alternativa', 'Erro de cálculo ou escolha'),
             ('formato_valido_escolheu_dica', 'Seguiu a dica proibida'),
             ('dica_fora_do_contrato', 'Seguiu a dica proibida'),
             ('formato_invalido_valor_incorreto', 'Formato inválido e valor incorreto'),
             ('outra_alternativa_fora_do_contrato', 'Formato inválido e valor incorreto'),
             ('sem_tag_answer', 'Sem resposta extraível'), ('sem_numero_final', 'Sem resposta extraível'),
             ('truncado_no_orcamento', 'Truncada no orçamento')]
    groups = []
    for _, label in order:
        if label not in groups:
            groups.append(label)
    keys = ['sft/A', 'sft/C', 'grpo/low', 'grpo/high']
    labels = ['SFT A', 'SFT C', 'GRPO low', 'GRPO high']
    totals = {g: [] for g in groups}
    for key in keys:
        categories = data['levels'][key]['categories']
        for group in groups:
            totals[group].append(sum(categories.get(c, 0) for c, g in order if g == group))
    present = [g for g in groups if any(totals[g])]
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    bottom = [0] * len(keys)
    shades = ['#ffffff', '#d9d9d9', '#bfbfbf', '#8c8c8c', '#595959', '#262626', '#a6a6a6', '#f0f0f0']
    hatch = ['', '', '///', '', '...', '', 'xx', '\\\\\\']
    for index, group in enumerate(present):
        ax.bar(labels, totals[group], bottom=bottom, color=shades[index % len(shades)], edgecolor='black',
               linewidth=0.5, hatch=hatch[index % len(hatch)], label=group)
        bottom = [b + v for b, v in zip(bottom, totals[group])]
    ax.set_ylabel('Respostas (de 120)')
    ax.set_ylim(0, 120)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8)
    return save(fig, 'fig3_falhas_v3.png')


def figura_v4():
    manifest = V4 / 'triagem_dev' / 'manifest.json'
    if not manifest.exists():
        return None
    data = json.loads(manifest.read_text(encoding='utf-8'))
    protocol = json.loads((V4 / 'protocolo_v4.json').read_text(encoding='utf-8'))
    keys = [('sft', 'A'), ('sft', 'C'), ('grpo', 'low'), ('grpo', 'high')]
    values = [data['summary'][m][lv]['accuracy'] for m, lv in keys]
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    ax.bar([f'{m.upper()} {lv}' for m, lv in keys], values, 0.55, color=GRAYS[1], edgecolor='black', linewidth=0.5)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, pct(v) + '%', ha='center', fontsize=8)
    ax.axhline(protocol['gate']['threshold'], color='black', linestyle='--', linewidth=0.8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Acurácia DEV (n = 120)')
    return save(fig, 'fig4_triagem_v4.png')


def main():
    made = [figura_piloto()]
    if DIAG.exists():
        made += [figura_triagem(), figura_falhas()]
    made.append(figura_v4())
    for path in made:
        if path:
            print(path.relative_to(REPO))


if __name__ == '__main__':
    main()
