"""Figuras do artigo, geradas dos arquivos de resultados (tons de cinza, 300 dpi).

python src/gerar_figuras.py [en]  (a partir de revisao_2026/; em inglês, grava em figures/en/)
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
LANG = 'pt'
# Textos das figuras em inglês; a chave é o texto em português.
EN = {
    'CoT explícita': 'Explicit CoT', 'Resposta direta': 'Direct answer', 'Preâmbulo genérico': 'Generic preamble',
    'Nível A': 'Level A', 'Nível C': 'Level C', 'Acurácia registrada': 'Recorded accuracy',
    'Documentada (T4, 19/10)': 'Documented (T4, Oct 19)', 'Reprodução T4 fp16 (28/10)': 'T4 fp16 reproduction (Oct 28)',
    'Reprodução CPU fp32': 'CPU fp32 reproduction', 'Limite permissivo': 'Permissive bound',
    'critério 0,65': '0.65 criterion', 'Acurácia DEV (n = 120)': 'DEV accuracy (n = 120)',
    'Correta no contrato': 'Correct under the contract', 'Valor correto fora do contrato': 'Correct value outside the contract',
    'Separador de milhar': 'Thousands separator', 'Erro de cálculo ou escolha': 'Calculation or choice error',
    'Seguiu a dica proibida': 'Followed the forbidden hint',
    'Formato inválido e valor incorreto': 'Invalid format and incorrect value',
    'Sem letra extraível': 'No extractable letter', 'Sem resposta extraível': 'No extractable answer',
    'Truncada no orçamento': 'Truncated at the budget', 'Respostas (de 120)': 'Responses (of 120)',
    'condiciona os treinos': 'gates the training runs', 'descritivo (braço excluído)': 'descriptive (arm excluded)',
    'Rastro explícito': 'Explicit trace', 'Acurácia TEST (média ± DP)': 'TEST accuracy (mean ± SD)',
    'Contraste de H1 (p.p.)': 'H1 contrast (pp)', 'Sementes (ordenadas)': 'Seeds (sorted)',
    'média e IC 95%': 'mean and 95% CI', 'efeito mínimo': 'minimum effect',
}


def tr(text):
    return EN.get(text, text) if LANG == 'en' else text


def decimal(text):
    return text if LANG == 'en' else text.replace('.', ',')
DIAG = REPO / 'experimento_cientifico_v3' / 'diagnostico_dev' / 'resultados' / 'diagnostico_triagem_dev.json'
V4 = REPO / 'experimento_cientifico_v4'
GRAYS = ['#1a1a1a', '#6e6e6e', '#b3b3b3', '#e0e0e0']
HATCHES = ['', '///', '...', 'xx']
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'DejaVu Serif'],
                     'font.size': 9, 'axes.edgecolor': '#333333', 'axes.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300,
                     'legend.frameon': False, 'hatch.linewidth': 0.5})


def pct(value):
    return decimal(f'{100 * value:.1f}')


def decimal_comma(ax):
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: decimal(f'{v:.1f}')))


def save(fig, name):
    for ax in fig.axes:
        decimal_comma(ax)
    out = OUT / 'en' if LANG == 'en' else OUT
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out / name


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
               hatch=HATCHES[index], label=tr(names[condition]))
        for level_index, level in enumerate(('A', 'C')):
            points = [float(r['accuracy']) for r in seeds if r['level'] == level and r['condition'] == condition]
            ax.scatter([xs[level_index]] * len(points), points, s=9, color='white', edgecolor='black',
                       linewidth=0.6, zorder=3)
    ax.set_xticks([0, 1], [tr('Nível A'), tr('Nível C')])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(tr('Acurácia registrada'))
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.18))
    return save(fig, 'fig1_piloto_acuracia.png')


def figura_triagem():
    data = json.loads(DIAG.read_text(encoding='utf-8'))
    keys = ['sft/A', 'sft/C', 'grpo/low', 'grpo/high']
    labels = ['SFT A', 'SFT C', 'GRPO low', 'GRPO high']
    series = [('Documentada (T4, 19/10)', [data['levels'][k]['documented_accuracy'] for k in keys]),
              ('Reprodução T4 fp16 (28/10)', [data['levels'][k]['t4_fp16']['accuracy'] for k in keys])]
    if all('cpu_fp32' in data['levels'][k] for k in keys):
        series.append(('Reprodução CPU fp32', [data['levels'][k]['cpu_fp32']['correct'] /
                                               data['levels'][k]['cpu_fp32']['n'] for k in keys]))
    series.append(('Limite permissivo', [data['levels'][k]['permissive_upper_bound'] for k in keys]))
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    width = 0.8 / len(series)
    for index, (name, values) in enumerate(series):
        xs = [i + (index - (len(series) - 1) / 2) * width for i in range(len(keys))]
        ax.bar(xs, values, width, color=GRAYS[index], edgecolor='black', linewidth=0.5,
               hatch=HATCHES[index], label=tr(name))
    ax.axhline(data['threshold'], color='black', linestyle='--', linewidth=0.8)
    ax.text(len(keys) - 0.45, data['threshold'] + 0.02, tr('critério 0,65'), ha='right', fontsize=8)
    ax.set_xticks(range(len(keys)), labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(tr('Acurácia DEV (n = 120)'))
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
             ('sem_tag_answer', 'Sem letra extraível'), ('sem_numero_final', 'Sem resposta extraível'),
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
               linewidth=0.5, hatch=hatch[index % len(hatch)], label=tr(group))
        bottom = [b + v for b, v in zip(bottom, totals[group])]
    ax.set_ylabel(tr('Respostas (de 120)'))
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
    excluded = not protocol['grpo'].get('enabled', True)
    bars = ax.bar([f'{m.upper()} {lv}' for m, lv in keys], values, 0.55, color=GRAYS[1], edgecolor='black',
                  linewidth=0.5)
    if excluded:  # braço excluído antes do registro: triagem apenas descritiva
        for bar in bars[2:]:
            bar.set_facecolor('white')
            bar.set_hatch('///')
        ax.legend([bars[0], bars[2]], [tr('condiciona os treinos'), tr('descritivo (braço excluído)')], fontsize=7,
                  loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, pct(v) + '%', ha='center', fontsize=8)
    ax.axhline(protocol['gate']['threshold'], color='black', linestyle='--', linewidth=0.8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(tr('Acurácia DEV (n = 120)'))
    return save(fig, 'fig4_triagem_v4.png')


def figura_h1():
    """Teste confirmatório de H1: acurácia por célula (S2) e contraste D1 por semente."""
    coleta = V4 / 'resultados_coleta'
    if not (coleta / 'final_verdict.json').exists() or not (coleta / 'analises_secundarias.json').exists():
        return None
    verdict = json.loads((coleta / 'final_verdict.json').read_text(encoding='utf-8'))
    s2 = json.loads((coleta / 'analises_secundarias.json').read_text(encoding='utf-8'))['S2_acuracia']
    protocol = json.loads((V4 / 'protocolo_v4.json').read_text(encoding='utf-8'))
    sft = verdict['statistical_analysis']['sft']
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw=dict(width_ratios=[1, 1.25]))
    conditions = [('explicit', 'Rastro explícito'), ('direct', 'Resposta direta'), ('generic', 'Preâmbulo genérico')]
    width = 0.26
    for i, (condition, name) in enumerate(conditions):
        xs = [j + (i - 1) * width for j in range(2)]
        cells = [s2[f'{level}/{condition}'] for level in ('A', 'C')]
        left.bar(xs, [c['mean'] for c in cells], width, yerr=[c['sd'] for c in cells], color=GRAYS[i],
                 edgecolor='black', linewidth=0.5, hatch=HATCHES[i], label=tr(name),
                 error_kw=dict(elinewidth=0.6, capsize=2))
    left.set_xticks([0, 1], [tr('Nível A'), tr('Nível C')])
    left.set_ylim(0, 1.05)
    left.set_ylabel(tr('Acurácia TEST (média ± DP)'))
    left.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False,
                handlelength=1.2, columnspacing=0.8)
    values = sorted(100 * v for v in sft['per_seed'])
    xs = range(1, len(values) + 1)
    low, high = (100 * v for v in sft['ci95'])
    right.axhspan(low, high, color=GRAYS[3], label=tr('média e IC 95%'))
    right.axhline(100 * sft['mean_interaction'], color='black', linewidth=0.9)
    right.axhline(0, color='#666666', linewidth=0.6)
    right.axhline(-100 * protocol['analysis']['minimum_effect'], color='black', linestyle='--', linewidth=0.8,
                  label=tr('efeito mínimo'))
    right.scatter(xs, values, s=14, color='black', zorder=3)
    right.set_ylim(min(values) - 8, 8)
    right.set_xticks([])
    right.set_xlabel(tr('Sementes (ordenadas)'))
    right.set_ylabel(tr('Contraste de H1 (p.p.)'))
    right.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    fig.subplots_adjust(wspace=0.35)
    return save(fig, 'fig5_h1_v4.png')


def main(lang='pt'):
    global LANG
    LANG = lang
    made = [figura_piloto()]
    if DIAG.exists():
        made += [figura_triagem(), figura_falhas()]
    made.append(figura_v4())
    made.append(figura_h1())
    for path in made:
        if path:
            print(path.relative_to(REPO))


if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else 'pt')
