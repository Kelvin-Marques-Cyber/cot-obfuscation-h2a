"""Tabelas do artigo. Cada tabela lê seus valores dos arquivos indicados na fonte.

Retorna {chave: (título, cabeçalho, linhas, alinhamentos, fonte)}. Com lang='en', as
tabelas citadas no artigo saem em inglês; as alterações do v4 vêm de
revisao_2026/traducao/mudancas_v4.en.json, tradução do arquivo congelado.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


def _csv(path):
    with Path(path).open(encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


COND = {'control': 'CoT explícita', 'cot_pressure': 'Resposta direta', 'output_supervision': 'Preâmbulo genérico'}
COND_EN = {'control': 'Explicit CoT', 'cot_pressure': 'Direct answer', 'output_supervision': 'Generic preamble'}
LEVEL = {'sft/A': 'SFT · A', 'sft/C': 'SFT · C', 'grpo/low': 'GRPO · low', 'grpo/high': 'GRPO · high'}


def build(root, repo, num, pct, frac, lang='pt'):
    results = root / 'results'
    tables = {}
    en = lang == 'en'

    def t(pt_text, en_text):
        return en_text if en else pt_text
    cond_names = COND_EN if en else COND

    tables['desenho_piloto'] = (
        'Desenho do estudo-piloto reanalisado',
        ['Elemento', 'Especificação registrada no notebook'],
        [['Modelo', 'Qwen2.5-7B-Instruct quantizado em 4 bits (Unsloth), LoRA r = 16, α = 16'],
         ['Níveis', 'A: adição ou subtração de dois inteiros (11–99); C: problemas de estoque com multiplicação'],
         ['Alvos de SFT', 'CoT explícita; resposta direta; preâmbulo genérico'],
         ['Sementes', '42, 43 e 44 (uma execução por célula)'],
         ['Itens', '150 de treino e 30 de avaliação por execução; 18 execuções; 540 avaliações'],
         ['Treino', '3 épocas, lote 2, acumulação 4, taxa 5 × 10⁻⁵; geração de até 200 tokens']],
        ['l', 'l'], 'notebook original (commit e21cef8), células 5, 13, 15 e 21.')

    summary = _csv(results / 'summary.csv')
    tables['piloto_resumo'] = (
        'Acurácia registrada no estudo-piloto por nível e alvo de treinamento',
        ['Nível', 'Alvo de SFT', 'Acertos', 'n', 'Média', 'DP entre sementes'],
        [[r['level'], COND[r['condition']], r['successes'], r['n'], num(float(r['mean']), 3),
          num(float(r['seed_sd']), 3)] for r in summary],
        ['l', 'l', 'r', 'r', 'r', 'r'], 'revisao_2026/results/summary.csv (reanálise dos dados publicados).')

    paired = _csv(results / 'paired_costs.csv')
    contrasts = {r['condition']: r for r in _csv(results / 'contrasts.csv')}
    rows = [[r['seed'], cond_names[r['condition']], num(float(r['cost_A']), 3), num(float(r['cost_C']), 3),
             num(float(r['difference_in_differences']), 3)] for r in paired]
    for condition, r in contrasts.items():
        rows.append([t('Média', 'Mean'), cond_names[condition], '', '', num(float(r['mean']), 3)])
    tables['piloto_contrastes'] = (
        t('Custo de acurácia em relação à CoT explícita e diferença entre níveis, por semente',
          'Accuracy cost relative to explicit CoT and difference between levels, by seed'),
        [t('Semente', 'Seed'), t('Alvo', 'Target'), t('Custo em A', 'Cost at A'), t('Custo em C', 'Cost at C'), 'C − A'],
        rows, ['l', 'l', 'r', 'r', 'r'], t('revisao_2026/results/paired_costs.csv e contrasts.csv.',
                                           'revisao_2026/results/paired_costs.csv and contrasts.csv.'))

    overlap = _csv(results / 'overlap_audit.csv')
    tables['sobreposicao'] = (
        'Enunciados compartilhados entre sementes no gerador do piloto',
        ['Nível', 'Partição', 'Sementes', 'Interseção', 'União das três'],
        [[r['level'], 'treino' if r['split'] == 'train' else 'avaliação', f"{r['seed_1']} e {r['seed_2']}",
          f"{r['intersection']}/{r['denominator']}", r['union_three']] for r in overlap],
        ['l', 'l', 'l', 'r', 'r'], 'revisao_2026/results/overlap_audit.csv.')

    stress = _csv(results / 'grading_stress_summary.csv')
    kinds = {'numero_correto': 'Número correto', 'numero_incorreto': 'Número incorreto',
             'alternativas': 'Duas alternativas', 'contradicao': 'Contradição',
             'contaminacao': 'Número correto fora da conclusão', 'xml_correto': 'Tag correta',
             'xml_incorreto': 'Tag incorreta', 'xml_duplicado': 'Tags duplicadas'}
    tables['corretor'] = (
        'Aceitações do corretor histórico e do contrato estrito no corpus construído',
        ['Classe', 'n', 'Aceitações esperadas', 'Corretor histórico', 'Contrato estrito'],
        [[kinds[r['kind']], r['n'], r['expected_accepts'], r['original_accepts'], r['strict_accepts']] for r in stress],
        ['l', 'r', 'r', 'r', 'r'], 'revisao_2026/results/grading_stress_summary.csv (2.000 textos construídos).')

    tables['desenho_v3'] = (
        t('Desenho registrado do estudo prospectivo v3.2', 'Registered design of prospective study v3.2'),
        [t('Braço', 'Arm'), t('Modelo-base', 'Base model'), t('Condições × níveis × sementes', 'Conditions × levels × seeds'),
         t('Treinos', 'Training runs'), t('Itens TEST por nível', 'TEST items per level'),
         t('Gerações TEST', 'TEST generations')],
        [['SFT', 'Qwen2.5-1.5B-Instruct', '3 × 2 × 20', '120', '220', num(120 * 220, 0)],
         ['GRPO', 'Qwen3-1.7B', '3 × 2 × 20', '120', '240', num(120 * 240, 0)],
         ['Total', '', '', '240', '', num(120 * 220 + 120 * 240, 0)]],
        ['l', 'l', 'c', 'r', 'r', 'r'],
        t('PROTOCOLO_PRE_REGISTRO_v3.md e scientific_runtime.py (commit fde856f); registro OSF sjwdf.',
          'PROTOCOLO_PRE_REGISTRO_v3.md and scientific_runtime.py (commit fde856f); OSF registration sjwdf.'))

    tables['cronologia'] = (
        'Cronologia do estudo v3.2 reconstruída a partir de registros verificáveis (UTC)',
        ['Data e hora', 'Evento', 'Registro'],
        [['16/10/2026', 'Pacote v3.1 recebido em ZIP', 'VALIDACAO.json (SHA256 b6d9…bee6)'],
         ['19/10 15:20', 'Implementação científica v3.2', 'commit fde856f'],
         ['19/10 15:31', 'Suporte Colab com recuperação de montagem', 'commit 4a2a133'],
         ['19/10 15:57', 'Arquivo anexado e removido do projeto OSF', 'log de 7hzjb'],
         ['19/10 16:11', 'Registro criado a partir do rascunho', 'osf.io/sjwdf'],
         ['19/10 16:45', 'Fim do embargo; registro público', 'log de 7hzjb'],
         ['19/10 17:45', 'Diagnóstico DEV sem nova inferência', 'commits cc2c4ad a 273d131']],
        ['l', 'l', 'l'], 'API pública do OSF (consulta em 28/10/2026) e histórico Git do repositório.')

    diag_path = repo / 'experimento_cientifico_v3' / 'diagnostico_dev' / 'resultados' / 'diagnostico_triagem_dev.json'
    if diag_path.exists():
        diag = _json(diag_path)
        rows = []
        for key, label in LEVEL.items():
            e = diag['levels'][key]
            t4, cpu = e['t4_fp16'], e.get('cpu_fp32')
            rows.append([label, f"{frac(e['documented_correct'], 120)} ({pct(e['documented_accuracy'])})",
                         f"{frac(t4['correct'], 120)} ({pct(t4['accuracy'])})",
                         f"{frac(cpu['correct'], cpu['n'])} ({pct(cpu['correct'] / cpu['n'])})" if cpu
                         else t('não executada', 'not run'),
                         pct(e['permissive_upper_bound'])])
        tables['triagem_v3'] = (
            t('Triagem DEV do v3.2: valores documentados e reproduções com o código congelado',
              'v3.2 DEV screening: documented values and reproductions with the frozen code'),
            [t('Nível', 'Level'), t('Documentado (T4, 19/10)', 'Documented (T4, Oct 19)'),
             t('Reprodução T4 fp16 (28/10)', 'T4 fp16 reproduction (Oct 28)'),
             t('Reprodução CPU fp32', 'CPU fp32 reproduction'), t('Limite permissivo (T4)', 'Permissive bound (T4)')],
            rows, ['l', 'r', 'r', 'r', 'r'],
            t('README do v3 (commit 273d131) e experimento_cientifico_v3/diagnostico_dev/resultados/.',
              'v3 README (commit 273d131) and experimento_cientifico_v3/diagnostico_dev/resultados/.'))
        names = {'correto_estrito': 'Correta no contrato',
                 'valor_correto_fora_do_contrato': 'Valor correto fora do contrato',
                 'alternativa_correta_fora_do_contrato': 'Valor correto fora do contrato',
                 'separador_de_milhar_lido_como_decimal': 'Separador de milhar lido como decimal',
                 'formato_valido_valor_incorreto': 'Formato válido, valor incorreto',
                 'formato_valido_outra_alternativa': 'Formato válido, valor incorreto',
                 'formato_valido_escolheu_dica': 'Formato válido, alternativa da dica',
                 'dica_fora_do_contrato': 'Alternativa da dica, fora do contrato',
                 'formato_invalido_valor_incorreto': 'Fora do contrato, valor incorreto',
                 'outra_alternativa_fora_do_contrato': 'Fora do contrato, valor incorreto',
                 'sem_tag_answer': 'Sem letra extraível (p. ex., "LETRA" copiado)',
                 'sem_numero_final': 'Sem resposta extraível',
                 'truncado_no_orcamento': 'Truncada no orçamento'}
        if en:
            names_en = {'Correta no contrato': 'Correct under the contract',
                        'Valor correto fora do contrato': 'Correct value outside the contract',
                        'Separador de milhar lido como decimal': 'Thousands separator read as decimal',
                        'Formato válido, valor incorreto': 'Valid format, incorrect value',
                        'Formato válido, alternativa da dica': 'Valid format, hint option',
                        'Alternativa da dica, fora do contrato': 'Hint option, outside the contract',
                        'Fora do contrato, valor incorreto': 'Outside the contract, incorrect value',
                        'Sem letra extraível (p. ex., "LETRA" copiado)': 'No extractable letter (e.g., "LETRA" copied)',
                        'Sem resposta extraível': 'No extractable answer',
                        'Truncada no orçamento': 'Truncated at the budget'}
            names = {k: names_en[v] for k, v in names.items()}
        groups = []
        for label in names.values():
            if label not in groups:
                groups.append(label)
        counts = {g: [] for g in groups}
        for key in LEVEL:
            categories = diag['levels'][key]['categories']
            for g in groups:
                counts[g].append(sum(v for k, v in categories.items() if names.get(k) == g))
        rows = [[g] + [str(v) for v in counts[g]] for g in groups if any(counts[g])]
        tables['falhas_v3'] = (
            t('Classificação das respostas da reprodução T4 (fp16) da triagem DEV v3.2',
              'Classification of the responses in the T4 (fp16) reproduction of the v3.2 DEV screening'),
            [t('Categoria', 'Category'), 'SFT · A', 'SFT · C', 'GRPO · low', 'GRPO · high'], rows,
            ['l', 'r', 'r', 'r', 'r'], 'experimento_cientifico_v3/diagnostico_dev/resultados/diagnostico_triagem_dev.json.')
    hyp = repo / 'experimento_cientifico_v3' / 'diagnostico_dev' / 'resultados' / 'hipoteses_tecnicas.json'
    if hyp.exists():
        data = _json(hyp)
        tables['hipoteses'] = ('Hipóteses de defeito técnico examinadas e conclusão',
                               ['Hipótese', 'Verificação', 'Resultado'],
                               [[h['hipotese'], h['verificacao'], h['resultado']] for h in data],
                               ['l', 'l', 'l'], 'experimento_cientifico_v3/diagnostico_dev/RELATORIO_DIAGNOSTICO_DEV.md.')

    v4 = repo / 'experimento_cientifico_v4'
    changes = v4 / 'calibracao' / 'resultados' / 'mudancas_v4.json'
    if changes.exists():
        source = _json(changes)
        if en:
            translated = _json(root / 'traducao' / 'mudancas_v4.en.json')
            if [c['elemento'] for c in translated] != [c['elemento'] for c in source]:
                raise ValueError('traducao/mudancas_v4.en.json não corresponde a mudancas_v4.json')
            source = [dict(elemento=c['element'], v3=c['v3'], v4=c['v4'], justificativa=c['rationale'])
                      for c in translated]
        tables['mudancas_v4'] = (t('Alterações do protocolo v4 em relação ao v3.2', 'Changes in protocol v4 relative to v3.2'),
                                 [t('Elemento', 'Element'), 'v3.2', 'v4', t('Justificativa', 'Rationale')],
                                 [[c['elemento'], c['v3'], c['v4'], c['justificativa']] for c in source],
                                 ['l', 'l', 'l', 'l'], t('experimento_cientifico_v4/PROTOCOLO_v4.md e protocolo_v4.json.',
                                                         'experimento_cientifico_v4/PROTOCOLO_v4.md and protocolo_v4.json.'))
    cal = v4 / 'calibracao' / 'resultados'
    if (cal / 'sft_calibration_decision.json').exists() and (cal / 'grpo_calibration_decision.json').exists():
        label = {'A': 'A: a + b', 'C1': t('C1: c de 3 a 9', 'C1: c from 3 to 9'),
                 'C2': t('C2: b e c de 12 a 99', 'C2: b and c from 12 to 99')}
        rows = []
        for rnd, name in (('1', 'sft_calibration_decision.json'), ('2', 'sft_calibration_decision_r2.json')):
            if not (cal / name).exists():
                continue
            for key, value in _json(cal / name)['summary'].items():
                cand, prompt = key.split('/')
                prompt = ((t('explícita', 'explicit') + (t(' + demonstrações', ' + demonstrations') if rnd == '2' else ''))
                          if prompt == 'explicit' else t('neutra', 'neutral'))
                rows.append([rnd, 'SFT', label[cand], prompt, 'A' if cand == 'A' else 'C', '—',
                             f"{round(value['accuracy'] * value['n'])}/{value['n']}"])
        glabel = {'G1': 'G1: 11–99', 'G0': 'G0: 2–20'}
        instr = {'v3': 'v3.2', 'r2a': 'r2a', 'r2b': 'r2b'}
        for rnd, name in (('1', 'grpo_calibration_decision.json'), ('2', 'grpo_calibration_decision_r2.json')):
            if not (cal / name).exists():
                continue
            for key, value in _json(cal / name)['summary'].items():
                parts = key.split('/')
                variant, cand, cond = (['v3'] + parts) if len(parts) == 2 else parts
                for level in ('low', 'high'):
                    v = value[level]
                    hint = (t('sem dica', 'no hint') if cond == 'sem_dica' else t('com dica', 'with hint'))
                    rows.append([rnd, 'GRPO', glabel[cand], f"{instr[variant]}, {hint}",
                                 level, pct(v['format_valid']), f"{round(v['accuracy'] * v['n'])}/{v['n']}"])
        tables['calibracao_v4'] = (t('Calibração exploratória do v4 em itens próprios (Tesla T4, fp16, 40 itens por linha)',
                                     'Exploratory v4 calibration on separate items (Tesla T4, fp16, 40 items per row)'),
                                   [t('Rodada', 'Round'), t('Braço', 'Arm'), t('Candidato', 'Candidate'),
                                    t('Instrução', 'Instruction'), t('Nível', 'Level'), t('Formato válido', 'Valid format'),
                                    t('Acertos', 'Correct')],
                                   rows, ['c', 'l', 'l', 'l', 'l', 'r', 'r'],
                                   t('experimento_cientifico_v4/calibracao/resultados/ (arquivos de decisão de cada rodada). '
                                     'Nota: no SFT, A = a + b com operandos de 100 a 999 e C = a + b·c − d com a e d de '
                                     '100 a 999 e b de 12 a 99; no GRPO, G1 e G0 são as faixas dos operandos.',
                                     'experimento_cientifico_v4/calibracao/resultados/ (decision files of each round). '
                                     'Note: in SFT, A = a + b with operands from 100 to 999 and C = a + b·c − d with a and d '
                                     'from 100 to 999 and b from 12 to 99; in GRPO, G1 and G0 are the operand ranges.'))
    monitors = cal / 'monitor_feasibility_decision.json'
    if monitors.exists():
        first = _json(monitors)
        second = _json(cal / 'monitor_feasibility_decision_r2.json') if (cal / 'monitor_feasibility_decision_r2.json').exists() else {}
        chosen = second.get('chosen_independent_monitor') or first.get('chosen_independent_monitor')
        reward = second.get('reward_monitor')

        def role(model_id):
            if 'Qwen2.5' in model_id:
                return t('recompensa', 'reward') + (t(' (escolhido)', ' (chosen)') if model_id == reward else '')
            if model_id == chosen:
                return t('independente (escolhido)', 'independent (chosen)')
            return (t('independente no v3.2', 'independent in v3.2') if '135M' in model_id
                    else t('candidato a independente', 'independent candidate'))

        rows = []
        for rnd, data in (('1', first), ('2', second)):
            for m in data.get('summary', []):
                gpu = ' (L4)' if 'L4' in str(m.get('hardware', '')) else ''
                rows.append([rnd + gpu, m['model_id'].split('/')[-1], role(m['model_id']),
                             pct(m['tpr']) if m['tpr'] is not None else t('não avaliável', 'not evaluable'),
                             pct(m['fpr']) if m['fpr'] is not None else t('não avaliável', 'not evaluable')])
        source = 'experimento_cientifico_v4/calibracao/resultados/monitor_feasibility_decision*.json.'
        third = repo / 'preparacao_v5' / 'resultados' / 'monitores_rodada3' / 'monitor_feasibility_decision_r3.json'
        if third.exists():
            r3 = _json(third)
            for m in r3['summary']:
                if m['role'] == 'reward':
                    papel = t('recompensa', 'reward') + (t(' (escolhido)', ' (chosen)')
                                                         if m['model_id'] == r3.get('reward_monitor') else '')
                else:
                    papel = (t('independente (escolhido)', 'independent (chosen)')
                             if m['model_id'] == r3.get('chosen_independent_monitor')
                             else t('candidato a independente', 'independent candidate'))
                rows.append(['3 (A100)', m['model_id'].split('/')[-1], papel,
                             pct(m['tpr']) if m['tpr'] is not None else t('não avaliável', 'not evaluable'),
                             pct(m['fpr']) if m['fpr'] is not None else t('não avaliável', 'not evaluable')])
            source = t('experimento_cientifico_v4/calibracao/resultados/monitor_feasibility_decision*.json (rodadas 1 e 2, '
                       'antes do congelamento do v4) e preparacao_v5/resultados/monitores_rodada3/ (rodada 3, exploratória, '
                       'depois do congelamento; não altera o v4).',
                       'experimento_cientifico_v4/calibracao/resultados/monitor_feasibility_decision*.json (rounds 1 and 2, '
                       'before the v4 freeze) and preparacao_v5/resultados/monitores_rodada3/ (round 3, exploratory, after '
                       'the freeze; does not change v4).')
        tables['monitores_v4'] = (
            t('Desempenho dos monitores em 320 controles construídos (rótulos por regra, não humanos)',
              'Monitor performance on 320 constructed controls (rule-based labels, not human)'),
            [t('Rodada', 'Round'), t('Modelo', 'Model'), t('Papel', 'Role'), 'TPR', 'FPR'], rows,
            ['c', 'l', 'l', 'r', 'r'], source)
    gate = v4 / 'triagem_dev' / 'manifest.json'
    if gate.exists():
        data = _json(gate)
        protocol = _json(v4 / 'protocolo_v4.json')
        threshold = protocol['gate']['threshold']
        rows = []
        for mode in ('sft', 'grpo'):
            active = protocol[mode].get('enabled', True)
            for level, e in data['summary'][mode].items():
                verdict = t('aprovado', 'passed') if e['accuracy'] >= threshold else t('reprovado', 'failed')
                rows.append([f'{mode.upper()} · {level}', f"{e['correct']}/{e['n']}", pct(e['accuracy']),
                             f"{e['format_valid']}/{e['n']}", str(e['truncated']),
                             verdict if active else t('não se aplica (descritivo)', 'not applicable (descriptive)')])
        tables['triagem_v4'] = (t('Triagem DEV do protocolo v4, executada uma vez antes do registro',
                                  'DEV screening of protocol v4, run once before registration'),
                                [t('Nível', 'Level'), t('Acertos', 'Correct'), t('Acurácia', 'Accuracy'),
                                 t('Formato válido', 'Valid format'), t('Truncadas', 'Truncated'),
                                 t('Critério 0,65', 'Criterion 0.65')],
                                rows, ['l', 'r', 'r', 'r', 'r', 'l'],
                                'experimento_cientifico_v4/triagem_dev/manifest.json.')
    secondary = repo / 'experimento_cientifico_v4' / 'resultados_coleta' / 'analises_secundarias.json'
    if secondary.exists():
        s = _json(secondary)
        names = {'explicit': t('Rastro explícito', 'Explicit trace'), 'direct': t('Resposta direta', 'Direct answer'),
                 'generic': t('Preâmbulo genérico', 'Generic preamble')}
        rows = []
        for level in ('A', 'C'):
            for condition in ('explicit', 'direct', 'generic'):
                key = f'{level}/{condition}'
                acc, fmt, trunc = s['S2_acuracia'][key], s['S4_formato_valido'][key], s['S4_truncamento'][key]
                rows.append([level, names[condition], f"{pct(acc['mean'])} ({pct(acc['sd'])})",
                             pct(fmt['mean']), pct(trunc['mean'])])
        tables['resultados_v4'] = (
            t('Acurácia TEST do v4 por nível e alvo de treinamento (20 sementes, 220 itens por nível)',
              'v4 TEST accuracy by level and training target (20 seeds, 220 items per level)'),
            [t('Nível', 'Level'), t('Alvo de treinamento', 'Training target'),
             t('Acurácia média (DP entre sementes)', 'Mean accuracy (SD across seeds)'),
             t('Formato válido', 'Valid format'), t('Truncadas', 'Truncated')],
            rows, ['l', 'l', 'r', 'r', 'r'],
            t('análises S2 e S4 (experimento_cientifico_v4/resultados_coleta/analises_secundarias.json).',
              'analyses S2 and S4 (experimento_cientifico_v4/resultados_coleta/analises_secundarias.json).'))
    validation = root / 'results' / 'validacao_2026_10_28.json'
    if validation.exists():
        data = _json(validation)
        tables['verificacoes'] = ('Verificações de software e execuções técnicas realizadas nesta revisão',
                                  ['Verificação', 'Ambiente', 'Resultado'],
                                  [[v['verificacao'], v['ambiente'], v['resultado']] for v in data['tabela']],
                                  ['l', 'l', 'l'], 'revisao_2026/results/validacao_2026_10_28.json.')
        tables['evidencias'] = ('Natureza da evidência que sustenta cada conclusão',
                                ['Conclusão', 'Categoria', 'Base'],
                                [[v['conclusao'], v['categoria'], v['base']] for v in data['evidencias']],
                                ['l', 'l', 'l'], 'síntese desta revisão; categorias definidas na seção 4.')
    return tables
