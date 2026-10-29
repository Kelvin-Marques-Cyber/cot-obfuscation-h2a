"""Tabelas do artigo. Cada tabela lê seus valores dos arquivos indicados na fonte.

Retorna {chave: (título, cabeçalho, linhas, alinhamentos, fonte)}.
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
LEVEL = {'sft/A': 'SFT · A', 'sft/C': 'SFT · C', 'grpo/low': 'GRPO · low', 'grpo/high': 'GRPO · high'}


def build(root, repo, num, pct, frac):
    results = root / 'results'
    tables = {}

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
    rows = [[r['seed'], COND[r['condition']], num(float(r['cost_A']), 3), num(float(r['cost_C']), 3),
             num(float(r['difference_in_differences']), 3)] for r in paired]
    for condition, r in contrasts.items():
        rows.append(['Média', COND[condition], '', '', num(float(r['mean']), 3)])
    tables['piloto_contrastes'] = (
        'Custo de acurácia em relação à CoT explícita e diferença entre níveis, por semente',
        ['Semente', 'Alvo', 'Custo em A', 'Custo em C', 'C − A'],
        rows, ['l', 'l', 'r', 'r', 'r'], 'revisao_2026/results/paired_costs.csv e contrasts.csv.')

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
        'Desenho registrado do estudo prospectivo v3.2',
        ['Braço', 'Modelo-base', 'Condições × níveis × sementes', 'Treinos', 'Itens TEST por nível', 'Gerações TEST'],
        [['SFT', 'Qwen2.5-1.5B-Instruct', '3 × 2 × 20', '120', '220', num(120 * 220, 0)],
         ['GRPO', 'Qwen3-1.7B', '3 × 2 × 20', '120', '240', num(120 * 240, 0)],
         ['Total', '', '', '240', '', num(120 * 220 + 120 * 240, 0)]],
        ['l', 'l', 'c', 'r', 'r', 'r'],
        'PROTOCOLO_PRE_REGISTRO_v3.md e scientific_runtime.py (commit fde856f); registro OSF sjwdf.')

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
                         f"{frac(cpu['correct'], cpu['n'])} ({pct(cpu['correct'] / cpu['n'])})" if cpu else 'não executada',
                         pct(e['permissive_upper_bound'])])
        tables['triagem_v3'] = (
            'Triagem DEV do v3.2: valores documentados e reproduções com o código congelado',
            ['Nível', 'Documentado (T4, 19/10)', 'Reprodução T4 fp16 (28/10)', 'Reprodução CPU fp32',
             'Limite permissivo (T4)'],
            rows, ['l', 'r', 'r', 'r', 'r'],
            'README do v3 (commit 273d131) e experimento_cientifico_v3/diagnostico_dev/resultados/.')
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
            'Classificação das respostas da reprodução T4 (fp16) da triagem DEV v3.2',
            ['Categoria', 'SFT · A', 'SFT · C', 'GRPO · low', 'GRPO · high'], rows,
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
        tables['mudancas_v4'] = ('Alterações do protocolo v4 em relação ao v3.2',
                                 ['Elemento', 'v3.2', 'v4', 'Justificativa'],
                                 [[c['elemento'], c['v3'], c['v4'], c['justificativa']] for c in _json(changes)],
                                 ['l', 'l', 'l', 'l'], 'experimento_cientifico_v4/PROTOCOLO_v4.md e protocolo_v4.json.')
    cal = v4 / 'calibracao' / 'resultados'
    if (cal / 'sft_calibration_decision.json').exists() and (cal / 'grpo_calibration_decision.json').exists():
        label = {'A': 'A: a + b (100–999)', 'C1': 'C1: a + b·c − d, c de 3 a 9',
                 'C2': 'C2: a + b·c − d, b e c de 12 a 99'}
        rows = []
        for rnd, name in (('1', 'sft_calibration_decision.json'), ('2', 'sft_calibration_decision_r2.json')):
            if not (cal / name).exists():
                continue
            for key, value in _json(cal / name)['summary'].items():
                cand, prompt = key.split('/')
                prompt = ('explícita' + (' + demonstrações' if rnd == '2' else '')) if prompt == 'explicit' else 'neutra'
                rows.append([rnd, 'SFT', label[cand], prompt, '—', '—',
                             f"{round(value['accuracy'] * value['n'])}/{value['n']}"])
        glabel = {'G1': 'G1: operandos 11–99', 'G0': 'G0: operandos 2–20'}
        instr = {'v3': 'v3.2', 'r2a': 'r2a', 'r2b': 'r2b'}
        for rnd, name in (('1', 'grpo_calibration_decision.json'), ('2', 'grpo_calibration_decision_r2.json')):
            if not (cal / name).exists():
                continue
            for key, value in _json(cal / name)['summary'].items():
                parts = key.split('/')
                variant, cand, cond = (['v3'] + parts) if len(parts) == 2 else parts
                for level in ('low', 'high'):
                    v = value[level]
                    rows.append([rnd, 'GRPO', glabel[cand], f"{instr[variant]}, {'sem' if cond == 'sem_dica' else 'com'} dica",
                                 level, pct(v['format_valid']), f"{round(v['accuracy'] * v['n'])}/{v['n']}"])
        tables['calibracao_v4'] = ('Calibração exploratória do v4 em itens próprios (Tesla T4, fp16, 40 itens por linha)',
                                   ['Rodada', 'Braço', 'Candidato', 'Instrução', 'Nível', 'Formato válido', 'Acertos'],
                                   rows, ['c', 'l', 'l', 'l', 'l', 'r', 'r'],
                                   'experimento_cientifico_v4/calibracao/resultados/ (arquivos de decisão de cada rodada).')
    monitors = cal / 'monitor_feasibility_decision.json'
    if monitors.exists():
        first = _json(monitors)
        second = _json(cal / 'monitor_feasibility_decision_r2.json') if (cal / 'monitor_feasibility_decision_r2.json').exists() else {}
        chosen = second.get('chosen_independent_monitor') or first.get('chosen_independent_monitor')
        reward = second.get('reward_monitor')

        def role(model_id):
            if 'Qwen2.5' in model_id:
                return 'recompensa' + (' (escolhido)' if model_id == reward else '')
            if model_id == chosen:
                return 'independente (escolhido)'
            return 'independente no v3.2' if '135M' in model_id else 'candidato a independente'

        rows = []
        for rnd, data in (('1', first), ('2', second)):
            for m in data.get('summary', []):
                rows.append([rnd, m['model_id'].split('/')[-1], role(m['model_id']),
                             pct(m['tpr']) if m['tpr'] is not None else 'não avaliável',
                             pct(m['fpr']) if m['fpr'] is not None else 'não avaliável'])
        tables['monitores_v4'] = (
            'Desempenho dos monitores em 320 controles construídos (rótulos por regra, não humanos)',
            ['Rodada', 'Modelo', 'Papel', 'TPR', 'FPR'], rows, ['c', 'l', 'l', 'r', 'r'],
            'experimento_cientifico_v4/calibracao/resultados/monitor_feasibility_decision*.json.')
    gate = v4 / 'triagem_dev' / 'manifest.json'
    if gate.exists():
        data = _json(gate)
        threshold = _json(v4 / 'protocolo_v4.json')['gate']['threshold']
        rows = []
        for mode in ('sft', 'grpo'):
            for level, e in data['summary'][mode].items():
                rows.append([f'{mode.upper()} · {level}', f"{e['correct']}/{e['n']}", pct(e['accuracy']),
                             str(e['format_valid']), str(e['truncated']),
                             'aprovado' if e['accuracy'] >= threshold else 'reprovado'])
        tables['triagem_v4'] = ('Triagem DEV do protocolo v4, executada uma vez antes do registro',
                                ['Nível', 'Acertos', 'Acurácia', 'Formato válido', 'Truncadas', 'Critério 0,65'],
                                rows, ['l', 'r', 'r', 'r', 'r', 'l'],
                                'experimento_cientifico_v4/triagem_dev/manifest.json.')
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
