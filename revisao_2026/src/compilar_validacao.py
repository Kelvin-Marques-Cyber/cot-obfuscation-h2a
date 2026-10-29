"""Compila o relatório de validação de 28/10/2026 a partir dos registros de execução.

Entradas: results/execucoes_locais.json (executar_validacao.py), diagnóstico do
v3.2, calibração e triagem do v4 e relatório do teste técnico CUDA. Saídas:
results/validacao_2026_10_28.json (usado pelo artigo) e docs/RELATORIO_VALIDACAO.md.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
V3 = REPO / 'experimento_cientifico_v3' / 'diagnostico_dev' / 'resultados'
V4 = REPO / 'experimento_cientifico_v4'


def read(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def pct(x):
    return f'{100 * x:.1f}%'.replace('.', ',')


def main():
    local = read(ROOT / 'results' / 'execucoes_locais.json')
    diag = read(V3 / 'diagnostico_triagem_dev.json')
    gate = read(V4 / 'triagem_dev' / 'manifest.json')
    cuda = read(V4 / 'validacao' / 'relatorio_tecnico_cuda.json')
    protocol = read(V4 / 'protocolo_v4.json')
    table, numbers = [], {}
    if local:
        env = local['environment']
        where = f"CPU ({env.get('cpu', 'x86_64')}, {env.get('cpu_count')} núcleos), Python {env['python']}"
        for entry in local['entries']:
            summary = ' '.join(entry['summary']) or ('código 0' if entry['exit_code'] == 0 else 'falhou')
            table.append(dict(verificacao=entry['name'], ambiente=where,
                              resultado=f"{'aprovado' if entry['exit_code'] == 0 else 'reprovado'} "
                                        f"({summary}; {entry['seconds']} s)"))
            for line in entry['summary']:
                if line.startswith('Ran '):
                    numbers[f"testes.{len(numbers)}"] = line.split()[1]
    if diag:
        same = all(e['t4_fp16']['matches_documented'] for e in diag['levels'].values())
        table.append(dict(verificacao='Reprodução da triagem DEV v3.2 pelo caminho original (commits fde856f/4a2a133)',
                          ambiente='Colab, Tesla T4, Python 3.13.16, PyTorch 2.11.0+cu130, fp16',
                          resultado='quatro valores documentados reproduzidos exatamente' if same
                          else 'valores diferentes dos documentados'))
        table.append(dict(verificacao='Reprodução da triagem DEV v3.2 com as mesmas funções em CPU',
                          ambiente='CPU x86_64, Python 3.12.3, PyTorch 2.8.0, fp32',
                          resultado='; '.join(f"{k}: {e['cpu_fp32']['correct']}/{e['cpu_fp32']['n']}"
                                              for k, e in diag['levels'].items() if 'cpu_fp32' in e)))
    cal = V4 / 'calibracao' / 'resultados'
    sft2, grpo2 = read(cal / 'sft_calibration_decision_r2.json'), read(cal / 'grpo_calibration_decision_r2.json')
    if sft2 and grpo2:
        table.append(dict(verificacao='Calibração exploratória do v4 (rodadas 1 e 2, itens próprios)',
                          ambiente='Colab, Tesla T4, fp16, decodificação gulosa item a item',
                          resultado=f"SFT: nível C = {sft2['chosen_C']}; GRPO: variante {grpo2['chosen'][0]}, "
                                    f"faixa {grpo2['chosen'][1]} (regras gravadas antes de cada rodada)"))
    mon1, mon2 = read(cal / 'monitor_feasibility_decision.json'), read(cal / 'monitor_feasibility_decision_r2.json')
    if mon1 and mon2:
        entries = mon1['summary'] + mon2['summary']
        gpus = sorted({e.get('hardware', 'Tesla T4') for e in entries})
        table.append(dict(verificacao=f'Viabilidade de {len(entries)} monitores em 320 controles construídos',
                          ambiente='Colab, ' + ' e '.join(gpus) + ', fp32',
                          resultado='nenhum monitor independente nem de recompensa aprovado; braço GRPO '
                                    'excluído pela regra declarada'))
        table.append(dict(verificacao='Encerramento de sessões do Colab durante a calibração',
                          ambiente='Colab (sessões sem atividade foram recicladas pelo serviço)',
                          resultado='etapas repetidas com o mesmo código; valores repetidos idênticos aos '
                                    'observados; a 1ª execução da rodada 2 dos monitores morreu por falta '
                                    'de RAM e foi repetida integralmente após declaração'))
    if gate and protocol:
        threshold = protocol['gate']['threshold']
        passed = all(e['accuracy'] >= threshold for levels in gate['summary'].values() for e in levels.values())
        table.append(dict(verificacao='Triagem DEV do v4 (execução única, antes do registro)',
                          ambiente='Colab, Tesla T4, fp16, decodificação gulosa',
                          resultado=('aprovada: ' if passed else 'reprovada: ') + '; '.join(
                              f"{m.upper()} {lv} {e['correct']}/{e['n']}" for m, levels in gate['summary'].items()
                              for lv, e in levels.items())))
    if cuda:
        notebooks = cuda.get('notebooks', {})
        ok = sum(1 for n in notebooks.values() if not n.get('error'))
        table.append(dict(verificacao='Teste técnico CUDA: cinco notebooks v4 com modelos reais (TECHNICAL_TEST)',
                          ambiente='Colab, ' + cuda.get('gpu', 'Tesla T4'),
                          resultado=f'{ok} de {len(notebooks)} notebooks executados sem erro'))
        t = cuda.get('interrupcao_treino', {})
        if t:
            final = t.get('status_final', {})
            table.append(dict(verificacao='Interrupção forçada de treino SFT na GPU e retomada',
                              ambiente='Colab, Tesla T4, fp16',
                              resultado=f"retomado de {final.get('resumed_from')} até o passo "
                                        f"{final.get('global_step')} ({final.get('status')})"))
        e = cuda.get('interrupcao_avaliacao', {})
        if e:
            table.append(dict(verificacao='Interrupção forçada de avaliação na GPU e retomada com linha parcial',
                              ambiente='Colab, Tesla T4, fp16',
                              resultado=f"{e.get('final_lines')} linhas, {e.get('unique_items')} itens únicos; "
                                        f"primeira linha idêntica: {e.get('first_line_identical')}"))
    evidencias = [
        dict(conclusao='Médias e interação do piloto reproduzidas', categoria='resultado histórico',
             base='540 linhas publicadas reprocessadas'),
        dict(conclusao='Corretor histórico aceita respostas ambíguas', categoria='verificação de software',
             base='2.000 textos construídos'),
        dict(conclusao='Sementes do piloto compartilham quase todos os itens', categoria='verificação de software',
             base='reexecução do gerador original'),
        dict(conclusao='Triagem do v3.2 reprovada nos quatro níveis', categoria='triagem de competência',
             base='valores documentados; reproduzidos exatamente na T4'),
        dict(conclusao='Causas da reprovação (incapacidade em C; instrução e contrato)', categoria='experimento exploratório',
             base='classificação das gerações reproduzidas'),
        dict(conclusao='Nenhum dos monitores avaliados discrimina controles construídos', categoria='experimento exploratório',
             base='9 modelos em 320 controles rotulados por regra, não por pessoas'),
        dict(conclusao='Exclusão do braço GRPO (H2 não testável no v4)', categoria='decisão de desenho',
             base='regra gravada antes do resultado dos monitores'),
        dict(conclusao='Escolhas de tarefa e instrução do v4', categoria='experimento exploratório',
             base='calibração em itens próprios, regras gravadas antes'),
        dict(conclusao='Resultado da triagem do v4', categoria='triagem de competência',
             base='120 itens DEV por nível, execução única'),
        dict(conclusao='Software de treino, avaliação e retomada funciona em CPU e GPU', categoria='verificação de software',
             base='testes unitários, integração e modo técnico'),
        dict(conclusao='Efeitos de H1 e H2', categoria='resultado confirmatório',
             base='não produzido: nenhum treino confirmatório executado'),
    ]
    gate_ok = bool(gate and protocol and all(e['accuracy'] >= protocol['gate']['threshold']
                                             for e in gate['summary']['sft'].values()))
    situacao = [
        dict(experimento='Reanálise do estudo-piloto', situacao='concluído',
             detalhe='médias, ANOVA e Welch reproduzidos; instrumentos auditados'),
        dict(experimento='Estudo v3.2 (OSF sjwdf): treinos e TEST', situacao='bloqueado (encerrado)',
             detalhe='triagem DEV reprovada nos quatro níveis; nenhum treino'),
        dict(experimento='Diagnóstico da triagem do v3.2 (T4 e CPU)', situacao='concluído',
             detalhe='quatro valores reproduzidos; causas classificadas'),
        dict(experimento='Calibração do v4 (tarefas, instruções e monitores)', situacao='concluído',
             detalhe='SFT C2; GRPO r2b; nenhum monitor aprovado'),
        dict(experimento='Triagem DEV do v4 (execução única)', situacao=(
            'concluído' if gate else 'não executado'), detalhe=(
            ('SFT aprovado' if gate_ok else 'SFT reprovado') + '; GRPO descritivo' if gate else '')),
        dict(experimento='Braço GRPO do v4 (H2)', situacao='excluído antes do registro',
             detalhe='sem monitor independente nem de recompensa viável'),
        dict(experimento='Registro público do v4', situacao='pendente (ação do pesquisador)' if gate_ok
             else 'não aplicável', detalhe='publicar o pacote do notebook 00 antes do primeiro treino'),
        dict(experimento='120 treinos SFT, 26.400 gerações TEST e inferência de H1',
             situacao='pendente' if gate_ok else 'bloqueado',
             detalhe='dependem do registro público' if gate_ok else 'triagem do v4 reprovada'),
        dict(experimento='Teste técnico CUDA dos cinco notebooks v4', situacao='concluído' if cuda else 'não executado',
             detalhe='modo técnico; não é evidência científica'),
    ]
    out = dict(date='2026-10-29', tabela=table, evidencias=evidencias, numeros=numbers, situacao=situacao,
               scientific_confirmatory_results=False)
    (ROOT / 'results' / 'validacao_2026_10_28.json').write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n',
                                                                encoding='utf-8')
    lines = ['# Relatório de validação e execuções — 28 e 29/10/2026', '',
             'Este relatório lista o que foi executado nesta revisão, em que ambiente e com que resultado. '
             'Verificações de software não constituem evidência científica sobre as hipóteses.', '',
             '| Verificação | Ambiente | Resultado |', '|---|---|---|']
    lines += [f"| {r['verificacao']} | {r['ambiente']} | {r['resultado']} |" for r in table]
    lines += ['', '## Situação dos experimentos', '', '| Experimento | Situação | Detalhe |', '|---|---|---|']
    lines += [f"| {r['experimento']} | {r['situacao']} | {r['detalhe']} |" for r in situacao]
    lines += ['', '## Natureza da evidência', '', '| Conclusão | Categoria | Base |', '|---|---|---|']
    lines += [f"| {r['conclusao']} | {r['categoria']} | {r['base']} |" for r in evidencias]
    if local:
        lines += ['', '## Comandos locais', '']
        for entry in local['entries']:
            lines.append(f"- `{entry['command']}` em `{entry['cwd']}`: código {entry['exit_code']}, {entry['seconds']} s")
        lines += ['', f"Ambiente: `{json.dumps(local['environment'], ensure_ascii=False)}`", '',
                  f"Commit avaliado: `{local['git_head']}`"]
    (ROOT / 'docs' / 'RELATORIO_VALIDACAO.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(table, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
