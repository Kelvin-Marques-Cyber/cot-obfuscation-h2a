"""Gera o artigo em DOCX (editável) e PDF a partir de revisao_2026/artigo.md.

Fluxo: figuras (gerar_figuras.py) -> expansão de tabelas, figuras, valores e
referências cruzadas -> pandoc com modelo de estilos -> pós-processamento das
tabelas -> LibreOffice converte o MESMO DOCX em PDF -> verificações (páginas,
fontes incorporadas, equivalência textual DOCX/PDF) -> PNG de todas as páginas.

Uso (a partir da raiz do repositório):
    python revisao_2026/src/gerar_artigo.py
Requer pandoc 3, LibreOffice com o componente Math, poppler-utils, python-docx
e matplotlib.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SOURCE = ROOT / 'artigo.md'
BUILD = ROOT / 'build'
DOCS = ROOT / 'docs'
NAME = 'Artigo_CoT_Versao_Final'
sys.path.insert(0, str(Path(__file__).resolve().parent))


# --------------------------------------------------------------- formatação
def num(value, digits=2):
    text = f'{value:,.{digits}f}'
    return text.replace(',', '§').replace('.', ',').replace('§', '.')


def pct(value, digits=1):
    return num(100 * value, digits) + '%'


def frac(k, n):
    return f'{k}/{n}'


def read_csv(path):
    with Path(path).open(encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


# ------------------------------------------------------------------- dados
DIAG = REPO / 'experimento_cientifico_v3' / 'diagnostico_dev' / 'resultados' / 'diagnostico_triagem_dev.json'
V4 = REPO / 'experimento_cientifico_v4'
VALIDATION = ROOT / 'results' / 'validacao_2026_10_28.json'


def values():
    """Todos os números citados no texto vêm daqui (chave -> texto formatado)."""
    out = {}
    report = read_json(ROOT / 'results' / 'report.json')
    summary = read_csv(ROOT / 'results' / 'summary.csv')
    contrasts = {r['condition']: r for r in read_csv(ROOT / 'results' / 'contrasts.csv')}
    stress = read_csv(ROOT / 'results' / 'grading_stress_summary.csv')
    out['piloto.linhas'] = str(report['n_rows'])
    out['piloto.celulas'] = str(report['n_cells'])
    out['piloto.anova_f'] = num(report['historical_anova']['F'], 2)
    out['piloto.anova_p'] = num(report['historical_anova']['p'] * 1e12, 2) + ' × 10⁻¹²'
    out['piloto.welch_t'] = num(report['historical_welch']['t'], 2)
    out['piloto.welch_p'] = num(report['historical_welch']['p'] * 1e7, 2) + ' × 10⁻⁷'
    for row in summary:
        out[f'piloto.acc.{row["level"]}.{row["condition"]}'] = num(float(row['mean']), 3)
    for condition, row in contrasts.items():
        out[f'piloto.dd.{condition}'] = num(float(row['mean']), 3)
        out[f'piloto.dd_ic.{condition}'] = f'[{num(float(row["conditional_t_low"]), 3)}; {num(float(row["conditional_t_high"]), 3)}]'
        out[f'piloto.signflip.{condition}'] = num(float(row['sign_flip_two_sided']), 2)
    necessity = report['necessity_reconstructed']['C']
    out['piloto.mcnemar_b'] = str(necessity['cot_only'])
    out['piloto.mcnemar_c'] = str(necessity['direct_only'])
    out['piloto.mcnemar_p'] = num(necessity['p_exact_mcnemar'], 3)
    out['piloto.ganho_liquido'] = num(necessity['net_gain'], 2)
    out['piloto.falsos_aceites'] = num(report['original_false_accepts'], 0)
    out['piloto.negativos'] = num(report['negative_cases'], 0)
    out['piloto.casos'] = num(report['constructed_stress_cases'], 0)
    out['piloto.taxa_aceite_indevido'] = pct(report['original_false_accepts'] / report['negative_cases'])
    out['piloto.itens_disjuntos'] = num(report['disjoint_generated_items'], 0)
    out['piloto.corr_rep'] = num(report['correlations']['score_repetition'], 3)
    out['piloto.corr_ttr'] = num(report['correlations']['score_ttr'], 3)
    del stress
    if DIAG.exists():
        diag = read_json(DIAG)
        for key, entry in diag['levels'].items():
            mode, level = key.split('/')
            base = f'v3.{mode}.{level}'
            out[base + '.doc'] = pct(entry['documented_accuracy'], 2 if mode == 'grpo' else 1)
            out[base + '.doc_k'] = frac(entry['documented_correct'], 120)
            out[base + '.cpu'] = pct(entry['reproduced_accuracy'])
            out[base + '.cpu_k'] = frac(entry['reproduced_correct'], 120)
            out[base + '.perm'] = pct(entry['permissive_upper_bound'])
            out[base + '.trunc'] = str(entry['truncated'])
            out[base + '.tokens'] = num(entry['mean_output_tokens'], 1)
            for name, count in entry['categories'].items():
                out[f'{base}.cat.{name}'] = str(count)
            if 't4_fp16' in entry:
                t4 = entry['t4_fp16']
                out[base + '.t4'] = pct(t4['accuracy'])
                out[base + '.t4_k'] = frac(t4['correct'], 120)
                out[base + '.t4_igual_cpu'] = str(t4.get('same_correctness_as_cpu', ''))
                out[base + '.t4_ident'] = str(t4.get('identical_generation_as_cpu', ''))
            if mode == 'grpo':
                out[base + '.formato'] = str(entry['format_valid'])
                out[base + '.dica'] = str(entry['chose_hint_letter'])
                for domain, d in entry['by_domain'].items():
                    out[f'{base}.dom.{domain}'] = frac(d['correct'], d['n'])
    if (V4 / 'triagem_dev' / 'manifest.json').exists():
        gate = read_json(V4 / 'triagem_dev' / 'manifest.json')
        for mode, levels in gate['summary'].items():
            for level, entry in levels.items():
                out[f'v4.gate.{mode}.{level}'] = pct(entry['accuracy'])
                out[f'v4.gate.{mode}.{level}.k'] = frac(entry['correct'], entry['n'])
    if VALIDATION.exists():
        for key, value in read_json(VALIDATION).get('numeros', {}).items():
            out['val.' + key] = str(value)
    return out


# ------------------------------------------------------------------ tabelas
def table_md(header, rows, align):
    sep = {'l': ':---', 'r': '---:', 'c': ':---:'}
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(sep[a] for a in align) + '|']
    for row in rows:
        lines.append('| ' + ' | '.join(str(c).replace('|', '\\|') for c in row) + ' |')
    return '\n'.join(lines)


def tabelas():
    import tabelas_artigo
    return tabelas_artigo.build(ROOT, REPO, num, pct, frac)


FIGURAS = {
    'piloto': ('fig1_piloto_acuracia.png', 'Acurácia registrada no estudo-piloto por nível e alvo de treinamento',
               'Reanálise dos 540 registros publicados. Barras: média das três sementes; círculos: sementes.'),
    'triagem': ('fig2_triagem_v3.png', 'Triagem DEV do estudo v3.2: valores documentados e reproduções',
                'README do v3 (commit 273d131) e reproduções desta revisão. Linha tracejada: critério de 0,65.'),
    'falhas': ('fig3_falhas_v3.png', 'Composição das respostas da reprodução em CPU da triagem DEV v3.2',
               'Classificação descritiva das 480 respostas; o desfecho registrado usa apenas o corretor congelado.'),
    'v4': ('fig4_triagem_v4.png', 'Triagem DEV do protocolo v4 antes do registro',
           'Arquivos de experimento_cientifico_v4/triagem_dev. Linha tracejada: critério de 0,65.'),
}


# ---------------------------------------------------------------- expansão
def expand(text, data, tables):
    order = {'tab': [], 'fig': []}
    for kind, key in re.findall(r'\{\{(tabela|figura):([\w-]+)\}\}', text):
        order['tab' if kind == 'tabela' else 'fig'].append(key)
    number = {('tab', k): i + 1 for i, k in enumerate(order['tab'])}
    number.update({('fig', k): i + 1 for i, k in enumerate(order['fig'])})

    def ref(match):
        kind, key = match.group(1), match.group(2)
        if (kind, key) not in number:
            raise KeyError(f'Referência sem destino: @{kind}:{key}')
        return ('Tabela ' if kind == 'tab' else 'Figura ') + str(number[kind, key])
    text = re.sub(r'@(tab|fig):([\w-]+)', ref, text)

    def value(match):
        key = match.group(1)
        if key not in data:
            raise KeyError('Valor sem fonte de dados: ' + key)
        return data[key]
    text = re.sub(r'\{\{v:([\w.\-]+)\}\}', value, text)

    def block(match):
        kind, key = match.group(1), match.group(2)
        if kind == 'tabela':
            title, header, rows, align, source = tables[key]
            return ('::: {custom-style="Legenda"}\nTabela %d – %s\n:::\n\n%s\n\n'
                    '::: {custom-style="Fonte"}\nFonte: %s\n:::\n' % (number['tab', key], title,
                                                                    table_md(header, rows, align), source))
        image, title, source = FIGURAS[key]
        path = ROOT / 'figures' / image
        if not path.exists():
            raise FileNotFoundError(path)
        return ('::: {custom-style="Legenda"}\nFigura %d – %s\n:::\n\n'
                '::: {custom-style="Figura"}\n![](%s){width=15cm}\n:::\n\n'
                '::: {custom-style="Fonte"}\nFonte: %s\n:::\n' % (number['fig', key], title, path, source))
    text = re.sub(r'^\{\{(tabela|figura):([\w-]+)\}\}\s*$', block, text, flags=re.M)
    text = text.replace('{{quebra}}', '```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```')
    leftover = re.findall(r'\{\{[^}]*\}\}', text)
    if leftover:
        raise ValueError('Marcadores não resolvidos: ' + ', '.join(leftover[:5]))
    return text, number


# ------------------------------------------------------- pós-processamento
def style_tables(docx_path):
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt
    from modelo_docx import find_style
    doc = Document(str(docx_path))
    table_style = find_style(doc, 'Texto de tabela')

    def border(parent, edge, size):
        element = OxmlElement(f'w:{edge}')
        element.set(qn('w:val'), 'single' if size else 'nil')
        element.set(qn('w:sz'), str(size))
        element.set(qn('w:space'), '0')
        element.set(qn('w:color'), '000000')
        parent.append(element)

    for table in doc.tables:
        tbl_pr = table._tbl.tblPr
        for old in tbl_pr.findall(qn('w:tblBorders')) + tbl_pr.findall(qn('w:tblW')):
            tbl_pr.remove(old)
        width = OxmlElement('w:tblW')
        width.set(qn('w:w'), '5000')
        width.set(qn('w:type'), 'pct')
        tbl_pr.append(width)
        borders = OxmlElement('w:tblBorders')
        for edge, size in (('top', 10), ('left', 0), ('bottom', 10), ('right', 0),
                           ('insideH', 0), ('insideV', 0)):
            border(borders, edge, size)
        tbl_pr.append(borders)
        for index, row in enumerate(table.rows):
            tr_pr = row._tr.get_or_add_trPr()
            tr_pr.append(OxmlElement('w:cantSplit'))
            if index == 0:
                tr_pr.append(OxmlElement('w:tblHeader'))
            for cell in row.cells:
                tc_pr = cell._tc.get_or_add_tcPr()
                if index == 0:
                    tc_borders = OxmlElement('w:tcBorders')
                    border(tc_borders, 'bottom', 6)
                    tc_pr.append(tc_borders)
                for paragraph in cell.paragraphs:
                    paragraph.style = table_style
                    paragraph.paragraph_format.keep_with_next = index < len(table.rows) - 1
                    for run in paragraph.runs:
                        run.font.size = Pt(9)
                        if index == 0:
                            run.font.bold = True
    for paragraph in doc.paragraphs:
        if paragraph.style.name == 'Figura':
            paragraph.paragraph_format.keep_with_next = True
    doc.core_properties.title = TITLE
    doc.core_properties.author = 'Kelvin e Silva Marques et al.'
    doc.core_properties.keywords = 'cadeia de pensamento; monitorabilidade; pré-registro; reprodutibilidade'
    doc.save(str(docx_path))


TITLE = ''


# --------------------------------------------------------------- verificação
def words(text):
    text = text.replace('­', '').replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    text = re.sub(r'-\n(\w)', r'\1', text)
    return re.findall(r'[0-9A-Za-zÀ-ÿ]+', text.lower())


def docx_text(path):
    from docx import Document
    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    return '\n'.join(parts)


def verify(docx_path, pdf_path):
    pdf_text = subprocess.run(['pdftotext', '-layout', str(pdf_path), '-'], capture_output=True,
                              text=True, check=True).stdout
    a, b = Counter(words(docx_text(docx_path))), Counter(words(pdf_text))
    page_numbers = {str(i) for i in range(1, 200)}
    missing = {w: c - b[w] for w, c in a.items() if c > b[w]}
    extra = {w: c - a[w] for w, c in b.items() if c > a[w] and w not in page_numbers}
    info = subprocess.run(['pdfinfo', str(pdf_path)], capture_output=True, text=True, check=True).stdout
    pages = int(re.search(r'Pages:\s+(\d+)', info).group(1))
    fonts = subprocess.run(['pdffonts', str(pdf_path)], capture_output=True, text=True).stdout.splitlines()[2:]
    not_embedded = [line for line in fonts if line.split() and line.split()[-5] != 'yes']
    return dict(pages=pages, docx_words=sum(a.values()), pdf_words=sum(b.values()),
                missing_in_pdf=dict(sorted(missing.items())[:40]), extra_in_pdf=dict(sorted(extra.items())[:40]),
                fonts=[line.split()[0] for line in fonts if line.split()], fonts_not_embedded=not_embedded,
                equivalent=not missing and not not_embedded)


def main():
    global TITLE
    import gerar_figuras
    import modelo_docx
    gerar_figuras.main()
    BUILD.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    text = SOURCE.read_text(encoding='utf-8')
    TITLE = re.search(r'^::: \{custom-style="Capa titulo"\}\n(.+?)\n:::', text, re.M | re.S).group(1).strip()
    expanded, numbers = expand(text, values(), tabelas())
    (BUILD / 'artigo_expandido.md').write_text(expanded, encoding='utf-8')
    reference = modelo_docx.build(BUILD / 'modelo.docx')
    docx_path = DOCS / f'{NAME}.docx'
    subprocess.run(['pandoc', str(BUILD / 'artigo_expandido.md'), '-f', 'markdown+pipe_tables', '-t', 'docx',
                    '--reference-doc', str(reference), '-o', str(docx_path)], check=True)
    style_tables(docx_path)
    lo_profile = BUILD / 'lo-profile'
    subprocess.run(['soffice', f'-env:UserInstallation=file://{lo_profile}', '--headless', '--convert-to', 'pdf',
                    '--outdir', str(BUILD), str(docx_path)], check=True, capture_output=True, timeout=600)
    pdf_path = DOCS / f'{NAME}.pdf'
    shutil.move(str(BUILD / f'{NAME}.pdf'), pdf_path)
    check = verify(docx_path, pdf_path)
    pages = BUILD / 'paginas'
    shutil.rmtree(pages, ignore_errors=True)
    pages.mkdir()
    subprocess.run(['pdftoppm', '-r', '70', '-png', str(pdf_path), str(pages / 'p')], check=True)
    check.update(docx_sha256=hashlib.sha256(docx_path.read_bytes()).hexdigest(),
                 pdf_sha256=hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
                 source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                 tables=len([k for k in numbers if k[0] == 'tab']), figures=len([k for k in numbers if k[0] == 'fig']))
    (DOCS / 'verificacao_documentos.json').write_text(json.dumps(check, indent=2, ensure_ascii=False) + '\n',
                                                      encoding='utf-8')
    print(json.dumps(check, indent=2, ensure_ascii=False))
    if not check['equivalent']:
        sys.exit('DOCX e PDF divergem ou há fontes não incorporadas; veja verificacao_documentos.json')


if __name__ == '__main__':
    main()
