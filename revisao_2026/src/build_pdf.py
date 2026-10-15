"""Diagramação acadêmica. Tabelas obtidas dos CSVs da reanálise."""
from pathlib import Path
import csv
import html
import re
import sys
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY,TA_CENTER,TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate,PageTemplate,Frame,Paragraph,
    Spacer,PageBreak,Table,TableStyle,Image,KeepTogether,Preformatted)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/"Artigo_CoT_Versao_Final.pdf"
import matplotlib
FONTROOT=Path(matplotlib.get_data_path())/"fonts/ttf"
for name,file in [("Serif","DejaVuSerif.ttf"),("Serif-Bold","DejaVuSerif-Bold.ttf"),
                  ("Serif-Italic","DejaVuSerif-Italic.ttf"),("Serif-BoldItalic","DejaVuSerif-BoldItalic.ttf"),
                  ("Sans","DejaVuSans.ttf"),("Sans-Bold","DejaVuSans-Bold.ttf"),
                  ("Mono","DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name,str(FONTROOT/file)))
pdfmetrics.registerFontFamily("Serif",normal="Serif",bold="Serif-Bold",
                             italic="Serif-Italic",boldItalic="Serif-BoldItalic")
pdfmetrics.registerFontFamily("Sans",normal="Sans",bold="Sans-Bold",italic="Sans",boldItalic="Sans-Bold")
PAGE_W,PAGE_H=A4
LEFT=3*cm;RIGHT=2*cm;TOP=2.45*cm;BOTTOM=2.2*cm
WIDTH=PAGE_W-LEFT-RIGHT
INK=colors.HexColor("#172b3b");GRAY=colors.HexColor("#555555")
STYLES={
 "body":ParagraphStyle("body",fontName="Serif",fontSize=10.2,leading=15.1,
    alignment=TA_JUSTIFY,firstLineIndent=18,spaceAfter=7,allowWidows=0,allowOrphans=0),
 "h1":ParagraphStyle("h1",fontName="Sans-Bold",fontSize=12.4,leading=17,
    spaceBefore=16,spaceAfter=10,textColor=INK,keepWithNext=True),
 "h2":ParagraphStyle("h2",fontName="Sans-Bold",fontSize=10.5,leading=15.5,
    spaceBefore=11,spaceAfter=7,textColor=INK,keepWithNext=True),
 "caption":ParagraphStyle("caption",fontName="Sans",fontSize=8.5,leading=11.5,
    alignment=TA_CENTER,spaceAfter=5,spaceBefore=7,keepWithNext=True),
 "note":ParagraphStyle("note",fontName="Sans",fontSize=8,leading=11,
    textColor=GRAY,spaceAfter=10,alignment=TA_LEFT),
 "cell":ParagraphStyle("cell",fontName="Sans",fontSize=8.1,leading=11),
 "th":ParagraphStyle("th",fontName="Sans-Bold",fontSize=8,leading=10.5,textColor=colors.white),
 "equation":ParagraphStyle("equation",fontName="Serif",fontSize=10,leading=15,
    alignment=TA_CENTER,spaceBefore=6,spaceAfter=10),
 "reference":ParagraphStyle("reference",fontName="Serif",fontSize=9.4,leading=13.7,
    alignment=TA_LEFT,spaceAfter=10,splitLongWords=True),
 "cover":ParagraphStyle("cover",fontName="Sans-Bold",fontSize=16,leading=23,
    alignment=TA_CENTER,textColor=INK,spaceAfter=16),
 "center":ParagraphStyle("center",fontName="Serif",fontSize=10.1,leading=16,alignment=TA_CENTER),
 "toc":ParagraphStyle("toc",fontName="Sans",fontSize=9.3,leading=15,
    leftIndent=0,firstLineIndent=0,spaceBefore=3),
 "code":ParagraphStyle("code",fontName="Mono",fontSize=8.2,leading=12,
    leftIndent=10,spaceBefore=6,spaceAfter=9)}


def rich(text):
    text=html.escape(text)
    text=re.sub(r"\*\*(.+?)\*\*",r"<b>\1</b>",text)
    text=re.sub(r"10\^(-?\d+)",r"10<super>\1</super>",text)
    def link(match):
        raw=match.group(1)
        url=raw.rstrip(".,;)")
        return f'<link href="{url}" color="#254c70">{url}</link>'+raw[len(url):]
    text=re.sub(r"(https://[^\s]+)",link,text)
    return text


def P(text,style="body"):
    return Paragraph(rich(str(text)),STYLES[style])


def read_csv(name):
    with (ROOT/"results"/name).open() as f:
        return list(csv.DictReader(f))


def f3(n):
    return f"{float(n):.3f}".replace(".",",")


def data_table(kind):
    names={"control":"CoT explícita","cot_pressure":"Resposta direta","output_supervision":"Preâmbulo genérico"}
    if kind=="design":
        return (1,"Condições de ajuste supervisionado disponíveis no piloto.",
            ["Nome no código original","Denominação adotada","Alvo imitado"],
            [["control","CoT explícita","Etapas de cálculo e resposta."],
             ["cot_pressure","Resposta direta","Resposta com justificativa final curta."],
             ["output_supervision","Preâmbulo genérico","Frase genérica seguida da resposta."]],
            [145,137,WIDTH-282],
            "Fonte: inspeção de build_assistant_target no notebook histórico. Não há otimização contra monitor nessas condições.")
    if kind=="summary":
        rows=[[r["level"],names[r["condition"]],f'{r["successes"]}/{r["n"]}',f3(r["mean"]),f3(r["seed_sd"])]
              for r in read_csv("summary.csv")]
        return (2,"Acurácia registrada e variabilidade entre sementes.",
            ["Nível","Alvo","Corretos / exposições","Média","DP amostral"],rows,
            [36,135,128,64,WIDTH-363],
            "Fonte: 540 linhas publicadas. DP calculado entre três sementes; exposições não equivalem a itens independentes.")
    if kind=="costs":
        rows=[];costs=read_csv("paired_costs.csv")
        for condition in ("cot_pressure","output_supervision"):
            g=[r for r in costs if r["condition"]==condition]
            avg=lambda k:sum(float(r[k]) for r in g)/len(g)
            rows.append([names[condition],f3(avg("cost_A")),f3(avg("cost_C")),f3(avg("difference_in_differences"))])
        return (3,"Custos médios de desempenho e diferença entre níveis.",
            ["Alvo comparado","Custo em A","Custo em C","Diferença C - A"],rows,
            [155,85,85,WIDTH-325],"Fonte: contrastes dentro de cada semente. Custo = controle - comparação.")
    if kind=="contrast_seeds":
        rows=[];data=read_csv("paired_costs.csv")
        for condition in ("cot_pressure","output_supervision"):
            for r in data:
                if r["condition"]==condition:
                    rows.append([names[condition],r["seed"],f3(r["cost_A"]),f3(r["cost_C"]),f3(r["difference_in_differences"])])
        return (4,"Contrastes pareados por semente.",
            ["Alvo","Semente","Custo A","Custo C","Diferença"],rows,
            [149,64,72,72,WIDTH-357],"Fonte: reanálise. O teste bicaudal por inversão de sinais tem p = 0,25 para cada alvo.")
    if kind=="necessity":
        return (5,"Reconstrução da comparação pareada do modelo base.",
            ["Nível","Ambos corretos","Só com CoT","Só direta","Ambos errados","Ganho líquido"],
            [["A","9","0","1","0","-0,10"],["C","0","5","2","3","+0,30"]],
            [32,93,82,72,94,WIDTH-373],
            "Fonte: agregados históricos; n = 10 por nível. McNemar exato: A, p = 1,00; C, p = 0,453125.")
    if kind=="overlap":
        return (6,"Interseção dos conjuntos reconstruídos entre sementes.",
            ["Partição","42 × 43","42 × 44","43 × 44","União das 3"],
            [["Treino, por nível","149/150","148/150","149/150","152"],
             ["Avaliação, por nível","29/30","28/30","29/30","32"]],
            [151,73,73,73,WIDTH-370],
            "Fonte: execução dos geradores históricos. Interseção treino/avaliação: um enunciado em A; zero em C.")
    if kind=="stress":
        labels={"numero_correto":"Número correto","numero_incorreto":"Número incorreto","alternativas":"Alternativas múltiplas",
                "contradicao":"Conclusão contraditória","contaminacao":"Menção contaminante","xml_correto":"XML correto",
                "xml_incorreto":"XML incorreto com menção","xml_duplicado":"XML com respostas duplicadas"}
        rows=[[labels[r["kind"]],r["n"],r["expected_accepts"],r["original_accepts"],r["strict_accepts"]]
              for r in read_csv("grading_stress_summary.csv")]
        return (7,"Aceitações dos corretores no corpus mecânico construído.",
            ["Classe","Casos","Esperado","Original","Estrito"],rows,
            [203,57,64,64,WIDTH-388],
            "Fonte: 250 casos por classe. Os textos foram construídos para teste de contrato; não são gerações de LLM.")
    if kind=="lexical":
        names2={"ausente":"Rastro ausente","curto_correto":"Equação curta correta","generico":"Preâmbulo genérico",
                "repetitivo":"Texto repetitivo","aritmetico_sem_conectivos":"Cálculo sem conectivos","texto_com_conectivos":"Texto com conectivos"}
        rows=[[names2[r["kind"]],f3(r["score"]),f3(r["ttr"]),f3(r["rep_ratio"])] for r in read_csv("trace_controls.csv")]
        return (8,"Controles construídos para o escore lexical histórico.",
            ["Controle","Escore","TTR","Repetição"],rows,[228,75,75,WIDTH-378],
            "Fonte: funções históricas. O rastro ausente recebe métricas convencionadas; os demais controles não possuem rótulo de intenção.")
    if kind=="hypotheses":
        return (9,"Alcance da evidência para as hipóteses investigadas.",
            ["Questão","Evidência disponível","Situação"],
            [["Custo por alvo e complexidade","Diferença consistente nos agregados.","Sustentação descritiva."],
             ["Necessidade causal de CoT","Ablação pequena e confundida.","Não estabelecida."],
             ["Emergência por supervisão de saída","SFT altera diretamente o alvo.","Não testada."],
             ["Recuperação por outro monitor","Sem rastros históricos ou monitor novo.","Não testada no piloto."],
             ["Generalização entre domínios","Mesmo gerador aritmético.","Não demonstrada."]],
            [139,191,WIDTH-330],"Fonte: síntese da auditoria, sem converter ausência de teste em evidência de ausência.")
    if kind=="extension":
        return (10,"Estrutura prospectiva do estudo multiescala.",
            ["Modelo previsto","Sementes / célula","Condições × níveis","Execuções"],
            [["Qwen3-4B","12","3 × 3","108"],["Qwen3-14B","6","3 × 3","54"],
             ["Llama-3.1-8B-Instruct","12","3 × 3","108"],["Total","-","27 células","270"]],
            [173,100,112,WIDTH-385],
            "Planejamento não executado. Máximo previsto: 4.000 passos por execução; avaliação com pelo menos 500 itens por domínio.")
    if kind=="all_cells":
        rows=[]
        for r in read_csv("reanalyzed_cells.csv"):
            rows.append([r["level"],names[r["condition"]],r["seed"],r["successes"],r["n"],f3(r["accuracy"]),f3(r["flag_rate"])])
        return (11,"Resultados históricos por célula.",
            ["Nível","Alvo","Seed","Corretos","n","Acurácia","Sinalização"],rows,
            [33,127,42,57,30,72,WIDTH-361],"Fonte: recálculo dos 18 agregados. Sinalização = escore lexical > 0,50.")
    if kind=="parser_examples":
        return (12,"Exemplos do contrato de resposta final, com gabarito 42.",
            ["Resposta gerada","Corretor estrito","Motivo"],
            [["Resposta: 42","Aceita","Valor único correto."],
             ["<answer>42</answer>","Aceita","Tag final única correta."],
             ["Resposta: 43; 42 seria incorreto.","Rejeita","Conclusão ambígua ou contraditória."],
             ["Intermediário: 42. <answer>43</answer>","Rejeita","Valor final incorreto."],
             ["<answer>42</answer><answer>43</answer>","Rejeita","Duas respostas finais."]],
            [254,89,WIDTH-343],"Fonte: contrato implementado. Tags são mostradas literalmente.")
    if kind=="evidence":
        return (13,"Correspondência entre desfecho e tipo de evidência.",
            ["Componente","Executado","O que permite concluir"],
            [["540 métricas históricas","Sim, reanálise.","Consistência dos agregados e contrastes."],
             ["2.000 textos construídos","Sim, teste mecânico.","Falha e correção no contrato testado."],
             ["4.500 itens disjuntos","Sim, geração.","Ausência de colisões no conjunto gerado."],
             ["53 testes de software","Sim, CPU.","Conformidade dos casos verificados."],
             ["Novos treinos e inferências GPU","Não.","Nenhum resultado novo de modelo."],
             ["Monitores e anotação humana","Não, módulos preparados.","Nenhuma estimativa empírica de concordância."],
             ["Fatorial multiescala","Não, notebooks preparados.","Hipóteses externas permanecem abertas."]],
            [159,114,WIDTH-273],"Fonte: registro da execução e delimitação metodológica.")
    raise ValueError(kind)


def make_table(kind):
    num,title,headers,rows,widths,note=data_table(kind)
    for i,h in enumerate(headers):
        if h=="Nível" and widths[i]<44:
            delta=44-widths[i]
            j=max((j for j in range(len(widths)) if j!=i),key=lambda j:widths[j])
            widths[i]+=delta
            widths[j]-=delta
    table=Table([[P(h,"th") for h in headers]]+[[P(c,"cell") for c in row] for row in rows],
                 colWidths=widths,repeatRows=1,hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),INK),("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f3f5f6")]),
        ("LINEBELOW",(0,-1),(-1,-1),.5,colors.HexColor("#9aa6ac"))]))
    return [KeepTogether([P(f"Tabela {num} - {title}","caption"),table,P(note,"note")])]


class ArticleDoc(BaseDocTemplate):
    def __init__(self,path):
        super().__init__(str(path),pagesize=A4,leftMargin=LEFT,rightMargin=RIGHT,
            topMargin=TOP,bottomMargin=BOTTOM,
            title="Validade externa da ofuscação de cadeia de pensamento sob supervisão de saída",
            author="Kelvin e Silva Marques et al.",subject="Reanálise reprodutível e validação de instrumentos")
        frame=Frame(LEFT,BOTTOM,WIDTH,PAGE_H-TOP-BOTTOM,id="normal",leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="all",frames=frame,onPage=self.decorate))
        self.heading_index=0
    def beforeDocument(self):
        self.heading_index=0
    def decorate(self,c,doc):
        c.saveState()
        if doc.page>1:
            c.setFont("Sans",7);c.setFillColor(GRAY)
            c.drawString(LEFT,PAGE_H-1.45*cm,"OFUSCAÇÃO DE CoT | REANÁLISE E VALIDAÇÃO DE INSTRUMENTOS")
            c.setStrokeColor(colors.HexColor("#b5bec4"));c.setLineWidth(.35)
            c.line(LEFT,PAGE_H-1.65*cm,PAGE_W-RIGHT,PAGE_H-1.65*cm)
            c.setFont("Sans",7.2)
            c.drawString(LEFT,1.35*cm,"Marques et al. | 2026")
            c.drawRightString(PAGE_W-RIGHT,1.35*cm,str(doc.page))
        c.restoreState()
    def afterFlowable(self,flowable):
        if isinstance(flowable,Paragraph) and flowable.style.name=="h1":
            text=flowable.getPlainText()
            if text in ("RESUMO","ABSTRACT","SUMÁRIO"):
                return
            key="heading_"+str(self.heading_index);self.heading_index+=1
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text,key,level=0,closed=False)
            self.notify("TOCEntry",(0,text,self.page,key))


def cover():
    authors=["Kelvin e Silva Marques","Dimitri Farias Fernandes","Hastrorllabyo da Silva Lima",
        "Keven dos Santos Xavier","Caio de Freitas Brandão","Kauan Magalhães Costa",
        "Guilherme Martins Bezerra","Pedro Henrique Gomes Ferreira",
        "Carlos Eduardo Rodrigues Ferreira","Luan dos Santos Gaspar","Francisco Glauber Rodrigues dos Santos"]
    return [Spacer(1,5),P("CENTRO UNIVERSITÁRIO ATENEU","center"),P("ENGENHARIA DA COMPUTAÇÃO","center"),
        Spacer(1,30),*[P(a,"center") for a in authors],Spacer(1,45),
        P("Validade externa da ofuscação de cadeia de pensamento sob supervisão de saída","cover"),
        P("Reanálise reprodutível de um estudo-piloto e validação dos instrumentos de avaliação","center"),
        Spacer(1,58),P("FORTALEZA","center"),P("2026","center"),PageBreak()]


def build():
    doc=ArticleDoc(OUTPUT);story=cover()
    lines=(ROOT/"artigo.md").read_text().splitlines();paragraph=[];references=False
    def flush():
        if paragraph:
            story.append(P(" ".join(paragraph),"reference" if references else "body"));paragraph.clear()
    i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:flush()
        elif line=="[[PAGEBREAK]]":flush();story.append(PageBreak())
        elif line=="[[TOC]]":
            flush();story.append(P("SUMÁRIO","h1"))
            toc=TableOfContents();toc.levelStyles=[STYLES["toc"]];toc.dotsMinLevel=0;story.append(toc)
        elif line.startswith("[[TABLE:"):
            flush();story.extend(make_table(line[len("[[TABLE:"):-2]))
        elif line.startswith("[[FIGURE:"):
            flush();file,caption=line[len("[[FIGURE:"):-2].split("|",1)
            img=Image(str(ROOT/"figures"/file),width=WIDTH,height=WIDTH*.53,kind="proportional")
            story.append(KeepTogether([img,P(caption,"caption"),Spacer(1,7)]))
        elif line.startswith("[[EQUATION:"):
            flush();story.append(P(line[len("[[EQUATION:"):-2],"equation"))
        elif line=="[[CODE]]":
            flush();block=[];i+=1
            while lines[i].strip()!="[[ENDCODE]]":block.append(lines[i]);i+=1
            story.append(Preformatted("\n".join(block),STYLES["code"]))
        elif line.startswith("## "):flush();story.append(P(line[3:],"h2"))
        elif line.startswith("# "):
            flush();title=line[2:];references=title=="REFERÊNCIAS";story.append(P(title,"h1"))
        else:paragraph.append(line)
        i+=1
    flush();OUTPUT.parent.mkdir(parents=True,exist_ok=True);doc.multiBuild(story);print(str(OUTPUT))


if __name__=="__main__":
    build()
