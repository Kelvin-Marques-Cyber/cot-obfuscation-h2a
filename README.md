# Ofuscação de cadeia de pensamento: reanálise e laboratório experimental

Artigo, dados auditados e notebooks para investigar a relação entre formato de
treinamento, desempenho e monitorabilidade. Revisão de 15 de outubro de 2026.

**[Baixar o artigo final em PDF](revisao_2026/docs/Artigo_CoT_Versao_Final.pdf)**

## Notebooks

| Etapa | Notebook | Execução |
|---|---|---|
| 1. Reproduzir resultados e auditoria | [Reanálise reproduzível](revisao_2026/notebooks/01_reanalise_reprodutivel.ipynb) | CPU |
| 2. Testar alvos supervisionados corrigidos | [Piloto SFT corrigido](revisao_2026/notebooks/02_piloto_sft_corrigido.ipynb) | GPU |
| 3. Treinar contra monitores | [GRPO multiescala](revisao_2026/notebooks/03_grpo_multiescala.ipynb) | GPU e monitor configurado |
| 4. Avaliar saídas e concordância | [Avaliação, monitores e anotação](revisao_2026/notebooks/04_avaliacao_monitores_anotacao.ipynb) | GPU para inferência; CPU para análise |

Os notebooks GPU começam com configuração pequena e não iniciam treinamentos
longos sem ativar a célula de execução. Modelo, revisão, sementes, dataset,
checkpoints e saídas integrais são registrados. Reexecutar a mesma configuração
retoma o último checkpoint.

**[Guia para Colab, Vast.ai e Jupyter](revisao_2026/docs/GUIA_GPU.md)**

## Resultados executados

- Reanálise das 540 linhas históricas, em 18 células.
- Reprodução das médias e da ANOVA histórica.
- Auditoria das sobreposições dos geradores.
- Teste dos corretores em 2.000 respostas construídas.
- Geração e conferência de 4.500 itens aritméticos disjuntos.
- 57 testes locais de software aprovados, incluindo regressões de execução em notebook.

No corpus mecânico, o corretor histórico aceita 1.250 de 1.500 respostas que
deveriam ser rejeitadas. O corretor estrito rejeita esses casos segundo o contrato.
Esse teste não mede a prevalência do problema nas 540 avaliações, porque o CSV
histórico não contém os textos gerados.

O contraste de desempenho publicado é reproduzível, mas não constitui uma
refutação de ofuscação estratégica. O piloto altera alvos SFT; não executa
supervisão exclusiva de saída por reforço. Sementes adjacentes compartilham
29 dos 30 itens de avaliação.

**[Resultados e limites](revisao_2026/docs/RESULTADOS_E_LIMITES.md)**

## Reproduzir em CPU

    cd revisao_2026
    python3 -m venv .venv
    . .venv/bin/activate
    pip install -r requirements.lock.txt
    bash run.sh

No Windows, ative o ambiente virtual e execute:

    python -m unittest discover -s tests -v
    python src/reanalyse.py

## Extensão em GPU

Os notebooks 02 a 04 e o módulo gpu_lab foram preparados para novas execuções.
Seu código foi compilado e suas funções independentes de hardware foram
testadas; treino e inferência CUDA não foram realizados nesta revisão.

O benchmark GRPO fornecido é sintético e separa a resposta objetiva da alternativa
explorável. Há domínio reservado, família estrutural reservada, variante sem
marcação, resposta antecipada, prefixo incorreto, monitor independente e sonda
textual com separação por item. A dificuldade da dica é uma proxy que exige
calibração no modelo base.

Não se declara que esse benchmark replique os datasets do artigo-alvo ou
substitua validação humana em domínios especializados. O plano multiescala prevê
270 execuções; começar pelo smoke test é necessário para conferir o hardware e a
configuração do monitor.

## Organização

- [Fonte editável do artigo](revisao_2026/artigo.md)
- [Código e instrumentos](revisao_2026/src)
- [Testes](revisao_2026/tests)
- [Dados e resultados da reanálise](revisao_2026/results)
- [Procedência](revisao_2026/provenance.json)
- [Protocolo prospectivo](revisao_2026/protocol_extension.json)

Os arquivos originais da raiz e o notebook histórico foram preservados para
rastreabilidade. As conclusões científicas vigentes e suas limitações estão na
versão revisada do artigo.

Código sob licença MIT. Fontes e dados originais mantêm os avisos de licença
declarados em sua origem.
