# Ofuscação de cadeia de pensamento: reanálise e laboratório experimental

Artigo, dados auditados e notebooks para investigar a relação entre formato de
treinamento, desempenho e monitorabilidade. Revisão de 15 de outubro de 2026.

**[Baixar o artigo final em PDF](revisao_2026/docs/Artigo_CoT_Versao_Final.pdf)**

## Experimento científico v3: sequência completa

| Etapa | Abrir no Colab | Ambiente |
|---|---|---|
| 00. Protocolo e datasets | [Protocolo](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/experimento_cientifico_v3/00_protocolo_e_datasets.ipynb) | CPU |
| 01. SFT replicado | [Treinamento SFT](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/experimento_cientifico_v3/01_treinamento_sft_replicado.ipynb) | GPU |
| 02. GRPO replicado | [Treinamento GRPO](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/experimento_cientifico_v3/02_treinamento_grpo_replicado.ipynb) | GPU |
| 03. Competência e teste cego | [Avaliação final](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/experimento_cientifico_v3/03_competencia_e_teste_cego.ipynb) | GPU |
| 04. Monitor e inferência | [Monitor independente e relatório](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/experimento_cientifico_v3/04_monitor_independente_e_inferencia.ipynb) | GPU |

Comece pelo **00** e use **Ambiente de execução → Executar tudo**. O pacote
mantém o desenho de **240 treinos e 55.200 gerações TEST**, com 20 sementes,
três condições e dois níveis por método. O notebook 00 exporta os hashes e o
protocolo para registro público anterior aos treinos. Informe a URL real no
campo `PUBLIC_REGISTRATION_URL` antes de iniciar 01/02. A etapa 04 depende
das anotações humanas descritas no próprio notebook.

Os cinco compartilham o mesmo `STUDY_ID` e retomam os resultados do Drive.
O padrão v3.2 usa `confirmatory_v3/estudo-principal-2026-v3-2/`, preservando
estudos anteriores. **[Instruções e protocolo](experimento_cientifico_v3/README.md)**.

Falhas de montagem do Drive têm duas tentativas de recuperação e verificação
de gravação. Se a conexão falhar, o 00 prepara um ZIP local; os treinos
aguardam armazenamento persistente. O modo local é temporário. Consulte as
instruções acima para reconectar ou baixar o ZIP pelo painel Arquivos.

## Laboratório de reanálise e testes curtos

| Etapa | Abrir no Colab | Ambiente |
|---|---|---|
| 1. Reproduzir resultados e auditoria | [Reanálise](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/revisao_2026/notebooks/01_reanalise_reprodutivel.ipynb) | CPU |
| 2. Testar alvos supervisionados corrigidos | [Piloto SFT](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/revisao_2026/notebooks/02_piloto_sft_corrigido.ipynb) | GPU T4 |
| 3. Treinar contra monitores | [GRPO](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/revisao_2026/notebooks/03_grpo_multiescala.ipynb) | GPU T4 |
| 4. Avaliar saídas e concordância | [Avaliação independente](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/revisao_2026/notebooks/04_avaliacao_monitores_anotacao.ipynb) | GPU T4 |

Abra e use **Ambiente de execução → Executar tudo**. Autorize o Drive quando
solicitado. Instalação, dados, treinos, avaliação e ZIP final já estão ativados.
O padrão `quick` usa modelos públicos pequenos e três condições em SFT/GRPO,
sem chave de API: os monitores são locais. `pilot` e `full` estão no formulário
da primeira célula para ampliar depois. O teste curto não é o estudo completo.

Modelo, revisão, sementes, dataset, checkpoints e saídas integrais são registrados.
Reexecutar retoma o trabalho pendente. Os quatro notebooks compartilham
`Meu Drive/CoT_Obfuscation_Lab/colab_v3/`; não é preciso copiar arquivos.
O código é baixado de um commit fixo, separado de clones antigos no Drive.
**Se já abriu um notebook antigo, reabra-o pelo link acima**: as cópias do Colab
mantêm o SHA antigo e não recebem correções automaticamente. Os notebooks
fixam uma revisão imutável do código, exibida na primeira célula.

**[Guia para Colab, Vast.ai e Jupyter](revisao_2026/docs/GUIA_GPU.md)**

## Resultados executados

- Reanálise das 540 linhas históricas, em 18 células.
- Reprodução das médias e da ANOVA histórica.
- Auditoria das sobreposições dos geradores.
- Teste dos corretores em 2.000 respostas construídas.
- Geração e conferência de 4.500 itens aritméticos disjuntos.
- 71 testes locais de software aprovados, incluindo recuperação de ambientes e troca de versão na mesma sessão do Colab.
- Integração real das APIs SFT/GRPO em CPU com miniatura aleatória, retomada de
  checkpoint, avaliação, análise e monitores locais; sem resultados experimentais novos.

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
Código, contratos e integração das APIs foram verificados, incluindo treinos
mínimos em CPU com pesos aleatórios. Treino quantizado e inferência CUDA com
os modelos públicos não foram realizados no ambiente desta revisão.

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
