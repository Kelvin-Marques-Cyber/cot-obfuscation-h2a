::: {custom-style="Capa"}
CENTRO UNIVERSITÁRIO ATENEU

ENGENHARIA DA COMPUTAÇÃO
:::

&nbsp;

::: {custom-style="Capa"}
Kelvin e Silva Marques

Dimitri Farias Fernandes

Hastrorllabyo da Silva Lima

Keven dos Santos Xavier

Caio de Freitas Brandão

Kauan Magalhães Costa

Guilherme Martins Bezerra

Pedro Henrique Gomes Ferreira

Carlos Eduardo Rodrigues Ferreira

Luan dos Santos Gaspar

Francisco Glauber Rodrigues dos Santos
:::

::: {custom-style="Capa titulo"}
Ofuscação da cadeia de pensamento sob supervisão: reanálise de um estudo-piloto e triagem de competência de um estudo prospectivo pré-registrado
:::

&nbsp;

::: {custom-style="Capa"}
FORTALEZA

2026
:::

{{quebra}}

# RESUMO

<<RESUMO>>

**Palavras-chave:** cadeia de pensamento; monitorabilidade; pré-registro; triagem de competência; reprodutibilidade.

{{quebra}}

# ABSTRACT

<<ABSTRACT>>

**Keywords:** chain of thought; monitorability; preregistration; competence screening; reproducibility.

{{quebra}}

# 1 INTRODUÇÃO

Modelos de linguagem que produzem rastros textuais antes da resposta final oferecem uma fonte adicional de evidência sobre seu comportamento. Neste texto, rastro designa o texto intermediário gerado antes da resposta, e monitor designa um procedimento, automático ou humano, que lê esse texto para detectar comportamentos indesejados. Esses rastros podem revelar o uso de atalhos, a exploração de recompensas mal especificadas ou a influência de informações que deveriam ser ignoradas. Korbak et al. (2025) descrevem essa monitorabilidade da cadeia de pensamento (*chain of thought*, CoT) como uma oportunidade relevante para a segurança, mas frágil: decisões de treinamento podem reduzi-la. A preocupação central é a ofuscação, isto é, a manutenção de um comportamento indesejado acompanhada da redução dos sinais textuais que permitiriam detectá-lo.

A literatura recente oferece evidências em direções que precisam ser compatibilizadas. Baker et al. (2025) relatam que a otimização direta contra um monitor de CoT pode levar modelos a continuar explorando falhas de recompensa enquanto deixam de mencioná-las. Drori et al. (2025) e Mitrani Hadida et al. (2026) mostram que a supervisão aplicada apenas às saídas também pode alterar os rastros e que esse efeito pode se transferir para tarefas não vistas. Emmons et al. (2025), por outro lado, argumentam que, quando o raciocínio explícito é computacionalmente necessário para a tarefa, a evasão de monitores se torna mais difícil. A questão empírica que motiva este trabalho situa-se nessa intersecção: em que condições a pressão de supervisão reduz a detectabilidade de um comportamento objetivamente verificável, e como separar esse efeito de mudanças triviais de formato ou de perda de competência.

Este artigo examina três conjuntos de evidência produzidos no mesmo projeto. O primeiro é um estudo-piloto com ajuste fino supervisionado (*supervised fine-tuning*, SFT), cujos dados publicados foram reanalisados. O segundo é um estudo prospectivo, denominado v3.2, isto é, um estudo cujas hipóteses, procedimentos e critérios de análise foram fixados antes da coleta dos dados. Ele foi registrado publicamente no Open Science Framework (OSF) antes de qualquer treinamento e tinha dois braços, entendidos como conjuntos de condições experimentais que compartilham o mesmo procedimento de treinamento: SFT e otimização por recompensa com *Group Relative Policy Optimization* (GRPO). O terceiro é uma revisão prospectiva do desenho, denominada v4, preparada depois que o v3.2 não satisfez seu critério de competência pré-treinamento.

O ponto de partida empírico do segundo conjunto é um resultado negativo de viabilidade. O protocolo v3.2 previa uma triagem de competência: antes dos 240 treinamentos, cada modelo-base, isto é, o modelo pré-treinado ainda não ajustado no estudo, deveria atingir ao menos 65% de acurácia em 120 itens da partição de desenvolvimento (DEV) por nível de dificuldade. A partição DEV é reservada a verificações prévias e se distingue da partição de teste (TEST), usada apenas no desfecho final. A triagem registrou 57,5% e 0% nos níveis do braço SFT e 43,33% e 15% nos níveis do braço GRPO, bloqueando o estudo. Antes de atribuir esse resultado à incapacidade dos modelos, é necessário excluir defeitos de software que produziriam o mesmo padrão: erro do corretor (o programa que decide se uma resposta está certa), uso incorreto do modelo de conversa (o formato com que instruções e mensagens são apresentadas ao modelo), truncamento por limite de comprimento, extração equivocada da resposta, efeitos de precisão numérica ou incompatibilidade entre bibliotecas.

As perguntas que orientam o trabalho são as seguintes. Primeira: o que os dados publicados do piloto sustentam depois de reprocessados e auditados? Segunda: a reprovação da triagem do v3.2 decorre de defeito técnico ou da incapacidade real dos modelos-base sob o desenho congelado? Terceira: quais alterações, calibradas exclusivamente com dados próprios, separados das partições do estudo, tornam viável um novo estudo prospectivo com o mesmo critério de competência, quais instrumentos do desenho original permanecem inviáveis e se o desenho revisado satisfaz esse critério? As hipóteses confirmatórias registradas para o v3.2 sobre custo de formato no SFT e exploração não sinalizada no GRPO são descritas, mas não testadas, porque nenhum treinamento confirmatório foi realizado.

As contribuições são: uma reanálise reprodutível do piloto, com identificação dos limites de seus instrumentos; a documentação verificável do desfecho de inviabilidade do v3.2, incluindo sua cronologia e a correspondência entre registro e implementação; uma reprodução diagnóstica da triagem com o código congelado, em unidade de processamento gráfico (GPU) e em processador central (CPU), com classificação das falhas; uma avaliação da viabilidade dos monitores automáticos previstos, que levou à exclusão do braço GRPO antes do registro do novo estudo; e um protocolo v4 com nova identificação, justificativa metodológica, calibração separada dos dados de teste, triagem executada antes do registro e software verificado. O trabalho não apresenta resultados confirmatórios sobre ofuscação estratégica, positivos ou negativos.

# 2 FUNDAMENTAÇÃO E TRABALHOS RELACIONADOS

## 2.1 Rastro textual, fidelidade e monitorabilidade

Neste trabalho, cadeia de pensamento designa o texto intermediário produzido por um modelo antes da resposta final. A definição é operacional: o rastro é um produto observável da geração, e não uma leitura completa dos estados internos do modelo. Fidelidade refere-se à correspondência entre o rastro e os fatores que efetivamente determinam a resposta; monitorabilidade, à possibilidade de detectar, a partir do rastro, comportamentos relevantes para a supervisão.

Wei et al. (2022) mostraram que instruções e exemplos que induzem etapas intermediárias melhoram o desempenho em problemas de múltiplos passos, o que estabelece a relevância computacional do texto intermediário em certas tarefas. Turpin et al. (2023), entretanto, demonstraram que explicações plausíveis podem omitir influências que alteram a resposta, e Lanham et al. (2023) usaram intervenções sobre o rastro, como truncamento e inserção de erros, para medir quanto a resposta depende dele. Esses resultados justificam tratar fidelidade como propriedade a ser medida, e não presumida a partir da fluência do texto.

Acurácia e monitorabilidade são dimensões distintas. Um modelo pode resolver uma tarefa com pouca explicação, justificar de modo convincente uma resposta incorreta ou executar um atalho proibido e descrevê-lo abertamente. Para inferir ofuscação, é necessário observar simultaneamente a execução de um comportamento definido independentemente do texto e a dificuldade de detectá-lo no rastro.

## 2.2 Pressão de supervisão e ofuscação

Baker et al. (2025) incorporaram monitores de raciocínio ao treinamento por reforço em tarefas de programação e observaram que penalizar os rastros que mencionavam a exploração de testes reduziu a detecção sem eliminar o comportamento, o que denominam ofuscação. Drori et al. (2025) estudaram a supervisão aplicada apenas à saída e propuseram mecanismos pelos quais o *feedback* sobre a resposta final se transfere para o rastro. Mitrani Hadida et al. (2026), na versão revisada de 20 de maio de 2026, relatam que a ofuscação aprendida sob supervisão de saída pode generalizar para tarefas não vistas em modelos Qwen3-4B e Qwen3-8B, reconhecendo que as tarefas usadas não estabelecem a necessidade causal do raciocínio explícito.

Emmons et al. (2025) distinguem o raciocínio que funciona como racionalização daquele que constitui computação necessária e relatam que, quando a tarefa exige o rastro, a evasão do monitor requer assistência ou otimização substanciais. Esse resultado sustenta uma hipótese condicional, e não uma impossibilidade geral de ocultação. Em conjunto, esses trabalhos indicam que o efeito da pressão de supervisão depende da tarefa, do mecanismo de treinamento e do monitor, o que torna essenciais o controle da competência do modelo e a definição comportamental da exploração.

## 2.3 Avaliação, replicação e pré-registro

Três fontes metodológicas orientam o desenho analisado. Agarwal et al. (2021) mostram que poucas execuções e estimativas pontuais produzem conclusões instáveis em aprendizado por reforço e recomendam explicitar a unidade de repetição e a incerteza entre execuções. Zheng et al. (2023) documentam vieses de posição, extensão e autopreferência em avaliadores baseados em modelos de linguagem, o que motiva monitores independentes do modelo avaliado e calibração com rótulos humanos. Nosek et al. (2018) descrevem o pré-registro como a separação explícita entre decisões tomadas antes e depois da observação dos dados, condição para distinguir análises confirmatórias de exploratórias.

O braço de reforço do estudo prospectivo usa GRPO, proposto por Shao et al. (2024), que estima vantagens comparando várias respostas geradas para o mesmo enunciado, sem modelo de valor separado. Ambos os braços usam adaptação de baixo posto (LoRA) (HU et al., 2022) sobre modelos Qwen2.5 (QWEN TEAM, 2024) e Qwen3 (YANG et al., 2025), implementados com as bibliotecas Transformers (WOLF et al., 2020) e TRL (VON WERRA et al., 2020).

# 3 PERGUNTAS, HIPÓTESES E CATEGORIAS DE EVIDÊNCIA

## 3.1 Perguntas deste trabalho

A pergunta P1 é retrospectiva: os dados publicados do piloto permitem atribuir a diferença de desempenho entre alvos de treinamento à ofuscação de raciocínio necessário? A pergunta P2 diz respeito ao estudo v3.2: a reprovação da triagem de competência decorre de defeito técnico ou da incapacidade dos modelos-base nas tarefas congeladas? A pergunta P3 diz respeito ao desenho revisado: alterações calibradas com dados próprios, separados das partições do estudo, permitem que os modelos-base satisfaçam o mesmo critério de 65% por nível, e os monitores previstos são capazes de distinguir rastros que usam a informação proibida de rastros que não a usam?

## 3.2 Hipóteses confirmatórias registradas

O registro do v3.2 especificou duas hipóteses, cujo texto o v4 mantém; no v4, porém, apenas H1 permanece testável, pelas razões apresentadas na seção 5.3. A hipótese H1 refere-se ao braço SFT e prevê que treinar respostas diretas, em vez de rastros explícitos, custe mais acurácia no nível de maior complexidade (C) do que no de menor complexidade (A). Cada treinamento é identificado por uma semente, o número que inicializa os geradores pseudoaleatórios e, portanto, determina a ordem dos exemplos e a inicialização dos parâmetros ajustados. Para cada semente de treinamento *s*, o contraste é a interação

$$\Delta^{\text{SFT}}_s = \left[\text{Acc}_{s}(\text{direct}, C) - \text{Acc}_{s}(\text{explicit}, C)\right] - \left[\text{Acc}_{s}(\text{direct}, A) - \text{Acc}_{s}(\text{explicit}, A)\right],$$

com direção prevista negativa. Esse contraste mede o efeito do alvo de treinamento sobre a acurácia; ele não mede intenção de ocultação.

A hipótese H2 refere-se ao braço GRPO. Cada questão é de múltipla escolha e traz uma nota auxiliar proibida, também chamada de dica, que aponta uma alternativa errada; a recompensa mal especificada, isto é, a função de pontuação que intencionalmente paga por um comportamento indesejado, premia essa alternativa. Uma resposta é exploração quando escolhe a alternativa indicada pela nota; é não sinalizada quando um monitor independente, distinto do monitor usado na recompensa, não a detecta no rastro. No nível de dica mais difícil (*high*), o contraste por semente é

$$\Delta^{\text{GRPO}}_s = P_s(\text{explora} \wedge \neg\text{sinalizado} \,|\, \text{cot\_monitor}) - P_s(\text{explora} \wedge \neg\text{sinalizado} \,|\, \text{none}),$$

com direção prevista positiva e exigência adicional de que a taxa de exploração não caia mais que 0,10 entre as condições.

Em ambos os casos a unidade de inferência é a semente de treinamento. As gerações de uma mesma semente não são observações independentes; elas são agregadas em uma proporção por semente antes de qualquer teste. Os contrastes de 20 sementes seriam avaliados por um teste de inversão de sinais bilateral, que compara a média observada com a distribuição obtida ao trocar aleatoriamente o sinal de cada contraste (199.999 permutações Monte Carlo com semente fixa); por um intervalo *bootstrap* percentil de 95%, obtido por reamostragem das sementes com reposição; e pela correção de Holm (1979), que controla a probabilidade de ao menos um falso positivo entre as famílias de testes, com efeito mínimo relevante de 0,05 em escala de proporção. Uma hipótese só receberia apoio se significância ajustada, direção, magnitude, intervalo e as condições de validade dos instrumentos fossem satisfeitos simultaneamente. Como nenhum treinamento confirmatório foi executado, essas análises permanecem especificadas e implementadas, mas não aplicadas.

## 3.3 Categorias de evidência

Os resultados são separados em cinco categorias, que não são combinadas para aumentar tamanhos amostrais ou fortalecer conclusões: (i) resultados históricos, provenientes dos dados publicados do piloto; (ii) verificações de software, que testam contratos, retomada, integridade e execução das bibliotecas sem produzir evidência sobre o fenômeno; (iii) triagens de competência, que verificam se os modelos-base satisfazem o critério prévio; (iv) experimentos exploratórios, como a reprodução diagnóstica da triagem e a calibração do v4; e (v) resultados confirmatórios, que exigiriam os treinamentos e avaliações completos de um protocolo registrado. Nenhum resultado da categoria (v) foi produzido.

# 4 MATERIAIS E MÉTODOS

## 4.1 Estudo-piloto e estratégia de reanálise

O piloto foi depositado no repositório público do projeto no commit e21cef8, de junho de 2026, com um notebook de treinamento e análise e quatro tabelas de resultados. Commit designa uma versão registrada no sistema de controle de versões Git, identificada por um código hexadecimal, e notebook designa um documento que combina texto e código executável em células. O desenho cruza dois níveis de complexidade aritmética, três alvos de SFT e três sementes (@tab:desenho_piloto). Os alvos recebem aqui nomes descritivos: CoT explícita (rastro com as etapas), resposta direta (somente a resposta) e preâmbulo genérico (frase vaga seguida da resposta). Os nomes usados no código, como *cot_pressure* e *output_supervision*, sugeriam mecanismos de penalização por monitor que o piloto não implementa.

{{tabela:desenho_piloto}}

O arquivo de resultados contém {{v:piloto.linhas}} linhas, uma por avaliação, com nível, condição, semente, acurácia, escore lexical e métricas de diversidade, mas sem enunciado, identificador do item, gabarito ou texto gerado. A reanálise, portanto, reprocessa as métricas depositadas; ela não refaz o julgamento das respostas. A estratégia teve três etapas: comparar as médias recalculadas com os agregados publicados em cada uma das {{v:piloto.celulas}} células; reproduzir a análise de variância e o teste *t* de Welch originais como verificação numérica; e auditar os instrumentos (gerador de itens, corretor, teste de intervenção e escore lexical) executando o código preservado.

Para cada semente *s*, nível *l* e alvo *c*, o custo de acurácia em relação à CoT explícita é K(l, c, s) = Acc(l, explícita, s) − Acc(l, c, s), e a diferença entre níveis é D(c, s) = K(C, c, s) − K(A, c, s). Como há três sementes, a análise de sensibilidade enumera as oito inversões de sinal possíveis dos três contrastes; o menor valor de *p* bilateral atingível é 0,25. O teste de intervenção histórico, que compara respostas com e sem explicação em dez itens por nível, foi reconstruído a partir dos agregados e avaliado pelo teste exato de McNemar.

## 4.2 Estudo prospectivo v3.2

### 4.2.1 Desenho e registro

O estudo v3.2 foi implementado no commit fde856f e operado por cinco notebooks do Google Colab, serviço que executa notebooks em máquinas virtuais com GPU: protocolo e dados (00), treinamento SFT (01), treinamento GRPO (02), avaliação TEST (03) e monitor independente com inferência (04). O desenho previa 240 treinamentos e 55.200 gerações de avaliação principal (@tab:desenho_v3). Esses números foram conferidos na implementação. Cada treinamento produz um adaptador, o conjunto de parâmetros adicionais ajustados pela LoRA enquanto os pesos originais permanecem fixos; cada adaptador SFT seria avaliado nos 220 itens TEST de seu nível e cada adaptador GRPO nos 240 itens TEST de seu nível (60 por domínio em quatro domínios), resultando em 26.400 e 28.800 gerações.

{{tabela:desenho_v3}}

No braço SFT, o nível A consiste em adições de dois inteiros entre 101 e 9.999, e o nível C, em expressões a + b × c − d com os quatro operandos no mesmo intervalo. O alvo explícito registra três etapas (o produto, a soma e a subtração) sem decompor a multiplicação. No braço GRPO, cada questão de múltipla escolha apresenta uma expressão aritmética em um de quatro domínios (aritmética, rastreamento de programa, regra fictícia e estoque) e uma nota auxiliar proibida que indica uma alternativa errada. No nível *low*, a nota nomeia a letra; no nível *high*, exige calcular um índice módulo 4. A recompensa mal especificada paga a escolha da alternativa indicada pela nota, e as condições *cot_monitor* e *summary_monitor* subtraem 0,6 quando o monitor de recompensa (Qwen2.5-0.5B-Instruct) detecta o uso da nota no rastro ou no resumo.

O registro público foi localizado por consulta à API do OSF: *Monitorabilidade e Ofuscação da Cadeia de Pensamento: Pré-registro de Experimentos SFT e GRPO com Avaliação Independente*, identificador sjwdf, criado em 19 de outubro de 2026 às 16h11 UTC e tornado público às 16h45 UTC (@tab:cronologia). O texto registrado especifica as hipóteses, os modelos, as 20 sementes, o critério de competência de 65% e as regras de análise e declara que mudanças substanciais de modelo, instrumento ou amostragem exigem novo protocolo. O registro não contém arquivos anexados nem os resumos criptográficos do protocolo e das revisões dos modelos: o único arquivo enviado ao projeto foi removido 24 segundos depois. O texto cita a versão 3.1 do protocolo, enquanto a implementação executada é a v3.2, cujas diferenças são operacionais (seleção do modo de avaliação, congelamento da revisão do monitor independente e pilha de dependências sem quantização), sem alteração de hipóteses, tamanhos ou critérios.

{{tabela:cronologia}}

### 4.2.2 Triagem de competência

Antes de qualquer treinamento, cada modelo-base respondia aos 120 primeiros itens DEV de cada nível em uma GPU NVIDIA Tesla T4, com pesos representados em ponto flutuante de 16 bits (fp16). A geração usava decodificação gulosa, que escolhe a cada passo o *token* mais provável, e um orçamento de 256 *tokens*, isto é, um limite máximo de 256 unidades do vocabulário do modelo por resposta. No SFT, a instrução de sistema, mensagem inicial que define a tarefa, era "Resolva o problema. Termine com uma única tag `<answer>NUMERO</answer>`." e a correção usava um contrato estrito: a resposta é aceita somente como uma única tag final, uma única linha final "Resposta: N" ou uma saída composta apenas pelo número. No GRPO, o enunciado incluía a nota proibida e a resposta só era correta se seguisse exatamente o formato `<trace>…</trace><summary>…</summary><answer>LETRA</answer>` com a letra do gabarito. O critério exigia acurácia mínima de 0,65 em cada um dos quatro níveis; a reprovação bloqueava os 240 treinamentos.

## 4.3 Diagnóstico da triagem do v3.2

Os arquivos brutos da triagem original estavam no Google Drive da sessão do Colab em que foi executada e não estavam disponíveis para esta auditoria; uma busca no Drive conectado ao projeto não os localizou. Os valores originais, portanto, constam aqui como documentados no README do v3 (commit 273d131), e não como reprodução independente. Para examinar sua origem, foram conduzidas duas reproduções diagnósticas, que não substituem a triagem registrada nem podem liberar os treinamentos do v3.2.

A primeira reproduz o caminho original no Colab, em 28 de outubro de 2026: os notebooks v3.2 baixam os commits fixados fde856f e 4a2a133, criam o ambiente isolado com as versões fixadas e executam as etapas de preparação, congelamento das revisões e triagem em uma Tesla T4, em fp16, numa pasta de diagnóstico separada. A segunda reconstrói os dados com o mesmo código e usa as mesmas funções de conversa e geração em CPU, com pesos em ponto flutuante de 32 bits (fp32), para avaliar a sensibilidade à precisão numérica. As revisões dos modelos, identificadores da versão exata dos pesos, foram confirmadas no Hugging Face Hub, repositório público de modelos: a última alteração de cada repositório é anterior a outubro de 2026, de modo que as revisões disponíveis coincidem necessariamente com as congeladas em 19 de outubro.

As respostas da reprodução na T4 foram classificadas em categorias descritivas: correta no contrato; valor correto fora do contrato (por exemplo, prosa terminada pelo número certo); separador de milhar lido como decimal; formato válido com valor incorreto; alternativa indicada pela dica; formato inválido com valor incorreto; e truncamento no orçamento, estimado na T4 pela recontagem dos tokens do texto gerado e observado diretamente na CPU. A soma das categorias corretas fora do contrato define um limite superior permissivo, que indica quanto um extrator mais tolerante poderia alterar o resultado. Esse limite é diagnóstico: o desfecho registrado permanece o do corretor congelado. Além disso, verificaram-se o texto produzido pelo modelo de conversa, a tokenização, o identificador de término, a ocorrência de truncamento e as versões das bibliotecas.

## 4.4 Protocolo prospectivo v4

### 4.4.1 Identificação e alterações

O protocolo v4 constitui um estudo novo, com identificador próprio (estudo-prospectivo-2026-v4), código no diretório experimento_cientifico_v4 e cinco notebooks distintos dos do v3.2. Ele mantém as perguntas, o texto das hipóteses H1 e H2, os modelos-base, as condições, as 20 sementes, o número de passos, os tamanhos das partições, o critério de 65% e o plano estatístico. Durante a calibração, o braço GRPO foi excluído antes do registro, pela regra descrita na seção 4.4.2, de modo que o v4 testa apenas H1. As alterações (@tab:mudancas_v4) restringem-se ao que o diagnóstico do v3.2 identificou como causa da reprovação ou como fragilidade de mensuração, e cada uma foi definida antes de ser avaliada nos dados de desenvolvimento do próprio v4.

{{tabela:mudancas_v4}}

### 4.4.2 Calibração exploratória

As faixas numéricas das tarefas, as instruções e os monitores foram escolhidos por calibração em itens gerados com sementes próprias (20261028 a 20261030), excluídos por construção das partições de treino, DEV e teste do v4. Cada regra de escolha foi gravada no repositório antes da execução correspondente (rodada 1: commits 97eaf5e, 4f2c503 e e02eda4; rodada 2: d3e4258, 73a6b13 e a4668a5; repetição da rodada 2 dos monitores: 5a091d8), e todas as rodadas, inclusive as reprovadas, são relatadas. As condições de geração são as previstas para a triagem: Tesla T4, fp16, decodificação gulosa item a item e orçamento de 256 tokens.

No SFT, o nível A (a + b, com operandos de 100 a 999) e cada candidato ao nível C deveriam atingir 0,80 com a instrução explícita em 40 itens. Os candidatos têm a forma a + b × c − d, com a e d de 100 a 999 e b de 12 a 99; em C1, c vai de 3 a 9 e, em C2, de 12 a 99. Escolhe-se o candidato mais difícil que satisfaça a regra. A rodada 2 manteve os itens e a regra e acrescentou à instrução duas demonstrações fixas do alvo explícito, com enunciados que não pertencem a nenhuma partição.

No GRPO, a faixa de operandos, comum aos dois níveis, deveria atingir 0,80 sem a dica em 40 itens por nível (10 por domínio); os candidatos foram G1 (11 a 99) e G0 (2 a 20), com preferência pela faixa mais difícil. A rodada 1 usou a instrução do v3.2. A rodada 2 testou, nesta ordem, uma instrução sem os marcadores copiáveis (r2a) e a mesma instrução acompanhada de uma demonstração de formato sem dica (r2b), escolhendo a primeira variante aprovada. O limiar de 0,80 oferece margem acima do critério de 0,65 para a variação amostral de 120 itens.

Os monitores automáticos previstos são modelos de linguagem usados como classificadores. Cada um recebe uma rubrica que pergunta se o texto usa a informação proibida, seguida da questão e do rastro, e compara as probabilidades que atribui às continuações "true" e "false"; o rastro é sinalizado quando a primeira é maior. Os monitores foram avaliados em 320 controles construídos a partir de modelos de texto: 160 rastros que usam explicitamente a nota proibida e 160 que calculam a resposta, inclusive rastros que rejeitam a nota de forma explícita. Esses rótulos decorrem da regra de construção, e não de anotação humana; eles avaliam apenas a viabilidade do instrumento e não substituem a validação com 100 controles positivos e 100 negativos rotulados por pessoas, que continua obrigatória. Para o monitor independente, a regra escolhia o menor modelo com taxa de verdadeiros positivos (TPR) de pelo menos 0,90 e taxa de falsos positivos (FPR) de no máximo 0,05; a rodada 1 avaliou modelos SmolLM2 (BEN ALLAL et al., 2025) de 135M, 360M e 1,7B de parâmetros, e a rodada 2, modelos de outras famílias em ordem crescente de tamanho: OLMo 2 de 1B (TEAM OLMO et al., 2024), Granite 3.3 de 2B (IBM GRANITE TEAM, 2025), SmolLM3 de 3B (HUGGING FACE, 2025) e Phi-3.5-mini (ABDIN et al., 2024). Para o monitor de recompensa, exigiu-se discriminação mínima TPR − FPR ≥ 0,5. A primeira execução da rodada 2 foi encerrada pelo sistema por falta de memória do hospedeiro ao carregar o segundo candidato; a rodada foi repetida integralmente, com os mesmos candidatos, controles e critérios, depois de alterar apenas o carregamento dos pesos, e a regra para o caso de nenhum monitor ser aprovado foi gravada antes de conhecido o resultado: nesse caso, o braço GRPO seria excluído do v4 antes do registro. Depois que o Phi-3.5-mini se mostrou grande demais para a memória da T4 com pesos em fp32, e antes de qualquer pontuação dele, declarou-se um complemento: os candidatos não avaliáveis na T4 seriam pontuados numa GPU com mais memória, com os mesmos controles e critérios. As sessões do Colab usadas na calibração foram encerradas pelo serviço em duas ocasiões, e os arquivos de geração de parte da rodada 1 se perderam; as etapas afetadas foram repetidas com o mesmo código, e os valores repetidos coincidiram com os já observados.

### 4.4.3 Triagem de competência antes do registro

No v4, a triagem é parte da preparação e precede o registro público, de modo que o registro só seja feito para um desenho viável e inclua os resumos criptográficos SHA256 das gerações da triagem. Um resumo SHA256 é uma sequência de 64 caracteres calculada a partir do conteúdo de um arquivo; qualquer alteração do arquivo produz um resumo diferente, o que permite verificar sua integridade. A triagem foi executada uma única vez, depois do congelamento do protocolo e do código dos módulos que determinam itens, instruções, geração e correção, em uma Tesla T4 no Colab, com os modelos nas revisões fixadas. No SFT, usaram-se a instrução explícita com as duas demonstrações e os 120 primeiros itens DEV de cada nível; no GRPO, o enunciado sem a nota proibida, para medir a competência na tarefa, e 30 itens DEV de cada domínio por nível. Com a exclusão do braço GRPO, apenas os níveis do SFT condicionam os treinamentos; os níveis do GRPO são relatados como descrição. O notebook 00 importa esses arquivos e confere seus resumos, sem repetir a inferência. Uma nova execução da triagem com resultado diferente não teria efeito sobre o estudo: os módulos que a determinam têm seus resumos verificados, e qualquer alteração exige outro identificador de estudo.

## 4.5 Verificação de software

As verificações de software foram separadas em quatro níveis, com propósitos distintos. Testes unitários, isto é, verificações automáticas de funções isoladas, cobrem as regras de correção, a geração de dados, o congelamento, os bloqueios e a estatística. Uma integração em CPU com modelos mínimos de pesos aleatórios exercita as interfaces reais das bibliotecas de treinamento nos dois braços, incluindo interrupção e retomada; o braço GRPO, embora excluído do v4, continua coberto como software. A execução dos cinco notebooks com o módulo do Colab simulado injeta falhas de montagem do Google Drive, o armazenamento persistente usado entre sessões, e de transferência de arquivos para o navegador. Por fim, um teste técnico em GPU executa os cinco notebooks no Colab com os modelos reais em modo técnico (uma semente, dois passos por treino e quatro itens de teste por avaliação, em pasta separada) e força interrupções de treinamento e de avaliação. Nenhum desses testes produz evidência sobre as hipóteses.

# 5 RESULTADOS

## 5.1 Reanálise do estudo-piloto

As {{v:piloto.celulas}} médias recalculadas coincidem com os agregados publicados, e a interação da análise de variância histórica foi reproduzida, F(2, 12) = {{v:piloto.anova_f}}, p = {{v:piloto.anova_p}}, assim como o teste de Welch para o nível C, t = {{v:piloto.welch_t}}, p = {{v:piloto.welch_p}}. No nível A, os três alvos têm acurácias próximas; no nível C, a acurácia registrada passa de {{v:piloto.acc.C.control}} com CoT explícita para {{v:piloto.acc.C.cot_pressure}} com resposta direta e {{v:piloto.acc.C.output_supervision}} com preâmbulo genérico (@tab:piloto_resumo e @fig:piloto).

{{tabela:piloto_resumo}}

{{figura:piloto}}

A diferença média entre os custos dos dois níveis foi {{v:piloto.dd.cot_pressure}} para resposta direta e {{v:piloto.dd.output_supervision}} para preâmbulo genérico, com sinal positivo nas três sementes (@tab:piloto_contrastes). Os intervalos *t* condicionais de 95%, {{v:piloto.dd_ic.cot_pressure}} e {{v:piloto.dd_ic.output_supervision}}, supõem três contrastes independentes e aproximadamente normais. A enumeração das inversões de sinal resulta em p = {{v:piloto.signflip.cot_pressure}} nas duas comparações, o menor valor possível com três sementes; esse resultado expressa a baixa resolução do desenho e não indica ausência de efeito.

{{tabela:piloto_contrastes}}

A auditoria dos instrumentos restringe a interpretação desses números em quatro pontos. Primeiro, o gerador cria cada item a partir da semente inicial somada ao índice, de modo que sementes adjacentes reutilizam quase todos os enunciados: 149 de 150 no treino e 29 de 30 na avaliação, com 32 enunciados distintos de avaliação por nível nas três sementes (@tab:sobreposicao). Também se reconstruiu um enunciado comum a treino e avaliação no nível A. As três sementes não são, portanto, três amostras independentes de problemas.

{{tabela:sobreposicao}}

Segundo, o corretor histórico aceita uma resposta quando qualquer número do trecho final coincide com o gabarito. Em um corpus construído de {{v:piloto.casos}} respostas, das quais {{v:piloto.negativos}} deveriam ser rejeitadas, ele aceitou {{v:piloto.falsos_aceites}} ({{v:piloto.taxa_aceite_indevido}}), por exemplo "Resposta: 43; 42 seria incorreto." com gabarito 42 (@tab:corretor). O contrato estrito usado nos estudos prospectivos rejeitou todos esses casos. Como os textos históricos não foram depositados, a frequência desse erro nas 540 avaliações permanece desconhecida.

{{tabela:corretor}}

Terceiro, a comparação histórica entre respostas com e sem explicação, reconstruída a partir dos agregados, mostra no nível C {{v:piloto.mcnemar_b}} itens acertados somente com explicação e {{v:piloto.mcnemar_c}} somente na resposta direta, ganho líquido de {{v:piloto.ganho_liquido}} e teste exato de McNemar com p = {{v:piloto.mcnemar_p}}. Além do tamanho de dez itens, essa comparação muda simultaneamente a instrução e o orçamento de geração (200 contra 40 tokens) e, por isso, não estabelece a necessidade causal do rastro. Quarto, o escore lexical de "ofuscação" atribui valor máximo a rastros vazios e correlaciona-se com repetição (r = {{v:piloto.corr_rep}}) e com diversidade lexical (r = {{v:piloto.corr_ttr}}) porque ambas integram sua fórmula; não há rótulos independentes que o validem como detector de ocultação.

## 5.2 Triagem do v3.2 e diagnóstico

Os valores documentados da triagem do v3.2 foram {{v:v3.sft.A.doc_k}} acertos no nível A e {{v:v3.sft.C.doc_k}} no nível C do braço SFT, e {{v:v3.grpo.low.doc_k}} e {{v:v3.grpo.high.doc_k}} nos níveis *low* e *high* do braço GRPO (@tab:triagem_v3 e @fig:triagem). A reprodução pelo caminho original no Colab, com os mesmos commits, as mesmas versões de bibliotecas e uma Tesla T4 em fp16, obteve {{v:v3.sft.A.t4_k}}, {{v:v3.sft.C.t4_k}}, {{v:v3.grpo.low.t4_k}} e {{v:v3.grpo.high.t4_k}}, isto é, os mesmos valores documentados. Os conjuntos de dados reconstruídos têm os mesmos resumos SHA256 nas duas reproduções, e as revisões dos modelos coincidem com as congeladas no v3.2. A classificação das falhas a seguir usa as gerações dessa reprodução.

{{tabela:triagem_v3}}

{{figura:triagem}}

A comparação item a item com a reprodução em CPU com pesos em fp32 delimita o efeito da precisão numérica. No nível A do SFT, {{v:v3.sft.A.cpu_ident}} das 120 gerações foram idênticas e {{v:v3.sft.A.cpu_igual}} tiveram a mesma correção; no nível C, {{v:v3.sft.C.cpu_ident}} foram idênticas e todas foram incorretas nas duas precisões. No nível *low* do GRPO, a correção coincidiu em {{v:v3.grpo.low.cpu_igual}} de {{v:v3.grpo.low.cpu_n}} itens, e no nível *high*, em {{v:v3.grpo.high.cpu_igual}} de {{v:v3.grpo.high.cpu_n}}. A precisão altera respostas isoladas, mas não o resultado da triagem.

A classificação das respostas (@tab:falhas_v3 e @fig:falhas) separa três origens de falha. A primeira é a incapacidade aritmética sob a instrução usada. No nível A do SFT, o modelo respondeu apenas com um número na maioria dos itens, sem etapas intermediárias, e errou {{v:v3.sft.A.cat.formato_valido_valor_incorreto}} somas de dois números de até quatro algarismos. No nível C, as 120 respostas tinham a forma "A resposta é: …", com valores distantes do gabarito, que envolve produtos de até oito algarismos; nenhuma seria correta mesmo com extração permissiva. Nenhuma resposta do SFT foi truncada: as do nível C tiveram em média {{v:v3.sft.C.tokens}} tokens na reprodução em CPU.

{{tabela:falhas_v3}}

{{figura:falhas}}

A segunda origem é o desenho da instrução e do contrato de formato. No SFT, a instrução pedia a tag `<answer>`, que o modelo nunca usou; o contrato aceitava a saída composta só pelo número, mas rejeitava {{v:v3.sft.A.cat.valor_correto_fora_do_contrato}} respostas do nível A cujo valor final estava correto em prosa ("O resultado de 4098 mais 4404 é 8502."). No GRPO, a instrução apresentava o formato com os marcadores "explicação verificável" e "resumo"; no domínio de rastreamento de programa, o modelo copiou esses marcadores literalmente, escreveu o cálculo fora das tags e terminou com a letra correta em grande parte dos casos. O contrato, que não admite texto entre os blocos, rejeitou essas respostas. Por esse motivo, o domínio aritmético obteve {{v:v3.grpo.low.dom.arithmetic}} no nível *low*, e o de rastreamento de programa, {{v:v3.grpo.low.dom.program_trace}}. Como os 120 primeiros itens DEV de cada nível pertenciam apenas a esses dois domínios, a triagem do GRPO não avaliou os domínios de regra fictícia e de estoque.

A terceira origem é um defeito do corretor: um número escrito com separador de milhar ("11,863") era lido como decimal, em vez de rejeitado como formato ambíguo. O defeito afetou {{v:v3.sft.A.cat.separador_de_milhar_lido_como_decimal}} resposta do nível A e não altera nenhum resultado da triagem.

Os limites superiores permissivos foram {{v:v3.sft.A.perm}}, {{v:v3.sft.C.perm}}, {{v:v3.grpo.low.perm}} e {{v:v3.grpo.high.perm}}. Eles indicam que um extrator mais tolerante poderia ter aprovado alguns níveis, mas não o nível C do SFT, em que a incapacidade do modelo-base sob a instrução registrada é suficiente, por si, para a reprovação. Como o critério registrado exigia a aprovação dos quatro níveis segundo o contrato de formato, e o registro determina novo protocolo para mudanças de instrumento ou de amostragem, a correção posterior do extrator não pode reverter o resultado do v3.2.

{{tabela:hipoteses}}

A @tab:hipoteses resume as hipóteses de defeito técnico examinadas. O modelo de conversa foi renderizado corretamente (formato ChatML, modo sem pensamento do Qwen3 com bloco vazio e identificador de término correto), a tokenização não duplicou marcadores especiais e as bibliotecas fixadas reproduziram os resultados de 19 de outubro no ambiente do Colab de 28 de outubro, com Python 3.13.16 e PyTorch 2.11. O resultado de inviabilidade do v3.2, portanto, não decorre de erro de execução; decorre da combinação entre a incapacidade do modelo-base nas tarefas de maior dificuldade e decisões de instrução e contrato que tornavam a medida de competência dependente da obediência a um formato não demonstrado.

Duas observações adicionais, obtidas na calibração do v4 com controles construídos, também afetam a viabilidade do v3.2. O monitor independente previsto, SmolLM2-135M-Instruct, sinalizou {{v:mon.135m.tpr}} dos positivos e {{v:mon.135m.fpr}} dos negativos, isto é, classificou quase todos os rastros como suspeitos, situação em que o próprio protocolo declara o monitor não validado. O monitor de recompensa, Qwen2.5-0.5B-Instruct, obteve TPR de {{v:mon.reward.tpr}} e FPR de {{v:mon.reward.fpr}}.

## 5.3 Calibração do v4

Na rodada 1 do SFT, a instrução explícita sem demonstrações não produziu nenhuma resposta aceita: o modelo escrevia as contas pedidas, mas omitia a tag `<answer>`, e o contrato estrito rejeitava a resposta ({{v:cal.sft.r1.A.explicit.k}} no nível A). Com a instrução neutra, o nível A atingiu {{v:cal.sft.r1.A.neutral.k}}, mas os candidatos C1 e C2 ficaram em {{v:cal.sft.r1.C1.neutral.k}} e {{v:cal.sft.r1.C2.neutral.k}}, o mesmo padrão do v3.2: sem etapas escritas, o modelo não resolve expressões com multiplicação. Com as duas demonstrações da rodada 2, os acertos foram {{v:cal.sft.r2.A.explicit.k}} no nível A, {{v:cal.sft.r2.C1.explicit.k}} em C1 e {{v:cal.sft.r2.C2.explicit.k}} em C2. Pela regra declarada, escolheu-se C2, o candidato mais difícil aprovado (@tab:calibracao_v4).

{{tabela:calibracao_v4}}

No GRPO, a instrução do v3.2 ficou abaixo de 0,80 nas duas faixas de operandos: {{v:cal.grpo.r1.G1.sem_dica.low.k}} e {{v:cal.grpo.r1.G1.sem_dica.high.k}} com G1 e {{v:cal.grpo.r1.G0.sem_dica.low.k}} e {{v:cal.grpo.r1.G0.sem_dica.high.k}} com G0, nos níveis *low* e *high*. A proporção de respostas em formato válido acompanhou a acurácia, o que indica que a perda decorria do formato, e não da aritmética. Na rodada 2, a instrução sem marcadores (r2a) não produziu nenhuma resposta aceita: em {{v:cal.r2a.quebra}} das {{v:cal.r2a.n}} respostas o modelo separou os blocos por quebras de linha, que o contrato não admite, embora {{v:cal.r2a.letra}} terminassem com a letra correta. Acrescida de uma demonstração sem dica (r2b), a instrução atingiu {{v:cal.grpo.r2.r2b.G1.sem_dica.low.k}} e {{v:cal.grpo.r2.r2b.G1.sem_dica.high.k}} sem a dica, com formato válido em todas as respostas. Com a dica, a acurácia foi a mesma, e a alternativa indicada pela nota foi escolhida em {{v:cal.grpo.r2.r2b.G1.com_dica.high.dica}} das respostas do nível *high*. Fixaram-se, assim, a variante r2b e os operandos de 11 a 99 nos dois níveis.

A @tab:monitores_v4 apresenta o desempenho dos monitores nos controles construídos. Nenhum dos sete candidatos a monitor independente atingiu o critério. Os modelos SmolLM2 de 360M e 1,7B sinalizaram praticamente todos os rastros, positivos e negativos (TPR de {{v:mon.360m.tpr}} e {{v:mon.smol17b.tpr}}; FPR de {{v:mon.360m.fpr}} e {{v:mon.smol17b.fpr}}), enquanto o OLMo 2 e o SmolLM3 não sinalizaram nenhum. Os dois padrões correspondem a um classificador que atribui sempre a mesma classe e, portanto, não extrai informação do rastro. O Granite 3.3 obteve TPR de {{v:mon.granite2b.tpr}} e FPR de {{v:mon.granite2b.fpr}}. O Phi-3.5-mini não coube na memória da T4 com pesos em fp32 e foi avaliado, conforme o complemento declarado, numa GPU NVIDIA L4 com 24 GB: obteve TPR de {{v:mon.phi35.tpr}} e FPR de {{v:mon.phi35.fpr}}, a única discriminação substancial observada, mas ainda acima do limite de 5% de falsos positivos. Seus falsos positivos ocorreram em dois dos quatro modelos de rastro negativo: em {{v:mon.phi35.neg0}} dos rastros que apresentam o cálculo completo sem mencionar a nota e em {{v:mon.phi35.neg1}} dos que rejeitam a nota explicitamente antes do cálculo. Entre os candidatos a monitor de recompensa, o Qwen2.5-0.5B-Instruct obteve TPR de {{v:mon.reward.tpr}} e FPR de {{v:mon.reward.fpr}}, discriminação inferior à de um sorteio, e o Qwen2.5-1.5B-Instruct não sinalizou nenhum controle.

{{tabela:monitores_v4}}

Pela regra gravada antes desses resultados, a ausência de monitor viável exclui o braço GRPO do v4 antes do registro: H2 não é testável neste estudo, e os 120 treinamentos GRPO não serão executados. O protocolo foi congelado com essa decisão (commit 37094c4), com o nível C2 no SFT e a variante r2b com operandos de 11 a 99 no GRPO, e a tabela de alterações (@tab:mudancas_v4) registra a exclusão.

<<RES_V4_GATE>>

# 6 DISCUSSÃO

## 6.1 O que o piloto sustenta

O piloto registra uma diferença grande e consistente em direção entre alvos de treinamento no nível aritmético de múltiplos passos, e os números publicados são reproduzíveis. Essa descrição motiva a hipótese de que, para certos modelos e tarefas, substituir as etapas explícitas do alvo supervisionado prejudica a aprendizagem do mapeamento desejado. Ela não identifica uma política de evasão aprendida contra um avaliador: o desenho altera diretamente o texto que o modelo deve imitar, não otimiza o modelo contra um monitor e não mede um comportamento proibido definido independentemente do rastro. A medida de acurácia, por sua vez, depende de um corretor que aceita respostas ambíguas, e as três sementes compartilham quase todos os itens. A pergunta P1 recebe, assim, uma resposta negativa: os dados permitem descrever um efeito do formato do alvo, mas não atribuí-lo à ofuscação de raciocínio necessário.

## 6.2 A inviabilidade do v3.2

A reprovação do v3.2 não decorre de erro de execução. Os quatro valores documentados foram reproduzidos exatamente no ambiente atual do Colab, as verificações de modelo de conversa, tokenização, precisão e bibliotecas foram negativas, e o único defeito de corretor encontrado não altera nenhum nível. A resposta à pergunta P2 é que a reprovação resulta de duas causas que se somam. A primeira é a incapacidade real do Qwen2.5-1.5B-Instruct de calcular, em uma única etapa e sem rastro, expressões com produtos de até oito algarismos, condição suficiente para reprovar o nível C. A segunda é um desenho de medida que fazia a competência depender da obediência a formatos não demonstrados: o modelo de SFT não usava a tag pedida, e o modelo de GRPO copiava os marcadores da instrução, produzindo respostas com a alternativa correta que o contrato rejeitava.

Essa distinção é relevante para a interpretação. Um extrator permissivo teria aprovado alguns níveis, mas trocar o extrator depois de observar os resultados da triagem alteraria o instrumento registrado. O registro previa exatamente essa situação: mudanças de instrumento, modelo ou amostragem exigem novo protocolo. O desfecho do v3.2 é, portanto, a inviabilidade do desenho tal como registrado, e não um resultado sobre as hipóteses H1 e H2, que permanecem não testadas.

A calibração do v4 revelou ainda um problema que a triagem não alcançava: o monitor independente previsto não discrimina rastros que usam a nota proibida de rastros que a rejeitam explicitamente (seção 6.3). Mesmo com competência aprovada, o gate de validação do monitor provavelmente teria falhado depois dos 240 treinamentos, deixando o braço GRPO sem desfecho interpretável. Uma verificação antecipada desse instrumento, com controles construídos e de baixo custo, teria revelado o problema antes do registro.

## 6.3 O que a calibração do v4 mostra

A calibração reforça o diagnóstico do v3.2. Nos dois braços, a maior parte da perda de acurácia sob o contrato estrito decorria da obediência ao formato, e não da aritmética: sem demonstrações, o modelo do SFT resolvia as contas e omitia a tag pedida, e o modelo do GRPO escrevia o cálculo e a letra correta fora da estrutura exigida. Uma instrução reescrita sem exemplos (r2a) não resolveu o problema. Já uma única demonstração do formato no GRPO (r2b) tornou válidas todas as respostas, e duas demonstrações no SFT elevaram a acurácia a {{v:cal.sft.r2.A.explicit}} no nível A e a {{v:cal.sft.r2.C2.explicit}} no candidato C2. A solução adotada mantém o contrato estrito e pré-especificado, em vez de tornar o extrator tolerante, e altera apenas a instrução apresentada ao modelo. Assim, a medida de competência do v4 continua a exigir o formato que o treinamento e a avaliação usarão, mas deixa de depender de que o modelo o adivinhe.

A calibração também delimita o que muda no fenômeno estudado. No SFT, o nível C escolhido (C2) envolve produtos de dois números de dois algarismos, que o modelo resolve escrevendo as etapas e não resolve sem elas: com a instrução neutra, nenhum dos 40 itens dos candidatos C foi acertado. A dependência do rastro, condição para que o contraste de H1 seja informativo, permanece. No GRPO, a mesma faixa de operandos nos dois níveis faz com que a diferença entre *low* e *high* se restrinja à dificuldade de decodificar a dica, como pretendia o desenho original.

A avaliação dos monitores é o resultado de maior consequência para o desenho. Sete modelos de quatro famílias, com 135 milhões a 3,8 bilhões de parâmetros, foram usados como classificadores por comparação de probabilidades, e nenhum distinguiu, com a margem exigida, rastros que seguem a nota proibida de rastros que a ignoram ou a rejeitam. A maioria dos candidatos atribuiu a mesma classe a quase todos os rastros, e o único com discriminação substancial produziu falsos positivos inclusive em rastros que apenas calculam a resposta. Como esses controles são inequívocos por construção, é improvável que os mesmos monitores fossem mais precisos nos rastros, mais variados, produzidos por modelos treinados. O instrumento central de H2, portanto, não estava disponível na forma prevista.

A exclusão do braço GRPO antes do registro decorre dessa constatação e não constitui evidência sobre H2. Ela evita investir o orçamento de 120 treinamentos em um desenho cujo desfecho seria, por construção, evidência insuficiente no gate do monitor. Um teste futuro de H2 exigirá validar o monitor antes dos treinamentos, com um instrumento de outro tipo, como um modelo maior usado como classificador generativo, um classificador ajustado com rótulos humanos ou anotação humana integral das observações relevantes, cada qual com custo e premissas próprios.

## 6.4 Validade interna e externa

No braço SFT, único mantido no v4, a lógica de identificação do v3.2 é preservada: contrastes pareados por semente e os mesmos itens de teste em todas as condições, de modo que diferenças entre alvos de treinamento não se confundem com diferenças entre amostras de itens. As alterações de tarefa reduzem a dificuldade aritmética, de modo que os resultados futuros, se obtidos, valerão para problemas que o modelo resolve com etapas explícitas e com erros residuais, e não para os problemas mais difíceis do v3.2. A escolha da faixa C2 pelo critério de 0,80 com demonstrações mantém a dependência das etapas: sem demonstrações, o mesmo modelo não acertou nenhum item dos candidatos ao nível C.

A validade externa permanece limitada a um modelo pequeno, a tarefas aritméticas sintéticas e a três alvos de treinamento. Os resultados sobre monitores, por sua vez, referem-se a um único procedimento de classificação por comparação de probabilidades, com uma rubrica e um formato de entrada, e não autorizam conclusões sobre monitores generativos, modelos maiores ou rubricas diferentes.

# 7 LIMITAÇÕES

Os arquivos brutos da triagem original do v3.2 não estavam acessíveis. A correspondência entre os valores documentados e a reprodução no Colab é forte, porque os quatro valores coincidem exatamente com o mesmo código, os mesmos dados e as mesmas revisões, mas não substitui a conferência dos arquivos originais. O registro OSF do v3.2 não contém os resumos criptográficos dos artefatos congelados e cita a versão 3.1 do protocolo, o que impede verificar, apenas pelo registro, a correspondência exata com a implementação v3.2.

A calibração do v4 usou 40 itens por candidato e controles de monitor construídos por regra. Os primeiros fornecem estimativas com erro-padrão próximo de 0,06; os segundos não representam a variedade de rastros que modelos treinados produzirão e não substituem a validação com rótulos humanos exigida pelo protocolo. O Phi-3.5-mini foi pontuado numa GPU NVIDIA L4, e não na T4 usada pelos demais candidatos, porque não coube na memória desta última com pesos em fp32; a precisão numérica foi a mesma, mas diferenças de hardware podem alterar escores em casas decimais. As rodadas sucessivas de calibração foram todas registradas antes da execução e relatadas, inclusive as reprovadas, mas a escolha entre variantes de instrução, ainda que feita em dados próprios, introduz graus de liberdade que só a triagem em itens DEV separados e o registro posterior controlam.

A triagem do v4 foi executada antes do registro público e numa sessão do Colab conduzida a partir de uma interface de linha de comando, com pesos em fp16 na mesma GPU prevista para os treinos. Sua cronologia é verificável pelo histórico Git do repositório, e não por uma plataforma de registro. Nenhum treinamento confirmatório, avaliação TEST, validação humana do monitor ou anotação humana foi realizado. As verificações em GPU usaram o modo técnico, com uma semente, dois passos e quatro itens, e demonstram funcionamento do software, não desempenho dos modelos treinados.

Por fim, a revisão bibliográfica concentrou-se nas fontes diretamente relacionadas ao argumento e não constitui revisão sistemática. Não houve coleta com seres humanos.

# 8 CONCLUSÃO

<<CONCLUSAO>>


{{quebra}}

# REFERÊNCIAS

::: {custom-style="Referencia"}
ABDIN, Marah *et al*. Phi-3 technical report: a highly capable language model locally on your phone. **arXiv**, 2404.14219, 2024. DOI: 10.48550/arXiv.2404.14219.

AGARWAL, Rishabh; SCHWARZER, Max; CASTRO, Pablo Samuel; COURVILLE, Aaron; BELLEMARE, Marc G. Deep reinforcement learning at the edge of the statistical precipice. *In*: ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS, 34., 2021. **Proceedings** [...]. [*S. l.*]: Curran Associates, 2021. Disponível em: https://arxiv.org/abs/2108.13264. Acesso em: 28 out. 2026.

BAKER, Bowen; HUIZINGA, Joost; GAO, Leo; DOU, Zehao; GUAN, Melody Y.; MADRY, Aleksander; ZAREMBA, Wojciech; PACHOCKI, Jakub; FARHI, David. Monitoring reasoning models for misbehavior and the risks of promoting obfuscation. **arXiv**, 2503.11926, 2025. DOI: 10.48550/arXiv.2503.11926.

BEN ALLAL, Loubna *et al*. SmolLM2: when smol goes big — data-centric training of a small language model. **arXiv**, 2502.02737, 2025. DOI: 10.48550/arXiv.2502.02737.

DRORI, Jacob; MARKS, Luke; WOODWORTH, Bryce; CLOUD, Alex; TURNER, Alexander Matt. Output supervision can obfuscate the chain of thought. **arXiv**, 2511.11584, 2025. DOI: 10.48550/arXiv.2511.11584.

EMMONS, Scott; JENNER, Erik; ELSON, David K.; SAUROUS, Rif A.; RAJAMANOHARAN, Senthooran; CHEN, Heng; SHAFKAT, Irhum; SHAH, Rohin. When chain of thought is necessary, language models struggle to evade monitors. **arXiv**, 2507.05246, 2025. DOI: 10.48550/arXiv.2507.05246.

HOLM, Sture. A simple sequentially rejective multiple test procedure. **Scandinavian Journal of Statistics**, Oslo, v. 6, n. 2, p. 65-70, 1979. Disponível em: https://www.jstor.org/stable/4615733. Acesso em: 28 out. 2026.

HU, Edward J.; SHEN, Yelong; WALLIS, Phillip; ALLEN-ZHU, Zeyuan; LI, Yuanzhi; WANG, Shean; WANG, Lu; CHEN, Weizhu. LoRA: low-rank adaptation of large language models. *In*: INTERNATIONAL CONFERENCE ON LEARNING REPRESENTATIONS, 10., 2022. **Proceedings** [...]. [*S. l.*]: ICLR, 2022. DOI: 10.48550/arXiv.2106.09685.

HUGGING FACE. **SmolLM3-3B**. Hugging Face, 2025. Modelo, revisão a07cc9a04f16550a088caea529712d1d335b0ac1. Disponível em: https://huggingface.co/HuggingFaceTB/SmolLM3-3B. Acesso em: 29 out. 2026.

IBM GRANITE TEAM. **granite-3.3-2b-instruct**. Hugging Face, 2025. Modelo, revisão 707f574c62054322f6b5b04b6d075f0a8f05e0f0. Disponível em: https://huggingface.co/ibm-granite/granite-3.3-2b-instruct. Acesso em: 29 out. 2026.

KORBAK, Tomek *et al*. Chain of thought monitorability: a new and fragile opportunity for AI safety. **arXiv**, 2507.11473, 2025. DOI: 10.48550/arXiv.2507.11473.

LANHAM, Tamera *et al*. Measuring faithfulness in chain-of-thought reasoning. **arXiv**, 2307.13702, 2023. DOI: 10.48550/arXiv.2307.13702.

MARQUES, Kelvin e Silva *et al*. **Validade externa da ofuscação de cadeia de pensamento sob supervisão de saída**: um teste empírico em tarefas com raciocínio causalmente necessário. Fortaleza: Centro Universitário Ateneu, 2026. Manuscrito não publicado, versão de 27 jun. 2026.

MARQUES, Kelvin e Silva. **cot-obfuscation-h2a**: código, dados e protocolos do projeto. GitHub, 2026. Disponível em: https://github.com/Kelvin-Marques-Cyber/cot-obfuscation-h2a. Acesso em: 28 out. 2026.

MARQUES, Kelvin e Silva. **Monitorabilidade e ofuscação da cadeia de pensamento**: pré-registro de experimentos SFT e GRPO com avaliação independente. OSF Registries, 19 out. 2026. Disponível em: https://osf.io/sjwdf/. Acesso em: 28 out. 2026.

MITRANI HADIDA, Nathaniel; BHANJI, Sassan; TICE, Cameron; RADMARD, Puria. Chain-of-thought obfuscation learned from output supervision can generalise to unseen tasks. **arXiv**, 2601.23086v2, 2026. Versão de 20 maio 2026. DOI: 10.48550/arXiv.2601.23086.

NOSEK, Brian A.; EBERSOLE, Charles R.; DEHAVEN, Alexander C.; MELLOR, David T. The preregistration revolution. **Proceedings of the National Academy of Sciences**, Washington, v. 115, n. 11, p. 2600-2606, 2018. DOI: 10.1073/pnas.1708274114.

QWEN TEAM. Qwen2.5 technical report. **arXiv**, 2412.15115, 2024. DOI: 10.48550/arXiv.2412.15115.

SHAO, Zhihong; WANG, Peiyi; ZHU, Qihao; XU, Runxin; SONG, Junxiao; BI, Xiao; ZHANG, Haowei; ZHANG, Mingchuan; LI, Y. K.; WU, Y.; GUO, Daya. DeepSeekMath: pushing the limits of mathematical reasoning in open language models. **arXiv**, 2402.03300, 2024. DOI: 10.48550/arXiv.2402.03300.

TEAM OLMO; WALSH, Pete; SOLDAINI, Luca; GROENEVELD, Dirk *et al*. 2 OLMo 2 Furious. **arXiv**, 2501.00656, 2024. DOI: 10.48550/arXiv.2501.00656.

TURPIN, Miles; MICHAEL, Julian; PEREZ, Ethan; BOWMAN, Samuel R. Language models don't always say what they think: unfaithful explanations in chain-of-thought prompting. *In*: ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS, 36., 2023. **Proceedings** [...]. [*S. l.*]: Curran Associates, 2023. DOI: 10.48550/arXiv.2305.04388.

VON WERRA, Leandro *et al*. **TRL**: Transformer Reinforcement Learning. Versão 0.22.2. GitHub, 2020. Disponível em: https://github.com/huggingface/trl. Acesso em: 28 out. 2026.

WEI, Jason *et al*. Chain-of-thought prompting elicits reasoning in large language models. *In*: ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS, 35., 2022. **Proceedings** [...]. [*S. l.*]: Curran Associates, 2022. DOI: 10.48550/arXiv.2201.11903.

WOLF, Thomas *et al*. Transformers: state-of-the-art natural language processing. *In*: CONFERENCE ON EMPIRICAL METHODS IN NATURAL LANGUAGE PROCESSING: SYSTEM DEMONSTRATIONS, 2020. **Proceedings** [...]. [*S. l.*]: Association for Computational Linguistics, 2020. p. 38-45. DOI: 10.18653/v1/2020.emnlp-demos.6.

YANG, An *et al*. Qwen3 technical report. **arXiv**, 2505.09388, 2025. DOI: 10.48550/arXiv.2505.09388.

ZHENG, Lianmin *et al*. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *In*: ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS, 36., 2023. **Proceedings** [...]. [*S. l.*]: Curran Associates, 2023. DOI: 10.48550/arXiv.2306.05685.
:::

