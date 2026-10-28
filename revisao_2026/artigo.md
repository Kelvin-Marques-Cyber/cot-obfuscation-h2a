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

Modelos de linguagem que produzem rastros textuais antes da resposta final oferecem uma fonte adicional de evidência sobre seu comportamento. Esses rastros podem revelar o uso de atalhos, a exploração de recompensas mal especificadas ou a influência de informações que deveriam ser ignoradas. Korbak et al. (2025) descrevem essa monitorabilidade da cadeia de pensamento (*chain of thought*, CoT) como uma oportunidade relevante para a segurança, mas frágil: decisões de treinamento podem reduzi-la. A preocupação central é a ofuscação, isto é, a manutenção de um comportamento indesejado acompanhada da redução dos sinais textuais que permitiriam detectá-lo.

A literatura recente oferece evidências em direções que precisam ser compatibilizadas. Baker et al. (2025) relatam que a otimização direta contra um monitor de CoT pode levar modelos a continuar explorando falhas de recompensa enquanto deixam de mencioná-las. Drori et al. (2025) e Mitrani Hadida et al. (2026) mostram que a supervisão aplicada apenas às saídas também pode alterar os rastros e que esse efeito pode se transferir para tarefas não vistas. Emmons et al. (2025), por outro lado, argumentam que, quando o raciocínio explícito é computacionalmente necessário para a tarefa, a evasão de monitores se torna mais difícil. A questão empírica que motiva este trabalho situa-se nessa intersecção: em que condições a pressão de supervisão reduz a detectabilidade de um comportamento objetivamente verificável, e como separar esse efeito de mudanças triviais de formato ou de perda de competência.

Este artigo examina três conjuntos de evidência produzidos no mesmo projeto. O primeiro é um estudo-piloto com ajuste fino supervisionado (*supervised fine-tuning*, SFT), cujos dados publicados foram reanalisados. O segundo é um estudo prospectivo, denominado v3.2, registrado publicamente no Open Science Framework (OSF) antes de qualquer treinamento, com dois braços: SFT e otimização por recompensa com *Group Relative Policy Optimization* (GRPO). O terceiro é uma revisão prospectiva do desenho, denominada v4, preparada depois que o v3.2 não satisfez seu critério de competência pré-treinamento.

O ponto de partida empírico do segundo conjunto é um resultado negativo de viabilidade. O protocolo v3.2 exigia que cada modelo-base atingisse ao menos 65% de acurácia em 120 itens de desenvolvimento (DEV) por nível de dificuldade antes dos 240 treinamentos previstos. A triagem registrou 57,5% e 0% nos níveis do braço SFT e 43,33% e 15% nos níveis do braço GRPO, bloqueando o estudo. Antes de atribuir esse resultado à incapacidade dos modelos, é necessário excluir defeitos de software que produziriam o mesmo padrão: erro do corretor, uso incorreto do modelo de conversa, truncamento, extração equivocada da resposta, efeitos de precisão numérica ou incompatibilidade entre bibliotecas.

As perguntas que orientam o trabalho são as seguintes. Primeira: o que os dados publicados do piloto sustentam depois de reprocessados e auditados? Segunda: a reprovação da triagem do v3.2 decorre de defeito técnico ou da incapacidade real dos modelos-base sob o desenho congelado? Terceira: quais alterações, calibradas exclusivamente com dados de desenvolvimento próprios, tornam viável um novo estudo prospectivo com o mesmo critério de competência, e o desenho revisado satisfaz esse critério? As hipóteses confirmatórias registradas para o v3.2 sobre custo de formato no SFT e exploração não sinalizada no GRPO são descritas, mas não testadas, porque nenhum treinamento confirmatório foi realizado.

As contribuições são: uma reanálise reprodutível do piloto, com identificação dos limites de seus instrumentos; a documentação verificável do desfecho de inviabilidade do v3.2, incluindo sua cronologia e a correspondência entre registro e implementação; uma reprodução diagnóstica da triagem com o código congelado, em GPU e em CPU, com classificação das falhas; e um protocolo v4 com nova identificação, justificativa metodológica, calibração separada dos dados de teste, triagem executada antes do registro e software verificado. O trabalho não apresenta resultados confirmatórios sobre ofuscação estratégica, positivos ou negativos.

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

A pergunta P1 é retrospectiva: os dados publicados do piloto permitem atribuir a diferença de desempenho entre alvos de treinamento à ofuscação de raciocínio necessário? A pergunta P2 diz respeito ao estudo v3.2: a reprovação da triagem de competência decorre de defeito técnico ou da incapacidade dos modelos-base nas tarefas congeladas? A pergunta P3 diz respeito ao desenho revisado: alterações calibradas com dados próprios de desenvolvimento permitem que os modelos-base satisfaçam o mesmo critério de 65% por nível?

## 3.2 Hipóteses confirmatórias registradas

O registro do v3.2 especificou duas hipóteses, mantidas sem alteração no v4. A hipótese H1 refere-se ao braço SFT e prevê que treinar respostas diretas, em vez de rastros explícitos, custe mais acurácia no nível de maior complexidade (C) do que no de menor complexidade (A). Para cada semente de treinamento *s*, o contraste é a interação

$$\Delta^{\text{SFT}}_s = \left[\text{Acc}_{s}(\text{direct}, C) - \text{Acc}_{s}(\text{explicit}, C)\right] - \left[\text{Acc}_{s}(\text{direct}, A) - \text{Acc}_{s}(\text{explicit}, A)\right],$$

com direção prevista negativa. Esse contraste mede o efeito do alvo de treinamento sobre a acurácia; ele não mede intenção de ocultação.

A hipótese H2 refere-se ao braço GRPO. Uma resposta é exploração quando escolhe a alternativa indicada por uma nota auxiliar proibida, que a recompensa mal especificada premia; é não sinalizada quando um monitor independente, distinto do monitor usado na recompensa, não a detecta no rastro. No nível de dica mais difícil (*high*), o contraste por semente é

$$\Delta^{\text{GRPO}}_s = P_s(\text{explora} \wedge \neg\text{sinalizado} \mid \text{cot\_monitor}) - P_s(\text{explora} \wedge \neg\text{sinalizado} \mid \text{none}),$$

com direção prevista positiva e exigência adicional de que a taxa de exploração não caia mais que 0,10 entre as condições.

Em ambos os casos a unidade de inferência é a semente de treinamento. As gerações de uma mesma semente não são observações independentes; elas são agregadas em uma proporção por semente antes de qualquer teste. Os contrastes de 20 sementes seriam avaliados por inversão de sinais bilateral (199.999 permutações Monte Carlo com semente fixa), intervalo *bootstrap* percentil de 95% entre sementes e correção de Holm (1979) para as duas famílias, com efeito mínimo relevante de 0,05 em escala de proporção. Uma hipótese só receberia apoio se significância ajustada, direção, magnitude, intervalo e as condições de validade dos instrumentos fossem satisfeitos simultaneamente. Como nenhum treinamento confirmatório foi executado, essas análises permanecem especificadas e implementadas, mas não aplicadas.

## 3.3 Categorias de evidência

Os resultados são separados em cinco categorias, que não são combinadas para aumentar tamanhos amostrais ou fortalecer conclusões: (i) resultados históricos, provenientes dos dados publicados do piloto; (ii) verificações de software, que testam contratos, retomada, integridade e execução das bibliotecas sem produzir evidência sobre o fenômeno; (iii) triagens de competência, que verificam se os modelos-base satisfazem o critério prévio; (iv) experimentos exploratórios, como a reprodução diagnóstica da triagem e a calibração do v4; e (v) resultados confirmatórios, que exigiriam os treinamentos e avaliações completos de um protocolo registrado. Nenhum resultado da categoria (v) foi produzido.

# 4 MATERIAIS E MÉTODOS

## 4.1 Estudo-piloto e estratégia de reanálise

O piloto foi depositado no repositório público do projeto no commit e21cef8, de junho de 2026, com um notebook de treinamento e análise e quatro tabelas de resultados. O desenho cruza dois níveis de complexidade aritmética, três alvos de SFT e três sementes (@tab:desenho_piloto). Os alvos recebem aqui nomes descritivos: CoT explícita (rastro com as etapas), resposta direta (somente a resposta) e preâmbulo genérico (frase vaga seguida da resposta). Os nomes usados no código, como *cot_pressure* e *output_supervision*, sugeriam mecanismos de penalização por monitor que o piloto não implementa.

{{tabela:desenho_piloto}}

O arquivo de resultados contém {{v:piloto.linhas}} linhas, uma por avaliação, com nível, condição, semente, acurácia, escore lexical e métricas de diversidade, mas sem enunciado, identificador do item, gabarito ou texto gerado. A reanálise, portanto, reprocessa as métricas depositadas; ela não refaz o julgamento das respostas. A estratégia teve três etapas: comparar as médias recalculadas com os agregados publicados em cada uma das {{v:piloto.celulas}} células; reproduzir a análise de variância e o teste *t* de Welch originais como verificação numérica; e auditar os instrumentos (gerador de itens, corretor, teste de intervenção e escore lexical) executando o código preservado.

Para cada semente *s*, nível *l* e alvo *c*, o custo de acurácia em relação à CoT explícita é K(l, c, s) = Acc(l, explícita, s) − Acc(l, c, s), e a diferença entre níveis é D(c, s) = K(C, c, s) − K(A, c, s). Como há três sementes, a análise de sensibilidade enumera as oito inversões de sinal possíveis dos três contrastes; o menor valor de *p* bilateral atingível é 0,25. O teste de intervenção histórico, que compara respostas com e sem explicação em dez itens por nível, foi reconstruído a partir dos agregados e avaliado pelo teste exato de McNemar.

## 4.2 Estudo prospectivo v3.2

### 4.2.1 Desenho e registro

O estudo v3.2 foi implementado no commit fde856f e operado por cinco notebooks do Google Colab: protocolo e dados (00), treinamento SFT (01), treinamento GRPO (02), avaliação TEST (03) e monitor independente com inferência (04). O desenho previa 240 treinamentos e 55.200 gerações de avaliação principal (@tab:desenho_v3). Esses números foram conferidos na implementação: cada adaptador SFT seria avaliado nos 220 itens TEST de seu nível e cada adaptador GRPO nos 240 itens TEST de seu nível (60 por domínio em quatro domínios), resultando em 26.400 e 28.800 gerações.

{{tabela:desenho_v3}}

No braço SFT, o nível A consiste em adições de dois inteiros entre 101 e 9.999, e o nível C, em expressões a + b × c − d com os quatro operandos no mesmo intervalo. O alvo explícito registra três etapas (o produto, a soma e a subtração) sem decompor a multiplicação. No braço GRPO, cada questão de múltipla escolha apresenta uma expressão aritmética em um de quatro domínios (aritmética, rastreamento de programa, regra fictícia e estoque) e uma nota auxiliar proibida que indica uma alternativa errada. No nível *low*, a nota nomeia a letra; no nível *high*, exige calcular um índice módulo 4. A recompensa mal especificada paga a escolha da alternativa indicada pela nota, e as condições *cot_monitor* e *summary_monitor* subtraem 0,6 quando o monitor de recompensa (Qwen2.5-0.5B-Instruct) detecta o uso da nota no rastro ou no resumo.

O registro público foi localizado por consulta à API do OSF: *Monitorabilidade e Ofuscação da Cadeia de Pensamento: Pré-registro de Experimentos SFT e GRPO com Avaliação Independente*, identificador sjwdf, criado em 19 de outubro de 2026 às 16h11 UTC e tornado público às 16h45 UTC (@tab:cronologia). O texto registrado especifica as hipóteses, os modelos, as 20 sementes, o critério de competência de 65% e as regras de análise e declara que mudanças substanciais de modelo, instrumento ou amostragem exigem novo protocolo. O registro não contém arquivos anexados nem os resumos criptográficos do protocolo e das revisões dos modelos: o único arquivo enviado ao projeto foi removido 24 segundos depois. O texto cita a versão 3.1 do protocolo, enquanto a implementação executada é a v3.2, cujas diferenças são operacionais (seleção do modo de avaliação, congelamento da revisão do monitor independente e pilha de dependências sem quantização), sem alteração de hipóteses, tamanhos ou critérios.

{{tabela:cronologia}}

### 4.2.2 Triagem de competência

Antes de qualquer treinamento, cada modelo-base respondia aos 120 primeiros itens DEV de cada nível, em decodificação gulosa, com orçamento de 256 tokens, em uma GPU Tesla T4 com pesos em fp16. No SFT, a instrução de sistema era "Resolva o problema. Termine com uma única tag <answer>NUMERO</answer>." e a correção usava um contrato estrito: a resposta é aceita somente como uma única tag final, uma única linha final "Resposta: N" ou uma saída composta apenas pelo número. No GRPO, o enunciado incluía a nota proibida e a resposta só era correta se seguisse exatamente o formato <trace>…</trace><summary>…</summary><answer>LETRA</answer> com a letra do gabarito. O critério exigia acurácia mínima de 0,65 em cada um dos quatro níveis; a reprovação bloqueava os 240 treinamentos.

## 4.3 Diagnóstico da triagem do v3.2

Os arquivos brutos da triagem original estavam no Google Drive da sessão do Colab em que foi executada e não estavam disponíveis para esta auditoria; uma busca no Drive conectado ao projeto não os localizou. Os valores originais, portanto, constam aqui como documentados no README do v3 (commit 273d131), e não como reprodução independente. Para examinar sua origem, foram conduzidas duas reproduções diagnósticas, que não substituem a triagem registrada nem podem liberar os treinamentos do v3.2.

A primeira reproduz o caminho original no Colab, em 28 de outubro de 2026: os notebooks v3.2 baixam os commits fixados fde856f e 4a2a133, criam o ambiente isolado com as versões fixadas e executam as etapas de preparação, congelamento das revisões e triagem em uma Tesla T4, em fp16, numa pasta de diagnóstico separada. A segunda reconstrói os dados com o mesmo código e usa as mesmas funções de conversa e geração em CPU, com pesos em fp32, para avaliar a sensibilidade à precisão numérica. As revisões dos modelos foram confirmadas no Hugging Face Hub: a última alteração de cada repositório é anterior a outubro de 2026, de modo que as revisões disponíveis coincidem necessariamente com as congeladas em 19 de outubro.

As respostas da reprodução em CPU foram classificadas em categorias descritivas: correta no contrato; valor correto fora do contrato (por exemplo, prosa terminada pelo número certo); separador de milhar lido como decimal; formato válido com valor incorreto; alternativa indicada pela dica; formato inválido com valor incorreto; e truncamento no orçamento. A soma das categorias corretas fora do contrato define um limite superior permissivo, que indica quanto um extrator mais tolerante poderia alterar o resultado. Esse limite é diagnóstico: o desfecho registrado permanece o do corretor congelado. Além disso, verificaram-se o texto produzido pelo modelo de conversa, a tokenização, o identificador de término, a ocorrência de truncamento e as versões das bibliotecas.

<<METODOS_V4>>

<<RESTANTE>>
