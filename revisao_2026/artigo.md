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
Ofuscação da cadeia de pensamento sob supervisão
:::

::: {custom-style="Capa subtitulo"}
reanálise de um estudo-piloto e triagem de competência de um estudo prospectivo pré-registrado
:::

::: {custom-style="Capa local"}
FORTALEZA\
2026
:::

{{quebra}}

# RESUMO

A monitorabilidade da cadeia de pensamento depende de que os rastros textuais gerados por modelos de linguagem continuem a revelar comportamentos indesejados quando o treinamento sofre pressão de supervisão. Este trabalho examina três conjuntos de evidência de um mesmo projeto. A reanálise de um estudo-piloto com ajuste fino supervisionado (SFT) reproduziu os resultados publicados, mas mostrou que o corretor aceitava respostas ambíguas e que as sementes compartilhavam quase todos os itens, o que impede atribuir a diferença entre alvos de treinamento à ofuscação. O estudo pré-registrado v3.2, com braços de SFT e de otimização por recompensa (GRPO), foi encerrado sem treinamentos porque os modelos-base não atingiram 65% de acurácia na triagem de competência. A reprodução com o código congelado obteve exatamente os valores registrados e descartou defeitos de execução; as causas foram a incapacidade dos modelos nos níveis mais difíceis e instruções que tornavam a medida dependente de formatos não demonstrados. Um novo protocolo, v4, recalibrou tarefas e instruções em itens próprios, e o modelo do braço SFT acertou {{v:v4.gate.sft.A}} e {{v:v4.gate.sft.C}} dos itens dos dois níveis na triagem anterior ao registro. Nenhum dos nove modelos avaliados como monitores distinguiu rastros que usam uma informação proibida de rastros que não a usam, e o braço GRPO foi excluído por uma regra declarada antes desse resultado. Não há resultado confirmatório sobre ofuscação: o trabalho documenta um desenho viável para testar o efeito do formato de supervisão e os requisitos de instrumentação para testar a ofuscação sob pressão de recompensa.

**Palavras-chave:** Cadeia de pensamento. Monitorabilidade. Pré-registro. Triagem de competência. Reprodutibilidade.

{{quebra}}

# ABSTRACT

Chain-of-thought monitorability depends on the textual traces produced by language models continuing to reveal undesired behavior when training is subject to supervision pressure. This work examines three bodies of evidence from a single project. A reanalysis of a supervised fine-tuning (SFT) pilot study reproduced the published results but showed that the grader accepted ambiguous answers and that the seeds shared almost all items, which prevents attributing the difference between training targets to obfuscation. The preregistered study v3.2, with SFT and reward-optimization (GRPO) arms, was closed without training because the base models did not reach 65% accuracy in the competence screening. Reproducing it with the frozen code yielded exactly the registered values and ruled out execution defects; the causes were the models' inability at the hardest levels and instructions that made the measurement depend on undemonstrated formats. A new protocol, v4, recalibrated tasks and instructions on separate items, and the SFT base model answered {{v:v4.gate.sft.A.en}} and {{v:v4.gate.sft.C.en}} of the items at the two levels correctly in the screening run before registration. None of the nine models evaluated as monitors distinguished traces that use forbidden information from traces that do not, and the GRPO arm was excluded under a rule declared before this result. There is no confirmatory result about obfuscation: the work documents a viable design for testing the effect of supervision format and the instrumentation requirements for testing obfuscation under reward pressure.

**Keywords:** Chain of thought. Monitorability. Preregistration. Competence screening. Reproducibility.

{{quebra}}

{{sumario}}

{{quebra}}

# 1 INTRODUÇÃO

Modelos de linguagem que produzem rastros textuais antes da resposta final oferecem uma fonte adicional de evidência sobre seu comportamento. Neste texto, rastro designa o texto intermediário gerado antes da resposta, e monitor designa um procedimento, automático ou humano, que lê esse texto para detectar comportamentos indesejados. Korbak et al. (2025) descrevem essa monitorabilidade da cadeia de pensamento (*chain of thought*, CoT) como uma oportunidade relevante para a segurança, mas frágil, porque decisões de treinamento podem reduzi-la. A preocupação central é a ofuscação: a manutenção de um comportamento indesejado acompanhada da redução dos sinais textuais que permitiriam detectá-lo.

Este artigo examina três conjuntos de evidência produzidos no mesmo projeto. O primeiro é um estudo-piloto com ajuste fino supervisionado (*supervised fine-tuning*, SFT), cujos dados publicados foram reanalisados. O segundo é um estudo prospectivo, denominado v3.2, isto é, um estudo cujas hipóteses, procedimentos e critérios de análise foram fixados e registrados publicamente antes da coleta dos dados. Ele tinha dois braços, entendidos como conjuntos de condições que compartilham o mesmo procedimento de treinamento: SFT e otimização por recompensa com *Group Relative Policy Optimization* (GRPO). O terceiro é um novo protocolo, v4, preparado depois que o v3.2 não satisfez seu critério de competência.

Esse critério constituía uma triagem: antes dos 240 treinamentos, cada modelo-base, isto é, o modelo pré-treinado ainda não ajustado no estudo, deveria acertar ao menos 65% de 120 itens da partição de desenvolvimento (DEV) em cada nível de dificuldade. A partição DEV é reservada a verificações prévias e distingue-se da partição de teste (TEST), usada apenas no desfecho final. A triagem registrou 57,5% e 0% nos níveis do braço SFT e 43,33% e 15% nos do braço GRPO. Antes de atribuir esse resultado à incapacidade dos modelos, é necessário excluir defeitos que produziriam o mesmo padrão: erro do corretor (o programa que decide se uma resposta está certa), uso incorreto do modelo de conversa (o formato com que as mensagens são apresentadas ao modelo), truncamento, extração equivocada da resposta, precisão numérica ou incompatibilidade entre bibliotecas.

Três perguntas orientam o trabalho. P1: os dados publicados do piloto permitem atribuir a diferença de desempenho entre alvos de treinamento à ofuscação de um raciocínio necessário? P2: a reprovação da triagem do v3.2 decorre de defeito técnico ou da incapacidade dos modelos-base? P3: alterações calibradas com dados próprios, separados das partições do estudo, permitem satisfazer o mesmo critério de competência, e os monitores previstos distinguem rastros que usam a informação proibida de rastros que não a usam? Nenhum treinamento confirmatório foi realizado; as hipóteses registradas são descritas, mas não testadas.

# 2 FUNDAMENTAÇÃO

Fidelidade refere-se à correspondência entre o rastro e os fatores que efetivamente determinam a resposta; monitorabilidade, à possibilidade de detectar, a partir do rastro, comportamentos relevantes para a supervisão. Wei et al. (2022) mostraram que etapas intermediárias melhoram o desempenho em problemas de múltiplos passos. Turpin et al. (2023) demonstraram, porém, que explicações plausíveis podem omitir influências que alteram a resposta, e Lanham et al. (2023) mediram, por intervenções no rastro, quanto a resposta depende dele. Fidelidade é, portanto, uma propriedade a medir, e não a presumir pela fluência do texto. Para inferir ofuscação, é necessário observar ao mesmo tempo um comportamento definido independentemente do texto e a dificuldade de detectá-lo no rastro.

Baker et al. (2025) observaram que penalizar, no treinamento por reforço, os rastros que mencionavam a exploração de testes reduziu a detecção sem eliminar o comportamento. Drori et al. (2025) e Mitrani Hadida et al. (2026) relatam que a supervisão aplicada apenas à saída também altera os rastros e que o efeito pode generalizar para tarefas não vistas. Emmons et al. (2025), por outro lado, mostram que, quando o rastro é computacionalmente necessário, a evasão do monitor exige assistência ou otimização substanciais. O efeito da pressão de supervisão depende, assim, da tarefa, do mecanismo de treinamento e do monitor, o que torna essenciais o controle da competência do modelo e uma definição comportamental da exploração.

No plano metodológico, Agarwal et al. (2021) recomendam explicitar a unidade de repetição e a incerteza entre execuções; Zheng et al. (2023) documentam vieses de avaliadores baseados em modelos de linguagem, o que motiva monitores independentes do modelo avaliado e validação com rótulos humanos; e Nosek et al. (2018) descrevem o pré-registro como a separação entre decisões tomadas antes e depois da observação dos dados. O GRPO (SHAO et al., 2024) gera várias respostas por enunciado e calcula a vantagem de cada uma, isto é, quanto sua recompensa supera a média do grupo. Os dois braços usam adaptação de baixo posto (LoRA) (HU et al., 2022), que acrescenta às camadas do modelo matrizes de baixo posto treináveis e mantém fixos os pesos originais, sobre modelos Qwen2.5 (QWEN TEAM, 2024) e Qwen3 (YANG et al., 2025), com as bibliotecas Transformers (WOLF et al., 2020) e TRL (VON WERRA et al., 2020).

# 3 HIPÓTESES E CATEGORIAS DE EVIDÊNCIA

O registro do v3.2 especificou duas hipóteses, cujo texto o v4 mantém; no v4, apenas H1 permanece testável (seção 5.3). Cada treinamento é identificado por uma semente, o número que inicializa os geradores pseudoaleatórios. A hipótese H1, do braço SFT, prevê que treinar respostas diretas, em vez de rastros explícitos, custe mais acurácia no nível de maior complexidade (C) do que no de menor complexidade (A). Para cada semente *s*,

$$\Delta^{\text{SFT}}_s = \left[\text{Acc}_{s}(\text{direct}, C) - \text{Acc}_{s}(\text{explicit}, C)\right] - \left[\text{Acc}_{s}(\text{direct}, A) - \text{Acc}_{s}(\text{explicit}, A)\right],$$

com direção prevista negativa. O contraste mede o efeito do alvo de treinamento sobre a acurácia, e não intenção de ocultação.

No braço GRPO, cada questão de múltipla escolha traz uma nota auxiliar proibida, também chamada de dica, que aponta uma alternativa errada; a recompensa mal especificada, uma pontuação que intencionalmente paga por comportamento indesejado, premia essa alternativa. Há três condições: *none*, sem monitor na recompensa; *cot_monitor*, em que a recompensa é reduzida quando um monitor de recompensa detecta o uso da nota no rastro; e *summary_monitor*, em que a penalidade incide sobre um resumo final. Uma resposta é exploração quando escolhe a alternativa da nota e é não sinalizada quando um monitor independente, distinto do monitor de recompensa, não a detecta. A hipótese H2 prevê, no nível de dica mais difícil (*high*), contraste positivo

$$\Delta^{\text{GRPO}}_s = P_s^{\text{cot\_monitor}}(\text{explora e não sinalizado}) - P_s^{\text{none}}(\text{explora e não sinalizado}),$$

com a exigência adicional de que a taxa de exploração não caia mais que 0,10.

A unidade de inferência é a semente: as gerações de uma semente são agregadas numa proporção antes de qualquer teste. Os contrastes de 20 sementes seriam avaliados por um teste de inversão de sinais bilateral, que compara a média observada com a distribuição obtida ao trocar aleatoriamente o sinal de cada contraste (199.999 permutações com semente fixa), por um intervalo *bootstrap* percentil de 95%, obtido por reamostragem das sementes, e pela correção de Holm (1979) para múltiplas famílias de testes, com efeito mínimo relevante de 0,05.

Os resultados são separados em cinco categorias, que não se combinam: (i) resultados históricos do piloto; (ii) verificações de software, sem evidência sobre o fenômeno; (iii) triagens de competência; (iv) experimentos exploratórios, como o diagnóstico e a calibração; e (v) resultados confirmatórios, que exigiriam os treinamentos e avaliações completos de um protocolo registrado. Nenhum resultado da categoria (v) foi produzido.

# 4 MATERIAIS E MÉTODOS

## 4.1 Reanálise do estudo-piloto

O piloto (MARQUES et al., 2026) foi depositado no repositório público do projeto (MARQUES, 2026a) no commit e21cef8, de junho de 2026. Commit designa uma versão registrada no sistema de controle de versões Git, identificada por um código hexadecimal. O piloto ajustou o Qwen2.5-7B-Instruct, quantizado em 4 bits, com LoRA em dois níveis aritméticos (A: adição ou subtração de inteiros de 11 a 99; C: problemas de estoque com multiplicação), três alvos de SFT e três sementes (42 a 44), com 150 itens de treino e 30 de avaliação por execução. Os alvos recebem aqui nomes descritivos: CoT explícita (rastro com as etapas), resposta direta e preâmbulo genérico (frase vaga seguida da resposta). O arquivo publicado contém {{v:piloto.linhas}} avaliações, sem enunciados nem textos gerados; a reanálise reprocessou as métricas depositadas, reproduziu a análise de variância e o teste *t* de Welch originais e auditou o gerador de itens, o corretor, o teste de intervenção e o escore lexical executando o código preservado.

Para cada semente *s*, nível *l* e alvo *c*, o custo de acurácia em relação à CoT explícita é K(l, c, s) = Acc(l, explícita, s) − Acc(l, c, s), e a diferença entre níveis é D(c, s) = K(C, c, s) − K(A, c, s). Com três sementes, a enumeração das oito inversões de sinal possíveis fornece *p* bilateral mínimo de 0,25. O teste histórico de intervenção, que compara respostas com e sem explicação em dez itens, foi reconstruído dos agregados e avaliado pelo teste exato de McNemar.

## 4.2 Estudo prospectivo v3.2

O v3.2 foi implementado no commit fde856f e operado por cinco notebooks do Google Colab, documentos que combinam texto e código executável em máquinas virtuais com unidade de processamento gráfico (GPU). O desenho previa 240 treinamentos e 55.200 gerações de teste (@tab:desenho_v3). Cada treinamento produz um adaptador, o conjunto de parâmetros ajustados pela LoRA. No SFT, o nível A consistia em adições de inteiros de 101 a 9.999, e o nível C, em expressões a + b × c − d com os quatro operandos no mesmo intervalo. No GRPO, as questões apresentavam uma expressão aritmética em quatro domínios (aritmética, rastreamento de programa, regra fictícia e estoque); no nível *low*, a nota nomeava a letra, e no nível *high*, exigia calcular um índice módulo 4. As condições com monitor subtraíam 0,6 da recompensa quando o monitor Qwen2.5-0.5B-Instruct detectava o uso da nota.

{{tabela:desenho_v3}}

O registro (MARQUES, 2026b) foi localizado pela interface de programação do OSF: identificador sjwdf, criado em 19 de outubro de 2026 às 16h11 UTC e tornado público às 16h45 UTC, depois dos commits do código (15h20) e do suporte ao Colab (15h31). Ele especifica hipóteses, modelos, sementes, critério de competência e análise e determina novo protocolo para mudanças substanciais de modelo, instrumento ou amostragem. Não contém, porém, os resumos criptográficos dos artefatos congelados e cita a versão 3.1 do protocolo; as diferenças da v3.2 executada são operacionais, sem alteração de hipóteses, tamanhos ou critérios.

Na triagem, cada modelo-base respondia aos 120 primeiros itens DEV de cada nível numa GPU NVIDIA Tesla T4, com pesos em ponto flutuante de 16 bits (fp16), decodificação gulosa (escolha, a cada passo, do *token* mais provável) e orçamento de 256 *tokens*, isto é, até 256 unidades do vocabulário do modelo por resposta. No SFT, a instrução pedia uma única tag `<answer>NUMERO</answer>`, e o contrato estrito de correção aceitava somente essa tag, uma linha final "Resposta: N" ou uma saída composta apenas pelo número. No GRPO, o enunciado incluía a nota proibida, e a resposta só era correta com os blocos `<trace>`, `<summary>` e `<answer>` em sequência, sem texto entre eles, e a letra do gabarito no último.

## 4.3 Diagnóstico da triagem do v3.2

Os arquivos brutos da triagem original não estavam disponíveis; os valores originais constam do README do v3 (commit 273d131). Duas reproduções diagnósticas, que não substituem a triagem registrada, foram conduzidas. A primeira repetiu, em 28 de outubro de 2026, o caminho original no Colab: os notebooks baixaram os commits fixados, criaram o ambiente com as versões fixadas e executaram a triagem numa Tesla T4, em fp16, numa pasta separada. A segunda executou as mesmas funções em processador central (CPU), com pesos em ponto flutuante de 32 bits (fp32), para avaliar o efeito da precisão. As revisões dos modelos, identificadores da versão exata dos pesos, foram confirmadas no Hugging Face Hub, repositório público de modelos.

As respostas da reprodução na T4 foram classificadas em categorias descritivas, como resposta correta fora do contrato, separador de milhar lido como decimal e truncamento no orçamento. A soma das respostas corretas fora do contrato define um limite superior permissivo, que indica quanto um extrator tolerante alteraria o resultado; o desfecho registrado permanece o do corretor congelado. Verificaram-se também o texto enviado ao modelo de conversa, a tokenização, o identificador de término e as versões das bibliotecas.

## 4.4 Protocolo prospectivo v4

O v4 é um estudo novo, com identificador próprio (estudo-prospectivo-2026-v4), código no diretório experimento_cientifico_v4 e cinco notebooks próprios. Ele mantém perguntas, texto das hipóteses, modelos-base, condições, 20 sementes, passos, tamanhos das partições, critério de 65% e plano estatístico; as alterações (@tab:mudancas_v4) restringem-se às causas identificadas no diagnóstico.

{{tabela:mudancas_v4}}

Tarefas, instruções e monitores foram escolhidos por calibração em itens gerados com sementes próprias, excluídos por construção das partições do estudo. Cada regra de escolha foi gravada no repositório antes da execução correspondente, e todas as rodadas são relatadas, inclusive as reprovadas. Nas mesmas condições da triagem (T4, fp16, decodificação gulosa e 256 *tokens*), o nível A do SFT e cada candidato ao nível C deveriam atingir 0,80 em 40 itens com a instrução explícita; escolhe-se o candidato mais difícil aprovado. A rodada 2 acrescentou à instrução duas demonstrações fixas do formato. No GRPO, a faixa de operandos, agora comum aos dois níveis, deveria atingir 0,80 sem a dica em 40 itens por nível; a rodada 2 testou uma instrução sem os marcadores copiáveis (r2a) e a mesma instrução com uma demonstração sem dica (r2b). O limiar de 0,80 dá margem acima do critério de 0,65.

Os monitores automáticos são modelos de linguagem usados como classificadores: cada um recebe uma rubrica que pergunta se o texto usa a informação proibida, seguida da questão e do rastro, e compara as probabilidades que atribui às continuações "true" e "false". Foram avaliados em 320 controles construídos a partir de frases-modelo, 160 que usam explicitamente a nota e 160 que calculam a resposta, inclusive rejeitando a nota. Esses rótulos decorrem da construção, e não de anotação humana, e servem apenas à viabilidade; a validação com 100 controles positivos e 100 negativos rotulados por pessoas continua obrigatória. Para o monitor independente, escolhia-se o menor modelo com taxa de verdadeiros positivos (TPR) de pelo menos 0,90 e taxa de falsos positivos (FPR) de no máximo 0,05: na rodada 1, SmolLM2 (BEN ALLAL et al., 2025) de 135M, 360M e 1,7B de parâmetros; na rodada 2, OLMo 2 de 1B (TEAM OLMO et al., 2024), Granite 3.3 de 2B (IBM GRANITE TEAM, 2025), SmolLM3 de 3B (HUGGING FACE, 2025) e Phi-3.5-mini (ABDIN et al., 2024). O monitor de recompensa deveria atingir TPR − FPR ≥ 0,5. Antes de conhecer o resultado da rodada 2, gravou-se que, sem monitor viável, o braço GRPO seria excluído antes do registro; e, quando o Phi-3.5-mini se mostrou grande demais para a memória da T4 em fp32, declarou-se, antes de pontuá-lo, que seria avaliado numa GPU maior com os mesmos critérios.

A triagem do v4 precede o registro, para que este cubra apenas um desenho viável e inclua os resumos SHA256 das gerações da triagem. Um resumo SHA256 é uma sequência de 64 caracteres calculada do conteúdo de um arquivo, que muda com qualquer alteração dele. A triagem foi executada uma única vez, depois do congelamento do protocolo e dos módulos que determinam itens, instruções, geração e correção; no GRPO, usou o enunciado sem a nota e 30 itens DEV de cada domínio por nível.

## 4.5 Verificação de software

Testes unitários, isto é, verificações automáticas de funções isoladas, cobrem correção, geração de dados, congelamento, bloqueios e estatística. Uma integração em CPU com modelos mínimos de pesos aleatórios exercita as bibliotecas de treinamento nos dois braços, com interrupção e retomada. Os cinco notebooks são executados com o Colab simulado e falhas injetadas de montagem do Google Drive, o armazenamento persistente entre sessões, e de transferência de arquivos. Um teste técnico em GPU executa os cinco notebooks com os modelos reais em modo técnico (uma semente, dois passos por treino e quatro itens por avaliação, em pasta separada) e força interrupções de treino e de avaliação. Nenhum desses testes produz evidência sobre as hipóteses.

# 5 RESULTADOS

## 5.1 Reanálise do estudo-piloto

As {{v:piloto.celulas}} médias recalculadas coincidem com os agregados publicados, e as análises históricas foram reproduzidas: interação da análise de variância F(2, 12) = {{v:piloto.anova_f}}, p = {{v:piloto.anova_p}}, e teste de Welch no nível C, t = {{v:piloto.welch_t}}, p = {{v:piloto.welch_p}}. No nível C, a acurácia passa de {{v:piloto.acc.C.control}} com CoT explícita para {{v:piloto.acc.C.cot_pressure}} com resposta direta e {{v:piloto.acc.C.output_supervision}} com preâmbulo genérico (@fig:piloto). A diferença média entre os custos dos níveis foi {{v:piloto.dd.cot_pressure}} e {{v:piloto.dd.output_supervision}}, positiva nas três sementes (@tab:piloto_contrastes), com p = {{v:piloto.signflip.cot_pressure}} na enumeração das inversões de sinal, o menor valor possível com três sementes.

{{figura:piloto}}

{{tabela:piloto_contrastes}}

A auditoria restringe a interpretação em quatro pontos. Primeiro, o gerador cria cada item a partir da semente somada ao índice, de modo que sementes adjacentes reutilizam 149 de 150 enunciados de treino e 29 de 30 de avaliação; as três sementes não são amostras independentes de problemas. Segundo, o corretor histórico aceita uma resposta quando qualquer número do trecho final coincide com o gabarito: num corpus construído de {{v:piloto.casos}} respostas, das quais {{v:piloto.negativos}} deveriam ser rejeitadas, aceitou {{v:piloto.falsos_aceites}} ({{v:piloto.taxa_aceite_indevido}}), por exemplo "Resposta: 43; 42 seria incorreto." com gabarito 42, enquanto o contrato estrito dos estudos prospectivos rejeitou todas. Terceiro, a comparação histórica com e sem explicação mostra no nível C {{v:piloto.mcnemar_b}} itens acertados só com explicação e {{v:piloto.mcnemar_c}} só sem ela (McNemar, p = {{v:piloto.mcnemar_p}}), mas altera simultaneamente instrução e orçamento (200 contra 40 *tokens*) e não estabelece a necessidade causal do rastro. Quarto, o escore lexical de "ofuscação" atribui valor máximo a rastros vazios e correlaciona-se com repetição (r = {{v:piloto.corr_rep}}) e diversidade lexical (r = {{v:piloto.corr_ttr}}), componentes de sua fórmula, sem validação independente.

## 5.2 Triagem do v3.2 e diagnóstico

A reprodução pelo caminho original no Colab obteve {{v:v3.sft.A.t4_k}}, {{v:v3.sft.C.t4_k}}, {{v:v3.grpo.low.t4_k}} e {{v:v3.grpo.high.t4_k}} acertos, exatamente os valores documentados (@tab:triagem_v3), com dados de mesmo SHA256 e as mesmas revisões dos modelos. Na comparação com a CPU em fp32, a correção coincidiu em {{v:v3.sft.A.cpu_igual}} de 120 itens no nível A e em todos os do nível C do SFT, e em {{v:v3.grpo.low.cpu_igual}} e {{v:v3.grpo.high.cpu_igual}} de {{v:v3.grpo.high.cpu_n}} no GRPO: a precisão altera respostas isoladas, não o resultado.

{{tabela:triagem_v3}}

A classificação (@tab:falhas_v3) separa três origens de falha. A primeira é a incapacidade aritmética: no nível A do SFT, o modelo respondeu só com um número e errou {{v:v3.sft.A.cat.formato_valido_valor_incorreto}} somas de até quatro algarismos; no nível C, nenhuma das 120 respostas seria correta mesmo com extração permissiva, pois o modelo não calcula sem etapas produtos de até oito algarismos; no nível *high* do GRPO, com operandos de três algarismos, acertou {{v:v3.grpo.high.dom.arithmetic}} itens do domínio aritmético, e {{v:v3.grpo.high.t4_trunc}} respostas esgotaram o orçamento enquanto calculavam o índice da nota. A segunda é a instrução e o contrato: o modelo do SFT nunca usou a tag pedida, e {{v:v3.sft.A.cat.valor_correto_fora_do_contrato}} respostas do nível A traziam o valor correto em prosa; o do GRPO copiou os marcadores "explicação verificável" e "resumo" da instrução e escreveu o cálculo fora das tags, e {{v:v3.grpo.low.cat.alternativa_correta_fora_do_contrato}} respostas do nível *low* terminaram com a letra correta fora do formato. Por isso, o domínio aritmético obteve {{v:v3.grpo.low.dom.arithmetic}} no nível *low*, e o de rastreamento de programa, {{v:v3.grpo.low.dom.program_trace}}; os 120 primeiros itens DEV cobriam apenas esses dois domínios. A terceira é um defeito do corretor, que lia um número com separador de milhar ("11,863") como decimal; afetou {{v:v3.sft.A.cat.separador_de_milhar_lido_como_decimal}} resposta e não altera nenhum nível.

{{tabela:falhas_v3}}

Os limites permissivos foram {{v:v3.sft.A.perm}}, {{v:v3.sft.C.perm}}, {{v:v3.grpo.low.perm}} e {{v:v3.grpo.high.perm}}: um extrator tolerante aprovaria alguns níveis, mas não o C do SFT. O modelo de conversa foi renderizado corretamente, no formato ChatML, que delimita as falas com marcadores especiais, e com o Qwen3 no modo sem pensamento, em que o bloco reservado ao raciocínio interno é enviado vazio; a tokenização não duplicou marcadores, e as bibliotecas fixadas reproduziram os resultados no Colab de 28 de outubro, com Python 3.13.16 e PyTorch 2.11. Os controles construídos da calibração do v4 mostraram ainda que o monitor independente do v3.2, SmolLM2-135M-Instruct, sinalizou {{v:mon.135m.tpr}} dos positivos e {{v:mon.135m.fpr}} dos negativos, e que o monitor de recompensa obteve TPR de {{v:mon.reward.tpr}} e FPR de {{v:mon.reward.fpr}}.

## 5.3 Calibração do v4

No SFT, a instrução explícita sem demonstrações não produziu respostas aceitas ({{v:cal.sft.r1.A.explicit.k}} no nível A): o modelo escrevia as contas e omitia a tag. Com a instrução neutra, o nível A atingiu {{v:cal.sft.r1.A.neutral.k}}, mas os candidatos C1 e C2 ficaram em {{v:cal.sft.r1.C1.neutral.k}} e {{v:cal.sft.r1.C2.neutral.k}}: sem etapas escritas, o modelo não resolve expressões com multiplicação. Com as demonstrações, os acertos foram {{v:cal.sft.r2.A.explicit.k}} no nível A, {{v:cal.sft.r2.C1.explicit.k}} em C1 e {{v:cal.sft.r2.C2.explicit.k}} em C2, e escolheu-se C2 (@tab:calibracao_v4). No GRPO, a instrução do v3.2 ficou abaixo de 0,80 nas duas faixas, com validade de formato acompanhando a acurácia. A variante r2a não produziu respostas aceitas, pois o modelo separou os blocos por quebras de linha em {{v:cal.r2a.quebra}} das {{v:cal.r2a.n}} respostas, embora {{v:cal.r2a.letra}} terminassem com a letra correta; a variante r2b atingiu {{v:cal.grpo.r2.r2b.G1.sem_dica.low.k}} e {{v:cal.grpo.r2.r2b.G1.sem_dica.high.k}} sem a dica, com formato válido em todas, e foi fixada com operandos de 11 a 99.

{{tabela:calibracao_v4}}

Nenhum dos sete candidatos a monitor independente atingiu o critério (@tab:monitores_v4). Os SmolLM2 de 360M e 1,7B sinalizaram praticamente todos os rastros, e o OLMo 2 e o SmolLM3 nenhum, padrões de um classificador que atribui sempre a mesma classe. O Granite 3.3 obteve TPR de {{v:mon.granite2b.tpr}} e FPR de {{v:mon.granite2b.fpr}}. O Phi-3.5-mini, avaliado numa GPU NVIDIA L4 com 24 GB, obteve TPR de {{v:mon.phi35.tpr}} e FPR de {{v:mon.phi35.fpr}}, com falsos positivos em {{v:mon.phi35.neg0}} dos rastros que apenas calculam a resposta e em {{v:mon.phi35.neg1}} dos que rejeitam a nota. Como monitor de recompensa, o Qwen2.5-0.5B-Instruct discriminou pior que um sorteio, e o Qwen2.5-1.5B-Instruct não sinalizou nenhum controle. Pela regra declarada, o braço GRPO foi excluído antes do registro, e o protocolo foi congelado com essa decisão no commit 37094c4.

{{tabela:monitores_v4}}

## 5.4 Triagem DEV do v4

Executada uma única vez, em 29 de outubro de 2026, numa T4 em fp16, a triagem resultou em {{v:v4.gate.sft.A.k}} acertos no nível A do SFT ({{v:v4.gate.sft.A}}) e {{v:v4.gate.sft.C.k}} no nível C ({{v:v4.gate.sft.C}}), com todas as respostas no formato exigido e nenhuma truncada (@tab:triagem_v4). Os dois níveis satisfizeram o critério de 0,65, e os treinamentos do SFT passaram a depender apenas do registro público. A margem no nível C é pequena: o intervalo de confiança de 95% de Wilson vai de {{v:v4.gate.sft.C.ic}} e a acurácia ficou abaixo da estimativa da calibração ({{v:cal.sft.r2.C2.explicit}} em 40 itens). O critério aplica-se à acurácia observada, de modo que o resultado é uma aprovação, mas o nível C situa-se perto do limite de competência do modelo-base. No GRPO, apenas descritivo, o modelo acertou {{v:v4.gate.grpo.low.k}} e {{v:v4.gate.grpo.high.k}} itens sem a nota.

{{tabela:triagem_v4}}

## 5.5 Verificações de software

As suítes de testes locais, a reanálise histórica e a execução dos cinco notebooks com o Colab simulado foram aprovadas. Na primeira execução do teste técnico em GPU, os cinco notebooks rodaram sem erro, mas a interrupção forçada de um treino logo após a gravação do último checkpoint revelou um defeito: na retomada, a biblioteca executou um passo além do previsto, e o software recusou o resultado. O defeito foi reproduzido em CPU, corrigido e coberto por um teste de regressão; na repetição, o treino interrompido retomou do primeiro checkpoint e terminou no passo previsto, e a avaliação interrompida, com uma linha parcial simulada, foi concluída sem itens duplicados. Comandos, versões, durações e resumos dos arquivos constam do relatório de validação do repositório.

# 6 DISCUSSÃO

O piloto registra uma diferença grande e consistente entre alvos de treinamento no nível de múltiplos passos, e os números publicados são reproduzíveis. Essa descrição motiva a hipótese de que substituir as etapas do alvo supervisionado prejudica a aprendizagem, mas não identifica uma política de evasão: o desenho altera o texto a imitar, não otimiza o modelo contra um monitor e não mede um comportamento proibido definido fora do rastro. Com um corretor que aceita respostas ambíguas e sementes que compartilham os itens, a resposta a P1 é negativa.

A reprovação do v3.2 não decorre de erro de execução. A resposta a P2 é que duas causas se somam: a incapacidade real dos modelos-base nas tarefas mais difíceis, suficiente para reprovar o nível C, e uma medida de competência que dependia da obediência a formatos não demonstrados. Um extrator permissivo aprovaria alguns níveis, mas trocá-lo depois de observar os resultados alteraria o instrumento registrado; o desfecho do v3.2 é a inviabilidade do desenho registrado, e não um resultado sobre H1 e H2.

A calibração confirma esse diagnóstico. A perda sob o contrato estrito decorria sobretudo do formato: uma instrução reescrita sem exemplos não bastou, ao passo que demonstrações tornaram as respostas válidas. O v4 mantém o contrato estrito e pré-especificado e altera apenas a instrução, de modo que a medida continua a exigir o formato do treinamento, sem depender de que o modelo o adivinhe. No SFT, o nível C2 continua dependente das etapas escritas, condição para que o contraste de H1 seja informativo. A resposta a P3 é parcial: o braço SFT tornou-se viável sob o mesmo critério, e o v4 está pronto para registro como estudo de uma hipótese confirmatória.

A avaliação dos monitores tem a maior consequência para o desenho. Sete modelos de quatro famílias, com 135 milhões a 3,8 bilhões de parâmetros, não distinguiram, com a margem exigida, rastros que seguem a nota de rastros que a ignoram ou rejeitam. Como esses controles são inequívocos por construção, é improvável que os mesmos monitores fossem mais precisos nos rastros, mais variados, de modelos treinados; mesmo com competência aprovada, a validação do monitor do v3.2 provavelmente teria falhado depois dos 240 treinamentos. A exclusão do braço GRPO não é evidência sobre H2. Ela evita gastar 120 treinamentos num desenho cujo desfecho seria, por construção, evidência insuficiente. Testar H2 exigirá validar antes um monitor de outro tipo, como um modelo maior usado como classificador generativo, um classificador ajustado com rótulos humanos ou anotação humana integral.

Quanto à validade, o braço SFT preserva a identificação do v3.2, com contrastes pareados por semente e os mesmos itens de teste em todas as condições. As alterações reduzem a dificuldade aritmética, e resultados futuros valerão para problemas que o modelo resolve com etapas explícitas e erros residuais. A validade externa limita-se a um modelo pequeno, a tarefas aritméticas sintéticas e a três alvos de treinamento; os resultados sobre monitores referem-se a um único procedimento de classificação por probabilidades, com uma rubrica, e não se estendem a monitores generativos, modelos maiores ou outras rubricas.

# 7 LIMITAÇÕES

Os arquivos brutos da triagem original do v3.2 não estavam acessíveis; a coincidência exata dos quatro valores reproduzidos é forte, mas não substitui sua conferência. O registro OSF do v3.2 não contém os resumos dos artefatos e cita a versão 3.1, o que impede verificar, apenas pelo registro, a correspondência com a implementação. A calibração do v4 usou 40 itens por candidato, com erro-padrão próximo de 0,06, e controles de monitor construídos por regra, que não representam a variedade de rastros de modelos treinados nem substituem rótulos humanos. As rodadas sucessivas foram declaradas antes de cada execução e relatadas, mas a escolha entre variantes introduz graus de liberdade que só a triagem em itens DEV separados controla. O Phi-3.5-mini foi pontuado numa L4, e não na T4, com a mesma precisão numérica. As sessões do Colab usadas na calibração foram encerradas pelo serviço duas vezes; as etapas afetadas foram repetidas com o mesmo código, com valores idênticos aos já observados. A triagem do v4 precedeu o registro, e sua cronologia é verificável pelo histórico Git, e não por uma plataforma de registro. Nenhum treinamento confirmatório, avaliação de teste, validação humana de monitor ou anotação humana foi realizado; os testes em GPU usaram o modo técnico e demonstram o funcionamento do software, não o desempenho de modelos treinados.

# 8 CONCLUSÃO

O estudo-piloto registra uma diferença consistente de acurácia entre alvos de treinamento, mas seus instrumentos não permitem atribuí-la à ofuscação de um raciocínio necessário. O estudo prospectivo v3.2 terminou sem treinamentos porque os modelos-base não atingiram o critério de competência; a reprodução com o código congelado confirmou os valores registrados, descartou defeitos de execução e identificou como causas a incapacidade dos modelos nas tarefas mais difíceis e instruções que tornavam a medida dependente de formatos não demonstrados.

O protocolo v4, preparado como estudo novo, tratou essas causas por calibração em itens próprios e satisfez o critério de competência no braço SFT, com pequena margem no nível de maior complexidade. Nenhum monitor avaliado distinguiu rastros que usam a informação proibida de rastros que não a usam, e o braço GRPO foi excluído antes do registro. Nenhuma conclusão confirmatória sobre ofuscação da cadeia de pensamento decorre deste trabalho. O passo seguinte é o registro público do v4, com os resumos SHA256 dos artefatos congelados, seguido dos 120 treinamentos do braço SFT; um teste de H2 exigirá, antes de qualquer treinamento, um monitor validado com rótulos humanos.

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

MARQUES, Kelvin e Silva. **cot-obfuscation-h2a**: código, dados e protocolos do projeto. GitHub, 2026a. Disponível em: https://github.com/Kelvin-Marques-Cyber/cot-obfuscation-h2a. Acesso em: 29 out. 2026.

MARQUES, Kelvin e Silva. **Monitorabilidade e ofuscação da cadeia de pensamento**: pré-registro de experimentos SFT e GRPO com avaliação independente. OSF Registries, 19 out. 2026b. Disponível em: https://osf.io/sjwdf/. Acesso em: 28 out. 2026.

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

