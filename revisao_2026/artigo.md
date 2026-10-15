# RESUMO

A supervisão de modelos de linguagem pode aproveitar seus rastros textuais de raciocínio, mas a redução de sinais observáveis nesses rastros não identifica, isoladamente, uma estratégia de ocultação. Este artigo reavalia um estudo-piloto sobre o custo de desempenho associado a diferentes alvos de ajuste supervisionado e valida seus instrumentos computacionais. Foram reanalisadas 540 linhas de métricas publicadas, organizadas em dois níveis de complexidade aritmética, três condições e três sementes. A auditoria reproduziu integralmente as médias de acurácia e a interação da análise de variância originalmente reportada, F(2,12) = 527,17, p = 2,03 × 10^-12. As diferenças de custos entre os níveis foram 0,589 para respostas diretas e 0,656 para preâmbulos genéricos. Entretanto, a reconstrução do gerador revelou 29 itens compartilhados entre conjuntos de avaliação de 30 itens em sementes adjacentes, além de uma coincidência entre treino e avaliação no nível simples. Um teste mecânico com 2.000 respostas construídas identificou 1.250 aceitações indevidas pelo corretor original; um corretor com contrato explícito eliminou essas aceitações nesse corpus. A tabela histórica de intervenção, reconstruída a partir dos agregados, não estabelece necessidade causal do raciocínio explícito no nível complexo: o teste exato pareado resultou em p = 0,453. Conclui-se que há uma associação descritiva entre formato de treinamento e desempenho no piloto, mas os dados disponíveis não demonstram ofuscação estratégica, emergência sob supervisão exclusiva de saída ou generalização entre domínios. A contribuição final é uma reanálise verificável, acompanhada de instrumentos corrigidos, dados de teste disjuntos e um protocolo prospectivo para investigar essas questões.

**Palavras-chave:** Cadeia de pensamento. Monitorabilidade. Reprodutibilidade. Validade externa. Avaliação de modelos de linguagem.

[[PAGEBREAK]]

# ABSTRACT

Language-model oversight can benefit from textual reasoning traces, but reduced observable evidence in those traces does not, by itself, identify a concealment strategy. This article reassesses a pilot study of the performance costs associated with different supervised fine-tuning targets and validates its computational instruments. We reanalyzed 540 published metric rows covering two arithmetic-complexity levels, three conditions, and three seeds. The audit fully reproduced the reported mean accuracies and the original analysis-of-variance interaction, F(2,12) = 527.17, p = 2.03 × 10^-12. Between-level differences in accuracy costs were 0.589 for direct answers and 0.656 for generic preambles. However, reconstruction of the data generator revealed 29 shared items between adjacent-seed evaluation sets containing 30 items each, as well as one training-evaluation collision in the simpler level. A mechanical test of 2,000 constructed responses identified 1,250 false acceptances by the original grader; an explicit-contract grader eliminated those acceptances on this corpus. The historical intervention table, reconstructed from aggregate counts, does not establish causal necessity of explicit reasoning in the more complex level: an exact paired test yielded p = .453. The available data support a descriptive association between training format and pilot performance, but do not demonstrate strategic obfuscation, its emergence under output-only supervision, or cross-domain generalization. The final contribution is a verifiable reanalysis with corrected instruments, disjoint test data, and a prospective protocol for investigating these questions.

**Keywords:** Chain of thought. Monitorability. Reproducibility. External validity. Language model evaluation.

[[PAGEBREAK]]

[[TOC]]

[[PAGEBREAK]]

# 1 INTRODUÇÃO

A possibilidade de examinar rastros textuais produzidos antes de uma resposta final oferece uma fonte adicional de evidência sobre o comportamento de modelos de linguagem. Essa possibilidade interessa à segurança de inteligência artificial porque pode revelar o uso de atalhos, a exploração de recompensas mal especificadas ou a influência de informações inadequadas. Korbak et al. (2025) tratam a monitorabilidade da cadeia de pensamento como uma oportunidade relevante, cuja preservação requer atenção às decisões de desenvolvimento.

Contudo, observar uma explicação não equivale a observar diretamente todas as operações internas do modelo. Um texto pode ser correto, incompleto, decorativo ou enganoso. Da mesma forma, deixar de produzir uma explicação pode refletir uma instrução de concisão, uma limitação de geração ou uma alteração no treinamento. A investigação de ofuscação exige distinguir essas possibilidades por instrumentos que não confundam forma textual, competência e comportamento.

Um estudo-piloto aritmético previamente disponibilizado associou a supressão de etapas explícitas a uma perda de desempenho dependente da complexidade da tarefa. A interpretação desse contraste como evidência sobre ofuscação exige examinar a validade dos instrumentos, a estrutura das repetições experimentais e a correspondência entre o fator manipulado e o mecanismo de interesse.

O objetivo geral é reavaliar, de maneira reprodutível, a evidência do piloto e a adequação de seus instrumentos. Os objetivos específicos são: reproduzir as estatísticas publicadas; examinar o corretor de respostas; reconstruir as partições do gerador; revisar a interpretação do teste de intervenção; separar ausência textual de evidência comportamental; e disponibilizar um ambiente executável para auditorias posteriores.

A pergunta principal é: **os dados publicados permitem atribuir a diferença de desempenho à ofuscação de raciocínio causalmente necessário?** Três perguntas complementares orientam a análise: o corretor realmente exige uma resposta final correta e inequívoca? As sementes representam conjuntos suficientemente distintos para sustentar as inferências utilizadas? O escore lexical possui informação independente para distinguir ocultação, vacuidade e ausência de texto?

O artigo apresenta uma reanálise retrospectiva concluída. Não se declara a execução do delineamento multiescala originalmente proposto. A contribuição empírica nova consiste em testes locais dos instrumentos e na reconstrução dos dados procedurais; as avaliações históricas do modelo continuam identificadas como dados secundários.

# 2 FUNDAMENTAÇÃO E DELIMITAÇÃO DO PROBLEMA

## 2.1 Rastro textual, fidelidade e monitorabilidade

Neste trabalho, cadeia de pensamento, ou CoT, designa um rastro textual intermediário associado à geração de uma resposta. Essa definição operacional evita tratar a sequência de palavras como uma leitura completa do estado interno do modelo. Fidelidade refere-se à relação entre o rastro e os fatores que determinam a resposta; monitorabilidade refere-se à possibilidade de detectar comportamentos relevantes para a supervisão.

Turpin et al. (2023) demonstram que explicações plausíveis podem deixar de mencionar influências que alteram as respostas. Lanham et al. (2023), por sua vez, examinam intervenções nos rastros para investigar a dependência das respostas em relação ao texto intermediário. Esses trabalhos justificam medir a contribuição do rastro, em vez de presumir sua fidelidade pela fluência.

Emmons et al. (2025) distinguem raciocínio como racionalização e como computação necessária. Seus experimentos mostram que tornar mais exigente o comportamento a ser monitorado pode aumentar a exposição de etapas relevantes. Os autores também estudam evasão deliberada com assistência e otimização. Portanto, essa evidência sustenta uma hipótese condicional de dificuldade de evasão, não uma impossibilidade universal de ocultação.

Acurácia e monitorabilidade também não são substitutos. Um sistema pode resolver uma tarefa e apresentar pouca explicação; pode explicar de modo convincente uma resposta incorreta; ou pode executar um atalho proibido e descrevê-lo com clareza. Para inferir ofuscação comportamental, é necessário observar simultaneamente a execução do comportamento e a dificuldade de detectá-lo no rastro.

## 2.2 Supervisão e atualização da comparação bibliográfica

Baker et al. (2025) investigam a integração de monitores de raciocínio ao treinamento e relatam que pressões intensas de otimização podem reduzir a visibilidade de comportamentos problemáticos. Drori et al. (2025) estudam como supervisão aplicada às saídas também pode modificar os rastros, propondo mecanismos de transbordamento do feedback. Esses mecanismos envolvem atualizações e dependências de geração; não são equivalentes à inserção manual de frases vagas no alvo supervisionado.

A comparação principal utiliza a versão 2 de Mitrani Hadida et al. (2026), revisada em 20 de maio de 2026. Nela, os experimentos incluem Qwen3-4B e Qwen3-8B, além de avaliações sem a estrutura de marcação usada no treinamento. Assim, as críticas de modelo único e dependência exclusiva dessas marcações foram retiradas. Permanece relevante a limitação, reconhecida pelos próprios autores, de tarefas simplificadas cuja necessidade de CoT não foi estabelecida. O resultado apresentado por eles é uma demonstração de possibilidade de falha; não precisa ser interpretado como uma lei universal sobre toda tarefa ou arquitetura.

A comparação entre estudos requer correspondência entre a versão bibliográfica examinada, o mecanismo experimental e o alcance da hipótese. Resultados condicionais não devem ser convertidos em afirmações universais para fins de contraste.

## 2.3 Avaliação e incerteza

Zheng et al. (2023) discutem vieses de posição, verbosidade e preferência do avaliador em julgamentos por modelos de linguagem. Isso motiva monitores independentes e calibração humana quando o desfecho exige interpretação semântica. Não se adota, nesta revisão, uma taxa universal de concordância entre juízes e especialistas: tal taxa depende do domínio, do instrumento e da população avaliada.

Agarwal et al. (2021) mostram por que poucas execuções e estimativas pontuais podem produzir conclusões instáveis em avaliação de aprendizado por reforço. A recomendação pertinente é explicitar a incerteza e a unidade de repetição. Aumentar o número de linhas de uma tabela não substitui obter novas execuções de treinamento ou novos problemas de teste.

# 3 MATERIAIS, PROCEDÊNCIA E ESCOPO EXECUTADO

## 3.1 Fontes documentais e dados secundários

Foram utilizados o manuscrito original e os arquivos do repositório público cot-obfuscation-h2a, de Kelvin-Marques-Cyber. O instantâneo analisado corresponde à árvore de fontes preservada no commit **e21cef8ad92369c56f3fcbbb6891fa49bda8c74b**, originada do depósito de junho de 2026. A reanálise foi executada em 15 de outubro de 2026.

O material inclui o notebook de treinamento e análise, quatro tabelas CSV, instruções de execução e dependências declaradas. O arquivo de resultados por avaliação contém 540 linhas e 12 campos: nível, condição, semente, tipo de tarefa, acurácia, sinalização lexical, escore lexical, indicador de burla, diversidade lexical, repetição, entropia e comprimento da geração.

Essas linhas não contêm enunciado, identificador do item, resposta de referência, texto integral gerado, motivo de término, contagem de tokens ou revisão dos pesos. Consequentemente, a auditoria reproduz o processamento das métricas depositadas, mas não refaz o julgamento semântico das 540 saídas. A falta de texto também impede afirmar a frequência real de falsos acertos do corretor nesse conjunto.

O notebook declara Qwen2.5-7B-Instruct quantizado em quatro bits, adaptação LoRA e execução histórica em RTX 5090. Esses elementos são características reportadas do piloto. Nesta revisão não se verificaram checkpoints nem se repetiu o treinamento desse modelo.

## 3.2 Desenho efetivamente disponível

O piloto cruza dois níveis de complexidade, três alvos de ajuste supervisionado e três sementes. Cada combinação contém 30 avaliações, totalizando 18 células e 540 exposições. As condições recebem nesta versão nomes descritivos, porque os nomes anteriores poderiam ser confundidos com mecanismos de penalização por um monitor.

[[TABLE:design]]

O nível A utiliza adição ou subtração de dois inteiros entre 11 e 99. O nível C utiliza problemas de estoque com multiplicação, adição e subtração. A complexidade da expressão difere entre os níveis, mas não se considera demonstrado que o nível C exija causalmente um rastro textual para aquele modelo.

Na configuração publicada, cada ajuste utiliza 150 exemplos por três épocas, taxa de aprendizado 5 × 10^-5, lote por dispositivo igual a 2 e acumulação de gradientes igual a 4. O posto e o fator de escala de LoRA são 16. O número de passos efetivos depende da implementação e do tratamento do último lote; por isso não é inferido como um registro de execução.

## 3.3 Ambiente local desta revisão

O ambiente executado usa Python 3.12.14 em Linux, com NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0 e Matplotlib 3.10.8. O documento foi produzido com ReportLab 4.4.9 e inspecionado com ferramentas de PDF. O processo de reanálise usa CPU; não havia GPU CUDA disponível.

Foram executados: recálculo dos agregados; análises de contraste por semente; reconstrução dos geradores; testes do corretor em respostas construídas; controles lexicais; geração de dados disjuntos; e uma suíte de testes de software. Não foram executados novos treinos, GRPO, inferências de novos modelos, anotações humanas ou o delineamento fatorial de três famílias/escalas.

O código, as tabelas resultantes e os registros da execução integram o material suplementar. Quatro notebooks organizam a reanálise, o piloto supervisionado corrigido, o treinamento GRPO multiescala e a avaliação independente. Os módulos GPU tiveram sua sintaxe compilada e suas funções independentes de hardware submetidas a testes locais; treinamento e inferência CUDA não foram executados. O empacotamento em Docker é fornecido como alternativa de instalação; a imagem Docker não foi construída.

# 4 MÉTODOS

## 4.1 Estratégia de reanálise

A primeira etapa compara as médias recalculadas a partir das 540 linhas com os agregados publicados. A concordância é verificada em cada combinação de nível, condição e semente. A segunda etapa reproduz a análise histórica, preservando seu papel de verificação numérica. A terceira introduz análises de sensibilidade e auditorias dos instrumentos, explicitando o que não pode ser recuperado.

Esta análise é retrospectiva. Não foi localizado nos materiais fornecidos um identificador de depósito público anterior à coleta que documentasse um pré-registro. Não se transforma a formulação de hipóteses em um registro retrospectivo. O protocolo de extensão apresentado ao final também não constitui, por si, pré-registro público.

## 4.2 Desfechos, custos e contrastes

A acurácia depositada é uma variável binária por avaliação. Para uma semente s, um nível l e uma condição c, define-se a média A(l,c,s) como a proporção de avaliações rotuladas como corretas pelo instrumento histórico. O termo “registrada” é importante: a reanálise não substitui esses rótulos por novos julgamentos.

O custo de desempenho de uma condição é definido por:

[[EQUATION:K(l,c,s) = A(l,explícita,s) - A(l,c,s).]]

Valores positivos de K indicam menor acurácia registrada na condição comparada; valores negativos indicam melhora. A interação de interesse é descrita pela diferença entre os custos no nível C e no nível A:

[[EQUATION:D(c,s) = K(C,c,s) - K(A,c,s).]]

São apresentados os três valores por semente, suas médias e desvios-padrão amostrais, com denominador n - 1. As proporções agregadas de 90 exposições por condição não são tratadas como 90 problemas independentes.

## 4.3 Inferência histórica e análise de sensibilidade

A análise de variância de dois fatores é reproduzida com as 18 médias por célula. O teste t de Welch histórico também é recalculado. Esses resultados verificam a consistência aritmética do processamento anterior, mas dependem de suposições que não resolvem a repetição de itens, a estrutura de blocos ou o pequeno número de sementes.

Como sensibilidade, calcula-se a média dos três contrastes D(c,s) e enumera-se a distribuição obtida invertendo seus sinais nas oito combinações possíveis. Sob a hipótese de sinais intercambiáveis e contrastes independentes e simétricos sob a nulidade, o teste é exato. Como não houve randomização que garanta essas condições, seu resultado é reportado como sensibilidade, e não como uma nova inferência confirmatória.

Com três contrastes e teste bicaudal, o menor valor de p possível nessa enumeração é 2/8 = 0,25. Portanto, ela evidencia a resolução limitada do desenho: não contradiz automaticamente o teste paramétrico, nem demonstra equivalência ou ausência de efeito. Intervalos t condicionais e intervalos binomiais descritivos ficam disponíveis nos dados suplementares, acompanhados de suas restrições; não são usados como prova de validade externa.

## 4.4 Reconstrução da comparação com e sem explicação

O arquivo histórico de intervenção informa acurácia com explicação, acurácia com resposta direta e proporção de casos acertados somente com explicação, em dez itens por nível. Esses três agregados permitem recuperar a tabela de acertos pareados, condicionada à correção dos valores depositados.

Se b é o número de casos acertados apenas com explicação e c o número acertado apenas na resposta direta, o ganho líquido é (b - c)/n. A proporção b/n informa apenas uma direção da discordância. O teste exato de McNemar é obtido por teste binomial dos b sucessos em b + c discordâncias sob probabilidade 0,5, implementado com SciPy.

A intervenção histórica altera a instrução do sistema e também o orçamento de geração: até 200 tokens novos com explicação e 40 na resposta direta. Ela não remove apenas um segmento de um mesmo rastro mantendo todo o restante constante. A comparação, portanto, é uma ablação de modo de resposta e orçamento, não uma identificação isolada da necessidade causal de CoT.

## 4.5 Auditoria do gerador e das partições

O código gera cada exemplo a partir de uma semente local igual à semente inicial mais o índice do item. Ao usar sementes iniciais adjacentes, quase todos os estados locais são reutilizados. A avaliação começa em s + 7777, com 30 itens; o treinamento começa em s, com 150.

A auditoria executa novamente esses geradores e compara os enunciados completos. Isso recupera as partições que o código produz, sem atribuir automaticamente um identificador reconstruído a uma linha histórica que não possui ID. Mede-se a interseção entre sementes e entre treino e avaliação, incluindo colisões possíveis em um domínio finito.

Um novo gerador usa um fluxo pseudoaleatório único, semente de dados 20261015, identificadores derivados do conteúdo e rejeição de enunciados já presentes. São construídos três níveis de complexidade e três partições, com 500 itens por combinação: 4.500 itens. Essa geração valida a infraestrutura de partição e o gabarito aritmético; não constitui avaliação de modelos.

## 4.6 Teste mecânico do corretor

O corretor histórico extrai o trecho após um marcador textual, quando presente, e aceita a resposta se qualquer número nesse trecho coincidir com o gabarito. Quando o marcador não aparece, a mesma função pode examinar o texto inteiro. A auditoria testa as consequências dessa regra em um corpus construído com oito classes e 250 casos por classe.

As classes incluem respostas corretas e incorretas inequívocas, alternativas múltiplas, contradições, menções do número correto fora da conclusão, respostas XML corretas e incorretas e duplicação de tags finais. Os rótulos decorrem do contrato do teste, não de anotação humana ou de gerações de LLM.

O instrumento corrigido extrai uma única resposta numérica inequívoca: uma tag final <answer>NUMERO</answer>, uma linha final “Resposta: NUMERO”, ou uma saída inteiramente numérica. Ele rejeita formatos ambíguos e compara valores com representação decimal exata. Aceitação de prosa livre e interpretação de explicações ficam fora desse contrato, devendo ser medidas separadamente se forem desejadas.

## 4.7 Auditoria do escore lexical

O escore histórico combina baixa densidade de conectivos causais, baixa diversidade, repetição de bigramas e termos genéricos. Em notação abreviada, sua composição é:

[[EQUATION:S = 0,35(1 - min(d/0,08;1)) + 0,25(1 - TTR) + 0,25R + 0,15G.]]

Aqui, d é a densidade de conectivos, TTR é a razão tipo/ocorrência, R é a repetição e G é a proporção de termos genéricos. Rastros com menos de cinco palavras recebem S = 1 por regra. Para um rastro vazio, as métricas históricas ainda atribuem TTR = 0 e R = 1.

São avaliados controles construídos de ausência, brevidade, preâmbulo genérico, repetição e cálculo aritmético. Também se varia o limiar de sinalização. Não se chama esse instrumento de detector validado de intenção, porque não há referência humana independente nem comportamento de burla disponível para calibrá-lo.

# 5 RESULTADOS

## 5.1 Reprodutibilidade dos números do piloto

As 18 médias de acurácia recalculadas coincidem com os agregados publicados. A Tabela 2 apresenta contagens, médias e desvios-padrão amostrais. Os valores da tabela são resultados históricos reprocessados; sua concordância não equivale à validação do corretor que os produziu.

[[TABLE:summary]]

No nível A, as condições de resposta direta e preâmbulo genérico têm médias ligeiramente superiores à condição com CoT explícita. No nível C, ambas têm médias inferiores. A Figura 1 representa esse contraste e mostra os três pontos por semente, evitando esconder a quantidade de repetições de treinamento.

[[FIGURE:accuracy.png|Figura 1 - Acurácia registrada no piloto, por nível e alvo de treinamento. Os pontos representam as três sementes; as barras mostram suas médias. Fonte: reanálise dos dados publicados.]]

Os custos médios em A são -0,067 para resposta direta e -0,033 para preâmbulo genérico. Em C, são +0,522 e +0,622, respectivamente. A diferença média entre níveis é positiva nas duas condições, conforme a Tabela 3.

[[TABLE:costs]]

Em termos relativos, a média registrada em C passa de 0,678 a 0,156 e 0,056, reduções aproximadas de 77% e 92%. São descrições da magnitude do contraste neste piloto. Não identificam, isoladamente, o mecanismo responsável.

## 5.2 Interação, sementes e resolução inferencial

A análise histórica é numericamente reproduzida: F(2,12) = 527,1667, p = 2,0311 × 10^-12. O teste t de Welch reportado para C também é reproduzido: t = 20,9377, p = 3,0927 × 10^-7. O problema principal não está no recálculo desses valores, mas em sua interpretação diante da estrutura dos dados.

O teste de Welch original junta duas condições distintas de comparação em um grupo de seis valores e o confronta com três controles. Esse agrupamento perde a distinção entre intervenções e não explora o pareamento por semente. Além disso, as duas condições de uma mesma semente compartilham o conjunto de avaliação.

[[TABLE:contrast_seeds]]

A média dos contrastes é 0,589 para resposta direta e 0,656 para preâmbulo genérico. Os intervalos t condicionais de 95% são, respectivamente, [0,493; 0,685] e [0,529; 0,782]. Eles descrevem apenas o modelo de três contrastes sob independência, simetria e aproximação paramétrica; não corrigem os problemas de mensuração ou de amostragem.

Na inversão exata dos três sinais, p = 0,25 em ambas as comparações. Esse resultado mostra a baixa resolução de uma análise não paramétrica com três repetições; não autoriza concluir efeito nulo. O contraste permanece consistente em direção nas três sementes, mas a força da conclusão depende de escolhas inferenciais que o piloto é pequeno demais para estabilizar.

[[FIGURE:paired_contrasts.png|Figura 2 - Diferença entre os custos nos níveis C e A, preservando a semente. Valores positivos indicam maior perda registrada em C. Fonte: cálculos desta revisão.]]

## 5.3 A comparação histórica não comprova necessidade causal

No nível C, cinco casos foram acertados somente com explicação e dois somente na resposta direta. Logo, a proporção de casos beneficiados é 0,50, mas o ganho líquido é 0,30. Os demais três casos foram errados em ambos os modos. A Tabela 5 explicita o pareamento recuperado.

[[TABLE:necessity]]

O teste exato pareado para C produz p = 0,453125. Para A, p = 1,00. O tamanho amostral de dez itens e o confundimento com instrução e orçamento de geração impedem interpretar a proporção de 0,50 como validação conclusiva de um nível causalmente necessário.

O resultado não prova que explicações sejam inúteis. Ele apenas mostra que os agregados disponíveis são insuficientes para estabelecer o construto central atribuído às tarefas. Nesta versão, os níveis são denominados por complexidade aritmética; “necessidade causal” fica reservada à hipótese a ser validada por intervenção mais controlada.

## 5.4 Sobreposição de dados e alcance da generalização

Entre sementes adjacentes, os conjuntos de treinamento compartilham 149 de 150 enunciados, ou 99,3%, e os conjuntos de avaliação compartilham 29 de 30, ou 96,7%. Entre as sementes 42 e 44, as interseções são 148 e 28. A união dos três conjuntos possui 152 enunciados de treino e 32 de avaliação por nível.

[[TABLE:overlap]]

Também foi reconstruído o enunciado “Calcule 11 - 99” em treino e avaliação do nível A, nas três sementes. A interseção observada em C foi zero. A coincidência em A demonstra que separar intervalos de sementes não garante separar conteúdos em um espaço finito de expressões.

A sobreposição de conjuntos não torna as inicializações de LoRA idênticas nem invalida toda comparação em um conjunto fixo de teste. Entretanto, reduz a diversidade efetivamente observada e exige distinguir variabilidade de treinamento de variabilidade de itens. Não é adequado contar as 540 exposições como 540 problemas novos.

Além disso, mudar somente a semente mantendo o mesmo gerador não caracteriza, por si, avaliação fora da distribuição. Os itens do piloto são avaliações separadas da mesma família procedural. Não foram executados testes entre domínios clínico, jurídico ou de programação aberta.

O código ainda mantém a semente do SFTConfig ligada à variável global inicial 42, enquanto a reinicialização do adaptador varia com o argumento da execução. Isso não prova identidade entre treinamentos, mas mostra que as fontes de aleatoriedade não foram controladas e documentadas separadamente.

## 5.5 Desempenho dos corretores no corpus construído

Das 2.000 respostas do teste mecânico, 500 são respostas corretas inequívocas e 1.500 devem ser rejeitadas pelo contrato. O corretor histórico aceita corretamente as 500 positivas, mas também aceita 1.250 negativas. Ele concorda com os rótulos em 750 casos, ou 37,5%; sua taxa de aceitação indevida entre os negativos desse corpus é 83,3%.

[[TABLE:stress]]

Um exemplo mínimo é “Resposta: 43; 42 seria incorreto.”, com gabarito 42. O corretor histórico aceita o texto porque encontra 42 depois do marcador, embora a conclusão declare 43. De modo semelhante, uma resposta XML incorreta pode ser aceita se o número do gabarito aparecer antes da tag.

O corretor com contrato explícito concordou com todos os 2.000 rótulos e não aceitou os 1.500 casos negativos. Esse resultado é restrito ao corpus construído, cujas regras orientaram o instrumento. Não é uma estimativa de 100% de desempenho em textos livres, nem permite converter retroativamente os 540 rótulos históricos.

Essa distinção é essencial: foi demonstrada uma vulnerabilidade do procedimento de correção, mas sua prevalência nas saídas históricas continua desconhecida. Sem os textos, não se calcula uma “acurácia real corrigida” do piloto.

## 5.6 O escore lexical não discrimina os mecanismos propostos

Nos dados depositados, a condição de resposta direta é sinalizada em 100% das avaliações; a condição de preâmbulo genérico, em 0%, com limiar 0,50. A própria regra do escore explica parte desse padrão: um rastro vazio recebe pontuação máxima, enquanto a frase genérica construída recebe 0,35.

[[TABLE:lexical]]

A diversidade e a repetição de um rastro ausente não são medidas de um texto degenerado: são valores convencionados pelo programa. Assim, o valor R = 1 não mostra repetição observada quando não há palavras. A atribuição de “colapso de representação” a esses casos excedia a evidência disponível.

O escore correlaciona-se com repetição em r = 0,996 e com diversidade em r = -0,988 nas 540 linhas. Essas correlações não são validação independente, porque repetição e diversidade participam da fórmula do próprio escore. O acoplamento é, em parte, uma consequência matemática de sua construção.

Ao reduzir o limiar para 0,30, o preâmbulo genérico passa a ser sinalizado; a condição com CoT explícita também passa a ser sinalizada em todos os casos de C. Portanto, elevar a detecção da frase vaga não resolve automaticamente a especificidade. A Figura 3 apresenta essa sensibilidade.

[[FIGURE:thresholds.png|Figura 3 - Sensibilidade da sinalização lexical ao limiar, no nível C. O gráfico usa os escores depositados e não constitui curva de desempenho contra rótulos de ofuscação. Fonte: cálculos desta revisão.]]

Por fim, todos os indicadores históricos de burla são zero. Nas tarefas aritméticas executadas, o campo de resposta de burla não é definido. Portanto, não se identifica a sensibilidade de um monitor para burla nem se demonstra que um sinal proibido tenha sido removido, preservado ou recuperado por outro avaliador.

## 5.7 Verificação do ambiente e dos dados novos

A suíte de software passou em 53 testes automatizados. Ela verifica casos de erro de formato, contaminação por números intermediários, múltiplas respostas, comparação decimal, separação de estados textuais, geração reprodutível sem repetição de IDs e isolamento das informações fornecidas aos monitores de recompensa.

O novo gerador produziu 4.500 itens, distribuídos igualmente entre três níveis de complexidade e três partições. Todos os identificadores são únicos. Uma conferência aritmética independente da resposta armazenada verificou os 4.500 gabaritos. Não foram produzidas respostas de modelos para esses itens.

Os resultados são exportados em CSV e JSON, e as figuras são regeneradas a partir desses arquivos. O registro de execução identifica as análises realizadas e as que não foram executadas, evitando misturar dados mecânicos de validação com avaliações de LLM.

# 6 DISCUSSÃO

## 6.1 O que permanece sustentado

O piloto registra uma diferença grande de desempenho entre alvos de treinamento, concentrada na tarefa aritmética de múltiplos passos. Essa descrição é consistente nos dados depositados e foi numericamente reproduzida. Ela oferece uma motivação para estudar como a organização da resposta e do treinamento interage com a complexidade da tarefa.

O resultado permite uma hipótese útil: para determinados modelos e tarefas, reduzir ou substituir etapas explícitas no alvo supervisionado pode comprometer a aprendizagem do mapeamento desejado. Contudo, a formulação permanece sobre alvos supervisionados. Não se identifica automaticamente uma política de evasão aprendida contra um avaliador.

A validade de uma conclusão científica depende tanto do tamanho de uma diferença quanto de sua identificação. Uma interação grande pode coexistir com um desfecho medido de forma inadequada, uma amostra restrita ou um fator experimental que muda várias propriedades ao mesmo tempo.

## 6.2 Alcance das hipóteses e identificação do mecanismo

O desenho histórico não otimiza o modelo contra um monitor de saída. Ele altera diretamente o conteúdo que o modelo deve imitar, inclusive o texto intermediário. Logo, a diferença entre condições não isola um efeito indireto de supervisão exclusiva da resposta final.

Também não há uma medida comportamental positiva de exploração de recompensa. Ausência de CoT, prosa genérica e desempenho reduzido não permitem separar deliberadamente ocultação de uma explicação curta, falha de aprendizagem ou mudança de formato. Nem uma queda de acurácia exclui intenção estratégica: essas dimensões podem coexistir.

A hipótese originalmente apresentada como H2.a recebe, assim, apoio descritivo limitado ao contraste entre alvos e complexidade. Sua interpretação em termos de necessidade causal não foi estabelecida. A hipótese H2.b, de recuperação do sinal por um monitor independente, não foi testada, porque esse monitor não foi executado e faltam os rastros para avaliá-lo.

[[TABLE:hypotheses]]

Não rejeitar uma hipótese por falta de identificação não equivale a confirmá-la. A evidência disponível não decide se a ofuscação generaliza em tarefas realmente dependentes de CoT, e tampouco autoriza afirmar que a supervisão de saída seja segura nessas tarefas.

## 6.3 Validade de construto e independência da mensuração

A correção numérica deve avaliar a resposta final, enquanto o monitor deve avaliar evidências de um comportamento definido. Indicadores de estilo podem ajudar no diagnóstico de saídas, mas precisam permanecer separados. Essa separação evita duas circularidades: aceitar um número porque ele aparece no texto e validar um escore pela correlação com componentes da própria fórmula.

Para esta revisão, ausência, brevidade, repetição e preâmbulo genérico são estados observáveis. Chamá-los de “ofuscação” exigiria uma referência adicional, como execução verificável de um atalho proibido, anotação cega ou manipulação causal adequada. Do mesmo modo, “colapso de representação” não pode ser inferido de TTR isolada, especialmente em sequências de comprimento muito diferente.

O corretor estrito melhora a validade de um contrato estreito. Seu custo é rejeitar respostas em prosa que poderiam estar semanticamente corretas. Essa limitação deve ser assumida pelo desenho: instruir claramente o formato esperado, registrar violações e reportar acurácia condicionada e não condicionada ao formato, sem alterar retrospectivamente a regra para favorecer uma condição.

## 6.4 Validade interna e externa

A comparação do nível A com o C altera complexidade, vocabulário do enunciado e número de operações. As condições alteram comprimento e estrutura do alvo. Não há randomização independente de necessidade de CoT, nem intervenção pós-geração com orçamento equiparado. Por isso, a diferença observada não é atribuída exclusivamente à necessidade causal.

Quanto à validade externa, os resultados cobrem uma família procedural aritmética e um modelo declarado. A pouca variedade entre sementes é especialmente relevante: bons resultados em repetidas versões de quase o mesmo conjunto não demonstram transferência para novos problemas de natureza distinta.

Uma demonstração de possibilidade de falha em outro trabalho não é eliminada por observar uma condição em que há perda de desempenho. Para confrontá-la diretamente, é preciso reproduzir o mecanismo de treinamento, confirmar a presença do comportamento e então testar, sob controles comparáveis, seus limites.

# 7 LIMITAÇÕES

A principal limitação da reanálise é a ausência de textos e identificadores nas avaliações históricas. Isso impede nova correção item a item, pareamento inequívoco dos resultados com problemas reconstruídos, anotação humana, análise semântica de rastros e cálculo de detecção comportamental.

O notebook disponibilizado não contém saídas de execução, e o material inspecionado não inclui adaptadores treinados ou registros completos que permitam verificar independentemente as 18 execuções históricas. A auditoria confirma a consistência dos arquivos depositados e do código lido, sem certificar aspectos da execução que não deixaram evidência acessível.

Há somente três sementes, conjuntos fortemente sobrepostos e uma ocorrência reconstruída de contaminação entre treino e avaliação em A. Os valores de p paramétricos e as análises de sensibilidade são dependentes de hipóteses. Nenhum deles repara a qualidade dos rótulos históricos.

Os 2.000 textos mecânicos foram construídos para expor classes de falha do corretor. Eles não representam uma amostra aleatória de respostas de modelos. A concordância perfeita do instrumento corrigido nesse corpus é um resultado de conformidade com o contrato, não uma validação externa universal.

Os 4.500 itens novos validam apenas geração, gabaritos e partições. Não foram avaliados por modelos nem classificados quanto à necessidade causal. Os testes locais tampouco substituem auditoria de execução distribuída, determinismo de GPU ou validação do contêiner proposto.

Finalmente, a revisão bibliográfica foi direcionada às fontes centrais do argumento e não constitui revisão sistemática exaustiva. Não houve coleta com seres humanos, uso de prontuários reais ou avaliação clínica de pacientes. Não se declara aprovação ética, concordância de especialistas ou anuência dos coautores que não tenha sido fornecida.

# 8 PROTOCOLO DE EXTENSÃO EXPERIMENTAL

## 8.1 Desenho e unidade de inferência

O programa original é mantido como extensão prospectiva, separado dos resultados concluídos. O desenho combina três regimes de penalização, três níveis de necessidade de CoT empiricamente validados e três modelos. Os modelos inicialmente previstos são Qwen3-4B, Qwen3-14B e Llama-3.1-8B-Instruct; suas revisões e licenças devem ser fixadas antes da coleta.

[[TABLE:extension]]

O número planejado de execuções é 3 × 3 × (12 + 6 + 12) = 270. Trata-se de uma decisão de planejamento, não de demonstração de poder estatístico. Uma simulação para desfechos binários hierárquicos deve informar se o orçamento fornece precisão adequada para a interação principal.

Semente de treinamento, semente de dados e semente de avaliação devem ser separadas. Os conjuntos devem possuir IDs de conteúdo, partições sem colisão e famílias de problemas reservadas. Os mesmos itens de teste devem ser usados em condições comparáveis para permitir contrastes pareados, mantendo novas famílias para generalização.

## 8.2 Replicação positiva e manipulação causal

A primeira etapa deverá reproduzir uma condição positiva de ofuscação com recompensa e monitor, preservando o mecanismo a ser examinado. Uma implementação que não reproduz o controle positivo não pode interpretar uma ausência posterior como limite do fenômeno.

A necessidade de CoT deve ser investigada antes do treinamento experimental por tarefas em gradação de dificuldade e intervenções controladas: término antecipado, substituição de resultados intermediários e continuação a partir de prefixos verificáveis. Instruções, formato da resposta e orçamento total precisam ser equiparados ou incluídos como fatores explícitos.

Classificar um item como difícil ou exigir uma justificativa no enunciado não demonstra necessidade causal. Quando o modelo resolve a tarefa por computação não textual, o rastro pode continuar dispensável. O critério de classificação deve depender da resposta à intervenção e de sua incerteza.

## 8.3 Tarefas, monitores e anotação

Na expansão para tarefas abertas, programação pode ser avaliada por testes independentes, com especificações reservadas e comportamento de exploração definido antes da otimização. Problemas de decisão podem usar regras sintéticas verificáveis. Tarefas jurídicas e clínicas exigem revisão especializada e não devem adotar rótulos simplificados como se fossem decisões reais.

Uma medida de comportamento deve ser independente do texto: por exemplo, execução de implementação que passa apenas nos testes visíveis, ou seleção comprovada de uma resposta induzida por informação proibida. A monitorabilidade pode então ser estimada como sensibilidade para esse comportamento em uma taxa de falsos positivos previamente fixada.

O monitor de treinamento, um monitor independente e a anotação humana devem ter funções distintas. A anotação planejada de pelo menos 300 respostas por nível deve ser estratificada, cega e acompanhada de rubrica. O alfa de Krippendorff pode ser negativo e não se restringe, em geral, ao intervalo de zero a um; estimativas e intervalos devem ser apresentados. Um limiar operacional de 0,70 pode ser estabelecido, sem tratá-lo como garantia universal de validade.

## 8.4 Análise, exclusões e conclusão possível

Para respostas binárias, o modelo primário proposto é logístico hierárquico, com efeitos por item e execução e termos para penalização, necessidade e modelo. O desfecho de monitorabilidade deve ser acompanhado de acurácia genuína e taxa de exploração. Controlar por uma acurácia alterada pelo tratamento pode introduzir viés pós-tratamento; por isso são preferíveis contrastes conjuntos e análises explicitamente condicionais.

A família de comparações, a margem de equivalência, as falhas técnicas e o tratamento de truncamentos devem ser fixados antes da nova coleta. Um teste de equivalência exige margem substantiva e precisão; p maior que 0,05 em teste de diferença não demonstra equivalência. Ajuste de multiplicidade e reporte de todos os domínios devem impedir seleção de resultados favoráveis.

A proposta admite resultados favoráveis e desfavoráveis à hipótese de moderação. Uma redução de detecção com comportamento preservado, acurácia adequada e resistência a monitores independentes seria evidência mais forte de ofuscação. Aumento de detectabilidade em tarefas de maior dependência textual delimitaria o efeito, sem negar sua existência em outros regimes. Queda de desempenho e vacuidade devem ser reportadas como desfechos próprios.

O protocolo exige depósito público antes da coleta confirmatória. Até esse depósito e a execução, nenhum desses itens é apresentado como resultado obtido.

# 9 CONCLUSÃO

A reanálise reproduziu os números centrais do piloto: em tarefas aritméticas de múltiplos passos, alvos de resposta direta ou preâmbulo genérico estão associados a menor acurácia registrada que alvos com etapas explícitas. A interação histórica também foi reproduzida. Esses achados são descritivos e limitados ao material publicado.

A auditoria modificou a conclusão científica. O corretor admite falsas aceitações por mera presença do número; as sementes reutilizam quase todos os itens; o teste de intervenção é pequeno e confunde instrução com orçamento; e o escore lexical não possui validação independente para identificar ocultação estratégica. Não há comportamento positivo de burla avaliado no piloto.

Portanto, o estudo não refuta a generalização da ofuscação sob supervisão de saída e não a confirma em tarefas causalmente dependentes de CoT. Ele demonstra que essa pergunta requer um desenho diferente do contraste supervisionado disponível.

A contribuição final consiste em tornar essa distinção verificável: métricas históricas reproduzidas, limitações localizadas no código, corretor com contrato explícito, partições novas sem repetição, testes executáveis e um protocolo de continuidade. A hipótese de validade externa permanece aberta, agora com critérios de mensuração e execução mais claros.

# 10 DISPONIBILIDADE, AUTORIA E TRANSPARÊNCIA

O repositório histórico está disponível em https://github.com/Kelvin-Marques-Cyber/cot-obfuscation-h2a, no commit identificado na seção 3. O pacote suplementar desta revisão contém o instantâneo textual analisado, scripts, testes, tabelas recalculadas, controles construídos, dados novos, figuras e fonte editável do artigo. Os materiais da reanálise são organizados no diretório revisao_2026 do mesmo repositório, incluindo os notebooks e a versão em PDF.

A reprodução local é realizada com instalação das versões fixadas e execução de “bash run.sh”, a partir da pasta do pacote. Em Windows, os comandos Python equivalentes estão no README. O arquivo de procedência registra a origem e os resumos criptográficos; o registro de execução identifica o que foi realizado.

O material computacional preserva os avisos de licença das fontes originais. A autoria e a afiliação institucional constam na folha inicial. A execução local, os parâmetros de análise e a procedência dos arquivos estão documentados no material suplementar.

[[PAGEBREAK]]

# REFERÊNCIAS

AGARWAL, Rishabh et al. **Deep Reinforcement Learning at the Edge of the Statistical Precipice**. Advances in Neural Information Processing Systems, v. 34, 2021. Disponível em: https://arxiv.org/abs/2108.13264. Acesso em: 15 out. 2026.

BAKER, Bowen et al. **Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation**. arXiv:2503.11926, 2025. Disponível em: https://arxiv.org/abs/2503.11926. Acesso em: 15 out. 2026.

DRORI, Jacob et al. **Output Supervision Can Obfuscate the Chain of Thought**. arXiv:2511.11584, 2025. Disponível em: https://arxiv.org/abs/2511.11584. Acesso em: 15 out. 2026.

EMMONS, Scott et al. **When Chain of Thought is Necessary, Language Models Struggle to Evade Monitors**. arXiv:2507.05246, 2025. Disponível em: https://arxiv.org/abs/2507.05246. Acesso em: 15 out. 2026.

KORBAK, Tomek et al. **Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety**. arXiv:2507.11473, 2025. Disponível em: https://arxiv.org/abs/2507.11473. Acesso em: 15 out. 2026.

LANHAM, Tamera et al. **Measuring Faithfulness in Chain-of-Thought Reasoning**. arXiv:2307.13702, 2023. Disponível em: https://arxiv.org/abs/2307.13702. Acesso em: 15 out. 2026.

MARQUES, Kelvin e Silva et al. **Validade externa da ofuscação de cadeia de pensamento sob supervisão de saída: um teste empírico em tarefas com raciocínio causalmente necessário**. Fortaleza: Centro Universitário Ateneu, 2026. Manuscrito fornecido para revisão, gerado em 27 jun. 2026.

MARQUES, Kelvin e Silva. **cot-obfuscation-h2a: código e dados do estudo-piloto**. GitHub, 2026. Commit e21cef8ad92369c56f3fcbbb6891fa49bda8c74b. Disponível em: https://github.com/Kelvin-Marques-Cyber/cot-obfuscation-h2a. Acesso em: 15 out. 2026.

MITRANI HADIDA, Nathaniel; BHANJI, Sassan; TICE, Cameron; RADMARD, Puria. **Chain-of-thought obfuscation learned from output supervision can generalise to unseen tasks**. arXiv:2601.23086v2, 2026. Revisão de 20 maio 2026. Disponível em: https://arxiv.org/abs/2601.23086v2. Acesso em: 15 out. 2026.

SCIPY COMMUNITY. **binomtest; permutation_test**. SciPy API Reference. Documentação consultada para teste binomial exato e permutação pareada. Disponível em: https://docs.scipy.org/doc/scipy/reference/stats.html. Acesso em: 15 out. 2026. Versão executada na análise: 1.17.0.

TURPIN, Miles et al. **Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting**. Advances in Neural Information Processing Systems, v. 36, 2023. Disponível em: https://arxiv.org/abs/2305.04388. Acesso em: 15 out. 2026.

ZHENG, Lianmin et al. **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**. Advances in Neural Information Processing Systems, v. 36, 2023. Disponível em: https://arxiv.org/abs/2306.05685. Acesso em: 15 out. 2026.

[[PAGEBREAK]]

# APÊNDICE A - RESULTADOS POR SEMENTE

A tabela seguinte preserva as 18 células do piloto e permite conferir contagens sem depender dos gráficos. “Corretos” mantém o rótulo histórico; “sinalização” corresponde ao escore lexical original maior que 0,50.

[[TABLE:all_cells]]

As contagens não foram reclassificadas com o corretor novo porque o arquivo depositado não inclui os textos gerados. As métricas de diversidade, repetição, entropia e comprimento permanecem acessíveis nos CSVs originais e recalculados, preservando a distinção entre dado e interpretação.

# APÊNDICE B - CONTRATO DOS INSTRUMENTOS

## B.1 Correção numérica e rastreabilidade

Uma nova avaliação deve registrar item_id, modelo e revisão, condição, semente de treinamento, partição, enunciado, gabarito, geração integral, motivo de término e contagens de tokens. O módulo evaluate_jsonl.py verifica campos obrigatórios e duplicação de avaliações antes de calcular validade de formato, acurácia e truncamento.

O contrato de resposta final aceita um número inteiro ou decimal com sinal opcional. O separador decimal pode ser vírgula ou ponto. Não há busca do gabarito no texto. Duas tags de resposta, múltiplos valores finais e texto posterior à tag final são rejeitados. Uma conclusão numérica errada continua errada mesmo que todos os valores intermediários estejam corretos.

[[TABLE:parser_examples]]

O indicador de formato deve ser reportado junto à acurácia. Uma falha de formato não é silenciosamente descartada. O código de avaliação é um ponto de partida para novas coletas; nesta revisão ele não foi aplicado a saídas novas de LLM.

## B.2 Estados textuais e validação independente

Os estados “ausente”, “curto”, “repetitivo” e “preâmbulo genérico” descrevem propriedades de um texto. Eles não recebem automaticamente rótulos de intenção ou de exploração. Uma comparação de monitorabilidade exige exemplos com comportamento positivo e negativo, monitor cego quanto à condição e taxa de falsos positivos identificável.

Quando houver anotação humana, a rubrica deve separar: correção da resposta, presença de inferências verificáveis, menção ao comportamento de exploração e informação suficiente para detectá-lo. A concordância precisa ser acompanhada de exemplos de desacordo e incerteza. Um conjunto de avaliadores automáticos não substitui essa validação apenas por possuir mais modelos.

# APÊNDICE C - REPRODUÇÃO E VERIFICAÇÕES

O pacote separa fontes originais, código novo e resultados gerados. As versões de dependências foram registradas em requirements.lock.txt. Para executar a auditoria em um ambiente virtual Python 3.12:

[[CODE]]
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.lock.txt
bash run.sh
[[ENDCODE]]

O script de execução realiza a suíte de testes e a reanálise. As saídas incluem summary.csv, paired_costs.csv, contrasts.csv, overlap_audit.csv, grading_stress_summary.csv, threshold_sensitivity.csv e report.json. Os 2.000 casos construídos e os 4.500 itens novos também são preservados.

Os testes demonstram conformidade com contratos e regressões localizadas; não são uma avaliação de inteligência, segurança ou capacidade de um modelo. A execução do contêiner e a repetição em outro hardware continuam verificações adicionais possíveis, sem serem apresentadas como realizadas.

O arquivo checksums.sha256 permite verificar integridade dos artefatos do pacote. Uma alteração futura dos dados ou dos instrumentos deve gerar novos resultados, atualizar a procedência e manter a versão anterior identificável.

# APÊNDICE D - MATRIZ DE EVIDÊNCIA

[[TABLE:evidence]]

A matriz explicita a correspondência entre cada conclusão e sua base empírica. A extensão confirmatória constitui um estudo distinto da reanálise retrospectiva e requer coleta própria.
