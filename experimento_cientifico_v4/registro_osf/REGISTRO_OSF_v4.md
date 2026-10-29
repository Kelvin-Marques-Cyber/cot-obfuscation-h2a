# Registro público do estudo prospectivo v4 no OSF

Este documento contém o texto de cada campo do modelo **OSF Preregistration**
(o mesmo usado no registro do v3.2, https://osf.io/sjwdf/) e os arquivos a
anexar. O registro precisa ser feito pela conta do proprietário do projeto no
OSF, **antes do primeiro treinamento**. Nenhum treino do v4 foi executado; os
únicos treinos existentes são os do teste técnico, descritos no campo
"Explanation of foreknowledge".

## Passo a passo

0. **Descartar a atualização aberta no v3.2.** Em 29/10/2026 a API pública do
   OSF mostrava o registro https://osf.io/sjwdf/ com `revision_state:
   in_progress`, e a página dele exibia "Updates being made". O v4 não é uma
   atualização do v3.2: é um registro novo. Abrir https://osf.io/sjwdf/ →
   menu **Updates** → **Continue Update** → **Delete Draft Update** →
   **Delete**. Não enviar essa atualização (arquivos também não podem ser
   anexados numa atualização).
1. Entrar no OSF e abrir o projeto https://osf.io/7hzjb/ (o mesmo do v3.2).
2. Aba **Registrations** → **New registration** → modelo **OSF Preregistration**.
3. Preencher título, descrição e campos com os textos abaixo. Os campos de
   escolha indicam a opção exata a marcar.
4. Anexar os quatro arquivos desta pasta e do repositório:

   | Arquivo | SHA256 |
   |---|---|
   | `pacote_registro_v4.zip` | `af36a8718ed985c26d28e80a7c3f02be0d4979a3b3995b1cc56d563598247823` |
   | `codigo_v4_b858d12.zip` | `773001fee94c8e9e09904fe18fa72ab47fcb586be527208f373e9ec2bb96e70f` |
   | `../analises_secundarias.py` | `6065d011fe027ed2c42fefcbb8af63c6d914f4b803c1a2170e12461363b1fbe0` |
   | `../PROTOCOLO_v4.md` | `b89537a3cc963dff7f9f47ef16ebd73e1c016f65d8b4f01a3e689369a4b7a296` |

   Sugestão: o ZIP do pacote em "Data collection procedures - File upload", o
   código e o script em "Statistical models - File upload" e o protocolo em
   "Study design" (campo de arquivo logo abaixo do texto).
5. Escolher **Make registration public immediately** (sem embargo) e
   confirmar. O OSF atribui data e hora ao registro; ele não pode ser editado
   depois, apenas retirado com justificativa pública.
6. Copiar a URL pública do registro (formato `https://osf.io/xxxxx/`) e
   informá-la no notebook 00 (`PUBLIC_REGISTRATION_URL`) ou ao executor. O
   runtime grava essa URL com os resumos do protocolo e só então libera os
   treinos.

**Escolha da GPU.** O campo "Data collection procedures" fixa uma única classe
de GPU para todo o estudo: **NVIDIA H100**, a classe mais rápida oferecida
pelo Colab (as outras são T4, L4, G4 e A100), escolhida em 29/10/2026, antes
do registro. A triagem DEV foi feita na Tesla T4 e não é repetida; o campo
declara essa diferença. O código e o protocolo congelados exigem apenas
"T4 ou superior" (`colab/sessao.py`, `PROTOCOLO_v4.md`, seção 9), de modo que a
troca não altera nenhum SHA256. Para usar outra classe, altere o nome nesse
parágrafo **antes** de registrar. Depois do registro, a classe não muda.

**Conferência do pacote.** O pacote foi gerado com o código do commit
`b858d128bcd545dd958e0dbed9978a7aea2591b3`, o mesmo fixado nos cinco notebooks,
pelas mesmas funções do notebook 00 (`prepare`, `verify-revisions`, `gate`,
`export`). Os SHA256 dos arquivos internos (`MANIFESTO_SHA256.json`) coincidem
com a tabela da seção 8 de `PROTOCOLO_v4.md`. O SHA256 do ZIP depende das
datas dos arquivos e muda se o pacote for gerado de novo; o conteúdo não muda.

---

## Title

Formato de supervisão e acurácia em aritmética de múltiplas etapas: estudo prospectivo v4 (SFT, Qwen2.5-1.5B-Instruct)

## Description

Este registro inaugura o estudo `estudo-prospectivo-2026-v4`, um estudo novo
do projeto cot-obfuscation-h2a. Ele não continua o estudo v3.2
(https://osf.io/sjwdf/), encerrado sem treinamentos porque os modelos-base não
atingiram o critério de competência pré-registrado. O v4 testa uma única
hipótese confirmatória (H1): se o treinamento supervisionado com respostas
diretas, em comparação com rastros explícitos de cálculo, reduz mais a
acurácia em expressões de múltiplas etapas (nível C, a + b·c − d) do que em
somas de dois termos (nível A). O desenho usa Qwen2.5-1.5B-Instruct, três
condições de supervisão (explicit, direct, generic), dois níveis e 20 sementes
de treinamento, num total de 120 treinamentos e 26.400 gerações TEST. As
tarefas, instruções e faixas de operandos foram calibradas em itens próprios,
excluídos das partições do estudo; a triagem de competência foi executada uma
vez, antes deste registro, e aprovou os dois níveis. O braço de otimização por
recompensa (GRPO) e a hipótese H2 foram excluídos antes deste registro porque
nenhum monitor avaliado atingiu os critérios de discriminação. Protocolo,
dados, código e resultados da triagem estão congelados por SHA256 nos arquivos
anexos e no repositório https://github.com/Kelvin-Marques-Cyber/cot-obfuscation-h2a
(commit b858d128bcd545dd958e0dbed9978a7aea2591b3).

English summary. This registration starts a new study (v4) that tests one
confirmatory hypothesis: whether supervised fine-tuning on direct answers,
compared with explicit calculation traces, costs more accuracy on multi-step
arithmetic (a + b·c − d) than on two-term addition. Qwen2.5-1.5B-Instruct,
three supervision formats, two levels, 20 training seeds (120 fine-tuning runs,
26,400 test generations). The reward-optimization arm and its monitorability
hypothesis were excluded before registration because no evaluated monitor met
the discrimination criteria. All materials are frozen by SHA256.

---

## Research questions or hypotheses (344-2)

**H1 — Formato de supervisão e acurácia (SFT, Qwen2.5-1.5B-Instruct).**
O treinamento supervisionado com respostas diretas, em comparação com rastros
explícitos de cálculo, reduz mais a acurácia em tarefas de múltiplas etapas
(nível C) do que em tarefas de uma etapa (nível A)?

Para cada semente de treinamento s:

D1(s) = [Acc(direct, C, s) − Acc(explicit, C, s)] − [Acc(direct, A, s) − Acc(explicit, A, s)].

A direção prevista é D1 < 0. O contraste mede o efeito do alvo de treinamento
sobre a acurácia observável; não mede intenção de ocultar raciocínio.

**H2 — não testada neste estudo.** A hipótese sobre pressão de recompensa
(GRPO) foi declarada não testável antes deste registro: nenhum dos sete
candidatos a monitor independente nem dos dois candidatos a monitor de
recompensa atingiu os critérios em 320 controles construídos (decisão gravada
no commit ad6b434, antes do resultado). Os 120 treinos GRPO não serão
executados. A triagem do modelo do braço GRPO é relatada só como descrição.

## Foreknowledge of data or evidence (344-4)

Marcar: **Data does not yet exist. No part of the data that will be used for
this analysis plan exists, and no part will be generated until after this plan
is registered.**

## Explanation of foreknowledge and managing unintended influences (344-14)

Os dados da análise de H1 são as respostas, no conjunto TEST, dos 120
adaptadores treinados com a configuração registrada. Nenhum desses adaptadores
existe. Os autores conhecem, porém, as seguintes evidências e materiais:

1. **Estudo-piloto e estudo v3.2.** O piloto (outras tarefas e corretor) e o
   encerramento do v3.2, com seu diagnóstico, estão publicados no repositório.
   Motivaram as mudanças do v4, descritas em `PROTOCOLO_v4.md`, seção 3.
2. **Itens sintéticos.** Os enunciados e gabaritos das partições
   train/dev/test são gerados por código determinístico e estão congelados por
   SHA256 (SFT: f7ee4224…). Gerar um item não produz dado de desfecho.
3. **Calibração.** Instruções, faixas de operandos e monitores foram
   calibrados em itens próprios (sementes 20261028–20261030), excluídos das
   partições do estudo, com regras gravadas no repositório antes de cada
   rodada.
4. **Triagem DEV.** O modelo-base respondeu uma vez a 120 itens DEV por nível
   (SFT A: 117/120; C: 82/120), na Tesla T4, em 29/10/2026, depois do congelamento do
   protocolo (commit 37094c4, 11h28 UTC) e antes deste registro. Os itens DEV
   não entram na análise de H1.
5. **Testes técnicos de GPU (exposição parcial do TEST).** Para verificar o
   software na GPU, os notebooks foram executados em modo técnico, na Tesla T4, nos commits
   6c41277 e 4b7d1fd: uma semente (113), as seis combinações de condição e
   nível, 2 passos de treino por adaptador e avaliação nos **4 primeiros
   itens TEST de cada nível** (8 itens no total). Esses adaptadores não
   pertencem ao estudo (pasta separada, 2 passos em vez de 120) e as
   respostas, publicadas em `experimento_cientifico_v4/validacao/`, não foram
   usadas em nenhuma decisão de desenho; o protocolo já estava congelado.
   Para tornar essa exposição verificável, a análise secundária S1 (campo
   "Other planned analysis") repete o contraste H1 sem esses 8 itens.

O código que treina, avalia e analisa está congelado por SHA256 e o runtime
recusa executar se algum arquivo divergir. O conjunto TEST só pode ser
avaliado depois que os 120 treinamentos estiverem concluídos, e a análise
confirmatória é calculada pelo código congelado, sem escolhas posteriores.

## Study type (344-17)

Marcar: **Simulation study** e **Non-randomized study** (as mesmas opções do
v3.2).

## Intention for causal interpretation (344-27)

Marcar: **Direct inference on causal relationship(s)**.

## Blinding of experimental treatments (344-32)

Marcar: **No blinding is involved.**

## Additional blinding during research or analysis (344-38)

Não há avaliação humana no v4. A correção de cada resposta é automática, por
regras de formato e de igualdade com o gabarito congeladas no código
(`cotv4/contrato.py`), o que elimina julgamento de codificadores. A estatística
confirmatória é calculada pelo código congelado (`cotv4/estatistica.py`,
`cotv4/runtime.py`), e a avaliação TEST fica bloqueada até que os 120
treinamentos estejam completos, de modo que nenhum resultado TEST parcial pode
orientar treinos posteriores.

## Study design (344-40)

Experimento computacional fatorial 3 × 2 com 20 réplicas pareadas.

- Modelo-base: Qwen/Qwen2.5-1.5B-Instruct, revisão
  989aa7980e4cf806f80c7fef2b1adb7bc71aa306.
- Fator condição (alvo de treinamento): **explicit** (rastro de cálculo
  seguido de `<answer>N</answer>`), **direct** (somente `<answer>N</answer>`) e
  **generic** (frase genérica fixa seguida de `<answer>N</answer>`).
- Fator nível: **A**, soma a + b com operandos de 100 a 999; **C**,
  a + b·c − d com a e d de 100 a 999 e b e c de 12 a 99.
- Sementes de treinamento: 113 a 132, as mesmas em todas as combinações.
- Itens: 220 por partição (train, dev, test) e por nível; cada adaptador é
  treinado nos 220 itens train do seu nível e avaliado nos 220 itens TEST do
  mesmo nível.
- Treinamento: ajuste fino supervisionado com LoRA (r = 8, α = 16), fp16,
  lote 1 com acumulação de gradiente 4, 120 passos, taxa de aprendizagem
  5 × 10⁻⁵, comprimento máximo 512, checkpoints a cada 20 passos.
- Avaliação: modo `full` (adaptador carregado), decodificação gulosa, item a
  item, orçamento de 256 tokens, mesmo prompt de sistema do treino.
- Total: 120 treinamentos e 26.400 gerações TEST.

O braço GRPO do desenho do v3.2 foi excluído antes deste registro (ver H2).

## Randomization (344-44)

Não há atribuição aleatória: todas as combinações de condição, nível e semente
são executadas. A variabilidade estocástica do treinamento é controlada pelas
20 sementes predefinidas, usadas de forma pareada entre condições. As mesmas
sementes e os mesmos itens TEST permitem comparações pareadas dentro de cada
semente. A ordem operacional dos treinos (nível, semente, condição) é fixada
pelo código e não é variável de interesse.

## Data collection procedures (344-47)

Os treinamentos e as avaliações são executados no Google Colab pelos
notebooks 01 e 03 do v4, ou por um executor que chama as mesmas etapas de
`cotv4.runtime` (`train` e `evaluate`), sempre com o código do commit
b858d128bcd545dd958e0dbed9978a7aea2591b3. O runtime confere o SHA256 do
protocolo, do código e dos dados antes de cada etapa e recusa executar se algo
divergir.

**Classe de GPU.** Todos os 120 treinamentos e as 120 avaliações TEST usam a
mesma classe de GPU, NVIDIA H100, em fp16 (a precisão do protocolo; bf16 não
é usado). A triagem DEV foi executada na Tesla T4 e não é repetida na H100.
Classes diferentes de GPU não produzem resultados numericamente idênticos, e
algumas respostas gulosas podem mudar; por isso a classe é única para todo o
estudo e todas as condições, níveis e sementes são comparadas na mesma
classe. Se a H100 ficar indisponível, a coleta aguarda; não há troca de classe
no meio do estudo. O nome da GPU de cada treinamento fica gravado no seu
manifesto; um treinamento ou uma avaliação executados em outra classe é falha
de integridade, tratada como descrito em "Data inclusion and exclusion".

Cada treinamento grava configuração, revisão do modelo, versões das
bibliotecas, GPU, perdas e SHA256 do adaptador. Interrupções são retomadas do
último checkpoint com os mesmos parâmetros; uma avaliação interrompida
descarta linhas parciais e continua sem duplicar itens. Os resultados ficam em
armazenamento persistente, fora da máquina virtual.

A coleta começa depois deste registro: o runtime só libera os treinos quando a
URL pública do registro é informada, e recusa a declaração se já existir
algum treino no estudo.

## Sample size (344-51)

120 treinamentos (3 condições × 2 níveis × 20 sementes) e 26.400 gerações TEST
(120 adaptadores × 220 itens), com 440 itens TEST distintos. A unidade de
replicação estatística é a semente de treinamento: 20 contrastes pareados D1(s).
Gerações de um mesmo adaptador não são tratadas como réplicas independentes.

## Sample size rationale (344-53)

O número de sementes (20) é o do v3.2, mantido sem alteração. Com os mesmos
parâmetros de planejamento do v3.2 (efeito mínimo relevante de 0,05 em escala
de proporção, desvio-padrão presumido de 0,07 entre sementes, potência nominal
de 80%) e uma única hipótese confirmatória (α bilateral de 0,05), a
aproximação normal

n ≈ [(z(0,975) + z(0,80)) × 0,07 / 0,05]²

resulta em cerca de 15,4 sementes; com 20 sementes, a potência nominal do
teste de significância para um efeito de 0,05 é de cerca de 89%. O
desvio-padrão de 0,07 é uma hipótese de planejamento, não uma estimativa de
dados. O critério de apoio exige também média observada de pelo menos 0,05 em
módulo; se o efeito verdadeiro for exatamente 0,05, essa exigência é atendida
em cerca de metade das repetições, de modo que a probabilidade de apoio nesse
caso-limite é menor que a potência do teste. A amostra não será ampliada nem
reduzida em função de resultados.

## Starting and stopping rules (344-55)

A coleta começa após este registro e a declaração da sua URL no estudo. Os 120
treinamentos são executados primeiro; a avaliação TEST só é liberada pelo
runtime quando todos estiverem concluídos. A coleta termina com os 120
treinamentos e as 26.400 gerações TEST completos e verificados. Não há análise
intermediária nem parada por resultado. Falhas de infraestrutura apenas
suspendem a coleta, que é retomada dos checkpoints. Se a integridade não puder
ser restaurada, o resultado é INSUFFICIENT_EVIDENCE, com relato das falhas.

## Manipulated variables (344-58)

- `condition`: explicit, direct, generic (alvo de treinamento).
- `level`: A, C (complexidade da tarefa).
- `train_seed`: 113 a 132 (réplica).

## Measured variables (344-62)

Por geração TEST: `accuracy` (1 se a resposta contém um único número
inequívoco segundo o contrato de formato e ele é igual ao gabarito; 0 caso
contrário), `format_valid` (1 se a resposta segue o contrato: uma única tag
`<answer>` no fim, uma única linha final "Resposta: N" ou somente o número),
`finish_reason` (`length` indica
truncamento no orçamento de 256 tokens), `output_tokens` e identificadores
(`item_id`, `run_id`, condição, nível, semente, revisão do modelo). O contrato
rejeita números com separador de milhar como formato ambíguo.

## Indices (344-66)

Acc(condição, nível, s) = proporção de respostas corretas nos 220 itens TEST do
nível, para o adaptador da semente s. Contraste por semente:
D1(s) = [Acc(direct, C, s) − Acc(explicit, C, s)] − [Acc(direct, A, s) − Acc(explicit, A, s)],
calculado com pareamento item a item dentro de cada nível. Estimativa
principal: média dos 20 valores de D1(s).

## Statistical models (344-71)

Teste bilateral de inversão de sinais sobre a média dos 20 contrastes D1(s),
com 199.999 permutações de Monte Carlo e semente fixa 20261015; intervalo de
confiança de 95% por *bootstrap* percentil, reamostrando sementes (20.000
reamostragens, semente 20261015); d_z de Cohen como tamanho de efeito pareado.
Não há teste omnibus nem ANOVA. A condição generic não entra no teste
confirmatório. Implementação: `cotv4/estatistica.py` e `final_report` em
`cotv4/runtime.py` (anexos).

## Transformations (344-75)

Nenhuma. Acurácias são proporções na escala de 0 a 1, agregadas por condição,
nível e semente; o contraste é calculado nessa escala. Não há padronização,
transformação logarítmica nem recodificação posterior.

## Inference criteria (344-77)

Há uma única família confirmatória (H1); a correção de Holm com α = 0,05
equivale, nesse caso, a comparar o valor-p com 0,05. H1 é apoiada
(`SUPPORT_FOR_SFT_CONTRAST`) somente se, ao mesmo tempo:

- o valor-p da inversão de sinais for ≤ 0,05;
- a média de D1(s) for ≤ −0,05;
- o limite superior do intervalo de 95% for < 0;
- todos os requisitos estiverem aprovados: registro público anterior aos
  treinos, triagem DEV aprovada e integridade completa (120 treinamentos,
  26.400 gerações pareadas, sem duplicação nem troca de conjunto).

Se algum requisito falhar, o veredito é `INSUFFICIENT_EVIDENCE`. Se os
requisitos forem cumpridos sem apoio, o veredito é
`NO_STATISTICAL_SUPPORT_FOR_PRESPECIFIED_CONTRAST`, que não demonstra
equivalência nem ausência do fenômeno. Nenhum resultado é interpretado como
prova de intenção ou de estados internos do modelo.

## Data inclusion and exclusion (344-79)

Entram na análise todas as gerações TEST, modo `full`, dos 120 adaptadores
registrados, nos 220 itens do nível correspondente. Nenhuma geração é excluída
por acurácia, formato, truncamento ou valor extremo: respostas fora do
contrato contam como incorretas. Duplicações, trocas de conjunto, divergência
de hash ou de revisão do modelo e execução em classe de GPU diferente da
registrada são falhas de integridade, corrigidas apenas
por retomada ou repetição da mesma execução com os mesmos parâmetros; se não
puderem ser corrigidas, o veredito é `INSUFFICIENT_EVIDENCE`.

## Missing data (344-81)

Não há imputação. A análise exige cobertura completa: cada adaptador com os 220
itens TEST do seu nível, sem duplicação. Execuções interrompidas são
retomadas dos checkpoints com os mesmos parâmetros. Se a cobertura completa não
puder ser obtida, o veredito é `INSUFFICIENT_EVIDENCE`, com relato da
frequência e das causas das ausências.

## Other planned analysis (344-83)

Análises secundárias, não confirmatórias, implementadas em
`analises_secundarias.py` (SHA256 6065d011…, anexo), com os mesmos
procedimentos estatísticos e sem correção para multiplicidade:

- **S1 — sensibilidade à exposição técnica:** contraste D1 sem os 4 primeiros
  itens TEST de cada nível (os usados nos testes técnicos de GPU), isto é, com
  216 itens por nível.
- **S2 — acurácia descritiva** por nível e condição (média e desvio-padrão
  entre sementes).
- **S3 — condição generic:** contrastes de mesma forma que D1 para
  generic − explicit e direct − generic.
- **S4 — formato e truncamento:** proporções de formato válido e de
  truncamento por nível e condição.

Qualquer outra análise será identificada como exploratória no relato.

## Context and additional information (344-86)

O v4 decorre do encerramento do v3.2 (https://osf.io/sjwdf/), cuja triagem de
competência reprovou os quatro níveis; o diagnóstico reproduziu os valores com
o código congelado e atribuiu a falha à incapacidade dos modelos nas tarefas
mais difíceis e a instruções que tornavam a medida dependente de formatos não
demonstrados. O v4 recalibrou tarefas e instruções em itens próprios e
executou a triagem uma vez, antes deste registro. Documentos:
`experimento_cientifico_v4/PROTOCOLO_v4.md` (protocolo e resumos SHA256),
`experimento_cientifico_v4/calibracao/` (regras e resultados da calibração),
`experimento_cientifico_v4/triagem_dev/` (triagem) e
`experimento_cientifico_v3/ENCERRAMENTO_v3_2.md` (encerramento do v3.2), no
repositório https://github.com/Kelvin-Marques-Cyber/cot-obfuscation-h2a.
