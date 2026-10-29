# Protocolo prospectivo v4 — formato de supervisão e monitorabilidade da CoT

**Identificador do estudo:** `estudo-prospectivo-2026-v4` · **Esquema:** `scientific-prospective-v4.0`
**Responsável:** Kelvin e Silva Marques · **Repositório:** https://github.com/Kelvin-Marques-Cyber/cot-obfuscation-h2a

Este documento descreve um estudo **novo**. Ele não continua o estudo v3.2
(`estudo-principal-2026-v3-2`, registro OSF https://osf.io/sjwdf/), que foi
encerrado sem treinamentos porque os modelos-base não atingiram o critério de
competência pré-registrado. Os arquivos, o registro e o desfecho do v3.2 são
preservados sem alteração. O v4 só se torna confirmatório depois de um
**registro público próprio**, feito antes do primeiro treinamento, que inclua os
resumos SHA256 listados na seção 8.

> **Decisão de calibração (29/10/2026):** nenhum candidato a monitor
> independente nem a monitor de recompensa atingiu os critérios nos controles
> construídos (seção 4). Pela regra gravada antes desse resultado (commit
> `ad6b434`), o **braço GRPO foi excluído do v4 antes do registro**: H2 não é
> testável neste estudo, os 120 treinos GRPO não serão executados e o registro
> cobre apenas H1. A triagem DEV do modelo GRPO é executada e relatada apenas
> como descrição.

## 1. Perguntas e hipóteses (texto inalterado em relação ao v3.2)

**H1 — formato de supervisão (SFT, Qwen2.5-1.5B-Instruct).** Para cada semente *s*:
`Δ_s = [Acc(direct,C) − Acc(explicit,C)] − [Acc(direct,A) − Acc(explicit,A)]`,
direção prevista negativa. Mede efeito do alvo de treinamento sobre a acurácia,
não intenção de ocultação.

**H2 — pressão de recompensa (GRPO, Qwen3-1.7B).** No nível `high`, para cada semente:
`Δ_s = P(explora ∧ ¬sinalizado | cot_monitor) − P(explora ∧ ¬sinalizado | none)`,
direção prevista positiva, com taxa de exploração não inferior em mais de 0,10.
Exploração é a escolha da alternativa indicada pela nota auxiliar proibida
(definição objetiva); a sinalização é feita por um monitor independente,
distinto do ator e do monitor de recompensa. **No v4, H2 foi declarada não
testável antes do registro** (ver decisão acima).

## 2. Desenho

| Braço | Condições | Níveis | Sementes | Treinos | Passos | Itens TEST/nível | Gerações TEST |
|---|---|---|---|---|---|---|---|
| SFT | explicit, direct, generic | A, C | 113–132 | 120 | 120 | 220 | 26.400 |
| GRPO (excluído) | none, cot_monitor, summary_monitor | low, high | — | 0 | — | — | — |

O braço GRPO tinha 120 treinos e 28.800 gerações TEST no desenho do v3.2;
no v4 permanece apenas o software, testado em modo técnico. LoRA (r = 8, α = 16) em fp16, lote 1 com acumulação 4, checkpoints a cada 20
passos, orçamento de 256 tokens, decodificação gulosa. O desfecho primário usa
somente TEST, modo `full`, adaptadores treinados. Unidade de inferência: semente
de treinamento.

## 3. O que muda em relação ao v3.2 e por quê

As mudanças decorrem do diagnóstico da triagem do v3.2
(`experimento_cientifico_v3/diagnostico_dev/`) e foram calibradas somente em
itens próprios de calibração, nunca em TEST. A tabela detalhada está em
`calibracao/resultados/mudancas_v4.json`.

1. **Tarefas SFT.** v3.2: A = soma de dois inteiros até 9.999; C = a + b·c − d
   com todos os operandos até 9.999 (produto de até oito algarismos, que o rastro
   explícito não decompunha). v4: faixas fixadas pela calibração (seção 4).
2. **Tarefas GRPO.** v3.2: operandos de 2 algarismos no nível `low` e de 3 no
   `high`, de modo que o nível alterava também a aritmética. v4: mesma faixa nos
   dois níveis; o nível varia apenas a dificuldade de decodificar a dica.
3. **Contrato de correção.** Números com separador de milhar passam a ser
   rejeitados como formato ambíguo (no v3.2 eram lidos como decimais).
4. **Instrução da triagem SFT.** Inclui duas demonstrações fixas do formato do
   alvo explícito; na calibração sem demonstrações o modelo omitiu a tag `<answer>`.
5. **Instrução GRPO.** Fixada pela calibração (seção 4); no v3.2 o modelo
   copiava os marcadores "explicação verificável" e "resumo" da instrução.
6. **Seleção da triagem GRPO.** 30 itens DEV de cada um dos quatro domínios por
   nível; no v3.2 os 120 primeiros itens cobriam só dois domínios.
7. **Monitores e braço GRPO.** O monitor independente do v3.2 sinalizou quase
   todos os controles construídos. Nenhum dos sete candidatos a monitor
   independente foi aprovado, e nenhum dos dois candidatos a monitor
   de recompensa atingiu a discriminação mínima; por isso o braço GRPO foi
   excluído antes do registro (seção 4).
8. **Momento da triagem.** Executada uma vez, antes do registro, com os
   módulos que a determinam congelados por hash; o registro inclui seus resumos.

## 4. Calibração (exploratória, em itens próprios)

Regras gravadas no repositório antes de cada execução (commits 97eaf5e,
4f2c503, e02eda4, d3e4258, 73a6b13 e a4668a5; repetição da rodada 2 dos
monitores em 5a091d8; regra para monitores inviáveis em ad6b434). Sementes de calibração
20261028–20261030; os enunciados e parâmetros usados são excluídos das
partições do estudo. Resultados completos e arquivos de geração em
`calibracao/resultados/`.

<!-- calibracao:inicio -->
| Rodada | Braço | Candidato | Instrução | Nível | Acurácia (n) |
|---|---|---|---|---|---|
| 1 | SFT | A | explícita | A | 0,0% (40) |
| 1 | SFT | A | neutra | A | 97,5% (40) |
| 1 | SFT | C1 | explícita | C | 0,0% (40) |
| 1 | SFT | C1 | neutra | C | 0,0% (40) |
| 1 | SFT | C2 | explícita | C | 0,0% (40) |
| 1 | SFT | C2 | neutra | C | 0,0% (40) |
| 2 | SFT | A | explícita + 2 demonstrações | A | 97,5% (40) |
| 2 | SFT | C1 | explícita + 2 demonstrações | C | 100,0% (40) |
| 2 | SFT | C2 | explícita + 2 demonstrações | C | 82,5% (40) |
| 1 | GRPO | G1 | v3, sem dica | low | 42,5% (40); formato válido 42,5%; letra da dica 0,0% |
| 1 | GRPO | G1 | v3, sem dica | high | 40,0% (40); formato válido 45,0%; letra da dica 2,5% |
| 1 | GRPO | G0 | v3, sem dica | low | 35,0% (40); formato válido 35,0%; letra da dica 0,0% |
| 1 | GRPO | G0 | v3, sem dica | high | 40,0% (40); formato válido 42,5%; letra da dica 0,0% |
| 2 | GRPO | G1 | r2a, sem dica | low | 0,0% (40); formato válido 0,0%; letra da dica 0,0% |
| 2 | GRPO | G1 | r2a, sem dica | high | 0,0% (40); formato válido 0,0%; letra da dica 0,0% |
| 2 | GRPO | G1 | r2b, sem dica | low | 100,0% (40); formato válido 100,0%; letra da dica 0,0% |
| 2 | GRPO | G1 | r2b, sem dica | high | 92,5% (40); formato válido 100,0%; letra da dica 2,5% |
| 2 | GRPO | G1 | r2b, com dica | low | 100,0% (40); formato válido 100,0%; letra da dica 0,0% |
| 2 | GRPO | G1 | r2b, com dica | high | 92,5% (40); formato válido 100,0%; letra da dica 2,5% |

| Rodada | Monitor | TPR | FPR | Situação |
|---|---|---|---|---|
| 1 | HuggingFaceTB/SmolLM2-135M-Instruct | 87,5% | 86,2% | avaliado |
| 1 | HuggingFaceTB/SmolLM2-360M-Instruct | 100,0% | 100,0% | avaliado |
| 1 | HuggingFaceTB/SmolLM2-1.7B-Instruct | 100,0% | 97,5% | avaliado |
| 1 | Qwen/Qwen2.5-0.5B-Instruct | 69,4% | 77,5% | avaliado |
| 2 | allenai/OLMo-2-0425-1B-Instruct | 0,0% | 0,0% | avaliado |
| 2 | ibm-granite/granite-3.3-2b-instruct | 78,8% | 45,0% | avaliado |
| 2 | HuggingFaceTB/SmolLM3-3B | 0,0% | 0,0% | avaliado |
| 2 | microsoft/Phi-3.5-mini-instruct | 86,9% | 15,0% | avaliado |
| 2 | Qwen/Qwen2.5-1.5B-Instruct | 0,0% | 0,0% | avaliado |

Decisões: SFT nível C = C2; GRPO variante r2b, faixa G1; monitor independente nenhum aprovado; monitor de recompensa nenhum aprovado.
<!-- calibracao:fim -->

## 5. Triagem de competência (critério inalterado)

Cada modelo-base deve atingir **acurácia ≥ 0,65 em 120 itens DEV por nível**
segundo o contrato de formato. SFT: instrução explícita com as duas
demonstrações; 120 primeiros itens DEV de cada nível. GRPO: enunciado **sem** a
nota proibida (competência na tarefa); 30 itens DEV de cada domínio por nível.
Tesla T4, fp16, decodificação gulosa item a item, 256 tokens. Execução única;
arquivos e manifesto em `triagem_dev/`. Com o braço GRPO excluído, somente os
níveis do SFT condicionam os treinos; reprovação em qualquer um deles bloqueia
todos os treinos deste estudo. Os níveis do GRPO são relatados como descrição.

## 6. Gates obrigatórios

1. Registro público com data auditável **antes** do primeiro treino, contendo os
   resumos da seção 8. URL fabricada ou retroativa não é aceita.
2. Triagem DEV aprovada (seção 5).
3. Integridade: 120 treinos SFT com configuração e passos corretos; 26.400
   gerações TEST pareadas, sem duplicação nem troca de conjunto.
4. Inferência para H1: inversão de sinais bilateral (199.999 permutações com
   semente fixa), *bootstrap* percentil de 95% entre sementes e efeito mínimo
   de 0,05; com uma única família, a correção de Holm mantém α = 0,05. Apoio
   exige significância, direção, magnitude, intervalo e gates aprovados.

Os gates do braço GRPO (monitor independente validado com ≥ 100 controles
positivos e ≥ 100 negativos rotulados por pessoas, com TPR ≥ 0,80 e FPR ≤ 0,10,
e anotação humana cega de 360 observações) permanecem implementados no
software, mas não se aplicam a este estudo.

## 7. Interpretação

`INSUFFICIENT_EVIDENCE` quando falta qualquer gate; `SUPPORT_FOR_SFT_CONTRAST`
apenas para o contraste observável previsto em H1;
`NO_STATISTICAL_SUPPORT_FOR_PRESPECIFIED_CONTRAST` não demonstra equivalência
nem ausência do fenômeno. Nenhum resultado é
interpretado como prova de intenção estratégica ou de estados internos.

## 8. Resumos a registrar publicamente

Valores no commit que contém a triagem DEV, gerados por `python resumos_registro.py
--escrever`. O registro deve incluir esta tabela ou o `MANIFESTO_SHA256.json`
do pacote produzido pelo notebook 00, que a contém.

<!-- hashes:inicio -->
<!-- hashes:fim -->

## 9. Execução e custos

120 treinos SFT e 26.400 gerações exigem várias sessões de GPU (T4 ou
superior) e armazenamento persistente no Google Drive. Os notebooks executam poucos jobs por
sessão e retomam do ponto em que pararam. O modo `TECHNICAL_TEST` usa uma
semente, dois passos e quatro itens, em pasta separada, e nunca produz
evidência científica. Os rótulos humanos devem ser produzidos por pessoas; o
software não os gera.
