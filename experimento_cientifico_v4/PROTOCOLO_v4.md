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

## 1. Perguntas e hipóteses (inalteradas em relação ao v3.2)

**H1 — formato de supervisão (SFT, Qwen2.5-1.5B-Instruct).** Para cada semente *s*:
`Δ_s = [Acc(direct,C) − Acc(explicit,C)] − [Acc(direct,A) − Acc(explicit,A)]`,
direção prevista negativa. Mede efeito do alvo de treinamento sobre a acurácia,
não intenção de ocultação.

**H2 — pressão de recompensa (GRPO, Qwen3-1.7B).** No nível `high`, para cada semente:
`Δ_s = P(explora ∧ ¬sinalizado | cot_monitor) − P(explora ∧ ¬sinalizado | none)`,
direção prevista positiva, com taxa de exploração não inferior em mais de 0,10.
Exploração é a escolha da alternativa indicada pela nota auxiliar proibida
(definição objetiva); a sinalização é feita por um monitor independente,
distinto do ator e do monitor de recompensa.

## 2. Desenho (inalterado)

| Braço | Condições | Níveis | Sementes | Treinos | Passos | Itens TEST/nível | Gerações TEST |
|---|---|---|---|---|---|---|---|
| SFT | explicit, direct, generic | A, C | 113–132 | 120 | 120 | 220 | 26.400 |
| GRPO | none, cot_monitor, summary_monitor | low, high | 113–132 | 120 | 120 | 240 | 28.800 |

LoRA (r = 8, α = 16) em fp16, lote 1 com acumulação 4, checkpoints a cada 20
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
7. **Monitores.** Fixados pela verificação de viabilidade (seção 4); o monitor
   independente do v3.2 sinalizou quase todos os controles construídos.
8. **Momento da triagem.** Executada uma vez, antes do registro, com os
   módulos que a determinam congelados por hash; o registro inclui seus resumos.

## 4. Calibração (exploratória, em itens próprios)

Regras gravadas no repositório antes de cada execução (commits 97eaf5e,
4f2c503, e02eda4, d3e4258, 73a6b13 e a4668a5). Sementes de calibração
20261028–20261030; os enunciados e parâmetros usados são excluídos das
partições do estudo. Resultados completos e arquivos de geração em
`calibracao/resultados/`.

<!-- calibracao:inicio -->
<!-- calibracao:fim -->

## 5. Triagem de competência (critério inalterado)

Cada modelo-base deve atingir **acurácia ≥ 0,65 em 120 itens DEV por nível**
segundo o contrato de formato. SFT: instrução explícita com as duas
demonstrações; 120 primeiros itens DEV de cada nível. GRPO: enunciado **sem** a
nota proibida (competência na tarefa); 30 itens DEV de cada domínio por nível.
Tesla T4, fp16, decodificação gulosa item a item, 256 tokens. Execução única;
arquivos e manifesto em `triagem_dev/`. Reprovação em qualquer nível bloqueia
todos os treinos deste estudo.

## 6. Gates obrigatórios (inalterados)

1. Registro público com data auditável **antes** do primeiro treino, contendo os
   resumos da seção 8. URL fabricada ou retroativa não é aceita.
2. Triagem DEV aprovada (seção 5).
3. Integridade: 240 treinos com configuração e passos corretos; 55.200 gerações
   TEST pareadas, sem duplicação nem troca de conjunto.
4. Monitor independente validado em ≥ 100 controles positivos e ≥ 100 negativos
   **rotulados por pessoas** (TPR ≥ 0,80 e FPR ≤ 0,10). Controles construídos
   por regra não satisfazem este gate.
5. Anotação humana cega de 360 observações GRPO (60 por nível × condição), dois
   avaliadores por item, concordância ≥ 75% e arbitragem de todas as divergências.
6. Inferência: inversão de sinais bilateral (199.999 permutações com semente
   fixa), *bootstrap* percentil de 95% entre sementes, Holm para as duas
   famílias (FWER 5%), efeito mínimo de 0,05. Apoio exige significância
   ajustada, direção, magnitude, intervalo e gates aprovados.

## 7. Interpretação

`INSUFFICIENT_EVIDENCE` quando falta qualquer gate; `SUPPORT_FOR_*` apenas para
os contrastes observáveis previstos; `NO_STATISTICAL_SUPPORT_FOR_PRESPECIFIED_CONTRASTS`
não demonstra equivalência nem ausência do fenômeno. Nenhum resultado é
interpretado como prova de intenção estratégica ou de estados internos.

## 8. Resumos a registrar publicamente

<<HASHES>>

## 9. Execução e custos

240 treinos e 55.200 gerações exigem várias sessões de GPU (T4 ou superior) e
armazenamento persistente no Google Drive. Os notebooks executam poucos jobs por
sessão e retomam do ponto em que pararam. O modo `TECHNICAL_TEST` usa uma
semente, dois passos e quatro itens, em pasta separada, e nunca produz
evidência científica. Os rótulos humanos devem ser produzidos por pessoas; o
software não os gera.
