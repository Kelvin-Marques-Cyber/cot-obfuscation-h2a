# Desvio 1 do v4: passos descartados pelo fp16 no treino

**Declarado em 04/11/2026**, depois do registro público (https://osf.io/hb64a/, 13:01 UTC) e
de 6 dos 120 treinamentos, **antes de qualquer geração TEST** e antes de retomar a coleta.

## O que aconteceu

O treino 6 (nível A, condição `generic`, semente 114) parou no passo 15 com
`FloatingPointError: grad_norm não finito no passo 15: nan`, levantado pelo guarda `FiniteLoss`
de `cotv4/treino.py` (código registrado, commit `b858d12`). As perdas dos passos 5 e 10 eram
finitas (3,0794 e 2,2069). Repetido com os mesmos parâmetros, como prevê o registro, o treino
reproduziu as mesmas perdas e o mesmo `nan` no passo 15: na A100 o treino é determinístico e a
repetição não corrige a falha. Pelo texto registrado, a coleta terminaria com
`INSUFFICIENT_EVIDENCE`. Evidência: `validacao/coleta_falha_treino6/`.

Em fp16, o escalonador de gradientes do PyTorch descarta o passo em que o gradiente estoura e
reduz a escala da perda; a norma registrada nesse passo é `nan`, e os parâmetros não são
alterados. O guarda original interrompia o treino sempre que isso coincidia com um passo de
registro (a cada 5 passos), embora a perda estivesse finita.

## A mudança

Somente o guarda de valores não finitos de `cotv4/treino.py` (commit
`9066d811e656cb7c0d3c2120ba06c7ae356691da`; teste em `tests/test_guarda_fp16.py`):

1. um passo descartado pelo otimizador em fp16 (`step_was_skipped`) é gravado em
   `passos_descartados.json` da execução e não interrompe o treino;
2. continua interrompendo o treino: perda não finita em qualquer passo registrado; norma não
   finita num passo que não foi descartado; treino em fp32 com norma não finita; mais de 10% dos
   passos descartados (mais de 12 de 120); adaptador final com algum peso não finito;
3. os passos descartados de cada treino entram no `status.json` (`skipped_fp16_steps`).

Nada mais muda: hipótese, dados (SHA256 `f7ee4224…`), sementes, configuração de treino,
avaliação, contrato de correção, estatística, critérios de inferência, classe de GPU e os módulos
da triagem (`avaliacao.py`, `contrato.py`, `modelos.py`, `tarefas.py`, com os mesmos hashes). O
guarda só observa o treino; num treino em que o guarda original não disparou, o resultado é o
mesmo com qualquer das duas versões.

## Consequências operacionais, fixadas antes de retomar

- O protocolo congelado inclui os hashes do código, então o estudo recomeça do zero numa pasta
  nova com o mesmo `STUDY_ID`. A pasta da primeira tentativa é preservada no Drive como
  `estudo-prospectivo-2026-v4-tentativa1-b858d12`, sem nenhuma alteração.
- Os notebooks passam a fixar o código `9066d81` (notebooks no commit seguinte). O registro hb64a
  é declarado de novo na pasta nova, que não tem nenhum treino; o resumo do protocolo gravado
  nessa declaração difere do pacote anexado ao registro exatamente pelo hash de `treino.py`.
- **Verificação:** os 5 treinos concluídos na primeira tentativa são refeitos. Seus adaptadores
  devem ter os mesmos SHA256 gravados na pasta arquivada; o resultado da comparação é relatado
  qualquer que seja. Se algum diferir, valem os adaptadores da pasta nova e a diferença é
  relatada como não determinismo.
- **Análise de sensibilidade S5 (exploratória, declarada agora):** o contraste D1 recalculado só
  com as sementes em que nenhum dos seis adaptadores teve passo descartado, com os mesmos
  procedimentos estatísticos; o número de passos descartados por treino é relatado por condição
  e nível.
- O veredito confirmatório continua sendo o do `final_report` do runtime, relatado como
  "teste confirmatório com um desvio declarado (desvio 1)".

## Texto para a atualização do registro no OSF (hb64a → Updates)

**Justificativa (aba Justification):**

> Desvio 1, declarado em 04/11/2026 antes de qualquer geração TEST. O treino 6 de 120 (nível A,
> generic, semente 114) parou no passo 15 porque o guarda do código registrado interrompe o treino
> quando a norma do gradiente registrada é nan. Em fp16 isso ocorre quando o escalonador de
> gradientes descarta um passo por estouro; a perda era finita. A repetição com os mesmos
> parâmetros, prevista no registro, reproduziu a falha de forma idêntica (treino determinístico).
> A única mudança é no guarda (commit 9066d811e656cb7c0d3c2120ba06c7ae356691da): passos descartados
> pelo fp16 são registrados e não interrompem o treino; continuam interrompendo perda não finita,
> norma não finita sem descarte, mais de 10% de passos descartados e adaptador com pesos não
> finitos. Hipótese, dados, sementes, configuração, avaliação, estatística e critérios não mudam.
> O estudo recomeça numa pasta nova; os 5 treinos já feitos são refeitos e devem reproduzir os
> mesmos adaptadores. Detalhes e evidência: experimento_cientifico_v4/DESVIO_01.md no repositório
> https://github.com/Kelvin-Marques-Cyber/cot-obfuscation-h2a.

**Campo "Data collection procedures":** substituir a frase "sempre com o código do commit
b858d128bcd545dd958e0dbed9978a7aea2591b3." por:

> com o código do commit b858d128bcd545dd958e0dbed9978a7aea2591b3, alterado somente no guarda de
> valores não finitos do treino pelo desvio 1 (commit 9066d811e656cb7c0d3c2120ba06c7ae356691da;
> ver DESVIO_01.md no repositório).
