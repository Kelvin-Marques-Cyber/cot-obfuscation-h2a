# Verificação declarada no desvio 1 (04/11/2026, 17:13 UTC)

Coleta retomada às 16:54 UTC numa pasta nova (mesmo `STUDY_ID`), com os notebooks do commit
`15f1bfa` e o código `9066d81`; o protocolo congelado da pasta nova registra `treino.py` com
SHA256 `a2d431c0…`, e o registro hb64a foi declarado antes do primeiro treino.

| Treino | Status | Adaptador comparado com a 1ª tentativa | Passos descartados pelo fp16 | Perda inicial → final |
|---|---|---|---|---|
| A · 113 · explicit | 120 passos | idêntico (`f773a412…`) | nenhum | 0,6734 → 0,0001 |
| A · 113 · direct | 120 passos | idêntico (`fb99ba0d…`) | nenhum | 0,5259 → 0,0001 |
| A · 113 · generic | 120 passos | idêntico (`5c0fc965…`) | nenhum | 3,0818 → 0,0002 |
| A · 114 · explicit | 120 passos | idêntico (`2b300444…`) | nenhum | 0,6720 → 0,0001 |
| A · 114 · direct | 120 passos | idêntico | nenhum | 0,5257 → 0,0001 |
| A · 114 · generic | 120 passos | sem referência (falhou na 1ª tentativa) | passo 15 | 3,0794 → 0,0002 |

Os cinco treinos concluídos na primeira tentativa reproduziram exatamente os mesmos
adaptadores: a correção do guarda não alterou nenhum treino em que o guarda original não
disparou. O treino que falhava terminou os 120 passos com um único passo descartado, o passo 15,
em que a primeira tentativa havia parado. Os SHA256 completos da primeira tentativa estão no
Drive do estudo, em `prospectivo_v4/tentativa1_adaptadores.json`.
