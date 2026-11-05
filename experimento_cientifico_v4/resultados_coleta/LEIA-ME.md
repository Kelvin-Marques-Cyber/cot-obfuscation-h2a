# Resultados da coleta confirmatória do v4

Coleta do estudo registrado em https://osf.io/hb64a/ (DOI 10.17605/OSF.IO/HB64A), com o desvio 1
(`../DESVIO_01.md`), executada de 04/11/2026 16:54 UTC a 05/11/2026 13:21 UTC em NVIDIA
A100-SXM4-80GB do Colab (VMs `v4d` e, depois da interrupção descrita em
`../validacao/coleta_interrupcao_vm/`, `v4f`), pelos notebooks do commit `15f1bfa` (código
`9066d81`), acionados por `../validacao/executores_colab/coleta_v4.py`.

## Veredito confirmatório (`final_verdict.json`, gerado pelo notebook 04)

**`SUPPORT_FOR_SFT_CONTRAST`**, sem pendências (`blocked_by` vazio): 120 treinos, 20 sementes,
26.400 gerações TEST pareadas.

| H1: D1 = (direct − explicit)_C − (direct − explicit)_A | Valor |
|---|---|
| Média entre as 20 sementes | −73,3 p.p. (−0,7325) |
| IC de 95% (bootstrap das sementes) | −73,5 a −73,0 p.p. |
| Valor-p (inversão de sinais, 199.999 permutações) | 5 × 10⁻⁶ (o mínimo possível) |
| d_z | −110,3 |
| Sementes com D1 < 0 | 20 de 20 (de −74,5 a −72,3 p.p.) |
| Critério registrado | Holm rejeita, média ≤ −5 p.p. e IC < 0: `SUPPORTED_FORMAT_COST` |

## Análises secundárias registradas (`analises_secundarias.json`, não confirmatórias)

Acurácia TEST por nível e alvo (S2; média entre sementes, DP entre parênteses):

| Nível | Rastro explícito | Resposta direta | Preâmbulo genérico |
|---|---|---|---|
| A | 98,1% (0,2) | 95,5% (0,1) | 94,8% (0,3) |
| C | 75,9% (0,6) | 0,0% (0,0) | 0,4% (0,2) |

- S1, sem os 4 itens por nível usados nos testes técnicos (216 itens por nível): −74,5 p.p.
  (IC −74,7 a −74,3).
- S3: generic − explicit, −72,2 p.p. (IC −72,5 a −71,9); direct − generic, −1,1 p.p. (IC −1,2 a −0,9).
- S4: formato válido em 100% das respostas de todas as células; nenhuma truncada.

## Análise exploratória do desvio 1 (`analise_desvio_01.json`)

- Passos descartados pelo fp16: 4 dos 120 treinos, um passo em cada (A · generic · 114, passo 15;
  A · direct · 118, passo 41; C · generic · 123, passo 17; C · generic · 130, passo 17).
- S5, só com as 16 sementes sem nenhum passo descartado: −73,2 p.p. (IC −73,5 a −72,9),
  p = 3,1 × 10⁻⁵ (o mínimo possível com 16 sementes, por enumeração exata).

## Leitura

No nível C, os modelos treinados para responder diretamente, ou com o preâmbulo genérico, não
acertam praticamente nenhum item: escrevem um número no formato exigido, mas errado. Com o rastro explícito, escrevem as contas e acertam 75,9%. No nível A, a
diferença entre os alvos é pequena. O contraste de H1 é, portanto, quase todo o desempenho do
rastro explícito no nível C. O resultado mostra que o formato de supervisão custa acurácia
quando a tarefa exige etapas escritas; não mostra ofuscação (`strategic_intent_proven`: false),
e, como as respostas diretas estão erradas, não há indício de que o cálculo tenha sido feito sem
ser escrito.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `final_verdict.json` | relatório final do runtime (`final_report`), cópia de `analysis/final_verdict.json` |
| `analises_secundarias.json` | S1 a S4, por `../analises_secundarias.py` (código registrado) |
| `analise_desvio_01.json` | passos descartados e S5, por `../analise_desvio_01.py` |
| `resultados_v4.zip` | exportação do notebook 04: protocolo, operações, logs, análises e as 120 avaliações TEST (`runs/sft/*/evaluation_test_*.jsonl`), com `MANIFESTO_SHA256.json` interno |
| `MANIFESTO_SHA256.json` | SHA256 dos quatro arquivos acima, gerado na VM por `pos_coleta_v4.py` |
| `execucao_v4f/` | log do executor e notebooks executados na `v4f` (os da `v4d` ficavam no disco local da VM e se perderam com ela) |

Os adaptadores (cerca de 36 GB) permanecem no Google Drive do proprietário, com os SHA256
gravados em cada `status.json` incluído no ZIP.
