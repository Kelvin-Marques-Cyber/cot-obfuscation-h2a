# Teste técnico CUDA após a correção (29/10/2026, commit `4b7d1fd`)

Colab, Tesla T4 (driver 580.82.07, 15 GiB), `TECHNICAL_TEST=True`: uma semente,
dois passos por treino, quatro itens TEST por avaliação, pasta separada. Não é
evidência científica.

- Cinco notebooks executados sem erro (`notebooks_executados/`): 00 com falha
  real de montagem do Drive (sem interface do navegador) e preparação em pasta
  temporária com ZIP salvo; 01 com 6/6 treinos SFT; 02 informando a exclusão do
  braço GRPO; 03 com 6/6 avaliações TEST; 04 com veredito `INSUFFICIENT_EVIDENCE`
  (teste técnico, sem registro público).
- Interrupção forçada de treino (`interrupcao_treino_*.log`): processo morto
  (SIGKILL) com `checkpoint-1` gravado; a retomada continuou de `checkpoint-1` e
  concluiu no passo 2, com perdas finitas e adaptador salvo (SHA256 em
  `relatorio_tecnico.json`).
- Interrupção forçada de avaliação (`interrupcao_avaliacao_*.log`): processo
  morto após 2 itens; uma linha parcial foi acrescentada ao arquivo para simular
  gravação interrompida; a retomada descartou a linha parcial e concluiu 4 itens
  únicos, com a primeira linha idêntica à anterior.
- Os conjuntos `data/*.jsonl`, as cópias da triagem e os ZIPs exportados foram
  omitidos (seus SHA256 constam do protocolo, de `triagem_dev/manifest.json` e
  das saídas dos notebooks).
