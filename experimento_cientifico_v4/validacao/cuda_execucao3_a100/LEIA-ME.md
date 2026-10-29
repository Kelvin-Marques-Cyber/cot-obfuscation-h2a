# Teste técnico CUDA na A100 (29/10/2026, commit `4b7d1fd`)

Colab, NVIDIA A100-SXM4-80GB (máquina High-RAM, driver 580.82.07, 80 GiB),
`TECHNICAL_TEST=True`: uma semente, dois passos por treino, quatro itens TEST por
avaliação, pasta separada. Mesmo executor (`../executores_colab/teste_tecnico_cuda.py`)
e mesmo commit do teste na T4 (`../cuda_execucao2/`). Não é evidência científica.

Motivo: verificar o software na classe de GPU escolhida para o estudo antes do
registro no OSF. A H100 foi pedida primeiro e recusada pelo Colab para a conta
(HTTP 400, sem direito de uso do acelerador); a A100 foi escolhida antes deste
teste, e o resultado não alterou nenhuma decisão de desenho.

- Cinco notebooks executados sem erro (`notebooks_executados/`), com as mesmas
  saídas da T4: 00 com falha real de montagem do Drive (sem interface do
  navegador) e preparação em pasta temporária com ZIP salvo; 01 com 6/6 treinos
  SFT; 02 informando a exclusão do braço GRPO; 03 com 6/6 avaliações TEST; 04
  com veredito `INSUFFICIENT_EVIDENCE` (teste técnico, sem registro público).
- Interrupção forçada de treino (`interrupcao_treino_*.log`): processo morto
  (SIGKILL) com `checkpoint-1` gravado; a retomada continuou de `checkpoint-1` e
  concluiu no passo 2, com perdas finitas e adaptador salvo.
- Interrupção forçada de avaliação (`interrupcao_avaliacao_*.log`): processo
  morto após 1 item; linha parcial acrescentada para simular gravação
  interrompida; a retomada a descartou e concluiu 4 itens únicos, com a
  primeira linha idêntica à anterior.
- Os conjuntos `data/*.jsonl`, as cópias da triagem e os ZIPs exportados foram
  omitidos; seus SHA256 coincidem com os do protocolo e de
  `triagem_dev/manifest.json` (SFT `f7ee4224…`, triagem `e17080a1…`).

Comparação com a T4 (mesmos 6 adaptadores de 2 passos, mesmos 24 pares
adaptador × item): perdas iguais até a terceira casa decimal; 23 de 24 respostas
idênticas; uma resposta (nível A, `direct`) mudou e passou de correta a
incorreta. É a diferença numérica entre classes de GPU que motiva a regra de
classe única no registro. Mediana de tempo: geração 0,31 s por item (T4:
0,34 s); treino de 2 passos 4,0 s (T4: 5,0 s).
