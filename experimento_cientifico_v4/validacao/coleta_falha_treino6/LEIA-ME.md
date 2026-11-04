# Falha do treino 6 da coleta confirmatória (04/11/2026)

Coleta do v4 (registro https://osf.io/hb64a/) na NVIDIA A100-SXM4-80GB, notebooks do commit
`4b7d1fd`, código `b858d12`, estudo no Google Drive.

- Treinos 1 a 5 (nível A; sementes 113 e 114) concluídos com 120 passos.
- Treino 6 (nível A, condição `generic`, semente 114, `run_id` 191d16c4b6131856cfb4) parou no passo
  15: `FloatingPointError: grad_norm não finito no passo 15: nan`, levantado pelo guarda
  `FiniteLoss` de `cotv4/treino.py`. As perdas registradas nos passos 5 e 10 eram finitas
  (3,0794 e 2,2069).
- Repetido com os mesmos parâmetros, como prevê o registro, o treino reproduziu as mesmas
  perdas e o mesmo `nan` no passo 15: o treino é determinístico nessa GPU e a repetição não
  corrige a falha.
- Interpretação técnica: em fp16, o escalonador de gradientes do PyTorch descarta o passo em
  que o gradiente estoura e reduz a escala da perda; a norma registrada nesse passo é `nan`.
  O guarda interrompe o treino sempre que isso coincide com um passo de registro (a cada 5).
- Nenhum dado TEST foi gerado.

Arquivos: status e manifesto do treino que falhou, log do executor e do treino e o notebook 01
executado na primeira tentativa. O log da segunda tentativa está no Drive do estudo
(`logs/train_sft.log`).
