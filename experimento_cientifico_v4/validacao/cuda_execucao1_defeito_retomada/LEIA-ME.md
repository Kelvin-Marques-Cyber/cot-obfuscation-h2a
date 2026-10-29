# Primeira execução do teste técnico CUDA (29/10/2026, commit `6c41277`)

Colab, Tesla T4, `TECHNICAL_TEST=True` (uma semente, dois passos por treino,
quatro itens TEST por avaliação, pasta separada). Não é evidência científica.

- Os cinco notebooks executaram sem erro (`relatorio_tecnico.json`,
  `notebooks_executados/`): 6/6 treinos SFT, braço GRPO informado como excluído,
  6/6 avaliações e veredito `INSUFFICIENT_EVIDENCE` (teste técnico, sem registro).
- **Defeito encontrado na interrupção forçada de treino:** o processo foi morto
  depois de gravar o checkpoint do último passo (`checkpoint-2`) e antes de
  registrar a conclusão. Na retomada, a biblioteca de treino executou um passo a
  mais (`checkpoint-3`), e o runtime recusou o resultado
  (`interrupcao/…/failure.txt`: "Treino terminou no passo 3, esperado 2").
  O teste de interrupção da avaliação não chegou a rodar por depender desse treino.
- O defeito foi reproduzido em CPU com modelos mínimos e corrigido no commit
  `b858d12`: um treino cujo último checkpoint já está no passo final é concluído
  a partir desse checkpoint, sem retomar o otimizador. O teste
  `test_kill_after_final_checkpoint_resumes_without_extra_step` cobre o caso.
- A execução repetida após a correção está em `../relatorio_tecnico_cuda.json`.
