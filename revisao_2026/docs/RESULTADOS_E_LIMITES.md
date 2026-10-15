# Resultados executados e limites

Data: 15/10/2026.

- 540 linhas históricas reanalisadas; 18 células e três sementes.
- Médias publicadas reproduzidas integralmente.
- ANOVA histórica reproduzida: F(2,12)=527,1667; p=2,0311e-12.
- Contrastes de custos C-A: 0,5889 e 0,6556.
- O teste exato por inversão de sinais de três contrastes tem p mínimo de 0,25;
  é uma análise de sensibilidade, não demonstração de ausência de efeito.
- Comparação pareada com/direta no nível C: b=5, c=2, McNemar exato p=0,453125.
- Avaliações de sementes adjacentes compartilham 29/30 enunciados.
- Há uma colisão reconstruída treino/avaliação em A: Calcule 11 - 99.
- Corretor histórico aceita 1.250 dos 1.500 negativos do corpus mecânico.
- Corretor estrito não aceita esses 1.500 negativos; o resultado é restrito ao contrato.
- Novo gerador aritmético produz 4.500 itens com IDs únicos e gabaritos conferidos.
- 53 testes de software passaram localmente.

Não foram executados novos treinos CUDA, GRPO, inferência de modelos,
monitor remoto ou anotação humana. Os notebooks GPU são entregues para essas
execuções. Os testes CPU incluem as regras de recompensa e o isolamento entre
rastro e resumo, não a execução de um treinador real.

O laboratório GRPO aceita dados externos no mesmo contrato JSONL. Para substituir
os dados sintéticos, use um arquivo com item_id, level, domain, split, question,
hint, gold, cheat e dataset_seed. gold e cheat são letras A-D diferentes.
Partições devem ser train/dev/test. Identificadores e enunciados repetidos entre
partições são rejeitados. Para usar o modo wrong_step em tarefas não numéricas,
o prefixo usa a alternativa de burla; não afirma que seja intervenção causal validada.

O PDF final é um estudo concluído de reanálise e validação de instrumentos.
Não apresenta como executado o estudo confirmatório multiescala.
