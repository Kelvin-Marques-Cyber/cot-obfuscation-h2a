# Preparação do v5: monitores maiores e validação humana

Esta pasta prepara um estudo futuro (v5) que possa testar a hipótese H2 sobre
pressão de recompensa no GRPO. Nada aqui altera o v4: o braço GRPO do v4 foi
excluído antes do registro, e o v4 testa somente H1. Tudo nesta pasta é
exploratório até que um protocolo v5 seja congelado e registrado.

O v4 deixou dois requisitos sem solução: nenhum monitor distinguiu os 320
controles construídos, e não havia rótulos humanos. Os dois scripts abaixo
atacam esses requisitos. As regras de cada um estão no topo do arquivo e foram
gravadas no repositório antes da execução correspondente; o horário de push
registrado pelo GitHub permite conferir a ordem.

| Script | O que faz |
|---|---|
| `monitores_rodada3.py` | Rodada 3 da viabilidade dos monitores: 7 modelos de 7B a 14B, mesmos controles, rubrica, procedimento e critérios das rodadas 1 e 2, em fp32 numa A100-SXM4-80GB |
| `rastros_anotacao.py` | Gera rastros reais do ator Qwen3-1.7B sob três instruções (proíbe, pede ou permite o uso da nota) e sorteia a amostra cega de 300 itens para anotação humana |
| `executor_colab.py` | Executa os scripts acima no Colab, no commit indicado, com o ambiente de GPU dos notebooks do v4 |

Os resultados ficam em `resultados/` e os rótulos humanos, em `anotacao/`. Os
rótulos são dados por pessoas, os coautores do projeto, na página de anotação;
nenhum rótulo é gerado por software.
