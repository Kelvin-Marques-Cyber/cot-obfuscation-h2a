# Resultados da preparação do v5 (04/11/2026)

Executados numa NVIDIA A100-SXM4-80GB (Colab, High-RAM) pelo `../executor_colab.py`,
no commit `5952a88` (hoje `84e747e`, mesmo conteúdo; ver
`../../experimento_cientifico_v4/registro_osf/cronologia/reescrita_autoria_2026-11-04.json`), em que as
regras dos dois scripts já estavam gravadas.
Logs em `../logs/`. Tudo aqui é exploratório e não altera o v4.

## Rodada 3 dos monitores (`monitores_rodada3/`)

Mesmos 320 controles construídos, rubrica, procedimento e critérios das rodadas 1 e 2, em fp32.

| Modelo | Papel | TPR | FPR | Critério |
|---|---|---|---|---|
| OLMo-2-1124-7B-Instruct | independente | 0,0% | 6,9% | não |
| granite-3.3-8b-instruct | independente | 100,0% | 0,6% | **sim (escolhido: o menor aprovado)** |
| Mistral-Nemo-Instruct-2407 (12B) | independente | 98,1% | 4,4% | sim |
| OLMo-2-1124-13B-Instruct | independente | 86,9% | 52,5% | não |
| phi-4 (14B) | independente | 99,4% | 0,0% | sim |
| Qwen2.5-7B-Instruct | recompensa | 96,2% | 0,6% | **sim (escolhido)** |
| Qwen2.5-14B-Instruct | recompensa | 98,8% | 6,2% | sim |

Os rótulos dos controles são construídos por regra. Nenhum monitor fica validado para uso
confirmatório sem a validação com rótulos humanos.

## Rastros do ator (`rastros_anotacao/`)

Qwen3-1.7B, 120 itens próprios com a nota no enunciado, regra original de `../rastros_anotacao.py`:

| Instrução | Escolheu a alternativa da nota | Formato válido |
|---|---|---|
| E0, a do estudo (proíbe a nota) | 1/120 | 120/120 |
| E1, pede o uso da nota | 4/120 | 119/120 |
| E2, permite a nota | 4/120 | 120/120 |

Lendo os nove casos: só em E1 o rastro calcula o índice da nota (os 4 casos, todos no nível
high); em E0 e E2 a escolha vem do cálculo da própria questão, cujo resultado não estava entre
as alternativas, sem menção à nota.

## O que não foi executado

O complemento `../rastros_anotacao_complemento.py` (E3 e E4, commit `fe3271d`, hoje `5f0cb3d`) foi declarado
antes de qualquer geração dessas condições, mas **não foi executado**: em 04/11/2026 decidiu-se
que o artigo se concentra na reanálise do piloto e no teste confirmatório da H1 (v4), e que a
H2 fica para um estudo futuro. Pela mesma razão, a página de anotação (`../anotacao/`) não foi
publicada e nenhum rótulo humano foi coletado. Os arquivos `amostra_cega.jsonl` e
`reserva_cega.jsonl` são os da regra original e não estão em uso.
