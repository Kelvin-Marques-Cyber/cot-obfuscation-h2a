# Executores usados nas sessões de GPU do Colab

Scripts executados numa máquina virtual do Colab (Tesla T4; Phi-3.5-mini numa
NVIDIA L4; teste técnico repetido na NVIDIA A100-SXM4-80GB, em
`../cuda_execucao3_a100/`), enviados e acionados por linha de comando (`google-colab-cli`), sem
alterar os notebooks. Cada um baixa o commit indicado, confere o `HEAD`, cria o
ambiente isolado de `requirements-gpu.txt` pelo mesmo código dos notebooks
(`colab/sessao.py`) e grava logs com código de saída e duração.

| Script | Uso | Argumentos |
|---|---|---|
| `calibracao.py` | calibrações e viabilidade dos monitores | `COMMIT PASTA "script [opções]"…` |
| `triagem_dev.py` | triagem DEV do v4 (execução única) | `COMMIT` (commit congelado `37094c4`) |
| `teste_tecnico_cuda.py` | cinco notebooks em `TECHNICAL_TEST=True` e interrupções forçadas | `COMMIT` dos notebooks |

A reprodução da triagem do v3.2 usou `../../experimento_cientifico_v3/diagnostico_dev/executor_colab_t4.py`.
Nenhum desses scripts produz resultado confirmatório.
