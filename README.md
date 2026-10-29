# Ofuscação da cadeia de pensamento: reanálise, estudo pré-registrado e protocolo revisado

Repositório do projeto `cot-obfuscation-h2a`, sob responsabilidade de Kelvin e
Silva Marques. Reúne o artigo, os dados, o código e os notebooks de três
conjuntos de trabalho que não devem ser confundidos:

| Conjunto | Situação em 28/10/2026 | Onde |
|---|---|---|
| Estudo-piloto (SFT) e reanálise | Concluído: números reproduzidos; instrumentos auditados | [`revisao_2026/`](revisao_2026) |
| Estudo prospectivo v3.2 (OSF [sjwdf](https://osf.io/sjwdf/)) | **Encerrado sem treinos**: triagem DEV reprovada; diagnóstico concluído | [`experimento_cientifico_v3/`](experimento_cientifico_v3) |
| Estudo prospectivo v4 (novo `STUDY_ID`) | <<SITUACAO_V4>> | [`experimento_cientifico_v4/`](experimento_cientifico_v4) |

**Artigo:** [PDF](revisao_2026/docs/Artigo_CoT_Versao_Final.pdf) ·
[DOCX editável](revisao_2026/docs/Artigo_CoT_Versao_Final.docx) ·
[fonte](revisao_2026/artigo.md) · [relatório de validação](revisao_2026/docs/RELATORIO_VALIDACAO.md)

Nenhum resultado confirmatório sobre as hipóteses H1 (custo do formato no SFT)
e H2 (exploração não sinalizada no GRPO) foi produzido até esta data.

## Estudo v4: os cinco notebooks

| Ordem | Notebook no Colab | Ambiente |
|---|---|---|
| 00 | [Protocolo, dados, triagem DEV e pacote de registro](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/experimento_cientifico_v4/00_protocolo_dados_e_triagem.ipynb) | CPU |
| 01 | [Treinamento SFT (120 treinos)](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/experimento_cientifico_v4/01_treinamento_sft.ipynb) | GPU T4+ |
| 02 | [Treinamento GRPO](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/experimento_cientifico_v4/02_treinamento_grpo.ipynb) | GPU T4+ |
| 03 | [Avaliação TEST (`full`)](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/experimento_cientifico_v4/03_avaliacao_test.ipynb) | GPU T4+ |
| 04 | [Monitor independente, anotação humana e inferência](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/experimento_cientifico_v4/04_monitor_humanos_e_inferencia.ipynb) | GPU T4+ |

Instruções de execução, retomada e falhas de Drive: [`experimento_cientifico_v4/README.md`](experimento_cientifico_v4/README.md). Pacote ZIP: [`experimento_cientifico_v4/notebooks_v4.zip`](experimento_cientifico_v4/notebooks_v4.zip).

## Estudo v3.2: desfecho

A triagem de competência exigia ≥ 65% em 120 itens DEV por nível. Os valores
documentados em 19/10/2026 (SFT A 57,5%; SFT C 0%; GRPO low 43,33%; GRPO high
15%) foram reproduzidos exatamente em 28/10/2026 pelo mesmo caminho dos
notebooks, numa Tesla T4. O diagnóstico descarta erro de execução, modelo de
conversa, precisão numérica e incompatibilidade de bibliotecas. A reprovação
decorre da incapacidade do modelo-base (nível C do SFT; aritmética com três
algarismos no GRPO-high, em que 48 respostas esgotaram o orçamento de 256
tokens) e de instruções/contratos de formato que rejeitavam respostas corretas
fora do formato pedido. Os monitores previstos também não discriminam
controles construídos. Detalhes: [`experimento_cientifico_v3/ENCERRAMENTO_v3_2.md`](experimento_cientifico_v3/ENCERRAMENTO_v3_2.md)
e [`diagnostico_dev/`](experimento_cientifico_v3/diagnostico_dev).

Os cinco notebooks do v3.2 permanecem disponíveis, fixados nos commits
originais, como registro do estudo encerrado.

## Verificar localmente

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r revisao_2026/requirements.lock.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r experimento_cientifico_v4/requirements-gpu.txt
python revisao_2026/src/executar_validacao.py
```

O executor roda as suítes do laboratório, do v3.2 e do v4, a reanálise
histórica, a integração com modelos mínimos de pesos aleatórios e os cinco
notebooks v4 com o Colab simulado, e grava comandos, versões, duração, códigos
de saída e hashes em `revisao_2026/results/execucoes_locais.json`.

## Gerar o artigo

```bash
python revisao_2026/src/gerar_artigo.py
```

Requer pandoc 3, LibreOffice com o componente Math e poppler-utils. O DOCX é
gerado pelo pandoc com o modelo de estilos `revisao_2026/src/modelo_docx.py`
(ABNT: A4, margens 3/2 cm, Liberation Serif) e o PDF é convertido do mesmo DOCX
pelo LibreOffice; o script confere a equivalência textual e grava
`revisao_2026/docs/verificacao_documentos.json`.

## Laboratório de reanálise (histórico)

Os notebooks de [`revisao_2026/notebooks/`](revisao_2026/notebooks) reproduzem a
reanálise e testes curtos do laboratório de 15/10/2026. Eles não fazem parte do
estudo prospectivo. [Guia do laboratório](revisao_2026/README.md).

## Procedência e histórico

Os commits deste repositório têm como autor Kelvin e Silva Marques; não há
trailers de coautoria automática. Nenhuma reescrita de histórico foi necessária,
e todos os commits fixados pelos notebooks e protocolos continuam acessíveis
(ver [auditoria Git](revisao_2026/docs/AUDITORIA_GIT.md)).

Código sob licença MIT. Os modelos usados (Qwen2.5, Qwen3, SmolLM2 e demais
citados no protocolo v4) mantêm suas licenças de origem.
