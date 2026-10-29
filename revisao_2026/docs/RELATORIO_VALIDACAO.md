# Relatório de validação e execuções — 28 e 29/10/2026

Este relatório lista o que foi executado nesta revisão, em que ambiente e com que resultado. Verificações de software não constituem evidência científica sobre as hipóteses.

| Verificação | Ambiente | Resultado |
|---|---|---|
| Laboratório de reanálise: testes unitários | CPU (Intel(R) Xeon(R) Processor @ 2.10GHz, 4 núcleos), Python 3.12.3 | aprovado (Ran 71 tests in 8.761s OK; 10.4 s) |
| Reanálise histórica (reanalyse.py) | CPU (Intel(R) Xeon(R) Processor @ 2.10GHz, 4 núcleos), Python 3.12.3 | aprovado (código 0; 2.5 s) |
| Estudo v3.2: testes de contrato e armazenamento | CPU (Intel(R) Xeon(R) Processor @ 2.10GHz, 4 núcleos), Python 3.12.3 | aprovado (Ran 29 tests in 1.036s OK; 1.2 s) |
| Estudo v4: contratos, armazenamento e sessão | CPU (Intel(R) Xeon(R) Processor @ 2.10GHz, 4 núcleos), Python 3.12.3 | aprovado (Ran 35 tests in 1.702s OK (skipped=1); 1.8 s) |
| Estudo v4: integração com modelos mínimos de pesos aleatórios (CPU) | CPU (Intel(R) Xeon(R) Processor @ 2.10GHz, 4 núcleos), Python 3.12.3 | aprovado (Ran 3 tests in 16.260s OK; 19.3 s) |
| Estudo v4: cinco notebooks com Colab simulado (montagem e download falhando) | CPU (Intel(R) Xeon(R) Processor @ 2.10GHz, 4 núcleos), Python 3.12.3 | aprovado (código 0; 7.2 s) |
| Reprodução da triagem DEV v3.2 pelo caminho original (commits fde856f/4a2a133) | Colab, Tesla T4, Python 3.13.16, PyTorch 2.11.0+cu130, fp16 | quatro valores documentados reproduzidos exatamente |
| Reprodução da triagem DEV v3.2 com as mesmas funções em CPU | CPU x86_64, Python 3.12.3, PyTorch 2.8.0, fp32 | sft/A: 70/120; sft/C: 0/120; grpo/low: 52/120; grpo/high: 17/120 |
| Calibração exploratória do v4 (rodadas 1 e 2, itens próprios) | Colab, Tesla T4, fp16, decodificação gulosa item a item | SFT: nível C = C2; GRPO: variante r2b, faixa G1 (regras gravadas antes de cada rodada) |
| Viabilidade de 9 monitores em 320 controles construídos | Colab, NVIDIA L4 e Tesla T4, fp32 | nenhum monitor independente nem de recompensa aprovado; braço GRPO excluído pela regra declarada |
| Encerramento de sessões do Colab durante a calibração | Colab (sessões sem atividade foram recicladas pelo serviço) | etapas repetidas com o mesmo código; valores repetidos idênticos aos observados; a 1ª execução da rodada 2 dos monitores morreu por falta de RAM e foi repetida integralmente após declaração |
| Triagem DEV do v4 (execução única, antes do registro) | Colab, Tesla T4, fp16, decodificação gulosa | aprovada no braço que condiciona os treinos: SFT A 117/120; SFT C 82/120; GRPO low 114/120 (descritivo); GRPO high 109/120 (descritivo) |
| Teste técnico CUDA, 1ª execução (commit 6c41277) | Colab, Tesla T4, fp16 | 5 de 5 notebooks sem erro; a interrupção após o checkpoint final revelou um passo extra na retomada, recusado pelo runtime; defeito reproduzido em CPU e corrigido (commit b858d12) |
| Teste técnico CUDA: cinco notebooks v4 com modelos reais (TECHNICAL_TEST), commit 4b7d1fd | Colab, Tesla T4, fp16 | 5 de 5 notebooks executados sem erro |
| Interrupção forçada de treino SFT na GPU e retomada | Colab, Tesla T4, fp16 | retomado de checkpoint-1 até o passo 2 (trained) |
| Interrupção forçada de avaliação na GPU e retomada com linha parcial | Colab, Tesla T4, fp16 | 4 linhas, 4 itens únicos; primeira linha idêntica: True |

## Situação dos experimentos

| Experimento | Situação | Detalhe |
|---|---|---|
| Reanálise do estudo-piloto | concluído | médias, ANOVA e Welch reproduzidos; instrumentos auditados |
| Estudo v3.2 (OSF sjwdf): treinos e TEST | bloqueado (encerrado) | triagem DEV reprovada nos quatro níveis; nenhum treino |
| Diagnóstico da triagem do v3.2 (T4 e CPU) | concluído | quatro valores reproduzidos; causas classificadas |
| Calibração do v4 (tarefas, instruções e monitores) | concluído | SFT C2; GRPO r2b; nenhum monitor aprovado |
| Triagem DEV do v4 (execução única) | concluído | SFT aprovado; GRPO descritivo |
| Braço GRPO do v4 (H2) | excluído antes do registro | sem monitor independente nem de recompensa viável |
| Registro público do v4 | pendente (ação do pesquisador) | publicar o pacote do notebook 00 antes do primeiro treino |
| 120 treinos SFT, 26.400 gerações TEST e inferência de H1 | pendente | dependem do registro público |
| Teste técnico CUDA dos cinco notebooks v4 | concluído | modo técnico; não é evidência científica |

## Natureza da evidência

| Conclusão | Categoria | Base |
|---|---|---|
| Médias e interação do piloto reproduzidas | resultado histórico | 540 linhas publicadas reprocessadas |
| Corretor histórico aceita respostas ambíguas | verificação de software | 2.000 textos construídos |
| Sementes do piloto compartilham quase todos os itens | verificação de software | reexecução do gerador original |
| Triagem do v3.2 reprovada nos quatro níveis | triagem de competência | valores documentados; reproduzidos exatamente na T4 |
| Causas da reprovação (incapacidade em C; instrução e contrato) | experimento exploratório | classificação das gerações reproduzidas |
| Nenhum dos monitores avaliados discrimina controles construídos | experimento exploratório | 9 modelos em 320 controles rotulados por regra, não por pessoas |
| Exclusão do braço GRPO (H2 não testável no v4) | decisão de desenho | regra gravada antes do resultado dos monitores |
| Escolhas de tarefa e instrução do v4 | experimento exploratório | calibração em itens próprios, regras gravadas antes |
| Resultado da triagem do v4 | triagem de competência | 120 itens DEV por nível, execução única |
| Software de treino, avaliação e retomada funciona em CPU e GPU | verificação de software | testes unitários, integração e modo técnico |
| Efeitos de H1 e H2 | resultado confirmatório | não produzido: nenhum treino confirmatório executado |

## Comandos locais

- `python -m unittest discover -s tests` em `revisao_2026`: código 0, 10.4 s
- `python src/reanalyse.py` em `revisao_2026`: código 0, 2.5 s
- `python -m unittest discover` em `experimento_cientifico_v3`: código 0, 1.2 s
- `python -m unittest tests.test_contratos tests.test_colab` em `experimento_cientifico_v4`: código 0, 1.8 s
- `python -m unittest tests.test_integracao_minima` em `experimento_cientifico_v4`: código 0, 19.3 s
- `python tests/integracao_notebooks.py --report <temporário>` em `experimento_cientifico_v4`: código 0, 7.2 s

Ambiente: `{"python": "3.12.3", "platform": "Linux-6.18.44-fc-v185-x86_64-with-glibc2.39", "torch": "2.8.0+cu128", "transformers": "4.56.2", "trl": "0.22.2", "peft": "0.17.1", "accelerate": "1.10.1", "datasets": "4.3.0", "numpy": "2.3.5", "pandas": "2.2.3", "scipy": "1.17.0", "matplotlib": "3.10.8", "docx": "1.2.0", "cpu": "Intel(R) Xeon(R) Processor @ 2.10GHz", "cpu_count": 4, "ram_gib": 15.7, "pandoc": "pandoc 3.1.3", "libreoffice": "LibreOffice 24.2.7.2 420(Build:2)"}`

Commit avaliado: `8459c4ccefcef296ef54f903352209b4799b91bd`
