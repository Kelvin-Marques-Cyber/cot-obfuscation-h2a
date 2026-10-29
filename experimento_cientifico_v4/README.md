# Estudo prospectivo v4 — formato de supervisão e monitorabilidade da CoT

Estudo **novo** (`STUDY_ID = estudo-prospectivo-2026-v4`), preparado depois do
encerramento do v3.2 por reprovação na triagem de competência. Protocolo:
[PROTOCOLO_v4.md](PROTOCOLO_v4.md) · parâmetros congelados:
[protocolo_v4.json](protocolo_v4.json) · calibração: [calibracao/](calibracao) ·
triagem DEV: [triagem_dev/](triagem_dev).

**Situação em 29/10/2026:** protocolo congelado (commit `37094c4`) e triagem DEV
executada uma única vez na Tesla T4: SFT A 117/120 e C 82/120, ambos ≥ 0,65. O
braço GRPO foi excluído antes do registro porque nenhum monitor foi aprovado na
calibração (H2 não testável; triagem GRPO apenas descritiva). **Próximo passo:**
publicar o registro com o pacote do notebook 00. Nenhum treino foi executado.

## Notebooks (abrir pelo link, nunca por cópia antiga)

| Ordem | Notebook no Colab | Ambiente |
|---|---|---|
| 00 | [Protocolo, dados, triagem DEV e pacote de registro](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/experimento_cientifico_v4/00_protocolo_dados_e_triagem.ipynb) | CPU |
| 01 | [Treinamento SFT (120 treinos)](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/experimento_cientifico_v4/01_treinamento_sft.ipynb) | GPU T4+ |
| 02 | [Treinamento GRPO](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/experimento_cientifico_v4/02_treinamento_grpo.ipynb) | GPU T4+ |
| 03 | [Avaliação TEST (`full`)](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/experimento_cientifico_v4/03_avaliacao_test.ipynb) | GPU T4+ |
| 04 | [Monitor independente, anotação humana e inferência](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/experimento_cientifico_v4/04_monitor_humanos_e_inferencia.ipynb) | GPU T4+ |

Pacote com os cinco notebooks, protocolo e dependências: [`notebooks_v4.zip`](notebooks_v4.zip).

Cada notebook baixa exatamente o commit `CODE_COMMIT` impresso na primeira
célula, confere o SHA256 de cada módulo e carrega o suporte com nome exclusivo
daquele commit; módulos de outras versões carregados na mesma sessão não são
reutilizados. O kernel não importa o pacote científico: cada etapa roda em
subprocesso e as etapas de GPU usam um ambiente virtual isolado que reutiliza o
PyTorch/CUDA do Colab (`--system-site-packages`) e instala apenas as versões
fixadas em [requirements-gpu.txt](requirements-gpu.txt).

## Como executar

1. **00 (CPU):** *Ambiente de execução → Executar tudo*; autorize o Drive.
   Prepara e congela protocolo, dados e revisões, importa a triagem DEV
   conferindo os SHA256 e baixa `pacote_registro_v4.zip`.
2. Publique o registro (OSF ou equivalente) **antes de qualquer treino**,
   incluindo os SHA256 do pacote (`MANIFESTO_SHA256.json`). Preencha
   `PUBLIC_REGISTRATION_URL` no 00 e execute tudo de novo.
3. **01 (GPU T4 ou superior):** executar tudo repetidas vezes, com o mesmo
   `STUDY_ID`, até 120/120 treinos SFT. Cada sessão faz `MAX_JOBS_PER_SESSION`
   treinos; treinos interrompidos retomam do último checkpoint e treinos
   concluídos não são refeitos.
4. **02 (braço GRPO):** excluído do v4 antes do registro, porque nenhum monitor
   foi aprovado na calibração; o notebook informa a exclusão e não treina.
5. **03 (GPU):** só abre TEST depois dos 120 treinos SFT; repita até concluir
   as 120 avaliações (26.400 gerações).
6. **04 (GPU):** relatório final de H1. As etapas de monitor independente e de
   anotação humana pertencem ao braço GRPO e não se aplicam a este estudo.

A primeira célula mostra a situação de cada etapa: **CONCLUÍDA** (arquivos
verificados), **BLOQUEADA** (critério do protocolo ou recurso ausente, com o
motivo) e **NÃO EXECUTADA/PENDENTE**.

## Drive, falhas e retomada

- `drive.mount` é tentado no máximo duas vezes; o Drive só é aceito se o ponto
  de montagem for real e um arquivo de teste for gravado e relido. Nenhuma
  pasta `MyDrive` é criada para simular persistência e nada é apagado.
- Sem Drive, o 00 prepara tudo em pasta **temporária** e oferece o ZIP; 01–04
  ficam pausados. Para reconectar: `STORAGE_MODE="auto"` e executar de novo a
  primeira célula; se persistir, *Desconectar e excluir ambiente de execução*.
- Se o download automático falhar, o ZIP permanece no caminho impresso: painel
  **Arquivos** → três pontos → *Fazer download*.
- Falta de GPU, de VRAM (< 14 GiB) ou de espaço em disco produz mensagem de
  bloqueio específica; falhas reais interrompem a célula com o final do log e
  geram um ZIP de diagnóstico sem pesos.
- `TECHNICAL_TEST=True` cria uma pasta separada (`…-teste-tecnico`) com uma
  semente, dois passos por treino e quatro itens TEST; nunca é evidência.

## Verificação local

```bash
python -m unittest tests.test_contratos tests.test_colab tests.test_integracao_minima
python tests/integracao_notebooks.py
python build_notebooks.py --code-ref <SHA do commit de código>
```

`test_integracao_minima` usa modelos mínimos de pesos aleatórios em CPU:
verifica as APIs de treino, retomada e avaliação, não o desempenho.
