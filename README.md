# Custo de acurácia da ofuscação de CoT sob supervisão de saída (estudo-piloto, H2.a)

Código, dados e figuras que reproduzem o **estudo-piloto** de uma reavaliação experimental de
**Mitrani Hadida et al. (2026)** sobre ofuscação da cadeia de pensamento (*chain-of-thought*, CoT)
sob supervisão de saída.

O piloto testa o **corolário H2.a (custo de acurácia)**: *quando a ofuscação é forçada, ela
coincide com queda de acurácia maior em tarefas cujo raciocínio é causalmente necessário
(load-bearing) do que em tarefas decorativas* — previsão de Emmons et al. (2025).

## Resultado principal

| Nível | Controle | CoT-pressure | Output-supervision |
|:--|:--:|:--:|:--:|
| **A** (CoT decorativa) | 0,933 | 1,000 | 0,967 |
| **C** (load-bearing) | 0,678 | 0,156 | 0,056 |

- **Nível A:** ofuscar **não custa** acurácia (a ofuscação é "gratuita").
- **Nível C:** a mesma ofuscação **colapsa** a acurácia (−77% e −92%) = degradação passiva.
- **Interação nível × penalização:** F(2, 12) = 527,17; p ≈ 2,03 × 10⁻¹².
- **Custo em C** (Controle vs. penalizado): t = 20,94; p ≈ 3,09 × 10⁻⁷.
- **Validação (intervenção de Lanham):** necessidade causal A = 0,00; C = 0,50.

Veredito: dentro dos limites de uma prova de conceito, os dados **corroboram a hipótese de
refutação** — o efeito de ofuscação é moderado pela necessidade causal do raciocínio.

## Escopo e limites (importante)

Esta é uma **prova de conceito**, não o estudo completo. Diferenças em relação ao delineamento
pleno descrito no artigo:

- **Ajuste supervisionado (SFT), não GRPO.** A ofuscação é *imposta* pelo alvo de treino, e não
  *emerge* de otimização por reforço contra um monitor. Logo, o piloto mede o **custo** da
  ofuscação (H2.a), não sua **emergência** (H1).
- **Tarefas aritméticas controladas** (1 passo vs. múltiplos passos), escolhidas pela correção
  numérica exata — e não as tarefas realistas (clínicas, jurídicas, programação) do plano pleno.
- **Um único modelo** (Qwen2.5-7B-Instruct) e **3 sementes**, com ajuste curto.

## Desenho do experimento

- **Fator 1 — necessidade causal:** Nível A (aritmética de 1 passo) × Nível C (problema de
  múltiplos passos).
- **Fator 2 — penalização (via alvo de treino):** Controle (CoT fiel) · CoT-pressure (sem CoT) ·
  Output-supervision (CoT vácuo).
- **3 sementes** (42, 43, 44) → **18 modelos**; 30 itens de avaliação por célula (540 avaliações).
- Métricas **desacopladas**: acurácia genuína medida em separado da ofuscação; ofuscação medida
  só pela qualidade do CoT (densidade causal, TTR, repetição, entropia).

## Arquivos

- `cot_obfuscation_experiment.ipynb` — notebook completo (instala dependências, treina, avalia,
  faz a análise e exporta os resultados).
- `h2a_resultados_brutos.csv` — uma linha por avaliação (540 linhas).
- `h2a_por_celula.csv` — agregado por nível × condição × semente.
- `h2a_custo_acuracia.csv` — custo de acurácia por nível.
- `h2a_necessidade_causal.csv` — teste de intervenção (necessidade causal por nível).
- `figuras/` — figuras 1–4 (300 dpi) usadas no artigo.

## Como reproduzir

Requer **GPU**. Testado em uma NVIDIA RTX 5090 (vast.ai) e compatível com Google Colab.

1. Abra o notebook em um ambiente com GPU (Jupyter no vast.ai ou Colab).
2. Rode todas as células (a célula 2 instala as dependências; detecta Colab vs. não-Colab).
3. Em terminal, opcionalmente, com [papermill](https://papermill.readthedocs.io):

```bash
pip install papermill
papermill cot_obfuscation_experiment.ipynb saida_executada.ipynb --log-output
```

**Versões fixadas e testadas:** `transformers==4.56.2`, `trl==0.22.2`, Unsloth (instalação
`--no-deps` quando há torch pré-instalado), PyTorch ≥ 2.4. Sementes: 42, 43, 44.

Para um teste rápido (~5 min), na célula de configuração use `seeds=(42,)`, `n_eval=8`,
`num_epochs=1`.

## Citação

Repositório de apoio ao artigo *"Validade externa da ofuscação de cadeia de pensamento sob
supervisão de saída: um teste empírico em tarefas com raciocínio causalmente necessário"*
(Centro Universitário Ateneu, 2026).

## Licença

Código sob licença MIT (ver `LICENSE`). Dados e figuras sob CC-BY 4.0.
