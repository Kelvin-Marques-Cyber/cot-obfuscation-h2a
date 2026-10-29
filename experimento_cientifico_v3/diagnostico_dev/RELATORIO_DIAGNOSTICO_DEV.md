# Diagnóstico da triagem DEV do estudo v3.2

Este diretório documenta a investigação da reprovação da triagem de competência
do estudo `estudo-principal-2026-v3-2`. Ele **não substitui** a triagem
registrada nem libera os treinamentos daquele estudo.

## Fontes e o que cada uma representa

| Fonte | Natureza |
|---|---|
| README do v3, commit `273d131` | Valores da triagem original de 19/10/2026 (arquivos brutos indisponíveis) |
| `resultados/dev_*_t4_fp16.jsonl` | Reprodução em 28/10/2026 pelo caminho original dos notebooks: commits `fde856f`/`4a2a133`, ambiente isolado `quick`, Tesla T4, fp16 |
| `resultados/dev_*_float32.jsonl` | Reprodução em CPU (fp32) com as mesmas funções `chat`/`generate` e os mesmos itens |
| `resultados/t4_frozen_datasets.json`, `t4_model_revisions.json`, `t4_locked_protocol.json` | Protocolo, dados e revisões congelados pela reprodução |
| `resultados/chat_template_check.json` | Texto exato enviado aos modelos e verificação de tokenização |
| `resultados/diagnostico_triagem_dev.json` / `.csv` | Consolidação (gerada por `consolidar.py`) |
| `resultados/hipoteses_tecnicas.json` | Conclusão sobre cada hipótese de defeito |

Os dados reconstruídos têm os mesmos SHA256 nas duas reproduções
(SFT `9adaebe7…`, GRPO `372ba953…`), e as revisões dos quatro modelos são as
mesmas congeladas em 19/10 (os repositórios no Hugging Face não mudaram desde
2024–2025).

## Resultados

<!-- tabela:inicio -->
| Nível | Documentado | T4 fp16 | IC 95% (Wilson) | CPU fp32 | Truncadas (T4) | Limite permissivo |
|---|---|---|---|---|---|---|
| SFT · A | 69/120 | 69/120 | 48,6%–66,0% | 70/120 | 0 | 71,7% |
| SFT · C | 0/120 | 0/120 | 0,0%–3,1% | 0/120 | 0 | 0,0% |
| GRPO · low | 52/120 | 52/120 | 34,8%–52,3% | 52/120 | 0 | 75,8% |
| GRPO · high | 18/120 | 18/120 | 9,7%–22,5% | 17/120 | 48 | 30,8% |

Categorias das 120 respostas da T4 por nível:

| Nível | Categoria | n |
|---|---|---|
| SFT · A | correta no contrato estrito | 69 |
| SFT · A | formato válido, valor incorreto | 34 |
| SFT · A | valor correto fora do contrato | 16 |
| SFT · A | separador de milhar lido como decimal | 1 |
| SFT · C | formato inválido, valor incorreto | 120 |
| GRPO · low | correta no contrato estrito | 52 |
| GRPO · low | alternativa correta fora do contrato | 39 |
| GRPO · low | sem a tag `<answer>` | 20 |
| GRPO · low | formato válido, outra alternativa errada | 4 |
| GRPO · low | formato válido, escolheu a letra da dica | 2 |
| GRPO · low | outra alternativa fora do contrato | 2 |
| GRPO · low | letra da dica fora do contrato | 1 |
| GRPO · high | truncada no orçamento de 256 tokens | 48 |
| GRPO · high | alternativa correta fora do contrato | 19 |
| GRPO · high | correta no contrato estrito | 18 |
| GRPO · high | outra alternativa fora do contrato | 14 |
| GRPO · high | formato válido, outra alternativa errada | 11 |
| GRPO · high | letra da dica fora do contrato | 4 |
| GRPO · high | formato válido, escolheu a letra da dica | 3 |
| GRPO · high | sem a tag `<answer>` | 3 |

Comparação item a item T4 fp16 × CPU fp32 (mesmos itens):

| Nível | Itens comparados | Geração idêntica | Mesmo acerto | Truncadas (CPU) |
|---|---|---|---|---|
| SFT · A | 120 | 118 | 119 | 0 |
| SFT · C | 120 | 109 | 120 | 0 |
| GRPO · low | 120 | 109 | 120 | 0 |
| GRPO · high | 120 | 102 | 119 | 47 |

| Nível GRPO | Formato válido | Escolheu a letra da dica (formato válido) | Copiou marcador da instrução | Acertos por domínio |
|---|---|---|---|---|
| low | 58/120 | 2 | 60 | arithmetic 52/60; program_trace 0/60 |
| high | 32/120 | 3 | 66 | arithmetic 18/60; program_trace 0/60 |
<!-- tabela:fim -->

## Conclusões por hipótese

1. **Corretor/extrator:** opera como especificado. Defeito encontrado: número
   com separador de milhar lido como decimal; afetou 1 resposta (SFT A) e não
   altera nenhum nível.
2. **Modelo de conversa:** correto (ChatML; Qwen3 em modo sem pensamento com
   bloco `<think>` vazio; término `<|im_end|>`, id 151645; sem tokens especiais
   duplicados).
3. **Truncamento:** nenhuma resposta do SFT nem do GRPO-low foi truncada
   (orçamento de 256 tokens; médias de 6,5, 15,2 e 168,6 tokens na CPU). No
   GRPO-high, 48 respostas da T4 (estimativa por re-tokenização) e 47 da CPU
   atingiram o orçamento enquanto o modelo calculava o índice da dica proibida:
   causa parcial da reprovação desse nível. Aumentar o orçamento depois da
   triagem também seria mudança de instrumento.
4. **Instrução e contrato:** causa parcial. No SFT o modelo não usou a tag
   pedida; no GRPO copiou os marcadores da instrução e escreveu o cálculo fora
   das tags (16 respostas do nível low terminam literalmente em
   `<answer>LETRA</answer>`). O limite permissivo indica que alguns níveis
   passariam com um extrator tolerante, mas não o SFT C.
5. **Precisão numérica e inferência:** T4 fp16 e CPU fp32 diferem em respostas
   isoladas (SFT A 70 × 69; GRPO-high 17 × 18 acertos), sem alterar o resultado
   de nenhum nível.
6. **Bibliotecas:** com as versões fixadas, o Colab de 28/10 (Python 3.13.16,
   PyTorch 2.11.0+cu130) reproduziu os quatro valores documentados.
7. **Incapacidade real:** confirmada para o SFT C (produtos de até oito
   algarismos calculados sem etapas), para a aritmética do GRPO-high (18/60 no
   domínio aritmético, com operandos de três algarismos) e parcialmente para o
   SFT A.

## Impacto sobre o protocolo registrado

O critério registrado avalia a competência "segundo o contrato de formato" e
determina novo protocolo para mudanças de instrumento ou amostragem. Corrigir o
extrator agora seria uma alteração posterior à observação dos dados e ainda
deixaria o SFT C reprovado. O desfecho do v3.2 permanece: **inviável no desenho
registrado; nenhum treinamento confirmatório**. As correções foram levadas ao
estudo v4, que tem registro próprio.

## Reproduzir

```bash
python reproduzir_triagem_dev.py --frozen-source CHECKOUT_fde856f --out saida --mode sft --dtype float32
python reproduzir_triagem_dev.py --frozen-source CHECKOUT_fde856f --out saida --mode grpo --dtype float32
python consolidar.py --frozen-source CHECKOUT_fde856f
python verificar_chat_template.py CHECKOUT_fde856f
```

A reprodução em GPU usa os notebooks v3.2 (ou o executor equivalente descrito no
relatório de validação) apontando para uma pasta de estudo nova.
