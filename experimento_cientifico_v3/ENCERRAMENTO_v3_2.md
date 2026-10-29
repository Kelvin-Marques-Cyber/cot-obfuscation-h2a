# Encerramento do estudo v3.2 (`estudo-principal-2026-v3-2`)

**Data do encerramento:** 28/10/2026 · **Registro:** OSF [sjwdf](https://osf.io/sjwdf/)
(criado em 19/10/2026 16:11:44 UTC, público desde 16:45 UTC) ·
**Código congelado:** `fde856f` (científico) e `4a2a133` (suporte Colab)

## Desfecho

O estudo foi encerrado **sem nenhum treinamento**. A triagem de competência
pré-registrada exigia acurácia ≥ 0,65 em 120 itens DEV por nível para os dois
modelos-base; os quatro níveis ficaram abaixo do critério. Conforme o registro,
mudanças de modelo, instrumento ou amostragem exigem novo protocolo prospectivo.
O desenho revisado é o estudo v4 ([`../experimento_cientifico_v4`](../experimento_cientifico_v4)),
com identificador, código, notebooks e registro próprios. As hipóteses H1 e H2
**não foram testadas** neste estudo; o desfecho não é evidência a favor nem
contra elas.

## Triagem

| Nível | Documentado (19/10, T4) | Reprodução T4 fp16 (28/10) | Critério |
|---|---|---|---|
| SFT · A | 69/120 (57,5%) | 69/120 | 0,65 |
| SFT · C | 0/120 (0%) | 0/120 | 0,65 |
| GRPO · low | 52/120 (43,33%) | 52/120 | 0,65 |
| GRPO · high | 18/120 (15%) | 18/120 | 0,65 |

Os arquivos brutos de 19/10 ficaram no Google Drive da sessão que executou a
triagem e não estavam acessíveis nesta auditoria; os valores originais são os
documentados no README (commit `273d131`). A reprodução de 28/10 seguiu o caminho
dos notebooks (mesmos commits, mesmo ambiente isolado, Tesla T4, fp16) numa
pasta de diagnóstico separada, sem tocar no estudo.

## Diagnóstico

Relatório completo: [`diagnostico_dev/RELATORIO_DIAGNOSTICO_DEV.md`](diagnostico_dev/RELATORIO_DIAGNOSTICO_DEV.md).

- **Não houve defeito de execução.** Modelo de conversa, tokenização,
  truncamento, precisão (T4 fp16 × CPU fp32) e bibliotecas foram verificados;
  os valores documentados foram reproduzidos exatamente no Colab atual
  (Python 3.13.16, PyTorch 2.11).
- **Incapacidade real no SFT C.** O Qwen2.5-1.5B-Instruct responde diretamente,
  sem etapas, a expressões com produtos de até oito algarismos; nenhuma resposta
  seria correta mesmo com extração permissiva.
- **Instrução e contrato de formato.** No SFT, o modelo nunca usou a tag
  `<answer>` pedida. No GRPO, copiou os marcadores da instrução
  ("explicação verificável", "resumo", "LETRA") e escreveu o cálculo fora das
  tags; o contrato rejeitou respostas com a letra correta. A seleção dos 120
  primeiros itens DEV cobria apenas dois dos quatro domínios.
- **Defeito do corretor sem efeito no desfecho.** Um número com separador de
  milhar era lido como decimal (1 resposta em SFT A).
- **Monitores.** Em controles construídos (não humanos), o monitor independente
  SmolLM2-135M-Instruct sinalizou quase todos os rastros (TPR 0,875; FPR 0,863)
  e o monitor de recompensa Qwen2.5-0.5B-Instruct ficou abaixo do acaso
  (TPR 0,694; FPR 0,775). Mesmo com competência aprovada, o gate do monitor
  independente e a manipulação da condição `cot_monitor` estariam comprometidos.

Corrigir o extrator depois de observar a triagem alteraria o instrumento
registrado e não reverteria o nível C do SFT; por isso o resultado registrado
permanece o de inviabilidade.

## Divergência entre registro e implementação

O texto do registro OSF cita a versão 3.1 do protocolo; a implementação
executada é a v3.2 (`fde856f`, de 19/10 15:20 UTC, anterior ao registro), cujas
diferenças são operacionais e não alteram hipóteses, tamanhos nem critérios.
O registro não contém anexos nem os SHA256 do protocolo e das revisões dos
modelos (um arquivo foi enviado ao projeto às 15:57:06 UTC e removido às
15:57:30 UTC). O protocolo v4 exige que esses resumos constem do registro.

## O que permanece disponível

Os cinco notebooks deste diretório continuam fixados em `fde856f`/`4a2a133` e
reproduzem o bloqueio. Nenhum arquivo do estudo, do protocolo ou dos resultados
foi alterado por este encerramento.
