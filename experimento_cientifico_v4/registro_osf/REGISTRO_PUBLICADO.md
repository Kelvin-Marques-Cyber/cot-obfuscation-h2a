# Registro público do v4

**https://osf.io/hb64a/** · OSF Preregistration · registrado em 04/11/2026 às 13:01:35 UTC,
público desde o registro, sem embargo, no projeto https://osf.io/7hzjb/.

Conferência feita pela API pública do OSF logo depois do registro:

- título, descrição e os 26 campos de texto e de escolha são idênticos ao rascunho salvo em
  `rascunho_osf_enviado.json`, preenchido a partir de `REGISTRO_OSF_v4.md`;
- os quatro anexos foram arquivados no registro com os SHA256 da tabela do roteiro:
  `pacote_registro_v4.zip` (af36a871…), `codigo_v4_b858d12.zip` (773001fe…),
  `analises_secundarias.py` (6065d011…) e `PROTOCOLO_v4.md` (b89537a3…).

O registro é posterior ao congelamento do protocolo (29/10/2026 11:28 UTC), à triagem DEV
(11:52 UTC) e aos carimbos de tempo da pasta `cronologia/` (29/10/2026 16:49:08 UTC), e
anterior a qualquer treinamento do estudo. A URL é declarada ao runtime pelo notebook 00
(`PUBLIC_REGISTRATION_URL`) ou por `validacao/executores_colab/coleta_v4.py`.

## Atualização 1 (desvio 1), publicada em 05/11/2026

Atualização do registro pelo fluxo "Updates" do OSF (resposta `6aec883f1dc5d8a4c80d61d8`),
submetida e aprovada pelo único administrador em 05/11/2026 às 13:44:53 UTC, com o texto do
fim de `../DESVIO_01.md`:

- **justificativa:** o parágrafo "Desvio 1, declarado em 04/11/2026 antes de qualquer geração
  TEST […]";
- **único campo alterado:** 344-47 ("Data collection procedures"), em que "sempre com o código do
  commit b858d12…" passou a "com o código do commit b858d12…, alterado somente no guarda de
  valores não finitos do treino pelo desvio 1 (commit 9066d81…; ver DESVIO_01.md no
  repositório)". Os outros 28 campos são idênticos aos da resposta original.

A atualização foi publicada depois da coleta, mas o desvio que ela descreve foi declarado,
carimbado (`cronologia/MANIFESTO_DESVIO_01.*`, 04/11/2026 14:21:10 UTC) e enviado ao GitHub
antes de qualquer geração TEST. Resposta da API: `atualizacao_01_osf.json`.
