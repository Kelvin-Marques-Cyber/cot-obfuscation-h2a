# Interrupção da coleta por perda da VM (04/11/2026)

Incidente operacional, sem alteração de protocolo, código, dados ou parâmetros. Registrado
antes de qualquer geração TEST.

## O que aconteceu

| Horário (UTC) | Evento |
|---|---|
| 16:54 | Coleta iniciada na VM `v4d` (NVIDIA A100-SXM4-80GB), código dos notebooks `15f1bfa`, registro hb64a |
| 21:16 | Termina a verificação periódica (uma célula executada a cada 10 minutos); a VM fica sem atividade de cliente |
| 22:08:11 | Último `status.json` gravado no Drive: treino 107 (nível C, `direct`, semente 128) concluído |
| antes de 22:19 | O Colab recolhe a VM `v4d` por inatividade; o processo da coleta termina com ela |
| 22:41 a 23:14 | Uma VM nova (`v4e`) é recolhida antes de montar o Drive; nenhum arquivo do estudo foi tocado |
| 23:24 | VM `v4f` (A100-SXM4-80GB) com o Drive montado |
| 23:28 | Retomada: notebook 00 aprovado; o notebook 01 para na verificação de `runtime.job_index` (abaixo) |

A VM do Colab só continua ativa enquanto um cliente executa células; o processo da coleta
roda em segundo plano e não conta como atividade. A verificação periódica era o que mantinha a
VM; sem ela, a VM foi recolhida cerca de 50 minutos depois.

## Efeito sobre os arquivos do estudo

O Drive montado na VM envia os arquivos de forma assíncrona. No treino 107
(`0a465f6cbebfabb33a85`), o `status.json` (pequeno, gravado por último) chegou ao Drive, mas o
adaptador e os checkpoints não: a pasta `adapter/` não existe e as pastas `checkpoint-*` estão
vazias. O `status.json` registra o SHA256 do adaptador produzido na `v4d`:
`cee192f2d3e4957e3634ee0f205cb4b85c4b4dfd98d9bf5200580fd5a5c6dec2`.

O runtime recusou esse treino ("Treino concluído incompatível com o protocolo"), como previsto:
ele exige o adaptador e confere o SHA256. Todos os outros 106 treinos concluídos foram
conferidos um a um na `v4f`: `adapter_config.json` presente e SHA256 do adaptador igual ao do
`status.json` (`verif_adaptadores_antes_retomada_v4f.json`). Nenhuma avaliação TEST havia
começado.

## O que foi feito

1. A pasta incompleta foi movida, sem alteração, para
   `prospectivo_v4/falhas_coleta/sincronizacao_v4d_0a465f6cbebfabb33a85/` no Drive (cópias do
   `status.json` e do `manifest.json` aqui, com prefixo `run_0a465f6cbebfabb33a85_`).
2. A coleta foi retomada com o mesmo `coleta_v4.py`, o mesmo commit e a mesma URL. O runtime
   refaz o treino C · `direct` · 128 do início, com os mesmos parâmetros (o diretório tem o mesmo
   `run_id`, derivado da configuração).
3. **Verificação:** como o treino na A100 é determinístico (desvio 1: cinco adaptadores
   refeitos com SHA256 idênticos), o adaptador refeito deve ter o SHA256 acima. O resultado é
   relatado abaixo qualquer que seja; se diferir, vale o adaptador refeito e a diferença é
   relatada como não determinismo.
4. Daqui em diante, uma célula é executada a cada 5 minutos até o fim da coleta, e o log e os
   notebooks executados são copiados para o Drive (`prospectivo_v4/coleta_logs/`), pois os da
   `v4d` ficavam no disco local e se perderam com ela. Os registros do runtime no Drive
   (`logs/`, `operations/`, `runs/`) não foram afetados.

## Resultado da verificação

O treino refeito na `v4f` (concluído antes de 23:57 UTC de 04/11/2026) produziu um adaptador
com SHA256 `cee192f2d3e4957e3634ee0f205cb4b85c4b4dfd98d9bf5200580fd5a5c6dec2`, **idêntico** ao
registrado na `v4d`, com as mesmas perdas inicial e final (1,2812 e 0,574) e nenhum passo
descartado. O conteúdo do estudo é, portanto, o mesmo que teria resultado sem a interrupção.

## Arquivos

- `estado_antes_retomada_v4f.json`: estado de cada treino no Drive antes da retomada.
- `verif_adaptadores_antes_retomada_v4f.json`: conferência dos adaptadores (107 treinos, 1 com problema).
- `run_0a465f6cbebfabb33a85_status.json` e `run_0a465f6cbebfabb33a85_manifest.json`: o treino afetado.
- `exec_v4f_tentativa1.log`: a primeira retomada, que parou na verificação do runtime.
