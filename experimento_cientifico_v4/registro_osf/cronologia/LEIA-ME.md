# Cronologia do v4 com prova independente do Git

Datas de commit são escritas por quem faz o commit e podem ser alteradas. Esta
pasta reúne duas provas que não dependem delas, obtidas em 29/10/2026, antes do
registro público no OSF.

## 1. Horários de push gravados pelo GitHub

`github_activity_2026-10-29.json` é a resposta bruta de
`GET https://api.github.com/repos/Kelvin-Marques-Cyber/cot-obfuscation-h2a/activity`
(obtida às 16h49 UTC). O campo `timestamp` de cada evento é gravado pelo
servidor do GitHub no momento do push. Trechos relevantes:

| Push (UTC) | Commit que passou a existir no GitHub | Significado |
|---|---|---|
| 29/10 10:49:49 | `ad6b434` | script de congelamento e regra declarada para monitores inviáveis |
| 29/10 11:28:14 | `37094c4` | protocolo v4 congelado |
| 29/10 11:52:04 | `aef52d9`, `6c41277` | resultado da triagem DEV (execução única) e notebooks fixados |
| 29/10 12:05:50 | `4b7d1fd` (inclui `b858d12`) | correção da retomada; código fixado no registro |
| 29/10 12:08:12 e 12:16:23 | `266045d`, `8459c4c` | registros dos testes técnicos na T4 |
| 29/10 16:13:23 | `f2ef270` (`main`) | kit de registro no OSF |
| 29/10 16:42:26 | `4114698` (hoje `6a770fa`, ver abaixo) | teste técnico na A100 e registro com a classe de GPU |

A ordem protocolo congelado (11:28:14) → resultado da triagem (11:52:04) está,
portanto, registrada pelo servidor, e não apenas pelo histórico local. A API
devolve só eventos recentes; por isso a resposta foi arquivada aqui e coberta
pelos carimbos do item 2.

## 2. Carimbos de tempo RFC 3161

`MANIFESTO_CRONOLOGIA.txt` lista os SHAs dos commits congelados e o SHA256 dos
arquivos do protocolo, da triagem, do pacote de registro, do texto do registro,
do teste técnico na A100 e da resposta do GitHub acima. Um SHA de commit
compromete todo o histórico anterior a ele, com conteúdo e datas. O manifesto
foi carimbado por duas autoridades independentes:

| Arquivo | Autoridade | Horário atestado |
|---|---|---|
| `MANIFESTO_CRONOLOGIA.freetsa.tsr` | FreeTSA (freetsa.org) | 29/10/2026 16:49:08 UTC |
| `MANIFESTO_CRONOLOGIA.digicert.tsr` | DigiCert (timestamp.digicert.com) | 29/10/2026 16:49:08 UTC |

Os carimbos provam que o manifesto, e portanto todos esses commits e arquivos,
já existiam nesse horário. A data do registro no OSF será posterior. Assim, a
triagem e o congelamento do protocolo ficam comprovadamente anteriores ao
registro sem depender das datas do Git.

`carimbar.sh` refaz tudo (resposta do GitHub, manifesto, carimbos e verificação) a
partir da raiz do repositório; deve ser executado de novo se algum arquivo
listado mudar antes do registro.

## Reescrita de autoria (04/11/2026)

Os commits da branch de trabalho posteriores a `f2ef270` foram reescritos para conter somente
a identidade do proprietário. Nenhum commit da `main`, e portanto nenhum SHA do manifesto
acima, foi alterado. `reescrita_autoria_2026-11-04.json` lista cada commit original, o commit
reescrito, a árvore (idêntica nos dois, ou seja, o mesmo conteúdo) e o horário de push do
original segundo `github_activity_2026-11-04.json`. Os dois arquivos foram carimbados pela
FreeTSA e pela DigiCert (`MANIFESTO_REESCRITA.*`, 04/11/2026 12:48:47 UTC). Assim continua
comprovado, por exemplo, que as regras da rodada 3 dos monitores (`5952a88`, hoje `84e747e`)
chegaram ao GitHub às 11:45:11 UTC, antes da execução.

## Como verificar

```bash
cd experimento_cientifico_v4/registro_osf/cronologia
openssl ts -verify -data MANIFESTO_CRONOLOGIA.txt -in MANIFESTO_CRONOLOGIA.freetsa.tsr -CAfile freetsa_cacert.pem -untrusted freetsa_tsa.crt
openssl ts -verify -data MANIFESTO_CRONOLOGIA.txt -in MANIFESTO_CRONOLOGIA.digicert.tsr -CAfile /etc/ssl/certs/ca-certificates.crt
openssl ts -reply -in MANIFESTO_CRONOLOGIA.digicert.tsr -text | grep "Time stamp"
```

Os certificados da FreeTSA podem ser baixados de novo em
https://freetsa.org/files/cacert.pem e https://freetsa.org/files/tsa.crt; o da
DigiCert está nos repositórios de certificados dos sistemas operacionais. Para
conferir o manifesto, recalcule os SHA256 com `sha256sum` a partir da raiz do
repositório e confira que cada commit listado existe (`git cat-file -t SHA`).
