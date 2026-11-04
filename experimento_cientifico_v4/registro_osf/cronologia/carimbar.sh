#!/usr/bin/env bash
# Gera MANIFESTO_CRONOLOGIA.txt, arquiva os horários de push do GitHub e pede
# carimbos RFC 3161 à FreeTSA e à DigiCert. Executar a partir da raiz do
# repositório, antes do registro no OSF, sempre que um arquivo listado mudar.
set -euo pipefail
D=experimento_cientifico_v4/registro_osf/cronologia
V=experimento_cientifico_v4
DAY=$(date -u +%Y-%m-%d)
ACTIVITY=$D/github_activity_$DAY.json

curl -sSf --max-time 60 -H 'Accept: application/vnd.github+json' \
  "https://api.github.com/repos/Kelvin-Marques-Cyber/cot-obfuscation-h2a/activity?per_page=100" -o "$ACTIVITY"

{
  echo "# Manifesto de cronologia do estudo-prospectivo-2026-v4"
  echo "# Gerado em $(date -u +%Y-%m-%dT%H:%M:%SZ), antes do registro público no OSF."
  echo "# Commits Git (cada SHA compromete todo o histórico anterior, com conteúdo e datas):"
  while read -r sha role; do
    git cat-file -e "$sha^{commit}"
    echo "commit $sha  $role"
  done <<'EOF'
37094c4afd4fe2983ca56322750a8f1359f12bbb protocolo_congelado_antes_da_triagem
aef52d9369d0b3f3013f91da06a6f4869daea6b1 triagem_DEV_executada
b858d128bcd545dd958e0dbed9978a7aea2591b3 codigo_fixado_nos_notebooks
4b7d1fd1aa9c5014af32f4b251e353cceb45e02d notebooks_fixados
f2ef2707a7997bfe7c07e7c4949a9d4934ff0ab3 main_kit_de_registro_OSF
EOF
  echo "# SHA256 dos arquivos (caminhos relativos à raiz do repositório):"
  sha256sum $V/PROTOCOLO_v4.md $V/protocolo_v4.json $V/analises_secundarias.py \
    $V/triagem_dev/manifest.json $V/triagem_dev/dev_*_base.jsonl \
    $V/registro_osf/pacote_registro_v4.zip $V/registro_osf/codigo_v4_b858d12.zip \
    $V/registro_osf/REGISTRO_OSF_v4.md $V/validacao/cuda_execucao3_a100/relatorio_tecnico.json \
    "$ACTIVITY"
} > $D/MANIFESTO_CRONOLOGIA.txt

cd $D
openssl ts -query -data MANIFESTO_CRONOLOGIA.txt -sha256 -cert -out MANIFESTO_CRONOLOGIA.tsq
curl -sSf --max-time 60 -H 'Content-Type: application/timestamp-query' --data-binary @MANIFESTO_CRONOLOGIA.tsq \
  https://freetsa.org/tsr -o MANIFESTO_CRONOLOGIA.freetsa.tsr
curl -sSf --max-time 60 -H 'Content-Type: application/timestamp-query' --data-binary @MANIFESTO_CRONOLOGIA.tsq \
  http://timestamp.digicert.com -o MANIFESTO_CRONOLOGIA.digicert.tsr
curl -sSf --max-time 60 https://freetsa.org/files/cacert.pem -o freetsa_cacert.pem
curl -sSf --max-time 60 https://freetsa.org/files/tsa.crt -o freetsa_tsa.crt
openssl ts -verify -data MANIFESTO_CRONOLOGIA.txt -in MANIFESTO_CRONOLOGIA.freetsa.tsr \
  -CAfile freetsa_cacert.pem -untrusted freetsa_tsa.crt
openssl ts -verify -data MANIFESTO_CRONOLOGIA.txt -in MANIFESTO_CRONOLOGIA.digicert.tsr \
  -CAfile /etc/ssl/certs/ca-certificates.crt
for t in freetsa digicert; do
  echo "$t: $(openssl ts -reply -in MANIFESTO_CRONOLOGIA.$t.tsr -text | grep 'Time stamp')"
done
