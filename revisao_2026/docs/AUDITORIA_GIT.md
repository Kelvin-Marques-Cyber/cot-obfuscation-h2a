# Auditoria de autoria e de SHAs congelados — 28/10/2026

## Escopo

Verificar se o histórico contém autoria, coautoria ou trailers automáticos de
assistentes, removê-los se existissem sem perder trabalho, e garantir que os
commits fixados por notebooks e protocolos continuem recuperáveis.

## Estado encontrado

- `main` remoto em `273d13192a17a5fc22594dbe10df18bdc7ff9a21`, idêntico ao
  local; nenhum commit posterior à referência informada.
- 23 commits. Autor e *committer* de todos: `Kelvin e Silva Marques
  <148489077+Kelvin-Marques-Cyber@users.noreply.github.com>`.
- Nenhuma mensagem contém `Co-authored-by`, `Generated with`, `Claude`,
  `Anthropic` ou assinatura de ferramenta; nenhum trailer Git
  (`git log --format=%(trailers)` vazio em todos os commits).
- Nenhum arquivo versionado menciona Claude, Anthropic, Codex, ChatGPT,
  OpenAI ou Copilot.
- *Contributors* calculados pelo GitHub (`GET /repos/.../contributors`):
  somente `Kelvin-Marques-Cyber` (23 contribuições).
- Colaboradores com permissão de acesso: somente `Kelvin-Marques-Cyber`
  (admin). Aplicativos GitHub instalados não são colaboradores e não foram
  alterados.

## Decisão

Não havia registro automático de autoria a remover; portanto **nenhuma
reescrita de histórico foi feita**. A reescrita só criaria risco para os SHAs
fixados. Os commits novos desta revisão usam a mesma identidade do
proprietário, sem assinatura de ferramenta e sem trailers de coautoria
(`git log --format='%an|%cn|%(trailers)' 273d131..main`). O branch de trabalho
usado durante a revisão foi incorporado à `main` por avanço rápido
(*fast-forward*) e excluído, e o *pull request* de rascunho aberto para ele foi
fechado sem *merge*.

Achado durante a revisão: dois arquivos de ambiente do diagnóstico
(`experimento_cientifico_v3/diagnostico_dev/resultados/*_float32_environment.json`)
registraram o caminho absoluto do diretório temporário da sessão de trabalho,
que inclui o nome da ferramenta de execução. O campo passou a registrar o
commit congelado (`fde856f…`) e o script deixou de gravar caminhos locais
(commit `495fdce`). As versões anteriores permanecem nos commits `4b28bd6` e
`0da71f2`: reescrevê-los mudaria os SHAs dos commits seguintes, entre eles os
que comprovam que cada regra de calibração do v4 foi gravada antes da execução
correspondente. Essa prova de precedência foi considerada mais importante que a
remoção de um caminho de arquivo que não constitui autoria.

## SHAs dependentes

| SHA | Uso | Recuperação em clone limpo |
|---|---|---|
| `fde856f895c841274b93b34f4675642508af102f` | código científico v3.2 (notebooks e protocolo) | ok, inclusive busca rasa por SHA |
| `4a2a133aff6638c910661b01821eeb43b72b4db4` | suporte Colab v3.2 | ok, inclusive busca rasa por SHA |
| `62b2d26dafc8417af7fe3d245188b055a638fad0` | notebooks do laboratório | ok, inclusive busca rasa por SHA |
| `4b498226f0ce5a302e32dbf773a99365302149c2` | testes do v3 | ok |
| `47d935f2bbfeb2638c0a734f4be91ac940df9c10` | testes do v3 | ok |
| `e21cef8ad92369c56f3fcbbb6891fa49bda8c74b` | procedência do piloto | ok |

O teste reproduz o que os notebooks fazem: `git init`, `git fetch --depth 1
origin <SHA>`, `git checkout --detach <SHA>` e conferência do `HEAD` e do
SHA256 de `scientific_runtime.py` (`0ccc8cd6…`). Os notebooks v4 fixam
commits publicados nesta revisão, verificados da mesma forma no Colab.

## Limites

Cópias, *forks*, caches de terceiros e históricos de eventos do GitHub estão
fora do controle deste repositório e não foram, nem poderiam ser, alterados.
