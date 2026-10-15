# Laboratório GPU

Os notebooks 02 a 04 foram preparados e tiveram o código Python compilado; as
execuções CUDA não foram realizadas no ambiente desta revisão. Comece pelo smoke
test. Não transforme resultados de teste de instalação em conclusão científica.

## Ordem

1. 01_reanalise_reprodutivel.ipynb: reproduz as tabelas históricas e a auditoria CPU.
2. 02_piloto_sft_corrigido.ipynb: executa o contraste de alvos supervisionados,
   com IDs, partições fixas e saídas integrais.
3. 03_grpo_multiescala.ipynb: monta e executa células independentes de GRPO com
   recompensa comportamental e penalização de rastro ou resumo.
4. 04_avaliacao_monitores_anotacao.ipynb: avalia checkpoints, modos de resposta,
   monitor independente e anotação cega, sem alterar o artigo automaticamente.

O notebook 01 funciona em CPU. Nele, mantenha INSTALL_DEPS = True na primeira
execução. Para os treinos dos notebooks 02 e 03 e a inferência do notebook 04,
ative GPU. Monte um diretório persistente antes de definir RUN_ROOT.
Em Vast.ai/Jupyter use um volume persistente. O diretório de execução contém
checkpoints, estado do otimizador, adaptador final, config e hash dos dados.
Rode novamente a mesma configuração para retomar o último checkpoint.

## Dependências e hardware

Instale uma distribuição de PyTorch/CUDA compatível com a sua GPU. As instruções
oficiais estão em https://pytorch.org/get-started/locally/. O instalador dos
notebooks preserva o torch existente. Em GPUs Blackwell, confirme que o binário
instalado suporta sua arquitetura. Não há promessa de VRAM suficiente para todo
modelo ou orçamento: faça primeiro a configuração pequena.

TRL e Transformers estão fixados no par de versões documentado no repositório.
Depois da instalação, reinicie o kernel se já tiver importado essas bibliotecas.
Execute pip check e salve pip freeze junto à execução.

Não se usa Unsloth, xformers ou instalação indiscriminada com --no-deps neste
laboratório novo. O notebook histórico é preservado sem alteração.

## Semântica das condições

O SFT compara três alvos: explicit, direct e generic. Não se chama isso de
supervisão exclusiva de saída.

O GRPO usa recompensa mal especificada para selecionar uma alternativa errada
indicada por informação auxiliar. A resposta objetiva e a alternativa explorável
são distintas por construção. As condições none, cot_monitor e summary_monitor
compartilham esse incentivo. Na última, o cliente do monitor recebe somente a
questão objetiva e o resumo; o teste automatizado verifica que o rastro não é
encaminhado ao monitor de recompensa.

Os datasets GRPO são sintéticos, com quatro domínios de superfície e dificuldade
controlada da dica. Não são uma replicação dos quatro conjuntos do artigo-alvo,
nem tarefas reais clínicas/jurídicas. low/medium/high são proxies; a classificação
causal exige avaliação no modelo base. O programa completo requer também dados
e rubricas de especialistas, conforme o protocolo.

## Monitor remoto

Use um servidor compatível com chat/completions. Configure em variáveis de
ambiente, sem gravar credenciais em células ou commits:

    MONITOR_BASE_URL=http://127.0.0.1:8000/v1
    MONITOR_MODEL=<modelo servido>
    MONITOR_REVISION=<revisão realmente servida>
    MONITOR_API_KEY=<se necessário>

Modelo e revisão são registrados; a revisão é uma declaração do operador do
endpoint. Ela deve corresponder ao peso realmente servido. Há cache SQLite
separado por endpoint/modelo/revisão/texto. Respostas inválidas e falhas de rede
interrompem o treino; não há substituição por recompensa zero ou heurística.
O operador é responsável por custos e política do endpoint escolhido.

## Limites e falhas

As células de execução transmitem stdout e stderr e gravam logs completos em
logs/. Em caso de falha, a exceção inclui as últimas linhas da etapa. Esse
diretório fica fora dos commits para preservar registros locais de execução.
As variáveis de ambiente são relidas a cada etapa; configure os monitores antes
de gerar o plano de treinamento.

Use um processo por GPU e uma configuração por subprocesso. Execuções falhas
produzem status failed; não são descartadas silenciosamente. Nunca misture
avaliações de budgets diferentes na mesma análise.

O modo early força uma resposta antes de um rastro livre. O modo wrong_step
introduz um prefixo incorreto verificável. O orçamento contabiliza também o
prefixo forçado. Essas ablações não constituem, sozinhas, identificação causal
conclusiva. Reporte erros de formato e truncamentos em todos os modos.

O monitor independente deve ter identidade distinta do monitor de treino. Use
saídas válidas e inválidas separadas. As métricas comportamentais não são
anotações de intenção. A exportação cega gera uma chave que deve permanecer
separada dos formulários enviados aos avaliadores.

## O que versionar

Versione código, configurações sem credenciais, manifestos, métricas, saídas
sintéticas e documentos. Não envie automaticamente caches, pesos ou tokens.
As pastas runs, dados de grande porte e checkpoints estão ignoradas no Git.
Os notebooks não fazem push nem upload de pesos por conta própria.
