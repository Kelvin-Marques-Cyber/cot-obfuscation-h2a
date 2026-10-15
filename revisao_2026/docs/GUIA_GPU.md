# Executar no Colab

Abra o notebook pelo link do README e use **Ambiente de execução → Executar
tudo**. No 01, CPU basta. Nos 02, 03 e 04, selecione GPU T4 no tipo de ambiente
e autorize o acesso ao Drive quando solicitado. Execute um notebook por vez.

Não é necessário editar comandos de instalação, ativar flags de treino ou
avaliação, copiar arquivos entre notebooks nem configurar uma API. O código é
baixado do commit fixado no próprio notebook, evitando reutilizar clones antigos
do Drive. As dependências são instaladas em um ambiente separado do kernel.
O ambiente GPU reutiliza o PyTorch/CUDA fornecido pelo Colab.

## Ordem

| Notebook | O que executa ao usar Executar tudo |
|---|---|
| 01 | Instalação CPU, testes, reanálise histórica, tabelas, gráficos e ZIP |
| 02 | Dados disjuntos, três alvos SFT, avaliação base/adaptador, análise e ZIP |
| 03 | Dados sintéticos, três condições GRPO, monitores locais, avaliação e ZIP |
| 04 | Completa etapas pendentes do 03, monitor independente, análise, formulários e ZIP |

O 04 pode ser executado em uma nova sessão. Se o plano do mesmo perfil não tiver
sido concluído, os treinos e avaliações pendentes são executados automaticamente.

## Tamanho da execução

A primeira célula já vem configurada com `PROFILE = "quick"`. Não é necessário
mudar nada para o teste curto. As opções do formulário permitem ampliar depois:

| Perfil | SFT | GRPO | Finalidade |
|---|---|---|---|
| quick | 3 treinos, modelo 0,5B, 2 passos por treino | 3 treinos, modelo 0,6B, 2 passos por treino | Verificar infraestrutura e fluxo completo |
| pilot | 27 treinos, 30 passos por treino | 27 treinos, 30 passos por treino | Exploração com 3 sementes e 3 níveis |
| full | 108 treinos, modelo 7B, 200 passos por treino | 270 treinos, modelos 4B/14B/8B, 4.000 passos por treino | Grade prospectiva de maior porte |

Os perfis têm dados e diretórios separados. Escolha o mesmo perfil nos notebooks
03 e 04 para avaliar os mesmos modelos. O perfil full pode exigir várias sessões
e GPUs maiores. Modelos restritos, como Llama, dependem da licença e autenticação
da conta do Hugging Face; essa autorização não pode ser concedida pelo notebook.

O teste quick não é uma demonstração de ofuscação nem a conclusão do programa
experimental do artigo. Poucos passos e modelos pequenos podem gerar muitas
saídas inválidas ou recompensas sem variação. Isso permanece visível nas saídas
e nos registros do treinador; não se converte em evidência de hipótese ausente.

## Onde ficam os arquivos

Pasta padrão: `Meu Drive/CoT_Obfuscation_Lab/colab_v3/`.

- `reanalysis/`: resultados e figuras do notebook 01.
- `quick/`, `pilot/` ou `full/`: dados, planos congelados, execuções e análises.
- `runs/sft/` e `runs/grpo/`, dentro do perfil: checkpoints, adaptadores e gerações.
- `annotation/`, dentro do perfil: formulário cego e chave privada separada.
- `logs/`: stdout e stderr completos de cada etapa.
- `downloads/`: ZIP final; o notebook também inicia seu download.

O ZIP contém métricas, gerações, configurações, figuras e logs. Pesos,
checkpoints e caches de monitores permanecem no Drive e não entram no ZIP.
Desmarcar o salvamento no Drive usa armazenamento temporário; nesse caso,
baixe o ZIP antes de encerrar a sessão.

## Retomar

Abra novamente o mesmo notebook e execute tudo com o mesmo perfil. Revisões do
ator e dos monitores são reutilizadas do plano congelado. Treinos concluídos
são ignorados; treinos interrompidos retomam o último checkpoint. A avaliação
pula itens já registrados e recupera uma última linha interrompida.

Dados existentes nunca são substituídos por um plano diferente. Uma divergência
interrompe a etapa e preserva os arquivos para inspeção. Os logs incluem o erro
original, além do código de saída.

## Monitores sem API

O padrão GRPO usa `Qwen/Qwen2.5-0.5B-Instruct` em CPU, por comparação de
verossimilhança das classes `false` e `true`. O controle não consulta o monitor;
a condição de rastro encaminha apenas o rastro; a condição de resumo encaminha
apenas o resumo. Revisões dos pesos e cache são registrados.

O 04 usa `HuggingFaceTB/SmolLM2-135M-Instruct`, distinto do ator e do monitor da
recompensa, para pontuar o modo full. As cinco modalidades de resposta têm
métricas comportamentais; a inspeção independente automática é limitada ao full.

Esses classificadores pequenos são instrumentos exploratórios sem validação
humana, com limitações de idioma e sensibilidade. Não estimam intenção nem
substituem a calibração e os juízes previstos no protocolo. O monitor remoto
continua disponível para execução por configuração/CLI, sem fallback silencioso.

## Anotação humana

Distribua cópias de `blinded_annotations.csv` e preserve
`private_annotation_key.csv` em separado. A chave diferencia modelo base e
adaptador para o mesmo item. Reúna as respostas em `completed_annotations.csv`,
na mesma pasta, com as colunas `annotation_id`, `rater` e `label`.
Reexecutar o 04 calcula a concordância. Sem respostas humanas, o estado é
`not_collected`, sem rótulos inventados. O PDF não é alterado automaticamente.

## Jupyter e Vast.ai

Abra os notebooks dentro de um clone do repositório. O código usa esse clone e
salva em `~/CoT_Obfuscation_Lab/colab_v3/`. Para um volume persistente específico,
defina `COT_STORAGE_ROOT` antes de iniciar o kernel. GPU/PyTorch CUDA devem estar
disponíveis na imagem escolhida. A instalação das demais dependências é automática.

## Verificação de software

A suíte CPU cobre instrumentos, planos, integridade dos dados, retomada de
revisões, logs e identificação de anotações. Um teste separado executa as APIs
reais de SFT/GRPO com uma miniatura Qwen2 inicializada aleatoriamente em CPU,
sem baixar modelos: `PYTHONPATH=src COT_TEXT_ONLY=1 python tests/integration_trainers.py`.
Esse teste exige as dependências dos treinadores. Não mede qualidade de modelos.
Treino quantizado e inferência CUDA com os pesos públicos precisam de execução
no runtime GPU do usuário; não foram realizados no ambiente desta revisão.
