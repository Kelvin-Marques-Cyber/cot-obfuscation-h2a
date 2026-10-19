# Experimento científico CoT — v3.2

Este é o conjunto de cinco notebooks do estudo prospectivo, incluindo o
protocolo. A implementação preserva 20 sementes, três condições, dois níveis
e 120 passos por treino: **120 treinos SFT + 120 GRPO** e **55.200 gerações TEST**.

| Ordem | Notebook no Colab | Ambiente |
|---|---|---|
| 00 | [Protocolo e datasets](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/experimento_cientifico_v3/00_protocolo_e_datasets.ipynb) | CPU |
| 01 | [Treinamento SFT replicado](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/experimento_cientifico_v3/01_treinamento_sft_replicado.ipynb) | GPU |
| 02 | [Treinamento GRPO replicado](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/experimento_cientifico_v3/02_treinamento_grpo_replicado.ipynb) | GPU |
| 03 | [Competência e teste cego](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/experimento_cientifico_v3/03_competencia_e_teste_cego.ipynb) | GPU |
| 04 | [Monitor independente e inferência](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/experimento_cientifico_v3/04_monitor_independente_e_inferencia.ipynb) | GPU |

## Executar no Colab

1. Abra **00**, deixe CPU, escolha **Ambiente de execução → Executar tudo**
   e autorize o Drive. O notebook congela dados, código e revisões dos dois
   atores e dos dois monitores. Baixa `pacote_para_pre_registro.zip`.
2. Publique o pacote em um registro público com data auditável antes dos
   treinos. Volte ao 00, preencha `PUBLIC_REGISTRATION_URL` e execute tudo.
   Informar uma URL não certifica por si só a cronologia do registro.
3. Abra **01**, selecione GPU T4 ou superior e execute tudo. A triagem DEV
   antecede os treinos; exige 65% de acurácia por nível nos dois modelos-base.
   Falhar bloqueia o estudo. O notebook conclui até dois jobs por sessão.
   Reexecute mantendo o mesmo estudo até completar 120 treinos SFT.
4. Abra **02** em GPU. Reexecute até completar 120 treinos GRPO; o padrão
   executa um por sessão. `MAX_JOBS_PER_SESSION` controla apenas a duração
   da sessão, sem alterar o protocolo.
5. Depois dos 240 treinos, execute **03** em GPU. Reexecute até concluir
   as 120 avaliações de cada método. Apenas `TEST/full` entra no desfecho
   primário, com 256 tokens por resposta.
6. Execute **04** em GPU repetidamente para pontuar as 120 avaliações GRPO
   com o monitor independente. Ele exporta uma amostra cega de 360 itens.
   Dois avaliadores preenchem os arquivos descritos no protocolo. Salve-os
   em `human/` no diretório do estudo e reexecute o 04 para o relatório.

### Quando a triagem DEV falhar (65%)

Os notebooks 01 e 02 fazem um diagnóstico **em CPU** antes de inicializar a GPU.
Se as quatro avaliações DEV já foram concluídas e alguma ficou abaixo de
65%, os notebooks reutilizam os JSONL existentes, conferem os SHA256 e a
contagem de acertos, mostram exemplos incorretos e **não iniciam a GPU nem
os treinos confirmatórios**. A interrupção será explicativa, em vez de um
`RuntimeError` longo. Os arquivos originais não são alterados.

Quando ainda falta um nível DEV, o notebook gera **somente os resultados
faltantes** e faz nova conferência. Se o critério científico não for atingido,
os 240 treinos permanecem bloqueados conforme o protocolo v3.2 e o OSF.
Não é permitido reduzir o limiar, retreinar para escolher uma triagem favorável
ou apagar avaliações DEV. Se o problema for incompetência real do modelo-base,
é necessário **outro estudo prospectivo** com mudanças previamente registradas.

**Exemplo já observado na triagem**: SFT-A 57,5%; SFT-C 0%; GRPO-low
43,33%; GRPO-high 15%. Nenhum nível atingiu 65%. As advertências sobre
`torch_dtype` e parâmetros de amostragem não são a causa desse bloqueio.

**Importante:** reabra [Notebook 01](https://colab.research.google.com/github/Kelvin-Marques-Cyber/cot-obfuscation-h2a/blob/main/experimento_cientifico_v3/01_treinamento_sft_replicado.ipynb)
pelo link do GitHub. Cópias antigas do Colab **não** recebem atualizações.
Mantenha o `STUDY_ID` original para preservar os arquivos já gerados.

Os arquivos ficam em:

`Meu Drive/CoT_Obfuscation_Lab/confirmatory_v3/estudo-principal-2026-v3-2/`

Mantenha o mesmo `STUDY_ID` nos cinco notebooks. Reexecutar retoma o trabalho;
trocar o ID inicia outro estudo. A v3.2 não altera diretórios v3.1 existentes.
Abra os links atualizados: cópias antigas salvas no Colab continuam usando
sua revisão anterior.

## Se a montagem do Drive falhar

Os cinco notebooks tratam `ValueError: mount failed`: fazem no máximo duas
tentativas, verificam o ponto de montagem e testam gravação antes de declarar
persistência. Um diretório `MyDrive` antigo não basta. A recuperação não apaga
arquivos, não força desmontagem e não altera estudos existentes.

Se o Drive permanecer indisponível, **00 termina a preparação localmente**
e oferece o ZIP. `STORAGE_MODE="local"` permite pular a conexão para essa
preparação. Os arquivos em `/content` são temporários: gere o pacote final
com o Drive conectado antes de registrar o estudo. Os notebooks 01–04
mostram a pendência e aguardam armazenamento persistente para executar
inferência e treinamento; não iniciam uma coleta temporária silenciosamente.

Para tentar a conexão de novo, mantenha `STORAGE_MODE="auto"` e reexecute
a primeira célula. Depois execute novamente o 00. Caso o download automático
não abra, o ZIP continua disponível no caminho impresso: use o painel
**Arquivos** do Colab para baixá-lo.

`SOURCE_COMMIT` mantém a implementação científica em `fde856f895c841274b93b34f4675642508af102f`.
`SETUP_COMMIT` fixa separadamente o suporte operacional atualizado. Assim,
esta correção de conexão não muda o contrato, os dados, o ID do estudo ou
os hashes dos modelos já congelados. Os checkouts e hashes são verificados.

A instalação é automática e preserva o PyTorch/CUDA do Colab. O estudo usa
LoRA fp16 sem quantização de 4 bits. O instalador compartilhado chama essa
pilha de dependências de `quick`; isso não diminui o número de treinos nem
os modelos deste estudo. Não são necessárias chaves de APIs de inferência.

## Registro e avaliação humana

Consulte [PROTOCOLO_PRE_REGISTRO_v3.md](PROTOCOLO_PRE_REGISTRO_v3.md).
O relatório mantém `INSUFFICIENT_EVIDENCE` se faltarem dados, competência,
registro, calibração ou anotação. Os dois contrastes usam a semente de
treinamento como unidade, inversão de sinais, bootstrap e correção de Holm.
Apoio estatístico aos contrastes observáveis não demonstra intenção interna.

O estudo exige várias sessões de GPU e armazenamento dos adaptadores e
checkpoints. Os ZIPs de diagnóstico e resultados excluem pesos e checkpoints;
estes continuam no Drive. O notebook 00 pode ser executado sem iniciar treinos.

## Verificação e reprodução

```bash
cd experimento_cientifico_v3
python -m unittest discover -v
python build_notebooks.py --source-ref SHA_CIENTIFICO --setup-ref SHA_DO_SUPORTE
```

Os cinco notebooks foram validados quanto ao formato e à sintaxe. O 00 foi
executado em CPU com dados reais dos geradores e consulta às revisões públicas.
O caso SFT/full possui teste de integração com uma miniatura de pesos
aleatórios em CPU. A execução completa em CUDA e os resultados científicos
ainda dependem da coleta. Veja [VALIDACAO.json](VALIDACAO.json).

Os testes de armazenamento reproduzem falha permanente de montagem,
recuperação na segunda tentativa, Drive já conectado, pasta antiga sem
montagem, falta de gravação e falha do download no navegador. A integração
`python integration_colab_storage.py` executa todas as células dos cinco
notebooks com dez falhas de montagem e duas de download simuladas, mantendo
os treinamentos pausados e produzindo os ZIPs. A autenticação real do Drive
depende da conta e da sessão do Colab; não foi exercitada por essa simulação.

O pacote original v3.1, fornecido em ZIP em 16/10/2026, é a origem deste
conjunto. As correções v3.2 integram o ambiente atualizado, fixam a seleção de
avaliações e congelam também a revisão do monitor independente antes da coleta.

