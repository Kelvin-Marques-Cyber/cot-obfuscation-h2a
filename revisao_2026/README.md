# Reanálise e ambiente de teste - ofuscação de CoT

Revisão de 15/10/2026. Este pacote acompanha o artigo final e permite reproduzir
os cálculos publicados e os novos testes dos instrumentos, sem GPU, sem chamadas
a APIs e sem custos de inferência.

## O que foi realmente executado

- Reanálise das 540 linhas de métricas depositadas, em 18 células.
- Auditoria das funções originais de geração, correção e pontuação lexical.
- Teste mecânico do corretor em 2.000 textos construídos, não produzidos por LLM.
- Reconstrução das partições do gerador e medição de suas sobreposições.
- Geração de 4.500 itens aritméticos com partições sem repetição de enunciados.
- Suíte automatizada dos contratos de análise.

Não foram executados novos treinos, novas inferências de LLM, GRPO ou anotação
humana. Os números do piloto são métricas históricas, não saídas reavaliadas: o
CSV original não contém o texto das gerações nem IDs dos itens.

## Reproduzir

Requer Python 3.12. Em um ambiente virtual:

    python3 -m venv .venv
    . .venv/bin/activate
    pip install -r requirements.lock.txt
    bash run.sh

No Windows, use .venv\Scripts\activate e depois:

    python -m unittest discover -s tests -v
    python src/reanalyse.py

O ambiente local usado na revisão já continha as versões do arquivo lock.
O Dockerfile é uma opção de empacotamento; não foi construído nesta revisão.
Pode ser construído com docker build -t cot-reanalysis . e executado com
docker run --rm -v "$PWD/results:/work/results" cot-reanalysis.

## Arquivos e procedência

- original/: notebook e CSVs do repositório cot-obfuscation-h2a, commit
  e21cef8ad92369c56f3fcbbb6891fa49bda8c74b; mantidos sem reescrita do conteúdo.
- src/instruments.py: corretor estrito, carregador de funções e gerador disjunto.
- src/reanalyse.py: todos os cálculos e gráficos desta revisão.
- src/evaluate_jsonl.py: contrato para avaliar novas gerações já coletadas.
- tests/: testes de contrato e regressão.
- results/: medições, corpora construídos e registros de execução.
- figures/: figuras regeneradas.
- artigo.md: fonte editável do texto final.
- protocol_extension.json: desenho prospectivo; não constitui pré-registro público.
- provenance.json e checksums.sha256: origem e integridade dos arquivos.

O parser estrito aceita uma única tag <answer>NUMERO</answer>, uma linha final
Resposta: NUMERO ou uma saída inteiramente numérica. Respostas fora desse
contrato são erro de formato, inclusive respostas semanticamente corretas em
prosa livre. Seu desempenho no corpus construído não estima desempenho geral.

O registro results/execution.log é a fonte para o total de testes vigente.
Os intervalos t exportados são condicionais a suposições fortes com três
sementes; os intervalos Wilson são descritivos e não corrigem agrupamento.
O teste por inversão de sinais é análise de sensibilidade, não um novo desenho
randomizado e não uma prova de efeito ausente.

## Licença

As cópias originais conservam os avisos da licença MIT e da licença de dados
declarada no repositório. Os acréscimos de código deste pacote são disponibilizados
sob MIT, com os mesmos termos de original/LICENSE. O texto do artigo mantém
a autoria indicada no manuscrito. Não se atribuem contribuições novas aos
coautores nem anuência à revisão sem confirmação dos próprios autores.
