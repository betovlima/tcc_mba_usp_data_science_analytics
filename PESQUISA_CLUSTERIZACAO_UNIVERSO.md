# Pesquisa exploratória — evolução do universo e agrupamento por semelhança

## Objetivo

Visualizar como a composição do universo de ativos mudou ao longo do histórico
do projeto e como os ativos se agrupam por semelhança quando o conjunto de
alternativas muda.

Esta branch foi criada porque não há segurança, neste momento, de que o primeiro
universo da pesquisa tivesse exatamente 20 ativos. Por isso o script não fixa um
tamanho inicial. Ele procura no histórico Git as composições literais de
`ASSETS` preservadas em `engine/configuracao.py` e apresenta primeiro uma
linha do tempo das listas encontradas.

## Branch

```text
research/universe-clustering-visualization-v1
```

## Script

```text
visualizar_evolucao_universo_spyder.py
```

O script é separado da reprodução científica. Ele não treina modelos, não roda
backtest, não acessa Alpaca, não consulta banco de dados e não altera os dados
congelados.

## Execução no Spyder

Abra o projeto no Spyder, abra
`visualizar_evolucao_universo_spyder.py`, reinicie o kernel e execute com F5.

Também é possível executar pelo console IPython do Spyder:

```python
%run visualizar_evolucao_universo_spyder.py
```

O script não força o backend `Agg`. Cada figura termina com `plt.show()`, de
modo que os gráficos aparecem no painel **Plots** do Spyder.

## Primeira figura: descoberta das etapas

A primeira figura mostra a quantidade de ativos em cada composição distinta
encontrada no histórico Git. O console imprime, para cada etapa:

- identificador da etapa;
- data;
- quantidade de ativos;
- ativos incluídos;
- ativos removidos;
- commit de origem.

A presença de branches históricas de pesquisa significa que a ordem temporal das
listas encontradas precisa ser validada antes de ser descrita como evolução
científica definitiva. O objetivo da primeira execução é justamente recuperar a
sequência correta sem assumir um universo inicial de 20, 25 ou outro tamanho.

## Mapas de agrupamento

Para cada ativo disponível no snapshot U67, o script calcula os 52 atributos já
definidos no experimento e resume cada atributo pela mediana no período de
análise.

Em cada etapa, esses valores são expressos relativamente aos demais ativos do
mesmo universo. Depois, todas as combinações ativo-etapa são projetadas com uma
única PCA, o que mantém os mesmos eixos nos diferentes gráficos.

Os grupos são obtidos de forma exploratória com K-Means. Eles não representam
setores econômicos, não demonstram causalidade e não constituem regra de seleção
de ativos.

## Interpretação para o texto científico

A leitura pretendida é simples:

- pontos próximos representam ativos com perfis semelhantes;
- pontos distantes representam perfis diferentes;
- a inclusão ou retirada de ativos modifica a referência relativa usada para
  comparar os integrantes do universo;
- uma mudança de grupo de um mesmo ativo entre etapas indica mudança de sua
  posição relativa dentro do conjunto analisado.

O gráfico deve ser tratado como análise exploratória da composição do universo,
não como evidência de que determinado grupo produz maior retorno.

## Arquivos auxiliares

Por padrão, o script grava em:

```text
output/universe_clustering/
```

os seguintes CSVs:

```text
etapas_universo_descobertas.csv
coordenadas_agrupamento.csv
resumo_atributos_por_ativo.csv
```

Os PNGs não são gravados por padrão. Para gravá-los, altere no início do script:

```python
SAVE_FIGURES = True
```

## Próximo passo

Executar a branch no Spyder e conferir a tabela e a primeira figura de evolução.
Somente depois dessa conferência devem ser escolhidas as etapas que entrarão nos
gráficos finais e no texto do trabalho.
