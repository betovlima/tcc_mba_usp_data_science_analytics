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


## Revisão após a primeira execução

A primeira execução local produziu apenas duas composições ao procurar mudanças
literais de `ASSETS` em `engine/configuracao.py`:

- U56;
- U67.

Essa descoberta era incompleta como narrativa histórica. A documentação
preservada da pesquisa registra explicitamente a sequência:

```text
U56
  + COLB, AMS, FOXF
= U59
  + THO, WDAY, EXR, XEL, SBFG, PAYX, MUX, SXC
= U67
```

A versão 1.1.0 do script passou a usar essa sequência documentada nos gráficos,
mantendo a leitura automática do histórico Git apenas como auditoria.

A mesma documentação registra outros universos de sensibilidade ou campanhas
diagnósticas, como U54/U55/U56 e expansões aleatórias U76/U79. Esses cenários
não são misturados à linha principal U56 -> U59 -> U67 porque tinham finalidade
experimental diferente.

Também foi corrigida uma ambiguidade da primeira figura: U56 e U67 são
quantidades documentadas do universo, enquanto os mapas atuais conseguem
representar 54 e 65 ativos, respectivamente, porque DOC e CLMT são excluídos
pela preparação estrutural da versão final. Na sequência U56 -> U59 -> U67,
os mapas passam a exibir:

- U56: 56 documentados, 54 representados;
- U59: 59 documentados, 57 representados;
- U67: 67 documentados, 65 representados.

Por fim, a quantidade de grupos deixou de ser fixada arbitrariamente em quatro.
O script testa K=2..6 e seleciona a quantidade de grupos com maior índice
silhouette, gravando a tabela de qualidade e mostrando o gráfico de seleção no
Spyder.

Ainda não há evidência suficiente nesta branch para afirmar que U56 foi o
universo primordial de toda a pesquisa. Essa questão deve permanecer aberta até
que as composições anteriores sejam recuperadas de artefatos ou commits mais
antigos.


## Fase 2 — busca de novos candidatos por agrupamento

Após revisar as tentativas históricas de assinatura e clusterização, esta
branch passa a testar a clusterização apenas como filtro de descoberta.

Motivação: a clusterização anterior encontrou grupos comportamentais estáveis,
mas não separou de forma suficiente candidatos que aumentavam o capital dos
que prejudicavam a estratégia. O problema principal apareceu na interação com
o incumbente e na dependência de caminho. Por isso, o novo desenho não usa o
cluster como decisão de inclusão.

Novo runner:

```text
pesquisar_candidatos_por_agrupamento_spyder.py
```

### Pergunta desta etapa

É possível localizar ativos externos ao U67 que:

1. apresentem comportamento semelhante a regiões já representadas no universo;
2. ocupem regiões próximas, porém pouco representadas;
3. possam depois ser avaliados em eventos de substituição contra o ativo
   atualmente mantido?

### Referência

A referência é o U67 atual após preparação estrutural. Os 52 atributos já
utilizados pelo experimento são resumidos por ativo. O escalonamento, a PCA e o
K-Means são ajustados somente nos ativos da referência.

Os candidatos externos são projetados nesse espaço sem reajustar a referência.
Isso evita que a própria amostra de candidatos mova os eixos ou os grupos usados
para defini-los.

### Busca inicial

Na primeira execução o runner:

- consulta o catálogo Alpaca;
- remove os símbolos já presentes no U67;
- mantém ações/ETFs negociáveis e margináveis das bolsas aceitas;
- ordena o catálogo por chave determinística;
- avalia inicialmente até 500 candidatos;
- baixa barras diárias SIP ajustadas por split;
- mantém cache local em `output/candidate_cluster_search/cache_bars/`;
- exige histórico longo e sem grandes lacunas;
- calcula os mesmos 52 atributos;
- projeta os candidatos no espaço comportamental do U67.

O limite de 500 controla o custo da primeira experiência. Ele não significa que
esses 500 sejam os melhores ativos do mercado.

### Dois tipos de candidatos

**Semelhantes**: os mais próximos de algum ativo já presente no U67.

**Complementares**: candidatos mais afastados do vizinho U67 mais próximo, mas
ainda dentro de uma faixa de suporte definida para evitar selecionar apenas
outliers extremos.

A distância é medida no espaço padronizado dos 52 atributos. O valor
`novelty_ratio` compara a distância do candidato ao seu vizinho U67 com a
distância típica entre os próprios ativos do U67.

### Saídas

```text
output/candidate_cluster_search/
├── candidate_catalog_sample.csv
├── candidate_history_quality.csv
├── candidate_feature_summary.csv
├── cluster_quality_u67.csv
├── u67_cluster_reference.csv
├── candidate_projection.csv
├── candidatos_cluster_shortlist.csv
├── candidate_cluster_search.json
├── 01_mapa_candidatos_u67.png
├── 02_distribuicao_distancia_comportamental.png
└── 03_shortlist_cluster_search.png
```

Os gráficos também são publicados no painel Plots do Spyder.

### Regra científica

Nenhum resultado de capital, P&L ou backtest participa da seleção desta fase.
A shortlist não deve ser incorporada automaticamente ao universo.

A próxima etapa, caso a shortlist seja tecnicamente coerente, será medir a
qualidade de substituição na unidade:

```text
(candidato, ativo atualmente mantido, data da decisão)
```

Somente depois dessa avaliação poderá ser definido um teste financeiro
congelado. O objetivo é evitar repetir a falha das tentativas anteriores, nas
quais semelhança de comportamento ou baixo risco individual não garantiram
melhoria conjunta do capital.


## Primeira execução da busca por candidatos — diagnóstico v1.0.0

Pacote analisado: `candidate_cluster_search.zip`.

A execução não deve ser interpretada como shortlist válida porque revelou um
problema de aquisição de barras.

Resultados observados:

- catálogo determinístico: 500 candidatos;
- candidatos com alguma barra disponível no cache/retorno: 105;
- classificados como `no_bars`: 395;
- candidatos que passaram a exigência de histórico: 21;
- candidatos com 52 atributos válidos: 21;
- U67 representado por 65 ativos, com DOC e CLMT excluídos;
- PCA PC1+PC2: aproximadamente 72,22% da variação;
- K=2 escolhido para o U67, silhouette aproximado 0,4637;
- distribuição dos grupos U67: 58 e 7 ativos.

A presença entre os falsos `no_bars` de tickers conhecidos com histórico
longo, como IBB, LKQ e SYK, mostrou que a baixa cobertura não era um resultado
econômico. A causa era a solicitação multi-símbolo de aproximadamente dez anos
de barras com limite agregado, que podia retornar somente uma parte dos
símbolos do lote.

Consequência: a shortlist produzida nessa execução, incluindo BSX, COP, AXGN,
CLF, AKBA, DBVT e outros, é apenas artefato diagnóstico e não deve avançar para
validação financeira ou de substituição.

### Correção v1.0.1

O runner passa a:

1. reutilizar o cache existente;
2. baixar cada ticker faltante individualmente;
3. exigir cobertura de resposta de pelo menos 90% do catálogo antes de
   interpretar qualquer shortlist;
4. abortar se a cobertura continuar incompleta;
5. sinalizar warrants, units, preferred, senior notes, rights e ETNs para que
   não entrem automaticamente na shortlist;
6. derivar a faixa de candidatos complementares da própria distribuição de
   distâncias entre vizinhos do U67, usando P75 a P95, em vez de premiar
   candidatos simplesmente muito afastados.

Nenhum resultado financeiro foi usado para fazer essas correções. O próximo
passo é repetir somente `pesquisar_candidatos_por_agrupamento_spyder.py`.
