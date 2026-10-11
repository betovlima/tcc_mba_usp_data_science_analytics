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


## Resultado da busca por candidatos — pacote v1.0.2

Pacote analisado: `candidate_cluster_search(1).zip`.

A correção do download resolveu o problema de cobertura:

- catálogo amostrado: 500;
- séries obtidas ou reutilizadas: 499;
- cobertura de barras: 99,80%;
- candidato sem barras: VGTX;
- candidatos com histórico suficiente e 52 atributos válidos: 166;
- candidatos rejeitados por histórico/continuidade temporal: 334.

A maior parte das rejeições ocorreu por início tardio da série, o que é
compatível com a exigência de uma janela longa comparável à do U67.

Referência U67:

- 67 ativos documentados;
- 65 representados após DOC e CLMT;
- K=2 escolhido;
- silhouette=0,463660;
- PC1=50,47% e PC2=21,75%;
- PC1+PC2=72,22% da variação do U67.

Distribuição do U67:

- grupo 1: 58 ativos;
- grupo 2: 7 ativos: AMS, FOXF, LKFT, MUX, PXLW, VNCE e YANG.

Faixa complementar derivada da própria geometria do U67:

- limite inferior: 1,2194166770;
- limite superior: 2,2569697026.

Shortlist produzida:

Semelhantes:
EIX, INGR, LXP, VOOG, RLI, SNPS, MLPA, SYK, THQ, GATX, DMO e TAP.

Complementares:
REFR, VTVT, FUSB, MHH, STRR, XNET, EBS, RCMT, ULE, MIN, VISN e VHC.

Interpretação importante:

- os 12 semelhantes ficaram todos no grupo 1;
- oito dos 12 complementares ficaram no grupo 2;
- os candidatos semelhantes apresentaram novelty_ratio entre aproximadamente
  0,42 e 0,75;
- os complementares selecionados ficaram aproximadamente entre 1,64 e 1,84,
  dentro da faixa de suporte definida pelo U67;
- o resultado não usa capital, P&L ou backtest.

### Bloqueio antes da validação de substituição

VISN apareceu entre os complementares, mas já havia sido rejeitado em campanha
anterior por transição estrutural de identidade. Por isso a shortlist v1.0.2
não deve avançar diretamente para backtest.

A versão v1.0.3 passa a excluir previamente símbolos com problemas estruturais
já identificados pela própria pesquisa e corrige também a sobreposição de
título do gráfico PCA.

Foi criado o runner:

```text
auditar_continuidade_shortlist_cluster_spyder.py
```

Ele consulta Corporate Actions somente para a shortlist, registra todos os
eventos sensíveis à identidade e classifica cada ativo como:

- `eligible_for_next_stage`;
- `review_required`;
- `excluded`.

Nenhum teste financeiro será executado antes dessa auditoria.


## Auditoria de continuidade da shortlist — pacote candidate_cluster_search(2)

A auditoria da shortlist v1.0.2 produziu:

- 24 candidatos avaliados;
- 20 classificados automaticamente como elegíveis;
- 2 excluídos;
- 2 inicialmente enviados para revisão.

Exclusões:

- STRR: o registro de Corporate Actions contém stock merger em 22 ago. 2025,
  com STRR como acquiree e HSON como acquirer. A série não deve ser tratada
  como continuidade simples do mesmo instrumento;
- VISN: já constava no histórico da pesquisa como transição de identidade e
  permanece excluído.

Casos inicialmente em revisão:

- SNPS: stock-and-cash merger em 18 jul. 2025 em que ANSS é acquiree e SNPS é
  acquirer. O candidato SNPS é a entidade sobrevivente; o evento, isoladamente,
  não quebra a continuidade histórica do instrumento SNPS;
- SYK: cash merger em 11 nov. 2020 em que WMGI é acquiree e SYK é acquirer.
  Também é aquisição feita pelo próprio candidato, não substituição de SYK por
  outra entidade.

A regra da auditoria foi refinada para distinguir fusões em que o candidato é
o acquiree, que podem quebrar sua continuidade, de aquisições em que o
candidato é o acquirer e permanece como instrumento avaliado.

Após essa revisão, SNPS e SYK podem avançar para a próxima etapa. O conjunto
da execução v1.0.2 fica, portanto, com 22 candidatos tecnicamente aceitáveis e
2 exclusões estruturais. Como a shortlist precisa ser refeita sem STRR e VISN,
ela ainda não deve ser congelada para teste financeiro.

Correções:

- candidate-cluster-search v1.0.4 inclui STRR entre exclusões estruturais já
  conhecidas e mantém VISN excluído;
- candidate-cluster-continuity-audit v1.0.1 reconhece aquisições em que o
  candidato é o acquirer como continuidade preservada, quando não há outro
  evento estrutural não resolvido.

Próxima execução:

1. executar novamente a busca por agrupamento, que reutiliza o cache;
2. executar a auditoria de continuidade;
3. somente se a nova shortlist estiver sem exclusões/revisões pendentes,
   congelar a lista para a análise de eventos de substituição.


## Auditoria final da shortlist v1.0.4 — pacote candidate_cluster_search(4)

A auditoria passou a apontar a fonte exata da shortlist e confirmou que o
arquivo analisado corresponde à lista nova de 24 candidatos, fingerprint:

```text
f343a02844a78704f0817d24cdd5e2df6a216b5e4237b1bc1dacebd4e80ade88
```

Resultado automático:

- 24 candidatos auditados;
- 23 elegíveis;
- 0 excluídos pelo conhecimento local já codificado;
- 1 em `review_required`: TVRD.

O Corporate Action de TVRD registrou em 16 abr. 2025:

- mudança de símbolo CARA -> TVRD;
- mudança de CUSIP 140755208 -> 140755307;
- reverse split 1-for-3.

A consulta documental externa mostrou que esse evento não foi apenas uma troca
de nome/ticker. O Nasdaq registrou business combination e change of control de
Cara Therapeutics com Tvardi Therapeutics. Após o fechamento, o negócio
conduzido pela companhia combinada passou a ser principalmente o negócio da
Legacy Tvardi. Portanto, a série CARA/TVRD não deve ser usada como uma única
continuidade econômica para esta pesquisa.

Decisão:

- TVRD passa a ser exclusão estrutural conhecida;
- o candidato não avança para a etapa de substituição;
- a busca deve preencher novamente a 12ª posição complementar;
- unidades de operating partnership também passam a ser sinalizadas antes da
  shortlist, evitando que instrumentos como ESBA sejam promovidos
  automaticamente sem revisão de compatibilidade com o universo.

Versões:

- candidate-cluster-search-v1.0.5;
- candidate-cluster-continuity-audit-v1.0.3.

Nenhum resultado de capital foi consultado para tomar essa decisão.


## Diagnóstico do pacote candidate_cluster_search(5)

O pacote revelou uma inconsistência de proveniência entre os artefatos da mesma
pasta.

Arquivos atuais da busca:

- `candidate_cluster_search.json`: v1.0.5;
- `candidate_projection.csv`: 162 candidatos válidos;
- TVRD não aparece no catálogo nem na projeção atual;
- ESBA aparece na projeção atual como próximo candidato complementar.

Entretanto:

- `candidatos_cluster_shortlist.csv` ainda contém TVRD;
- a pasta `continuity_audit` ainda corresponde à shortlist anterior;
- o audit é v1.0.2 e mantém TVRD em review_required.

Portanto, a pasta combinou resultados de execuções diferentes e não pode ser
usada para congelar a shortlist.

A análise da projeção atual mostra que, sem TVRD, ESBA seria o próximo
complementar. ESBA é `EMPIRE STATE REALTY OP, L.P. SERIES ES`, uma unidade de
operating partnership, e não deve ser promovida automaticamente como se fosse
uma ação ordinária comparável sem revisão de instrumento.

Correções v1.0.6 / audit v1.0.4:

1. a busca limpa todos os resultados antigos antes de rodar, preservando apenas
   `cache_bars`;
2. a regra de instrumento passa a reconhecer `OP, L.P.` como
   `operating_partnership_unit`;
3. a shortlist é validada como subconjunto da projeção corrente;
4. a busca aborta se a shortlist contiver exclusão estrutural ou instrumento
   sinalizado;
5. a busca grava lista e fingerprint exatos da shortlist no JSON;
6. a auditoria exige que o fingerprint e a ordem dos ativos coincidam com o
   JSON da busca antes de consultar Corporate Actions.

Com ESBA sinalizado, o próximo candidato complementar pela geometria atual é
GME, sujeito à nova execução e à auditoria de continuidade. Nenhum resultado
financeiro foi usado nessa substituição.


## Pacote candidate_cluster_search(6) — shortlist tecnicamente fechada

A busca v1.0.6 produziu uma shortlist coerente e vinculada à auditoria pelo
fingerprint:

```text
a6eca6ae6f7eb22ccebb9dcd5c10210887d7828b996c6913ebbcb3204298bf2f
```

Resumo da busca:

- catálogo amostrado: 500;
- séries obtidas/reutilizadas: 499;
- cobertura: 99,80%;
- candidatos com histórico suficiente e 52 atributos: 162;
- U67 representado por 65 ativos;
- K=2;
- silhouette=0,463660;
- PC1+PC2=72,2225% da variação do U67.

Shortlist:

Semelhantes:
EIX, INGR, LXP, VOOG, RLI, SNPS, MLPA, SYK, THQ, GATX, DMO, TAP.

Complementares:
REFR, VTVT, FUSB, MHH, XNET, EBS, RCMT, ULE, MIN, VHC, UPLD, GME.

A auditoria v1.0.4 confirmou 23 elegíveis e manteve somente GME em
`review_required` por um evento classificado pelo provedor como `spin_off`
em outubro de 2025.

A revisão documental mostrou que o evento de GME foi uma distribuição especial
de warrants aos acionistas: um warrant para cada dez ações de GME. O ativo GME
permaneceu sendo a ação ordinária da mesma companhia; o novo instrumento
GME.WS foi distribuído separadamente. Portanto, o evento não representa quebra
de continuidade econômica do ticker GME para o objetivo desta pesquisa.

A auditoria v1.0.5 passa a reconhecer especificamente esse evento como:

```text
warrant_distribution_parent_continuity_preserved
```

Com essa correção metodológica, a expectativa é 24/24 candidatos elegíveis,
sem exclusão ou revisão pendente.

Foi criado o runner:

```text
congelar_shortlist_cluster_spyder.py
```

Ele somente congela a shortlist quando:

- busca = candidate-cluster-search-v1.0.6;
- auditoria = candidate-cluster-continuity-audit-v1.0.5;
- 24 candidatos;
- 24 elegíveis;
- 0 review_required;
- 0 excluded;
- fingerprint e ordem dos ativos coincidem.

O congelamento ocorre antes de qualquer resultado financeiro da nova shortlist.


## Pacote candidate_cluster_search(7) — auditoria concluída

A execução v1.0.6 da busca e a auditoria v1.0.5 ficaram finalmente alinhadas
na mesma shortlist e no mesmo conjunto de artefatos.

Identificador da shortlist:

```text
a6eca6ae6f7eb22ccebb9dcd5c10210887d7828b996c6913ebbcb3204298bf2f
```

Resultado final da auditoria:

- shortlist: 24 candidatos;
- elegíveis para a próxima etapa: 24;
- review_required: 0;
- excluded: 0;
- resultados financeiros usados na seleção: não;
- backtest executado antes da seleção: não.

Semelhantes:

```text
EIX, INGR, LXP, VOOG, RLI, SNPS, MLPA, SYK, THQ, GATX, DMO, TAP
```

Complementares:

```text
REFR, VTVT, FUSB, MHH, XNET, EBS, RCMT, ULE, MIN, VHC, UPLD, GME
```

Casos sensíveis resolvidos:

- SNPS: candidato é o acquirer; continuidade preservada;
- SYK: candidato é o acquirer; continuidade preservada;
- GME: distribuição separada de warrants GME.WS; continuidade da ação GME
  preservada.

A etapa de descoberta por agrupamento está, portanto, encerrada e pronta para
congelamento formal antes de qualquer análise financeira dos 24 candidatos.

Próximo runner:

```text
congelar_shortlist_cluster_spyder.py
```

Somente após o congelamento a pesquisa deve construir o conjunto temporal de
eventos de substituição `(candidato, ativo mantido, data da decisão)`.


## Confirmação do pacote candidate_cluster_search(8)

O pacote recebido confirmou, sem pendências, a shortlist produzida pela busca
v1.0.6 e auditada pela continuidade v1.0.5.

Resultado observado nos artefatos:

- 24 candidatos na shortlist;
- 24 classificados como `eligible_for_next_stage`;
- 0 em `review_required`;
- 0 excluídos;
- lista e ordem da shortlist coincidem entre busca e auditoria;
- a auditoria reconheceu SNPS e SYK como aquisições em que o candidato é o
  acquirer e, portanto, preserva a continuidade;
- GME foi reconhecido como distribuição separada de warrants, preservando a
  continuidade da ação ordinária.

Shortlist confirmada:

Semelhantes:
EIX, INGR, LXP, VOOG, RLI, SNPS, MLPA, SYK, THQ, GATX, DMO, TAP.

Complementares:
REFR, VTVT, FUSB, MHH, XNET, EBS, RCMT, ULE, MIN, VHC, UPLD, GME.

O ZIP recebido não contém os artefatos de
`dados/cluster_candidate_search/`, portanto ele confirma a auditoria, mas não
permite afirmar que o runner `congelar_shortlist_cluster_spyder.py` já tenha
sido executado.

Próximo passo operacional: executar o congelamento formal da shortlist antes de
qualquer construção de eventos de substituição ou abertura de resultados
financeiros.


## Congelamento formal da shortlist concluído

A execução de `congelar_shortlist_cluster_spyder.py` concluiu com sucesso:

- version: `cluster-shortlist-freeze-v1.0.0`;
- count: 24;
- `financial_outcomes_used_for_selection=False`;
- `backtest_executed_before_freeze=False`;
- a seleção foi marcada como bloqueada antes de qualquer avaliação financeira.

Shortlist congelada:

Semelhantes:
EIX, INGR, LXP, VOOG, RLI, SNPS, MLPA, SYK, THQ, GATX, DMO, TAP.

Complementares:
REFR, VTVT, FUSB, MHH, XNET, EBS, RCMT, ULE, MIN, VHC, UPLD, GME.

### Próxima etapa: congelar os dados dos candidatos

Antes de construir eventos `(candidato, incumbente, data)`, os dados dos 24
candidatos devem ser congelados com o mesmo contrato do U67: Alpaca SIP,
1Day, RAW, Corporate Actions e o mesmo cutoff temporal.

Foi criado:

```text
construir_snapshot_shortlist_cluster_spyder.py
```

O runner:

1. exige a shortlist formalmente congelada;
2. lê o cutoff diretamente do snapshot U67;
3. baixa RAW e Corporate Actions dos 24 candidatos;
4. cria snapshot em `dados/cluster_candidate_search/snapshot`;
5. valida a preparação dos 24 ativos;
6. aborta se surgir qualquer exclusão estrutural nova;
7. não executa backtest nem consulta capital;
8. reutiliza o snapshot existente em execuções posteriores.
