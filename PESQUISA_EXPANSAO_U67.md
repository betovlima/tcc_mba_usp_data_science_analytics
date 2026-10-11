# Pesquisa de expansão do universo U67 inspirada na Busca de Ativos do sistema operacional

## Objetivo

Partir do universo atual U67 sem reconstruir sua origem e testar somente a
adição de novos ativos externos.

O U67 permanece como baseline fixo. Nenhum ativo existente será removido,
substituído ou reordenado como condição para esta pesquisa.

Na execução financeira, o pipeline continuará aplicando exatamente a preparação
estrutural já usada pela reprodução oficial. Portanto, o contrato continua
sendo 67 símbolos configurados e 65 ativos efetivos no replay atual, devido às
exclusões estruturais já documentadas de DOC e CLMT.

## Pergunta

Existem ativos externos ao U67 que, quando adicionados ao universo atual,
aumentam o capital final da Strategy sem depender apenas de um ranking
preditivo?

## Princípio recuperado do sistema operacional

A Busca de Ativos do sistema operacional mostrou que ranking preditivo e contribuição econômica
não são equivalentes. O ranking deve ser usado apenas para reduzir o espaço de
busca; a inclusão deve ser decidida pela contribuição do candidato quando
rodado dentro da Strategy.

Assim, esta pesquisa separa:

1. triagem técnica;
2. ranking preditivo;
3. replay econômico marginal;
4. validação conjunta dos ativos aprovados.

## Baseline

```text
U67 atual
requested = 67
effective no replay atual = 65
capital de referência oficial em 2026-10-06 ≈ US$ 76.927.051,39
```

O baseline é recalculado pelo próprio runner antes de avaliar candidatos. Não
será usado valor histórico digitado para decidir aprovação.

## Universo externo

A busca consulta o catálogo US Equity ativo da Alpaca e remove apenas os
símbolos já existentes no U67 e exclusões estruturais conhecidas.

Critérios de mercado derivados da versão madura do Asset Discovery:

```text
exchange: NYSE, NASDAQ, AMEX, ARCA ou BATS
tradable = true
preço >= US$ 5
median dollar volume >= US$ 10 milhões
nonzero volume ratio >= 0,98
```

O candidato deve possuir histórico suficiente para ser comparado com a janela
da Strategy e deve passar pela auditoria de identidade/continuidade antes do
replay financeiro.

## Dados

Para manter comparabilidade com o U67 oficial:

```text
feed = SIP
timeframe = 1Day
adjustment = RAW
início = 2016-01-01
cutoff = 2026-10-06
Corporate Actions = Alpaca
splits = normalizados pelo mesmo pipeline do U67
```

Não será usado o download SPLIT-adjusted da pesquisa de agrupamento para decidir
inclusão financeira.

## Ranking

A primeira versão reproduzirá a ideia central do Learning-to-Rank da Busca de
Ativos do sistema operacional para priorizar os candidatos de maior interesse. O ranking não é
um gate econômico.

Um candidato bem ranqueado pode ser rejeitado se o replay marginal piorar o
capital.

## Replay marginal

Para cada candidato selecionado para a etapa financeira:

```text
baseline = U67
candidate = U67 + ativo
delta = capital(candidate) - capital(baseline)
```

Serão registradas, no mínimo:

- capital final;
- delta absoluto e percentual;
- CAGR;
- Sharpe;
- MaxDD;
- pior fold;
- número de operações/switches;
- participação direta do novo ativo;
- quantidade de decisões alteradas em relação ao U67.

## Regra da primeira rodada

Nenhum ativo é removido do U67.

A primeira rodada testa candidatos individualmente contra o mesmo baseline U67.
Os ativos com delta de capital positivo formam apenas um conjunto provisório.

Depois será executado:

```text
U67 + todos os positivos da rodada
```

porque contribuições marginais individuais podem interagir quando os ativos são
adicionados juntos.

Se o conjunto piorar o baseline, serão avaliadas combinações/subconjuntos sem
alterar os ativos originais do U67.

## Regra de interpretação

```text
U67 é o ponto de partida fixo.
A pesquisa procura U67+n.
Não procura reconstruir U67.
Não procura substituir ativos do U67 nesta fase.
Não usa quantidade de ativos como objetivo.
O objetivo é aumentar o capital preservando robustez.
```

## Saídas previstas

```text
output/u67_asset_expansion/
    candidate_catalog.csv
    candidate_screening.csv
    candidate_rank.csv
    continuity_audit.csv
    u67_baseline_metrics.json
    marginal_replay.csv
    positive_candidates.csv
    combined_positive_replay.json
    decision_changes.csv
    trades_new_assets.csv
    pesquisa_u67_asset_expansion.json
```

## Critério de avanço

A pesquisa só avança para uma segunda rodada de adição se a primeira produzir
um universo U67+n que supere o U67 de referência e não apresente deterioração
inaceitável das métricas de risco.

A decisão final será baseada no resultado conjunto do universo expandido, não
na soma dos deltas marginais individuais.


## Implementação v1.0.1

A execução desta pesquisa está concentrada em `pesquisar_expansao_u67_spyder.py`.
O arquivo foi desenhado para execução completa pelo Spyder com F5 e não altera
o snapshot oficial `dados/u67`.

O fluxo implementado é:

1. validar e reproduzir o U67 congelado antes de qualquer busca;
2. consultar o catálogo ativo US Equity da Alpaca e construir uma amostra
   externa determinística;
3. executar uma triagem rápida de mercado e ordenar os candidatos com
   Learning-to-Rank LightGBM LambdaMART inspirado na Busca de Ativos do sistema operacional;
4. baixar, somente para os candidatos que chegam ao replay, barras SIP/1Day/RAW
   e Corporate Actions em cache local separado;
5. rejeitar séries com quebra de identidade ou continuidade, sem reconstrução,
   bridge ou correção manual;
6. executar U67 + 1 ativo contra o mesmo calendário, folds, custos e regras do
   baseline;
7. formar o conjunto provisório dos candidatos com delta de capital positivo;
8. executar U67 + todos os positivos;
9. quando o conjunto completo não melhora o capital, construir de forma
   gulosa um subconjunto dos positivos individuais, sempre preservando todo o
   U67 original.

### Reuso exato do baseline

Os modelos LightGBM do U67 são treinados uma vez por fold e seus caches de
utilidade são reutilizados nos cenários marginais. Como a implementação oficial
ajusta um regressor independente por ativo, cada cenário treina somente o
modelo do novo ativo e combina seu cache com os caches do U67. A
`switch_margin` continua sendo recalibrada para cada universo testado.

A comparação usa explicitamente o calendário do U67 como
`calendar_override`. Assim, a entrada de um ativo não pode alterar os folds ou
as sessões de decisão da pesquisa.

### Busca externa

A v1.0.0 usa uma amostra determinística de até 500 símbolos externos elegíveis
do catálogo ativo da Alpaca. A triagem exige os critérios registrados nesta
pesquisa e usa dados recentes somente para o ranking. Os dados que entram no
replay financeiro são baixados novamente no formato RAW e normalizados pelo
mesmo pipeline de Corporate Actions e splits usado pelo U67.

Por padrão, os 24 candidatos mais bem ranqueados que passam pela triagem rápida
seguem para a validação completa. Esses limites são parâmetros operacionais da
primeira campanha e ficam registrados no artefato final.

### Integridade e limitações

O catálogo de descoberta contém ações atualmente ativas. Portanto, a pesquisa
registra explicitamente que o viés de sobrevivência do universo externo não foi
eliminado por completo. O ranking também é tratado como exploratório, pois foi
treinado com o histórico congelado completo do U67 para priorizar a busca.

Não foi criado um corte automático de Sharpe ou MaxDD, pois não existe nesta
pesquisa uma definição numérica prévia de "deterioração inaceitável". A
aprovação econômica automática da primeira rodada exige delta positivo de
capital; as métricas de risco são preservadas integralmente para a decisão
posterior.

Além das saídas originalmente previstas, a execução grava
`scenario_fold_calibration.csv`, contendo a calibração de margem de cada
cenário, e gera `pacote_analise_u67_asset_expansion.zip` ao final.
