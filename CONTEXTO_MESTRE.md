# CONTEXTO_MESTRE

## Baseline preservado

A reproducao oficial permanece Control vs Soft Horizon Consensus, com dados
congelados em `dados/pesquisa/`. O experimento Directional Change nao altera o
Control oficial nem o pipeline operacional do Market Cycle Trader.

## Pesquisa ativa — v1.3.0-dev.1

Branch: `feature/v1.3.0-dev.1-directional-change-lightgbm`

Hipotese: reduzir o giveback entre o topo observado durante uma posicao e a
saida usando um classificador causal Directional Change + LightGBM.

Arquivos principais:

- `pesquisas/directional_change_lightgbm.py`;
- `pesquisar_directional_change_spyder.py`;
- `tests/test_directional_change_lightgbm.py`;
- `docs/changes/v1.3.0-dev.1-directional-change-lightgbm.md`.

Regra experimental: o challenger somente antecipa uma saida para CASH quando o
Control manteria a posicao e a probabilidade calibrada de reversao e alta. Uma
decisao de rotacao/saida ja tomada pelo Control nunca e bloqueada pelo overlay.

Protocolo: mesmos dados congelados, mesmo calendario, mesmos folds, purge e
custos do Control. O futuro e usado somente na construcao dos labels de treino.

Promocao: nenhum resultado deve ser levado ao MCT antes de demonstrar ganho OOS
no TCC. Avaliar capital, CAGR, Sharpe, MaxDD, pior fold e metricas Peak Exit.

Status: codigo de pesquisa implementado; aguarda execucao completa do backtest
local para produzir resultados reais.


## Resultado v1.3.0-dev.1

A primeira versao Directional Change + LightGBM nao foi aprovada:

- Control: US$ 10.082.425,91;
- DC v1: US$ 3.653.966,86;
- delta relativo: -63,76%;
- distancia mediana do topo: 2,3844% -> 2,3437%;
- captura mediana do topo: 32,76% -> 26,53%;
- 52 gatilhos DC;
- balanced accuracy de calibracao entre aproximadamente 50,5% e 53,0%;
- precision entre aproximadamente 20,0% e 30,0%.

Diagnostico: o alvo generico de drawdown em cinco sessoes gerou falsos
positivos e saidas de um pregao para CASH seguidas de reentrada. A v1 nao deve
ser promovida para o MCT.

## Pesquisa ativa — v1.3.0-dev.2

Branch: `feature/v1.3.0-dev.2-directional-change-top-turn`

A v2 preserva as features Directional Change, mas substitui o alvo por um
evento Top-Turn first-passage condicionado a tendencia de alta proxima de uma
maxima. A calibracao prioriza precision com F0.5 e exige duas confirmacoes
consecutivas antes de uma saida.

Script Spyder:
`pesquisar_directional_change_top_turn_spyder.py`

Ao final, o script gera automaticamente um ZIP compacto de analise e emite um
sinal sonoro. O ZIP e o unico pacote necessario para enviar em futuras analises.

Nenhum resultado da v2 existe antes da execucao real.
