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
