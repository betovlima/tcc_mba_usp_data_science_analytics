# Restrições de execução do TCC

Este arquivo registra as regras metodológicas obrigatórias do projeto `tcc_mba_usp_data_science_analytics`.

## Regra principal

O resultado do TCC deve ser produzido por código contido neste próprio repositório, executando localmente todas as etapas da estratégia.

Durante a execução do experimento acadêmico é proibido:

- importar `market_cycle_trader_api` ou qualquer outro módulo do projeto Market Cycle Trader;
- executar scripts do repositório Market Cycle Trader;
- fazer checkout, clone ou bootstrap do código do Market Cycle Trader para compor o backtest;
- chamar endpoints HTTP da API do Market Cycle Trader;
- reutilizar modelos treinados, previsões, decisões, resultados de backtests ou artefatos processados pelo Market Cycle Trader;
- usar Strategy persistida do Market Cycle Trader como mecanismo de execução.

O Market Cycle Trader pode ser consultado somente durante o desenvolvimento como referência histórica para compreender a metodologia original. A implementação final deve ser reescrita e mantida neste repositório, de forma auditável e independente.

## Etapas que precisam existir no próprio TCC

O projeto deve executar, com implementação própria:

1. carregamento e validação dos dados históricos brutos;
2. alinhamento temporal do universo de ativos;
3. engenharia de features;
4. construção do target;
5. construção dos folds walk-forward e períodos de purge/calibração;
6. treinamento dos modelos em cada fold;
7. geração de scores/previsões fora da amostra;
8. política de seleção, persistência, rotação e CASH;
9. execução causal das operações na sessão seguinte;
10. aplicação de slippage, taxas e demais custos definidos no experimento;
11. contabilidade do capital composto;
12. cálculo das métricas, folds, drawdown, Sharpe, CAGR e artefatos finais.

## Meta de reconstrução

O valor histórico de aproximadamente `US$ 43.759.854,82` é uma referência de reconstrução, não uma constante a ser forçada no código.

A implementação do TCC deve chegar ao resultado por consequência das regras e dos dados. Não é permitido inserir o capital esperado como atalho, utilizar resultados intermediários previamente processados ou ajustar o algoritmo apenas para fazer o número final coincidir.

## Fonte de dados

A execução acadêmica deve trabalhar com dados históricos brutos disponíveis localmente. Qualquer parâmetro metodológico herdado do experimento histórico precisa ser declarado explicitamente no código ou na documentação do TCC, de forma que a execução seja reproduzível sem acesso ao processamento do Market Cycle Trader.
