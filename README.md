# TCC MBA USP — Data Science & Analytics

Simulação histórica de rotação de capital com LightGBM e validação temporal progressiva.

## Execução

```bash
git clone https://github.com/betovlima/tcc_mba_usp_data_science_analytics.git
cd tcc_mba_usp_data_science_analytics
python -m pip install -r requirements.txt
python backtest.py
```

A fonte de mercado é o Yahoo Finance, por meio da biblioteca `yfinance`.

## Fluxo do experimento

```text
Yahoo Finance
  ↓
OHLCV diário
  ↓
características técnicas
  ↓
alvo multihorizonte
  ↓
validação temporal progressiva
  ↓
LightGBM por janela e ativo
  ↓
calibração da margem de troca
  ↓
política de rotação
  ↓
execução no pregão seguinte
  ↓
custos e capital composto
  ↓
métricas e arquivos de auditoria
```

Todas as etapas da estratégia são executadas pelo código deste repositório. O motor não importa módulos, modelos treinados, previsões ou resultados processados do projeto Market Cycle Trader.

## Estrutura principal

```text
backtest.py
requirements.txt
analysis/
tcc_engine/
  __init__.py
  caracteristicas.py
  configuracao.py
  diagnosticos_rotacao.py
  execucao.py
  metricas.py
  modelo.py
  politica_rotacao.py
  rotacao_capital.py
  simulacao.py
  validacao_temporal.py
```

Os nomes de arquivos, funções, classes, comentários e variáveis de domínio do motor são mantidos em português. Permanecem em inglês somente nomes definidos por bibliotecas externas e campos estáveis do contrato de saída, como parâmetros do LightGBM e colunas OHLCV.

## Configuração

Os parâmetros do experimento estão em:

```text
tcc_engine/configuracao.py
```

Parâmetros principais:

```text
capital inicial                  10.000
horizontes do alvo               5, 10, 20, 40 e 60 sessões
linhas mínimas de treinamento    700
calibração                       126 sessões
teste por janela                 504 sessões
separação temporal               60 sessões
permanência mínima               2 sessões
margem base de rotação           0.0005
margens candidatas               0, 0.0025, 0.005, 0.01
semente aleatória                42
```

Os hiperparâmetros abaixo mantêm os nomes exigidos pelo LightGBM:

```text
n_estimators          329
learning_rate         0.020731
max_depth             3
num_leaves            6
min_child_samples     18
min_child_weight      5.0
subsample             0.85
colsample_bytree      0.88067
reg_alpha             0.050837
reg_lambda            3.596305
```

## Universo

```text
NVDA, MSFT, META, TSLA, AMD, JPM, SPY, AVGO, NFLX,
ORCL, COST, LLY, XOM, CAT, WMT, V, HD, ADC, ADEA,
ADI, ADM, GKOS, VNCE, CORT, UNFI, DNN, MKSI, APD,
DDS, RACE, UNF, TX, CEF, YANG, KKR, BXMT, SCSC
```

Período solicitado:

```text
2016-01-01 → 2026-09-04
```

## Arquivos gerados

Cada execução grava:

```text
output/market_data.csv
output/backtest_result.json
output/equity_curve.csv
output/folds.csv
output/trades.csv
output/summary.txt
```

`market_data.csv` contém as séries históricas usadas pelo backtest e também alimenta a planilha de análise.

## Planilha de auditoria

Após o backtest:

```bash
python analysis/build_excel.py
```

Arquivo gerado:

```text
analysis/tcc_backtest_output_analysis.xlsx
```

## Referência histórica

A reprodução certificada anterior foi preservada na tag:

```text
certified-43m-standalone
```

A `main` usa Yahoo Finance e representa o experimento acadêmico atual. O resultado histórico de US$ 43.759.854,82 não é usado como constante nem como alvo de ajuste.

## Limitação metodológica

O universo de 37 ativos foi obtido retrospectivamente e é tratado como universo congelado. A validação temporal progressiva é aplicada às decisões do modelo, não ao processo histórico de seleção desse universo.
