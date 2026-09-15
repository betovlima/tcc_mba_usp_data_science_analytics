# TCC MBA USP — Data Science & Analytics

Backtest de rotação de capital com LightGBM e validação walk-forward.

## Execução

```bash
git clone https://github.com/betovlima/tcc_mba_usp_data_science_analytics.git
cd tcc_mba_usp_data_science_analytics
python -m pip install -r requirements.txt
python backtest.py
```

Fonte de mercado: Yahoo Finance via `yfinance`.

Parâmetros fixos do download:

```text
interval       1d
auto_adjust    True
actions        False
repair         False
```

O `end` do `yfinance` é exclusivo; o script soma um dia para incluir `END_DATE`.

## Spyder

`backtest.py` usa células `# %%`.

```text
# %% 0  Imports e configuração
# %% 1  Início da execução
# %% 2  Download e validação das séries OHLCV
# %% 3  Preparação metodológica
# %% 4  LightGBM + walk-forward + rotação
# %% 5  Objetos de resultado
# %% 6  Gravação dos artefatos
# %% 7  Métricas finais
```

- `F5`: executa tudo.
- `Ctrl+Enter`: executa somente a célula atual.

Após a célula 2, `frames` contém uma série temporal por ativo e `market_data` contém o snapshot consolidado usado no backtest.

## Fluxo

```text
Yahoo Finance
  ↓
OHLCV diário
  ↓
features
  ↓
target multi-horizonte
  ↓
folds walk-forward + purge
  ↓
LightGBM por fold
  ↓
predições OOS
  ↓
ranking e política de rotação
  ↓
execução na sessão seguinte
  ↓
custos e compound
  ↓
métricas
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

## Output

Cada execução gera:

```text
output/market_data.csv
output/backtest_result.json
output/equity_curve.csv
output/folds.csv
output/trades.csv
output/summary.txt
```

### `market_data.csv`

Snapshot exato das séries usadas pelo motor:

```text
symbol
timestamp
open
high
low
close
volume
```

O `backtest_result.json` registra também o SHA-256 desse snapshot.

## Excel

Após o backtest:

```bash
python analysis/build_excel.py
```

Arquivo:

```text
analysis/tcc_backtest_output_analysis.xlsx
```

Abas gerais:

```text
00_Guia
01_KPIs
02_Equity
03_Folds
04_Trades
05_Cycles
06_Assets
07_Monthly
08_Dictionary
09_JSON
10_Reconciliation
11_Ativos
```

Além delas, o Excel cria uma aba para cada ativo:

```text
NVDA
MSFT
META
TSLA
...
SCSC
```

Cada aba de ativo contém a série temporal efetivamente usada pelo backtest:

```text
Date
Open
High
Low
Close
Volume
Daily Return
Running Peak
Drawdown
```

As abas dos ativos são geradas a partir de `output/market_data.csv`; o Excel não baixa novamente os dados.

## Configuração do modelo

Principais parâmetros em `tcc_engine/config.py`:

```text
capital inicial                  10,000
modelo                           LightGBM Utility
n_estimators                     329
learning_rate                    0.020731
max_depth                        3
num_leaves                       6
min_child_samples                18
min_child_weight                 5.0
subsample                        0.85
colsample_bytree                 0.88067
reg_alpha                        0.050837
reg_lambda                       3.596305
holding mínimo                   2 sessões
switch margin base               0.0005
margens de calibração            0, 0.0025, 0.005, 0.01
random_state                     42
```

## Referência histórica

A reprodução certificada anterior, baseada no snapshot histórico antigo, foi preservada na tag:

```text
certified-43m-standalone
```

A `main` usa Yahoo Finance e constitui um novo experimento. Portanto, o resultado de US$ 43.759.854,82 não é tratado como resultado esperado da nova fonte de dados.

## Limitação metodológica

O universo de 37 ativos foi obtido retrospectivamente e é tratado como universo congelado. A validação walk-forward é aplicada às decisões do modelo, não ao processo histórico de seleção desse universo.
