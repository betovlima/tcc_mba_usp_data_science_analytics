# TCC MBA USP — Data Science & Analytics

Backtest reproduzível de rotação de capital entre ativos com LightGBM e validação walk-forward.

## Execução

```bash
git clone https://github.com/betovlima/tcc_mba_usp_data_science_analytics.git
cd tcc_mba_usp_data_science_analytics
python -m pip install -r requirements.txt
python backtest.py
```

No Spyder, abra `backtest.py` e pressione `F5`.

MongoDB local padrão:

```text
URI        mongodb://localhost:27017
Database   extrema_backtest
Collection alpaca_market_bars
Interval   1Day
Feed       sip
Adjustment all
```

Opcionalmente:

```text
TCC_MONGO_URI=mongodb://localhost:27017
TCC_MONGO_DATABASE=extrema_backtest
```

A execução rejeita MongoDB remoto.

## Auditoria no Excel

Após o backtest:

```bash
python analysis/build_excel.py
```

Arquivo gerado:

```text
analysis/tcc_backtest_output_analysis.xlsx
```

A planilha importa o `output/` e reconstrói no Excel as principais métricas, folds, curva de capital, operações, ciclos, análise mensal e reconciliação contábil.

## Fluxo do experimento

```text
OHLCV bruto
  ↓
features
  ↓
target multi-horizonte
  ↓
folds walk-forward
  ↓
treinamento LightGBM por fold
  ↓
predições OOS
  ↓
ranking e política de rotação
  ↓
execução na sessão seguinte
  ↓
custos e compound
  ↓
métricas e artefatos
```

Todo o processamento é executado pelo código local deste repositório.

## Fonte de dados

Entrada de runtime:

```text
extrema_backtest.alpaca_market_bars
```

Campos utilizados:

```text
symbol
timestamp
open
high
low
close
volume
```

Não são usados como entrada:

- Strategy persistida;
- modelo treinado persistido;
- previsões persistidas;
- resultado de backtest anterior;
- jobs ou serviços externos de processamento.

## Universo congelado

37 ativos:

```text
NVDA, MSFT, META, TSLA, AMD, JPM, SPY, AVGO, NFLX,
ORCL, COST, LLY, XOM, CAT, WMT, V, HD, ADC, ADEA,
ADI, ADM, GKOS, VNCE, CORT, UNFI, DNN, MKSI, APD,
DDS, RACE, UNF, TX, CEF, YANG, KKR, BXMT, SCSC
```

Histórico bruto:

```text
2016-01-01 → 2026-09-04
```

O período inicial é usado para construção de features, targets e treino. A avaliação econômica é feita somente nas sessões OOS.

## Configuração certificada

Principais parâmetros:

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

## Resultado reproduzido

```text
Capital inicial        US$ 10,000.00
Capital final          US$ 43,759,854.819224246
CAGR                   293.8231%
Sharpe                 2.557370
Max Drawdown           -28.1934%
Rotações               315
Compras                316
Vendas                 316
CASH days              0
Exposição              100%
Sessões OOS            1,538
```

O capital final reproduz exatamente a referência histórica certificada.

## Walk-forward

```text
Fold 1   US$ 10,000.00       → US$ 69,042.61       +590.43%
Fold 2   US$ 69,042.61       → US$ 2,805,962.94    +3,964.10%
Fold 3   US$ 2,805,962.94    → US$ 43,759,854.82   +1,459.53%
```

O capital final de cada fold é o capital inicial do fold seguinte.

## Fingerprints

A execução valida os principais fingerprints antes do treinamento:

```text
Strategy configuration
509b940659a89a7348be3690882213c839ce1a43b7e44057656074f5b2517a6e

Model settings
b4d112d678f79ca931c24630831e6464ebfe492f46364f54632c2980618803ab

Execution request
8aa99e2c5a9e4cdf666cbfa406896b1aee82f2fbe9ea65d68ad077e8b8be73a6

Market OHLCV
2db920471bc6ff8925081735c4d8218adf879a1363fae7fd239da940d6ebe30c
```

Os hashes são usados somente para validar reprodutibilidade; não participam da decisão de investimento.

## Output

Cada execução gera:

```text
output/backtest_result.json
output/equity_curve.csv
output/folds.csv
output/trades.csv
output/summary.txt
```

### `backtest_result.json`

Resultado canônico: métricas, configuração, folds, metadados e diagnósticos agregados.

### `equity_curve.csv`

Curva OOS por sessão. Permite recalcular retorno, CAGR, Sharpe, drawdown e benchmark.

### `folds.csv`

Janelas de treino, calibração, purge e teste, com capital e desempenho por fold.

### `trades.csv`

Livro-razão completo de BUY/SELL e diagnósticos da decisão. Inclui scores, ranking, margens, MFE, MAE, custos e métricas contrafactuais.

### `summary.txt`

Resumo textual da execução.

## Auditoria financeira no Excel

A planilha gerada por `analysis/build_excel.py` contém:

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
```

Principais validações reproduzidas por fórmulas do Excel:

```text
capital final
retorno total
CAGR
Sharpe
Max Drawdown
compras e vendas
rotações
holding médio
retorno geométrico por posição
taxas
folds
reconciliação do capital
```

A reconciliação principal é:

```text
capital inicial
+ PnL realizado nas posições encerradas
- taxas das compras
= capital final
```

## Estatísticas dos ciclos

Na execução certificada:

```text
Ciclos encerrados       316
Vencedores              204
Perdedores              112
Win rate                64.56%
Retorno médio vencedor  +6.58%
Retorno médio perdedor  -3.46%
Payoff                  1.90
Profit Factor           2.79
Mediana por posição     +1.37%
Melhor posição          +84.34%
Pior posição            -14.32%
MFE médio               +6.63%
MAE médio               -3.43%
Profit capture médio    45.97%
```

## Turnover

`turnover_ratio` nesta implementação é:

```text
soma do valor bruto negociado / capital inicial
```

Não deve ser interpretado como turnover anual tradicional.

## Estrutura

```text
tcc_mba_usp_data_science_analytics/
├── backtest.py
├── requirements.txt
├── README.md
├── analysis/
│   └── build_excel.py
└── tcc_engine/
    ├── config.py
    ├── capital_rotation.py
    ├── research_challengers.py
    └── ...
```

`output/` é gerado localmente e permanece fora do Git.

## Limitação metodológica

O universo de 37 ativos foi obtido retrospectivamente e é tratado aqui como universo congelado.

O resultado demonstra reprodutibilidade do motor, do protocolo walk-forward e da política de rotação sobre esse universo. Ele não demonstra generalização fora da amostra do processo histórico de seleção dos 37 ativos.
