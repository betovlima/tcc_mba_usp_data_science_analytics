# TCC MBA USP — Data Science & Analytics

Backtest reproduzível de rotação de capital entre ativos com LightGBM e validação walk-forward.

## Execução

```bash
git clone https://github.com/betovlima/tcc_mba_usp_data_science_analytics.git
cd tcc_mba_usp_data_science_analytics
python -m pip install -r requirements.txt
python backtest.py
```

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

## Execução no Spyder

`backtest.py` é organizado em células `# %%`.

- `F5`: executa o arquivo completo.
- `Ctrl+Enter`: executa somente a célula atual.
- As variáveis permanecem no namespace e podem ser abertas no Variable Explorer.

Células:

```text
# %% 0  Imports e configuração
# %% 1  Início da execução
# %% 2  Carregamento e validação do OHLCV
# %% 3  Preparação metodológica
# %% 4  LightGBM + walk-forward + política de rotação
# %% 5  Objetos de resultado para inspeção
# %% 6  Gravação dos artefatos
# %% 7  Métricas finais
```

Após a célula 2:

```text
raw
frames
```

Após a célula 4:

```text
results
result
```

Após a célula 5:

```text
predictions
trades
folds
metrics
equity
payload
```

Isso permite estudar o experimento etapa por etapa sem alterar a execução completa por `F5` ou terminal.

## Auditoria no Excel

Após o backtest:

```bash
python analysis/build_excel.py
```

Arquivo gerado:

```text
analysis/tcc_backtest_output_analysis.xlsx
```

A planilha importa o `output/` e reconstrói as principais métricas, folds, curva de capital, operações, ciclos, análise mensal e reconciliação contábil.

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

O período inicial é usado para features, targets e treino. A avaliação econômica é feita somente nas sessões OOS.

## Configuração principal

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

Os parâmetros ficam declarados em `tcc_engine/config.py`.

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

## Walk-forward

```text
Fold 1   US$ 10,000.00       → US$ 69,042.61       +590.43%
Fold 2   US$ 69,042.61       → US$ 2,805,962.94    +3,964.10%
Fold 3   US$ 2,805,962.94    → US$ 43,759,854.82   +1,459.53%
```

O capital final de cada fold é o capital inicial do fold seguinte.

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

Livro-razão de BUY/SELL e diagnósticos da decisão. Inclui scores, ranking, margens, MFE, MAE, custos e métricas contrafactuais.

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

Principais validações por fórmulas do Excel:

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

Reconciliação principal:

```text
capital inicial
+ PnL realizado nas posições encerradas
- taxas das compras
= capital final
```

## Estatísticas dos ciclos

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

Não é turnover anual tradicional.

## Estrutura

```text
tcc_mba_usp_data_science_analytics/
├── backtest.py
├── requirements.txt
├── README.md
├── analysis/
│   ├── build_excel.py
│   └── tcc_backtest_output_analysis.xlsx
└── tcc_engine/
    ├── config.py
    ├── capital_rotation.py
    ├── research_challengers.py
    └── ...
```

`output/` é gerado localmente e permanece fora do Git.

## Limitação metodológica

O universo de 37 ativos foi obtido retrospectivamente e é tratado como universo congelado.

O resultado demonstra reprodutibilidade do motor, do protocolo walk-forward e da política de rotação sobre esse universo. Não demonstra generalização fora da amostra do processo histórico de seleção dos 37 ativos.
