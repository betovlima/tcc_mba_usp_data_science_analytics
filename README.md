# TCC MBA USP — Data Science & Analytics

Backtest reproduzível de rotação de capital entre ativos com LightGBM e validação walk-forward.

## Fonte de dados

O experimento usa séries históricas diárias da Alpaca, com um arquivo CSV por ativo em:

```text
dados/series_historicas/
```

Exemplo:

```text
NVDA.csv
MSFT.csv
META.csv
TSLA.csv
...
SCSC.csv
```

Cada arquivo contém:

```text
timestamp
open
high
low
close
volume
```

Parâmetros usados na Alpaca:

```text
timeframe   1Day
feed        SIP
adjustment  all
```

O `backtest.py` não consulta MongoDB e não baixa dados. Ele executa somente sobre os arquivos históricos locais.

## Preparação das séries históricas

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

Copie o arquivo de exemplo:

```bash
copy .env.example .env
```

No Linux/macOS:

```bash
cp .env.example .env
```

Preencha:

```text
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
```

Depois execute:

```bash
python baixar_series_alpaca.py
```

O script baixa os 37 ativos diretamente da Alpaca e grava cada série em seu próprio arquivo dentro de `dados/series_historicas/`.

A aquisição dos dados fica separada do experimento. Depois que os CSVs existem, o backtest não precisa de conexão com a Alpaca.

## Execução do backtest

```bash
python backtest.py
```

No Spyder, `backtest.py` está organizado em células `# %%`:

```text
# %% 0  Imports e configuração
# %% 1  Início da execução
# %% 2  Carregamento e validação das séries históricas
# %% 3  Preparação metodológica
# %% 4  LightGBM + walk-forward + política de rotação
# %% 5  Objetos de resultado para inspeção
# %% 6  Gravação dos artefatos
# %% 7  Métricas finais
```

- `F5`: executa todo o arquivo.
- `Ctrl+Enter`: executa somente a célula atual.
- As variáveis ficam disponíveis no Variable Explorer.

## Fluxo do experimento

```text
Alpaca SIP
  ↓
CSV estático por ativo
  ↓
OHLCV
  ↓
features
  ↓
target multi-horizonte
  ↓
folds walk-forward
  ↓
treinamento LightGBM por fold
  ↓
predições fora da amostra
  ↓
ranking e política de rotação
  ↓
execução na sessão seguinte
  ↓
custos e capital composto
  ↓
métricas e artefatos
```

Todo o processamento da estratégia é executado pelo código local deste repositório. Não são usadas APIs de processamento, Strategy persistida, modelos treinados, previsões ou resultados anteriores do Market Cycle Trader.

## Universo congelado

37 ativos:

```text
NVDA, MSFT, META, TSLA, AMD, JPM, SPY, AVGO, NFLX,
ORCL, COST, LLY, XOM, CAT, WMT, V, HD, ADC, ADEA,
ADI, ADM, GKOS, VNCE, CORT, UNFI, DNN, MKSI, APD,
DDS, RACE, UNF, TX, CEF, YANG, KKR, BXMT, SCSC
```

Período:

```text
2016-01-01 → 2026-09-04
```

O período inicial é usado para criação das variáveis, alvos e treinamento. A avaliação econômica ocorre somente nas sessões fora da amostra.

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

## Referência histórica certificada

O estado anterior do experimento, ainda lendo o mesmo histórico Alpaca a partir do MongoDB local, reproduziu:

```text
Capital inicial        US$ 10,000.00
Capital final          US$ 43,759,854.82
CAGR                   293.82%
Sharpe                 2.557
Max Drawdown           -28.19%
Rotações               315
```

A alteração atual muda somente a forma de armazenamento e leitura do OHLCV: em vez de MongoDB, um CSV por ativo. O motor da estratégia permanece local no projeto.

## Resultados

Cada execução gera:

```text
output/backtest_result.json
output/equity_curve.csv
output/folds.csv
output/trades.csv
output/summary.txt
```

`output/` permanece fora do Git.

## Auditoria no Excel

Após o backtest:

```bash
python analysis/build_excel.py
```

Arquivo gerado:

```text
analysis/tcc_backtest_output_analysis.xlsx
```

A planilha reconstrói as principais métricas, folds, curva de capital, operações, ciclos, análise mensal e reconciliação financeira.

## Estrutura

```text
tcc_mba_usp_data_science_analytics/
├── backtest.py
├── baixar_series_alpaca.py
├── .env.example
├── requirements.txt
├── dados/
│   └── series_historicas/
│       ├── NVDA.csv
│       ├── MSFT.csv
│       ├── ...
│       └── SCSC.csv
├── analysis/
│   └── build_excel.py
└── tcc_engine/
```

## Limitação metodológica

O universo de 37 ativos foi obtido retrospectivamente e é tratado como universo congelado.

O resultado demonstra o comportamento do motor, da validação walk-forward e da política de rotação sobre esse universo. Ele não demonstra generalização fora da amostra do processo histórico de seleção dos 37 ativos.
