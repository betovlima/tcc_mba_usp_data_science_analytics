# TCC MBA USP — Data Science & Analytics

Backtest reproduzível de rotação de capital entre ativos com LightGBM e validação walk-forward.

## Entrada congelada do experimento

A fonte atual é a API End-of-Day da Tiingo. A aquisição dos dados fica separada do backtest:

```text
Tiingo EOD
  ↓
congelar_series_tiingo.py
  ↓
snapshot local versionável
  ↓
backtest.py
```

O snapshot usa somente os preços brutos:

```text
open
high
low
close
volume
```

Os campos ajustados da Tiingo (`adjOpen`, `adjHigh`, `adjLow`, `adjClose`, `adjVolume`) não são gravados nem usados pelo modelo.

Também são preservados, separadamente:

```text
dividendo
fator_split
```

Esses eventos ainda não alteram o OHLCV nem a carteira. Eles serão usados posteriormente para estudar um tratamento causal de eventos corporativos.

## 1. Congelar a fotografia da Tiingo

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

Crie `.env` a partir do exemplo e informe:

```text
TIINGO_API_KEY=SEU_TOKEN
```

Depois execute uma única vez:

```bash
python congelar_series_tiingo.py
```

O script faz uma consulta por ativo e grava:

```text
dados/series_historicas/NVDA.csv
dados/series_historicas/MSFT.csv
...
dados/series_historicas/SCSC.csv
dados/series_historicas/manifesto_tiingo.json
```

Cada CSV possui:

```text
timestamp
open
high
low
close
volume
dividendo
fator_split
```

O `manifesto_tiingo.json` registra a data do congelamento, período, universo e contagem de eventos. Não são usados hashes.

Depois de congelada, essa fotografia deve ser mantida sem substituição silenciosa. Uma nova coleta representa um novo snapshot de dados e deve ser validada separadamente.

## 2. Executar o backtest

```bash
python backtest.py
```

O backtest não consulta internet, não usa token e não baixa dados. Ele lê somente a fotografia local congelada e executa:

```text
OHLCV bruto congelado
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

No Spyder, `backtest.py` continua organizado em células `# %%`.

## Universo

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

## Resultado de referência — Tiingo RAW direto

Antes de congelar a fotografia, a execução direta Tiingo → memória de 16/09/2026 produziu:

```text
Capital inicial        US$ 10,000.00
Capital final          US$ 45,766.77
Retorno total          357.67%
CAGR                   28.22%
Sharpe                  0.711
Max Drawdown           -56.09%
Rotações                238
```

Esse resultado é uma referência para validar o snapshot congelado. Se a Tiingo devolver os mesmos registros no momento do congelamento, o backtest local deverá reproduzir numericamente essa execução.

O resultado também mostra que OHLCV bruto sem tratamento de splits não é a metodologia final: splits permanecem como descontinuidades econômicas artificiais na série. A fotografia RAW é preservada justamente para permitir que o tratamento causal seja desenvolvido de forma reproduzível.

## Comparações já observadas

```text
Tiingo RAW atual        ≈ US$ 45.8 mil
Alpaca RAW              ≈ US$ 76.2 mil
Yahoo Finance           ≈ US$ 1.25 milhão
Alpaca SPLIT            ≈ US$ 3.34 milhões
Alpaca ALL atual        ≈ US$ 22.02 milhões
Snapshot histórico      ≈ US$ 43.76 milhões
```

Esses valores não são usados para escolher a fonte pelo maior capital. Eles demonstram o impacto da origem e da política de ajuste dos dados sobre o experimento.

## Resultados gerados

Cada execução cria somente:

```text
output/backtest_result.json
output/equity_curve.csv
output/folds.csv
output/trades.csv
output/summary.txt
```

O arquivo residual `output/market_data.csv` de versões antigas é removido automaticamente, evitando misturar dados de execuções anteriores.

## Estrutura principal

```text
tcc_mba_usp_data_science_analytics/
├── backtest.py
├── congelar_series_tiingo.py
├── baixar_series_alpaca.py
├── exportar_series_certificadas_mongo.py
├── dados/
│   └── series_historicas/
│       ├── manifesto_tiingo.json
│       ├── NVDA.csv
│       ├── ...
│       └── SCSC.csv
├── analysis/
│   └── build_excel.py
└── tcc_engine/
```

## Limitação metodológica

O universo de 37 ativos foi obtido retrospectivamente e é tratado como universo congelado. O resultado demonstra o comportamento do motor e da validação walk-forward nesse universo, não a generalização fora da amostra do processo histórico de seleção dos ativos.
