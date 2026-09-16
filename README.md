# TCC MBA USP — Data Science & Analytics

Backtest reproduzível de rotação de capital entre ativos com LightGBM e validação walk-forward.

## Fonte de dados do teste atual

O `backtest.py` consulta diretamente a API End-of-Day da Tiingo no início da execução e mantém as séries em memória até o fim do processamento.

Para este teste são usados somente os campos brutos:

```text
open
high
low
close
volume
```

Os campos ajustados (`adjOpen`, `adjHigh`, `adjLow`, `adjClose`, `adjVolume`) não entram no treinamento.

A Tiingo também informa `divCash` e `splitFactor`. Esses eventos são mantidos separadamente em memória apenas para auditoria e não são aplicados ao OHLCV neste teste.

Fluxo atual:

```text
Tiingo EOD RAW
  ↓
37 séries mantidas em memória
  ↓
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

Nenhum CSV intermediário é usado pelo `backtest.py` neste teste.

## Preparação

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

Copie o arquivo de exemplo:

Windows:

```bash
copy .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

Preencha o token da Tiingo:

```text
TIINGO_API_KEY=
```

As variáveis da Alpaca permanecem no exemplo porque `baixar_series_alpaca.py` continua disponível como utilitário de comparação:

```text
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
```

## Execução do backtest

```bash
python backtest.py
```

No Spyder, `backtest.py` está organizado em células `# %%`:

```text
# %% 0  Imports e configuração
# %% 1  Início da execução e credencial da Tiingo
# %% 2  Download das séries RAW diretamente para memória
# %% 3  Preparação metodológica
# %% 4  LightGBM + walk-forward + política de rotação
# %% 5  Objetos de resultado para inspeção
# %% 6  Gravação dos artefatos
# %% 7  Métricas finais
```

- `F5`: executa todo o arquivo.
- `Ctrl+Enter`: executa somente a célula atual.
- As variáveis ficam disponíveis no Variable Explorer.

## Eventos corporativos no teste atual

A série utilizada pelo modelo não é retroativamente ajustada. Os eventos recebidos da Tiingo ficam disponíveis na variável:

```text
eventos_corporativos
```

Por ativo, são preservados:

```text
timestamp
dividendo
fator_split
```

Eles ainda não alteram preço, volume, quantidade de ações ou retorno. Essa separação permite investigar posteriormente um tratamento causal de splits e dividendos sem utilizar eventos futuros para modificar observações passadas.

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

## Referências de comparação

A pesquisa já produziu resultados diferentes conforme a fonte e a política de ajuste dos dados. O resultado histórico certificado permanece apenas como referência experimental:

```text
Capital inicial        US$ 10,000.00
Capital final          US$ 43,759,854.82
CAGR                   293.82%
Sharpe                 2.557
Max Drawdown           -28.19%
Rotações               315
```

Esse valor não é tratado como objetivo a ser reproduzido pela Tiingo. O teste atual busca medir o comportamento da mesma estratégia quando alimentada por OHLCV bruto de uma fonte independente.

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

O `backtest_result.json` registra explicitamente:

```text
market_data_source       tiingo_eod_memoria
market_data_adjustment   raw
corporate_actions_applied false
```

## Utilitários históricos

`baixar_series_alpaca.py` continua disponível para baixar séries da Alpaca e comparar fontes.

`exportar_series_certificadas_mongo.py` existe somente para recuperar o snapshot histórico usado na reprodução certificada. Ele não participa do teste Tiingo.

## Auditoria no Excel

Após o backtest:

```bash
python analysis/build_excel.py
```

Arquivo gerado:

```text
analysis/tcc_backtest_output_analysis.xlsx
```

## Estrutura

```text
tcc_mba_usp_data_science_analytics/
├── backtest.py
├── baixar_series_alpaca.py
├── exportar_series_certificadas_mongo.py
├── .env.example
├── requirements.txt
├── analysis/
│   └── build_excel.py
└── tcc_engine/
```

## Limitação metodológica

O universo de 37 ativos foi obtido retrospectivamente e é tratado como universo congelado.

O resultado demonstra o comportamento do motor, da validação walk-forward e da política de rotação sobre esse universo. Ele não demonstra generalização fora da amostra do processo histórico de seleção dos 37 ativos.
