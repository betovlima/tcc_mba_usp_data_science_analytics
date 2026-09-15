# TCC MBA USP — Data Science & Analytics

Simulação histórica de rotação de capital com LightGBM e validação temporal progressiva.

## Execução

```bash
git clone https://github.com/betovlima/tcc_mba_usp_data_science_analytics.git
cd tcc_mba_usp_data_science_analytics
python -m pip install -r requirements.txt
python backtest.py
```

Fonte de mercado: Yahoo Finance por meio da biblioteca `yfinance`.

Os parâmetros do `yfinance` permanecem com os nomes definidos pela própria biblioteca:

```text
interval       1d
auto_adjust    True
actions        False
repair         False
```

O parâmetro `end` do `yfinance` não inclui a data final. O script soma um dia para incluir `DATA_FIM`.

## Padrão de código

Comentários, funções auxiliares, variáveis locais e textos controlados pelo projeto devem ser escritos em português.

As funções seguem `snake_case`, com nomes descritivos, por exemplo:

```text
serializar_json
ler_argumentos
carregar_dados
montar_ciclos
montar_mensal
montar_resumo_ativos
preparar_serie_ativo
estilizar_aba_ativo
gerar_planilha
```

Nomes definidos por bibliotecas externas ou pelo contrato interno do motor são preservados quando a tradução quebraria compatibilidade. No `backtest.py`, esses nomes são importados com apelidos em português sempre que possível.

## Execução no Spyder

`backtest.py` usa células `# %%`.

```text
# %% 0  Importações e configuração
# %% 1  Início da execução
# %% 2  Download e validação das séries OHLCV
# %% 3  Preparação metodológica
# %% 4  LightGBM + validação temporal + rotação
# %% 5  Objetos de resultado
# %% 6  Gravação dos artefatos
# %% 7  Métricas finais
```

- `F5`: executa o arquivo completo.
- `Ctrl+Enter`: executa somente a célula atual.

Após a célula 2, `quadros_por_ativo` contém uma série temporal por ativo e `dados_mercado` contém a cópia consolidada usada na simulação.

## Fluxo

```text
Yahoo Finance
  ↓
OHLCV diário
  ↓
atributos técnicos
  ↓
alvo multihorizonte
  ↓
janelas de validação temporal
  ↓
LightGBM por janela
  ↓
previsões fora da amostra
  ↓
ranqueamento e política de rotação
  ↓
execução na sessão seguinte
  ↓
custos e capital composto
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

## Arquivos gerados

Cada execução gera:

```text
output/market_data.csv
output/backtest_result.json
output/equity_curve.csv
output/folds.csv
output/trades.csv
output/summary.txt
```

Os nomes desses arquivos e alguns campos internos permanecem estáveis por fazerem parte do contrato de saída do experimento.

### `market_data.csv`

Cópia exata das séries usadas pelo motor:

```text
symbol
timestamp
open
high
low
close
volume
```

`backtest_result.json` registra também o SHA-256 desse conjunto de dados.

## Excel

Após a simulação:

```bash
python analysis/build_excel.py
```

Arquivo gerado:

```text
analysis/tcc_backtest_output_analysis.xlsx
```

Abas gerais:

```text
00_Guia
01_Indicadores
02_Curva
03_Janelas
04_Operacoes
05_Ciclos
06_Ativos
07_Mensal
08_Dicionario
09_JSON
10_Reconciliacao
11_Lista_Ativos
```

Além delas, existe uma aba para cada ativo.

Cada aba contém:

```text
Data
Abertura
Máxima
Mínima
Fechamento
Volume
Retorno diário
Pico acumulado
Queda desde o pico
```

As séries são lidas de `output/market_data.csv`. A planilha não baixa os dados novamente.

## Configuração do modelo

Os parâmetros principais estão em `tcc_engine/config.py`.

Os nomes abaixo são mantidos porque correspondem aos parâmetros esperados pelo LightGBM e pelo motor:

```text
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
random_state                     42
```

Parâmetros da política:

```text
capital inicial                  10,000
permanência mínima               2 sessões
margem base de rotação           0.0005
margens de calibração            0, 0.0025, 0.005, 0.01
```

## Referência histórica

A reprodução certificada anterior foi preservada na tag:

```text
certified-43m-standalone
```

A `main` usa Yahoo Finance e representa um novo experimento. O resultado histórico de US$ 43.759.854,82 não é tratado como resultado esperado da nova fonte de dados.

## Limitação metodológica

O universo de 37 ativos foi obtido retrospectivamente e é tratado como universo congelado. A validação temporal progressiva é aplicada às decisões do modelo, não ao processo histórico de seleção desse universo.
