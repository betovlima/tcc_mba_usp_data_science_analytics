# Séries históricas

Este diretório contém a entrada estática do backtest: um CSV por ativo.

Formato esperado de cada arquivo:

```text
timestamp,open,high,low,close,volume
```

Exemplos:

```text
NVDA.csv
MSFT.csv
META.csv
TSLA.csv
...
SCSC.csv
```

As séries são obtidas diretamente da Alpaca com:

```text
timeframe   1Day
feed        SIP
adjustment  all
```

Para criar ou atualizar todos os arquivos:

```bash
python baixar_series_alpaca.py
```

O `backtest.py` apenas lê esses arquivos. Ele não consulta MongoDB nem baixa dados durante a execução do experimento.
