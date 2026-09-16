# Séries históricas congeladas

Este diretório contém a entrada estática do backtest.

A fotografia atual deve ser criada com:

```bash
python congelar_series_tiingo.py
```

A fonte é Tiingo End-of-Day e somente os campos brutos de mercado são usados:

```text
timestamp
open
high
low
close
volume
```

Cada CSV também preserva:

```text
dividendo
fator_split
```

Os eventos corporativos ainda não são aplicados ao OHLCV. Eles ficam armazenados para permitir tratamento causal posterior.

Arquivos esperados:

```text
NVDA.csv
MSFT.csv
...
SCSC.csv
manifesto_tiingo.json
```

Depois de gerado e validado, o snapshot deve permanecer congelado. O `backtest.py` não consulta Tiingo, Alpaca, Yahoo, MongoDB ou qualquer outra fonte externa.

Uma nova coleta da Tiingo deve ser tratada como um novo snapshot e validada antes de substituir a fotografia usada no experimento.
