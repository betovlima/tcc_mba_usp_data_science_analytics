# Séries históricas

Este diretório contém a entrada estática e congelada do backtest: um CSV por ativo.

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

## Base certificada do experimento

Para reproduzir o resultado histórico certificado, os CSVs devem representar exatamente o mesmo snapshot OHLCV usado pelo backtest certificado.

A Alpaca permite baixar dados com:

```text
timeframe   1Day
feed        SIP
adjustment  all
```

Porém, `adjustment=all` incorpora eventos corporativos. Dividendos e outros eventos processados posteriormente podem reajustar retroativamente preços históricos. Portanto, um novo download feito em outra data pode ser correto e ainda assim não ser numericamente idêntico ao snapshot usado anteriormente.

Para migrar uma única vez o snapshot certificado armazenado no MongoDB local para os CSVs deste diretório:

```bash
python exportar_series_certificadas_mongo.py
```

Depois dessa migração, o `backtest.py` lê somente os arquivos CSV. O MongoDB não participa da execução do experimento.

## Download atual da Alpaca

O script abaixo continua disponível para obter uma fotografia atual dos dados da Alpaca:

```bash
python baixar_series_alpaca.py
```

Um novo download não deve ser usado para substituir silenciosamente a base certificada sem uma nova validação do backtest.
