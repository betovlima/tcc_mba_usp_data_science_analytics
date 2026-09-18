# TCC MBA USP — Data Science & Analytics

Backtest reproduzível de rotação de capital entre ativos com LightGBM e validação walk-forward.

## Metodologia atual dos dados

A entrada do experimento é separada em três camadas:

```text
Tiingo EOD
  ↓
OHLCV bruto congelado
  ↓
normalização causal somente de desdobramentos reais
  ↓
features / targets / LightGBM / rotações
```

Os arquivos de preço contêm somente:

```text
timestamp
open
high
low
close
volume
```

Os campos ajustados da Tiingo (`adjOpen`, `adjHigh`, `adjLow`, `adjClose`, `adjVolume`) não são gravados nem usados.

Dividendos são preservados separadamente apenas para auditoria e **não ajustam os preços, as features ou os targets** nesta etapa.

## Por que não usamos diretamente `splitFactor` do endpoint EOD

A documentação da Tiingo informa que o `splitFactor` presente no endpoint EOD pode representar split, reverse split ou distribuição. Portanto ele não é tratado como uma fonte confiável de desdobramentos puros para o modelo.

Os desdobramentos usados pelo backtest vêm do endpoint específico de Corporate Actions:

```text
/tiingo/corporate-actions/<ticker>/splits
```

Esse endpoint fornece:

```text
exDate
splitFrom
splitTo
splitFactor
splitStatus
```

Somente eventos com `splitStatus = a` são congelados e usados.

## Execucao oficial desta branch no Spyder

A branch `research/tiingo-starter-frozen-dataset-v1` usa exclusivamente Tiingo
como fonte do snapshot experimental e o universo completo de 56 ativos.

No Spyder, execute os scripts abaixo nesta ordem, sempre com o diretorio de
trabalho apontando para a raiz do repositorio:

```text
congelar_series_tiingo.py
congelar_desdobramentos_tiingo.py
backtest.py
```

O primeiro script exige somente `TIINGO_API_KEY` no arquivo `.env`.
Os dois passos seguintes trabalham sobre os arquivos locais congelados.

## 1. Congelar o OHLCV bruto

Com `TIINGO_API_KEY` configurada no `.env`:

```bash
python congelar_series_tiingo.py
```

O script grava:

```text
dados/series_historicas/<ATIVO>.csv
dados/eventos_corporativos/<ATIVO>.csv
dados/manifesto_tiingo.json
```

A série histórica permanece imutável e contém somente OHLCV bruto.

## 2. Congelar os desdobramentos reais

Depois execute:

```bash
python congelar_desdobramentos_tiingo.py
```

Esse script consulta somente a API específica de splits e grava:

```text
dados/desdobramentos/<ATIVO>.csv
dados/manifesto_desdobramentos_tiingo.json
```

Cada arquivo possui:

```text
timestamp
split_de
split_para
fator_split
status
```

O script possui retomada automática caso a Tiingo retorne HTTP 429 por limite horário.

## 3. Normalização causal

O `backtest.py` mantém duas estruturas no Spyder:

```text
series_historicas_brutas
series_historicas
```

`series_historicas_brutas` contém exatamente o OHLCV congelado.

`series_historicas` é construída em memória para o modelo. Para cada desdobramento, o fator começa a valer **na data do evento e somente dali para frente**.

Exemplo de split 10:1:

```text
antes do evento      fator acumulado = 1
na data do split     fator acumulado = 10
depois do split      fator acumulado = 10
```

Os preços usados pelo modelo são:

```text
preço_modelo = preço_RAW × fator_acumulado
```

O volume é convertido para a mesma unidade econômica:

```text
volume_modelo = volume_RAW / fator_acumulado
```

Com isso, um split não aparece para o modelo como uma queda artificial de aproximadamente 90%, e nenhum evento futuro modifica o passado.

Dividendos continuam fora dessa normalização.

## 4. Executar o backtest

```bash
python backtest.py
```

Fluxo:

```text
OHLCV RAW congelado
+
desdobramentos reais congelados
  ↓
normalização causal em memória
  ↓
features
  ↓
target multi-horizonte
  ↓
folds walk-forward
  ↓
LightGBM por fold
  ↓
predições fora da amostra
  ↓
ranking e política de rotação
  ↓
capital composto
```

O backtest não consulta a internet.

## Diagnóstico dos desdobramentos

Cada execução também cria:

```text
output/diagnostico_desdobramentos.csv
```

O arquivo mostra, para cada split, o retorno observado no RAW e o retorno depois da normalização causal. Isso permite verificar explicitamente se a ruptura mecânica foi removida.

## Universo congelado

56 ativos:

```text
NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA, AMD, JPM, SPY,
AVGO, NFLX, CRM, ORCL, COST, LLY, XOM, CAT, WMT, V, HD,
ADC, ADEA, ADI, ADM, DDS, CNQ, VRTS, XSD, CXW, ENS, CORT,
CCK, CEF, GKOS, RACE, TX, UNF, BXMT, PXLW, KKR, SCSC, LKFT,
DNN, VNCE, UNFI, DOC, CLMT, APD, MGM, MAN, MYE, YANG, MKSI,
MCS, ECC
```

Período:

```text
2016-01-01 → 2026-09-04
```

## Referências históricas

O resultado de aproximadamente US$ 45,8 mil obtido com Tiingo RAW puro foi apenas um diagnóstico. Ele mostrou o efeito destrutivo de entregar splits não tratados ao motor e não é considerado baseline econômico.

O controle histórico independente continua sendo a tag:

```text
certified-43m-standalone
```

que reproduz aproximadamente:

```text
Capital inicial        US$ 10,000.00
Capital final          US$ 43,759,854.82
```

O objetivo do novo processamento não é forçar a Tiingo a reproduzir esse número. O objetivo é construir uma base metodologicamente limpa: OHLCV bruto congelado, sem ajustes retroativos de dividendos, e tratamento causal explícito somente dos desdobramentos reais.

## Artefatos gerados

```text
output/backtest_result.json
output/equity_curve.csv
output/folds.csv
output/trades.csv
output/summary.txt
output/diagnostico_desdobramentos.csv
```

## Limitação metodológica

O universo de 37 ativos foi obtido retrospectivamente e é tratado como universo congelado. O experimento avalia o motor e a validação walk-forward dentro desse universo; ele não demonstra generalização fora da amostra do processo histórico de seleção dos ativos.


## 5. Otimizacao do LightGBM sem trocar a familia de modelo

Depois de certificar o baseline com `backtest.py`, a primeira fase de busca
usa somente LHS (Latin Hypercube Sampling) sobre os hiperparametros do
LightGBM. Nenhum dado, feature, target, fold, politica de rotacao ou custo e
alterado.

No Spyder, execute:

```text
tunar_lightgbm_tiingo.py
```

O padrao e 32 candidatos LHS mais o `candidate_000`, que reproduz o baseline.
A campanha e retomavel e grava os resultados em:

```text
output/tiingo_56_lightgbm_lhs_v1/
```

Somente depois de concluir o LHS, execute o CARO adaptativo:

```text
tunar_lightgbm_tiingo_caro.py
```

O CARO usa os resultados do LHS como warm-up e mantem o mesmo dataset Tiingo
de 56 ativos e o mesmo processamento split-causal do baseline.



## Aceleracao GPU e cache em memoria

O motor tenta automaticamente o melhor backend disponivel para o LightGBM.

No Windows, o LightGBM nao oferece o backend `device_type=cuda`; a aceleracao
suportada e `device_type=gpu` via OpenCL. Em Linux, o motor tenta CUDA
primeiro, depois GPU/OpenCL e por fim CPU.

A configuracao pode ser sobrescrita no `.env`:

```text
TCC_LIGHTGBM_DEVICE=auto
TCC_LIGHTGBM_GPU_DEVICE_ID=-1
```

Valores aceitos para `TCC_LIGHTGBM_DEVICE`: `auto`, `cpu`, `gpu`,
`cuda`. Se a instalacao local do LightGBM nao possuir o backend acelerado,
o motor detecta isso com um treino minimo e volta automaticamente para CPU.

Durante campanhas LHS/CARO, o painel de features, os folds e as matrizes
X/y de cada fase sao mantidos em RAM e reutilizados entre candidatos. Isso
evita recalcular features e reler/slicar os mesmos dados dezenas de vezes.

