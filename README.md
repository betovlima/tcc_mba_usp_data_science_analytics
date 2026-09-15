# TCC MBA USP — Data Science & Analytics

Projeto de conclusão do MBA em Data Science & Analytics da USP, focado na construção reproduzível e validação de um backtest de rotação de capital entre ativos usando LightGBM e validação walk-forward.

Este `README.md` é o documento único do projeto para execução, metodologia, arquitetura, limitações e evolução. Não serão criados READMEs separados por versão ou etapa.

## Regra principal de reprodutibilidade

A execução do TCC pode reaproveitar do projeto original somente a base histórica diária dos ativos.

Fonte externa permitida:

```text
MongoDB local
└── extrema_backtest
    └── alpaca_market_bars
        └── OHLCV diário
```

Não são lidos durante a execução:

- Strategy persistida;
- parâmetros de Strategy armazenados no MongoDB;
- modelos treinados;
- previsões persistidas;
- resultados de backtests anteriores;
- jobs da API;
- artefatos do Front;
- hashes de Strategy ou commits do MCT;
- caches de modelos ou de resultados.

Os parâmetros metodológicos do experimento ficam declarados no próprio código deste repositório.

## Fluxo completo

Cada execução reconstrói o experimento desde os candles brutos:

```text
OHLCV bruto do MongoDB local
        ↓
1. limpeza e organização dos candles
        ↓
2. construção das features
        ↓
3. construção dos targets multi-horizonte
        ↓
4. criação dos folds walk-forward
        ↓
5. treinamento LightGBM do zero por fold e por ativo
        ↓
6. previsões somente fora da amostra
        ↓
7. política de rotação
        ↓
8. simulação de BUY / SELL / custos
        ↓
9. curva de capital
        ↓
10. CAGR / Sharpe / Max Drawdown / demais métricas
```

Nenhuma etapa intermediária é carregada pronta.

## Execução

### 1. Instalação

```bash
git clone https://github.com/betovlima/tcc_mba_usp_data_science_analytics.git
cd tcc_mba_usp_data_science_analytics
python -m pip install -r requirements.txt
```

### 2. MongoDB local

Por padrão:

```text
URI        mongodb://localhost:27017
Database   extrema_backtest
Collection alpaca_market_bars
Interval   1Day
Feed       sip
Adjustment all
```

Opcionalmente, crie um `.env` local:

```text
TCC_MONGO_URI=mongodb://localhost:27017
TCC_MONGO_DATABASE=extrema_backtest
```

O programa rejeita MongoDB remoto para esta execução acadêmica.

### 3. Spyder

Abra:

```text
backtest.py
```

e pressione `F5`.

### 4. Terminal

```bash
python backtest.py
```

## Estrutura

```text
tcc_mba_usp_data_science_analytics/
├── backtest.py
├── requirements.txt
├── README.md
├── .gitignore
└── tcc_engine/
```

`backtest.py` é o ponto de entrada estável. `tcc_engine/` contém somente o código necessário para construir features, targets, modelos, política, execução e métricas dentro do próprio projeto.

## Configuração do experimento

Período bruto:

```text
2016-01-01 → 2026-09-04
```

Capital inicial:

```text
US$ 10.000,00
```

Universo fixo de 37 ativos:

```text
NVDA, MSFT, META, TSLA, AMD, JPM, SPY, AVGO, NFLX,
ORCL, COST, LLY, XOM, CAT, WMT, V, HD, ADC, ADEA,
ADI, ADM, GKOS, VNCE, CORT, UNFI, DNN, MKSI, APD,
DDS, RACE, UNF, TX, CEF, YANG, KKR, BXMT, SCSC
```

O projeto não executa nova descoberta ou seleção de ativos. O foco do TCC é a reconstrução reproduzível do backtest para esse universo.

### Target e validação temporal

```text
Horizontes do target        5, 10, 20, 40 e 60 sessões
Pesos                       0,10 / 0,15 / 0,20 / 0,30 / 0,25
Treino mínimo               700 sessões
Calibração                  126 sessões
Teste por fold              504 sessões
Teste mínimo                126 sessões
Purge                       60 sessões
Holding mínimo              2 sessões
Random state                42
```

### LightGBM

O LightGBM é treinado novamente a cada execução e a cada fold. Os hiperparâmetros usados no experimento são declarados em `tcc_engine/config.py` e não são carregados do MongoDB.

Parâmetros principais:

```text
n_estimators        329
learning_rate       0.020731
max_depth           3
num_leaves          6
min_child_samples   18
min_child_weight    5.0
subsample           0.85
colsample_bytree    0.88067
reg_alpha           0.050837
reg_lambda          3.596305
random_state        42
```

## Walk-forward

A avaliação usa separação temporal. O histórico anterior ao período de teste é usado para treinamento e calibração; o modelo é avaliado somente em dados posteriores que não participaram do treinamento.

Não é usado split aleatório.

O `purge` cria uma separação adicional entre treino/calibração e teste para reduzir contaminação temporal dos targets futuros.

## Saídas

Cada execução grava em `output/`:

```text
backtest_result.json
summary.txt
equity_curve.csv
trades.csv
folds.csv
```

Esses arquivos são resultados da execução. Eles nunca são usados como entrada de uma execução posterior.

### `backtest_result.json`

Contém os parâmetros acadêmicos básicos e as métricas produzidas pela execução atual. Não contém certificação contra capital histórico nem hashes de Strategy do MCT.

### `equity_curve.csv`

Curva de capital fora da amostra e informações de decisão disponíveis no resultado.

### `trades.csv`

Operações geradas pela própria execução atual.

### `folds.csv`

Janelas temporais e métricas dos folds walk-forward.

### `summary.txt`

Resumo textual do backtest atual.

## Referência histórica

O projeto original chegou anteriormente a aproximadamente US$ 43,76 milhões partindo de US$ 10 mil para esse universo e metodologia.

Esse valor é somente uma referência histórica para comparação posterior. Ele não é armazenado no código como condição de sucesso, não interrompe a execução e não participa de nenhuma decisão do backtest.

O resultado válido do TCC é sempre o que for reconstruído a partir do OHLCV bruto pela execução atual.

## Limitação metodológica

O backtest é avaliado em walk-forward, porém o universo final de 37 ativos foi obtido anteriormente por análise retrospectiva. Portanto, o experimento avalia a reprodutibilidade da estratégia para um universo fixo; ele não demonstra generalização fora da amostra do processo de escolha desses 37 ativos.

## Evolução do projeto

Todas as próximas alterações metodológicas, instruções de execução e decisões do TCC serão registradas nesta seção do mesmo `README.md`.

### 2026-09-15 — execução independente

- removida a dependência de Strategy persistida;
- removidos hashes de Strategy, commit, request e modelo da execução acadêmica;
- removida a certificação obrigatória contra resultado histórico;
- MongoDB restrito ao OHLCV diário;
- LightGBM continua sendo treinado do zero em cada execução;
- features, targets, folds, decisões, operações e métricas continuam sendo reconstruídos pelo projeto;
- documentação consolidada neste único `README.md`.
