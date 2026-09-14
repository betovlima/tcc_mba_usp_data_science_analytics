# USP MBA Data Science & Analytics — Capstone Project

Projeto de conclusão do MBA em Data Science & Analytics da USP focado na construção reproduzível de um backtest quantitativo de mercado.

## Regra de documentação

Este `README.md` é o único README do projeto. Toda evolução, forma de execução, decisão metodológica e histórico de mudanças deve ser registrada aqui. Não criar READMEs adicionais por versão, pasta ou experimento.

## Objetivo

O objetivo atual é reconstruir o backtest de forma independente, partindo somente dos dados históricos dos ativos.

A execução **não pode reutilizar** Strategy salva, modelos treinados, resultados de backtests anteriores, snapshots de configuração, hashes do Market Cycle Trader ou código importado do projeto original.

A única fonte externa permitida durante a execução é:

```text
MongoDB local
└── alpaca_market_bars
    └── OHLCV diário dos ativos
```

Todo o restante é calculado novamente pelo próprio `backtest.py`.

## Fluxo completo

```text
OHLCV bruto
   ↓
limpeza e alinhamento
   ↓
engenharia de features
   ↓
construção do target multi-horizonte
   ↓
folds walk-forward
   ↓
treinamento LightGBM do zero
   ↓
calibração da margem de rotação
   ↓
treinamento final de cada fold
   ↓
previsões fora da amostra
   ↓
política de rotação
   ↓
execução no próximo open
   ↓
contabilidade do capital
   ↓
CAGR / Sharpe / Max Drawdown / folds
```

## Universo do experimento

O universo é definido explicitamente no código como parte do desenho experimental. Ele não é lido de nenhuma Strategy do banco.

```text
NVDA, MSFT, META, TSLA, AMD, JPM, SPY, AVGO, NFLX,
ORCL, COST, LLY, XOM, CAT, WMT, V, HD, ADC, ADEA,
ADI, ADM, GKOS, VNCE, CORT, UNFI, DNN, MKSI, APD,
DDS, RACE, UNF, TX, CEF, YANG, KKR, BXMT, SCSC
```

Não há busca de ativos neste projeto.

## Metodologia implementada

O `backtest.py` contém, em um único arquivo, toda a cadeia necessária para reproduzir o experimento:

- leitura de candles `open`, `high`, `low`, `close` e `volume`;
- retornos de 1, 2, 3, 5, 10, 20, 40, 60 e 120 sessões;
- volatilidade móvel;
- relações entre volatilidades;
- distâncias e cruzamentos de médias exponenciais;
- RSI de 14 períodos;
- ATR normalizado;
- posição em canais de 20, 50, 100 e 200 sessões;
- eficiência de tendência;
- aceleração de momentum;
- expansão de range;
- variáveis de volume;
- target de utilidade futura em 5, 10, 20, 40 e 60 sessões;
- penalização de downside e drawdown;
- componente de captura de movimento e persistência de tendência;
- walk-forward expansivo com treino, purge, calibração e teste;
- LightGBM treinado novamente em cada fold;
- calibração da margem de troca de ativo;
- política de manter, rotacionar ou ir para CASH;
- execução da mudança de posição na abertura da sessão seguinte;
- custos regulatórios e slippage configurados no próprio experimento;
- curva de capital e benchmark equal-weight;
- métricas finais e métricas por fold.

## Configuração principal

A configuração metodológica está escrita de forma explícita em `ExperimentConfig`, dentro do próprio `backtest.py`.

Principais parâmetros:

```text
capital inicial                 10.000
histórico                       2016-01-01 até 2026-09-04
horizontes do target            5, 10, 20, 40, 60 sessões
pesos dos horizontes            0.10, 0.15, 0.20, 0.30, 0.25
mínimo de treino                700 sessões
calibração                      126 sessões
teste por fold                  504 sessões
mínimo de teste                 126 sessões
purge                           60 sessões
holding mínimo                  2 sessões
margem base de rotação          0.005
candidatos de margem            0, 0.0025, 0.005, 0.01
random_state                    42
```

LightGBM:

```text
n_estimators                    300
learning_rate                   0.035
max_depth                       3
num_leaves                      8
min_child_samples               20
min_child_weight                5.0
subsample                       0.85
subsample_freq                  0
colsample_bytree                0.85
reg_alpha                       0.10
reg_lambda                      2.0
max_bin                         255
```

Esses valores são parâmetros declarados do experimento. Eles não são carregados de uma Strategy existente.

## Estrutura do projeto

```text
tcc_mba_usp_data_science_analytics/
├── README.md
├── requirements.txt
├── .gitignore
└── backtest.py
```

Os artefatos gerados localmente ficam em `output/`, ignorado pelo Git.

## Instalação

```bat
git clone https://github.com/betovlima/tcc_mba_usp_data_science_analytics.git
cd tcc_mba_usp_data_science_analytics
python -m pip install -r requirements.txt
```

## MongoDB local

Por padrão:

```text
URI:      mongodb://localhost:27017
Database: extrema_backtest
Collection: alpaca_market_bars
```

Opcionalmente, crie `.env` na raiz:

```text
TCC_MONGO_URI=mongodb://localhost:27017
TCC_MONGO_DATABASE=extrema_backtest
```

Nenhuma outra coleção do banco faz parte do experimento.

O código filtra o histórico diário por:

```text
interval   = 1Day
feed       = sip
adjustment = all
```

## Execução no Spyder

Abra:

```text
backtest.py
```

Pressione:

```text
F5
```

O mesmo arquivo também pode ser executado pelo terminal:

```bat
python backtest.py
```

## O que deve aparecer no console

A execução mostra as etapas em ordem:

```text
[1/8] carregamento dos dados
[2/8] features e targets
[3/8] construção dos folds walk-forward
[4/8] treinamento para calibração
[5/8] calibração da margem de rotação
[6/8] treinamento final dos folds
[7/8] simulação fora da amostra
[8/8] métricas e artefatos
```

Assim é possível acompanhar no Spyder exatamente onde o backtest está trabalhando.

## Artefatos gerados

Cada execução grava:

```text
output/backtest_result.json
output/equity_curve.csv
output/trades.csv
output/folds.csv
```

Esses arquivos são resultados da execução atual; não são usados como entrada em execuções futuras.

## Reprodutibilidade

O princípio adotado neste projeto é:

> dados de mercado entram; o experimento inteiro é reconstruído.

Portanto:

- não existe leitura de `strategy_profiles`;
- não existe leitura de `backtest_runs`;
- não existe leitura de modelos persistidos;
- não existe checkout de commit do Market Cycle Trader;
- não existe `MCT_REFERENCE_SOURCE`;
- não existe validação por hash de Strategy;
- não existe comparação `PASS/FAIL` contra um capital previamente conhecido;
- não existe acelerador histórico importado;
- não existe dependência da API ou do Front.

O resultado econômico é consequência da execução atual.

## Ressalva acadêmica

O universo de 37 ativos foi definido antes desta reconstrução e não é objeto de validação neste projeto. Portanto, o backtest testa a política de rotação e o protocolo de aprendizado sobre esse universo fixo; ele não demonstra que o processo histórico de escolha dos 37 ativos generaliza fora da amostra.

Essa distinção deve ser mantida no texto final do TCC.

## Histórico do projeto

### 2026-09-14 — Inicialização

- criado o repositório do MBA Capstone Project;
- definido um único `README.md` como documento vivo;
- criado `requirements.txt` independente da stack web;
- criado `backtest.py` como entrypoint para Spyder.

### 2026-09-14 — Primeira reprodução certificada

- foi criada uma primeira versão que importava o motor histórico do Market Cycle Trader e conferia o resultado contra o capital conhecido;
- essa abordagem reproduziu o resultado, mas foi considerada inadequada para o TCC porque reutilizava Strategy, código histórico e metadados de certificação.

### 2026-09-14 — Reconstrução independente a partir de OHLCV

- removido o checkout automático do Market Cycle Trader;
- removida a leitura da Strategy #10;
- removidos hashes de commit, Strategy, request e packages;
- removida a comparação automática com capital histórico;
- removido o acelerador histórico de previsões;
- removidas dependências de FastAPI, Pydantic, Alpaca SDK e XGBoost;
- mantida como única fonte externa a coleção local `alpaca_market_bars`;
- implementada engenharia de features no próprio projeto;
- implementada construção do target multi-horizonte no próprio projeto;
- implementado walk-forward no próprio projeto;
- implementado treinamento LightGBM do zero em cada fold;
- implementada calibração de margem no próprio projeto;
- implementada política de rotação no próprio projeto;
- implementada simulação de operações e capital no próprio projeto;
- implementado cálculo das métricas no próprio projeto;
- mantido `backtest.py` como único entrypoint para Spyder.
