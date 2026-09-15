# TCC MBA USP — Data Science & Analytics

Backtest quantitativo reproduzível com rotação de capital entre ativos usando LightGBM e validação walk-forward.

## Execução

### 1. Instalar

```bash
git clone https://github.com/betovlima/tcc_mba_usp_data_science_analytics.git
cd tcc_mba_usp_data_science_analytics
python -m pip install -r requirements.txt
```

### 2. MongoDB local

Fonte de dados usada pelo experimento:

```text
URI        mongodb://localhost:27017
Database   extrema_backtest
Collection alpaca_market_bars
Interval   1Day
Feed       sip
Adjustment all
```

Configuração opcional em `.env`:

```text
TCC_MONGO_URI=mongodb://localhost:27017
TCC_MONGO_DATABASE=extrema_backtest
```

Somente OHLCV diário é utilizado como entrada de mercado.

### 3. Executar

```bash
python backtest.py
```

No Spyder, abra `backtest.py` e pressione `F5`.

## Arquitetura

```text
OHLCV MongoDB
    ↓
backtest.py
    ↓
tcc_engine/
    ├── features e targets
    ├── walk-forward
    ├── LightGBM
    ├── calibração da margem
    ├── política de rotação
    ├── execução e custos
    └── métricas
    ↓
output/
```

O runtime não lê Strategy persistida, modelo treinado, previsão histórica ou resultado anterior. O código do motor necessário ao experimento está versionado localmente em `tcc_engine/`.

## Experimento congelado

```text
Histórico bruto       2016-01-01 → 2026-09-04
Capital inicial       US$ 10.000
Universo              37 ativos
Modelo                LightGBM Utility
Target                5, 10, 20, 40 e 60 sessões
Treino mínimo         700 sessões
Calibração            126 sessões
Teste por fold        504 sessões
Purge                  60 sessões
Holding mínimo        2 sessões
```

Universo:

```text
NVDA, MSFT, META, TSLA, AMD, JPM, SPY, AVGO, NFLX,
ORCL, COST, LLY, XOM, CAT, WMT, V, HD, ADC, ADEA,
ADI, ADM, GKOS, VNCE, CORT, UNFI, DNN, MKSI, APD,
DDS, RACE, UNF, TX, CEF, YANG, KKR, BXMT, SCSC
```

O universo de 37 ativos é fixo. A seleção histórica desses ativos não é refeita durante o backtest.

## Reprodutibilidade

Antes do treinamento, o programa valida fingerprints do experimento certificado:

- configuração do modelo;
- request de execução;
- OHLCV congelado.

Esses hashes são controles de integridade e não entram nas decisões de trading.

Resultado reproduzido:

```text
Capital final     US$ 43.759.854,82
Retorno total     437.498,55%
CAGR              293,82%
Sharpe            2,5574
Max Drawdown      -28,19%
Rotações          315
CASH days         0
Sessões OOS       1.538
```

O período OOS inicia em `2020-07-22`. O histórico anterior é usado para features e treinamento.

## Walk-forward

| Fold | Capital inicial | Capital final | Retorno |
|---|---:|---:|---:|
| 1 | US$ 10.000,00 | US$ 69.042,61 | +590,43% |
| 2 | US$ 69.042,61 | US$ 2.805.962,94 | +3.964,10% |
| 3 | US$ 2.805.962,94 | US$ 43.759.854,82 | +1.459,53% |

O capital final de cada fold é o capital inicial do fold seguinte.

## Output

Cada execução grava em `output/`:

### `backtest_result.json`

Resultado canônico do experimento: parâmetros, métricas, folds, fingerprints e metadados.

### `equity_curve.csv`

Curva diária OOS da estratégia e benchmark. Permite recalcular retorno, CAGR, Sharpe e drawdown.

### `folds.csv`

Janelas de treino, calibração, purge e teste, além do desempenho de cada fold.

### `trades.csv`

Livro-razão das operações e diagnósticos das decisões. Contém BUY, SELL, FINAL_SELL, scores, ranking, margens, MFE, MAE e campos contrafactuais.

### `summary.txt`

Resumo textual da execução.

Os arquivos de `output/` são resultados. Não são usados como entrada de uma nova execução.

## Auditoria em Excel

Arquivo versionado:

```text
analysis/tcc_backtest_output_analysis.xlsx
```

A planilha reconstrói a camada financeira do backtest a partir do output e compara os cálculos com o resultado do motor.

Validações principais:

```text
Capital final                  diferença 0
Retorno total                  diferença 0
CAGR                           diferença 0
Sharpe                         diferença numérica ~0
Max Drawdown                   diferença 0
Compras                        316
Vendas                         316
Rotações                       315
Taxas totais                   US$ 43.241,14
```

Reconciliação contábil:

```text
US$ 10.000,00
+ PnL realizado nas saídas
- taxas das compras
= US$ 43.759.854,82
```

A planilha também contém:

- análise dos 316 ciclos completos de posição;
- análise por ativo;
- retornos mensais;
- análise dos folds;
- dicionário dos campos de `trades.csv`;
- reconciliação motor x fórmulas do Excel.

### Estatísticas dos ciclos

```text
Ciclos                  316
Vencedores              204
Perdedores              112
Win rate                64,56%
Retorno médio vencedor  +6,58%
Retorno médio perdedor  -3,46%
Payoff                  1,90
Profit Factor           2,79
Mediana por ciclo       +1,37%
Melhor ciclo            +84,34%
Pior ciclo              -14,32%
MFE médio               +6,63%
MAE médio               -3,43%
Profit capture médio    45,97%
```

O Excel consegue reproduzir métricas, contabilidade e análise das decisões já produzidas. Ele não substitui o treinamento do LightGBM nem recria os scores a partir do OHLCV; essa etapa permanece no `tcc_engine`.

## Turnover

`turnover_ratio` neste experimento é:

```text
soma do valor bruto negociado / capital inicial
```

Não deve ser interpretado como turnover anual tradicional de carteira.

## Estrutura

```text
tcc_mba_usp_data_science_analytics/
├── analysis/
│   └── tcc_backtest_output_analysis.xlsx
├── tcc_engine/
├── backtest.py
├── requirements.txt
├── TCC_EXECUTION_CONSTRAINTS.md
└── README.md
```

## Limitação metodológica

O motor é avaliado walk-forward, mas o universo final de 37 ativos foi obtido em estudo histórico retrospectivo. Portanto, o resultado demonstra reprodutibilidade do mecanismo para o universo congelado; não demonstra generalização fora da amostra do processo de seleção dos ativos.
