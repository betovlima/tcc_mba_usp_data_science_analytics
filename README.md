# TCC MBA USP — Rotação de Capital com Machine Learning

Este projeto implementa uma pesquisa reproduzível de rotação de capital entre
ativos financeiros usando Machine Learning. O objetivo é estudar se uma
política que escolhe dinamicamente onde manter o capital pode melhorar o
crescimento composto quando comparada a uma estratégia simples de comprar e
manter.

O experimento utiliza dados diários de mercado, LightGBM, validação temporal
walk-forward e uma única conta de capital reinvestida ao longo do tempo. O
projeto não depende de MongoDB nem de serviços do Market Cycle Trader em tempo
de execução.

## O que o sistema faz

O pipeline baixa da Alpaca barras OHLCV diárias em formato RAW usando o feed
SIP e consulta Corporate Actions para tratar eventos como splits. Os arquivos
são armazenados localmente em CSV, um por ativo, e um `manifest.json` registra
a identidade do snapshot e hashes SHA-256 para permitir reproduções posteriores.

Antes do treinamento, o pipeline valida a continuidade dos dados. Quando um
ativo apresenta um problema estrutural de identidade, histórico ou origem, ele
é excluído de forma explícita em vez de ter sua série reconstruída manualmente.
Na execução validada, DOC foi excluído por uma mudança estrutural associada à
operação DOC -> PEAK.

Os splits são normalizados em memória. Depois disso o sistema calcula variáveis
de retorno, tendência, volatilidade, médias móveis, RSI, ATR, posição em canais
de preço, eficiência de tendência e volume. O LightGBM aprende uma utilidade
multi-horizonte usando horizontes de 5, 10, 20, 40 e 60 sessões.

A avaliação é cronológica. Cada fold possui período de treinamento, calibração,
purge temporal e teste fora da amostra. Assim, uma decisão em uma determinada
data utiliza apenas informações disponíveis antes dela.

As mudanças de posição são executadas na abertura da sessão seguinte. Todo o
capital pertence a uma única conta e é reinvestido após cada rotação.

## Pesquisa sobre rotação de capital

A pesquisa compara três comportamentos sobre o mesmo capital inicial de
**US$ 10.000**.

**Control** é a política-base. O LightGBM estima a utilidade dos ativos e a
estratégia decide permanecer no ativo atual, trocar para outro ativo ou ficar em
caixa. Uma margem mínima evita rotações quando a vantagem prevista é pequena.

**Soft Horizon Consensus** utiliza exatamente o mesmo LightGBM e a mesma
política-base, mas consulta a concordância entre os horizontes de 5, 10, 20, 40
e 60 sessões. Quando o suporte entre horizontes é menor, a margem necessária
para uma troca aumenta. O Soft não escolhe outro ativo por conta própria: ele
apenas aceita a decisão-base ou bloqueia uma rotação marginal.

**Comprar e manter** é o benchmark. O capital inicial é distribuído em pesos
iguais entre os ativos com preços completos na janela de execução e as posições
são mantidas. Esse benchmark usa o mesmo período histórico e o mesmo capital
inicial da estratégia de rotação.

A reprodução validada utiliza dados de 2016-01-01 até 2026-09-17. Foram
solicitados 56 ativos e 55 permaneceram elegíveis. A avaliação fora da amostra
contém 1.546 sessões distribuídas em três folds.

### Resultado da reprodução validada

| Estratégia | Capital inicial | Capital final | Retorno total | CAGR | Sharpe | Máx. Drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | US$ 10.000,00 | US$ 10.094.316,30 | +100.843,16% | 207,83% | 2,102 | -31,22% |
| Soft Horizon Consensus | US$ 10.000,00 | US$ 9.851.632,93 | +98.416,33% | 206,62% | 2,101 | -31,22% |
| Comprar e manter | US$ 10.000,00 | US$ 39.001,99 | +290,02% | 24,76% | 1,154 | -28,15% |

No experimento validado, o Control terminou com aproximadamente **258,8 vezes**
o capital final do benchmark comprar-e-manter, enquanto o Soft terminou com
aproximadamente **252,6 vezes** esse benchmark. O Soft modificou somente 5 das
1.546 decisões da política-base e terminou 2,40% abaixo do Control.

Esses números descrevem um backtest histórico com validação temporal fora da
amostra. Eles não constituem previsão nem garantia de desempenho futuro.

## Tecnologias utilizadas

- **Python 3.12** como ambiente de referência do projeto e do CI.
- **Pandas** e **NumPy** para séries temporais, transformação e cálculo numérico.
- **Matplotlib** para gráficos reproduzíveis destinados à análise e ao TCC.
- **LightGBM** para os modelos de regressão de utilidade.
- **scikit-learn** como dependência do ecossistema de modelagem usado pelo
  LightGBM.
- **threadpoolctl** para controle do paralelismo numérico.
- **alpaca-py** e **requests** para obtenção dos dados de mercado.
- **python-dotenv** para carregar as credenciais locais da Alpaca.
- **CSV + JSON + SHA-256** para congelamento e auditoria do snapshot de dados.
- **pytest** para testes automatizados.
- **Ruff** para análise estática do código.
- **GitHub Actions** para executar Ruff e pytest em cada atualização relevante.
- **Spyder** é opcional e pode ser usado para executar o experimento célula por
  célula e inspecionar as variáveis intermediárias.

## Estrutura do projeto

```text
.
├── dados/
│   └── reproducao_v1/
│       ├── raw_bars/
│       ├── corporate_actions/
│       └── manifest.json
├── engine/
│   ├── config.py
│   ├── diagnostics.py
│   ├── execution.py
│   ├── lightgbm.py
│   └── rotation.py
├── reproducao/
│   ├── artefatos.py
│   ├── dados.py
│   ├── experimento.py
│   └── preparacao.py
├── visualizacao/
│   └── heatmap_mensal.py
├── tests/
├── reproduzir_experimento_spyder.py
├── gerar_graficos.py
├── requirements.txt
└── .env.example
```

A pasta `dados/reproducao_v1/` e a pasta `output/` são locais e não são
versionadas. Depois que o snapshot é criado e validado, o experimento pode ser
reexecutado offline usando os mesmos CSVs.

## Pré-requisitos

Para criar um snapshot novo são necessários:

1. Python 3.12.
2. Acesso à internet.
3. Uma conta Alpaca.
4. Uma API Key e uma Secret Key da Alpaca com acesso aos dados históricos
   necessários, incluindo o feed SIP utilizado pelo experimento.
5. Git para baixar o projeto.

A Alpaca utiliza **duas credenciais**, não um único token: uma API Key e uma
Secret Key. Elas são usadas somente para criar ou substituir o snapshot local.
Depois disso, com o manifesto e os CSVs presentes, a reprodução pode funcionar
sem acessar a Alpaca.

Nunca adicione as credenciais ao Git.

## Instalação

Clone o projeto e entre no diretório:

```bash
git clone https://github.com/betovlima/tcc_mba_usp_data_science_analytics.git
cd tcc_mba_usp_data_science_analytics
```

Crie um ambiente virtual:

```bash
python -m venv .venv
```

No Windows usando Git Bash:

```bash
source .venv/Scripts/activate
```

No Linux ou macOS:

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuração das credenciais da Alpaca

Copie o arquivo de exemplo:

```bash
cp .env.example .env
```

Preencha o arquivo `.env`:

```text
ALPACA_API_KEY=sua_api_key
ALPACA_SECRET_KEY=sua_secret_key
```

O arquivo `.env` é ignorado pelo Git.

Na primeira execução, se ainda não existir
`dados/reproducao_v1/manifest.json`, o sistema carrega essas credenciais e
baixa os dados necessários.

## Executando a pesquisa

O modo mais simples é executar:

```bash
python reproduzir_experimento_spyder.py
```

O fluxo executado é:

```text
Alpaca RAW/SIP
    -> CSV OHLCV por ativo
    -> Corporate Actions
    -> manifest.json + SHA-256
    -> validação estrutural
    -> normalização de splits
    -> features e targets
    -> folds walk-forward
    -> LightGBM CPU
    -> Control
    -> Soft Horizon Consensus
    -> comparação e exportação
```

Por padrão:

```python
FORCAR_DOWNLOAD = False
```

Mantenha esse valor em `False` para preservar e reutilizar o snapshot
existente. Altere para `True` somente quando a intenção for substituir os CSVs
locais e construir deliberadamente um novo snapshot.

## Reexecução offline

Se estes itens já existirem:

```text
dados/reproducao_v1/raw_bars/
dados/reproducao_v1/corporate_actions/
dados/reproducao_v1/manifest.json
```

e `FORCAR_DOWNLOAD = False`, o programa valida os hashes e reutiliza os dados
locais. Nesse caso não é necessário baixar novamente os dados da Alpaca.

## Execução no Spyder

Abra `reproduzir_experimento_spyder.py` e execute as células `# %%` de cima
para baixo:

```text
0  configuração
1  credenciais ou replay offline
2  barras OHLCV RAW
3  Corporate Actions
4  manifesto e SHA-256
5  preparação dos dados
6  configurações e folds
7  Control
8  Soft Horizon Consensus
9  comparação
10 exportação
```

O Variable Explorer permite inspecionar, entre outras:

```text
frames
ativos_elegiveis
folds
config_control
config_soft
control_result
control_metrics
soft_result
soft_metrics
comparacao
```

## Gráficos para análise e TCC

Os gráficos são gerados como pós-processamento dos artefatos da reprodução.
Eles não alteram o backtest nem os dados congelados.

A primeira família de gráficos reproduz dois heatmaps usados no MCT:

1. **P/L realizado mensal**: soma o P/L realizado das saídas executadas em cada
   mês e mostra também total anual, total por mês e total geral.
2. **Retorno mensal**: usa o último capital observado em cada mês e calcula
   `capital no fim do mês / capital no fim do mês anterior - 1`. O primeiro
   mês é omitido porque não existe um mês anterior para comparação.

O heatmap de retorno mensal inclui uma coluna **Soma anual** e uma linha
**Soma por mês**. Esses totais são somas aritméticas dos percentuais exibidos,
assim como solicitado para a leitura visual do heatmap; eles não representam
retorno composto anual ou acumulado.

Para gerar o heatmap de P/L realizado do Control:

```bash
python gerar_graficos.py --chart realized-pnl --variant control
```

Para gerar o heatmap de retorno mensal do Control:

```bash
python gerar_graficos.py --chart monthly-return --variant control --mode simulation
```

A visão do benchmark comprar-e-manter usa:

```bash
python gerar_graficos.py --chart monthly-return --variant control --mode reference
```

E o excesso mensal Control menos benchmark:

```bash
python gerar_graficos.py --chart monthly-return --variant control --mode excess
```

Para gerar todos os heatmaps das duas variantes:

```bash
python gerar_graficos.py --chart all --variant all --mode all
```

Os arquivos são gravados em:

```text
output/reproducao_v1/graficos/
├── monthly_realized_pnl_control.csv
├── monthly_realized_pnl_heatmap_control.png
├── monthly_realized_pnl_heatmap_control.svg
├── monthly_returns_control.csv
├── monthly_return_heatmap_control_simulation.png
├── monthly_return_heatmap_control_simulation.svg
└── ...
```

Os CSVs mensais são mantidos junto das figuras para permitir auditoria dos
valores sem depender da imagem.

## Testes e análise estática

Antes de executar uma pesquisa completa, rode:

```bash
python -m ruff check engine reproducao visualizacao reproduzir_experimento_spyder.py gerar_graficos.py tests --select F401,F811,F821,F841
python -m pytest -q
```

## Resultados gerados

A execução cria:

```text
output/reproducao_v1/
├── summary.json
├── summary.txt
├── comparison.csv
├── control_predictions.csv
├── control_trades.csv
├── soft_horizon_consensus_predictions.csv
├── soft_horizon_consensus_trades.csv
├── folds.csv
├── data_diagnostics.csv
├── data_audit.json
└── structural_exclusions.csv
```

`summary.json` concentra as métricas principais. `comparison.csv` resume a
comparação Control versus Soft. Os arquivos de predictions e trades permitem
auditar as decisões individuais, enquanto `data_audit.json` e
`structural_exclusions.csv` documentam a integridade do snapshot e eventuais
exclusões estruturais.
