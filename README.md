# TCC MBA USP — Data Science & Analytics

Reproducao independente do experimento de rotacao de capital com LightGBM,
validacao temporal walk-forward e comparacao entre Control e Soft Horizon
Consensus.

## Versao

```text
TCC reproduction: 1.1.0-dev.1
Branch de refatoracao: refactor/v1.1.0-minimal-engine
Backend: CPU
Banco de dados: nenhum
Fonte: Alpaca
Feed: SIP
Timeframe: 1Day
Download: RAW
```

A versao 1.0.6 reduz o snapshot local ao conjunto minimo necessario para
reproducao. A pasta derivada `normalized_bars/` foi removida porque nunca era
lida: a normalizacao de splits e refeita em memoria a partir de `raw_bars/` e
`corporate_actions/`. A logica matematica, os hiperparametros do LightGBM,
os folds, Control e Soft permanecem inalterados.

## Pipeline oficial

```text
Alpaca RAW/SIP
      |
      v
CSV OHLCV por ativo
      |
      +--> CSV Corporate Actions por ativo
      |
      v
manifest.json + SHA-256
      |
      v
validacao estrutural de identidade
      |
      +--> identidade quebrada => excluir ativo
      |
      v
normalizacao local de splits
      |
      v
features + targets 5/10/20/40/60 dias
      |
      v
folds walk-forward
      |
      v
LightGBM CPU
      |
      +--------------------+
      |                    |
      v                    v
   CONTROL        SOFT HORIZON CONSENSUS
      |                    |
      +---------+----------+
                v
          comparacao final
```

## Estrutura do repositorio

```text
.
├── .github/
│   └── workflows/
├── dados/
│   ├── README.md
│   └── reproducao_v1/
│       └── README.md
├── reproducao/
│   ├── artefatos.py
│   ├── dados.py
│   ├── experimento.py
│   ├── preparacao.py
│   └── reference_10_8_74_raw_snapshot_diagnostics.json
├── engine/
├── tests/
├── reproduzir_experimento_spyder.py
├── requirements.txt
├── .env.example
└── README.md
```

Nao existem scripts historicos de Tiingo, MongoDB, CARO, tuning antigo ou
backtests antigos no estado atual da `main`. O historico permanece acessivel
pelos commits do Git.

## Dados locais

Nenhum dado de mercado e versionado no Git. A primeira execucao cria:

```text
dados/reproducao_v1/
├── raw_bars/
│   └── <ATIVO>.csv
├── corporate_actions/
│   └── <ATIVO>.csv
└── manifest.json
```

Depois do snapshot ser congelado, as execucoes seguintes podem ser feitas
offline. O manifesto valida SHA-256 de cada arquivo antes do treinamento.

Os dados normalizados nao sao persistidos. Eles sao derivados em memoria a
cada execucao a partir dos dados RAW e dos eventos corporativos, evitando
duplicacao de arquivos no snapshot.

## Universo e janela

```text
Ativos solicitados:       56
Ativos elegiveis esperados: 55
Inicio:                    2016-01-01
Analysis end:              2026-09-17
Bar snapshot as-of end:    2026-09-17
```

A regra metodologica para um ativo com problema estrutural de historico,
identidade, continuidade ou fonte e: excluir o ativo e registrar a exclusao.
Nao reconstruir, fazer bridge ou ajuste manual.

No snapshot de referencia, DOC e excluido por mudanca estrutural DOC -> PEAK.

## Contrato de dados de referencia

Quando a Alpaca reproduz a fotografia de dados validada:

```text
eligible assets       = 55
eligible RAW rows     = 148060
splits applied        = 17
simulation sessions   = 1546
last execution session = 2026-09-16
```

Esses valores sao usados como auditoria de reproducao, nao como criterio de
tuning.

## Control e Soft

Os dois usam exatamente o mesmo LightGBM.

Control:

```python
soft_horizon_consensus = {"enabled": False}
```

Soft Horizon Consensus:

```python
soft_horizon_consensus = {
    "enabled": True,
    "penalty_strength": 1.0,
}
```

O Soft nao escolhe um ativo fora da decisao-base. Ele pode aceitar a troca
proposta pelo Control ou bloquear uma troca marginal e manter a posicao atual.

## LightGBM

```text
n_estimators      = 329
learning_rate     = 0.020731
max_depth         = 3
num_leaves        = 6
min_child_samples = 18
min_child_weight  = 5
subsample         = 0.85
colsample_bytree  = 0.88067
reg_alpha         = 0.050837
reg_lambda        = 3.596305
max_bin           = 255
n_jobs            = -1
random_state      = 42
backend           = CPU
```

Horizontes:

```text
5, 10, 20, 40, 60
```

Pesos:

```text
0.10, 0.15, 0.20, 0.30, 0.25
```

## Spyder

Abra `reproduzir_experimento_spyder.py` e execute as celulas `# %%` de cima
para baixo:

```text
0  Configuracao
1  Credenciais ou replay offline
2  OHLCV RAW
3  Corporate Actions
4  Manifesto e SHA-256
5  Preparacao estrutural e splits
6  Control/Soft e folds
7  CONTROL
8  SOFT HORIZON CONSENSUS
9  Comparacao
10 Exportacao
```

As principais variaveis permanecem disponiveis no Variable Explorer:

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

## Execucao

Instalacao:

```bash
python -m pip install -r requirements.txt
```

Configure `.env`:

```text
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
```

Execute:

```bash
python reproduzir_experimento_spyder.py
```

## Resultados

Os resultados sao gerados somente localmente:

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

O diretorio `output/` nao e versionado.

## Resultado de reproducao validado

A reproducao CPU validada antes da limpeza estrutural produziu:

```text
Control = US$ 10.094.316,30
Soft    = US$  9.851.632,93
```

Esse resultado e referencia de reproducao, nao alvo de tuning.

## Testes

```bash
python -m pytest -q
```

Os testes verificam ausencia de dependencias de banco, configuracao CPU,
parametros congelados do LightGBM, equivalencia da configuracao Control/Soft,
estrutura Spyder e presenca da politica Soft Horizon Consensus.


## Refatoracao 1.1.0

A branch `refactor/v1.1.0-minimal-engine` reduz o motor ao fluxo efetivamente
usado pelo TCC. O pacote `tcc_engine/` foi renomeado para `engine/`.

Estrutura alvo:

```text
engine/
├── __init__.py
├── config.py
├── diagnostics.py
├── execution.py
├── lightgbm.py
└── rotation.py
```

Foram removidos modos historicos de alocacao, cash gates, selective opportunity,
risk overlay, IQN, hard horizon voting e suporte GPU. O experimento oficial e
CPU-only e compara somente Control vs Soft Horizon Consensus.


### v1.1.0-dev.1

Corrige a refatoracao da semantica de quantidade. O checkpoint v1.0.6 usava
`whole_shares=False`, portanto a execucao sempre permitiu quantidades
fracionarias. Essa semantica agora e fixa no motor, sem um campo de configuracao
redundante. Foi adicionado um teste de contrato entre os atributos acessados
pelo runtime e `StandaloneBacktestConfig` para impedir novas remocoes
inconsistentes.
