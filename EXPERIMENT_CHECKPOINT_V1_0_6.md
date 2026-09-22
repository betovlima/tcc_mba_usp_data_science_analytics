# Checkpoint do experimento — v1.0.6

Este documento congela o estado do experimento antes da refatoracao estrutural
do motor.

## Identidade

- Versao: 1.0.6
- Backend: CPU
- Modelo: LightGBM Utility
- Strategy mode: `COMPOUND_ROTATION_SWING_LIGHTGBM`
- Variantes: Control e Soft Horizon Consensus
- Banco de dados: nenhum
- Fonte de dados: Alpaca
- Feed: SIP
- Timeframe: 1Day
- Download: RAW
- Reproducao: CSV por ativo + Corporate Actions + manifesto SHA-256
- Execucao suportada: script sequencial e celulas `# %%` no Spyder

## Janela e universo

- Inicio: 2016-01-01
- Analysis end: 2026-09-17
- Ultima sessao executada: 2026-09-16
- Ativos solicitados: 56
- Ativos elegiveis: 55
- DOC excluido por mudanca estrutural de identidade DOC -> PEAK
- RAW rows elegiveis: 148060
- Splits aplicados: 17
- Sessoes de simulacao: 1546

Regra metodologica para problemas estruturais de identidade, continuidade,
historico ou fonte: excluir o ativo e registrar a exclusao; nao reconstruir,
fazer bridge ou ajuste manual.

## Walk-forward

- Fold 1: 2020-07-22 -> 2022-07-21, 504 sessoes
- Fold 2: 2022-07-22 -> 2024-07-24, 504 sessoes
- Fold 3: 2024-07-25 -> 2026-09-16, 538 sessoes

## LightGBM congelado

- n_estimators: 329
- learning_rate: 0.020731
- max_depth: 3
- num_leaves: 6
- min_child_samples: 18
- min_child_weight: 5.0
- subsample: 0.85
- colsample_bytree: 0.88067
- reg_alpha: 0.050837
- reg_lambda: 3.596305
- max_bin: 255
- n_jobs: -1
- random_state: 42

Targets multi-horizonte:

- horizontes: 5, 10, 20, 40, 60
- pesos: 0.10, 0.15, 0.20, 0.30, 0.25

## Control

Control usa LightGBM + politica-base de rotacao + switch margin, sem a camada
Soft Horizon Consensus.

Resultado de reproducao CPU validado:

- capital final: US$ 10,094,316.30
- CAGR: 207.83%
- Sharpe: 2.1021
- Max Drawdown: -31.22%
- pior fold: +275.05%

## Soft Horizon Consensus

Soft usa exatamente o mesmo LightGBM e a mesma politica-base do Control,
adicionando somente o modificador continuo multi-horizonte.

Configuracao:

- enabled: true
- penalty_strength: 1.0

Resultado de reproducao CPU validado:

- capital final: US$ 9,851,632.93
- CAGR: 206.62%
- Sharpe: 2.1013
- Max Drawdown: -31.22%
- pior fold: +275.05%

Soft alterou 5 das 1546 decisoes da politica-base, bloqueando cinco switches
marginais. Todas as intervencoes ocorreram no Fold 2.

## Estrutura de dados minima

```text
dados/reproducao_v1/
├── raw_bars/
│   └── <ATIVO>.csv
├── corporate_actions/
│   └── <ATIVO>.csv
└── manifest.json
```

Os dados normalizados nao sao persistidos; os splits sao normalizados em memoria
em cada reproducao.

## Estado de qualidade

- sem MongoDB
- sem Tiingo
- sem CARO
- sem XGBoost/XGB no runtime
- Ruff habilitado para F401, F811, F821 e F841
- pytest e Ruff aprovados no CI da v1.0.6

## Objetivo da proxima branch

A proxima etapa deve reduzir o motor ao estritamente necessario para reproduzir
Control e Soft, removendo modos historicos e infraestrutura generica herdada do
Market Cycle Trader.

A refatoracao deve manter como invariantes:

- mesmos dados e folds;
- mesmas 1546 sessoes;
- mesmo Control;
- mesmo Soft;
- mesmas 5 intervencoes do Soft;
- mesmos resultados numericos dentro da tolerancia definida pelos testes de
  regressao;
- backend CPU.

A pasta `tcc_engine/` deve ser eliminada ou renomeada para `engine/`. A
preferencia e `engine/` caso ainda exista um nucleo reutilizavel apos a
reducao.
