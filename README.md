# TCC MBA USP — Data Science & Analytics

Projeto independente e reproduzivel para o experimento de rotacao de capital com
**LightGBM**, validacao temporal walk-forward e comparacao entre:

- **Control**: LightGBM + politica-base de rotacao;
- **Soft Horizon Consensus**: o mesmo Control + modificador continuo de margem
  baseado no consenso dos horizontes 5, 10, 20, 40 e 60 dias.

## Versao oficial desta refatoracao

```text
TCC reproduction: 1.0.0
Branch: reproduction/v1.0.0-cpu-control-soft-spyder
Backend oficial: CPU
Banco de dados: nenhum
Fonte do snapshot: Alpaca
Feed: SIP
Timeframe: 1Day
Download: RAW
```

O motor matematico foi vendorizado da linha MCT API 10.8.84 para este
repositorio. **Nenhum codigo do Market Cycle Trader e importado em runtime.**

---

## Objetivo

Este repositorio e a base de reproducao do experimento academico. A execucao
deve poder ser repetida por outra pessoa sem MongoDB, sem API do MCT e sem
estado persistido externo.

A linha oficial e:

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
regra de identidade estrutural
      |
      +--> ticker estruturalmente quebrado => EXCLUIR
      |
      v
normalizacao local de splits
      |
      v
features e targets multi-horizonte
      |
      v
folds walk-forward cronologicos
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

---

## Sem banco de dados

O caminho oficial nao usa MongoDB, SQL ou qualquer outro banco.

Os dados ficam em:

```text
dados/reproducao_v1/
├── raw_bars/
│   ├── NVDA.csv
│   ├── AAPL.csv
│   ├── ...
│   └── ECC.csv
├── corporate_actions/
│   ├── NVDA.csv
│   ├── AAPL.csv
│   ├── ...
│   └── ECC.csv
├── normalized_bars/
│   ├── NVDA.csv
│   ├── AAPL.csv
│   ├── ...
│   └── ECC.csv
└── manifest.json
```

O diretorio e local e esta no `.gitignore`.

O `manifest.json` guarda SHA-256 de cada arquivo. Depois de congelado, o
experimento pode ser repetido **offline**, sem nova chamada a Alpaca.

---

## Universo

A requisicao congelada possui 56 tickers:

```text
NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA, AMD, JPM, SPY,
AVGO, NFLX, CRM, ORCL, COST, LLY, XOM, CAT, WMT, V, HD,
ADC, ADEA, ADI, ADM, DDS, CNQ, VRTS, XSD, CXW, ENS, CORT,
CCK, CEF, GKOS, RACE, TX, UNF, BXMT, PXLW, KKR, SCSC,
LKFT, DNN, VNCE, UNFI, DOC, CLMT, APD, MGM, MAN, MYE,
YANG, MKSI, MCS, ECC
```

A regra metodologica para problemas estruturais e:

> nao reparar, reconstruir, fazer bridge ou ajuste manual de identidade.

No snapshot de referencia, `DOC` e excluido por mudanca estrutural
`DOC -> PEAK`. O universo elegivel fica com 55 ativos.

---

## Periodo

```text
Inicio:                 2016-01-01
Analysis end:           2026-09-17
Bar snapshot as-of end: 2026-09-17
```

Quando os dados coincidirem com a fotografia de referencia 10.8.74, a auditoria
esperada e:

```text
eligible assets = 55
eligible RAW rows = 148060
splits applied = 17
simulation sessions = 1546
last execution session = 2026-09-16
```

A auditoria e informativa; ela nao e usada para escolher parametros.

---

## LightGBM

Control e Soft usam a **mesma configuracao LightGBM**:

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

Os horizontes do target sao:

```text
5, 10, 20, 40, 60
```

com pesos:

```text
0.10, 0.15, 0.20, 0.30, 0.25
```

---

## Control vs Soft

### Control

```text
LightGBM
  -> scores/ranking
  -> politica-base
  -> switch margin calibrado
  -> decisao
```

No Control:

```python
soft_horizon_consensus = {"enabled": False}
```

### Soft Horizon Consensus

O Soft parte da mesma politica-base e adiciona uma camada que avalia o suporte
dos modelos independentes dos horizontes.

```python
soft_horizon_consensus = {
    "enabled": True,
    "penalty_strength": 1.0,
}
```

Ele nao escolhe um novo ativo fora da decisao-base. Ele pode aceitar a troca
proposta pelo Control ou bloquear uma troca marginal e manter o ativo atual.

Portanto:

```text
Control = grupo de referencia
Soft    = variante experimental
```

---

## Execucao no Spyder

Abra:

```text
reproduzir_experimento_spyder.py
```

O arquivo foi escrito com celulas `# %%`.

Execute de cima para baixo:

```text
0  Imports, caminhos e configuracao
1  Credenciais ou reutilizacao offline
2  Download/reuso OHLCV RAW
3  Download/reuso Corporate Actions
4  Manifesto e SHA-256
5  Exclusoes estruturais + normalizacao de splits
6  Configuracoes Control/Soft + folds
7  CONTROL
8  SOFT HORIZON CONSENSUS
9  Comparacao
10 Exportacao
```

As variaveis permanecem no Variable Explorer. Exemplos:

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

Isso permite estudar cada fase sem transformar a reproducao em uma unica
funcao opaca.

---

## Primeira execucao: criar o snapshot

Crie um `.env`:

```text
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
```

No arquivo do Spyder:

```python
FORCAR_DOWNLOAD = False
```

Se os CSVs ainda nao existirem, a Alpaca sera consultada.

O log de barras preserva o formato da pesquisa original:

```text
[alpaca-bars] 1/56 downloading NVDA...
[alpaca-bars] NVDA rows=2692 first=... last=...
```

Corporate actions:

```text
[corporate-actions] progress=40/56 chunk=... pages=...
```

Auditoria:

```text
[data-audit] eligible_rows=... reference_rows=148060
             row_mismatches=... ca_mismatches=... split_mismatches=...
```

Treinamento:

```text
[final] CONTROL progress=...
[technical] CONTROL model=lightgbm ...
[final] completed CONTROL capital=...

[final] SOFT_HORIZON_CONSENSUS progress=...
[technical] SOFT_HORIZON_CONSENSUS model=lightgbm ...
[final] completed SOFT_HORIZON_CONSENSUS capital=...
```

---

## Reproducao offline

Depois que `dados/reproducao_v1/manifest.json` existir:

```python
FORCAR_DOWNLOAD = False
```

A etapa 1 entra automaticamente em:

```text
[snapshot] modo=offline-reuse
```

Todos os hashes sao validados antes do modelo. Nao e necessario ter acesso a
Alpaca para repetir o experimento com o snapshot congelado.

---

## Execucao completa fora do Spyder

O mesmo arquivo pode ser executado sequencialmente:

```bash
python reproduzir_experimento_spyder.py
```

Por compatibilidade:

```bash
python backtest.py
```

executa o mesmo roteiro.

---

## Artefatos

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

---

## Referencias de resultados

A ultima execucao CPU usada para migrar esta linha apresentou aproximadamente:

```text
Control = US$ 10.094.316
Soft    = US$  9.851.633
```

Esse numero e **referencia de reproducao**, nao objetivo de tuning.

A antiga execucao API 10.8.74 registrou:

```text
Control = US$ 5.551.143,96
Soft    = US$ 7.376.955,56
```

As diferencas entre execucoes historicas devem ser tratadas como objeto de
auditoria e nao como criterio para alterar parametros ate recuperar um capital
especifico.

---

## Testes

```bash
pytest -q
```

Os testes verificam, entre outros pontos:

- ausencia de banco no caminho oficial;
- universo e datas congelados;
- backend CPU;
- hiperparametros LightGBM;
- Control e Soft usando o mesmo LightGBM;
- diferenca Control/Soft apenas na camada Soft;
- celulas Spyder presentes;
- motor contendo a politica Soft Horizon Consensus.

---

## Scripts historicos

Arquivos antigos de Tiingo, Mongo, CARO, tuning e diagnosticos permanecem no
repositorio apenas para rastreabilidade historica. Eles **nao fazem parte da
reproducao oficial v1.0.0**.

O caminho oficial e somente:

```text
reproduzir_experimento_spyder.py
  -> reproducao/
  -> tcc_engine/
  -> dados/reproducao_v1/
  -> output/reproducao_v1/
```

---

## Linhagem tecnica

```text
Projeto-fonte do motor:
betovlima/market_cycle_trader_api

API de referencia:
10.8.84

Branch-fonte:
research/api-v10.8.84-soft-horizon-7m-cpu-isolation

TCC reproduction:
1.0.0
```

O codigo necessario foi copiado para este repositorio. A reproducao nao depende
do repositorio MCT depois do clone.
