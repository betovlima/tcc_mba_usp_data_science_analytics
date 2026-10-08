# TCC MBA USP — Rotação de Capital com Machine Learning

Implementação reproduzível do TCC de rotação de capital com LightGBM e
validação temporal walk-forward.

A árvore atual foi reduzida ao fluxo necessário para reproduzir e auditar o
U67 Control. Pesquisas experimentais encerradas continuam disponíveis no
histórico Git, mas não fazem mais parte do runtime oficial.

## Execução oficial

O único runner principal é:

```text
reproduzir_experimento.py
```

A reprodução atual:

1. carrega o snapshot U67 congelado em `dados/u67/`;
2. valida os 67 ativos, o cutoff e os hashes de integridade;
3. usa barras `1Day`, feed `SIP`, ajuste `RAW` e Corporate Actions congelados;
4. normaliza splits em memória;
5. remove somente identidades com problema estrutural documentado;
6. preserva o calendário científico derivado do U56 elegível;
7. executa LightGBM Control com calibração walk-forward e purge temporal;
8. simula as rotações com o mesmo contrato financeiro;
9. compara o resultado com o job MCT auditado;
10. exporta previsões, operações, diagnósticos, gráficos e pacote ZIP.

Não há dependência de MongoDB.

## Referência de paridade

O runner está travado para a referência operacional auditada:

```text
MCT job: 20261007T095423-60e489c0
analysis_end: 2026-10-06
requested assets: 67
effective assets: 65
expected exclusions: CLMT, DOC
MCT ending capital: US$ 76,927,051.38897176
```

O capital inicial permanece em US$ 10.000. O resultado é um backtest histórico
e não representa previsão ou garantia de desempenho futuro.

## Dados

A reprodução lê somente:

```text
dados/u67/
├── raw_bars/
├── corporate_actions/
└── manifest.json
```

A coleta foi separada em `preparar_snapshot_u67.py`. Esse script é usado
somente para criar ou substituir o snapshot congelado e é o único fluxo que
precisa de credenciais da Alpaca. O `reproduzir_experimento.py` não acessa a
Alpaca.

O `manifest.json` registra a identidade do snapshot e hashes de integridade
dos CSVs. Esses hashes são usados para validar os dados e não participam do
treinamento do modelo.

## Estrutura do código

```text
.
├── preparar_snapshot_u67.py
├── reproduzir_experimento.py
├── engine/
│   ├── configuracao.py
│   ├── diagnosticos.py
│   ├── execucao.py
│   ├── modelo_lightgbm.py
│   └── rotacao.py
├── reproducao/
│   ├── artefatos.py
│   ├── dados.py
│   ├── experimento.py
│   ├── graficos.py
│   └── preparacao.py
├── dados/
├── tests/
├── requirements.txt
├── CONTEXTO_MESTRE.md
└── .env.example
```

A pasta `pesquisas/`, o antigo `buscar_ativos.py`, o módulo de Directional
Change e auxiliares de migração legados foram removidos do runtime final.
O histórico dessas pesquisas continua preservado pelo Git.

## Execução

No terminal:

```bash
python -m pip install -r requirements.txt
python reproduzir_experimento.py
```

No Spyder, abra `reproduzir_experimento.py`, reinicie o kernel e execute o
arquivo completo.

Somente para criar ou substituir o snapshot com `preparar_snapshot_u67.py`,
configure localmente:

```text
ALPACA_API_KEY=sua_api_key
ALPACA_SECRET_KEY=sua_secret_key
```

Nunca adicione credenciais ao Git.

## Artefatos

A execução grava os resultados em:

```text
output/reproducao/
```

Entre os principais artefatos:

```text
u67_requested_assets.csv
u67_effective_assets.csv
u67_runtime_exclusions.csv
u67_fold_margins.csv
u67_fold_calibration_candidates.csv
u67_market_data_hashes.csv
u67_predictions.csv
u67_trades.csv
u67_runtime_environment.json
reproducao_u67_mct_parity.json
graficos/
pacote_reproducao_u67_mct_parity.zip
```

## Testes e análise estática

```bash
python -m ruff check engine reproducao preparar_snapshot_u67.py reproduzir_experimento.py tests --select F401,F811,F821,F841
python -m pytest -q
```

O GitHub Actions executa esses checks nas branches de desenvolvimento e
release.
