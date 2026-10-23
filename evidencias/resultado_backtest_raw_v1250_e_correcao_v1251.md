# Resultado observado do replay RAW econômico v1.25.0 e recuperação v1.25.1

**Data de execução:** 23/10/2026. **Branch única:** `research/v1.23.1-fresh-snapshot-20261022`. **Main intocada:** `4667d2ccee3f3441ee4d2b362ce734311cedd5e3`. **PR #18:** Draft, sem merge.

## 1. Evidência recebida e nível de confirmação

O pesquisador enviou o **log final e traceback**, não o ZIP em si. O log informa:

```text
[fixed75] capital=US$ 42,600,974.32
pacote=output/validacao_economica_raw_75/20261023T152638959386Z/pacote_validacao_economica_raw.zip
[sound] completion mechanisms=PlaySound:SystemExclamation,MessageBeep,Beep
KeyError: 'strategy_ending_capital'
validar_backtest_economico_raw_spyder.py:111
```

**Capital observado pelo executor:** **US$ 42.600.974,32** (antes do erro do wrapper de relatório). **Snapshot exigido pelo código:** SHA-256 `595b7a03dc2c8978487975cd3554b1ea65608741893f13208938dc624cb273cc`, 75 ativos fixos; cutoff 06/10/2026; capital inicial US$ 10.000.

O procedimento original `reproduzir_universo_fixo75()` concluiu a simulação, gravou gráficos, `reproducao_universo.json`, `predictions.csv`, `trades.csv`, ZIP e executou o som de conclusão. A exceção aconteceu **depois** de o wrapper verificar que os hashes dos 150 CSVs de entrada eram iguais antes e depois. Porém **o ZIP efetivo ainda não foi recebido nem auditado independentemente nesta conversa**. Não afirmar Sharpe, drawdown, número de operações, validade de folds, integridade interna das curvas ou ausência de viés antes de examinar os artefatos.

## 2. Comparação financeira de referência

| Experimento | Capital final | Interpretação |
|---|---:|---|
| Vintage histórico antigo, método anterior, cutoff 06/10 | US$ 587.165.941,13 | Reproduzível apenas no snapshot antigo |
| Vintage Alpaca renovado, método anterior, mesmo cutoff | US$ 89.941.496,49 | Reproduzido em campanha prévia |
| **Alpaca RAW econômico e custódia split-aware, mesmo cutoff** | **US$ 42.600.974,32** | **Observado no log da v1.25.0; pacote aguarda auditoria** |

A v1.25.0 é **52,634795% menor** que a variante atualizada do processamento antigo e **92,744645% menor** que o resultado vintage de US$ 587,17 milhões. Essas diferenças **não podem ser atribuídas exclusivamente aos splits**: o novo pipeline alterou em conjunto atributos/rótulos via visão econômica causal, a calibração interna baseada nessa visão, a contabilização da custódia, o benchmark e o uso do modelo com o mesmo snapshot. Para identificar contribuições isoladas seriam necessários comparativos pré-registrados sem reajustar parâmetros.

A redução histórica de US$ 587,17 milhões para US$ 89,94 milhões, por outro lado, foi rastreada em ensaio anterior à restauração contrafactual de **73 barras RAW diferentes em maio/2018, em 31 ativos**, sem estabelecer qual vintage é mais correto.

**Overfitting permanece sem resolução:** 196 de 225 avaliações de modelos anteriormente apresentaram alertas de possível sobreajuste; ranking OOS ~51,20% com intervalo exploratório incluindo 50%. Esse capital simulado é um resultado retrospectivo, não uma validação prospectiva independente de seleção. Sem impacto de mercado, slippage 0 bps e com frações de ações admitidas.

## 3. Defeito reproduzido em inspeção de código e reparo

A exceção foi causada por usar `result["metrics"]["strategy_ending_capital"]` no wrapper. Esta chave existe no **`RotationRunResult.metrics` interno de `engine/rotacao.py`**, mas o executor retorna `summarize_metrics(result, folds, initial)`, cujo contrato em `reproducao/experimento.py` publica **`result["metrics"]["ending_capital"]`**. O próprio `reproducao/universo_fixo75.py` lê essa chave para imprimir o valor correto.

**Correção v1.25.1:** `validar_backtest_economico_raw_spyder.py` commit `beb69edcf98a09ff60b08764e764191ee3184ee5`: função `capital_do_resultado` consome a chave sumarizada correta, exige sua presença, confere `checkpoint.observed_capital` com tolerância de US$ 0,01 e conserva snapshot/estratégia inalterados. Novo treino **não é necessário para recuperar o resultado anterior**.

**Recuperação sem recomputação:** `recuperar_validacao_economica_raw_spyder.py`, commit `3fc65e9ec3c801747a139ff2c55ae02ceb7c7f44`, fixa o `run_id=20261023T152638959386Z` e lê apenas o snapshot e os artefatos locais da campanha já finalizada. Checa status/schema/cutoff/75 ativos/hash do snapshot, métricas e checkpoint, compara capital final com última linha da curva `predictions.csv` e com caixa da `FINAL_SELL` de `trades.csv`, testa integridade do ZIP original e calcula SHA dos 150 arquivos de origem. **Não treina, não recalibra nem simula**. Preserva o ZIP original bit a bit e cria um novo `pacote_validacao_economica_raw_v1251_finalizado.zip` com `interpretacao_resultado_v1251.json`, além de registrar SHA do ZIP original.

**Teste de regressão:** `tests/test_recuperacao_resultado_v1251.py`, commit `e0a281f7bd4d9551c021764a481564ed03d20b94`. Atesta a chave correta, rejeita campos faltantes e inconsistências entre checkpoint/curva/trade, valida cópia sem mutação do ZIP e impede sobregravação de evidência. 

## 4. Próxima ação local

```powershell
git fetch origin
git switch research/v1.23.1-fresh-snapshot-20261022
git pull --ff-only origin research/v1.23.1-fresh-snapshot-20261022
```

No Spyder, reiniciar o kernel e pressionar **F5 em `recuperar_validacao_economica_raw_spyder.py`**, **não** no replay completo. Solicitar o arquivo:

`output/validacao_economica_raw_75/20261023T152638959386Z/pacote_validacao_economica_raw_v1251_finalizado.zip`.

Somente após auditar esse ZIP verificar métricas por fold, operações, contabilidade de carteira, benchmarking e desempenho preditivo; **não mudar dados, hiperparâmetros, universo, branch ou main** com base no capital observado.
