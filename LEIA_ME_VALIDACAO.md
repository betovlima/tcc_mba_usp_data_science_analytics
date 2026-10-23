# Diagnóstico preditivo OOS (entrega isolada)

Esta entrega não modifica `main`, a lógica de trades, calibrações, parâmetros nem snapshots.

## Contrato mínimo

`previsoes_oos_por_ativo.csv` precisa conter: `data_decisao`, `ativo`, `fold`,
`previsao`, `observado`, `label_maturado`, `split_validado`.
O alvo `observado` é **forward_risk_adjusted_utility**, não retorno financeiro
realizado ao encerrar uma posição. `split_validado` só pode ser verdadeiro após
a auditoria de eventos corporativos na construção de cada alvo (não é presumido
para a série RAW).

Usar no Spyder após obter as linhas OOS e validar as datas e os eventos:

```python
import pandas as pd
from pathlib import Path
from reproducao.validacao_preditiva import gerar_relatorios
entrada = pd.read_csv('output/previsoes_oos_por_ativo.csv')
resumo = gerar_relatorios(entrada, Path('output/validacao_preditiva'))
print(resumo)
```

O helper `exportar_previsoes_oos` pode ser chamado a partir de `states`,
`baseline_frames`, `baseline_symbols`, `common_dates` e caches dos candidatos
na execução histórica. Ele **deliberadamente marca split_validado=False** até
que o responsável pelo pipeline ateste a correção dos alvos. Não conectar
automaticamente ao executor 1.25.1 antes de publicar seu código-fonte: a
versão disponível no GitHub é 1.23.0 e tem caminho de treinamento distinto.

**Limitações:** MAE/RMSE são da utilidade, não do lucro; NDCG usa deslocamento
por sessão para trabalhar com relevância não negativa; universo/candidatos
selecionados full-window continuam contaminados por seleção retrospectiva.
Nenhum resultado OOS é emitido quando faltam evidências.

## Auditoria de vazamento nos folds

`auditar_limites_temporais(states=..., common_dates=..., horizon_maximo=...)`
identifica quantos rótulos da janela de treino ainda dependem de sessões
posteriores ao início da calibração ou teste. Um alerta exige confirmar o
uso efetivo desses rótulos e aplicar separação temporal apropriada no
experimento futuro. Essa auditoria não altera a reprodução histórica.

## Execucao integrada na branch desta entrega

**Executor realmente disponivel no remoto:** `reproduzir_experimento.py`
(v1.23.0, snapshot congelado da campanha historica). O codigo fonte
das execucoes v1.25.0/v1.25.1 dos ZIPs recentes nao esta publicado no
repositório consultado. NAO interpretar o replay v1.23 como reproduzindo
a campanha economica raw v1.25.1 de US$ 42,6 milhoes.

No console Git (sem mover ou alterar a main):

```bash
git fetch origin
git switch research/v1.25.2-diagnosticos-preditivos
git pull --ff-only origin research/v1.25.2-diagnosticos-preditivos
```

No Spyder, com os **mesmos dados locais congelados da v1.23** e seus
hashes conferidos, execute o arquivo `reproduzir_experimento.py` com F5.
O executor usa os mesmos modelos, margens e folds, testa a reproducao ao
centavo e, depois, gera:

```text
output/reproducao_universo/
    previsoes_oos_por_ativo.csv
    validacao_preditiva/
        diagnostico_preditivo.json
        metricas_por_fold.csv
        metricas_por_ativo_fold.csv
        ranking_por_sessao.csv
        auditoria_splits.csv
        auditoria_limites_temporais.csv
        erro_por_fold.png
        spearman_temporal.png
    pacote_reproducao_universo.zip
```

Os PNGs sao gerados apenas quando ha pares OOS validos; um erro de
auditoria fica no campo `predictive_audit` do
`reproducao_universo.json`, sem alterar o resultado economico.

**Cuidado com a interpretacao:** a auditoria de splits confere se os
OHLCV da modelagem correspondem a normalizacao deterministica das
Corporate Actions locais. Nao confirma, por verificacao externa,
a completude das Corporate Actions. Os intervalos dos folds ainda
exigem conferir os avisos de rótulos incompletos na auditoria temporal.
O conjunto de 75 ativos da v1.23 foi escolhido retrospectivamente,
entao essas metricas OOS nao eliminam por si o viés de selecao.
