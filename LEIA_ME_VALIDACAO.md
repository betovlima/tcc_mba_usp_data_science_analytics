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
