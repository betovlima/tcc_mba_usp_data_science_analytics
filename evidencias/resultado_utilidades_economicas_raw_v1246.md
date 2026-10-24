# Auditoria dos rótulos econômicos completos — execução v1.24.6 (23/10/2026)

**Branch:** `research/v1.23.1-fresh-snapshot-20261022`. **HEAD do código executado:** `2b3cf0f9c604ee8e8de9fed4c6dade8a3f249442`. **Script:** `auditar_utilidades_economicas_raw_spyder.py`, commit `8313aee49a6f17acc44afb2780b9b1ba16db3577`; **testes:** `tests/test_auditar_utilidades_economicas_raw.py`, commit `f2a710cc6e7256397141a17553d965bcf8f94cc8`. **Main imutável:** `4667d2ccee3f3441ee4d2b362ce734311cedd5e3`, PR #18 Draft.

## Fonte e integridade

ZIP original do pesquisador: `pacote_utilidades_economicas_raw.zip`, SHA-256 **`63be9712d1e3f846ec2f9ca8060369eb7caddcf33cf3f5c49dbd2308e19cee03`**. ZIP sem erros (`ZipFile.testzip() = None`); contém `resumo.json`, `por_ativo.csv` (75×7) e `divergencias_labels.csv` (6×8). Snapshot U75 SHA-256 **`595b7a03dc2c8978487975cd3554b1ea65608741893f13208938dc624cb273cc`**. **Fonte OHLCV RAW intocada:** `raw_modified=false`; modelo não treinado, capital não calculado.

## Indicadores reais

| Indicador | Valor |
|---|---:|
| Ativos fixos auditados | 75 |
| Ativos com splits | 17 |
| Splits registrados | 21 |
| Decisões hipotéticas com 15 rótulos completos comparados | **2.210** |
| Células numéricas comparadas | **33.150** |
| Diferenças materiais, tolerância `rtol=atol=1e-9` | **6** |
| Equivalência dentro da tolerância | **33.144/33.150 = 99,98190%** |
| Valor máximo de diferença absoluta | **0,00833333333333336** |
| Linhas de decisão com diferença | **3** |
| Ativos com diferenças | **2: YANG e CAPR** |
| Divergências nos retornos líquidos individuais (5 horizontes) | 0 |
| Divergências nas utilidades individuais por horizonte (5 horizontes) | 0 |
| Divergências em `forward_trend_persistence` | 3 |
| Divergências em `forward_risk_adjusted_utility` | 3 |

### Valores afetados conforme CSV

| Ativo | Decisão | Maturidade do alvo máximo | Componente | Econômico (RAW + quantidade) | Normalização legada em memória | Diferença |
|---|---|---|---|---:|---:|---:|
| YANG | 16/10/2024 | 14/01/2025 | `forward_trend_persistence` | 0,404520 | 0,396186 | **+0,008333333** |
| YANG | 16/10/2024 | 14/01/2025 | `forward_risk_adjusted_utility` | −0,098952 | −0,100618 | **+0,001666667** |
| CAPR | 20/05/2019 | 14/08/2019 | `forward_trend_persistence` | 0,561441 | 0,553107 | **+0,008333333** |
| CAPR | 20/05/2019 | 14/08/2019 | `forward_risk_adjusted_utility` | 0,458503 | 0,456836 | **+0,001666667** |
| CAPR | 28/05/2019 | 21/08/2019 | `forward_trend_persistence` | 0,578249 | 0,569915 | **+0,008333333** |
| CAPR | 28/05/2019 | 21/08/2019 | `forward_risk_adjusted_utility` | 0,493179 | 0,491513 | **+0,001666667** |

## Interpretação delimitada

A origem das seis diferenças está **exclusivamente na persistência de tendência e no termo final que a incorpora**. No motor, `forward_trend_persistence = .5*mean(future_closes>entry_open)+.5*mean(diff(future_closes)>0)`. Para 60 fechamentos, **um resultado diferente na comparação `future_close > entry_open` contribui exatamente `0,5/60=0,008333333…`**; essa assinatura coincide com os três desvios observados e é compatível com comparações numericamente limítrofes, mas **não prova** quais barras ou quais operações de ponto flutuante os causaram. O termo final recebe **0,001666667**, consistente com peso de persistência de 0,2. É necessário auditar cada comparação de preços com precisão hexadecimal, ULP e diferença antes/depois dos fatores.

**Não reclassificar automaticamente igualdade como tendência:** uma tolerância artificial pode mudar regras/labels da pesquisa em milhares de decisões e invalidar a paridade histórica. Não alterar o RAW nem ajustar a regra de negócio para suprimir seis diferenças sem identificar a causa.

**Conclusão atual:** equivalência numérica **quase completa**, mas não completa, dos 15 alvos sob o critério implementado. O resultado da v1.24.4 de 11.980/11.980 retornos simples de preço concordantes permanece verdadeiro. Esta análise não demonstra que os 75 modelos terão os mesmos scores ou capital, nem endossa o snapshot que maximiza retorno.

## Próximo ensaio isolado — v1.24.7

Auditar, para **YANG 16/10/2024, CAPR 20/05/2019 e CAPR 28/05/2019**, exatamente as 60 comparações de fechamentos de cada rótulo com o preço de entrada, nas duas escalas, e identificar **dia, valores float64, hexadecimal, número de ULPs e comparação booleana** para cada possível empate ou inversão. Verificar também os 59 passos `diff(close)>0`. Persistindo diferença que não seja exclusivamente precisão numérica, investigar lógica de quantidade antes de modificar o motor ou o experimento oficial. O ensaio deve ler o mesmo snapshot, com `split_normalize` **somente em memória**, sem modificar nenhum OHLCV original, sem treino/otimização/backtest.
