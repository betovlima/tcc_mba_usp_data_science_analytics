# Resultado de maturidade temporal dos rótulos no universo fixo U75 — v1.24.5

**Data:** 23/10/2026. **Branch:** `research/v1.23.1-fresh-snapshot-20261022`. **Código executado:** `auditar_maturidade_labels_spyder.py`, commit `65c3d63d09d338645504533b6fa6455fab56c407`. **Testes da rotina:** `tests/test_auditar_maturidade_labels.py`, commits `8ca1ad655b9144a79e2ad7ed905fc9a164c555c4`, `da817a6a3696157337b55504dcdb9d10f61a54de`. **HEAD de pesquisa anterior à auditoria:** `fa0ee25f0cff1de542de77d63e5be40655326df9`. **`main` preservada:** `4667d2ccee3f3441ee4d2b362ce734311cedd5e3`. PR #18 Draft.

**Arquivo recebido:** `pacote_maturidade_labels_u75.zip`, **SHA-256 `48750a807bfc895e3d3ce8b0acd92d098569c8a4e09dc11c32d988a8ca2f185a`**; contém `resumo.json` e `maturidade_por_ativo_fold.csv` (450 linhas e 10 colunas). **Fotografia dos 75 ativos** SHA-256 `595b7a03dc2c8978487975cd3554b1ea65608741893f13208938dc624cb273cc`.

## 1. Pergunta experimental e método

O treinamento utiliza rótulos de utilidade futuros `t+5, t+10, t+20, t+40, t+60`. Isso é admissível em aprendizado supervisionado **somente depois** do pregão futuro necessário à construção do rótulo. A auditoria reconstruiu o protocolo do motor Control `_construir_contexto_execucao` com os **75 CSVs RAW estritos**, sem reescrever preços, baixar dados, ajustar LightGBM, calcular scores, negociações ou capital. Para **três folds × duas fases de ajuste × 75 ativos**, examinou quais linhas são realmente elegíveis ao ajuste (`forward_risk_adjusted_utility` e 52 atributos presentes) e calculou a maturidade `t+60` no calendário de sessões do próprio ativo, não simplesmente em dias corridos.

A primeira avaliação subsequente é a calibração, para os fits de calibração, ou o período fora da amostra, para os fits finais. Nenhuma linha cujo label amadurece na data da primeira sessão avaliada ou depois dela é admitida pelo critério testado.

## 2. Resultado observado

| Indicador | Resultado |
|---|---:|
| Ativos auditados | **75/75** |
| Folds cronológicos | **3** |
| Fases auditadas por fold | **2** (`calibration_model_fit`, `final_model_fit`) |
| Grupos ativo × fold × fase | **450/450** |
| Total das linhas de treinamento contabilizadas nos grupos | **583.650** |
| Horizontes de labels de utilidade | **5, 10, 20, 40, 60 sessões** |
| Máximo horizonte auditado | **60 sessões** |
| Purge configurado | **60 sessões** |
| Labels amadurecendo tardiamente | **0** |
| Grupos com violação | **0** |
| Arquivos RAW alterados | **0** |
| Treinamentos/replays financeiros executados | **0** |

**A soma de 583.650 inclui linhas repetidas nos conjuntos de treinamento progressivos dos folds** e nas duas fases; não representa 583.650 amostras estatisticamente independentes. Como os 75 ativos tiveram a mesma quantidade de linhas elegíveis em cada grupo, as datas de maturidade/avaliação foram iguais entre ativos, mas as checagens foram registradas individualmente.

### Datas limites por fold

| Fold | Fase | Linhas elegíveis por ativo | Último dia de features do treino | Data em que amadurece seu alvo t+60 | Primeira sessão avaliada |
|---|---|---:|---|---|---|
| 1 | Calibração | 700 | 30/07/2019 | 23/10/2019 | **24/10/2019** |
| 1 | Final | 886 | 24/04/2020 | 21/07/2020 | **22/07/2020** |
| 2 | Calibração | 1.204 | 29/07/2021 | 22/10/2021 | **25/10/2021** |
| 2 | Final | 1.390 | 25/04/2022 | 21/07/2022 | **22/07/2022** |
| 3 | Calibração | 1.708 | 01/08/2023 | 25/10/2023 | **26/10/2023** |
| 3 | Final | 1.894 | 26/04/2024 | 24/07/2024 | **25/07/2024** |

As datas da coluna de maturidade são os pregões de que o rótulo necessita, e a informação só estaria completa depois do respectivo fechamento. A avaliação começa no próximo pregão, inclusive no fold 2, cuja distância é de três dias corridos por causa de um fim de semana.

## 3. Interpretação, sem extrapolações

**Demonstrado:** nas 450 combinações reais auditadas, segundo o mesmo calendário e filtro de elegibilidade usado pelo ajustador, nenhum label `t+60` teria informações posteriores ao começo de sua avaliação. Por consequência, os horizontes menores não ultrapassam a mesma fronteira, sob a hipótese de calendários ordenados e iguais àqueles considerados na montagem dos alvos. O protocolo de **purge de 60 sessões é suficiente nos cortes medidos**.

**Não demonstrado:** que todos os resultados de capital estejam livres de lookahead, que preços RAW ingênuos produzam labels de utilidade economicamente corretos nas janelas de split, que todas as fontes/atributos estejam point-in-time ou que o modelo tenha realizado lucro prospectivamente. **A rotina de auditoria usa o motor de construção de features sobre RAW sem reexpressão de preços**, enquanto o checkpoint v1.23.0 usava transformação local de splits. As datas de avaliação e o filtro real de treinamento foram inspecionados, mas o teste **não ajustou modelos LightGBM nem reproduziu o capital oficial**. A correspondência exata de linhas com o treino histórico é uma inferência do código, não a execução observada de novo fit em RAW.

O achado de US$ 587.165.941,13 versus US$ 89.941.496,49 é independente: causado, em experimento anterior, por 73 revisões nas próprias barras RAW de maio/2018. Este teste não modifica essa conclusão.

## 4. Próxima bateria v1.24.6 — utilidade econômica completa sem alterar RAW

O ensaio anterior v1.24.4 demonstrou que retornos simples obtidos pelo fator de quantidade foram iguais à normalização antiga em 11.980/11.980 janelas. Isso não cobria o alvo final do LightGBM, que inclui penalidades de **downside e drawdown**, movimento capturado, persistência e a agregação ponderada dos cinco horizontes.

Foi implementado o teste `auditar_utilidades_economicas_raw_spyder.py` (commit `8313aee49a6f17acc44afb2780b9b1ba16db3577`) e o teste automatizado `tests/test_auditar_utilidades_economicas_raw.py` (commit `f2a710cc6e7256397141a17553d965bcf8f94cc8`). O programa utiliza os 75 ativos e o snapshot anterior, computa em memória os rótulos econômicos a partir das barras RAW e multiplicadores de quantidade por splits entre a entrada e o horizonte de saída, e compara **15 valores de rótulos** (`5 utilidades + 5 retornos líquidos + 5 agregados/diagnósticos`) contra o legado reexpresso **apenas em memória**. Exames nas janelas próximas aos splits, sem treinamento, otimização, modificação dos CSVs ou resultado financeiro.

**Resultado real da v1.24.6 ainda não disponível**. O próximo pacote a produzir pelo Spyder (F5) é `output/utilidades_economicas_raw/<ID>/pacote_utilidades_economicas_raw.zip`, com `resumo.json`, `por_ativo.csv` e `divergencias_labels.csv`. Zero diferenças seria um achado de equivalência das fórmulas examinadas, **não** autorização automática para executar backtest RAW sem regras de custódia, custos e disponibilidade histórica dos anúncios.

**Regras de continuidade:** manter um único universo de 75 ativos, RAW sem modificação, a mesma branch, PR #18 Draft, sem merge ou revisão do resultado oficial.
