# Resultado do teste v1.24.2 — splits futuros e atributos históricos do TCC

**Data:** 23/10/2026. **Branch:** `research/v1.23.1-fresh-snapshot-20261022`. **Código da campanha:** `auditar_impacto_splits_futuros_spyder.py`, commit `cea46639bb1c46c408f447064e1047ecbefe74b3`. **Testes:** `tests/test_auditar_impacto_splits_futuros.py`, commit `98c9d6dbf77af48e978a98f0e8d93f524f96eaab`. **HEAD anterior à publicação deste relatório:** `cbd0ecc901fa7c8a936c4570a0029721f3334f96`. **`main` preservada:** `4667d2ccee3f3441ee4d2b362ce734311cedd5e3`.

**Artefato recebido:** `pacote_impacto_splits_futuros.zip`; **SHA-256 `530eb722e87d9f0d79a84bd9ebb3c96608ee97d7479e925ae991c255a5f9dc8c`**. O pacote contém `resumo.json`, `impacto_por_ativo.csv` (75 linhas, 19 campos) e `impacto_por_atributo.csv` (**zero linhas**, quatro colunas). **Snapshot U75:** `595b7a03dc2c8978487975cd3554b1ea65608741893f13208938dc624cb273cc`. **Corte contrafactual:** 28/07/2020.

## 1. Pergunta e desenho experimental

Comparar a **normalização legada calculada somente na memória** com (A) todos os splits presentes no snapshot coletado em 2026 e (B) apenas os splits cuja `ex_date` já havia ocorrido até 28/07/2020. Nos dois caminhos, preservar integralmente o arquivo RAW de origem, o mesmo universo de 75 ativos, e verificar especificamente os **52 atributos derivados da estratégia**, **não os preços de custódia** ou os scores de um modelo ajustado. **Data efetiva após o corte não significa que o anúncio era desconhecido**; falta `first_seen_at_utc` no snapshot retrospectivo.

## 2. Resultado medido

| Indicador | Resultado |
|---|---:|
| Ativos verificados | **75** |
| Splits efetivos posteriores ao corte | **14**, em 12 ativos |
| Ativos com OHLCV reexpresso pelo caminho legado que inclui splits futuros | **12** |
| Barras históricas com diferenças de OHLCV entre caminhos (somadas por ativo) | **13.800** |
| Barras com diferenças de preço | **13.800** |
| Barras com diferenças de volume | **13.800** |
| Linhas de atributos comuns entre os dois caminhos | **11.412** (951 por ativo afetado) |
| Campos de entrada do modelo examinados | **52** |
| Células de atributos comparadas | **593.424** |
| Linhas de atributos com qualquer mudança material | **0** |
| Células de atributos com mudança material | **0** |
| Linhas de atributos presentes em apenas um caminho | **0** |
| Arquivos RAW originais modificados | **Nenhum** |
| Modelos ajustados / decisões / capital recalculados | **Não** |

Tolerância de comparação numérica: `rtol=1e-10`, `atol=1e-12`. O resultado **não afirma equivalência bit a bit dos atributos**, mas **ausência de diferenças materiais conforme o critério implementado**. A contagem de 11.412 linhas refere-se aos **12 ativos impactados**, pois o script não reconstruiu atributos para os outros 63 que não possuíam splits futuros.

## 3. Ativos com eventos futuros

| Símbolo | Splits após 28/07/2020 | Ex-dates efetivas futuras | Barras antigas reexpressas | Células de atributos modificadas materialmente |
|---|---:|---|---:|---:|
| NVDA | 2 | 20/07/2021; 10/06/2024 | 1.150 | 0 |
| AAPL | 1 | 31/08/2020 | 1.150 | 0 |
| AMZN | 1 | 06/06/2022 | 1.150 | 0 |
| GOOGL | 1 | 18/07/2022 | 1.150 | 0 |
| TSLA | 2 | 31/08/2020; 25/08/2022 | 1.150 | 0 |
| AVGO | 1 | 15/07/2024 | 1.150 | 0 |
| NFLX | 1 | 17/11/2025 | 1.150 | 0 |
| WMT | 1 | 26/02/2024 | 1.150 | 0 |
| CNQ | 1 | 11/06/2024 | 1.150 | 0 |
| PXLW | 1 | 09/06/2025 | 1.150 | 0 |
| YANG | 1 | 04/11/2024 | 1.150 | 0 |
| MUX | 1 | 28/07/2022 | 1.150 | 0 |

**Por que zero atributos mudaram:** antes da data efetiva de qualquer split futuro, a rotina retrospectiva multiplica toda a série anterior de um ativo por **fatores constantes de preço** e por seus inversos de volume. Os 52 atributos do Control examinados (retornos, volatilidades percentuais, distância relativa à EMA, RSI, ATR percentual, indicadores de canal e razões de volume etc.) são invariantes ou numericamente quase invariantes a esta mudança de unidade dentro da janela inteiramente anterior ao evento. Esse comportamento **não autoriza** mudar o RAW: os 13.800 registros de OHLCV reexpressos são transformações do processamento legado e **não mudanças nos CSVs fornecidos**.

## 4. Conclusão delimitada

**A hipótese de que os 14 splits efetivados depois de 28/07/2020 contaminaram os 52 atributos de entrada do LightGBM até aquela data não foi confirmada por este teste**; neste ponto de decisão, a transformação de escala não modificou materialmente esses atributos.

**Ainda não verificado:** efeito dos splits sobre atributos **durante e depois** da efetivação, rótulos de utilidade que cruzam a data efetiva, preços de execução, custos por ação, posição e custódia, treinamento/calibração completos e o ranking dos 75 ativos em cada sessão. A ausência de data comprovada de disponibilidade do anúncio persiste. Portanto **não afirmar que o sistema inteiro estava livre de vazamento ou contabilização incorreta**.

**Resultado histórico independente:** a discrepância financeira entre US$ 587.165.941,13 e US$ 89.941.496,49 em 06/10/2026 já havia sido rastreada até 73 barras **RAW** diferentes de maio/2018 em 31 ativos. Esse achado não é explicado nem negado pelos 14 splits e não deve ser confundido com a ausência de diferenças em atributos de 28/07/2020.

## 5. Próximo experimento

**Campanha v1.24.3:** usar os splits efetivamente registrados no snapshot dos mesmos 75 ativos, sem alterar RAW, e comparar o motor de atributos e os **rótulos de treinamento** ao redor das datas efetivas. A janela de até 60 sessões anteriores deve captar alvos que abrangem o evento; a janela de 60 sessões seguintes deve captar retornos/indicadores distorcidos quando o RAW não é tratado economicamente. **Comparar apenas em memória**, registrando diferença percentual observada no preço entre dois pregões e mudança de utilidade/atributos. Não treinar/negociar/otimizar até integrar eventos à contabilização de posições e alvos temporais.

A publicação desta auditoria não modifica `main`, universo efetivo, snapshot ou o resultado congelado da release v1.23.0.
