# Resultado da mitigação de overfitting — v1.25.7 (23/10/2026)

**Branch:** `research/v1.23.1-fresh-snapshot-20261022`. **Status:** pacote de execução recebido, ZIP íntegro, quatro variantes concluídas, métricas e resultados verificados contra a reprodução original congelada. **Nenhum código/estratégia operacional alterado** nesta auditoria de resultados.

## Proveniência e verificações

- Pacote recebido: `pacote_mitigacao_overfitting_v1257.zip`, SHA-256 `5abbccc485251046750adecfc2d712f2085ac6fe07fe356fbcbdc01737adcc42`.
- ZIP: 31 arquivos, `testzip() = None` (integridade de leitura ok).
- Contrato: `status=completed`, `version=1.25.7`, `study_type=retrospective_predefined_interventions`, `independent_prospective_holdout=false`.
- Identidade congelada: SHA-256 `e440f59da5e684f1de59cf447abfedd9aed3f7817b3d5d681058fe631276a575`, 75 ativos (65+10 efetivos; DOC/CLMT excluídos), três folds temporais, 225 ajustes por variante.
- **Controle integral reproduzido**: `controle_original/predictions.csv` com 1.560 linhas e 34 colunas é **idêntico célula a célula** a `predictions.csv` do ZIP anterior `pacote_reproducao_universo(20261023-222155).zip`; capital final **US$ 587.165.941,13**. Todos os 225 valores de `train_rmse`, `oos_rmse` e gap são iguais aos da auditoria v1.25.4 para esse controle.
- Cada variante tem 225 diagnósticos completos, curva de 1.560 sessões e benchmark `buy_hold_equity` idêntico ao controle. Sem troca de data, ativo, benchmark nem universo.
- **Os testes são internos e retrospectivos**, porque o universo e o período 2020–2026 já foram considerados anteriormente. Resultados não são validação prospectiva independente.

## Efeito quantitativo por intervenção

| Variante | Gap mediano treino/teste | RMSE OOS agregado | Mudança relativa RMSE vs controle | Alertas overfit /225 | Skill positivo /225 | Decisões diferentes /1560 | Capital histórico simulado | Mudança capital vs controle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Controle: 329 árvores | +161,32% | 0,284124 | 0% | 196 | 29 | 0 | US$ 587.165.941,13 | — |
| Bagging ativo (`subsample_freq=1`) | +164,35% | 0,285038 | **+0,32% (pior)** | 195 | 30 | 369 (23,65%) | US$ 4.933.303,14 | **−99,16%** |
| Mais observações/folha (`min_child_samples=50`) | +144,33% | 0,282337 | **−0,63% (melhor)** | 193 | 32 | 603 (38,65%) | US$ 2.317.448,35 | **−99,61%** |
| Menos árvores (`n_estimators=160`) | **+95,52%** | **0,272234** | **−4,18% (melhor)** | **180** | **45** | 438 (28,08%) | US$ 2.879.578,84 | **−99,51%** |

**Diagnóstico importante:** a variante 160 árvores melhorou o RMSE OOS individual em **206 de 225** ajustes (**69/75** no fold 1, **68/75** no fold 2, **69/75** no fold 3); piorou em 19. Ainda ficou **17,15% pior em RMSE** que a previsão ingênua constante, a **média do alvo de treino** (0,272234 versus 0,232375). O controle era 22,27% pior que essa mesma referência. A regularização atenua, **não elimina** a fragilidade de generalização.

Por subgrupo de 75 ativos: nos dez selecionados, o gap mediano passou de +202,15% (controle) para +104,59% (160 árvores); nos 65 de base, de +160,22% para +93,15%. Mesmo assim, os modelos 160 árvores dos dez selecionados só apresentaram skill OOS positivo em **7 dos 30** pares (eram 5 dos 30 no controle).

## Desempenho financeiro e riscos

| Variante | Sharpe | Drawdown máximo | Rotações | Pior fold (retorno simples) |
|---|---:|---:|---:|---:|
| Controle | 2,729 | −41,68% | 354 | +580,00% |
| Bagging | 1,766 | −43,08% | 342 | +202,05% |
| Folha mínima 50 | 1,603 | −47,29% | 374 | +189,89% |
| 160 árvores | 1,692 | −50,70% | 311 | +354,47% |

As quatro variantes encerram com capital acima do investimento de US$ 10 mil, mas **nenhuma supera o controle** neste vintage. O efeito de capital **não** é uma medida direta do overfitting: é uma consequência composta das decisões de ativos, margens recalibradas por fold e dinâmica de exposição. Exemplos de margens selecionadas por fold: controle **0 / 0,01 / 0**; bagging **0 / 0,01 / 0,005**; folha 50 **0 / 0 / 0,005**; 160 árvores **0,01 / 0 / 0,01**. Portanto, não atribuir diferenças de lucro exclusivamente ao número de árvores.

## Por fold: magnitude de generalização (gap mediano)

| Variante | Fold 1 | Fold 2 | Fold 3 |
|---|---:|---:|---:|
| Controle | 247,69% | 148,44% | 96,25% |
| Bagging | 251,84% | 150,21% | 99,75% |
| Folha 50 | 211,53% | 136,31% | 91,93% |
| 160 árvores | **139,62%** | **82,11%** | **57,27%** |

## Conclusão científica

1. **Bagging isolado não mitigou sobreajuste neste experimento**: RMSE OOS agregado e gap mediano aumentaram.
2. **Aumentar `min_child_samples` melhorou pouco a predição OOS** e reduziu o gap, mas alterou muitas escolhas e reduziu muito o resultado histórico.
3. **Reduzir para 160 árvores é uma mitigação real, porém parcial, de overfitting PREDITIVO**: RMSE OOS caiu, skill OOS melhorou e gap caiu na maioria dos pares. Contudo, **não resolveu o problema de modelo pior que a referência simples**; os resultados econômicos e risco pioraram substancialmente no histórico congelado.
4. A divergência entre ganhos de precisão e perda de capital reforça que **a perda RMSE por ativo não coincide necessariamente com a utilidade de ranqueamento/decisão da política**. Mudanças pequenas ou moderadas nos scores podem mudar centenas de escolhas, calibração e compounding; a causalidade fina de cada efeito **não** foi isolada.
5. **Não selecionar nenhum modelo para produção com base nestes resultados históricos**, nem declarar a variante original robusta apenas por maior capital. Os 587 mi pertencem a vintage e universo escolhidos retrospectivamente; o estudo anterior de drift de 73 barras de maio/2018 já mostrou grande sensibilidade econômica do sistema.

## Próxima pesquisa recomendada, sem repetir busca por capital

Pré-especificar uma avaliação de **métricas de decisão e ranking fora da amostra** para as quatro variantes **usando as previsões diárias já geradas quando elas contêm scores suficientes**; se for indispensável recuperar scores por ativo, fazê-lo em experimento separado preservando hashes e sem utilizar o teste para selecionar parâmetros. Comparar ranking cross-sectional, regret, top-N, benefício de rotação vs manter, custo e taxa de decisões instáveis; incluir atribuição de mudanças das margens, separando modelo da política de rotação. Depois usar avaliação temporal aninhada para decisões de modelagem e um período futuro intocado para validade prospectiva. Não aumentar capital em cima de OOS já inspecionado.

**Status:** auditoria de resultados concluída e documentada. Não há experimento prospectivo independente, nem alteração ou promoção de estratégia.
