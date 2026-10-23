# Auditoria de generalização preditiva do vintage US$ 587.165.941,13 — v1.25.4

**Data da evidência:** 23/10/2026. **Branch:** `research/v1.23.1-fresh-snapshot-20261022`. **Status:** execução local concluída, evidência recebida e verificada por leitura dos dois arquivos enviados pelo pesquisador. **Sem alteração dos modelos ou da estratégia nesta análise.**

## Proveniência e integridade

- Pacote recebido: `pacote_generalizacao_587m_v1254.zip`, SHA-256 **`0b455cbdfc5dc4c83818d30ba8e868634d91e84e5ac5ab49b869c505bcc9a060`**.
- Conteúdo: `model_generalization_225.csv`, `resumo_generalizacao.json` e três gráficos de RMSE, mapa de calor e distribuição dos gaps.
- CSV complementar: `ultima_execucao_metricas_brutas.csv`, 225 linhas/18 colunas; coincide exatamente nos valores com as colunas correspondentes do CSV final. Sem métricas ausentes no pacote final. Campos de treinamento, OOS e os indicadores estão presentes para todos os 225 pares.
- Fonte declarada: `somente_modelos_reajustados`, isto é, os modelos foram reajustados a partir do snapshot histórico congelado, **sem novo replay financeiro**.
- Snapshot original confirmado no resumo da execução: `e440f59da5e684f1de59cf447abfedd9aed3f7817b3d5d681058fe631276a575`. O capital de US$ 587.165.941,13 é **checkpoint de referência histórica**, não medição financeira nova.
- Universo efetivo: 75 ativos, três folds, 225 ajustes LightGBM; 195 de ativos base e 30 dos dez selecionados.

## Definições matemáticas

Para cada ativo/fold, o script mede `RMSE_treino` nos exemplos usados no ajuste final e `RMSE_OOS` nas datas de decisão seguintes ao ajuste temporal.

`gap_percentual = 100 × (RMSE_OOS / RMSE_treino − 1)`.

O comparador ingênuo (**naive mean**) prevê a **média do alvo no conjunto de treino**, mantida fixa na janela OOS; **não é uma previsão sempre igual a zero**. A habilidade relativa é `skill = 1 − RMSE_modelo / RMSE_naive_mean`. No relatório, `overfit_warning` significa **skill_treino > 0 e skill_OOS ≤ 0**. Essas medidas são de **generalização preditiva**, não são probabilidades de sobreajuste.

## Resultados empíricos

| Indicador | Medido |
|---|---:|
| Ajustes válidos | 225/225 |
| Gaps positivos (erro OOS > treino) | **224/225 (99,56%)** |
| Gap mediano (cada ativo/fold com mesmo peso) | **+161,32%** |
| Gap quartis 25%–75% | **+103,09% a +222,16%** |
| Alertas de overfit (treino melhor que média, OOS não) | **196/225 (87,11%)** |
| Skill OOS positivo frente à média de treino | **29/225 (12,89%)** |
| Alertas de underfit (regra do script) | 0/225 |
| RMSE OOS agregado por erros quadráticos e número de rótulos | **0,284124** |
| RMSE OOS do comparador de média (mesmo agregado) | **0,232375** |
| RMSE OOS modelo / baseline − 1 | **+22,27% (modelo pior)** |
| Amostras OOS agregadas nos 225 ajustes | **112.725** (observações ativo-data, não independentes) |

**Correção do diálogo anterior:** a cifra anterior `0,2841 / 0,2560 ≈ +10,96%` não descreve o comparador naive mean auditado neste pacote. Neste pacote, a comparação com a média histórica treinada é **0,284124 / 0,232375 ≈ 1,222694**, isto é, **22,27% maior RMSE fora da amostra**.

### Por fold

| Fold | RMSE treino mediano | RMSE OOS mediano | Mediana do gap | Pares com OOS > treino |
|---|---:|---:|---:|---:|
| 1 | 0,065331 | 0,259751 | **+247,69%** | 75/75 |
| 2 | 0,086702 | 0,207636 | **+148,44%** | 74/75 |
| 3 | 0,100622 | 0,217743 | **+96,25%** | 75/75 |

As medianas dos gaps devem ser calculadas por par ativo/fold. Não calcular uma diferença percentual entre as duas medianas e supor que seja o mesmo valor.

### Subgrupos

| Grupo | Modelos | Gap mediano | Alertas overfit | Skill OOS positivo |
|---|---:|---:|---:|---:|
| 65 ativos base | 195 | **+160,22%** | 171/195 | 24/195 |
| 10 ativos selecionados | 30 | **+202,15%** | 25/30 | 5/30 |

Exemplos de maiores gaps medidos: `CRT` fold 1 **+733,15%**, `PNBK` fold 1 **+732,19%**, `CNQ` fold 1 **+709,96%**. O único gap negativo foi `NBR` fold 2, **−3,82%**. Isso não fornece por si só ranking de contribuição causal de cada ativo ao capital composto.

## Interpretação científica e limitações

Há evidência **forte e quantitativa** de um **gap de generalização preditiva** e do fato de o LightGBM, apesar do bom ajuste em treino, falhar frequentemente em superar uma referência ingênua em OOS. Chamar isso de apenas "suposição de overfitting" subestima a evidência. Ao mesmo tempo:

1. O gap por si só não permite decompor quanto é overfitting do ajuste, deslocamento de distribuição (*regime shift*) ou mudança de variância do alvo; os períodos de treino e teste são cronologicamente distintos.
2. Ativos, datas e folds possuem dependências; as 225 linhas não são 225 experimentos independentes.
3. O comparador é uma regressão ingênua da utilidade, não uma estratégia financeira `buy-and-hold` ou de rotação; RMSE pior **não prova** que a sequência Top1 e a política de troca destruam capital.
4. Os US$ 587 milhões são uma reprodução condicionada a um snapshot, universo e hiperparâmetros selecionados retrospectivamente. O teste OOS *interno* não substitui um teste prospectivo nunca utilizado pela pesquisa e não quantifica que parcela do capital tenha sido causada por sobreajuste.
5. Estes números dizem respeito ao **vintage legado congelado**, não devem ser transferidos sem medição para os modos RAW econômicos v1.25.x nem usados para escolher o vintage com maior capital.

**Conclusão:** O experimento v1.25.4 cumpre o objetivo de quantificar o gap de treino/teste. Há **evidência observacional forte de sobreajuste/fragilidade de generalização preditiva**. A robustez *econômica* do algoritmo de rotatividade depende de auditoria distinta das escolhas e do desempenho fora do desenvolvimento da estratégia. **Não otimizar em cima destes 225 testes para tentar recuperar o valor histórico.**

## Próximos testes pré-especificados

- Examinar acertos de **escolha e de rotação**, incluindo comparação da ação escolhida com `manter` e comparadores simples, em janelas cronológicas não utilizadas no desenvolvimento; métricas monetárias, custos e perdas.
- Medir estabilidade de scores e da regra de troca com os dois snapshots já congelados sem confundir a revisão de 73 barras de 2018 com o resultado RAW econômico v1.25.x.
- Validar economicamente sob avaliação temporal rigorosa e independente, sem alteração de ativos ou hiperparâmetros motivada pelos resultados do teste.
