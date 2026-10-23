# Resultado da auditoria dos retornos econômicos com splits — TCC v1.24.4

**Data:** 23/10/2026. **Branch:** `research/v1.23.1-fresh-snapshot-20261022`. **Código executado:** `auditar_retornos_economicos_splits_spyder.py`, commit `0052964dc5c35bb0ab0b8f435ffa986793c49482`. **Testes da campanha:** `tests/test_auditar_retornos_economicos_splits.py`, commit `78c0100c1b451ae086ec7ef46e3d5cc9f3750caf`. **HEAD antes da auditoria:** `72efd8b4f067b8adbbd8e9392167a23668581027`. **Main mantida:** `4667d2ccee3f3441ee4d2b362ce734311cedd5e3`; PR #18 Draft.

**Pacote recebido:** `pacote_retornos_economicos_splits.zip`, SHA-256 **`8b1903bbaeb0ed4694b9bb3abc705782d138bed02610a02ff8d63feb85a41ef3`**. Contém `resumo.json`, `retornos_por_decisao_e_horizonte.csv` (11.980 linhas × 16 campos) e `comparacao_por_ativo.csv` (16 linhas × 5 campos). **Snapshot dos 75 ativos:** SHA-256 `595b7a03dc2c8978487975cd3554b1ea65608741893f13208938dc624cb273cc`.

## 1. Hipótese e escopo efetivamente testados

O pesquisador exige **preços e volumes RAW preservados**. Para operações hipotéticas que compram na abertura do pregão seguinte à decisão e encerram no fechamento da sessão t+h, calcular:

1. **Retorno RAW ingênuo:** `close_saida/open_entrada - 1`, sem considerar splits.
2. **Retorno econômico por custódia:** `close_saida * produto(new_rate/old_rate) / open_entrada - 1`, considerando apenas splits com `entry_day < ex_date <= exit_day`. Quem compra na própria ex-date **não** recebe um multiplicador retrospectivo de quantidade.
3. **Controle matemático legado:** normalizar preços por splits **somente em memória** e calcular retorno entre abertura e fechamento nas mesmas datas.

São **retornos simples de preço**. O teste não contempla dividendos em dinheiro, custos de transação, spread, slippage, marcação diária, risco intrahorizonte, frações/cash-in-lieu, políticas de rotação ou patrimônio composto. A equivalência (2) × (3) decorre também da mesma álgebra de fator de split; **não é uma validação externa de preços ou execução econômica integral**.

## 2. Resultados medidos e verificados no ZIP

| Indicador | Resultado |
|---|---:|
| Universo fixo | **75 ativos** |
| Ativos com evento de split registrado | **17** |
| Ativos com janelas de retorno avaliáveis | **16**; RACE tem evento fora do período das barras |
| Janelas por decisão e horizonte | **11.980** |
| Horizontes | **5, 10, 20, 40, 60 sessões** |
| Janelas efetivamente atravessando split | **2.596** |
| Janelas sem split entre compra e saída | **9.384** |
| Janelas em que RAW ingênuo divergiu do econômico | **2.596** |
| Janelas com sinais diferentes no retorno ingênuo versus econômico | **1.427** das 2.596 com split |
| Retorno econômico diferente do controle legado sob `np.isclose` | **0/11.980** |
| Máximo erro absoluto econômico versus legado | **4,440892098500626 × 10⁻¹⁶** |
| Maior distância de retorno simples RAW-ingênuo versus econômico | **217,1013424585** unidades de retorno (21.710,13 pontos percentuais) |
| RAW alterado, treinamento de modelos, simulação de patrimônio | **Não** |

**Desagregação por horizonte:**

| Horizonte | Total de comparações | Atravessando split |
|---|---:|---:|
| 5 | 2.396 | 80 |
| 10 | 2.396 | 180 |
| 20 | 2.396 | 380 |
| 40 | 2.396 | 780 |
| 60 | 2.396 | 1.176 |
| **Total** | **11.980** | **2.596** |

Os rótulos de disponibilidade `label_matures_after_close_on` coincidem em todas as linhas com `exit_day`. **Esse carimbo no CSV não prova, sozinho, que o ajustador LightGBM efetivamente excluiu rótulos imaturos; isso requer auditoria dos limites dos folds e amostras usadas no ajuste.**

## 3. Exemplos reais de resultados

O efeito pode ser muito grande em agrupamentos (*reverse splits*). Em **NBR**, a janela de decisão 12/03/2020, entrada 13/03 e saída 08/06, horizonte de 60 sessões, atravessou um grupamento 50:1 (multiplicador de quantidade 0,02). Retorno RAW ingênuo **220,531982** (22.053,20%), contra **3,430640** (343,06%) economicamente neutralizado pelo grupamento. Ambos são retornos de preço de uma janela hipotética e **não lucros efetivos da estratégia Control**.

**Igualdade observada:** 11.980/11.980 retornos econômicos por quantidade são equivalentes aos da normalização antiga, com erro máximo na escala de precisão de ponto flutuante. Portanto, **não é necessário reescrever as barras RAW para obter o retorno de preço economicamente comparável**, desde que se aplique corretamente o fator de quantidade na janela.

## 4. Interpretação científica e limites

- **Demonstrado neste conjunto de janelas:** preservar RAW e ajustar **somente as quantidades econômicas** permite reproduzir os retornos simples de preço do legado. O algoritmo antigo fazia reexpressão retrospectiva, mas esse procedimento não é necessário para essa parte dos rótulos.
- **Não demonstrado:** que os 52 atributos, as sete utilidades (penalidade por downside, drawdown, movimento capturado, persistência), os custos e o backtest final permaneçam iguais. As auditorias v1.24.2 e v1.24.3 mostraram que os atributos e targets em janelas que atravessam eventos exigem tratamento específico. Também não foi testada a compatibilidade dos preços RAW com os dividendos e as regras de custódia da corretora.
- **Maturidade temporal:** um label supervisionado que usa preços até t+60 **só fica disponível depois do fechamento de t+60**. Atribuí-lo ao instante t é permitido como índice de supervisão histórica **apenas se** o treinamento cronológico não o utiliza antes de completar o horizonte. Os 60 pregões de purge configurados precisam ser verificados sobre os folds reais e as amostras efetivamente selecionadas.
- **Integridade:** o resultado dos US$ 587.165.941,13 no dataset antigo não é substituído nem confirmado por este teste de retornos; a diferença para o dataset novo foi atribuída em ensaio independente a 73 barras RAW revisadas em maio de 2018. A v1.24.4 não calcula capital, não promove parâmetros e não altera a `main`.

## 5. Próxima campanha proposta: auditoria de maturidade real dos labels

Inspecionar, para os **mesmos 75 ativos e seus três folds walk-forward**, os índices dos conjuntos de treinamento de calibração e treinamento final, a distância do último rótulo potencialmente usado até o início de sua avaliação e os horizontes 5/10/20/40/60. Não supor vazamento nem segurança sem medir. Distinguir o timestamp da última observação de features (t) do instante em que se conhece seu alvo (fechamento t+h). Registrar as margens temporais e falhar em qualquer amostra de treino cujo alvo amadureça depois do início do conjunto de avaliação. Script novo deverá operar offline, sem retreinar ou alterar séries.
