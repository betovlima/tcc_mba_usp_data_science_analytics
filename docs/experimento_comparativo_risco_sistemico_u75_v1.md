# U75: comparacao simultanea de metodos de previsao do risco coletivo

Versao \`u75-systemic-risk-benchmark-v1.0.0\`, exclusivamente TCC e **shadow**.

## Executar no Spyder

Atualize sua branch de pesquisa e abra \`pesquisar_risco_sistemico_comparativo_u75_spyder.py\`, executando com F5. Requer os mesmos 75 CSVs RAW e corporate actions, contrato Strategy #30 e ZIP original exigidos pelos experimentos recentes. Dependencias: \`requirements.txt\` (mais scipy, dependencia do scikit-learn). **Nao e necessario reexecutar o LightGBM Strategy #30** para comparar somente a capacidade estatistica de prever eventos coletivos.

O executor usa os ativos declarados no manifesto congelado e valida as assinaturas, nao fixa contagem de universo ou datas no algoritmo estatistico. Permite \`TCC_SYSTEMIC_OUTPUT_DIR\`, \`STRATEGY30_SNAPSHOT_DIR\` e \`CONTROL_STRATEGY30_ZIP\`. Nao baixa Alpaca, nao acessa banco nem sistema operacional.

Abre **sete graficos no Spyder** com \`plt.show()\`, preservando arquivos PNG e tabelas CSV em pasta isolada por execucao:
\`output/comparacao_risco_sistemico_u75/<timestamp>/\`. O ZIP inclui \`resultado.json\`, \`comparacao_metricas.csv\`, \`comparacao_por_ano.csv\`, \`previsoes_por_modelo.csv\`, \`auditoria_folds.csv\`, \`eventos_e_antecedencia.csv\`, \`atributos_coletivos.csv\`, \`rotulos_futuros_poshoc.csv\`, \`snapshot_identidade.csv\`, \`resumo.md\`, graficos PNG e log persistido de progresso. Em caso de falha, gera \`erro_execucao.txt\` e ZIP de diagnostico.

## Metodos de fato implementados

- **Baseline:** probabilidade historica de evento maturado, com suavizacao.
- **CUSUM** de retornos coletivos negativos padronizados pela volatilidade anterior, com calibrador logistico.
- **BOCPD**, run-length Bayesiano com observacao Student-t e hazard predefinido, com calibrador logistico.
- **Markov/HMM:** tres estados gaussianos diagonais, parametros ajustados com EM/Baum-Welch exclusivamente no treino; probabilidades filtradas no fechamento t, calibradas para o evento.
- **PCA/Absorption Ratio:** decomposicao da correlacao Ledoit-Wolf anterior, quota da primeira componente e soma das 20% componentes maiores, com breadth.
- **DCC_proxy:** atualizacao recursiva tipo DCC com parametros fixos e residuos padronizados por EWMA; **nao e DCC-GARCH ajustado por maxima verossimilhanca**. A aproximacao e explicitamente identificada nos resultados.
- **EVT-POT:** cauda de perdas diarias do indice coletivo com distribuicao generalizada de Pareto (GPD), recalculada com janela movel apenas sobre o passado; calibrador logistico.
- **LedoitWolf:** distancia de Mahalanobis calculada com covariancia shrinkage anterior e coexcedencias.
- **Regressao Logistica:** todas as features coletivas.
- **LightGBM:** mesmas features, restricao de profundidade e regularizacao predefinidas.

Metodos *nao usam sinais dos ativos individualmente para disparar CASH*. A unidade da previsao e a **queda do conjunto**.

## Alvo e metodologia

Para cada horizonte h (1, 5, 10 e 20 pregoes), um evento futuro exige:
1. retorno logaritmico do indice U75 equal-weight entre t+1 e t+h abaixo do quantil inferior de 15% observado **somente no treino do fold**;
2. ao menos 60% do universo com retorno acumulado negativo entre t e t+h.

A amostra de treinamento inclui apenas exemplos cujo **desfecho esteja maturado antes do inicio do ano de avaliacao**. A avaliacao e walk-forward anual expansiva e usa o **mesmo conjunto de datas** para todos os modelos. Sem interpolar dados faltantes. Alerta definido no quantil 90% das probabilidades in-sample de treinamento, com criterio estrito para valores constantes. Metricas: Brier, logloss, AUC ROC, average precision, precisao/sensibilidade de alertas, VP/FP/FN, contagem de eventos e graficos cronologicos/calibracao.

**ATENCAO:** quantil, requisito de breadth, risco e memoria sao hiperparametros cientificos predefinidos, ainda nao validados. A primeira execucao sobre o historico ja inspecionado e exploratoria, nao prova generalizacao nem ausencia de sobreajuste. Eventos rotulados em janelas sobrepostas nao sao crises independentes; a tabela de onset apenas identifica inicio da sequencia de decisoes rotuladas como evento.

Este laboratorio **nao executa CASH** e **nao descobre nem inclui novos ativos**. A proxima fase, condicionada a resultados preditivos, compara risco coletivo com a melhor oportunidade elegivel entre os ativos U75 e CASH, com custos, execucao na abertura seguinte e politica consistente de reentrada. O Control Strategy #30 continua congelado, com a limitacao conhecida de elegibilidade historica t+1; este estudo nao declara que a removeu.
