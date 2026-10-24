# Protocolo pré-execução — mitigação de sobreajuste LightGBM v1.25.7

**Data:** 23/10/2026. **Branch única:** `research/v1.23.1-fresh-snapshot-20261022`; a `main` e a pesquisa operacional MCT não são alteradas. **Executor:** `pesquisar_mitigacao_overfitting_spyder.py`.

## Pergunta e hipótese

O LightGBM histórico apresenta 224/225 ajustes com RMSE fora da amostra acima do treino, gap mediano +161,32%, e somente 29/225 modelos superam a média do alvo calculada no treino. **Objetivo:** verificar se três intervenções simples de regularização, cada uma isolada, reduzem o erro preditivo OOS e/ou melhoram a estabilidade das decisões *sem necessariamente ampliar o patrimônio histórico*. Não há garantia de melhoria.

## Contrato congelado

Dados: vintage antigo que reproduz US$ 587.165.941,13 até 06/10/2026; `dados/u67` + cache candidato antigo; hash dos arquivos verificado contra `evidencias/manifest_universo_congelado.json` antes e após treino. Universo efetivo de 75 ativos, exclusões de DOC e CLMT por problema estrutural, 3 folds, mesma função alvo, engenharia de atributos, purga temporal, mesma calibração de margem e custo do motor legado v1.23.0. Nenhum download Alpaca ou Mongo e nenhuma mudança em controle de produção.

**Controle — sem mudanças:** `n_estimators=329`, `min_child_samples=18`, `subsample=.85`, `subsample_freq=0`, demais parâmetros originais. O controle DEVE reproduzir exatamente US$ 587.165.941,13 até US$ 0,01; divergência interrompe a campanha.

**Intervenções pré-especificadas, não cumulativas:**
1. `bagging_ativo`: apenas `subsample_freq=1`, mantendo `subsample=.85`. O valor original de frequência 0 desabilita a amostragem de linhas nas iterações; o teste ativa a regularização já prevista pelo 0,85.
2. `folha_min_50`: apenas `min_child_samples=50` (antes 18).
3. `menos_arvores_160`: apenas `n_estimators=160` (antes 329).

**Early stopping** exige validação interna própria e não é falsamente apresentado como parte deste experimento. Alterá-lo exigiria um protocolo separado sem utilizar os folds OOS para escolher melhor época.

## Medidas pré-definidas

- Predição: `RMSE_OOS` ponderado por quantidade de rótulos válidos, `RMSE` da previsão ingênua de **média do alvo no treino**, mediana por ativo-fold de `100 × (RMSE_OOS/RMSE_treino − 1)`, `overfit_warning`, proporção dos ajustes com skill OOS positivo.
- Decisão: quantidade de seleções `selected_asset` diferentes das 1.560 sessões do controle.
- Economia: capital final da mesma política e benchmark, CAGR, Sharpe, drawdown máximo, pior fold, rotações e a curva de capital.
- Figuras: RMSE OOS por variante, gap mediano, curvas de capital, mudanças na sequência de ativos.
- Saída: `output/mitigacao_overfitting_v1257/<run_id>/pacote_mitigacao_overfitting_v1257.zip`, com métricas completas por modelo/fold, calibração e decisões de cada variante, verificações e gráficos PNG/SVG.
- Resultados parciais da campanha são gravados em `estado_parcial.json` e cada variante possui diretório próprio. A execução não implementa resume automático e não sobrescreve dados originais.

## Regras de interpretação

O experimento é **retrospectivo exploratório**: o universo de 75 ativos e o histórico de 2020 a 2026 já influenciaram decisões de pesquisa anteriores. Portanto, mesmo que uma variante melhore RMSE, capital ou estabilidade, **isso não valida uma nova estratégia prospectiva nem autoriza selecioná-la pelo máximo do teste OOS**. Reduzir gap às custas de maior erro de teste não representa melhoria. Mudança das escolhas não prova estabilidade; medir estabilidade contra novas perturbações exige experiência independente. Se o controle falhar, a resposta válida é auditar o contrato, nunca alterar dados para recuperar capital.

A aprovação de qualquer novo modelo operacional exigiria desenho prospectivo independente. Nenhuma promoção ou alteração da estratégia é realizada pelo código.

## Estado

Código e protocolo publicados na branch de pesquisa. **Ainda não há resultados de execução da versão v1.25.7**; o código precisa ser executado no ambiente Spyder onde reside o cache original congelado.
