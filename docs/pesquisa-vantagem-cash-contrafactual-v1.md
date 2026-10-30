# Pesquisa v1.0.6: vantagem contrafactual CASH versus continuar Control

**Branch:** `research/optimal-switching-cash-reentry-v1`
**Versao:** `cash-counterfactual-continuation-v1.0.6`
**Executor Spyder:** `pesquisar_vantagem_cash_contrafactual_spyder.py`

## Escopo

O universo cientifico permanece **um conjunto unico de 75 ativos**.
Mesmo calendario (AAPL com atributos preparados, 2506 sessoes), 3 folds,
1560 sessoes fora da amostra, custos e modelo Control de referencia,
com checkpoint inalterado **US$ 587.165.941,13**. Sem acessar banco de
dados ou baixar historicos, sem tocar a `main` ou qualquer aplicacao operacional.

## Definicao contrafactual

Para cada fechamento t do periodo de calibracao, considerando a posicao
que o Control manteria naquela data, criam-se dois ramos historicos:

- **Continuar Control:** seguir a decisao da politica Control em t e
  continuar seguindo a propria politica ate t+10.
- **CASH agora:** manter CASH na proxima sessao e, da decisao seguinte
  em diante, seguir novamente a mesma politica Control ate t+10.

Em cada ramo, **posicao e tempo de permanencia sao evoluidos
separadamente**. Isso permite que as rotacoes futuras sejam diferentes
devido aos estados de carteira distintos. Os custos proporcionais e os
retornos close(t) -> close(t+1), com execucao na abertura seguinte,
seguem a mesma regra de treinamento do motor oficial:

`Y_t = log(W_cash,t+10 / W_control,t+10)`

Os rotulos Y_t usam precos futuros **somente de calibration**, nunca do
periodo OOS. Nenhum rotulo cruza a data final da calibracao. Um valor
positivo significa que, historicamente, a espera de uma decisao superou
seguir Control, descontada a aproximacao de custos da etapa de treinamento.

## Estimacao

Usa 8 atributos observaveis no fechamento: retorno 5/20, volatilidade 20,
distancia do topo 20, eficiencia de tendencia, escore relativo do candidato
Control, indicador de estar em CASH e tempo ja investido. O escore Control
serve como atributo contextual, nao como probabilidade ou retorno esperado.

Uma regressao Ridge com alpha 10.0, fixo antes deste teste, e ajustada
somente com os exemplos contrafactuais da calibracao do fold. Uma
separacao interna cronologica, com embargo de pelo menos dez sessoes
nos rotulos, produz MAE e acuracia direcional de diagnostico.
Hiperparametros nao sao escolhidos pelo resultado OOS. Em seguida,
o mesmo modelo e reajustado em toda a calibracao (ainda anterior ao
teste). Os coeficientes e a padronizacao sao exportados, sem pickle.

Em cada nova decisao do teste, o modelo estima a vantagem CASH menos
continuar Control. Se positiva, o overlay escolhe CASH; caso contrario,
executa o Control. Reavaliacoes diarias podem prolongar o intervalo em
CASH e permitir retorno posterior.

## Limites e cuidados

Nao e solucao exata do problema de Bellman com 76 estados. Os rotulos
usam custos proporcionais de treinamento, e o replay aplica as taxas
completas. A regressao aprende a partir de menos de 126 sessoes por fold;
a estimativa pode ser instavel. A calibracao do Control tambem seleciona
uma margem usando essa mesma janela, fonte potencial de vies. A selecao
historica do universo nao prova elegibilidade prospectiva.

**As versoes anteriores ja foram avaliadas no mesmo OOS. Portanto, apesar
do isolamento cronologico dentro de cada fold, nao se pode declarar a
v1.0.6 validada externamente como nova descoberta sem um periodo
completamente independente.** Nao otimizar parametros olhando o capital
final.

## Reproducao e pacote

Atualizar a branch, abrir
`pesquisar_vantagem_cash_contrafactual_spyder.py` e executar F5.
O executor valida o checkpoint antes de ajustar o modelo novo e gera
sempre um ZIP:

`output/counterfactual_cash_continuation/pacote_optimal_switching_cash_<run_id>.zip`

Dentro do pacote: `resultado.json`, series de capital e negocios do
Control e contrafactual, `calibration_counterfactual_labels.csv`,
`inner_validation_predictions.csv`, `counterfactual_fold_models.csv`,
`counterfactual_model_manifest.json`, `counterfactual_decisions.csv`,
`resumo_para_analise.md` e `manifesto_pacote.json`, com SHA-256.
Em falha, `erro_execucao.json` e evidencias ja produzidas sao
compactados. Dados RAW, eventos corporativos e segredos nao sao incluidos.
