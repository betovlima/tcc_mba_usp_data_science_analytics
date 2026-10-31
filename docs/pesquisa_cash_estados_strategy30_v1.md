# Pesquisa CASH com estados de reentrada: Strategy #30

## Hipotese experimental

Control oficial: US$ 80.685.454,79 em 1576 sessoes, 75 ativos,
2020-07-22 a 2026-10-28. Ele foi reproduzido independentemente no TCC.

A pesquisa pergunta se uma politica adicional consegue decidir quando
ficar em CASH e quando voltar ao ativo selecionado pelo mesmo Control,
sem usar informacao do pregão seguinte.

## Metodo predeclarado

1. Recarregar os 75 arquivos RAW/SIP e corporate actions de
   `dados/strategy30_snapshot`. Validar o ZIP original da Strategy #30.
2. Treinar os modelos LightGBM Control com a revisao congelada da fonte.
   Em cada fold, selecionar a margem de rotacao na calibracao anterior ao
   teste e ajustar o modelo final antes do intervalo OOS. Nenhuma margem
   e forçada a reproduzir os 80 milhoes.
3. Nas **126 sessoes de calibracao**, simular seis trajetorias de estado:
   Control puro e janelas fixas de CASH 1/7, 2/11, 4/15, 8/20 e 13/31.
   A agenda CASH nao utiliza retornos futuros.
4. Para cada estado atingido, criar o rotulo de 10 sessoes:
   log(patrimonio se CASH for escolhido nesta decisao e depois Control)
   menos log(patrimonio seguindo Control desde esta decisao). Cada ramo
   mantem posicao e dias investidos proprios, e somente o historico
   de calibracao pode formar o rotulo.
5. Treinar uma unica regressao Ridge com alpha=10 e atributos conhecidos
   na data: return_5, return_20, vol_20, distancia ao topo em 20 dias,
   eficiencia de tendencia, score Control, estado CASH e dias em posicao.
   Validacao cronologica interna 75/25 com purga de pelo menos 10 sessoes.
6. Refazer o backtest **Control puro** com o mesmo motor e verificar todas
   as 1576 decisoes/folds e capital por sessao do ZIP congelado, ate um
   centavo. Se divergir, abortar sem validar nenhuma estrategia CASH.
7. Rodar o overlay CASH no simulador financeiro original com as mesmas
   taxas e datas, uma decisao por pregão, guardando entradas e reentradas
   efetivas, capital por fold, Sharpe, MaxDD e taxas.

## Execucao no Spyder

Branch de pesquisa: `research/control30-cash-state-aware-v1`.
Execute `pesquisar_cash_estados_strategy30_spyder.py` com F5, apos
`git pull` da branch.

Nao baixe dados novamente: o snapshot da paridade validada e reutilizado.
O arquivo ZIP original de resultados deve permanecer em `Downloads`
ou `evidencias/` (ou configurar `CONTROL_STRATEGY30_ZIP`).

A saida `output/cash_state_aware_strategy30/` inclui CSVs de rotulos,
validacao cronologica, metricas por fold, serie Control reproduzida,
serie CASH, decisoes, periodos efetivos CASH, comparacao por sessao
e resumo. O ZIP usa SHA-256 e e produzido mesmo em erro.

## Limites de interpretacao

Amostras historicas de multiplas trajetorias da mesma sessao sao
correlacionadas: o numero de linhas nao representa observacoes
independentes. Rotulos historicos aproximam taxas; o simulador OOS usa
custo real, mas nao se pode prometer identidade dos dois procedimentos.

O rotulo considera CASH apenas na decisao inicial e Control nas
seguintes. No teste, a politica pode ficar em CASH consecutivamente:
essa diferenca continua sendo risco metodologico documentado.

O periodo OOS ja foi inspecionado muitas vezes em estudos anteriores.
Nao se deve declarar generalizacao prospectiva nem ausencia de
overfitting com base apenas nele. Os parametros estao fixados antes
da execucao desta versao e nao devem ser selecionados por capital OOS.

Nao alterar o motor original nem a aplicacao operacional.
O Control anterior de US$ 587,17 milhoes permanece como referencia
historica arquivada, nao como alvo numerico do experimento atual.
