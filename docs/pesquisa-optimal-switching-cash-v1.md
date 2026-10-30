# Pesquisa: Optimal Switching CASH/Reentrada v1.0.0

**Branch:** `research/optimal-switching-cash-reentry-v1`.
**Estado:** prototipo cientifico implementado, execucao financeira longa deve
ser feita localmente (Spyder). Nao e alteracao do MCT operacional.

## Problema

O Control classifica o melhor ativo do universo mesmo quando esse ativo pode
ter retorno economico desfavoravel. A pesquisa testa, separadamente do ranking
LightGBM, uma politica matematica para abandonar a exposicao e recomprar.

## Desenho e equacao

A politica Control por fold continua produzindo o **ativo candidato**. Um
modelo estatistico independente estima para **cada ativo e fold** tres regimes
a partir do momentum logaritmico dos ultimos 20 fechamentos dividido pela
volatilidade realizada correspondente. Os limites dos tres regimes sao
tercis calculados **somente no trecho de ajuste historico daquele fold**.

Para cada regime, a media dos retornos realizaveis entre
`abertura(t+1)` e `abertura(t+2)`, a media das perdas
`max(-retorno, 0)` e a matriz de transicao empirica (suavizada com
Dirichlet(1)) sao ajustadas **apenas ate o fim da janela de treinamento**.
A data t+2 nunca ultrapassa a data de corte do ajuste. O modelo nao
reutiliza a utilidade nem o ranking LightGBM para estimar risco absoluto.

Os dois estados da carteira sao `p=0` (CASH) e `p=1` (investido no
ativo indicado pelo Control). A programacao dinamica finita usa:

`V_h(z,p) = max_a [ E(r|z,a) - lambda E(perda|z,a) - c(p,a)
                  + sum_z' P(z'|z) V_(h-1)(z',a) ]`

com horizonte de 10 decisoes, `lambda=1` predefinido e retorno de
CASH igual a zero; a taxa livre de risco e tributacao nao sao modeladas.
Sair e reentrar nao dependem de um stop percentual fixo. O modelo pode
escolher CASH mesmo se a utilidade relativa do melhor ativo for positiva.

Os custos de troca usados no Bellman sao uma **aproximacao proporcional**
(slippage, comissao e SEC na venda); o replay financeiro compartilhado
aplica **as taxas efetivas calculadas pelo motor**. A troca entre dois
ativos distintos e avaliada pelo Control, e nao como terceiro estado
na DP: isso limita a otimalidade do prototipo e sera estudado posteriormente.

## Reproducao

A primeira campanha usa apenas o **U67 base congelado** (67 solicitados;
65 ativos efetivos). O script exige SHA-256 do snapshot identico ao
manifesto historico de corte **06/10/2026**. Nao baixa dados da Alpaca,
nao acessa banco, nao altera os CSVs nem o codigo oficial do Control.

1. Atualizar a branch de pesquisa e instalar `requirements.txt`.
2. Abrir `pesquisar_optimal_switching_cash_spyder.py` no Spyder,
   com diretorio de trabalho na raiz, e executar F5.
3. O Control e obrigatoriamente reproduzido ao centavo: **US$ 76.927.051,39**.
   Qualquer divergencia aborta o experimento **antes** de comparar.
4. O overlay usa exatamente os mesmos folds, caches de previsao, taxas,
   janelas temporais e simulador; so altera a escolha de exposicao/CASH.
5. Resultados em `output/optimal_switching_cash/<timestamp>/`:
   `resultado.json`, posicoes/operacoes Control e Switching,
   `switching_decisions.csv`, `switching_fold_models.csv`.
   O output anterior nao e sobrescrito.

## Integridade e limitacoes

- Decisao no fechamento de t e execucao na abertura de t+1.
- Preco futuro nao e usado em caracteristicas do dia de decisao.
- O corte historico exclui a queda de CAPR entre 20 e 30/10/2026.
- Nenhum resultado financeiro novo foi medido na criacao da branch;
  o experimento completo precisa ser executado.
- A selecao historica de ativos e a realizabilidade financeira nao
  sao comprovadas por este teste. O universo de 75 ativos selecionados
  requer cache externo preservado e validado por SHA; nao comparar os
  US$ 587 milhoes diretamente com os resultados do U67 base.
- Nao escolher novos parametros observando o desempenho do teste.
- Nao alterar a politica operacional do MCT antes de validar a hipotese
  em janelas independentes e executar os testes de custos e liquidez.
