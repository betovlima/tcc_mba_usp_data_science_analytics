# Pesquisa: capitalização de lucro com saída por utilidade econômica

## Pergunta

Na mesma série histórica congelada do TCC, a capitalização de ganhos
em alvos de 1,5%, 3% e 5% melhora quando a engine consegue abandonar
antecipadamente uma posição cuja **utilidade estimada perdeu
atratividade**, sem exigir um prazo fixo de 5, 10 ou 20 pregões?

## Regras, sem informação futura

A entrada em um ativo é proposta pelo LightGBM no fechamento D e
executada na abertura D+1, usando o patrimônio inteiro.
Depois da compra, mantém-se ordem limite de realização de lucro:
se a abertura alcançar o alvo, vende na abertura; senão, se a máxima
do dia alcançar, vende pelo limite. Não há recompra no mesmo dia.

Enquanto houver posição, a saída pode ser antecipada **na abertura D+1**
quando o sinal conhecido no fechamento D atender ao critério específico
do cenário. Se uma saída de utilidade estiver programada e a abertura
D+1 já ultrapassar o lucro-alvo, a venda é contabilizada como realização
do alvo, e não como venda por utilidade.

Os dois gatilhos de utilidade usam **exclusivamente as previsões fora
da amostra já geradas pela engine original**, sem reestimar nenhum modelo:

1. `negative`: utilidade prevista do ativo mantido menor ou igual
   à alternativa CASH, cuja pontuação é 0. Esse gatilho pode sair
   mesmo durante a permanência mínima, pois é uma proteção independente.
2. `native_switch`: a política Control original recomenda mudar de
   ativo ou ir para CASH, respeitando seus parâmetros de permanência
   mínima e margem de troca calibrada em cada fold. Aqui a recomendação
   apenas encerra a posição, deixando o dinheiro em CASH. A **próxima
   compra** espera a indicação válida após esse fechamento.
3. `combined`: basta um dos dois gatilhos.
4. `none`: manter até o lucro-alvo, sem saídas por utilidade. É a
   reprodução da pesquisa anterior.

A política Control original com rotações imediatas continua como
comparação separada. Portanto, não confundir `native_switch` com o
Control: na variante nova, a venda não é seguida de uma compra na mesma
abertura.

## Comparações: 16 cenários

- Control original sem alvo fixo;
- 3 alvos com as rotações originais;
- 3 alvos sem rotação e sem saída antecipada;
- 3 alvos com gatilho `negative`;
- 3 alvos com gatilho `native_switch`;
- 3 alvos com gatilho `combined`.

Todos partem de **US$ 10.000** e usam os mesmos folds cronológicos,
mesmas séries CSV, mesmas taxas e mesmo ranking LightGBM. Não se
seleciona limiar novo por busca no retorno realizado; o limiar CASH=0
e a margem de troca são os da política já existente.

O checkpoint do Control deve ser reproduzido até o centavo,
US$ 587.165.941,13, e o replay sem TP deve coincidir com a curva
diária original, tolerância máxima de US$ 0,02. Em caso de falha,
o script interrompe. Os caches de utilidade são capturados no instante
da própria reprodução oficial, associados à identidade de cada
universo, fold e data; exige-se o mesmo vetor de scores, inclusive
a coluna CASH, sem preenchimento de valores futuros.

O estudo é exploratório: o histórico já foi visto em pesquisas
anteriores. Para conclusão de generalização, será necessária nova
validação cronológica fora das janelas usadas para conceber a regra.

## Execução

Abra a branch `research/compound-utility-exit-v1`.
No terminal:

```bash
git fetch origin
git switch research/compound-utility-exit-v1
python -m pytest tests/test_capitalizacao_saida_utilidade.py -q
python pesquisar_capitalizacao_saida_utilidade_spyder.py
```

No Spyder, reinicie o kernel e execute com F5 o arquivo
`pesquisar_capitalizacao_saida_utilidade_spyder.py`.

O treinamento requer os CSVs congelados de `dados/u67` e de
`dados/.u67_expansion_cache/full_raw_20261006`.
O cache adicional **não está versionado no Git**. O experimento não
baixa dados novos nem altera os modelos.

## Resultados gerados

`output/pesquisa_capitalizacao_saida_utilidade/`:

- `comparacao.csv` com patrimônio, drawdown, Sharpe, saídas por alvo,
  por utilidade e períodos em CASH;
- `resultado.json` com métricas por fold e proveniência;
- 16 pares de CSV de patrimônio diário e operações;
- `comparacao_capital.png` e `comparacao_drawdown.png`;
- figuras específicas por alvo (1,5%, 3%, 5%);
- `reproducao_controle/` com a verificação do checkpoint.

Não há garantia de execução integral no preço limite com dados OHLC.
Não se modela fila de ordens, spread real, volume disponível, impacto
de mercado ou negociações fracionadas. A pesquisa não é recomendação
de operação financeira.

O Control original e a branch main permanecem intactos.
