# Pesquisa: capitalizar lucros mantendo a posição até atingir o alvo

## Hipótese

Partir de US$ 10.000. Quando a carteira está em CASH, usar o mesmo
LightGBM do TCC e seu ranking de utilidade para escolher o ativo.
Comprar na abertura do pregão seguinte. **Não vender apenas porque o
ranking mudou**. Manter uma ordem de realização de lucro de 1,5%, 3%
ou 5% sobre o preço efetivo de compra. Após alcançar o alvo, vender,
manter CASH até o fechamento do pregão e usar o próximo sinal válido
para reinvestir o capital integral, incluindo lucros anteriores.
O mesmo ativo pode ser recomprado.

Essa é uma hipótese diferente da pesquisa anterior, em que as rotações
normais podiam encerrar uma posição antes do alvo, inclusive com perda.

**Não há stop loss e não há prazo máximo de permanência.** Isso é
intencional para isolar a hipótese matemática. Se o alvo não for
atingido, a posição pode permanecer aberta durante anos e seu
patrimônio será marcado diariamente pelo fechamento efetivo.
A única liquidação forçada ocorre no último pregão histórico, para
contabilizar a posição pelo valor de mercado e pelas taxas de venda.

## Como comparar corretamente

O arquivo `pesquisar_capitalizacao_ate_alvo_spyder.py` usa o
mecanismo oficial de reprodução congelada v1.23.0.
Primeiro exige que o Control reproduza US$ 587.165.941,13 com
diferença de no máximo um centavo e que o replay de referência
reproduza a curva diária original com diferença inferior a dois
centavos. Se não conseguir, interrompe o estudo.

Depois, utilizando os **mesmos modelos, folds, dados e sinalizadores**
de posição do processo original, simula 7 cenários:

| Código | Regra |
| --- | --- |
| control | Política original, com rotações |
| tp_015_com_rotacao | TP de 1,5% e rotações originais |
| tp_015_ate_alvo | TP de 1,5%, sem rotações |
| tp_030_com_rotacao | TP de 3% e rotações originais |
| tp_030_ate_alvo | TP de 3%, sem rotações |
| tp_050_com_rotacao | TP de 5% e rotações originais |
| tp_050_ate_alvo | TP de 5%, sem rotações |

Avaliamos capital final, Sharpe, drawdown máximo, número efetivo de
saídas por lucro, períodos em CASH, maior duração de uma posição,
perdas não realizadas e total de taxas. A pesquisa recalcula a
carteira financeira inteira para cada cenário, inclusive a próxima
escolha de ativo quando a estratégia volta a CASH.

## Comparar com a projeção matemática

A projeção de rendimento diário de 3% em **todos** os 1.560 pregões
usa `C=10.000*(1+0,03)^1560`. Ela pressupõe um ganho positivo em
**cada pregão**, sem exceções.

A nova estratégia tem ganho de 3% **somente nas operações cujo preço
atingiu a meta**, que pode levar mais de um pregão, e pode encerrar
o período com perda. Por isso, `C=10.000*(1+0,03)^N`, onde N é o número
efetivo de saídas por alvo, é apenas uma referência otimista,
desconsiderando custos e a liquidação final. O valor apurado pelo
backtest usa todos os preços e eventos, não substitui perdas por zero.

## Mercado e cronologia

O preço de entrada é a abertura seguinte a uma decisão de fechamento.
A ordem limite fica ativa na sessão da compra e nas sessões seguintes.
Se a abertura estiver acima do alvo, a venda é simulada na abertura;
senão, se a máxima diária tocar o alvo, a venda é simulada no alvo.
Não se usa a máxima futura para escolher o ativo.
**Não se recompra no mesmo pregão da venda**, mesmo se houver outro
sinal. Se a série de preços desaparecer, a execução para com erro,
sem inventar cotação nem um preço de venda.

A execução inferida de barras diárias não comprova liquidez, spread,
tamanho disponível, fila de execução ou impacto sobre os preços. Os
resultados não são evidência de retorno realizável com capitais muito
altos e são exploratórios porque a escolha dos alvos não foi feita
em um período temporal independente.

## Execução

Na raiz do projeto:

```bash
git fetch origin
git switch research/compound-hold-until-target-v1
python -m pytest tests/test_capitalizacao_ate_alvo.py -q
python pesquisar_capitalizacao_ate_alvo_spyder.py
```

No Spyder, basta abrir `pesquisar_capitalizacao_ate_alvo_spyder.py`,
reiniciar o kernel e executar com F5.

A reprodução exige os CSVs congelados em `dados/u67` e
`dados/.u67_expansion_cache/full_raw_20261006`; esta segunda pasta
não é versionada. Os hashes do manifesto são verificados antes do
treinamento. **Não se faz download** nem se modifica a reprodução
oficial ou a branch main.

## Resultados

A pasta `output/pesquisa_capitalizacao_ate_alvo` contém:
- `comparacao.csv`, `resultado.json`;
- sete arquivos de patrimônio e sete de operações (CSV);
- `comparacao_capital.png` e `comparacao_drawdown.png`;
- `reproducao_controle/` com a validação do checkpoint congelado.

Os resultados numéricos dos sete cenários só existem após rodar
o script com os dados originais. Não inferir retornos a partir do
número de pregões ou de sinais do modelo.
