# Pesquisa exploratória de realização de lucro (1,5%, 3% e 5%)

## Referência
O estudo usa o TCC v1.23.0, com base congelada e checkpoint de US$ 587.165.941,13. Os US$ 76,71 milhões pertencem a uma configuração do sistema externo e NÃO são a referência deste experimento. O capital inicial é US$ 10 mil.

As quatro curvas usam os mesmos modelos LightGBM, fold, margem de troca, universo e dados do Control. A única diferença é o encerramento de posições ao alcançar uma das metas de lucro. Os sinais são recalculados com o estado efetivo da carteira depois de cada saída.

## Execução
No fechamento D, o modelo escolhe manter, rotacionar ou ficar em CASH. As compras programadas são executadas na abertura D+1. A posição dispõe de uma ordem limite hipotética de venda a preço de compra multiplicado por (1 + alvo). A ordem pode disparar no próprio dia da compra. Se o pregão abre acima do limite, o preço de execução é a abertura; caso contrário, se a máxima atinge o limite, o preço é o limite. As taxas e deslizamento seguem a configuração original.

Depois do alvo, a posição fica em CASH até o encerramento do dia. No próximo pregão a política pode comprar o mesmo ativo, outro ativo, ou continuar em CASH. Sem disparo do alvo, as regras originais permanecem, inclusive permanência mínima, escolha por utilidade e troca.

CASH tem rendimento zero. Não modelamos liquidez, spread, fila de execução ou execução garantida a partir de máximas diárias. Os resultados são exploratórios porque os alvos foram escolhidos após a observação dos dados históricos. Ganhar 3% ou 5% em todos os pregões, usado na conta teórica, é muito diferente de realizar lucro de 3% ou 5% quando o mercado permitir.

## Como executar
1. Selecione a branch research/take-profit-targets-015-03-05-v1.
2. Garanta os CSVs congelados em dados/u67 e dados/.u67_expansion_cache/full_raw_20261006. A segunda pasta não é versionada no Git.
3. Instale as dependências de requirements.txt, reinicie o kernel Spyder.
4. Execute pesquisar_take_profit_alvos_spyder.py com F5 ou pelo comando: python pesquisar_take_profit_alvos_spyder.py.

A rotina original verifica os hashes SHA-256 antes de treinar. O script para se o controle nativo divergir de US$ 587.165.941,13 por mais de um centavo ou se o replay sem alvos divergir da curva diária nativa em mais de dois centavos. Não faz download de dados.

## Saídas
Na pasta output/pesquisa_take_profit_alvos:
- reproducao_controle/: resultados da reprodução oficial em pasta isolada
- control_patrimonio.csv e control_operacoes.csv
- tp_015_patrimonio.csv e tp_015_operacoes.csv
- tp_030_patrimonio.csv e tp_030_operacoes.csv
- tp_050_patrimonio.csv e tp_050_operacoes.csv
- comparacao.csv e resultado.json: capital, drawdown, Sharpe, pregões em CASH, custo e registros por fold
- comparacao_patrimonio.png e comparacao_drawdown.png

## Testes
Execute: python -m pytest tests/test_take_profit_alvos.py -q

O script só pode gerar resultados financeiros reais no ambiente onde o histórico congelado completo estiver disponível. Não modifica a main, a engine científica entregue nem o sistema externo em produção.
