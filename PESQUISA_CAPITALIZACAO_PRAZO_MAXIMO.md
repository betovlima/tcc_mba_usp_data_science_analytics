# Pesquisa: capitalização com prazo máximo de permanência

## Objetivo

Reinvestir integralmente o capital em cada nova compra e realizar lucro quando
a posição alcançar **1,5%, 3% ou 5%**. Caso não alcance, encerrar depois de
**5, 10 ou 20 pregões completos** e aguardar a próxima proposta da engine.
Não realizar rotação apenas porque surgiu um ativo com ranking melhor,
enquanto o prazo da posição atual não tiver terminado.

O mesmo ativo pode voltar a ser comprado se for a melhor proposta no
fechamento posterior à venda. O patrimônio diário inclui perdas não
realizadas, e a ordem de venda por prazo pode realizar prejuízo.

## Calendário e execução sem informação futura

- A primeira sessão investida é a data de compra, contada como **1**.
- Enquanto posicionado, mantemos uma ordem limite hipotética para realizar lucro.
- Se a máxima de uma sessão atinge o alvo, o replay vende no nível limite.
  Se a abertura supera o limite, vende na abertura.
- Se após N sessões completas o alvo não tiver sido atingido, **vende na
  abertura da sessão N+1**, ao preço real de abertura disponível no histórico.
  Se essa abertura já superar o alvo, a venda é registrada como lucro-alvo.
- Não se usa a máxima ou a mínima da sessão N+1 para decidir adiar a venda por
  prazo. Não há recompra intradiária.
- A decisão de nova entrada usa o fechamento da sessão de venda e pode ser
  executada na abertura da sessão seguinte.
- A engine só propõe ativos quando está em CASH. Quando está posicionada,
  mudanças de ranking não provocam rotatividade.
- Sem alvo ou prazo cumprido até o fim da amostra, liquida-se no último
  fechamento apenas para calcular o valor econômico e os custos.
- Uma venda por tempo pode realizar perda muito maior do que o esperado em
  razão de gaps. O prazo máximo não é um stop de perda.

## Cenários

A análise produz **16 resultados**, para comparação homogênea:

- 1 controle original com rotações;
- 3 alvos com rotações originais;
- 3 alvos sem prazo, mantendo indefinidamente até atingir o alvo;
- 9 combinações de alvo {1,5%, 3%, 5%} × prazo {5, 10, 20 sessões}.

São usados os **mesmos modelos LightGBM**, os três folds, os dados históricos
congelados e o capital inicial de US$ 10 mil. Nenhuma estratégia é escolhida
ou calibrada com base nos resultados deste novo grid.

A reprodução oficial do TCC precisa validar primeiro seu checkpoint de
**US$ 587.165.941,13**, com tolerância de US$ 0,01, e o replay Control precisa
reproduzir a curva diária com erro máximo de US$ 0,02. Sem isso, a pesquisa
interrompe antes de publicar resultados.

O experimento é exploratório; esses prazos foram propostos após o estudo do
histórico e não constituem avaliação independente fora da amostra. OHLC
diário não demonstra execução real para grandes volumes, nem considera
impacto de mercado, ordens parcialmente executadas e fila de liquidez.

## Executar

No terminal da raiz do repositório:

```bash
git fetch origin
git switch research/compound-target-max-holding-v1
python -m pytest tests/test_capitalizacao_prazo_maximo.py -q
python pesquisar_capitalizacao_prazo_maximo_spyder.py
```

No Spyder, abra `pesquisar_capitalizacao_prazo_maximo_spyder.py`, reinicie o
kernel e execute com F5.

É necessário ter o snapshot já congelado em `dados/u67` e o cache dos
candidatos em `dados/.u67_expansion_cache/full_raw_20261006`. O programa
verifica os hashes antes de treinar, não baixa dados e não modifica a main.

## Saídas

Na pasta `output/pesquisa_capitalizacao_prazo_maximo/`:

- `comparacao.csv` e `resultado.json`;
- `matriz_capital_por_alvo_e_prazo.csv`;
- `matriz_capital.png` com patrimônio final em cada combinação;
- gráficos de patrimônio e drawdown com todos os cenários;
- gráficos por alvo: `comparacao_capital_tp_015.png` e equivalentes para 3%
  e 5%, também para drawdown;
- CSVs com operações e patrimônio diário de cada cenário;
- `reproducao_controle/` com a reprodução original validada.

Métricas incluem capital final, Sharpe, drawdown máximo, tempo em CASH,
saídas por lucro, saídas por prazo, quantas saídas por prazo tiveram prejuízo,
duração máxima das operações e taxas de transação.
