# Pesquisa: confirmacao de reentrada e protecao por posicao

## Motivacao

Na pesquisa anterior o Control terminou em US$ 587.165.941,13, enquanto
realizacoes de lucro em 1,5%, 3% e 5% terminaram com patrimonio menor.
No episodio CAPR de julho de 2026, a compra do mesmo ticker no pregao
seguinte a venda por lucro foi uma causa observada para a continuidade
da exposicao a uma queda forte.

Esta nova pesquisa isola duas hipoteses:
1. **Confirmacao antes de recomprar o mesmo ativo**: apos a saida forcada
   por lucro ou stop, exigir que o mesmo ticker seja a melhor proposta
   valida em **dois fechamentos consecutivos** antes de comprá-lo. Qualquer
   outro ticker que seja a melhor proposta pode ser comprado normalmente.
   A confirmacao so usa dados conhecidos no fechamento da decisao.
2. **Stop independente de utilidade**: sair quando a perda no preco da
   posicao atingir **10%**, mesmo que o modelo continue classificando
   aquele ticker como o melhor e mesmo durante a permanencia minima.

O objetivo e diagnosticar vulnerabilidade a reentradas e a grandes quedas.
Nao afirmar que esses valores de 2 e 10% sao otimos. O experimento usa
limiares fixados antes da simulacao de seu resultado.

## Cenarios e identificacao

A rotina `pesquisar_protecao_reentrada_spyder.py` compara:
- control original, sem TP e sem stop;
- tp_015, tp_030, tp_050, identicos aos tres cenarios anteriores;
- para cada TP: sufixo `_confirm2`, `_stop10` e `_confirm2_stop10`;
- `stop10_only`, sem realizacao de lucro, apenas stop.

Sao **14 cenarios**. Mesmo universo e mesmas previsoes do LightGBM,
mesmas datas e folds, mesmo saldo inicial e taxas, com a simulacao
financeira recalculada desde o primeiro pregao para cada carteira.

## Regras de execucao e limitacoes

O sinal do fechamento D decide a ordem para a abertura D+1.
O limite de venda por lucro ou protecao permanece ativo durante o pregao.
Se a abertura estiver alem do stop, a venda e contabilizada na abertura,
podendo realizar perda acima de 10%. Se a minima do dia atingir o stop,
contabiliza-se venda pelo nivel. Se maxima e minima atingirem os dois
limites no mesmo dia, o replay usa o **stop primeiro**, conservador,
e conta a barra como ambigua.

No dia de venda por alvo ou stop a carteira fica em CASH e nao compra
novamente no mesmo pregao. A estrategia reavalia no fechamento.
Sem protecao disparada valem as rotacoes e a permanencia minima normais.
Os limites nao modelam liquidez, spread, slippage de mercado ou
ordem parcial. O estudo so e valido como simulacao historica de
execucao de ordens com OHLC diario.

A confirmacao de dois fechamentos **nao e novo modelo** e nao verifica
se o score melhorou. Ela exige persistencia da proposta positiva de
maior utilidade. Para evitar otimizar no proprio teste, nao selecionar
um vencedor apenas pelo lucro do historico ja examinado.

## Como executar

No terminal do TCC, sobre a branch desta pesquisa:

    git fetch origin
    git switch research/tp-reentry-protection-v1
    python -m pytest tests/test_protecao_reentrada.py -q
    python pesquisar_protecao_reentrada_spyder.py

Tambem e possivel abrir o script no Spyder, reiniciar o kernel e
executar com F5.

O ambiente precisa dos dados congelados, incluindo
`dados/.u67_expansion_cache/full_raw_20261006`, que nao estao
versionados. O script valida os hashes, reproduz o Control
e verifica a curva diaria antes de gerar as comparacoes.
Nao faz novos downloads nem altera a estrategia principal.

## Saida

`output/pesquisa_protecao_reentrada/` inclui:
- `comparacao.csv`, `resultado.json`;
- operacoes e patrimonio, por cenario;
- `comparacao_patrimonio.png`, `comparacao_drawdown.png`;
- figuras separadas para cada alvo para evitar linhas demais;
- logs de venda por stop, TP e bloqueios de reentrada;
- resultados por fold e episodios de perda no historico.

A `main` e o TCC entregue permanecem inalterados.
