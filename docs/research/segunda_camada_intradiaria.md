# Pesquisa do TCC: segunda camada de ML para permanencia intradiaria

## Objetivo

Avaliar uma **camada 2** de permanencia sobre o ativo previamente escolhido
pela Winner (camada 1). Universo da Winner: **77 ativos solicitados**,
dos quais **75 foram efetivos no Control historico congelado**.

Nao refazer o ranking intradiario, nao trocar diretamente para outro ativo e
nao alterar Winner Live, sistema operacional de producao ou main.

Esta branch nasce da branch
`research/permanencia-minima-contrafactual-v1`, preservando o Control.
Os testes da camada 2 sao **exploratorios** e independentes das ordens reais.

## Metodos no comparador

| Nome no script | Implementacao | Limite |
| --- | --- | --- |
| `multirrolling` | t de regressao em janelas de 6, 12 e 24 candles, combinado sem consulta ao futuro | Determina direcao/forca, nao a utilidade liquida de uma ordem |
| `bocpd` | Filtro bayesiano online Normal-Normal com hazard predefinido | Hipotese gaussiana e forte para eventos extremos |
| `hmm` | Duas emissoes GaussianMixture, transicoes estimadas no treino e filtro forward | Aproximacao HMM de dois estados; nao e Markov-switching completo |
| `lightgbm` | Classificador causal de retorno futuro liquido de custo hipotetico em 30 minutos | Requer amostra historica ampla |
| `pca_lightgbm` | PCA de 15 indicadores com LightGBM | Proxy de reducao de dimensionalidade, NAO e o PVB-FPCA profundo do artigo |
| `fitted_q` | Fitted Q para optimal stopping com CASH terminal e HOLD ate proxima hora | Piloto de valor de permanencia, nao um robô RL de producao |
| `ofi_l1` | **Nao avaliado** se nao houver quotes L1 sincronizados | OHLCV e volume de candles nao representam OFI do livro de ofertas |

Assim, **seis tecnicas executaveis** compartilham o mesmo pipeline de dados,
e **OFI** e explicitamente registrada como NAO_AVALIAVEL por insuficiencia
de dados, nunca simulada com informacao inventada.

## Obtencao automatica do historico no computador local

O script principal prepara automaticamente `dados_intraday/winner_positions.csv`
a partir da selecao congelada do Control original, incluida no proprio script.
Sao 1.560 sessoes, de 2020-07-22 ate 2026-10-06. A copia foi extraida
do `comparacao_diaria_ativos_e_patrimonio.csv` da pesquisa anterior,
sem reconstruir sinais nem baixar cotacoes diarias.

**Nao e necessario copiar o ZIP anterior para a pasta do projeto.**
Caso `pesquisa_permanencia_minima.zip` esteja disponivel, o leitor ainda
aceita o arquivo original. A origem e o SHA-256 ficam registrados na saida.

Ao pressionar **F5**, o script procura o repositorio da API operacional na
estrutura habitual de pastas do computador. Ele **importa somente o leitor
historico RAW/SIP 5Min de acoes** e utiliza a configuracao de acesso Alpaca
ja instalada com a API. Nao e necessario iniciar um segundo treino, nao
existe endpoint de ordem no TCC e nenhuma operacao Live e enviada.

**Importante:** isto usa a biblioteca de dados da API local, nao seu endpoint
HTTP de grafico: o endpoint de grafico apresenta somente precos recentes e
nao entrega o OHLCV necessario para o treinamento. O codigo do servico
operacional deve estar acessivel localmente com suas dependencias e suas
credenciais historicas ja configuradas. Nenhuma credencial e salva no ZIP.

Somente os pares de ativo e mes em que o Control esteve comprado sao
consultados. O resultado fica no cache privado
`dados_intraday/cotacoes_5min/<ATIVO>_<AAAA-MM>.csv`.
Na proxima execucao com F5, os CSVs validados sao reutilizados.
`FORCAR_DOWNLOAD = True` permite atualizar o cache, se necessario.

A primeira execucao pode exigir muitas consultas historicas. Falhas de
permissao, rate limit, suspensoes ou falta de barras nao geram cotacoes
sinteticas. O ZIP inclui `aquisicao`, `origem_control` e `cobertura`,
e os indicadores sao identificados como **parciais** ate completar as
sessoes da carteira. Nenhum resultado de patrimonio final e inferido sem
um replay economico completo.

**Reprodutibilidade:** a selecao vem exclusivamente do snapshot congelado,
nao do sinal atual do mercado. O modelo continua em uma branch de pesquisa,
sem modificacoes no motor ou no ambiente operacional.

## Executar no Spyder

Abrir `pesquisar_segunda_camada_intradiaria_spyder.py` e pressionar F5.

O script primeiro executa testes sinteticos de integridade/causalidade.
Se faltarem candles reais, produz somente `status.json` informando a
ausencia de resultados; **nao cria AUC nem patrimonio inventados**.

Para avaliar dados historicos, fornecer CSVs em:

```text
dados_intraday/
    CAPR_2026-07.csv
    ADEA_2022-10.csv
    ...
    winner_positions.csv
```

Formato de candles (timestamp absoluto em UTC; inicio do candle):

```csv
timestamp,symbol,open,high,low,close,volume
2026-07-28T13:30:00+00:00,CAPR,6.78,7.22,6.68,6.90,524641
```

Pode conter candles de **um minuto**, que serao agregados em grupos
completos de cinco minutos, ou candles de cinco minutos (ja alinhados).

`winner_positions.csv`:

```csv
session,symbol
2026-07-28,CAPR
```

Uma e somente uma posicao por sessao, correspondente a posicao efetiva
do Control naquele dia. O script filtra por essa selecao. Sem este CSV,
o teste e apenas **metodologico**, nao uma politica condicionada a Winner.

Somente acoes que o Control realmente manteve/comprou precisam de
candles intradiarios. **Nao publicar dados brutos SIP licenciados no
repositorio publico.** Manter os CSVs em armazenamento local, com SHA-256
documentado. Nao inserir API keys nem arquivos de ambiente no script.

## Relatorios e limites

A saida sobrescreve atomicamente apenas:

`output/pesquisa_vantagem_rotacao.zip`

Contem, conforme a cobertura de dados:
- `status.json` com a origem e limites da amostra;
- `comparacao_metodos.csv`, com AUC, acuracia balanceada e Brier;
- `previsoes_horarias.csv` com o score causal e retorno futuro **somente
  para auditoria retrospectiva**, nunca como entrada de uma decisao;
- `codigo_executado.py` e `arquivos_sha256.json`.

**Importante:** AUC de direcao a 30 min nao e uma medida de ganho nem
garante boa deteccao de reversoes. Para validar melhora da permanencia,
falta uma etapa de replay de HOLD/CASH, limites de liquidez,
custos, atrasos de dados, gaps, halts e caixa na abertura seguinte.
O capital do Control de **US$ 587.165.941,13** nao e recalculado por
esse script. Nenhum patrimonio da camada 2 deve ser divulgado ate
existir replay integral e causal.

## Causalidade e validacao

- Os horarios de avaliacao sao 10h30 ate 15h30 em Nova York.
- Cada decisao usa o candle de 5 min iniciado cinco minutos antes, ja
  completo, e somente features anteriores.
- O alvo de 30 minutos futuros e utilizado **apenas no treinamento e
  avaliacao retrospectiva**, nunca para construir indicadores no instante.
- Janelas reiniciadas por ativo/sessao, evitando mistura de splits/gaps
  entre sessoes nas features.
- Separacao de sessoes inteiras em treino, validacao e teste, com expurgo
  de uma sessao em cada fronteira. A divisao 60/20/20 e provisoria;
  no replay final, substituir pelos **folds congelados do TCC**, com
  os mesmos pontos de corte e a mesma elegibilidade por ativo.
- A selecao Winner deve ser obtida da auditoria historica congelada,
  nunca prevista a posteriori.
- O Fold 3 historico, especialmente CAPR, **nao e um holdout totalmente
  inedito** apos os experimentos que usaram seus resultados para
  formular esta pesquisa.
- O desempenho por ativo, fold, regime, custos e capital inicial
  precisa ser apresentado antes de qualquer proposta de adocao.

## Testes executados nesta entrega

1. Criacao e verificacao de 78 candles sintéticos de cinco minutos
   numa unica sessao, com seis horarios de decisao.
2. Verificacao de que adicionar dados de uma sessao seguinte nao altera
   os indicadores da sessao anterior.
3. Rejeicao de timestamps duplicados.
4. Smoke test local com 18 sessoes **sinteticas**: os seis metodos
   produziram scores e metricas nos mesmos pontos horarios.
   **Nao interpretar as metricas desse smoke test como mercado.**

Ainda nao foi possivel executar uma comparacao cientifica intradiaria
dos 77 ativos nem recalcular patrimonio, pois o ZIP original do TCC
contem series **diarias**; a amostra de 18 avaliacoes de CAPR, ADEA e
YANG nao e suficiente para treinar/validar todos os modelos.

## Referencias metodologicas

- Bramante, Facchinetti e Zappa (2019): *Online Detection of Financial Time Series Peaks and Troughs: A Probability-Based Approach*.
- Grillenzoni (2014): *Sequential Smoothing for Turning Point Detection with Application to Financial Decisions*.
- Adams e MacKay (2007): *Bayesian Online Changepoint Detection*.
- Hamilton (1989): *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle*.
- Cont, Kukanov e Stoikov (2014): *The Price Impact of Order Book Events*.
- Sutton e Barto (2018): *Reinforcement Learning: An Introduction*.

As implementacoes aqui sao **prototipos**, nao reproducoes numericas
exatas dos artigos.
