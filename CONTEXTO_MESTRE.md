# CONTEXTO_MESTRE

## Baseline preservado

A reproducao oficial permanece Control vs Soft Horizon Consensus, com dados
congelados em `dados/pesquisa/`. A pesquisa Directional Change nao altera o
Control oficial nem o MCT.

## Regra de continuidade da pesquisa

Esta linha de pesquisa evolui exclusivamente nesta unica branch de pesquisa:
`research/reversal-bocpd-comparison`.
A `main` permanece intocada durante a pesquisa; o historico fica nos commits
desta branch.

Evoluir sempre os mesmos arquivos:

- `pesquisas/directional_change_lightgbm.py`;
- `pesquisar_directional_change_spyder.py`;
- `tests/test_directional_change_lightgbm.py`.

Nao criar arquivos ou branches novos para representar cada tentativa. O
historico e as diferencas entre tentativas ficam nos commits do Git.

Os resultados locais sao sobrescritos em `output/directional_change/`. O
arquivo `pacote_analise.zip` e recriado ao final de cada execucao e e o unico
pacote necessario para analise.

## Resultado da primeira tentativa Directional Change

A primeira formulacao nao foi aprovada:

- Control: US$ 10.082.425,91;
- Directional Change: US$ 3.653.966,86;
- delta relativo: -63,76%;
- distancia mediana do topo: 2,3844% -> 2,3437%;
- captura mediana do topo: 32,76% -> 26,53%;
- 52 gatilhos adicionais;
- balanced accuracy de calibracao aproximadamente 50,5% a 53,0%;
- precision aproximadamente 20% a 30%.

Diagnostico: o alvo generico de drawdown em cinco sessoes gerava falsos
positivos e saidas curtas para CASH seguidas de reentrada.

## Estado atual da pesquisa

A implementacao corrente reformula o alvo como um evento de virada perto do
topo:

- tendencia de alta nas escalas Directional Change;
- preco proximo da maxima recente;
- retorno de 20 sessoes positivo;
- primeiro evento futuro: queda relevante antes de nova continuacao da alta;
- calibracao orientada a precision com F0.5;
- duas confirmacoes consecutivas antes da saida.

O futuro e usado somente na construcao do label de treino. As features e as
decisoes OOS permanecem causais.

Ao final do processamento no Spyder:

1. os CSV/JSON correntes sao atualizados;
2. `pacote_analise.zip` e recriado;
3. um aviso sonoro e emitido.

## Resultado Top-Turn reproduzido

A implementacao Top-Turn terminou com:

- Control: US$ 10.082.425,91;
- Top-Turn: US$ 12.486.768,00;
- delta: +US$ 2.404.342,09;
- vantagem relativa: +23,8469%;
- Sharpe: 2,1011 -> 2,1565;
- MaxDD praticamente inalterado;
- 10 intervencoes Top-Turn em 1.547 sessoes.

O ganho ficou concentrado principalmente no fold 2 e em poucas intervencoes,
especialmente TSLA. Por isso a pesquisa atual nao altera o modelo: ela mede a
robustez por ablation leave-one-trigger-out.

## Pesquisa corrente — ablation

A versao corrente executa o mesmo treinamento Top-Turn uma unica vez e depois
faz replays OOS sem retreinar:

- remove cada gatilho original individualmente;
- mede o capital final sem aquele gatilho;
- calcula a contribuicao marginal de cada intervencao;
- remove em conjunto todos os gatilhos de cada ativo;
- remove em conjunto todos os gatilhos de cada fold;
- executa um replay com todos os gatilhos desabilitados para verificar paridade
  com o Control.

O objetivo da ablation agrupada e verificar se a vantagem de +23,8469% depende
principalmente de um ativo especifico, especialmente TSLA, ou de um unico fold.

Correcao metodologica: a ablation agrupada usa supressao de escopo completo.
Ao testar um ativo, qualquer gatilho Top-Turn desse ativo e bloqueado durante
todo o OOS. Ao testar um fold, qualquer gatilho Top-Turn daquele fold e
bloqueado. Nao se limita mais apenas as datas dos gatilhos originais.

Os resultados entram no mesmo `comparison_directional_change.json` e,
portanto, no mesmo `pacote_analise.zip`. Nenhum arquivo de pesquisa adicional
e criado e a `main` permanece intocada.


## Fechamento da linha Top-Turn

Checkpoint final validado antes de encerrar esta branch de pesquisa:

- research_version: 1.3.0-dev.5;
- Control: US$ 10.082.425,91;
- Top-Turn: US$ 12.486.768,00;
- delta: +US$ 2.404.342,09;
- vantagem relativa: +23,8469%;
- CAGR: 207,62% -> 218,50%;
- Sharpe: 2,1011 -> 2,1565;
- MaxDD: -31,2194% -> -31,2192%;
- pior fold: +275,05% -> +276,27%;
- distancia mediana do topo: 2,3844% -> 2,3514%;
- captura mediana do topo: 32,76% -> 36,43%;
- 10 gatilhos Top-Turn em 1.547 sessoes OOS.

Ablation de escopo completo:

- sem todos os gatilhos: US$ 10.082.425,91, exatamente o Control;
- sem LKFT: US$ 12.173.712,96;
- sem NFLX: US$ 12.766.219,07;
- sem NVDA: US$ 12.423.034,94;
- sem TSLA: US$ 10.167.194,50;
- sem fold 1: US$ 12.446.191,96;
- sem fold 2: US$ 10.167.194,50;
- sem fold 3: US$ 12.423.034,94.

A contribuicao agrupada de TSLA/fold 2 foi US$ 2.319.573,50, cerca de
96,5% da vantagem total observada sobre o Control. Portanto o resultado
Top-Turn e reproduzivel e supera o Control neste backtest, mas a evidencia de
generalizacao permanece limitada pela concentracao em TSLA/fold 2.

Esta branch fica encerrada como checkpoint experimental. A proxima pesquisa
deve comparar este Top-Turn congelado com uma tecnica de deteccao de mudanca
de regime diferente, sem alterar a main.


## Pesquisa ativa — Top-Turn vs BOCPD

Branch ativa: `research/reversal-bocpd-comparison`.

Base congelada desta comparacao: checkpoint Top-Turn do commit
`bf9e11338ee9a43b070d65714ca2522332efd771`.

Objetivo: comparar a descoberta Top-Turn com Bayesian Online Change Point
Detection usando o mesmo Control, snapshot congelado, folds walk-forward,
custos e semantica de execucao next-open.

BOCPD:

- usa retornos diarios padronizados causalmente pela volatilidade passada;
- acompanha a distribuicao posterior do run length;
- transforma a massa posterior em run lengths curtos em um score de mudanca
  descendente;
- usa apenas contexto proximo da maxima recente e retorno de 20 sessoes
  positivo como elegibilidade generica;
- calibra hazard lambda e threshold somente no bloco de calibracao de cada
  fold;
- exige duas confirmacoes consecutivas;
- somente antecipa uma saida para CASH quando o Control manteria a posicao.

A comparacao atual executa Control, Top-Turn congelado e BOCPD e grava todos os
resultados no mesmo `output/directional_change/pacote_analise.zip`.

Nenhum resultado BOCPD existe antes da execucao local completa.


## Resultado BOCPD e proxima tecnica

A comparacao OOS executada para BOCPD produziu:

- Control: US$ 10.082.425,91;
- Top-Turn: US$ 12.486.768,00;
- BOCPD: US$ 10.808.663,89;
- BOCPD vs Control: +7,2030%;
- BOCPD usou 6 gatilhos OOS;
- BOCPD melhorou o pior fold para aproximadamente +301,78%;
- os gatilhos BOCPD apareceram em regioes temporais proximas de varios
  gatilhos Top-Turn, mas com timing geralmente mais tardio;
- o ganho do BOCPD nao dependeu do mesmo episodio TSLA/fold 2 que concentrou
  a vantagem do Top-Turn.

A pesquisa ativa agora adiciona uma quarta tecnica, HSMM de duracao explicita,
mantendo Control, Top-Turn e BOCPD congelados como comparadores.

HSMM:

- tres estados latentes ordenados como baixa, neutro e alta;
- emissoes Gaussianas sobre retorno diario padronizado e tendencia EMA
  padronizada;
- parametros de emissao estimados somente no treino de cada fold;
- distribuicoes explicitas de duracao por estado estimadas no treino;
- filtro semi-Markov causal;
- score de reversao baseado na probabilidade anterior de alta e na massa
  posterior atual de baixa/deterioracao;
- threshold calibrado somente no bloco de calibracao;
- duas confirmacoes consecutivas antes de antecipar saida para CASH.

A execucao corrente compara:

- Control;
- Top-Turn;
- BOCPD;
- HSMM.

Ao final tambem sao gerados graficos em
`output/directional_change/graficos/`:

- `capital_comparison.png`;
- `trigger_timeline.png`;
- `peak_distance_comparison.png`;
- `peak_capture_comparison.png`;
- `triggers_<ATIVO>.png` para cada ativo com algum gatilho de reversao.

Os graficos sao sobrescritos em cada execucao e entram no mesmo
`pacote_analise.zip`.


## Pesquisa ativa — Hazard/Survival e Plots no Spyder

Versao corrente: `1.6.0-dev.1`.

A comparacao passa a executar, no mesmo universo e protocolo:

- Control;
- Top-Turn;
- BOCPD;
- HSMM;
- Hazard/Survival.

Hazard/Survival usa um modelo logistico de hazard discreto por intervalo.
A construcao pessoa-periodo trata:

- queda como evento de interesse;
- continuacao da alta como evento concorrente que censura o processo de queda;
- ausencia de evento dentro das cinco sessoes como censura a direita no fim
  da janela;
- threshold calibrado apenas no bloco de calibracao de cada fold;
- duas confirmacoes consecutivas antes de antecipar uma saida para CASH.

Correcao estrutural de CLMT:

- o snapshot congelado permanece intacto;
- o Corporate Action de 2024-07-11 registra `name_change` com o ticker CLMT
  preservado, mas CUSIP alterado de `131476103` para `131428104`;
- CLMT passa a ser excluido explicitamente do pipeline de modelagem como
  `structural_identity_change`;
- a serie nao e reconstruida nem conectada artificialmente.

Graficos:

- os PNGs continuam em `output/directional_change/graficos/`;
- antes de fechar cada Figure, o script a publica no console IPython;
- quando executado no Spyder, o backend Matplotlib e configurado como
  `inline`, permitindo navegar pelas figuras na aba Plots;
- os graficos adicionais sao `relative_equity_vs_control.png`,
  `trigger_peak_distance.png` e `trigger_peak_capture.png`;
- os graficos por ativo passam a incluir tambem Hazard/Survival.

O `pacote_analise.zip` antigo continua sendo apagado antes da criacao do
novo arquivo e o caminho permanece fixo:
`output/directional_change/pacote_analise.zip`.


## Correcao metodologica 1.6.0-dev.2

A primeira execucao Hazard/Survival de 1.6.0-dev.1 mostrou tres problemas que
nao devem ser tratados como resultado final:

- CLMT ainda apareceu na execucao local, indicando modulo de preparacao
  desatualizado/cacheado no kernel; o script agora importa a lista de exclusoes
  estruturais conhecidas e aborta imediatamente se algum ativo excluido
  continuar em `frames`;
- os graficos `trigger_peak_distance.png` e
  `trigger_peak_capture.png` casavam cada gatilho com a saida posterior
  seguinte; agora o casamento exige o mesmo ativo e o mesmo timestamp de
  execucao;
- o modelo logistico de hazard usava `class_weight="balanced"`, o que
  deslocava as probabilidades de hazard e fazia a probabilidade acumulada ficar
  quase sempre proxima de 1. A verossimilhanca do hazard agora e estimada sem
  balanceamento artificial de classes para preservar o significado
  probabilistico.

Essas alteracoes sao correcoes de metodologia/implementacao, nao tuning sobre
o resultado observado. A versao corrente e `1.6.0-dev.2`.


## Storytelling mensal/anual de ativos

Versao corrente: `1.6.0-dev.3`.

Os graficos genericos de comparacao deixam de ser o foco principal. O script
passa a gerar calendarios ano x mes orientados a narrativa:

- `story_control_monthly.png`;
- `story_top_turn_monthly.png`;
- `story_bocpd_monthly.png`;
- `story_hsmm_monthly.png`;
- `story_hazard_survival_monthly.png`;
- `market_monthly_leaders.png`;
- `market_monthly_laggards.png`;
- `asset_story_<ATIVO>.png` somente para ativos que tiveram gatilhos.

Nos calendarios de estrategia, cada celula mostra o ativo dominante no mes,
o retorno mensal da estrategia e a porcentagem de sessoes em que esse ativo
foi dominante.

Nos calendarios individuais de ativo, cada celula mostra o retorno mensal do
ativo e marcadores dos sinais observados naquele mes:
`TT` Top-Turn, `BO` BOCPD, `HS` HSMM e `HZ` Hazard/Survival.

Todos os calendarios usam somente a janela OOS para manter a narrativa
comparavel com os resultados da estrategia. As figuras continuam sendo salvas
em `output/directional_change/graficos/`, publicadas na aba Plots do Spyder e
incluidas no mesmo `pacote_analise.zip`.


## Checkpoint Top-Turn e inicio da pesquisa Bottom-Turn — v1.7.0-dev.1

A pesquisa passa a focar o ciclo completo de entrada e saida. BOCPD, HSMM e
Hazard/Survival permanecem preservados no historico e na documentacao, mas nao
sao executados na campanha ativa de Bottom-Turn.

### O Top-Turn atual se enquadra principalmente em tres camadas

| Camada | Familia / tecnologia | Papel no Top-Turn |
| --- | --- | --- |
| Estrutura de mercado | Directional Change (DC) | Transforma movimentos de preco em eventos/regimes de alta, baixa, overshoot e reversao. |
| Previsao | Supervised Machine Learning | Classifica se uma reversao para baixo tende a acontecer antes da continuacao da alta. |
| Modelo | LightGBM / Gradient Boosted Decision Trees | Aprende a probabilidade do evento first-passage a partir das features de mercado e Directional Change. |
| Execucao | Hybrid / Rule-based decision overlay | Exige contexto, threshold e duas confirmacoes; so antecipa a saida quando o Control manteria a posicao. |

Descricao curta oficial:

`Top-Turn = Directional Change + supervised ML com LightGBM + regras causais de confirmacao e execucao.`

O alvo Top-Turn e first-passage com eventos concorrentes: perto de uma maxima,
pergunta se uma queda relevante acontece antes de uma continuacao relevante da
alta dentro de cinco sessoes.

### Familias de metodos ja pesquisadas para reversao de topo

1. Event-based / Directional Change:
   Directional Change + LightGBM -> Top-Turn.
2. Bayesian Change Point:
   BOCPD.
3. Latent Regime Models:
   HSMM de duracao explicita.
4. Survival / Event History:
   Hazard/Survival discreto.
5. Deep Learning:
   nao faz parte da linha ativa; nao ha rede neural/neuronios no experimento
   corrente.

### Ultimo checkpoint OOS valido antes do Bottom-Turn

Execucao `1.6.0-dev.3`, com DOC e CLMT excluidos estruturalmente:

| Metodo | Capital final | vs. Control | Sharpe | MaxDD |
| --- | ---: | ---: | ---: | ---: |
| Control | US$ 5.092.399,32 | - | 1,9287 | -36,65% |
| Top-Turn | US$ 6.306.816,02 | +23,85% | 1,9844 | -36,65% |
| BOCPD | US$ 5.333.893,64 | +4,74% | 1,9431 | -36,65% |
| HSMM | US$ 4.614.917,81 | -9,38% | 1,9064 | -37,23% |
| Hazard/Survival | US$ 3.192.675,93 | -37,31% | 1,8201 | -36,53% |

Top-Turn continua sendo a referencia de topo. A remocao estrutural de CLMT
alterou o caminho absoluto da carteira, mas a vantagem relativa do Top-Turn
permaneceu aproximadamente +23,85%.

### Pesquisa ativa: Bottom-Turn

Bottom-Turn e a contraparte simetrica do Top-Turn:

- familia principal: Directional Change + LightGBM;
- horizonte inicial: 5 sessoes;
- threshold adaptativo: 1,5 x ATR, limitado entre 2% e 8%;
- elegibilidade: maioria dos regimes DC em baixa, preco a ate 5% da minima de
  20 sessoes e retorno de 20 sessoes negativo;
- alvo positivo: recuperacao relevante ocorre antes de nova continuacao da
  queda;
- alvo negativo: nova continuacao da queda ocorre primeiro, ou a recuperacao
  nao vence dentro da janela;
- duas confirmacoes consecutivas;
- atua somente em CASH -> ativo;
- o ativo continua sendo escolhido pela politica Control/LightGBM;
- Bottom-Turn pode apenas confirmar ou atrasar a entrada;
- nao altera rotacoes ativo -> ativo nem posicoes ja abertas.

A campanha ativa compara quatro cenarios:

| Cenario | Entrada | Saida | Objetivo |
| --- | --- | --- | --- |
| Control | Control | Control | baseline |
| Bottom-Turn | Bottom-Turn | Control | medir apenas melhoria de compra |
| Top-Turn | Control | Top-Turn | referencia de melhoria de venda |
| Top+Bottom | Bottom-Turn | Top-Turn | medir o ciclo completo |

### Metricas de fundo

A pesquisa passa a exportar `bottom_entry_<cenario>.csv` com:

- `bottom_price_before_entry`;
- `bottom_timestamp_before_entry`;
- `entry_distance_from_bottom_pct`;
- `bottom_capture_pct`;
- `days_from_bottom_to_entry`;
- `post_entry_return_5d_pct`, `10d`, `20d`;
- `continued_drawdown_5d_pct`, `10d`, `20d`.

A distancia do fundo segue a mesma filosofia da distancia do topo: quanto menor
a distancia de entrada em relacao ao fundo observado durante o periodo em
CASH, melhor o timing. Gap de entrada abaixo do fundo anterior e tratado como
distancia zero.

### Storytelling visual do ciclo

Os calendarios mensais permanecem no Spyder/ZIP. Nos graficos individuais:

- `TT↓` = saida Top-Turn;
- `BT↑` = entrada Bottom-Turn.

O objetivo visual e acompanhar:
`queda -> BT↑ -> recuperacao -> alta -> TT↓ -> queda`.

### Governanca

- Branch unica ativa: `research/reversal-bocpd-comparison`.
- `main` permanece intocada.
- Mesmos arquivos de pesquisa; historico fica nos commits.
- Snapshot congelado; sem Alpaca durante tuning.
- DOC e CLMT continuam excluidos estruturalmente.
- Nenhum resultado Bottom-Turn existe antes do replay OOS completo.
- O ZIP antigo e apagado e recriado sempre em
  `output/directional_change/pacote_analise.zip`.
