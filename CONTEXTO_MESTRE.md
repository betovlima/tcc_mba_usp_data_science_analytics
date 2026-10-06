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


### Correcao do sinal sonoro — v1.7.0-dev.2

O sinal de conclusao anterior usava `winsound.Beep()` e retornava assim que a
chamada terminava sem excecao. Em alguns ambientes Windows/Spyder essa chamada
pode ser aceita sem produzir som audivel no dispositivo configurado.

A funcao `sinal_sonoro_conclusao()` passa a tentar, em sequencia:

1. `winsound.PlaySound("SystemExclamation", SND_ALIAS | SND_SYNC)`;
2. `winsound.MessageBeep(MB_ICONASTERISK)`;
3. os dois tons `winsound.Beep()` anteriores;
4. bell do terminal somente se nenhum mecanismo Windows estiver disponivel.

O script imprime `[sound] completion mechanisms=...` no final para registrar
quais mecanismos foram executados. Essa alteracao nao muda modelos, dados,
folds, sinais ou resultados do backtest.


## Resultado Bottom-Turn v1 — execucao OOS 1.7.0-dev.1

Pacote validado com `execution_schema=top-bottom-cycle-v1`, 1.547 sessoes OOS,
DOC e CLMT fora do universo estrutural.

### Comparacao dos quatro cenarios

| Cenario | Capital final | vs Control | CAGR | Sharpe | MaxDD | Worst fold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | US$ 5.092.399,32 | - | 175,31% | 1,9287 | -36,65% | +124,00% |
| Bottom-Turn | US$ 5.729.695,93 | +12,51% | 180,63% | 1,9624 | -36,65% | +152,01% |
| Top-Turn | US$ 6.306.816,02 | +23,85% | 185,04% | 1,9844 | -36,65% | +124,72% |
| Top+Bottom | US$ 5.327.682,44 | +4,62% | 177,33% | 2,0239 | -31,22% | +160,49% |

Top-Turn permanece com o maior capital final. Top+Bottom melhora Sharpe,
drawdown e pior fold, mas termina 15,53% abaixo do Top-Turn em capital.

### Evidencia Bottom-Turn puro

Bottom-Turn puro gerou somente 1 entrada confirmada e 13 bloqueios. O unico
gatilho ocorreu em CXW em 2020-08-10, depois de 13 sessoes uteis em CASH desde
o inicio OOS. A entrada ficou 3,29% acima do fundo observado, dois dias uteis
depois dele. Retornos posteriores: +14,20% em 5 sessoes, +9,77% em 10 e +7,95%
em 20; drawdown posterior maximo medido nessas janelas foi -0,80%.

A vantagem de +12,51% sobre Control nao representa generalizacao por varios
fundos: praticamente toda a diferenca nasce desse unico atraso da implantacao
inicial no fold 1. Os folds 2 e 3 mantem retornos praticamente iguais ao
Control.

### Evidencia Top+Bottom

No ciclo combinado houve 7 entradas Bottom-Turn, 318 bloqueios e 6 saidas
Top-Turn. Entradas confirmadas:

- CXW 2020-08-10;
- LKFT 2020-10-20;
- XSD 2021-05-14;
- VRTS 2022-05-04;
- LKFT 2023-08-24;
- MYE 2024-08-13;
- LLY 2025-08-01.

Tempo aproximado em CASH antes dessas entradas: 13, 5, 14, 208, 51, 32 e
10 dias uteis, respectivamente. O caso VRTS permaneceu cerca de 208 dias uteis
em CASH apos a saida NFLX de 2021-07-16, mostrando que o gate obrigatorio pode
bloquear a politica por tempo excessivo.

Metricas medianas das 7 entradas Top+Bottom:

- distancia do fundo: 3,20%;
- captura do fundo: 85,56%;
- dias do fundo ate entrada: 2;
- retorno +5d: +1,75%;
- retorno +10d: -0,54%;
- retorno +20d: -0,89%;
- drawdown posterior +5d: -1,55%;
- drawdown posterior +20d: -6,42%.

O caso LKFT de outubro/2020 foi especialmente util: Top-Turn sozinho recomprou
em 2020-10-14 a 145,59 e sofreu -15,45% nos 5 dias seguintes; Bottom-Turn
esperou ate 2020-10-20 e recomprou a 132,11, aproximadamente 9,26% abaixo,
reduzindo o drawdown imediato. Isso mostra que existe sinal util em pelo menos
alguns episodios.

### Calibracao Bottom-Turn

Thresholds por fold: 0,60 / 0,55 / 0,55. Balanced accuracy ficou entre
aproximadamente 50,5% e 51,4%, precision entre 23,6% e 29,0% e recall entre
14,4% e 28,4%. Portanto, a classificacao global ainda e fraca; o valor potencial
vem de poucos eventos economicamente relevantes, como ocorreu com Top-Turn.

### Conclusao metodologica do v1

Bottom-Turn v1 ainda nao pode ser considerado detector de fundos validado.
Ha exemplos promissores, mas o desenho como veto obrigatorio de toda entrada
CASH -> ativo e excessivamente restritivo. O ganho do Bottom-Turn puro depende
de um unico evento inicial e o ciclo Top+Bottom sacrifica capital por longos
periodos em CASH.

Top-Turn continua congelado como referencia de topo. A proxima formulacao de
Bottom-Turn deve preservar o objetivo de timing de fundo sem permitir bloqueios
indefinidos da politica-base. Qualquer nova regra deve ser definida
conceitualmente antes do replay e nao ajustada para perseguir este resultado.


## Checkpoint historico de capital e campanha Bottom-Turn v2 — v1.8.0-dev.1

### Como registrar no TCC os US$ 12,49 milhoes

O resultado historico de Top-Turn de aproximadamente US$ 12.486.768,00 deve
ser preservado no relato cientifico, mas sempre identificado como pertencente
ao universo anterior a exclusao estrutural de CLMT.

Checkpoint historico pre-exclusao CLMT:

| Cenario | Capital final | Top-Turn vs Control |
| --- | ---: | ---: |
| Control | US$ 10.082.425,91 | - |
| Top-Turn | US$ 12.486.768,00 | +23,85% |

Depois da identificacao da mudanca estrutural de identidade/CUSIP de CLMT e da
aplicacao da regra de exclusao estrutural do TCC, o universo corrigido passou a
produzir:

| Cenario | Capital final | Top-Turn vs Control |
| --- | ---: | ---: |
| Control | US$ 5.092.399,32 | - |
| Top-Turn | US$ 6.306.816,02 | +23,85% |

A queda no capital absoluto nao deve ser interpretada como lucro isolado de
CLMT. A estrategia e path-dependent: retirar um ativo muda rankings, rotacoes,
capital disponivel e todas as decisoes compostas posteriores. O ponto
metodologicamente importante e que a vantagem relativa do Top-Turn permaneceu
aproximadamente +23,85% mesmo apos a correcao estrutural do universo.

No texto final do TCC, os dois checkpoints podem ser usados para mostrar a
importancia da integridade do universo, da reproducibilidade e da sensibilidade
de estrategias de rotacao composta a mudancas estruturais nos dados. O baseline
cientifico corrente e sempre o universo corrigido; o resultado pre-exclusao e
historico e nao deve ser apresentado como resultado final vigente.

### Bottom-Turn v2

Versao ativa: `1.8.0-dev.1`.
Execution schema: `top-bottom-cycle-v2`.

A formulacao v1 mostrou que um veto Bottom-Turn indefinido podia manter a
carteira em CASH por centenas de sessoes. A v2 muda somente a arquitetura da
intervencao, sem alterar features, target, LightGBM, thresholds ou calibracao.

Protocolo congelado antes do replay:

1. Top-Turn permanece congelado como detector de saida.
2. Bottom-Turn so pode ser armado por uma saida efetivamente disparada pelo
   Top-Turn.
3. A janela Bottom-Turn dura no maximo 5 sessoes de decisao, igual ao horizonte
   probabilistico definido previamente.
4. Dentro da janela, duas confirmacoes Bottom-Turn permitem a reentrada no
   ativo que o Control ja escolheu.
5. Se nao houver confirmacao ate a quinta sessao, a quinta ainda pode ser
   bloqueada; a partir da sessao seguinte a decisao volta integralmente ao
   Control.
6. A entrada inicial da carteira nao e bloqueada pelo Bottom-Turn.
7. Entradas normais nao originadas de uma saida Top-Turn nao sao bloqueadas.
8. Rotacoes ativo -> ativo continuam sob a politica-base.
9. Nao ha tuning de threshold ou features com base no resultado do v1.

A campanha ativa compara apenas:

- Control;
- Top-Turn;
- Top-Turn + Bottom-Turn v2.

Bottom-Turn v1, BOCPD, HSMM e Hazard permanecem no historico e nos checkpoints,
mas nao sao executados na campanha v2.

Objetivo do experimento: verificar se o sinal de fundo consegue melhorar a
reentrada apos uma boa saida Top-Turn sem sacrificar capital por exposicao
excessiva a CASH.


### Correcao do winsound.SND_SYNC — v1.8.0-dev.2

Em Windows/Spyder foi observado:

`AttributeError: module 'winsound' has no attribute 'SND_SYNC'`.

A falha ocorria somente depois da criacao de `pacote_analise.zip`, portanto
nao invalida o backtest nem o pacote ja gerado. A causa era a suposicao de que
todas as versoes de `winsound` expunham `SND_SYNC`.

A funcao de conclusao foi tornada auxiliar e nao-fatal:

- `PlaySound` usa apenas `SND_ALIAS`; reproducao sincrona e o comportamento
  padrao quando `SND_ASYNC` nao e fornecido;
- constantes opcionais usam `getattr`;
- falhas de qualquer mecanismo de audio nao propagam excecao;
- `MessageBeep`, `Beep` e bell de terminal continuam como fallbacks;
- foi adicionado teste simulando `winsound` sem `SND_SYNC`.

Essa versao altera apenas o sinal sonoro de conclusao. Modelos, dados,
features, folds, thresholds, sinais e resultados OOS permanecem inalterados.


### Fail-fast contra pacote v1 durante campanha v2 — v1.8.0-dev.3

Foi observado novamente um pacote com `research_version=1.8.0-dev.1`, mas
`execution_schema=top-bottom-cycle-v1` e artefatos `comparison_cycle.json`,
`bottom_turn_*` e `top_bottom_*`. Isso prova que um script Spyder antigo
pode importar o modulo novo e, por isso, exibir uma versao de pesquisa nova
mesmo executando o fluxo antigo.

Para impedir nova ambiguidade, `criar_pacote_analise()` agora recusa gerar o
ZIP da campanha v2 se nao existir `comparison_cycle_v2.json` ou se o
`execution_schema` nao for exatamente `top-bottom-cycle-v2`.

Essa e uma correcao de rastreabilidade/empacotamento. Nao altera modelos,
features, thresholds, folds, sinais ou resultados OOS.


### Guard de schema antes do replay e Period mensal sem warning — v1.8.0-dev.4

Uma execucao longa chegou ao final com o modulo em campanha v2, mas o runner
Spyder ainda estava em fluxo incompatível; o fail-fast do empacotamento
impediu a criacao de um ZIP incorreto apenas no fim.

A partir de v1.8.0-dev.4 o runner importa `EXPECTED_EXECUTION_SCHEMA` do
modulo e compara com seu `EXECUTION_SCHEMA` antes da validacao do snapshot e
antes de qualquer treinamento/replay. Se houver mistura de arquivos de versoes
diferentes, a execucao para imediatamente.

Tambem foram removidos os warnings de Pandas ao converter timestamps UTC em
`Period[M]`: os timestamps sao primeiro normalizados para UTC e depois o
timezone e removido deliberadamente, pois um periodo mensal nao representa
fuso horario. Isso afeta apenas a geracao dos calendarios mensais, nao os dados,
folds, sinais ou resultados OOS.


## Resultado Bottom-Turn v2 — OOS 1.8.0-dev.4

Pacote validado com `research_version=1.8.0-dev.4`,
`execution_schema=top-bottom-cycle-v2`, 1.547 sessoes OOS e protocolo
`post_top_only=True, max_wait_sessions=5`.

### Resultado agregado

| Cenario | Capital final | vs Control | vs Top-Turn | CAGR | Sharpe | MaxDD | Worst fold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | US$ 5.092.399,32 | - | - | 175,31% | 1,9287 | -36,65% | +124,00% |
| Top-Turn | US$ 6.306.816,02 | +23,85% | - | 185,04% | 1,9844 | -36,65% | +124,72% |
| Top+Bottom v2 | US$ 5.244.421,51 | +2,99% | -16,85% | 176,62% | 1,9453 | -36,65% | +132,55% |

Bottom-Turn v2 resolveu o problema arquitetural de bloqueios indefinidos, mas
nao melhorou o Top-Turn. O capital final ficou US$ 1.062.394,51 abaixo do
Top-Turn.

### Comportamento da janela de cinco sessoes

No Top+Bottom v2 ocorreram:

- 8 saidas Top-Turn;
- 39 decisoes de entrada bloqueadas dentro das janelas;
- 7 expiracoes da janela;
- apenas 1 entrada confirmada pelo Bottom-Turn;
- somente 4 observacoes OOS com probabilidade Bottom-Turn disponivel dentro
  das janelas.

A unica confirmacao Bottom-Turn ocorreu em LKFT, outubro/2020, na quinta
sessao da janela. Portanto, o v2 se comportou majoritariamente como um cooldown
fixo de cinco sessoes, e nao como um detector de fundos capaz de liberar
reentradas de forma recorrente.

### Entrada LKFT confirmada

Top-Turn sozinho recomprou LKFT em 2020-10-14 a 145,59.
Bottom-Turn confirmou reentrada em 2020-10-20 a 132,11, aproximadamente
9,26% abaixo do preco da recompra Top-Turn. Esse episodio continua sendo uma
evidencia positiva individual para o conceito de fundo.

### Desempenho por fold versus Top-Turn

- fold 1: Top+Bottom v2 aproximadamente +3,48% relativo ao Top-Turn;
- fold 2: aproximadamente -18,02%;
- fold 3: aproximadamente -1,98%.

A principal deterioracao ocorreu no fold 2.

### Caso TSLA junho/2023

Top-Turn sozinho executou:

- 2023-06-14: venda TSLA a 260,17;
- 2023-06-15: recompra TSLA a 248,40;
- 2023-06-21: nova saida Top-Turn a 275,13;
- 2023-06-22: nova recompra a 250,77.

O Top+Bottom v2 vendeu em 2023-06-14, permaneceu em CASH durante a janela e
somente recomprou TSLA em 2023-06-23 a 259,29. Assim, perdeu a alta entre
248,40 e 275,13 e deixou de estar posicionado para o segundo gatilho Top-Turn
de 2023-06-21.

A relacao de equity Top+Bottom v2 / Top-Turn estava aproximadamente +3,48% na
saida de 2023-06-14 e caiu para cerca de -9,64% na reentrada de 2023-06-23.
Esse episodio explica parcela importante da perda do fold 2 e mostra que uma
janela de espera pode destruir uma sequencia lucrativa de saida-reentrada-
nova-saida.

Tambem desapareceram do caminho combinado dois gatilhos que existiam no
Top-Turn isolado: NFLX 2021-09-10 e TSLA 2023-06-21. Isso e efeito de path
dependence: ao alterar o momento de reentrada, a carteira pode nao estar mais
posicionada quando um gatilho posterior surgiria.

### Qualidade mediana das reentradas

Top-Turn isolado:

- distancia do fundo: 0,43%;
- captura do fundo: 83,61%;
- dias apos o fundo: 1;
- retorno +5d: +0,25%;
- retorno +10d: -0,96%;
- retorno +20d: +4,45%;
- drawdown +20d: -4,27%.

Top+Bottom v2:

- distancia do fundo: 2,72%;
- captura do fundo: 76,59%;
- dias apos o fundo: 2;
- retorno +5d: +0,55%;
- retorno +10d: +2,53%;
- retorno +20d: +5,35%;
- drawdown +20d: -3,14%.

A espera v2 entrou mais longe do minimo observado, mas apresentou mediana de
retorno posterior e drawdown um pouco melhores. Isso nao compensou o custo de
oportunidade das altas perdidas nem a perda de gatilhos Top-Turn subsequentes.

### Conclusao cientifica do v2

1. O limite de cinco sessoes resolveu o erro arquitetural do v1.
2. Bottom-Turn ainda nao esta validado como detector recorrente de fundos.
3. A maior parte do comportamento v2 vem do cooldown temporal, nao do modelo:
   1 confirmacao contra 7 expiracoes.
4. Top-Turn permanece a referencia economica no universo corrigido:
   US$ 6.306.816,02 e +23,85% sobre Control.
5. O proximo teste deve incluir um baseline `Top-Turn + cooldown fixo de
   5 sessoes sem ML` para separar o valor do simples atraso do valor real da
   previsao Bottom-Turn.
6. Nao ajustar thresholds/features do Bottom-Turn olhando este resultado antes
   dessa ablacao.


## Ablacao Top-Turn + Cooldown5 — protocolo congelado 1.9.0-dev.1

Objetivo: separar o efeito economico de simplesmente esperar cinco sessoes
apos uma saida Top-Turn do valor incremental do modelo Bottom-Turn.

A campanha usa a mesma branch, os mesmos arquivos e o mesmo snapshot congelado.
Nao ha ajuste de thresholds, features, target, calibracao ou horizonte a partir
do resultado Bottom-Turn v2.

Cenarios executados:

1. Control;
2. Top-Turn;
3. Top-Turn + Cooldown5, sem Bottom-Turn/ML;
4. Top-Turn + Bottom-Turn v2.

Regras do baseline Cooldown5:

- somente uma saida Top-Turn real arma o cooldown;
- as cinco sessoes seguintes permanecem em CASH;
- nenhuma probabilidade Bottom-Turn e consultada;
- na sexta sessao a politica Top-Turn volta a operar normalmente;
- entrada inicial nao e bloqueada;
- rotacoes ativo-para-ativo fora dessa janela nao sao alteradas;
- o valor 5 nao foi otimizado: corresponde ao horizonte predeclarado do
  experimento Bottom-Turn.

Implementacao: Cooldown5 reutiliza no mesmo treinamento/replay os mesmos
modelos de utilidade, modelos Top-Turn, caches e thresholds usados no fluxo
Top+Bottom v2. Assim, a diferenca Cooldown5 vs Bottom-Turn v2 isola a regra de
reentrada, sem introduzir uma nova rodada de treinamento.

Interpretacao predeclarada:

- se Bottom-Turn v2 for aproximadamente igual ao Cooldown5, nao ha evidencia
  de valor incremental relevante do ML de fundo;
- se Bottom-Turn v2 superar Cooldown5 de forma material e consistente por
  folds, ha evidencia de valor incremental do sinal Bottom-Turn;
- se ambos perderem para Top-Turn, o custo de esperar domina o beneficio de
  tentar confirmar o fundo;
- nenhum threshold sera retunado olhando esta mesma janela OOS antes dessa
  ablacao ser interpretada.

Artefato principal esperado:
`comparison_cycle_ablation.json`.


## Resultado da ablacao Top-Turn + Cooldown5 — OOS 1.9.0-dev.1

Pacote validado com:

- `research_version=1.9.0-dev.1`;
- `execution_schema=top-bottom-cooldown-ablation-v1`;
- snapshot SHA-256 `4e2fd225cc0ea05da56dad8f0628ca989ad332812a5fa3a7dc796b2b8a6d5128`;
- 1.547 sessoes OOS;
- exclusoes estruturais DOC e CLMT mantidas.

### Resultado agregado

| Cenario | Capital final | vs Control | vs Top-Turn | CAGR | Sharpe | MaxDD | Worst fold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | US$ 5.092.399,32 | - | - | 175,31% | 1,9287 | -36,65% | +124,00% |
| Top-Turn | US$ 6.306.816,02 | +23,85% | - | 185,04% | 1,9844 | -36,65% | +124,72% |
| Top-Turn + Cooldown5 | US$ 5.617.130,69 | +10,30% | -10,94% | 179,73% | 1,9650 | -36,65% | +149,06% |
| Top+Bottom v2 | US$ 5.244.421,51 | +2,99% | -16,85% | 176,62% | 1,9453 | -36,65% | +132,55% |

O Cooldown5, mesmo sem ML de fundo, superou o Top+Bottom v2 em
US$ 372.709,18 (+7,11% em capital do Cooldown sobre o v2, ou -6,64% do v2
em relacao ao Cooldown). Ainda assim, ambos perderam para o Top-Turn puro.

### Resultado decisivo da ablacao

A diferenca Bottom-Turn v2 vs Cooldown5 nasceu praticamente toda de um unico
evento no fold 1.

- Cooldown5: reentrada LKFT em 2020-10-21 a 123,35.
- Bottom-Turn v2: confirmacao ML antecipou a reentrada para 2020-10-20 a
  132,11.
- O Bottom-Turn comprou aproximadamente 7,10% mais caro do que o Cooldown5.
- Depois de 2020-10-21, as duas estrategias selecionaram o mesmo ativo em
  todas as sessoes restantes da amostra; a diferenca de patrimonio permaneceu
  praticamente multiplicativa.
- Razao final Top+Bottom v2 / Cooldown5 = 0,933648, isto e, -6,635%.

Portanto, a unica intervencao efetiva do ML de fundo nesta campanha foi
economicamente negativa contra o baseline sem ML.

### Comparacao por fold

Cooldown5 vs Top-Turn:

- fold 1: +10,83%;
- fold 2: -18,01%;
- fold 3: -1,98%.

Bottom-Turn v2 vs Cooldown5:

- fold 1: -6,63%;
- fold 2: aproximadamente -0,003%;
- fold 3: aproximadamente -0,001%.

Bottom-Turn v2 vs Top-Turn:

- fold 1: +3,48%;
- fold 2: -18,02%;
- fold 3: -1,98%.

Isso confirma duas conclusoes diferentes:

1. esperar cinco sessoes pode ajudar em episodios especificos, sobretudo no
   fold 1, mas nao generalizou para os folds seguintes;
2. o Bottom-Turn ML nao mostrou valor incremental sobre essa espera fixa.

### Cobertura do Bottom-Turn v2

No replay combinado foram observados:

- 8 saidas Top-Turn;
- 39 bloqueios de entrada;
- 7 expiracoes;
- 1 unica entrada confirmada pelo Bottom-Turn;
- apenas 4 observacoes de probabilidade Bottom-Turn disponiveis dentro das
  janelas de decisao.

O Cooldown5 teve 8 janelas armadas, 40 bloqueios e 8 expiracoes. A unica
diferenca operacional relevante entre os dois caminhos foi a liberacao
antecipada de LKFT pelo Bottom-Turn em outubro/2020.

### Conclusao cientifica

A ablacao responde negativamente a pergunta predeclarada: nesta especificacao,
nao ha evidencia de que o LightGBM Bottom-Turn agregue valor economico alem de
uma espera temporal simples apos uma saida Top-Turn.

Mais forte ainda: nesta amostra, a unica decisao realmente tomada pelo modelo
Bottom-Turn piorou o resultado frente ao Cooldown5.

O Top-Turn puro permanece a referencia economica atual no universo corrigido,
com US$ 6.306.816,02, +23,85% sobre Control.

Nao retunar thresholds/features do Bottom-Turn sobre este mesmo OOS. Se a linha
Bottom-Turn continuar, a proxima etapa deve ser uma reformulacao metodologica
predeclarada e validada em novo protocolo, e nao ajuste de parametros para
recuperar este resultado.


## Sensibilidade do universo 54/55/56 — protocolo 1.10.0-dev.1

A revisao da exclusao de CLMT mostrou que o snapshot oficial de referencia
`10.8.74` possui 56 tickers configurados em `ASSETS`, mas registra
**55 ativos elegiveis**. O diagnostico de referencia inclui CLMT e nao inclui
DOC.

Isso corrige uma interpretacao anterior: o checkpoint historico de
Control ~US$ 10,08 milhoes e Top-Turn ~US$ 12,49 milhoes deve ser tratado,
ate reproducao da campanha abaixo, como resultado do universo elegivel de
55 ativos com CLMT presente e DOC excluido, e nao como um universo elegivel de
56 ativos.

### Distincao de identidade

CLMT:
- corporate action 2024-07-11 marcado como `name_change`;
- ticker antigo = CLMT e ticker novo = CLMT;
- CUSIP mudou de 131476103 para 131428104;
- para esta campanha a exclusao e explicitamente sobrescrita, sem alterar
  precos, retornos ou corporate actions do snapshot.

DOC:
- o DOC antigo possui CUSIP 71943U104;
- em 2024-03-01 houve `stock_merger`: DOC -> PEAK, taxa do acquiree 1,0 e
  taxa do acquirer 0,674;
- depois PEAK passou a usar o ticker DOC/CUSIP 42250P103;
- portanto a serie por ticker DOC atravessa identidades economicas distintas.
  Sua inclusao no universo de 56 e apenas diagnostica e nao deve ser promovida
  automaticamente a baseline cientifico.

### Campanha congelada

Versao: `1.10.0-dev.1`.
Schema: `universe-sensitivity-54-55-56-v1`.

Executar somente Control e Top-Turn, sem Bottom-Turn, BOCPD, HSMM ou Hazard,
mantendo snapshot, parametros LightGBM, parametros Top-Turn, folds, custos e
protocolo OOS inalterados.

Cenarios:

1. `u54_current`: exclusoes atuais de DOC e CLMT.
2. `u55_clmt`: CLMT restaurado por override explicito; DOC continua excluido.
3. `u56_raw`: CLMT e DOC incluidos por override; diagnostico de sensibilidade
   por ticker, nao baseline cientifico.

Perguntas da campanha:
- o U55 reproduz o checkpoint historico de ~US$10,08M Control /
  ~US$12,49M Top-Turn?
- a vantagem relativa Top-Turn vs Control continua ~23,85%?
- qual parcela da mudanca U54 -> U55 decorre de restaurar CLMT?
- o que acontece ao forcar DOC no U56, sabendo que ha quebra de identidade?

Nenhum threshold, feature, target ou parametro sera alterado depois de observar
os resultados desta campanha. O artefato principal sera
`comparison_universe_sensitivity.json`.

## Resultado U54/U55/U56 e campanha de contribuicao marginal — 1.11.0-dev.1

### Resultado observado na sensibilidade de universo

O replay 1.10.0-dev.1 confirmou:

| Universo | Control | Top-Turn | Top-Turn vs Control |
| --- | ---: | ---: | ---: |
| U54 atual | US$ 5.092.399,32 | US$ 6.306.816,02 | +23,85% |
| U55 + CLMT | US$ 10.082.425,91 | US$ 12.486.768,00 | +23,85% |
| U56 + CLMT + DOC | US$ 10.082.425,91 | US$ 12.486.768,00 | +23,85% |

DOC foi um controle negativo forte: U55 e U56 produziram o mesmo capital e o
mesmo caminho de selecao. A inclusao de CLMT, ao contrario, alterou
materialmente a trajetoria.

A margem candidata calibrada por fold foi:

- U54: 0,01 / 0,01 / 0,00;
- U55: 0,00 / 0,01 / 0,00;
- U56: 0,00 / 0,01 / 0,00.

Como a margem-base do sistema e 0,0005, o primeiro fold opera efetivamente com
0,01 no U54 e 0,0005 no U55/U56. A mudanca de politica aparece antes de CLMT
ser necessariamente a posicao selecionada, portanto o ganho U54 -> U55 nao
pode ser atribuido apenas ao lucro direto de operacoes em CLMT.

### Hipotese de contribuicao marginal para a rotacao

Hipotese predeclarada: um ativo pode contribuir para o universo de duas formas
distintas:

1. valor direto de investimento, quando o proprio ativo e selecionado e gera
   retorno;
2. valor indireto de informacao/calibracao, quando sua presenca altera a
   politica de rotacao e melhora decisoes entre os demais ativos.

A segunda componente e chamada provisoriamente de contribuicao marginal de
rotacao. Nesta etapa nao se procura novos ativos e nao se cria um score por
ajuste retrospectivo. Primeiro sera testado se o efeito indireto de CLMT existe
de forma separavel.

### Desenho fatorial 2x2 congelado

Versao: 1.11.0-dev.1.
Schema: rotation-contribution-factorial-v1.
Branch unica mantida: research/reversal-bocpd-comparison.

Fator A, disponibilidade de CLMT:
- 0 = U54, CLMT nao investivel;
- 1 = U55, CLMT investivel.

Fator B, politica de rotacao:
- 0 = margens naturais calibradas no U54;
- 1 = margens naturais calibradas no U55.

Celulas:

1. A0/B0: U54 natural, baseline.
2. A0/B1: U54 investivel + margens U55. Esta e a celula
   information-only e testa se a politica aprendida com CLMT melhora o
   universo mesmo quando CLMT nao pode ser comprado.
3. A1/B0: U55 investivel + margens U54. Esta e a celula
   investability-only e mede o valor de permitir CLMT mantendo a politica
   do universo sem CLMT.
4. A1/B1: U55 natural, efeito completo observado.

O override contrafactual somente pode usar valores que ja pertencem ao conjunto
congelado de candidatos de rotation_switch_margin_candidates. Qualquer valor
novo e recusado pelo codigo. Portanto, este experimento nao introduz tuning.

A decomposicao principal sera feita no Control, pois ele isola a politica de
rotacao sem adicionar outro mecanismo. Top-Turn sera executado nas mesmas
celulas como verificacao end-to-end; sua leitura e secundaria porque o universo
tambem participa da calibracao do overlay.

DOC permanece como controle negativo U55 -> U56 e nao e promovido ao universo
cientifico por causa da quebra documentada de identidade/ticker.

### Criterio de interpretacao predeclarado

- se A0/B1 superar A0/B0, existe evidencia de valor indireto da politica
  calibrada com CLMT, mesmo sem CLMT investivel;
- se A1/B0 superar A0/B0, existe valor direto de investibilidade de CLMT sob a
  politica U54;
- a diferenca residual em escala logaritmica e tratada como interacao entre
  disponibilidade e politica;
- nenhuma busca por novos tickers sera feita antes da interpretacao deste
  fatorial;
- somente se a contribuicao indireta se confirmar sera aberta uma campanha
  posterior para procurar a assinatura em outros ativos sem usar o mesmo OOS
  como criterio de tuning.

Artefato principal esperado:
rotation_contribution_factorial.json.

## Resultado fatorial de contribuicao CLMT — 1.11.0-dev.1

Pacote validado com schema rotation-contribution-factorial-v1.

### Decomposicao Control

| Celula | Capital final |
| --- | ---: |
| U54 natural, A0/B0 | US$ 5.092.399,32 |
| U54 + politica U55, A0/B1 | US$ 8.529.013,20 |
| U55 + politica U54, A1/B0 | US$ 6.020.007,45 |
| U55 natural, A1/B1 | US$ 10.082.425,91 |

O efeito completo de CLMT foi +97,99%. A celula information-only, na qual
CLMT nao pode ser comprado mas as margens por fold sao as aprendidas no U55,
terminou em US$ 8.529.013,20, +67,49% sobre o U54 natural. A celula
investability-only terminou em US$ 6.020.007,45, +18,22%.

Em escala logaritmica, apropriada para decompor crescimento composto:
- politica/calibracao: 75,50% do efeito total;
- investibilidade direta: 24,50%;
- interacao multiplicativa: aproximadamente zero (-0,0019%).

Top-Turn reproduziu praticamente a mesma decomposicao:
- U54 natural: US$ 6.306.816,02;
- information-only: US$ 10.562.864,25;
- investability-only: US$ 7.455.688,24;
- U55 completo: US$ 12.486.768,00.

### Localizacao temporal do efeito

A mudanca de politica ocorre exclusivamente no fold 1:
- U54: margem calibrada 0,01;
- U55: margem calibrada 0,00;
- 33 de 1.547 sessoes OOS mudam de ativo, todas no fold 1;
- primeira divergencia: 2020-07-28;
- capital ao fim do fold 1: U54 US$ 22.399,56 contra
  information-only US$ 37.505,09;
- retorno do fold 1: +124,00% contra +275,05%;
- MaxDD do fold 1 melhora de -36,65% para -29,46%.

Nos folds 2 e 3 as margens sao iguais entre U54 e U55, e o caminho
information-only volta a selecionar os mesmos ativos do U54. A vantagem criada
no fold 1 persiste por composicao do capital.

A investibilidade direta de CLMT com politica U54 nao altera o fold 1. Ao fim
do fold 2 seu efeito relativo ainda e negativo, aproximadamente -3,99%, e
somente no fold 3 passa a ser positivo, encerrando em +18,22%.

DOC permanece controle negativo: U55 e U56 tem capital, margens e caminho
identicos.

### Conclusao

O fatorial confirma que a maior parte do salto U54 -> U55 nao vem de comprar
CLMT. Ela vem da politica de rotacao induzida pela presenca de CLMT na
calibracao. Isso e evidencia forte de contribuicao marginal de rotacao.

Ainda nao se pode declarar uma assinatura generalizavel de novo ativo, porque
o efeito de politica observado e concentrado em um unico fold. O proximo passo
deve procurar outros ativos que produzam o mesmo tipo de influencia usando
somente as janelas de treino/calibracao, sem escolher candidatos pelo capital
OOS.

## Campanha signature leave-one-out calibration-only — 1.12.0-dev.1

Versao: 1.12.0-dev.1.
Schema: rotation-contribution-signature-loo-v1.
Branch unica: research/reversal-bocpd-comparison.

Objetivo: descobrir quais ativos possuem contribuicao marginal para a
calibracao da politica antes de qualquer nova selecao por resultado OOS.

Protocolo congelado:
1. usar U55 com CLMT restaurado apenas como universo diagnostico;
2. em cada fold, treinar os modelos LightGBM uma unica vez na janela de treino;
3. medir a superficie completa dos candidatos congelados de switch margin;
4. remover um ativo por vez somente do conjunto de escolhas da calibracao;
5. recalcular a margem otima e o objetivo de calibracao sem retreinar os
   modelos dos demais ativos;
6. registrar margin flip, contribuicao marginal ao objetivo, contribuicao com
   margem fixa, forca do flip e estatisticas cross-sectional de score;
7. nao criar score composto arbitrario nesta etapa;
8. nao executar backtest OOS para selecionar ou ordenar candidatos;
9. usar DOC/U56 somente como controle negativo estrutural;
10. somente depois de interpretar esta campanha sera definido um teste OOS
    confirmatorio para uma hipotese de assinatura predeclarada.

Artefatos esperados:
- rotation_contribution_signature_loo.json;
- rotation_contribution_signature_loo.csv;
- rotation_contribution_signature_aggregate.csv;
- rotation_contribution_margin_surface.csv;
- pacote_analise.zip.

## Mudanca de direcao: expansao aleatoria U56 -> U76 — 1.13.0-dev.1

A campanha 1.12.0-dev.1 de assinatura leave-one-out calibration-only foi
preparada, mas foi substituida antes de uma execucao cientifica pelo desenho
abaixo. Ela nao deve ser interpretada como resultado.

### Pergunta da nova pesquisa

Em vez de procurar imediatamente uma assinatura derivada de CLMT, a nova
campanha expande o universo diagnostico de 56 para 76 tickers com 20 ativos
adicionais escolhidos aleatoriamente de forma reprodutivel. Cada ativo e
tratado como um objeto com propriedades de mercado, modelo, ranking,
calibracao, selecao e contribuicao marginal ao lucro.

Versao: 1.13.0-dev.1.
Schema: object-universe-expansion-76-v1.
Branch unica: research/reversal-bocpd-comparison.

### Amostragem dos 20 ativos

Seed congelado: 20261004.

Catalogo de origem: Alpaca em 2026-10-04. A amostragem parte de ativos US
equity ativos, negociaveis e marginaveis das bolsas NYSE, NASDAQ, AMEX, ARCA e
BATS, com ticker simples e fora dos 56 tickers originais. Depois da ordem
pseudoaleatoria deterministica, o candidato precisa possuir serie diaria SIP
RAW praticamente continua desde janeiro/2016 ate setembro/2026, pelo menos
2.600 barras e nenhuma lacuna superior a 10 dias.

Os 20 objetos congelados sao:

FAF, IJR, GAB, ELS, AEIS, VWOB, BDJ, DGX, ESP, BWZ, PSF, DBA, HEEM, NPKI,
MHK, BLKB, ARCO, AGM, NWFL e SKOR.

A verificacao previa mostrou serie continua para os 20. Candidatos como EGLE
e RWL foram rejeitados automaticamente pelo criterio de continuidade historica.

### Dados

O snapshot original dados/pesquisa permanece intocado. Os 20 novos ativos sao
baixados uma unica vez para dados/pesquisa_expansao_76, com SIP, 1Day, RAW,
mesma data inicial e mesmo corte temporal do snapshot oficial. Corporate
Actions tambem sao congeladas e o manifesto da extensao registra como pai o
SHA-256 do snapshot original.

O diretorio da extensao passa a ser permitido pelo .gitignore para que, depois
da primeira aquisicao local, seus CSVs e manifesto possam ser versionados.

### Objeto ativo

Cada um dos 76 ativos recebe uma linha em asset_objects_76.csv com propriedades
como:

- coorte original_56 ou random_20;
- comprimento e cobertura historica;
- retorno bruto do preco, CAGR, volatilidade e drawdown;
- mediana de dollar volume;
- correlacao e beta em relacao ao SPY;
- media/desvio dos scores LightGBM;
- frequencia top-1, top-3 e percentil medio de ranking;
- participacao nas sessoes selecionadas e numero de linhas de trade;
- numero de folds em que sua retirada muda a margem calibrada;
- contribuicao media ao objetivo de calibracao;
- capital U76 sem o objeto;
- contribuicao marginal absoluta e percentual ao capital final;
- contribuicoes marginais de Sharpe e MaxDD.

### Desenho de computacao

Os modelos LightGBM dos 76 ativos sao treinados apenas uma vez por fold, tanto
na fase de calibracao como no fit final. Isso e valido porque os modelos sao
independentes por ativo.

Depois sao executados:
1. U56 original, com os mesmos folds congelados do U76;
2. U76 completo;
3. 76 replays leave-one-out, removendo um objeto por vez.

Cada leave-one-out recalibra o switch margin usando somente o subconjunto de
75 ativos, mas nao retreina os modelos dos outros 75. Assim isolamos a
contribuicao do objeto para a competicao e para a politica de rotacao sem pagar
o custo de 76 treinamentos completos.

A contribuicao marginal e local e path-dependent. Ela nao e aditiva: os 76
valores leave-one-out nao devem ser somados como se fossem efeitos
independentes.

### Saidas esperadas

- object_universe_76.json;
- asset_objects_76.csv;
- random_20_objects.csv;
- asset_leave_one_out_76.csv;
- object_property_profit_correlations.csv;
- u76_full_predictions.csv;
- u76_full_trades.csv;
- pacote_analise.zip.

O primeiro resultado a interpretar sera o efeito agregado U56 -> U76. Depois
sera analisado quais propriedades distinguem objetos que aumentam o capital de
objetos que o reduzem. Correlacoes propriedade-lucro nesta campanha sao
exploratorias e nao podem, sozinhas, ser usadas como regra final de selecao sem
uma validacao posterior.


## Correcao U76 benchmark/calendar — 1.13.1-dev.1

A primeira execucao da campanha 1.13.0-dev.1 concluiu o treinamento dos 76
modelos no fold 3, mas abortou antes do primeiro replay U56 com:

ValueError: No asset has complete prices for the benchmark execution window.

Causa: a campanha U76 permitia que o conjunto expandido escolhesse a fonte do
calendario de mercado. Os frames U56 eram entao reindexados no calendario
escolhido pelo U76. Pequenas diferencas de sessoes podiam deixar todos os 56
objetos com ao menos uma lacuna na janela, enquanto o benchmark equal-weight
exige pelo menos um ativo com preco completo em toda a janela.

Alem do erro tecnico, permitir que os 20 objetos aleatorios alterassem o
calendario seria um confundidor cientifico. A expansao deve alterar apenas a
competicao entre objetos e a calibracao da rotacao, nao a definicao das
sessoes do experimento.

Correcao congelada:
- versao 1.13.1-dev.1;
- schema permanece object-universe-expansion-76-v1;
- calendario escolhido apenas pelo U56 original e fixo em U56, U76 e LOO;
- benchmark calculado uma unica vez sobre o U56 original e reutilizado em
  todos os 78 replays;
- _simular_exato ganhou benchmark_override opcional, sem mudar o padrao;
- preparar_painel_rotacao e _construir_contexto_execucao ganharam override
  opcional de calendario, sem alterar campanhas anteriores por padrao.

Assim, diferencas U56 -> U76 e U76 -> U76-sem-objeto ficam atribuiveis a
disponibilidade/calibracao/rotacao dos objetos sob a mesma linha temporal e
o mesmo benchmark.

## Guard contra runner local obsoleto — 1.13.2-dev.1

A segunda tentativa local mostrou engine/rotacao.py novo, com suporte a
benchmark_override, mas pesquisar_directional_change_spyder.py ainda estava
na chamada antiga de _simular_exato, sem benchmark_override. O schema da
1.13.1 permaneceu v1 e por isso o guard anterior nao detectou essa mistura.

Correcao:
- versao 1.13.2-dev.1;
- schema object-universe-expansion-76-v2;
- runner possui SCRIPT_RESEARCH_VERSION literal e compara com RESEARCH_VERSION;
- qualquer combinacao de runner antigo + modulo novo aborta antes do snapshot
  e, principalmente, antes de treinar os 456 modelos.

A pesquisa cientifica nao mudou em relacao a 1.13.1: calendario e benchmark
continuam congelados no U56 original. Esta versao apenas torna impossivel
repetir silenciosamente a mistura de arquivos locais observada no Spyder.

## Segundo lote aleatorio vs U56 vencedor — 1.14.0-dev.1

Objetivo solicitado: repetir a expansao com outros 20 ativos aleatorios, sem
reutilizar o primeiro lote, procurar novos objetos que melhorem o U56 e medir
a correlacao desses objetos com o universo vencedor de 56 ativos.

Para evitar cherry-picking, os 20 nomes nao sao escolhidos por lucro ou por
resultado de backtest. O lote e congelado antes da execucao usando seed
2026100402, o mesmo catalogo Alpaca e os mesmos filtros de elegibilidade de
historico. Os 20 nomes do primeiro lote sao excluidos da amostragem.

Segundo lote congelado:
VIOV, MBSD, MVIS, EWD, OPHC, CASY, COLB, EES, GNK, VUZI, FOXF, AMS, IQLT,
ISCF, FUTY, KB, PRN, CE, XTNT e UEC.

ONTO apareceu antes de alguns desses nomes na ordem pseudoaleatoria, mas foi
rejeitado por transicao estrutural NANO -> ONTO observada em Corporate
Actions. A exclusao segue a regra do TCC de nao fazer bridge de identidade.

Desenho principal:
- U56 vencedor e reproduzido com calendario e benchmark congelados;
- U76_B2 adiciona os 20 candidatos simultaneamente para medir efeito de grupo;
- cada candidato e testado isoladamente como U56 + 1 objeto;
- modelos dos 76 ativos sao treinados uma unica vez por fold, pois sao
  independentes por ativo;
- cada insercao recalibra somente a competicao e o switch margin usando o
  conjunto congelado de candidatos;
- o efeito individual U56 + candidato e a medida principal de contribuicao;
- o efeito conjunto dos 20 e secundario, porque interacoes nao sao aditivas.

Correlacoes registradas para cada candidato:
- retorno diario vs retorno equal-weight do U56;
- retorno diario vs retorno da estrategia U56;
- media/mediana/absoluta das correlacoes com os 56 objetos originais;
- objeto U56 mais e menos correlacionado;
- correlacao do score LightGBM do candidato com score medio e melhor score
  cross-sectional do U56;
- correlacao do score do candidato com o decision_score da estrategia U56;
- frequencia em que o candidato supera o melhor score do U56;
- sessoes em que sua insercao muda a selecao da carteira;
- folds em que sua insercao muda o switch margin;
- P&L e taxa de acerto das operacoes do proprio candidato.

Saidas:
- random_batch2_u56_correlation.json;
- random_batch2_candidates.csv;
- random_batch2_property_correlations.csv;
- u56_winner_predictions.csv / trades.csv;
- u76_batch2_predictions.csv / trades.csv;
- pacote_analise.zip.

As correlacoes desta campanha sao exploratorias. Elas servem para formular a
proxima hipotese de assinatura, nao para selecionar retrospectivamente ativos
sem uma terceira amostra intocada.

## Correcao batch2 para modelo ausente em calibracao — 1.14.1-dev.1

A primeira execucao do segundo lote treinou normalmente os 76 modelos finais,
mas no fold 1 a fase de calibracao produziu 75 modelos. Um objeto nao atingiu
o minimo de linhas validas depois do drop de features/target na janela de
treino inicial. Isso e um estado permitido pelo engine: o objeto fica
indisponivel nessa calibracao e sua utilidade permanece -inf ate haver modelo.

O runner 1.14.0 assumia incorretamente que todo simbolo existiria no dicionario
de modelos de calibracao e indexava diretamente o dict, causando KeyError ao
montar o replay U76_B2.

Correcao:
- versao 1.14.1-dev.1;
- _run_subset agora inclui apenas modelos realmente ajustados, preservando os
  simbolos no cache e na competicao com -inf quando o modelo nao existe;
- o mesmo tratamento defensivo foi aplicado aos modelos finais;
- cada fold imprime calibration_missing/final_missing uma unica vez, tornando
  explicito qual objeto ainda nao era elegivel para modelagem naquela janela;
- o desenho cientifico, seed, 20 candidatos, calendario, benchmark e folds
  permanecem inalterados.

Esse comportamento espelha a semantica do engine LightGBM: um ativo sem
amostra minima em uma fase nao deve abortar todo o universo nem receber modelo
artificial; ele simplesmente nao compete naquela fase.


## Assinatura provisoria e validacao no terceiro lote — 1.15.0-dev.1

Ate este ponto existe uma assinatura provisoria, nao uma assinatura validada.

Evidencia replicada entre os lotes 1 e 2:
- objetos que invadem muitas decisoes tendem a degradar o capital;
- alta frequencia de vencer o melhor objeto U56 tende a ser negativa;
- alta variabilidade do score LightGBM tende a ser negativa;
- baixa correlacao de retorno com o U56, isoladamente, nao identifica utilidade;
- os melhores objetos do lote 2 atuaram de forma seletiva: COLB, AMS e FOXF;
- COLB mostrou efeito de caminho, AMS alpha direto e FOXF efeito positivo menor.

Assinatura congelada selective-specialist-v0.1:
- candidate_beats_u56_best_share > 0 e <= 5%;
- candidate_score_std <= 0.15;
- candidate_score_mean <= 0.11;
- abs(model_score_corr_u56_mean) <= 0.25.

Esses limites foram derivados exploratoriamente do lote 2 e por isso nao
constituem validacao. O terceiro lote e o primeiro teste confirmatorio dessa
regra sem reajustar os limites depois de ver seu capital.

Novo desenho:
- U56 permanece controle cientifico intacto;
- U59 = U56 + COLB + AMS + FOXF, removendo neutros e negativos do lote 2;
- ARCO nao entra no U59 porque veio do primeiro protocolo e seu efeito positivo
  foi residual; permanece apenas como evidencia historica separada;
- 20 ativos totalmente novos sao sorteados excluindo lotes 1 e 2;
- cada candidato e classificado pela assinatura antes do replay de capital;
- cada candidato e testado em U56+1 para validacao comparavel ao lote 2;
- cada candidato e testado em U59+1 para valor incremental no universo
  enriquecido;
- U79 mede o efeito conjunto U59 + 20 novos.

Terceiro lote congelado com seed 2026100503:
MG, VSTM, HEWJ, GBAB, CRESY, BGT, DBJP, UBND, PPLT, RXL, JPIN, REXR, QVAL,
REM, CEVA, TRC, SCHA, CALM, DRN e EVH.

Rejeitados antes do congelamento:
FLG (NYCB -> FLG), LBTYA (transicao de identidade/CUSIP), DCOY (SLRX -> DCOY),
BLOX (grande descontinuidade historica), VISN (COMM -> VISN), COR (ABC -> COR)
e ECON (historico inferior a 2.600 barras).

A assinatura sera considerada apenas parcialmente suportada se a direcao
predita distinguir candidatos positivos dos nao positivos no terceiro lote.
Se falhar, os limites nao serao reajustados retroativamente nesta mesma amostra.


## Checkpoint positivo apos validacao do lote 3 — 1.15.0

Execucao validada pelo pacote `pacote_analise(6).zip`, schema
`signature-validation-batch3-u59-v1`.

Resultados principais:
- U56: capital final US$ 10.082.425,91; Sharpe 2,1011; MaxDD -31,22%;
- U59 = U56 + COLB + AMS + FOXF: US$ 30.080.091,01; Sharpe 2,3375;
  MaxDD -31,22%; ganho de +198,34% sobre U56;
- U79 = U59 + 20 candidatos do lote 3: US$ 5.673.953,48; a inclusao
  indiscriminada do lote reduziu o capital em -81,14% contra U59 e elevou o
  MaxDD para -68,69%.

No teste marginal contra U59, somente tres ativos do lote 3 aumentaram o
capital:
- REXR: +10,49%;
- MG: +9,69%;
- CALM: +2,15%.

VSTM foi positivo contra U56, mas negativo (-9,42%) contra U59, confirmando que
a contribuicao depende do universo de competicao. Os dez objetos classificados
como `dormant` foram neutros. Todos os seis `invasive_or_unstable` ficaram
negativos contra U59. A regra `selective-specialist-v0.1` acertou 17/20 contra
U56, mas com recall baixo; por isso permanece como primeira assinatura
confirmatoria, nao como regra final.

Conjunto de adicionais positivos conhecido neste checkpoint:
- lote 2: COLB, AMS, FOXF;
- lote 3 contra U59: MG, REXR, CALM.

O conjunto positivo de seis ativos fica congelado como referencia de descoberta:
`COLB, AMS, FOXF, MG, REXR, CALM`. ARCO permanece fora deste checkpoint por
ter sido medido em protocolo anterior e nao diretamente comparavel.

A proxima campanha nao escolhera outro lote aleatorio e nao usara insercao
individual com replay da estrategia para procurar candidatos. A descoberta
passa a ser feita por pre-selecao inteligente em duas etapas: propriedades de
mercado/historico e comportamento de score LightGBM relativo ao universo de
referencia. O capital OOS dos novos candidatos so podera ser consultado depois
de a lista ter sido congelada.


## Selecao inteligente sem replay por candidato — 1.16.0-dev.1

A partir do checkpoint positivo do lote 3, a pesquisa deixa de sortear novos
lotes aleatorios e deixa de usar insercao individual com replay da estrategia
como mecanismo de descoberta.

Objetivo:
- procurar candidatos no catalogo Alpaca de forma deterministica;
- usar como exemplos positivos conhecidos COLB, AMS, FOXF, MG, REXR e CALM;
- excluir candidatos ja testados e rejeicoes estruturais conhecidas;
- nao consultar capital OOS de nenhum novo candidato durante a selecao.

Stage 1 usa uma janela barata de scouting desde 2023-01-01 e propriedades de
preco, volatilidade, drawdown, liquidez, momentum, eficiencia de tendencia,
correlacao e beta vs SPY. Uma regressao logistica balanceada, treinada apenas
nos 40 candidatos ja rotulados dos lotes 2 e 3, ordena o catalogo elegivel.
Os top 100 vao para um snapshot integral congelado.

Stage 2 baixa/congela historia integral e Corporate Actions dos top 100,
remove problemas estruturais sem bridge, treina LightGBM por fold e mede apenas
o comportamento dos scores contra a referencia fixa U56. Nenhum replay da
carteira e executado por candidato.

Assinatura exploratory selective-specialist-v0.2:
- 0 < beats_u56_best_share <= 5%;
- score_std <= 0,15;
- score_mean <= 0,16;
- abs(corr_score_com_melhor_U56) <= 0,10;
- pelo menos 85% das sessoes de score esperadas.

A v0.2 foi formulada depois de observar o lote 3 e nao e tratada como validada
naquele lote. A lista final contem 20 ativos. Se menos de 20 passarem o filtro
forte, as vagas restantes sao preenchidas pelo ranking congelado, sem olhar
capital. O artefato de congelamento e intelligent_selected_20.csv.

Schema: intelligent-candidate-screen-v1.
Versao: 1.16.0-dev.1.
Branch unica mantida: research/reversal-bocpd-comparison.


## Branch permanente da pesquisa de assinatura inteligente

Data de abertura: 2026-10-05.

Branch ativa unica desta linha a partir deste ponto:
`research/intelligent-asset-signature-v1`.

A branch anterior `research/reversal-bocpd-comparison` fica congelada no commit
`91f33998d37d7b085e0d6403d0cc0b641caa01cf`. A `main` permanece intocada.
Todas as proximas versoes, correcoes, testes e descobertas desta pesquisa devem
ser mantidas nesta nova branch, sem abrir branches paralelas enquanto esta
linha estiver ativa.

Checkpoint cientifico que antecede a nova linha:
- commit: `4b6b71414ce6f7047dc465679e57629ba8a4c453`;
- tag planejada: `research-positive-assets-v1.15.0`;
- U56 permanece a referencia historica de 56 ativos;
- lote 2 adicionou COLB, AMS e FOXF, formando U59;
- contra U59, o lote 3 encontrou MG, REXR e CALM como contribuicoes positivas;
- VSTM foi positivo contra U56, mas negativo contra U59 e nao entra no conjunto;
- ARCO permanece fora deste checkpoint por ter sido medido em protocolo anterior.

Conjunto positivo conhecido para a nova pesquisa:
`COLB, AMS, FOXF, MG, REXR, CALM`.

Esse conjunto define o `U62 candidato = U56 + 6 positivos`. Ele ainda nao e um
baseline cientifico validado em conjunto, porque MG, REXR e CALM foram medidos
individualmente contra U59. A expressao U62 candidato deve ser mantida ate o
replay conjunto confirmar o comportamento dos seis simultaneamente.

Correcao metodologica obrigatoria para a selecao inteligente:
- novos candidatos devem ser avaliados como possiveis acrescimos ao U62 candidato;
- nao usar apenas U56 como contexto de score, porque o caso VSTM demonstrou que
  a contribuicao pode mudar de sinal quando o universo vencedor muda;
- os 100 novos ativos sao apenas pool de pesquisa e nao sao adicionados todos ao
  universo operacional/cientifico;
- a descoberta nao pode usar backtest individual de capital como filtro;
- primeiro usar propriedades historicas e comportamento de score/modelo para
  ordenar candidatos;
- congelar a lista dos 20 selecionados antes de consultar seu resultado de capital;
- somente depois executar um teste confirmatorio contra o U62 candidato.

Estado da versao 1.16.0-dev.1:
- implementou o primeiro rascunho da selecao sem forca bruta;
- ainda usa U56 como referencia de score no Stage 2;
- portanto nao deve ser executada como campanha cientifica final;
- a proxima versao deve corrigir o contexto para U62 candidato antes da execucao.

Regra de continuidade documental nesta branch:
para cada nova versao, acrescentar ao final deste arquivo, sem apagar o historico:
1. versao e commit;
2. pergunta/hipotese testada;
3. universo e ativos incluidos/excluidos;
4. snapshot/dados e regras de congelamento;
5. metodo de selecao e parametros congelados;
6. artefatos gerados;
7. resultado observado, quando existir;
8. interpretacao e limites do resultado;
9. decisao tomada e proximo passo.

Esse registro e obrigatorio para permitir retomada fiel da pesquisa em novas
conversas sem depender do historico do chat.


## Execucao diagnostica 1.16.0-dev.1 — nao promover

Pacote analisado: `pacote_analise(7).zip`.
Schema observado: `intelligent-candidate-screen-v1`.
Versao observada: `1.16.0-dev.1`.
Runtime: 334,3155 s.

Esta execucao e diagnostica e nao deve ser promovida como campanha cientifica
confirmatoria porque o Stage 2 ainda usa `stage2_score_reference=U56`. A decisao
metodologica vigente exige que os novos candidatos sejam avaliados no contexto
do `U62 candidato = U56 + COLB + AMS + FOXF + MG + REXR + CALM`.

O pacote confirmou que nenhum replay de capital foi usado para selecionar os
novos candidatos:
- random_sampling=false;
- candidate_strategy_replays=0;
- selection_uses_new_candidate_capital=false.

Stage 1:
- 2.271 ativos passaram a janela curta de historico;
- 100 foram levados ao pool integral;
- somente 44 desses 100 possuíam a cobertura integral exigida;
- 43 ficaram elegiveis para modelagem apos filtros estruturais;
- GOGL foi excluido por stock_merger GOGL -> CMBT em 2025-08-20.

A lista de 20 produzida por esta versao NAO deve ser congelada como lista de
validacao porque foi ordenada contra U56. Ela foi:
CNC, EQNR, GERN, BCRX, PLUG, MOH, CSIQ, WST, EDU, JD, IRD, MTEX, AMC, WTI,
ILMN, CYTK, UCO, IOVA, CYRX e FLOT.

Somente CNC e EQNR passaram integralmente pela assinatura v0.2. Os outros 18
entraram pelo mecanismo de preenchimento. Entre os 43 modelaveis havia:
- 2 selective_specialist;
- 17 invasive_or_unstable;
- 17 dormant;
- 7 insufficient_score_data.

Auditoria adicional feita apos a execucao, usando somente os 40 candidatos ja
rotulados dos lotes 2 e 3, mostrou que o classificador logistico do Stage 1 nao
generaliza entre os dois lotes: AUC aproximadamente 0,353 ao treinar no lote 3
e testar no lote 2, e 0,451 no sentido inverso. Portanto a probabilidade
`raw_winner_probability` nao deve ser tratada como uma assinatura preditiva nem
como filtro principal da proxima versao.

Outra descoberta de engenharia: a janela curta favoreceu muitos instrumentos
sem historia integral. Dos top 100, 56 falharam o requisito de cobertura de
10 anos. A proxima versao deve separar melhor triagem barata de elegibilidade
historica e nao desperdiçar a maior parte do pool com ativos que depois sao
removidos.

Decisoes para a proxima versao:
1. usar U62 candidato como contexto de score, mantendo o calendario cientifico
   fixo no U56;
2. nao promover os 20 nomes desta execucao;
3. remover o preenchimento forcado com classes invasive/dormant apenas para
   chegar a 20 nomes;
4. tratar Stage 1 como recuperacao/coarse screening, nao como prova de vencedor,
   ate existir validacao cruzada aceitavel;
5. congelar novos candidatos somente depois de uma assinatura calculada no
   contexto U62 e sem consultar capital dos candidatos novos.


## Correcao U62 da selecao inteligente — 1.16.1-dev.1

A versao 1.16.0-dev.1 foi mantida apenas como diagnostico porque calculava a
assinatura dos novos candidatos contra U56. O caso VSTM ja havia demonstrado
que a contribuicao pode mudar de sinal quando o universo vencedor muda.

A versao 1.16.1-dev.1 corrige o contexto cientifico:
- U62 candidato = U56 + COLB + AMS + FOXF + MG + REXR + CALM;
- os seis ativos conhecidos positivos permanecem no contexto de score;
- o calendario continua congelado no U56 original para nao introduzir outro
  confundidor;
- os novos candidatos sao medidos contra o melhor e a media de score do U62
  candidato, nao contra U56;
- nenhum replay da estrategia por candidato e executado durante a descoberta.

O Stage 1 tambem foi endurecido:
- a janela barata de 2023 foi removida;
- a triagem exige o proprio inicio congelado da pesquisa e pelo menos 2.600
  barras antes de um ativo entrar no pool;
- o pool-alvo subiu de 100 para 500 para reduzir o risco de perder candidatos
  devido ao classificador bruto, cuja generalizacao entre lotes foi fraca;
- se houver menos de 500 ativos elegiveis, usa todos os elegiveis disponiveis.

Selecao final:
- assinatura: selective-specialist-u62-v0.1;
- nao existe preenchimento forcado;
- somente candidatos que passam integralmente a assinatura sao congelados;
- no maximo 20 sao selecionados;
- se apenas 7 passarem, ficam 7; nao entram dormant, invasive_or_unstable ou
  insufficient_score_data apenas para completar vinte.

Artefato final da lista:
`intelligent_selected_candidates.csv`.

Schema: `intelligent-candidate-screen-u62-v1`.
Versao: `1.16.1-dev.1`.
Branch ativa: `research/intelligent-asset-signature-v1`.

Esta versao ainda nao possui resultado de execucao. A lista produzida por ela
sera a primeira lista nova que podera ser congelada para teste confirmatorio de
capital contra o U62 candidato.


## Separacao definitiva entre busca e avaliacao financeira — 1.17.0-dev.1

Decisao do usuario em 2026-10-05: busca de ativos e avaliacao financeira deixam
de compartilhar o mesmo runner. A partir desta versao existem exatamente dois
arquivos ativos para esta linha de pesquisa, ambos preparados com celulas "# %%"
e comentarios para execucao no Spyder.

### 1. buscar_ativos_spyder.py

Responsabilidade exclusiva: procurar e classificar novos ativos.

Regras:
- nao executa a estrategia para medir capital dos novos candidatos;
- random_sampling=false;
- candidate_strategy_replays=0;
- selection_uses_new_candidate_capital=false;
- baseline/contexto de score = U59 vencedor;
- U59 = U56 + COLB + AMS + FOXF;
- o calendario continua fixo no U56 original;
- o catalogo Alpaca e filtrado por elegibilidade, continuidade e historico
  integral desde o inicio congelado da pesquisa;
- o pool-alvo para avaliacao por score e de ate 500 ativos;
- o Stage 1 e apenas recuperacao/coarse screening. A probabilidade bruta nao
  deve ser interpretada como prova de vencedor porque sua generalizacao entre
  lotes anteriores foi fraca;
- o Stage 2 treina LightGBM e compara o comportamento de score dos candidatos
  contra o U59, sem replay financeiro;
- assinatura exploratoria: selective-specialist-u59-v0.1;
- nao existe preenchimento forcado para chegar a 20;
- sao congelados no maximo 20 ativos aprovados;
- se menos de 20 passarem, a lista fica menor;
- a lista congelada e gravada em
  dados/pesquisa_smart_candidates/selected_candidates.csv e tambem no pacote da
  busca, com research_version, execution_schema, search_reference e hash do
  snapshot para impedir que o runner financeiro leia uma lista antiga.

Saidas principais:
- output/busca_ativos/asset_search.json;
- output/busca_ativos/intelligent_stage1_ranked.csv;
- output/busca_ativos/intelligent_candidates_ranked.csv;
- output/busca_ativos/intelligent_selected_candidates.csv;
- output/busca_ativos/pacote_busca_ativos.zip.

### 2. avaliar_resultado_financeiro_spyder.py

Responsabilidade exclusiva: executar a estrategia e medir resultado financeiro.

Baseline congelado:
- U59 = U56 + COLB + AMS + FOXF;
- capital historicamente reproduzido na execucao validada anterior:
  US$ 30.080.091,01;
- o runner deve reproduzir esse baseline antes de promover qualquer extensao.

Regras:
- nao procura novos ativos;
- por padrao, le somente a lista congelada por buscar_ativos_spyder.py;
- rejeita lista sem identidade da versao/schema atuais;
- avalia U59 como baseline;
- se existir lista congelada valida, avalia U59 + lista congelada como grupo;
- o replay individual U59 + 1 candidato existe apenas como diagnostico
  financeiro posterior e permanece desligado por padrao
  (AVALIAR_CANDIDATOS_INDIVIDUALMENTE=False);
- replay individual nunca deve voltar a ser usado como mecanismo de descoberta;
- calendario e benchmark permanecem fixos no U56.

Saidas principais:
- output/avaliacao_financeira/financial_evaluation.json;
- output/avaliacao_financeira/financial_summary.csv;
- predictions/trades do U59 e do grupo quando aplicavel;
- output/avaliacao_financeira/pacote_avaliacao_financeira.zip.

### Ordem recomendada no Spyder

Para reproduzir apenas o baseline financeiro:
1. executar avaliar_resultado_financeiro_spyder.py.

Para descobrir e depois confirmar novos ativos:
1. executar buscar_ativos_spyder.py;
2. revisar/congelar o pacote de busca;
3. executar avaliar_resultado_financeiro_spyder.py;
4. comparar o grupo selecionado contra U59.

O antigo arquivo monolitico pesquisar_directional_change_spyder.py foi removido
da branch ativa para evitar execucao acidental de um protocolo obsoleto. Seu
historico continua preservado no Git.

### Correcao conceitual em relacao a 1.16.1

A proposta U62 candidato foi abandonada como referencia imediata da busca.
MG, REXR e CALM continuam sendo evidencia historica positiva individual contra
U59, mas ainda nao foram validados simultaneamente como conjunto. Por decisao do
usuario, o ponto financeiro consolidado permanece U59, o conjunto que atingiu
aproximadamente US$ 30 milhoes. A busca de ate 500 novos candidatos parte desse
U59.

Versao: 1.17.0-dev.1.
Schemas:
- intelligent-asset-search-u59-v1;
- financial-evaluation-u59-v1.
Branch ativa: research/intelligent-asset-signature-v1.


### Implementacao consolidada da 1.17.0-dev.1

Implementacao tecnica concluida ate o commit:
\`7686d6b9d29a9f171f5ebf220c205bbf04222f52\`.

Arquivos ativos desta campanha:
- \`buscar_ativos_spyder.py\`;
- \`avaliar_resultado_financeiro_spyder.py\`.

O antigo runner monolitico \`pesquisar_directional_change_spyder.py\` foi
removido da branch ativa.

O arquivo de busca invalida qualquer lista congelada anterior no inicio de uma
nova execucao. Uma nova lista somente volta a existir se a busca terminar e
gravar \`selected_candidates.csv\` com:
- research_version;
- execution_schema;
- search_reference=U59_WINNER;
- smart_snapshot_sha256.

O arquivo financeiro confere esses quatro campos e tambem exige que o hash da
lista seja exatamente o hash do snapshot SMART atual. Assim, uma busca
interrompida ou um arquivo antigo nao pode ser usado silenciosamente.

O runner financeiro registra ainda como referencia historica:
\`HISTORICAL_U59_ENDING_CAPITAL = 30080091.008142874\`
e imprime a diferenca entre a reproducao corrente e esse valor.

Nao havia status de CI ou workflow automatico associado ao commit de
implementacao. A validacao efetiva desta versao deve ser feita pela execucao
local no Spyder e pelo pacote produzido por cada runner.


## Resultados executados 1.17.0-dev.1

Pacotes analisados em 2026-10-05: pacote_avaliacao_financeira.zip e pacote_busca_ativos.zip.

Avaliacao financeira:
- schema financial-evaluation-u59-v1;
- U59 reproduzido exatamente em US$ 30.080.091,008142874;
- CAGR 267,4033%;
- Sharpe 2,33754894;
- MaxDD -31,2189%;
- worst fold +275,0509%;
- buy-and-hold US$ 38.978,53;
- selected_assets vazio, portanto este pacote validou somente o baseline U59.

Busca inteligente:
- schema intelligent-asset-search-u59-v1;
- random_sampling=false;
- candidate_strategy_replays=0;
- selection_uses_new_candidate_capital=false;
- referencia de score U59_WINNER;
- pool alvo 500; pool efetivo 453; modelaveis 446;
- classes: dormant 188, invasive_or_unstable 111, selective_specialist 72, uncertain 63, insufficient_score_data 12;
- exclusoes estruturais: AMTD e GOGL;
- sem preenchimento forcado.

Lista congelada dos 20 selective_specialist:
SGA, THO, XNTK, CIVB, WDAY, EXR, PAYX, FMBH, SBFG, ALNY, SXC, ICCC, XEL, EBMT, VLRS, PDFS, SITC, FDX, FNWB, MUX.

Proximo passo obrigatorio:
reexecutar avaliar_resultado_financeiro_spyder.py com AVALIAR_LISTA_CONGELADA=True e AVALIAR_CANDIDATOS_INDIVIDUALMENTE=False. O teste primario e U59 + os 20 congelados contra o baseline de US$ 30.080.091,01. Nao alterar lista nem thresholds antes desse replay.


## Resultado financeiro da lista congelada 1.17.0-dev.1

Pacote analisado: pacote_avaliacao_financeira(1).zip.
Schema: financial-evaluation-u59-v1.
Versao: 1.17.0-dev.1.

Teste primario, sem alterar a lista congelada:
U59 versus U59 + SGA, THO, XNTK, CIVB, WDAY, EXR, PAYX, FMBH, SBFG,
ALNY, SXC, ICCC, XEL, EBMT, VLRS, PDFS, SITC, FDX, FNWB e MUX.

Resultados:
- U59: US$ 30.080.091,008142874;
- U59 + lista congelada: US$ 2.017.935,5138941268;
- delta: -US$ 28.062.155,494248748;
- diferenca percentual: -93,2915% contra U59;
- CAGR: U59 267,4033% versus grupo 136,8624%;
- Sharpe: U59 2,33755 versus grupo 1,74288;
- MaxDD: U59 -31,2189% versus grupo -33,6579%;
- worst fold: U59 +275,0509% versus grupo +290,6203%.

O grupo nao perdeu dinheiro em termos absolutos; o problema foi grande perda de
captura de oportunidade. O dano se concentrou nos folds 2 e 3:
- fold 1: grupo terminou 4,15% acima do U59;
- fold 2: grupo terminou com apenas 15,96% do capital acumulado do U59;
- fold 3: grupo terminou com 6,71% do capital final do U59.

Os novos ativos ocuparam aproximadamente:
- 5,75% das sessoes no fold 1;
- 36,11% no fold 2;
- 25,60% no fold 3.

Entre as posicoes fechadas dos novos ativos, 81 vendas somaram PnL realizado
aproximado de -US$ 926.785, contra +US$ 2,866 milhoes nas demais posicoes do
grupo. Quatro ativos congelados nao chegaram a ser selecionados em nenhuma
sessao: CIVB, EXR, SXC e XNTK.

Os maiores diagnosticos negativos entre os selecionados foram PDFS, ICCC,
VLRS, FDX e SITC. Isto e apenas diagnostico da execucao do grupo, nao validacao
individual de efeito marginal.

Interpretacao cientifica:
a assinatura selective-specialist-u59-v0.1 NAO foi validada economicamente
como regra suficiente para expandir o U59. Ela selecionou candidatos com
comportamento de score aparentemente complementar, mas o conjunto alterou a
rotacao e reduziu fortemente a captura das oportunidades que geravam o
crescimento extraordinario do U59, principalmente no fold 2.

Nao retunar thresholds usando este resultado e reapresentar o mesmo lote como
confirmatorio. Qualquer nova regra derivada deste fracasso passa a ser
exploratoria e precisa de nova validacao congelada.


## Diagnostico individual dos 20 congelados — 1.17.0-dev.1

Pacote analisado: pacote_avaliacao_financeira(2).zip.
Schema: financial-evaluation-u59-v1.
Versao: 1.17.0-dev.1.
Runtime: 153,217 s.

O baseline U59 foi novamente reproduzido em US$ 30.080.091,008142874.
O teste do grupo dos 20 permaneceu em US$ 2.017.935,5138941268
(-93,2915% vs U59).

O replay individual U59 + 1 ativo, feito apenas depois do congelamento da lista,
produziu 8 positivos, 1 neutro e 11 negativos. Portanto, entre os 20 ativos
previamente classificados como selective_specialist, a precisao economica
observada para sinal positivo foi 40%.

Positivos individuais:
- THO: US$ 39.061.905,24; +29,8597% vs U59;
- WDAY: US$ 34.926.832,65; +16,1128%;
- EXR: US$ 34.599.017,24; +15,0230%;
- XEL: US$ 33.355.376,81; +10,8886%;
- SBFG: US$ 31.758.236,39; +5,5789%;
- PAYX: US$ 30.923.853,30; +2,8051%;
- MUX: US$ 30.714.006,84; +2,1074%;
- SXC: US$ 30.614.920,79; +1,7780%.

Neutro:
- CIVB: US$ 30.080.091,01; 0,0000%.

Negativos:
- XNTK: -0,6064%;
- SGA: -12,1390%;
- ICCC: -16,8931%;
- PDFS: -17,8729%;
- FDX: -19,9311%;
- FNWB: -20,3256%;
- FMBH: -24,4327%;
- ALNY: -25,8435%;
- EBMT: -27,4971%;
- SITC: -39,3984%;
- VLRS: -62,1047%.

Media dos oito efeitos positivos: +10,5192%.
Mediana dos oito positivos: +8,2337%.
Media dos onze efeitos negativos: -24,2768%.
Mediana dos onze negativos: -20,3256%.

Descoberta diagnostica, NAO regra validada:
dentro destes 20, candidate_beats_u59_best_share apresentou associacao inversa
com o efeito financeiro (Pearson aproximadamente -0,683; Spearman -0,598).
Os positivos tiveram media dessa medida de aproximadamente 0,6303%, enquanto
os nao positivos ficaram em aproximadamente 2,0470%. O limiar <=5% usado pela
assinatura parece largo demais neste lote. Tambem houve associacao moderada
entre menor abs_score_corr_u59_best e melhor resultado financeiro.

Essa observacao e post-hoc. Nao pode ser transformada em nova regra e validada
sobre estes mesmos 20. Qualquer threshold novo deve ser tratado como
exploratorio e testado em nova lista congelada.

Outro resultado importante: ativos que sao positivos isoladamente podem
interagir mal quando adicionados simultaneamente. O fracasso do grupo dos 20
nao implica que todos sejam ruins; o replay individual encontrou oito
contribuicoes positivas reais contra U59.

Proximo experimento exploratorio recomendado:
avaliar U59 + os oito positivos individuais juntos
(THO, WDAY, EXR, XEL, SBFG, PAYX, MUX, SXC) para medir interacao e
nao-aditividade. Esse teste nao e confirmatorio, pois os oito foram escolhidos
apos observar os resultados financeiros individuais. Depois, usar os rotulos
dos 20 somente para desenvolver uma nova assinatura e valida-la em candidatos
novos e previamente congelados.


## Experimento exploratorio dos oito positivos — 1.17.1-dev.1

Apos o replay individual da lista congelada 1.17.0-dev.1, oito ativos foram
positivos contra o U59:
THO, WDAY, EXR, XEL, SBFG, PAYX, MUX e SXC.

Novo experimento financeiro, explicitamente exploratorio:
U59 + os oito positivos individuais, todos simultaneamente.

Arquivo:
avaliar_resultado_financeiro_spyder.py

Versao financeira:
1.17.1-dev.1

Schema:
financial-evaluation-u59-positive8-v1

Configuracao congelada desta rodada:
- EXECUTAR_BASELINE_U59=True;
- AVALIAR_LISTA_CONGELADA=True;
- AVALIAR_GRUPO_CONGELADO=False;
- AVALIAR_GRUPO_POSITIVOS_DIAGNOSTICOS=True;
- AVALIAR_CANDIDATOS_INDIVIDUALMENTE=False.

A lista original continua sendo a lista congelada da busca 1.17.0-dev.1.
O runner financeiro exige SOURCE_SEARCH_VERSION=1.17.0-dev.1 e verifica o hash
do snapshot SMART. Nao e necessario nem permitido refazer a busca para esta
rodada.

Objetivo:
medir se oito ativos que foram positivos individualmente conseguem coexistir
no mesmo universo sem destruir a vantagem do U59.

Interpretacao:
este teste NAO e confirmatorio, porque os oito foram escolhidos apos observar
seus resultados financeiros individuais. Serve apenas para estudar interacao
e nao-aditividade entre vencedores.

Saidas adicionais:
- u59_plus_positive8_predictions.csv;
- u59_plus_positive8_trades.csv;
- pacote_avaliacao_financeira_positivos8.zip.


## Pacote financeiro rejeitado por versao antiga

O arquivo pacote_avaliacao_financeira(3).zip, gerado em 2026-10-05, nao
corresponde ao experimento exploratorio dos oito positivos.

O pacote reporta:
- research_version=1.17.0-dev.1;
- execution_schema=financial-evaluation-u59-v1;
- evaluate_group=true;
- evaluate_individual_candidates=true;
- u59_plus_diagnostic_positive8 ausente/null.

Ele repetiu a campanha anterior com o grupo congelado de 20 e os replays
individuais. Portanto, nenhum resultado deste pacote deve ser interpretado como
resultado de U59 + THO, WDAY, EXR, XEL, SBFG, PAYX, MUX e SXC.

A branch correta possui avaliar_resultado_financeiro_spyder.py em
1.17.1-dev.1, schema financial-evaluation-u59-positive8-v1, com:
AVALIAR_GRUPO_CONGELADO=False,
AVALIAR_GRUPO_POSITIVOS_DIAGNOSTICOS=True e
AVALIAR_CANDIDATOS_INDIVIDUALMENTE=False.

Necessario atualizar a branch local, reiniciar o kernel do Spyder e executar
novamente apenas o runner financeiro.


## Resultado U59 + oito positivos exploratorios — 1.17.1-dev.1

Pacote analisado: pacote_avaliacao_financeira_positivos8.zip.
Schema: financial-evaluation-u59-positive8-v1.
Versao financeira: 1.17.1-dev.1.
Versao da busca de origem: 1.17.0-dev.1.
Runtime: 93,310 s.

Experimento exploratorio:
U59 + THO, WDAY, EXR, XEL, SBFG, PAYX, MUX e SXC.

Baseline U59 reproduzido:
- capital final US$ 30.080.091,008142874;
- CAGR 267,4033%;
- Sharpe 2,33754894;
- MaxDD -31,2189%;
- worst fold +275,0509%.

Grupo dos oito:
- capital final US$ 58.557.157,67496595;
- delta vs U59 +US$ 28.477.066,666823074;
- ganho relativo +94,6708%;
- CAGR 309,4001%;
- Sharpe 2,51874371;
- MaxDD -30,3591%;
- worst fold +282,5896%.

O grupo superou o U59 em todos os folds:
- fold 1: +2,0100% no capital acumulado ao fim do fold;
- fold 2: +14,2223%;
- fold 3/final: +94,6708%.

Uso dos oito no replay conjunto:
- THO: 18 sessoes;
- WDAY: 41 sessoes;
- EXR: 2 sessoes;
- XEL: 7 sessoes;
- SBFG: 2 sessoes;
- PAYX: 4 sessoes;
- MUX: 2 sessoes;
- SXC: 0 sessoes.

O resultado mostra que os oito positivos individuais conseguiram coexistir e,
em conjunto, quase dobraram o capital final do U59, com Sharpe maior e MaxDD
ligeiramente menor. Entretanto, continua sendo resultado exploratorio e nao
confirmatorio, porque a escolha dos oito foi feita depois de observar seus
replays individuais.

Interpretacao:
- a falha do grupo original de 20 decorreu principalmente dos falsos positivos;
- ha evidencia de que um subconjunto de vencedores marginais pode ser
  compativel em grupo;
- complementaridade/compatibilidade entre ativos importa e nao pode ser
  inferida apenas pelo fato de cada ativo ser selective_specialist;
- o proximo desenvolvimento deve usar os 20 rotulados para melhorar a assinatura
  e depois validar a nova regra em candidatos completamente novos e congelados.


## Correcao da visualizacao da pesquisa — 1.17.2-dev.1

Foi identificado que os novos runners buscar_ativos_spyder.py e
avaliar_resultado_financeiro_spyder.py nao chamavam o modulo de graficos.
Os graficos existentes eram atualizados apenas por reproduzir_experimento_spyder.py
e continuavam representando a comparacao antiga Control/Soft. Portanto as
descobertas U59, lista congelada de 20, diagnostico individual e U59 + 8 nao
estavam sendo refletidas visualmente.

A versao financeira 1.17.2-dev.1 corrige isso sem criar nova branch.

Foi adicionada em reproducao/graficos.py a funcao
gerar_graficos_pesquisa_financeira, chamada automaticamente pelo runner
financeiro. Cada nova execucao passa a recriar output/avaliacao_financeira/
graficos_pesquisa com CSV, PNG e SVG auditaveis.

Graficos previstos:
- evolucao_capital_pesquisa: U56, U59, U59 + 20 e U59 + 8;
- curvas_capital_cenarios_log: curvas de capital da rodada atual;
- vantagem_relativa_vs_u59: vantagem acumulada sessao a sessao;
- capital_por_fold: comparacao do capital acumulado ao fim de cada fold;
- efeito_individual_candidatos: os 20 efeitos marginais contra U59;
- uso_ativos_adicionados: numero de sessoes em que cada novo ativo foi escolhido;
- assinatura_distribuicao_classes, quando o CSV da busca estiver presente;
- assinatura_beats_vs_correlacao, quando o CSV da busca estiver presente.

Os dados que alimentam cada grafico tambem sao gravados em CSV na mesma pasta.
O pacote financeiro passa a incluir recursivamente esses artefatos.

Versao financeira: 1.17.2-dev.1.
Schema: financial-evaluation-u59-positive8-v2.
Branch: research/intelligent-asset-signature-v1.

Esta mudanca e de visualizacao/auditoria. Nao altera modelos, thresholds,
ativos, folds, capital ou a interpretacao cientifica dos resultados existentes.


## Verificacao dos graficos da pesquisa — 1.17.2-dev.1

Pacote verificado:
pacote_avaliacao_financeira_positivos8_graficos.zip.

A integracao dos graficos funcionou. O pacote contem 31 arquivos, incluindo
7 conjuntos de visualizacao com CSV + PNG + SVG:
- assinatura_beats_vs_correlacao;
- assinatura_distribuicao_classes;
- capital_por_fold;
- curvas_capital_cenarios_log;
- efeito_individual_candidatos;
- evolucao_capital_pesquisa;
- uso_ativos_adicionados;
- vantagem_relativa_vs_u59.
(Os dois primeiros sao diagnosticos da busca; os demais acompanham a campanha
financeira. O JSON lista 25 artefatos de graficos incluindo diretorio e formatos.)

A versao executada esta correta:
- research_version=1.17.2-dev.1;
- shared_module_version=1.17.0-dev.1;
- source_search_version=1.17.0-dev.1;
- schema=financial-evaluation-u59-positive8-v2.

Os graficos reproduzem os resultados esperados:
- U56: US$ 10.082.425,91;
- U59: US$ 30.080.091,01;
- U59 + 20: US$ 2.017.935,51;
- U59 + 8: US$ 58.557.157,67.

A curva de vantagem relativa U59+8 vs U59 mostra:
- primeira diferenca em 2022-05-19, inicialmente -1,12%;
- minimo de -5,53% em 2024-02-12;
- depois de 2024-02-12 nao volta a ficar negativa;
- cruza +10% em 2024-02-21;
- cruza +20% em 2024-03-14;
- cruza +50% em 2025-11-12;
- termina em +94,67% em 2026-09-17.

Esse comportamento mostra que a contribuicao dos novos ativos e temporal/regime
dependente, reforcando que a assinatura matematica precisa incluir contexto
temporal e nao apenas caracteristicas estaticas do ticker.

No replay conjunto, os novos ativos foram selecionados em:
WDAY 41 sessoes, THO 18, XEL 7, PAYX 4, EXR 2, SBFG 2, MUX 2 e SXC 0.
Entre vendas dos novos ativos, WDAY teve PnL realizado aproximado de
US$ 9,83 milhoes. Isso nao deve ser interpretado como efeito causal isolado,
pois a presenca dos demais ativos altera ranking, scores e caminho do capital.

Nova conclusao visual importante:
o grafico atual de assinatura separa selective_specialist de nao selecionados,
mas ainda nao mostra diretamente vencedor financeiro vs falso positivo.
Para a fase de generalizacao, os proximos graficos devem vincular as features
pre-replay aos rotulos financeiros dos 20 congelados, sem usar esses rotulos
como validacao da mesma regra.


## Analise estatistica consolidada da assinatura — sem novos replays

Decisao metodologica: interromper o ciclo de novos replays sobre os mesmos
dados. O conjunto existente ja e suficiente para caracterizar matematicamente
a assinatura em desenvolvimento. Novos replays sobre os mesmos candidatos
aumentariam principalmente o risco de overfitting/post-hoc, nao a informacao.

Base analisada:
- 40 candidatos historicos rotulados para Stage1: batch2=20 e batch3=20;
  em cada batch, 3 positivos e 17 nao positivos (prevalencia historica 15%);
- 446 candidatos modelaveis na busca 1.17.0, 72 classificados como
  selective_specialist e 20 congelados para replay financeiro;
- entre os 20 congelados: 8 positivos, 1 neutro, 11 negativos;
- trajetorias por sessao de U59, U59+20 e U59+8, incluindo trades e
  diagnosticos de regime.

1. Enriquecimento da selecao nova.
A taxa de positivos nos 20 congelados foi 40%, contra 15% nos dois lotes
historicos. Isso corresponde a enriquecimento de 2,67x. Usando 15% apenas como
referencia historica de prevalencia, P(X>=8 | n=20,p=0,15)=0,0059 em teste
binomial unilateral. IC Wilson 95% para 8/20: aproximadamente 21,9%-61,3%.
Esse teste sustenta que a busca nova enriqueceu candidatos, mas nao estima a
prevalencia real dos 446 porque somente 20 receberam replay financeiro.

2. Features estaticas nao generalizam.
Regressao logistica usando apenas CAGR, volatilidade, drawdown, liquidez,
positive-day-share, momentum, trend efficiency, corr SPY e beta ficou perto
do acaso em validacao cruzada entre os lotes historicos:
- batch2 -> batch3: AUC aproximadamente 0,48;
- batch3 -> batch2: AUC aproximadamente 0,51.
Conclusao: a assinatura nao esta em atributos estaticos do ticker.

3. Features contextuais mostram sinal consistente.
Nos 20 novos congelados, candidate_beats_u59_best_share foi a variavel mais
informativa:
- Pearson com delta de capital: aproximadamente -0,683;
- Spearman: aproximadamente -0,598;
- AUC para menor-is-better: aproximadamente 0,802;
- mediana nos vencedores: 0,226% das sessoes;
- mediana nos nao vencedores: 1,907%.
Estimador robusto Theil-Sen: cada +1 ponto percentual em beats-share ficou
associado a cerca de -12,6 pontos percentuais no efeito de capital, IC 95%
aproximado [-16,4; -4,8], dentro desta amostra selecionada.

Entre candidatos ativos do batch3, as direcoes foram coerentes:
- menor beats-share: AUC 0,738;
- menor abs(corr score vs melhor baseline): AUC 0,857;
- menor score_std: AUC 1,000;
- maior positive_score_share: AUC 0,619.
Nos 20 novos, as mesmas orientacoes produziram AUCs aproximadas de 0,802,
0,615, 0,635 e 0,583 respectivamente.
Portanto a geometria da assinatura e mais estavel do que um threshold isolado:
baixa frequencia de dominancia, baixa interferencia/correlacao, score mais
estavel e predominantemente positivo.

4. Mecanismo de rotacao e esparso.
Comparando U59+8 contra U59 nas 1547 sessoes:
- apenas 93 sessoes (6,01%) mudaram o ativo selecionado;
- apenas 76 sessoes (4,91%) selecionaram diretamente um dos oito novos;
- essas 76 sessoes responderam por cerca de 86,0% da vantagem relativa em
  log-capital;
- 17 sessoes em que a inclusao dos novos ativos alterou a escolha para outro
  ativo legado responderam por mais 17,4%;
- sessoes em que ambos os universos escolheram o mesmo ativo responderam por
  aproximadamente -3,5% da diferenca em log-capital.
Isso confirma uma assinatura de especialista raro, nao de ativo
persistentemente dominante.

5. Dependencia temporal/regime.
Contribuicao por fold do U59+8:
- fold1: 13 sessoes com novos ativos, vantagem acumulada final +2,01%;
- fold2: 34 sessoes, vantagem acumulada +14,22%;
- fold3: 29 sessoes, vantagem acumulada +94,67%.
O fold3 sozinho respondeu por aproximadamente 80% do log-excesso total.

Nas sessoes com novos ativos, a mediana de breadth-20 foi aproximadamente
49,3%, contra 58,2% nas sessoes em que U59 e U59+8 escolheram o mesmo ativo.
Mann-Whitney bilateral p aproximado 0,016 antes de correcao por multiplos
testes. O sinal e sugestivo, nao confirmatorio. No fold3, breadth-20 mediano
foi 44,8% nas sessoes dos novos ativos contra 58,2% nas sessoes iguais, e o
SPY return-20 mediano foi cerca de -1,06% contra +1,65%.
Conclusao: a assinatura possui componente de ativacao por regime, sobretudo
em mercados de menor breadth.

6. Interacao entre os oito positivos.
Soma dos deltas isolados dos oito: aproximadamente +US$ 25,31M, equivalente
a +84,15 pontos percentuais sobre o U59.
Delta observado dos oito juntos: +US$ 28,48M, +94,67%.
Residual descritivo de interacao: aproximadamente +US$ 3,16M, ou +10,52
pontos percentuais. Nao e Shapley nem efeito causal, mas mostra ausencia de
interacao destrutiva liquida no conjunto dos oito.

7. THO e WDAY cumprem papeis diferentes.
WDAY foi selecionado em 41 sessoes e somou cerca de US$ 9,83M de PnL realizado
nas vendas do replay conjunto. THO foi selecionado em 18 sessoes. Em
decomposicao de log-excesso por ativo selecionado, THO respondeu por cerca de
0,370 de log-excesso e WDAY por 0,147. Portanto WDAY domina PnL nominal devido
tambem ao tamanho do capital/timing, enquanto THO possui maior contribuicao
relativa percentual no caminho. Nao atribuir causalidade isolada a nenhum
deles.

8. Forma matematica recomendada.
A assinatura deve ser modelada como processo em dois niveis:
(a) ativacao/especializacao rara; e
(b) qualidade condicional quando ativa, com interacao de regime.
Uma forma de desenvolvimento e:

P(deltaC>0 | X,R) = sigmoid(
  beta0
  - beta1*rank(beats_share)
  - beta2*rank(abs(corr_best))
  - beta3*rank(score_std)
  + beta4*rank(positive_score_share)
  + gamma' * regime
  + interacoes
)

Os ranks devem ser relativos ao conjunto candidato/contexto, nao ao ticker.
Os coeficientes numericos ainda sao de desenvolvimento; a conclusao robusta e
a orientacao/sinal das dimensoes, nao um threshold final.

Conclusao metodologica:
nao realizar nova rodada financeira sobre os mesmos candidatos para procurar
uma regra melhor. A proxima etapa e analise/modelagem offline dos dados ja
existentes, com leave-one-cohort-out, bootstrap e regularizacao. Uma futura
amostra intocada sera necessaria apenas para comprovar generalizacao externa,
nao para continuar descobrindo a estrutura nos dados atuais.


## Analise standalone dos oito ativos do cenario de US$ 58,56M

Foi criado, na mesma branch research/intelligent-asset-signature-v1, o runner:
analisar_topos_fundos_ativos_58m_spyder.py

Versao: 1.17.3-dev.1.
Schema: standalone-top-bottom-eight-assets-v1.

Objetivo:
estudar THO, WDAY, EXR, XEL, SBFG, PAYX, MUX e SXC individualmente, sem
reinseri-los no universo completo e sem executar novo backtest financeiro.

O runner:
- le o OHLCV congelado do snapshot dados/pesquisa_smart_candidates;
- reutiliza somente as entradas/saidas ja observadas no replay U59+8 de
  US$ 58,56M;
- detecta topos e fundos Directional Change em 2%, 4% e 8%;
- separa data do extremo (ex-post descritiva) da data de confirmacao (causal);
- calcula features locais/causais na entrada e na saida;
- mede MFE, MAE, pico durante a posicao, devolucao do pico ate a saida,
  captura do movimento e distancia temporal/preco a fundos/topos;
- gera um grafico por ativo com fechamento, topos/fundos DC 4% e as entradas/
  saidas reais do U59+8;
- exporta tabelas de eventos, swings, operacoes e resumo por ativo.

Arquivos principais esperados:
- standalone_asset_summary.csv;
- standalone_trade_details.csv;
- standalone_directional_change_events.csv;
- standalone_directional_swings.csv;
- standalone_turn_geometry.csv;
- standalone_asset_analysis.json;
- graficos/<ATIVO>_topos_fundos_entradas_saidas.png/svg;
- pacote_analise_ativos_58m_standalone.zip.

Este experimento nao procura novos ativos, nao recalibra a assinatura, nao
treina LightGBM e nao altera o capital. E um diagnostico morfologico/temporal
dos vencedores ja encontrados, destinado a investigar se compartilham
caracteristicas de entrada apos fundos, persistencia ate topos e padroes de
saida/devolucao do pico.


## Resultado da analise standalone de topos/fundos dos oito ativos — 1.17.3-dev.1

Pacote analisado:
pacote_analise_ativos_58m_standalone.zip

Versao:
1.17.3-dev.1

Schema:
standalone-top-bottom-eight-assets-v1

A analise nao executou novo backtest nem treinou modelo. Ela usou o OHLCV
congelado dos oito ativos do cenario de US$ 58,56M e as entradas/saidas ja
observadas no replay U59+8.

Amostra de execucao:
- 22 operacoes fechadas;
- 7 dos 8 ativos foram efetivamente negociados;
- SXC nao foi selecionado em nenhuma operacao no U59+8;
- WDAY teve 10 trades, THO 5, XEL 2, PAYX 2, EXR 1, SBFG 1 e MUX 1.

Padrao dominante de entrada:
- 16/22 entradas (72,7%) ocorreram com retorno de 20 sessoes negativo;
- 17/22 (77,3%) com RSI14 abaixo de 0,50;
- 18/22 (81,8%) abaixo da EMA20;
- 17/22 (77,3%) na metade inferior do canal de 20 sessoes;
- 15/22 (68,2%) a no maximo 10% do minimo de 20 sessoes;
- 14/22 (63,6%) estavam pelo menos 10% abaixo da maxima de 20 sessoes.

Medianas pooled na entrada:
- return20 = -6,45%;
- RSI14 = 0,409;
- distancia da EMA20 = -3,76%;
- distancia da maxima de 20 = -13,76%;
- distancia da minima de 20 = +7,42%;
- posicao no canal de 20 = 0,268.

Directional Change mostra que o timing esta mais ligado a fundos curtos do que
a grandes fundos:
- DC 2%: mediana 2,5 sessoes apos fundo confirmado; 72,7% das entradas em ate
  5 sessoes; preco mediano +1,93% acima do extremo do fundo;
- DC 4%: mediana 6 sessoes; 40,9% em ate 5 sessoes; +2,62% acima do fundo;
- DC 8%: mediana 16,5 sessoes; 27,3% em ate 5 sessoes; +4,27% acima do fundo.

Padrao de saida:
- mediana da posicao = 2 sessoes;
- mediana do MFE = +4,96%;
- mediana do MAE = -1,65%;
- mediana da devolucao do pico da posicao ate a saida = -2,20%;
- 12/22 saidas ocorreram na mesma sessao do pico da posicao;
- 17/22 em ate 1 sessao apos o pico;
- 21/22 em ate 2 sessoes;
- todas as 22 em ate 4 sessoes.

A passagem entrada -> saida deslocou a posicao, em mediana, para cima:
- RSI14: 0,409 -> 0,436;
- distancia EMA20: -3,76% -> -1,51%;
- distancia da maxima de 20: -13,76% -> -9,90%;
- posicao no canal de 20: 0,268 -> 0,415.
Em teste de Wilcoxon pareado, a reducao do ATR percentual e a aproximacao da
maxima de 20 sessoes permaneceram significativas apos correcao Holm entre as
features testadas. Os demais sinais sao descritivos devido a n pequeno e
dependencia entre trades do mesmo ativo.

Morfologia por ativo:
- WDAY e o caso mais robusto na execucao conjunta: 10 trades, 90% vencedores,
  retorno mediano +3,81%, MFE mediano +6,59%, MAE mediano -1,40%;
- THO: 5 trades, 60% vencedores, retorno mediano +1,88%, entrada tipicamente
  apos pullback forte e grande espaco ex-post ate o proximo topo DC4;
- XEL: 2 trades, comportamento de recuperacao semelhante ao padrao pooled;
- PAYX, MUX e EXR tiveram pouca amostra e nao sustentam inferencia individual;
- SBFG foi uma excecao morfologica de momentum: unica entrada ocorreu com
  RSI alto, retorno20 positivo, acima da EMA20 e no topo do canal;
- SXC nao possui entradas/saidas para caracterizacao operacional no grupo.

A geometria standalone dos ativos em DC4 mostrou amplitudes de swing medianas
relativamente semelhantes, mas frequencias muito diferentes:
- eventos/ano aproximados: MUX 46,6; SXC 33,8; THO 31,1; WDAY 28,3;
  SBFG 19,7; EXR 16,5; PAYX 16,3; XEL 12,0.
A razao mediana de swing de alta / magnitude do swing de baixa em DC4 foi
maior que 1 em todos exceto MUX; XEL foi o mais assimetrico (~1,41).

Conclusao:
o padrao comum nao e "comprar no fundo absoluto" nem "comprar perto do topo".
A maior parte das entradas do U59+8 ocorre durante pullbacks, perto de fundos
locais de pequena escala (2%), com momentum ainda fraco/negativo e preco abaixo
da EMA20. A estrategia captura uma recuperacao curta, normalmente sai muito
proxima do pico da propria posicao e raramente permanece tempo suficiente para
capturar o topo estrutural seguinte de 4% ou 8%.

Isso sugere uma assinatura temporal em dois niveis:
(1) ativacao de reversao curta apos micro-fundo;
(2) saida rapida apos expansao positiva, antes de um topo estrutural maior.

A analise e descritiva e usa entradas/saidas do replay conjunto U59+8. Ela nao
prova que os mesmos pontos seriam gerados por uma estrategia independente de
um unico ativo.


## Buy-and-hold dos oito ativos do cenario de US$ 58,56M

Foi calculado um benchmark simples de comprar e manter usando a mesma serie
standalone plotada para THO, WDAY, EXR, XEL, SBFG, PAYX, MUX e SXC.

Janela comum da serie analisada:
2016-10-17 a 2026-09-17.

Hipoteses do calculo:
- US$ 10.000 investidos em cada ativo no primeiro fechamento da serie;
- fracoes de acoes permitidas;
- manutencao ate o ultimo fechamento da serie;
- precos ajustados apenas por splits, coerentes com o pipeline;
- sem reinvestimento de dividendos, taxas ou impostos.

Resultados:
- THO: 82,27 -> 68,84; retorno -16,32%; US$ 8.367,57;
- WDAY: 86,08 -> 199,28; retorno +131,51%; US$ 23.150,56;
- EXR: 76,29 -> 139,03; retorno +82,24%; US$ 18.223,88;
- XEL: 40,34 -> 73,63; retorno +82,52%; US$ 18.252,35;
- SBFG: 12,69 -> 29,41; retorno +131,76%; US$ 23.175,73;
- PAYX: 55,74 -> 116,51; retorno +109,02%; US$ 20.902,40;
- MUX: 33,50 -> 18,59; retorno -44,51%; US$ 5.549,25;
- SXC: 7,45 -> 9,66; retorno +29,66%; US$ 12.966,44.

Capital inicial total: US$ 80.000,00.
Capital final total: US$ 130.588,19.
Lucro: US$ 50.588,19.
Retorno agregado do portfolio equal-dollar: +63,24%.

Normalizado para US$ 10.000 de capital total, divididos igualmente entre os
oito ativos, o valor final seria aproximadamente US$ 16.323,52.

Interpretacao:
o benchmark evidencia que o resultado de US$ 58,56M da estrategia nao decorre
simplesmente de possuir oito ativos que tiveram buy-and-hold extraordinario.
THO e MUX, por exemplo, terminaram a serie abaixo do preco inicial, embora THO
tenha sido um importante contribuinte nas rotacoes. Isso reforca a natureza
temporal/contextual da assinatura e o valor do timing de entrada/saida.


## Benchmark buy-and-hold do universo completo do cenario de US$ 58,56M

Correcao de escopo: o cenario exploratorio que atingiu aproximadamente
US$ 58,56 milhoes nao possui 58 ativos. Ele e o U67:
- U56 congelado;
- + COLB, AMS e FOXF = U59;
- + THO, WDAY, EXR, XEL, SBFG, PAYX, MUX e SXC = U67.

Foi criado o runner:
calcular_buy_hold_universo_58m_spyder.py

Versao: 1.17.4-dev.1.
Schema: buy-hold-u67-58m-v1.

Objetivo:
calcular US$10.000 de buy-and-hold para cada um dos 67 ativos e somar todos
os montantes finais, usando os mesmos snapshots congelados e a mesma
normalizacao de splits do pipeline, sem executar novo backtest ou treinar
modelo.

O runner produz dois benchmarks:
1. own_history: cada ativo inicia no primeiro fechamento disponivel da propria
   serie;
2. common_window: todos os 67 usam a mesma janela temporal comum.

Capital inicial em ambos: 67 x US$10.000 = US$670.000.

Arquivos:
- buy_hold_u67_own_history.csv;
- buy_hold_u67_common_window.csv;
- buy_hold_u67_summary.json;
- pacote_buy_hold_universo_58m_u67.zip.

A execucao local e necessaria porque COLB, AMS e FOXF pertencem ao snapshot
local pesquisa_expansao_76_b2 e o pacote standalone dos oito ativos nao
contem o OHLC completo dos 59 ativos do baseline.


## Resultado buy-and-hold do universo completo U67

Pacote:
pacote_buy_hold_universo_58m_u67.zip

Versao:
1.17.4-dev.1

Schema:
buy-hold-u67-58m-v1

Universo:
U67 = U59 + THO, WDAY, EXR, XEL, SBFG, PAYX, MUX, SXC.

Todos os 67 ativos possuem inicio em 2016-01-04 e fim em 2026-09-17 no
benchmark produzido, portanto own_history e common_window coincidiram.

Hipotese:
- US$ 10.000 por ativo;
- capital inicial total US$ 670.000;
- fracoes de acoes permitidas;
- sem dividendos reinvestidos, taxas ou impostos;
- precos normalizados por splits pelo pipeline existente.

Resultado:
- capital final: US$ 7.848.543,23;
- lucro: US$ 7.178.543,23;
- retorno agregado: +1.071,42%;
- CAGR aproximado do portfolio equal-dollar: 25,86% a.a.

Concentracao:
- 56 ativos positivos e 11 negativos;
- mediana de retorno individual: +164,48%;
- NVDA: US$ 10 mil -> US$ 2.710.410,87 (+27.004,11%);
- AMD: US$ 10 mil -> US$ 1.967.833,94 (+19.578,34%);
- os cinco maiores contribuidores (NVDA, AMD, CORT, TSLA, AVGO) responderam
  por aproximadamente 74,7% do lucro agregado.

Comparacao com a estrategia U59+8:
- estrategia: US$ 10.000 -> US$ 58.557.157,67;
- buy-and-hold U67 com US$ 670.000 iniciais -> US$ 7.848.543,23;
- mesmo sem normalizar capital inicial, a estrategia termina 7,46x acima;
- normalizando o buy-and-hold para US$ 10.000 de capital total, o final seria
  US$ 117.142,44;
- nessa base comum, a estrategia termina aproximadamente 499,88x acima do
  buy-and-hold equal-dollar.

Correcao do benchmark anterior dos oito ativos:
o calculo anterior dos oito usou a serie ja truncada pela construcao de
features, iniciando em 2016-10-17. O benchmark U67 usa corretamente o OHLC
congelado desde 2016-01-04. Portanto, para comparacoes de buy-and-hold, usar
este benchmark U67 como referencia oficial e nao o calculo parcial anterior
dos oito.


## Assinatura matematica contextual — 1.18.0-dev.1

A pesquisa entrou oficialmente na fase de modelagem estatistica da assinatura,
sem novos backtests durante o ajuste.

Branch:
research/intelligent-asset-signature-v1

Runner:
modelar_assinatura_matematica_spyder.py

Schema:
contextual-marginal-signature-math-v1

Dados contextuais congelados no Git:
- dados/assinatura_matematica/contextual_batch3.csv;
- dados/assinatura_matematica/contextual_smart20.csv.

Pergunta:
quais propriedades contextuais, ja disponiveis antes do replay financeiro,
separam candidatos que aumentam o capital daqueles que degradam ou nao alteram
o universo?

A analise confirmou que um unico modelo monotono sobre todos os 40 candidatos
e inadequado porque existe uma massa de candidatos dormant. No batch3, os 10
candidatos com beats_best_share=0 foram todos nao positivos. Portanto a
assinatura foi formalizada como um processo hurdle em dois niveis:

1. ativacao:
A = 1[beats_best_share > 0]

2. qualidade entre ativos:
S = 1 - mean(
    rank_active(beats_best_share),
    rank_active(abs(score_corr_best)),
    rank_active(score_std)
)

Os ranks sao relativos a cada coorte e calculados somente entre candidatos
ativos. Valores maiores de S representam especialista mais raro, menos
correlacionado com o melhor score do universo e com score mais estavel.

Amostra contextual:
- 40 candidatos totais;
- 30 ativos;
- 10 dormant;
- 11 positivos entre os 30 ativos;
- batch3: 20 totais, 10 ativos, 3 positivos;
- smart20: 20 ativos, 8 positivos.

Controle negativo de features estaticas:
uma regressao logistica com CAGR, volatilidade, drawdown, liquidez,
positive-day-share, momentum, trend efficiency, corr SPY e beta ficou proxima
do acaso ao atravessar coortes:
- batch2 -> batch3: AUC 0,4792;
- batch3 -> batch2: AUC 0,5098.
Isso reforca que a assinatura nao esta em caracteristicas estaticas do ticker.

Replicacao direcional entre candidatos ativos:
- menor beats-share: AUC batch3 0,7381; smart20 0,8021;
- menor abs(corr_best): AUC batch3 0,8571; smart20 0,6146;
- menor score_std: AUC batch3 1,0000; smart20 0,6354;
- maior positive_score_share: AUC batch3 0,6190; smart20 0,5833;
- maior score_mean: AUC batch3 0,5714; smart20 0,5833.

Resultado do score S:
- AUC pooled = 0,8110;
- average precision pooled = 0,7154;
- Spearman S vs efeito de capital = 0,6379;
- p de Spearman = 0,000149;
- bootstrap por coorte, AUC 95% aproximadamente [0,629; 0,950];
- bootstrap por coorte, Spearman 95% aproximadamente [0,400; 0,800].

Por coorte:
- batch3 ativo: AUC 0,9524; AP 0,9167; Spearman 0,6322;
- smart20: AUC 0,7448; AP 0,6882; Spearman 0,5987.

Leave-one-cohort-out do score de uma dimensao:
- treina batch3, testa smart20: AUC 0,7448;
- treina smart20, testa batch3: AUC 0,9524.
Esta e robustez interna, nao validacao externa, pois a composicao do score foi
sintetizada usando as coortes de desenvolvimento ja observadas.

Calibracao logistica de desenvolvimento nos 30 candidatos ativos:
P(deltaCapital > 0 | A=1)
    = logistic(-3,536889 + 5,920287 * S)

Exemplos apenas de calibracao interna:
- S=0,25 -> P aproximada 11,3%;
- S=0,50 -> P aproximada 36,0%;
- S=0,75 -> P aproximada 71,2%;
- S=1,00 -> P aproximada 91,6%.

A camada temporal dos 22 trades dos oito vencedores permanece explicativa e
nao entra no score classificador porque nao existe um controle equivalente de
trades para os candidatos negativos. Ela continua sustentando a morfologia de
micro-fundo/pullback -> recuperacao curta -> saida perto do pico.

Limitacoes obrigatorias:
- smart20 e uma amostra range-restricted porque os 20 ja haviam passado pelo
  filtro anterior;
- batch3 usa features contextuais relativas a U56, mas o alvo financeiro usado
  nesta sintese e relativo a U59;
- ranks por coorte reduzem a incompatibilidade de escala/contexto, mas nao a
  eliminam;
- o score foi sintetizado com dados ja observados;
- nenhuma afirmacao de generalizacao externa pode ser feita ainda.

Decisao:
a estrutura matematica de desenvolvimento esta suficientemente definida para
ser congelada. O proximo replay financeiro, quando ocorrer, nao deve ser usado
para ajustar S. Ele deve ser uma unica validacao prospectiva em ativos
intocados, escolhidos sem acesso ao seu delta de capital.


## Execucao oficial e congelamento da assinatura matematica — v1.18

Pacote oficial recebido e auditado:
pacote_assinatura_matematica_v118.zip

SHA-256 do pacote:
32f7ead20b87fac3c6d9c17ef4fbd45ee27e73394200db2c13b4478d23993f9d

SHA-256 de signature_math.json:
0c35546efcd6e681eabb9af01e99cc6f927f98202c61710f163a38c1e08710d2

A execucao reproduziu os resultados esperados sem discrepancias:
- total contextual=40;
- ativos=30;
- dormant=10;
- positivos entre ativos=11;
- AUC pooled=0,8110;
- AP pooled=0,7154;
- Spearman S vs efeito de capital=0,6379, p=0,000149;
- bootstrap AUC 95%=[0,629; 0,950];
- batch3 AUC=0,9524;
- smart20 AUC=0,7448;
- features estaticas permaneceram perto do acaso entre coortes
  (AUC 0,4792 e 0,5098).

A calibracao logistica pooled foi reproduzida:
P(deltaCapital>0 | A=1)=logistic(-3,536889 + 5,920287*S).

Entretanto, a regressao treinada apenas no batch3 apresentou slope=12,4596 e
a treinada apenas no smart20 slope=4,2899. As probabilidades absolutas,
portanto, ainda nao devem ser tratadas como calibradas externamente.
Curiosamente, a fronteira P=0,5 permaneceu proxima:
- batch3: S=0,6206;
- smart20: S=0,5916;
- pooled: S=0,5974.

Decisao estatistica:
congelar S como SCORE CONTINUO DE ORDENACAO, e nao como probabilidade
operacional calibrada. O arquivo machine-readable congelado e:
dados/assinatura_matematica/signature_v1_18_frozen.json.

Formula congelada:
A = 1[beats_best_share > 0]
S = 1 - mean(
    rank_active(beats_best_share),
    rank_active(abs(score_corr_best)),
    rank_active(score_std)
)

Nao alterar componentes, sinais ou pesos depois de observar novos resultados
financeiros. A referencia de validacao continua U59_WINNER; U67 permanece
resultado exploratorio/post-hoc.

Proximo passo permitido:
uma unica validacao prospectiva usando candidatos cujo capital ainda nao foi
consultado. O conjunto de candidatos pode ser obtido do
intelligent_candidates_ranked.csv ja existente, sem nova busca Alpaca.
A selecao deve ser congelada por S antes de qualquer replay financeiro.


## Congelamento da validacao prospectiva one-shot — 1.18.1-dev.1

Foi criado o runner:
congelar_validacao_prospectiva_v118_spyder.py

Objetivo:
aplicar a assinatura matematica v1.18 ja congelada aos candidatos cujo
resultado financeiro ainda nao foi consultado e congelar uma unica coorte de
validacao antes de qualquer replay.

Este runner NAO executa backtest e NAO consulta capital.

Fonte:
- output/busca_ativos/intelligent_candidates_ranked.csv;
- busca 1.17.0-dev.1, schema intelligent-asset-search-u59-v1;
- referencia de score U59_WINNER;
- assinatura machine-readable congelada em
  dados/assinatura_matematica/signature_v1_18_frozen.json.

Guardas:
- ranked deve conter exatamente 446 candidatos;
- os 20 candidatos Smart20 cujo capital ja foi revelado sao excluidos;
- portanto o pool prospectivo deve conter exatamente 426 candidatos;
- candidatos com dados de score insuficientes nao entram na validacao;
- nenhum resultado de capital pode participar da selecao.

Formula aplicada ao pool prospectivo:
A = 1[beats_best_share > 0]
S = 1 - mean(
    rank_active(beats_best_share),
    rank_active(abs(score_corr_best)),
    rank_active(score_std)
)

Os ranks sao recalculados dentro da nova coorte prospectiva, apenas entre
candidatos ativos, conforme a definicao congelada.

Desenho amostral predeclarado:
- dividir os candidatos ativos em tres tercis por S;
- selecionar 8 do tercil alto;
- selecionar 8 do tercil medio;
- selecionar 8 do tercil baixo;
- selecionar 8 dormant A=0;
- total planejado = 32 candidatos.

A escolha dentro de cada estrato usa SHA-256 deterministico com salt congelado:
tcc-v118-prospective-u59-one-shot-v1

Isso evita escolha manual por ticker, setor, score stage1 ou qualquer
informacao financeira. O objetivo e ter cobertura do espectro do score para
testar monotonicidade, AUC/Spearman e a hipotese hurdle A=0.

Artefatos:
- prospective_validation_cohort_v118.csv;
- prospective_untouched_pool_scored.csv;
- prospective_validation_freeze.json;
- pacote_congelamento_validacao_prospectiva_v118.zip;
- copia local da coorte congelada em
  dados/assinatura_matematica/prospective_validation_cohort_v118.csv.

Regra:
depois que a coorte for congelada, ela nao pode ser alterada por qualquer
motivo relacionado ao resultado financeiro. O proximo runner podera somente
revelar os resultados individuais contra U59 e calcular as metricas
predeclaradas. Nao havera segunda tentativa de selecao apos observar capital.


## Coorte prospectiva congelada e plano estatistico pre-registrado — 1.18.2-dev.1

Pacote de congelamento recebido e auditado antes de qualquer resultado
financeiro:
pacote_congelamento_validacao_prospectiva_v118.zip

SHA-256 do pacote:
287efd76dbc84afb38055fbe25b529633dd5cc2ab68f2c457aa52f2490cb9bc5

SHA-256 exato da coorte:
a1fe00ea2a7c7691d55396366be0375e64294ab43d97d4949a2433d106443978

O freeze reproduziu:
- 446 candidatos no ranked original;
- 426 candidatos financeiramente intocados apos excluir Smart20;
- 414 com dados de score elegiveis;
- 226 ativos;
- 188 dormant;
- 12 insuficientes;
- estratos ativos: 76 high-S, 75 mid-S, 75 low-S.

Coorte one-shot congelada de 32 ativos:

High-S:
HMN, LOCO, ACU, HBCP, LYV, PNFP, TCBI, OVLY.

Mid-S:
WDC, PROV, EDU, RARE, RDCM, RELL, NUE, JOUT.

Low-S:
NSIT, ENTG, PLUG, GOGO, STRA, CETX, IOVA, TANH.

Dormant A=0:
EQIX, FTQI, ILF, IPAC, MTG, QUAL, VAW, XSLV.

A coorte exata e seus metadados foram gravados no Git antes da abertura do
gabarito:
- dados/assinatura_matematica/prospective_validation_cohort_v118.csv;
- dados/assinatura_matematica/prospective_validation_freeze_v118.json.

Plano estatistico pre-registrado:
dados/assinatura_matematica/prospective_validation_plan_v118.json

Endpoint primario:
Spearman entre S e capital_pct_vs_u59 nos 24 candidatos ativos, com hipotese
unilateral rho > 0 e p por 20.000 permutacoes deterministicas.

Criterio de suporte prospectivo:
rho > 0 E p_perm < 0,05.

Endpoints secundarios, sem substituir o criterio primario:
- ROC AUC de S para delta de capital positivo nos 24 ativos;
- gradiente high/mid/low em taxa positiva e mediana do efeito;
- Fisher unilateral high-S vs low-S;
- comportamento dos oito dormant;
- replay economico conjunto dos oito high-S congelados.

O baseline U59 deve reproduzir US$ 30.080.091,008142874 dentro da tolerancia
congelada antes que qualquer efeito individual seja revelado. Se nao
reproduzir, o runner aborta antes da abertura do gabarito.

Runner de abertura do gabarito:
validar_assinatura_prospectiva_v118_spyder.py

Versao:
1.18.2-dev.1

Schema:
prospective-signature-financial-validation-v1

O runner treina os modelos uma unica vez por fold para U59 + 32 candidatos e
reutiliza os caches para os replays individuais. Nao faz nova busca, nao baixa
dados, nao troca candidatos e nao permite segunda selecao.

Regra final:
independentemente do resultado, a assinatura v1.18 nao pode ser reajustada com
esses 32 e continuar sendo chamada de mesma validacao prospectiva.


## Correcao do guard de hash da coorte prospectiva — 1.18.2-dev.2

Erro observado antes de qualquer replay financeiro:
o runner validar_assinatura_prospectiva_v118_spyder.py abortou ao comparar o
SHA-256 bruto do CSV da coorte congelada.

Esperado do pacote original:
a1fe00ea2a7c7691d55396366be0375e64294ab43d97d4949a2433d106443978

Observado no working tree local:
34b17a047821c0cd9de73f934b5afdec75cc0c194d448deb8413a632e5815a18

A causa e de serializacao/working-tree: o hash bruto de CSV e sensivel a
normalizacao CRLF/LF e representacao textual de floats. Esse comportamento
pode mudar quando o arquivo passa pelo Git/checkout no Windows sem que a
coorte cientifica tenha mudado.

A correcao NAO altera a coorte, score, plano estatistico ou resultados.
Nenhum capital havia sido revelado quando a correcao foi feita.

Novo guard:
- preserva o SHA-256 bruto do pacote original para proveniencia;
- calcula adicionalmente um hash semantico canonico dos campos cientificamente
  relevantes, independente de CRLF/LF e com floats canonizados a 12 casas;
- exige o hash semantico congelado:
  37fd32c3fc8c2a424f9a1f26a1fc0764ee8f6361eaec2ac7a13cd60a6439d2b6;
- continua validando ordem, ativos, estratos, activation/dormant, S,
  percentis, beats-share, abs corr, score_std, sessoes e identidades do
  protocolo;
- continua comparando a lista e ordem dos 32 ativos com o freeze JSON.

Versao do runner:
1.18.2-dev.2

Commit da correcao:
614c3e134d08aa914b3481f5d67bd15362a634fc

Como a falha ocorreu antes do baseline U59 e antes da abertura dos 32
resultados, a validade prospectiva permanece intacta.


## Resultado oficial da validacao prospectiva one-shot — v1.18

Pacote analisado:
pacote_validacao_prospectiva_financeira_v118.zip

SHA-256 do pacote:
f05a0efac2b8dff4e85fb641a3fdf9cf50e0271012f93cc33babebe8704b74df

Runner:
1.18.2-dev.2

Schema:
prospective-signature-financial-validation-v1

O U59 reproduziu exatamente o checkpoint congelado:
US$ 30.080.091,008142874
erro relativo = 0.

Endpoint primario pre-registrado nos 24 candidatos ativos:
Spearman(S, capital_pct_vs_u59) = 0,6217391304
p unilateral por 20.000 permutacoes = 0,0006999650
bootstrap 95% de rho = [0,2516; 0,8451]

Pelo criterio pre-registrado rho>0 e p<0,05, o endpoint primario foi
ATENDIDO. Portanto existe suporte prospectivo para a capacidade do score S de
ORDENAR contribuicao marginal de capital entre candidatos ativos.

Porem o resultado deve ser interpretado com precisao:
- AUC para classificar efeito positivo = 0,6421;
- p unilateral por permutacao da AUC = 0,18199;
- average precision = 0,33663;
- portanto a assinatura NAO validou, neste teste, um classificador binario
  confiavel de vencedores positivos.

Resultados por estrato:
High-S:
- 2/8 positivos (25%);
- 1/8 zero;
- mediana delta = -2,827%;
- media delta = -12,993%;
- minimo -63,768%; maximo +9,249%.

Mid-S:
- 3/8 positivos (37,5%);
- mediana delta = -11,532%;
- media delta = -17,431%;
- minimo -59,171%; maximo +19,159%.

Low-S:
- 0/8 positivos;
- mediana delta = -82,032%;
- media delta = -74,189%;
- minimo -99,090%; maximo -39,521%.

Dormant A=0:
- 0/8 positivos;
- 8/8 efeito exatamente zero;
- mediana e media delta = 0.

Vencedores prospectivos individuais:
- ACU +9,2491%;
- HBCP +3,6461%;
- RDCM +4,1743%;
- RELL +19,1592%;
- NUE +4,0780%.

Controle high-S vs low-S em taxa de positivos:
2/8 vs 0/8, Fisher unilateral p=0,2333, portanto nao significativo.

A relacao continua de ranking e fortemente explicada pela separacao entre
candidatos low-S muito destrutivos e os estratos high/mid. Em analise de
sensibilidade pos-resultado, apenas high+mid juntos nao apresentam relacao
monotona interna relevante (Spearman ~0,0176). Esta analise de sensibilidade e
exploratoria e NAO altera o endpoint primario.

Decomposicao exploratoria dos componentes nos 24 ativos:
- menor beats_best_share vs efeito: Spearman orientado +0,7654;
- menor score_std vs efeito: +0,6070;
- menor abs(score_corr_best) vs efeito: +0,1226.
Nao retunar pesos ou remover componentes com base nisso; registrar apenas como
resultado descritivo prospectivo.

Teste economico secundario dos oito high-S juntos:
U59 + high-S8 terminou em US$ 6.660.799,10
delta = -US$ 23.419.291,91
delta percentual = -77,8565%
Sharpe = 2,0980
MaxDD = -31,2200%.

Portanto a assinatura validou ORDENACAO RELATIVA de dano/beneficio individual,
mas NAO validou a regra operacional "adicionar em conjunto os ativos de maior
S". O score parece especialmente eficaz em identificar/evitar candidatos
destrutivos de baixo S. Ele ainda e insuficiente, sozinho, para selecionar um
grupo economicamente superior ao U59.

Conclusao cientifica:
- evidencia prospectiva de generalizacao do ranking continuo: SIM;
- evidencia prospectiva de classificacao binaria de positivos: NAO;
- evidencia prospectiva de que high-S em conjunto melhora o U59: NAO;
- hurdle dormant A=0: fortemente corroborado nesta amostra (8/8 efeito zero);
- nao ajustar a assinatura v1.18 usando estes 32 e chamar o resultado de mesma
  validacao prospectiva.

A v1.18 deve ser mantida congelada como resultado cientifico. Qualquer modelo
novo que use estas 32 observacoes passa a ser uma nova hipotese de
desenvolvimento e exigiria, futuramente, outra amostra realmente intocada.


## Redacao do rascunho tecnico-cientifico do TCC — v1

A linha experimental da assinatura contextual foi encerrada apos a validacao
prospectiva one-shot v1.18. A etapa seguinte passou a ser exclusivamente de
redacao, organizacao e revisao academica, sem novos replays para buscar
resultados melhores.

Foi preparado um primeiro rascunho completo usando o template oficial
"Template TCC - Implementação de Algoritmo(s) de Machine Learning (251, 252)"
e as normas do Manual de Instrucoes e Normas para TCC do MBA USP/Esalq.

Titulo de trabalho provisório:
"Assinatura contextual para seleção de ativos em estratégia de rotação com
aprendizado de máquina".

Estrutura adotada:
- Resumo e Palavras-chave;
- Considerações Iniciais;
- Implementação de Algoritmo(s) de Machine Learning;
- Resultados e Discussão;
- Conclusões;
- Referências.

O texto separou explicitamente:
- U59 como baseline financeiro congelado;
- U59+8 positivos como resultado exploratorio pos-hoc;
- v1.18 como assinatura matematica congelada;
- validacao prospectiva one-shot como teste confirmatorio do ranking continuo;
- falha da AUC binaria como resultado negativo;
- falha economica de U59+High-S8 como evidencia de nao aditividade e de que
  ordenacao marginal e composicao conjunta sao problemas distintos.

Foram incluidos no rascunho:
- tabela da estrutura cronologica dos tres folds;
- tabela comparativa U56, U59, U59+20 Smart20, U59+8 positivos e U59+High-S8;
- tabela dos resultados prospectivos por estrato;
- grafico de capital final dos principais cenarios;
- grafico da assinatura contextual na amostra de desenvolvimento;
- grafico da morfologia temporal de entrada/saida;
- grafico do score congelado S contra efeito financeiro prospectivo.

Referencias cientificas usadas na fundamentacao/discussao:
Ke et al. (2017), Gu et al. (2020), Gama et al. (2014),
Bailey et al. (2017) e Lopez de Prado (2018).

Controle editorial:
- titulo provisório com 14 palavras, abaixo do limite institucional de 15;
- resumo com 229 palavras, abaixo do limite institucional de 250;
- documento renderizado com 13 paginas;
- todas as 13 paginas foram inspecionadas visualmente sem clipping,
  sobreposicao ou tabelas quebradas;
- ficaram em aberto somente dados pessoais/editoriais que precisam de
  confirmacao do autor: nome/titulacao/e-mail do orientador e e-mail do autor.

Artefato produzido fora do repositorio para revisao:
TCC_Assinatura_Contextual_Rascunho_v1.docx

Decisao:
a partir deste ponto, alteracoes devem ser editoriais, bibliograficas ou de
clareza metodologica. Nao executar novas campanhas financeiras para reescrever
a conclusao da validacao v1.18.


## Nova hipotese de filtro de aderencia de capital — v1.19

A pedido do autor, foi aberta uma NOVA linha experimental depois do encerramento
da validacao prospectiva v1.18.

Regra de integridade:
a v1.18 permanece congelada e concluida. Os 32 resultados prospectivos agora
podem ser usados como dados de desenvolvimento apenas porque a v1.19 e uma
hipotese nova. Nenhum resultado v1.19 deve ser descrito como extensao da mesma
validacao prospectiva v1.18.

Motivacao:
a validacao v1.18 mostrou que o score contextual foi muito melhor para ordenar
dano/beneficio e identificar candidatos destrutivos do que para classificar
vencedores positivos. Em particular, o estrato low-S teve 0/8 positivos e
mediana de aproximadamente -82% contra U59.

Foi definida uma nova hipotese de "aderencia ao crescimento de capital" usando
apenas os dois componentes que apresentaram associacao mais forte e replicada
com dano financeiro:
- beats_best_share;
- score_std.

Amostra de desenvolvimento v1.19:
- Smart20: 20 candidatos ativos;
- prospectiva v1.18: 24 candidatos ativos;
- total: 44 candidatos ativos, todos com efeito financeiro medido contra U59;
- os oito dormant da v1.18 foram excluidos do desenvolvimento de oportunidade,
  pois produziram efeito financeiro zero.

Alvo de baixa aderencia:
harm10 = 1[capital_pct_vs_u59 <= -10%].

Score de risco congelado:
R = mean(
    rank_active(beats_best_share),
    rank_active(score_std)
)

Menor R = maior aderencia / menor risco de dano.

Evidencia de desenvolvimento, que NAO e validacao externa da v1.19:
- AUC Smart20 para harm10 = 0,8200;
- AUC prospectiva24 para harm10 = 0,8630;
- AUC pooled = 0,8432;
- Spearman R vs delta de capital = -0,6081;
- p de Spearman = 1,19e-05;
- Mann-Whitney unilateral de risco maior nos harm10: p = 5,88e-05;
- nos 10% de menor risco da amostra de desenvolvimento: 5 candidatos,
  0 harm10 e 4 positivos.

A remocao de abs(score_corr_best) define uma NOVA hipotese e nao altera a
formula v1.18.

Modelo machine-readable:
dados/assinatura_matematica/capital_adherence_filter_v1_19_frozen.json

Runner de congelamento:
congelar_filtro_aderencia_capital_v119_spyder.py

Versao:
1.19.0-dev.1

Schema:
capital-adherence-freeze-v1

Desenho do novo teste:
- partir dos mesmos 446 candidatos da busca 1.17.0;
- excluir os 20 Smart20 e os 32 da validacao prospectiva v1.18, todos com
  capital ja conhecido;
- restam 394 candidatos financeiramente intocados;
- manter apenas candidatos elegiveis e ativos A=1;
- recalcular R dentro desse pool ainda intocado;
- selecionar deterministicamente os 8 menores R;
- congelar a lista antes de qualquer replay financeiro.

Endpoint economico planejado:
capital final de U59 + adherence8 > capital final de U59.

Diagnosticos secundarios:
- numero de candidatos individuais com delta <= -10%;
- efeitos individuais contra U59;
- comparacao descritiva com o resultado exploratorio U59+positive8 =
  US$58.557.157,67.

Nao ha garantia de ganho. Se o grupo falhar, o resultado deve ser preservado e
nao sera permitida segunda selecao usando o mesmo pool apos observar o capital.


## Congelamento oficial e abertura planejada do filtro de aderencia — v1.19

Pacote recebido e auditado antes de qualquer resultado financeiro novo:
pacote_congelamento_aderencia_capital_v119.zip

SHA-256 do pacote:
8919f98a4613a67c55b595441234bc9b74cf0fb1c8ae49498ea7454f230ab38c

SHA-256 bruto da selecao:
dcb6b8746719c24a59b6b9a2c056366446372d9c38811a0885b1abab877ef3b2

O congelamento reproduziu exatamente a evidencia de desenvolvimento v1.19:
- 44 candidatos ativos conhecidos;
- 25 com dano >=10% contra U59;
- AUC Smart20 = 0,8200;
- AUC prospectiva24 = 0,8630;
- AUC pooled = 0,8432;
- Spearman R vs delta de capital = -0,6081;
- p = 1,194e-05;
- Mann-Whitney unilateral = 5,883e-05.

Pool ainda intocado:
- 446 candidatos originais;
- 52 resultados conhecidos excluidos;
- 394 candidatos restantes;
- 382 com dados de score elegiveis;
- 202 ativos A=1;
- 192 dormant ou inelegiveis.

Os oito menores riscos R foram congelados, todos dentro dos ~4% de menor risco
do pool ativo:
1. TRST  R=0,0284653
2. DBB   R=0,0309406
3. MCD   R=0,0445545
4. CSB   R=0,0655941
5. WMK   R=0,0668317
6. WFC   R=0,0680693
7. RJF   R=0,0705446
8. FIBK  R=0,0742574

Nenhum capital desses oito foi consultado para produzir a selecao.

A lista e o freeze foram gravados no Git:
- dados/assinatura_matematica/capital_adherence_cohort_v119.csv
- dados/assinatura_matematica/capital_adherence_freeze_v119.json

Plano financeiro pre-registrado:
dados/assinatura_matematica/capital_adherence_validation_plan_v119.json

Endpoint primario confirmatorio da v1.19:
capital final de U59 + adherence8 > capital final de U59.

Diagnosticos secundarios:
- quantidade de efeitos individuais <= -10%;
- quantidade de positivos;
- mediana/media do efeito individual;
- comparacao descritiva, nao confirmatoria, com U59+positive8 =
  US$58.557.157,67.

Runner de abertura do gabarito:
validar_filtro_aderencia_capital_v119_spyder.py

Versao:
1.19.1-dev.1

Schema:
capital-adherence-financial-validation-v1

O runner reproduz primeiro o U59 em US$30.080.091,008142874 e aborta antes de
qualquer resultado novo se houver divergencia. Em seguida revela os oito
efeitos individuais e executa uma unica vez U59+adherence8, reutilizando o
mesmo treino/caches por fold.

Regra:
nao trocar candidatos, nao alterar R e nao realizar segunda selecao com este
pool depois de observar os resultados.


## Resultado oficial do filtro de aderencia de capital — v1.19

Pacote analisado:
pacote_validacao_aderencia_capital_v119.zip

SHA-256 do pacote:
17c57407b546ac652d815a77140193688bca2bbaafd7002b0b14b5e0c4cb43d9

Runner:
1.19.1-dev.1

Schema:
capital-adherence-financial-validation-v1

O U59 reproduziu exatamente o checkpoint congelado:
US$ 30.080.091,008142874
erro relativo = 0.

Coorte v1.19 congelada:
TRST, DBB, MCD, CSB, WMK, WFC, RJF, FIBK.

Endpoint primario pre-registrado:
capital final U59+adherence8 > capital final U59.

Resultado:
- U59: US$ 30.080.091,01;
- U59+adherence8: US$ 21.972.866,56;
- delta: -US$ 8.107.224,45;
- delta percentual: -26,9521%;
- endpoint primario: NAO ATENDIDO;
- benchmark exploratorio U59+positive8 = US$58.557.157,67 nao foi alcancado.

Metricas U59+adherence8:
- CAGR = 249,1261%;
- Sharpe = 2,26638;
- MaxDD = -31,2191%;
- worst fold = +241,5744%.

Efeitos individuais:
- TRST: 0,0000%;
- DBB: -18,2926% (harm10=true);
- MCD: -4,4643%;
- CSB: -2,9900%;
- WMK: -4,5670%;
- WFC: +1,3289%;
- RJF: -0,2454%;
- FIBK: -2,9152%.

Diagnosticos:
- 1/8 positivo;
- 1/8 harm10;
- 1/8 zero;
- mediana individual = -2,9526%;
- media individual = -4,0182%.

Interpretacao:
o filtro v1.19 teve sucesso descritivo em reduzir a incidencia de candidatos
individualmente catastroficos. Na amostra de desenvolvimento, 25/44 ativos
(56,8%) tinham dano <= -10%; na nova selecao, somente 1/8 (12,5%) teve harm10.
A comparacao e exploratoria, pois a taxa historica nao foi pre-registrada como
endpoint inferencial da v1.19.

Contudo, a hipotese economica principal falhou. A baixa incidencia de dano
individual nao garantiu aumento conjunto de capital. Oito candidatos
individualmente pouco destrutivos ainda alteraram a sequencia de escolhas e
deslocaram oportunidades do U59.

Mecanismo de caminho observado no replay conjunto:
- apenas 24/1547 sessoes (1,55%) tiveram ativo selecionado diferente do U59;
- 17/1547 sessoes (1,10%) selecionaram diretamente um dos oito novos ativos;
- 7 sessoes tiveram mudanca indireta para outro ativo legado;
- TRST e FIBK nunca foram selecionados;
- selecoes diretas por novo ativo: CSB 4, WMK 4, WFC 3, DBB 2, MCD 2, RJF 2;
- os sete trades fechados nos novos ativos somaram aproximadamente
  +US$244.797 de PnL realizado, apesar da estrategia conjunta terminar
  US$8,107 milhoes abaixo do U59.

Isso mostra que o prejuizo conjunto nao veio simplesmente de trades perdedores
nos novos ativos. Ele veio principalmente de custo de oportunidade e
dependencia de caminho: a introducao dos novos candidatos mudou quais ativos
legados foram escolhidos e quando.

Exemplo dominante:
em 23-24 ago. 2022, DBB substituiu DNN por duas sessoes e a razao de capital
(U59+adherence8)/U59 caiu aproximadamente 17,39% apenas nesse episodio.
Outros deslocamentos relevantes envolveram NVDA/XOM, NVDA/AMD, NVDA/MYE e
GKOS. Alguns episodios tiveram efeito positivo, mas o saldo acumulado foi
negativo.

Decomposicao descritiva do log-ratio final:
- sessoes com novo ativo selecionado: log-excess -0,0884;
- sessoes com ativo legado diferente: -0,0643;
- sessoes com mesma selecao: -0,1614, refletindo a propagacao por
  capital/quantidade/caminho apos divergencias anteriores;
- log-ratio total = -0,3141, equivalente a -26,95%.

Conclusao cientifica da v1.19:
- o filtro de aderencia reduziu danos individuais severos: evidencia
  descritiva favoravel;
- a regra conjunta U59+adherence8 melhorou o capital: NAO;
- baixa aderencia individual e compatibilidade conjunta continuam sendo
  problemas diferentes;
- a principal informacao nova e que minimizar risco marginal nao basta:
  e necessario modelar explicitamente o custo de oportunidade causado pelo
  deslocamento dos incumbentes e as interacoes entre candidatos;
- nao retunar R com estes oito e nao realizar segunda selecao no mesmo pool
  chamando-a de confirmacao v1.19.


## Simplificacao dos runners e reproducao oficial U59

A pedido do autor, os runners experimentais de teste acumulados durante as
campanhas anteriores foram removidos do diretorio raiz. A evidencia cientifica
dos experimentos permaneceu preservada em CONTEXTO_MESTRE.md, nos dados
congelados e no historico Git.

Branch mantida:
research/intelligent-asset-signature-v1

Runners principais remanescentes:
- buscar_ativos.py
- reproduzir_experimento.py

O arquivo migrar_snapshot_pesquisa.py foi mantido apenas como utilitario de
manutencao/migracao de snapshots legados; nao integra o fluxo normal.

Arquivos experimentais removidos do diretorio raiz:
- analisar_topos_fundos_ativos_58m_spyder.py
- avaliar_resultado_financeiro_spyder.py
- calcular_buy_hold_universo_58m_spyder.py
- congelar_filtro_aderencia_capital_v119_spyder.py
- congelar_validacao_prospectiva_v118_spyder.py
- modelar_assinatura_matematica_spyder.py
- validar_assinatura_prospectiva_v118_spyder.py
- validar_filtro_aderencia_capital_v119_spyder.py
- buscar_ativos_spyder.py, substituido por buscar_ativos.py
- reproduzir_experimento_spyder.py, substituido por reproduzir_experimento.py

Nova reproducao oficial:
reproduzir_experimento.py

Versao do runner:
1.20.0-dev.1

Schema:
u59-control-reproduction-v1

Objetivo unico:
reproduzir o checkpoint U59 de:
US$ 30.080.091,008142874
a partir de capital inicial de US$10.000.

Universo:
- U56 congelado em dados/pesquisa;
- + COLB, AMS e FOXF de dados/pesquisa_expansao_76_b2;
- total esperado: 59 ativos.

Caracteristicas:
- apenas snapshots CSV congelados;
- sem MongoDB;
- sem Market Cycle Trader;
- sem download Alpaca;
- LightGBM Control;
- calendario fixado no U56 original;
- calibracao walk-forward por fold;
- aborta se o capital final divergir do checkpoint U59 em mais da tolerancia
  congelada.

Artefatos esperados:
- output/reproducao/u59_assets.csv
- output/reproducao/u59_fold_margins.csv
- output/reproducao/u59_predictions.csv
- output/reproducao/u59_trades.csv
- output/reproducao/reproducao_u59.json
- output/reproducao/pacote_reproducao_u59_30m.zip

README, dados/README.md, testes automatizados e GitHub Actions foram
atualizados para os novos nomes e para a reproducao U59.

Validacao automatizada apos a limpeza:
GitHub Actions reproduction-tests no commit
28fa42b74fb9cbe45e5bef25ff809620c697bc1f
concluiu com sucesso em 06 out. 2026.

A execucao financeira completa de reproduzir_experimento.py ainda deve ser
feita localmente para confirmar que o novo runner simplificado reproduz
exatamente os US$30.080.091,008142874.
