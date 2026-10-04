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
