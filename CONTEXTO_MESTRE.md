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
