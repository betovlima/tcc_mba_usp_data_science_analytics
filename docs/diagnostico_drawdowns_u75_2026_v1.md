# Diagnóstico do risco da posição versus risco coletivo do U75 (2026)

**Versão documental:** `u75-drawdown-risk-decomposition-v1.0.0`  
**Fonte:** `market_cycle_trader_20261030T232618-19911c69.zip`  
**SHA-256:** `dd9d27b847c318ce7d907c1b819fa35314c8cbf1d7f8f32dcebca0645452858d`  
**Fonte do cálculo:** série `strategy_equity`, `selected_asset` e diagnósticos de mercado em `PORTFOLIO_tcc_u77_fixed_challenger_predictions.csv`; métricas do mesmo ZIP.  
**Reprodução:** `strategy_equity / strategy_equity.cummax() - 1`, com identificação dos picos e fundos por data. Não refaz os modelos, não consulta o MCT operacional e não utiliza preços externos.

## Por que o gráfico mudou nossa hipótese

O gráfico do usuário marca duas quedas fortes, no início de junho e em outubro de 2026. O CSV confirma **um terceiro episódio ainda mais severo em julho**, que não devemos omitir. Esses episódios **não demonstram, por si só, que os 75 ativos entraram simultaneamente em regime de queda**: a carteira é concentrada na posição selecionada. Temos de distinguir a **deterioração do U75 como conjunto** da **cauda específica de um ativo escolhido**.

**Check da fonte:** execução com 75 ativos elegíveis (apesar da pasta ter `u77` no nome), capital em 30/10/2026 de **US$ 82.628.123,40844664**, `market_exposure=1.0`, `cash_days=0`, MaxDD do backtest **−41,682843%** e 209 sessões em 2026. São dados de simulação, não ordens reais.

## Três episódios que devem orientar as perguntas de risco

| Episódio | Pico | Fundo | Queda entre pico e fundo | Posição observada | Breadth U75 de 20 pregões, pico→fundo | SPY 20 pregões, pico→fundo |
|---|---:|---:|---:|---|---|---|
| 28/05 a 08/06/2026 | US$ 63.308.741,12 | US$ 50.183.758,20 | **−20,7317%** | AMS, oito sessões | **56,00%→53,33%** positivos | **+5,45%→+0,82%** |
| 27/07 a 31/07/2026 | US$ 69.980.504,04 | US$ 40.810.640,31 | **−41,6828%** | WDAY no pico, CAPR nos quatro pregões seguintes | **54,67%→53,33%** positivos | **+0,63%→−0,55%** |
| 30/09 a 26/10/2026 | US$ 109.905.024,60 | US$ 74.340.984,03 | **−32,3589%** | CAPR em 15 de 19 sessões; WDAY e ADEA em 2 cada | **33,33%→42,67%** positivos | **−0,37%→−1,48%** |

**Interpretação correta:** `universe_breadth_20` é a fração dos 75 ativos cujo retorno acumulado de 20 sessões é positivo, não a fração que subiu no pregão. A mudança entre os pontos extremos **não estabelece** o momento causal de início da deterioração. `spy_return_20` também é um indicador retrospectivo disponível no fechamento, não o retorno futuro da decisão.

### Maio–junho: AMS

- AMS permaneceu como `selected_asset` durante as **oito sessões** desde 28/05 até 08/06; também aparece como `raw_best_asset` nelas.
- A amplitude de queda da **posição concentrada** foi grande, mas a breadth de 20 pregões permaneceu acima de 50% nos dois extremos; os indicadores do pacote não sustentam uma crise coletiva evidente.
- A mesma posição teve fortes recuperações em sessões intermediárias e posteriores; uma saída definida com a informação do fundo seria vazamento. Precisamos medir riscos **disponíveis antes** das perdas e custo dos alarmes falsos.

### Julho: WDAY→CAPR e pior drawdown

- O Control estava em WDAY em 27/07 e mudou para CAPR em 28/07; CAPR era o Top-1 registrado nas sessões seguintes.
- Em **30/07**, o patrimônio simulado caiu **36,2253% em apenas um pregão** enquanto `selected_asset=CAPR`.
- Entre 27/07 e 31/07, a carteira perdeu **41,6828%**, mas a breadth de 20 pregões no fundo ainda estava em **53,33%**.
- Um detector exclusivo de risco coletivo, baseado apenas na queda agregada do U75, pode **não sinalizar** um choque específico da posição. O risco marginal da posição selecionada deve ter estimativa distinta.

### Setembro–outubro: CAPR em ambiente de breadth enfraquecida

- No **pico em 30/09**, só **33,33%** dos ativos tinham retorno positivo em 20 sessões; em cinco sessões, apenas **25,33%**.
- CAPR estava na posição em 30/09 e continuou predominando no período (15 de 19 sessões). Houve rotações transitórias para WDAY e ADEA.
- O fundo em 26/10 ficou **32,3589%** abaixo do pico. A carteira recuperou parte da perda, mas terminou 30/10 ainda **24,8186% abaixo** do topo de 30/09.
- Ao contrário de maio e julho, há sinais descritivos de fraqueza transversal no início do episódio. Não significa que uma regra baseada apenas em breadth teria gerado proteção lucrativa: faltam teste temporal, política, execução e contrafactual.

## Formulação científica corrigida: dois riscos condicionais

**A. Risco coletivo**, estimado sobre os 75 ativos em t: breadth, fator comum robusto (PCA), covariância com shrinkage, coexcedências de cauda, mudança de distribuição (CUSUM / estados de Markov filtrados), Expected Shortfall da distribuição agregada.

**B. Risco da posição efetiva e do Top-1**, estimado em t: risco de cauda individual, estabilidade da utilidade prevista, decomposição da exposição em um fator de mercado U75 excluindo o próprio ativo e componente residual. Por exemplo, regressão temporal robusta (janelas de treino anteriores):

`r_(a,t) = beta_(a,t-1) * r_(U75\a,t) + epsilon_(a,t)`.

O termo `epsilon` representa a variação específica não explicada pelo fator coletivo. Estimar risco de eventos extremos tanto do fator quanto do residual, com validação cronológica e sem inventar thresholds. A existência de residual grande **ex post** não prova que poderia ser previsto em t.

**C. Escolha alternativa contextual**, somente depois de medir A e B: comparar a melhor opção interna (Top-1), a **posição efetiva** que precisa ser vendida ou mantida, os externos elegíveis em t e CASH, respeitando o custo de transição e a distribuição condicional dos resultados. Um ativo externo não é bom por ser apenas parecido ou mais forte no momentum. Não transformar CASH em stop fixo de um ativo isolado; preservar a política de reavaliação diária e permitir CASH apenas quando o risco do universo/portfólio justificar a não exposição.

## O experimento a implementar

1. Preparar painel causal por sessão do fator coletivo, dos retornos de cada ativo, da posição efetiva e do Top-1; calcular esses estados **sem consultar o retorno do pregão seguinte**. Corrigir/auditar o problema de elegibilidade de t+1 herdado do ranking antes de alegar qualquer evidência independente.
2. Calcular estimativas de perdas extremas condicionais do **fator U75 e da posição**, além de detecção de mudanças de regime, sem misturar alvo de retorno do índice com drawdown da carteira.
3. Fazer previsões **shadow** de risco para 1, 5 e 10 pregões; auditar acertos **antes** do início das quedas, falso alarme, lead time, calibração e robustez por regime. Os três episódios aqui listados são **diagnósticos já inspecionados**, não podem ser reutilizados como amostra independente para ajustar e depois anunciar acerto OOS. Usar teste cronológico genuinamente não consultado ou novos dados posteriores.
4. Quando sinais estiverem validados, buscar alternativas externas e simular uma política realista e sequencial com decisões no fechamento t, execução na abertura t+1, posição efetiva, overnight, preços, custos e CASH sem trava temporal. Medir ES95 de perdas, drawdown por episódio, perdas evitadas e oportunidades sacrificadas. Capital final apenas complementar.

O objetivo não é recuperar US$ 587 milhões, otimizar hiperparâmetros para capital terminal ou recriar o CARO. O relatório documenta hipóteses matemáticas sustentadas por **observações da execução enviada**, sem afirmar que tais alertas já funcionam.
