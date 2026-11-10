# TCC | Risco absoluto da Winner: entrada em CASH e permanência em posição

**Estado:** hipótese e protocolo de pesquisa; nenhuma regra de produção implementada.
**Control:** backtest de produção congelado em 10/11/2026, 1.584 sessões, 75 ativos elegíveis, 0 sessões em CASH. Preservar integralmente a Winner Control.
**Objetivo:** testar se uma avaliação probabilística e generalizável dos retornos e dos riscos permite **não comprar** e **não permanecer investido** quando o melhor ativo relativo é uma oportunidade ruim em termos absolutos.

## Dois estados distintos

1. **Capital inicialmente em CASH.** O ranking seleciona o melhor candidato relativo, mas a decisão precisa comparar o ganho líquido e o risco estimados desse candidato contra **continuar em CASH**, inclusive quando o universo inteiro cai. Treinar somente com operações executadas da Winner é insuficiente: o Control atual não passou dias em CASH. São necessárias observações históricas por ativo e estados CASH contrafactuais construídos sem usar informações posteriores a cada decisão.
2. **Capital aplicado no ativo atual.** Se ele se deteriora e os demais candidatos estão piores, não há obrigação de manter. Devem competir três ações: **manter o atual**, **rotacionar para um concorrente** e **vender para CASH**, incluindo custo de venda, custo de nova compra, risco de execução e custo de deixar passar uma recuperação. Não assumir que a venda é possível durante halt.

**Falha comum:** a utilidade relativa/ranking não mede automaticamente rentabilidade e risco em relação à alternativa CASH. Uma melhora de ranking, isoladamente, não constitui retorno absoluto esperado positivo.

## Proposta estatística

Modelo **compartilhado por ativo e sessão** (*pooled*, sem treinar um modelo específico por ticker), capaz de estimar uma distribuição preditiva condicional do retorno executável futuro. Começar com LightGBM para quantis 10%, 50% e 90% (e um modelo de média/retorno líquido), mais calibração temporal, usando exclusivamente dados disponíveis até o instante de decisão. Comparar com baseline estatístico/histórico; HMM ou detecção bayesiana de mudança de regime somente se melhorar a previsão fora da amostra.

Variáveis candidatas do histórico já discutido, sempre com versão temporal:
- **Movimento do ativo:** retornos 1, 2, 3, 5, 10, 20, 40, 60 e 120 sessões; volatilidade realizada; ATR; distâncias e inclinações EMA 10/20/50/100; volume e variação de volume; drawdown desde o topo.
- **Mercado/universo:** retornos e volatilidade SPY; amplitude das 75 séries (parcela em alta/abaixo de EMAs); dispersão dos retornos; correlação média entre ativos; concentração da queda e um regime probabilístico de estresse.
- **Estado da carteira:** CASH vs posição comprada; ativo e preço de entrada da posição; ganho/perda desde entrada; MAE/MFE até a decisão; custo de troca; permanência em posição; posição relativa no ranking da Winner.
- **Liquidez e execução:** volume observado, fração da ordem em relação ao volume negociado, spread onde houver dados, ocorrência de halt e qualidade de preços; não imputar preço executável se o mercado estiver suspenso.

Para cada ativo `i`, no instante `t` e horizonte `h`, produzir `E_t[R_{i,h}]`, quantis condicionais e perda esperada na cauda (Expected Shortfall). O modelo deve ser calibrado e testado fora da amostra. O objetivo de pesquisa é comparar **utilidade líquida ajustada ao risco** entre ações admissíveis, por exemplo:

`U_t(i | estado) = E_t[log(1 + R_{i,h})] - lambda * ES_t(perda_i) - custo_t(estado -> i) - penalidade_de_liquidez_t(i)`

`U_t(CASH | estado) = rendimento_estimado_do_caixa - custo_t(estado -> CASH)`

`lambda` e os custos não são sinais técnicos de stop: devem refletir preferências de risco e execução auditável. Suas escolhas devem ser feitas **somente com dados de desenvolvimento**, nunca para maximizar o teste. Se o objetivo principal for patrimônio final, comparar também aprendizado/otimização *decision-focused* em desenvolvimento, sem recalibrar no teste.

As ações possíveis são as mesmas, mas a situação atual altera seus custos:
- **CASH:** escolher CASH ou ativo com maior utilidade estimada.
- **Comprado em A:** escolher manter A, comprar outro B (venda + compra) ou CASH (venda), desde que a operação seja executável.

Nenhum cruzamento EMA nem percentual fixo de perda determina a ação. Esses indicadores são **features**, e a decisão decorre das estimativas condicionais aprendidas.

## Desenho de teste que impede conclusões falsas

1. **Fonte:** exigir OHLCV e eventos corporativos versionados para todos os 75 ativos em todo o período da produção, inclusive até 09/11/2026, com assinaturas verificadas. O ZIP de produção não traz esse OHLCV completo; falhar explicitamente se não houver cobertura. Evitar baixar/reconciliar séries automaticamente durante análise sem congelar a nova fonte.
2. **Causalidade:** features calculadas no fechamento `t`, decisão depois de `t`, primeiro preço potencialmente executável em `t+1` ou na janela intradiária elegível. Labels apenas de preços posteriores à execução hipotética. Usar *purge/embargo* para horizontes sobrepostos e considerar eventos de suspensão, não simular venda às 09h30 de CAPR em 29/07/2026.
3. **Folds:** preservar os três folds cronológicos definidos; seleção de features, modelos e hiperparâmetros só em desenvolvimento/validação, fold final intocado para avaliar. Se o fold final já tiver sido consultado em pesquisas exploratórias, reconhecer o problema e exigir novo teste prospectivo.
4. **Três experimentos separados:** (A) apenas capacidade de previsão e calibração de risco; (B) decisão **CASH -> ativo/CASH**; (C) decisão **posição -> manter/rotacionar/CASH**. Depois testar a combinação de B e C no mesmo capital.
5. **Comparação:** patrimônio final após reinvestimento, CAGR, Sharpe, MaxDD, tempo de recuperação, dias em CASH, custos, rotatividade, ordens não executáveis, quedas evitadas, recuperações perdidas, perda de oportunidade e arrependimento da decisão. Evidência de cauda via cobertura dos quantis e pinball loss contra baseline; intervalos por bootstrap em blocos.
6. **Abordagem de risco:** não propor que análise técnica antecipe notícia/halts. Bloquear resultados que dependam de fills impossíveis ou de liquidez maior que a histórica. Considerar que o universo fixado retrospectivamente limita inferências sobre seleção de ativos prospectiva.

## Limite do módulo experimental já criado

O módulo `reproducao/risco_reversao_ml.py` hoje prevê o risco de **retorno realizado da carteira Control no próximo pregão**, baseado nos diagnósticos exportados. É um diagnóstico inicial, **não resolve nenhum dos dois problemas**: não possui resultados contrafactuais para todos os candidatos e tampouco observações da alternativa CASH. Não deve ser usado como política de compra, manutenção ou venda.

**Próxima entrega técnica:** construir base causal ativo×sessão, validar cobertura de OHLCV até o cutoff de produção, gerar previsões condicionais por ativo em folds cronológicos e avaliar separadamente CASH e posição comprada. Somente então testar impacto sobre patrimônio, sem alterar o sistema operacional de negociacao.

## Referências metodológicas

- Engle e Manganelli, *CAViaR*: https://www.nber.org/papers/w7341
- Regime-switching por HMM: https://www.mdpi.com/1911-8074/13/12/311
- Moreira e Muir, *Volatility Managed Portfolios*: https://www.nber.org/papers/w22208
- Wang et al., *Machine Learning Meets Markowitz* (2026): https://www.nber.org/papers/w34861
- Wolff, *Stock picking with machine learning*: https://onlinelibrary.wiley.com/doi/10.1002/for.3021
