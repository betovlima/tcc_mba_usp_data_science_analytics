# Pesquisa v1: preservação estatística de capital sob risco sistêmico do U75

**Estado:** revisão teórica e desenho de experimento, sem política homologada.  
**Escopo:** apenas o ambiente de pesquisa TCC. O Market Cycle Trader (MCT) permanece operacional e inalterado.  
**Origem:** branch `research/u75-stress-dynamic-cash-v1`, política U75 v1.0.1, proteção coletiva diária.  
**Data de referência:** 31/10/2026 (horário de São Paulo), pacote de backtest `20261101T005056-0acb4c89` (UTC).  
**Princípio obrigatório:** acionar CASH exclusivamente por evidência estatística de risco do **conjunto U75**, nunca pela perda isolada de um ativo escolhido. Nenhum prazo fixo de CASH ou limite numérico arbitrário.

## 1. Evidência observada, sem atribuição causal ainda

No pacote fornecido pelo usuário, `PORTFOLIO_tcc_u77_fixed_challenger` é o identificador do motor base, mas o contrato da execução é explicitamente `u75-daily-market-protection-v1` e `u75_daily_cash_protection_enabled=true`. O universo requerido tem 77 ativos, com **75 efetivos**. O intervalo efetivo é 22/07/2020–30/10/2026, com 1.578 sessões.

- Capital final MCT: US$ 76.710.842,01. Drawdown máximo: **−41,6828%**, em 31/07/2026; patrimônio US$ 37.888.054,69 versus topo de US$ 64.968.968,19.
- Posição efetiva no fundo: **CAPR**. Na sessão: `u75_market_stress=false`, 26 candidatos técnicos, `u75_cash_override=false`.
- Todo o período: 163 observações com estresse coletivo (`u75_market_stress=true`), porém apenas **2 sessões efetivas em CASH** (17/06/2022 e 19/04/2024).
- Em 2026: 209 sessões, 16 observações sob estresse coletivo e **zero sessões em CASH**; em observações de estresse existiam de 6 a 12 candidatos. A regra exige **zero** e cobertura válida de 75/75.
- A referência científica TCC U75 v1.0.1 anterior apresentou Control US$ 80.685.454,79 contra nova proteção US$ 74.907.294,49; ambas as saídas efetivas daquela execução foram desfavoráveis. Não comparar estes valores diretamente ao MCT sem fixar dados, calendário e condições de execução.

**Limitação:** séries de preços completos do U75 e contrafactuais individuais não estão contidos neste relatório. Uma queda do capital de uma posição singular não constitui, logicamente, prova de queda sistêmica do U75. Portanto, uma política apenas sistêmica pode corretamente permanecer investida durante drawdowns idiossincráticos severos. Não há garantia de que seja possível eliminar a perda de CAPR sem violar a regra de escopo.

## 2. Revisão da proteção atual

Código original `engine/u75_market_protection.py`: marca estresse ao combinar participação de retornos positivos em 5 e 20 sessões, participação abaixo da média de 50 sessões e mediana de retorno de 20 sessões, usando limiares exploratórios de 35%, 45%, 55% e −1,5%. Exige adicionalmente inexistência de candidato técnico e a cobertura integral de 75 ativos; tomada de decisão no fechamento e execução na abertura seguinte. A restrição antiga de permanência obrigatória por cinco pregões em CASH foi removida.

As quatro constantes não possuem validação estatística de nível de erro, nível de confiança, capacidade preditiva fora da amostra ou ótimo de utilidade. E o critério zero candidatos torna raríssima a intervenção. Não substituí-las por outros números escolhidos em retrospectiva.

## 3. Métodos com embasamento acadêmico e limitações

### 3.1 Risco de perda na cauda: Expected Shortfall/CVaR

Rockafellar e Uryasev (2000) formularam uma otimização convexa de *Conditional Value-at-Risk* (CVaR), também conhecido como perda média condicional na cauda. Defina o retorno de um índice sintético de amplitude igualitária U75 calculado causalmente:

`R_U(t) = (1/75) * sum_{i=1}^{75} R_i(t)`.

Com perdas `L_U = -R_U` e quantil de risco `alpha`, a grandeza é

`CVaR_alpha(L_U) = min_eta [eta + E[max(L_U - eta, 0)] / (1-alpha)]`.

Aproximar a distribuição condicional `L_U,t+1 | F_t` somente com informações disponíveis no fechamento t, incluindo alargamento da cauda e incerteza do estimador. O parâmetro de tolerância `alpha` e limite de perda são escolhas de risco **pré-declaradas**; não surgem automaticamente de uma prova matemática. O método dá solução matemática para o problema **assumido**, não garante acerto de previsão ou lucro na prática.

Hipótese H1: CVaR condicional do U75 em t antecipa uma frequência maior de grandes perdas agregadas em t+1 do que os quatro limiares da regra atual.

### 3.2 Dependência multivariada e correlação de queda

Longin & Solnik (2001) encontraram correlações mais elevadas em mercados internacionais em baixa, não simplesmente em toda volatilidade; Ang & Chen (2002) documentaram assimetria na dependência de carteiras de ações nos movimentos negativos. Isto sugere medir **a perda de diversificação do U75**, não apenas a contagem de retornos positivos.

Estimar séries transversais dos 75 retornos diários, volatilidades e correlação conjunta especialmente na cauda negativa. Para estabilizar a matriz `Sigma_t` em dimensão 75 e histórico finito, testar shrinkage Ledoit–Wolf (2004). Evitar estimar inversa de covariância amostral instável em janelas curtas.

Hipótese H2: a combinação de amplitude de queda e dependência entre ativos fornece informação adicional à amplitude isolada para predizer um regime de risco coletivo.

### 3.3 Turbulência multivariada: distância de Mahalanobis

Kritzman & Li (2010) utilizaram a distância de Mahalanobis de vetores de retornos para medir turbulência financeira:

`T_U(t) = (r_t - mu_t)' * Sigma_t^(-1) * (r_t - mu_t)`,

com `r_t` de dimensão 75, e `mu_t`/`Sigma_t` estimados exclusivamente no passado, preferencialmente com regularização da covariância. A medida capta comportamento incomum, mas **não distingue sozinha uma alta coordenada de uma baixa coordenada**. Para proteção, condicioná-la a um indicador de amplitude negativa e perdas no U75.

Hipótese H3: turbulência multivariada + breadth negativo identifica alterações no risco coletivo sem fazer inferência a partir de um ativo isolado.

### 3.4 Mudança de regime probabilística

Adams & MacKay (2007) desenvolveram *Bayesian Online Changepoint Detection*: distribuição posterior do tempo desde a última mudança de parâmetros em uma série. Como entrada, utilizar indicadores **agregados U75** (retorno do índice sintético, amplitude, volatilidade, correlação, perda de cauda), com modelo de observação e prior do hazard declarados e calibrados dentro do treino. Exigir persistência estatística empiricamente medida, não permanência fixa em CASH.

Hipótese H4: a probabilidade causal de mudança de regime melhora a calibração de probabilidade de risco coletivo, mas pode ter atraso ou falsos alarmes.

### 3.5 Gestão de exposição sob risco de drawdown

Grossman & Zhou (1993) deduzem políticas ótimas sob um limite relativo ao pico de riqueza em um modelo contínuo específico. Busseti, Ryu & Boyd (2016) desenvolvem uma aproximação convexa que impõe um limite matemático de probabilidade de drawdown na distribuição assumida; a garantia depende da validade dos pressupostos. Moreira & Muir (2017) documentam melhorias históricas de Sharpe com exposição reduzida quando a volatilidade sobe; não demonstram garantia de preservação em todos os mercados.

No nosso estudo, essas abordagens só são elegíveis se o **gatilho primário** for sistêmico U75. Um stop diretamente por CAPR ou por drawdown da única posição descumpre a restrição do estudo. Reduzir continuamente a fração investida também implicaria mudar as regras de execução atuais; avaliar apenas como braço de pesquisa separado, sem equipará-lo à política binária CASH.

## 4. Desenho experimental sem limiares escolhidos a posteriori

- **C0**: U75 v1.0.1 original, intocada.
- **C1**: Control sem proteção sistêmica, preservando modelo, custos, calendário, perfil, universe e elegibilidade.
- **S1**: indicador probabilístico coletivo de risco de cauda baseado em retornos U75 e CVaR condicional.
- **S2**: S1 + correlação assimétrica/turbulência U75 com shrinkage e breadth negativo.
- **S3**: S2 + componente bayesiano de mudança de regime (apenas se S1/S2 forem promissores). Se a complexidade não melhorar estabilidade, descartar.

Formalização geral de elegibilidade: `CASH(t+1) = 1` **somente** se, com informações `F_t`, a probabilidade calibrada do evento adverso **agregado U75** exceder o limiar escolhido *ex-ante* pelo orçamento de risco ou função de custo `C(false-negative), C(false-positive)`, e houver cobertura de dados admitida pelo protocolo. O estado deve ser reavaliado uma vez por pregão; o risco sistêmico é a única causa possível para saída. Na falta de dados suficientes, reportar `indeterminado` e aplicar uma regra de segurança definida antes de testar, não reconstruir dados faltantes.

### 4.1 Protocolo de dados e causalidade

Usar o mesmo universo efetivo U75 congelado, barras RAW/SIP Alpaca, Corporate Actions, calendário de negociação e exclusões estruturais reproduzíveis. Não reconstruir ativos excluídos. Registrar checksums de cada série; não comparar métricas de snapshots distintos. Features e distribuição condicional só até fechamento t; ordens simuladas na abertura de t+1. Emulação rigorosa do spread, taxas, gaps de abertura, liquidação e ausência de rendimento sobre CASH se esse for o contrato de comparação.

### 4.2 Treino, calibração e teste

Nested expanding walk-forward com purga/embargo conforme horizontes usados. A janela externa é intocada; no treino interno estimar parâmetros de distribuição, shrinkage, hazard e ponto de decisão maximizando utilidade **ajustada a custo do risco** com critério previamente registrado. Submeter decisões completas e não apenas previsão de classe à avaliação. Nada de ajuste manual retrospectivo nos eventos de 2022, julho ou outubro de 2026.

### 4.3 Critérios de evidência

Medir: capital final e crescimento logarítmico líquido, CAGR, Sharpe, Sortino, máximo drawdown, Expected Shortfall dos retornos do **portfólio**, pior sessão, duração de recuperação, sessões CASH, falso alarme/risco perdido, perdas evitadas e ganho não realizado por intervenção. Reportar ICs por bootstrap em blocos temporalmente adequados, comparações pareadas com mesma janela e distribuição de resultados por fold. Ao comparar muitas candidatas, aplicar Hansen (2005) SPA ou White (2000) Reality Check, com atenção a fragilidade de amostras de crises raras. Reportar também Probability of Backtest Overfitting (PBO) se houver grade de múltiplos experimentos, sem tratá-lo como prova de previsão futura.

**Critério de adoção:** não basta reduzir drawdown de 2026; a melhoria deve ser estável fora da amostra, não depender de um único evento, reduzir risco de cauda com custo de crescimento documentado e respeitar o mandato U75-only. Caso contrário, manter o controle.

## 5. Bibliografia primária verificada

1. Rockafellar, R. T.; Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk*. https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf
2. Ledoit, O.; Wolf, M. (2004). *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices*. Journal of Multivariate Analysis 88, 365–411. https://doi.org/10.1016/S0047-259X(03)00096-4
3. Longin, F.; Solnik, B. (2001). *Extreme Correlation of International Equity Markets*. Journal of Finance 56, 649–676. https://doi.org/10.1111/0022-1082.00340
4. Ang, A.; Chen, J. (2002). *Asymmetric Correlations of Equity Portfolios*. Journal of Financial Economics 63, 443–494. https://doi.org/10.1016/S0304-405X(02)00068-5
5. Kritzman, M.; Li, Y. (2010). *Skulls, Financial Turbulence, and Risk Management*. Financial Analysts Journal 66(5). https://doi.org/10.2469/faj.v66.n5.3
6. Adams, R. P.; MacKay, D. J. C. (2007). *Bayesian Online Changepoint Detection*. https://arxiv.org/abs/0710.3742
7. Grossman, S. J.; Zhou, Z. (1993). *Optimal Investment Strategies for Controlling Drawdowns*. Mathematical Finance 3(3), 241–276. https://doi.org/10.1111/j.1467-9965.1993.tb00044.x
8. Busseti, E.; Ryu, E.; Boyd, S. (2016). *Risk-Constrained Kelly Gambling*. Journal of Investing 25(3), 118–134. https://web.stanford.edu/~boyd/papers/kelly.html
9. Moreira, A.; Muir, T. (2017). *Volatility-Managed Portfolios*. Journal of Finance 72(4), 1611–1644. https://doi.org/10.1111/jofi.12513
10. Hansen, P. R. (2005). *A Test for Superior Predictive Ability*. Journal of Business & Economic Statistics 23(4), 365–380. https://doi.org/10.1198/073500105000000063
11. White, H. (2000). *A Reality Check for Data Snooping*. Econometrica 68(5), 1097–1126. https://doi.org/10.1111/1468-0262.00152

**Conclusão de pesquisa, não homologação:** nenhuma técnica oferece proteção universal contra gaps e perdas extraordinárias sob distribuição desconhecida; fundamentação matemática prova propriedades **condicionais do modelo**, enquanto a utilidade para o U75 depende de validação causal fora da amostra. Não alterar MCT ou o Winner com base apenas nesta revisão.
