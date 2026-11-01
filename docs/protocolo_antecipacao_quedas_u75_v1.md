# Protocolo científico: antecipação de períodos desfavoráveis do universo U75

**Versão do protocolo:** `u75-downside-regime-avoidance-v1.1.0`  
**Branch:** `research/u75-downside-regime-avoidance-v1`  
**Natureza:** correção de objetivo e plano de pesquisa. **Não é um algoritmo executado, nem resultado validado.**  
**Ambiente:** exclusivamente TCC. Não alterar nem consultar o MCT operacional para tomar decisões ou treinar modelos.

## 1. Objetivo vigente

**Antecipar condições de queda coletiva do universo original U75 usando somente dados conhecidos até o fechamento t, reduzir a exposição a períodos desfavoráveis e selecionar uma alternativa que preserve melhor o patrimônio.**

A política desejada é: avaliar U75 → se permanecer atrativo, seguir o Control → caso haja risco coletivo adverso, pesquisar opções externas ao U75 → se existir alternativa cuja vantagem e risco sejam sustentados por evidência, considerar a rotação → se nenhuma alternativa justificável estiver disponível e a exposição ao U75 continuar desfavorável, considerar CASH → reavaliar tudo no fechamento seguinte.

**Não usar US$ 587.165.941,13, US$ 80.685.454,79 ou qualquer capital histórico como alvo da otimização.** Em suas janelas históricas respectivas, estes valores podem funcionar somente como checkpoints de identidade/paridade técnica do Control congelado. O objetivo econômico é **perda evitada e risco reduzido em períodos previamente identificáveis**, descontado o custo das proteções equivocadas, não maximização do capital terminal histórico.

### Regras já fixadas pelo autor

- **Uma avaliação de rotação por pregão**. Sinal no fechamento t, execução na abertura t+1, posição efetiva acompanhada ao longo da sessão. Não supor entrada e saída de CASH no mesmo dia.
- **CASH apenas pela condição do universo/risco do portfólio**, nunca como stop arbitrário de um único ativo. Se não existe opção interna nem externa satisfatória, pode ser a melhor alternativa. Sem permanência mínima/máxima fixada.
- Distinguir queda coletiva do U75, queda isolada de ativo e queda específica da posição efetivamente mantida pelo Control; uma proteção coletiva não deve ser julgada pelo erro em evento puramente idiossincrático.
- Nenhum limiar, período de permanência, penalidade, corte de risco ou universo de candidatos escolhido **depois de olhar o teste**. Parâmetros científicos precisam de fundamento econômico/estatístico e determinação causal nas janelas anteriores.
- Dados RAW/SIP e Corporate Actions em CSV por ativo, sem banco, no TCC. Não reconstruir/colar ("bridge") ativos com identidade ou série estruturalmente inválida; registrar sua exclusão reprodutível.
- Não promover hipótese exploratória ao MCT. Não alterar Control congelado para fazer a nova política parecer melhor.
- Não assumir que os 52 atributos sejam suficientes, nem otimizar o LightGBM apenas em capital final, nem fazer busca CARO para perseguir checkpoints históricos.

## 2. A diferença entre as perguntas científicas

| Pergunta | Unidade de avaliação | Resultado a medir |
|---|---|---|
| **A: É possível antecipar a deterioração coletiva do U75?** | Data de decisão t e U75 inteiro | Calibração do risco previsto, acerto/alarme falso, antecedência à queda, robustez temporal |
| **B: Nos períodos de risco elevado, existe opção externa que justifique sair do U75?** | Data, estado do U75, alternativa externa ou CASH | Vantagem contrafactual do episódio, risco de perda, custos e oportunidade sacrificada |
| **C: A proteção reduz o dano à carteira?** | Política sequencial completa em mesmo período/dados | Drawdown máximo, ES95, perdas evitadas, tempo de recuperação, rotações, custos, dias CASH e capital terminal como métrica secundária |

**A é uma previsão, B é uma decisão contextual, C é um efeito econômico.** Não usar o desempenho de uma única sessão abertura→fechamento como substituto de retorno de episódio ou proteção completa do capital.

## 3. Formalizar o risco com dados causais

Definir o universo U75 congelado de cada protocolo, com identidade, calendário e hashes documentados. Para medir o regime do próprio universo, propor um índice equal-weight dos retornos dos 75 ativos **apenas como observação agregada**, sem confundir com patrimônio da carteira concentrada. Conferir preços válidos e cobertura sem usar t+1 como elegibilidade no instante t.

Atributos candidatos já existentes: proporção de ativos com retornos 5/20 sessões positivos, fração abaixo da média de 50 sessões, mediana dos retornos 20 sessões, dispersão e volatilidade cross-sectional, correlações condicionais, comportamento conjunto nas piores sessões. Os 52 `ROTATION_FEATURES` podem alimentar modelos de ativos, mas não substituir medidas agregadas de queda sistêmica.

Alvos avaliáveis **somente após maturação**, a serem predefinidos antes do teste:
- Probabilidade de retorno coletivo negativo em horizontes econômicos relevantes;
- Probabilidade e intensidade de perda extrema coletiva (limites estimados em dados de treino, jamais escolhidos ao observar o teste);
- Risco previsto da política em exposição, com estado efetivo da carteira conhecido em t;
- Duração e amplitude de episódios de deterioração, usados apenas como rótulos futuros de avaliação.

Avaliar discriminação e calibração (Brier, curvas de calibração, precisão/recall de eventos raros, falsos alarmes) e **lead time**. Para distribuição de retornos, reportar Expected Shortfall empírico e intervalos por reamostragem em blocos. Não tratar condição de estresse técnico no fechamento como previsão validada de queda futura.

## 4. Política de proteção a comparar

**P0 — Control intocado:** rotação interna U75, referência pareada, sem novo protetor.

**P1 — Sinal de risco somente (shadow):** mesmas posições da P0, gerar previsões de risco e alertas; não opera. Mede qualidade do alerta sem interferência econômica.

**P2 — Proteção sistêmica via CASH:** permitir CASH somente quando o risco coletivo e a ausência de opção suficientemente favorável justificarem. Reavaliar diariamente; sem prazo artificial em CASH. Teste separa os efeitos das saídas desnecessárias.

**P3 — Proteção sistêmica com busca externa:** quando o U75 estiver desfavorável, avaliar ativos externos disponíveis naquela data, com vantagem prevista/risco e custos. Pode manter U75 se a alternativa não compensar. CASH somente quando não houver opção justificável.

O mesmo calendário, dados, custos, estado de portfólio e protocolo de decisão devem valer para P0–P3. A alternativa externa exige reprodução da política completa de holding/rotação; não basta comparar a oportunidade intradiária do dia seguinte.

## 5. Desenho de pesquisa e critérios de decisão

1. **Primeiro, auditoria temporal:** remover ou isolar a dependência herdada de preço da próxima sessão na elegibilidade do motor Control. A paridade de capital não comprova causalidade.
2. **Descoberta histórica separada de avaliação:** definir informação disponível em cada data, inclusive se o ativo externo já estava listado/negociável e se seu histórico tinha cobertura; não selecionar um pool retrospectivo apenas pelos ativos vencedores.
3. **Treino aninhado e cronológico:** usar somente datas anteriores à avaliação para seleção de atributos, calibração de risco, busca de candidatos e hiperparâmetros. Embargo coerente com a maturação do alvo. Testes cronológicos posteriores intactos. Separar resultados exploratórios de evidência prospectiva nova.
4. **Primeira etapa somente shadow:** validar se o detector identifica quedas futuras do U75 com antecedência e false positives economicamente suportáveis. Comparar com indicadores técnicos já existentes e um baseline ingênuo; sem buscar retorno máximo.
5. **Depois, teste de decisão:** rodar P0–P3 numa simulação de posições efetivas e custos reais, incluindo entrada/saída e reentrada em CASH, sem inventar preços de execução.
6. **Avaliação econômica primária:** piora/melhora em ES95, distribuição de perdas nos episódios de U75 adverso, drawdown por episódio, perdas efetivamente evitadas versus custo de proteção, recuperação e retorno perdido por alertas falsos. Capital terminal, CAGR e Sharpe são contexto secundário.
7. **Robustez:** resultados por ano/regime, intervalos por blocos, estabilidade da calibração do detector, análises por eventos reais e correção de múltiplas tentativas quando aplicável. Não escolher parâmetros pela janela testada.

## 6. O que os resultados já existentes significam

- O diagnóstico anterior identificou **163 dias classificados como estresse coletivo**, com apenas **duas intervenções CASH** na regra restritiva (estresse E zero candidatos técnicos). Essas intervenções não melhoraram o resultado histórico do Control no replay pareado. O sinal é descritivo, não uma previsão de queda calibrada.
- Na auditoria operacional de 1.578 sessões, **somente uma das dez piores sessões de retorno da carteira** coincidiu com o sinal sistêmico. Isso não demonstra por si que o detector deveria prever quedas isoladas de ativos como CAPR.
- O teste de 24 ativos externos por proximidade e regressão Ridge foi global por pregão e não condicionou descoberta/seleção às janelas de risco coletivo do U75; portanto **não respondeu à hipótese de proteção durante regime adverso**. Sua média foi negativa, o que desaconselha promoção daquela regra.
- O piloto de reotimização de hiperparâmetros U75 via LHS + GP/EI usou objetivo de performance de rotação, não qualidade de previsão de queda coletiva nem ES95 evitado. A nova configuração perdeu frente aos parâmetros congelados sob a mesma calibração. **Isso não refuta a hipótese de antecipar riscos do U75**, apenas não apoia a promoção daquela reotimização.
- Os checkpoints US$ 587,17 milhões (snapshot até 06/10/2026) e US$ 80,69 milhões (Strategy #30 até 28/10/2026) são bases de reprodução diferentes. Nunca comparar seus valores finais como se fossem mesmas janelas e mesmos inputs.

## 7. Entrega experimental seguinte

Criar um executor Spyder **independente**, com seções `# %%`, sem downloads involuntários, sem MCT, usando CSVs congelados e manifestos. Inicialmente apenas a fase **P1 shadow**:
- `indice_u75_diario.csv`: séries coletivas e atributos disponíveis em t;
- `risco_previsto_oos.csv`: probabilidade/risco para horizontes predefinidos, decisões e maturação de labels;
- `episodios_queda_u75.csv`: eventos de queda coletiva com indicadores de antecedência;
- `calibracao_risco.csv`, `metricas_por_fold.csv`, `falsos_alertas.csv`;
- relatório de cobertura/hashes, limitações, figuras e ZIP de auditoria.

Só avançar para P2–P3 após verificar se existe previsão de risco coletiva minimamente confiável fora do ajuste. Evitar definir limiar numérico arbitrário para disparo de CASH, veto ou compra. Nenhum resultado novo é alegado por este protocolo.

## 8. Métodos matemáticos e estatísticos prioritários (v1.1.0, 01/11/2026)

**Registro de alinhamento com os diálogos anteriores:** os mecanismos de proteção precisam decorrer de quantidades estatísticas calculáveis e auditáveis sobre o universo U75 completo, e não de travas manuais de CASH ou de uma otimização sobre capital final. O classificador supervisionado, se estudado, é benchmark e não pré-requisito de abordagem matemática.

### 8.1 O objeto é o risco COLETIVO, não a perda de um ativo isolado

Para os retornos diários \(r_{i,t}\) dos 75 ativos e conjunto \(V_t\) de dados elegíveis estritamente conhecidos em \(t\):

- Força interna: \(B_t=|V_t|^{-1}\sum_{i\in V_t}\mathbb{1}[r_{i,t}>0]\).
- Retorno equal-weight *descritivo*: \(\bar r_t=|V_t|^{-1}\sum_{i\in V_t}r_{i,t}\).
- Dependência coletiva: proporção da variância de retornos explicada pela primeira componente principal, com fator estimado em janela móvel anterior à data de decisão.
- Frequência conjunta de perdas extremas: coexcedência da cauda negativa e suas incertezas, estimadas somente com observações anteriores à decisão.

Não substituir o universo congelado por outro devido à falta de dados futuros. Critérios de cobertura devem ser definidos antes do teste; lacunas precisam aparecer em CSV de auditoria, sem classificação automática de risco baixo.

### 8.2 Covariância robusta e turbulência multivariada

Estimar \(\widehat\mu_{t-1}\) e \(\widehat\Sigma_{t-1}\) em janelas anteriores a \(t\), utilizando shrinkage de Ledoit-Wolf ou outra estimativa robusta predefinida. Definir turbulência observável no fechamento:

\[
D_t^2=(\mathbf r_t-\widehat\mu_{t-1})^\top
\widehat\Sigma_{t-1}^{-1}
(\mathbf r_t-\widehat\mu_{t-1}).
\]

Uma turbulência alta caracteriza retorno incomum em múltiplos ativos, **não necessariamente queda**: cruzar com \(\bar r_t\), \(B_t\), comportamento conjunto das caudas e direção do componente comum. Para evitar instabilidade numérica, registrar condicionamento da matriz e eventuais exceções estatísticas. Não usar \(\Sigma\) calculada com retornos do período de teste inteiro.

Fundamentação: Ledoit e Wolf (2004), *A well-conditioned estimator for large-dimensional covariance matrices* (Journal of Multivariate Analysis, 88, 365–411); medida de turbulência de Mahalanobis da literatura de risco sistêmico (Kritzman e Li).

### 8.3 Expected Shortfall condicional: quantificar risco de cauda

Para perda futura de horizonte \(h\), \(L_{t,h}\), estimar **usando apenas informação \(\mathcal F_t\)**:

\[
\mathrm{ES}_{\alpha,t,h} =
\mathbb E[L_{t,h}\mid L_{t,h}\geq
\mathrm{VaR}_{\alpha,t,h},\mathcal F_t].
\]

Em distribuições discretas, usar a definição integral generalizada da cauda, com massa no quantil tratada corretamente. Diferenciar ES condicional *estimado em t* de ES realizado *ex post*. Primeiro definir se o alvo é a perda do índice equal-weight do U75, da posição efetiva do Control ou de ambas, registrando as duas séries separadamente. Testar métodos como simulação histórica filtrada e estimação condicional por regimes com validação temporal e incerteza.

Fundamentação: Rockafellar e Uryasev (2001), *Conditional Value-at-Risk for General Loss Distributions*.

### 8.4 Mudanças de regime e detecção sequencial

Comparar métodos estatísticos independentes: (a) CUSUM e detecção de mudanças nas distribuições/retornos agregados; (b) PCA dinâmica; (c) modelo oculto de Markov com probabilidades **filtradas** usando dados até t. **Não usar probabilidades suavizadas por todo o histórico**, porque dependem de informação futura. A detecção de uma ruptura já observada não é previsão de queda subsequente: medir separadamente tempo de detecção, antecedência real, falsos positivos, probabilidades de queda por horizonte e estabilidade dos regimes.

Não combinar todas as técnicas com pesos fixos inventados. Estudar cada componente isoladamente, com benchmark técnico e ingênuo; quando houver evidência de contribuição incremental, definir combinação por seleção exclusivamente temporal.

### 8.5 Critério de decisão derivado de distribuição e custos, não de regra arbitrária

Para política \(a\) entre U75, ativo externo e CASH, estimar a distribuição condicional do resultado financeiro de cada alternativa, riscos e custos. Uma rotação só se justifica quando o benefício de reduzir risco ou perdas futuras tiver evidência estatística suficiente **para compensar os custos e a oportunidade perdida** no cenário de avaliação. Reportar a fronteira risco-retorno, em vez de converter múltiplas medidas em um score com penalidades de pesos escolhidos retroativamente.

Primeiro validar previsões estatísticas **shadow/P1** (sem transação). Só então executar contrafactuais de decisão **P2/P3** com uma única avaliação diária, entradas na abertura seguinte, custos, reentrada sem duração mínima forçada em CASH e política completa do Control.

### 8.6 Critérios de rigor

- No fechamento t, apenas recursos observáveis em t. Ajustes de covariância, PCA, parâmetros de regime, quantis de cauda e calibração ocorrem em janelas do passado; maturação de alvos com embargo correspondente.
- Para cada horizonte pré-especificado, reportar qualidade da previsão de perda coletiva (erro de calibração, Brier, curva de confiabilidade, lead time e alarmes falsos), além de ES95, drawdown e perdas efetivamente evitadas por política após o replay.
- Usar validação cronológica/aninhada, comparação com benchmark ingênuo, estimativa de incerteza por reamostragem em blocos e controle do número de alternativas investigadas.
- Os trabalhos de Bellman/Optimal Switching CASH, regressão Ridge contrafactual e indicadores de estresse já foram executados; **a combinação prospetiva de ES condicional, Ledoit-Wolf, turbulência, PCA e regimes para U75 ainda não foi demonstrada**. Não apresentar metodologia como resultado.
- Hiperparametrização LightGBM/CARO e os US$ 587 milhões são história/checkpoint técnico, **não objetivo científico deste estudo**.

### Referências metodológicas a consultar na redação

- Ledoit, O.; Wolf, M. (2004). *A well-conditioned estimator for large-dimensional covariance matrices*. Journal of Multivariate Analysis, 88(2), 365–411. DOI: 10.1016/S0047-259X(03)00096-4.
- Rockafellar, R. T.; Uryasev, S. P. (2001). *Conditional Value-at-Risk for General Loss Distributions*. Manuscrito de 2001 (SSRN 267256).
- Kritzman, M.; Li, Y. (2010). *Skulls, Financial Turbulence, and Risk Management*. Financial Analysts Journal, referência de contexto a conferir na revisão bibliográfica antes de citar formalmente.

**Estado da v1.1.0: atualização de protocolo; não implementa o detector e não produz resultados financeiros.**
