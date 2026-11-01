# Protocolo científico: antecipação de períodos desfavoráveis do universo U75

**Versão do protocolo:** `u75-downside-regime-avoidance-v1.0.0`  
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
