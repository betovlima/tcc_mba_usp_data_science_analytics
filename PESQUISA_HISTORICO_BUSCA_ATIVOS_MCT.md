# Reconstrução histórica da Busca de Ativos do Market Cycle Trader

## Objetivo

Este documento reconstrói como o universo de ativos evoluiu até o U67 atual e
separa três coisas que, ao longo do desenvolvimento, acabaram aparecendo juntas:

1. o universo inicial escolhido manualmente;
2. a Busca de Ativos / Asset Discovery implementada no Market Cycle Trader;
3. pesquisas posteriores que acrescentaram ativos ao U56 e depois foram
   incorporadas novamente ao MCT.

A reconstrução usa conversas, logs, artefatos preservados e o histórico Git do
repositório `betovlima/market_cycle_trader_api`. Quando não existe evidência
suficiente para atribuir um ativo a uma campanha específica, isso é indicado
explicitamente.

---

## 1. Resumo executivo

O número atual de 67 ativos não surgiu de uma única execução do Asset Discovery.

A trajetória recuperada é, em linhas gerais:

```text
MCT muito inicial: U10
        ↓
universo de pesquisa: U20
        ↓
U24 = U20 + ADC + ADEA + ADI + ADM
        ↓
U25 = U24 + HD
        ↓
expansões Asset Discovery / validação econômica
        ↓
U32 → U36 → U44 → U51 → U52 → U53 → U56
        ↓
pesquisa posterior fora do Asset Discovery original
U56 + COLB + AMS + FOXF = U59
        ↓
U59 + THO + WDAY + EXR + XEL + SBFG + PAYX + MUX + SXC = U67
        ↓
U67 copiado de volta ao MCT
```

O estado atual configurado contém 67 símbolos. Na reprodução operacional que
serviu de referência recente, DOC e CLMT foram filtrados antes do modelo,
produzindo 65 ativos efetivos. A validade científica dessa exclusão foi
questionada em auditorias posteriores; portanto, "67 configurados / 65
efetivos" descreve a execução reproduzida, não deve ser reinterpretado como
prova de que 65 é o universo conceitualmente correto.

O ponto mais importante para a próxima pesquisa é histórico: os maiores saltos
de capital apareceram quando o sistema procurou contribuição econômica marginal
de ativos dentro da estratégia, e não quando simplesmente aumentou a quantidade
de ativos. Houve também expansões grandes que destruíram capital.

---

## 2. Antes do Asset Discovery: U10, U20, U24 e U25

### U10

Existe evidência de uma versão inicial do MCT com 10 ativos:

```text
NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA, AMD, JPM, SPY
```

Esse universo pertence à fase inicial do projeto e não deve ser confundido com
a linha de Asset Discovery posteriormente usada para formar o U56.

### U20

Em agosto de 2026 já existia um universo de pesquisa de 20 ativos:

```text
NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA, AMD, JPM, SPY,
AVGO, NFLX, CRM, ORCL, COST, LLY, XOM, CAT, WMT, V
```

As conversas posteriores descrevem essa lista como a base inicial recomendada
durante o desenvolvimento. Não foi recuperado um registro que mostre uma
campanha automática responsável pela passagem U10 → U20.

### U24

Quatro ativos aparecem depois no contexto de pesquisa/tuning:

```text
ADC, ADEA, ADI, ADM
```

Assim:

```text
U20 + ADC + ADEA + ADI + ADM = U24
```

Eles já estavam presentes em campanhas Latin Hypercube/CARO. Não foi recuperado
um log que prove que esses quatro foram escolhidos pelo Asset Discovery.

### U25

HD aparece como `research_candidate_asset` e passa a compor o universo de 25:

```text
U24 + HD = U25
```

Uma comparação preservada mostrou, naquele contexto específico:

```text
24 ativos: US$ 421.104
25 ativos com HD: US$ 482.817,85
diferença: +US$ 61.713,48 (+14,66%)
```

Esse número pertence ao experimento daquele momento e não é diretamente
comparável aos checkpoints posteriores, pois parâmetros/campanhas mudaram.

O Winner #7, promovido em 13/08/2026, já usava os 25 ativos e registrou:

```text
capital final: US$ 1.109.354,57
CAGR: +117,98%
Sharpe: 1,966
MaxDD: -31,45%
Worst Fold: +98,06%
```

A composição U25 ficou:

```text
NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA, AMD, JPM, SPY,
AVGO, NFLX, CRM, ORCL, COST, LLY, XOM, CAT, WMT, V,
HD, ADC, ADEA, ADI, ADM
```

---

## 3. Primeira geração do Asset Discovery

A primeira geração recuperada do Asset Discovery trabalhava em lotes
controlados, inicialmente de oito ativos.

### Universo externo

O scanner consultava ativos US Equity ativos na Alpaca e aceitava:

```text
AMEX, ARCA, BATS, NASDAQ, NYSE
```

O símbolo precisava ser negociável e não podia já fazer parte da Strategy.

### Filtros de mercado

A implementação legada mantinha, entre outros, os seguintes filtros:

```text
preço mínimo: US$ 5
mediana de dollar volume: US$ 5 milhões
proporção mínima de volume não zero: 95%
```

O filtro comportamental usava aproximadamente 756 sessões e observava:

- cauda negativa de retornos;
- gaps negativos;
- volatilidade anualizada;
- drawdown;
- pior perda diária;
- pior gap;
- pior perda em dez sessões.

Os limites restaurados no código legado eram aproximadamente:

```text
downside tail 1% <= 12%
gap downside tail 1% <= 10%
volatilidade anualizada <= 90%
MaxDD <= 75%
pior dia <= 30%
pior gap <= 25%
pior janela de 10 sessões <= 35%
```

Os ativos podiam terminar como `candidate`, `watchlist` ou `rejected`.
Falhas técnicas eram separadas de rejeições analíticas.

Essa geração ainda era essencialmente um scanner de qualidade/risco. Ela não
era o mecanismo econômico final que produziu a maior parte da expansão.

---

## 4. Expansão U25 → U32 e experimentos U33-U36

A Strategy #22 revision 4 registra explicitamente 32 ativos. Comparando sua
lista com o U25 congelado, os sete novos ativos são:

```text
DDS, CNQ, VRTS, XSD, CXW, ENS, CORT
```

Portanto a composição registrada permite reconstruir:

```text
U25 + DDS + CNQ + VRTS + XSD + CXW + ENS + CORT = U32
```

O histórico não preserva um append log individual para cada um dos sete; a
atribuição do conjunto é obtida pela diferença entre listas de revisões.

Depois ocorreram vários testes de inclusão.

### PAYX

Uma revision 5 testou PAYX como 33º ativo:

```text
baseline U32: US$ 1.953.776,88
U32 + PAYX:  US$ 1.629.747,74
delta: -16,58%
```

PAYX foi rejeitado naquele contexto. Isso é importante porque PAYX reapareceria
muito depois como um dos oito positivos do U59 → U67. A utilidade de um ativo
mudou conforme o restante do universo mudou.

### CCK

CCK foi testado em um universo de 33 ativos:

```text
capital: US$ 2.099.217,01
delta aproximado: +7,44%
```

Apesar do aumento de capital, o pior fold caiu cerca de 10,36 pontos
percentuais e o gate de robustez rejeitou a inclusão naquele momento. CCK,
porém, aparece posteriormente no universo consolidado.

### CEF

CEF foi acrescentado à Strategy #22 revision 8, levando a 34 ativos:

```text
capital: US$ 2.206.133
CAGR: 142,11%
Sharpe: 2,1210
MaxDD: -33,20%
```

Foi aprovado pelo Full Strategy Validation.

### GKOS

GKOS aparece na revision 9, com 35 ativos. O backtest daquela revisão terminou
em aproximadamente:

```text
US$ 2.097.887
```

### RACE

RACE aparece na revision 10, formando U36:

```text
capital: US$ 3.803.231
CAGR: 164,64%
Sharpe: 2,304
MaxDD: -28,06%
```

RACE teve sete posições completas, quatro positivas, com P&L direto de cerca de
US$ 197,8 mil.

Esses resultados mostram uma característica que depois se repetiria:
aprovação/rejeição era dependente do universo e do protocolo vigente.

---

## 5. Reconstrução do U44

A Strategy #10 revision 2 foi registrada com 44 ativos e os seguintes 19 ativos
adicionais em relação à base de 25:

```text
DDS, CNQ, VRTS, XSD, CXW, ENS, CORT, CCK, CEF, GKOS,
RACE, TX, UNF, BXMT, PXLW, KKR, SCSC, LKFT, DNN
```

Logo, a composição U44 é explicitamente compatível com:

```text
U25 + 19 ativos = U44
```

A revision 1 da Strategy #10 tinha 26 ativos; CCS estava presente nessa linha e
saiu na revision 2. Não foi recuperada a lista integral da revision 1 nem o
motivo exato da retirada de CCS. Portanto, não é seguro afirmar que CCS era
simplesmente o 26º ativo da base U25, embora a contagem seja compatível com essa
hipótese.

No mesmo registro histórico:

```text
Strategy #10 rev.1 / 26 ativos: ~US$ 1.074.831
Strategy #10 rev.2 / 44 ativos: ~US$ 5.394.605
```

Esse é um dos primeiros saltos claros associados à ampliação do universo
mantendo a lógica central da Strategy.

---

## 6. U44 → U51: composição parcialmente reconstruída

O universo U56 final contém, depois dos 19 ativos do U44 e antes de MYE/YANG,
os seguintes sete símbolos:

```text
VNCE, UNFI, DOC, CLMT, APD, MGM, MAN
```

A aritmética produz exatamente 51:

```text
44 + 7 = 51
```

Entretanto, não foi recuperado um append log que declare literalmente
"U44 → U51 adicionou estes sete". Eles aparecem no universo posterior e são os
sete símbolos que completam a composição até MAN, mas essa etapa deve ser
tratada como reconstrução, não como evidência individual de campanha.

O checkpoint recuperado de 51 ativos foi aproximadamente:

```text
US$ 13,81 milhões
```

---

## 7. U51 → U52: MYE confirmado

Esta etapa tem evidência direta.

MYE foi o único ativo acrescentado na passagem de 51 para 52:

```text
sem MYE: US$ 13.812.229
com MYE: US$ 15.522.587
delta:   +US$ 1.710.358
delta%:  +12,38%
```

MYE foi selecionado diretamente apenas quatro vezes, com P&L direto de cerca de
US$ 102,8 mil, mas alterou 29 de 1.536 decisões. A maior parte do ganho,
portanto, não veio do P&L direto de MYE, mas da mudança do caminho das decisões.

Esse episódio foi uma das primeiras evidências fortes de dependência de
trajetória.

---

## 8. U52 → U53: YANG

YANG é o 28º candidato da lista consolidada e aparece como o 53º ativo na ordem
preservada do U56.

Há registro de um checkpoint com 53 ativos em torno de:

```text
US$ 19,863 milhões
```

Não foi recuperada uma comparação isolada "antes/depois de YANG" que permita
atribuir esse salto causalmente ao ativo. Logo, U52 → U53 = +YANG é uma
reconstrução consistente com a ordem do universo, mas não possui a mesma força
documental da etapa MYE.

Mais tarde, YANG apresentou contribuição negativa, neutra e positiva conforme
o universo mudava, reforçando que a contribuição marginal de um ativo não é
estável.

---

## 9. U53 → U56: MKSI, MCS e ECC

Os artefatos preservados da execução de 56 ativos mostram a parte final da
ordem:

```text
53/56 — YANG
54/56 — MKSI
55/56 — MCS
56/56 — ECC
```

Também existe uma campanha Asset Discovery concluída com 92 candidatos e
`asset_count=56`, cuja lista termina:

```text
YANG, MKSI, MCS, ECC
```

Isso sustenta a reconstrução:

```text
U53 + MKSI + MCS + ECC = U56
```

O texto dos artefatos não registra uma tabela de capital isolando esses três
ativos como um lote único, portanto não se deve atribuir todo o ganho U53 → U56
individualmente a eles.

O checkpoint histórico U56 mais conhecido dessa fase foi:

```text
capital final: US$ 23.712.122,62
CAGR: ~255,71%
Sharpe: ~2,317
MaxDD: ~-28,24%
```

A composição U56 ficou:

```text
NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA, AMD, JPM, SPY,
AVGO, NFLX, CRM, ORCL, COST, LLY, XOM, CAT, WMT, V,
HD, ADC, ADEA, ADI, ADM,
DDS, CNQ, VRTS, XSD, CXW, ENS, CORT, CCK, CEF, GKOS,
RACE, TX, UNF, BXMT, PXLW, KKR, SCSC, LKFT, DNN,
VNCE, UNFI, DOC, CLMT, APD, MGM, MAN, MYE, YANG,
MKSI, MCS, ECC
```

---

## 10. Como o Asset Discovery do MCT evoluiu tecnicamente

### 10.1 Scanner de mercado e risco

A primeira implementação usava filtros de preço, liquidez, volume e risco
comportamental, processando lotes de oito.

### 10.2 Remoção da implementação legada

Em 24/08/2026:

```text
a4dd70e023 — refactor: remove legacy Asset Discovery implementation
```

O scanner anterior foi removido e a pesquisa passou a ser reconstruída em torno
da Strategy.

### 10.3 Learning-to-Rank + contribuição econômica

Em 25/08/2026:

```text
d7e023bfea — feat: rank Asset Discovery results by marginal economic contribution
```

A nova geração treinava um LightGBM LambdaMART com 14 atributos:

```text
return_5, return_10, return_20, return_40, return_60,
volatility_10, volatility_20, volatility_60,
drawdown_20, drawdown_60,
position_60, trend_efficiency_20,
volume_ratio_20, dollar_volume_log
```

O alvo do ranker considerava retorno futuro de 20 sessões com penalização de
adverse excursion.

A validação do ranker usava:

```text
4 folds cronológicos
expanding walk-forward
purge de 20 sessões
NDCG@5
comparação contra Momentum e Random
```

O ranker servia para ordenar candidatos, mas a decisão econômica passou a ser
avaliada por replay da Strategy.

### 10.4 Integridade histórica

O candidato precisava ser comparável ao universo de referência. A lógica passou
a exigir cobertura do calendário XNYS requerido pela Strategy.

Uma lacuna interna longa, a partir de 20 sessões, era tratada como possível
reutilização/quebra de identidade do ticker.

Mais tarde a política ficou ainda mais rígida: histórico completo desde
01/01/2016 até o snapshot, sem sessões obrigatórias faltantes.

### 10.5 Marginal Capital Replay

O mecanismo central comparava:

```text
Strategy baseline
versus
Strategy baseline + candidato
```

mantendo modelo, período, custos, folds e configuração.

As métricas incluíam:

- capital final;
- CAGR;
- Sharpe;
- MaxDD;
- pior fold;
- número de switches;
- exposição e caixa.

Em versões importantes, um candidato só persistia se a contribuição econômica
fosse positiva.

### 10.6 Validação causal

Em 28-29/08 foi introduzida uma tentativa de validação causal com janela
posterior à seleção:

```text
ec1f145c9b — require nested causal validation and final certification
d7ffb8d6c1 — enforce causal validation window boundaries
75a118f9cf — parallelize causal validation and remove research cap
```

O protocolo separava seleção, validação e certificação, com corte temporal e
sem usar ganho futuro para selecionar o candidato.

### 10.7 Append no mesmo universo

Em 29/08:

```text
e2e9ac0044 — append selected certified assets to research strategy
```

A regra passou a ser:

- preservar todos os ativos existentes;
- acrescentar apenas os candidatos explicitamente selecionados;
- exigir certificação;
- persistir histórico somente dos selecionados;
- não criar uma Strategy paralela só para o candidato.

### 10.8 Retorno ao replay de histórico completo

Ainda em 29/08:

```text
cbbac30214 — restore full-history capital contribution protocol
```

A persistência voltou a usar a regra:

```text
positive_full_history_strategy_capital
```

Ou seja, o MCT passou novamente a privilegiar candidatos que melhoravam o
capital final da Strategy no histórico completo.

Essa escolha é crucial para interpretar os resultados: ela encontrou ativos
com forte impacto econômico, mas também cria risco de seleção retrospectiva no
mesmo histórico.

---

## 11. O experimento que mostrou o perigo de simplesmente adicionar ativos

Em setembro, o modo predictive-first tentou acelerar a descoberta.

Commit principal:

```text
34568f8504 — switch asset discovery to predictive-first mode
```

A seleção automática deixou de executar o replay econômico individual completo
para cada candidato.

Também foi introduzida uma medida de estado baseada nas folhas do LightGBM:

```text
similaridade = fração de árvores em que candidato e ativo-base caem na mesma folha
novidade = 1 - maior similaridade com um ativo-base
```

Essa fase foi rápida, mas uma expansão posterior aumentou o universo até cerca
de 72 ativos e o capital caiu para aproximadamente:

```text
US$ 1,552 milhão
```

Ao retirar as inclusões problemáticas, o U56 voltou para aproximadamente:

```text
US$ 23,712 milhões
```

Entre os 16 ativos acrescentados na expansão problemática estavam:

```text
ADT, ALV, AX, BABA, DCOM, DXCM, HNI, IDCC,
KBH, PB, QID, RF, SGRY, SNA, UTI, VTI
```

Esse episódio é evidência direta de que:

```text
mais ativos != mais capital
```

e foi uma das razões para restaurar os gates econômicos e de histórico.

---

## 12. Política madura do Asset Discovery

Depois das regressões, o MCT passou a acumular várias proteções.

### Aderência de mercado

Na política restaurada:

```text
preço >= US$ 5
median dollar volume >= US$ 10 milhões
nonzero volume ratio >= 0,98
```

### Histórico completo

O candidato deve cobrir toda a janela da Strategy antes de ser persistido.

### Descoberta preditiva primeiro

O ranker pode ser usado para reduzir o universo de busca e ordenar candidatos.

### Gate econômico antes do append

A política final exige validação do universo selecionado completo:

```text
full_strategy_history_replay
capital_improves = True
full_history_integrity = True
research_context = True
```

O código atual explicita:

```text
selection_policy =
predictive_discovery_then_selected_universe_full_strategy_economic_validation
```

e:

```text
economic_append_gate =
exact_selected_universe_must_increase_final_strategy_capital
```

A política também registra que o replay marginal individual não é obrigatório
antes do append final; o conjunto selecionado deve ser validado economicamente.

---

## 13. Pesquisas paralelas que NÃO são a formação principal do U56

Algumas linhas de pesquisa importantes não devem ser confundidas com a
cronologia de inclusão do universo.

### Clustering diário do estado dos ativos

Foi introduzido em:

```text
5062d22663 — add daily unsupervised asset state clustering
```

e depois removido do pipeline principal. Era diagnóstico, não a regra que
formou o universo final.

### Busca sequencial/pruning posterior

Uma pesquisa posterior partiu dos ativos disponíveis e chegou a:

```text
U42 ≈ US$ 33.443.684,50
U37 ≈ US$ 43.759.854,82
```

A poda removeu cinco ativos e aumentou o capital. Isso novamente confirmou que
quantidade de ativos não é objetivo por si só.

Essa linha não substituiu o U56 histórico na formação do U67 final.

---

## 14. U56 → U59 não veio do Asset Discovery original

Esta etapa ocorreu em pesquisa posterior.

Um lote de 20 novos ativos foi sorteado sem usar o backtest para escolhê-los.
Três apresentaram contribuição positiva:

```text
COLB
AMS
FOXF
```

Resultados individuais recuperados:

```text
COLB: ~+73,01%
AMS:  ~+51,05%
FOXF: ~+3,40%
```

O universo resultante:

```text
U56 + COLB + AMS + FOXF = U59
```

teve, no protocolo daquela pesquisa:

```text
capital: US$ 30.080.091,01
CAGR: 267,40%
Sharpe: 2,3375
MaxDD: -31,22%
```

Essa seleção pertence à pesquisa posterior, não ao Asset Discovery original do
MCT.

---

## 15. U59 → U67 também foi pesquisa posterior

Uma pré-seleção contextual congelou candidatos sem usar capital na fase de
triagem. Depois, testes financeiros individuais identificaram oito ativos
positivos:

```text
THO, WDAY, EXR, XEL, SBFG, PAYX, MUX, SXC
```

Portanto:

```text
U59 + 8 = U67
```

O checkpoint exploratório daquela pesquisa ficou em:

```text
US$ 58.557.157,67
```

Mais tarde o U67 foi copiado para o MCT como Strategy dedicada. Com dados e
pipeline operacional atualizados, a reprodução recente registrou:

```text
2026-09-17: US$ 78.782.538,31
2026-10-06: US$ 76.927.051,39
```

Esses valores não devem ser comparados diretamente aos checkpoints históricos
sem considerar snapshot, cutoff e regras de preparação.

---

## 16. O que está comprovado e o que continua incompleto

### Comprovado

- composição exata do U20, U25, U32, U44, U56 e U67;
- MYE como único incremento U51 → U52;
- presença ordenada YANG, MKSI, MCS e ECC no final do U56;
- arquitetura e evolução do Asset Discovery no Git;
- uso de Learning-to-Rank;
- uso de Marginal Capital Replay;
- tentativas de validação causal;
- retorno ao full-history replay;
- predictive-first e sua posterior proteção por histórico/aderência/gate
  econômico;
- U56 → U59 por COLB/AMS/FOXF em pesquisa posterior;
- U59 → U67 pelos oito ativos posteriores;
- uma expansão ampla pode destruir capital, como ocorreu no experimento que
  chegou a ~72 ativos.

### Parcialmente reconstruído

- a passagem exata U44 → U51. Os sete símbolos necessários para completar a
  lista são VNCE, UNFI, DOC, CLMT, APD, MGM e MAN, mas o append log dessa
  transição não foi recuperado;
- U52 → U53 com YANG é consistente com a ordem preservada, mas não há
  comparação isolada de YANG naquele momento;
- U53 → U56 com MKSI/MCS/ECC é fortemente sustentado pelo contador 53/56,
  54/56, 55/56, 56/56 e pela lista final, porém não existe uma tabela
  before/after do lote de três.

### Não comprovado

- uma única cadeia automática que explique cada um dos 67 ativos;
- que todo ativo adicional do U56 tenha sido escolhido pelo mesmo algoritmo;
- que o capital crescente em cada marco seja atribuível ao último ativo
  adicionado;
- que o universo U67 seja um ótimo global;
- que ampliar o universo seja monotonicamente benéfico.

---

## 17. Conclusão histórica

A Busca de Ativos do MCT passou por três ideias distintas:

```text
1. filtrar ativos negociáveis e comportamentalmente aceitáveis;
2. ordenar candidatos por capacidade preditiva;
3. medir se sua presença altera economicamente a Strategy.
```

O terceiro mecanismo foi o que mais se relacionou aos grandes saltos históricos
de capital.

Ao mesmo tempo, o histórico contém evidência forte de seleção dependente do
contexto:

- PAYX foi ruim em U32 e positivo muito depois;
- MYE produziu grande efeito indireto apesar de poucas operações próprias;
- YANG mudou de sinal conforme o universo;
- expansões amplas chegaram a destruir mais de 90% do capital final comparado ao
  baseline;
- uma poda 42 → 37 aumentou capital.

Por isso, o próximo teste não deve procurar "mais candidatos" nem repetir
clustering. Ele deve testar diretamente se a lógica que historicamente formou o
universo, a busca marginal por aumento de capital, sobrevive quando seleção e
avaliação são temporalmente separadas.

---

## 18. Teste proposto para esta branch

Nome provisório:

```text
MCT Sequential Marginal Search — Causal Reconstruction
```

Pergunta:

> Se reconstruirmos a busca marginal do MCT começando de um universo congelado
> e permitirmos que cada candidato seja escolhido apenas com dados disponíveis
> até um cutoff de desenvolvimento, os ativos aceitos continuam acrescentando
> valor em um período futuro que nunca participou da seleção?

Desenho:

```text
universo inicial congelado
        ↓
pool de candidatos congelado
        ↓
replay marginal somente no período de desenvolvimento
        ↓
aceita o melhor candidato positivo
        ↓
congela a inclusão
        ↓
repete até nenhum candidato melhorar o desenvolvimento
        ↓
uma única avaliação no holdout futuro intocado
```

O experimento deve registrar duas curvas:

```text
capital no período usado para selecionar
capital no holdout nunca usado para selecionar
```

A hipótese só recebe suporte se a sequência que melhora o período de
desenvolvimento também produzir ganho fora da amostra.

Esse desenho reproduz a ideia econômica que historicamente funcionou no MCT,
mas remove justamente a fragilidade mais importante do protocolo
`positive_full_history_strategy_capital`: selecionar e validar repetidamente
no mesmo histórico.


---

## 19. Adendo após recuperação dos artefatos primários do MCT

A reconstrução inicial acima foi confrontada com JSONs históricos das campanhas
reais do Market Cycle Trader preservados na biblioteca do projeto. Esses
artefatos permitem corrigir uma simplificação importante: a evolução do número
de ativos NÃO foi uma única sequência monotônica. Houve Strategies paralelas,
testes, rollbacks e recomposições do universo.

### 19.1 Campanha real de 25/08/2026 partindo do U25

Uma campanha preservada partiu exatamente do U25 e executou Marginal Capital
Replay em oito candidatos:

```text
RARE, CBL, CCS, PTCT, SCCO, ITW, NXST, NBTB
```

Baseline:

```text
capital = US$ 1.067.794,9987
CAGR = 115,63%
Sharpe = 1,9449
MaxDD = -31,45%
switches = 260
worst fold = +98,06%
```

Resultados econômicos marginais:

```text
CCS   +0,65895%   marginal_rank=1
CBL   +0,35521%   marginal_rank=2
ITW    0,00000%   marginal_rank=3
NBTB   0,00000%   marginal_rank=4
SCCO  -2,19762%   marginal_rank=5
PTCT -14,09692%   marginal_rank=6
NXST -31,61311%   marginal_rank=7
RARE -32,21420%   marginal_rank=8
```

Esse artefato é especialmente relevante porque o ranking preditivo não
coincidiu com a contribuição econômica. RARE era o primeiro colocado do ranker
e acabou sendo o pior no replay marginal. CCS era apenas o terceiro do ranking
preditivo e terminou como o melhor candidato econômico.

Conclusão: na fase historicamente bem-sucedida da Busca de Ativos, o
Learning-to-Rank servia para reduzir/ordenar a busca, mas a contribuição para a
Strategy precisava ser medida economicamente.

### 19.2 Strategies paralelas confirmadas

Também foram recuperadas campanhas contemporâneas que não pertencem à linhagem
final do U56.

Uma Strategy #7 chegou a 31 ativos adicionando ao U25:

```text
TQQQ, EWY, GAP, BB, ALK, ATI, SQQQ
```

Outra Strategy #8 tinha 27 ativos:

```text
U25 + AGI + TQQQ
```

e testou:

```text
VSAT, LB, CB, IJH
```

Essas composições não sobreviveram até o U56 final. Portanto, a sequência
histórica não pode ser escrita como se cada novo número de ativos fosse uma
versão cumulativa única.

### 19.3 Linha causal de 40 e 42 ativos

A Strategy #22 revision 14 teve 40 ativos. Em relação ao U25, os quinze extras
registrados foram:

```text
DDS, CNQ, VRTS, XSD, CXW, ENS, CORT, CCK, CEF, GKOS,
RACE, TX, UNF, BXMT, CNX
```

Uma Strategy #24 posterior chegou a 42 com:

```text
U40 + HL + BOX
```

Campanhas causais executadas sobre esse U42 chegaram a reter zero candidatos
em algumas rodadas.

Essa linha U40/U42 é diferente da Strategy #10 que posteriormente consolidou o
U44/U56. Em particular, CNX/HL/BOX não aparecem no U56 final, enquanto
PXLW/KKR/SCSC/LKFT/DNN aparecem.

### 19.4 Protocolo causal realmente implementado no MCT

Os artefatos confirmam que o MCT chegou a implementar separação temporal
explícita:

```text
método: historical_selection_then_validation_then_certification

selection cutoff: 2025-08-27
validation:       2025-08-28 a 2026-02-27
certification:    2026-03-02 a 2026-08-28

holdout total: 252 sessões
validation:    126 sessões
certification: 126 sessões
```

O artefato registrou:

```text
historical_gain_used_for_selection = false
promotion_uses_holdout_only = true
promotion_uses_certification_only = true
survivorship_bias_fully_eliminated = false
```

Esse protocolo foi tecnicamente mais rigoroso, mas produziu campanhas com
poucos ou nenhum candidato promovível. Em seguida, o projeto restaurou a
política de contribuição positiva no histórico completo.

Essa decisão histórica é central para o próximo experimento: não é suficiente
"recriar a validação causal" exatamente como antes. É necessário entender se a
busca marginal que elevou o capital no histórico completo tinha sinal fora da
amostra ou se parte do ganho vinha da reutilização do mesmo histórico durante
seleção.

### 19.5 Campanha preditiva de 92 candidatos no U56

Foi recuperada a lista integral de uma campanha posterior que partiu do U56.
Ela aprovou integridade preditiva para 92 candidatos, mas registrou
explicitamente que nenhum replay econômico de histórico completo foi executado
naquela etapa.

Isso confirma a distinção entre:

```text
predictive candidate
!=
economic contributor
```

A campanha seguinte que promoveu um subconjunto preditivo sem uma proteção
econômica equivalente chegou a expandir o universo e produzir forte deterioração
do capital. Esse episódio foi uma das razões para o retorno posterior ao gate
econômico da Strategy completa.

### 19.6 Conclusão corrigida da reconstrução

A formação histórica do universo atual deve ser entendida em três camadas:

```text
Camada A — universo base
U10 -> U20 -> U24 -> U25

Camada B — pesquisa/Asset Discovery do MCT
múltiplas Strategies e campanhas:
U25 -> alternativas U27/U31/...
        -> U32/U36
        -> linha U40/U42
        -> recomposição Strategy #10
        -> U44 -> ... -> U56

Camada C — pesquisas posteriores
U56 + COLB + AMS + FOXF = U59
U59 + THO + WDAY + EXR + XEL + SBFG + PAYX + MUX + SXC = U67
```

Assim, o número 67 não é a saída de um único algoritmo de busca. É o resultado
acumulado de seleção manual inicial, campanhas de Asset Discovery com várias
políticas, recomposições/rollbacks e duas campanhas de pesquisa posteriores ao
U56.
