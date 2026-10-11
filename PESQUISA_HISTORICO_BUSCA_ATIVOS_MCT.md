# Histórico da Busca de Ativos do Market Cycle Trader

## Objetivo

Este documento reconstrói, de forma auditável, como o universo de ativos do
Market Cycle Trader (MCT) cresceu até o universo usado posteriormente pelo TCC.
O objetivo não é justificar retrospectivamente o universo atual, mas recuperar
o processo real de descoberta, validação, inclusão, exclusão e rollback que
ocorreu no MCT.

A reconstrução é necessária antes de qualquer novo teste porque a contagem atual
não surgiu de um único algoritmo nem de um número-alvo pré-definido. Houve
várias gerações de Asset Discovery, mudanças no critério de aceitação, campanhas
que aumentaram o universo, expansões que falharam e pesquisas posteriores que já
não pertenciam ao Asset Discovery original.

## Hierarquia de evidência

Para evitar transformar lembranças em fatos, este histórico usa quatro níveis de
evidência.

1. **Evidência direta de código/Git do MCT**: código, commits e documentação
   existentes no repositório `betovlima/market_cycle_trader_api`.
2. **Evidência direta de artefatos exportados do MCT**: JSON/TXT/XLSX preservados
   das campanhas e backtests.
3. **Evidência de conversas anteriores**: resultados e decisões registrados
   durante a execução das pesquisas.
4. **Inferência**: diferença entre duas listas conhecidas ou sequência provável
   quando o artefato intermediário exato não foi preservado.

Sempre que a origem não for do nível 1 ou 2, isso é explicitado.

---

# 1. Antes do Asset Discovery formal: 20 -> 24 -> 25

## 1.1 Universo-base de 20 ativos

A pesquisa original usava 20 ativos-base:

```text
NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA, AMD, JPM, SPY,
AVGO, NFLX, CRM, ORCL, COST, LLY, XOM, CAT, WMT, V
```

Uma configuração histórica v1.13.31 separava esses 20 como
`calendar_anchor_assets`, mantendo os ativos pesquisados fora do conjunto que
definia calendário/folds. Conversas e logs de agosto de 2026 preservaram a mesma
ordem.

**Status da evidência:** conversa histórica + estrutura de configuração antiga.
O artefato original da primeira criação desses 20 não foi localizado.

## 1.2 20 -> 24

No experimento referido na época como **"Teste de novos ativos 4"**, quatro
ativos foram adicionados:

```text
ADC, ADEA, ADI, ADM
```

O universo passou de 20 para 24 ativos.

O usuário registrou em 9 de agosto de 2026 que o Winner de aproximadamente
US$ 230 mil utilizava esses 24 ativos.

**Importante:** o trabalho era pesquisa de candidatos compatível com a ideia que
mais tarde se consolidou como Asset Discovery, mas o material recuperado não
permite afirmar que já usava a implementação formal chamada "Asset Discovery".

## 1.3 24 -> 25

O universo operacional que se tornou o Winner histórico passou a conter:

```text
20 ativos-base
+ HD
+ ADC, ADEA, ADI, ADM
= 25 ativos
```

A composição de 25 está preservada diretamente em vários artefatos posteriores
do MCT e em campanhas de Asset Discovery que usaram esse Winner como fonte:

```text
NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA, AMD, JPM, SPY,
AVGO, NFLX, CRM, ORCL, COST, LLY, XOM, CAT, WMT, V,
HD, ADC, ADEA, ADI, ADM
```

Winner #7, promovida em agosto de 2026, utilizava exatamente esses 25 ativos.

**Conclusão desta fase:** o número 25 também não era um objetivo matemático.
Ele era o resultado acumulado da base de 20 e de cinco ativos incorporados pela
pesquisa de candidatos.

---

# 2. Primeira geração do Asset Discovery formal

Antes de 24 de agosto de 2026 existia uma implementação operacional de Asset
Discovery no MCT com:

- universo atual de ações americanas da Alpaca;
- ativos ativos/tradáveis;
- exchanges AMEX, ARCA, BATS, NASDAQ e NYSE;
- exclusão dos ativos já presentes na Strategy;
- lotes limitados, originalmente até 8 ativos avaliáveis;
- cursor para continuar a busca em campanhas posteriores;
- reavaliação periódica;
- filtros de preço, liquidez e qualidade de volume;
- filtro de risco/comportamento;
- classificação `candidate`, `watchlist` ou `rejected`.

Configuração histórica padrão encontrada no código:

```text
batch_size = 8
recheck_days = 30
min_price = 5
min_median_dollar_volume = 5,000,000
min_nonzero_volume_ratio = 0.95
behavior_lookback_days = 1125
behavior_lookback_sessions = 756
```

O filtro de risco comportamental media, entre outros:

- cauda inferior de retorno de 1%;
- cauda inferior de gaps de 1%;
- volatilidade anualizada;
- máximo drawdown;
- pior perda diária;
- pior gap;
- pior perda em 10 sessões.

Essa geração fazia um **filtro de elegibilidade e comportamento**, mas ainda não
respondia satisfatoriamente à pergunta econômica principal:

> adicionar este ativo melhora a trajetória da Strategy inteira?

Commit que encerrou essa geração:

```text
2026-08-24
a4dd70e refactor: remove legacy Asset Discovery implementation
```

O frontend também removeu a interface legada na mesma data.

---

# 3. Segunda geração: ranking + contribuição econômica marginal

Em 25 de agosto foi introduzida uma nova linha de Asset Discovery:

```text
d7e023b feat: rank Asset Discovery results by marginal economic contribution
```

A arquitetura passou a separar dois problemas:

1. **descobrir/rankear candidatos**;
2. **verificar se a presença do candidato melhora a Strategy completa**.

## 3.1 Learning-to-Rank

O ranker foi implementado com LightGBM LambdaMART:

```text
objective = lambdarank
metric = ndcg
target horizon = 20 sessões
```

Os 14 atributos do ranking eram:

```text
return_5
return_10
return_20
return_40
return_60
volatility_10
volatility_20
volatility_60
drawdown_20
drawdown_60
position_60
trend_efficiency_20
volume_ratio_20
dollar_volume_log
```

O alvo de ranking usava uma utilidade futura de 20 sessões, incorporando retorno
e excursão adversa.

A validação do ranker era temporal, expanding walk-forward com purge:

```text
4 folds
purge = 20 sessões
```

O desempenho era comparado contra momentum e ranking aleatório por NDCG@5.

O ranking, porém, era apenas **triagem**. Nas versões corrigidas ele não deveria
ser suficiente para persistir um ativo na Strategy.

## 3.2 Marginal Capital Replay

A pergunta econômica ficou explícita:

```text
Strategy base
versus
Strategy base + candidato
```

O critério desejado era contribuição marginal positiva no capital final.

Em 26 de agosto:

```text
ef07bde feat(discovery): add full-strategy candidate validation
```

Em 27 de agosto:

```text
9befc62 perf(asset-discovery): fast-scan candidates before full replay validation
```

A busca passou a:

1. escanear candidatos rapidamente;
2. ordenar os candidatos;
3. validar histórico completo na ordem;
4. parar quando atingisse o número de aderentes desejado.

---

# 4. Primeiras expansões do universo de 25

## 4.1 Campanha de 25 de agosto: +4 ativos comprovados

Um artefato de Strategy preserva diretamente o `discovery_origin` da campanha:

```text
run_id = asset-discovery-20260825T230459-45ecac9f
snapshot_end = 2026-08-25
```

Ativos selecionados:

```text
DDS, CNQ, VRTS, XSD
```

Cada um teve 2.676 linhas históricas persistidas.

Ranking registrado:

| Ativo | rank | raw_score |
|---|---:|---:|
| XSD | 2 | 0.2864635761 |
| VRTS | 3 | 0.2020376590 |
| CNQ | 4 | 0.0982152600 |
| DDS | 6 | 0.0521515745 |

Essa é a primeira expansão pós-25 para a qual foi localizado um vínculo de
proveniência completo entre campanha e Strategy.

## 4.2 Até U32

Por 26-27 de agosto, uma Strategy #22 rev.4 já continha 32 ativos:

```text
25 anteriores
+ DDS, CNQ, VRTS, XSD
+ CXW, ENS, CORT
= 32
```

A sequência exata das três últimas inclusões entre 29 e 32 não foi localizada
em um único artefato. Portanto:

- a presença das três no U32 é confirmada;
- a ordem individual CXW/ENS/CORT deve ser considerada **não recuperada**.

Há registro histórico de CORT tendo contribuição marginal positiva em replay,
mas esse resultado pertence às conversas de execução e não a um artefato que
tenha sido localizado nesta auditoria.

---

# 5. Crescimento econômico U32 -> U36

Essa fase é importante porque mostra que o processo não era simplesmente
"adicionar todos os melhores do ranking".

## 5.1 PAYX foi testado e rejeitado

Com U32 em torno de US$ 1,953 milhão, PAYX foi testado como 33º ativo.

Resultado recuperado:

```text
baseline ~ US$ 1.953.777
base + PAYX ~ US$ 1.629.748
delta ~ -16,58%
```

PAYX não foi mantido naquele universo.

## 5.2 CCK

```text
U32 -> U33
+ CCK
capital ~ US$ 2.099.217
delta ~ +7,44%
```

## 5.3 CEF

```text
U33 -> U34
+ CEF
capital ~ US$ 2.206.133
delta ~ +5,09%
```

## 5.4 GKOS

```text
U34 -> U35
+ GKOS
```

## 5.5 RACE

```text
U35 -> U36
+ RACE
capital ~ US$ 3,416 mi -> US$ 3,803 mi
delta ~ +11,33%
```

RACE apareceu em sete posições e teve P/L atribuído de aproximadamente
US$ 197,8 mil na análise da época.

Essa sequência demonstra o comportamento central que queremos preservar no
novo teste: **candidato ranqueado não significava candidato aceito; a
contribuição no sistema de rotação importava**.

---

# 6. Uma campanha com evidência direta de Marginal Capital Replay

Um JSON de 28 de agosto preserva uma campanha em que:

```text
baseline assets = 40
baseline ending capital = 4.990.517,905
fast scan ranked = 22 ativos
full-history validated = 8 ativos
```

Apenas HL ficou elegível para persistência.

Resultado de HL:

```text
baseline        = US$ 4.990.517,905
base + HL       = US$ 7.201.723,486
delta           = US$ 2.211.205,581
delta percentual = +44,308%
```

O motivo gravado foi:

```text
positive_marginal_capital
```

Esse artefato é uma demonstração concreta de que, naquele estágio, o sistema
estava usando ganho marginal de capital como gate, e não simplesmente a posição
no ranking.

Nem todo ativo aprovado em campanhas paralelas acabou no ramo canônico que
levou ao U56. Por isso, HL não deve ser automaticamente inserido na genealogia
da Strategy #10 final.

---

# 7. U44: primeiro checkpoint grande diretamente preservado

Um export de 1º de setembro contém exatamente 44 ativos.

Comparado aos 25 originais, os 19 adicionais eram:

```text
BXMT, CCK, CEF, CNQ, CORT, CXW, DDS, DNN, ENS, GKOS,
KKR, LKFT, PXLW, RACE, SCSC, TX, UNF, VRTS, XSD
```

A lista completa U44 era:

```text
AAPL, ADC, ADEA, ADI, ADM, AMD, AMZN, AVGO, BXMT, CAT,
CCK, CEF, CNQ, CORT, COST, CRM, CXW, DDS, DNN, ENS,
GKOS, GOOGL, HD, JPM, KKR, LKFT, LLY, META, MSFT, NFLX,
NVDA, ORCL, PXLW, RACE, SCSC, SPY, TSLA, TX, UNF, V,
VRTS, WMT, XOM, XSD
```

Conversas da época registraram:

```text
Strategy #10 rev.1 = 26 ativos
Strategy #10 rev.2 = 44 ativos
rev.1 capital ~ US$ 1.074.831
rev.2 capital ~ US$ 5.394.605
```

Foi registrado também que CCS estava na rev.1 e saiu na rev.2. A comparação de
listas sugere que rev.2 removeu CCS e incorporou o conjunto acima, mas a
cronologia individual dentro desse salto não está integralmente preservada.

---

# 8. U44 -> U51

Outro export direto contém exatamente 51 ativos.

A diferença entre U44 e U51 é:

```text
VNCE
UNFI
DOC
CLMT
APD
MGM
MAN
```

Logo:

```text
44 + 7 = 51
```

O backtest de Strategy #10 rev.3 foi registrado com:

```text
51 ativos
capital final = US$ 14.065.953,30
CAGR = 227,12%
Sharpe = 2,294
MaxDD = -28,24%
319 rotações
OOS = 2020-07-22 até 2026-08-31
```

Houve reruns próximos com capital ligeiramente diferente devido ao snapshot de
mercado, mas a composição de 51 permaneceu a mesma.

---

# 9. U51 -> U56

## 9.1 MYE

A passagem 51 -> 52 adicionou MYE.

Resultado registrado:

```text
capital ~ US$ 13,81 mi -> US$ 15,52 mi
delta ~ +12,38%
sessões alteradas ~ 29 de ~1.536
lucro direto atribuído a MYE ~ US$ 103 mil
```

Esse caso foi importante para perceber que a maior parte do benefício podia vir
do **efeito de trajetória** e não apenas do P/L direto do novo ticker.

## 9.2 YANG

A passagem 52 -> 53 adicionou YANG.

Checkpoint histórico:

```text
53 ativos
capital ~ US$ 19,86 mi
```

## 9.3 MKSI, MCS e ECC

O conjunto que consolidou o U56 adicionou, em revisões intermediárias:

```text
MKSI, MCS, ECC
```

O artefato final U56 preserva exatamente:

```text
25 originais
+ DDS, CNQ, VRTS, XSD, CXW, ENS, CORT, CCK, CEF, GKOS,
  RACE, TX, UNF, BXMT, PXLW, KKR, SCSC, LKFT, DNN,
  VNCE, UNFI, DOC, CLMT, APD, MGM, MAN, MYE, YANG,
  MKSI, MCS, ECC
= 56
```

Strategy #10 rev.12 foi registrada com:

```text
56 ativos
capital histórico ~ US$ 23.712.122,62
snapshot = 2026-09-04
```

A ordem exata de inclusão individual de MKSI/MCS/ECC não está tão bem
preservada quanto U44 e U51. A composição final, porém, é inequívoca.

---

# 10. Correções metodológicas do Asset Discovery

Entre 28 e 29 de agosto o fluxo passou por várias tentativas de tornar a seleção
mais "causal":

```text
ec1f145 feat(asset-discovery): require nested causal validation and final certification
d7ffb8d enforce causal validation boundaries
75a118f parallelize validation and remove research cap
e2e9ac0 allow certified assets to be appended to Strategy
d101763 restore current-campaign candidate selection
```

Entretanto, em 29 de agosto foi identificada a necessidade de voltar ao critério
econômico completo:

```text
cbbac30 fix(asset-discovery): restore full-history capital contribution protocol
```

O protocolo corrigido v10.3 passou a declarar:

```text
method = full_strategy_history_replay
history_must_start_with_strategy = True
continuous_history_required = True
approval_metric = ending_capital_delta_rate
approval_rule = candidate_or_selected_universe_must_increase_final_strategy_capital
```

A persistência passou a usar o motivo:

```text
positive_full_history_strategy_capital
```

Também ficou registrado explicitamente:

```text
survivorship_bias_fully_eliminated = false
```

Isto é importante: o protocolo ficou melhor que as versões anteriores, mas não
transforma essas campanhas retrospectivas em validação prospectiva intocada.

---

# 11. O experimento U72 e o rollback

A evidência mais importante contra a ideia "mais ativos é sempre melhor" veio
depois do U56.

Uma Strategy #10 chegou a **72 ativos**, adicionando estes 16 ao U56:

```text
ADT, ALV, AX, BABA, DCOM, DXCM, HNI, IDCC,
KBH, PB, QID, RF, SGRY, SNA, UTI, VTI
```

A composição U72 está preservada diretamente em export.

O problema: essa expansão não passou pelo mesmo gate econômico completo que
havia construído o U56. Os próprios logs da campanha registravam:

```text
No full-history economic replay was executed.
```

O capital do baseline expandido caiu para aproximadamente:

```text
US$ 1.551.904,20
```

contra aproximadamente:

```text
US$ 23.712.122,62
```

no U56 histórico.

Uma validação posterior tentou adicionar mais 12 candidatos ao U72 e também
falhou de forma extrema.

**Conclusão:** U56 não apareceu porque "56 era o número ideal". U56 apareceu
como ponto recuperado após uma expansão permissiva demonstrar que aumentar o
universo sem contribuição econômica validada podia destruir a trajetória.

---

# 12. Predictive-first e por que o ranking sozinho foi insuficiente

Em setembro a busca foi novamente acelerada.

Commit:

```text
34568f8 perf: switch asset discovery to predictive-first mode
```

Além do ranker, passou a existir uma medida de similaridade de estado via
acordo de folhas do LightGBM:

```text
state_novelty_score = 1 - max_leaf_agreement
```

A ideia era encontrar candidatos com estados técnicos transferíveis em relação
ao universo-base.

Em uma campanha de 5 de setembro:

```text
baseline = U56
amostra externa determinística = 100
universo Alpaca ativo ~ 13 mil símbolos
candidatos preditivos = 92
```

Naquela execução, o log dizia explicitamente que a contribuição econômica
completa **não seria executada automaticamente**.

Isso explica por que a existência de 92 "predictive candidates" nunca deve ser
lida como "92 ativos aprovados".

Uma tentativa posterior de Marginal Capital Replay nesses 92 registrou muitos
erros técnicos:

```text
persistent candidates = 0
low adherence = 1
history rejected = 2
research context rejected = 1
technical failures = 88
```

Esse resultado não prova que 92 candidatos eram economicamente ruins. Ele prova
que aquela execução não entregou uma avaliação econômica utilizável para a
maioria do conjunto.

---

# 13. Pesquisas posteriores sobre seleção de ativos

Depois do Asset Discovery do produto, foram criadas várias pesquisas para
entender por que um candidato aparentemente bom podia piorar a conta conjunta.

## 13.1 Asset Timing vs Buy & Hold

Pergunta:

> o timing da própria Strategy consegue operar este ativo fora da amostra melhor
> do que comprar e manter o mesmo ativo?

Gate padrão:

- maioria dos folds OOS vence buy-and-hold do próprio ativo;
- mediana do excesso por fold positiva;
- retorno composto de timing > buy-and-hold;
- períodos investidos melhores que períodos em CASH.

O universo era congelado antes de um único backtest final.

## 13.2 Rotation Leadership

Passou a estudar oportunidades de liderança dentro da rotação, tentando evitar
confundir "ativo intrinsecamente bom" com "ativo útil para substituir outro".

## 13.3 Universe Scale

Testou universos aninhados:

```text
56 -> 82 -> 250 -> 500
```

com os mesmos parâmetros/modelo.

Pergunta: aumentar apenas o conjunto de oportunidades, mantendo a Strategy,
melhora ou degrada?

## 13.4 Marginal Rotation Contribution v1

A v1 procurava eventos em que o candidato passava a superar o líder e agregava
rótulos futuros de ganho relativo.

Ela selecionou UTI, mas na validação posterior:

```text
U56 = US$ 9.716,60
U57 (+UTI) = US$ 9.517,34
delta = -2,0507%
```

Foi descoberto que somar rótulos futuros sobrepostos não equivale ao crescimento
de uma conta única.

## 13.5 Marginal Rotation Contribution v2

A v2 corrigiu esse defeito:

- replay cronológico da conta;
- posição e capital atravessam folds;
- execução na abertura seguinte;
- custos reais do motor;
- contribuição:
  `log(capital_expandido / capital_atual)`;
- trajetória recalculada após cada inclusão.

Essa mudança é metodologicamente relevante para o próximo teste.

## 13.6 Cross-Asset Utility Signature

Foi testado um único LightGBM pooled, treinado apenas nos ativos protegidos, sem
ticker como atributo:

```text
ROTATION_FEATURES -> forward_risk_adjusted_utility
```

O objetivo era descobrir se estados úteis aprendidos no universo-base eram
transferíveis para ativos externos em uma escala comum.

Essa linha e seus sucessores mostraram que score/assinatura individual não
resolve automaticamente a interação de trajetória da carteira.

---

# 14. U56 -> U59 -> U67: outra fase, não confundir com o Asset Discovery original

O crescimento posterior que levou ao U67 ocorreu em pesquisa separada.

## 14.1 U56 -> U59

Foram adicionados:

```text
COLB, AMS, FOXF
```

Checkpoint:

```text
U59 ~ US$ 30.080.091,01
```

## 14.2 Smart20 e U67

Um conjunto de 20 candidatos foi testado individualmente.

Oito apresentaram resultado positivo naquele protocolo:

```text
THO, WDAY, EXR, XEL, SBFG, PAYX, MUX, SXC
```

A inclusão indiscriminada dos 20 foi ruim:

```text
U59 + Smart20 ~ US$ 2.017.935,51
```

A inclusão apenas dos oito positivos produziu o U67 histórico:

```text
U67 = U59
    + THO, WDAY, EXR, XEL, SBFG, PAYX, MUX, SXC

capital histórico ~ US$ 58.557.157,67
```

Isso reforçou novamente que "candidato promissor" e "conjunto de candidatos
promissor" são problemas diferentes.

Mais tarde, com o snapshot atualizado e a reprodução operacional, o mesmo U67
chegou ao checkpoint de aproximadamente US$ 76,93 milhões em 6 de outubro de
2026. Esse valor atualizado não muda a genealogia do universo.

---

# 15. Genealogia canônica recuperada

A sequência mais defensável hoje é:

```text
U20
  + ADC, ADEA, ADI, ADM
= U24

U24
  + HD
= U25

U25
  + DDS, CNQ, VRTS, XSD
  + CXW, ENS, CORT
= U32

U32
  + CCK
= U33

U33
  + CEF
= U34

U34
  + GKOS
= U35

U35
  + RACE
= U36

... campanhas intermediárias ...

U44
  = U25 + 19 candidatos confirmados

U44
  + VNCE, UNFI, DOC, CLMT, APD, MGM, MAN
= U51

U51
  + MYE
= U52

U52
  + YANG
= U53

U53
  + MKSI, MCS, ECC
= U56

U56
  + 16 ativos sem replay econômico completo
= U72  [FALHOU / rollback]

U56
  + COLB, AMS, FOXF
= U59

U59
  + THO, WDAY, EXR, XEL, SBFG, PAYX, MUX, SXC
= U67
```

No TCC atual, U67 é a lista nominal. DOC e CLMT são excluídos na preparação
estrutural, resultando em 65 ativos efetivamente entregues ao modelo.

---

# 16. O que realmente determinou o número de ativos?

Não foi um valor ótimo escolhido a priori.

O número evoluiu porque, em diferentes fases, o MCT tentou responder a três
perguntas:

1. **Elegibilidade**: o ativo possui histórico, liquidez e comportamento
   compatíveis?
2. **Ranking**: entre muitos candidatos, quais merecem ser avaliados primeiro?
3. **Contribuição conjunta**: ao inserir o ativo, a Strategy inteira termina
   melhor?

O ponto mais importante da história é que a terceira pergunta foi a única que
conseguiu impedir de forma consistente expansões destrutivas.

Quando o processo se afastou dela e passou a promover uma expansão apenas por
ranking/qualificação preditiva, surgiu o U72 e o resultado desabou.

Ao mesmo tempo, os trabalhos posteriores mostraram que até o replay marginal
retrospectivo pode superajustar o período observado. Portanto, para um novo
experimento não basta reproduzir o mesmo replay de 2016-2026 e escolher o maior
capital.

---

# 17. Lacunas que ainda permanecem

A pesquisa recuperou muito mais do que estava documentado no TCC, mas alguns
elos não possuem artefato intermediário completo:

- a criação original exata dos 20 ativos-base;
- a ordem individual de CXW, ENS e CORT entre U29 e U32;
- cada revisão individual entre U36 e U44;
- a ordem individual de MKSI, MCS e ECC entre U53 e U56;
- alguns universos paralelos, por exemplo Strategy #24 com 48 ativos, não fazem
  parte da genealogia canônica da Strategy #10 e não devem ser misturados nela.

Essas lacunas não alteram as composições diretamente preservadas de U25, U44,
U51, U56, U72 e do U67 final.

---

# 18. Implicação para o próximo teste

O teste seguinte não deve repetir PCA, clustering ou assinatura de ativos.

A hipótese historicamente mais fiel é:

> O mecanismo que construiu o universo lucrativo do MCT funcionava porque
> candidatos passavam por uma triagem preditiva e, principalmente, por um teste
> de contribuição marginal na trajetória completa da Strategy. Esse ganho,
> porém, precisa ser demonstrado fora do período usado para selecionar o ativo.

Portanto, a nova pesquisa deve reproduzir o **MCT Asset Discovery econômico**,
mas com fronteira temporal bloqueada.

Desenho recomendado para a próxima implementação:

1. usar um universo-base congelado anterior à expansão;
2. congelar um conjunto externo de candidatos antes de observar o período final;
3. aplicar filtros de identidade, histórico e qualidade iguais para todos;
4. usar o ranker somente para ordenar/limitar custo computacional;
5. executar Marginal Capital Replay cronológico estilo v2 somente no período de
   seleção;
6. congelar os candidatos aceitos;
7. executar **uma única validação posterior intocada**, na qual nenhum ativo
   pode ser adicionado/removido;
8. comparar contra a mesma Strategy-base no mesmo período;
9. registrar separadamente contribuição direta do ticker e efeito de trajetória.

Esse teste responde à pergunta que a história do MCT ainda não respondeu de
forma limpa:

> os ativos que pareciam aumentar marginalmente o capital no período de seleção
> continuam aumentando o capital em um período futuro que não participou da
> escolha?

Se a resposta for não, a expansão histórica deve ser tratada como seleção
retrospectiva. Se for sim, teremos evidência muito mais forte de que o mecanismo
do MCT Asset Discovery realmente generaliza.

---

## Referências de implementação e histórico consultadas

### MCT API

Repositório:

```text
betovlima/market_cycle_trader_api
```

Commits centrais:

```text
a4dd70e  remove legacy Asset Discovery implementation
d7e023b  rank Asset Discovery results by marginal economic contribution
ef07bde  add full-strategy candidate validation
9befc62  fast-scan candidates before full replay validation
ec1f145  require nested causal validation and final certification
75a118f  parallelize validation and remove research cap
e2e9ac0  allow certified assets to be appended to Strategy
d101763  restore current-campaign candidate selection
cbbac30  restore full-history capital contribution protocol
34568f8  switch Asset Discovery to predictive-first mode
```

Branches históricas relevantes:

```text
feature/10.0.0-causal-asset-discovery
feature/10.3.0-full-history-asset-discovery
research/asset-marginal-rotation-contribution-v1
research/asset-marginal-rotation-contribution-v2
research/asset-rotation-independent-validation-v1
research/asset-rotation-leadership-v1
research/asset-rotation-universe-scale-v1
research/asset-signature-leave-one-out
research/cross-asset-utility-signature-v1
```

Documentos consultados no Git:

```text
scripts/README_asset_marginal_rotation_contribution_v2.md
scripts/README_asset_rotation_universe_scale.md
scripts/README_asset_timing_vs_buyhold.md
scripts/README_cross_asset_utility_signature.md
```

Artefatos históricos privados consultados incluem exportações de Asset
Discovery de 24/25 de agosto, 28 de agosto, 1º de setembro e 5-7 de setembro de
2026, além de dumps de Strategy e backtests preservados no projeto.

