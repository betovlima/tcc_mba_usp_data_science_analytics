# Pesquisa de expansão incremental do universo U67

## Objetivo

Partir do universo U67 oficial sem reconstruí-lo e testar somente a adição de
ativos externos.

O U67 permanece como baseline fixo. Nenhum ativo existente é removido,
substituído ou reordenado. O contrato continua sendo 67 símbolos configurados
e 65 ativos efetivos no replay atual, em razão das exclusões estruturais já
documentadas de DOC e CLMT.

## Pergunta de pesquisa

Quais ativos externos, quando adicionados individualmente ao U67, aumentam de
fato o capital final da estratégia?

A partir da versão 1.1.0, essa pergunta é respondida diretamente pelo replay
financeiro. Preço, liquidez, volume, momentum e Learning-to-Rank não eliminam
candidatos antes do teste econômico.

## Baseline

```text
U67 atual
requested = 67
effective no replay atual = 65
capital de referência oficial em 2026-10-06 ≈ US$ 76.927.051,39
```

O baseline é recalculado pelo próprio runner antes de avaliar qualquer
candidato. A campanha é abortada se o capital não reproduzir o checkpoint
oficial até o centavo.

## Amostra congelada

A versão 1.1.0 reutiliza exatamente os 500 símbolos da primeira campanha,
preservados em:

```text
pesquisa_u67_amostra_500.csv
```

Isso permite comparar o novo desenho com a execução anterior sem trocar a
amostra depois de observar os resultados.

O catálogo US Equity ativo da Alpaca ainda é consultado e registrado como
contexto, mas não funciona como um novo gate para os 500 símbolos congelados.

## Regra de seleção

A cadeia de decisão passa a ser:

```text
500 símbolos congelados
        ↓
auditoria estrutural mínima
        ↓
U67 + candidato
        ↓
delta de capital final
        ↓
positivos
        ↓
validação conjunta
```

Foram retirados do caminho de decisão os dois cortes anteriores:

```text
500 → 117 por filtros de mercado
117 → 24 por Learning-to-Rank
```

Essas etapas podem continuar sendo estudadas como variáveis explicativas, mas
não decidem quem merece o replay financeiro.

## Auditoria estrutural obrigatória

A auditoria anterior ao replay não procura desempenho. Ela somente impede
comparações inválidas.

Um candidato é rejeitado antes do replay quando não é possível construir uma
série comparável ao U67, por exemplo:

- histórico insuficiente para a janela exigida;
- sessões obrigatórias ausentes;
- quebra estrutural de identidade;
- preços de execução incompletos;
- impossibilidade de obter o histórico RAW ou Corporate Actions necessários.

Não há bridge entre fontes, preenchimento artificial de histórico nem correção
manual da série.

## Dados financeiros

```text
feed = SIP
timeframe = 1Day
adjustment = RAW
início = 2016-01-01
cutoff = 2026-10-06
Corporate Actions = Alpaca
splits = normalizados pelo mesmo pipeline do U67
```

Os dados de candidatos ficam em cache local separado e não alteram
`dados/u67`.

## Replay marginal direto

Para todo candidato estruturalmente válido:

```text
baseline = U67
candidate = U67 + ativo
delta = capital(candidate) - capital(baseline)
```

O U67 usa o mesmo calendário, folds, custos, regras de execução e configuração
da reprodução oficial. Os modelos do U67 são treinados uma vez por fold e seus
caches de utilidade são reutilizados. Para cada cenário marginal é treinado
somente o modelo do novo ativo, e a `switch_margin` é recalibrada para o
universo testado.

São registrados:

- capital final;
- delta absoluto e percentual;
- CAGR;
- Sharpe;
- MaxDD;
- pior fold;
- rotações;
- decisões alteradas;
- operações do novo ativo.

O rótulo primário de descoberta é:

```text
positive_capital_delta = delta_capital > 0
```

As métricas de risco são preservadas para análise, mas não existe um gate
numérico de risco inventado depois de observar os resultados.

## Uso de memória na campanha de 500

A versão 1.1.0 processa os candidatos sequencialmente. Frames e caches de
candidatos com delta não positivo são descartados depois do replay.

Somente os candidatos positivos permanecem em memória para a validação conjunta.
Isso permite ampliar a busca sem manter centenas de modelos de candidatos
simultaneamente.

A execução também grava checkpoints parciais de replay e auditoria para
facilitar diagnóstico se uma campanha longa for interrompida.

## Validação conjunta

Depois dos testes individuais:

```text
U67 + todos os candidatos positivos
```

é executado porque contribuições marginais individuais podem interagir.

Se o conjunto completo não superar o baseline, o runner aplica a busca gulosa
já existente sobre os positivos individuais, sem remover qualquer ativo do
U67.

## Interpretação da assinatura

A assinatura de interesse não é definida pelos filtros antigos. Ela será
estudada a partir da variável econômica observada:

```text
contribuição marginal real do ativo ao U67
```

Assim, ativos positivos e negativos da mesma amostra poderão ser comparados
posteriormente para investigar quais características temporais e contextuais
antecedem contribuição econômica positiva.

## Saídas da versão 1.1.0

```text
output/u67_asset_expansion/
    candidate_catalog.csv
    candidate_sample.csv
    candidate_download.csv
    continuity_audit.csv
    marginal_replay.csv
    positive_candidates.csv
    combined_positive_replay.json
    decision_changes.csv
    trades_new_assets.csv
    scenario_fold_calibration.csv
    u67_baseline_metrics.json
    pesquisa_u67_asset_expansion.json
    pacote_analise_u67_asset_expansion.zip
```

Durante uma execução ainda incompleta também podem existir:

```text
marginal_replay_partial.csv
continuity_audit_partial.csv
```

## Limitações

A amostra original foi derivada de um catálogo de ativos ativos e, portanto, o
viés de sobrevivência do universo externo não está totalmente eliminado.

A versão 1.1.0 reduz um problema diferente: dentro dos 500 símbolos congelados,
nenhum filtro de preço, liquidez, volume ou ranking escolhe antecipadamente os
ativos que recebem o replay financeiro.

A campanha continua sendo exploratória. Uma assinatura aprendida a partir dos
resultados deverá ser confirmada posteriormente em ativos não usados para
construí-la.


## Implementação v1.2.0: otimização conjunta dos 15 positivos

A campanha direta sobre os 500 símbolos produziu 15 candidatos com delta
marginal positivo:

```text
CRT, NBR, CAPR, PNBK, PENN, JBL, LEU, JBSS, STAG, MCHB,
EML, MAR, LAD, TOWN, ACIW
```

Antes da busca conjunta, a v1.2.0 exige reproduzir o checkpoint:

```text
U67 + 15 positivos = US$ 290.727.954,41
```

com tolerância absoluta de um centavo. Se esse checkpoint divergir, a busca é
interrompida.

O modo padrão de `pesquisar_expansao_u67_spyder.py` passa a ser
`subset_optimization`. O runner não repete a campanha dos 500 ativos. Ele
reutiliza somente os 15 positivos e os dados RAW armazenados no cache local,
reajusta seus modelos por fold e executa uma busca determinística em duas
direções:

```text
forward selection
    começar em U67
    testar a adição de cada candidato restante
    aceitar a adição que produz o maior capital
    parar quando nenhuma adição melhora o capital

backward elimination
    partir de U67 + todos os 15
    testar a remoção de um candidato por vez
    aceitar a remoção que produz o maior capital
    repetir até nenhuma remoção melhorar

backward from forward
    aplicar a mesma eliminação ao conjunto encontrado pelo forward
```

O vencedor é o conjunto com maior capital entre `all_positive`, `forward`,
`backward_from_all` e `backward_from_forward`. Empates econômicos dentro do
critério de busca são resolvidos de forma determinística e favorecem o conjunto
menor no comparativo final.

Essa é uma busca heurística, não uma enumeração exaustiva das 32.767
combinações não vazias possíveis dos 15 ativos. A escolha evita um custo
computacional muito maior e mantém toda a trilha de cenários efetivamente
avaliados.

As novas saídas ficam em:

```text
output/u67_subset_optimization/
    candidate_download.csv
    continuity_audit.csv
    subset_scenarios_partial.csv
    subset_scenarios.csv
    subset_search.csv
    decision_changes.csv
    trades_new_assets.csv
    scenario_fold_calibration.csv
    selected_subset_replay.json
    pesquisa_u67_subset_optimization.json
    pacote_analise_u67_subset_optimization.zip
```

O modo anterior permanece no mesmo arquivo como `RUN_MODE =
"direct_marginal"` para reprodução da campanha de 500 ativos.


## Implementação v1.3.0: assurance, generalização e universo dinâmico

O resultado de descoberta de aproximadamente US$ 587,17 milhões passa a ser
tratado como Challenger, não como resultado final.

O modo padrão do runner é agora:

```text
RUN_MODE = "assurance_validation"
```

A versão exige reproduzir até o centavo três checkpoints conhecidos:

```text
U67 = US$ 76.927.051,39
U67 + 15 positivos = US$ 290.727.954,41
U67 + 10 selecionados = US$ 587.165.941,13
```

Depois executa cinco famílias de garantia.

### 1. Overfitting e underfitting do LightGBM

Para cada ativo e fold são gravados:

```text
train MAE / RMSE
OOS MAE / RMSE
naive mean train MAE / RMSE
naive mean OOS MAE / RMSE
train skill vs naive
OOS skill vs naive
generalization gap
```

Os warnings não usam um limiar arbitrário de gap:

```text
overfit_warning:
    train skill > 0
    e OOS skill <= 0

underfit_warning:
    train skill <= 0
    e OOS skill <= 0
```

Assim, o modelo é comparado a um preditor ingênuo que usa somente a média do
target observada no período de ajuste.

### 2. Validação temporal da seleção do universo

O conjunto de 15 candidatos positivos permanece congelado para esta fase.
A seleção do subconjunto é refeita usando somente folds anteriores:

```text
fold 1
    selecionar universo
    testar no fold 2

folds 1 + 2
    selecionar universo
    testar no fold 3
```

O score de seleção ignora o fold futuro. Como os modelos/políticas são
independentes por fold, o cálculo dos cenários completos não altera o score
histórico usado na decisão.

Esta validação ainda não é uma redescoberta nested dos 500 ativos. O pool dos
15 foi descoberto usando a janela controlada completa. O artefato registra
explicitamente essa limitação.

### 3. Leave-one-asset-out

Para cada um dos 10 ativos do Challenger, o runner executa novamente o cenário
sem aquele ativo e mede:

```text
capital
delta contra o Challenger
Sharpe
MaxDD
worst fold
```

O objetivo é detectar dependência excessiva de um único ativo.

### 4. Stress de custos

O Challenger é reavaliado com:

```text
base
slippage = 5 bps
slippage = 10 bps
taxas regulatórias = 2x
taxas regulatórias = 3x
```

O stress altera execução e recalibração da margem, mas preserva os modelos já
treinados para isolar sensibilidade operacional a custos.

### 5. Reprodutibilidade

O Challenger é executado novamente na mesma campanha. São comparados:

```text
capital final até o centavo
sequência de selected_asset
diferença máxima absoluta da curva de capital
```

### Universo não fixo

A lista atual não representa uma lista permanente de ativos.

O objetivo final da pesquisa é estimar algo equivalente a:

```text
valor_marginal(ativo, universo_atual, regime, tempo)
```

para suportar futuramente três decisões:

```text
ADICIONAR
MANTER
REMOVER
```

usando apenas informação disponível no momento da decisão.

Nesta fase o U67 congelado continua sendo a bancada experimental para medir
efeitos de forma controlada. A transição para atualização diária será feita
somente depois que a assinatura temporal e contextual estiver suficientemente
validada.

### Saídas da v1.3.0

```text
output/u67_assurance_validation/
    candidate_download.csv
    continuity_audit.csv
    model_generalization.csv
    temporal_universe_validation.csv
    temporal_universe_search.csv
    leave_one_asset_out.csv
    cost_stress.csv
    reproducibility.json
    assurance_scenarios.csv
    assurance_scenarios_partial.csv
    scenario_fold_calibration.csv
    pesquisa_u67_assurance_validation.json
    pacote_analise_u67_assurance_validation.zip
```


## Implementação v1.4.0: qualidade de decisão e selection overfitting

A assurance v1.3.0 mostrou que MAE/RMSE absolutos não são suficientes para
explicar o desempenho econômico. A v1.4.0 passa a testar diretamente a função
que interessa à política: ordenar alternativas de investimento.

O modo padrão passa a ser:

```text
RUN_MODE = "decision_validation"
```

Os checkpoints U67, U67+15 e U67+10 continuam obrigatórios e devem reproduzir
até o centavo antes dos novos diagnósticos.

### Diagnóstico cross-sectional de ranking

Em cada sessão OOS, a previsão disponível no momento da decisão é comparada à
`forward_risk_adjusted_utility` que somente se torna conhecida depois.

São avaliados três escopos:

```text
U67 baseline
U67 + 10 Challenger
pool dos 15 candidatos positivos
```

As métricas incluem:

```text
Spearman / Information Coefficient
NDCG@5
NDCG@10
sign accuracy
top-1 exact hit
top-1 previsto dentro do top-3 realizado
top-1 previsto dentro do top-5 realizado
percentil realizado do top-1 previsto
overlap top-3
utility realizada do top-1 previsto
utility oracle
regret do top-1
percentil realizado do ativo efetivamente selecionado pela política
```

Essas métricas são somente diagnósticos de avaliação. A utility futura não
participa da decisão histórica.

### Reconstrução da paisagem de 180 subconjuntos

A busca forward/backward da v1.2.0 é repetida com os mesmos 15 candidatos. A
execução exige exatamente 180 cenários únicos e o mesmo subconjunto de 10
ativos. Para cada cenário são preservados:

```text
capital final
CAGR
Sharpe
MaxDD
worst fold
Sharpe diário
série diária de retornos
```

Essa matriz permite medir risco de seleção retrospectiva.

### Deflated Sharpe Ratio

A v1.4.0 calcula o Deflated Sharpe Ratio do Challenger em três leituras:

```text
1. 180 trials brutos da busca de subconjuntos
2. número efetivo de trials estimado pelos autovalores da matriz de correlação
3. sensibilidade conservadora com 326 = 146 marginais + 180 subconjuntos
```

O terceiro número não é tratado como quantidade de testes independentes.
Ele serve somente como cenário conservador de sensibilidade.

Além do DSR, é preservado o Probabilistic Sharpe Ratio e o Sharpe esperado do
melhor trial sob múltiplos testes.

### Probability of Backtest Overfitting

O CSCV/PBO divide a matriz de retornos em 10 blocos e avalia todas as
combinações simétricas de cinco blocos em sample e cinco fora de sample:

```text
C(10,5) = 252 divisões
```

São produzidos dois PBOs:

```text
log_growth
    alinhado ao critério usado para maximizar capital

Sharpe
    diagnóstico complementar de risco-retorno
```

Para cada divisão, registra-se qual cenário venceu no in-sample e sua posição
relativa no OOS.

### Piloto de universo dinâmico

A seleção temporal permanece limitada ao pool fixo dos 15 positivos nesta
versão:

```text
fold 1 -> selecionar -> testar fold 2
folds 1+2 -> selecionar -> testar fold 3
```

Além do resultado temporal, a v1.4.0 converte a mudança de composição em ações:

```text
ADICIONAR
MANTER
REMOVER
```

A transição entre folds é somente um piloto de baixa frequência. A política
final deverá redescobrir candidatos externos e reavaliar ativos internos em
atualização diária ou outra frequência definida antes da execução.

### Saídas da v1.4.0

```text
output/u67_decision_validation/
    candidate_download.csv
    continuity_audit.csv
    decision_rank_sessions.csv
    decision_rank_summary.csv
    full_window_subset_search.csv
    multiple_testing_scenarios.csv
    multiple_testing_daily_returns.csv
    multiple_testing_summary.json
    pbo_log_growth.csv
    pbo_sharpe.csv
    dynamic_universe_temporal_validation.csv
    dynamic_universe_transitions.csv
    dynamic_universe_search.csv
    scenario_fold_calibration.csv
    pesquisa_u67_decision_validation.json
    pacote_analise_u67_decision_validation.zip
```

O objetivo desta etapa não é encontrar mais capital. É determinar se o
resultado de descoberta sobrevive a diagnósticos de ranking e múltiplos testes
e se a composição ótima mostra comportamento temporal compatível com um
universo não permanente.


## Hotfix v1.4.1: serialização JSON e recuperação sem rerun

Uma execução completa da v1.4.0 chegou ao final dos cálculos, incluindo mais de
300 cenários totais após as buscas temporais, mas falhou somente ao escrever o
JSON final:

```text
ValueError: Out of range float values are not JSON compliant: nan
```

A causa foi um `NaN` legítimo em campos diagnósticos agregados. O contrato do
artefato usa `allow_nan=False`, portanto valores não finitos precisam ser
convertidos explicitamente para `null` antes da serialização.

A v1.4.1 adiciona sanitização recursiva para:

```text
NaN       -> null
+Infinity -> null
-Infinity -> null
np.integer -> int
np.bool_   -> bool
```

A correção não altera modelos, decisões, retornos, DSR, PBO, ranking ou seleção
temporal.

Como a falha ocorreu depois dos cálculos e os CSV/JSON intermediários já tinham
sido gravados, foi criado:

```text
finalizar_u67_decision_validation_spyder.py
```

Esse recovery valida os 180 cenários, os checkpoints U67+15 e U67+10, recupera
os hashes dos 15 candidatos, recompõe somente o JSON final e cria o ZIP de
análise. Ele não treina modelos, não executa backtests e não apaga a pasta de
output existente.

Para uma execução que apresentou exatamente essa falha, usar o recovery em vez
de repetir a campanha completa.


## Resultados da v1.4.0 / recovery v1.4.1

A campanha de validação de decisão foi concluída e recuperada sem rerun após o
hotfix de serialização. Os checkpoints permaneceram exatos:

```text
U67                         = US$ 76.927.051,38897176
U67 + 15 positivos          = US$ 290.727.954,41339606
U67 + 10 Challenger         = US$ 587.165.941,1330212
subconjuntos full-window    = 180
```

### Qualidade de ranking

No universo U67+10, considerando 75 ativos e 1.503 sessões OOS:

```text
Spearman IC médio                       = 0,03419
NDCG@5 médio                            = 0,52660
NDCG@10 médio                           = 0,53826
sign accuracy                           = 63,10%
top-1 exato                             = 2,26%
top-1 previsto no top-3 realizado       = 8,25%
top-1 previsto no top-5 realizado       = 12,71%
percentil realizado médio do top-1      = 55,38%
```

O sinal cross-sectional global é pequeno, mas há concentração no topo. Como
referência puramente combinatória, em 75 ativos um top-1 aleatório acertaria o
melhor ativo em 1,33% das sessões, estaria no top-3 em 4,0% e no top-5 em
6,67%. Portanto, o modelo mostra sinal útil no topo, mas ainda fraco para uma
ordenação completa e estável do universo.

No pool dos 15 candidatos positivos:

```text
Spearman IC médio                       = 0,01444
top-1 exato                             = 14,11%
top-1 previsto no top-3 realizado       = 25,62%
top-1 previsto no top-5 realizado       = 39,19%
percentil realizado médio do top-1      = 57,36%
```

A expectativa combinatória de top-1 exato para 15 ativos é 6,67%. Assim, a
habilidade de identificar ocasionalmente o melhor candidato existe, mas o IC
próximo de zero confirma que a classificação completa permanece ruidosa.

### Deflated Sharpe Ratio

O Challenger apresentou:

```text
Sharpe anualizado                       = 2,72854
DSR com 180 trials                      ≈ 0,9999999999997
DSR com 326 trials conservadores        ≈ 0,9999999999995
```

A dimensão efetiva estimada pela matriz de correlação foi apenas 1,2792. Essa
medida isolada é otimista porque os 180 cenários compartilham grande parte do
mesmo universo e das mesmas operações. A interpretação principal deve usar os
180 trials brutos e a sensibilidade conservadora de 326 trials, que também
mantiveram DSR próximo de 1.

### Probability of Backtest Overfitting

O CSCV com 10 blocos e 252 divisões produziu:

```text
PBO por log-growth                      = 53,57%
mediana do percentil OOS                = 48,48%

PBO por Sharpe                          = 59,52%
mediana do percentil OOS                = 36,74%
```

Esses valores são um alerta. Quando a própria rotina de otimização escolhe o
vencedor em diferentes subconjuntos temporais, esse vencedor cai abaixo da
mediana OOS em mais da metade dos casos. Portanto, a busca do melhor subconjunto
não pode ser considerada estável apenas pelo capital final.

Há, porém, uma distinção importante. Recalculando a posição OOS do Challenger
fixo de 10 ativos em todas as 252 divisões, sem reescolher a estratégia:

```text
log-growth:
    percentil OOS mediano                = 93,92%
    abaixo da mediana OOS                = 0,00%
    top-decile OOS                       = 58,33%

Sharpe:
    percentil OOS mediano                = 82,04%
    abaixo da mediana OOS                = 7,54%
    top-decile OOS                       = 33,73%
```

Assim, o PBO elevado descreve principalmente a instabilidade do mecanismo de
seleção do vencedor entre muitos subconjuntos. O Challenger específico é
forte quando tratado como cenário fixo, mas ainda foi descoberto usando a
janela completa e, portanto, não deve ser promovido como evidência prospectiva
independente.

### Correlação entre cenários

As 180 séries de retorno são fortemente correlacionadas:

```text
correlação par-a-par média               ≈ 0,879
mediana                                  ≈ 0,879
10º percentil                            ≈ 0,746
90º percentil                            ≈ 0,988
séries de retorno exatamente distintas  = 158 de 180
```

Isso explica por que a dimensão efetiva de trials é pequena e reforça a
necessidade de interpretar DSR e PBO em conjunto.

### Universo dinâmico

A validação temporal repetiu a composição apenas com informação dos folds
anteriores.

Etapa 1:

```text
seleção usando fold 1:
    NBR, PENN, EML

holdout fold 2:
    retorno universo dinâmico = 77,917619
    retorno U67               = 77,898220
```

A vantagem foi muito pequena e deve ser tratada como empate econômico.

Etapa 2:

```text
seleção usando folds 1+2:
    CRT, NBR, PNBK, PENN, JBL, STAG, MCHB, TOWN

holdout fold 3:
    retorno universo dinâmico = 35,536979
    retorno U67               = 24,484656
```

A segunda etapa apresentou vantagem material no holdout. A transição entre as
duas composições foi:

```text
MANTER:
    NBR, PENN

REMOVER:
    EML

ADICIONAR:
    CRT, PNBK, JBL, STAG, MCHB, TOWN
```

Esse comportamento é compatível com a hipótese central da pesquisa: o universo
não deve ser uma lista permanente.

### Conclusão da v1.4

A v1.4 não elimina o risco de selection overfitting. Ela mostra duas evidências
simultâneas:

1. o Challenger fixo de 10 ativos é muito forte em DSR e em sua posição OOS
   dentro das divisões CSCV;
2. a rotina de escolher o vencedor entre muitos subconjuntos é instável, com
   PBO acima de 50%.

Portanto, a próxima etapa não deve otimizar novamente o mesmo conjunto. Ela
deve validar a regra de construção do universo em um desenho nested, no qual
candidatos externos e ativos internos sejam redescobertos usando somente
informação disponível até cada data de atualização.


## Recentragem do objetivo: pertencimento dinâmico, não ranking global

A análise conjunta dos artefatos v1.1-v1.4 e do cache congelado confirma que o
objetivo científico não é ordenar globalmente todos os ativos. O ranking
cross-sectional da v1.4 permanece apenas como diagnóstico da política
LightGBM. A pergunta principal passa a ser explicitamente:

```text
Dado o universo U_t e o contexto observável em t,
o ativo X agrega valor como opção disponível?
```

Para ativos externos, isso corresponde à contribuição de inclusão. Para ativos
já presentes, corresponde à contribuição de permanência versus remoção.

A matriz diária dos 180 cenários permite recuperar, sem novo download, a
contribuição relativa de cada um dos 15 candidatos quando testado
individualmente contra o U67 em cada fold. O resultado é:

| ativo | fold 1 vs U67 | fold 2 vs U67 | fold 3 vs U67 | full-window |
|---|---:|---:|---:|---:|
| CRT | -0,98% | +69,49% | ~0,00% | +69,50% |
| NBR | +41,24% | +0,01% | +5,66% | +56,49% |
| CAPR | -0,98% | ~0,00% | +36,91% | +36,91% |
| PNBK | -0,98% | +4,09% | +21,19% | +26,15% |
| PENN | +18,84% | +0,01% | ~0,00% | +20,03% |
| JBL | -0,98% | +6,61% | +11,96% | +19,36% |
| LEU | -15,35% | -39,29% | +102,90% | +5,30% |
| JBSS | -0,98% | ~0,00% | +5,21% | +5,21% |
| STAG | -0,98% | +4,78% | ~0,00% | +4,78% |
| MCHB | -0,98% | +4,30% | ~0,00% | +4,30% |
| EML | +2,52% | ~0,00% | ~0,00% | +3,54% |
| MAR | -0,98% | +3,26% | ~0,00% | +3,26% |
| LAD | -0,98% | -3,24% | +6,58% | +3,13% |
| TOWN | -0,98% | +1,33% | ~0,00% | +1,33% |
| ACIW | -0,98% | +1,07% | ~0,00% | +1,07% |

Os percentuais por fold são a diferença relativa de capital do cenário
`U67 + ativo` contra o U67 dentro daquele fold, derivados das séries diárias
já preservadas. Valores próximos de zero significam pouca ou nenhuma
contribuição econômica naquele período.

A tabela mostra diretamente que positividade full-window não significa
pertencimento permanente. LEU é o caso extremo: prejudica fortemente os folds
1 e 2 e só se torna muito favorável no fold 3. CAPR é essencialmente um ativo
de contribuição tardia; CRT concentra quase todo o benefício no fold 2; EML é
útil no início e depois perde relevância.

Isso também explica por que o Challenger full-window não pode ser convertido
em lista fixa. CAPR e LAD fazem parte dos 10 finais, mas não eram escolhidos
pela seleção que usava apenas folds 1+2. Seu valor aparece posteriormente.

### Perguntas já respondidas com os dados disponíveis

Os dados existentes já suportam as seguintes conclusões:

- a contribuição de um ativo muda materialmente com o tempo;
- positividade isolada full-window não garante utilidade conjunta;
- a composição útil mudou entre as duas decisões temporais observadas;
- a seleção temporal realizou ADICIONAR/MANTER/REMOVER de forma coerente com
  a hipótese de universo não permanente;
- filtros estáticos simples de preço, liquidez, momentum ou ranking global não
  são uma definição adequada da assinatura.

Ainda permanecem sem resposta causal suficiente:

- quais ativos do próprio U67 deveriam ser removidos, pois o U67 foi mantido
  fixo em todos os experimentos desta linha;
- qual é a regra de pertencimento em frequência diária ou quase diária;
- quanto de histerese/permanência mínima é necessário para evitar churn;
- se a assinatura aprendida em uma janela permanece válida quando candidatos
  externos são redescobertos de forma nested sem acesso ao futuro.

O cache atual contém os dados de mercado necessários para aprofundar essas
perguntas sem novo download dos 500 candidatos. O que falta são rótulos
contrafactuais temporais mais densos, produzidos pelo próprio replay controlado.


## Reprodução histórica conferida em 21/10/2026

A campanha **v1.4.1**, `RUN_MODE="assurance_validation"`, voltou a reproduzir **US$ 76.927.051,38897176** no Control inicial, **US$ 290.727.954,41339606** no grupo de 15 e **US$ 587.165.941,1330212** com os dez selecionados. O pacote recebido (`pacote_analise_u67_assurance_validation(1).zip`, SHA-256 `0f06561e9685b3e84194580c50a6808e4c84d30a55584b97bd19dae2fafd6c37`) registra `status=completed` e replays com capital, escolhas e curva idênticos (`exact_to_cent=true`, `selected_asset_sequence_identical=true`, `max_equity_absolute_difference=0.0`).

**Condições essenciais:** cutoff 06/10/2026, snapshot `dados/u67` com SHA `e440f59da5e684f1de59cf447abfedd9aed3f7817b3d5d681058fe631276a575`, 67 originais/65 efetivos (excluindo DOC e CLMT), mais 15 caches RAW e Corporate Actions em `dados/.u67_expansion_cache/full_raw_20261006/`. Esses 15 pares de CSV são arquivos locais não versionados no Git. A garantia de reprodução requer preservar os arquivos **e seus hashes** listados no pacote, e não apenas os tickers.

O código v1.4.1 possui downloads da Alpaca na preparação dos candidatos. Para não substituir o cache, a execução de confirmação empregou interceptação **somente em memória** das rotinas de download no Spyder; o script versionado, os modelos e a estratégia não foram modificados. Um novo download ou cutoff não é a mesma experiência. Consulte [CONTEXTO_MESTRE.md](CONTEXTO_MESTRE.md) para o procedimento offline completo.

**Interpretação:** a exatidão é de reprodução histórica, não de validação prospectiva de seleção de universo. Os 15 candidatos foram descobertos na janela completa. Os testes de seleção temporal num pool fixo, os 38/45 alertas de overfit e os valores PBO de 53,57%/59,52% continuam relevantes. Prioridade científica posterior: assinatura temporal/contextual e descoberta nested de ativos sem vazamento de informação futura.


## Encerramento do executor oficial v1.23.0 (21/10/2026)

A rotina `reproduzir_experimento.py` passou a reproduzir, sem realizar nova busca ou atualização de dados, o conjunto fixo de dez candidatos já selecionados nesta pesquisa, com 77 ativos solicitados e 75 efetivos. A execução real no Spyder registrou **US$ 587.165.941,1330212**, exatamente o checkpoint de `decision_validation`/`assurance_validation`, e `reproduced_exact_to_cent=true`. O pacote `pacote_reproducao_universo.zip`, SHA-256 `0c3518387556f147897baf121cb089f532f8355a8bb074c91fe0a6b5ec4b6ac8`, inclui 1.560 previsões/sessões e 710 registros de negociação. A saída é `output/reproducao_universo/`.

Essa validação confirma a **reprodução de um cenário selecionado retrospectivamente**, não uma seleção de novos candidatos fora da amostra. A investigação sobre assinatura temporal de pertencimento, o alerta de PBO e a ausência de redescoberta nested prospectiva permanecem assuntos científicos abertos. O manifesto `evidencias/manifest_universo_congelado.json` fixa os hashes dos arquivos históricos adicionais; os respectivos CSVs continuam fora do controle de versão. Ver seção 28 do `MANIFEST.md` e seção 8 de `CONTEXTO_MESTRE.md`.
