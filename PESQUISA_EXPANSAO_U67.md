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
