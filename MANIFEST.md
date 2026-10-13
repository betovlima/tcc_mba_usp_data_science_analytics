# MANIFEST — TCC MBA USP Data Science & Analytics

## 1. Finalidade deste documento

Este arquivo consolida em um único lugar o histórico técnico, as decisões
metodológicas, a arquitetura final, os resultados, as refatorações, os testes e
a avaliação crítica do projeto no contrato científico descrito abaixo.

Release e baseline científico documentados:

```text
versão: 1.22.1
tag: v1.22.1
branch de entrega: main
branch integrada: fix/u67-eligibilidade-sem-futuro
main anterior à integração: 262e5388354f258de2480151a0b9f1b9ff147db7
execução científica validada: 3f8d1b1a2784721201938071471e91fd4282f628 (dev.10)
data da atualização: 2026-10-10
```

Esta release integra à `main` a remoção da consulta aos preços da sessão
seguinte na seleção de candidatos e a correção de compra e manutenção no
mesmo universo fixo. Inclui os dados congelados, os diagnósticos de teste,
a procedência recuperada dos hiperparâmetros e as sete figuras do TCC.
As seções 20 a 23 preservam as verificações realizadas antes da integração;
a seção 24 registra a entrega autorizada em 2026-10-10 e a memória para
continuidade. As menções à `main` preservada nas seções históricas referem-se
ao estado anterior a esta release.

Este MANIFEST substitui a documentação histórica fragmentada que existia em
arquivos de contexto, checkpoints antigos, notas de mudanças por versão e
README específico da pasta de dados. A partir desta consolidação, o README fica
restrito à execução e este arquivo passa a ser a fonte documental única do
projeto.

---

## 2. Objetivo científico

O projeto avalia o crescimento do capital e o risco de uma estratégia de rotação
orientada por utilidade ajustada ao risco, estimada por regressão supervisionada.
A pesquisa utiliza um universo fixo de ativos. Sua única referência financeira
é compra e manutenção com alocação inicial igualitária no mesmo conjunto de
ativos, com o mesmo capital inicial, datas e funções de custos.

O objetivo da implementação final é permitir uma reprodução independente do
experimento usando:

- dados históricos congelados;
- preparação determinística dos dados;
- validação temporal walk-forward;
- purge temporal;
- calibração cronológica;
- simulação financeira com regras explícitas;
- checkpoints de regressão do próprio TCC;
- artefatos de auditoria;
- testes automatizados e análise estática.

O projeto final não depende de banco de dados durante a reprodução e não depende
de outra aplicação para executar o experimento.

---

## 3. Estado final do projeto

O runtime científico foi reduzido ao seguinte núcleo:

```text
.
├── preparar_snapshot_u67.py
├── reproduzir_experimento.py
├── avaliar_elegibilidade_u67.py
├── gerar_figuras_tcc.py
├── engine/
│   ├── configuracao.py
│   ├── diagnosticos.py
│   ├── execucao.py
│   ├── modelo_lightgbm.py
│   └── rotacao.py
├── reproducao/
│   ├── artefatos.py
│   ├── dados.py
│   ├── experimento.py
│   ├── graficos.py
│   └── preparacao.py
├── dados/
├── tests/
├── requirements.txt
├── README.md
└── MANIFEST.md
```

A árvore atual foi deliberadamente simplificada. Código de pesquisas encerradas,
estratégias descartadas, layouts antigos de dados, módulos genéricos não usados
e referências históricas que não participavam da reprodução final foram
removidos.

---

## 4. Separação entre aquisição e reprodução

A arquitetura final separa dois processos.

### 4.1 Preparação do snapshot

`preparar_snapshot_u67.py` é o único fluxo que pode acessar a fonte de dados
externa.

Ele:

1. carrega as credenciais locais;
2. baixa barras diárias;
3. baixa Corporate Actions;
4. grava primeiro em `dados/.u67_build/`;
5. cria o manifesto;
6. calcula hashes de integridade;
7. valida o conjunto;
8. publica o snapshot em `dados/u67/`.

A área `dados/.u67_build/` é temporária e não deve ser versionada.

### 4.2 Reprodução científica

`reproduzir_experimento.py` não baixa dados.

Ele deve trabalhar exclusivamente com:

```text
dados/u67/
├── raw_bars/
├── corporate_actions/
└── manifest.json
```

O snapshot é validado antes do início do experimento.

Essa separação foi uma das mudanças mais importantes do projeto: a coleta de
dados deixou de ser parte implícita do backtest e passou a ser uma etapa
independente e auditável.

---

## 5. Contrato dos dados

O contrato final usa:

```text
fonte: Alpaca
feed: SIP
timeframe: 1Day
adjustment: RAW
início histórico: 2016-01-01
cutoff final: 2026-10-06
```

Corporate Actions são mantidos separadamente.

Splits são normalizados em memória. Os dados normalizados não são persistidos
como uma segunda série histórica, evitando duas fontes concorrentes de verdade.

Os CSVs são lidos com:

```text
float_precision="round_trip"
```

Essa decisão foi incorporada após uma investigação de pequenas diferenças
numéricas provocadas pela serialização e releitura de `float64`. O parser
round-trip preservou os valores de ponto flutuante necessários para reproduzir
os checkpoints monetários exatamente.

O manifesto registra identidade, datas, arquivos e hashes SHA-256. Os hashes
servem para integridade e auditoria; não são features do modelo e não participam
do treinamento.

---

## 6. Universo U67

O universo fixo de entrada contém os 67 símbolos informados pelo autor. A lista
completa é definida por `ASSETS`; `U67_REQUESTED_ASSETS` aponta para essa mesma
lista, sem divisões entre conjuntos anteriores, candidatos e referência.

A configuração é centralizada em:

```text
engine/configuracao.py
```

O pipeline aplica exclusões estruturais documentadas para:

```text
CLMT
DOC
```

Portanto:

```text
ativos solicitados: 67
ativos efetivos: 65
exclusões estruturais esperadas: 2
```

Os mesmos 65 ativos efetivos são utilizados pela rotação e por compra e
manutenção. A referência distribui inicialmente 1/65 do capital por ativo,
desconta as taxas de compra, mantém as quantidades sem rebalanceamento e
liquida todas as posições no fechamento final. Um preço ausente, não positivo
ou não finito interrompe o cálculo, em vez de reduzir silenciosamente seu
universo. Os ativos efetivamente comprados pela rotação podem ser um subconjunto
do universo estudado; isso não altera a composição da referência.

O calendário é selecionado diretamente entre os históricos do universo efetivo.
No snapshot auditado, a fonte continua sendo AAPL, com 2.506 datas preparadas e
1.560 sessões de execução. Os limites dos três folds permanecem idênticos aos
registrados na execução v1.22.0-dev.8.

A regra metodológica adotada ao longo do projeto é conservadora: quando um ativo
apresenta problema estrutural de identidade, continuidade histórica ou
consistência da série, ele é excluído explicitamente. O projeto não reconstrói
manualmente uma continuidade artificial entre identidades diferentes.

---

## 7. Modelo e parâmetros científicos finais

Modelo oficial:

```text
LightGBM Control
strategy_mode = COMPOUND_ROTATION_SWING_LIGHTGBM
backend oficial = CPU
```

Parâmetros LightGBM congelados:

```text
n_estimators        = 329
learning_rate       = 0.020731
max_depth           = 3
num_leaves          = 6
min_child_samples   = 18
min_child_weight    = 5.0
subsample           = 0.85
subsample_freq      = 0
colsample_bytree    = 0.88067
reg_alpha           = 0.050837
reg_lambda          = 3.596305
max_bin             = 255
n_jobs              = -1
repetitions         = 1
seed_step           = 1000
random_state        = 42
```

Targets multi-horizonte:

```text
horizontes = 5, 10, 20, 40, 60
pesos      = 0.10, 0.15, 0.20, 0.30, 0.25
```

Parâmetros principais da validação e rotação:

```text
minimum_training_rows             = 700
walk_forward_calibration_days     = 126
walk_forward_test_days            = 504
walk_forward_min_test_days        = 126
purge_days                        = 60
downside_penalty                  = 0.20
drawdown_penalty                  = 0.35
minimum_holding_days              = 2
minimum_expected_edge             = 0.001
cash_threshold                    = 0.0
switch_margin                     = 0.0005
switch_margin_candidates          = 0.0, 0.0025, 0.005, 0.01
```

Capital inicial:

```text
US$ 10,000.00
```

Custos configurados:

```text
slippage_bps       = 0.0
commission_rate    = 0.0
SEC fee rate       = 0.0000206
TAF fee/share      = 0.000195
TAF fee cap        = 9.79
CAT fee/share      = 0.000003
```

---

## 8. Validação temporal

O projeto evoluiu para uma validação cronológica explícita.

A reprodução final usa:

- folds walk-forward;
- janela de calibração anterior ao período de decisão;
- purge temporal de 60 dias;
- ajuste final usando somente informações permitidas pelo tempo;
- avaliação fora da amostra;
- seleção da switch margin por score de calibração;
- replay financeiro posterior à definição da política.

A data final é tratada como data de mercado inclusiva. Essa correção foi
introduzida após identificar que barras NYSE em `04:00 UTC` ou `05:00 UTC`
podiam ficar fora da janela quando a comparação era feita contra
`00:00 UTC` da mesma data.

---

## 9. Política financeira e simulação

A simulação trabalha com uma posição de capital por vez e registra rotações
entre ativos.

O replay inclui:

- compra;
- venda;
- troca de ativo;
- holding mínimo;
- margem mínima para troca;
- custos regulatórios;
- fechamento final;
- curva de capital;
- métricas de risco;
- benchmark equal-weight;
- diagnósticos de operações.

Os módulos responsáveis foram separados em:

```text
engine/execucao.py
engine/rotacao.py
engine/diagnosticos.py
```

---

## 10. Backtest Analytics

A etapa de análise foi incorporada ao fluxo oficial e gera artefatos
reproduzíveis para inspeção do comportamento da estratégia.

Entre os artefatos estão:

- retornos mensais;
- P/L realizado mensal;
- heatmaps;
- rotações de capital;
- matriz de transição entre ativos;
- dados mensais das rotações;
- planilha consolidada XLSX;
- PNG e SVG;
- CSVs de suporte.

O diretório de gráficos é recriado a cada execução para evitar mistura de
artefatos antigos e novos.

---

## 11. Artefatos finais

A reprodução grava em:

```text
output/reproducao/
```

Principais arquivos:

```text
u67_requested_assets.csv
u67_effective_assets.csv
u67_runtime_exclusions.csv
u67_fold_margins.csv
u67_fold_calibration_candidates.csv
u67_market_data_hashes.csv
u67_predictions.csv
u67_trades.csv
u67_runtime_environment.json
reproducao_u67.json
graficos/
pacote_reproducao_u67.zip
```

O ZIP é criado somente depois da validação do artefato JSON principal e do
schema de execução.

---

## 12. Checkpoints do TCC

Os checkpoints atuais são internos ao próprio projeto e servem apenas para
detectar regressões.

```text
analysis_end = 2026-10-06

capital final esperado
US$ 76,927,051.38897176

checkpoint 2026-09-17
US$ 78,782,538.31270888

tolerância monetária
US$ 0.01
```

Na reprodução validada:

```text
delta final = US$ 0.00
delta no checkpoint intermediário = US$ 0.00
reprodução exata até o centavo = true
```

Esses valores não são entrada do modelo e não são utilizados para ajustar
parâmetros durante a execução. Eles são comparados somente depois que a
simulação termina.

---

## 13. Métricas finais validadas

```text
capital final = US$ 76,927,051.38897176
CAGR          = 322.775261%
Sharpe        = 2.56904268
Max Drawdown  = -30.358970%
Worst Fold    = 282.589554%
```

Interpretação:

- o CAGR resume o crescimento composto equivalente do período;
- o Sharpe indica uma relação historicamente elevada entre retorno e
  volatilidade;
- o Max Drawdown mostra que a estratégia chegou a ficar aproximadamente 30,36%
  abaixo de um topo anterior;
- o Worst Fold mostra que o pior período walk-forward também permaneceu
  fortemente positivo.

Esses resultados são históricos. Não constituem previsão nem garantia de
retorno futuro.

---

## 14. Evolução do projeto

### 14.1 Checkpoint v1.0.6

O projeto consolidou um primeiro experimento reproduzível com:

- 56 ativos solicitados;
- 55 ativos elegíveis;
- dados Alpaca SIP/RAW;
- LightGBM;
- três folds walk-forward;
- comparação entre Control e uma variante Soft Horizon Consensus;
- ausência de banco de dados;
- snapshot por CSV e Corporate Actions;
- CI com Ruff e pytest.

Resultados então registrados:

```text
Control
capital final = US$ 10,094,316.30
CAGR          = 207.83%
Sharpe        = 2.1021
Max Drawdown  = -31.22%
Worst Fold    = +275.05%

Soft Horizon Consensus
capital final = US$ 9,851,632.93
CAGR          = 206.62%
Sharpe        = 2.1013
Max Drawdown  = -31.22%
Worst Fold    = +275.05%
```

A variante Soft alterou somente cinco decisões e não superou o Control. Esse
resultado contribuiu para a decisão posterior de remover o caminho Soft do
runtime final.

### 14.2 Isolamento do refresh de dados

Foi identificado que o fluxo de download podia apontar para o snapshot
congelado. A arquitetura foi corrigida para separar snapshot científico de
downloads temporários.

A lição dessa etapa permaneceu no desenho final: o dado congelado nunca deve
ser alterado implicitamente por uma reprodução.

### 14.3 Universo único de ativos

O projeto abandonou a separação manual entre ativos de referência e candidatos.

A configuração passou a ter uma fonte única para o universo, reduzindo regras
especiais herdadas de etapas anteriores.

### 14.4 Padronização do engine em português

Arquivos e métodos internos foram renomeados para tornar o código mais coerente
com o TCC:

```text
configuracao.py
diagnosticos.py
execucao.py
modelo_lightgbm.py
rotacao.py
```

Após a renomeação, uma falha de imports residuais mostrou que testes de nomes
não eram suficientes. Foi então adicionado teste de importação de todo o grafo
de módulos do runtime.

### 14.5 Backtest Analytics

Os gráficos e tabelas passaram a fazer parte do fluxo oficial. A geração de
artefatos foi testada automaticamente, incluindo CSV, PNG, SVG e XLSX.

### 14.6 Data final dinâmica e correção de inclusividade

Durante etapas intermediárias, execuções temporárias passaram a acompanhar
novas sessões disponíveis.

Em seguida foi corrigida a semântica da data final para garantir que a sessão
da própria data de corte fosse incluída mesmo quando a barra diária estivesse
indexada algumas horas depois de `00:00 UTC`.

Na versão final, a janela voltou a ser congelada para fins de reprodução.

### 14.7 Consolidação do universo fixo

As versões intermediárias de preparação foram substituídas pelo contrato final
U67. Elas integram o histórico de desenvolvimento. Os resultados apresentados
para responder ao objetivo científico utilizam o universo fixo final e a
comparação entre rotação e compra e manutenção nesse mesmo conjunto.

### 14.8 Investigação de precisão numérica

Uma campanha de validação independente detectou que pequenas diferenças de
`float64` podiam surgir entre dados em memória e dados serializados em CSV.

A correção adotada foi a leitura com `float_precision="round_trip"`.

Depois da correção, a reprodução chegou aos mesmos checkpoints monetários até o
centavo.

### 14.9 Refatoração final

A etapa final removeu grande volume de infraestrutura que já não participava do
experimento científico.

Foram removidos, entre outros:

- pasta de pesquisas experimentais;
- módulo Directional Change;
- testes exclusivos dessas pesquisas;
- runner antigo de busca de ativos;
- auxiliares de migração;
- caminhos Soft Horizon Consensus;
- execução genérica de variantes descartadas;
- compatibilidades de snapshots antigos;
- layouts legados de dados;
- diagnósticos históricos fora do contrato final;
- dependências residuais de arquivos já removidos;
- referências a sistemas externos.

Funções genéricas ainda úteis, como criação do pacote ZIP e sinal sonoro de
conclusão, foram preservadas e movidas para módulos adequados.

### 14.10 Snapshot U67 único

A estrutura de dados convergiu para um único contrato:

```text
dados/u67/
```

Aquisição e reprodução foram separadas definitivamente.

### 14.11 Supressão seletiva de warnings

O runner passou a ocultar apenas `Pandas4Warning`, quando a classe estiver
presente na versão instalada do pandas.

A implementação evita `warnings.filterwarnings("ignore")` global, mantendo
outros warnings visíveis.

---

## 15. Qualidade e testes

O workflow oficial usa Python 3.12.

Na execução histórica da `main` anterior à correção de elegibilidade:

```text
Ruff: sucesso
pytest: 34 passed
workflow: sucesso
```

Ambiente observado nesse CI:

```text
Python        = 3.12.15
numpy         = 2.5.3
pandas        = 3.0.6
scikit-learn  = 1.9.1
LightGBM      = 4.7.0
matplotlib    = 3.11.2
openpyxl      = 3.1.5
alpaca-py     = 0.44.0
pytest        = 8.4.2
ruff          = 0.16.10
```

O CI executa:

```bash
python -m ruff check engine reproducao preparar_snapshot_u67.py reproduzir_experimento.py avaliar_elegibilidade_u67.py gerar_figuras_tcc.py tests --select F401,F811,F821,F841
python -m pytest -q
```

Entre as proteções adicionadas ao longo do projeto estão:

- importação de todos os módulos do runtime;
- contrato dos parâmetros LightGBM;
- universo e exclusões;
- semântica temporal inclusiva;
- precisão round-trip dos CSVs;
- artefatos de Backtest Analytics;
- ausência de dependência de banco de dados;
- remoção de modos e nomes aposentados;
- isolamento do snapshot;
- prevenção de referências históricas removidas;
- supressão seletiva de warnings.

A validação da release v1.22.1 está registrada na seção 24; o ambiente e os
34 testes acima pertencem à execução histórica, não à suíte atual.

---

## 16. Avaliação crítica do estado final

### 16.1 Pontos fortes

**Reprodutibilidade computacional.** O projeto possui checkpoints monetários
internos e a execução validada os reproduziu até o centavo.

**Separação de responsabilidades.** Aquisição, preparação, modelagem, simulação,
analytics e empacotamento estão separados em módulos distintos.

**Validação temporal.** Walk-forward, calibração cronológica e purge reduzem o
risco de vazamento temporal em comparação com uma divisão aleatória comum.

**Política explícita para problemas estruturais.** Ativos com ruptura de
identidade são excluídos de forma documentada, em vez de receber correções
manuais difíceis de reproduzir.

**Redução de complexidade.** O runtime final contém apenas o caminho Control que
é realmente utilizado.

**Auditoria.** Predições, operações, margens por fold, hashes, ambiente e
gráficos são exportados.

**CI.** Ruff e pytest são executados automaticamente sobre a árvore científica.

### 16.2 Limitações que devem aparecer no TCC

**Resultados excepcionalmente altos.** CAGR acima de 300% e crescimento de
US$ 10 mil para dezenas de milhões exigem interpretação cautelosa. O resultado
é histórico e não deve ser apresentado como expectativa de desempenho futuro.

**Seleção retrospectiva da amostra.** O universo é fixo para este estudo. Sua
definição não foi validada prospectivamente antes da janela de avaliação. A
generalização para outros ativos e períodos exige investigação adicional.

**Custos potencialmente otimistas.** A configuração final usa
`slippage_bps=0` e `commission_rate=0`. Existem taxas regulatórias, mas a
ausência de slippage explícito pode superestimar a execução real, especialmente
em períodos de maior volatilidade ou menor liquidez.

**Dependências não totalmente travadas.** `requirements.txt` usa intervalos de
versão, não um lockfile de versões exatas. O CI registra um ambiente validado,
mas uma instalação futura pode resolver versões menores diferentes.

**Determinismo entre plataformas.** O checkpoint foi reproduzido exatamente no
ambiente validado, mas a configuração ainda permite paralelismo do LightGBM
(`n_jobs=-1`) e `deterministic_execution=False`. Isso significa que
identidade bit a bit entre todos os sistemas operacionais e bibliotecas não é
formalmente garantida.

**Dados incluídos na release.** A `main` de origem não continha os CSVs.
Esta revisão inclui os 134 CSVs e o manifesto original, preservando seus bytes
e hashes. `.gitattributes` impede a conversão automática das quebras de linha
do snapshot entre sistemas operacionais. A inclusão resolve a ausência do
snapshot na release; não elimina as demais limitações metodológicas.

**Elegibilidade corrigida.** O código anterior consultava a abertura e o
fechamento seguintes para decidir se o ativo poderia receber uma previsão.
Essa dependência foi removida. No snapshot congelado, o filtro antigo não
excluía candidatos nas datas usadas, e a correção não alterou a curva. Esse
resultado delimita o efeito observado neste experimento, sem validar outras
possíveis fontes de informação retrospectiva.

### 16.3 Avaliação geral

A arquitetura final é substancialmente mais clara e reproduzível do que as
etapas intermediárias. O projeto passou de um ambiente com múltiplas linhas de
pesquisa, modos e layouts de dados para um único protocolo científico.

A principal evidência técnica de qualidade é a combinação de:

```text
snapshot congelado
+ validação temporal
+ pipeline único
+ checkpoints internos
+ reprodução exata até o centavo
+ CI automatizado
```

O snapshot está incluído na release. As próximas verificações científicas
devem se concentrar na execução econômica e na interpretação dos resultados,
considerando a seleção retrospectiva da amostra e os custos assumidos.

---

## 17. O que é científico e o que é infraestrutura

### Científico

- universo U67;
- exclusões estruturais;
- features;
- LightGBM e seus parâmetros;
- targets e pesos;
- folds walk-forward;
- purge;
- calibração;
- política de rotação;
- custos;
- benchmark;
- checkpoints;
- métricas.

### Infraestrutura

- organização de diretórios;
- nomes internos dos módulos;
- geração de ZIP;
- sinal sonoro;
- logging;
- gráficos;
- planilha;
- filtros de warning;
- CI;
- documentação.

Refatorações de infraestrutura não devem alterar os resultados científicos sem
uma decisão metodológica explícita.

---

## 18. Regras de manutenção

1. Manter um único caminho oficial de reprodução.
2. Não introduzir dependência de banco de dados no runner científico.
3. Não baixar dados durante `reproduzir_experimento.py`.
4. Alterar o snapshot somente por fluxo explícito e auditável.
5. Preservar exclusões estruturais documentadas.
6. Não reconstruir manualmente séries com ruptura de identidade.
7. Toda alteração metodológica deve produzir nova versão científica.
8. Refatorações sem efeito científico devem manter os checkpoints.
9. Ruff e pytest devem passar antes de merge.
10. Qualquer mudança legítima nos checkpoints deve ser explicada no manifesto.
11. O README deve permanecer operacional e curto; documentação técnica,
    histórica e avaliativa pertence a este MANIFEST.

---

## 19. Estado de encerramento desta consolidação

O estado científico descrito por este documento é:

```text
versão científica        = 1.22.1
universo solicitado      = 67
universo efetivo         = 65
modelo                   = LightGBM Control
cutoff                   = 2026-10-06
capital inicial          = US$ 10,000.00
capital final checkpoint = US$ 76,927,051.38897176
CAGR                     = 322.775261%
Sharpe                   = 2.56904268
Max Drawdown             = -30.358970%
Worst Fold               = 282.589554%
compra e manutenção      = mesmos 65 ativos efetivos
capital final referência = US$ 33,295.28176973073
Ruff local               = aprovado
pytest local             = ver seção 24
```

A versão dev.9 corrigiu a referência financeira. A versão dev.10 corrige a
elegibilidade e registra diagnósticos fora da amostra. A release v1.22.1
consolida esse protocolo na main, sem modificar parâmetros ou resultados.

---

## 20. Comparação no mesmo universo fixo

A revisão de 9 de outubro de 2026 removeu do fluxo oficial a construção de um
universo separado para compra e manutenção e a injeção de um benchmark externo
no replay. O motor calcula a referência com os mesmos `frames`, `symbols` e
datas da estratégia. O JSON registra `benchmark_assets`,
`benchmark_asset_count` e `benchmark_same_universe`. O CSV
`u67_buy_hold_assets.csv` registra os ativos e seus pesos iniciais.

### 20.1 Verificação executada

Utilizaram-se `u67.zip` e `pacote_reproducao_u67.zip` fornecidos pelo autor.
Conferiram-se a identidade do snapshot e seus 134 hashes. Prepararam-se os 65
ativos com o parser round-trip e recalculou-se compra e manutenção com as
funções oficiais de taxas e deslizamento.

As decisões diárias registradas em `u67_predictions.csv` foram reaplicadas no
motor financeiro. O replay manteve as 674 operações, os limites dos três folds,
as 1.560 sessões e toda a curva de capital da estratégia, com diferença máxima
absoluta de US$ 0,00. Os dois checkpoints foram preservados até o centavo.
Esta validação não repetiu o treinamento dos modelos; verifica o calendário,
a referência e a contabilidade das decisões já registradas.

### 20.2 Referência recalculada

```text
ativos de compra e manutenção = mesmos 65 ativos efetivos
peso inicial por ativo       = 1/65, antes das taxas de compra
capital inicial              = US$ 10,000.00
capital final                = US$ 33,295.28176973073
retorno acumulado            = 232.952818%
CAGR                         = 21.384775%
Sharpe                       = 1.04791659
drawdown máximo              = -27.494624%
retorno no fold 1             = 45.511891%
retorno no fold 2             = 44.277723%
retorno no fold 3             = 58.593334%
```

Esses valores foram produzidos pelo recálculo da referência. O pacote histórico
v1.22.0-dev.8 permanece como evidência das decisões da rotação; sua antiga curva
de compra e manutenção foi substituída na comparação científica desta revisão.

### 20.3 Validação do código

Ruff passou nas regras F401, F811, F821 e F841. A suíte local passou com 44
casos, incluindo alocação igualitária com taxas, liquidação final e interrupção
quando um ativo tem preços ausentes ou inválidos. Um teste de replay também
verificou que a referência inclui todo o universo declarado, mesmo quando a
rotação compra apenas um dos ativos.

As limitações de execução e elegibilidade futura já identificadas não foram
alteradas por esta correção. O resultado histórico não demonstra desempenho
prospectivo nem isola a contribuição individual dos componentes da utilidade.

---

## 21. Elegibilidade com a informação da decisão

### 21.1 Correção e escopo

Em 10 de outubro de 2026, criou-se `fix/u67-eligibilidade-sem-futuro` a partir
da `main` `262e5388354f258de2480151a0b9f1b9ff147db7`. A revisão de compra e
manutenção da seção 20, commit `192dc7bd457beed66a62a13c368b20865da748c9`,
foi incorporada porque ainda não fazia parte da `main`. O protocolo permanece
restrito ao universo fixo fornecido pelo autor e à referência de compra e
manutenção nesse mesmo conjunto. Não houve busca de ativos ou ajuste de
hiperparâmetros nesta etapa.

Os caminhos individual e em lote consultavam a linha seguinte e exigiam
abertura e fechamento positivos e finitos antes de admitir o ativo na decisão
atual. A versão dev.10 utiliza apenas a presença do modelo, da sessão atual e
de seus 52 atributos finitos. Uma previsão pode ser produzida com os dados
disponíveis até a sessão de decisão, mesmo sem a linha seguinte.

Os preços posteriores são lidos para executar a ação já definida e calcular
o resultado realizado. Se o preço necessário estiver ausente ou inválido, a
simulação é interrompida com identificação do ativo, campo e sessão. O fluxo
não escolhe outro ativo a partir dessa informação posterior.

### 21.2 Desenho da medição

`avaliar_elegibilidade_u67.py` ajusta uma vez os modelos de calibração e os
modelos finais de cada um dos três folds. As previsões desses mesmos ajustes
alimentam dois controles: o filtro histórico e a elegibilidade corrigida.
Cada controle calibra sua margem com as mesmas alternativas e dados. Assim,
uma diferença entre eles corresponde à condição de elegibilidade e às suas
consequências sobre a política, sem confundir ajustes distintos do modelo.

A máscara histórica existe apenas no script de auditoria. O caminho oficial
`reproduzir_experimento.py` utiliza a regra corrigida. As datas finais de cada
cache, que não geram execução posterior dentro do experimento, são excluídas
da contagem de decisões e dos erros preditivos.

Preservaram-se o cutoff de 6 de outubro de 2026, o parser round-trip, as
exclusões estruturais DOC/CLMT, os três folds, os parâmetros, os custos e os
US$ 10 mil iniciais. Conferiram-se os 134 hashes e a identidade do snapshot:

```text
e440f59da5e684f1de59cf447abfedd9aed3f7817b3d5d681058fe631276a575
```

Uma verificação adicional confrontou a máscara com a função em lote do commit
dev.9, usando previsões constantes para isolar a regra de disponibilidade.
Houve concordância nas datas de calibração e teste dos três folds. Todos os
atributos atuais nessas datas eram finitos. Essa verificação técnica é
distinta do treinamento usado para produzir os resultados financeiros.

### 21.3 Efeito observado no snapshot

| Medida | Resultado |
| --- | ---: |
| Capital final com o filtro histórico | US$ 76.927.051,38897176 |
| Capital final com a elegibilidade corrigida | US$ 76.927.051,38897176 |
| Diferença no capital final entre os controles | US$ 0,00 |
| Diferença máxima absoluta na curva | US$ 0,00 |
| Sessões com decisão alterada | 0 |
| Operações em cada controle | 674 |
| Sessões de execução | 1.560 |
| Capital final de compra e manutenção | US$ 33.295,28176973073 |

O filtro histórico não excluiu candidatos em nenhuma das 24.375 combinações
de ativo e decisão de calibração, nem nas 101.400 combinações de teste. As
margens escolhidas permaneceram 0,0005; 0,0100; 0,0005, respectivamente.
Os dois controles preservaram todas as operações e a curva completa. A curva
da rotação também coincidiu com o registro histórico dev.8, com diferença
máxima de US$ 0,00.

Portanto, a retirada da dependência futura não alterou o resultado financeiro
observado neste snapshot. A correção elimina a dependência lógica demonstrada
pelos testes, mas este resultado não garante que ela seria irrelevante em
outros históricos ou na execução futura. Também não elimina a seleção
retrospectiva da amostra, não mede a contribuição causal de cada componente
da utilidade e não mede impacto de mercado ou capacidade de execução.

### 21.4 Diagnósticos preditivos

Foram avaliadas as previsões dos modelos finais nas sessões de decisão de
teste de cada fold, contra o alvo `forward_risk_adjusted_utility`. Os erros
agregam pares de ativo e sessão com previsão e rótulo finitos; rótulos sem
horizonte futuro completo são excluídos, sem preenchimento por zero.

| Fold | Pares válidos | MAE | RMSE |
| --- | ---: | ---: | ---: |
| 1 | 32.760 | 0,2171303552 | 0,2851669999 |
| 2 | 32.760 | 0,1835953272 | 0,2418070904 |
| 3 | 32.045 | 0,1856207765 | 0,2433666188 |

MAE e RMSE estão na escala da função de utilidade; não representam erros de
preço, percentuais de retorno ou taxas de acerto. O terceiro fold tem menos
rótulos válidos porque o snapshot termina antes de se completar o horizonte
de 60 sessões para as últimas decisões. As importâncias por ganho dos 52
atributos de cada modelo final são exportadas em
`final_fit_feature_importance_gain`. Pertencem ao ajuste de treino e não
medem importância fora da amostra ou contribuição causal.

Os diagnósticos ficam no JSON da reprodução oficial e no relatório pareado.
O campo vazio do pacote histórico dev.8 não foi reinterpretado como se já
contivesse esses resultados.

### 21.5 Evidências e validação

A branch inclui o snapshot original: 134 CSVs e `manifest.json`, totalizando
21.789.973 bytes. O manifesto do snapshot conserva a identificação de sua
geração histórica; a versão do código passou para dev.10. Os arquivos de
dados têm conversão de quebras de linha desativada no Git, para conservar os
hashes em clones feitos em diferentes sistemas.

A análise estática passou nas regras F401, F811, F821 e F841. A suíte local
passou com 80 casos. Os testes verificam que alterar apenas a abertura ou o
fechamento futuros não muda os escores ou o ativo escolhido, que uma previsão
funciona sem a linha seguinte, que preços inválidos são tratados na execução
e que os diagnósticos de teste excluem rótulos incompletos. Um teste também
valida a identidade e todos os hashes do snapshot incluído.

O fluxo oficial também foi executado por completo, incluindo novo treinamento,
simulação, gráficos, planilha e pacote de reprodução. Seus dois checkpoints
foram reproduzidos exatamente: US$ 78.782.538,31270888 em 17 de setembro de
2026 e US$ 76.927.051,38897176 em 6 de outubro de 2026. As 1.560 sessões,
as escolhas de ativos, a curva completa e as 674 operações coincidiram com
o registro dev.8. Os diagnósticos de teste foram efetivamente exportados,
com 65 modelos por fold e importâncias dos atributos em todos eles.

`evidencias/eligibilidade_v1.22.0-dev.10.json` conserva a configuração,
o ambiente, as métricas dos controles, os diagnósticos, os checkpoints e as
verificações. Os CSVs detalhados e o pacote oficial são gerados pelos comandos
abaixo. O JSON de evidência não altera o pacote histórico original.

O ambiente da medição usa Python 3.12.14, NumPy 2.3.5, pandas 2.2.3,
scikit-learn 1.8.0, LightGBM 4.7.0 e threadpoolctl 3.6.0, em Linux. A
configuração manteve `deterministic_execution=False` e `n_jobs=-1`. A
concordância observada com o registro anterior não é uma garantia universal
de identidade numérica entre ambientes.

Para repetir a verificação, execute na raiz do projeto:

```bash
python avaliar_elegibilidade_u67.py
python reproduzir_experimento.py
```

O primeiro comando grava os dois controles em `output/elegibilidade/`. O
segundo executa o fluxo oficial corrigido e grava os artefatos em
`output/reproducao/`. Nenhum dos comandos baixa dados.

### 21.6 Próxima etapa do TCC

O texto deve substituir a pendência sobre o impacto da retirada do filtro
pela medição documentada nesta seção. Também pode apresentar os diagnósticos
de teste com sua escala e número de observações, distinguindo-os das
importâncias calculadas no ajuste. A revisão deve preservar as limitações
econômicas e a interpretação da política completa frente a compra e
manutenção, sem atribuir causalidade individual aos seus componentes.

---

## 22. Procedência dos hiperparâmetros (conferência documental de 2026-10-10)

O arquivo original e57d19fd-37d6-4be0-b54a-08e81f9b2892.zip, recuperado nos
anexos históricos, contém o manifesto e os CSVs da campanha
20260813T170641-tune-27cdfc7d. Ela foi concluída em 13 de agosto de 2026:
20 configurações além do controle, todas concluídas, semente 42, seis
propostas iniciais por hipercubo latino e 14 propostas adaptativas.
O modelo probabilístico registrado foi
gaussian_process_adaptive_trust_region_cei_v2.

Os oito domínios foram: 220–380 árvores; taxa de aprendizado 0,020–0,050;
profundidade 2–4; 4–12 folhas, limitadas pela profundidade; 15–30 observações
mínimas por folha; proporção de atributos 0,75–0,95; regularização L1
0–0,50 e L2 1–4. Os outros parâmetros do modelo foram conservados do controle.

Processos gaussianos estimaram capital final, Sharpe, drawdown máximo e
retorno do pior fold. A aquisição combinou melhoria esperada de capital
sujeita às condições de risco, exploração da incerteza e probabilidade de
satisfazer as condições. O registro do candidato selecionado identifica
2.048 propostas na aquisição, 512 cenários de Monte Carlo e peso de
exploração 0,15. A atualização do incumbente exigia ao menos 3% de melhoria
no capital, tolerância absoluta 0,05 no Sharpe, 0,03 no drawdown e retorno
positivo no pior fold. O ranking final ordenava as configurações com
todos os folds positivos pela pontuação acumulada de retornos
logarítmicos penalizados por perdas e aumentos de drawdown, com desempate
por capital final e Sharpe.

O candidato 18 foi o primeiro no ranking e satisfez as condições de
atualização. Seu vetor completo de 16 opções coincide exatamente com
engine/configuracao.py. Esta conferência não executou novo treinamento
nem modificou resultados do experimento dev.10.

A configuração de origem, recuperada em
extrema_backtest.strategy_profiles.json, tem o mesmo hash
fac52b7e8ba25a916b5570a69de29a2fc970ffb1166868aa8de26330dba9dc67
registrado na campanha. Ela documenta a composição exploratória então
configurada, recorte iniciado em 2016-01-01, barras diárias, Alpaca/SIP
e ajuste all. A campanha fixou o corte em 2026-08-11 e registrou a
assinatura de dados
d4fc5bf5460e01d4a76208f5931c21abefa98657ad9785fd8a67fb408a828e0c
em todas as 21 configurações. A identificação dessa composição serve
à procedência dos parâmetros. O experimento do TCC conserva sua lista
fixa e compra e manutenção como única referência financeira.

As métricas usadas na seleção cobriram 2020-07-22 a 2026-08-11, com três
folds cronológicos. O registro conserva retornos por fold, mas não todas
as datas de fronteira do treinamento e da calibração. A base original
congelada também não foi recuperada: os outros backtests examinados
possuem assinatura diferente e não devem preencher essa lacuna.
É necessário recuperar o snapshot original e os limites exatos das
janelas para repetir a campanha integralmente. O commit da execução
original pode ser conferido no log do job, se disponível.

A seleção foi exploratória e consultou resultados em período que se
sobrepõe à avaliação atual. A manutenção posterior dos hiperparâmetros
e a validação cronológica do ajuste não demonstram independência da
avaliação em relação a essa seleção. A calibração da margem no
experimento dev.10 continua sendo uma busca em grade por fold, distinta
da otimização bayesiana dos hiperparâmetros.

evidencias/selecao_hiperparametros_20260813.json registra as fontes e seus
hashes, o contrato de dados, os domínios, a configuração selecionada, a
conferência com o código atual e as pendências específicas. A referência
ao código do período é uma verificação de implementação, não prova do
commit exato da campanha. A versão científica permanece dev.10; esta
alteração é documental e permanece na branch de trabalho.

---

## 23. Técnicas de Data Science e figuras do TCC (2026-10-10)

A revisão do texto detalhou a unidade ativo-sessão, a engenharia dos 52
atributos, a construção do alvo contínuo, a regressão por árvores com
boosting de gradiente, a perda quadrática e a função dos controles de
complexidade. Também explicou o uso do hipercubo latino e dos processos
gaussianos na campanha histórica, a calibração da margem por grade, o
reajuste antes do teste e a separação cronológica dos rótulos.

gerar_figuras_tcc.py produz sete figuras: fluxo dos dados, janelas
temporais, capital em escala logarítmica, drawdown, retornos por ano,
MAE/RMSE no teste e participação dos oito grupos de atributos no ganho
dos ajustes finais. As duas curvas existentes foram reapresentadas e
cinco figuras foram acrescentadas. O Word preserva dez tabelas e inclui
cinco gráficos nativos editáveis com suas planilhas incorporadas.

evidencias/series_financeiras_v1.22.0-dev.10.csv conserva as 1.560 datas,
as duas curvas financeiras e a identificação do fold, extraídas de
output/reproducao/u67_predictions.csv. Seu SHA-256 é
f9e873cc33e91e006817ac31781261d4ef440745d8b7cf82208026186a29b9b9.
Os erros e os ganhos provêm de evidencias/eligibilidade_v1.22.0-dev.10.json.
figuras/procedencia_figuras.json identifica essas fontes, a execução
científica 3f8d1b1a2784721201938071471e91fd4282f628 e os hashes das imagens.

Os retornos por ano usam o capital da última sessão de cada intervalo
dividido pelo capital do encerramento anterior, menos um; a primeira
base é US$ 10.000. Os anos 2020 e 2026 são parciais e não foram
anualizados. A composição dos fatores anuais reproduz o capital final.
O ganho de cada grupo é a soma dos ganhos normalizados de seus atributos
em cada modelo, seguida da média simples entre os 65 modelos finais do
fold. A soma dos oito grupos é um. Trata-se de ganho no treinamento,
sem atribuição causal ao retorno ou medição de importância no teste.
Os grupos têm quantidades diferentes de atributos.

Verificaram-se a igualdade exata das colunas exportadas com o registro
financeiro, a contagem das sessões, a correspondência dos capitais
finais, os três folds e as somas dos ganhos. Duas gerações sucessivas
preservaram os bytes dos 18 arquivos derivados. A análise estática de
gerar_figuras_tcc.py passou. As 29 páginas do Word foram conferidas após
renderização, inclusive equações, legendas e tabelas.

Esta revisão não executou novo treinamento ou backtest, não modificou
o motor financeiro e não constitui uma nova rodada dos 80 testes
registrados na execução corrigida. A versão científica permanece
v1.22.0-dev.10, o universo da pesquisa continua fixo e compra e
manutenção permanece a única referência financeira. A pendência sobre
o snapshot e as fronteiras da campanha de hiperparâmetros foi preservada.
As alterações continuam em fix/u67-eligibilidade-sem-futuro; a main
permanece no commit 262e5388354f258de2480151a0b9f1b9ff147db7.

---

## 24. Release v1.22.1 e memória de continuidade (2026-10-10)

### 24.1 Entrega e autorização

Em 2026-10-10, o autor solicitou explicitamente integrar a branch
`fix/u67-eligibilidade-sem-futuro` à `main`, criar uma tag e atualizar os
arquivos Markdown de contexto. Essa autorização substitui, para esta entrega,
a orientação anterior de preservar a `main` enquanto a revisão estava em
andamento. A tag escolhida foi `v1.22.1`, pois `v1.22.0` já existia.

A referência imutável da entrega é
[v1.22.1](https://github.com/betovlima/tcc_mba_usp_data_science_analytics/tree/v1.22.1).
O merge conserva os commits científicos e documentais da branch. A tag
anotada identifica o commit de integração na `main`; o workflow só a cria
após Ruff e pytest aprovarem esse commit. O job de publicação está restrito
ao push na `main` com o título desta release e recusa substituir uma tag
existente que aponte para outro commit.

Este MANIFEST continua sendo a memória técnica e metodológica única.
`README.md` conserva as instruções de execução e aponta para esta seção.
Não criar arquivos de contexto paralelos que possam divergir destas decisões.

### 24.2 Conteúdo consolidado e proveniência

O núcleo científico foi validado no commit
`3f8d1b1a2784721201938071471e91fd4282f628`, versão dev.10. A procedência dos
hiperparâmetros foi documentada em `04e28b9`, e a revisão de Data Science e
das figuras, em `e2b1699`. A versão 1.22.1 consolida essas alterações:

- elegibilidade definida com informações disponíveis na sessão da decisão;
- compra e manutenção recalculada no mesmo universo fixo da rotação;
- snapshot original incluído, com identidade e 134 hashes conferidos;
- MAE e RMSE fora da amostra por fold e ganhos dos atributos no ajuste;
- registro documental da seleção histórica dos hiperparâmetros;
- sete figuras reproduzíveis, tabelas derivadas e hashes de procedência.

Os hiperparâmetros, a função de utilidade, os atributos, as regras financeiras
e os dados não foram modificados nesta preparação de release. A atualização
dos identificadores em `engine/configuracao.py` e `reproduzir_experimento.py`
passa a identificar novas execuções como 1.22.1. Os arquivos de evidência
dev.10 e o manifesto original dos dados mantêm os nomes e versões da execução
que efetivamente os produziu; não devem ser renomeados como se fossem novos
experimentos.

Os resultados preservados são US$ 76.927.051,38897176 na rotação,
US$ 33.295,28176973073 em compra e manutenção, 674 operações e 1.560 sessões.
A correção de elegibilidade produziu diferença máxima de US$ 0,00 na curva
deste snapshot. Isso mede o efeito da correção neste registro, sem atribuir
contribuição causal isolada aos componentes da política.

### 24.3 Critério da sugestão inicial de ativos

A consulta ao histórico em 2026-10-10 recuperou mensagens do assistente de
2026-07-26 às 04:23:40 e 04:38:54 UTC. O registro mais antigo encontrado
apresentou AAPL como alvo e sugeriu SPY, QQQ, MSFT, AMZN, GOOGL, META e NVDA
como ativos considerados líquidos para ampliar a base de aprendizado e
fornecer contexto. Uma mensagem às 04:44:38 UTC explicou a ampliação da
amostra e o uso de séries relacionadas no aprendizado daquela etapa.

Esse é um critério qualitativo recuperado de uma recomendação exploratória.
Não foi encontrado nesses registros um ranking de volume, corte mínimo de
liquidez, base quantitativa de seleção ou justificativa individual completa.
Portanto, não afirmar que a lista foi formada pelos ativos mais negociados
do mercado nem que houve um filtro estatístico de liquidez documentado.

O histórico também registra a lista de dez ativos confirmada pelo autor em
2026-07-30: NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA, AMD, JPM e SPY.
Esse registro confirma a lista usada naquela etapa; não documenta, sozinho,
os critérios originais de cada inclusão. A primeira recomendação recuperada,
a lista operacional posterior e a lista fixa atual pertencem a etapas
distintas e não devem ser tratadas como uma única seleção datada.

O autor informou que, depois das sugestões iniciais, foram realizados testes
de força bruta para buscar inclusões. Os registros posteriores recuperados
tratam de contribuição dos candidatos no backtest. Essa história explica a
formação exploratória da amostra; não constitui uma comparação científica
entre universos nem uma seleção prospectiva comprovada. A pesquisa final
usa a lista fixa informada pelo autor e compara somente rotação e compra e
manutenção nesse mesmo conjunto, observadas as exclusões estruturais já
documentadas. No texto acadêmico, explicar o procedimento sem citar o nome
da aplicação usada na exploração.

### 24.4 Validação desta entrega

Em Python 3.12.14, a análise estática passou nas regras F401, F811, F821 e
F841, incluindo `gerar_figuras_tcc.py`. A suíte existente passou com
**80 testes**, incluindo a identidade do snapshot. Houve um aviso de
depreciação de `websockets.legacy`, sem falha dos testes.

`gerar_figuras_tcc.py` foi executado novamente: as verificações de capital,
composição dos retornos e ganhos normalizados passaram, e os 18 arquivos
derivados conservaram exatamente os mesmos bytes. A estrutura do YAML do CI,
a condição de publicação da tag e a sintaxe Bash desse job foram conferidas.
O workflow passou a executar também em pull requests para a `main`.

Nesta entrega não se repetiu o treinamento completo. Sua execução e os
checkpoints permanecem documentados na seção 21. Uma aprovação de CI não deve
ser descrita como novo backtest ou nova validação financeira integral.

### 24.5 Decisões que devem orientar a próxima sessão

Manter a pesquisa no universo fixo informado pelo autor, com compra e
manutenção como única referência financeira. No resumo, descrever a
formulação matemática, deixando a biblioteca de implementação para a
metodologia. Não inventar resultados, referências ou critérios de seleção;
priorizar autores brasileiros quando houver fonte adequada e verificável.

A reprodução ajusta os modelos com hiperparâmetros fixos, calibra a margem
por grade em cada fold, simula as duas políticas e exporta os diagnósticos.
Não repete a campanha bayesiana histórica. As sete figuras desta revisão são
geradas separadamente por `gerar_figuras_tcc.py`, a partir das evidências
preservadas, sem novo treinamento.

A campanha histórica tem parâmetros, domínios, critérios e períodos de
avaliação recuperados. Continuam ausentes o snapshot original da campanha e
as fronteiras completas de treinamento e calibração. O hash recuperado não
autoriza substituir essa base por outro backtest. A sobreposição entre
seleção exploratória e período avaliado deve continuar explícita.

A próxima etapa científica é medir a sensibilidade a custos e condições de
execução, incluindo liquidez e participação no volume, com desenho e dados
documentados antes de apresentar resultados. A contribuição individual dos
componentes da utilidade também permanece sem experimentos de isolamento.
Não transformar essas pendências em resultados já disponíveis.


---

## 25. Pesquisa pós-release de universo dinâmico U67+n (2026-10-12)

### 25.1 Escopo e separação da entrega acadêmica oficial

Esta seção registra uma linha exploratória posterior à release `v1.22.1`.
Ela não reescreve os resultados oficiais preservados nas seções anteriores e
não altera a `main`. A pesquisa permanece na branch
a branch de pesquisa atual.

O U67 congelado continua sendo a bancada experimental. A pergunta desta linha
não é encontrar uma lista fixa definitiva, mas estimar quando um ativo agrega
valor como opção disponível no universo. O objetivo operacional futuro é uma
política de estado dinâmico:

```text
ADICIONAR
MANTER
REMOVER
```

usando somente informação disponível no momento da decisão. Portanto, um ativo
pode ser útil em uma janela e deixar de ser útil em outra. A lista não deve ser
interpretada como permanente.

### 25.2 Campanha direta e otimização do subconjunto

A v1.1.0 removeu como gates os filtros de preço, liquidez, volume e o
Learning-to-Rank. Dos 500 símbolos congelados, somente condições estruturais de
dados puderam impedir o replay financeiro. A comparação econômica passou a ser
direta:

```text
U67 + candidato versus U67
```

A campanha produziu 15 candidatos com delta marginal positivo. A v1.2.0
reproduziu `U67 + 15` em US$ 290.727.954,41339606 e aplicou forward selection
e backward elimination. O Challenger resultante contém 10 ativos:

```text
CRT, NBR, CAPR, PNBK, PENN, JBL, STAG, MCHB, LAD, TOWN
```

e reproduziu capital final de US$ 587.165.941,1330212. A busca avaliou 180
subconjuntos únicos. Esse valor é um resultado de descoberta/otimização, não
uma estimativa prospectiva independente.

### 25.3 Assurance v1.3.0

A execução `u67-asset-expansion-v1.3.0` reproduziu exatamente os três
checkpoints:

```text
U67                         = US$ 76.927.051,38897176
U67 + 15 positivos          = US$ 290.727.954,41339606
U67 + 10 Challenger         = US$ 587.165.941,1330212
```

A reexecução do Challenger preservou o capital até o centavo, a sequência de
`selected_asset` e a curva de capital, com diferença máxima absoluta igual a
zero.

Os diagnósticos preditivos registraram 45 combinações candidato-fold. Houve 38
warnings de overfitting, nenhum warning de underfitting e apenas 7 combinações
com skill OOS positivo contra o preditor ingênuo da média. No próprio U67,
171 das 195 combinações ativo-fold também receberam warning de overfitting e
24 tiveram skill OOS positivo. Portanto, o alerta não é exclusivo dos novos
ativos e deve ser investigado na dimensão de ranking/decisão, não apenas por
MAE/RMSE de magnitude absoluta.

A validação temporal do subconjunto, ainda usando o pool fixo de 15 candidatos,
produziu:

```text
seleção com fold 1
    NBR, PENN, EML
holdout fold 2
    retorno 77,917619 contra 77,898220 do U67

seleção com folds 1+2
    CRT, NBR, PNBK, PENN, JBL, STAG, MCHB, TOWN
holdout fold 3
    retorno 35,536979 contra 24,484656 do U67
```

A similaridade de Jaccard com os 10 do Challenger foi 0,181818 na primeira
etapa e 0,80 na segunda. Isso é compatível com a hipótese de universo variável
no tempo, mas ainda não constitui redescoberta nested dos 500 ativos.

O leave-one-asset-out dos 10 não encontrou remoção individual que aumentasse o
capital. As perdas relativas ao retirar cada ativo foram aproximadamente:
CRT -42,90%, NBR -36,10%, CAPR -27,26%, PNBK -20,73%, JBL -17,51%, PENN
-16,69%, MCHB -4,13%, LAD -3,28%, TOWN -2,84% e STAG -1,25%.

O stress de custos manteve o resultado acima do U67, mas mostrou sensibilidade
material a slippage:

```text
base                 US$ 587,17 mi
slippage 5 bps       US$ 411,67 mi
slippage 10 bps      US$ 288,63 mi
taxas regulatórias 2x: -0,90% versus base
taxas regulatórias 3x: -1,80% versus base
```

### 25.4 Interpretação metodológica após a assurance

O contraste entre MAE/RMSE OOS fracos e forte desempenho econômico sugere que
a qualidade relevante pode estar na ordenação relativa entre ativos, não na
previsão precisa da magnitude de `forward_risk_adjusted_utility`.

Por isso, a próxima validação não deve procurar mais capital. Ela deve medir:

- Information Coefficient/Spearman cross-sectional por sessão;
- NDCG@5 e NDCG@10;
- acerto do top-1 e presença do top-1 previsto no top-3/top-5 realizado;
- overlap de top-3;
- percentil realizado do ativo previsto e do ativo efetivamente selecionado;
- regret da escolha prevista frente ao melhor ativo realizado;
- sign accuracy como diagnóstico secundário.

A avaliação usa a utility realizada posterior somente como rótulo de avaliação.
Ela não pode entrar na decisão histórica.

### 25.5 Selection overfitting e v1.4.0

A v1.4.0 introduz `RUN_MODE = "decision_validation"` e mantém os checkpoints
anteriores congelados. Ela refaz de forma determinística a paisagem de 180
subconjuntos da v1.2 para gerar séries diárias comparáveis.

Serão calculados:

```text
Deflated Sharpe Ratio
    - 180 trials brutos da busca de subconjuntos
    - número efetivo de trials via dimensão efetiva da matriz de correlação
    - sensibilidade conservadora usando 146 testes marginais + 180 subconjuntos

Probability of Backtest Overfitting / CSCV
    - 10 blocos
    - 252 divisões simétricas
    - métrica log-growth, alinhada ao objetivo de capital
    - métrica Sharpe, como diagnóstico complementar
```

O cálculo com 326 trials é explicitamente uma análise de sensibilidade:
os testes são correlacionados e pertencem a famílias diferentes, portanto não
devem ser tratados como 326 experimentos independentes.

A v1.4.0 também repete a seleção temporal em fronteiras de fold e grava a
transição entre universos como `ADICIONAR`, `MANTER` ou `REMOVER`. Isso é
um piloto em baixa frequência. Ainda não é a política diária final.

### 25.6 Próxima fronteira científica

Após a v1.4.0, o passo seguinte é uma redescoberta nested do universo externo:
em cada data de atualização, candidatos e ativos internos devem ser avaliados
somente com histórico disponível até aquela data. A política deverá decidir
adição, permanência e remoção, congelar o universo resultante e medir o efeito
somente no período posterior.

A regra de universo dinâmico deverá incluir controles contra churn, como
histerese, evidência mínima e/ou permanência mínima. Esses controles ainda não
foram calibrados e não devem ser descritos como resultados disponíveis.

Esta seção é a memória técnica da pesquisa exploratória pós-release. As
alterações concretas continuam versionadas na mesma branch de pesquisa; a
`main` e a tag `v1.22.1` permanecem referências imutáveis da entrega
acadêmica consolidada.


### 25.7 Hotfix de serialização v1.4.1

Na primeira execução completa da v1.4.0, os cálculos da validação de decisão
foram concluídos, mas a escrita de
`pesquisa_u67_decision_validation.json` falhou porque um campo diagnóstico
continha `NaN` e o serializador estava configurado com `allow_nan=False`.

O problema é exclusivamente de artefato. Não há evidência de falha no treino,
replay, checkpoints, ranking, DSR, PBO ou seleção temporal anterior ao ponto de
serialização.

A v1.4.1 introduz conversão recursiva de valores não finitos para `null`.
Também inclui `finalizar_u67_decision_validation_spyder.py`, que finaliza uma
execução v1.4.0 já calculada a partir dos arquivos existentes, sem repetir os
backtests. O recovery exige os artefatos intermediários, confere os 180
subconjuntos e os checkpoints de US$ 290.727.954,41339606 e
US$ 587.165.941,1330212, recompõe os hashes dos candidatos e então cria o JSON
e o pacote ZIP final.

A orientação para a execução afetada é preservar
`output/u67_decision_validation/`, atualizar a branch e executar somente o
script de finalização. Rodar novamente o runner principal apagaria esse output
no início e repetiria desnecessariamente a campanha.


### 25.8 Resultado da validação de decisão v1.4

A v1.4.0 foi concluída e o pacote final foi recuperado pela v1.4.1 sem repetir
modelos ou backtests. Os 180 subconjuntos da busca full-window e os checkpoints
de US$ 290.727.954,41339606 e US$ 587.165.941,1330212 foram preservados.

O diagnóstico de ranking no Challenger U67+10 mostrou IC cross-sectional médio
de 0,03419, NDCG@5 de 0,52660, NDCG@10 de 0,53826, sign accuracy de 63,10%,
top-1 exato de 2,26%, top-1 dentro do top-3 realizado em 8,25% e dentro do
top-5 em 12,71%. O sinal de ordenação global é pequeno, mas há enriquecimento
no topo em relação à expectativa combinatória.

O Deflated Sharpe Ratio permaneceu praticamente igual a 1 tanto com 180 trials
quanto na sensibilidade conservadora de 326 trials. A dimensão efetiva de
trials foi 1,2792, resultado de correlação média aproximada de 0,879 entre as
180 séries. Por isso, a dimensão efetiva não deve ser usada isoladamente como
prova de robustez.

O CSCV/PBO produziu 53,57% para log-growth e 59,52% para Sharpe. Isso indica
instabilidade material do processo de escolher o melhor cenário dentro da
amostra. Em contraste, o Challenger fixo de 10 ativos teve percentil OOS
mediano de 93,92% por log-growth e 82,04% por Sharpe nas mesmas 252 divisões,
ficando abaixo da mediana em 0% e 7,54% dos casos, respectivamente. A leitura
correta é que o cenário fixo é forte, mas o processo de seleção entre muitos
subconjuntos ainda apresenta selection overfitting.

O piloto temporal confirmou mudança de composição. Com apenas o fold 1,
NBR/PENN/EML foram escolhidos e praticamente empataram com o U67 no fold 2.
Com folds 1+2, a seleção passou para CRT/NBR/PNBK/PENN/JBL/STAG/MCHB/TOWN e
superou materialmente o U67 no fold 3. Na transição, NBR e PENN foram mantidos,
EML removido e CRT/PNBK/JBL/STAG/MCHB/TOWN adicionados.

Estado da evidência após v1.4:

```text
reprodutibilidade                 PASS
integridade dos checkpoints       PASS
stress de custos                  PASS com sensibilidade a slippage
leave-one-asset-out               PASS local
ranking cross-sectional           SINAL FRACO/MODERADO NO TOPO
Deflated Sharpe Ratio             PASS
PBO da rotina de seleção          WARNING
universo temporal                 PARTIAL PASS
redescoberta nested completa      PENDENTE
```

O Challenger de US$ 587,17 milhões permanece uma referência forte de
descoberta, não um resultado prospectivo final. A próxima fronteira científica
é a redescoberta nested de candidatos e a decisão temporal de
ADICIONAR/MANTER/REMOVER sem usar a janela futura.


### 25.9 Foco científico consolidado após v1.4

O ranking global não é objetivo da linha de expansão. Ele foi usado somente
para diagnosticar por que o modelo podia apresentar erro absoluto OOS elevado
e ainda gerar decisões econômicas úteis. O alvo científico consolidado é
pertencimento dinâmico ao universo.

A partir das séries diárias já preservadas, foi recuperada a contribuição
individual por fold dos 15 positivos. Os casos mais informativos foram:

```text
NBR   fold1 +41,24% | fold2 ~0%    | fold3 +5,66%
CRT   fold1 -0,98%  | fold2 +69,49%| fold3 ~0%
CAPR  fold1 -0,98%  | fold2 ~0%    | fold3 +36,91%
LEU   fold1 -15,35% | fold2 -39,29%| fold3 +102,90%
EML   fold1 +2,52%  | fold2 ~0%    | fold3 ~0%
LAD   fold1 -0,98%  | fold2 -3,24% | fold3 +6,58%
```

Isso demonstra diretamente que o sinal de pertencimento é temporal e que um
rótulo estático de ativo bom/ruim é inadequado. A positividade full-window é
somente evidência de descoberta dentro da janela controlada.

A linha ainda não mediu remoção contrafactual dos 65 ativos efetivos do U67.
Logo, não existe evidência para afirmar quais ativos do baseline deveriam sair.
Também não existe ainda densidade temporal suficiente para calibrar frequência
diária, histerese ou permanência mínima.

Os dados congelados enviados para a campanha são suficientes para continuar
sem novo download de candidatos; a próxima necessidade é gerar rótulos
contrafactuais temporais mais densos e nested, não coletar mais OHLCV.


### 25.10 Congelamento do Challenger U67+10 para integração operacional

Solicitação do autor em 2026-10-12: encerrar esta etapa de pesquisa com a tag
`research-u67plus10-challenger-v1.4.1` e transferir ao ambiente operacional
uma **nova estratégia de universo fixo**, sem substituir a política atualmente
vencedora.

A composição requisitada para a estratégia operacional é o universo U67
original, preservado em sua ordem e identidade, mais dez ativos, exatamente
nesta ordem:

```text
CRT,NBR,CAPR,PNBK,PENN,JBL,STAG,MCHB,LAD,TOWN
```

A fonte científica documentada tem 77 ativos solicitados e 75 efetivos
na janela controlada, com exclusões estruturais DOC e CLMT. O checkpoint
congelado de descoberta é US$ 587.165.941,1330212 para o cutoff 2026-10-06.

Condições da transferência:

- O universo fixo solicitado é apenas uma fotografia operacional da pesquisa,
  sem substituir a hipótese científica futura de ADICIONAR/MANTER/REMOVER.
- O ambiente operacional utiliza dados da Alpaca atualizados; o checkpoint
  congelado é referência histórica, não uma meta de capital futura.
- A nova estratégia é distinta da versão U67 preexistente, para preservar
  o histórico e permitir comparações.
- O autor escolhe quando utilizar o backtest para decidir a adoção no catálogo
  operacional; a pesquisa não estabelece homologações adicionais, vetos por
  PBO, etapas extras de aprovação nem requisito de novo estudo antes da escolha.
- Capital, Sharpe, MaxDD, pior fold, trades, exclusões e hashes permanecem
  artefatos de análise, e não bloqueios administrativos introduzidos nesta linha.

O histórico da v1.4.1 registra PBO de 53,57% (log-growth) e 59,52% (Sharpe)
como diagnósticos científicos, não como impedimentos automáticos de adoção.

O fechamento acadêmico oficial `v1.22.1` permanece independente. A tag
de pesquisa solicitada deve apontar para o commit de encerramento desta
branch, não para a `main`. A criação/push da tag anotada deve ocorrer via Git
após conferir o HEAD, pois os conectores usados para esta entrega só expõem
operações de escrita em branches, não criação de refs de tag.

### 25.11 Correção de escopo de transferência

Em 2026-10-12, o autor reiterou que não autorizou restrições adicionais de
pesquisa sobre o cadastro, o backtest nem a possibilidade de adotar o
Challenger como Winner no ambiente operacional. A redação anterior sobre
etapas extras de homologação foi retirada. Esta alteração de documentação
não modifica os resultados numéricos nem a política de universo congelada.


### 25.12 Pesquisa experimental de persistência preditiva multi-horizonte (2026-10-12)

**Versão experimental:** horizon-persistence-oos-v1.0.2.
**Branch separada:** research/horizon-persistence-oos-v1, criada a partir de
da pesquisa U67+10 no commit 13e5ed39.
Esta pesquisa **não modifica** os checkpoints, a reprodução oficial, o
checkpoint congelado U67+10 ou o universo original. Não existe promoção
automática nem mudança do resultado acadêmico validado.

**Hipótese:** a antiga permanência mínima fixa de dois pregões pode ser
substituída por evidência individual dos horizontes 5/10/20/40/60, acompanhando
a previsão registrada no instante de compra e suas revisões posteriores. O
modelo oficial atual aprende **uma única utilidade agregada** como alvo, com
pesos (0.10, 0.15, 0.20, 0.30, 0.25); ele não emite diretamente cinco
previsões individuais. O experimento ajusta cabeças LightGBM adicionais por
ativo e horizonte usando as colunas de alvo existentes
forward_horizon_utility_{h} — não pressupõe previsões que o sistema
original não produzia.

**Três braços de replay**, mantendo o mesmo universo efetivo, preços,
folds walk-forward, custos e modelos agregados:

1. control_holding_2 — política congelada, incluindo a margem de troca
   calibrada; o capital final deve reproduzir US$ 587.165.941,1330212.
2. ablation_holding_0 — mesma utilidade agregada, modelos, margens e
   custos; apenas o holding mínimo passa a zero.
3. horizon_confidence — sem holding mínimo; o primeiro filtro econômico
   da margem de troca permanece; uma segunda evidência usa previsões
   individuais por horizonte, pesos das frações ainda não vencidas das
   previsões feitas na entrada e erro absoluto p75 calculado estritamente
   antes da janela fora da amostra. Multiplicadores 0/0.5/1.0 são
   selecionados por score de calibração cronológica antes de cada fold
   de teste. O p75 **não** é uma probabilidade calibrada de ganho.

A nova regra emprega um estimador heurístico explícito: horizonte restante
r_h=max(0,1-idade_em_sessoes/h), previsão atual do ativo comprado e
previsão original no instante da entrada com mistura 50/50, e previsão
atual de cada concorrente. A troca é admitida se o ganho esperado no
horizonte em comparação com essa referência excede
multiplicador*p75_erro. O efeito de truncar horizontes não representa
matematicamente uma distribuição preditiva do retorno residual; por isso a
regra é uma hipótese mensurável de pesquisa, não confiança probabilística.
Quando o horizonte original se esgota, usam-se apenas previsões novas.
Em troca de fold, se uma posição foi herdada e a previsão exata da compra
anterior não está disponível, a observação atual é registrada como
substituto marcado na auditoria. A simulação financeira não recebe preços
futuros na seleção de operações.

**Dados e causalidade:** o script exige o snapshot U67 versionado no Git
e os arquivos RAW e Corporate Actions locais da campanha de dez ativos,
com SHA-256 registrado na saída. Não baixa ativos adicionais. A rotina
exige continuidade e ausência de quebra estrutural nos ativos externos.
Treino de cada horizonte é truncado em h observações antes do próximo
intervalo; o cálculo de erro descarta qualquer rótulo que só maturaria
depois da janela de calibração. Os mesmos dez símbolos congelados são
usados sem nova busca de subconjuntos.

**Execução:** abrir pesquisar_persistencia_horizontes_spyder.py no Spyder e
executar com F5; alternativamente:
python pesquisar_persistencia_horizontes_spyder.py

**Artefatos:** output/persistencia_horizontes_u77/{experiment.json,
comparison.csv, horizon_calibration.csv, horizon_decisions.csv,
control_holding_2_equity.csv, ablation_holding_0_equity.csv,
horizon_confidence_equity.csv e *trades.csv}.
Todas as saídas são de pesquisa. Nenhum resultado do novo experimento foi
observado no momento da implementação. Os testes unitários validam
causalidade de rótulos, memória da previsão de entrada, possibilidade de
troca imediata e saída para CASH; não equivalem a uma execução financeira
completa. A avaliação do efeito de múltiplas comparações e da capacidade
de execução econômica permanece relevante antes de qualquer adoção futura.

**Separação permanente:** apenas após execução e análise dos três braços,
com auditoria OOS e deliberação do autor, uma versão operacional distinta
poderá ser proposta. Esta pesquisa não altera a aplicação operacional.


#### 25.12.1 Hotfix H05 — eliminação do double purge (v1.0.2, 2026-10-12)

**Incidente observado na primeira execução local, confirmado pelo log do autor:**

- control_holding_2: US$ 587.165.941,13 (reproduzido ao centavo).
- ablation_holding_0: US$ 59.060.362,54.
- RuntimeError H5: insuficientes residuos maturados para medir a incerteza
  antes do OOS: 0.

O erro aconteceu antes de calcular qualquer resultado do braço multi-horizonte.
O fold 1 da calibração contém 700 observações de treino, o mínimo
rotation_minimum_training_rows=700. A primeira implementação fazia
safe_train_dates=state.train_dates[:-h], removendo mais 5 datas
(695 linhas disponíveis) embora o protocolo oficial já separasse o treino
da calibração com purge de 60 pregões. A função de treinamento retornava
um dicionário de modelos vazio e a medição de erro recebia previsões
não finitas, resultando em zero resíduos válidos. Não houve corrupção de
dados, alteração da estratégia base nem conclusão sobre a política nova.

A correção preserva todas as datas pré-calibração e pré-OOS cujo rótulo
de horizonte h matura antes do primeiro pregão de avaliação. Para cada
fold e h, a validação exige estritamente:

    (train_end_exclusive - 1) + h < evaluation_start_index

A verificação funciona para H05/H10/H20/H40/H60 com purge 60, sem
novo truncamento das 700 observações mínimas. Também verifica que
o dicionário de modelos não está vazio e mantém o descarte de
rótulos de calibração não maturados antes da decisão OOS. Novos testes
de regressão cobrem quantidade original de treinamento e rejeição de
rótulos futuros.

**A pesquisa nova ainda precisa ser executada novamente pelo autor.**
O backtest completo não está validado pela CI unitária; os resultados
Control e ablação observados não antecipam desempenho do método
multi-horizonte. Reexecutar pesquisar_persistencia_horizontes_spyder.py
após atualizar esta branch e enviar o ZIP produzido.


#### 25.12.2 Resultado executado dos três braços (v1.0.2, 2026-10-12)

**Fonte observada:** pacote local
pacote_persistencia_horizontes_u77.zip fornecido pelo autor, script_version
horizon-persistence-oos-v1.0.2, status completed e cutoff 2026-10-06.
Snapshot U67 SHA-256:
e440f59da5e684f1de59cf447abfedd9aed3f7817b3d5d681058fe631276a575.
Universo requisitado 77, efetivo 75, exclusões CLMT e DOC.
A campanha usa os dez ativos novos congelados e os mesmos três folds.

| Política | Capital final USD | Sharpe | MaxDD | CAGR | Rotações |
|---|---:|---:|---:|---:|---:|
| Control, holding=2 | 587.165.941,13 | 2,728536 | -41,6828% | 486,575% | 354 |
| Ablação, holding=0 | 59.060.362,54 | 2,257679 | -41,6828% | 305,150% | 450 |
| Horizonte/confiança heurística | 473.069,88 | 1,264438 | -74,7582% | 86,1475% | 129 |

O Control reproduziu o checkpoint de US$ 587.165.941,1330212 ao
centavo. A retirada da permanência mínima reduziu o capital simulado em
89,94145% e aumentou as rotações de 354 para 450. A nova política de
confiança alcançou apenas 0,0805683% do capital do Control e não
demonstrou vantagem. Não se deve exportar ou promover a nova política
com base nesta execução.

**Capitais acumulados nos finais de fold:**

| Fold | Control USD | Sem holding USD | Multi-horizonte USD |
|---|---:|---:|---:|
| 1 | 67.999,95 | 37.425,26 | 25.249,95 |
| 2 | 10.968.842,01 | 3.205.786,89 | 39.859,00 |
| 3 | 587.165.941,13 | 59.060.362,54 | 473.069,88 |

No braço multi-horizonte, as 1.560 decisões registradas incluem:
751 CURRENT_BEST, 666 HORIZON_CONFIDENCE_NOT_MET,
129 ROTATE_HORIZON_CONFIDENT, 13 SWITCH_MARGIN e uma entrada inicial.
O fold 2, especificamente, registrou 411 bloqueios de confiança,
apenas 4 rotações e 79 decisões em que o ativo atual era o melhor.
O multiplicador calibrado foi 0,0 no fold 1; 1,0 no fold 2; e 0,0
no fold 3. São escolhas de limiar heurístico, não probabilidades
estatisticamente calibradas.

O Control registrou 355 aplicações do MIN_HOLD_GUARD, porém somente
162 dessas decisões identificavam um ativo bruto superior ao
incumbente. Logo, contar 355 como oportunidades perdidas seria
incorreto. Comparando as posições diárias, a ablação diferiu do
Control em 164 de 1.560 sessões (10,5%), e o multi-horizonte em
712 sessões (45,6%). Os efeitos cumulativos refletem trajetórias
distintas de composição de capital, não apenas diferenças isoladas
de taxa de transação.

**Interpretação e limitações:** A política experimental faz uma
comparação adicional com uma mistura da previsão atual e da previsão
da compra em cinco horizontes, cujo erro absoluto p75 é calculado
em períodos anteriores ao OOS. Esse p75 não é calibrado como
probabilidade condicional de outra ação superar a posição corrente,
nem mede diretamente a diferença de utilidade de pares. No fold 2,
o limiar impediu quase toda a rotação. O experimento demonstra
fracasso da heurística específica, **não** refutação da possibilidade
de usar memória preditiva por horizonte.

Em 73 linhas de decisão a origem da previsão de entrada foi
fold_boundary_proxy (posição herdada de outro fold), e não a
previsão histórica original da compra. Esse caso deve ser
resolvido em uma próxima versão para obter a equivalência de estado
do replay prospectivo.

**Próxima hipótese a testar, em pesquisa separada nesta mesma branch:**
manter o comportamento do Control holding=2 como fallback, habilitando
um override antecipado estritamente quando evidência calibrada de
vantagem líquida específica do par candidato-incumbente for suficiente.
Para isolar o efeito, comparar baseline, override antecipado e
limiar calibrado, preservando o protocolo temporal. Registrar
decisões antecipadas, PnL contrafactual, custos, calibragem e
estabilidade por fold. Não escolher thresholds com base na janela
OOS final, não confundir o score agregado com probabilidade e não
impor aprovação ou gate novo à adoção: trata-se de pesquisa
científica para subsidiar decisão posterior.

**Separação entre defeitos:** eventual diferença do contador de
holding de produção precisa de correção operacional independente;
nenhum resultado aqui justifica manter um contador desatualizado.


#### 25.12.3 Versão v1.0.3 — override antecipado de rotação, sem veto adicional

**Hipótese operacional em pesquisa (não promovida):** a política Control
com permanência mínima de dois pregões apresentou desempenho muito superior
ao holding=0 e à primeira heurística de confiança por horizonte. Em lugar de
substituir toda a decisão Control por um veto de confiança, a nova experiência
mantém as decisões do Control como fallback e tenta **somente antecipar**
uma troca durante o período em que o Control ainda impediria a saída por
holding mínimo. Depois de satisfeita a permanência, não aplica qualquer
veto adicional à decisão Control.

**Código:** mesma branch research/horizon-persistence-oos-v1, no mesmo
arquivo pesquisar_persistencia_horizontes_spyder.py,
script_version horizon-persistence-oos-v1.0.3.

**Três braços:** controle original holding=2, ablação holding=0 e
early_horizon_override. Todos compartilham o universo U67+10 de 75 ativos,
calendário, modelos agregados, margens do Control, custos financeiros e
checkpoint histórico congelado do Control, que deve reproduzir
US$ 587.165.941,1330212 ao centavo antes das variantes.

**Política de override:** primeiro calcular a decisão oficial do Control
com a posição e idade efetivamente presentes na **própria simulação**
experimental. Somente se o Control mantiver o ativo, ainda houver holding
inferior a 2 pregões e existir uma previsão original de entrada registrada
pelo próprio modelo para essa posição, verificar a melhor alternativa.
Exigir vantagem positiva da utilidade agregada contra CASH e o incumbent,
considerando a mesma switch margin do Control. Em seguida, calcular a
vantagem predita do par por H05/H10/H20/H40/H60 e comparar tanto com a
previsão atual quanto com a previsão original da entrada, usando pesos
efetivos de horizontes ainda não expirados. Antecipar a troca apenas se
ambas as vantagens de par superarem o multiplicador do erro quantílico
correspondente. Não preencher previsões de entrada desconhecidas
com uma observação sintética. Se houver transição de fold e a posição
original for desconhecida, **não permitir override até a próxima
entrada real**, preservando a decisão Control.

**Erro de diferenças por par:** para cada data de estimação, comparar a
previsão de utilidade individual por horizonte do líder agregado com a
dos demais símbolos disponíveis e o retorno/risco realizado de cada par.
Agregam-se os erros absolutos pelo p75 dentro do dia, e usa-se o p75
desses erros diários para o horizonte. Somente rótulos totalmente
maturados antes da data de corte participam da estimação. Isso evita
contar 74 pares da mesma sessão como 74 observações cronológicas
independentes. **O indicador continua sendo uma heurística de margem
de erro, não uma probabilidade calibrada.**

**Calibração cronológica separada:** o primeiro segmento de 2/3 das
126 datas de calibração estima os erros por horizonte; o 1/3 final
seleciona exclusivamente entre os multiplicadores 0,0; 0,5; 1,0,
sem utilizar labels ou métricas da janela OOS. O protocolo original
preserva o purge de 60 dias para treinar os modelos de horizonte.
Um multiplicador pode permitir nenhuma troca antecipada; isso é um
resultado legítimo de comparação, não uma promoção obrigatória.

**Saídas v1.0.3 (separadas da v1.0.2):**

    output/persistencia_horizontes_u77_v103/
      experiment.json
      comparison.csv
      horizon_calibration.csv
      horizon_decisions.csv
      override_by_fold.csv
      control_decision_reasons.csv
      no_hold_decision_reasons.csv
      control_holding_2_equity.csv / *_trades.csv
      ablation_holding_0_equity.csv / *_trades.csv
      early_horizon_override_equity.csv / *_trades.csv
      pacote_persistencia_horizontes_u77_v103.zip

**Interpretação:** preservar a lógica do Control como fallback significa
não vetar as ações que ele autorizaria **dado o estado da variante no
mesmo dia**. Uma rotação antecipada modifica o ativo subsequente,
portanto a trajetória completa da variante pode divergir da
trajetória do Control: não se afirma identidade de ordens ou de capital
após um override. A análise medirá frequência, delta financeiro,
custos, drawdown, estabilidade por fold e causalidade sem reotimizar
o resultado original do TCC.

**Status de resultados:** A implementação em branch e os testes de
regressão não representam uma execução financeira completa desta
versão. Os valores de 2026-10-12 na seção anterior pertencem à
v1.0.2 e não se transferem automaticamente à v1.0.3. Os
resultados ficam pendentes de execução no ambiente de pesquisa com
os dados candidatos congelados. Nenhuma alteração foi feita na
aplicação operacional, no Winner ou na main do TCC.


#### 25.12.4 Versão v1.0.4 — duração mínima da permanência

**Motivação e evidência recebida (v1.0.3, 2026-10-12):** a execução do
pacote `pacote_persistencia_horizontes_u77_v103.zip` reproduziu o Control
(holding=2) em US$ 587.165.941,13 (Sharpe 2,7285, MaxDD -41,6828%).
O override antecipado encerrou em US$ 353.355.356,01 (Sharpe 2,6147);
holding=0 ficou em US$ 59.060.362,54. Houve 20 overrides apenas no
primeiro fold, com seleção do multiplicador zero nesse fold. A v1.0.3
não foi promovida. Não atribuir diferenças agregadas a uma única causa:
trajetórias e capital reinvestido divergem após decisões diferentes.

**Hipótese predefinida:** verificar se duração mínima fixa entre 1, 2, 3
e 4 sessões explica a vantagem do Control; comparar ainda uma política
que seleciona um desses quatro valores exclusivamente na calibração
anterior a cada fold (sem escolher o melhor pelo OOS). Holding=2 e
holding=0 permanecem como referências. A v1.0.4 NÃO antecipa rotações
com previsões por horizonte; mantém previsões agregadas, mesmos
hiperparâmetros, margens selecionadas para o Control, custos, universo
U67+10 e calendário congelado.

**Reprodução:** branch `research/horizon-persistence-oos-v1`;
arquivo existente `pesquisar_persistencia_horizontes_spyder.py`;
`VERSION=horizon-persistence-oos-v1.0.4`. O checkpoint Control
US$ 587.165.941,1330212 deve ser reproduzido ao centavo ANTES
de qualquer comparação. Só aceita os 75 ativos válidos, inclusive
os dez externos já congelados no cache local, com hashes externos
registrados. Falta de dados deve interromper a pesquisa, sem download,
bridge entre fontes ou alteração manual de ativo.

**Seleção por fold:** cada fold usa seu modelo e a margem já calibrada
pelo Control. Os holdings 1,2,3,4 são avaliados nos últimos 1/3 das
datas de calibração do próprio fold (aproximadamente 42 sessões).
O melhor score de crescimento ajustado por risco escolhe um único
valor para a janela OOS subsequente. O empate prefere holding=2,
depois o valor mais próximo de 2. Em nenhum momento a seleção lê
retornos ou métricas OOS. Como a margem Control já foi calibrada
utilizando a janela histórica e a amostra para holding é curta,
há risco de sobreajuste INTRA-calibração e elevada variância;
o método é exploratório e não uma comprovação de superioridade.
Os holdings fixos são comparadores registrados antes da execução,
não candidatos selecionáveis pelo resultado final OOS.

**Diagnóstico causalmente disponível:** registrar nas decisões em que
o Control bloqueou uma troca por permanência mínima as variáveis
`vol_20`, `vol_60`, `return_5`, `trend_efficiency_20`,
`distance_from_high_20` da data da decisão, sem rótulos futuros.
Esse CSV é observacional, não identifica sozinho o efeito causal
de regime ou da mudança de holding; não há regra adaptativa
treinada por essas variáveis na v1.0.4.

**Outputs separados da v1.0.3:**

    output/persistencia_horizontes_u77_v104/
      experiment.json
      comparison.csv
      fold_comparison.csv
      calibrated_holding_by_fold.csv
      hold_guard_point_in_time_features.csv
      control_decision_reasons.csv
      trades_by_asset.csv
      trades_by_month.csv
      *_equity.csv / *_trades.csv
      pacote_persistencia_horizontes_u77_v104.zip

**Status:** implementação publicada apenas nesta branch de pesquisa.
Testes unitários de seleção e causalidade adicionados. Backtest financeiro
v1.0.4 pendente de execução no Spyder com cache histórico completo.
Não altera a `main`, a reprodução oficial, a aplicação operacional
ou qualquer Winner. Os resultados devem ser interpretados por fold,
Sharpe, drawdown, custos e capital final, sem promoção automática.


#### 25.12.5 Versão v1.0.5 — ganho percentual previsto relativo ao ativo atual

**Hipótese fixada antes do backtest:** quando outra ação do mesmo universo
apresenta retorno percentual **líquido previsto** maior que o da ação
mantida em carteira nos horizontes de curto e médio prazo, pode valer
a pena substituí-la. H05 e H10 formam o grupo curto; H20 e H40 formam
o grupo médio. O H60 permanece somente no alvo da política Control
original. Comparação é por ativo contra o incumbent na mesma data e
não por preço nominal de uma ação.

**Alvo numérico:** para cada H05/H10/H20/H40, criar
`forward_horizon_net_return_pct_H` a partir de
`expm1(forward_horizon_net_log_return_H)`, cujo log já desconta
custos round-trip conforme configuração congelada. Cada regressão
LightGBM estima diretamente retorno percentual líquido esperado
condicional às características da data. O alvo utiliza futuro
**somente no ajuste do modelo**, nunca para gerar decisões,
selecionar ativos ou regras OOS. Um retorno líquido previsto é
estimativa do modelo, não lucro garantido. Os custos e a liquidez
efetiva continuam limitações.

**Treinamento causal:** utilizar os mesmos folds e o embargo original
de 60 sessões. Cada novo modelo H é treinado somente com rótulos
que estejam inteiramente maturados antes do começo de seu
período OOS, verificados com
`select_purged_horizon_training_dates`. Em cada fold treinar
4 regressões por ativo, reutilizando o mesmo pipeline, features
contemporâneas, parâmetros LightGBM e snapshot U67+10.
Usar as previsões para a data atual (não o retorno observado
nos próximos H dias). Nenhuma consulta Alpaca e nenhuma troca
de fonte para corrigir ativos.

**Duas regras pré-definidas, ambas experimentais:**
1. `relative_short_medium_group`: candidato deve ter previsão
   de retorno líquido positivo nos dois grupos, superar o ativo
   atual em ao menos 0,005 de retorno (0,5 ponto percentual)
   nas médias H05/H10 e H20/H40; se o Control estiver
   trocando para outro candidato, a nova escolha deve superar
   também este destino nos dois grupos.
2. `relative_all_four_positive`: mesmas condições, e vantagem
   estritamente positiva em cada H05/H10/H20/H40 individual,
   além de utilidade agregada superar o incumbent pela margem
   Control. Isso é mais conservador que a regra 1.

**Restrições:** manter holding mínimo=2, todas as previsões
agregadas do Control, limiar de caixa e limiar de entrada.
Preservar integralmente a ação do Control quando houver saída
para caixa, quando o tempo mínimo não tiver vencido, ou quando
não houver candidato qualificado; não acrescentar veto contra
rotações do Control. Permitir que a regra experimental selecione
um ativo alternativo quando supera as referências pelo critério
de retorno dos horizontes. A diferença em decisões subsequentes
pode aumentar com trajetórias distintas. Não interpretar o
limiar 0,005 como uma margem otimizada, nem selecionar vencedor
olhando o OOS.

**Auditoria e limites:** registrar ranking dos dez primeiros
candidatos por data/variante, previsões individuais e médias,
diferenças em pontos percentuais frente ao ativo atual e destino
Control, ações executadas, motivo, folds, custos, Sharpe,
drawdown, capital final e os hashes dos dez históricos externos.
A vantagem em retorno projetado não é causal nem garante
execução real; comparar também com o baseline em cada fold.
Como as mesmas janelas OOS já foram examinadas nas v1.0.2 a
v1.0.4, o experimento é exploratório e a confirmação exigirá
outra janela cronológica, sem ajustá-la a esses resultados.

**Checkpoint de regressão:** o Control holding=2 deve gerar
US$ 587.165.941,1330212 ao centavo, sem mutações na main.

**Saídas:** `output/persistencia_horizontes_u77_v105/`,
com `comparison.csv`, `fold_comparison.csv`,
`group_candidate_ranking_top10.csv`,
`four_horizons_candidate_ranking_top10.csv`,
`group_policy_decisions.csv`,
`four_horizons_policy_decisions.csv`,
`horizon_model_training.csv`, curvas e operações por variante,
`experiment.json` e ZIP `pacote_persistencia_horizontes_u77_v105.zip`.

**Status:** código e testes unitários da v1.0.5 publicados na
branch de pesquisa. Backtest financeiro pendente de execução
local no Spyder com o cache U67+10. Não promove estratégia
operacional ou altera a reprodução acadêmica oficial.


#### 25.12.6 v1.0.6 — análise de captura e seleção expandida de auditoria

**Motivação:** v1.0.5 confirmou Control holding=2 US$ 587.165.941,13
no universo U67+10. Regras adicionais com ganhos projetados
superiores em H05/H10/H20/H40 reduziram o resultado financeiro.
A maioria das operações do Control durou menos de cinco a dez
sessões. É necessário testar a correspondência entre horizonte
preditivo e duração operacional antes de formular novas trocas.

**Dois estudos na MESMA execução, com seleções independentes:**
A. `capture` considera H02/H03/H05/H10, priorizando
   horizontes de captura observados na carteira. Compara todos
   os candidatos com o ativo atual na data de decisão, com
   previsão percentual líquida e um desconto incremental
   conservador de custo de troca.
B. `expanded_audit` classifica os mesmos candidatos de maneira
   independente em H01/H02/H03/H05/H10/H15/H20/H30/H40/H60.
   Também exporta o melhor candidato isolado por horizonte.
   Inclui granularidade mais ampla, mas não amplia o prazo
   máximo além do embargo causal original de 60 sessões.
   H90/H120 exigiriam reprojetar o protocolo temporal, não
   sendo testados nesta versão.

**Sem nova estratégia de rotação:** ambos os seletores são
EXCLUSIVAMENTE diagnósticos. A simulação financeira executa
somente o Control, sem ordens experimentais. O Control deve
reproduzir o checkpoint ao centavo antes do estudo. Não
existe promoção do melhor candidato ou do melhor horizonte
por desempenho observado fora da amostra. O mesmo fold
histórico já foi consultado em outras campanhas e, portanto,
a pesquisa ainda é exploratória.

**Alvos e preços:** usar o mesmo ajuste e snapshot de dados.
H05/H10/H20/H40/H60 reutilizam exatamente os targets
`forward_horizon_net_log_return` previamente definidos
pelo pipeline. H01/H02/H03/H15/H30 produzem rótulos
suplementares usando abertura na sessão seguinte,
fechamento da sessão t+H e custo round-trip equivalente,
reindexando no calendário oficial ANTES de aplicar o
deslocamento; não se comprimem lacunas ou se criam dias
sintéticos. O log líquido é convertido a retorno percentual
por `expm1`. É proibido usar retornos observados futuros
para treinar dentro do fold OOS ou para classificar candidatos
na data de decisão.

**Calibração e treino:** novos LightGBM com as features,
parâmetros, universo e folds oficiais. Há uma regressão
de retorno percentual para cada ativo e horizonte,
treinada por fold exclusivamente com labels que
amadurecem antes do período de teste. H60 usa o embargo
de 60 pregões já existente; nenhum alvo superior a H60
é treinado sem ajustar o desenho experimental.

**Ranking:** converter a diferença entre percentual
previsto de candidato e incumbente em pontos percentuais.
Descontar um custo incremental round-trip de troca como
proxy conservadora (não um backtest executável);
normalizar o delta por H para gerar um score diário
comparável entre horizontes; ordenar por maior média
do ganho diário previsto, com desempate determinístico.
Registrar cinco melhores candidatos por data e seleção,
quantidade de horizontes com vantagem e as previsões
individuais. Uma vantagem prevista negativa não gera
`best_candidate`, mas o ranking pode incluir pontuações
negativas para auditoria.

Na **selecao de captura**, o candidato somente e escolhido se
pelo menos tres dos quatro horizontes apresentarem vantagem
individual e as medias H02/H03 (curto) e H05/H10 (medio)
forem ambas positivas. A **auditoria expandida** nao herda
esse filtro, preservando uma amostra mais ampla para medir
concordancia e divergencias entre horizontes. Essas regras
foram fixadas antes do backtest v1.0.6.

**Avaliação fora da amostra, separada da seleção:**
após congelar os rankings, juntar retornos realizados
do ativo atual e do candidato escolhido por H.
Somente calcular um resultado quando t+H se completar
DENTRO do mesmo fold OOS, evitando dados do fold
seguinte ou do futuro não observado. Resultado realizado
é uma diferença contrafactual de retorno fixo, não
um P&L de carteira executado ou um efeito causal.
Relatar taxa de acerto, ganho realizado, MAE, RMSE,
acurácia de sinal e quantidade de observações maduras,
por fold e horizonte. Registrar hashes dos históricos
complementares, preservando exclusões estruturais.

**Saídas isoladas:**
`output/persistencia_horizontes_u77_v106/`:
`control_holding_2_equity.csv`,
`control_holding_2_trades.csv`,
`control_decision_reasons.csv`,
`capture_ranked_top5.csv`, `capture_selected.csv`,
`capture_quality_by_fold_horizon.csv`,
`expanded_audit_ranked_top5.csv`,
`expanded_audit_selected.csv`,
`expanded_audit_best_per_horizon.csv`,
`expanded_audit_quality_by_fold_horizon.csv`,
`horizon_model_training.csv`, `comparison.csv`,
`experiment.json` e o ZIP
`pacote_persistencia_horizontes_u77_v106.zip`.

**Estado:** código de pesquisa na branch
`research/horizon-persistence-oos-v1`; backtest v1.0.6
pendente no Spyder com o cache histórico privado.
Sem downloads de corretora e sem alteração na main,
na execução oficial ou na aplicação operacional.


#### 25.12.7 v1.0.7 — pares de ativos, erro histórico e comparação com a decisão Control

**Causa e hipótese:** a pesquisa v1.0.6 exibiu vantagens previstas
em H02/H03/H05/H10 que não foram sustentadas, em média, pelos ganhos
realizados. A escolha do maior valor entre dezenas de ativos pode
produzir otimismo no vencedor do ranking. O novo experimento investiga
se comparar diretamente o ganho percentual previsto do candidato
com o ativo realmente mantido **e com o ativo que a política Control
escolheria** produz oportunidades mais confiáveis.

**Separação de evidências:** não se treina um modelo LightGBM dedicado
a cada par, pois isso exigiria outro desenho computacional e temporal.
A v1.0.7 usa as previsões de retorno percentual líquido das regressões
por ativo/H e calcula o **delta de retorno entre pares**, estimando
diretamente sobre os pares historicamente escolhidos um fator
de contração `slope` em [0,1] e o quantil p75 do erro absoluto.
Isso calibra a diferença condicionada ao procedimento de seleção
e combate o otimismo do máximo de muitos candidatos, mas
não garante nem demonstra superioridade causal.

**Calibração cronológica:** em cada um dos três folds, modelos
auxiliares H02/H03/H05/H10 são ajustados apenas no período de treino,
separado da calibração por embargo. Produzem previsões inéditas para
as datas da calibração. Simula-se a posição histórica do Control
naquela janela para formar um conjunto de referências; candidatos
são ordenados usando exclusivamente previsões contemporâneas.
O primeiro 2/3 dessa janela fornece pares vencedores para
estimar contração e erro. Um rótulo do período de ajuste só entra
se o seu horizonte maturou estritamente ANTES do início do
último 1/3 da calibração. O último 1/3 serve como
**checagem histórica** e não é usado para determinar slope/p75.
A margem do próprio Control foi anteriormente calibrada na
janela histórica total; portanto essa checagem não deve ser
tratada como holdout inteiramente independente.

**Confiabilidade:** aplicar slope às diferenças projetadas
candidato-Control e candidato-incumbente. Exigir ganho curto
(H02/H03) e médio (H05/H10) positivo contra ambos, vantagem
positiva em pelo menos 3 horizontes, e o score diário ajustado
superior ao limite derivado exclusivamente dos erros p75 da
primeira parte da calibração. Nenhum fator ou limiar é escolhido
pelas métricas OOS. Se não houver ao menos 15 dias cronológicos
distintos com pares maturados para cada horizonte, interromper
com erro em vez de preencher estimativas. Exportar seleção
bruta e seleção com confiança, mesmo que a última esteja vazia.

**Custo:** uma estimativa incremental específica de cada par
usa exclusivamente os fechamentos conhecidos na data da decisão,
slippage, comissão, SEC, TAF e CAT configurados. Considera
venda+compra quando Control mantém o ativo, e a diferença
positiva de custo de compra quando Control já giraria para
outro ativo. O proxy não substitui a contabilidade exata
nem modela impacto de mercado, liquidez, quantidades e caps
finais de taxas; quantidades futuras não são observadas.
As métricas financeiras do Control são as da simulação
original, jamais substituídas por essa estimativa.

**Sem transações experimentais:** Control holding=2 é a
única carteira simulada e deve reproduzir
US$ 587.165.941,1330212 ao centavo. Toda oportunidade
de troca é contrafactual em horizonte fixo, não operação
executada nem estimativa automática de capital final.

**Auditoria expandida independente:** H01/H02/H03/H05/H10/
H15/H20/H30/H40/H60 continuam sendo avaliados na mesma
execução, mas não participam do ajuste de slope, da seleção
de confiança ou da escolha de horizontes da pesquisa
de captura. Ranking consolidado e líder individual
por horizonte são auditados com custos estimados
específicos do candidato e retornos realizados apenas
quando t+H termina dentro do próprio fold. O script
usa `decision_to_fold` para não duplicar a data de
fronteira entre folds contíguos. Limite máximo H60
preserva o embargo original de 60 sessões.

**Outputs separados:** `output/persistencia_horizontes_u77_v107/`,
com `control_holding_2_equity.csv`,
`control_holding_2_trades.csv`,
`control_decision_reasons.csv`,
`capture_pair_ranked_top5.csv`,
`capture_pair_raw_selected.csv`,
`capture_pair_confident_selected.csv`,
`capture_pair_calibration_by_fold.csv`,
`capture_pair_historical_validation.csv`,
`capture_pair_validation_quality.csv`,
`capture_pair_raw_quality.csv`,
`capture_pair_confident_quality.csv`,
`expanded_audit_ranked_top5.csv`,
`expanded_audit_selected.csv`,
`expanded_audit_quality_by_fold_horizon.csv`,
`expanded_audit_best_per_horizon.csv`,
`expanded_audit_per_horizon_quality.csv`,
`horizon_model_training.csv`, `comparison.csv`,
`experiment.json` e pacote
`pacote_persistencia_horizontes_u77_v107.zip`.

**Governança:** a main, a reprodução oficial e a aplicação
operacional não são modificadas. Não há promoção automática
de estratégia. Os mesmos folds já foram analisados nas
campanhas v1.0.2–v1.0.6, de modo que esta avaliação
continua exploratória; confirmação externa exige novo
período cronológico ainda não utilizado na pesquisa.
Resultados financeiros da v1.0.7 pendentes de execução
local com os históricos complementares congelados.

#### 25.12.8 v1.0.8 — modelo direto de diferenças de retorno

**Motivo:** na v1.0.7, a maior parte dos ganhos relativos previstos nao se confirmou e nenhuma troca passou pelo filtro de confianca. A v1.0.8 investiga o aprendizado supervisionado direto da diferenca de retorno entre candidato e ativo de referencia em H02/H03/H05/H10, preservando o Control como unica carteira simulada.

**Correcao da cobertura:** todos os rotulos H01/H02/H03/H05/H10/H15/H20/H30/H40/H60 sao reconstruidos individualmente no calendario congelado a partir da abertura em t+1 e fechamento em t+H com custo-alvo round-trip. Um label curto pode amadurecer nas ultimas 60 sessoes mesmo quando H60 nao amadureceu. Lacunas permanecem ausentes. Esses rotulos somente alimentam treinos anteriores ao teste e verificacoes posteriores, nunca a escolha em t.

**Modelo direto:** para cada horizonte curto, o LightGBM aprende R_candidato - R_referencia a partir de 13 indicadores disponiveis em t para cada ativo e suas 13 diferencas (39 colunas). Os pares de treinamento usam duas referencias rotativas deterministicas por data, sem conhecimento de rentabilidade futura, nas duas orientacoes. Treino cronologico em duas etapas por fold: antes da calibracao e antes do teste, sempre respeitando o embargo original de 60 sessoes. CPU deterministica e parametros congelados.

**Calibracao:** os primeiros dois tercos da janela historica fornecem vencedores de ranking escolhidos apenas por previsao e rotulos plenamente maduros antes da divisao, para estimar vies mediano e erro absoluto p75 por H. O ultimo terco e destinado a verificacao historica, sem ajuste por resultados OOS. Nessa etapa a referencia historica usa a margem-base do Control, nao a margem selecionada a posteriori nessa janela.

**Selecao:** comparar candidato simultaneamente com o ativo atual e com o destino escolhido pelo Control; exigir vantagem nos grupos H02/H03 e H05/H10, tres horizontes positivos contra Control e, para a variante conservadora, limite inferior corrigido por vies e erro historico positivo em ambos os grupos. As taxas incrementais estimadas por candidato incluem slippage, comissao, SEC, TAF e CAT conforme parametros e cotacao disponivel na data da decisao. Essa estimativa nao substitui execucao com liquidez e impacto de mercado.

**Auditoria independente:** a mesma execucao treina modelos individuais H01 a H60 sem influenciar os modelos de pares. Exporta rank consolidado, lideres por H, ganhos realizados pos-selecao, qualidade por fold e completude de labels em horizon_label_coverage.csv. Decisoes de fronteira nao sao duplicadas; outcomes que atravessam o final do fold sao ignorados.

**Saidas:** output/persistencia_horizontes_u77_v108/pacote_persistencia_horizontes_u77_v108.zip, incluindo captura de pares (ranking, escolhas brutas e confiantes, calibracao e qualidade historica/OOS), auditoria expandida, coverage, hashes do U67+10, experiment.json e trades/curva do Control.

**Status:** somente branch research/horizon-persistence-oos-v1; main, experimento oficial e aplicacao operacional inalterados. Os folds OOS foram vistos em pesquisas anteriores: conclusoes sao exploratorias e exigem uma nova janela futura para confirmacao. O backtest completo da v1.0.8 permanece pendente no Spyder.

#### 25.12.9 v1.0.9 — comparacao integral de rankings, mesma execucao

**Objetivo:** auditar a capacidade de ordenacao dos 75 ativos usando as
previsoes e os historicos congelados, sem alterar o Control. Executar
oito classificadores sobre exatamente o mesmo conjunto de ativos
e datas elegiveis, com tie-break por simbolo, e produzir logs,
ranking integral e metricas por horizonte e fold.

**Metodos registrados ANTES de observar novos resultados:**
1. predicted_return: media igualmente ponderada dos retornos previstos
   por pregao em H02/H03/H05/H10;
2. net_advantage: media por pregao das diferencas esperadas contra
   destino Control, descontado custo incremental do ativo;
3. borda: combinacao das quatro posicoes transversais, sem depender
   das magnitudes extremas das previsoes;
4. topsis: distancia normalizada entre solucao ideal e anti-ideal
   com pesos fixos: H02/H03/H05/H10, cada qual 0,15; volatilidade
   0,15; friccao 0,15; dispersao entre horizontes 0,10;
5. pareto: fronteiras nao dominadas em quatro ganhos diarios
   e menor volatilidade, com desempate deterministico;
6. probabilistic: media das probabilidades aproximadas de vantagem
   sobre o destino Control em H02/H03/H05/H10, com distribuicao
   Normal de erro, medias e desvios estimados APENAS em previsoes
   cronologicamente anteriores ao OOS. A independencia implicita
   entre erros de ativos e apenas aproximacao; nao tratar como
   probabilidade calibrada sem verificacao;
7. robust: pior ganho previsto apos ajuste de vies e um desvio
   historico aproximado, por horizonte e por pregao;
8. learning_to_rank: comparador supervisionado adicional LambdaMART
   (LGBMRanker) com features de mercado disponiveis em t e
   relevancia historica H05 agrupada por sessao. Diferentemente
   dos demais, este metodo treina novo modelo; seus labels
   amadurecem integralmente antes do fold OOS e obedecem embargo.
   Nao usa ganho OOS no treinamento.

**Amostra e avaliacao:** os rankings e scores usam somente informacoes
disponiveis no fechamento da sessao de decisao, modelos treinados
anteriormente e calibracao historica anterior ao periodo OOS.
Para cada metodo, usar a mesma lista de candidatos com previsoes
finitas em todos os horizontes, dados completos de features e
custos estimados. A politica Control e o benchmark fixo.
Calcular retornos realizados e rankings verdadeiros apenas APOS
congelar todas as ordenacoes por data. Nao substituir Top-1 por
Top-2 quando o rotulo futuro do primeiro estiver ausente;
marcar esse caso como nao avaliavel. Nao permitir outcomes
que atravessem o final do fold.

**Metricas:** correlacao de Spearman, Kendall, NDCG@5,
retorno realizado do Top-1 e da media Top-5 relativo ao Control,
taxa de acerto do Top-1 e arrependimento do Top-1 comparado
ao melhor retorno observado. Relatar por dia/horizonte/fold,
e agregar os resultados de modo macro (media dos 3 folds),
sem selecionar pesos/limiares a partir do OOS.
A metrica PRINCIPAL descritiva foi fixada como media macro-fold
do NDCG@5 (medida de posicionamento no topo da lista).
O metodo que obtiver o maior NDCG@5 sera chamado de melhor
ordenacao OBSERVADA nesta amostra, nunca de vencedor
confirmado e nunca sera promovido automaticamente.

**Logs e arquivos:** manter no mesmo
`output/persistencia_horizontes_u77_v109/`:
`ranking_methods.log` (inicio, metodos, fold, lideres
periodicos e leaderboard completo),
`ranking_all_assets.csv` (cada posicao para todos os
metodos e datas com previsoes e desfechos maduros),
`ranking_oos_quality_by_session.csv`,
`ranking_methods_by_fold.csv`,
`ranking_methods_leaderboard.csv`,
`ranking_historical_error_calibration.csv`, alem de toda
a auditoria v1.0.8 e
`pacote_persistencia_horizontes_u77_v109.zip`.
O ZIP deve conter os logs e CSVs.

**Governanca e limitacoes:** somente a branch
`research/horizon-persistence-oos-v1`.
A main, a reproducao Control congelada e a aplicacao
operacional permanecem inalteradas. O checkpoint do Control
US$ 587.165.941,13 continua obrigatorio. Nenhuma nova
politica e simulada financeiramente ou promovida.
Como os folds OOS anteriores ja foram vistos muitas vezes,
o leaderboard e exploratorio e nao constitui prova
de generalizacao. Efeito de slippage real, liquidez,
capacidade e descontinuidade de ativos nao e modelado
pela ordenacao.

**Status:** implementacao programatica concluida;
backtest local Spyder e leaderboard financeiro v1.0.9
pendentes de execucao com o cache historico U67+10.

#### 25.12.10 v1.0.10 — rotacoes matematicas autonomas

**Questao da pesquisa:** nao e obrigatorio superar o capital final
do Control; pretende-se testar politicas mais autonomas, precisas,
reproduziveis e robustas que realizem rotacoes com menos decisoes
desfavoraveis. O Control permanece como comparador congelado, nunca
como entrada para a decisao das novas politicas.

**Previsoes:** reutilizar as mesmas estimativas LightGBM individuais
e cronologicas para H02/H03/H05/H10 da pesquisa v1.0.9.
Nenhuma politica AUTONOMA treina um modelo de ordenacao.
A auditoria legada v1.0.9 ainda executa seu comparador LambdaMART
em separado, sem participar das novas escolhas; isso prolonga
o tempo total de execucao. Os modelos de previsao continuam
existindo, mas a ordenacao das novas politicas e matematica.

**Quatro politicas de decisao independentes**, cada uma com sua
propria carteira simulada, estado e duracao de permanencia:
- `net_gain`: ordenar pela media de retorno diario previsto,
  descontando custo de entrada/troca em t;
- `borda_pareto`: combinar posicoes nos quatro horizontes e
  favorecer candidatos na fronteira de Pareto, incluindo menor risco;
- `topsis`: combinar ganhos previstos, risco, custo e dispersao,
  com pesos fixos e normalizacao transversal;
- `consensus`: exigir intersecao do Top 5 de Borda, TOPSIS e ganho
  liquido, mais nao dominancia de Pareto.

**Decisao:** permanecer pelo menos dois pregoes antes de trocar ou
sair para caixa; abrir posicao somente se a vantagem esperada for
positiva. Troca entre ativos requer melhoria liquida prevista
positiva contra a POSICAO ATUAL e vantagem prevista positiva nos
grupos H02/H03 e H05/H10. Consenso exige tambem tres de quatro
horizontes positivos. Se a expectativa media diaria do ativo atual
for negativa e a permanencia minima tiver sido cumprida, e permitido
sair para caixa. Nao escolher parametros com resultados dos folds
OOS. Os custos estimados na decisao usam apenas fechamento presente
e parametros de slippage/comissao/SEC/TAF/CAT conhecidos;
a contabilidade das simulacoes usa as funcoes originais de taxas
e execucao do Control.

**Experimento financeiro:** apos reproduzir o checkpoint do Control
US$ 587.165.941,13 ao centavo, executar quatro replays separados
via `_simular_exato` sobre o MESMO calendario OOS, universo de 75
ativos e modelo de custos; nenhuma operacao real. Comparar capital
final, Sharpe, drawdown, pior fold, fees, quantidade de trocas,
exposicao, tempo de permanencia e politica original.
Manter o Control como checkpoint, sem alteracao na implementacao
`engine/rotacao.py` nem na branch `main`.

**Precisao das rotacoes:** depois da simulacao, usar somente
retornos futuros ja maturados no mesmo fold e medir se cada
acao de TROCAR (incluindo caixa) teria apresentado retorno
liquido superior a permanecer no ativo anterior em H02/H03/H05/H10.
Nao permitir desfechos atravessando o final do fold; esses
retornos sao apenas diagnostico ex post de qualidade da decisao,
nao resultados financeiros de uma politica nem insumos da escolha.
Avaliar mesma definicao tambem sobre as decisoes do Control.
A metrica primaria diagnostica pre-fixada e precisao H05 por fold,
calculando a media macro dos folds. Para evitar falsas vitorias
com pouquissimas trocas, rankings de qualidade so recebem posicao
quando ha pelo menos 10 trocas com H05 maduro em CADA fold.
Metodos sem amostra suficiente permanecem reportados sem rank.
Desempate alfabetico; nenhuma promocao por resultados OOS.

**Artefatos v1.0.10:**
`output/persistencia_horizontes_u77_v110/` com as trilhas
v1.0.9 preservadas e novos
`autonomous_<metodo>_equity.csv`,
`autonomous_<metodo>_trades.csv`,
`autonomous_rotation_decisions.csv`,
`autonomous_rotation_realized_quality.csv`,
`autonomous_rotation_quality_by_fold.csv`,
`autonomous_financial_summary.csv`,
`autonomous_quality_leaderboard.csv`,
`rotation_policy.log`, `execution_full.log`, `experiment.json`
e `pacote_persistencia_horizontes_u77_v110.zip`.
Logs completos de stdout e stderr aparecem no console Spyder
e sao gravados no arquivo. Aviso sonoro no fim em Windows
por `winsound.Beep`: sucesso com dois tons altos, erro com
dois tons baixos. Fora de Windows usa fallback ASCII bell,
dependendo do terminal. Em caso de erro, `execution_full.log`
contem stack trace e deve ser enviado mesmo se nao gerar ZIP.

**Limitacoes importantes:** a simulacao nao inclui impacto real de
mercado, spread e restricoes de capacidade proporcionais ao
capital acumulado. O universo inclui ativos de liquidez variavel.
Os tres folds OOS ja foram vistos em varias pesquisas; todos
os resultados desta versao sao exploratorios e exigem uma nova
janela temporal intocada para confirmacao. Precisao positiva
H05 nao prova melhor curva de capital. A saida para caixa e
experimental. O estudo e isolado de sistemas operacionais.

**Status:** codigo versionado e testes automaticos de regressao,
sem execucao completa com os historicos congelados do Spyder.


#### 25.12.10.A Marco de encerramento da v1.0.10 (13/10/2026)

**Status da campanha:** execucao Spyder completa, auditoria de
artefatos realizada, resultados congelados como referencia de
pesquisa exploratoria. **Nao promover** nenhuma politica; nao
alterar a `main`, o Control ou a aplicacao operacional.

**Artefato local imutavel recebido:**
`pacote_persistencia_horizontes_u77_v110.zip`
SHA-256:
`c44d7b4c6fa7268afeefab4d6b8d82bc3673e2b57b5b4a83e63cd495e6ba7ad2`.
O ZIP do usuario nao foi versionado no Git; o hash permite sua
identificacao precisa apos transferencia externa.

**Checkpoint:** 75 ativos, tres folds, Control reproduzido com
capital US$ 587.165.941,1330212. As quatro politicas autonomas
foram simuladas, cada uma com sua carteira independente. Todos
os indicadores seguintes vem do pacote v1.0.10 e nao sao
garantia de desempenho futuro.

| Politica | Precisao H05 macro-fold | Trocas H05 maduras | Capital final (US$) | MaxDD |
|---|---:|---:|---:|---:|
| Control | 58,7604% | 352 | 587.165.941,13 | -41,68% |
| Net gain | 51,9033% | 195 | 165.481,13 | -70,66% |
| TOPSIS | 50,9317% | 187 | 120.939,53 | -75,47% |
| Borda + Pareto | 50,1932% | 209 | 156.107,33 | -76,73% |
| Consensus | 49,0961% | 200 | 216.922,97 | -65,43% |

**Diagnostico do mecanismo Control, sem afirmacao causal:**
`control_decision_reasons.csv` registra 1.560 decisoes:
831 `HOLD_CURRENT_BEST`, 355 `MIN_HOLD_GUARD`,
354 `ROTATE_TO_BEST_ASSET`, 19 `SWITCH_MARGIN_GUARD`
e uma entrada inicial. Nas 355 sessoes do guard,
em 193 o ativo atual continuava em primeiro no ranking
bruto; em **162** o ranking indicava outro ativo, mas a
permanencia minima impediu trocar.
As margens efetivas de troca eram 0,0005 nos folds 1/3
e 0,01 no fold 2, escolhidas no historico de calibracao.
Control estima **utilidade ajustada ao risco**
(`forward_risk_adjusted_utility`), incluindo penalidades
de downside/drawdown e componentes de movimento/persistencia,
enquanto as novas politicas ordenam previsoes de retornos
percentuais H02/H03/H05/H10. Assim, diferem
**o alvo da previsao e a politica**, e nao somente o hold.

**Auditoria retrospectiva adicional sobre o guard:**
para 161 dos 162 dias de bloqueio com rival na lideranca,
foi possivel obter o retorno H05 dos dois ativos nos
arquivos `ranking_all_assets.csv`.
Se o melhor bruto fosse comprado imediatamente, seu
retorno H05 superaria o retorno de manter em 85/161
dias (52,8%), mas a diferenca media foi aproximadamente
**-0,24 ponto percentual**, sem simular custos e
trajetoria decorrentes da troca. Esse exercicio
diagnostico nao prova que o guard cause a precisao
maior; bloqueia tanto oportunidades positivas quanto
negativas.

**Conclusao:** nenhuma politica autonoma alcancou o
Control em precisao de troca H05. Nao confundir
`precisao condicional entre as trocas executadas` com
qualidade de TODAS as decisoes de manter/rotacionar/caixa.
O Control escolhe quando operar segundo utilidade de
risco calibrada, favorecendo uma amostra selecionada
de rotacoes. O motivo de sua maior precisao deve ser
investigado mediante **ablacao controlada**, nao
atribuido ao hold por inferencia.

**Proposta subsequente, ainda NAO implementada:**
comparar Control completo com ablacões de hold=0,
margem fixa e mesmo sinal de utilidade; isolar
efeito do alvo de previsao vs efeito da politica.
Para reproduzir autonomia sem gates rigidamente fixos,
pesquisar uma **meta-politica probabilistica de acao**
que aprenda, somente no treino com previsoes OOF e
labels maturos, a vantagem estimada de
`manter / trocar / caixa`, condicionada ao estado
atual, custos, incerteza e regimes. Validar por
walk-forward purgado e posteriormente por janela nova.
Nao escolher pesos/hyperparametros usando o OOS ja visto.

**Governanca de fechamento:** esta secao e o
registro versionado do marco. Recomenda-se criar a
tag anotada local `research-v1.0.10-rotacoes-autonomas`
apontando ao commit deste registro e enviá-la
ao GitHub. Isso nao autoriza merge para `main`.


#### 25.12.11 v1.0.11 — ablação fatorial da utilidade Control

**Pergunta:** a maior precisão das rotações do Control decorre
predominantemente do seu sinal de utilidade ajustado ao risco,
da permanência obrigatória ou da margem de troca? A v1.0.10
testou modelos de retornos percentuais e regras matemáticas
diferentes ao mesmo tempo; portanto não isolou tais efeitos.

**Pré-registro:** o Control congelado H2 + margem calibrada
é o checkpoint obrigatório de US$ 587.165.941,13.
Nenhuma previsão LightGBM do Control será re-treinada
ou ajustada para as variantes. Cada fold reutiliza o MESMO
cache preditivo OOS de utilidade ajustada ao risco,
o MESMO universo de 75 ativos, custos, calendário,
configurações de risco e simulador.

**Fatorial 2 x 3** (seis cenários, sem varredura de
parâmetros a posteriori):
- Fator hold = 0 ou 2 pregões;
- Fator margem = zero (ambas as margens zeradas),
  base de configuração fixa, ou calibrada no histórico
  anterior ao teste, igual à escolha oficial do fold.

O cenário H2 + calibrada **reutiliza diretamente a
simulação do Control**; não cria uma segunda versão
potencialmente divergente do checkpoint. Para as outras
cinco combinações, o código produz novas decisões
pela própria `_politica_utilidade` usando exatamente
as mesmas previsões: NUNCA copia a ação final do Control
para executar a estratégia alternativa.

**Dois tipos de avaliação complementares:**
1. *Ablação pareada de uma decisão*: todas as combinações
   são reaplicadas à MESMA data, ao MESMO ativo atual
   e ao MESMO holding observado no Control; mede quantas
   ações mudariam ao retirar apenas a margem ou o bloqueio.
   O cenário oficial deve concordar 100% ou o experimento
   aborta. Os desfechos H05 das ações alternativas são
   auditados somente quando já maturados dentro do fold.
   Esta é uma comparação pontual; NÃO é backtest alternativo.
2. *Replay financeiro completo*: cada combinação possui
   sua própria sequência de ativos, contador de permanência,
   ordens simuladas e carteira; compara a trajetória real
   da política frente ao checkpoint, conservando fees,
   execução no próximo pregão e folds. Como o estado
   diverge ao longo do tempo, diferenças finais refletem
   efeitos acumulados, não somente um bloqueio isolado.

**Métricas predefinidas:** precisão de rotações H05
contra permanecer no ativo anterior (primária descritiva),
também H02/H03/H10, número de rotações, média do ganho
relativo realizado por troca, Sharpe, capital, MaxDD,
pior fold, custos e diferenças de ação no mesmo estado.
O aprendizado de ranking da v1.0.9 e as políticas de
retorno da v1.0.10 seguem como auditorias separadas,
não entram na escolha por utilidade.
Resultados OOS pré-examinados são exploratórios:
NENHUMA seleção/promocão automática; validação
confirmatória exige nova janela não explorada.

**Arquivos:** `output/persistencia_horizontes_u77_v111/`,
contendo `utility_ablation_financial.csv`,
`utility_ablation_quality_by_fold.csv`,
`utility_ablation_quality_by_decision.csv`,
`utility_ablation_policy_decisions.csv`,
`utility_ablation_one_step.csv`,
`utility_ablation.log`, seis curvas de capital e arquivos
de operações `utility_hold*_margin_*_equity.csv` e
`utility_hold*_margin_*_trades.csv`, todos os artefatos
de auditoria anteriores, `comparison.csv`,
`experiment.json`, `execution_full.log` e
`pacote_persistencia_horizontes_u77_v111.zip`.

**Correção do pacote:** o ZIP agora é criado SOMENTE
depois de gravar no log a conclusão e o aviso sonoro,
após fechar stdout/stderr, e validado integralmente.
O sinal sonoro continua diferenciado entre sucesso
e erro. Em erros, guardar `execution_full.log`
independentemente de haver ZIP.

**Limitações:** a função de utilidade Control contém
ganho/riscos futuros apenas NOS LABELS de treinamento,
não nas features OOS. Sua otimização de margens utiliza
histórico de calibração, não o OOS atual. O backtest
de capital composto não modela completamente liquidez,
impacto real e capacidade. O contrafactual de uma
única sessão H05 não mede o resultado de mudar toda
a trajetória. Os mesmos folds já foram examinados;
um vencedor histórico não é melhoria confirmada.

**Estado:** implementação versionada no ramo isolado;
execução completa Spyder com snapshot congelado pendente
de resultados. O Control e a main seguem intocados.
