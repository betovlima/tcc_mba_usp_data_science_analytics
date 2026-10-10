# MANIFEST — TCC MBA USP Data Science & Analytics

## 1. Finalidade deste documento

Este arquivo consolida em um único lugar o histórico técnico, as decisões
metodológicas, a arquitetura final, os resultados, as refatorações, os testes e
a avaliação crítica do projeto no contrato científico descrito abaixo.

Baseline científico documentado:

```text
versão: 1.22.0-dev.10
main de referência para a correção: 262e5388354f258de2480151a0b9f1b9ff147db7
branch de trabalho: fix/u67-eligibilidade-sem-futuro
data da atualização: 2026-10-10
```

Esta revisão remove da seleção de candidatos a consulta aos preços da sessão
seguinte. Incorpora também a correção de compra e manutenção já validada na
branch `fix/u67-buy-hold-same-universe`, pois essa revisão ainda não estava na
`main`. A seção 20 preserva sua validação histórica. A seção 21 documenta a
medição da elegibilidade, os diagnósticos de teste e os dados congelados
incluídos nesta branch. A `main` não foi alterada.

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

Na execução da `main` correspondente ao baseline deste manifesto:

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
python -m ruff check engine reproducao preparar_snapshot_u67.py reproduzir_experimento.py avaliar_elegibilidade_u67.py tests --select F401,F811,F821,F841
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

**Dados disponíveis nesta branch.** A `main` de origem não continha os CSVs.
Esta revisão inclui os 134 CSVs e o manifesto original, preservando seus bytes
e hashes. `.gitattributes` impede a conversão automática das quebras de linha
do snapshot entre sistemas operacionais. A inclusão resolve a ausência do
snapshot nesta branch; não elimina as demais limitações metodológicas.

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

O snapshot está incluído nesta branch. As próximas verificações científicas
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
versão científica        = 1.22.0-dev.10
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
pytest local             = ver seção 21
```

A versão dev.9 corrigiu a referência financeira. A versão dev.10 corrige a
elegibilidade e registra diagnósticos fora da amostra.

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
