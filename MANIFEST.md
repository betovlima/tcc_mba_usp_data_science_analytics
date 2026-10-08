# MANIFEST — TCC MBA USP Data Science & Analytics

## 1. Finalidade deste documento

Este arquivo consolida em um único lugar o histórico técnico, as decisões
metodológicas, a arquitetura final, os resultados, as refatorações, os testes e
a avaliação crítica do projeto até o estado científico atualmente consolidado
na `main`.

Baseline científico documentado:

```text
versão: 1.22.0-dev.8
main de referência: 4655e2ea54ba870d23ebf67ffb0bb3f30bb2180f
data da consolidação: 2026-10-08
```

Esta consolidação é documental. Ela não altera parâmetros científicos, dados,
folds, regras de rotação ou resultados do experimento.

---

## 2. Objetivo científico

O projeto avalia uma estratégia de rotação de capital entre ativos financeiros
usando aprendizado de máquina supervisionado com LightGBM.

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

O universo oficial de entrada contém 67 ativos.

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

### 14.7 Expansão do universo

A pesquisa avançou por checkpoints intermediários, incluindo U59 e depois U67.

Esses checkpoints foram úteis para testar a estabilidade do motor e o impacto
do universo, mas foram posteriormente substituídos pelo contrato final U67
consolidado no código atual.

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
python -m ruff check engine reproducao preparar_snapshot_u67.py reproduzir_experimento.py tests --select F401,F811,F821,F841
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

**Risco de seleção de universo.** O U67 foi alcançado depois de várias etapas de
pesquisa e expansão do universo. Isso pode introduzir viés de seleção ou
data-snooping quando o desempenho final é interpretado fora do contexto do
processo de pesquisa.

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

**Snapshot remoto.** No estado da `main` usado para esta consolidação, a
árvore do GitHub contém a documentação da pasta `dados/`, mas não contém os
CSVs de `dados/u67/`. Portanto, a reprodução a partir de um clone limpo ainda
depende de gerar ou adicionar o snapshot U67. Para uma entrega acadêmica
autossuficiente, esse snapshot deve ser versionado ou distribuído por um meio
oficial com hash verificável.

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

A principal pendência para uma reprodução acadêmica completamente
autossuficiente é garantir que o snapshot U67 esteja disponível junto da
entrega final ou em repositório de dados associado.

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
versão científica        = 1.22.0-dev.8
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
Ruff                     = aprovado
pytest                   = 34 aprovados
```

A versão científica não foi alterada por esta consolidação documental.
