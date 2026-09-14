# USP MBA Data Science & Analytics — Capstone Project

Projeto de conclusão do MBA em Data Science & Analytics da USP focado na construção, documentação e validação reproduzível de um backtest de estratégia quantitativa de mercado.

## Regra de documentação

Este `README.md` é o único README do projeto. Toda evolução relevante, arquitetura, forma de execução, requisitos, decisões metodológicas e histórico de mudanças devem ser mantidos neste arquivo. Não criar READMEs adicionais por versão, pasta ou experimento.

## Objetivo

O objetivo deste repositório é reproduzir e documentar academicamente um backtest já obtido no projeto Market Cycle Trader. Não existe neste projeto busca de novos ativos, otimização aberta do universo, CARO, tuning de hiperparâmetros ou tentativa de descobrir uma estratégia melhor.

A pergunta aqui é simples:

> Dadas uma configuração, um universo de ativos, um snapshot histórico e um código de backtest congelados, conseguimos repetir o mesmo resultado de forma auditável?

## Backtest de referência

A reprodução foi congelada a partir do experimento histórico de poda exata do Market Cycle Trader.

- código-fonte histórico: `betovlima/market_cycle_trader_api`
- commit congelado: `17019d95bfce6f0fbcd153e097b1968d9cfce1ca`
- Strategy: `#10`
- Strategy ID: `strategy-87713a05860748719ec18d0a086dcce7`
- revisão observada na certificação: `15`
- SHA-256 da configuração: `509b940659a89a7348be3690882213c839ce1a43b7e44057656074f5b2517a6e`
- início do histórico: `2016-01-01`
- fim do snapshot: `2026-09-04`
- capital inicial: `US$ 10.000,00`
- capital final de referência: `US$ 43.759.854,82`
- CAGR aproximado: `293,82%`
- Sharpe aproximado: `2,557`
- Max Drawdown aproximado: `-28,19%`

O resultado de referência foi reexecutado antes da criação deste repositório. O universo inicial de 42 ativos terminou com aproximadamente `US$ 33,44 milhões`; após a poda histórica, o universo final de 37 ativos terminou com aproximadamente `US$ 43,76 milhões`.

A ordem efetivamente observada na nova reprodução da poda foi `AAPL`, `GOOGL`, `AMZN`, `CRM`, `MAN`. Essa busca já terminou e **não faz parte da execução deste projeto**.

## Universo congelado — 37 ativos

```text
NVDA, MSFT, META, TSLA, AMD, JPM, SPY, AVGO, NFLX,
ORCL, COST, LLY, XOM, CAT, WMT, V, HD, ADC, ADEA,
ADI, ADM, GKOS, VNCE, CORT, UNFI, DNN, MKSI, APD,
DDS, RACE, UNF, TX, CEF, YANG, KKR, BXMT, SCSC
```

`backtest.py` usa essa lista fixa. Nenhum ativo é adicionado ou removido durante a execução.

## Ressalva acadêmica importante

O motor do backtest usa validação walk-forward para treinamento e teste dos modelos ao longo do tempo. Porém o **universo final de 37 ativos foi escolhido retrospectivamente usando o período histórico completo até 04/09/2026**.

Portanto, `US$ 43,76 milhões` não deve ser apresentado como evidência de que a seleção do universo generaliza fora da amostra. Neste trabalho ele é tratado como um **resultado histórico congelado cuja reprodução computacional será estudada e auditada**.

Essa distinção deve ser preservada no texto final do trabalho.

## Onde existe Machine Learning

O backtest de referência utiliza **LightGBM Utility**. Em cada fold do walk-forward, os modelos são treinados somente com a janela permitida pelo protocolo daquele fold. O modelo estima uma utilidade de mercado construída a partir de múltiplos horizontes e features técnicas.

Neste projeto não existe uma nova etapa de Machine Learning. O projeto apenas repete o treinamento previsto pelo próprio backtest congelado:

```text
dados históricos do fold
        ↓
features da estratégia
        ↓
LightGBM Utility
        ↓
política de rotação
        ↓
execução no próximo open
        ↓
capital e métricas
```

Não existe treinamento para selecionar os 37 ativos. Não existe novo classificador, nova rede neural ou novo target criado para o TCC.

## Onde não existe Machine Learning

São determinísticos ou diretamente calculados pelo motor:

- construção das janelas walk-forward;
- separação temporal de treino, calibração, purge e teste;
- regras da política de rotação;
- aplicação de slippage e custos;
- compra, venda, rotação e liquidação final;
- contabilidade do capital;
- CAGR;
- Sharpe;
- Max Drawdown;
- número de rotações;
- comparação do capital final com a referência;
- fingerprints SHA-256 da configuração, dados e request de execução.

O Machine Learning propõe scores dentro do protocolo da estratégia; o replay determina o resultado econômico.

## Tecnologias

- **Python**: implementação e execução do experimento;
- **Spyder**: ambiente principal para execução interativa do projeto;
- **pandas / NumPy**: séries temporais, features e cálculos numéricos;
- **LightGBM**: modelo de regressão usado pela estratégia histórica;
- **SciPy / scikit-learn**: dependências científicas da linhagem do motor;
- **exchange-calendars**: calendário XNYS e sessões de pregão;
- **MongoDB local / PyMongo**: Strategy e candles históricos congelados;
- **SHA-256**: fingerprints de configuração, dados e request;
- **Git**: congelamento do código histórico exato usado na referência.

FastAPI, frontend, autenticação, administração e componentes de produção do Market Cycle Trader não fazem parte do fluxo acadêmico.

## Estrutura do projeto

```text
tcc_mba_usp_data_science_analytics/
├── README.md
├── requirements.txt
├── .gitignore
└── backtest.py
```

Durante a execução podem surgir duas pastas locais ignoradas pelo Git:

```text
.mct_reference/        # checkout técnico do commit histórico, somente se necessário
output/                # artefatos gerados pelo backtest
```

A documentação continua exclusivamente neste `README.md`.

## Instalação

Clone o projeto:

```bat
git clone https://github.com/betovlima/tcc_mba_usp_data_science_analytics.git
cd tcc_mba_usp_data_science_analytics
```

Instale as dependências usando o mesmo interpretador Python configurado no Spyder:

```bat
python -m pip install -r requirements.txt
```

Para confirmar qual Python o Spyder está usando:

```python
import sys
print(sys.executable)
print(sys.version)
```

## MongoDB local

Por padrão o projeto usa:

```text
URI:      mongodb://localhost:27017
Database: extrema_backtest
```

O banco precisa conter a Strategy #10 e o histórico de mercado usado na reprodução certificada.

Opcionalmente, crie um arquivo local `.env` na raiz do projeto, nunca commitado:

```text
TCC_MONGO_URI=mongodb://localhost:27017
TCC_MONGO_DATABASE=extrema_backtest
```

## Código histórico congelado

O `backtest.py` só aceita o commit:

```text
17019d95bfce6f0fbcd153e097b1968d9cfce1ca
```

A execução procura primeiro um checkout local já existente desse commit. O worktree histórico `mct_exact_pruning_validation`, se estiver acessível na árvore de diretórios, pode ser reutilizado.

Se necessário é possível indicar explicitamente o checkout no `.env`:

```text
MCT_REFERENCE_SOURCE=C:\caminho\para\mct_exact_pruning_validation
```

Se nenhum checkout do commit exato for encontrado, `backtest.py` tenta criar `.mct_reference` e fazer checkout detached do commit congelado. Esse bootstrap é a única situação em que Git/internet podem ser necessários. O backtest e os candles continuam sendo executados a partir do MongoDB local.

O script recusa silenciosamente usar outra revisão do código: o `HEAD` encontrado precisa ser exatamente igual ao commit de referência.

## Execução no Spyder

O único entrypoint é:

```text
backtest.py
```

Abra esse arquivo no Spyder e pressione:

```text
F5
```

Não é necessário chamar scripts auxiliares manualmente.

Também é possível executar pelo terminal:

```bat
python backtest.py
```

## O que `backtest.py` valida antes da execução

Antes de iniciar o replay, o script verifica:

- commit exato do código histórico;
- conexão com MongoDB local;
- ID da Strategy #10;
- SHA-256 da configuração da Strategy;
- data inicial da Strategy;
- presença dos 37 ativos congelados na Strategy;
- histórico local de cada ativo até `2026-09-04`.

Depois calcula e registra:

- SHA-256 dos candles OHLCV efetivamente usados;
- SHA-256 do request de execução;
- versões das bibliotecas Python;
- caminho e commit do código histórico;
- métricas econômicas do replay;
- quantidade e intervalo das sessões de decisão;
- tempo de execução.

## Critério de certificação

O capital final é o critério primário de igualdade econômica.

Referência:

```text
US$ 43.759.854,82
```

A tolerância relativa configurada é:

```text
1e-6
```

Ou seja, diferenças numéricas minúsculas de runtime podem ser aceitas, mas uma execução economicamente diferente deve falhar.

O log termina com:

```text
Certification  : PASS
```

ou:

```text
Certification  : FAIL
```

Uma execução `FAIL` não deve ser utilizada como resultado do TCC antes de reconciliar código, configuração e dados.

## Artefato da execução

Cada execução grava um único artefato estável:

```text
output/backtest_result.json
```

Esse arquivo contém referência congelada, fingerprints do ambiente e dados, métricas obtidas e status da certificação. A pasta `output/` não é versionada no Git.

## Princípios de reprodutibilidade

- um único entrypoint para execução;
- um único README para toda a documentação;
- nenhuma busca de ativos durante o backtest;
- nenhum tuning automático durante a reprodução;
- código histórico pinado por commit;
- Strategy validada por hash;
- snapshot de mercado limitado a `2026-09-04`;
- dados de entrada fingerprintados por SHA-256;
- request efetivo fingerprintado por SHA-256;
- resultado comparado automaticamente com o capital certificado;
- segredos e artefatos locais fora do Git.

## Métricas principais

A execução reporta, entre outras métricas:

- capital inicial e final;
- retorno acumulado;
- CAGR — taxa de crescimento anual composta;
- Sharpe — retorno ajustado ao risco;
- Max Drawdown — maior perda acumulada entre topo e fundo;
- número de rotações;
- dias em CASH;
- pior fold;
- quantidade de sessões de decisão;
- tempo total de replay.

## Status atual

O harness reproduzível está implementado em `backtest.py` e pinado ao código histórico responsável pelo resultado certificado. O próximo passo operacional é executar o arquivo no ambiente local/Spyder e verificar `Certification : PASS`. Se houver divergência, os hashes gravados pelo próprio programa servem para localizar exatamente se a diferença veio do código, Strategy, request, dados ou ambiente numérico.

## Histórico do projeto

### 2026-09-14 — Inicialização

- criado o repositório do MBA Capstone Project;
- definido o foco em reprodução do backtest, sem nova pesquisa de Asset Discovery;
- adotado um único `README.md` como documento vivo;
- criado `requirements.txt` independente da stack web do Market Cycle Trader;
- criado `backtest.py` como entrypoint estável para Spyder.

### 2026-09-14 — Backtest certificado congelado

- congelado o commit histórico `17019d95bfce6f0fbcd153e097b1968d9cfce1ca`;
- congelada a Strategy #10 e o SHA-256 de sua configuração;
- congelado o snapshot final em `2026-09-04`;
- congelado o universo final de 37 ativos;
- definido `US$ 43.759.854,82` como capital final de referência;
- implementada leitura exclusivamente do histórico de mercado no MongoDB local;
- implementados fingerprints SHA-256 da configuração, candles e execution request;
- implementada comparação automática do capital reproduzido com a referência;
- implementado `output/backtest_result.json` como artefato estável de auditoria;
- documentada a ressalva acadêmica de que a escolha dos 37 ativos é retrospectiva e não constitui validação OOS da seleção do universo.
