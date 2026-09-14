# USP MBA Data Science & Analytics — Capstone Project

Projeto de conclusão do MBA em Data Science & Analytics da USP, focado na construção, documentação e validação reproduzível de um backtest de estratégia quantitativa de mercado.

## Regra de documentação

Este `README.md` é o único README do projeto. Toda evolução relevante, arquitetura, forma de execução, requisitos, decisões metodológicas e histórico de mudanças devem ser mantidos neste arquivo. Não criar READMEs adicionais por versão, pasta ou experimento.

## Objetivo

O objetivo deste repositório não é continuar uma busca aberta por novos ativos ou novas estratégias. O foco é reconstruir de forma independente, auditável e reproduzível o backtest já desenvolvido no projeto Market Cycle Trader, preservando as hipóteses, os dados de entrada, a configuração da estratégia, a lógica de walk-forward e as métricas necessárias para permitir repetição e análise acadêmica.

## Princípios de reprodutibilidade

- O backtest deve poder ser executado a partir de um único entrypoint no Spyder.
- Dependências Python ficam declaradas em `requirements.txt`.
- Segredos, tokens e credenciais não entram no Git.
- Toda configuração que afete o resultado deve ser explícita e versionada.
- Datas de treino, teste e corte devem ser explícitas para evitar look-ahead.
- Seeds e limites de threads devem ser fixados quando necessários para execução determinística.
- O resultado reproduzido deve registrar capital final, retorno, CAGR, Sharpe, drawdown máximo e métricas por período/fold quando aplicável.
- Alterações metodológicas relevantes devem ser registradas na seção **Histórico do projeto** deste mesmo README.

## Estrutura inicial

```text
tcc_mba_usp_data_science_analytics/
├── README.md
├── requirements.txt
├── .gitignore
└── backtest.py
```

A estrutura deve permanecer pequena. Novos módulos só devem ser criados quando houver ganho claro de organização ou testabilidade; a documentação continua centralizada neste README.

## Ambiente recomendado

- Python 3.13+
- Spyder
- MongoDB local, enquanto os dados históricos usados pelo backtest ainda forem carregados a partir da base local

O Spyder é apenas o ambiente de desenvolvimento e execução e, por isso, não é instalado pelo `requirements.txt` do projeto.

## Instalação

Clone o projeto e entre na pasta:

```bat
git clone https://github.com/betovlima/tcc_mba_usp_data_science_analytics.git
cd tcc_mba_usp_data_science_analytics
```

Instale as dependências no mesmo interpretador Python usado pelo Spyder:

```bat
python -m pip install -r requirements.txt
```

No Spyder, confirme o interpretador em:

```python
import sys
print(sys.executable)
print(sys.version)
```

## Execução no Spyder

O entrypoint estável do projeto é:

```text
backtest.py
```

A intenção é que a execução normal seja feita abrindo esse arquivo no Spyder e pressionando `F5`, sem depender de vários scripts manuais ou comandos auxiliares.

Enquanto a migração do motor do backtest estiver em andamento, o arquivo executa apenas a validação do ambiente. A próxima etapa do projeto é migrar o backtest certificado para esse entrypoint sem carregar dependências da API web do Market Cycle Trader.

## Escopo da migração do backtest

A migração deverá preservar, de forma explícita:

- universo de ativos usado no backtest de referência;
- intervalo histórico;
- configuração completa da estratégia;
- construção das features;
- modelo utilizado pela estratégia;
- walk-forward e separação treino/teste;
- custos, slippage e regras de rotação;
- política de decisão;
- cálculo de capital e métricas;
- seeds e parâmetros de determinismo;
- identificação/hash da configuração de referência;
- comparação automática do resultado reproduzido com o resultado certificado.

O código do projeto acadêmico deve conter apenas o necessário para reproduzir e analisar o backtest. Componentes da API, frontend, autenticação, administração e execução em produção do Market Cycle Trader não fazem parte deste repositório.

## Dados

Os dados usados para o trabalho não devem depender de arquivos temporários ou resultados manuais não rastreáveis. A fonte de dados e o procedimento de extração/congelamento do dataset serão registrados aqui quando a migração do backtest for concluída.

Arquivos locais de dados, dumps, bancos, credenciais e artefatos grandes não devem ser commitados sem decisão explícita sobre licenciamento e reprodutibilidade.

## Métricas principais

A reprodução deve reportar, no mínimo:

- capital inicial e final;
- retorno acumulado;
- CAGR — taxa de crescimento anual composta;
- Sharpe — retorno ajustado ao risco;
- Max Drawdown — maior perda acumulada entre topo e fundo;
- número de decisões/rotações;
- desempenho por fold ou período fora da amostra quando aplicável.

## Status atual

Estrutura inicial criada. O motor do backtest ainda será migrado e validado contra o resultado de referência antes de qualquer análise acadêmica final.

## Histórico do projeto

### 2026-09-14 — Inicialização

- criado o repositório do MBA Capstone Project;
- definido o foco em reprodução do backtest, não em nova busca de ativos;
- adotado um único `README.md` como documento vivo do projeto;
- definido `backtest.py` como entrypoint estável para Spyder;
- criado `requirements.txt` independente da stack FastAPI/Frontend do Market Cycle Trader;
- estabelecidos princípios de determinismo, isolamento temporal e validação contra um resultado certificado.
