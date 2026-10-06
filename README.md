# TCC MBA USP — Rotação de Capital com Machine Learning

Este repositório contém a implementação do TCC sobre rotação de capital entre
ativos financeiros usando LightGBM e validação temporal walk-forward.

A execução oficial não depende de MongoDB nem do Market Cycle Trader. Ela usa
os snapshots CSV congelados da pesquisa e não baixa dados novos durante a
reprodução.

## Checkpoint oficial U67

O checkpoint oficial passa a ser o **U67**:

- U56 do snapshot-base;
- + COLB, AMS e FOXF, formando o U59;
- + THO, WDAY, EXR, XEL, SBFG, PAYX, MUX e SXC, formando o U67.

O capital inicial é de **US$ 10.000**. A execução oficial deve reproduzir:

| Métrica | Referência |
| --- | ---: |
| Capital final | **US$ 58.557.157,67** |
| CAGR | **309,4001%** |
| Sharpe | **2,51874371** |
| MaxDD | **-30,3591%** |
| Worst fold | **+282,5896%** |
| Universo | 67 ativos |
| Modelo | LightGBM Control |
| Período congelado | até 17 set. 2026 |
| Banco de dados | não utilizado |
| Download na reprodução | não realizado |

O U67 terminou 94,6708% acima do checkpoint U59 de US$ 30.080.091,01. O
resultado é um backtest histórico e não constitui previsão ou garantia de
desempenho futuro.

## Dois runners ativos

O diretório raiz mantém somente os dois runners de pesquisa e reprodução:

~~~text
buscar_ativos.py
reproduzir_experimento.py
~~~

### reproduzir_experimento.py

É o runner oficial do checkpoint U67. O script:

1. valida os snapshots congelados;
2. carrega o U56;
3. acrescenta COLB, AMS e FOXF para formar o U59;
4. acrescenta THO, WDAY, EXR, XEL, SBFG, PAYX, MUX e SXC para formar o U67;
5. fixa o calendário temporal no U56 original;
6. executa calibração walk-forward e LightGBM Control;
7. simula o U67;
8. exige a reprodução de **US$ 58.557.157,67496595**;
9. exporta previsões, operações, margens por fold e pacote de auditoria.

No terminal:

~~~bash
python reproduzir_experimento.py
~~~

No Spyder, abra o arquivo, reinicie o kernel e execute com F5.

A execução oficial usa os seguintes snapshots congelados presentes no ambiente
de pesquisa:

~~~text
dados/pesquisa/
dados/pesquisa_expansao_76_b2/
dados/pesquisa_smart_candidates/
~~~

O runner não substitui esses dados por downloads atuais. Os snapshots B2 e
SMART precisam estar disponíveis no diretório de trabalho usado para a
reprodução.

### buscar_ativos.py

É o runner preservado para pesquisa de candidatos. Ele permanece separado da
reprodução oficial. Pode acessar a Alpaca quando a campanha exigir coleta de
novos dados, mas essa coleta nunca é executada por
`reproduzir_experimento.py`.

## Estrutura atual

~~~text
.
├── buscar_ativos.py
├── reproduzir_experimento.py
├── CONTEXTO_MESTRE.md
├── dados/
│   └── pesquisa/
├── engine/
├── pesquisas/
├── reproducao/
├── tests/
├── requirements.txt
└── .env.example
~~~

Os runners e arquivos de trabalho das pesquisas de assinatura que não
generalizaram foram removidos da árvore atual. Os resultados negativos e a
sequência dos experimentos continuam preservados no histórico Git e no
`CONTEXTO_MESTRE.md`.

## Metodologia resumida

As séries diárias são preparadas a partir dos snapshots congelados. Splits são
normalizados em memória e problemas estruturais de identidade são tratados de
forma explícita conforme as regras da pesquisa.

O LightGBM estima utilidade multi-horizonte com horizontes de 5, 10, 20, 40 e
60 sessões. A validação é cronológica e usa folds walk-forward com
treinamento, calibração, purge temporal e teste fora da amostra.

As mudanças de posição são executadas na abertura da sessão seguinte. Todo o
capital pertence a uma única conta e é reinvestido ao longo da trajetória.

## Artefatos da reprodução

A execução grava em:

~~~text
output/reproducao/
~~~

os principais arquivos:

~~~text
u67_assets.csv
u67_fold_margins.csv
u67_predictions.csv
u67_trades.csv
reproducao_u67.json
pacote_reproducao_u67_58m.zip
~~~

O arquivo `reproducao_u67.json` registra o universo, a identidade dos
snapshots, as margens por fold, as métricas e o erro de reprodução em relação
ao checkpoint oficial.

## Instalação

~~~bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

Para a reprodução oficial não são necessárias credenciais da Alpaca.

Para `buscar_ativos.py`, quando houver coleta de dados, use um arquivo
`.env` local:

~~~text
ALPACA_API_KEY=sua_api_key
ALPACA_SECRET_KEY=sua_secret_key
~~~

Nunca adicione credenciais ao Git.

## Testes e análise estática

~~~bash
python -m ruff check engine reproducao pesquisas buscar_ativos.py reproduzir_experimento.py tests --select F401,F811,F821,F841
python -m pytest -q
~~~

O GitHub Actions executa os mesmos checks nas branches de pesquisa e release.
