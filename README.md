# TCC MBA USP — Rotação de Capital com Machine Learning

Este repositório contém a implementação reproduzível do TCC sobre rotação de
capital entre ativos financeiros usando LightGBM e validação temporal
walk-forward.

A execução oficial não depende de MongoDB nem do Market Cycle Trader. Os dados
utilizados para a reprodução ficam congelados em CSV e são validados por
manifestos e hashes antes da execução.

## Resultado oficial reproduzível

O universo financeiro oficial é o **U59**:

- 56 ativos do snapshot-base em dados/pesquisa/;
- mais COLB, AMS e FOXF, preservados em dados/pesquisa_expansao_76_b2/.

O capital inicial é de **US$ 10.000**. O checkpoint que o runner oficial deve
reproduzir é:

| Métrica | Referência |
| --- | ---: |
| Capital final | **US$ 30.080.091,01** |
| Universo | 59 ativos |
| Modelo | LightGBM Control |
| Período congelado | até 17 set. 2026 |
| Banco de dados | não utilizado |
| Download na reprodução | não realizado |

O resultado descreve um backtest histórico e não constitui previsão ou
garantia de desempenho futuro.

## Entradas principais

O repositório foi simplificado para dois runners de pesquisa no diretório
raiz:

~~~text
buscar_ativos.py
reproduzir_experimento.py
~~~

### buscar_ativos.py

É o runner de busca de candidatos. Ele usa a infraestrutura de pesquisa
preservada para avaliar candidatos em relação ao U59. A busca é uma atividade
de pesquisa separada da reprodução oficial e pode exigir credenciais da
Alpaca quando houver coleta de novos dados.

No terminal:

~~~bash
python buscar_ativos.py
~~~

No Spyder, abra buscar_ativos.py e execute com F5.

### reproduzir_experimento.py

É a reprodução oficial do resultado U59. O script:

1. valida os snapshots congelados;
2. carrega o U56;
3. acrescenta COLB, AMS e FOXF;
4. fixa o calendário temporal no U56 original;
5. executa a calibração walk-forward e o LightGBM Control;
6. simula o U59;
7. exige a reprodução de **US$ 30.080.091,008142874**;
8. exporta previsões, operações, margens por fold e o pacote de auditoria.

No terminal:

~~~bash
python reproduzir_experimento.py
~~~

No Spyder, abra reproduzir_experimento.py, reinicie o kernel e execute com F5.

A execução usa apenas os dados congelados e não acessa a Alpaca.

## Estrutura do projeto

~~~text
.
├── buscar_ativos.py
├── reproduzir_experimento.py
├── CONTEXTO_MESTRE.md
├── dados/
│   ├── pesquisa/
│   ├── pesquisa_expansao_76_b2/
│   ├── pesquisa_smart_candidates/
│   └── assinatura_matematica/
├── engine/
├── pesquisas/
├── reproducao/
├── tests/
├── requirements.txt
└── .env.example
~~~

Os antigos runners experimentais de análise, congelamento e validação foram
removidos do diretório raiz após seus resultados terem sido registrados. A
evidência científica correspondente continua preservada nos dados congelados,
no histórico Git e no CONTEXTO_MESTRE.md.


## Metodologia resumida

As séries diárias são preparadas a partir dos snapshots versionados. Splits são
normalizados em memória e problemas estruturais de identidade são tratados de
forma explícita conforme as regras congeladas da pesquisa.

O LightGBM estima utilidade multi-horizonte com horizontes de 5, 10, 20, 40 e
60 sessões. A validação é cronológica e utiliza folds walk-forward com
treinamento, calibração, purge temporal e teste fora da amostra.

As mudanças de posição são executadas na abertura da sessão seguinte. Todo o
capital pertence a uma única conta e é reinvestido ao longo da trajetória.

## Dados congelados

A reprodução U59 depende de dois conjuntos versionados:

~~~text
dados/pesquisa/
dados/pesquisa_expansao_76_b2/
~~~

O primeiro contém os 56 ativos-base. O segundo contém, entre outros ativos da
campanha histórica, as três inclusões necessárias para o U59:

~~~text
COLB
AMS
FOXF
~~~

A reprodução oficial não substitui esses arquivos por downloads atuais.

## Resultados gerados

A execução de reproduzir_experimento.py grava em:

~~~text
output/reproducao/
~~~

os principais artefatos:

~~~text
u59_assets.csv
u59_fold_margins.csv
u59_predictions.csv
u59_trades.csv
reproducao_u59.json
pacote_reproducao_u59_30m.zip
~~~

O arquivo reproducao_u59.json contém o universo, os snapshots, as margens
selecionadas por fold, as métricas e o erro de reprodução em relação ao
checkpoint oficial.

## Instalação

Crie um ambiente Python e instale:

~~~bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

Para a reprodução oficial não são necessárias credenciais da Alpaca.

Para buscar_ativos.py, quando houver coleta de dados, use um arquivo .env
local:

~~~text
ALPACA_API_KEY=sua_api_key
ALPACA_SECRET_KEY=sua_secret_key
~~~

Nunca adicione credenciais ao Git.

## Testes e análise estática

Execute:

~~~bash
python -m ruff check engine reproducao pesquisas buscar_ativos.py reproduzir_experimento.py tests --select F401,F811,F821,F841
python -m pytest -q
~~~

O GitHub Actions executa os mesmos checks nas branches de pesquisa.

## Histórico científico

Os experimentos anteriores, inclusive as análises de assinatura contextual,
validação prospectiva e filtro de aderência de capital, permanecem documentados
em CONTEXTO_MESTRE.md e no histórico Git.

Esses experimentos não são necessários para reproduzir o checkpoint U59 de
aproximadamente US$ 30 milhões.
