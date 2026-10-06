# TCC MBA USP — Rotação de Capital com Machine Learning

Este repositório contém a implementação reproduzível do TCC sobre rotação de
capital entre ativos financeiros usando LightGBM e validação temporal
walk-forward.

A execução oficial não depende de MongoDB nem do Market Cycle Trader. Os dados
utilizados para a reprodução ficam congelados em CSV e são validados por
manifestos e hashes antes da execução.

## Baseline de trabalho atual

A partir desta etapa, o baseline usado nas análises e visualizações é o
**U67**, formado por:

- U59 = U56 + COLB + AMS + FOXF;
- mais THO, WDAY, EXR, XEL, SBFG, PAYX, MUX e SXC.

O capital inicial é de **US$ 10.000** e o checkpoint de trabalho é:

| Métrica | Referência |
| --- | ---: |
| Capital final | **US$ 58.557.157,67** |
| Universo | 67 ativos |
| Modelo | LightGBM Control |
| Período congelado | até 17 set. 2026 |
| Banco de dados | não utilizado |
| Download na reprodução | não realizado |

O U59 de **US$ 30.080.091,01** permanece preservado como checkpoint histórico
confirmatório. O U67 passa a ser a referência prática das próximas análises,
mas deve continuar identificado no TCC como resultado exploratório, pois os
oito ativos adicionais foram escolhidos depois da observação de seus efeitos
financeiros individuais.

Os resultados descrevem backtests históricos e não constituem previsão ou
garantia de desempenho futuro.

## Entradas principais

O repositório foi simplificado para dois runners de pesquisa no diretório
raiz:

~~~text
buscar_ativos.py
reproduzir_experimento.py
~~~

### buscar_ativos.py

É o runner de busca de candidatos. A próxima evolução da busca deve usar o
U67 como referência de trabalho, preservando o U59 como checkpoint histórico. A busca é uma atividade
de pesquisa separada da reprodução oficial e pode exigir credenciais da
Alpaca quando houver coleta de novos dados.

No terminal:

~~~bash
python buscar_ativos.py
~~~

No Spyder, abra buscar_ativos.py e execute com F5.

### reproduzir_experimento.py

É o runner do baseline de trabalho U67. O script:

1. valida os snapshots disponíveis;
2. carrega o U56;
3. acrescenta COLB, AMS e FOXF para reconstruir o U59;
4. acrescenta THO, WDAY, EXR, XEL, SBFG, PAYX, MUX e SXC;
5. fixa o calendário temporal no U56 original;
6. executa a calibração walk-forward e o LightGBM Control;
7. simula o U67;
8. exige a reprodução de **US$ 58.557.157,67496595**;
9. exporta previsões, operações, margens e os gráficos de rotação.

No terminal:

~~~bash
python reproduzir_experimento.py
~~~

No Spyder, abra reproduzir_experimento.py, reinicie o kernel e execute com F5.

A execução não acessa a Alpaca. O U67 depende também do snapshot local
dados/pesquisa_smart_candidates, que contém os oito ativos adicionais.

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

O U67 usa:

~~~text
dados/pesquisa/
dados/pesquisa_expansao_76_b2/
dados/pesquisa_smart_candidates/
~~~

Os dois primeiros reconstroem o U59. O terceiro fornece os oito ativos
adicionais do baseline de trabalho:

~~~text
THO
WDAY
EXR
XEL
SBFG
PAYX
MUX
SXC
~~~

O runner não substitui esses arquivos por downloads atuais.

## Resultados gerados

A execução de reproduzir_experimento.py grava em:

~~~text
output/reproducao/
~~~

os principais artefatos:

~~~text
u67_assets.csv
u67_fold_margins.csv
u67_predictions.csv
u67_trades.csv
reproducao_u67.json
pacote_reproducao_u67_58m.zip
~~~

O arquivo reproducao_u67.json contém o universo, os snapshots, as margens
selecionadas por fold, as métricas, o status científico do baseline e o erro
de reprodução em relação ao checkpoint de US$ 58,56 milhões.

## Instalação

Crie um ambiente Python e instale:

~~~bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

Para a reprodução do baseline U67 não são necessárias credenciais da Alpaca,
desde que o snapshot local dos oito ativos esteja presente.

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

O U59 de aproximadamente US$ 30 milhões permanece como checkpoint histórico.
As análises correntes passam a usar o U67 de aproximadamente US$ 58,56 milhões
como baseline de trabalho.
