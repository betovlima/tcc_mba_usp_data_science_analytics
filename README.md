# TCC MBA USP — Rotação de Capital com Machine Learning

Release: **v1.22.1**, integrada à `main` a partir de
`fix/u67-eligibilidade-sem-futuro`. A memória da release, as evidências e as
pendências estão na seção 24 de [MANIFEST.md](MANIFEST.md).

## Requisitos

- Python 3.12
- dependências de `requirements.txt`

Instale:

```bash
python -m pip install -r requirements.txt
```

## 1. Verifique os dados

A reprodução usa:

```text
dados/u67/
├── raw_bars/
├── corporate_actions/
└── manifest.json
```

A release inclui o snapshot congelado em `dados/u67/`. Vá direto para a
execução; os hashes são conferidos antes do treinamento.

Se o snapshot ainda não existir, crie um arquivo `.env` local com:

```text
ALPACA_API_KEY=sua_api_key
ALPACA_SECRET_KEY=sua_secret_key
```

e execute:

```bash
python preparar_snapshot_u67.py
```

## 2. Execute a reprodução

No terminal:

```bash
python reproduzir_experimento.py
```

No Spyder, abra `reproduzir_experimento.py`, reinicie o kernel e execute o
arquivo completo.

## 3. Resultados

Os artefatos são gravados em:

```text
output/reproducao/
```

## 4. Confira o efeito da correção de elegibilidade

No terminal:

```bash
python avaliar_elegibilidade_u67.py
```

No Spyder, abra `avaliar_elegibilidade_u67.py` e execute o arquivo completo
com F5. O script compara o filtro antigo e o corrigido com os mesmos modelos
ajustados, sem baixar dados ou alterar os parâmetros. Os resultados ficam em
`output/elegibilidade/`. O controle com o filtro antigo serve à auditoria;
`reproduzir_experimento.py` usa a regra corrigida.

## 5. Gere as figuras do TCC

No terminal:

```bash
python gerar_figuras_tcc.py
```

No console do Spyder, com a pasta do projeto como diretório de trabalho:

```python
%run gerar_figuras_tcc.py
```

As sete figuras em PNG e SVG, as tabelas derivadas e sua procedência ficam em
`figuras/`. O comando usa as evidências preservadas da execução dev.10; não
treina modelos nem executa outro backtest. Ele é separado da reprodução.

A reprodução usa os hiperparâmetros fixados pela campanha histórica e calibra
a margem de troca por busca em grade em cada janela temporal. Ela não repete a
otimização bayesiana que selecionou os hiperparâmetros.

A documentação técnica, histórica e a avaliação completa do projeto estão em
`MANIFEST.md`.


## 6. Pesquisa de expansão incremental U67+n

A pesquisa que mantém o U67 fixo e testa somente a adição de novos ativos está
em:

```text
pesquisar_expansao_u67_spyder.py
```

A versão 1.4.1 usa por padrão o modo `decision_validation`. Ela preserva
os checkpoints da assurance e mede qualidade de ranking/decisão, Deflated
Sharpe Ratio, Probability of Backtest Overfitting e um piloto temporal de
`ADICIONAR / MANTER / REMOVER` ativos. Os modos `assurance_validation`,
`subset_optimization` e `direct_marginal` permanecem disponíveis para
reproduzir as etapas anteriores.

Ela exige as credenciais locais da Alpaca no arquivo `.env`, mas não recria nem
altera `dados/u67`. Os dados dos candidatos são mantidos em cache local
ignorado pelo Git.

No terminal:

```bash
python pesquisar_expansao_u67_spyder.py
```

No Spyder, abra `pesquisar_expansao_u67_spyder.py`, reinicie o kernel e execute
o arquivo completo com F5.

No modo padrão de validação da decisão, os resultados ficam em:

```text
output/u67_decision_validation/
```

Os modos anteriores continuam usando `output/u67_assurance_validation/`,
`output/u67_subset_optimization/` e `output/u67_asset_expansion/`.

O desenho e as limitações da campanha estão documentados em
`PESQUISA_EXPANSAO_U67.md`.

Se uma execução v1.4.0 já concluiu os cálculos e falhou somente ao serializar
`NaN` no JSON final, não repita o backtest. Preserve a pasta de output e rode:

```bash
python finalizar_u67_decision_validation_spyder.py
```

No Spyder, abra esse arquivo e execute com F5.


## Pesquisa experimental U67+10 — Persistência multi-horizonte

Esta seção descreve uma **branch de pesquisa**, não uma atualização da
reprodução oficial v1.22.1. A política congelada não é alterada.

Na branch research/horizon-persistence-oos-v1, o arquivo
pesquisar_persistencia_horizontes_spyder.py compara:

- Control U67+10 com permanência mínima de 2 sessões;
- Control U67+10 com permanência mínima de 0 sessões;
- política nova baseada na revisão das previsões H05/H10/H20/H40/H60
  e no erro medido antes de cada fold fora da amostra.

Requer os dados U67 versionados e os 10 históricos candidatos já baixados
pela pesquisa U67+n na máquina local, em:

dados/.u67_expansion_cache/full_raw_20261006/raw_bars/
dados/.u67_expansion_cache/full_raw_20261006/corporate_actions/

Os seguintes símbolos precisam estar presentes nos dois diretórios:
CRT, NBR, CAPR, PNBK, PENN, JBL, STAG, MCHB, LAD, TOWN.

O script **não consulta Alpaca**; se esses arquivos faltarem, reporta os
nomes ausentes. Reinicie o kernel do Spyder, abra
pesquisar_persistencia_horizontes_spyder.py e execute com F5; ou use:

    python pesquisar_persistencia_horizontes_spyder.py

Arquivos gerados apenas após uma execução completa em:
output/persistencia_horizontes_u77/

Leia a seção 25.12 de MANIFEST.md para critérios de causalidade,
limitações da confiança heurística e interpretação de resultados.


### Versão experimental 1.0.3 — antecipação sem veto adicional

Use a mesma branch e o mesmo arquivo
pesquisar_persistencia_horizontes_spyder.py.
A versão 1.0.3 substitui, somente para esta nova campanha,
a heurística que bloqueava rotações do Control por uma política
**Control + antecipações de troca por evidência multi-horizonte**.

O script continua exigindo os dez históricos externos em cache
e não consulta a corretora. Execute com F5 no Spyder depois
de atualizar a branch e reiniciar o kernel.

O pacote novo é separado daquele produzido pela versão 1.0.2:

    output/persistencia_horizontes_u77_v103/pacote_persistencia_horizontes_u77_v103.zip

A campanha deve reproduzir o checkpoint Control ao centavo antes
de avaliar a nova hipótese. Veja a seção 25.12.3 de MANIFEST.md.


### Versão experimental 1.0.4 — permanência de 1 a 4 pregões

A partir desta atualização, o mesmo arquivo
`pesquisar_persistencia_horizontes_spyder.py` executa por padrão a
**v1.0.4**; a versão v1.0.3 permanece reproduzível pelo commit histórico.
A v1.0.4 compara holdings fixos 1, 2, 3 e 4 e um holding selecionado
exclusivamente pela calibração anterior a cada fold. Inclui holding=0
como ablação histórica. Não treina modelos adicionais por horizonte
nem faz download da Alpaca. Exige os mesmos dez históricos em cache.

No terminal do Git, a partir da pasta do repositório:

    git fetch origin
    git switch research/horizon-persistence-oos-v1
    git pull --ff-only origin research/horizon-persistence-oos-v1

Depois reinicie o kernel do Spyder e execute:

    %run pesquisar_persistencia_horizontes_spyder.py

O script reproduz o checkpoint Control ao centavo antes dos novos
cenários. Os resultados da v1.0.4 ainda dependem dessa execução.
O ZIP final, exclusivo dessa versão, fica em:

    output/persistencia_horizontes_u77_v104/pacote_persistencia_horizontes_u77_v104.zip

Consulte a seção 25.12.4 de MANIFEST.md para os critérios de seleção,
limitações de validação e interpretação do CSV de variáveis históricas.


### Versão experimental 1.0.5 — comparar ganhos de ativos no curto e médio prazo

A execução padrão do arquivo existente
`pesquisar_persistencia_horizontes_spyder.py` passa a ser v1.0.5.
O código testa candidatos contra o ativo **realmente mantido** na
simulação da própria variante: H05/H10 (curto) e H20/H40 (médio).
Treina regressões auxiliares de retorno percentual líquido com
apenas os rótulos passados plenamente maturados. A política
Control permanece como fallback, com holding=2. H60 continua
no Control original.

Na pasta do TCC, use o console do Spyder:

    !git fetch origin
    !git switch research/horizon-persistence-oos-v1
    !git pull --ff-only origin research/horizon-persistence-oos-v1

Reinicie o kernel e execute:

    %run pesquisar_persistencia_horizontes_spyder.py

Os dez históricos candidatos do cache local U67+10 são
obrigatórios. Nenhum download será feito.
O pacote de resultados está separado das versões anteriores:

    output/persistencia_horizontes_u77_v105/pacote_persistencia_horizontes_u77_v105.zip

Antes de simular candidatos, o código verifica o checkpoint
US$ 587.165.941,13 da política Control. Ver detalhes e
limitações na seção 25.12.5 de `MANIFEST.md`.


### Versão experimental 1.0.6 — captura H02/H03/H05/H10 + auditoria ampliada

`pesquisar_persistencia_horizontes_spyder.py` executa
a v1.0.6 por padrão. A mesma execução gera dois estudos
**independentes e apenas diagnósticos**:

- **Captura:** retorno previsto do candidato versus o ativo
  em carteira, nos horizontes H02/H03/H05/H10.
- **Auditoria ampliada:** seleção em
  H01/H02/H03/H05/H10/H15/H20/H30/H40/H60,
  com ranking consolidado e o líder de cada horizonte.

O ranking de captura exige vantagem prevista em pelo menos
três horizontes, com ganho médio positivo nos grupos
H02/H03 e H05/H10. A auditoria não aplica esse filtro.

O Control original, com holding=2, é o único backtest
financeiro; candidatos NÃO são automaticamente negociados.
O processo calcula diferenças realizadas somente depois
de selecionados os candidatos, com maturidade dentro do
próprio fold. Não há download de dados e o cache histórico
U67+10 anterior é obrigatório.

No Spyder:

    !git fetch origin
    !git switch research/horizon-persistence-oos-v1
    !git pull --ff-only origin research/horizon-persistence-oos-v1

Reinicie o kernel e execute:

    %run pesquisar_persistencia_horizontes_spyder.py

A execução usa CPU para treinar dez regressões por ativo
e fold; pode demorar consideravelmente mais que a v1.0.5.
O Control deve reproduzir US$ 587.165.941,13.
O pacote será escrito exclusivamente em:

    output/persistencia_horizontes_u77_v106/pacote_persistencia_horizontes_u77_v106.zip

Leia a seção 25.12.6 do `MANIFEST.md` para os critérios
de causalidade e as limitações da análise contrafactual.


### Versão experimental 1.0.7 — comparar pares com calibração histórica

A versão padrão do mesmo script
`pesquisar_persistencia_horizontes_spyder.py` é agora v1.0.7.
Para H02/H03/H05/H10, compara cada candidato tanto com
o ativo em carteira quanto com a próxima escolha do Control.
O ganho previsto relativo é reduzido conforme erros
observados em pares selecionados na janela histórica.
Uma checagem cronológica fica separada da estimação de
confiabilidade. As decisões financeiras do Control
permanecem intocadas.

Na mesma execução, a auditoria independente continua
com H01/H02/H03/H05/H10/H15/H20/H30/H40/H60.
Os resultados só são medidos se o horizonte termina
dentro do fold, sem reutilizar fronteiras de decisão.
Somente o Control é simulado financeiramente.

Atualize a branch no console do Spyder:

    !git fetch origin
    !git switch research/horizon-persistence-oos-v1
    !git pull --ff-only origin research/horizon-persistence-oos-v1

Reinicie o kernel e execute:

    %run pesquisar_persistencia_horizontes_spyder.py

O script requer os mesmos dez históricos externos congelados
do U67+10, não faz download de dados e pode demorar
significativamente: dez horizontes finais mais quatro
de calibração por ativo e por fold. O resultado fica em:

    output/persistencia_horizontes_u77_v107/pacote_persistencia_horizontes_u77_v107.zip

O checkpoint Control de US$ 587.165.941,13 é condição
obrigatória antes da pesquisa. O novo estudo não produziu
ainda resultado financeiro, e os rankings são diagnósticos,
não ordens nem prova de lucratividade. Consulte a seção
25.12.7 em `MANIFEST.md`.

### v1.0.8 — aprendizado direto entre pares e auditoria independente

A versao padrao do arquivo `pesquisar_persistencia_horizontes_spyder.py` e agora v1.0.8. Reconstroi os rotulos de cada H independentemente, corrigindo a falta artificial de resultados curtos perto do fim do historico. Para H02/H03/H05/H10, treina LightGBM para prever diretamente a diferenca de retorno candidato versus referencia, com calibracao historica do erro do vencedor. A comparacao considera tanto o ativo atual quanto a acao que o Control escolheria.

Na mesma execucao continua a auditoria H01/H02/H03/H05/H10/H15/H20/H30/H40/H60, isolada do novo modelo. Somente o Control holding=2 opera; nenhum candidato e negociado. A pesquisa exige os mesmos dez ativos externos do cache congelado e nao consulta Alpaca.

Atualize a branch no console do Spyder:

    !git fetch origin
    !git switch research/horizon-persistence-oos-v1
    !git pull --ff-only origin research/horizon-persistence-oos-v1

Reinicie o kernel e execute:

    %run pesquisar_persistencia_horizontes_spyder.py

O checkpoint obrigatorio e US$ 587.165.941,13. Esta versao adiciona ate 24 ajustes de modelos de pares aos 2.250 treinamentos individuais da auditoria; pode demorar substancialmente. O pacote esperado e:

    output/persistencia_horizontes_u77_v108/pacote_persistencia_horizontes_u77_v108.zip

Resultados da v1.0.8 ainda aguardam execucao. Para o protocolo cientifico e as limitacoes, consulte a secao 25.12.8 de `MANIFEST.md`.

### Pesquisa v1.0.9 — todos os rankings em uma mesma execucao com logs

A versao padrao de `pesquisar_persistencia_horizontes_spyder.py`
e agora v1.0.9. Executa e compara **oito metodos**:
retorno previsto, vantagem liquida contra Control, Borda,
TOPSIS, fronteiras de Pareto, ranking probabilistico,
ranking robusto e Learning to Rank (LambdaMART).
Somente Learning to Rank treina um modelo de ordenacao
adicional; os outros sete sao pos-processamentos de
previsoes congeladas. Todos compartilham o mesmo
universo elegivel e os mesmos horizontes H02/H03/H05/H10.

O criterio descritivo PRINCIPAL e **NDCG@5**, com
agregacao macro dos folds. Tambem sao gerados Spearman,
Kendall, top-1, top-5, taxa de vitoria sobre Control
e top-1 regret. A auditoria separada H01–H60,
o modelo direto de pares e o Control permanecem
na mesma execucao, sem alteracao da politica financeira.

No console do Spyder:

    !git fetch origin
    !git switch research/horizon-persistence-oos-v1
    !git pull --ff-only origin research/horizon-persistence-oos-v1

Depois de reiniciar o kernel, execute:

    %run pesquisar_persistencia_horizontes_spyder.py

O script usa apenas snapshots congelados U67+10, sem
download de mercado, e deve reproduzir o Control ao
centavo antes do estudo. Podera consumir bastante tempo
para treinar os modelos de previsao anteriores, os
modelos de calibracao e um LambdaMART por fold.

Artefatos da execucao:

    output/persistencia_horizontes_u77_v109/pacote_persistencia_horizontes_u77_v109.zip

O ZIP inclui `ranking_methods.log` e
`ranking_all_assets.csv` com todos os candidatos e
oito rankings por dia, alem de
`ranking_methods_leaderboard.csv`,
`ranking_methods_by_fold.csv`,
`ranking_oos_quality_by_session.csv`,
`ranking_historical_error_calibration.csv` e
arquivos anteriores de auditoria. Nenhuma regra nova
e promovida automaticamente. Os folds ja foram vistos
em campanhas previas; o ranking obtido e exploratorio.
Ver `MANIFEST.md` secao 25.12.9.

### Pesquisa v1.0.10 — rotacao autonoma com rankings matematicos

A versao padrao de `pesquisar_persistencia_horizontes_spyder.py`
executa agora quatro politicas matematicas independentes,
utilizando as mesmas previsoes H02/H03/H05/H10:
`net_gain`, `borda_pareto`, `topsis`, `consensus`.
As politicas novas nao treinam modelos de ordenacao;
a auditoria legada v1.0.9 continua treinando LambdaMART
separadamente, sem usar suas previsoes nas rotacoes autonomas.
Essa auditoria faz a execucao demorar mais. O Control
e apenas a referencia financeira e de precisao ex post;
nao fornece destino ou estado as politicas alternativas.

Cada politica pode manter, rotacionar e ir para caixa;
minimo de dois pregoes por posicao e regras fixas de
melhoria marginal prevista, custos e risco.
Sao realizados replays financeiros separados com o
mesmo simulador, universo, fees e calendario. O checkpoint
congelado do Control US$ 587.165.941,13 e verificado
antes das alternativas.

Alem de capital, Sharpe, drawdown e operacoes, sao
reportadas precisao das trocas e vantagem observada
versus manter o ativo, pelos horizontes H02/H03/H05/H10
e pelos tres folds. A metrica primaria descritiva e
precisao de troca H05, com pelo menos 10 resultados
maturos em cada fold para obter posicao no ranking.
Os folds OOS ja foram estudados repetidamente e nenhuma
estrategia e promovida automaticamente.

**Execucao no console do Spyder:**

    !git fetch origin
    !git switch research/horizon-persistence-oos-v1
    !git pull --ff-only origin research/horizon-persistence-oos-v1
    !git log -1 --oneline

Reinicie o kernel e execute:

    %run pesquisar_persistencia_horizontes_spyder.py

**Resultados:**
`output/persistencia_horizontes_u77_v110/pacote_persistencia_horizontes_u77_v110.zip`

O pacote contem as tabelas de capital por politica,
operacoes, qualidade de decisoes, leaderboard,
rankings ja conhecidos e logs. `execution_full.log`
captura a execucao completa. Na conclusao, o Windows
produz um sinal sonoro curto de sucesso; em erros,
dois tons diferentes e o trace ficam preservados no log.
Nao ha chamadas de mercado ou operacoes reais.
Consulte MANIFEST.md 25.12.10.


### Pesquisa v1.0.11 — explicar a precisão do Control por ablação

A v1.0.11 conserva o **Control congelado** (capital
US$ 587.165.941,13; 75 ativos, 3 folds) e testa
seis variantes da MESMA previsão de utilidade:
permanência mínima de 0 ou 2 pregões, cruzada com
margem zero, margem base ou margem calibrada pelo
histórico pré-OOS. A variante H2/margem calibrada
é o Control reproduzido e não é reexecutada.
Não há novos modelos de decisão probabilística.

Para diferenciar o bloqueio imediato dos efeitos
indiretos de mudar a carteira, geramos duas análises:
`utility_ablation_one_step.csv` aplica as seis regras
ao MESMO ativo e tempo de permanência do Control em
cada data, e seis simulações completas com estado
independente gravam operações, capital e qualidade
das rotações. A métrica diagnóstica principal é
precisão de trocas H05 versus manter, por fold.
O OOS existente não será usado para tuning.

A v1.0.11 ainda executa as auditorias anteriores.
A ordenação autônoma matemática da v1.0.10
continua separada; o comparador Learning to Rank
v1.0.9 permanece somente no diagnóstico histórico.
Todos os custos e as execuções continuam simulados,
sem ordens reais e sem alteração do Control.

**Atualização do Spyder:**

    !git fetch origin
    !git switch research/horizon-persistence-oos-v1
    !git pull --ff-only origin research/horizon-persistence-oos-v1
    !git log -1 --oneline

Reinicie o kernel:

    %run pesquisar_persistencia_horizontes_spyder.py

**Pacote para análise:**
`output/persistencia_horizontes_u77_v111/pacote_persistencia_horizontes_u77_v111.zip`

O ZIP é finalizado após a conclusão do script, o
registro dos logs completos e o sinal sonoro do
Windows. Em caso de falha, envie
`output/persistencia_horizontes_u77_v111/execution_full.log`.
O sinal de sucesso usa tons diferentes do alerta
de erro. Ver o protocolo e as limitações da
seção 25.12.11 em `MANIFEST.md`.
