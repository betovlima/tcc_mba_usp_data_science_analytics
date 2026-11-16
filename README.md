# TCC MBA USP — Rotação de Capital com Machine Learning

**Branch de correção em teste: universo efetivo de 77 ativos.** A release `v1.23.0` registra a reprodução histórica anterior e não certifica a nova execução. O executor usa a política Control/LightGBM, com uma única lista oficial de 77 ativos definida em `engine/configuracao.py`. O encerramento da release está na seção 28 de [MANIFEST.md](MANIFEST.md); as seções anteriores preservam os experimentos históricos.

## Requisitos

- Python 3.12
- dependências de `requirements.txt`

Instale:

```bash
python -m pip install -r requirements.txt
```

## 1. Verifique os dados

A execução oficial usa **um único `ASSETS` com 77 símbolos**. Os arquivos RAW permanecem em duas pastas congeladas por compatibilidade de armazenamento. Essas pastas não definem dois universos:

```text
dados/u67/                         # snapshot base versionado
  raw_bars/
  corporate_actions/
  manifest.json
dados/.u67_expansion_cache/full_raw_20261006/   # cache local, nao versionado
  raw_bars/                      # 10 candidatos selecionados
  corporate_actions/
evidencias/manifest_universo_congelado.json     # hashes SHA-256 exigidos
```

A release inclui o snapshot congelado em `dados/u67/`. Vá direto para a
execução; os hashes são conferidos antes do treinamento.

**Não execute nenhum script de download para reproduzir os dados congelados:**
o executor usa os hashes históricos e não baixa novas cotações. Sem os dez CSVs locais e os arquivos de
eventos corporativos com os hashes do manifesto, o programa aborta antes do treino.
O snapshot base está no Git, mas o cache dos candidatos precisa ser restaurado
de uma execução histórica preservada.

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
output/reproducao_universo/
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
`reproduzir_experimento.py` reutiliza as rotinas congeladas de decisão da pesquisa de expansão, com a mesma regra causal.

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

O executor oficial em teste consome **exclusivamente `ASSETS`**, com 77 ativos efetivos, incluindo DOC e CLMT.
As duas pastas de dados são somente fontes históricas: a preparação percorre
**1/77 até 77/77 em uma única passagem**. Em cada um dos três folds, os 77
ativos são treinados, calibrados e avaliados **juntos**, pelo mesmo modelo
LightGBM Control e pela mesma política de rotação de capital. Não há treinamento
separado da base seguido de inclusão posterior dos dez ativos.
O capital da execução anterior não é um checkpoint válido para esse novo universo.
Não há nova busca de ativos nem otimização do universo. A reprodução usa os hiperparâmetros fixados pela campanha histórica e calibra
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


## Reproduzir a campanha histórica de US$ 587 milhões

A campanha antiga confirmou os capitais de **US$ 76.927.051,39** (base anterior), **US$ 290.727.954,41** (expansão ampla) e **US$ 587.165.941,13** (dez selecionados), para cutoff de **06/10/2026**. **Esses números são históricos e não servem como assert de sucesso para o universo efetivo de 77 ativos.** Esta é a campanha de pesquisa `u67-asset-expansion-v1.4.1`, não uma alteração do algoritmo ou das configurações do motor `v1.22.1`.

**Antes de rodar:** leia [CONTEXTO_MESTRE.md](CONTEXTO_MESTRE.md), que contém o contrato dos snapshots, a diferença entre a execução financeira inicial e a assurance de expansão, a preparação offline no Spyder, as verificações de hashes e as limitações científicas. O cache extra dos 15 candidatos não está versionado; não é possível prometer US$ 587 milhões em um clone limpo sem os mesmos arquivos. Evite executar `preparar_snapshot_u67.py` ou o runner de expansão com downloads ativos durante a reprodução.

A integração da pesquisa histórica (commit `13e5ed39ed0e700d2eb8c90bc8bee64adc7fdf9a`) à `main` já ocorreu no commit `9d2e279938a2ad271f3ef5e0ac86a18e130c7e4f`. A referência de tag para esta reprodução é `research-u67plus10-challenger-v1.4.1` (confira sua publicação no repositório antes de usá-la). Os resultados completos e alertas de seleção estão em [MANIFEST.md](MANIFEST.md), seção 26, e em [PESQUISA_EXPANSAO_U67.md](PESQUISA_EXPANSAO_U67.md).


## Executor oficial do universo selecionado (v1.23.0)

`reproduzir_experimento.py` executa **um único replay financeiro com os 77 ativos declarados em `ASSETS`**, sem divisão do treinamento entre uma base e ativos acrescentados depois. O executor reutiliza o motor Control/LightGBM, a calibração temporal e a política de rotação. A saída é `output/reproducao_universo/pacote_reproducao_universo.zip`.

O código valida primeiro a identidade do snapshot, a composição do universo e os **20 hashes dos CSVs** (barras e eventos de dez ativos). Depois treina e calibra **os 77 ativos juntos, em cada fold**. O resultado da primeira execução válida com os 77 efetivos será um novo checkpoint, ainda não homologado. Nenhuma chamada de download é feita pelo executor. O cenário é histórico, não uma previsão de resultado futuro. **O resultado publicado em 21/10/2026 pertence à campanha anterior.** A branch de correção mantém o mesmo snapshot e o mesmo cache, mas não afirma que reproduziu aquele capital usando 77 ativos efetivos. O ZIP conferido é `pacote_reproducao_universo.zip`, SHA-256 `0c3518387556f147897baf121cb089f532f8355a8bb074c91fe0a6b5ec4b6ac8`. O pacote contém 29 arquivos, incluindo `predictions.csv` (1.560 sessões), `trades.csv` (710 registros), `reproducao_universo.json` e análises gráficas. O checkpoint foi confirmado também pelos modos de pesquisa `decision_validation` e `assurance_validation` sobre o mesmo recorte congelado. O cache local continua **não versionado**: clones limpos não têm, por si só, os arquivos suficientes para refazer o replay.


## Correção do universo efetivo de 77 ativos (branch de teste)

A branch `fix/tcc-universo-77-efetivos` usa a única constante `ASSETS` em
`engine/configuracao.py` e prepara os **77 ativos de uma vez**, incluindo DOC e CLMT. Os arquivos
historicamente guardados em duas pastas são unidos **antes** do treinamento.
Cada fold ajusta 77 modelos em uma única rodada e o replay opera com a mesma
lista completa. Qualquer ausência interrompe a execução.

- **DOC:** o arquivo histórico de `DOC` anterior a 04/03/2024 segue a
  companhia sucessora Healthpeak, negociada anteriormente como `PEAK`, e
  não deve ser interpretado como a antiga Physicians Realty Trust. A troca
  de símbolo é auditada pelo registro de Corporate Actions.
- **CLMT:** a conversão societária de 11/07/2024 preservou a proporção 1:1,
  exigindo verificação de CUSIP e Corporate Actions.

O tratamento está **em teste**, sem aprovação para negociação real. A
simulação histórica ainda precisa ser auditada quanto à correspondência do
símbolo negociável `PEAK` antes de 04/03/2024. Em caso de ausência ou
inconsistência de identidade, a execução deve falhar explicitamente.

Execute o mesmo `reproduzir_experimento.py` no Spyder (F5) com o cache
histórico preservado. A saída permanece em
`output/reproducao_universo/pacote_reproducao_universo.zip`. O capital novo
não será comparado automaticamente ao checkpoint da execução anterior.
