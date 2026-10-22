# TCC MBA USP — Rotação de Capital com Machine Learning

**Release oficial: `v1.23.0` — universo selecionado reproduzido.** O executor usa o motor Control histórico (`v1.22.1`) e as rotinas de expansão pesquisadas na `v1.4.1`. O encerramento da release está na seção 28 de [MANIFEST.md](MANIFEST.md); as seções anteriores preservam os experimentos históricos.

## Requisitos

- Python 3.12
- dependências de `requirements.txt`

Instale:

```bash
python -m pip install -r requirements.txt
```

## 1. Verifique os dados

A execução oficial usa **duas fontes congeladas**:

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

**Não execute `preparar_snapshot_u67.py` para reproduzir o checkpoint:**
isso substituiria a fonte histórica. Sem os dez CSVs locais e os arquivos de
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

O executor oficial reproduz o conjunto fixo de 77 ativos solicitados (75 efetivos), cujo capital histórico esperado é **US$ 587.165.941,13**; não realiza nova busca de ativos ou otimização de universo. A reprodução usa os hiperparâmetros fixados pela campanha histórica e calibra
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

O checkpoint histórico foi novamente confirmado em **21/10/2026**: Control de 65 ativos **US$ 76.927.051,39**; Control com 15 candidatos **US$ 290.727.954,41**; Control com dez selecionados **US$ 587.165.941,13**, para cutoff de **06/10/2026**. Esta é a campanha de pesquisa `u67-asset-expansion-v1.4.1`, não uma alteração do algoritmo ou das configurações do motor `v1.22.1`.

**Antes de rodar:** leia [CONTEXTO_MESTRE.md](CONTEXTO_MESTRE.md), que contém o contrato dos snapshots, a diferença entre a execução financeira inicial e a assurance de expansão, a preparação offline no Spyder, as verificações de hashes e as limitações científicas. O cache extra dos 15 candidatos não está versionado; não é possível prometer US$ 587 milhões em um clone limpo sem os mesmos arquivos. Evite executar `preparar_snapshot_u67.py` ou o runner de expansão com downloads ativos durante a reprodução.

A integração da pesquisa histórica (commit `13e5ed39ed0e700d2eb8c90bc8bee64adc7fdf9a`) à `main` já ocorreu no commit `9d2e279938a2ad271f3ef5e0ac86a18e130c7e4f`. A referência de tag para esta reprodução é `research-u67plus10-challenger-v1.4.1` (confira sua publicação no repositório antes de usá-la). Os resultados completos e alertas de seleção estão em [MANIFEST.md](MANIFEST.md), seção 26, e em [PESQUISA_EXPANSAO_U67.md](PESQUISA_EXPANSAO_U67.md).


## Executor oficial do universo selecionado (v1.23.0)

`reproduzir_experimento.py` agora executa o **mesmo replay financeiro de dez candidatos** usado na pesquisa de expansão, e não somente a referência base. O arquivo do executor usa nomenclatura genérica; a adaptação preserva a rotina histórica por trás de `reproducao/experimento.py`, sem duplicar a matemática. A saída é `output/reproducao_universo/pacote_reproducao_universo.zip`.

O código valida primeiro a identidade do snapshot, a composição do universo e os **20 hashes dos CSVs** (barras e eventos de dez ativos). Depois treina a referência e os candidatos por fold, calibra conjuntamente e compara o capital simulado ao checkpoint congelado. Nenhuma chamada de download é feita pelo executor. O cenário é histórico, não uma previsão de resultado futuro. **A execução completa foi concluída no Spyder em 21/10/2026**, sobre o cache histórico preservado, com `status=completed`, `reproduced_exact_to_cent=true`, capital de **US$ 587.165.941,1330212** e diferença de **US$ 0,00**. O ZIP conferido é `pacote_reproducao_universo.zip`, SHA-256 `0c3518387556f147897baf121cb089f532f8355a8bb074c91fe0a6b5ec4b6ac8`. O pacote contém 29 arquivos, incluindo `predictions.csv` (1.560 sessões), `trades.csv` (710 registros), `reproducao_universo.json` e análises gráficas. O checkpoint foi confirmado também pelos modos de pesquisa `decision_validation` e `assurance_validation` sobre o mesmo recorte congelado. O cache local continua **não versionado**: clones limpos não têm, por si só, os arquivos suficientes para refazer o replay.

## Experimento v1.23.2 — comparacao do TCC com a data final atualizada

**Objetivo restrito:** investigar o efeito de atualizar os historicos e de avancar o corte temporal mantendo *o mesmo universo vencedor fixo*, que gerou US$ 587.165.941,13 em 06/10/2026. Nao descobrir, classificar ou otimizar novos ativos.

- Branch em uso: `research/v1.23.1-fresh-snapshot-20261022`; **revisao experimental v1.23.2**, sem alterar a `main`.
- Entrada no Spyder: `executar_atualizacao_series_spyder.py` (reiniciar kernel, F5).
- `TARGET_CUTOFF = "2026-10-22"` e `EVALUATION_CUTOFFS = ("2026-10-06", "2026-10-21")`.
- **Uma coleta de 77 ativos solicitados** de Alpaca RAW/SIP/1Day e seus eventos corporativos. O pacote historico original (67 + 10) define a lista uma unica vez. A pesquisa nao executa descoberta de candidatos, nem altera a composicao.
- **75 ativos efetivos obrigatorios.** Exclusoes estruturais permanecem DOC e CLMT. Um novo problema de integridade interrompe a execucao com erro; nao descartar ativos silenciosamente nem reparar series manualmente.
- Dados localmente isolados em `dados/.u67_fresh_runs/asof_<DATA>_<ID>/universe_77/` (nao versionados no Git), manifesto e SHA-256. Os arquivos do snapshot oficial `dados/u67/` e os manifestos de `evidencias/` sao preservados.
- Dois replays financeiros com **o mesmo snapshot de 77 simbolos, os mesmos algoritmos, atributos, parametros, custos, treinamento e politica**, alterando apenas a data final. A divisao interna do motor historico entre base e adicionais existe unicamente para manter os calculos de v1.23.0; nao e etapa de pesquisa, escolha ou coleta em separado.
- **Corte A — 06/10:** mede alteracoes do resultado em 06/10 com historicos atualizados, comparando ao TCC v1.23.0 (US$ 587.165.941,13).
- **Corte B — 21/10:** mede o efeito de avancar a data final na mesma fonte e permite comparar com a execucao externa encerrada em 21/10.
- Saida para compartilhar: `output/fresh_universe/asof_<DATA>_<ID>/pacote_comparacao_cutoffs.zip`; inclui `comparacao_cutoffs.json`, manifestos, dados financeiros e operacoes dos dois replays, sem RAWs. O checkpoint oficial nao e imposto aos experimentos; diferenças devem ser explicadas, nao ajustadas.
- Nenhuma ordem, nenhuma atualizacao de Mongo e nenhuma modificacao da aplicacao operacional.
- Testes CI validam a separacao, integridade de referencias e orquestracao de 77 ativos com mocks. A reproducao economica completa depende de executar F5 localmente com credenciais da Alpaca e enviar o pacote.

Para baixar a branch no terminal Git:

```powershell
git fetch origin
git switch research/v1.23.1-fresh-snapshot-20261022
git pull --ff-only origin research/v1.23.1-fresh-snapshot-20261022
```

