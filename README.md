# TCC MBA USP — Rotação de Capital com Machine Learning

Release do motor Control de referência: **v1.22.1**; pesquisa de expansão reproduzida em **v1.4.1** e já integrada à `main`. A memória da release, as evidências e as
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


## Reproduzir a campanha histórica de US$ 587 milhões

O checkpoint histórico foi novamente confirmado em **21/10/2026**: Control de 65 ativos **US$ 76.927.051,39**; Control com 15 candidatos **US$ 290.727.954,41**; Control com dez selecionados **US$ 587.165.941,13**, para cutoff de **06/10/2026**. Esta é a campanha de pesquisa `u67-asset-expansion-v1.4.1`, não uma alteração do algoritmo ou das configurações do motor `v1.22.1`.

**Antes de rodar:** leia [CONTEXTO_MESTRE.md](CONTEXTO_MESTRE.md), que contém o contrato dos snapshots, a diferença entre a execução financeira inicial e a assurance de expansão, a preparação offline no Spyder, as verificações de hashes e as limitações científicas. O cache extra dos 15 candidatos não está versionado; não é possível prometer US$ 587 milhões em um clone limpo sem os mesmos arquivos. Evite executar `preparar_snapshot_u67.py` ou o runner de expansão com downloads ativos durante a reprodução.

A integração `research/mct-final-582` → `main` já ocorreu no commit `9d2e279938a2ad271f3ef5e0ac86a18e130c7e4f`. A referência de tag para esta reprodução é `research-u67plus10-challenger-v1.4.1` (confira sua publicação no repositório antes de usá-la). Os resultados completos e alertas de seleção estão em [MANIFEST.md](MANIFEST.md), seção 26, e em [PESQUISA_EXPANSAO_U67.md](PESQUISA_EXPANSAO_U67.md).
