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
