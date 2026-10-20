# TCC MBA USP — Rotação de Capital com Machine Learning

## Entradas do projeto

Este repositório mantém **somente dois scripts de execução na raiz**:

| Arquivo | Finalidade |
| --- | --- |
| `buscar_ativos.py` | Pesquisa exploratória de ativos (PCA, agrupamentos e, opcionalmente, investigação U67+n) sem alterar a composição do universo oficial |
| `reproduzir_experimento.py` | Reproduz o experimento científico oficial usando o snapshot congelado versionado em `dados/u67/`, sem baixar dados nem consultar outra aplicação operacional |

A metodologia, os resultados e o histórico científico continuam em
[MANIFEST.md](MANIFEST.md). A reorganização não altera os modelos, parâmetros,
configurações científicas nem os dados congelados.

### Instalação

Python 3.12:

```bash
python -m pip install -r requirements.txt
```

### Reprodução oficial

```bash
python reproduzir_experimento.py
```

No Spyder, abra `reproduzir_experimento.py` e execute com F5 após reiniciar
o kernel. O resultado é salvo em `output/reproducao/` e o script valida o
snapshot local `dados/u67/` antes de iniciar o treinamento.

### Busca de ativos

```bash
python buscar_ativos.py
```

No Spyder, execute `buscar_ativos.py` com F5. No início do script, selecione:

- `MODO_BUSCA = "agrupamentos"`: explora o catálogo Alpaca e produz
  shortlist por atributos técnicos, PCA e clusters; **não incorpora candidatos
  automaticamente** ao universo científico.
- `MODO_BUSCA = "expansao_u67"`: mantém a investigação incremental U67+n e seus
  checkpoints históricos. O módulo preservado usa por padrão
  `RUN_MODE = "decision_validation"`. Os outros modos de pesquisa são
  documentados em [PESQUISA_EXPANSAO_U67.md](PESQUISA_EXPANSAO_U67.md).

A busca consulta a Alpaca e exige credenciais locais. Use `.env` conforme
`.env.example`. Diferentemente da reprodução, a busca pode coletar dados
de candidatos e gravar cache em `output/` ou `dados/` ignorados pelo Git.
Ela não altera `dados/u67/`.

**Distinção de resultados:** a referência reproduzida no contrato científico
U67 é de aproximadamente US$ 76,93 milhões. A campanha exploratória
U67+10 registrou um cenário de aproximadamente US$ 587,17 milhões.
Este último não é promovido automaticamente a novo resultado oficial;
a metodologia e suas limitações estão no `MANIFEST.md`.

## Módulos internos e manutenção

As rotinas auxiliares de auditoria, preparação do snapshot e figuras foram
movidas para `reproducao/`, sem virar novos scripts na raiz. Quando necessário:

```bash
python -m reproducao.avaliar_elegibilidade_u67
python -m reproducao.gerar_figuras_tcc
```

A preparação do snapshot é **separada e explícita**; a reprodução oficial
nunca a executa. Apenas para uma nova campanha de coleta, com credenciais
Alpaca e posterior revisão e versionamento dos dados:

```bash
python -m reproducao.preparar_snapshot_u67
```

Há também `python -m reproducao.finalizar_decision_validation` para recuperar
a serialização de uma campanha histórica já calculada.

Os arquivos de `engine/`, `reproducao/`, `dados/u67/`, `evidencias/`,
`figuras/`, `documento/` e os testes preservam a verificabilidade do estudo.
Scripts exploratórios redundantes foram retirados da versão corrente;
continuam recuperáveis no histórico Git.

## Testes

```bash
python -m pytest -q
python -m ruff check engine reproducao buscar_ativos.py reproduzir_experimento.py tests --select F401,F811,F821,F841
```

O código oficial não depende de banco de dados nem de outra aplicação operacional.
