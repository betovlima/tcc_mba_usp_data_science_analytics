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


## Replay isolado U67+10 versus Paper (branch de pesquisa)

Script: `reproduzir_u77_paper_20260804_spyder.py` (versao
`u77-paper-replay-v1.0.0`). Esta pesquisa nao modifica o experimento oficial
`v1.22.1` nem conecta a conta Paper/Live, MongoDB ou qualquer executor de ordens.

**Pergunta respondida:** qual seria o capital e quais seriam as posicoes
simuladas se o Challenger U67+10 (77 ativos pedidos, 75 efetivos)
comecasse com US$ 10.000 em CASH na abertura de **04/08/2026** e
terminasse na sexta-feira **16/10/2026**?

**Requisitos:** instalar `requirements.txt`, configurar `.env`
local com `ALPACA_API_KEY` e `ALPACA_SECRET_KEY` (SIP historico)
e executar na raiz do TCC:

```powershell
python -m pip install -r requirements.txt
python -m unittest tests.test_u77_paper_replay -v
python reproduzir_u77_paper_20260804_spyder.py
```

No Spyder, abrir o script e executar F5. O script baixa novamente barras
RAW/SIP de 2016-01-01 a 2026-10-16 e Corporate Actions ate 16/10;
nao reutiliza o cache da campanha dos US$ 587 milhoes, nao usa banco e
nao escreve em `dados/u67`.

Como DOC e CLMT possuem exclusoes estruturais documentadas, seus
historicos nao sao reconstruidos. Qualquer nova exclusao estrutural ou
ausencia dos 75 ativos exigidos aborta a pesquisa, em vez de produzir
um resultado com universo diferente.

Cada execucao cria um novo `run_id`, gravando:

- `dados/.u77_paper_replay/<run_id>/`: snapshot bruto, eventos e hashes
- `output/u77_paper_replay/<run_id>/resultado.json`: capital e metadados
- `output/u77_paper_replay/<run_id>/decisoes_e_patrimonio.csv`:
  uma linha por sessao, ranking e capital
- `output/u77_paper_replay/<run_id>/operacoes.csv`:
  operacoes, quantidade, preco, custos, lucro realizado
- `output/u77_paper_replay/<run_id>/margens_calibracao.csv`
- `output/u77_paper_replay/<run_id>/auditoria_dados.csv`
- `output/u77_paper_replay/<run_id>/snapshot_manifest.json`
- `output/u77_paper_replay/<run_id>/pacote_analise_u77_paper_replay_*.zip`:
  os artefatos para comparacao com a auditoria Paper

**Envie somente o ZIP gerado em `output/` para a comparacao inicial.**
Preserve localmente o snapshot sob `dados/.u77_paper_replay` para
reproducao posterior. Nenhum dado nem ZIP e enviado automaticamente ao Git.

**Limitacoes:** o motor e a politica cientifica walk-forward do TCC
nao recriam automaticamente o treinamento diario/intradiario do Paper.
Os precos atualizados podem divergir do snapshot congelado de outubro.
A ultima operacao `FINAL_SELL` corresponde somente a liquidacao
contabil no backtest; ela nao e uma venda que o robo teria enviado.
O universo U77 foi escolhido retrospectivamente; nao e evidencia
prospectiva nem promessa de lucro.

**Branch:** `research/u77-paper-replay-20260804-20261016`. A `main`
permanece intocada.
