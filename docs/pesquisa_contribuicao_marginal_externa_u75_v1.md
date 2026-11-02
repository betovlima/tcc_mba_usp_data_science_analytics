# Pesquisa v1.0.0: contribuicao marginal de ativos externos ao U75

**Branch:** `research/u75-external-marginal-capital-v1`  
**Executor Spyder:** `pesquisar_contribuicao_marginal_externa_u75_spyder.py`  
**Escopo:** TCC, nao altera MCT, Control, regras CASH, snapshots ou `main`.  
**Estado:** codigo de teste disponibilizado; **sem resultado financeiro real ate execucao local**.

## Pergunta

Se adicionar um candidato externo ao **universo inteiro de 75 ativos**, sem retirar
nenhum dos atuais e sem mudar a regra de rotacao, a trajetoria de posicoes e
patrimonio do Control muda? O efeito incremental e positivo ou negativo?
A inclusao pode modificar a classificacao de outros ativos, a calibracao e a
sequencia de decisoes mesmo sem o externo terminar comprado.

**Control da comparacao:** Strategy #30, 75 ativos, 1.576 sessoes ate
28/10/2026, capital historico US$ 80.685.454,79. O checkpoint antigo de
US$ 587.165.941,13, ate 06/10/2026, pertence a outra campanha e **nao**
e a referencia desta pesquisa.

## Fase exploratoria v1.0.0

1. Verificar o ZIP Strategy #30 e identidade dos 75 RAW/SIP + Corporate Actions
   em `dados/strategy30_snapshot`. Abortar se ausentes/incompativeis.
2. Carregar a shortlist **ja congelada** de 24 ativos externos em
   `dados/cluster_candidate_search` utilizando o validador existente.
   Ela nao sera redescoberta e nao ha novo download.
3. Retreinar o Control de 75 ativos pelo motor independente, mesmos folds,
   parametros e custos. Exigir **paridade exata ao centavo por sessao**, ativo
   selecionado e fold identicos ao ZIP oficial. Sem essa prova, nao comparar.
4. Para cada externo do lote, adicionar **apenas esse** candidato, formar
   76 ativos, verificar igualdade do calendario e dos folds, recalibrar a
   margem por treino/calibracao historica, retreinar o LightGBM do conjunto
   e executar a simulacao financeira completa.
5. Comparar decisoes e patrimonio **na mesma data de execucao**:
   delta de patrimonio, delta de crescimento logaritmico composto, Sharpe,
   drawdown maximo, custos, numero de rotacoes, dias do externo,
   dias alterados **sem compra do externo**, detalhamento por fold.
6. Registrar hashes RAW/eventos do candidato, exclusoes/erros, CSVs diarios
   e ZIP. Nao escolher o "vencedor" como se fosse prospectivo.

**Nao ha otimizacao de CASH.** A politica do Control e reutilizada sem
alterar regras, limiares ou criterios de permanencia em CASH.

### Limites

- Os 24 candidatos foram pesquisados anteriormente com informacoes da janela
  historica completa. O resultado **nao comprova descoberta causal** ou
  capacidade de aumentar capital futuro.
- O Control possui dependencia historica de elegibilidade do pregao seguinte
  identificada em auditoria; igualar os dois backtests nao corrige isso.
- O snapshot externo termina em 06/10/2026; o Control vai ate 28/10/2026.
  Nao completar a serie automaticamente nem recriar precos. Um candidato
  que nao tiver cobertura suficiente podera ser reprovado, com erro exportado.
- Execucoes sequenciais de muitos candidatos apresentam selecao multipla:
  os maiores ganhos ex post serao tratados como **hipoteses**, nao como
  inferencias confirmatorias ou motivo para alterar o MCT.
- Para uma pesquisa nova e defensavel, deve-se construir um pool mais amplo
  com candidatos elegiveis conhecidos em cada data, selecionar no passado e
  avaliar exclusivamente em sessoes posteriores nunca utilizadas no desenho.
  A selecao pode ser ADICIONAR / MANTER / EXCLUIR, mas somente prospectiva.
- Nunca reconstruir, fazer bridge ou reparar manualmente ativos com problemas
  estruturais. Excluir com justificativa auditavel.

## Execucao com Spyder

Necessario o ZIP original Control Strategy #30 e os snapshots locais
`dados/strategy30_snapshot/` e `dados/cluster_candidate_search/`,
exatamente como usados em pesquisas anteriores. A ausencia de dados e
falha explicita, nao justificativa para usar outros precos.

No **Git Bash**, em seu repositorio:

```bash
git status --short
git fetch origin
git switch main
git pull --ff-only origin main
git switch --track origin/research/u75-external-marginal-capital-v1
```

No **Spyder**, com a raiz do repositorio como diretorio de trabalho,
abrir e executar com F5:

```python
# -*- coding: utf-8 -*-
# Lote inicial de 3 candidatos, ordem alfabetica;
# configure os indices abaixo nas variaveis de ambiente caso queira outro lote.
%run pesquisar_contribuicao_marginal_externa_u75_spyder.py
```

Variaveis de ambiente (opcionais) para a execucao em lotes:

```bash
# Git Bash, se iniciar o Spyder deste mesmo terminal:
export U75_EXTERNAL_BATCH_START=0
export U75_EXTERNAL_BATCH_SIZE=3
```

O script inicia por padrao pelo indice 0 e processa 3 candidatos; mudar
`START_INDEX` e `BATCH_SIZE` no topo do arquivo ou ajustar variaveis
de ambiente **antes de iniciar o Spyder** para processar lotes 3, 6, 9...
Nao mover nem sobrescrever snapshots. Cada execucao cria uma pasta
identificada por timestamp UTC em:

```text
output/contribuicao_marginal_externa_u75/<UTC>/
    contrato_e_lote.json
    auditoria_paridade.json
    auditoria_paridade_diaria.csv
    control_predictions.csv
    baseline_metricas.json
    comparacao_capital.csv
    comparacao_por_fold.csv
    decisoes_<SYMBOL>.csv
    trades_<SYMBOL>.csv
    identidade_<SYMBOL>.json
    candidatos_reprovados.json
    resultado.json
    resumo.md
    pacote_*.zip
```

Se a paridade do Control falhar, o script **aborta antes dos candidatos**,
salvando auditoria e pacote de erro. Se um candidato falhar, o problema e
registrado e o lote continua, sem inventar serie substituta.

Testes unitarios dos comparadores (sem Alpaca):

```bash
python -m unittest discover -s tests -p 'test_contribuicao_marginal_externa_u75.py' -v
```

## Hipoteses para a fase 2

A fase 1 testa se um ativo externo consegue **mudar o resultado**,
nao se e possivel **descobrir antecipadamente** qual incluir.
A fase 2 deve conter descoberta aninhada antes de cada fold e teste
sem ajuste ex post, com universo historicamente elegivel e
controle de multiplas tentativas. Avaliar 75 vs 75+1 vs pequenos grupos
selecionados **somente pelo periodo de treinamento** e analisar a
sequencia de decisoes alteradas.

**Criterio de interpretacao:** patrimonio maior em um replay historico
e evidência descritiva, mas nao basta para declarar capacidade
de escolha prospectiva de novos ativos.

## Complemento v1.0.1: fechamento dos 24 candidatos (02/11/2026)

Esta versão adiciona duas salvaguardas:

- O executor verifica arquivos de paridade, Control e decisões individuais
  antes de declarar que o pacote final está completo. Sua ausência faz a
  execução falhar com resultado de erro, sem falsificar evidências.
- O comparador passou a registrar `top1_control`, `top1_challenger`,
  `external_top1_days` e `external_top1_not_held_days` **quando**
  `raw_best_asset` está disponível nos dois replays. O Top-1 é o maior
  **score previsto** no fechamento, e não o melhor retorno efetivo do futuro.
  A posição efetiva continua sujeita à política de rotação original.

O pacote histórico do lote 0 a 2 em 02/11/2026 apresentou resultados
de DMO, EBS e EIX, mas **não trouxe todos os arquivos exigidos**:
faltaram `auditoria_paridade.json`, `auditoria_paridade_diaria.csv`,
`control_predictions.csv`, `baseline_metricas.json`, detalhes de DMO
e outros. O JSON registra paridade; a auditoria granular no ZIP não está completa.
Preservar o ZIP sem o alterar.

### Execução final sem novas escolhas retrospectivas

Atualizar a branch já existente, sem misturar com a `main`:

```bash
git status --short
git fetch origin
git switch research/u75-external-marginal-capital-v1
git pull --ff-only origin research/u75-external-marginal-capital-v1
python -m unittest discover -s tests -p 'test_contribuicao_marginal_externa_u75.py' -v
python -m unittest discover -s tests -p 'test_consolidacao_externa_u75.py' -v
```

No IPython do Spyder, primeiro os **21 candidatos ainda não processados**
(indices 3 a 23 da lista alfabética congelada). O executável retreina o
Control, compara a paridade e usa o motor original, sem downloads:

```python
import os
os.environ["U75_EXTERNAL_BATCH_START"] = "3"
os.environ["U75_EXTERNAL_BATCH_SIZE"] = "21"
%run pesquisar_contribuicao_marginal_externa_u75_spyder.py
```

Para obter auditoria completa dos primeiros três candidatos sem sobrescrever
o pacote antigo, execute um novo lote (se os artefatos originais não puderem
ser recuperados integralmente):

```python
os.environ["U75_EXTERNAL_BATCH_START"] = "0"
os.environ["U75_EXTERNAL_BATCH_SIZE"] = "3"
%run pesquisar_contribuicao_marginal_externa_u75_spyder.py
```

Finalmente, ainda no Spyder:

```python
%run consolidar_contribuicao_externa_u75_spyder.py
```

A consolidação lê **todos os ZIPs** `output/contribuicao_marginal_externa_u75/
pacote_contribuicao_marginal_externa_u75_*.zip`, verifica hashes,
controle, snapshots, resultados repetidos e evidências por candidato.
Reexecuções duplicadas com valores financeiros divergentes causam falha,
sem escolher o melhor resultado. Se o candidato foi repetido com resultados
iguais, a consolidação usa a execução com evidência mais completa.

Produz:

```text
output/consolidacao_externa_u75/<UTC>/
  resultado_consolidado.json
  ranking_capital_exploratorio.csv
  resumo_final.md
```

Somente se os 24 candidatos tiverem resultados e auditoria completa, sem
erros pendentes, o estado será `completed_exploratory`.
Caso contrário, será `incomplete`, com faltas explícitas. **Mesmo
`completed_exploratory` não significa que a escolha entre 24 modelos
tenha sido validada prospectivamente.**

### Encerramento científico da campanha

- O experimento confirma efeitos marginais **históricos** de incluir 1
  candidato por vez no universo do Control, com comparação pareada do
  patrimônio e decisões.
- A proposta anterior de escolher um externo pelo retorno intradiário
  superior ao líder Top-1 não apresentou benefício OOS robusto no estudo
  publicado em 01/11/2026. Não usar esse outcome futuro para selecionar ativos.
- A seleção dos 24 candidatos foi retrospectiva, e o Control original
  possui dependência de elegibilidade `t+1`. Portanto, descrever os ganhos
  encontrados como **exploratórios**; não declarar uma descoberta causal
  automática do mercado amplo.
- Não ampliar o conjunto de candidatos após examinar os resultados desta
  campanha, nem promover qualquer candidato ao MCT sem outro protocolo
  prospectivo independente. Esta é a campanha final acordada de expansão
  externa para o TCC.
