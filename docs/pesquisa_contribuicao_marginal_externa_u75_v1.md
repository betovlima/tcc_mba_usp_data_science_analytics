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
