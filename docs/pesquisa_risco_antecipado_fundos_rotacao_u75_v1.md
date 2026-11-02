# Experimento de risco antecipado, fundos e preservacao da rotacao U75

**Versão:** `u75-early-risk-bottom-reversal-v1.0.0`  
**Branch:** `research/u75-early-risk-bottom-reversal-v1`  
**Executor:** `pesquisar_risco_reversao_u75_spyder.py`  
**Fonte:** Strategy #30, U75 congelado (75 ativos), replay de 1.576 sessoes ate 28/10/2026.  
**Ambiente:** somente TCC, sem operacoes ou alteracao de MCT, sem retreino em producao.

## Pergunta principal

Conseguimos identificar **risco de continuidade de queda** antes da perda
principal e, separadamente, **oportunidades de recuperacao** em ativos que
ja recuaram, sem perder a capacidade de rotacao do Control e sem induzir
permanencia indevida em CASH?

Nao se vende porque a acao esta perto do fundo. Uma perda iminente deve
ser prevista **antes** do recuo; a deteccao de recuperacao serve para abrir
outras oportunidades de compra depois de deterioracoes.

## Invariantes

- U75, parametros do Control, modelos LightGBM originais, folds, dados e
  simulador financeiro nao sao modificados.
- Treino/calibracao dos **75 modelos** ocorre uma unica vez por fold.
  Os mesmos modelos alimentam os tres cenarios. Control deve reproduzir
  exatamente patrimonio, ativo por pregao e fold do ZIP Strategy #30,
  com tolerancia de US$ 0,01. Falha de paridade **bloqueia todos os
  replays experimentais**.
- Decisao no fechamento `t`, execucao na abertura `t+1`, mesma regra
  de custos/overnight e reavaliacao diaria, sem entrar/sair no mesmo dia.
- O Control executa sua decisao **antes** da analise adicional. Se ele
  ja escolheu uma alternativa de baixa probabilidade de queda, a decisao
  e preservada. Apenas evidencias de risco ou recuperacao podem desviar
  a acao. Se o risco e elevado, procurar outra acao primeiro; CASH e
  ultima alternativa quando nao existe candidato admissivel.
- Nenhuma utilizacao dos retornos futuros nos atributos de decisao.
  Rotulos futuros so entram no ajuste se amadureceram antes do
  primeiro pregao de teste do fold.
- Nenhum ajuste de hiperparametros pelo patrimonio historico observado.
- As grandes quedas de 2026 ja foram analisadas: **este teste nao
  constitui validacao nova e intocada**, mesmo com separacao temporal
  entre ajuste e teste.

## Modelos adicionais e dados

Os dois classificadores sao regressao logistica (StandardScaler +
LogisticRegression) treinada em painel de 75 ativos por fold, com os
atributos originais disponiveis no fechamento:

`return_5`, `return_20`, `return_60`, `vol_20`,
`vol_ratio_5_20`, `ema_distance_20`,
`distance_from_high_20`, `distance_from_low_20`,
`trend_efficiency_20`, `momentum_acceleration_5_20`,
`rsi_14`, `atr_pct_14`.

### A: risco de continuidade de queda

Rotulo positivo se o minimo dos cinco pregoes posteriores ao fechamento
`t` estiver **8% ou mais abaixo da abertura de `t+1`**.
O alvo so amadurece em `t+5`. Inferencia usa somente atributos de `t`.
O alerta experimental requer `p_downside >= 0.40`.

### B: oportunidade de recuperacao apos queda

Somente observacoes que no fechamento ja estao **10% ou mais abaixo
do maximo dos ultimos 20 pregoes** compoem o conjunto de treinamento B.
O rotulo e positivo se `close(t+10)/open(t+1)-1 >= 5%`.
Na inferencia, a regra exige `p_recovery >= 0.60` e risco A abaixo
de 0.40.

Esses valores sao **hipoteses fixadas antes desta nova execucao**, nao
limites cientificamente comprovados nem parametros que devem ser ajustados
depois de olhar o capital resultante. Se houver uma unica classe no
treino, o modelo devolve a frequencia observada. Sem pelo menos 500
registros maduros, o respectivo modelo fica indisponivel e a politica
preserva a escolha do Control.

### Como decidir

Usamos as **utilidades LightGBM originais**, com scores de ativos ja
elegiveis no motor. O risco penaliza o score de ativos com probabilidade
de queda acima do limiar, e a recuperacao pode atribuir bonificacao pequena
a ativos de fundo com risco baixo.

- A penalizacao de risco e `0.08 * max(p_downside - 0.40, 0)`.
- A bonificacao B e `0.015 * (p_recovery - 0.60)` quando o sinal B e
  valido (caso contrario zero).
- O vencedor da revisao precisa ter score original positivo e score
  ajustado positivo.
- Sem alta de risco no alvo original, a politica A **nao o modifica**.
- A politica A+B aceita uma nova rotacao por recuperacao apenas se a
  utilidade ajustada for ao menos `0.0025` superior a alternativa
  original do Control.
- Se o alvo original apresentar risco elevado, o sistema escolhe o
  melhor ativo admissivel pelo score ajustado. Na ausencia completa de
  alternativa, pode escolher CASH e reavaliara a abertura de uma nova
  posicao no dia seguinte.

Nota: uso da palavra *fundo* e probabilistico. Nenhum algoritmo deste
experimento usa o menor preco futuro para escolher o ponto exato de compra.
Fundo detectado *ex post* nao pode ser apresentado como sinal negociavel.

## Tres cenarios financeiros, mesmo treino

| Cenario | Rotacao Control | A: risco | B: recuperacao | CASH |
|---|---|---|---|---|
| Control | original | nao | nao | original |
| Control + risco | primeiro | sim | nao | ultima alternativa |
| Control + risco + reversao | primeiro | sim | sim | ultima alternativa |

Medidas: patrimonio, Sharpe, MaxDD, ES95 empirico da carteira,
pior pregao, dias negativos, custos, rotações, dias em CASH e
sobrescritas das decisoes originais. Diagnosticos preditivos A e B
incluem Brier e frequencia de eventos, separados por fold. Para cada
evento de 2026, e importante conferir se a primeira decisao ocorreu
ANTES da perda, nao no proprio fundo.

## Execucao

No Git Bash do repositorio TCC:

```bash
git status --short
git fetch origin
git switch --track origin/research/u75-early-risk-bottom-reversal-v1
python -m unittest discover -s tests -p 'test_risco_reversao_antecipado_u75.py' -v
python -m unittest discover -s tests -p 'test_catalogo_ativos_futuros.py' -v
```

Se a branch ja existe localmente, usar:

```bash
git switch research/u75-early-risk-bottom-reversal-v1
git pull --ff-only origin research/u75-early-risk-bottom-reversal-v1
```

No Spyder, com diretorio de trabalho na raiz do TCC, abra e execute F5:

```python
%run pesquisar_risco_reversao_u75_spyder.py
```

Requer localmente: `dados/strategy30_snapshot/`, o ZIP original
Strategy #30 aceito por `pesquisar_controle_strategy30_spyder.py` e
as dependencias instaladas em `requirements.txt`. Nao baixa CSVs,
nao consulta bancos e nao modifica dados congelados.

O ZIP e gerado automaticamente em:

```text
output/risco_reversao_antecipado_u75/
  pacote_risco_reversao_antecipado_u75_<UTC>.zip
  <UTC>/
    protocolo_cientifico.json
    auditoria_snapshot.csv
    predicoes_risco_reversao.csv
    treino_modelos_folds.csv
    metricas_preditivas_por_fold.csv
    margens_control_por_fold.csv
    auditoria_paridade.json
    auditoria_paridade_diaria.csv
    control_predictions.csv
    control_trades.csv
    risco_decisoes.csv
    risco_predictions.csv
    risco_trades.csv
    risco_reversao_decisoes.csv
    risco_reversao_predictions.csv
    risco_reversao_trades.csv
    comparacao_dia_a_dia.csv
    patrimonio_tres_politicas.png
    drawdowns_tres_politicas.png
    resultado.json
    resumo.md
```

Se a execucao falhar, o pacote sinaliza `failed` e inclui
`erro_execucao.json`; nao simular sucesso nem sobrescrever resultados.

## Trilha separada: novos ativos **para eventos futuros**

`pesquisar_catalogo_ativos_futuros_spyder.py` faz somente GET ao
catalogo atual Alpaca de acoes norte-americanas negociaveis. Registra
IDs e tickers e compara com o inventario anterior, destacando novas
listagens e mudancas de identidade, **sempre fora do U75 congelado**.

No Spyder, executar:

```python
%run pesquisar_catalogo_ativos_futuros_spyder.py
```

Requer credenciais locais `.env` Alpaca. Usa catalogo com status atual,
sem montar historia retrospectiva a partir da lista presente (evitar
*survivorship bias*). A primeira execucao e somente inventario,
nao lista de ativos promovidos. Para um candidato tornar-se elegivel
em operacao futura serao necessarios: identidade consistente de ativo,
RAW/SIP e Corporate Actions, validacao de liquidez e de disponibilidade
em `t`, modelo treinado antes da avaliacao, teste da contribuicao
marginal ao capital e regras de promocao predefinidas. Nenhum ativo
descoberto e incluido automaticamente no MCT.

## Limites conhecidos, essenciais para o TCC

O motor Control historico possui uma dependencia identificada de
disponibilidade de precos da proxima sessao para elegibilidade dos
scores. A paridade exata comprova *equivalencia a referencia*, nao
elimina essa dependencia. Ate que esse detalhe seja corrigido e a
pesquisa seja repetida com dados posteriores **nao utilizados para
desenhar as hipoteses**, nao considerar os resultados evidencias
independentes de capacidade de protecao futura.

O proprio modelo A avalia um evento de minima de cinco pregoes e o
modelo B um resultado de dez pregoes: os resultados se sobrepoem entre
datas proximas. Bootstrap ingênuo por dia seria otimista; priorizar
epidosios nao sobrepostos e intervalos em blocos.

**Nenhuma variante sera promovida ao MCT com base nesta execucao.**
