# Pesquisa econômica de rotação, risco de queda e recuperação

**Versão:** `u75-economic-rotation-recovery-v1.0.0`
**Branch:** `research/u75-economic-rotation-recovery-v1`
**Programa Spyder:** `pesquisar_vantagem_economica_rotacao_u75_spyder.py`

## Motivação e hipótese

O experimento anterior `u75-early-risk-bottom-reversal-v1.0.0` protegeu
contra as quedas de AMS/CAPR de 2026, mas o patrimônio final caiu de
US$ 80.685.454,79 (Control) para US$ 7.547.558,11 (Control com risco).
O modelo B não mudou nenhuma decisão. Esses resultados já foram inspecionados.

**Hipótese desta versão:** a vantagem econômica prevista de uma
alternativa, após penalização de perdas de cauda e custo adicional de
transição, pode preservar melhor as rotações positivas do Control
que um veto por probabilidade de perda. Recuperações depois de quedas
também podem entrar no ranking, mesmo com risco de novas mínimas.

Não alterar o U75, os modelos e hiperparâmetros LightGBM do Control,
o MCT, as regras de uma avaliação por pregão ou as fontes de dados.

## Comparações e modelos

1. **Control:** único benchmark oficial Strategy #30, paridade obrigatória.
2. **Control + vantagem econômica (risco):** regressão Ridge em todos
   os ativos estima o log-retorno de abertura `t+1` até fechamento
   `t+5`, usando apenas atributos conhecidos no fechamento de `t`.
   O risco de queda usa o classificador A anterior e a severidade média
   de perdas reais da classe de queda calculada somente no treino.
3. **Control + vantagem econômica + recuperação:** acrescenta um
   estimador de retorno de recuperação para ativos que já recuaram
   pelo menos 10% do máximo dos últimos 20 pregões. A regressão
   original tem horizonte 5, e a expectativa condicional derivada
   de probabilidades B e médias de retorno de 10 dias é convertida
   à escala de cinco dias para comparação. Não existe o filtro rígido
   `p_downside < 0.40` que bloqueou o modelo B anterior.

As duas avaliações usam o **mesmo ranking LightGBM** por fold e
a mesma margem calibrada do Control, sem retreinamento separado do
Control por cenário. Somente a política posterior muda.

### Comparação de retornos, perdas e custos

A referência financeira dos candidatos é o retorno logarítmico
estimado no horizonte de cinco pregões.

- `S_risco = E[retorno_5] - 0.40 * p_queda * E[severidade | queda]`.
- `S_conjunto = 0.65 * E[retorno_5] + 0.35 *
  (5/10) * E[retorno_10 | probabilidades B]
  - 0.40 * p_queda * E[severidade | queda]`.

As médias condicionais B são calculadas usando **somente resultados
já maturados antes do período de teste**. A estimativa de retorno
B não é um segundo bônus somado integralmente a retorno já previsto.

Para substituir a escolha do Control, exigir vantagem incremental
estimada pelo menos `0.005` (50 pontos-base em log-retorno
aproximado) após o **custo adicional** estimado da transição
(`0.003`, 30 pontos-base como aproximação fixa). A execução financeira
usa depois as taxas reais do simulador Control. O modelo respeita o
bloqueio de permanência mínima do Control, não tenta trocar quando
seu modelo não tem dados e não reinicia o estado da carteira.

**CASH é o último recurso**: somente quando não existe nenhuma
oportunidade com retorno ajustado positivo e a escolha do Control
tem retorno projetado igual ou inferior a `-0.010`. Sem cobertura
de 90% das oportunidades elegíveis, a decisão original é preservada.
CASH continua sujeito à reavaliação diária, sem permanência fixa.

Todos esses coeficientes são **pré-declarados para este novo teste**
e não podem ser alterados com base nos resultados históricos.
O custo de 30 pontos-base é um filtro de seleção aproximado,
não uma reprodução individualizada das taxas executadas.

## Validação cronológica

O rótulo do modelo de retorno `t+5` só é usado no treino quando
`t+5` é anterior ao primeiro teste do fold. O mesmo vale para os
rótulos de recuperação `t+10`. Nenhum atributo da decisão consulta
o retorno futuro. Se faltam preços ou há qualquer lacuna na janela
de resultado, o exemplo fica sem alvo, não se reconstrói um preço.

A **paridade exata de 1.576 sessões** com o ZIP oficial Strategy #30,
inclusive posições, folds e patrimônio ao centavo, é obrigatória
antes dos dois replays experimentais.

A avaliação continua exploratória porque as grandes perdas de 2026
já orientaram as hipóteses. O motor original possui uma dependência
conhecida de disponibilidade de preço de `t+1` na elegibilidade:
reproduzir o Control não elimina essa limitação causal. Uma decisão
de promoção exige outro motor livre da limitação e teste posterior
não utilizado para a concepção dos modelos.

## Como executar no Windows / Git Bash

```bash
git status --short
git fetch origin
git switch --track origin/research/u75-economic-rotation-recovery-v1
python -m unittest discover -s tests -p "test_vantagem_economica_rotacao_u75.py" -v
```

Se a branch local já existir, use `git switch
research/u75-economic-rotation-recovery-v1` e
`git pull --ff-only origin research/u75-economic-rotation-recovery-v1`.

No Spyder (F5), com `dados/strategy30_snapshot/` e o ZIP oficial
Strategy #30 no local esperado:

```python
%run pesquisar_vantagem_economica_rotacao_u75_spyder.py
```

Nenhum dado é baixado e nada é escrito no ambiente operacional.
O programa gera um pacote com SHA-256 em:

```text
output/vantagem_economica_rotacao_u75/
  pacote_vantagem_economica_rotacao_u75_<UTC>.zip
  <UTC>/
    protocolo_cientifico.json
    auditoria_snapshot.csv
    predicoes_risco_reversao.csv
    treino_modelos_folds.csv
    metricas_preditivas_por_fold.csv
    estimativas_economicas_ativos.csv
    treino_economico_por_fold.csv
    margens_control_por_fold.csv
    auditoria_paridade.json
    auditoria_paridade_diaria.csv
    control_predictions.csv
    control_trades.csv
    economico_risco_decisoes.csv
    economico_risco_predictions.csv
    economico_risco_trades.csv
    economico_reversao_decisoes.csv
    economico_reversao_predictions.csv
    economico_reversao_trades.csv
    comparacao_dia_a_dia.csv
    maiores_diferencas_pregao.csv
    patrimonio_tres_politicas.png
    drawdowns_tres_politicas.png
    resultado.json
    resumo.md
```

Se falhar, a execução grava um ZIP `failed` com `erro_execucao.json`.
As decisões auditáveis mostram a escolha original, proposta nova,
motivo, retorno econômico previsto, risco, possível recuperação,
custo incremental e cobertura do modelo.

## Critérios de análise

Primeiro, verificar se **as rotações originais com fortes ganhos
permanecem acessíveis**. Depois, comparar episódio por episódio
a perda evitada, a oportunidade abandonada, a hora dos avisos,
o desempenho por fold, ES95, drawdown, Sharpe, patrimônio,
quantidade de rotações e custo. A política de recuperação precisa
efetivamente produzir decisões distintas, se as previsões trouxerem
vantagem suficiente; caso contrário, registrar ausência de contribuição,
sem relaxar os critérios retroativamente.

O estudo de descoberta de novos ativos para datas futuras continua
separado em `pesquisar_catalogo_ativos_futuros_spyder.py`; nenhum novo
ativo pode entrar retrospectivamente no U75 congelado.
