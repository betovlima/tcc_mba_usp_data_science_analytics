# Resultado da auditoria FIX-004: contribuição marginal de episódios

<!-- [TCC-DFL:RESULT-002] -->
Data: 27/09/2026. Branch única: `research/decision-focused-oof-v2`.
Versão de trabalho: `v1.3.0-dev.3`; **sem nova versão/branch/tag**.
Origem: ZIP `decision_focused(3).zip` fornecido pelo pesquisador;
SHA-256 `c471ed2387e6f55efa1f2bc03089d2cc1688a67c26bb66a0bdbe8f6274246955`.
Workflow que executou o código da auditoria:
https://github.com/betovlima/tcc_mba_usp_data_science_analytics/actions/runs/36349759992
Código do run: `649b59a8287fdc080845832623740e18f6e3d841`;
conclusão verificada: SUCCESS, inclusive testes, verificação de snapshot,
backtest e upload. Todos os números abaixo derivam dos arquivos
`decision_focused/active/` dentro do ZIP, especialmente
`intervention_reconciliation.json`,
`intervention_episode_attribution.csv`,
`intervention_episode_daily_pairs.csv`,
`intervention_bridge_trades.csv` e `summary.json`.

## Validação da execução

- Control: US$ 10.082.425,91.
- Soft: US$ 9.840.028,41.
- Regressão OOF: US$ 7.639.150,63.
- DFL pareado: US$ 2.105.511,08.
- DFL softmax por decisão: US$ 4.716.464,39.
- 3 episódios, 6 datas de override em 1.547 decisões.
- Replay BASELINE vs Control oficial: diferença máxima absoluta de curva **US$ 0**.
- Bridge final EP003 vs regressão original: diferença máxima absoluta **US$ 0**.
- Soma dos efeitos marginais: **−US$ 2.443.275,28**.
- Diferença final regressão menos Control: **−US$ 2.443.275,28**.
- Erro de reconciliação: **US$ 0**.

### Atribuição cronológica, condicional aos episódios anteriores

| Episódio | Decisão original | Sessões override | Delta após 20 sessões | Contribuição até final | Fração da perda total | Diferença final relativa ao cenário anterior |
|---|---|---:|---:|---:|---:|---:|
| EP001 | 19/01/2023: VNCE -> YANG | 1 | −US$ 1.499,79 | −US$ 95.603,28 | 3,91% | −0,9482% |
| EP002 | 08–11/10/2024: mantém YANG vs MYE | 4 | −US$ 12.719,65 | −US$ 96.989,26 | 3,97% | −0,9712% |
| EP003 | 14/01/2026: mantém CORT vs AVGO | 1 | −US$ 1.412.175,96 | −US$ 2.250.682,75 | 92,12% | −22,7575% |

A soma dos efeitos é telescópica: BASELINE -> EP001 -> EP002 -> EP003.
Não interpretar cada número como efeito independente ou causal; mudar a
ordem das intervenções pode modificar o respectivo efeito marginal.

Capitais sucessivos dos quatro cenários:
BASELINE US$ 10.082.425,91;
EP001 US$ 9.986.822,64;
EP002 US$ 9.889.833,38;
EP003 US$ 7.639.150,63.

## Anatomia EP003: uma decisão local muda outras ações

1. Em 14/01/2026 o Control escolheu AVGO a partir de CORT; a regressão
   manteve CORT (único override local).
2. Execução 15/01: cenário sem intervenção já tem AVGO; com intervenção,
   CORT. O delta de capital é **−US$ 120.146,17**.
3. Execução 16/01: ambos passam a AVGO; a diferença é
   **−US$ 165.732,81** (−3,56% relativamente ao cenário sem episódio).
   Esse PRIMEIRO dia no mesmo ativo NÃO significa reconvergência da
   trajetória: capital, quantidade e tempo de posição continuam distintos.
4. Execuções 20–21/01: Control contrafactual volta para CORT, mas a
   regressão fica em AVGO por consequência do estado/tempo de posição.
   Deltas de capital: −US$ 466.175,26 e −US$ 650.737,76.
5. Execução 22/01: cenário sem episódio vende CORT e registra
   P&L realizado de **+US$ 986.548,86**, voltando a AVGO;
   cenário com episódio não participou desse trecho em CORT.
   Delta diário de capital no fechamento: **−US$ 1.223.153,86**.
6. Após isso, ambos podem carregar o MESMO ativo nominal mas bases de
   capital/quantidades diferentes. Em 20 sessões o efeito acumula
   −US$ 1.412.175,96, chegando a −US$ 2.250.682,75 no fim do replay,
   pelo composto sobre capitais distintos.

O evento não é simplesmente a diferença de retorno de CORT e AVGO
em 15/01. O override deslocou o relógio de holding, a rotação de 20/01
e a exposição à alta de CORT de 20–22/01. A característica relevante
para estudos futuros é **estado da carteira e decisões subsequentes**.
Não afirmar que o modelo previu um evento específico ou que o efeito
seria reprodutível fora deste histórico.

As diferenças de fees por episódio foram −US$ 120,47, −US$ 116,06
e −US$ 1.816,85, medidas sobre a trajetória COMPLETA; fees menores
podem refletir menor capital negociado e não demonstram eficiência
de execução. Nenhum cenário passou tempo em CASH nesta auditoria.

## Limite da auditoria e decisões científicas

A coluna `first_same_asset_after_start` denota apenas o **primeiro
reencontro nominal do ativo**, não a reconvergência permanente do estado
ou da riqueza. Em EP003, por exemplo, a igualdade nominal de 16/01
é interrompida por nova diferença de exposição em 20–21/01.
O capital nunca se equalizou.

O resultado foi analisado após outros experimentos nesses folds. Os
folds atuais são desenvolvimento, não teste confirmatório intocado.
A descoberta apoia a investigação de robustez de intervenções a
trajetórias/holding, mas NÃO autoriza ajustar novos thresholds neste
mesmo OOS para apresentar ganho estatisticamente independente.
Dados congelados, modelos e versão permanecem sem alterações.
Próximo passo: definir uma avaliação futura independente antes
de eventual mudança de política ou protocolo.
