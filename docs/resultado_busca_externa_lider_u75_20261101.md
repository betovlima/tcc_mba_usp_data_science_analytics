# Resultado observado: busca externa por assinatura do líder U75

**Versão do relatório:** `u75-external-leader-oos-results-v1.0.0`.  
**Executor:** `u75-external-leader-search-v1.0.2`.  
**Pacote recebido:** `pacote_busca_externa_lider_u75_20261101T033836Z.zip`.  
**Estado:** `completed`.  
**Escopo:** exclusivamente TCC; nenhuma alteração no Control, no MCT ou na estratégia operacional. **Não é backtest de patrimônio.**

## Origem e cobertura

- Backtest MCT: `market_cycle_trader_20261101T021336-83e49685.zip`, SHA-256 `6c0f55543734250168f90ca2d04db6d7a39a1a14e8e952e7400aa8efd9adfebe`, 1.578 registros.
- Universo de referência: **75 ativos**; líder = `raw_best_asset` da execução registrada; não equivale necessariamente à posição efetiva.
- Snapshot Control: `market_cycle_trader_20261029T010353-f60d8486.zip` e identidade `1bbb19790c49d7066d369556c884faba66ba3f7fb42361ab5ce4755d8be67033`.
- Snapshot externo: `18dbaf4f1529d6952052cac25379fc9f8f6486e59f575861173d2f01c07a3225`. Shortlist previamente congelada de **24 candidatos**, todos fora dos 75 ativos.
- `decision_coverage.csv`: 1.578 decisões; 1.561 `ok`, 16 `no_external_candidates` (07 a 28/10/2026), 1 `insufficient_u75_reference` (29/10/2026). Não interpolar, reconstruir ou extrapolar dados externos após 06/10/2026.
- `candidate_pairs.csv`: 37.464 combinações de data/candidato. Avaliação OOS: 1.194 sessões, anos 2022–2026, **última decisão em 06/10/2026**, última execução em 07/10/2026.
- Folds anuais com expanding-window: treino de datas/rótulos maturados antes da primeira decisão de teste, 2022, 2023, 2024, 2025 e 2026.
- Modelo: regressão Ridge com `ridge_alpha=1.0`, 12 atributos relativos e distância. Custo de referência `reference_cost_bps=0`, isto é, **a avaliação não contabiliza custos reais**.

## Resultados fora do treino

O alvo é **diferença de log-retorno abertura→fechamento da sessão de execução** entre candidato e líder. Os valores abaixo representam diferenças médias de retornos por pregão em pontos percentuais aproximados, **não a variação de patrimônio e não uma estratégia composta**.

| Política de escolha | Média diária | Mediana diária | Acertos com excesso positivo | Observações |
|---|---:|---:|---:|---:|
| Líder U75, sem substituição | 0,00 p.p. | 0,00 p.p. | Referência | 1.194 |
| Externo mais próximo | **−0,5107 p.p.** | −0,4068 p.p. | 527/1.193, **44,17%** | 1.193 |
| Ridge com opção de manter líder | **−0,3798 p.p.** | 0,00 p.p. | 399/1.194, **33,42%** | 1.194 |

IC95% exploratório da média por bootstrap mensal (repetições: 1.000, seed 42):
- Proximidade: **[−0,7477; −0,2987] p.p.**
- Ridge: **[−0,6687; −0,0931] p.p.**

Os intervalos são exploratórios e refletem o procedimento de reamostragem mensal do script; não incorporam a incerteza da seleção retrospectiva da shortlist, nem eliminam outros possíveis vieses do protocolo.

### Quebra temporal: média diária em pontos percentuais

| Ano | Sessões | Proximidade | Ridge | Ridge escolheu externo |
|---|---:|---:|---:|---:|
| 2022 | 251 | −0,7332 | −0,3828 | 193 |
| 2023 | 250 | −0,6334 | −0,9458 | 187 |
| 2024 | 252 | −0,2811 | +0,1185 | 159 |
| 2025 | 250 | −0,4483 | −0,3988 | 188 |
| 2026 até 06/10 | 191 | −0,4416 | −0,2677 | 127 |

A vantagem pequena de 2024 depende de outlier: **07/08/2024, EBS vs LLY, vantagem realizada +0,42933 em log-retorno**. Excluindo somente esse dia, a média anual Ridge passa de +0,1185 p.p. para aproximadamente **−0,0520 p.p.**. A exclusão é apenas **diagnóstico ex post**, não regra de negociação.

### Comportamento do modelo e limite superior retrospectivo

- Em **854/1.194 = 71,52%** dos dias o Ridge recomendou substituir o líder.
- Nessas 854 trocas, a vantagem média **prevista** foi **+0,3422 p.p.**, mas a média **realizada** foi **−0,5310 p.p.**; 399/854 = **46,72%** superaram o líder.
- Correlação de Pearson entre o score previsto selecionado e a vantagem realizada nesses dias: aproximadamente **−0,0068** (próxima de zero).
- Ridge, em média, foi 0,1305 p.p./sessão menos negativo que a proximidade na comparação pareada, mas o IC95% por blocos mensais para essa diferença é aproximadamente **[−0,1625; +0,4230] p.p.**; não permite concluir superioridade do Ridge sobre a proximidade com segurança.
- Em 1.113/1.193 dias (**93,29%**) existiu algum candidato com retorno de abertura→fechamento maior que o líder. A vantagem média do **melhor conhecido após a execução** foi **+6,2705 p.p.**. Esse `oracle_delta` é um limite superior **ex post**, não reproduzível como estratégia causal e não pode ser usado para seleção/calibração no mesmo teste.

## Conclusões sustentadas

1. A hipótese **"semelhança com o líder implica melhor retorno na sessão seguinte"** não foi sustentada nesta shortlist.
2. A hipótese **"Ridge linear com 12 atributos relativos prevê vantagem suficiente para substituição"** não foi sustentada nesta shortlist.
3. A regressão seleciona muitas trocas apesar da expectativa superestimada de vantagem; é especialmente importante medir calibração e comparar contra não intervir.
4. A existência retrospectiva de ativos melhores em diversos pregões não equivale à capacidade de escolhê-los antecipadamente.
5. O protocolo **não mede proteção de patrimônio**, drawdown, exposição overnight, preços de abertura de execução após decisão, permanência, liquidez, custos reais nem efeito sequencial das rotações. **Não calcular patrimônio composto a partir de `model_delta`.**
6. Os 24 candidatos provêm de shortlist histórica previamente selecionada usando a janela completa; mesmo com folds anuais, **esta não é validação prospectiva independente da descoberta externa**.
7. Os dados do líder vêm de snapshot/backtest MCT e os preços comparativos de snapshots TCC com cortes diferentes. A auditoria de dependência futura na elegibilidade original continua pendente.

## Próxima decisão científica

**Rejeitar promoção operacional desta hipótese na forma atual.** Não escolher novas configurações pelo desempenho OOS já observado; não aumentar arbitrariamente o pool nem o número de trials para buscar um vencedor.

Antes de qualquer novo ranking externo, definir um experimento independente com:
- descoberta do universo externo **anterior ao início de cada fold**, com snapshots e calendário causais;
- avaliação da vantagem sobre a **posição efetiva**, além do líder bruto, e alvo de episódio da política completa (não apenas intradiário);
- custos, liquidez, slippage, condições da próxima abertura e risco da trajetória;
- controle fixo sem substituições, e avaliação pareada em janelas posteriores ainda não utilizadas;
- comparativo de drawdown máximo, Expected Shortfall, patrimônio final e testes de seleção múltipla.

O estudo sobre risco sistêmico U75 e o estudo sobre expansão externa são perguntas diferentes. Uma shortlist externa não deve ser assumida como hedge de CASH ou preservação de patrimônio sem teste de correlação, risco e política completa.
