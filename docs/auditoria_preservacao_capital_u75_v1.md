# Auditoria U75 de preservação de patrimônio, v1

**Status:** avaliação descritiva e reprodutível, sem alteração de política e sem autorização de promoção.  
**Branch:** `research/u75-capital-preservation-audit-v1`.  
**Fonte da execução:** `market_cycle_trader_20261101T005056-0acb4c89.zip`, SHA-256 `6eeafa90937b501f66356c3d1e6019bbde342430d92905570d32b4bcffa2228e`.  
**Período analisado:** 22/07/2020 a 30/10/2026, 1.578 sessões.  
**Proveniência:** contrato `u75-daily-market-protection-v1` e flag `u75_daily_cash_protection_enabled=true` no metrics do próprio pacote. O nome do diretório interno usa o motor `tcc_u77_fixed_challenger`, não deve ser confundido com um universo efetivo de 77 ativos; os dados registram 75 ativos elegíveis.

## 1. O que preservação significa neste estudo

Separar quatro resultados:

1. **Manutenção do principal:** o patrimônio nunca caiu abaixo de US$ 10 mil nos pregões observados, mas isso não é garantia para outras trajetórias.
2. **Manutenção dos picos:** falhou em evitar perdas muito grandes em relação ao topo anterior.
3. **Risco de perdas extremas:** avaliar distribuição de retornos diários, especialmente Expected Shortfall (ES) empírico de 95%, calculado como média dos retornos abaixo do percentil 5 da série. Essa métrica é **descritiva, não uma trava operacional**.
4. **Efeito causal de CASH:** ainda não demonstrado neste pacote. O benchmark `buy_hold_equity` não é a política Control de rotação sem overlay U75.

## 2. Resultados observados

| Indicador | Rotação U75 com proteção | Referência do mesmo pacote |
|---|---:|---:|
| Capital inicial | US$ 10.000,00 | US$ 10.000,00 |
| Capital final | US$ 76.710.842,01 | US$ 29.900,99 |
| Sharpe registrado | 2,3035 | 0,9496 |
| Drawdown máximo | −41,68% | −26,05% |
| ES95 empírico por sessão (retorno dos piores 5%) | −7,80% | −2,93% |

Patrimônio máximo: US$ 102.034.355,76 em 30/09/2026; fechamento final estava **24,82% abaixo** desse topo. As 1.578 sessões registraram **163 sinais de estresse coletivo**, **2 overrides U75 de CASH** e **2 sessões efetivas em CASH** (17/06/2022, 19/04/2024). Em 2026, as 209 sessões ficaram investidas, sem nenhum dia em CASH. Das dez piores sessões de retorno da carteira, apenas uma tinha `u75_market_stress=true`.

### Principais drawdowns (pico até fundo, e recuperação se observada)

| Pico | Fundo | Perda | Ativo no fundo | Estresse U75 no fundo | Recuperação |
|---|---|---:|---|---|---|
| 27/07/2026 | 31/07/2026 | −41,68% | CAPR | Não | 14/08/2026 |
| 04/12/2024 | 03/03/2025 | −36,92% | AMD | Não | 26/03/2025 |
| 30/09/2026 | 26/10/2026 | −32,36% | CAPR | Não | Não recuperado até 30/10/2026 |
| 02/02/2022 | 11/05/2022 | −31,57% | VRTS | Sim | 11/08/2022 |
| 26/09/2024 | 07/10/2024 | −30,36% | YANG | Não | 09/10/2024 |

O retorno diário de 30/07/2026 foi −36,23%, enquanto o modelo manteve CAPR, `u75_market_stress=false` e 30 candidatos técnicos. Esse movimento pode ser idiossincrático, e não constitui evidência de colapso sistêmico da cesta U75.

## 3. O que esta auditoria **não** prova

- Lucro acumulado ou Sharpe alto não prova controle satisfatório do drawdown máximo.
- As duas sessões de CASH não demonstram perdas evitadas, sem comparação financeira contrafactual Control e Overlay na **mesma série, janelas, execução, calendário, universo, dados e custos**.
- Uma trava **exclusivamente sistêmica** não pode ser considerada defeituosa só por não evitar uma queda isolada de CAPR. Se o objetivo for proteger o patrimônio concentrado, a gestão do risco da seleção/rotação do próprio U75 precisa ser estudada separadamente do CASH coletivo, sem introduzir um stop arbitrário para um ativo.
- O período já foi examinado: escolher novos limites ou modelos usando julho/outubro de 2026 e depois reportar acerto nesse mesmo período é viés de seleção. Walk-forward interno não o elimina automaticamente.
- Regras de elegibilidade que dependem da sessão seguinte, previamente encontradas no motor congelado, precisam permanecer visíveis como limitação científica até a análise de sensibilidade ser concluída. Não misturar uma correção desse tema com a política de CASH.

## 4. Critério para a próxima etapa

**Hipótese principal:** existe um sinal causal de risco agregado do U75, conhecido no fechamento t, que prevê uma perda coletiva relevante em t+1 e que melhora a *utilidade ajustada ao risco* após custos de sair/reentrar.

Usar os artefatos científicos já existentes em `pesquisar_diagnostico_u75_spyder.py`, `pesquisar_optimal_switching_cash_spyder.py` e os estudos contrafactuais, sem reimplementá-los. No TCC, comparar o Control congelado com a proteção U75 v1.0.1 e somente depois testar uma nova variável independente baseada em Expected Shortfall condicional da cesta de 75 ativos e, se justificar, covariância robusta, turbulência e correlação negativa. As decisões devem ser feitas em t e executadas na abertura t+1. Preservar as políticas atuais intactas, sem duração fixa de CASH.

**Análises obrigatórias:** delta de patrimônio entre Control e Overlay, drawdown, ES95 de retornos da carteira, duração de recuperação, valor sacrificado nas saídas desnecessárias, perdas efetivamente evitadas e resultados pareados por período. Escolher parâmetros somente nos treinos internos; avaliar em teste temporal separado, com incerteza por bootstrap em blocos e correção por múltiplos testes quando cabível.

**Escopo operacional:** MCT permanece intocado. Esta auditoria não usa Mongo, não consulta Alpaca, não reconstrói séries e não executa ordens. O backtest novo repetido pelo usuário deve ser analisado **somente após recebermos seu próprio ZIP**, não por suposição de que reproduziu o resultado anterior.
