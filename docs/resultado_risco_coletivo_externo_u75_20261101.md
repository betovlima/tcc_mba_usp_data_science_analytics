# Auditoria do piloto matemático U75: risco coletivo e escolhas externas condicionadas

**Versão documental:** `u75-risk-gated-external-oos-audit-v1.0.0`  
**Executor:** `u75-risk-gated-external-v1.0.0`  
**Pacote recebido:** `pacote_risco_coletivo_externo_u75_20261101T200501Z.zip`  
**SHA-256 do pacote:** `7b9e52ab6b5056ebfe2701b44cae69c579a66c0a4077f12384ebba0295c89d20`  
**Estado:** `completed`, TCC apenas. Nenhuma alteração no MCT, Control, snapshots, decisão operacional, treinamento ou modelo.

## Pergunta de pesquisa

É possível antecipar a deterioração coletiva de um universo U75 pela covariância robusta, turbulência de Mahalanobis, PCA e medidas transversais de risco; e, nos momentos alertados, encontrar externamente uma alternativa preferível ao Top-1, apenas com dados conhecidos em t?

**Não** otimizar o capital histórico de US$ 587 milhões ou outro checkpoint; não interpretar medidas do índice equal-weight como perdas da posição efetiva.

## Identidade dos dados

- U75 congelado conforme Strategy #30. Decisões totais do backtest MCT: 1.578; ZIP `market_cycle_trader_20261101T021336-83e49685.zip` e SHA-256 `6c0f55543734250168f90ca2d04db6d7a39a1a14e8e952e7400aa8efd9adfebe`.
- Snapshot base: `1bbb19790c49d7066d369556c884faba66ba3f7fb42361ab5ce4755d8be67033`.
- Shortlist externa: 24 ativos, snapshot `18dbaf4f1529d6952052cac25379fc9f8f6486e59f575861173d2f01c07a3225`.
- Estatística de cauda: limiar inferior de **5% dos retornos logarítmicos futuros de cinco pregões** do índice equal-weight, recalibrado **somente no treino de cada fold**. O evento futuro é rótulo de validação, não sinal conhecido no fechamento.
- Ledoit-Wolf: janela de 126 pregões anteriores; análogos padronizados `ceil(sqrt(N))`; testes OOS anuais com apenas rótulos maturados anteriormente.
- 1.210 datas de decisão OOS no arquivo, 1.205 com retorno futuro observado; cinco datas no fim da série ainda não tinham cinco sessões de maturação do alvo.

## Detecção fora do treinamento

| Item | Resultado |
|---|---:|
| Sessões mensuráveis | 1.205 |
| Eventos efetivos de cauda | 66 |
| Alertas emitidos e mensuráveis | 130 |
| Verdadeiros positivos | 4 |
| Falsos positivos | 126 |
| Falsos negativos | 62 |
| Verdadeiros negativos | 1.013 |
| Precisão dos alertas | **3,077%** |
| Sensibilidade (recall) dos eventos | **6,061%** |
| Frequência base de eventos | 5,477% |
| Brier da probabilidade condicional (menor melhor) | **0,054132** |
| Brier da previsão incondicional do treino | **0,051599** |
| AUC ROC | **0,4633** |
| Average precision (AUPRC) | **0,0681**, frequência base 0,0548 |

O Brier ficou **0,002533 pior** que a frequência histórica como previsão. Estimativa exploratória por 5.000 reamostragens de **58 blocos mensais**, seed 42, para `Brier_modelo - Brier_baseline`: **[+0,000305; +0,004451]**. Com eventos futuros sobrepostos em horizontes de cinco dias, os blocos mensais não são independentes estritamente; esse é diagnóstico, não inferência confirmatória.

| Ano | Sessões com rótulo | Alertas | Eventos | Acertos | Falsos alertas |
|---|---:|---:|---:|---:|---:|
| 2022 | 251 | 2 | 46 | 2 | 0 |
| 2023 | 250 | 21 | 3 | 0 | 21 |
| 2024 | 252 | 44 | 9 | 2 | 42 |
| 2025 | 250 | 12 | 8 | 0 | 12 |
| 2026 | 202 | 51 | **0** | 0 | 51 |

Os alertas corretos em datas são **14/01/2022, 04/04/2022, 29/07/2024 e 30/07/2024**. Os dois últimos fazem parte do **mesmo episódio temporal** de retornos futuros sobrepostos; não tratá-los como quatro quedas coletivas independentes.

Em 2022, com limiar treinado até 2021 de **−3,262% em log-retorno de cinco pregões**, houve 46 eventos: a base de cauda era maior que os ~5% esperados do histórico. Em 2026, o limiar aprendido era **−4,052%**, mas não houve evento desse tamanho na série de índice equal-weight. Isso é uma indicação de mudança de distribuição entre anos e falta de estabilidade/calibração da previsão.

Indicadores médios durante **eventos efetivos** vs **não eventos**:
- Participação dos ativos em queda no retorno de 20 pregões: 55,86% vs 47,46%;
- Participação da primeira componente principal na variância: 25,99% vs 19,58%.
- Curiosamente, nas datas em que **o detector alertou**, a participação média da primeira componente era só 15,17%, contra 20,51% nas demais; diferença descritiva com confundimento temporal, sem causalidade demonstrada.

## Escolhas externas condicionadas aos alertas

Dos 130 alertas, 124 possuem resultado intradiário da alternativa pesquisada; **6 datas de 07/10 a 16/10/2026** receberam `no_external_snapshot`. O snapshot externo termina em 06/10/2026; **não reconstruir** nem inventar a trajetória desses seis sinais.

| Resultado | Quantidade |
|---|---:|
| `EXTERNAL` | 120 |
| `TOP1` | 4 |
| `CASH` | 0 |
| Dados externos ausentes | 6 |
| Superou o Top-1 em log-retorno intradiário seguinte | 67 / 124 |
| Perdeu | 53 / 124 |
| Empatou | 4 / 124 |
| Diferença média intradiária | **+0,00162769 log = +0,1628 ponto percentual** |
| Diferença mediana intradiária | **+0,002335 log = +0,2335 ponto percentual** |

O intervalo exploratório de 95% para a média, reamostrando 38 meses em 5.000 repetições com seed 42, é **[−0,9908; +1,4006] pontos percentuais de log-retorno por sessão**. Portanto não há vantagem estatística robusta demonstrada.

Diferenças médias em pontos percentuais de log-retorno por ano, nos alertas com outcome:

| Ano | N | Vantagem intradiária |
|---|---:|---:|
| 2022 | 2 | −0,339 |
| 2023 | 21 | −0,594 |
| 2024 | 44 | −1,143 |
| 2025 | 12 | +1,132 |
| 2026 | 45 | +1,556 |

**Nos quatro alertas verdadeiros**, a seleção externa apresentou **−5,7779 pontos percentuais de log-retorno intradiário médio** contra o Top-1. Os casos são:

| Data | Top-1 | Externo | Diferença de log-retorno |
|---|---|---|---:|
| 14/01/2022 | PENN | MLPA | +2,481 p.p. |
| 04/04/2022 | NFLX | RCMT | −3,158 p.p. |
| 29/07/2024 | MUX | EBS | **−21,864 p.p.** |
| 30/07/2024 | AVGO | EBS | −0,570 p.p. |

São somente quatro datas e incluem resultados futuros de cinco sessões sobrepostos; não é inferência estável sobre o benefício de proteção.

## Diagnóstico técnico e limitações

1. Este é um estudo **P1/shadow**, não a execução de uma política completa de carteira. O retorno da sessão seguinte de abertura a fechamento **não é** o ganho realizado com posições, custos, overnight, entrada em CASH ou reentrada.
2. A regra de alerta usou comparação unilateral aproximada de probabilidades com análogos históricos; não garante taxa nominal de erro sob rótulos sobrepostos e autocorrelação. O cálculo de ES95 por k vizinhos, com k=20 a 37, usa aproximadamente **uma ou duas observações na cauda** e tende a ser instável.
3. O detector mediu a cauda negativa do **índice diário equal-weight dos 75 ativos**, não da carteira concentrada no Top-1. O problema de proteção coletiva deve permanecer separado da exposição específica do ativo escolhido.
4. O Top-1 é `raw_best_asset` do backtest MCT já congelado; elegibilidade dependente de dados de t+1 permanece limitação conhecida. Mesmo um seletor externo causal não elimina automaticamente essa limitação.
5. Os 24 candidatos externos são shortlist histórica e não descoberta prospectiva do mercado inteiro. A pontuação simples `return_20/(sqrt(20)*vol_20)` praticamente sempre elegeu externo em alerta (120/124), mas não constitui previsão de vantagem ou redução condicional de ES.
6. Comparar diretamente os períodos de alerta com os grandes drawdowns CAPR ou da posição efetiva exige medir ambos no mesmo calendário, evitando atribuir perda idiossincrática ao detector coletivo.

## Decisão científica

**Não promover o detector, o ranking externo de momentum ou qualquer decisão CASH para MCT. Não ajustar limiares à amostra OOS observada.** Não concluir que covariância robusta, Mahalanobis ou PCA são inúteis: apenas esta combinação/kNN/critério de alerta não antecipou o risco desejado.

Próxima investigação, antes de novo replay financeiro:

- Avaliar risco coletivo real em **horizontes múltiplos e episódios** (1, 5, 10 e 20 pregões), com ponto de entrada da queda medido explicitamente, inclusive o período de 2022;
- Comparar processos matemáticos independentes: **CUSUM** para alteração sequencial, **Markov filtrado** para regimes e **ES condicional** com amostra histórica suficiente; recalibrar e validar ano a ano sem consultar os rótulos de teste;
- Testar incrementalmente covariância shrinkage, coexcedências de cauda, breadth e PCA como antecedentes, não só contemporâneos. Não escolher pesos ou limites depois dos resultados;
- Definir escolha externa pelo benefício condicional de risco/retorno vs **Top-1 e posição efetiva** (inclusive CASH), com dados de universo disponível em t e validação aninhada. O candidato não é bom apenas por ter momentum alto;
- Quando houver evidência de detecção e de vantagem externa, reproduzir política P0–P3 com custos/overnight e medir perdas efetivamente evitadas, ES95, drawdown e oportunidades perdidas. Capital terminal é secundário.

Relatório apenas documental e reproduzível; não se alega novo resultado financeiro além dos números conferidos no pacote.
