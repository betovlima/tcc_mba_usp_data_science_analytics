# Auditoria do experimento contextual CASH, 02/11/2026

**Fonte única dos números:** `pacote_vantagem_cash_contextual_u75_20261102T092523149278Z.zip`, fornecido pelo pesquisador, gerado pelo executor `u75-contextual-cash-advantage-v1.0.0`.

**Integridade:** ZIP testado sem erros, 24 arquivos; `resultado.json` e `andamento_atual.json` com status `completed`. Control Strategy #30 reproduzido com paridade de 1.576/1.576 sessões, zero divergências, 75 ativos, 3 folds, patrimônio de US$ 80.685.454,78647688.

## Desempenho

| Cenário | Patrimônio final | Dif. vs Control | MaxDD | ES95 diário | CASH | Episódios |
|---|---:|---:|---:|---:|---:|---:|
| Control | US$ 80,685 mi | 0% | -41,68% | -7,79% | 0 | 0 |
| Contexto mercado | US$ 37,194 mi | -53,90% | **-53,75%** | -7,63% | 88 | 64 |
| Contexto mercado+ativo | US$ 38,102 mi | -52,78% | -41,68% | **-7,56%** | 84 | 62 |
| Contexto mercado+ativo+forma | US$ 23,932 mi | -70,34% | -41,68% | -7,61% | 94 | 68 |

Os 3 cenários mantiveram perda de -36,2253% no pior pregão, CAPR em 30/07/2026. Nenhuma política reduz o MaxDD original de forma material; a variante exclusivamente coletiva piorou esse indicador em aproximadamente 12 pontos percentuais.

## Diagnóstico dos sinais de cinco pregões

Leitura dos arquivos `<cenario>_avaliacao_futura.csv`, rotulados somente **após** a execução. Retorno futuro: log-retorno do ativo escolhido entre aberturas de `t+1` e `t+h+1` com `h=5`; não equivale ao resultado de carteira que revisa posições todo dia.

| Cenário | Alarmes com desfecho | Quedas entre alarmes | Taxa de quedas nos alarmes | Taxa de quedas sem alerta | Recall das quedas | Média de log-retorno futuro nos alarmes |
|---|---:|---:|---:|---:|---:|---:|
| Mercado | 85 | 31 | 36,47% | 41,99% | 4,73% | +5,25% |
| Mercado+ativo | 83 | 37 | 44,58% | 41,94% | 5,60% | +0,29% |
| Mercado+ativo+forma | 93 | 45 | 48,39% | 41,27% | 6,87% | -0,43% |

A proteção é fragmentada: **52 de 64**, **46 de 62** e **51 de 68** episódios CASH duraram **um único pregão**, respectivamente. Isso indica incompatibilidade prática entre o sinal estimado a cinco pregões e a regra de reavaliação/reentrada diária.

**Nenhum modelo merece promoção.** Mercado coletivo protege pior que o próprio padrão de quedas observado nos dias sem alarme; com DTW a concentração de episódios negativos melhora modestamente, mas o custo de perder altas produz capital final 70,34% menor.

## Episódios que já conhecíamos

As porcentagens são entre os fechamentos indicados, comparação sempre entre os mesmos limites:

| Intervalo | Control | Mercado | Mercado+ativo | Mercado+ativo+DTW |
|---|---:|---:|---:|---:|
| AMS 28/05–08/06/2026 | -20,73% | -19,57% | -20,73% | -20,73% |
| CAPR 27/07–31/07/2026 | -41,68% | -36,85% | -41,68% | -41,68% |
| CAPR 30/09–26/10/2026 | -32,36% | -28,55% | -21,19% | -21,19% |
| CAPR 29–30/07/2026 | -36,23% | -36,23% | -36,23% | -36,23% |
| CAPR 13–14/08/2026 | +57,96% | -13,19% | +57,96% | +57,96% |
| VNCE 22–28/12/2023 | +130,29% | +63,03% | +130,29% | +130,29% |

### Fato decisivo: saída antecipada e retorno prematuro

- No fechamento de **27/07/2026**, o cenário mercado recomendou CASH para CAPR, e ficou CASH em **28/07**; em seguida, **voltou a CAPR em 29/07** após o sinal se tornar falso, e continuou exposto na queda de **30/07 (-36,23%)**.
- Depois do choque, ficou CASH de **31/07 até 13/08**, mas reentrou em CAPR **na abertura de 14/08** (operação registrada com preço de US$ 7,66), perdendo a exposição anterior à forte recuperação. O Control ganhou +57,96% entre fechamentos 13 e 14/08, enquanto o cenário de mercado registrou -13,19%; as políticas tinham histórias distintas de posição e preço médio.
- De setembro a outubro, a política mercado+ativo e a variante com DTW diminuíram a queda **local** de CAPR, mas **não** o máximo global, determinado pelo episódio anterior de julho.
- O cenário mercado também ficou parcialmente fora da alta de VNCE em dezembro de 2023.

## Interpretação científica e próxima hipótese

Este experimento **não comprova previsão útil de crashes**. A questão mais relevante já não é simplesmente reconhecer a forma, a amplitude ou a volatilidade: o que determina o desempenho é se o risco detectado persiste durante o período em que CASH evitaria perdas **líquidas de abertura, reentrada e custos**, versus a oportunidade de ganhos.

Uma próxima hipótese pré-registrada poderia estudar política temporal coerente de permanência/reentrada ou estimar a vantagem **marginal de capital** da decisão, usando simulação contrafactual causal e futura, com exame de robustez. **Não ajustar depois de ver CAPR/VNCE e chamar a repetição de validação independente.**

Restrições mantidas: nunca misturar sistema operacional com pesquisa; Control é referência histórica imutável, incluindo limitação conhecida de elegibilidade t+1; não afirmar ausência de lookahead total nem generalização sem avaliação futura nova. Protocolo atual é piloto retrospectivo, tem rótulos sobrepostos e limite unilateral Gaussiano descritivo sem cobertura nominal garantida.
