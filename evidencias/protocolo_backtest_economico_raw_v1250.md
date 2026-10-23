# Pré-registro experimental — TCC v1.25.0: replay financeiro econômico sobre RAW

**Data:** 23/10/2026. **Branch única:** `research/v1.23.1-fresh-snapshot-20261022`. **PR:** #18 Draft. **Main protegida:** `4667d2ccee3f3441ee4d2b362ce734311cedd5e3`.

## 1. Pergunta da próxima execução

Qual é o capital final da política Control, com o **mesmo universo fixo de 75 ativos** e os mesmos parâmetros/protocolo walk-forward, sobre **um único snapshot RAW congelado e verificável**, quando eventos de splits são contabilizados prospectivamente na quantidade de ações e todos os retornos/atributos do aprendizado preservam coerência econômica?

Esta pergunta **não** resolve por si só se o capital observado é realizável no mercado ou se a escolha retrospectiva de universo/hiperparâmetros introduziu sobreajuste.

## 2. Contrato, sem alteração após olhar resultados

| Parâmetro | Valor |
|---|---|
| Executor | `validar_backtest_economico_raw_spyder.py`, v1.25.0 |
| Schema | `fixed75-raw-economic-custody-v1` |
| Snapshot imutável exigido | `asof_2026-10-22_*/universe_75` |
| SHA-256 do manifesto do snapshot | `595b7a03dc2c8978487975cd3554b1ea65608741893f13208938dc624cb273cc` |
| Fonte | Alpaca SIP, timeframe 1Day, Adjustment.RAW já preservado |
| Período final a comparar | até **06/10/2026**; não misturar pregões posteriores |
| Universe | exatamente 75; exclusões DOC e CLMT preservadas |
| Capital inicial | US$ 10.000 |
| LightGBM | configuração Control existente; sem retuning externo |
| Folds | três folds walk-forward existentes, com purge 60 sessões |
| Calibração | **somente a calibração temporal interna já prevista** nos folds, com margens candidatas já definidas |
| Frações de ações | permitidas como no simulador anterior; cash-in-lieu não implementado |
| Impacto de mercado | não simulado |
| Slippage configurado | 0 bps no contrato atual |
| Resultado financeiro deste teste | **ainda não existe**; dependerá do F5 local |

## 3. Mudanças arquiteturais, opt-in e isoladas

**A. Dados RAW são a fonte de negociação e custódia.** `prepare_model_frames(strict_raw=True, apply_split_adjustments=False)` faz leitura auditável, não reexpressa historicamente preços/volumes, nem remove barras inválidas silenciosamente. Os **150 CSVs locais** (75 OHLCV e 75 Corporate Actions) são hasheados antes e depois; qualquer mudança aborta.

**B. Atributos e rótulos são calculados em uma vista derivada em RAM, e jamais gravada como substituto do RAW.** Em `reproducao/visao_economica_splits.py`, a cada `ex_date` já efetiva, o preço futuro passa a ser expresso por ação-base com a quantidade acumulada; volumes são divididos pelo fator. Antes da `ex_date`, a vista não usa eventos futuros. O construtor original de 52 features e dos 15 alvos de utilidade é reutilizado sobre essa vista econômica. Nenhum rótulo futuro é usado como feature atual, e o purge já foi auditado separadamente.

**C. Calibração interna usa a mesma vista econômica de preços**, para que margens não sejam escolhidas a partir de perdas/ganhos fictícios em desdobramentos. O protocolo de candidatos de margem é o pré-existente, não uma nova pesquisa de margens.

**D. Replay executa ordens com preços RAW e corrige a custódia pelo fator de quantidade APÓS a `ex_date`.** Antes da venda na sessão de ex-date, quantidade é multiplicada pelo fator, preço médio e extremos de diagnóstico em unidades de ação são divididos pelo fator. Comprar na própria ex-date não aplica fator pela segunda vez. A carteira de pesos iguais também atualiza quantidades em cada `ex_date`. O comportamento legado é inalterado quando o argumento de eventos é omitido.

**E. Relatórios versionados:** `reproducao_universo.json`, `predictions.csv`, `trades.csv`, `fold_calibration.csv`, figuras do replay, `interpretacao_resultado.json` com SHA-256 dos 150 arquivos e `pacote_validacao_economica_raw.zip`; todos dentro de `output/validacao_economica_raw_75/<ID>/`.

## 4. Referências financeiras anteriores — não são resultado novo

Os números abaixo foram obtidos em campanhas distintas, **não** pelo programa v1.25.0:

- **US$ 587.165.941,13**, reprodução do legado com snapshot vintage anterior, cutoff 06/10/2026.
- **US$ 89.941.496,49**, replay legado com atualização Alpaca RAW, mesmo cutoff. A restauração contrafactual de **73 barras OHLCV de maio/2018** em 31 ativos recuperou exatamente US$ 587,17 milhões.
- **US$ 76.152.561,34**, replay com snapshot renovado e cutoff posterior de 21/10/2026 — não comparar diretamente com o corte de 06/10/2026.

O valor novo será relatado **mesmo que diminua drasticamente** ou se aproxime dos US$ 587 milhões; nenhuma configuração será alterada em resposta ao desempenho do teste.

## 5. Regras de interpretação e avaliação de generalização

**A suspeita de overfitting continua aberta e relevante.** O diagnóstico já realizado teve 196/225 alertas de possível sobreajuste; 224/225 RMSE de teste maiores do que no treino, 29/225 skill positivo no teste e concordância OOS de ranking de 51,1981%, com intervalo exploratório de 95% 49,63%–52,43% incluindo 50%. Não transformar sucesso financeiro retrospectivo em alegação de previsibilidade confirmada. O protocolo walk-forward é cronológico, mas universo e configuração foram avaliados/selecionados retrospectivamente.

**Critério de validade do próximo ZIP:** hashes idênticos antes e depois, 75 ativos completos sem exclusões adicionais, cenário identificado como RAW econômico com splits contábeis, folds e custos documentados, capital reconciliado com curva e trades e base de comparação mesmo cutoff. Se falhar, declarar experimento **inconclusivo**, não substituir snapshot ou parâmetros silenciosamente.

**Limitações não resolvidas:** dados vintage ainda divergem em 73 barras de 2018, first_seen histórico de anúncios de splits não existe (somente ex-date para correção econômica), frações e cash-in-lieu, custo de oportunidade, impacto de mercado, execução de lotes muito grandes, visibilidade de outras informações e viés de seleção retrospectiva. A auditoria não prova desempenho prospectivo independente.

**Continuação científica após receber o ZIP:** revisar reconciliação econômica, contagens de splits, decisões/trades, robustez por fold e modelos que permaneceram com habilidade preditiva fora da amostra; somente uma validação em observações nunca utilizadas no processo de seleção pode abordar de fato a robustez prospectiva.
