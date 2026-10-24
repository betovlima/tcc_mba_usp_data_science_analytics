# Resultado auditado — temporalidade de splits no universo único de 75 ativos

**Data da análise:** 23/10/2026. **Branch:** `research/v1.23.1-fresh-snapshot-20261022`. **Versão executada:** `1.24.1`. **HEAD do código que criou o teste:** `d86f148c9b39653a37191459c930ff934ef1e9f7`. **Main imutável:** `4667d2ccee3f3441ee4d2b362ce734311cedd5e3`. **PR:** #18 (Draft).

**Evidência recebida:** `pacote_auditoria_temporal_splits.zip`; **SHA-256 `840f42f48d4277bd6c4e65cc81c121ed073aa68169a8e9266c0d609587372ea4`**. Arquivos do pacote: `resultado_temporal_splits.json`, `eventos_splits_e_disponibilidade.csv` e `integridade_prefixos_75.csv`.

**Snapshot-base dos 75 ativos:** SHA-256 `595b7a03dc2c8978487975cd3554b1ea65608741893f13208938dc624cb273cc`. **Data de referência da decisão:** `2020-07-28`.

## Evidências efetivamente observadas

- **75/75 ativos** verificados, sem DOC ou CLMT, **1.150 barras por ativo** até a decisão de 28/07/2020, isto é, **86.250 registros RAW** analisados por fingerprint.
- `future_split_injected_into_raw=false` e `raw_bar_values_modified=false` para todos os 75 ativos. Os fingerprints foram guardados por ativo, e os CSVs reais não foram modificados pela auditoria.
- **21 registros de splits** no arquivo retrospectivo de Corporate Actions, distribuídos por **17 ativos distintos**. **7 eventos com ex-date anterior ou igual a 28/07/2020** e **14 eventos posteriores**, estes distribuídos por **12 ativos**. Data efetiva futura **não é sinônimo de anúncio futuro**.
- **Todos os 21** apresentam `first_seen_at_utc` vazio. O campo `historical_visibility_verifiable` é `false` para **21/21**: **não há comprovação, nesse pacote, da data de disponibilidade do evento na API**. `evidence_available_at_decision=false` significa **falta de prova**, não prova de que o evento era desconhecido em 28/07/2020.
- Entre os **14 eventos com efetivação posterior** estão AAPL 4:1 (31/08/2020), TSLA 5:1 (31/08/2020), NVDA 4:1 (20/07/2021) e 10:1 (10/06/2024), AMZN 20:1 (06/06/2022), GOOGL 20:1 (18/07/2022), AVGO 10:1 (15/07/2024), além de CNQ, NFLX, MUX, PXLW, WMT e YANG.
- **Controle sintético NVDA:** o script injetou **somente em memória** um split com ex-date 12/08/2020. Se o primeiro registro comprovado fosse 01/08/2020, não seria conhecido em 28/07; se fosse 20/07, seria conhecido, embora ainda não efetivado; sem `first_seen_at_utc` permanece desconhecido. Todas essas afirmações são casos **sintéticos**, não cronologia demonstrada da Nvidia.
- Foram comparadas **52 colunas de atributos em 951 linhas** do motor de engenharia sobre o prefixo RAW. Reportou-se `features_equal_before_event=true`, `raw_preserved=true`, `model_retrained=false` e `score_and_rotation_parity_tested=false`. A igualdade vale para o controle sintético executado, não prova que o backtest real de 75 ativos teria rotações iguais.
- `legacy_split_normalization_changes_pre_event_prices=true`: confirmado que o algoritmo anterior **altera retrospectivamente valores de barras anteriores** com o multiplicador do split, inclusive ao receber metadados de evento futuro. Isso exige auditar o impacto nos atributos reais antes de afirmar qualquer vazamento financeiro.

## Interpretação científica

**Demonstrado:** o snapshot não registra o momento de conhecimento dos anúncios, impossibilitando afirmar disponibilidade histórica; os dados RAW podem ser preservados; o mecanismo legado é capaz de transformar preços passados por splits efetivados posteriormente.

**Não demonstrado:** uso material de informação futura nas previsões do Control, mudança de ranking, renda fictícia ou impacto em US$ por causa de splits. Os 21 eventos não equivalem a 21 vazamentos. O teste de LightGBM sintético passou em CI, mas o ZIP não apresenta replay dos 75 ativos nem teste de isolamento de cada split real.

**Próxima investigação, sem otimização:** comparar no snapshot real, para cada um dos 75 ativos, o prefixo OHLCV e os **atributos derivados** do motor quando a normalização legada inclui **todos os splits registrados em 2026** versus quando inclui somente splits **já efetivos até 28/07/2020**. O corte por ex-date serve para medir o efeito de eventos posteriores; não é uma reconstrução perfeita da informação histórica disponível, pois falta `first_seen_at_utc`. Gerar CSV por ativo/campo e contagens materialmente diferentes, sem modificar a fonte ou executar backtest financeiro. Em especial, distinguir **alteração de nível de preço** (pode ser enorme) de **alteração nos 52 atributos relativos** (pode ser zero por invariância de escala). Somente depois investigar previsões/rotações sem lookahead com modelo completo e contabilidade de splits em custódia.

**Integridade:** manter `main` intocada, preservar v1.23.0, manter a pesquisa sempre nos 75 ativos e não substituir dados RAW por preços ajustados nem utilizar o snapshot que render mais capital como se fosse necessariamente verdadeiro.
