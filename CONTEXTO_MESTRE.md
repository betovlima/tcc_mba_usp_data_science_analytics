## Auditoria recebida v1.24.6 — quase equivalência das utilidades; rastreio de precisão v1.24.7 (23/10/2026)

**Branch exclusiva:** `research/v1.23.1-fresh-snapshot-20261022`. **`main` preservada:** `4667d2ccee3f3441ee4d2b362ce734311cedd5e3`. PR #18 permanece Draft e sem merge.

**Pacote recebido:** `pacote_utilidades_economicas_raw.zip`, SHA-256 **`63be9712d1e3f846ec2f9ca8060369eb7caddcf33cf3f5c49dbd2308e19cee03`**, snapshot fixo U75 SHA `595b7a03dc2c8978487975cd3554b1ea65608741893f13208938dc624cb273cc`. **Código da auditoria:** v1.24.6, script `auditar_utilidades_economicas_raw_spyder.py` commit `8313aee49a6f17acc44afb2780b9b1ba16db3577` e testes `f2a710cc6e7256397141a17553d965bcf8f94cc8`. **Resultado científico integral:** `evidencias/resultado_utilidades_economicas_raw_v1246.md` (criado pelo commit `cc8a13c6507cc29812f3196ab0f460d160dd3a03`).

**Valores efetivos:** 75 ativos, 17 com split, 21 eventos registrados, **2.210 decisões hipotéticas × 15 valores de utilidade = 33.150 células**. **6 diferenças materiais** (`rtol=atol=1e-9`), ou **33.144/33.150 concordantes (99,98190%)**. Máximo erro absoluto **0,00833333333333336**. Somente **3 decisões** tiveram diferença, em apenas dois campos cada:
- YANG 16/10/2024, rótulo t+60 maduro em 14/01/2025: `forward_trend_persistence` econômico **0,404520** vs legado **0,396186** (diferença +0,008333333); `forward_risk_adjusted_utility` **−0,098952** vs **−0,100618** (diferença +0,001666667).
- CAPR 20/05/2019, rótulo t+60 maduro em 14/08/2019: persistência **0,561441** vs **0,553107**; utilidade final **0,458503** vs **0,456836**.
- CAPR 28/05/2019, rótulo t+60 maduro em 21/08/2019: persistência **0,578249** vs **0,569915**; utilidade final **0,493179** vs **0,491513**.
Nenhuma diferença material nas **5 utilidades individuais por horizonte nem nos 5 retornos líquidos individuais**; tampouco nas demais componentes agregadas. Os CSVs RAW permaneceram intactos; não foram ajustados modelos, decisões, capital ou lucro.

**Hipótese numérica ainda não comprovada nos dados reais:** `trend_persistence = 0.5×mean(close_future > abertura_entrada)+0.5×mean(diff(close_future)>0)`. Uma única comparação `close > abertura` entre 60 sessões que inverte booleana por empate/precisão float64 produz exatamente `0,5/60=0,0083333333`; a utilidade final propaga aproximadamente `0,001666667` (peso 0,2). O encaixe matemático sugere efeito de arredondamento ao aplicar fatores de split, mas o arquivo v1.24.6 contém apenas linhas de diferença no label, não as 60 barras por decisão, portanto **a causa não está identificada definitivamente**. Não modificar critérios ou tolerâncias para maximizar concordância sem auditar as barras.

**Próximo ensaio v1.24.7 implementado:** `diagnosticar_empates_utilidade_splits_spyder.py` commit `bf13e05bba9fa918aa2fe9549c24fe8c50cd7dfb`; `tests/test_diagnosticar_empates_utilidade_splits.py` commit `74d4da3d68a3f93c5b6f6e202035c9fa230f9740`. Em memória, compara, nas **três decisões específicas**, 60 resultados de `close > entrada` e 59 de `close_(t+1) > close_t` em RAW+quantidade versus controle legado. Registra pregões discrepantes, valores float64, representação hexadecimal e distância em ULPs, sem reescrever RAW, treinar ou operar. Teste unitário construtivo prova que um fechamento **um ULP maior** pode tornar-se empate após divisão por 3 no float64, mas não atribui automaticamente esse mecanismo aos dados YANG/CAPR.

**Para obter evidência real:** atualizar a mesma branch, reiniciar kernel, F5 em `diagnosticar_empates_utilidade_splits_spyder.py`; enviar `output/empates_utilidade_splits/<ID>/pacote_empates_utilidade_splits.zip`. Até então, classificar como quase equivalência de labels, **não equivalência completa**; não publicar backtest RAW nem substituir o checkpoint histórico dos US$ 587 milhões. O uso de labels supervisionados só após maturidade já foi auditado no teste anterior (450 grupos, zero violações), mas nenhum destes programas reproduziu a performance de capital.

---

## Maturidade dos rótulos U75 v1.24.5 aprovada; teste da utilidade completa v1.24.6 (23/10/2026)

**Branch:** `research/v1.23.1-fresh-snapshot-20261022`; **`main` preservada:** `4667d2ccee3f3441ee4d2b362ce734311cedd5e3`; PR #18 Draft, sem merge. **Relatório completo:** `evidencias/resultado_maturidade_labels_u75_v1245.md`, commit de criação `7fc18de7eb2492a19c3c29750221b92956244f45`.

**Evidência recebida:** `pacote_maturidade_labels_u75.zip`, **SHA-256 `48750a807bfc895e3d3ce8b0acd92d098569c8a4e09dc11c32d988a8ca2f185a`**. Snapshot U75 SHA-256 `595b7a03dc2c8978487975cd3554b1ea65608741893f13208938dc624cb273cc`. **Código executado:** `auditar_maturidade_labels_spyder.py` v1.24.5, commit `65c3d63d09d338645504533b6fa6455fab56c407`; testes `8ca1ad655b9144a79e2ad7ed905fc9a164c555c4`, `da817a6a3696157337b55504dcdb9d10f61a54de`.

**Resultado observado:** **75 ativos × 3 folds × 2 fases de treinamento = 450 grupos auditados** sobre o calendário próprio de cada ativo, usando dados RAW estritos, máximo de **60 sessões de horizonte** e purge de **60 sessões**. **583.650 linhas de treino somadas sobre os grupos**, contando reutilizações entre folds e fases. **Zero rótulos que amadurecem depois ou na data de início da avaliação; zero grupos em violação.** O último alvo disponível amadureceu no pregão anterior à primeira sessão avaliada em todos os seis limites.

| Fold | Fase | Linhas elegíveis / ativo | Último alvo amadurecido | Primeira avaliação |
|---|---|---:|---|---|
| 1 | Calibração | 700 | 23/10/2019 | 24/10/2019 |
| 1 | Final | 886 | 21/07/2020 | 22/07/2020 |
| 2 | Calibração | 1.204 | 22/10/2021 | 25/10/2021 |
| 2 | Final | 1.390 | 21/07/2022 | 22/07/2022 |
| 3 | Calibração | 1.708 | 25/10/2023 | 26/10/2023 |
| 3 | Final | 1.894 | 24/07/2024 | 25/07/2024 |

**Conclusão limitada:** o protocolo cronológico de ajuste dos rótulos `t+60` passou na auditoria de elegibilidade e maturidade; **não está provado** que todos os atributos, fontes e eventos corporativos sejam point-in-time, nem que os rótulos calculados ingênua e diretamente de RAW sem contabilizar splits sejam economicamente válidos. A auditoria reconstruiu quais amostras seriam elegíveis ao fit, sem treinar LightGBM ou executar nova carteira. Os US$ 587,17 milhões da v1.23.0 e a discrepância por 73 barras RAW de 2018 permanecem resultados históricos separados.

**Próximo ensaio v1.24.6 implementado e validado em CI:** `auditar_utilidades_economicas_raw_spyder.py` (commit `8313aee49a6f17acc44afb2780b9b1ba16db3577`), testes `tests/test_auditar_utilidades_economicas_raw.py` (commit `f2a710cc6e7256397141a17553d965bcf8f94cc8`). **176 testes aprovados, 34 avisos** no GitHub Actions, sem falhas. Este diagnóstico offline deriva **15 rótulos econômicos completos** (cinco utilidades por horizonte, cinco retornos líquidos por horizonte, cinco componentes agregados) de preços RAW **sem tocar no histórico**, aplicando fatores de quantidade nos caminhos futuros. Compara os rótulos com a formulação histórica normalizada, calculada somente em memória, em janelas próximas aos splits registrados nos **mesmos 75 ativos**. Nenhum treinamento, nenhuma otimização, nenhuma simulação de capital. **Os resultados reais de utilidade ainda não existem**, não inferir equivalência pelo sucesso dos testes sintéticos.

**Próximo passo no Spyder:** atualizar a mesma branch, reiniciar kernel, pressionar F5 em `auditar_utilidades_economicas_raw_spyder.py`. Enviar `output/utilidades_economicas_raw/<ID>/pacote_utilidades_economicas_raw.zip` para comparar `resumo.json`, `por_ativo.csv` e `divergencias_labels.csv` e registrar a identidade SHA do pacote e o resultado encontrado. Não alterar CSVs RAW, snapshots, versão oficial, universo de 75 ou main.

---

## Auditoria de retornos RAW/custódia concluída — v1.24.4; maturidade dos alvos v1.24.5 (23/10/2026)

**Branch de pesquisa:** `research/v1.23.1-fresh-snapshot-20261022`. **Main intocada:** `4667d2ccee3f3441ee4d2b362ce734311cedd5e3`. **Relatório técnico do ZIP:** `evidencias/resultado_retornos_economicos_splits_v1244.md`, commit `ec60bf9d486754961da1fad71f08b4f79ad5c46f`.

**Evidência recebida:** `pacote_retornos_economicos_splits.zip`, **SHA-256 `8b1903bbaeb0ed4694b9bb3abc705782d138bed02610a02ff8d63feb85a41ef3`**; snapshot único U75 `595b7a03dc2c8978487975cd3554b1ea65608741893f13208938dc624cb273cc`. O script produtor da v1.24.4 está no commit `0052964dc5c35bb0ab0b8f435ffa986793c49482` e seus testes em `78c0100c1b451ae086ec7ef46e3d5cc9f3750caf`.

**Resultados efetivamente medidos:** **11.980 retornos** de horizontes 5/10/20/40/60, sobre 75 ativos (17 com splits no histórico; **16 com janelas auditáveis**). **2.596 janelas atravessaram splits**, com diferença entre retorno ingênuo calculado dos preços RAW e retorno econômico após aplicar fator de quantidade apenas durante a posição. Nos **11.980 casos**, retorno econômico por quantidade foi numericamente igual à normalização de preços legada, com **0 divergências e erro absoluto máximo 4,440892098500626 × 10⁻¹⁶**. As outras 9.384 não atravessaram splits. **1.427 das 2.596 janelas tiveram sinal de retorno diferente** nos dois cálculos. Em casos de grupamento, o retorno RAW ingênuo pode divergir centenas de pontos percentuais; exemplo NBR entrada em 13/03/2020, saída em 08/06/2020, fator de quantidade 0,02: retorno ingênuo 220,531982 (22.053,20%), retorno econômico 3,430640 (343,06%). Não interpretar retornos hipotéticos como resultado da estratégia.

**Por horizonte:** 5 sessões: 80 janelas atravessando split; 10: 180; 20: 380; 40: 780; 60: 1.176 (2.396 comparações em cada horizonte). **RAW não modificado; não houve treino, simulação ou lucro calculado.** A coincidência econômica com o legado é uma identidade de retorno simples de preço; não avalia dividendos, custos, mark-to-market, penalidade de drawdown, labels de utilidade completos nem custódia fracionária.

**Risco científico ainda aberto:** `label_matures_after_close_on = exit_day` é um campo correto no pacote, porém sua presença **não prova** que os modelos LightGBM usaram apenas labels já maduros. O protocolo de folds define purge de 60 sessões e horizontes máximos de 60, mas dias ausentes no calendário específico de um ativo podem atrasar a maturidade econômica de um label, mesmo quando o calendário comum tem 60 sessões. É preciso conferir linhas **efetivamente elegíveis** para os ajustes do modelo por ativo.

**Próximo teste v1.24.5 (código implementado, resultado real pendente):** `auditar_maturidade_labels_spyder.py`, commit `65c3d63d09d338645504533b6fa6455fab56c407`, testes `tests/test_auditar_maturidade_labels.py`, commits `8ca1ad655b9144a79e2ad7ed905fc9a164c555c4` e `da817a6a3696157337b55504dcdb9d10f61a54de`. O script lê somente RAW estrito dos **75 ativos**, reconstrói o painel/3 folds através das funções do motor atual sem retreinar e verifica se, para cada amostra de treino elegível, `t+60` no **calendário real do próprio ativo** ocorre antes do início da avaliação subsequente. Audita os fits da fase de calibração e do ajuste final; gera número de linhas elegíveis, última maturidade e qualquer violação. Falha explicitamente quando há amostras prematuras. **Sem downloads, mutação de OHLCV, ordens nem backtest econômico.**

**Próximo comando:** atualizar `research/v1.23.1-fresh-snapshot-20261022`, reiniciar Spyder e executar `auditar_maturidade_labels_spyder.py` (F5). Saída: `output/maturidade_labels_u75/<ID>/pacote_maturidade_labels_u75.zip`. Ainda não se conhecem contagens de linhas dos folds reais nem se há violações. Mesmo zero violações de maturidade não provará ausência de outros vazamentos, nem valida a integridade da escala dos labels de utilidade quando usados preços RAW sem tratamento econômico.

**Política preservada:** único universo fixo de 75 ativos, snapshots Alpaca RAW imutáveis, resultados anteriores congelados, PR #18 Draft, nenhum merge na main.

---

## Auditoria v1.24.3 concluída e rótulos econômicos v1.24.4 (23/10/2026)

**Branch:** `research/v1.23.1-fresh-snapshot-20261022`; **main intocada:** `4667d2ccee3f3441ee4d2b362ce734311cedd5e3`. **Relatório integral da execução:** `evidencias/resultado_janelas_splits_v1243.md` (commit `03c9c280ff2c27bb6d915effeda0ae18d5f3ebc0`).

**Arquivo recebido:** `pacote_janelas_splits.zip`, SHA-256 **`1a73fcddfa794121eff21af8d1b87555f906268341835258699c16c304538326`**; base U75 `595b7a03dc2c8978487975cd3554b1ea65608741893f13208938dc624cb273cc`. Teste v1.24.3 rodado com o script commit `5f90ff28ef53cc94eb25039411d94e262ed15f66`, testes `bee74758394dc88cd0a9ea8d2c612d62a224fa6b`.

**Resultado empírico:** 21 splits na fonte, **20 durante a cobertura de preço** e RACE (04/01/2016) fora da janela. Comparando somente em memória RAW com normalização legada de cada evento nos 60 pregões anteriores/posteriores: **0 células de 52 atributos pré-evento materialmente modificadas** (1.080 linhas analisadas); **18/20 eventos** com alvos de utilidade futura anteriores diferentes (**1.062 linhas, 4.464 células**); **19/20 eventos** com indicadores posteriores alterados (**1.130 linhas, 38.606 células**); **0 células de rótulos posteriores modificadas** nas janelas analisadas. YANG 24/03/2016 e APD 03/10/2016 não têm rótulos pré-evento suficientes para serem comparados — o zero é falta de cobertura do aquecimento, não prova de ausência de efeito. Não existe reentrenamento ou backtest financeiro nessa auditoria.

**Exemplos reais dos gaps RAW na abertura:** NVDA 10/06/2024 −90,04% (neutro por split −0,43%); AMZN 06/06/2022 −94,88% (neutro +2,37%); PXLW reverso 09/06/2025 +1.088,49% (neutro −0,96%); NBR reverso 23/04/2020 +4.938,95% (neutro +0,78%). A reexpressão legada foi calculada na memória apenas como controle; os CSVs RAW mantêm seus valores.

**Interpretação científica:** mudanças de labels pré-evento ocorrem porque o alvo de utilidade futuro cruza a ex-date. Isso é **um problema de avaliação de rentabilidade econômica se splits forem ignorados**, mas o uso supervisionado do retorno ex-post depois de sua realização **não é, por si, lookahead**. Falta verificar que o treino purga/atrasa labels até seu horizonte ficar conhecido. As diferenças dos atributos pós-evento surgem quando janelas dos indicadores atravessam a quebra de unidade nos preços RAW. Esse fenômeno é distinto das 73 barras RAW revisadas de maio/2018 que explicaram a discrepância de US$ 587 milhões em outra campanha.

**Próximo teste v1.24.4 implementado:** `auditar_retornos_economicos_splits_spyder.py` (commit `0052964dc5c35bb0ab0b8f435ffa986793c49482`), testes `tests/test_auditar_retornos_economicos_splits.py` (commit `78c0100c1b451ae086ec7ef46e3d5cc9f3750caf`). Reconstrói, **sem modificar a série RAW**, o retorno de PREÇO por operação hipotética entrando na abertura da sessão seguinte à decisão e encerrando no fechamento em horizonte 5/10/20/40/60, aplicando fatores de **quantidade** de splits efetivados após a compra e até a saída. Compara (1) retorno ingênuo RAW, (2) retorno econômico pela quantidade e (3) retorno de preço legadamente normalizado apenas em memória como verificação numérica. **Não calcula utilidade completa, custos, drawdown, dividendos, capital nem ranking; não treina nem otimiza.** Registra também `label_matures_after_close_on` (fechamento da sessão de saída); a regra científica é não admitir esse alvo no treinamento antes dessa data.

**Para executar:** atualizar a mesma branch, reiniciar kernel, F5 em `auditar_retornos_economicos_splits_spyder.py`. Saída `output/retornos_economicos_splits/<ID>/pacote_retornos_economicos_splits.zip`. Os números reais de v1.24.4 **não foram produzidos ainda**. Mantenha 75 ativos, `main` sem alteração, PR #18 Draft e release oficial congelada.

---

## Resultado v1.24.2 e teste de fronteiras de splits v1.24.3 (23/10/2026)

**Branch de pesquisa:** `research/v1.23.1-fresh-snapshot-20261022`. **`main` intocada:** `4667d2ccee3f3441ee4d2b362ce734311cedd5e3`. **Relatório científico completo da execução:** `evidencias/resultado_impacto_splits_futuros_v1242.md` (criado pelo commit `99813bda592f1d0bf1433e08c2c615b5f3da9248`).

**ZIP analisado:** `pacote_impacto_splits_futuros.zip`, SHA-256 `530eb722e87d9f0d79a84bd9ebb3c96608ee97d7479e925ae991c255a5f9dc8c`; snapshot U75 SHA-256 `595b7a03dc2c8978487975cd3554b1ea65608741893f13208938dc624cb273cc`. **Código produtor:** `auditar_impacto_splits_futuros_spyder.py`, commit `cea46639bb1c46c408f447064e1047ecbefe74b3` e testes `98c9d6dbf77af48e978a98f0e8d93f524f96eaab`.

**Achado quantitativo no corte 28/07/2020:** 14 splits com ex-date futura, em 12 de 75 ativos (NVDA, AAPL, AMZN, GOOGL, TSLA, AVGO, NFLX, WMT, CNQ, PXLW, YANG, MUX). A rotina **legada**, ao incorporar esses eventos, reexpressou **13.800 barras** pré-decisão (1.150 por ativo), alterando níveis de preço **e** volume. Entretanto, em **11.412 linhas de atributos × 52 atributos = 593.424 células** comparadas (951 por ativo), houve **zero diferenças materiais** (`rtol=1e-10; atol=1e-12`), e nenhuma data de atributo ficou disponível em só um dos dois cenários. Dos 75 ativos, somente os 12 com splits posteriores tiveram reconstrução de features nesta comparação; os outros 63 não foram apresentados como janelas de features comparadas. Nenhum arquivo RAW foi modificado.

**Interpretação correta:** alterar retrospectivamente o nível absoluto dos preços não se traduz necessariamente em modificar retornos, razões e distâncias normalizadas no período inteiro anterior à efetivação, pois esses 52 atributos são invariantes à escala constante. **Esse teste refuta, sob sua tolerância, a alegação de mudança desses atributos antes de 28/07/2020, mas não prova segurança global do LightGBM, dos rótulos ou das decisões.** A data `ex_date` não atesta quando o anúncio esteve disponível ao sistema; `first_seen_at_utc` permanece sem evidência histórica. Os US$ 587,17 milhões versus US$ 89,94 milhões são explicados por 73 diferenças nas próprias barras RAW de maio/2018 em experimento separado, **não** por este teste de splits.

**Nova investigação v1.24.3:** `auditar_janelas_splits_spyder.py` (implementação commit `5f90ff28ef53cc94eb25039411d94e262ed15f66`, testes `tests/test_auditar_janelas_splits.py`, commit `bee74758394dc88cd0a9ea8d2c612d62a224fa6b`). Examina os eventos reais do snapshot no único universo de 75 ativos: para cada split durante a cobertura RAW, compara apenas em memória atributos calculados pelo motor e alvos de utilidade multihorizonte nos **60 pregões anteriores e 60 posteriores** ao evento, usando como controle a antiga normalização por split. Registra também gap de preço RAW da abertura e gap neutralizado **apenas como cálculo econômico de auditoria**, sem reescrever cotação, treinar, negociar ou publicar capital.

**Próxima saída a obter com F5 no Spyder:** `output/auditoria_janelas_splits/<ID>/pacote_janelas_splits.zip`, com `eventos_e_janelas.csv` e `resumo.json`. Os números do experimento sobre splits efetivos ainda **não foram gerados nesta conversa**. Mesmo havendo diferenças em rótulos, elas não demonstram automaticamente vazamento de informação de anúncio; exigem distinção entre alvos supervisionados ex-post e informação disponível no instante da decisão. As ações de custódia em split ainda precisam de integração e teste antes de qualquer backtest RAW economicamente válido.

Preservar: 75 ativos, RAW imutável, experimentos v1.23.0 publicados, `main` e PR #18 Draft, sem merge e sem otimização.

---

## Auditoria temporal de splits concluída e teste de impacto real — v1.24.2 (23/10/2026)

**Branch:** `research/v1.23.1-fresh-snapshot-20261022`. **`main` preservada:** `4667d2ccee3f3441ee4d2b362ce734311cedd5e3`. **Relatório integral das evidências:** `evidencias/resultado_auditoria_temporal_splits_v1241.md`, criado em `6cf7207d5db1b64366be7d27b0fbac5f3464a25c`.

**Pacote recebido e auditado:** `pacote_auditoria_temporal_splits.zip`, SHA-256 `840f42f48d4277bd6c4e65cc81c121ed073aa68169a8e9266c0d609587372ea4`. **Código executado:** v1.24.1, código HEAD `d86f148c9b39653a37191459c930ff934ef1e9f7`. **Snapshot do universo único de 75:** `595b7a03dc2c8978487975cd3554b1ea65608741893f13208938dc624cb273cc`.

**Achados reais (cutoff 28/07/2020):** 75 ativos, **1.150 linhas históricas por ativo, 86.250 linhas ao todo**, sem edição RAW. **21 splits registrados em 17 ativos**, sendo **7 já efetivos e 14 com ex-date posterior** ao cutoff, estes distribuídos por 12 ativos (11 forward e 3 reverse). **Todos os 21 têm `first_seen_at_utc` ausente**: disponibilidade do anúncio em 2020 não comprovada; `evidence_available_at_decision=false` significa *sem evidência*, **não** anúncio comprovadamente desconhecido. O fato de a data efetiva ser futura não determina a data do anúncio.

**Controle sintético v1.24.1:** em memória foi inserido um split fictício de NVDA com ex-date 12/08/2020 e first_seen posterior a 28/07. O motor reportou **52 atributos iguais em 951 linhas** ao usar RAW não modificado; **não treinou o LightGBM nesse pacote** e **não comparou scores ou rotações reais dos 75**. A normalização legada mostrou alterar preços antes da ex-date quando se introduz um evento futuro, fato que precisa ser medido no universo histórico real.

**Nova bateria v1.24.2 — medir impacto retroativo real, não inferir sem dados:** script `auditar_impacto_splits_futuros_spyder.py` (commit `cea46639bb1c46c408f447064e1047ecbefe74b3`), testes sintéticos `tests/test_auditar_impacto_splits_futuros.py` (commit `98c9d6dbf77af48e978a98f0e8d93f524f96eaab`). Executar localmente F5, somente leitura, usando a mesma fotografia Alpaca U75. Para cada ativo, compara **dois cenários de normalização legada somente na memória**, um com todos os splits conhecidos pelo snapshot de 2026, outro limitado aos splits **já efetivados** até 28/07/2020. Compara quantas **linhas OHLCV antigas** e quantas **células de 52 atributos do TCC** mudam materialmente antes da decisão. Não escreve essas séries, não treina modelos, não consulta Alpaca, não altera universo nem capital.

**Saída do próximo teste:** `output/impacto_splits_futuros/<ID>/pacote_impacto_splits_futuros.zip`, com `resumo.json`, `impacto_por_ativo.csv` e `impacto_por_atributo.csv`. **Ainda não há resultados desses impactos em dados reais.** O corte por ex-date é apenas comparação do efeito de efetivações futuras sobre o histórico; **não substitui prova de conhecimento do anúncio** nem autoriza ajustes em preços RAW.

**Condição de conclusão científica:** quantificar primeiro se eventos efetivados depois da decisão modificaram apenas os níveis históricos ou também os atributos relativos do LightGBM. Alteração retrospectiva de OHLCV **não implica necessariamente** alteração de scores/decisões (vários atributos são invariantes à escala). Mesmo que os atributos mudem, não declarar vazamento econômico sem isolar treino temporal e inferência completos. Preservar snapshots e a reprodução de US$ 587.165.941,13 sem sobrescrever resultados oficiais.

---

## Testes point-in-time de splits — v1.24.1 (23/10/2026)

**Branch:** `research/v1.23.1-fresh-snapshot-20261022`. **Base da `main`, não modificada:** `4667d2ccee3f3441ee4d2b362ce734311cedd5e3`. **Relatório completo:** `evidencias/auditoria_temporal_splits_v1241.md` (commit de criação `196c9e4045278aa32f01f601f6e978f93606b701`).

**Commits de implementação:** `41dde42d74e408a107805356103d6fcbbc5df796` (`reproducao/splits_temporais.py`); `2d15b011eb33f288a6bf5627e524dcbdf7583355` (`auditar_temporalidade_splits_spyder.py`); `d411219b0253d9e5e2b07a9a1b731a38acbcbef1` (testes de disponibilidade/atributos/custódia); `a2c7ca93c20c1d5d04328e6dbbb319c1be85150e` (teste de ranking LightGBM sintético). **CI do último código:** **141 testes aprovados**, com advertências de dependência, sem falhas.

**Regra experimental:** separar `first_seen_at_utc` (primeira observação comprovada do anúncio pelo sistema), `ex_date` (data econômica de efetivação) e `process_date` (data de processamento, **não substitui** primeira observação). Um evento futuro **não anunciado e desconhecido** não pode modificar atributos ou decisões calculados antes de sua disponibilidade. Um evento **anunciado mas não efetivado** pode constar dos metadados conhecidos, porém não permite mudar quantidades de ações antes da ex-date.

**Resultados dos controles sintéticos:** a política RAW mantém inalterados os atributos do motor real e o ranking de um mini-LightGBM ao acrescentar anúncio posterior à data de decisão. O teste de aritmética de custódia confirma que um split 1:10 transforma 1 ação × US$ 100 em 10 ações × US$ 10 sem alterar a base de custo, e rejeita a reaplicação do mesmo evento. O comportamento legado `split_normalize` modifica retrospectivamente OHLCV anteriores ao evento: esse é o mecanismo que motivou a mudança de política, não um procedimento autorizado da versão RAW estrita.

**Próxima execução real, sem downloads e sem modificar dados:** atualizar branch no Git e pressionar F5 em `auditar_temporalidade_splits_spyder.py`. O script lê o snapshot dos **75 ativos efetivos** em `dados/.u75_fresh_runs`, verifica o prefixo até **28/07/2020** e registra os splits e a evidência de disponibilidade histórica. Gera `output/auditoria_temporal_splits/<ID>/pacote_auditoria_temporal_splits.zip`. Ainda não existe resultado local do universo inteiro; não afirmar paridade das previsões OOS/LightGBM completo antes de completar simulação economicamente correta com contabilidade dos splits.

**Limitação científica:** snapshot retrospectivo não prova quando o anúncio foi disponibilizado historicamente. Sem log contemporâneo `first_seen_at_utc`, classificar visibilidade como **desconhecida**, nunca presumir. A nova aritmética de custódia é um teste unitário; faltam integração com motor de negociação, fracionários/cash-in-lieu e retornos/labels ao atravessar splits. **Nenhum capital de novo backtest RAW foi produzido ou validado**. Preservar a reprodução oficial v1.23.0 (US$ 587.165.941,13), os dados congelados, os resultados anteriores e a aplicação operacional.

---

## Política de integridade RAW — v1.24.0 (23/10/2026)

**Decisão explícita do pesquisador:** a pesquisa contemporânea do TCC deve **preservar OHLCV e os timestamps recebidos da Alpaca**, com `feed=SIP` e `adjustment=RAW`. Não aplicar normalização por splits, ajuste por dividendos, interpolação, reconstrução histórica nem remoção silenciosa de linhas. Conversão de timestamp para UTC e de colunas numéricas para tipos numéricos é permitida **somente como interpretação dos mesmos valores**; erros, datas duplicadas, ausência, preços não positivos, volumes negativos ou ordem inconsistente devem provocar falha explícita, não correção automática. **Universo fixo: 75 ativos**, sem DOC/CLMT.

**Auditoria do código anterior:** `reproducao/preparacao.py::split_normalize()` multiplicava retrospectivamente OHLC pelo fator `old_rate/new_rate` e volume pelo fator `new_rate/old_rate`. Isso foi introduzido para manter continuidade econômica depois de desdobramentos, mas é uma **transformação de valores RAW**, contrária à política agora exigida. O carregador anterior em `reproducao/dados.py::_normalize_alpaca_frame` também fazia `drop_duplicates(keep="last")`, `dropna` e filtro de preços/volumes inválidos **sem rejeição explícita**. Não foi constatado nos ZIPs que tais filtros realmente descartaram registros nesta campanha; identificou-se o risco no código.

**Nova implementação (exclusiva para pesquisa RAW):**
- `reproducao/preparacao.py` acrescenta `strict_raw=True` e `apply_split_adjustments=False`: preserva OHLCV integral, mantém ordem e quantidade das barras, rejeita inválidos e registra `price_transform_policy=alpaca_raw_unchanged` e `actual_total_splits_applied=0`. Commit `4658d5e6c40de7e7bc91c830b53453521fb9f7c1`.
- `reproducao/dados.py` acrescenta `strict_raw=True` à ingestão: **não reordena**, **não descarta duplicados**, **não substitui valores inválidos**; aborta com diagnóstico. Fonte continua `Adjustment.RAW` e `DataFeed.SIP`. Commit `ca51c18182085825ff71567af802c89beb8dae0d`.
- Novo `auditar_series_alpaca_raw_spyder.py` (commit `ddbf2b512d8fb1b78febdc4ad05b36cb8095792b`): no Spyder, F5, consulta **somente os 75 ativos efetivos** em um **novo snapshot isolado** `dados/.u75_strict_raw_runs`, com manifesto SHA-256. Compara os valores float64 de OHLCV do CSV com os valores disponibilizados na preparação para o modelo, verificando **igualdade exata para as 75 séries**. Gera `output/auditoria_raw_sem_ajustes/<ID>/pacote_auditoria_raw_sem_ajustes.zip` e **não realiza backtest**, para não apresentar rentabilidade economicamente inválida.
- Testes novos `tests/test_raw_immutable_v1240.py` (commit inicial `7a3dabf3d92ea94633e42e97012e4d56ccf85601`; higiene `e1000a3a8e20e1745ccb806333a5309108224948`): valor float64 idêntico, nenhuma exclusão silenciosa, falha em timestamps duplicados/invertidos e em preços/volumes inválidos, ausência de ajuste por splits e dividendos.

**Preservação do experimento antigo:** o comportamento padrão das rotinas de preparação legadas permanece o mesmo **apenas para reproduzir os checkpoints de v1.23.0 a v1.23.9**, incluindo US$ 587.165.941,13 e US$ 76.152.561,34. Isso não valida que valores transformados sejam séries RAW; devem ser identificados como `Alpaca RAW + transformação local de splits`. Nunca reescrever esses resultados ou os snapshots para forçar uma comparação.

**Condição prévia para futuro backtest estritamente RAW:** em um desdobramento `n:1`, a Alpaca reporta o preço pós-evento sem ajuste, mas a carteira passa a ter `n` vezes as ações. Simular sem aumentar a quantidade de ações cria prejuízo falso; ajustar preços viola a política de dados. O motor legado usa normalização de splits para evitar essa descontinuidade, e ainda **não possui contabilidade completa de quantidades no evento corporativo**. Por isso a v1.24.0 **não publica capital nem previsão de retorno em RAW sem esse suporte**. A etapa científica seguinte é implementar eventos de splits **somente no razão de posições/custódia e nos cálculos econômicos**, sem modificar, reescrever ou substituir as barras originais, e provar por teste de conservação de patrimônio que `1 × US$ 100 == 10 × US$ 10` num split de 10:1. Dividendos também devem ser avaliados separadamente como fluxos de caixa apenas quando o escopo econômico os incorporar; nunca descontar os preços em silêncio.

**Branch e proveniência:** `research/v1.23.1-fresh-snapshot-20261022`, PR #18 Draft, main mantida em `4667d2ccee3f3441ee4d2b362ce734311cedd5e3`. A campanha v1.24.0 representa **nova política de dados e potencial alteração metodológica material**; não comparar seu futuro capital diretamente aos US$ 587 milhões como se a única diferença fosse o cutoff.

---

## Auditoria RAW e decisões recentes — v1.23.9 (23/10/2026)

**Branch:** `research/v1.23.1-fresh-snapshot-20261022`. **Novo diagnóstico offline:** `diagnosticar_decisoes_outubro_2026_spyder.py`, commit `e879e34ff945ece2f0e684f3d140e2e88bc93a1c`; testes de contrato `tests/test_diagnosticar_decisoes_outubro.py`, commit `575d047af09f65769aa6c97d127ee7fede6b810f`. **Auditoria detalhada:** `evidencias/auditoria_raw_dividendos_decisoes_outubro_v1239.md`, commit `d1883c4cd122430233ceca499a32cb4c16d2a5a8`. **`main` de referência:** `4667d2ccee3f3441ee4d2b362ce734311cedd5e3` (intacta).

**Pergunta respondida:** ambos os conjuntos Alpaca foram obtidos com `Adjustment.RAW`, `DataFeed.SIP`, barras diárias. O pipeline `split_normalize` faz ajuste **somente de splits**, não de dividendos; foram 20 aplicações de splits nos 75 ativos. Sete eventos `cash_dividend` adicionais de setembro/2026 não implicam desconto de dividendos dos preços do TCC. A auditoria direta das barras **RAW** encontra **73 barras modificadas em 31 ativos efetivos**, concentradas em 01–03/05/2018: volume mudou nas 73, abertura em 16, máxima em 57, mínima em 54 e fechamento em apenas 2. **44 outros ativos** não tiveram alterações materiais antes de 06/10/2026. Restaurar essas 73 barras em cópia isolada, sem mudar Corporate Actions, recuperou US$ **587.165.941,13** com negociações byte a byte iguais ao TCC histórico. Não há prova da razão da revisão da Alpaca nem de qual vintage é o correto.

**Avaliação realizada sobre os ZIPs já recebidos (sem retreinamento):** patrimônio marcado em 06/10 **US$ 89.943.393,94095525**; em 21/10 **US$ 76.152.561,33521412**; onze pregões adicionais, delta **−US$ 13.790.832,61 (−15,33279%)**. Quatro novas posições encerradas: **WDAY +2,92%**, **CAPR −7,45%**, **ADEA −3,03%**, **CAPR −9,95%**. Pior pregão: 21/10 **−11,05%**. Scores nas quatro entradas: **0,386285; 0,392323; 0,345593; 0,455803**. Scores são utilidade estimada, **não** probabilidade de retorno positivo, e não foram disponibilizados vetores completos dos 75 scores para julgar alternativas.

**Artefatos e hashes:** `pacote_comparacao_cutoffs_75.zip` SHA-256 `da1001963f1a57cc2576aa6e55cffbaf31150bb20c824e6fca2419c381175e79`; `pacote_auditoria_drift_historico.zip` SHA-256 `a6ce4625b3bbca7ad7440e94e3623f1fff20fe01bf0fc0b03d5a6e5592aa414c`; pacote novo de diagnóstico `pacote_diagnostico_outubro_2026.zip` SHA-256 `6be2c89a77532786c12980fef6a9a77aaade01a79d4172e15433ccbfe869a497`.

**Reproduzir o diagnóstico no Spyder:** atualizar a mesma branch, reiniciar kernel, executar `diagnosticar_decisoes_outubro_2026_spyder.py` (F5). O código usa somente o ZIP previamente gravado em `output/fresh_universe_75` e gera `output/diagnostico_decisoes_outubro_2026/pacote_diagnostico_decisoes_outubro_2026.zip`. Não chama a Alpaca, não altera séries oficiais e não modifica estratégia. Em paralelo, permanece pendente o **teste fatorial volume-only vs OHLC-only** da v1.23.8, que poderá isolar o papel de preços e volumes dos 73 registros. Não confundir resultados retrospectivos com validação prospectiva de scores.

---

## Resultado do contrafactual histórico de 2018 e teste fatorial — v1.23.8

**Branch de pesquisa:** `research/v1.23.1-fresh-snapshot-20261022`. **Motor Control unificado de 75 ativos, referência executada:** v1.23.7, commit `78d473fb83d5ce2b89c3282f6b6ace4054c53bdc`. **`main` intacta:** `4667d2ccee3f3441ee4d2b362ce734311cedd5e3`. **Relatório científico integral:** `evidencias/auditoria_contrafactual_2018_v1237.md`, introduzido no commit `557171eb83e190ff355c43fad00178d1de70623c`.

**Pacote de evidência recebido:** `pacote_contrafactual_2018.zip`, **SHA-256 `612d068d0fac97423a34a6da14c98b51ce2e3fffbb172954b619fc1990d9d21b`**; fonte U75 renovada SHA `595b7a03dc2c8978487975cd3554b1ea65608741893f13208938dc624cb273cc`, contrafactual SHA `0cae0311283e7d7419bef061b482e31c72eb972420c2f4178b0733a3d07a06f6`.

| Corte em 06/10/2026, mesmo Control U75 | Capital final |
|---|---:|
| Snapshot renovado de 75 ativos (sem intervenção) | US$ 89.941.496,49095525 |
| Mesmo snapshot, restaurando somente 73 barras OHLCV antigas de maio/2018 | **US$ 587.165.941,1330212** |
| Release histórica v1.23.0 com snapshot congelado | **US$ 587.165.941,1330212** |

A restauração controlada de **73 barras em 31 ativos efetivos**, das quais 12 em 01/05, 30 em 02/05 e 31 em 03/05/2018, foi **suficiente para recuperar exatamente o checkpoint econômico histórico**: +US$ 497.224.444,64 em relação ao controle, capital multiplicado por 6,52831. Contagens por campo: volume 73, máxima 57, mínima 54, abertura 16, fechamento 2 (sobreposição de campos por barra). DOC e CLMT não integraram a pesquisa.

**Verificação reforçada:** o `trades.csv` restaurado tem **os mesmos bytes do arquivo histórico oficial**, com 710 registros; 1.560 escolhas e valores econômicos coincidentes. As únicas 552 diferenças de células do `predictions.csv` frente ao original são metadados `fold_test_end` do terceiro fold, não operações. A referência renovada tem 690 operações; as escolhas divergem em 192/1.560 sessões, começando em **28/07/2020, CXW (renovado) versus MGM (restaurado)**. Margens calibradas por fold: renovado **0,0100 / 0,0000 / 0,0100**; restaurado **0,0000 / 0,0100 / 0,0000**, iguais às originais.

**Conclusão limitada aos dados e ao motor analisados:** as 73 diferenças pré-OOS de maio de 2018 são suficientes para explicar integralmente a divergência financeira entre estes dois replays. **Não está provado** quais linhas/campos/ativos dominam o efeito, nem qual versão de preços da Alpaca é economicamente correta, nem a possibilidade de realizar prospectivamente os US$ 587 milhões. Registros originais e scripts oficiais não devem ser sobrescritos. A dissertação deve destacar a forte sensibilidade do LightGBM/rotação a pequenas revisões históricas concentradas.

**Novo teste fatorial v1.23.8 (não confundir com resultado já obtido):** `pesquisar_atribuicao_ohlc_volume_spyder.py` (implementação `592e371b9becdf774588774e79400d6cda52cfa1`, limpeza `860872a0ae8959dfbf550aa1aa1ae42aef3bfd33`, testes `a4f237a2997f51b5146dc34578f0ddcf3f2f4c13`). Apenas os dois cenários intermediários são calculados: **restaurar somente volume** ou **restaurar somente OHLC** sobre duas cópias descartáveis, recuperando por hash e manifesto os replays original e integral restaurado já disponíveis em `output/causal_drift_2018/`. Os quatro capitais compõem um fatorial 2×2 que permite examinar efeitos separados e interação não linear. **Ainda sem números financeiros desses novos dois cenários.** Nenhum download, ordem, alteração de 75 ativos ou troca de parâmetros. Saída `output/fatorial_drift_2018/<ID>/pacote_fatorial_ohlc_volume.zip`. Executar após atualizar a mesma branch, reiniciar o kernel e pressionar F5.

---

## Auditoria financeira final da unificação dos 75 ativos — v1.23.7

**Branch:** `research/v1.23.1-fresh-snapshot-20261022`; **revisão testada:** `78d473fb83d5ce2b89c3282f6b6ace4054c53bdc`; **base da `main` preservada:** `4667d2ccee3f3441ee4d2b362ce734311cedd5e3`. **Relatório integral:** `evidencias/auditoria_universo_unico_75_v1237_20261022.md`, adicionado no commit `ecf16c9080ea94ef5c89ceb89d247f7b9b5b5a1b`. **Pacote local v1.23.7:** `pacote_comparacao_cutoffs_75.zip`, SHA-256 `da1001963f1a57cc2576aa6e55cffbaf31150bb20c824e6fca2419c381175e79`.

| Execução | Cutoff 06/10/2026 | Cutoff 21/10/2026 |
|---|---:|---:|
| Controle experimental v1.23.7 — **75 ativos treinados conjuntamente** | **US$ 89.941.496,49095525** | **US$ 76.152.561,33521412** |
| Controle experimental v1.23.4 — 75 efetivos, motor legado | US$ 89.941.496,49095525 | US$ 76.152.561,33521412 |
| Plataforma operacional, mesma estratégia e data | — | US$ 76.152.561,33521412 |
| TCC oficial v1.23.0 — dados congelados, cutoff 06/10 | US$ 587.165.941,1330212 | não executado |

**Provas de equivalência nesta fotografia:** `snapshot_75_manifest.json` possui exatamente 75 símbolos. Todos os **150 arquivos RAW/Corporate Actions** destes ativos mantêm SHA-256 idêntico ao snapshot anterior de 77, sendo DOC e CLMT os únicos símbolos descartados. A nova identidade U75 é `595b7a03dc2c8978487975cd3554b1ea65608741893f13208938dc624cb273cc`, proveniente da U77 `1c4d16173eaa5219a1bcc18cfa431cf1c096dbeaa26ca53aa03e9fcf047cff5c`. Os `predictions.csv` e `trades.csv` de v1.23.7 e v1.23.4 são **100% idênticos por célula**, respectivamente 1.560 e 1.571 decisões e 690 e 698 registros de negociação; as margens e pontuações de calibração são idênticas, diferindo apenas o rótulo `scenario` herdado do motor (`U67_BASELINE`). **Nenhum resultado econômico se alterou com a consolidação dos 75 ativos.**

Métricas até 21/10: Sharpe **2,310440236459**, drawdown máximo **−41,682843%**, 348 rotações, benchmark de comprar/manter no mesmo universo **US$ 30.822,89**. O replay terminado em 06/10 liquida a posição CAPR e paga US$ 1.897,45; o replay contínuo marca US$ 89.943.393,94 nesse dia, daí a diferença contábil entre encerrar ou continuar.

**Consequência metodológica:** não há justificativa para manter duas etapas 65+10 na pesquisa contemporânea: a execução conjunta dos 75 produziu exatamente os mesmos resultados. Isso **não explica nem recupera** os US$ 587 milhões sobre o snapshot antigo. É preciso concluir o teste contrafactual das **73 barras de maio de 2018 em 31 ativos efetivos**, por `pesquisar_causa_drift_2018_spyder.py` na mesma branch, com resultado isolado `pacote_contrafactual_2018.zip`. Não alterar o TCC oficial, a `main` ou a aplicação operacional. DOC/CLMT não participaram dos resultados comparados; este experimento não mede o resultado hipotético de incluí-los.

---

## Política definitiva do universo efetivo — v1.23.7 (22/10/2026)

**Decisão do pesquisador:** o **único universo da pesquisa é composto por 75 ativos**, sem DOC e CLMT. Os símbolos excluídos permanecem apenas nos manifestos de proveniência histórica e na reprodução oficial já congelada, **nunca na coleta, treinamento, calibração, simulação ou auditoria prospectiva do fluxo de pesquisa v1.23.7**. Nenhum ativo deve ser acrescentado, descoberto, substituído ou reabilitado automaticamente.

A convenção anterior de 67 símbolos originalmente solicitados (65 elegíveis) acrescidos de dez selecionados pertence **exclusivamente à história do projeto**. Ela não define mais dois grupos de modelagem. Os 75 símbolos efetivos são derivados uma única vez do manifesto vencedor, removendo DOC e CLMT, e são tratados como **um painel fechado e único desde o primeiro fold**.

**Identificação Git:** branch `research/v1.23.1-fresh-snapshot-20261022`. Código do novo executor unificado: commit `d2f219cdd78cbee7770fd17f9634a432411f0ac1` (`reproducao/universo_fixo75.py`); runner: `8c2fd0ad05cf9af06021e0e0ad506cf2a18cbb7e` (`executar_atualizacao_series_spyder.py`); contrafactual de 75 ativos: `1562b251e4d71718a3741d71bb3a01cd77441877` e `26e8feaf2db4903e543cb296dfe815d155019800`; auditoria offline limitada a 75: `87096850d6e0bbf0657be5c3642cf9263ea00e2e`.

**Operação v1.23.7:**

- `executar_atualizacao_series_spyder.py` (Spyder F5) usa **75 séries SIP/RAW/1Day**, sem DOC/CLMT, tanto na aquisição quanto no modelo. Se o usuário já possui o snapshot isolado legado de 77 em `dados/.u67_fresh_runs`, o programa primeiro valida integralmente seu SHA-256 e cria **uma nova cópia com somente os 75 arquivos correspondentes**, com manifesto novo e SHA próprio, sob `dados/.u75_fresh_runs`. O snapshot anterior não é modificado; a migração é offline. Sem snapshot legado, baixam-se somente os 75.
- Os 75 modelos LightGBM por ativo são ajustados **em conjunto dentro do mesmo ciclo de folds**, e a margem de rotação é calibrada sobre os 75 ativos, sem formação de uma referência de 65 ativos seguida por adição de dez.
- O programa avalia 06/10/2026 e 21/10/2026 com a mesma fotografia de 75 ativos, gera `output/fresh_universe_75/asof_<DATA>_<ID>/pacote_comparacao_cutoffs_75.zip`.
- `auditar_drift_historico_spyder.py` considera apenas os 75. Das alterações históricas antes observadas, esperam-se **73 barras de 31 ativos efetivos**, pois duas barras eram de CLMT e não pertencem à pesquisa. As duas fontes históricas originais continuam inalteradas.
- `pesquisar_causa_drift_2018_spyder.py` usa somente os 75 e faz a intervenção temporária nas 73 barras, com o mesmo motor unificado e cutoff de 06/10/2026. Não cria novos ativos, não escreve nos snapshots originais e não utiliza a aplicação operacional para pesquisa.

**Advertência metodológica indispensável:** o resultado US$ 587.165.941,13 pertence ao executor histórico v1.23.0 e foi reproduzido com 75 ativos **efetivamente negociáveis**, mas o procedimento histórico ainda distinguia 65 ativos-base dos dez adicionais no treinamento. O novo procedimento de **treinamento conjunto de 75** é uma **mudança de desenho experimental**. Portanto, **não prometer que a execução v1.23.7 reproduzirá exatamente US$ 587 milhões ou US$ 76,15 milhões**, nem atribuir diferenças exclusivamente à atualização das séries sem um controle que preserve o motor antigo. Essas referências e todos os hashes continuam arquivados para comparação honesta.

**Regra de integridade:** nenhum novo snapshot/resultado de v1.23.7 substitui `dados/u67`, `evidencias/manifest_universo_congelado.json` ou a release científica `v1.23.0`. A `main` permanece intocada. As razões documentadas para exclusão de DOC e CLMT são estruturais; não se deve afirmar, sem ensaio controlado específico, que eles necessariamente destruiriam o capital em qualquer cenário.

---

## Auditoria de divergencias de maio de 2018 — v1.23.6 (23/10/2026 UTC)

**Branch:** `research/v1.23.1-fresh-snapshot-20261022` (**sem merge na main**). O replay financeiro de referência segue sendo **v1.23.4, commit `bd99b592c03dadec2cf45ade0b09e8ee07148d3b`**. A presente revisão v1.23.6 só acrescenta testes e instrumento de sensibilidade; não altera o modelo ou a política de rotação.

**Evidência recebida:** `pacote_auditoria_drift_historico.zip`, SHA-256 `a6ce4625b3bbca7ad7440e94e3623f1fff20fe01bf0fc0b03d5a6e5592aa414c`, originado do teste offline v1.23.5. Original: SHA do snapshot base `e440f59da5e684f1de59cf447abfedd9aed3f7817b3d5d681058fe631276a575`. Atualizado: SHA do snapshot unificado `1c4d16173eaa5219a1bcc18cfa431cf1c096dbeaa26ca53aa03e9fcf047cff5c`.

- **77 séries alinhadas até 06/10/2026**, cada qual com **2.705 datas idênticas**, sem lacunas; 12 sessões posteriores presentes no snapshot renovado.
- **75 divergências OHLCV materiais em 32 ativos**, limitadas aos dias **1º, 2 e 3 de maio de 2018**. As demais 45 séries são materialmente idênticas nas datas antigas. Duas das 75 barras são de CLMT, previamente excluído do modelo por problema estrutural, restando **73 divergências em 31 ativos efetivos**.
- Em **02/05/2018**, volume agregado nos ativos com diferenças: **12.152.097 (antigo)** versus **216.979.152 (renovado)**. Em **03/05/2018**, **10.871.735** versus **196.836.941**, respectivamente.
- Somente dois fechamentos diferiram: **SBFG 02/05/2018 (18,930 × 18,954)** e **CAPR 03/05/2018 (1,290 × 1,280)**. As outras divergências correspondem a abertura, máxima, mínima e/ou volume.
- **Sete eventos de dividendo** foram acrescentados no snapshot novo, todos em setembro de 2026 (CRM, ADC, BXMT, XEL, CRT, STAG, TOWN); a auditoria não constatou alteração em splits de datas anteriores ao corte. A diferença pós-normalização de splits permanece em 75 barras.
- Os próprios históricos de **MGM e CXW**, associados à primeira decisão diferente em **28/07/2020**, não divergiram materialmente. A diferença de decisão pode refletir efeitos indiretos em treinamento, calibração ou seleção do modelo, sem causalidade ainda comprovada.

**Próximo teste:** `pesquisar_causa_drift_2018_spyder.py` (F5 no Spyder), em **uma cópia descartável**, reconstrói os valores originais dessas 75 barras a partir de arquivos de hash verificado, sem editar qualquer CSV ou manifesto original. Executa **dois replays com cutoff 06/10/2026 e mesmas séries fora desses registros**, mantendo eventos corporativos, universo fixo, folds, regras e custos. O primeiro usa snapshot renovado inalterado; o segundo, a cópia isolada com as 75 barras restauradas para o experimento. Saída: `output/causal_drift_2018/<ID>/pacote_contrafactual_2018.zip`.

**Proveniência do novo código:** implementação inicial `e02b315d47474c21e94223bbee1738a799077e9c`; testes sintéticos de imutabilidade e intervenção `df3f500f4e180803e722273d11ba9b0858b62000`. Os scripts também conferem a composição dos 77 símbolos, hashes e exatamente 75 diferenças históricas de 32 ativos. **A cópia criada não é fonte oficial** e não deve ser usada para alterar o checkpoint de US$ 587.165.941,13 ou promover estratégias.

**Hipótese a verificar:** se os resultados dos dois replays mudarem fortemente apenas pela substituição das observações, teremos evidência experimental de sensibilidade financeira aos insumos de maio de 2018. Se não mudarem, será necessário investigar eventos, configuração do calendário de folds e demais diferenças controláveis; não presumir a explicação.

---

## Auditoria de paridade e plano de testes de insumos — revisão documental v1.23.5 (22/10/2026)

**Escopo da branch:** `research/v1.23.1-fresh-snapshot-20261022`. O experimento financeiro que gerou os resultados abaixo permanece na **v1.23.4**; a v1.23.5 adiciona *somente auditoria e testes*, sem mudar LightGBM, política, hiperparâmetros, custos, universo ou `main`.

**Proveniência Git:**
- TCC `main` usado como referência: `4667d2ccee3f3441ee4d2b362ce734311cedd5e3` (release histórica `v1.23.0`).
- Runner TCC experimental v1.23.4 analisado: `bd99b592c03dadec2cf45ade0b09e8ee07148d3b`.
- Implementação do diagnóstico offline: `bc4a61cecfbbfcc894886d065ef7a2e22023fa70` (ajuste posterior `1578b7edcf5be556d8730ee3ea28cf4d002d6761`).
- Testes automatizados do diagnóstico: `d2fb205ceb475ac91bc7834c6728b7d359af8556`.
- Execução externa de comparação: `adb764df4bbf36bcf63af1c9017b9a3dcb6fb844` (API v10.8.125).
- Identidade snapshot histórico base U67: `e440f59da5e684f1de59cf447abfedd9aed3f7817b3d5d681058fe631276a575`; identidade snapshot único renovado U77: `1c4d16173eaa5219a1bcc18cfa431cf1c096dbeaa26ca53aa03e9fcf047cff5c`.
- Pacote de comparação v1.23.4, SHA-256 `fb9b87c43d6233ae8334e6dc083b42280bb0778cbcb31c698a2c12766963ab61`.

**Resultado reproduzido:**
| Execução | Corte | Patrimônio final |
|---|---|---:|
| TCC histórico congelado | 06/10/2026 | US$ 587.165.941,13 |
| TCC com séries renovadas | 06/10/2026 | US$ 89.941.496,49 |
| TCC com séries renovadas | 21/10/2026 | US$ 76.152.561,34 |
| Plataforma operacional, mesma data | 21/10/2026 | US$ 76.152.561,34 |

A execução renovada e a plataforma operacional apresentaram 1.571/1.571 escolhas idênticas e 698/698 operações equivalentes no período. A diferença entre TCC congelado e TCC renovado já aparece em 28/07/2020: MGM versus CXW; 192 das 1.560 decisões da janela até 06/10 selecionaram ativos diferentes.

**Importante:** diferença de SHA-256 dos CSVs NÃO comprova revisão retroativa. A simples inclusão de registros após 06/10 altera o hash. Também não se pode afirmar que uma revisão específica foi responsável pelo capital sem checar primeiro preços brutos, calendário e Corporate Actions.

**Próximo experimento (somente leitura, sem corretora):** abrir `auditar_drift_historico_spyder.py` no Spyder e executar com F5. O programa valida hashes históricos e atuais, confronta **os 77 ativos fixos** (65+10 efetivos, exclusões DOC/CLMT preservadas) somente até 06/10, compara OHLCV bruto nas mesmas datas, compara valores após normalização de splits e registra mudanças de eventos corporativos anteriores ao corte e splits posteriores. Uma análise focaliza entradas anteriores a 28/07/2020. Gera `output/auditoria_drift_historico/pacote_auditoria_drift_historico.zip`. Não faz downloads, não atualiza CSVs e não modifica a reprodução oficial.

**Pré-requisito local:** dados oficiais em `dados/u67`, cache original dos dez adicionais em `dados/.u67_expansion_cache/full_raw_20261006` com hashes exatos e snapshot novo em `dados/.u67_fresh_runs/asof_2026-10-22_*/universe_77`. Se faltar algum original, o programa deve interromper — nunca baixar ou reconstruir no lugar. O ZIP de backtest não contém os RAWs necessários para esse teste.

**Interpretação acadêmica:** distinguir robustez da política contra revisões de fonte, reprodução exata condicionada ao snapshot congelado e testes efetivamente fora da amostra; evitar afirmar ausência de vazamento futuro ou causalidade da primeira decisão divergente sem a auditoria por linha.

---

## Experimento v1.23.4 — comparacao do TCC com a data final atualizada

**Objetivo restrito:** investigar o efeito de atualizar os historicos e de avancar o corte temporal mantendo *o mesmo universo vencedor fixo*, que gerou US$ 587.165.941,13 em 06/10/2026. Nao descobrir, classificar ou otimizar novos ativos.

- Branch em uso: `research/v1.23.1-fresh-snapshot-20261022`; **revisao experimental v1.23.4**, sem alterar a `main`.
- Entrada no Spyder: `executar_atualizacao_series_spyder.py` (reiniciar kernel, F5).
- `TARGET_CUTOFF = "2026-10-22"` e `EVALUATION_CUTOFFS = ("2026-10-06", "2026-10-21")`.
- **Correção da falha apos 10/10 modelos (v1.23.4):** no replay experimental, `expected_capital` e `checkpoint.delta` sao `null`, pois nao existe capital esperado fixo quando os dados mudam. A expressao anterior `capital - None` gerava `TypeError` depois de todo o treinamento. O checkpoint historico oficial continua usando a comparacao numerica exata.
- **Retomada economica (v1.23.4):** com `REUSE_LAST_COMPLETE_SNAPSHOT = True` (padrao), `executar_atualizacao_series_spyder.py` procura o snapshot mais recente `dados/.u67_fresh_runs/asof_<DATA>_*/universe_77/`, confere manifesto SHA-256, todos os 77 ativos fixos e barra final de AAPL e **nao faz nova coleta da Alpaca**. Reutiliza o mesmo identificador de execucao, produz os dois replays e o ZIP. Modelos nao foram salvos antes da excecao, portanto o treinamento dos folds e refeito. Se o manifesto estiver ausente, realiza nova coleta; se for inconsistente, para com erro. Definir `REUSE_LAST_COMPLETE_SNAPSHOT = False` somente se desejar deliberadamente uma nova fotografia.
- **Correção do erro 403 SIP (v1.23.3):** a coleta preserva `feed=sip` e `adjustment=raw`, mas limita `end` a, no máximo, a hora UTC atual menos 20 minutos. Isso evita requisitar o início de 23/10/2026 enquanto ainda estamos em 22/10/2026. O cutoff analítico permanece 22/10/2026, e a barra de AAPL dessa data deve ser verificada. Não trocar automaticamente SIP por IEX. Se o `403` persistir, verificar a mensagem retornada pela Alpaca e as permissões da credencial.
- **Uma coleta de 77 ativos solicitados** de Alpaca RAW/SIP/1Day e seus eventos corporativos. O pacote historico original (67 + 10) define a lista uma unica vez. A pesquisa nao executa descoberta de candidatos, nem altera a composicao.
- **75 ativos efetivos obrigatorios.** Exclusoes estruturais permanecem DOC e CLMT. Um novo problema de integridade interrompe a execucao com erro; nao descartar ativos silenciosamente nem reparar series manualmente.
- Dados localmente isolados em `dados/.u67_fresh_runs/asof_<DATA>_<ID>/universe_77/` (nao versionados no Git), manifesto e SHA-256. Os arquivos do snapshot oficial `dados/u67/` e os manifestos de `evidencias/` sao preservados.
- Dois replays financeiros com **o mesmo snapshot de 77 simbolos, os mesmos algoritmos, atributos, parametros, custos, treinamento e politica**, alterando apenas a data final. A divisao interna do motor historico entre base e adicionais existe unicamente para manter os calculos de v1.23.0; nao e etapa de pesquisa, escolha ou coleta em separado.
- **Corte A — 06/10:** mede alteracoes do resultado em 06/10 com historicos atualizados, comparando ao TCC v1.23.0 (US$ 587.165.941,13).
- **Corte B — 21/10:** mede o efeito de avancar a data final na mesma fonte e permite comparar com a execucao externa encerrada em 21/10.
- Saida para compartilhar: `output/fresh_universe/asof_<DATA>_<ID>/pacote_comparacao_cutoffs.zip`; inclui `comparacao_cutoffs.json`, manifestos, dados financeiros e operacoes dos dois replays, sem RAWs. O checkpoint oficial nao e imposto aos experimentos; diferenças devem ser explicadas, nao ajustadas.
- Nenhuma ordem, nenhuma atualizacao de Mongo e nenhuma modificacao da aplicacao operacional.
- Testes CI validam a separacao, integridade de referencias e orquestracao de 77 ativos com mocks. A reproducao economica completa depende de executar F5 localmente com credenciais da Alpaca e enviar o pacote.

Para baixar a branch no terminal Git:

```powershell
git fetch origin
git switch research/v1.23.1-fresh-snapshot-20261022
git pull --ff-only origin research/v1.23.1-fresh-snapshot-20261022
```

---

# CONTEXTO_MESTRE — reprodução científica do TCC

**Atualização:** 21/10/2026. **Projeto:** `betovlima/tcc_mba_usp_data_science_analytics`. Este documento é a referência inicial para um novo assistente ou nova sessão. **Separação de ambientes:** o TCC utiliza dados congelados em pesquisa; a aplicação operacional independente utiliza cotações Alpaca atuais.

## 1. Git, estado e identidade

- **Branch principal:** `main`. O commit da pesquisa histórica `13e5ed39ed0e700d2eb8c90bc8bee64adc7fdf9a` já foi incorporado à `main` pelo merge **`9d2e279938a2ad271f3ef5e0ac86a18e130c7e4f`**, de 18/10/2026 UTC. Seu HEAD histórico era `13e5ed39ed0e700d2eb8c90bc8bee64adc7fdf9a`. **Não fazer novo merge nem abrir branches duplicadas.**
- **Motor Control de referência:** versão `1.22.1`. **Programa de pesquisa:** `pesquisar_expansao_u67_spyder.py`, versão `u67-asset-expansion-v1.4.1`, esquema `u67-asset-expansion-v5`.
- **Tag de congelamento acordada:** `research-u67plus10-challenger-v1.4.1`. A tag deve apontar ao estado documentado da `main`; **verificar com `git ls-remote --tags origin` se foi efetivamente publicada**. Não afirmar que existe antes da confirmação.
- **Reprodução já atestada:** `reproduzir_experimento.py` reproduziu Control base; `assurance_validation` confirmou os cenários de expansão em 21/10/2026. Não mudar código, limites financeiros, folds, modelos ou fonte de dados para alcançar o checkpoint.

## 2. Dados congelados que produzem o resultado histórico

| Item | Contrato obrigatório |
|---|---|
| Base | `dados/u67/manifest.json` e 67 arquivos RAW/Corporate Actions versionados |
| Identidade da base | `e440f59da5e684f1de59cf447abfedd9aed3f7817b3d5d681058fe631276a575` |
| Fonte / datas | Alpaca SIP, RAW, 1Day; início 01/01/2016, corte **06/10/2026** |
| Base efetiva | 65 ativos; excluir DOC e CLMT por identidade estrutural, sem reparar/concatenar séries |
| Expansão testada | 15: CRT, NBR, CAPR, PNBK, PENN, JBL, LEU, JBSS, STAG, MCHB, EML, MAR, LAD, TOWN, ACIW |
| Dez selecionados | **CRT, NBR, CAPR, PNBK, PENN, JBL, STAG, MCHB, LAD, TOWN** |
| Cache dos 15 | `dados/.u67_expansion_cache/full_raw_20261006/raw_bars/*.csv` e `corporate_actions/*.csv` (ignorado pelo Git) |
| Evidência externa | `pacote_analise_u67_assurance_validation(1).zip`, SHA-256 `0f06561e9685b3e84194580c50a6808e4c84d30a55584b97bd19dae2fafd6c37` |

**Risco crítico:** houve execução com um snapshot local diferente (`8183a2b718bb...`) que retornou **US$ 45.493.884,10360689** no Control inicial; após restaurar o snapshot original versionado, voltou a **US$ 76.927.051,38897176**. Validar um manifesto local não prova que ele é o manifesto histórico: comparar o SHA exato. **Não executar `preparar_snapshot_u67.py` durante a reprodução e não baixar novamente barras/eventos dos 15 ativos.** Antes de restaurar arquivos, criar backup fora do Git.

## 3. Capitais e evidência recebida

| Cenário no cutoff 06/10/2026 | Capital (US$) |
|---|---:|
| Control base, 65 efetivos | **76.927.051,38897176** |
| Base + 15 positivos | **290.727.954,41339606** |
| Base + 10 selecionados, 75 efetivos | **587.165.941,1330212** |

A execução `assurance_validation` enviada pelo autor declarou `status=completed`, `all_15_exact_to_cent=true` e `selected_10_exact_to_cent=true`. `reproducibility.json` confirmou `exact_to_cent=true`, `selected_asset_sequence_identical=true`, `max_equity_absolute_difference=0.0` **entre dois replays da campanha restaurada**. O Sharpe dos dez foi **2,7285355362** e o MaxDD **-41,68283638%**; com 10 bps de slippage, capital caiu para **US$ 288.628.398,50**.

**O ZIP e o cache de candidatos não foram anexados ao Git**; o ZIP inclui os SHA-256 de cada barra e arquivo de Corporate Actions dos 15 candidatos. Clonar o repositório sozinho **não garante** a reprodução do cenário expandido. A evidência histórica deve ser preservada de forma separada e seu checksum conferido.

## 4. Como executar sem alterar os dados (Spyder)

**Git é usado no console do Git; Python exclusivamente no console IPython do Spyder.** Na reprodução histórica, trabalhar na `main` atualizada e preservar alterações locais antes de usar `git restore`.

No **console Git** (somente consulta/atualização de referências e estado):

```bash
git status --short
git fetch origin
git branch --show-current
git ls-remote --tags origin research-u67plus10-challenger-v1.4.1
```

Para restaurar **somente se necessário**, após backup externo de `dados/u67`, usar no console Git:

```bash
git restore --source=origin/main --worktree -- dados/u67
```

No **console Spyder** (diretório de trabalho é a raiz do TCC), reiniciar kernel e executar:

```python
from pathlib import Path
from datetime import datetime
import runpy
import shutil
import pandas as pd
from reproducao.dados import SnapshotPaths, validate_snapshot

ROOT = Path.cwd().resolve()
assert (ROOT / "pesquisar_expansao_u67_spyder.py").is_file(), "Abra o Spyder na raiz do TCC"
assert validate_snapshot(SnapshotPaths.u67(ROOT))["snapshot_sha256"] == "e440f59da5e684f1de59cf447abfedd9aed3f7817b3d5d681058fe631276a575"

namespace = runpy.run_path(str(ROOT / "pesquisar_expansao_u67_spyder.py"), run_name="assurance_offline")
env = namespace["main"].__globals__
env["RUN_MODE"] = "assurance_validation"
env["OUT"] = ROOT / "output" / "u67_assurance_validation"
if env["OUT"].exists():
    backup = env["OUT"].with_name(env["OUT"].name + "_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    shutil.copytree(env["OUT"], backup)
    print("Backup da saída:", backup)

def usar_somente_cache(_credentials, ativos):
    rows = []
    for asset in ativos:
        valid_bars = env["_full_raw_cache_valid"](asset)
        valid_actions = (env["FULL_DATA"].corporate_actions / f"{asset}.csv").is_file()
        if not (valid_bars and valid_actions):
            raise RuntimeError(f"Cache histórico faltante/incompleto: {asset}")
        rows.append({"asset": asset, "status": "cached", "reason": None})
    return pd.DataFrame(rows)

def bloquear_download(*args, **kwargs):
    raise RuntimeError("Download proibido na reprodução histórica")

env["load_alpaca_credentials"] = lambda *_: None
env["_prepare_full_candidate_files"] = usar_somente_cache
env["download_raw_bars"] = bloquear_download
env["download_corporate_actions"] = bloquear_download
result = env["main"]()
```

**Observação:** esse bloqueio cobre as funções de download do script histórico v1.4.1 e não substitui validação de hash dos arquivos do cache. O programa grava em `output/u67_assurance_validation/` e pode demorar após `[assurance-train] 15/15 ACIW` sem mostrar novos logs: trabalha em simulações de subconjuntos. Um primeiro arquivo de progresso esperado é `assurance_scenarios_partial.csv`. Não reiniciar o kernel enquanto houver CPU trabalhando.

## 5. Como interpretar e continuar a pesquisa

- Os checkpoints são **reprodução histórica**, não evidência de lucro futuro. A rotina original selecionou candidatos com informação de toda a janela; a busca de subconjuntos apresentou **PBO 53,57% (log-growth)** e **59,52% (Sharpe)**.
- A pesquisa temporal em pool fixo selecionou `NBR,PENN,EML` usando fold 1 para avaliar fold 2; e `CRT,NBR,PNBK,PENN,JBL,STAG,MCHB,TOWN` usando folds 1+2 para avaliar fold 3. O pool já era conhecido: **não é descoberta nested prospectiva**.
- A auditoria registrou **38 alertas de overfit em 45 candidato-fold**, segundo `model_generalization.csv`. Não esconder isso nas redações.
- Priorizar uma **assinatura contextual/temporal de pertencimento** `S(ativo, U_t, contexto_t)` para ADICIONAR/MANTER/REMOVER sem olhar capital posterior à decisão. Diferenciar a pesquisa matemática posterior v1.18/v1.19, que teve outro baseline (U59), antes de tentar transferir o modelo.
- **Separação permanente:** o TCC utiliza CSVs congelados e a plataforma operacional independente utiliza preços atuais da Alpaca. O capital do experimento histórico não é previsão nem condição automática para decisões novas.

## 6. Documentos para releitura antes de qualquer modificação

1. [MANIFEST.md](MANIFEST.md), seção 26, com resultados e contexto da promoção.
2. [PESQUISA_EXPANSAO_U67.md](PESQUISA_EXPANSAO_U67.md), metodologia da campanha e interpretação da assinatura temporal.
3. [README.md](README.md), instruções gerais e requisitos.

**Regra para a próxima sessão:** verificar branch/commit/snapshot/cache/hash antes de propor alteração; não criar branches nem mudar `main` por suposição. Alterações futuras só mediante solicitação explícita do autor; documentar versão, causa, testes e limites.


## 7. Mudança de entrada oficial (v1.23.0)

A entrada `reproduzir_experimento.py` usa a função reutilizável `reproduzir_universo_congelado` em `reproducao/experimento.py` e deve calcular o checkpoint histórico do **universo selecionado**, US$ 587.165.941,1330212, com 77 ativos solicitados (75 efetivos). Evitar reintroduzir marcadores de contagem no nome do executor. O programa de pesquisa histórico mantém suas funções originais para garantir paridade numérica. `evidencias/manifest_universo_congelado.json` fixa os SHA-256 das barras e eventos corporativos dos dez candidatos; o executor não consulta Alpaca. O pacote gerado é `output/reproducao_universo/pacote_reproducao_universo.zip`. O usuário usa comandos Git em seu console Git e Python no Spyder.

O checkpoint de US$ 587 milhões é **meta de reprodução na mesma amostra congelada**, não inferência prospectiva. **A verificação integral foi concluída com sucesso em 21/10/2026**, no ambiente Spyder do autor: capital final observado de US$ 587.165.941,1330212, diferença US$ 0,00, `status=completed`, 1.560 sessões e 354 rotações. Pacote recebido `pacote_reproducao_universo.zip`, SHA-256 `0c3518387556f147897baf121cb089f532f8355a8bb074c91fe0a6b5ec4b6ac8`. Os dez pares de arquivos adicionais continuam locais e não versionados.


## 8. Ponto de recuperação oficial — release v1.23.0

**Estado aprovado para congelamento da release em 21/10/2026:** a nova entrada `reproduzir_experimento.py` é o executor oficial do universo selecionado, com nomes genéricos, delegando a rotina financeira ao código validado na campanha histórica. Foram confirmados a identidade do snapshot `e440f59da5e684f1de59cf447abfedd9aed3f7817b3d5d681058fe631276a575`, os 20 hashes dos dez arquivos de barras e dez eventos, 77 ativos solicitados/75 efetivos e capital de **US$ 587.165.941,1330212**, exato ao centavo. `reproducao_universo.json` do ZIP declarou `reproduced_exact_to_cent=true`. O arquivo de evidência é `pacote_reproducao_universo.zip` (SHA-256 `0c3518387556f147897baf121cb089f532f8355a8bb074c91fe0a6b5ec4b6ac8`); não foi adicionado ao Git.

A tag de referência da versão é **`v1.23.0`**, criada somente após sucesso dos testes do commit documental. **Antes de afirmar que a tag foi publicada, verificá-la no GitHub**. O código da versão já estava na `main`; não é necessário fazer novo merge. A tag histórica `research-u67plus10-challenger-v1.4.1` não deve ser movida. A CI cobre 94 testes/checagens unitárias e estáticas, mas não roda o replay financeiro que depende do cache não versionado.

**Próxima sessão:** usar `git fetch origin`, `git switch main`, `git pull --ff-only origin main` em console Git; executar `reproduzir_experimento.py` com F5 no Spyder sem download de histórico; comparar o SHA-256 do ZIP local contra a evidência desta seção. Não misturar os comandos Git em células Python. A pesquisa futura de assinatura de pertencimento temporal deve tratar os US$ 587 milhões como checkpoint histórico com seleção retrospectiva, não como validação prospectiva de descoberta automática.
