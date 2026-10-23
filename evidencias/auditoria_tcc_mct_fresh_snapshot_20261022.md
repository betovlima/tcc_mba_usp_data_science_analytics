# Auditoria de paridade TCC × MCT e sensibilidade ao snapshot — 22/10/2026

## Identificacao e rastreabilidade

- **Branch de pesquisa:** `research/v1.23.1-fresh-snapshot-20261022` (não criar outra branch; não fazer merge na `main` até concluir a auditoria de insumos).
- **Revisão testada do TCC:** v1.23.4, commit **`bd99b592c03dadec2cf45ade0b09e8ee07148d3b`**, resultado do executor `executar_atualizacao_series_spyder.py` com dados Alpaca renovados.
- **Base `main` do TCC antes da pesquisa:** commit **`4667d2ccee3f3441ee4d2b362ce734311cedd5e3`**; release oficial etiquetada `v1.23.0`. Não alteradas.
- **MCT Winner #30 — execução estudada:** commit em runtime **`adb764df4bbf36bcf63af1c9017b9a3dcb6fb844`**, API v10.8.125; identificador `20261022T151609-7d2de3e9`.
- **TCC com snapshot atual:** SHA-256 de identidade **`1c4d16173eaa5219a1bcc18cfa431cf1c096dbeaa26ca53aa03e9fcf047cff5c`**, 77 ativos solicitados e 75 elegíveis.
- **TCC oficial congelado (67 da base):** SHA-256 de identidade **`e440f59da5e684f1de59cf447abfedd9aed3f7817b3d5d681058fe631276a575`**. Os dez adicionais originais vivem no cache histórico local e têm seus 20 hashes fixados em `evidencias/manifest_universo_congelado.json`.
- **ZIP da comparação atualizada:** `pacote_comparacao_cutoffs.zip`, SHA-256 **`fb9b87c43d6233ae8334e6dc083b42280bb0778cbcb31c698a2c12766963ab61`**.
- **ZIP do TCC congelado conferido:** `pacote_reproducao_universo(4).zip`, SHA-256 **`7a3aadd62b3f373a80c7df45ca702eb5397b4070c55e590f5e6fb1f6dd7afbbc`**.
- **ZIP do MCT Winner #30 conferido:** `market_cycle_trader_20261022T151609-7d2de3e9(1).zip`, SHA-256 **`7c799e242ec32efe64c391c81093f45493483a40b93ade55f92d4d64559d8998`**.
- **Fonte da auditoria:** conteúdos desses três ZIPs recebidos, sem download adicional de séries ou acesso a ordens.

## Resultado financeiro em janelas comparáveis

| Cenário | Corte inclusivo | Capital final (US$) | Significado |
|---|---|---:|---|
| TCC v1.23.0 com CSVs congelados | 06/10/2026 | 587.165.941,1330212 | checkpoint oficial reproduzido |
| TCC v1.23.4 com snapshot renovado em 22/10 | 06/10/2026 | 89.941.496,49095525 | efeito de **trocar o snapshot**, mantendo o corte |
| TCC v1.23.4, **mesmo** snapshot renovado | 21/10/2026 | 76.152.561,33521412 | efeito conjunto até o corte comparável ao MCT |
| MCT Winner #30 | 21/10/2026 | 76.152.561,33521412 | mesma saída da simulação do TCC atualizado |

- Diferença entre TCC congelado e TCC renovado **em 06/10**: **−US$ 497.224.444,64** (cerca de −84,68%). Esta comparação separa o efeito de *reprocessar com uma fotografia de dados mais nova* da simples inclusão dos pregões adicionais.
- Variação entre dois cortes com a **mesma fotografia atualizada**: **−US$ 13.788.935,16** (cerca de −15,33%). Os dois cortes são replays com liquidação final diferente; a liquidação ao terminar em 06/10 cobra aproximadamente US$ 1.897,45, ao passo que a simulação contínua em 06/10 ainda não liquidou.
- **Paridade TCC renovado × MCT (21/10):** capital idêntico; 1.571/1.571 sessões com o mesmo ativo; 698/698 operações com as mesmas datas/ações/ativos e valores de execução, 348 rotações, Sharpe 2,310440236459 e drawdown máximo −41,682843%. Diferenças de arredondamento na curva diária inferiores a US$ 0,000001.
- **Primeira divergência TCC antigo × renovado:** **28/07/2020**, com MGM no snapshot antigo e CXW no atualizado; 192/1.560 sessões da janela histórica tiveram ativos diferentes. Esta divergência é anterior ao avanço temporal para outubro de 2026.
- **Folds (capital composto até 06/10):** Fold 1, **US$ 67.999,95 antigo × US$ 51.308,86 renovado**; Fold 2, **US$ 10.968.842,01 × US$ 4.160.778,02**; Fold 3, **US$ 587.165.941,13 × US$ 89.941.496,49**. A diferença é amplificada pelo reinvestimento.

## O que a auditoria prova, e o que não prova

1. **Comprovado:** TCC renovado v1.23.4 e Winner #30 produzem as mesmas decisões e resultados nesta execução documentada. Isso valida *paridade de saída observada*, não equivalência universal do código.
2. **Comprovado:** o resultado de US$ 587 milhões é reproduzível **com o snapshot original**, mas não se manteve quando insumos históricos foram renovados. A diferença não pode ser atribuída apenas às onze sessões adicionais em outubro.
3. **Ainda não demonstrado:** quais linhas OHLCV, splits ou outros corporate actions anteriores ao corte mudaram e quais explicam diretamente a decisão de 28/07/2020. **Hashes de arquivo diferentes não provam revisão retroativa de preços:** acrescentar pregões também muda SHA-256.
4. **Dos dez adicionais**, todos os arquivos RAW diferem por hash; os arquivos de eventos de CRT, STAG e TOWN também diferem por hash. Esta observação não equivale a afirmar que eventos com efeito pré-06/10 foram alterados.
5. **Viés temporal:** reconstruir os preços de 2020 a partir de um snapshot de 2026 pode envolver ajustes retrospectivos por splits. A inspeção do pipeline deve separar a diferença de preços brutos da diferença gerada pela normalização dos eventos, antes de concluir causalidade.
6. **Limitação de acesso:** os ZIPs não trazem os 77 CSVs RAW. O teste definitivo das revisões por linha deve ser rodado **localmente** sobre os dois conjuntos preservados; não se deve baixar de novo para preencher essa lacuna.

## Próxima bateria de testes — sem otimização

Executar `auditar_drift_historico_spyder.py` (introduzido na mesma branch após esta auditoria), usando:

- **Origem 1:** `dados/u67` para os 67 originais e `dados/.u67_expansion_cache/full_raw_20261006` para os dez adicionais. Conferir os hashes antes de comparar.
- **Origem 2:** último snapshot `dados/.u67_fresh_runs/asof_2026-10-22_*/universe_77`, validado integralmente por SHA-256. Nunca sobrescrever nenhuma origem.
- **Janela:** registros de preços e eventos até 06/10/2026; registrar novos pregões além da janela separadamente.
- **Diagnósticos:** comparar por ativo/data os OHLCV brutos (exatos e com tolerância numérica), preços após normalização de splits, diferenças de eventos corporativos por data efetiva e diferenças anteriores a **28/07/2020**. Gerar arquivos de evidência, incluindo primeira divergência e variáveis afetadas.
- **Critério:** identificar observações divergentes e classificá-las; **não** substituir símbolos, corrigir dados manualmente, alterar modelo, parâmetros, política Control ou Worker do MCT.
- **Saída:** `output/auditoria_drift_historico/` e ZIP de diagnóstico. Não versionar dados brutos no Git.
- **Próxima decisão:** se forem detectadas mudanças nos dados de entrada de 2020 ou eventos com ajuste retrospectivo, investigar seus efeitos antes de qualquer afirmação sobre a origem causal da diferença financeira.

## Integridade histórica

A reprodução original em `reproduzir_experimento.py`, seus hashes oficiais, o modo Control e a `main` devem permanecer imutáveis durante estes testes. A auditoria local é somente leitura sobre os dois snapshots e produção de novos relatórios em `output/`.

## 30. Resultado da auditoria por linha — snapshot antigo x renovado (22/10/2026)

**Evidência efetivamente recebida:** `pacote_auditoria_drift_historico.zip`, SHA-256 **`a6ce4625b3bbca7ad7440e94e3623f1fff20fe01bf0fc0b03d5a6e5592aa414c`**; executado em `2026-10-22T23:58:08.453028+00:00` na branch `research/v1.23.1-fresh-snapshot-20261022`.

- Snapshot base antigo **`e440f59da5e684f1de59cf447abfedd9aed3f7817b3d5d681058fe631276a575`** e novo unificado **`1c4d16173eaa5219a1bcc18cfa431cf1c096dbeaa26ca53aa03e9fcf047cff5c`** (conferidos pelos manifestos locais).
- **77/77 ativos solicitados** têm exatamente as mesmas **2.705 linhas até 06/10/2026** e os mesmos timestamps. Todos receberam **12 sessões novas posteriores à data**. Foram **zero datas ausentes e zero datas novas** na janela comparável.
- **32/77 ativos têm modificações materiais** em datas comuns, **75 barras no total, exclusivamente em 01, 02 e 03 de maio de 2018**. Os outros **45 ativos** não apresentam alteração material de OHLCV no intervalo.
- O ativo **CLMT**, excluído estruturalmente do universo efetivo, está entre os 32 afetados (duas barras). Portanto, **73 das 75 barras alteradas pertencem a 31 ativos efetivos**; não se deve atribuir diretamente contribuição econômica às duas barras de CLMT.
- 01/05/2018: 12 barras, **somente volume**. 02/05/2018: 31 barras, todas com volume e máxima alterados (28 mínimas, 7 aberturas e 1 fechamento). 03/05/2018: 32 barras, todas com volume alterado (28 máximas, 28 mínimas, 9 aberturas e 1 fechamento).
- Volume somado **apenas entre os ativos com divergência naquela data**: 02/05, **12.152.097 antigo vs. 216.979.152 renovado**; 03/05, **10.871.735 vs. 196.836.941**. O histórico original contém valores de volume muito inferiores aos da nova consulta. Essa coincidência de datas em muitos ativos é um sinal forte de anomalia localizada dos *insumos históricos*, mas **o motivo da anomalia não foi comprovado**.
- Dos 75 registros OHLCV, **somente dois fechamentos diferiram:** **SBFG em 02/05/2018 (18,930 vs. 18,954)** e **CAPR em 03/05/2018 (1,290 vs. 1,280)**. As demais divergências de preços foram predominantemente abertura, máxima e mínima, além do volume.
- **Corporate Actions:** sete novos registros de `cash_dividend` no snapshot atualizado, todos com ex-date em setembro de 2026: **CRM, ADC, BXMT, XEL, CRT, STAG e TOWN**. Não houve eventos de split anteriores a 06/10 incluídos/removidos nas evidências.
- **Normalização por splits:** as mesmas **75 barras** foram apontadas após normalização; nenhuma nova data com diferença surgiu no arquivo `divergencias_ohlcv_splits.csv`.
- **MGM e CXW** apresentam séries OHLCV equivalentes na janela até 06/10; a decisão diferente de 28/07/2020 requer investigação de **efeitos indiretos sobre o treinamento/calibração** e da composição dos folds, não simples divergência de preço desses dois ativos.

### Interpretação e limites

A comparação de arquivos agora localiza as diferenças em maio de 2018, **antes de todas as 1.560 decisões OOS comparadas**. Isso cria uma hipótese plausível de amplificação das diferenças por LightGBM + decisões de rotação + reinvestimento, mas **não prova a cadeia causal**. Os sete dividendos de setembro de 2026 e possíveis diferenças nas fronteiras de folds devido às 12 sessões adicionais constituem fatores a avaliar separadamente. Em especial, diferenças grandes de capital não permitem inferir automaticamente vazamento de futuro ou erro de algoritmo.

**Status: diagnóstico descritivo concluído.** A próxima análise é um **teste de sensibilidade controlada** em uma cópia descartável do snapshot atualizado: conservar os mesmos 77 símbolos, o mesmo calendário, os eventos e o mesmo cutoff 06/10/2026, alterando somente as 75 barras divergentes de maio de 2018 para seus valores do snapshot histórico verificado. Executar lado a lado o Control no snapshot renovado integral e nessa cópia. **Não alterar nenhum dos dois snapshots originais; não efetuar correções manuais em séries estruturais nem promover a cópia de diagnóstico como fonte oficial.**

- **Nova revisão experimental de diagnóstico:** `v1.23.6` na mesma branch `research/v1.23.1-fresh-snapshot-20261022`; script `pesquisar_causa_drift_2018_spyder.py`.
- **Implementação inicial:** commit **`e02b315d47474c21e94223bbee1738a799077e9c`**; testes de isolamento iniciados no commit **`df3f500f4e180803e722273d11ba9b0858b62000`**.
- **Saída a obter localmente:** `output/causal_drift_2018/<ID>/pacote_contrafactual_2018.zip`. Somente nesse resultado poderá ser quantificada a contribuição financeira condicional às diferenças de 2018. A referência para a comparação é **US$ 89.941.496,49** (snapshot renovado em 06/10), frente ao checkpoint histórico **US$ 587.165.941,13**.

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

