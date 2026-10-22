# Auditoria de paridade TCC × MCT e sensibilidade ao snapshot — 22/10/2026

## Identificacao e rastreabilidade

- **Branch de pesquisa:** \`research/v1.23.1-fresh-snapshot-20261022\` (não criar outra branch; não fazer merge na \`main\` até concluir a auditoria de insumos).
- **Revisão testada do TCC:** v1.23.4, commit **\`bd99b592c03dadec2cf45ade0b09e8ee07148d3b\`**, resultado do executor \`executar_atualizacao_series_spyder.py\` com dados Alpaca renovados.
- **Base \`main\` do TCC antes da pesquisa:** commit **\`4667d2ccee3f3441ee4d2b362ce734311cedd5e3\`**; release oficial etiquetada \`v1.23.0\`. Não alteradas.
- **MCT Winner #30 — execução estudada:** commit em runtime **\`adb764df4bbf36bcf63af1c9017b9a3dcb6fb844\`**, API v10.8.125; identificador \`20261022T151609-7d2de3e9\`.
- **TCC com snapshot atual:** SHA-256 de identidade **\`1c4d16173eaa5219a1bcc18cfa431cf1c096dbeaa26ca53aa03e9fcf047cff5c\`**, 77 ativos solicitados e 75 elegíveis.
- **TCC oficial congelado (67 da base):** SHA-256 de identidade **\`e440f59da5e684f1de59cf447abfedd9aed3f7817b3d5d681058fe631276a575\`**. Os dez adicionais originais vivem no cache histórico local e têm seus 20 hashes fixados em \`evidencias/manifest_universo_congelado.json\`.
- **ZIP da comparação atualizada:** \`pacote_comparacao_cutoffs.zip\`, SHA-256 **\`fb9b87c43d6233ae8334e6dc083b42280bb0778cbcb31c698a2c12766963ab61\`**.
- **ZIP do TCC congelado conferido:** \`pacote_reproducao_universo(4).zip\`, SHA-256 **\`7a3aadd62b3f373a80c7df45ca702eb5397b4070c55e590f5e6fb1f6dd7afbbc\`**.
- **ZIP do MCT Winner #30 conferido:** \`market_cycle_trader_20261022T151609-7d2de3e9(1).zip\`, SHA-256 **\`7c799e242ec32efe64c391c81093f45493483a40b93ade55f92d4d64559d8998\`**.
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

Executar \`auditar_drift_historico_spyder.py\` (introduzido na mesma branch após esta auditoria), usando:

- **Origem 1:** \`dados/u67\` para os 67 originais e \`dados/.u67_expansion_cache/full_raw_20261006\` para os dez adicionais. Conferir os hashes antes de comparar.
- **Origem 2:** último snapshot \`dados/.u67_fresh_runs/asof_2026-10-22_*/universe_77\`, validado integralmente por SHA-256. Nunca sobrescrever nenhuma origem.
- **Janela:** registros de preços e eventos até 06/10/2026; registrar novos pregões além da janela separadamente.
- **Diagnósticos:** comparar por ativo/data os OHLCV brutos (exatos e com tolerância numérica), preços após normalização de splits, diferenças de eventos corporativos por data efetiva e diferenças anteriores a **28/07/2020**. Gerar arquivos de evidência, incluindo primeira divergência e variáveis afetadas.
- **Critério:** identificar observações divergentes e classificá-las; **não** substituir símbolos, corrigir dados manualmente, alterar modelo, parâmetros, política Control ou Worker do MCT.
- **Saída:** \`output/auditoria_drift_historico/\` e ZIP de diagnóstico. Não versionar dados brutos no Git.
- **Próxima decisão:** se forem detectadas mudanças nos dados de entrada de 2020 ou eventos com ajuste retrospectivo, investigar seus efeitos antes de qualquer afirmação sobre a origem causal da diferença financeira.

## Integridade histórica

A reprodução original em \`reproduzir_experimento.py\`, seus hashes oficiais, o modo Control e a \`main\` devem permanecer imutáveis durante estes testes. A auditoria local é somente leitura sobre os dois snapshots e produção de novos relatórios em \`output/\`.