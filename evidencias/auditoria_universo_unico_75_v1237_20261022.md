# Auditoria conclusiva — execução Control com um único universo de 75 ativos (v1.23.7)

**Data:** 22/10/2026 (cortes financeiros em 06/10/2026 e 21/10/2026).  
**Branch:** `research/v1.23.1-fresh-snapshot-20261022`.  
**Código da revisão executada:** commit `78d473fb83d5ce2b89c3282f6b6ace4054c53bdc`.  
**Origem da branch / main preservada:** `4667d2ccee3f3441ee4d2b362ce734311cedd5e3`.  
**PR:** #18, mantido Draft.  
**Pacote recebido:** `pacote_comparacao_cutoffs_75.zip`. **SHA-256 do ZIP:** `da1001963f1a57cc2576aa6e55cffbaf31150bb20c824e6fca2419c381175e79`.  
**Pacote comparativo anterior (v1.23.4):** `pacote_comparacao_cutoffs.zip`; SHA-256 `fb9b87c43d6233ae8334e6dc083b42280bb0778cbcb31c698a2c12766963ab61`.  
**ZIP da Winner operacional comparado:** `market_cycle_trader_20261022T151609-7d2de3e9(1).zip`; SHA-256 `7c799e242ec32efe64c391c81093f45493483a40b93ade55f92d4d64559d8998`.  
**Release científica original:** v1.23.0; ZIP `pacote_reproducao_universo(4).zip`, SHA-256 `7a3aadd62b3f373a80c7df45ca702eb5397b4070c55e590f5e6fb1f6dd7afbbc`.

## 1. Identidade do único universo de 75

O novo `snapshot_75_manifest.json` contém **75 ativos solicitados e 75 efetivos**, sem `DOC` ou `CLMT`. O arquivo `universe_75.csv` confirma exatamente a lista utilizada. Não houve descoberta nem expansão do universo.

- **Snapshot novo U75:** `595b7a03dc2c8978487975cd3554b1ea65608741893f13208938dc624cb273cc`.
- **Snapshot anterior U77 usado para migração offline:** `1c4d16173eaa5219a1bcc18cfa431cf1c096dbeaa26ca53aa03e9fcf047cff5c`.
- **Conferência por arquivo:** todos os **150/150 hashes SHA-256** dos 75 pares RAW + Corporate Actions são idênticos aos mesmos arquivos no snapshot anterior. Dos 154 arquivos anteriores, foram retirados apenas os quatro `raw_bars/DOC.csv`, `corporate_actions/DOC.csv`, `raw_bars/CLMT.csv` e `corporate_actions/CLMT.csv`.
- Todas as séries contêm 2.717 linhas RAW até 22/10/2026; o snapshot já estava disponível localmente e foi reutilizado, não rebaixado.

## 2. Resultado financeiro

| Simulação | Data final | Capital final US$ | Sessões OOS | Negociações |
|---|---|---:|---:|---:|
| Oficial v1.23.0 com dados históricos congelados | 06/10/2026 | 587.165.941,1330212 | 1.560 | referência histórica |
| TCC v1.23.4 com dados renovados, executor legado (75 efetivos) | 06/10/2026 | 89.941.496,49095525 | 1.560 | 690 |
| **TCC v1.23.7, treino e calibração em um universo único de 75** | **06/10/2026** | **89.941.496,49095525** | **1.560** | **690** |
| TCC v1.23.4 com dados renovados | 21/10/2026 | 76.152.561,33521412 | 1.571 | 698 |
| **TCC v1.23.7, universo único de 75** | **21/10/2026** | **76.152.561,33521412** | **1.571** | **698** |
| Winner operacional #30, no mesmo corte | 21/10/2026 | 76.152.561,33521412 | 1.571 | 698 |

**Comparação linha a linha v1.23.7 × v1.23.4:** os dois `predictions.csv` são inteiramente idênticos nos respectivos cortes (1.560 e 1.571 linhas, 34 colunas). Os dois `trades.csv` também são inteiramente idênticos (690 e 698 linhas, 101 colunas), incluindo datas, ativos, quantidades, preços e todas as colunas diagnósticas. As calibrações têm as mesmas margens e pontuações; **a única diferença nos CSVs de calibração é o rótulo textual `scenario`, que ainda diz `U67_BASELINE` na nova versão**, herança da função de treinamento reutilizada. Não é indicativo de treino de apenas 65: o executor passa os 75 ativos à função. Os folds escolheram as margens 0,01, 0,00 e 0,01, respectivamente.

**Paridade com a Winner operacional:** 1.571/1.571 ativos escolhidos em igualdade; 698/698 registros de negociação em igualdade de ação, ativo, quantidades, preços e taxas. Os CSVs usam formatos de timestamp distintos (ambos equivalentes em UTC), e as diferenças de `strategy_equity` são da ordem de 3 × 10⁻⁸ por arredondamento de ponto flutuante. Capital final registrado exatamente idêntico, `US$ 76.152.561,33521412`.

Métricas TCC v1.23.7 em 21/10: **Sharpe 2,310440236459238; máximo drawdown −41,682843%; buy-and-hold US$ 30.822,892636814973; 348 rotações** (349 compras, 348 vendas intermediárias e uma liquidação final).

## 3. Cortes temporais e retorno acumulado

Os folds 1 e 2, bem como a calibração, são os mesmos entre 06/10 e 21/10. Capital composto ao fim dos folds: `US$ 51.308,85624000585` (fold 1), `US$ 4.160.778,0228136093` (fold 2) e `US$ 89.941.496,49095525` ou `US$ 76.152.561,33521412` (fold 3 conforme o corte).

- Mudança apenas da **fotografia histórica** na mesma data 06/10: de 587.165.941,13 para 89.941.496,49, diferença **−84,6821%**. A comparação com os 587 milhões mantém ressalvas de seleção retrospectiva do universo original.
- Mudança somente do **cutoff** sobre a mesma fotografia atualizada: de 89.941.496,49 (06/10) para 76.152.561,34 (21/10), diferença **−15,3310%**.
- Os 1.560 ativos escolhidos dos dois replays v1.23.7 são iguais até 06/10. Na última sessão de 06/10, o replay curto liquida `CAPR` e paga `US$ 1.897,45` de taxas; o replay longo mantém a posição e marca `US$ 89.943.393,94095525`, explicando a única diferença de patrimônio na janela sobreposta.

## 4. Interpretação metodológica e limites

1. **Conclusão suportada:** neste snapshot e nestes dois cortes, consolidar a preparação/treinamento da pesquisa em um painel de 75 ativos, em lugar do executor histórico de base+adicionais, preservou integralmente decisões, negociações, calibração numérica e patrimônio. Não é necessário manter o tratamento em dois grupos para produzir estes resultados observados.
2. **Não suportado:** dizer que DOC e CLMT “sempre destroem capital”. Eles foram **excluídos estruturalmente** dos dois replays econômicos comparados; os resultados presentes não medem causalmente sua inclusão. Os dados deles, porém, não são necessários no fluxo científico atual.
3. **Não suportado:** afirmar que a diferença para os 587 milhões é exclusivamente causada pelas alterações OHLCV de maio de 2018. A auditoria anterior identificou 73 barras materiais de 31 ativos elegíveis, mas ainda falta o experimento de intervenção.
4. **Novo teste mantido:** executar `pesquisar_causa_drift_2018_spyder.py` em modo **75 ativos efetivos**, substituindo **apenas 73 barras** de 1–3/05/2018 numa cópia temporária do snapshot atual, com controles idênticos de folds, dados corporativos e custo; comparar o capital em 06/10. O teste não deve alterar a `main`, o TCC v1.23.0, a Winner ou snapshots originais.
5. **Estado documental:** preservar `CONTEXTO_MESTRE.md`, `MANIFEST.md`, `README.md` e este registro; um novo capital obtido no contrafactual será etiquetado explicitamente como experimento de sensibilidade, **não** como resultado oficial.

## 5. Rastreabilidade do código

- Código de treinamento único U75: `d2f219cdd78cbee7770fd17f9634a432411f0ac1`.
- Runner de coleta/avaliação U75: `8c2fd0ad05cf9af06021e0e0ad506cf2a18cbb7e`.
- Testes de reuso/migração U77→U75: `78d473fb83d5ce2b89c3282f6b6ace4054c53bdc`.
- Todos integrados em `research/v1.23.1-fresh-snapshot-20261022`. CI do commit `78d473fb83d5ce2b89c3282f6b6ace4054c53bdc`: **110 testes aprovados**, com dois avisos de dependências previamente documentados.
