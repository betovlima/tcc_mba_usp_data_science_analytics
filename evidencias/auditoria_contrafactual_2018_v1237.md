# Auditoria experimental — restauração das 73 barras de maio de 2018 (TCC v1.23.7)

**Análise:** 22/10/2026. **Branch:** `research/v1.23.1-fresh-snapshot-20261022`.  
**Commit do motor e runner do contrafactual:** `78d473fb83d5ce2b89c3282f6b6ace4054c53bdc` (código da campanha v1.23.7; o contrafactual unificado foi atualizado nos commits `1562b251e4d71718a3741d71bb3a01cd77441877` e `26e8feaf2db4903e543cb296dfe815d155019800`).  
**Main preservada:** `4667d2ccee3f3441ee4d2b362ce734311cedd5e3`. **PR:** #18, Draft.  
**Arquivo auditado:** `pacote_contrafactual_2018.zip`. **SHA-256 verificado:** `612d068d0fac97423a34a6da14c98b51ce2e3fffbb172954b619fc1990d9d21b`.  
**Origem renovada:** snapshot `595b7a03dc2c8978487975cd3554b1ea65608741893f13208938dc624cb273cc`.  
**Snapshot experimental restaurado:** `0cae0311283e7d7419bef061b482e31c72eb972420c2f4178b0733a3d07a06f6`.  
**Base histórica do TCC:** `e440f59da5e684f1de59cf447abfedd9aed3f7817b3d5d681058fe631276a575`.  
**Referência imutável:** `pacote_reproducao_universo(4).zip`, SHA-256 `7a3aadd62b3f373a80c7df45ca702eb5397b4070c55e590f5e6fb1f6dd7afbbc`.

## 1. Intervenção realmente testada

Dois replays Control com os **mesmos 75 ativos efetivos**, sem DOC e CLMT, período OOS e corte final em **06/10/2026**. O contrafactual copiou o snapshot Alpaca renovado e restaurou **73 registros OHLCV, entre 31 ativos**, com valores do conjunto histórico conferido por SHA-256. Os três dias permitidos foram 01, 02 e 03/05/2018. A rotina declarou `original_files_modified=false` e `corporate_actions_unchanged=true`; as verificações de integridade da execução conferem a identidade e o universo. Nenhum novo download foi necessário.

| Data de maio de 2018 | Barras alteradas no universo efetivo |
|---|---:|
| 01/05/2018 | 12 |
| 02/05/2018 | 30 |
| 03/05/2018 | 31 |
| **Total** | **73 (31 ativos)** |

Por campo, os 73 registros contêm alterações em volume (73), máxima (57), mínima (54), abertura (16) e fechamento (2). As contagens de campos não devem ser somadas como número de barras; uma barra pode ter múltiplos campos alterados.

## 2. Resultado financeiro definitivo do teste controlado

| Experimento | Capital final 06/10/2026 (US$) | Sessões OOS | Registros de negociação |
|---|---:|---:|---:|
| Snapshot renovado, 75 ativos | **89.941.496,49095525** | 1.560 | 690 |
| Cópia renovada com 73 barras históricas restauradas | **587.165.941,1330212** | 1.560 | 710 |
| TCC v1.23.0, dados históricos congelados | **587.165.941,1330212** | 1.560 | 710 |

**Efeito da intervenção nesta execução:** aumento de **US$ 497.224.444,642066**, equivalente a **+552,83%** em relação ao replay atualizado, ou multiplicação do capital por **6,52831**. A diferença percentual do replay atualizado frente ao congelado é **−84,6821%**.

**Validação independente de arquivos:** `fresh_with_original_2018_bars/trades.csv` é **byte a byte idêntico** ao `trades.csv` da reprodução original (710 linhas e 101 colunas). Nos `predictions.csv` são iguais as 1.560 escolhas, os valores econômicos e as outras 33 colunas; as **únicas 552 células divergentes** são `fold_test_end` no fold 3: o replay sobre o snapshot renovado preserva o fim de cobertura modelado em 22/10/2026, enquanto o pacote congelado registrou 06/10/2026. A simulação encerrou efetivamente em 06/10 em ambos os casos. As margens e pontuações de calibração do cenário restaurado coincidem com o TCC oficial; o `scenario` textual mudou de `TCC_SELECTED_UNIVERSE` para `U67_BASELINE`.

A verificação do controle também foi independente: `fresh_reference/predictions.csv` e `fresh_reference/trades.csv` são byte a byte idênticos aos arquivos da v1.23.7 anteriormente enviados para o corte de 06/10.

## 3. Onde o capital se separa

| Fold | Renovado | Com 73 barras restauradas |
|---|---:|---:|
| 1, 2020–2022 | US$ 51.308,85624000585 | US$ 67.999,9480329993 |
| 2, 2022–2024 | US$ 4.160.778,0228136093 | US$ 10.968.842,009933945 |
| 3, 2024–2026 | US$ 89.941.496,49095525 | **US$ 587.165.941,1330212** |

Os calendários, folds e sessões OOS são os mesmos nos dois replays. **192 das 1.560 escolhas diferem** entre controle e restaurado. A primeira divergência é em **28/07/2020**, com **CXW** no controle renovado e **MGM** no restaurado. Os patrimônios nesse dia foram **US$ 11.174,693058** e **US$ 11.433,537718**, respectivamente. A diferença de capital é amplificada ao reinvestir e tomar outras rotações.

## 4. Calibração — mecanismo observável

Os mesmos quatro candidatos de margem `(0, 0.0025, 0.005, 0.01)` foram avaliados nos dois casos. Contudo, as margens selecionadas mudaram:

| Fold | Renovado | Restaurado e TCC original |
|---|---:|---:|
| 1 | 0,0100 | 0,0000 |
| 2 | 0,0000 | 0,0100 |
| 3 | 0,0100 | 0,0000 |

**Inferência delimitada:** a mudança de 73 observações históricas altera cálculos de atributos e/ou ajustes de modelos; os escores de calibração mudam e a política escolhe margens diferentes. Isso explica *como* a intervenção pode influenciar decisões posteriores. O material ainda não isola se **volume**, **abertura/máxima/mínima/fechamento**, interações entre campos ou poucos ativos/datas são os mecanismos dominantes.

## 5. Conclusão científica e limitações

**Suficiência demonstrada, nesta execução:** restaurar essas 73 barras do histórico antigo foi suficiente para recuperar integralmente o resultado econômico original, inclusive operações e calibração, preservando os demais dados do snapshot novo. Os valores até 2018 usados no treino/calibração estavam na janela anterior ao início OOS, e as decisões mudaram a partir de 2020.

**Não demonstrado:** que a versão histórica da Alpaca era mais correta, que as revisões foram maliciosas, que uma única linha causou todo o efeito, ou que o capital original teria sido alcançado numa operação prospectiva. O modelo é **altamente sensível à versão dos dados históricos**, um limite de robustez e reprodutibilidade entre snapshots que deve ser reportado com destaque no TCC.

O estudo também não permite inferir uma taxa de retorno realizável, nem comparar causalmente universos diferentes: **somente 75 ativos efetivos** foram usados. A reprodução oficial de US$ 587 milhões é válida *condicionada ao dataset congelado* e não deve ser sobrescrita pela cópia contrafactual.

## 6. Próxima investigação científica proposta

Realizar um teste fatorial **volume versus preços OHLC** com quatro cenários mantendo o mesmo motor e snapshot: (1) novo integral; (2) somente volume de 2018 restaurado; (3) somente OHLC de 2018 restaurado; (4) as 73 barras inteiras restauradas. Comparar capitais, margens por fold, escolhas e impacto de interação. Se necessário, reduzir para os dias 01, 02 e 03/05 ou ativos individuais em estudos posteriores; evitar escolher combinações pelo melhor resultado. O objetivo é **atribuir o efeito de uma anomalia nos insumos**, não otimizar o capital.

Até executar esses controles, não é científico concluir que volume sozinho ou preços sozinhos explicam a divergência.
