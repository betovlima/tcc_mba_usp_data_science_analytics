# Auditoria RAW, dividendos e decisões de outubro — TCC v1.23.9

**Branch:** `research/v1.23.1-fresh-snapshot-20261022` (sem merge). **Código experimental v1.23.9:** `e879e34ff945ece2f0e684f3d140e2e88bc93a1c` (`diagnosticar_decisoes_outubro_2026_spyder.py`). **Testes de contrato:** `575d047af09f65769aa6c97d127ee7fede6b810f`. **Release original imutável:** v1.23.0, `main` em `4667d2ccee3f3441ee4d2b362ce734311cedd5e3`. 

**Evidências conferidas:** `pacote_comparacao_cutoffs_75.zip`, SHA-256 `da1001963f1a57cc2576aa6e55cffbaf31150bb20c824e6fca2419c381175e79`; `pacote_auditoria_drift_historico.zip`, SHA-256 `a6ce4625b3bbca7ad7440e94e3623f1fff20fe01bf0fc0b03d5a6e5592aa414c`; `pacote_contrafactual_2018.zip`, SHA-256 `612d068d0fac97423a34a6da14c98b51ce2e3fffbb172954b619fc1990d9d21b`. **Pacote deste diagnóstico gerado sobre as saídas recebidas:** `pacote_diagnostico_outubro_2026.zip`, SHA-256 `6be2c89a77532786c12980fef6a9a77aaade01a79d4172e15433ccbfe869a497`.

## 1. Respondendo à dúvida: mudou de RAW para série ajustada por dividendos?

**Não.** O endpoint de barras históricas da Alpaca é chamado explicitamente com `adjustment=Adjustment.RAW`, `feed=DataFeed.SIP` e `TimeFrame.Day` no arquivo `reproducao/dados.py`. As saídas monetárias registram `feed=sip`, `adjustment=raw` e `timeframe=1Day` nas duas campanhas.

A preparação interna em `reproducao/preparacao.py::split_normalize` ajusta somente ações corporativas com `action_type` igual a `forward_split` ou `reverse_split`. Preços anteriores ao split são multiplicados por `old_rate/new_rate` e volumes por `new_rate/old_rate`. O manifesto atualizado registra **20 aplicações de split** nos 75 ativos. O processo **não ajusta preços por `cash_dividend`**. Os arquivos de eventos contêm mais sete dividendos de setembro/2026 (CRM, ADC, BXMT, XEL, CRT, STAG, TOWN), mas não há efeito desses sete registros sobre a normalização por splits.

Isso não equivale a atribuir todas as alterações históricas às mudanças no feed. A política de agregação da Alpaca por trás dos dois snapshots precisa de verificação externa com a fornecedora; o que foi auditado é o conteúdo de cada versão gravada e sua identidade SHA.

## 2. Diferenças efetivas em datas antigas comuns, antes do novo cutoff

Dos **75 ativos do único universo da pesquisa**, **31** tiveram barras RAW materialmente diferentes, totalizando **73 barras nos dias 1–3 de maio de 2018**. Os **outros 44** tiveram OHLCV equivalente nas mesmas datas até 06/10/2026. Nenhuma data de negociação faltou ou foi acrescentada dentro da janela comum. As 12 sessões após 06/10 anexadas ao snapshot novo não constituem mudança retroativa dos preços passados.

| Data | Barras diferentes |
|---|---:|
| 01/05/2018 | 12 |
| 02/05/2018 | 30 |
| 03/05/2018 | 31 |
| **Total** | **73** |

**Diferenças de atributos entre os 73 registros:** volume em **73** barras, máxima em 57, mínima em 54, abertura em 16 e fechamento em apenas **duas** (SBFG 02/05/2018: 18,930 → 18,954; CAPR 03/05/2018: 1,290 → 1,280). As categorias são sobrepostas. Exemplos: NVDA volume em 02/05 de **362.289 antigo vs 7.068.474 novo**; em 03/05 de **312.289 vs 11.350.383**. CAPR, em 03/05, passou de **204 vs 228.386** em volume. Dividendos em dinheiro não produzem alterações dessa natureza no campo volume de uma solicitação RAW.

No experimento contrafactual documentado, o restauro **somente das 73 barras** numa cópia isolada das séries renovadas recuperou exatamente **US$ 587.165.941,1330212**, o capital da reprodução original, inclusive as 710 operações. Isso comprova suficiência da intervenção no conjunto analisado; **não comprova que os preços/volumes antigos são mais corretos**.

## 3. Rotações e resultados entre 06/10 e 21/10/2026

A comparação correta do período adicional usa o capital **marcado** no dia 06/10 no replay contínuo, US$ **89.943.393,94095525**, não o capital de liquidação do replay encerrado em 06/10 (US$ 89.941.496,49095525). O capital em 21/10 foi **US$ 76.152.561,33521412**: diferença de **−US$ 13.790.832,60574 (−15,33279%)**, em **11 sessões**. A pior queda diária foi **21/10, −11,04576%**.

| Entrada | Saída | Ativo | Score escolhido na entrada | Retorno da posição | Lucro/perda realizado |
|---|---|---|---:|---:|---:|
| 08/10 | 12/10 | WDAY | 0,386285 | +2,9172% | +US$ 2.669.227 |
| 12/10 | 16/10 | CAPR | 0,392323 | −7,4454% | −US$ 7.018.088 |
| 16/10 | 20/10 | ADEA | 0,345593 | −3,0303% | −US$ 2.644.714 |
| 20/10 | 21/10 | CAPR | 0,455803 | −9,9541% | −US$ 8.420.022 |

A posição CAPR fechada em 08/10 havia começado em 25/09 e **não deve ser contabilizada como nova entrada neste período**. Os lucros/perdas realizados de posições **não podem ser somados diretamente** ao delta diário de patrimônio sem reconciliar marcações e datas.

O `decision_score` não é probabilidade de rentabilidade futura nem retorno efetivamente previsto para o dia seguinte: é utilidade estimada por modelo de múltiplos horizontes. Os quatro scores acima foram associados às datas de entrada, mas a amostra tem apenas quatro posições concluídas, insuficiente para inferência sobre calibração. O score em 21/10 (0,67129) é de uma sessão de **liquidação final**, não um novo sinal de compra.

**Limitação dos artefatos:** embora as 1.571 escolhas e operações estejam disponíveis, o pacote não apresenta uma matriz integral de scores de todos os 75 ativos por sessão. Campos como `raw_best_score`, `current_score` e `final_action_score` no CSV de operações são NaN neste intervalo. Não é possível reconstruir fielmente a classificação das alternativas rejeitadas sem ampliar a coleta de diagnósticos antes do backtest — e sem mudar regras de decisão.

## 4. Testes da v1.23.9 e próximo experimento

- Script **somente leitura** `diagnosticar_decisoes_outubro_2026_spyder.py` lê o ZIP U75 da campanha existente, verifica feed RAW, universo 75, 12 datas (base + 11 pregões), pares BUY/venda realizados e escreve três arquivos: `sessoes.csv`, `posicoes_encerradas.csv` e `resumo.json`. Saída: `output/diagnostico_decisoes_outubro_2026/pacote_diagnostico_decisoes_outubro_2026.zip`. **Não há novo ajuste de modelo, consulta à corretora ou mudança nos 75 ativos**.
- Testes em `tests/test_diagnosticar_decisoes_outubro.py`: regra de fonte `Adjustment.RAW`/`DataFeed.SIP`; cash dividend deixa as barras inalteradas em `split_normalize`; split multiplica preço/volume pelo fator adequado; valida separação de scores e retorno realizado; recusa janela incompleta.
- Prosseguir na análise de **volume-only versus OHLC-only** já publicada no script `pesquisar_atribuicao_ohlc_volume_spyder.py` da v1.23.8. Essa atribuição ainda não possui resultado financeiro. Não fazer correções manuais nem selecionar o snapshot com maior capital.

**Regra científica:** fonte RAW e ajuste por split são conceitos distintos; mudança concentrada de volume em barras RAW não deve ser classificada como ajuste por dividendos sem evidência. A grande diferença de capital derivou da sensibilidade do aprendizado a observações passadas revisadas; o recuo em outubro é um segundo efeito, decorrente de retornos negativos em posições mantidas nas 11 sessões adicionais.
