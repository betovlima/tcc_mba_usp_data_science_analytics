# v1.25.6 — melhoria da nitidez dos gráficos de overfitting

**Data:** 23/10/2026. **Branch:** `research/v1.23.1-fresh-snapshot-20261022`. **Arquivo modificado:** `consolidar_evidencias_overfitting_spyder.py`.

## Motivação

A execução v1.25.5 concluiu corretamente a consolidação e entregou sete gráficos em PNG (~165 DPI), com possível serrilhado/pixelização ao ampliar ou inserir em documentos. O problema é de **formato de saída**, não de valor financeiro ou de estatística.

## Modificação

- Os mesmos sete gráficos passam a ser exportados em **PNG 320 DPI** e **SVG vetorial** (dois arquivos por figura), sem mudar os dados, escalas, séries, cálculos e rótulos científicos.
- O pacote agora se chama `pacote_consolidacao_overfitting_v1256.zip` e contém 14 imagens (7 PNG + 7 SVG), `resumo_consolidado.json` e `LEIA_ME_GRAFICOS.md`.
- Relatório associa cada PNG ao SVG correspondente.
- Nenhum treinamento, acesso à Alpaca, backtest financeiro, alteração da estratégia ou atualização do snapshot.
- Contrato SHA-256 continua exigindo o vintage histórico `e440f59da5e684f1de59cf447abfedd9aed3f7817b3d5d681058fe631276a575`.

## Verificação local

O script v1.25.6 foi executado utilizando:

- `pacote_generalizacao_587m_v1254(1).zip`, SHA-256 `0b455cbdfc5dc4c83818d30ba8e868634d91e84e5ac5ab49b869c505bcc9a060`.
- `pacote_reproducao_universo(20261023-222155).zip`, SHA-256 `b62982a462fe0c8b8c1ea1a3c9df5ea4d719716956d4b6680f361847eaa1685d`.

**Status observado:** `completed`, sete PNGs, sete SVGs, ZIP gerado; figuras SVG verificadas no pacote; exemplo PNG de capital: **3477 × 1560 px**. Indicadores preservados: 225 modelos, gap mediano **+161,318209%**, **224/225** RMSE OOS > treino, **222/354** rotações com valor adicional positivo, PBO de seleção histórico **53,57%** em log-growth e **59,52%** em Sharpe.

**Limitação:** melhoria exclusivamente visual. O conjunto segue retrospectivo, sem novo período prospectivo independente. Não inferir ausência de overfitting econômico a partir da qualidade gráfica.
