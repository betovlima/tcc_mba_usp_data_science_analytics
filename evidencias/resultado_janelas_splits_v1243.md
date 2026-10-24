# Auditoria experimental das janelas em torno de splits — TCC v1.24.3

**Data:** 23/10/2026. **Branch:** `research/v1.23.1-fresh-snapshot-20261022`. **Código executado:** `auditar_janelas_splits_spyder.py`, commit `5f90ff28ef53cc94eb25039411d94e262ed15f66`; testes `tests/test_auditar_janelas_splits.py`, commit `bee74758394dc88cd0a9ea8d2c612d62a224fa6b`. **HEAD anterior a este relatório:** `d6f474c52cdb011a13eb14291bcb9803a5467a99`. **`main` inalterada:** `4667d2ccee3f3441ee4d2b362ce734311cedd5e3`. PR #18 Draft.

**Pacote auditado:** `pacote_janelas_splits.zip`; SHA-256 **`1a73fcddfa794121eff21af8d1b87555f906268341835258699c16c304538326`**. Arquivos: `resumo.json` e `eventos_e_janelas.csv` (**21 linhas × 30 colunas**). **Snapshot fonte, 75 ativos:** `595b7a03dc2c8978487975cd3554b1ea65608741893f13208938dc624cb273cc`.

## 1. Desenho experimental

Para cada um dos **21 splits** encontrados no arquivo retrospectivo de Corporate Actions do universo fixo de **75 ativos**, comparar, **somente em memória**, dados diários RAW intactos contra a antiga rotina `split_normalize` aplicada apenas ao evento examinado. Medir, dentro dos limites de dados disponíveis, **até 60 sessões anteriores e 60 posteriores**: (i) 52 atributos explicativos e (ii) sete alvos de treinamento (utilidades horizontes 5, 10, 20, 40, 60, utilidade agregada e retorno log agregado). Registrar gaps RAW e gap neutro à quantidade de ações na abertura posterior. **Não há treinamento nem backtest financeiro neste pacote.**

Critério de diferença material dos valores numéricos `np.isclose(rtol=1e-10, atol=1e-12, equal_nan=True)`. As somas abaixo se referem a **janelas por split**; não representam observações estatisticamente independentes entre eventos ou campos.

## 2. Totais efetivamente observados

| Indicador | Resultado |
|---|---:|
| Splits no snapshot | **21** |
| Splits dentro da cobertura diária RAW | **20** (13 forward, 7 reverse) |
| Split sem cobertura anterior suficiente | **RACE 04/01/2016** (fora da cobertura) |
| Eventos com pelo menos uma mudança de atributo anterior | **0/20** |
| Linhas de atributo anteriores comparadas | **1.080** |
| Células de atributo anteriores materialmente diferentes | **0** |
| Eventos com alteração de alvos ANTES da ex-date | **18/20** |
| Linhas de alvo anteriores diferentes | **1.062** |
| Células de alvo anteriores diferentes | **4.464** |
| Eventos com alteração de atributos DEPOIS da ex-date | **19/20** |
| Linhas de atributo posteriores diferentes | **1.130** |
| Células de atributo posteriores diferentes | **38.606** |
| Eventos com alteração de alvos DEPOIS da ex-date | **0/20** |
| Células de alvo posteriores diferentes | **0** |
| Arquivos RAW modificados | **0** |
| LightGBM reentrenado / novas previsões e rotações | **Não** |
| Capital recalculado | **Não** |

A maioria dos 18 eventos com alvos anteriores afetados apresentou **59 das 60 sessões** e **248 células de rótulos** alteradas. Os dois casos sem alterações de rótulos anteriores (**YANG, ex-date 24/03/2016; APD, ex-date 03/10/2016**) ocorrem perto do começo do histórico, sem observações de atributo suficientes para montar a janela antecedente. **YANG 24/03/2016 também não apresentou atributos posteriores avaliáveis** devido ao aquecimento requerido; APD teve 50 linhas posteriores com 1.562 células alteradas. Não classificar esses eventos como economicamente inofensivos.

## 3. Exemplos numéricos extraídos do pacote

| Ativo e data efetiva | Gap RAW da abertura | Gap de abertura neutro ao split |
|---|---:|---:|
| **NVDA 10/06/2024**, 10:1 | **−90,04%** | **−0,43%** |
| **AMZN 06/06/2022**, 20:1 | **−94,88%** | **+2,37%** |
| **AAPL 31/08/2020**, 4:1 | **−74,48%** | **+2,07%** |
| **TSLA 31/08/2020**, 5:1 | **−79,91%** | **+0,44%** |
| **PXLW 09/06/2025**, reverso 12:1 | **+1.088,49%** | **−0,96%** |
| **NBR 23/04/2020**, reverso 50:1 | **+4.938,95%** | **+0,78%** |

O gap ajustado pela quantidade foi calculado, para diagnóstico apenas, como `(preco_abertura_seguinte × new_rate / old_rate) / fechamento_anterior − 1`. **Os preços e volumes RAW de origem não foram alterados.** Para os 20 splits cobertos, o maior gap neutro em módulo foi **4,57%** (YANG 24/03/2016, início da cobertura).

## 4. O que isso demonstra — e o que não demonstra

**Demonstrado:** janelas de atributos anteriores ao evento permaneceram invariantes, conforme a tolerância usada; no entanto, os alvos supervisionados que cruzam o evento e os atributos históricos imediatamente posteriores diferiram amplamente entre cálculo ingênuo sobre RAW e a normalização legada. Os preços RAW registram a alteração de unidade de cotação ocasionada pelo desdobramento, e é incorreto interpretar automaticamente um gap bruto de −90%/ +900% como perda/ganho econômico.

**Não demonstrado:** que o modelo tenha usado rótulos ainda imaturos no treinamento; que haja vazamento de anúncio antes da data conhecida; que rótulos historicamente ajustados sejam incorretos; que a versão antiga tenha produzido capital artificial; ou que os resultados oficiais sejam válidos/inválidos após troca de política. Um **rótulo de retorno futuro** somente se torna disponível após seu horizonte se encerrar, sendo legítimo na base de treinamento futura se for calculado economicamente e se o esquema walk-forward respeitar a maturidade dos rótulos. A ausência de `first_seen_at_utc` continua impedindo reconstituição perfeita da informação pública a cada data passada.

**Achado independente:** a diferença entre US$ 587,17 milhões e US$ 89,94 milhões já foi atribuída em ensaio separado à restauração de 73 registros **RAW** de maio/2018. Não confundir esse fenômeno com mudanças de indicadores ao atravessar splits.

## 5. Próxima verificação recomendada

Construir **retornos econômicos futuros diretamente das barras RAW com fatores de quantidade efetivados entre a entrada e a saída**, sem aplicar ajustes retrospectivos à série de OHLCV. Por horizonte 5, 10, 20, 40 e 60, calcular (a) retorno ingênuo usando exclusivamente os preços RAW, (b) retorno econômico contabilizando quantidade por split depois da compra e até a saída, (c) retorno do legado reexpresso em memória apenas como verificação numérica. Auditar igualdade (b) × (c), identificar divergências (a) × (b) e registrar as datas quando o alvo se torna conhecido. Sem treinar políticas nem reportar capital até completar custos, rótulos de risco/path e custódia fracionária.
