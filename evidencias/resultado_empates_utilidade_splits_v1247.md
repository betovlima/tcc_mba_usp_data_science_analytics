# Resultado do diagnóstico de empates numéricos das utilidades — v1.24.7

**Data:** 23/10/2026. **Branch exclusiva:** `research/v1.23.1-fresh-snapshot-20261022`. **`main` preservada:** `4667d2ccee3f3441ee4d2b362ce734311cedd5e3`. PR #18 Draft. Script produtor: `diagnosticar_empates_utilidade_splits_spyder.py` commit `bf13e05bba9fa918aa2fe9549c24fe8c50cd7dfb`. Testes: `tests/test_diagnosticar_empates_utilidade_splits.py`, commits `74d4da3d68a3f93c5b6f6e202035c9fa230f9740`, `8fec45997c57cd58fdd6b8c5f7a5201a44488bf2`.

**Pacote recebido:** `pacote_empates_utilidade_splits.zip`; **SHA-256 `92676f9700294ac7a54363545c6b16bc33f53eb7457eb0f6172bbde447eaab12`**. `ZipFile.testzip() == None`; arquivos `resumo.json`, `resumo_decisoes.csv` (3 linhas × 18 colunas) e `comparacoes_divergentes.csv` (3 linhas × 19 colunas). Snapshot U75 preservado SHA-256 `595b7a03dc2c8978487975cd3554b1ea65608741893f13208938dc624cb273cc`.

## 1. Teste efetivamente realizado

O programa inspecionou as **três decisões reais** com diferença residual nos 15 rótulos da v1.24.6. Em cada decisão comparou exatos `float64` de 60 fechamentos futuros com a abertura na entrada (`close > entry`) e 59 diferenças entre fechamentos adjacentes (`diff(close)>0`). O cenário econômico preserva OHLCV RAW e aplica somente multiplicadores de quantidade; o controle legado reexpressa os preços exclusivamente **em memória**.

## 2. Resultado — divergências localizadas exatamente

| Ativo | Decisão | Sessão em que mudou `close>entry` | Preço econômico: `close-entry` | Controle legado: `close-entry` | Distância em ULP do cenário econômico | Distância em ULP do legado |
|---|---|---|---:|---:|---:|---:|
| YANG | 16/10/2024 | **19/11/2024** (24º fechamento futuro) | **+4,4408920985e−16** | **0** | **1** | 0 |
| CAPR | 20/05/2019 | **07/06/2019** (13º fechamento) | **+5,5511151231e−17** | **0** | **1** | 0 |
| CAPR | 28/05/2019 | **12/08/2019** (53º fechamento) | **+5,5511151231e−17** | **0** | **1** | 0 |

Exemplos exatos dos 64 bits:
- **YANG:** econômico `close=0x1.f5c28f5c28f5dp+1`, `entry=0x1.f5c28f5c28f5cp+1`; legado `close=entry=0x1.399999999999ap+6`. Valores exibidos arredondados: US$ 3,92 e 78,40.
- **CAPR 20/05/2019:** econômico `close=0x1.47ae147ae147cp−2`, `entry=0x1.47ae147ae147bp−2`; legado `close=entry=0x1.999999999999ap+1`. Valores exibidos: US$ 0,32 e 3,20.
- **CAPR 28/05/2019:** econômico `close=0x1.353f7ced91688p−2`, `entry=0x1.353f7ced91687p−2`; legado `close=entry=0x1.828f5c28f5c29p+1`. Valores exibidos: US$ 0,302 e 3,02.

Nos três casos, o cenário econômico reportou `close > entry = True`, enquanto o legado classificou **empate** (`False`), e a distância foi **um único ULP**. **Zero diferenças em `diff(close)>0`** nas 59 transições de cada decisão.

### Identidade exata das diferenças de utilidade

O cálculo original contém `trend_persistence = 0.5 * mean(close_future > entry) + 0.5 * mean(diff(close_future) > 0)`. Em cada decisão, a diferença de uma comparação positiva nos 60 fechamentos produz **`0.5/60 = 0.0083333333333`** na persistência, exatamente como medido. Seu peso na utilidade final é **0,2**, produzindo **`0.0016666666667`** de diferença em `forward_risk_adjusted_utility`. Logo, **as seis diferenças finais são plenamente explicadas pelas três classificações estritas numéricas**.

**Limite epistemológico crucial:** os hashes/representações comprovam que a discrepância surge em números próximos por 1 ULP após transformações/cálculos em `float64`. O pacote **não contém o texto decimal original das células RAW**, portanto ainda não distingue se a diferença de 1 ULP já existia no CSV de origem ou se foi introduzida só ao aplicar fator de quantidade/método de reexpressão. Não afirmar origem exata do erro sem leitura literal das fontes.

## 3. Recomendação científica

**Não alterar `>` por `>=`, tolerância relativa ou arredondamento arbitrário** no motor com base em apenas três eventos; isso pode alterar rótulos fora do ensaio e romper a reprodução do checkpoint. Preservar todos os arquivos RAW. Em vez disso, auditar de forma independente o texto **literal** das células do CSV (preço RAW da abertura e fechamento) com `Decimal`/frações racionais exatas dos splits, para saber se os três pares são economicamente iguais na precisão efetivamente registrada ou se contêm uma diferença decimal explícita.

A execução v1.24.7 não recalculou previsões, ranking, rotatividade ou capital. Os resultados históricos **US$ 587.165.941,13 e US$ 89.941.496,49** continuam ligados a snapshots diferentes por 73 barras RAW revisadas de maio/2018, problema independente deste estudo de empate.

## 4. Próxima verificação

Campanha v1.24.8: ler os **CSV RAW sem conversão numérica no momento da coleta textual** e comparar as três entradas e saídas com aritmética decimal exata aplicada **somente no diagnóstico**; registrar valor textual, eventos realmente atravessados, comparação exata e divergência com float64. Gerar hashes dos CSVs antes/depois, sem alterar o modelo ou os preços. Esse teste separa *precisão do registro original* de *arredondamento produzido pela aritmética*, antes de decidir se é justificável propor qualquer correção de lógica.
