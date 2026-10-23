# Auditoria dos decimais armazenados nos dados RAW — resultado v1.24.8

**Data da análise:** 23/10/2026. **Branch:** `research/v1.23.1-fresh-snapshot-20261022`. **Código executado:** `auditar_precisao_decimal_splits_spyder.py`, commits `e19e0f8a5e48edc73a40215fe31149e61edf4e79`, `4215b2122c5f8f8dfd14c404227f83a7ecfe6aee` e `a15a9ad82793cbe6bc1b53b804afa79c5df5267c`. **`main` de referência, não alterada:** `4667d2ccee3f3441ee4d2b362ce734311cedd5e3`. **PR #18:** Draft.

**Pacote recebido:** `pacote_decimais_raw_splits.zip`, **SHA-256 `df22e249611f456fb4c742b6c350f491ca9dad5e7e503f61303fec9d76839253`**; `ZipFile.testzip() = None`. Contém `resultado.json` (resumo e três decisões). **Snapshot-base de 75 ativos:** SHA `595b7a03dc2c8978487975cd3554b1ea65608741893f13208938dc624cb273cc`. Contrato de 75 símbolos conferido, mas cotação literal lida de **apenas CAPR e YANG**. O script não alterou os arquivos ou treinou modelos; nenhum backtest financeiro.

## 1. Resultado efetivo dos valores literais no snapshot

Em **3/3 decisões**, a comparação racional exata entre os **textos decimais gravados** dos CSVs e os multiplicadores de quantidade dos splits resultou em **diferença estritamente positiva**, sem empate literal, exatamente como informa o JSON (`exact_decimal_equal=0`, `exact_decimal_gt=3`).

| Ativo | Decisão | Literal `open` da entrada | Literal `close` futuro | Split realmente atravessado | Diferença econômica racional exata |
|---|---|---|---|---|---|
| CAPR | 20/05/2019 | `0.32000000000000001` | `3.2000000000000002` | reverso 10:1, 05/06/2019 (quantidade ×1/10) | **+1×10⁻¹⁷** |
| CAPR | 28/05/2019 | `0.30199999999999999` | `3.02` | reverso 10:1, 05/06/2019 (quantidade ×1/10) | **+1×10⁻¹⁷** |
| YANG | 16/10/2024 | `3.9199999999999999` | `78.400000000000006` | reverso 20:1, 04/11/2024 (quantidade ×1/20) | **+4×10⁻¹⁶** |

Os três pares são **empates em representações decimais curtas que recuperam o mesmo `float64`**: CAPR `0.32` versus `3.2/10`; CAPR `0.302` versus `3.02/10`; YANG `3.92` versus `78.4/20`. Verificações locais: `repr(float('0.32000000000000001')) == '0.32'`, `repr(float('3.2000000000000002')) == '3.2'`; idem para os outros quatro literais. **Para todos os seis números, `format(float(literal), '.17g') == literal`**, coincidindo com a instrução real de exportação em `reproducao/dados.py`: `frame.to_csv(target, index=False, float_format="%.17g")`.

**O que isto demonstra:** as pequenas diferenças decimais literais estão presentes no snapshot, e a forma das seis strings é consistente exatamente com a **serialização de `float64` em 17 algarismos significativos**. Na análise anterior v1.24.7, a reexpressão legada produziu empate `float64` em cada par, enquanto o caminho RAW × quantidade tinha fechamento 1 ULP acima da abertura, alterando o booleano `close > entry`. Isso explica integralmente as seis diferenças de labels da v1.24.6: três mudanças na persistência de **0,5/60 = 0,008333333…**, três na utilidade agregada de **0,001666667**.

**Limite de proveniência:** o CSV foi gravado pelo programa após valores passarem pela representação numérica da biblioteca/API, e **não é uma captura byte a byte da resposta JSON original da Alpaca**. Portanto não é cientificamente válido afirmar com este resultado que a Alpaca enviou exatamente `0.32000000000000001`, ou que `0.32` foi necessariamente o decimal de origem. A igualdade decimal obtida usando `repr(float)` é uma **interpretação canônica possível** dos mesmos floats, **não** uma prova da exatidão do preço fornecido pela corretora. O teste comparou três pares; não generalizar a ausência de problemas aos 75 ativos.

## 2. Hashes dos dois arquivos consultados

| Arquivo | SHA-256 |
|---|---|
| `CAPR.csv` OHLCV RAW | `19f8507f5ba29aa010cd8c4bfc3fc505fd7c9cef4115f6d5e326615c3db20657` |
| `CAPR.csv` Corporate Actions | `37ffee05ca73077b2c902f670b824ed9bd45e7286e0d4d79a48bdab39260a534` |
| `YANG.csv` OHLCV RAW | `bf69506c20af10fb1c1731081a976c6737444cea86e5519c8d678b0ac1ed2747` |
| `YANG.csv` Corporate Actions | `42e24cce1dc3bbf538f6c0ef2793110815373172bdfe8719207dfcfa32b814af` |

## 3. Decisão técnica

**Não mudar os CSVs RAW, nem `>` para `>=`, nem introduzir arredondamento, epsilon ou ajuste ad hoc** para forçar paridade com a normalização legada. Os registros preservam exatamente o texto da campanha; uma mudança de serialização geraria novos hashes e uma nova versão de dados, o que quebraria a comparabilidade com a reprodução oficial.

A conclusão da sequência v1.24.4–1.24.8 permanece delimitada: **11.980/11.980 retornos simples de preço** equivalentes; **33.144/33.150 comparações de componentes de utilidade** concordantes; as outras seis diferenças reduzidas a **três decisões** e a desvios de **1 ULP** em comparações estritas. O fenômeno decorre da representação/aritmética numérica no fluxo do pipeline, e **não há evidência de divergência da razão econômica do split** nesses três casos. Não se reproduziram scores, rankings, negociação, custos ou capital sobre RAW imutável.

**Próxima verificação pontual v1.24.9:** adicionar um diagnóstico de serialização **somente leitura** que para as seis células: compare texto CSV versus `format(float(text), '.17g')`, representação curta `repr(float(text))`, bits `float.hex()` e sinais em aritmética racional; registrar hashes. Isso estabelece tecnicamente a contribuição da serialização da biblioteca. **Não é nova busca por capital nem motivo para mudar dados históricos.** Se desejarmos comprovar os bytes originais retornados da API, será necessário um novo experimento separado com captura explícita do payload original, jamais reescrever silenciosamente o snapshot antigo.
