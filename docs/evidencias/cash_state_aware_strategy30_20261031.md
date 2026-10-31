# Evidência congelada: CASH com estados e reentrada versus Control Strategy #30

**Situação:** execução concluída; comparação financeira inferior ao Control.
**Identificador de execução:** `20261031T001851972255Z`.
**Versão do experimento:** `cash-state-aware-strategy30-v1.0.0`.
**Tag de congelamento:** `research-cash-state-aware-strategy30-v1.0.0-20261031`.
**Branch de origem:** `research/control30-cash-state-aware-v1`.
**Commit testado do código:** `6ca625b338895c009aee53212f53e0dbf5526167`.
**Commit do motor científico de referência:** `f78b718ae4a459e34d91cc78a486dd6bbba197b5`.
**Commit anterior de paridade validada:** `346e8f1dfd7f27a369a14eb96afeca332819076c`.

> A tag aponta para o **commit documental que contém este arquivo** e as
> configurações de pesquisa. Seu hash deve ser recuperado com
> `git rev-list -n 1 research-cash-state-aware-strategy30-v1.0.0-20261031`.
> O hash do commit de **código testado** acima não será alterado pela
> documentação. Não tentar escrever o hash do próprio commit dentro do
> arquivo versionado: isso criaria uma referência circular.

## 1. Artefato e cadeia de integridade

- Pacote original: `pacote_optimal_switching_cash_20261031T001851972255Z.zip`.
- **SHA-256 do pacote ZIP:** `584f379fbe9c39f0a088a1508e495d1b5fb6b8a07ce6addc45b5a3bb511d5c0b`.
- Arquivos de evidência no ZIP: **14**, todos com SHA-256 individual
  validado contra `manifesto_pacote.json`; `ZipFile.testzip() == None`.
- Status do manifesto: `completed`; data declarada no manifesto:
  `2026-10-31T00:26:36.531432+00:00`.
- SHA-256 de `resultado.json`:
  `3bad8bd6e64431632dfddd8e22539364fa76f1bc89e220137503f3b2742d123c`.
- SHA-256 de `control_parity.json`:
  `98e42a5ddf101669827bf41bab9b7c2931e9f9602f34686edf75a6b61e88341c`.
- SHA-256 do ZIP do Control operacional originalmente importado:
  `8f41720fc7af78ec1a07bff3d72d637a081326601b92f1e2bce9f117ee3ccb7a`.
- Snapshot de CSVs RAW/SIP e corporate actions:
  `dados/strategy30_snapshot/`; ID SHA-256 do manifesto:
  `6d792f4119ca6c6a78b7e61601b1b852bb1be8c9d7dd2a5fd4cdfa3e436ac06a`.
- Pacote de análise não contém dados RAW e não usa banco.

**O ZIP não está embutido nesta tag.** Guarde o arquivo binário original
separadamente sem modificá-lo; a tag fixa o código, esta descrição e os
hashes necessários para verificar a identidade da evidência. O arquivo
`manifesto_pacote.json` dentro do ZIP contém os hashes individuais.

## 2. Protocolo de avaliação

Universo de 75 ativos, período fora da amostra de **22/07/2020 a
28/10/2026**, total de **1.576 pregões**, em três folds de 504, 504 e
568 sessões. Capital inicial de US$ 10.000 para ambas as políticas,
em dólar nominal simulado. Controle e alternativa utilizam o mesmo
calendário, dados e regras de taxas no simulador.

Control: Strategy #30 com LightGBM e rotação entre ativos, sem
CASH efetivo nesta execução.

Alternativa: regressão Ridge, `alpha=10`, horizonte de **10 sessões**,
treino/validação cronológicos de 75%/25% na janela de calibração,
rótulos com expurgo temporal. São simulados seis estados de calibração
pré-definidos: `control_shadow`, `cash_1_of_7`,
`cash_2_of_11`, `cash_4_of_15`, `cash_8_of_20` e
`cash_13_of_31`.

As margens efetivas de rotação selecionadas na calibração foram
fold 1: **0,0100**, fold 2: **0,0005** e fold 3: **0,0100**.
Não houve imposição de margens pelo resultado do teste fora da amostra.

**Paridade do Control no ZIP:** `fully_reproduced=true`;
1.576 sessões de referência e 1.576 de reprodução; **zero** sessões
divergentes nas decisões, nos folds ou no patrimônio além do limite de
US$ 0,01. Patrimônio final da reprodução: US$ 80.685.454,78647688,
diferença final igual a zero.

## 3. Resultado financeiro consolidado

| Indicador | Control Strategy #30 | CASH com estados/reentrada |
|---|---:|---:|
| Capital inicial | US$ 10.000,00 | US$ 10.000,00 |
| Capital final | **US$ 80.685.454,79** | **US$ 16.733.706,35** |
| Ganho líquido simulado sobre o capital inicial | US$ 80.675.454,79 | US$ 16.723.706,35 |
| Retorno acumulado sobre US$ 10 mil | +806.754,55% | +167.237,06% |
| Multiplicação do capital inicial | 8.068,55 vezes | 1.673,37 vezes |
| Sharpe | **2,317719** | **2,195560** |
| Drawdown máximo | −41,682843% | −41,682875% |
| Rotações entre ativos | 348 | 236 |
| Sessões em CASH | 0 | 348 |
| Taxas de execução | US$ 66.481,95 | US$ 15.783,77 |

O CASH terminou com **20,7394%** do capital do Control, uma diferença
**desfavorável de US$ 63.951.748,44**, ou **−79,2606%** em patrimônio
final relativo. O Sharpe caiu **0,122159 ponto**. A diferença de drawdown
máximo foi de cerca de **−0,000032 ponto percentual**, materialmente nula
na precisão de leitura usual.

As taxas da alternativa foram US$ 50.698,18 menores (redução de
76,26%), mas isso **não compensou** o custo de oportunidade das
decisões de ficar fora dos ativos.

Esses ganhos são retornos **hipotéticos de backtest**, não lucro real,
não validação prospectiva e não garantia de desempenho futuro.

## 4. Resultado por fold

Os retornos percentuais abaixo são calculados sobre o patrimônio no
**início de cada fold**, não sobre os US$ 10.000 de forma independente.

| Fold (período OOS) | Sessões | Control final | CASH final | Retorno Control no fold | Retorno CASH no fold |
|---|---:|---:|---:|---:|---:|
| 1 (2020-07-22 a 2022-07-21) | 504 | US$ 51.308,86 | US$ 32.685,75 | +413,09% | +226,86% |
| 2 (2022-07-22 a 2024-07-24) | 504 | US$ 4.160.778,02 | US$ 638.500,26 | +8.009,28% | +1.853,45% |
| 3 (2024-07-25 a 2026-10-28) | 568 | US$ 80.685.454,79 | US$ 16.733.706,35 | +1.839,19% | +2.520,78% |

O CASH teve crescimento **proporcional maior apenas no fold 3**, mas
iniciou esse período com patrimônio muito inferior, devido às perdas
relativas acumuladas nos dois folds anteriores. Escolher só o fold 3
posteriormente seria seleção pela informação do teste.

## 5. CASH efetivo e qualidade de previsão

- **348** sessões efetivas em CASH, em **115** períodos contíguos.
- **3,0261 pregões por período**, em média (348/115).
- Período máximo em CASH: **18 pregões**.
- **115** saídas de ativo para CASH e **115** reentradas em ativo.
- Reentradas na mesma sessão: **zero**. Uma decisão por pregão.
- Amostras de calibração: 463, 406 e 379 por fold, **1.248 no total**.
- Amostras com estado atual CASH: **101 por fold, 303 no total**.
- Validação interna: 90, 92 e 95 previsões, **277 no total**,
  mas apenas **22 datas distintas por fold** (66 pares de datas/fold).
- Acerto direcional do Ridge por fold: **53,33%**, **47,83%** e
  **38,95%**.
- Referência trivial de sempre preferir Control: **63,33%**,
  **75,00%** e **38,95%**, respectivamente.
- Acerto agregado ponderado por previsão: **46,57%** no Ridge e
  **58,84%** na referência trivial. As observações correlacionadas
  não devem ser tratadas como datas independentes.

A política demonstrou funcionamento mecânico de saída e reentrada,
**mas não superioridade preditiva consistente** em validação nem
superioridade financeira na avaliação fora da amostra.

## 6. Conclusão e limites científicos

**Decisão:** arquivar a experiência como evidência **negativa de
melhoria** em relação ao Control, mantendo Strategy #30 como baseline
da pesquisa. Não promover esta política para uso operacional.

Limites declarados no próprio pacote: etiquetas contrafactuais de uma
saída CASH seguida do Control diferem da política OOS que pode escolher
CASH em sessões consecutivas; custos do rótulo histórico são
aproximados, embora o replay financeiro use taxas de referência; os
estados de múltiplas trajetórias no mesmo dia são correlacionados;
o período OOS já foi examinado por pesquisas anteriores e não é um
teste independente inédito. A execução reproduz um resultado histórico,
mas não comprova ausência de sobreajuste.

## 7. Como verificar localmente

```bash
git fetch origin
git switch research/control30-cash-state-aware-v1
git pull --ff-only origin research/control30-cash-state-aware-v1
git show research-cash-state-aware-strategy30-v1.0.0-20261031
git rev-list -n 1 research-cash-state-aware-strategy30-v1.0.0-20261031
sha256sum pacote_optimal_switching_cash_20261031T001851972255Z.zip
```

O hash do ZIP deve coincidir com o SHA-256 documentado acima. O comando
`sha256sum` requer Bash/Git Bash, ou ferramenta equivalente no sistema.
