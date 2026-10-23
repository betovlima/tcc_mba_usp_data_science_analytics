# Auditoria temporal de splits e integridade das decisões — v1.24.1

**Data:** 23/10/2026. **Branch de pesquisa:** `research/v1.23.1-fresh-snapshot-20261022`. **PR:** #18 (Draft). **`main` de referência, não alterada:** `4667d2ccee3f3441ee4d2b362ce734311cedd5e3`.

**Versão:** v1.24.1 — teste isolado sobre a política RAW estrita v1.24.0, **sem alterar o código do backtest oficial v1.23.0** nem parâmetros, universo de 75 ativos, política Control, snapshot RAW ou a plataforma operacional.

**Commits da implementação e testes:**
- `41dde42d74e408a107805356103d6fcbbc5df796`: `reproducao/splits_temporais.py`, políticas distintas para visibilidade de anúncio, efetivação e contabilidade sintética.
- `2d15b011eb33f288a6bf5627e524dcbdf7583355`: `auditar_temporalidade_splits_spyder.py`, auditoria offline dos 75 ativos e ensaio por injeção de split futuro sem modificar os arquivos.
- `d411219b0253d9e5e2b07a9a1b731a38acbcbef1`: `tests/test_splits_temporais_v1241.py`, testes sintéticos de informação, atributos e custódia.
- `a2c7ca93c20c1d5d04328e6dbbb319c1be85150e`: regressão com LightGBM real (ranking de dois ativos sintéticos, mantendo o mesmo conjunto de barras e colocando um evento futuro apenas nos metadados).
- Leitura RAW estrita anterior: `4658d5e6c40de7e7bc91c830b53453521fb9f7c1`, `ca51c18182085825ff71567af802c89beb8dae0d`; guarda de incompatibilidade entre modos: `e89b2a8740073129a79b9067aaae471b072df850`.

## Hipótese falsificável

> Se acrescentarmos um split **não conhecido** e cuja data efetiva é posterior ao instante da decisão, os dados RAW, os atributos históricos, os scores e o ranking calculados somente com informações anteriores à decisão **não poderão mudar**.

O teste também separa o caso de um split **já anunciado** mas ainda não efetivado: seu metadado pode ser considerado conhecido, mas a quantidade da posição não muda antes de sua ex-date. Não inferir anúncio a partir do campo `process_date` do arquivo retrospectivo: **não é registro comprovado de primeira disponibilidade**.

## Ensaios automáticos na CI

1. `first_seen_at_utc` posterior ao instante de decisão: evento ainda não conhecido; evento com `first_seen_at_utc` anterior: anúncio conhecido, **ainda não efetivado**.
2. Registro sem `first_seen_at_utc`: visibilidade histórica **desconhecida**; não preencher de forma fictícia com `process_date` ou `ex_date`.
3. Split 1:10 com uma ação a US$ 100: após ex-date, custo unitário de US$ 10 e quantidade 10; valor e base de custo US$ 100 preservados, sem editar as cotações. Proibir aplicar o mesmo evento duas vezes; incluir teste inverso 10:1.
4. A rotina histórica `split_normalize` **de fato modifica preços RAW anteriores** quando recebe um split futuro; registrar isso como característica **legada**, não usar como entrada do modo RAW.
5. Dois arquivos de Corporate Actions, um sem eventos e outro com split futuro não anunciado, são processados em **modo RAW estrito**: todos os preços/volumes e atributos retornados pelo **motor real de engenharia de atributos** devem ser exatamente idênticos.
6. Miniatura de LightGBM sobre duas séries sintéticas: previsões de utilidade e ativo com maior score precisam coincidir ao acrescentar o evento futuro não conhecido, com mesmas barras e mesmo corte de treinamento.

**Limitações dos testes sintéticos:** igualdade de atributos não prova, isoladamente, equivalência do replay Control com 75 ativos, folds, rotatividade, previsões OOS e contabilidade completa de splits. O mini-LightGBM não utiliza o dataset financeiro original e não substitui reexecução real.

## Auditoria local sobre os 75 ativos reais

Executar no Spyder `auditar_temporalidade_splits_spyder.py` (F5) após atualizar **a mesma branch**. Ele requer snapshot local já validado sob `dados/.u75_fresh_runs/asof_2026-10-22_*/universe_75` e **não baixa nem modifica dados**.

Com o snapshot, o executor:
- verifica que existem exatamente os 75 ativos fixos;
- lê os OHLCV com parsing numérico estrito e registra fingerprint dos prefixos históricos até 28/07/2020;
- inventaria splits nos 75 arquivos de Corporate Actions: `ex_date`, proporções e presença/ausência de evidência de `first_seen_at_utc`;
- injeta **somente em memória** um split sintético com ex-date 12/08/2020 e `first_seen_at_utc` 01/08/2020, posterior à decisão histórica de 28/07/2020;
- compara os **atributos reais da engenharia do TCC** antes do evento, registra que RAW permanece intacto e mostra que a antiga normalização retrospectiva alteraria o passado;
- gera `output/auditoria_temporal_splits/<ID>/pacote_auditoria_temporal_splits.zip` com `resultado_temporal_splits.json`, `eventos_splits_e_disponibilidade.csv` e `integridade_prefixos_75.csv`.

**Ainda não há pacote local dessa auditoria de 75 ativos nesta revisão.** Não inventar contagem de anúncios historicamente disponíveis nem afirmar que a API oferecia cada evento antes de um dado pregão: o snapshot foi coletado anos depois e não apresenta comprovante suficiente de `first_seen_at_utc` para reconstituir o conhecimento de 2020.

## Próximo bloqueio e critério de aceitação

**Antes de promover qualquer backtest sobre RAW imutável**, implementar e testar contabilidade realista de posições nos splits, incluindo quantidades fracionárias/cash-in-lieu quando aplicável, além de rótulos de treinamento e cálculo de retorno que atravessem desdobramentos sem retornos artificiais. Não normalizar ou sobrescrever OHLCV para isso. O módulo de custódia desta revisão é **aritmética sintética**, **não** foi integrado à simulação de negociação.

Depois disso, verificar com um replay completo e sem seleção de parâmetros se a inclusão de um evento futuro desconhecido não muda as decisões anteriores. Se falhar, documentar e corrigir a dependência temporal; se passar, não afirmar que isso resolve outros possíveis vazamentos ou que valida retrospectivamente os US$ 587 milhões.

**Resultado histórico preservado:** os US$ 587.165.941,13 pertencem ao TCC v1.23.0 condicionado aos seus CSVs e à normalização legada. A v1.24.1 não substitui esse checkpoint nem os resultados do contrafactual de 73 barras de maio de 2018.
