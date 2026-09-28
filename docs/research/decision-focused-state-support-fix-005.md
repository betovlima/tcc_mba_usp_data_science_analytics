# FIX-005 — Cobertura de estados no aprendizado Decision-Focused

**Escopo:** auditoria descritiva, sem retreinar LightGBM ou trocar política.
**Versão de trabalho:** `v1.3.0-dev.3`.
**Branch:** `research/decision-focused-oof-v2`, única.
**Fonte dos resultados preliminares:** `decision_focused(5).zip`,
SHA-256 `2bad4be01b6dc8b0f22a051d3cbe2e34d28aafab4320aed74f67d37eb9203d1a`.
**Não é novo teste confirmatório:** os folds de 2020–2026 já foram examinados.
**Testes do código:** 66 testes aprovados e Ruff aprovado no workflow
\`reproduction-tests\` run [36416400586](https://github.com/betovlima/tcc_mba_usp_data_science_analytics/actions/runs/36416400586), commit \`2f0fee3\`.
**Execução econômica com os novos CSVs:** [run 36416400770](https://github.com/betovlima/tcc_mba_usp_data_science_analytics/actions/runs/36416400770); confirmar conclusão e outputs antes de declarar identidade de capital.

## Problema observado

O alvo OOF é sequencial em 20 sessões, mas a distribuição de estados
de treinamento pode não cobrir os estados de carteira efetivamente
visitados no teste: o modelo recebe features relativas ao incumbente,
rank e holding. Em particular, o universo de estados após uma intervenção
pode diferir do conjunto visto antes do fold. A investigação aqui é
**cobertura**, não causalidade da perda nem qualidade do preditor.

As 7.717 linhas de rótulos correspondem a **1.886 grupos**
`(fold, date, state_source, incumbent, holding)` em **870 datas**.
Existem várias ações candidatas por estado/data e labels com horizonte
de 20 pregões sobrepostos; não considerar as linhas independentes.
Um candidato idêntico à ação-base tem vantagem exatamente zero.

## Comparação cronológica: treino elegível vs Control OOS

Foram considerados APENAS os rótulos com
`decision_date < validation_start` E
`outcome_end < validation_start`. Os registros do conjunto de validação
interna foram excluídos antes de calcular estados vistos. Isso reproduz
a fronteira temporal de elegibilidade, mas não substitui a própria
auditoria de treino do modelo.

| Fold | Linhas elegíveis | Datas elegíveis | Pares distintos (ativo, holding) | Sessões Control OOS | Incumbente não visto em treino | Par exato não visto em treino |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 483 | 94 | 80 | 504 | 343 (68,06%) | 465 (92,26%) |
| 2 | 2.038 | 381 | 243 | 504 | 96 (19,05%) | 265 (52,58%) |
| 3 | 3.598 | 647 | 289 | 539 | 259 (48,05%) | 385 (71,43%) |
| Total | 6.119 | 1.122 datas-fold* | — | 1.547 | 698 (45,12%) | 1.115 (72,07%) |

*Somatório por fold, não união de datas em calendários distintos.
Os 1.598 rótulos restantes da exportação integral são reservados à
validação interna ou não elegíveis pelo corte de desfecho.

**Interpretação adequada:** ausência de par exato `(ativo,holding)` não
prova incapacidade de generalização. O LightGBM aprende atributos
contínuos/relativos de candidatos, e pode funcionar em posições ou
durações não observadas literalmente. Mas demonstra que avaliar apenas
número de linhas do dataset esconde cobertura bastante desigual.

No fold 3, o ativo CORT aparece como incumbente em 70 datas da fonte
`OOF_CONTROL` e em 4 datas da fonte `OOF_LEARNED_STATE`,
enquanto AVGO não aparece como incumbente em nenhum rótulo deste fold.
Nos OOS do fold 3, o Control visitou AVGO como incumbente em 51
sessões (52 na regressão). Essa associação NÃO permite concluir que
a falta de exemplos AVGO causou a decisão de 14/01/2026; a regressão
mantinha CORT nessa decisão e o modelo dispõe de atributos relativos.

## FIX-005 — instrumento reprodutível

Novo módulo INTERNO `reproducao/diagnostico_estado.py`, chamado
exclusivamente pelo ponto de entrada existente
`pesquisar_decision_focused_spyder.py`. Incluído registro observacional
`research_holding_days_at_decision` na política experimental para
que o diagnóstico da regressão utilize o próprio estado dela,
**sem copiar holding do Control nem usar o resultado do teste para treino**.

Exports em `output/decision_focused/active/`:

- `state_support_folds.csv`: cobertura de incumbente e do par exato
  ativo/holding por fold e por política;
- `state_support_assets.csv`: estados não vistos por ativo;
- `state_support_oos_sessions.csv`: data, política, ativo, holding
  e indicadores de suporte por sessão;
- `state_support_checks.json`: invariantes da auditoria.

Todas as contagens usam rótulos anteriores à validação como referência.
O OOS é apenas OBSERVADO; nenhuma ação, parâmetro, limite,
treino ou escolha de modelo é alterado. O "estado" usado na tabela é
**parcial**: `(incumbente, holding)`; não equivale a capital, quantidade,
entry price, margens, histórico de exposição e posição completa.
A nova coluna de holding é coletada no momento da decisão, sem antecipar
a próxima barra. O código falha se houver falta de campos, duplicidade
de datas, erro de alinhamento de calendários ou labels-base não nulos.

## Próxima hipótese a avaliar, não implementada aqui

Comparar desempenho de um futuro modelo de Q condicionado ao estado e
sua cobertura OOF contra Control usando pré-registro e janela temporal
nova, posterior a 17/09/2026, congelada separadamente.
A cobertura parcial identificada agora não é motivo para fazer
tuning corretivo nos mesmos três folds. As mudanças desta entrega
são de instrumentação e testes; a pesquisa econômica continua
com Control US$ 10.082.425,91 e regressão US$ 7.639.150,63
até validação efetiva de um novo replay completo.
