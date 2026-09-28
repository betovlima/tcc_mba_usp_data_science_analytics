# CONTEXTO MESTRE — Pesquisa TCC Decision-Focused

<!-- [TCC-DFL:CONTEXT-START] -->
<!-- [TCC-DFL:BRANCH] research/decision-focused-oof-v2 -->
<!-- [TCC-DFL:WORKING-VERSION] v1.3.0-dev.3; não incrementar por correção pequena -->
<!-- [TCC-DFL:CHECKPOINT-BASE] 682b0cc4d0daa8b822f05f0eb4e65a98b3ade184 -->
<!-- [TCC-DFL:RESUME] git rev-parse HEAD na branch ativa é a fonte do commit mais recente. -->
<!-- [TCC-DFL:CONTEXT-END] -->

**Leia este arquivo primeiro em qualquer novo chat antes de propor mudanças no código.**
O histórico continua neste mesmo arquivo, atualizado por commits na mesma branch.

## Regra de continuidade solicitada pelo pesquisador

- Trabalhar **somente** na branch `research/decision-focused-oof-v2` até que a pesquisa esteja estável. Não criar branch para cada correção, ajuste, execução ou retomada de chat.
- Não criar nova tag ou incrementar a versão experimental a cada commit. A versão de trabalho permanece `v1.3.0-dev.3` enquanto estivermos estabilizando este experimento. Quando aprovado, discutir tag de fechamento e integração na `main`.
- Cada alteração recebe commit claro e um registro curto abaixo, identificado por marcador único `[TCC-DFL:FIX-NNN]`, contendo **causa, arquivos, impacto, testes, status e próximo passo**.
- Preservar resultados históricos e suas fontes. Nunca substituir uma execução negativa por outra, nem mudar dados congelados para conseguir resultado positivo.
- Na retomada, consultar o HEAD remoto e o log mais recente, porque o SHA anotado em um documento que está no próprio commit não pode conter de forma confiável seu próprio SHA. Nunca inferir que uma execução ainda está rodando apenas por este texto.
- Não alterar a `main`, o Control/Soft oficial, o snapshot congelado ou a tag original durante a pesquisa.
- O script independente está organizado por células `# %%` para Spyder. Exibir comandos do console do Spyder e comandos Git em cada entrega concreta.

## Identificação técnica

- Repositório: https://github.com/betovlima/tcc_mba_usp_data_science_analytics
- Branch única em desenvolvimento: `research/decision-focused-oof-v2`.
- Commit inicial desta fase: `682b0cc4d0daa8b822f05f0eb4e65a98b3ade184`.
- Origem da pesquisa anterior: `research/decision-focused-counterfactual-v1`, commit `7847bdf66501b5f341931caeaeb9169e8a414519`.
- Base oficial preservada: `main` em `f9cf29fdb736676d0d3e26780481be99813c602a`; tag oficial anterior `v1.2.0`.
- Versão experimental atual: `v1.3.0-dev.3`; esta numeração é identificador do protocolo e não muda por correções de implementação.
- **Único script principal da pesquisa: `pesquisar_decision_focused_spyder.py`.** Não criar executáveis `_v2`, `_v3` etc. O módulo `reproducao/decision_focused_v2.py` é somente implementação interna importada pelo script principal, não um segundo ponto de entrada.
- Núcleo: `reproducao/decision_focused_v2.py`, `reproducao/dfl_softmax.py`; mecanismo de rótulos em `reproducao/decision_focused.py`.
- Testes: `tests/test_decision_focused.py`, `tests/test_decision_focused_v2.py`.
- Documentação detalhada: `docs/changes/v1.3.0-dev.3-oof-decision-focused.md`.
- Output ativo, não versionado: `output/decision_focused/active/` (histórico anterior permanece preservado e separado).
- **Único workflow da pesquisa ativa:** `.github/workflows/research-decision-focused.yml`.

## Resultado histórico preservado

A v1 testou Control, Soft, regressão contrafactual e ranking pareado ponderado por regret. Foi um **resultado negativo**, e não um erro a ser ocultado. Registro versionado:
`docs/results/v1.3.0-dev.2-negative-result.json`. A saída original teve a identificação inconsistente `v1.3.0-dev.1`; o protocolo completado depois da correção do cache é identificado como `v1.3.0-dev.2`.

Referências observadas no ZIP fornecido pelo pesquisador: Control US$ 10.082.425,91; Soft US$ 9.840.028,41; regressão US$ 867.787,52; ranking US$ 315.934,23. Apenas 265 datas distintas produziram 1.074 rótulos. **Não usar o OOS desta execução para escolher hiperparâmetros ou limiares retroativamente.**

## Protocolo de pesquisa da versão corrente

1. Gerar rótulos contrafactuais em blocos cronológicos OOF dos modelos-base internos, anteriores ao teste do fold externo; purge de 60 sessões e horizonte de rótulo de 20 sessões.
2. Separar uma janela temporal posterior de calibração, purgando qualquer rótulo de treinamento cujo desfecho a alcance.
3. Treinar modelo piloto apenas com prefixo histórico; gerar exemplos adicionais em estados visitados pelo próprio piloto, sem usar a calibração nem o teste final.
4. Treinar três variantes no corpus histórico: regressão de vantagem, ranking pareado sensível a regret e softmax diferenciável de regret por decisão. A última **não** diferencia o backtest sequencial de ponta a ponta.
5. Calibrar a intervenção exclusivamente na janela interna reservada; a política recua para Control quando a evidência não supera a guarda calibrada.
6. Executar Control/Soft inalterados e as três variantes no mesmo backtest sequencial OOS. Reportar capital, CAGR, Sharpe, drawdown, pior fold, número de intervenções, decisões por ativo e exposição CASH.
7. Usar somente `dados/pesquisa` congelados com SHA-256; se LF/CRLF diferir, aceitar cópia temporária somente quando os bytes reconstruídos igualarem os hashes originais. Sem download, Alpaca, MCT ou MongoDB.

Limitações reconhecidas: rótulos sobrepostos, possível drift de estados, capacidade distinta dos modelos internos e do Control final, viés do universo retrospectivo, restrições de execução, e inexistência de garantia causal ou de rentabilidade futura.

## Status dos testes e da execução

<!-- [TCC-DFL:CHECKPOINT-TESTS] -->
Em 27/09/2026, o commit `682b0cc4d0daa8b822f05f0eb4e65a98b3ade184` passou no `reproduction-tests`: **56 testes aprovados**, Ruff aprovado. O workflow `36336715959` **concluiu com sucesso**, incluindo verificação SHA-256, execução dos backtests e upload do artifact. Resultados auditados e identidade do ZIP constam em `docs/results/v1.3.0-dev.3-oof-run-36336715959.md`. Nenhuma das três variantes experimentais supera Control.
Workflow: https://github.com/betovlima/tcc_mba_usp_data_science_analytics/actions/runs/36336715959

<!-- [TCC-DFL:FIX-001] -->
**Fix 001 — protocolo de continuidade (27/09/2026).**
Causa: criação de branches/versões a cada correção dificulta a retomada quando a conversa atinge o limite.
Arquivos: este `CONTEXTO_MESTRE.md`.
Impacto: documental; sem modificação de dados, parâmetros, modelos, critérios ou resultado.
Testes: não requerem execução nova para esta alteração de documentação. Último resultado técnico verificado consta acima.
Status: branch única estabelecida; os novos commits nesta branch devem atualizar este registro.
Próximo passo: verificar resultado do workflow completo; investigar quaisquer falhas nessa **mesma branch**, registrar novo marcador `[TCC-DFL:FIX-002]` e só então interpretar resultados.

## Procedimento de retomada em outro chat

Use esta instrução no primeiro pedido: **"Leia `CONTEXTO_MESTRE.md` na branch `research/decision-focused-oof-v2` do repositório `betovlima/tcc_mba_usp_data_science_analytics`. Confira o HEAD, os últimos commits e o estado do workflow de pesquisa. Continue na MESMA branch, sem criar outra ou alterar a main, e registre as mudanças com o próximo marcador FIX neste arquivo."**

Terminal Git, em um clone existente:

```bash
git fetch origin
git switch research/decision-focused-oof-v2
git pull --ff-only origin research/decision-focused-oof-v2
git rev-parse HEAD
git log -5 --oneline
```

Se ainda não existir a branch localmente: `git switch -c research/decision-focused-oof-v2 --track origin/research/decision-focused-oof-v2`.

Console Spyder, com diretório de trabalho na raiz do projeto:

```python
import subprocess
subprocess.run(["python", "-m", "pytest", "-q"], check=True)
runfile("pesquisar_decision_focused_spyder.py", wdir=".")
```

**Critério de fechamento:** testes e backtest completos auditados, documentação e artefatos coerentes, discussão dos resultados e aprovação da versão estável antes de tag/merge.


<!-- [TCC-DFL:FIX-002] -->
**Fix 002 — identidade de artefatos e checkpoint de resultado (27/09/2026).**
Causa: o arquivo recebido `decision_focused(1).zip` pertence à versão **v1.3.0-dev.2**, não ao novo protocolo OOF; a anotação anterior registrava o workflow como ainda em andamento. Não misturar métricas ou considerar ZIP v1 como teste v2.
Arquivos: `docs/results/v1.3.0-dev.3-oof-run-36336715959.md` e este `CONTEXTO_MESTRE.md`.
Impacto: documental e de rastreabilidade; nenhum ajuste no modelo, loss, snapshot, parâmetros ou horizonte. Pesquisa continua em `v1.3.0-dev.3` na MESMA branch.
Testes: run `36336715959` **success**; 56 testes; snapshot de 112 arquivos verificado; artifact GitHub ID `10937284707`, SHA-256 `8173ada4391acd35b50eeb661ced371b7f9af2768fdba869f5e246ab6f0ad8fa`.
Resultados (capital final): Control US$ 10.082.425,91; Soft US$ 9.840.028,41; regressão OOF US$ 7.639.150,63; ranking pareado US$ 2.105.511,08; DFL softmax US$ 4.716.464,39. Regressão fez somente 6 overrides locais, ranking 55 e softmax 30 de 1.547 decisões. Nenhuma superou Control. O histórico OOS já foi consultado após a v1; não tratá-lo como holdout totalmente intocado para decidir novos hiperparâmetros.
Status: resultado negativo da v2 registrado, v1 preservado separadamente. Sem merge nem tag.
Próximo passo: atribuir efeito das intervenções e estudar um teste confirmatório independente. Não fazer tuning retroativo nos mesmos folds. Próximo marcador: `[TCC-DFL:FIX-003]` quando houver nova mudança concreta.


<!-- [TCC-DFL:FIX-003] -->
**Fix 003 — retorno ao script original como único ponto de entrada (27/09/2026).**
Causa: a pesquisa atual foi implementada em `pesquisar_decision_focused_v2_spyder.py` sem evoluir `pesquisar_decision_focused_spyder.py`, gerando executáveis concorrentes e confusão na execução.
Arquivos: `pesquisar_decision_focused_spyder.py` (agora executa o protocolo OOF), exclusão de `pesquisar_decision_focused_v2_spyder.py`, unificação em `.github/workflows/research-decision-focused.yml`, exclusão do workflow duplicado, testes e documentação.
Impacto: um único script e um único workflow para a pesquisa; a saída ativa usa `output/decision_focused/active/` para não sobrescrever resultados históricos. O protocolo de modelagem, os dados congelados e as políticas Control/Soft não são alterados. Código histórico permanece acessível pelo Git e métricas anteriores em `docs/results/`.
Testes: Ruff aprovado e **57 testes aprovados** no run 36345280599 (commit 934c3e7), incluindo teste que exige script e workflow únicos. Nova execução completa pelo arquivo original: run 36345280608, ainda em andamento na última consulta; verifique seu estado antes de declarar resultado. Não confundir o run histórico 36336715959 (script sufixado) com o novo.
Status: mesma branch `research/decision-focused-oof-v2`, mesma versão de trabalho `v1.3.0-dev.3`; nenhuma tag, branch nova ou merge.
Próximo passo: executar somente `pesquisar_decision_focused_spyder.py`, verificar comparação e registrar mudanças concretas em `[TCC-DFL:FIX-004]`.


<!-- [TCC-DFL:FIX-004] -->
**Fix 004 — auditoria econômica das intervenções OOF (27/09/2026).**
Causa: o ZIP anterior não exportava Control/Soft completos nem permitia atribuir
a diferença entre Control e regressão aos três episódios com um replay de
carteira completa. As seis intervenções locais são 19/01/2023 (VNCE -> YANG),
08–11/10/2024 (YANG vs MYE) e 14/01/2026 (CORT vs AVGO).
Arquivos: `reproducao/auditoria_intervencoes.py` (NOVO MÓDULO INTERNO, não
um novo script de execução), `reproducao/decision_focused_v2.py`,
`pesquisar_decision_focused_spyder.py` (ÚNICO ponto de entrada),
`tests/test_auditoria_intervencoes.py` e workflow ÚNICO
`.github/workflows/research-decision-focused.yml`.
Impacto: apenas diagnóstico e exportação. Control, Soft, LightGBM, dados CSV
congelados, hiperparâmetros, critérios de decisão, loss e folds INTACTOS.
Exporta séries e trades diários de Control/Soft em
`output/decision_focused/active/{control,soft}_{predictions,trades}.csv`;
episódios em `intervention_episode_attribution.csv`, curvas completas
`intervention_bridge_curves.csv`, diferença diária
`intervention_episode_daily_pairs.csv`, operações de cada trajetória
`intervention_bridge_trades.csv`, verificações e reconciliação
`intervention_reconciliation.json`.
Método: replay exato BASELINE (nenhum override), depois EP001, EP002 e EP003
adicionados cronologicamente. Para cada episódio, os cenários possuem passado
idêntico, diferem apenas nas ações do episódio, e depois ambos seguem a
MESMA política-base. Os efeitos finais são MARGINAIS CONDICIONAIS à ordem,
não efeitos independentes/causais. A soma telescópica é confrontada com
a diferença de capital entre a regressão real e a política-base; o bridge final
deve reproduzir TODA a curva e os ativos da regressão original. O replay
base deve reproduzir TODA a curva e os ativos do Control oficial.
Testes: **62 aprovados**, Ruff aprovado no run `36349760069`, commit
`649b59a8287fdc080845832623740e18f6e3d841`. Testes sintéticos incluem
agrupamento por sessão, reconciliação, integridade da trajetória e fechamento
diante de inconsistências. Workflow completo (mesma base de código) run
`36349759992`: **SUCCESS**; arquivo enviado pelo pesquisador contém
reconciliação completa. Ver `docs/results/decision-focused-fix-004-episode-attribution.md`.
Status: código, testes e auditoria econômica completos. Mantida a mesma
branch e versão de trabalho `v1.3.0-dev.3`; nenhuma alteração no modelo.
Próximo passo: conferir `intervention_episode_attribution.csv` e
`intervention_reconciliation.json`, verificar efeito de 20 sessões,
persistência, fees, posições e reconciliação de capital antes de
qualquer hipótese de novo modelo. NÃO ajustar parâmetros no OOS visto.
Próximo marcador: `[TCC-DFL:FIX-005]` somente se houver correção concreta.


<!-- [TCC-DFL:RESULT-002] -->
**Resultado auditado do FIX-004 — ZIP decision_focused(3).zip (27/09/2026).**
Registro completo: `docs/results/decision-focused-fix-004-episode-attribution.md`.
SHA-256 do ZIP: `c471ed2387e6f55efa1f2bc03089d2cc1688a67c26bb66a0bdbe8f6274246955`.
O run `36349759992` do commit `649b59a` teve conclusão SUCCESS,
incluindo backtest, snapshot e artefatos. BASELINE e Control oficial têm
curvas e posições **idênticas**; bridge EP003 e regressão OOF também;
erro de reconciliação = **US$ 0,00**. O Control termina com
US$ 10.082.425,91 e a regressão com US$ 7.639.150,63; perda total
US$ 2.443.275,28. Efeitos marginais cumulativos: EP001 (19/01/2023)
−US$ 95.603,28 (3,91%); EP002 (08–11/10/2024) −US$ 96.989,26 (3,97%);
EP003 (14/01/2026) −US$ 2.250.682,75 (92,12%).
EP003: a regressão segura CORT em 15/01, passa a AVGO em 16/01;
os ativos coincidem em 16/01, MAS as carteiras não reconvergem.
Em 20–21/01 o Control contrafactual retorna a CORT, a regressão
permanece AVGO e perde exposição ao movimento que gerou P&L realizado
de +US$ 986.548,86 na venda de CORT de 22/01 do cenário-base.
`first_same_asset_after_start` significa somente primeira igualdade
nominal do ativo, não igualdade de capital/holding nem convergência
duradoura; não tirar conclusões causais ou independentes do OOS.
**Próximo passo:** discussão metodológica sobre estados de carteira
e janela confirmatória externa; não ajustar modelo/thresholds olhando
esse OOS. Nada foi alterado na política nesta análise documental.
Marcador futuro `[TCC-DFL:FIX-005]` reservado para correção concreta,
sempre na MESMA branch e script original.


<!-- [TCC-DFL:ANALYSIS-003] -->
**Análise entregue — dependência sequencial e validação confirmatória (27/09/2026).**
Origem: `decision_focused(4).zip`, SHA-256
`38f409435cbb7c337e527e15d56e9266db5b19f33c8aa113ed2b4c7da2a0202a`.
Documento integral:
`docs/research/decision-focused-sequential-state-analysis.md`.
Diagnóstico: EP001 teve 2 sessões futuras de ativos distintos + 916 de
mesmo ativo/capital diferente; EP002, 4 + 482; EP003, 3 + 166.
Após última divergência de ativo (21/01/2026) em EP003, restaram
165 sessões de símbolos iguais com riqueza distinta. A relação
`W_com/W_sem` passou de 0,772429504 no primeiro pregão após a
última divergência para 0,772424604 no final. Diferença nominal
no horizonte de 20 sessões: −US$ 1.412.175,96; no fim:
−US$ 2.250.682,75; aumento do gap nominal após 20 sessões
de −US$ 838.506,78 é sobretudo capital composto diferente,
NÃO prova de novas ações divergentes depois desse horizonte.
O target atual JÁ usa replay contrafactual de 20 sessões,
portanto não o descrever como retorno isolado de um dia.
Hipótese futura: aprendizado/avaliação de trajetórias condicionais
ao estado completo da carteira; não treinado/testado nesta etapa.
Protocolo confirmatório: pré-registro com parâmetros/cutoffs fixos
antes de nova janela pós-17/09/2026, snapshot futuro separado,
purge máximo coerente com outcomes até 60 sessões; nenhum ajuste
pelos três folds históricos já examinados.
Impacto: **somente análise/documentação**. Nenhum modelo, dado,
script, workflow ou versão alterado. Branch única e
`pesquisar_decision_focused_spyder.py` mantidos; sem tag/merge.
Testes: sem execução nova necessária para documento; resultado e
reconciliação históricos seguem íntegros. Próximo marcador de
código permanece `[TCC-DFL:FIX-005]` apenas se alteração concreta.


<!-- [TCC-DFL:REPRO-001] -->
**Reprodução recebida em 28/09/2026: `decision_focused(5).zip`.**
SHA-256 `2bad4be01b6dc8b0f22a051d3cbe2e34d28aafab4320aed74f67d37eb9203d1a`.
Comparação byte a byte com `decision_focused(4).zip`: 30/31 arquivos
regulares idênticos; somente `active/summary.json` possui 53 diferenças,
todas em campos `*_seconds` de duração computacional.
Curvas, decisões, operações, rótulos e métricas NÃO mudaram.
Control US$ 10.082.425,91; regressão US$ 7.639.150,63; reconciliação
de três episódios e seis overrides fecha −US$ 2.443.275,28 com erro zero.
Registro: `docs/results/decision-focused-fix-004-episode-attribution.md`.
Evidência adicional de reprodutibilidade do mesmo snapshot/protocolo,
NÃO validação confirmatória de modelo novo. Nenhum código ou parâmetro
alterado, mesma branch `research/decision-focused-oof-v2`, mesma
versão `v1.3.0-dev.3` e mesmo script original.


<!-- [TCC-DFL:FIX-005] -->
**Fix 005 — instrumentação de cobertura temporal de estados (28/09/2026).**
Causa: 7.717 linhas de labels ocultam múltiplos candidatos por
estado/data e cobertura limitada de incumbentes/holding no treino.
Documento detalhado:
`docs/research/decision-focused-state-support-fix-005.md`.
Origem diagnóstica: `decision_focused(5).zip` SHA-256
`2bad4be01b6dc8b0f22a051d3cbe2e34d28aafab4320aed74f67d37eb9203d1a`.
Avaliados rótulos de TREINO seguros (decision_date e outcome_end
estritamente antes de validation_start): Fold1 483 linhas/94 datas;
Fold2 2038/381; Fold3 3598/647. No OOS Control, incumbentes não
vistos no treino: 343/504 (68,06%), 96/504 (19,05%),
259/539 (48,05%). Pares exatos (ativo, holding) não vistos:
465/504 (92,26%), 265/504 (52,58%), 385/539 (71,43%).
No fold3 AVGO não aparece como incumbent nos labels disponíveis,
mas aparece como incumbente OOS no Control em 51 sessões. Isso NÃO
prova causalidade nem falha de generalização do LightGBM; suas
features de candidato/incumbente podem generalizar.
Implementação: módulo INTERNO `reproducao/diagnostico_estado.py`,
captura puramente observacional de
`research_holding_days_at_decision` na `reproducao/decision_focused_v2.py`,
exportação no ÚNICO `pesquisar_decision_focused_spyder.py`,
testes `tests/test_diagnostico_estado.py` e workflow único atualizado.
Novos arquivos de output: `state_support_folds.csv`,
`state_support_assets.csv`, `state_support_oos_sessions.csv`,
`state_support_checks.json`. Estados aqui PARCIAIS: ativo/holding,
não riqueza/quantidade/entry price completo. OOS usado para auditoria
descritiva, NUNCA para retreinamento/guards.
Testes: `reproduction-tests` run `36416400586` (commit `2f0fee3`):
**66 aprovados, Ruff aprovado**. Workflow completo
`decision-focused-research` run `36416400770`: consultar conclusão
e artefatos para conferir invariância dos capitais; não afirmar
resultado novo enquanto o backtest não encerrar.
Impacto: instrumentação apenas; NÃO alterados dataset congelado,
LightGBM, loss, Control, Soft, decisão da regressão, hiperparâmetros
ou horizonte. Mesma branch e versão de trabalho `v1.3.0-dev.3`;
não criar novo entrypoint, branch, tag ou merge.
Próximo passo: analisar os CSVs novos e, só depois, decidir desenho
de hipótese state-aware pré-registrada e avaliação futura separada.
Marcador seguinte `[TCC-DFL:FIX-006]` só se houver mudança concreta.


<!-- [TCC-DFL:FIX-006] -->
**Fix 006 — fallback seguro para módulos antigos retidos no Spyder (28/09/2026).**
Causa: a execução local do usuário concluiu a pesquisa mas falhou na
auditoria descritiva em `pesquisar_decision_focused_spyder.py:85` com
`ValueError: Colunas ausentes em Regressao:
['research_holding_days_at_decision']`. O código da branch já produz
esse campo em `_criar_politica_v2` e o mesmo pipeline passou no GitHub
run `36416400770` em processo novo. A discrepância é compatível
com um módulo `reproducao.decision_focused_v2` retido em memória no
kernel Spyder após `git pull`. Não inferir erro de modelo,
de alocação ou da fonte Alpaca a partir desse stacktrace.
Arquivos: `reproducao/diagnostico_estado.py`,
`pesquisar_decision_focused_spyder.py`, `tests/test_diagnostico_estado.py`,
este CONTEXTO_MESTRE.md. Nenhum executável novo.
Correção A: no início do ÚNICO entrypoint, `importlib.reload`
explícito de `decision_focused_v2` e `diagnostico_estado`
antes de iniciar a pesquisa, vinculando as funções recarregadas.
Correção B: `recuperar_holding_regressao` deriva o holding da
TRAJETÓRIA EXATA da regressão (par `previous_asset` /
`selected_asset`, ordenado por timestamp de execução), sem
copiar o holding do Control. Cada data de decisão reconstrói
estado pré-decisão: CASH=0, nova posição=1, permanência +1.
Valida datas, ordem, continuidade do incumbente e ativo da
regressão; caso o campo auxiliar exista, confronta-o com o valor
reconstruído e falha caso haja diferença. Isso funciona também com
o objeto `pesquisa` já computado por um módulo antigo, evitando
novo treinamento se o namespace estiver disponível. Após recuperar
os estados, atualiza somente os diagnósticos exportados.
Testes: sintéticos para estado no replay, campo ausente/presente,
divergência de holding, descontinuidade, desalinhamento e presença
de reload no entrypoint. Consultar run `36441325476` para resultado
dos testes e run `36441325445` para backtest completo do commit
`6869240a2222d0618b9682cc55a45558fc3a7c85`; NÃO alegar
sucesso antes de consultar suas conclusões.
Impacto esperado: corrigir APENAS a auditoria, sem alteração de
LightGBM, Control, Soft, regressão de investimento, loss,
snapshot, fold, horizonte ou dados financeiros. Mesma branch
`research/decision-focused-oof-v2`, versão de trabalho
`v1.3.0-dev.3`; sem tag/merge.
Próximos passos: conferir 1.547 sessões nos CSVs de suporte,
`intervention_reconciliation.json` com erro 0, e confirmar que
capitais reproduzem os valores congelados antes de fechar FIX-006.
Próximo marcador `[TCC-DFL:FIX-007]` somente se nova mudança concreta.
