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
`36349759992`: verifique conclusão e artefato antes de declarar resultados.
Status: código e testes validados; replays econômicos completos em validação
na mesma branch e versão de trabalho `v1.3.0-dev.3`.
O resultado anterior documentado permanece o último resultado econômico
confirmado até uma execução completa COM estes novos replays.
Próximo passo: conferir `intervention_episode_attribution.csv` e
`intervention_reconciliation.json`, verificar efeito de 20 sessões,
persistência, fees, posições e reconciliação de capital antes de
qualquer hipótese de novo modelo. NÃO ajustar parâmetros no OOS visto.
Próximo marcador: `[TCC-DFL:FIX-005]` somente se houver correção concreta.
