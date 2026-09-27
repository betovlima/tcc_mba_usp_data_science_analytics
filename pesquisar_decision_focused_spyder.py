"""Ponto de entrada UNICO da pesquisa Decision-Focused, por secoes do Spyder.

[TCC-DFL:FIX-003] Consolida neste arquivo a evolucao OOF/DFL do experimento.
Historico v1.3.0-dev.2 permanece em docs/results; nao usar executavel paralelo.
Nao muda dados congelados nem Control/Soft oficiais.
"""
# %% 0 - Imports e configuracao; somente dados da pesquisa congelada
import json
from pathlib import Path

from engine.configuracao import CONFIG, EXPERIMENT_VERSION
from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.snapshot_portatil import preparar_snapshot_verificado
from reproducao.preparacao import prepare_model_frames
from reproducao.experimento import build_folds, build_variant_configs, run_variant
from reproducao.decision_focused_v2 import RESEARCH_VERSION, executar_pesquisa_v2
from reproducao.auditoria_intervencoes import _validar_equivalencia

RAIZ = Path(__file__).resolve().parent
DADOS_ORIGEM = SnapshotPaths.research(RAIZ)
DIRETORIO_RESULTADOS = RAIZ / "output" / "decision_focused" / "active"
HORIZONTE = 20
EXECUTAR_BASELINES = True

print(f"[decision-focused] version={RESEARCH_VERSION} backend=CPU", flush=True)
print(f"[decision-focused] reference={EXPERIMENT_VERSION} frozen_only=true", flush=True)

# %% 1 - Validacao bit a bit do snapshot versionado
DADOS, snapshot_audit = preparar_snapshot_verificado(
    DADOS_ORIGEM, DIRETORIO_RESULTADOS / "snapshot_verified",
)
manifesto = validate_snapshot(DADOS)
frames, exclusoes, diagnosticos_dados, auditoria_dados = prepare_model_frames(
    DADOS, assets=CONFIG.assets, comparar_snapshot_referencia=True,
)
config_control, config_soft = build_variant_configs(frames, CONFIG)
datas, folds = build_folds(frames, config_control)
print(f"[decision-focused] eligible={len(frames)} folds={len(folds)}", flush=True)

# %% 2 - Control e Soft inalterados; sempre executar na comparacao oficial
control_result = soft_result = None
control_metrics = soft_metrics = None
if EXECUTAR_BASELINES:
    control_result, control_metrics = run_variant(
        "CONTROL", frames, config_control, folds,
    )
    soft_result, soft_metrics = run_variant(
        "SOFT_HORIZON_CONSENSUS", frames, config_soft, folds,
    )

# %% 3 - OOF historico, estados on-policy, calibracao temporal e backtest OOS
pesquisa = executar_pesquisa_v2(
    frames, config_control, horizonte=HORIZONTE,
)
print(
    f"[decision-focused] REGRESSION={pesquisa.regression_metrics['ending_capital']:,.2f} "
    f"DFL_PAIRWISE={pesquisa.dfl_metrics['ending_capital']:,.2f} "
    f"DFL_SOFTMAX={pesquisa.softmax_metrics['ending_capital']:,.2f}",
    flush=True,
)

# %% 4 - Auditoria e exportacao; preserva resultado anterior
# [TCC-DFL:FIX-004] Identidade economica do replay BASELINE com Control.
# Nao ha treino adicional, nem mudanca na politica durante a auditoria.
auditoria_intervencoes = pesquisa.auditoria_intervencoes
if control_result is not None:
    error_control = _validar_equivalencia(
        auditoria_intervencoes.baseline_result, control_result,
        context="replay base vs Control oficial",
        rtol=1e-8, atol=1e-4,
    )
    auditoria_intervencoes.checks["baseline_vs_official_control_max_abs"] = error_control
print(
    "[decision-focused] audit episodes="
    f"{auditoria_intervencoes.checks['episodes']} "
    f"reconciliation={auditoria_intervencoes.checks['marginal_reconciliation_abs_error']:.9f}",
    flush=True,
)
DIRETORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
# Exportar a serie COMPLETA das duas politicas oficiais para permitir
# comparacao diaria dos estados, capital e custos (sem reconstruir graficos).
for label, result in (("control", control_result), ("soft", soft_result)):
    if result is None:
        continue
    result.predictions.to_csv(DIRETORIO_RESULTADOS / f"{label}_predictions.csv")
    result.trades.to_csv(DIRETORIO_RESULTADOS / f"{label}_trades.csv", index=False)
auditoria_intervencoes.episodios.to_csv(
    DIRETORIO_RESULTADOS / "intervention_episode_attribution.csv", index=False,
)
auditoria_intervencoes.curvas.to_csv(
    DIRETORIO_RESULTADOS / "intervention_bridge_curves.csv", index=False,
)
auditoria_intervencoes.pares_diarios.to_csv(
    DIRETORIO_RESULTADOS / "intervention_episode_daily_pairs.csv", index=False,
)
auditoria_intervencoes.operacoes.to_csv(
    DIRETORIO_RESULTADOS / "intervention_bridge_trades.csv", index=False,
)
(DIRETORIO_RESULTADOS / "intervention_reconciliation.json").write_text(
    json.dumps(auditoria_intervencoes.checks, indent=2, ensure_ascii=False, default=str),
    encoding="utf-8",
)
pesquisa.labels.to_csv(DIRETORIO_RESULTADOS / "oof_counterfactual_labels.csv", index=False)
pesquisa.audit.to_csv(DIRETORIO_RESULTADOS / "inner_oof_audit.csv", index=False)
pesquisa.calibration.to_csv(DIRETORIO_RESULTADOS / "temporal_calibration.csv", index=False)
pesquisa.regression_decisions.to_csv(
    DIRETORIO_RESULTADOS / "regression_decisions.csv", index_label="decision_date",
)
pesquisa.dfl_decisions.to_csv(
    DIRETORIO_RESULTADOS / "dfl_decisions.csv", index_label="decision_date",
)
pesquisa.softmax_decisions.to_csv(
    DIRETORIO_RESULTADOS / "softmax_dfl_decisions.csv", index_label="decision_date",
)
for name, result in (
    ("regression", pesquisa.regression_result),
    ("dfl", pesquisa.dfl_result),
    ("softmax_dfl", pesquisa.softmax_result),
):
    result.predictions.to_csv(DIRETORIO_RESULTADOS / f"{name}_predictions.csv")
    result.trades.to_csv(DIRETORIO_RESULTADOS / f"{name}_trades.csv", index=False)

archived_v1 = json.loads(
    (RAIZ / "docs" / "results" / "v1.3.0-dev.2-negative-result.json").read_text(
        encoding="utf-8",
    )
)
summary = {
    "research_version": RESEARCH_VERSION,
    "reference_experiment_version": EXPERIMENT_VERSION,
    "baseline_commit": "f9cf29fdb736676d0d3e26780481be99813c602a",
    "snapshot_audit": snapshot_audit,
    "data_mode": "frozen, SHA-256 verified; no provider calls",
    "label_horizon_sessions": HORIZONTE,
    "method": "chronological nested out-of-fold and on-policy state augmentation",
    "calibration": "prior temporal validation only, no OOS threshold selection",
    "dfl_pairwise_definition": "regret-weighted pairwise ranking surrogate",
    "dfl_softmax_definition": "differentiable expected per-decision regret with group softmax; not end-to-end sequential replay",
    "control": control_metrics,
    "soft": soft_metrics,
    "oof_counterfactual_regression": pesquisa.regression_metrics,
    "oof_decision_focused_surrogate": pesquisa.dfl_metrics,
    "oof_softmax_decision_focused": pesquisa.softmax_metrics,
    "intervention_audit": auditoria_intervencoes.checks,
    "intervention_attribution_method": (
        "Cumulative chronological episode toggles; same prior decisions, "
        "then baseline policy; marginal effects depend on inclusion order."
    ),
    "previous_negative_result": {
        "research_version": archived_v1["research_version"],
        "source_archive_sha256": archived_v1["source_archive_sha256"],
        "variants": archived_v1["variants"],
    },
}
(DIRETORIO_RESULTADOS / "summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False, default=str),
    encoding="utf-8",
)
print("[decision-focused] completed output=", DIRETORIO_RESULTADOS, flush=True)
