"""Pesquisa isolada, executavel por secoes no console do Spyder.

A pesquisa oficial em reproduzir_experimento_spyder.py nao e modificada.
Nao realiza download; exige os CSVs congelados no Git e valida SHA-256.
"""
# %% 0 - Imports e configuracao (somente snapshot congelado)
import json
from pathlib import Path

from engine.configuracao import CONFIG, EXPERIMENT_VERSION
from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.preparacao import prepare_model_frames
from reproducao.experimento import build_folds, build_variant_configs, run_variant
from reproducao.decision_focused import executar_pesquisa
from reproducao.snapshot_portatil import preparar_snapshot_verificado

RAIZ = Path(__file__).resolve().parent
SNAPSHOT_ORIGEM = SnapshotPaths.research(RAIZ)
SNAPSHOT = SNAPSHOT_ORIGEM
SAIDA = RAIZ / "output" / "decision_focused"
EXECUTAR_BASELINES = True
HORIZONTE = 20
print("[research] version=1.3.0-dev.1 source=research-frozen backend=CPU", flush=True)
print(f"[research] reference_version={EXPERIMENT_VERSION} baseline=main", flush=True)

# %% 1 - Auditoria dos dados; esta execucao nunca acessa a Alpaca
SNAPSHOT, snapshot_audit = preparar_snapshot_verificado(
    SNAPSHOT_ORIGEM, SAIDA / "snapshot_verified",
)
manifesto = validate_snapshot(SNAPSHOT)
print(f"[research] snapshot_audit={snapshot_audit}", flush=True)
frames, exclusoes, diagnosticos, auditoria = prepare_model_frames(
    SNAPSHOT, assets=CONFIG.assets, comparar_snapshot_referencia=True,
)
config_control, config_soft = build_variant_configs(frames, CONFIG)
datas, folds = build_folds(frames, config_control)
print(f"[research] eligible={len(frames)} folds={len(folds)}", flush=True)

# %% 2 - Baselines congelados: mesma politica, mesmos folds
if EXECUTAR_BASELINES:
    control_result, control_metrics = run_variant(
        "CONTROL", frames, config_control, folds,
    )
    soft_result, soft_metrics = run_variant(
        "SOFT_HORIZON_CONSENSUS", frames, config_soft, folds,
    )
else:
    control_result = soft_result = None
    control_metrics = soft_metrics = None

# %% 3 - Rotulos de vantagem contrafactual, regressao e ranking por regret
pesquisa = executar_pesquisa(frames, config_control, horizonte=HORIZONTE)
print(
    "[research] OOS regression="
    f"{pesquisa.regressao_metricas['ending_capital']:,.2f} "
    f"DFL={pesquisa.dfl_metricas['ending_capital']:,.2f}", flush=True,
)

# %% 4 - Exportacao independente; nao sobrescreve output/reproducao
SAIDA.mkdir(parents=True, exist_ok=True)
pesquisa.rotulos.to_csv(SAIDA / "counterfactual_labels.csv", index=False)
pesquisa.auditoria_folds.to_csv(SAIDA / "temporal_audit.csv", index=False)
pesquisa.decisoes_regressao.to_csv(SAIDA / "regression_decisions.csv", index_label="decision_date")
pesquisa.decisoes_dfl.to_csv(SAIDA / "dfl_decisions.csv", index_label="decision_date")
pesquisa.regressao.predictions.to_csv(SAIDA / "regression_predictions.csv")
pesquisa.regressao.trades.to_csv(SAIDA / "regression_trades.csv", index=False)
pesquisa.dfl.predictions.to_csv(SAIDA / "dfl_predictions.csv")
pesquisa.dfl.trades.to_csv(SAIDA / "dfl_trades.csv", index=False)
comparacao = {
    "control": control_metrics,
    "soft": soft_metrics,
    "counterfactual_regression": pesquisa.regressao_metricas,
    "decision_focused_surrogate": pesquisa.dfl_metricas,
    "research_version": "1.3.0-dev.2",
    "reference_experiment_version": EXPERIMENT_VERSION,
    "data_mode": "dados/pesquisa (frozen, verified bytes)",
    "snapshot_audit": snapshot_audit,
    "label_horizon_sessions": HORIZONTE,
    "test_fold_count": len(folds),
    "label_generation": "one forced action, then same control policy",
    "dfl_definition": "regret-weighted pairwise ranking surrogate; not end-to-end differentiable DFL",
}
(SAIDA / "summary.json").write_text(
    json.dumps(comparacao, indent=2, ensure_ascii=False, default=str),
    encoding="utf-8",
)
print(f"[research] output={SAIDA}", flush=True)
