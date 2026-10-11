"""Teste decisivo da hipótese de substituição dos 24 candidatos congelados.

Pergunta científica
-------------------
Quando um candidato congelado teria força suficiente para substituir a decisão
do U67, as informações conhecidas naquela sessão conseguem distinguir uma
substituição futura favorável de uma desfavorável?

Esta fase foi desenhada para impedir nova exploração indefinida:
- a shortlist de 24 ativos já está congelada;
- nenhum ativo é adicionado/removido com base no resultado deste script;
- não há backtest de capital dos candidatos nesta fase;
- a unidade é (candidato, incumbente, data de decisão);
- o alvo usa dados futuros e é proibido como atributo;
- a validação segura simultaneamente tempo e candidato fora do treino;
- o critério de suporte é declarado antes da leitura dos resultados.

Se o critério não for atingido, a linha de pesquisa por agrupamento deve ser
encerrada, sem procurar novo K, nova PCA ou nova shortlist com base no resultado.

Execução no Spyder
------------------
1. Execute reproduzir_experimento.py ao menos uma vez na versão atual.
2. Execute construir_snapshot_shortlist_cluster_spyder.py.
3. Abra este arquivo e execute com F5.

Saídas:
output/substituicoes_cluster/
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import shutil
import time
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from engine.configuracao import CONFIG, U67_REQUESTED_ASSETS
from engine.execucao import aplicar_deslizamento, calcular_taxas_referencia
from engine.modelo_lightgbm import (
    _ajustar_modelos_lightgbm,
    _construir_contexto_execucao,
)
from engine.rotacao import (
    _politica_agendada,
    _politica_utilidade,
    _precalcular_utilidades_modelo,
    _simular_exato,
    construir_quadro_rotacao,
)
from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.experimento import build_control_config
from reproducao.preparacao import prepare_model_frames


ROOT = Path(__file__).resolve().parent
U67 = SnapshotPaths.u67(ROOT)
FROZEN_DIR = ROOT / "dados" / "cluster_candidate_search"
CANDIDATE_SNAPSHOT = SnapshotPaths.from_root(
    FROZEN_DIR / "snapshot"
)
FREEZE_JSON = FROZEN_DIR / "cluster_shortlist_freeze.json"
FREEZE_CSV = FROZEN_DIR / "cluster_shortlist_frozen.csv"
CANDIDATE_SNAPSHOT_JSON = (
    FROZEN_DIR / "cluster_candidate_snapshot.json"
)

U67_REPRO_DIR = ROOT / "output" / "reproducao"
U67_RESULT_JSON = U67_REPRO_DIR / "reproducao_u67.json"
U67_FOLD_MARGINS = U67_REPRO_DIR / "u67_fold_margins.csv"

OUT = ROOT / "output" / "substituicoes_cluster"

SCRIPT_VERSION = "cluster-substitution-test-v1.0.0"
ANALYSIS_SCHEMA = "cluster-candidate-substitution-holdout-v1"
EXPECTED_FREEZE_VERSION = "cluster-shortlist-freeze-v1.0.0"
EXPECTED_SNAPSHOT_VERSION = "cluster-candidate-snapshot-v1.0.0"
EXPECTED_U67_CAPITAL = 76_927_051.38897176
EXPECTED_ANALYSIS_END = "2026-10-06"

# Evento que realmente poderia mudar a decisão da política:
# candidato supera o melhor U67 e também satisfaz a margem de troca.
EVENT_DEFINITION = (
    "incumbent!=CASH; min-hold satisfied; candidate_score>best_u67_score; "
    "candidate_score>=incumbent_score+effective_switch_margin; "
    "candidate_score>cash_threshold"
)

# Critério pré-declarado antes de observar o resultado deste teste.
# A hipótese só recebe suporte se o classificador fixo generalizar com
# AUC >= 0.60 e os limites inferiores de dois bootstraps em blocos forem > 0.50.
SUPPORT_AUC_MIN = 0.60
BOOTSTRAP_LOWER_MIN = 0.50
BOOTSTRAP_REPETITIONS = 1000
MIN_EVENTS = 200
MIN_CANDIDATES_WITH_EVENTS = 8
ASSET_HOLDOUT_GROUPS = 4
RANDOM_SEED = 42

HORIZONS = (5, 10, 20, 40, 60)

RELATIVE_TECHNICAL_FEATURES = (
    "return_5",
    "return_20",
    "return_60",
    "return_120",
    "vol_20",
    "vol_60",
    "ema_distance_20",
    "ema_20_vs_50",
    "ema_slope_50_10",
    "rsi_14",
    "atr_pct_14",
    "distance_from_high_20",
    "channel_position_20",
    "trend_efficiency_20",
    "trend_efficiency_60",
    "momentum_acceleration_5_20",
    "momentum_acceleration_20_60",
    "volume_ratio_5_20",
)

BASELINE_CONTEXT_COLUMNS = (
    "current_asset",
    "current_score",
    "best_asset",
    "best_score",
    "second_asset",
    "second_score",
    "effective_switch_margin",
    "holding_days_at_decision",
    "current_asset_rank",
    "universe_score_mean",
    "universe_score_std",
    "current_score_zscore",
    "best_score_zscore",
    "positive_score_count",
    "finite_score_count",
    "final_action_asset",
    "decision_reason",
    "position_return_since_entry",
    "position_drawdown_from_peak",
    "position_mfe_so_far",
    "position_mae_so_far",
    "days_current_not_top1",
    "consecutive_days_current_not_top1",
    "spy_return_5",
    "spy_return_20",
    "spy_realized_volatility_20",
    "universe_breadth_5",
    "universe_breadth_20",
)

CORE_FEATURES = (
    "candidate_score",
    "incumbent_score",
    "best_u67_score",
    "gap_vs_incumbent",
    "gap_vs_best_u67",
    "strength_vs_margin",
    "candidate_score_zscore",
    "score_mean_20",
    "score_std_20",
    "positive_share_20",
    "beats_best_share_20",
    "beats_incumbent_share_20",
    "score_mean_60",
    "score_std_60",
    "positive_share_60",
    "beats_best_share_60",
    "beats_incumbent_share_60",
    "holding_days_at_decision",
    "current_asset_rank",
    "universe_score_mean",
    "universe_score_std",
    "current_score_zscore",
    "best_score_zscore",
    "positive_score_count",
    "finite_score_count",
    "position_return_since_entry",
    "position_drawdown_from_peak",
    "position_mfe_so_far",
    "position_mae_so_far",
    "days_current_not_top1",
    "consecutive_days_current_not_top1",
    "spy_return_5",
    "spy_return_20",
    "spy_realized_volatility_20",
    "universe_breadth_5",
    "universe_breadth_20",
    *tuple(f"delta_{feature}" for feature in RELATIVE_TECHNICAL_FEATURES),
)

SCORE_ONLY_FEATURES = (
    "candidate_score",
    "gap_vs_incumbent",
    "gap_vs_best_u67",
    "strength_vs_margin",
    "candidate_score_zscore",
)


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"Arquivo ausente: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_contract() -> tuple[dict, pd.DataFrame, tuple[str, ...]]:
    freeze = _read_json(FREEZE_JSON)
    snapshot = _read_json(CANDIDATE_SNAPSHOT_JSON)

    if freeze.get("script_version") != EXPECTED_FREEZE_VERSION:
        raise RuntimeError(
            "Shortlist congelada em versão inesperada: "
            f"{freeze.get('script_version')}"
        )
    if freeze.get("selection_locked") is not True:
        raise RuntimeError("A shortlist não está bloqueada.")
    if freeze.get("financial_outcomes_used_for_selection") is not False:
        raise RuntimeError(
            "A shortlist não pode ter usado resultados financeiros."
        )
    if freeze.get("backtest_executed_before_freeze") is not False:
        raise RuntimeError(
            "A shortlist deve ter sido congelada antes do backtest."
        )

    if snapshot.get("script_version") != EXPECTED_SNAPSHOT_VERSION:
        raise RuntimeError(
            "Snapshot dos candidatos em versão inesperada."
        )
    if snapshot.get("status") != "candidate_snapshot_ready":
        raise RuntimeError(
            "Snapshot dos candidatos não está pronto."
        )
    if snapshot.get("structural_exclusions"):
        raise RuntimeError(
            "Há exclusões estruturais no snapshot dos candidatos."
        )
    if snapshot.get("analysis_end") != EXPECTED_ANALYSIS_END:
        raise RuntimeError(
            "Cutoff do snapshot de candidatos inesperado."
        )

    shortlist = pd.read_csv(FREEZE_CSV)
    assets = tuple(
        shortlist["asset"].astype(str).str.strip().str.upper()
    )
    frozen_assets = tuple(
        str(asset).strip().upper()
        for asset in (freeze.get("assets") or [])
    )
    snapshot_assets = tuple(
        str(asset).strip().upper()
        for asset in (snapshot.get("assets") or [])
    )
    if assets != frozen_assets or assets != snapshot_assets:
        raise RuntimeError(
            "A lista congelada não coincide com o snapshot dos candidatos."
        )
    if len(assets) != 24 or len(set(assets)) != 24:
        raise RuntimeError("Esperados exatamente 24 candidatos.")

    return freeze, shortlist, assets


def _load_u67_reproduction() -> tuple[dict, pd.DataFrame]:
    payload = _read_json(U67_RESULT_JSON)
    if payload.get("status") != "completed":
        raise RuntimeError(
            "A reprodução oficial U67 atual não está concluída."
        )

    metrics = payload.get("metrics") or {}
    ending = float(metrics.get("ending_capital") or 0.0)
    if not math.isclose(
        ending,
        EXPECTED_U67_CAPITAL,
        rel_tol=0.0,
        abs_tol=0.01,
    ):
        raise RuntimeError(
            "Checkpoint U67 divergente. Execute reproduzir_experimento.py "
            "na branch atual antes deste teste. "
            f"observado={ending:,.8f} esperado={EXPECTED_U67_CAPITAL:,.8f}"
        )

    if not U67_FOLD_MARGINS.exists():
        raise RuntimeError(
            "u67_fold_margins.csv ausente. Execute reproduzir_experimento.py."
        )
    margins = pd.read_csv(U67_FOLD_MARGINS)
    required = {
        "fold_id",
        "selected_margin",
        "effective_margin",
    }
    if not required.issubset(margins.columns):
        raise RuntimeError(
            "u67_fold_margins.csv não possui o contrato esperado."
        )
    return payload, margins


def _prepare_data(
    candidate_assets: tuple[str, ...],
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    list[str],
    tuple[str, ...],
    object,
    object,
    pd.DatetimeIndex,
    list[dict],
    pd.DatetimeIndex,
    dict[pd.Timestamp, int],
]:
    validate_snapshot(U67)
    validate_snapshot(CANDIDATE_SNAPSHOT)

    u67_raw, u67_exclusions, _, _ = prepare_model_frames(
        U67,
        assets=U67_REQUESTED_ASSETS,
        csv_float_precision="round_trip",
    )
    if len(u67_raw) != 65 or {
        str(item.get("symbol"))
        for item in u67_exclusions
    } != {"CLMT", "DOC"}:
        raise RuntimeError(
            "O contrato efetivo U67 precisa continuar em 65 ativos."
        )

    candidate_raw, candidate_exclusions, _, _ = prepare_model_frames(
        CANDIDATE_SNAPSHOT,
        assets=candidate_assets,
        csv_float_precision="round_trip",
    )
    if candidate_exclusions or tuple(candidate_raw) != candidate_assets:
        raise RuntimeError(
            "Os 24 candidatos congelados não sobreviveram integralmente "
            "à preparação atual."
        )

    config_u67 = build_control_config(u67_raw, CONFIG).copiar_modelo(
        update={"analysis_end_date": EXPECTED_ANALYSIS_END}
    )

    (
        frames_u67,
        common_dates,
        _calendar_source,
        u67_symbols,
        folds,
        all_decision_dates,
        decision_to_fold,
        _decision_metadata,
    ) = _construir_contexto_execucao(
        u67_raw,
        config_u67,
    )

    config_all = build_control_config(
        {**u67_raw, **candidate_raw},
        CONFIG,
    ).copiar_modelo(
        update={"analysis_end_date": EXPECTED_ANALYSIS_END}
    )

    candidate_frames = {
        symbol: construir_quadro_rotacao(
            candidate_raw[symbol],
            config_all,
        ).reindex(common_dates)
        for symbol in candidate_assets
    }

    return (
        frames_u67,
        candidate_frames,
        u67_symbols,
        candidate_assets,
        config_u67,
        config_all,
        common_dates,
        folds,
        all_decision_dates,
        decision_to_fold,
    )


def _margin_map(
    folds: list[dict],
    margins: pd.DataFrame,
) -> dict[int, dict[str, float]]:
    result = {}
    for row in margins.itertuples(index=False):
        result[int(row.fold_id)] = {
            "selected_margin": float(row.selected_margin),
            "effective_margin": float(row.effective_margin),
        }
    expected = {int(fold["fold_id"]) for fold in folds}
    if set(result) != expected:
        raise RuntimeError(
            "Os folds do checkpoint U67 não coincidem com o contexto atual."
        )
    return result


def _fit_and_score(
    frames_u67: dict[str, pd.DataFrame],
    candidate_frames: dict[str, pd.DataFrame],
    u67_symbols: list[str],
    candidate_assets: tuple[str, ...],
    config_u67,
    config_all,
    common_dates: pd.DatetimeIndex,
    folds: list[dict],
    margins: dict[int, dict[str, float]],
) -> tuple[
    dict[int, object],
    dict[pd.Timestamp, dict],
    dict[int, dict[pd.Timestamp, np.ndarray]],
]:
    fold_policies: dict[int, object] = {}
    decision_diagnostics: dict[pd.Timestamp, dict] = {}
    candidate_cache_by_fold: dict[
        int, dict[pd.Timestamp, np.ndarray]
    ] = {}

    for position, fold in enumerate(folds, start=1):
        fold_id = int(fold["fold_id"])
        final_fit_dates = common_dates[
            : int(fold["final_fit_end_index"])
        ]
        decision_dates = pd.DatetimeIndex(
            fold["decision_dates"]
        )
        score_dates = decision_dates[:-1]

        print(
            f"[fit] fold={fold_id} {position}/{len(folds)} "
            f"u67={len(u67_symbols)} candidates={len(candidate_assets)}",
            flush=True,
        )

        u67_models = _ajustar_modelos_lightgbm(
            frames_u67,
            u67_symbols,
            final_fit_dates,
            config_u67,
            phase=f"cluster_substitution_u67_fold_{fold_id}",
            technical_log_callback=lambda message: print(
                f"[technical] {message}",
                flush=True,
            ),
        )
        u67_cache, _ = _precalcular_utilidades_modelo(
            u67_models,
            frames_u67,
            u67_symbols,
            decision_dates,
            config_u67,
        )

        fold_diag: dict[pd.Timestamp, dict] = {}
        selected_margin = margins[fold_id]["selected_margin"]
        effective_margin = margins[fold_id]["effective_margin"]
        fold_policies[fold_id] = _politica_utilidade(
            u67_models,
            frames_u67,
            u67_symbols,
            config_u67,
            effective_margin,
            decision_diagnostics=fold_diag,
            fold_id=fold_id,
            calibrated_switch_margin=selected_margin,
            utility_cache=u67_cache,
        )
        # O dicionário é preenchido no replay da política.
        decision_diagnostics.update(fold_diag)

        candidate_models = _ajustar_modelos_lightgbm(
            candidate_frames,
            list(candidate_assets),
            final_fit_dates,
            config_all,
            phase=f"cluster_substitution_candidates_fold_{fold_id}",
            technical_log_callback=lambda message: print(
                f"[technical] {message}",
                flush=True,
            ),
        )
        candidate_cache, _ = _precalcular_utilidades_modelo(
            candidate_models,
            candidate_frames,
            list(candidate_assets),
            score_dates,
            config_all,
        )
        candidate_cache_by_fold[fold_id] = candidate_cache

        # A closure da política mantém fold_diag; precisamos guardar a mesma
        # referência para consolidar depois do replay.
        fold_policies[fold_id]._cluster_diag = fold_diag  # type: ignore[attr-defined]

    return fold_policies, decision_diagnostics, candidate_cache_by_fold


def _replay_u67_with_diagnostics(
    fold_policies: dict[int, object],
    decision_to_fold: dict[pd.Timestamp, int],
    frames_u67: dict[str, pd.DataFrame],
    u67_symbols: list[str],
    all_decision_dates: pd.DatetimeIndex,
    config_u67,
    folds: list[dict],
) -> tuple[object, pd.DataFrame]:
    scheduled = _politica_agendada(
        fold_policies,
        decision_to_fold,
    )

    merged_diag: dict[pd.Timestamp, dict] = {}
    for policy in fold_policies.values():
        merged_diag.update(
            getattr(policy, "_cluster_diag", {})
        )

    decision_metadata = {}
    for fold in folds:
        for timestamp in fold["decision_dates"][:-1]:
            decision_metadata[pd.Timestamp(timestamp)] = {
                "fold_id": int(fold["fold_id"]),
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
            }

    result = _simular_exato(
        "cluster_substitution_u67_reference",
        scheduled,
        frames_u67,
        u67_symbols,
        all_decision_dates,
        config_u67,
        calcular_taxas_referencia,
        aplicar_deslizamento,
        decision_metadata=decision_metadata,
        policy_decision_diagnostics=merged_diag,
        model_label="TCC U67 Control reference",
        method_line=(
            "- Replay apenas para reconstruir o estado/incumbente de cada "
            "sessão; candidatos não participam do capital."
        ),
    )

    observed = float(
        result.metrics.get("strategy_ending_capital") or 0.0
    )
    if not math.isclose(
        observed,
        EXPECTED_U67_CAPITAL,
        rel_tol=0.0,
        abs_tol=0.01,
    ):
        raise RuntimeError(
            "O replay de referência U67 não reproduziu o checkpoint. "
            f"observado={observed:,.8f} esperado={EXPECTED_U67_CAPITAL:,.8f}"
        )

    baseline = result.predictions.reset_index().copy()
    baseline["decision_date"] = pd.to_datetime(
        baseline["decision_date"],
        utc=True,
    )
    if baseline["decision_date"].duplicated().any():
        raise RuntimeError(
            "Replay U67 produziu decision_date duplicada."
        )
    return result, baseline


def _safe_value(
    frame: pd.DataFrame,
    timestamp: pd.Timestamp,
    column: str,
) -> float:
    if column not in frame.columns or timestamp not in frame.index:
        return float("nan")
    value = pd.to_numeric(
        frame.at[timestamp, column],
        errors="coerce",
    )
    return float(value) if pd.notna(value) else float("nan")


def _target_difference(
    candidate_frame: pd.DataFrame,
    comparator_frame: pd.DataFrame | None,
    timestamp: pd.Timestamp,
) -> dict[str, float]:
    output: dict[str, float] = {}

    candidate_multi = _safe_value(
        candidate_frame,
        timestamp,
        "forward_risk_adjusted_utility",
    )
    comparator_multi = (
        _safe_value(
            comparator_frame,
            timestamp,
            "forward_risk_adjusted_utility",
        )
        if comparator_frame is not None
        else 0.0
    )

    output["target_delta_utility_multi"] = (
        candidate_multi - comparator_multi
        if np.isfinite(candidate_multi)
        and np.isfinite(comparator_multi)
        else np.nan
    )
    output["target_positive_multi"] = (
        float(output["target_delta_utility_multi"] > 0.0)
        if np.isfinite(output["target_delta_utility_multi"])
        else np.nan
    )

    for horizon in HORIZONS:
        candidate_value = _safe_value(
            candidate_frame,
            timestamp,
            f"forward_horizon_utility_{horizon}",
        )
        comparator_value = (
            _safe_value(
                comparator_frame,
                timestamp,
                f"forward_horizon_utility_{horizon}",
            )
            if comparator_frame is not None
            else 0.0
        )
        output[f"target_delta_utility_{horizon}"] = (
            candidate_value - comparator_value
            if np.isfinite(candidate_value)
            and np.isfinite(comparator_value)
            else np.nan
        )

    return output


def _build_sequence(
    baseline: pd.DataFrame,
    candidate_cache_by_fold: dict[
        int, dict[pd.Timestamp, np.ndarray]
    ],
    candidate_assets: tuple[str, ...],
    shortlist: pd.DataFrame,
) -> pd.DataFrame:
    baseline_by_date = baseline.set_index("decision_date")
    metadata = shortlist.set_index("asset")
    rows = []

    for fold_id, cache in sorted(candidate_cache_by_fold.items()):
        for timestamp, utilities in cache.items():
            if timestamp not in baseline_by_date.index:
                continue
            base = baseline_by_date.loc[timestamp]

            current_score = pd.to_numeric(
                base.get("current_score"),
                errors="coerce",
            )
            best_score = pd.to_numeric(
                base.get("best_score"),
                errors="coerce",
            )
            margin = pd.to_numeric(
                base.get("effective_switch_margin"),
                errors="coerce",
            )
            mean = pd.to_numeric(
                base.get("universe_score_mean"),
                errors="coerce",
            )
            std = pd.to_numeric(
                base.get("universe_score_std"),
                errors="coerce",
            )

            for idx, asset in enumerate(candidate_assets, start=1):
                score = float(utilities[idx])
                if not np.isfinite(score):
                    continue

                gap_current = (
                    score - float(current_score)
                    if pd.notna(current_score)
                    else np.nan
                )
                gap_best = (
                    score - float(best_score)
                    if pd.notna(best_score)
                    else np.nan
                )
                stable_std = (
                    float(std)
                    if pd.notna(std) and float(std) > 1e-12
                    else np.nan
                )

                row = {
                    "decision_date": timestamp,
                    "walk_forward_fold": int(fold_id),
                    "candidate_asset": asset,
                    "candidate_role": str(
                        metadata.at[asset, "search_role"]
                    ),
                    "cluster_novelty_ratio": float(
                        metadata.at[asset, "novelty_ratio"]
                    ),
                    "candidate_score": score,
                    "incumbent_asset": str(
                        base.get("current_asset") or "CASH"
                    ),
                    "incumbent_score": (
                        float(current_score)
                        if pd.notna(current_score)
                        else np.nan
                    ),
                    "best_u67_asset": str(
                        base.get("best_asset") or "CASH"
                    ),
                    "best_u67_score": (
                        float(best_score)
                        if pd.notna(best_score)
                        else np.nan
                    ),
                    "baseline_action_asset": str(
                        base.get("final_action_asset")
                        or base.get("selected_asset")
                        or "CASH"
                    ),
                    "effective_switch_margin": (
                        float(margin)
                        if pd.notna(margin)
                        else np.nan
                    ),
                    "gap_vs_incumbent": gap_current,
                    "gap_vs_best_u67": gap_best,
                    "strength_vs_margin": (
                        gap_current / float(margin)
                        if pd.notna(margin)
                        and float(margin) > 0
                        and np.isfinite(gap_current)
                        else np.nan
                    ),
                    "beats_incumbent": bool(
                        np.isfinite(gap_current)
                        and gap_current > 0.0
                    ),
                    "clears_switch_margin": bool(
                        np.isfinite(gap_current)
                        and pd.notna(margin)
                        and gap_current >= float(margin)
                    ),
                    "beats_best_u67": bool(
                        np.isfinite(gap_best)
                        and gap_best > 0.0
                    ),
                    "candidate_score_zscore": (
                        (score - float(mean)) / stable_std
                        if pd.notna(mean) and np.isfinite(stable_std)
                        else np.nan
                    ),
                }
                for column in BASELINE_CONTEXT_COLUMNS:
                    if column not in row:
                        row[column] = base.get(column)
                rows.append(row)

    sequence = pd.DataFrame(rows)
    if sequence.empty:
        raise RuntimeError("Nenhum score de candidato foi produzido.")

    sequence = sequence.sort_values(
        ["candidate_asset", "decision_date"],
        ignore_index=True,
    )

    grouped = sequence.groupby(
        "candidate_asset",
        sort=False,
    )
    for window in (20, 60):
        min_periods = max(5, window // 4)
        sequence[f"score_mean_{window}"] = grouped[
            "candidate_score"
        ].transform(
            lambda values, w=window, m=min_periods: values.rolling(
                w,
                min_periods=m,
            ).mean()
        )
        sequence[f"score_std_{window}"] = grouped[
            "candidate_score"
        ].transform(
            lambda values, w=window, m=min_periods: values.rolling(
                w,
                min_periods=m,
            ).std(ddof=0)
        )
        sequence[f"positive_share_{window}"] = grouped[
            "candidate_score"
        ].transform(
            lambda values, w=window, m=min_periods: (
                values.gt(0.0).astype(float).rolling(
                    w,
                    min_periods=m,
                ).mean()
            )
        )
        sequence[f"beats_best_share_{window}"] = grouped[
            "beats_best_u67"
        ].transform(
            lambda values, w=window, m=min_periods: (
                values.astype(float).rolling(
                    w,
                    min_periods=m,
                ).mean()
            )
        )
        sequence[f"beats_incumbent_share_{window}"] = grouped[
            "beats_incumbent"
        ].transform(
            lambda values, w=window, m=min_periods: (
                values.astype(float).rolling(
                    w,
                    min_periods=m,
                ).mean()
            )
        )

    return sequence


def _build_events(
    sequence: pd.DataFrame,
    frames_u67: dict[str, pd.DataFrame],
    candidate_frames: dict[str, pd.DataFrame],
    config_u67,
) -> pd.DataFrame:
    min_hold = int(config_u67.rotation_min_holding_days)
    cash_threshold = float(config_u67.rotation_cash_threshold)

    actionable = sequence.loc[
        sequence["incumbent_asset"].ne("CASH")
        & sequence["holding_days_at_decision"].fillna(0).ge(min_hold)
        & sequence["beats_best_u67"]
        & sequence["clears_switch_margin"]
        & sequence["candidate_score"].gt(cash_threshold)
    ].copy()

    rows = []
    for row in actionable.itertuples(index=False):
        record = row._asdict()
        timestamp = pd.Timestamp(row.decision_date)
        candidate = str(row.candidate_asset)
        incumbent = str(row.incumbent_asset)
        baseline_action = str(row.baseline_action_asset)

        candidate_frame = candidate_frames[candidate]
        incumbent_frame = frames_u67.get(incumbent)
        baseline_frame = (
            frames_u67.get(baseline_action)
            if baseline_action != "CASH"
            else None
        )

        for feature in RELATIVE_TECHNICAL_FEATURES:
            candidate_value = _safe_value(
                candidate_frame,
                timestamp,
                feature,
            )
            incumbent_value = (
                _safe_value(
                    incumbent_frame,
                    timestamp,
                    feature,
                )
                if incumbent_frame is not None
                else np.nan
            )
            record[f"delta_{feature}"] = (
                candidate_value - incumbent_value
                if np.isfinite(candidate_value)
                and np.isfinite(incumbent_value)
                else np.nan
            )

        primary = _target_difference(
            candidate_frame,
            baseline_frame,
            timestamp,
        )
        record.update(primary)

        secondary = _target_difference(
            candidate_frame,
            incumbent_frame,
            timestamp,
        )
        for key, value in secondary.items():
            record[f"{key}_vs_incumbent"] = value

        rows.append(record)

    events = pd.DataFrame(rows)
    if events.empty:
        raise RuntimeError(
            "Nenhum evento acionável de substituição foi encontrado."
        )

    assets_sorted = sorted(
        events["candidate_asset"].unique()
    )
    holdout_map = {
        asset: idx % ASSET_HOLDOUT_GROUPS
        for idx, asset in enumerate(assets_sorted)
    }
    events["asset_holdout_group"] = events[
        "candidate_asset"
    ].map(holdout_map).astype(int)

    return events.sort_values(
        ["decision_date", "candidate_asset"],
        ignore_index=True,
    )


def _make_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scale",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def _holdout_predictions(
    events: pd.DataFrame,
    features: tuple[str, ...],
    model_name: str,
) -> pd.DataFrame:
    valid = events.dropna(
        subset=["target_positive_multi"]
    ).copy()
    valid["target_positive_multi"] = valid[
        "target_positive_multi"
    ].astype(int)

    folds = sorted(
        int(value)
        for value in valid["walk_forward_fold"].unique()
    )
    if len(folds) < 2:
        return pd.DataFrame()

    outputs = []
    for test_fold in folds[1:]:
        for holdout_group in range(ASSET_HOLDOUT_GROUPS):
            train = valid.loc[
                valid["walk_forward_fold"].lt(test_fold)
                & valid["asset_holdout_group"].ne(holdout_group)
            ].copy()
            test = valid.loc[
                valid["walk_forward_fold"].eq(test_fold)
                & valid["asset_holdout_group"].eq(holdout_group)
            ].copy()

            if train.empty or test.empty:
                continue
            if train["target_positive_multi"].nunique() < 2:
                continue

            usable = [
                feature
                for feature in features
                if feature in train.columns
                and pd.to_numeric(
                    train[feature],
                    errors="coerce",
                ).notna().any()
            ]
            if not usable:
                continue

            x_train = train[usable].apply(
                pd.to_numeric,
                errors="coerce",
            )
            x_test = test[usable].apply(
                pd.to_numeric,
                errors="coerce",
            )
            y_train = train["target_positive_multi"].to_numpy(
                dtype=int
            )

            model = _make_model()
            model.fit(x_train, y_train)
            probability = model.predict_proba(x_test)[:, 1]

            chunk = test[
                [
                    "decision_date",
                    "walk_forward_fold",
                    "candidate_asset",
                    "candidate_role",
                    "asset_holdout_group",
                    "target_positive_multi",
                    "target_delta_utility_multi",
                ]
            ].copy()
            chunk["model_name"] = model_name
            chunk["probability"] = probability
            chunk["train_rows"] = len(train)
            chunk["feature_count"] = len(usable)
            outputs.append(chunk)

    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True)


def _metrics(predictions: pd.DataFrame) -> dict:
    if predictions.empty:
        return {
            "rows": 0,
            "auc": None,
            "average_precision": None,
            "brier": None,
            "positive_rate": None,
        }

    y = predictions["target_positive_multi"].to_numpy(
        dtype=int
    )
    p = predictions["probability"].to_numpy(dtype=float)
    auc = (
        float(roc_auc_score(y, p))
        if len(np.unique(y)) == 2
        else None
    )
    ap = (
        float(average_precision_score(y, p))
        if len(np.unique(y)) == 2
        else None
    )
    return {
        "rows": int(len(predictions)),
        "auc": auc,
        "average_precision": ap,
        "brier": float(brier_score_loss(y, p)),
        "positive_rate": float(np.mean(y)),
    }


def _bootstrap_auc(
    predictions: pd.DataFrame,
    block_column: str,
) -> dict:
    if predictions.empty:
        return {
            "block": block_column,
            "repetitions": 0,
            "lower_95": None,
            "median": None,
            "upper_95": None,
        }

    groups = {
        key: frame
        for key, frame in predictions.groupby(
            block_column,
            sort=False,
        )
    }
    keys = list(groups)
    if len(keys) < 2:
        return {
            "block": block_column,
            "repetitions": 0,
            "lower_95": None,
            "median": None,
            "upper_95": None,
        }

    rng = np.random.default_rng(RANDOM_SEED)
    aucs = []
    for _ in range(BOOTSTRAP_REPETITIONS):
        sampled = rng.choice(
            keys,
            size=len(keys),
            replace=True,
        )
        frame = pd.concat(
            [groups[key] for key in sampled],
            ignore_index=True,
        )
        y = frame["target_positive_multi"].to_numpy(dtype=int)
        if len(np.unique(y)) < 2:
            continue
        p = frame["probability"].to_numpy(dtype=float)
        aucs.append(float(roc_auc_score(y, p)))

    if not aucs:
        return {
            "block": block_column,
            "repetitions": 0,
            "lower_95": None,
            "median": None,
            "upper_95": None,
        }

    values = np.asarray(aucs, dtype=float)
    return {
        "block": block_column,
        "repetitions": int(len(values)),
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _candidate_summary(events: pd.DataFrame) -> pd.DataFrame:
    valid = events.dropna(
        subset=["target_positive_multi"]
    ).copy()
    return (
        valid.groupby(
            ["candidate_asset", "candidate_role"],
            as_index=False,
        )
        .agg(
            events=("target_positive_multi", "size"),
            positive_rate=("target_positive_multi", "mean"),
            median_delta_utility=(
                "target_delta_utility_multi",
                "median",
            ),
            mean_delta_utility=(
                "target_delta_utility_multi",
                "mean",
            ),
            median_gap_vs_best=(
                "gap_vs_best_u67",
                "median",
            ),
        )
        .sort_values(
            ["positive_rate", "median_delta_utility"],
            ascending=[False, False],
            ignore_index=True,
        )
    )


def _role_summary(events: pd.DataFrame) -> pd.DataFrame:
    valid = events.dropna(
        subset=["target_positive_multi"]
    ).copy()
    return (
        valid.groupby(
            "candidate_role",
            as_index=False,
        )
        .agg(
            events=("target_positive_multi", "size"),
            candidates=("candidate_asset", "nunique"),
            positive_rate=("target_positive_multi", "mean"),
            median_delta_utility=(
                "target_delta_utility_multi",
                "median",
            ),
            mean_delta_utility=(
                "target_delta_utility_multi",
                "mean",
            ),
        )
    )


def _plot_candidate_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    ordered = summary.sort_values(
        "positive_rate",
        ascending=True,
    )
    fig, ax = plt.subplots(figsize=(11.0, 8.0))
    ax.barh(
        ordered["candidate_asset"],
        100.0 * ordered["positive_rate"],
    )
    ax.axvline(
        50.0,
        linestyle="--",
        linewidth=1,
    )
    ax.set_xlabel(
        "Substituições com utilidade futura superior à decisão U67 (%)"
    )
    ax.set_ylabel("Candidato congelado")
    ax.set_title(
        "Qualidade futura dos desafios acionáveis por candidato"
    )
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    fig.savefig(
        OUT / "01_taxa_positiva_por_candidato.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()


def _plot_holdout(predictions: pd.DataFrame) -> None:
    if predictions.empty:
        return
    ordered = predictions.sort_values("probability")
    bins = pd.qcut(
        ordered["probability"],
        q=min(10, ordered["probability"].nunique()),
        duplicates="drop",
    )
    calibration = (
        ordered.assign(probability_bin=bins)
        .groupby(
            "probability_bin",
            observed=True,
            as_index=False,
        )
        .agg(
            predicted=("probability", "mean"),
            observed=("target_positive_multi", "mean"),
            events=("target_positive_multi", "size"),
        )
    )

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.plot(
        100.0 * calibration["predicted"],
        100.0 * calibration["observed"],
        marker="o",
    )
    ax.plot(
        [0, 100],
        [0, 100],
        linestyle="--",
        linewidth=1,
    )
    ax.set_xlabel("Probabilidade prevista (%)")
    ax.set_ylabel("Frequência positiva observada (%)")
    ax.set_title(
        "Validação fora de tempo e fora do candidato"
    )
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(
        OUT / "02_calibracao_holdout.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()


def _package() -> Path:
    path = OUT / "pacote_substituicoes_cluster.zip"
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for item in sorted(OUT.iterdir()):
            if item == path or not item.is_file():
                continue
            archive.write(item, arcname=item.name)
    return path


def main() -> None:
    started = time.perf_counter()
    print("=" * 78)
    print("TCC - TESTE DECISIVO DE SUBSTITUIÇÕES DA SHORTLIST")
    print(f"version={SCRIPT_VERSION} schema={ANALYSIS_SCHEMA}")
    print(f"event_definition={EVENT_DEFINITION}")
    print("candidate_capital_backtest=NO")
    print("candidate_selection_change=NO")
    print(
        f"support_rule=AUC>={SUPPORT_AUC_MIN:.2f} "
        f"AND bootstrap_date_lower>{BOOTSTRAP_LOWER_MIN:.2f} "
        f"AND bootstrap_asset_lower>{BOOTSTRAP_LOWER_MIN:.2f}"
    )
    print("=" * 78)

    freeze, shortlist, candidate_assets = _load_contract()
    _u67_payload, fold_margin_frame = _load_u67_reproduction()

    (
        frames_u67,
        candidate_frames,
        u67_symbols,
        candidate_assets,
        config_u67,
        config_all,
        common_dates,
        folds,
        all_decision_dates,
        decision_to_fold,
    ) = _prepare_data(candidate_assets)

    margins = _margin_map(folds, fold_margin_frame)

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    (
        fold_policies,
        _diag_placeholder,
        candidate_cache_by_fold,
    ) = _fit_and_score(
        frames_u67,
        candidate_frames,
        u67_symbols,
        candidate_assets,
        config_u67,
        config_all,
        common_dates,
        folds,
        margins,
    )

    reference_result, baseline = _replay_u67_with_diagnostics(
        fold_policies,
        decision_to_fold,
        frames_u67,
        u67_symbols,
        all_decision_dates,
        config_u67,
        folds,
    )

    sequence = _build_sequence(
        baseline,
        candidate_cache_by_fold,
        candidate_assets,
        shortlist,
    )
    events = _build_events(
        sequence,
        frames_u67,
        candidate_frames,
        config_u67,
    )

    valid_events = events.dropna(
        subset=["target_positive_multi"]
    ).copy()
    event_count = int(len(valid_events))
    candidate_count = int(
        valid_events["candidate_asset"].nunique()
    )

    score_predictions = _holdout_predictions(
        valid_events,
        SCORE_ONLY_FEATURES,
        "score_only",
    )
    full_predictions = _holdout_predictions(
        valid_events,
        CORE_FEATURES,
        "contextual_substitution",
    )

    score_metrics = _metrics(score_predictions)
    full_metrics = _metrics(full_predictions)
    date_bootstrap = _bootstrap_auc(
        full_predictions,
        "decision_date",
    )
    asset_bootstrap = _bootstrap_auc(
        full_predictions,
        "candidate_asset",
    )

    sufficient_sample = bool(
        event_count >= MIN_EVENTS
        and candidate_count >= MIN_CANDIDATES_WITH_EVENTS
        and full_metrics.get("rows", 0) >= MIN_EVENTS // 2
    )

    auc = full_metrics.get("auc")
    date_lower = date_bootstrap.get("lower_95")
    asset_lower = asset_bootstrap.get("lower_95")
    supported = bool(
        sufficient_sample
        and auc is not None
        and auc >= SUPPORT_AUC_MIN
        and date_lower is not None
        and date_lower > BOOTSTRAP_LOWER_MIN
        and asset_lower is not None
        and asset_lower > BOOTSTRAP_LOWER_MIN
    )

    if not sufficient_sample:
        scientific_decision = "INSUFFICIENT_EVENT_SAMPLE_STOP"
    elif supported:
        scientific_decision = "SUPPORTED_ADVANCE_TO_ONE_LOCKED_CAPITAL_REPLAY"
    else:
        scientific_decision = "NOT_SUPPORTED_STOP_CLUSTER_LINE"

    candidate_summary = _candidate_summary(valid_events)
    role_summary = _role_summary(valid_events)

    sequence.to_csv(
        OUT / "candidate_score_sequences.csv.gz",
        index=False,
        compression="gzip",
    )
    events.to_csv(
        OUT / "candidate_substitution_events.csv.gz",
        index=False,
        compression="gzip",
    )
    candidate_summary.to_csv(
        OUT / "candidate_event_summary.csv",
        index=False,
    )
    role_summary.to_csv(
        OUT / "role_event_summary.csv",
        index=False,
    )
    score_predictions.to_csv(
        OUT / "holdout_predictions_score_only.csv",
        index=False,
    )
    full_predictions.to_csv(
        OUT / "holdout_predictions_contextual.csv",
        index=False,
    )
    baseline.to_csv(
        OUT / "u67_reference_predictions_rebuilt.csv",
        index=False,
    )

    _plot_candidate_summary(candidate_summary)
    _plot_holdout(full_predictions)

    payload = {
        "script_version": SCRIPT_VERSION,
        "analysis_schema": ANALYSIS_SCHEMA,
        "status": "completed",
        "shortlist_fingerprint": freeze.get(
            "shortlist_fingerprint"
        ),
        "candidate_count_frozen": len(candidate_assets),
        "candidate_count_with_actionable_events": candidate_count,
        "score_rows": int(len(sequence)),
        "actionable_events": int(len(events)),
        "valid_target_events": event_count,
        "event_definition": EVENT_DEFINITION,
        "primary_target": (
            "candidate forward risk-adjusted utility minus forward "
            "risk-adjusted utility of the action selected by the U67 "
            "reference policy on the same decision date"
        ),
        "secondary_target": (
            "candidate forward risk-adjusted utility minus incumbent "
            "forward risk-adjusted utility"
        ),
        "financial_capital_backtest_performed": False,
        "candidate_selection_changed_after_outcomes": False,
        "validation": {
            "random_row_split_allowed": False,
            "time_holdout": (
                "each test fold is predicted only from earlier folds"
            ),
            "candidate_holdout": (
                "alphabetically assigned candidate groups are absent "
                "from each corresponding training set"
            ),
            "asset_holdout_groups": ASSET_HOLDOUT_GROUPS,
            "score_only": score_metrics,
            "contextual": full_metrics,
            "bootstrap_by_decision_date": date_bootstrap,
            "bootstrap_by_candidate": asset_bootstrap,
        },
        "predeclared_support_rule": {
            "minimum_events": MIN_EVENTS,
            "minimum_candidates_with_events": (
                MIN_CANDIDATES_WITH_EVENTS
            ),
            "minimum_auc": SUPPORT_AUC_MIN,
            "bootstrap_date_lower_must_exceed": (
                BOOTSTRAP_LOWER_MIN
            ),
            "bootstrap_candidate_lower_must_exceed": (
                BOOTSTRAP_LOWER_MIN
            ),
        },
        "sufficient_sample": sufficient_sample,
        "hypothesis_supported": supported,
        "scientific_decision": scientific_decision,
        "stop_rule": (
            "If scientific_decision is NOT_SUPPORTED_STOP_CLUSTER_LINE, "
            "do not tune clustering, PCA, shortlist, K or thresholds from "
            "these outcomes. Close this research line."
        ),
        "u67_reference_ending_capital_validation": float(
            reference_result.metrics[
                "strategy_ending_capital"
            ]
        ),
        "candidate_role_summary": role_summary.to_dict(
            orient="records"
        ),
        "runtime_seconds": float(
            time.perf_counter() - started
        ),
    }

    (
        OUT / "substitution_validation.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    package = _package()

    print("\n" + "=" * 78)
    print("RESULTADO PRÉ-REGISTRADO")
    print(f"events={event_count} candidates={candidate_count}")
    print(
        "score_only_auc="
        f"{score_metrics.get('auc')}"
    )
    print(
        "contextual_auc="
        f"{full_metrics.get('auc')}"
    )
    print(
        "bootstrap_date_95="
        f"[{date_bootstrap.get('lower_95')}, "
        f"{date_bootstrap.get('upper_95')}]"
    )
    print(
        "bootstrap_candidate_95="
        f"[{asset_bootstrap.get('lower_95')}, "
        f"{asset_bootstrap.get('upper_95')}]"
    )
    print(f"scientific_decision={scientific_decision}")
    print(f"package={package}")
    print("=" * 78)


if __name__ == "__main__":
    main()
