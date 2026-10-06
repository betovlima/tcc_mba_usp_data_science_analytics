"""Gera a base de eventos candidato -> incumbente para o baseline U67.

Esta etapa foi criada para estudar o mecanismo de substituicao ANTES de
introduzir Deep Learning.

Ela NAO executa replay financeiro completo e NAO baixa dados da Alpaca.
O trabalho pesado e somente uma passagem de score LightGBM dos candidatos
contra as mesmas datas walk-forward do U67. Os scores sao cacheados para que
as proximas analises nao precisem treinar novamente.

A unidade primaria deixa de ser "ativo" e passa a ser:

    (candidato, incumbente, data de decisao)

Um evento de substituicao ocorre quando o score do candidato supera o score do
incumbente U67 naquela data. O resultado futuro e gravado em colunas target_*
e nunca deve ser usado como feature no ajuste de modelos.

Saidas principais:
- candidate_score_sequences.csv.gz
- candidate_substitution_events.csv.gz
- candidate_event_summary.csv
- substitution_strength_bins.csv
- event_analysis.json
- pacote_eventos_substituicao_u67.zip
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import time
import zipfile

import matplotlib

if (
    os.name != "nt"
    and not os.environ.get("DISPLAY")
    and not os.environ.get("WAYLAND_DISPLAY")
):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine.configuracao import CONFIG
from engine.modelo_lightgbm import (
    _ajustar_modelos_lightgbm,
    _construir_contexto_execucao,
)
from engine.rotacao import (
    _custo_troca_proporcional,
    _precalcular_utilidades_modelo,
    preparar_painel_rotacao,
)
from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.experimento import build_variant_configs
from reproducao.preparacao import prepare_model_frames


ROOT = Path(__file__).resolve().parents[1]
BASE = SnapshotPaths.research(ROOT)
B2 = SnapshotPaths.from_root(
    ROOT / "dados" / "pesquisa_expansao_76_b2"
)
SMART = SnapshotPaths.from_root(
    ROOT / "dados" / "pesquisa_smart_candidates"
)

RANKED_FILE = (
    ROOT
    / "output"
    / "busca_ativos"
    / "intelligent_candidates_ranked.csv"
)
U67_PREDICTIONS_FILE = (
    ROOT / "output" / "reproducao" / "u67_predictions.csv"
)
U67_RESULT_FILE = (
    ROOT / "output" / "reproducao" / "reproducao_u67.json"
)

OUT = ROOT / "output" / "eventos_substituicao_u67"
SCORE_CACHE = OUT / "candidate_score_sequences.csv.gz"
CACHE_META = OUT / "score_cache_metadata.json"

ANALYSIS_VERSION = "1.0.0-dev.1"
ANALYSIS_SCHEMA = "u67-candidate-substitution-events-v1"
SCORE_CACHE_SCHEMA = "u67-candidate-score-sequences-v1"
SHOW_PLOTS = True

EXPECTED_U67_CAPITAL = 58_557_157.67496595
U59_ADDITIONS = ("COLB", "AMS", "FOXF")
POSITIVE8 = (
    "THO",
    "WDAY",
    "EXR",
    "XEL",
    "SBFG",
    "PAYX",
    "MUX",
    "SXC",
)

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

BASELINE_COLUMNS = (
    "decision_date",
    "timestamp",
    "walk_forward_fold",
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
    "spy_return_5",
    "spy_return_20",
    "spy_realized_volatility_20",
    "universe_breadth_5",
    "universe_breadth_20",
    "position_return_since_entry",
    "position_drawdown_from_peak",
    "position_mfe_so_far",
    "position_mae_so_far",
    "days_current_not_top1",
    "consecutive_days_current_not_top1",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_bucket(asset: str, buckets: int = 5) -> int:
    value = hashlib.sha256(
        str(asset).encode("utf-8")
    ).hexdigest()
    return int(value[:8], 16) % buckets


def _save_figure(fig, stem: str) -> dict[str, str]:
    png = OUT / f"{stem}.png"
    svg = OUT / f"{stem}.svg"
    fig.savefig(
        png,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        svg,
        bbox_inches="tight",
        facecolor="white",
    )
    if SHOW_PLOTS:
        fig.canvas.draw_idle()
        plt.show(block=False)
    else:
        plt.close(fig)
    return {
        "png": str(png.relative_to(ROOT)),
        "svg": str(svg.relative_to(ROOT)),
    }


def _validate_u67_result() -> dict:
    if not U67_RESULT_FILE.exists():
        raise RuntimeError(
            "reproducao_u67.json ausente. Execute o U67 congelado uma vez "
            "antes desta fase."
        )
    payload = json.loads(
        U67_RESULT_FILE.read_text(encoding="utf-8")
    )
    observed = float(payload.get("observed_ending_capital"))
    if payload.get("status") != "reproduced":
        raise RuntimeError(
            "O U67 atual nao esta marcado como reproduced."
        )
    if not math.isclose(
        observed,
        EXPECTED_U67_CAPITAL,
        rel_tol=1e-9,
        abs_tol=0.01,
    ):
        raise RuntimeError(
            "Checkpoint U67 divergente. "
            f"observado={observed:,.8f} "
            f"esperado={EXPECTED_U67_CAPITAL:,.8f}"
        )
    return payload


def _load_baseline_predictions() -> pd.DataFrame:
    if not U67_PREDICTIONS_FILE.exists():
        raise RuntimeError(
            "u67_predictions.csv ausente em output/reproducao."
        )
    frame = pd.read_csv(U67_PREDICTIONS_FILE)
    missing = [
        column
        for column in BASELINE_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise RuntimeError(
            "u67_predictions.csv sem colunas necessarias: "
            + ",".join(missing)
        )

    frame = frame.loc[:, BASELINE_COLUMNS].copy()
    frame["decision_date"] = pd.to_datetime(
        frame["decision_date"],
        utc=True,
    )
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
    )
    frame["current_asset"] = (
        frame["current_asset"]
        .fillna("CASH")
        .astype(str)
        .str.upper()
    )
    frame["best_asset"] = (
        frame["best_asset"]
        .fillna("CASH")
        .astype(str)
        .str.upper()
    )

    numeric_columns = [
        column
        for column in frame.columns
        if column not in {
            "decision_date",
            "timestamp",
            "current_asset",
            "best_asset",
            "second_asset",
        }
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    if frame["decision_date"].duplicated().any():
        raise RuntimeError(
            "u67_predictions.csv possui decision_date duplicada."
        )

    return frame.sort_values(
        "decision_date",
        ignore_index=True,
    )


def _load_ranked_assets() -> tuple[str, ...]:
    if not RANKED_FILE.exists():
        raise RuntimeError(
            "intelligent_candidates_ranked.csv ausente. "
            "Esta fase nao refaz a busca; ela usa o ranking ja produzido."
        )
    ranked = pd.read_csv(RANKED_FILE)
    if "asset" not in ranked.columns:
        raise RuntimeError(
            "intelligent_candidates_ranked.csv sem coluna asset."
        )
    assets = tuple(
        sorted(
            set(
                ranked["asset"]
                .astype(str)
                .str.strip()
                .str.upper()
            )
        )
    )
    if not assets:
        raise RuntimeError("Ranking de candidatos vazio.")
    return assets


def _prepare_frames(
    ranked_assets: tuple[str, ...],
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    list[str],
    list[dict],
    object,
    dict,
]:
    frames_u56, exclusions_u56, _, _ = prepare_model_frames(
        BASE,
        assets=CONFIG.assets,
        comparar_snapshot_referencia=False,
        allow_structural_assets=frozenset({"CLMT", "DOC"}),
    )
    if len(frames_u56) != 56:
        raise RuntimeError(
            f"U56 esperado=56 observado={len(frames_u56)} "
            f"exclusoes={exclusions_u56}"
        )

    frames_b2, exclusions_b2, _, _ = prepare_model_frames(
        B2,
        assets=U59_ADDITIONS,
        comparar_snapshot_referencia=False,
    )
    if exclusions_b2 or len(frames_b2) != 3:
        raise RuntimeError(
            "Nao foi possivel reconstruir U59 integralmente."
        )

    frames_positive8, exclusions_positive8, _, _ = (
        prepare_model_frames(
            SMART,
            assets=POSITIVE8,
            comparar_snapshot_referencia=False,
        )
    )
    if exclusions_positive8 or len(frames_positive8) != 8:
        raise RuntimeError(
            "Os oito ativos do baseline U67 nao estao integralmente "
            "disponiveis no snapshot SMART."
        )

    raw_u67 = {
        **frames_u56,
        **frames_b2,
        **frames_positive8,
    }
    if len(raw_u67) != 67:
        raise RuntimeError(
            f"U67 esperado=67 observado={len(raw_u67)}"
        )

    candidate_assets = tuple(
        asset
        for asset in ranked_assets
        if asset not in raw_u67
    )
    frames_candidates, candidate_exclusions, _, _ = (
        prepare_model_frames(
            SMART,
            assets=candidate_assets,
            comparar_snapshot_referencia=False,
        )
    )
    candidate_symbols_raw = sorted(frames_candidates)
    if not candidate_symbols_raw:
        raise RuntimeError(
            "Nenhum candidato modelavel disponivel."
        )

    config_u56, _ = build_variant_configs(
        frames_u56,
        CONFIG,
    )
    _, reference_calendar, reference_source = (
        preparar_painel_rotacao(
            frames_u56,
            config_u56,
        )
    )

    all_raw = {
        **raw_u67,
        **frames_candidates,
    }
    config_all, _ = build_variant_configs(
        all_raw,
        CONFIG,
    )

    (
        frames_all,
        common_dates,
        calendar_source,
        symbols_all,
        folds,
        _,
        decision_to_fold,
        _,
    ) = _construir_contexto_execucao(
        all_raw,
        config_all,
        calendar_override=reference_calendar,
        calendar_source_label=(
            f"U56_FIXED:{reference_source}"
        ),
    )

    candidate_symbols = [
        symbol
        for symbol in candidate_symbols_raw
        if symbol in frames_all
    ]
    frames_u67 = {
        symbol: frames_all[symbol]
        for symbol in sorted(raw_u67)
    }

    context = {
        "common_dates": common_dates,
        "calendar_source": calendar_source,
        "symbols_all": symbols_all,
        "decision_to_fold": decision_to_fold,
        "candidate_exclusions": candidate_exclusions,
    }
    return (
        frames_u67,
        frames_all,
        candidate_symbols,
        folds,
        config_all,
        context,
    )


def _cache_fingerprint(
    manifests: dict,
) -> dict:
    return {
        "score_cache_schema": SCORE_CACHE_SCHEMA,
        "base_snapshot_sha256": (
            manifests["base"].get("snapshot_sha256")
        ),
        "b2_snapshot_sha256": (
            manifests["b2"].get("snapshot_sha256")
        ),
        "smart_snapshot_sha256": (
            manifests["smart"].get("snapshot_sha256")
        ),
        "ranked_sha256": _sha256(RANKED_FILE),
        "u67_predictions_sha256": _sha256(
            U67_PREDICTIONS_FILE
        ),
    }


def _cache_is_valid(fingerprint: dict) -> bool:
    if not SCORE_CACHE.exists() or not CACHE_META.exists():
        return False
    try:
        saved = json.loads(
            CACHE_META.read_text(encoding="utf-8")
        )
    except Exception:
        return False
    return saved.get("fingerprint") == fingerprint


def _score_candidates(
    frames_all: dict[str, pd.DataFrame],
    candidate_symbols: list[str],
    folds: list[dict],
    config_all,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    baseline_by_date = baseline.set_index("decision_date")
    chunks = []

    for fold_position, fold in enumerate(folds, start=1):
        fold_id = int(fold["fold_id"])
        fit_dates = pd.DatetimeIndex(
            frame_date
            for frame_date in (
                frames_all[candidate_symbols[0]].index[
                    : int(fold["final_fit_end_index"])
                ]
            )
        )
        decision_dates = pd.DatetimeIndex(
            fold["decision_dates"][:-1]
        )
        decision_dates = decision_dates.intersection(
            baseline_by_date.index[
                baseline_by_date[
                    "walk_forward_fold"
                ].astype("Int64") == fold_id
            ]
        )

        print(
            f"[score] fold={fold_id} "
            f"{fold_position}/{len(folds)} "
            f"candidates={len(candidate_symbols)} "
            f"dates={len(decision_dates)}",
            flush=True,
        )

        models = _ajustar_modelos_lightgbm(
            frames_all,
            candidate_symbols,
            fit_dates,
            config_all,
            phase=(
                f"u67_substitution_fold_{fold_id}_final"
            ),
            technical_log_callback=lambda message: print(
                f"[technical] {message}",
                flush=True,
            ),
        )
        cache, _ = _precalcular_utilidades_modelo(
            models,
            frames_all,
            candidate_symbols,
            decision_dates,
            config_all,
        )

        fold_baseline = baseline_by_date.loc[
            decision_dates
        ].copy()

        rows = []
        for timestamp in decision_dates:
            utilities = cache.get(pd.Timestamp(timestamp))
            if utilities is None:
                continue
            values = np.asarray(utilities, dtype=float)
            base_row = fold_baseline.loc[
                pd.Timestamp(timestamp)
            ]

            for position, asset in enumerate(
                candidate_symbols,
                start=1,
            ):
                if position >= len(values):
                    continue
                score = float(values[position])
                if not np.isfinite(score):
                    continue

                current_score = float(
                    base_row["current_score"]
                )
                best_score = float(base_row["best_score"])
                margin = float(
                    base_row["effective_switch_margin"]
                )
                if not np.isfinite(current_score):
                    current_score = 0.0
                if not np.isfinite(best_score):
                    best_score = 0.0
                if not np.isfinite(margin) or margin <= 0:
                    margin = float(
                        config_all.rotation_switch_margin
                    )

                gap_current = score - current_score
                gap_best = score - best_score

                top_scores = [
                    pd.to_numeric(
                        base_row.get("best_score"),
                        errors="coerce",
                    ),
                    pd.to_numeric(
                        base_row.get("second_score"),
                        errors="coerce",
                    ),
                ]
                finite_top = [
                    float(value)
                    for value in top_scores
                    if pd.notna(value)
                ]
                rank_bucket = (
                    1
                    + sum(
                        value > score
                        for value in finite_top
                    )
                )
                if rank_bucket > 2:
                    rank_bucket = 3

                rows.append(
                    {
                        "decision_date": timestamp,
                        "execution_date": base_row[
                            "timestamp"
                        ],
                        "walk_forward_fold": fold_id,
                        "candidate_asset": asset,
                        "asset_holdout_bucket": (
                            _asset_bucket(asset)
                        ),
                        "candidate_score": score,
                        "incumbent_asset": base_row[
                            "current_asset"
                        ],
                        "incumbent_score": current_score,
                        "best_u67_asset": base_row[
                            "best_asset"
                        ],
                        "best_u67_score": best_score,
                        "effective_switch_margin": margin,
                        "gap_vs_incumbent": gap_current,
                        "gap_vs_best_u67": gap_best,
                        "strength_vs_margin": (
                            gap_current / margin
                        ),
                        "beats_incumbent": (
                            gap_current > 0.0
                        ),
                        "clears_switch_margin": (
                            gap_current >= margin
                        ),
                        "beats_best_u67": (
                            gap_best > 0.0
                        ),
                        "top_rank_bucket": rank_bucket,
                        "holding_days_at_decision": (
                            base_row[
                                "holding_days_at_decision"
                            ]
                        ),
                        "universe_score_mean": base_row[
                            "universe_score_mean"
                        ],
                        "universe_score_std": base_row[
                            "universe_score_std"
                        ],
                        "candidate_score_zscore": (
                            (
                                score
                                - float(
                                    base_row[
                                        "universe_score_mean"
                                    ]
                                )
                            )
                            / float(
                                base_row[
                                    "universe_score_std"
                                ]
                            )
                            if (
                                pd.notna(
                                    base_row[
                                        "universe_score_std"
                                    ]
                                )
                                and float(
                                    base_row[
                                        "universe_score_std"
                                    ]
                                )
                                > 0
                            )
                            else np.nan
                        ),
                        "spy_return_5": base_row[
                            "spy_return_5"
                        ],
                        "spy_return_20": base_row[
                            "spy_return_20"
                        ],
                        "spy_realized_volatility_20": (
                            base_row[
                                "spy_realized_volatility_20"
                            ]
                        ),
                        "universe_breadth_5": base_row[
                            "universe_breadth_5"
                        ],
                        "universe_breadth_20": base_row[
                            "universe_breadth_20"
                        ],
                    }
                )

        chunks.append(pd.DataFrame(rows))

    if not chunks:
        raise RuntimeError(
            "Nenhum score de candidato foi produzido."
        )

    sequence = pd.concat(
        chunks,
        ignore_index=True,
    )
    sequence = sequence.sort_values(
        ["candidate_asset", "decision_date"],
        ignore_index=True,
    )

    grouped = sequence.groupby(
        "candidate_asset",
        sort=False,
    )
    for window in (5, 20, 60):
        sequence[
            f"score_mean_{window}"
        ] = grouped["candidate_score"].transform(
            lambda values, w=window: values.rolling(
                w,
                min_periods=max(2, w // 4),
            ).mean()
        )
        sequence[
            f"score_std_{window}"
        ] = grouped["candidate_score"].transform(
            lambda values, w=window: values.rolling(
                w,
                min_periods=max(2, w // 4),
            ).std(ddof=0)
        )
        sequence[
            f"positive_share_{window}"
        ] = grouped["candidate_score"].transform(
            lambda values, w=window: (
                values.gt(0.0)
                .astype(float)
                .rolling(
                    w,
                    min_periods=max(2, w // 4),
                )
                .mean()
            )
        )
        sequence[
            f"beats_best_share_{window}"
        ] = grouped["beats_best_u67"].transform(
            lambda values, w=window: (
                values.astype(float)
                .rolling(
                    w,
                    min_periods=max(2, w // 4),
                )
                .mean()
            )
        )
        sequence[
            f"beats_incumbent_share_{window}"
        ] = grouped["beats_incumbent"].transform(
            lambda values, w=window: (
                values.astype(float)
                .rolling(
                    w,
                    min_periods=max(2, w // 4),
                )
                .mean()
            )
        )

    for lag in (1, 5, 20):
        sequence[
            f"score_change_{lag}"
        ] = grouped["candidate_score"].diff(lag)

    return sequence


def _load_or_score(
    fingerprint: dict,
    frames_all: dict[str, pd.DataFrame],
    candidate_symbols: list[str],
    folds: list[dict],
    config_all,
    baseline: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    if _cache_is_valid(fingerprint):
        print(
            "[cache] candidate score sequences reutilizadas",
            flush=True,
        )
        sequence = pd.read_csv(
            SCORE_CACHE,
            compression="gzip",
        )
        sequence["decision_date"] = pd.to_datetime(
            sequence["decision_date"],
            utc=True,
        )
        sequence["execution_date"] = pd.to_datetime(
            sequence["execution_date"],
            utc=True,
        )
        return sequence, True

    sequence = _score_candidates(
        frames_all,
        candidate_symbols,
        folds,
        config_all,
        baseline,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    sequence.to_csv(
        SCORE_CACHE,
        index=False,
        compression="gzip",
    )
    CACHE_META.write_text(
        json.dumps(
            {
                "analysis_version": ANALYSIS_VERSION,
                "fingerprint": fingerprint,
                "candidate_count": int(
                    sequence["candidate_asset"].nunique()
                ),
                "rows": int(len(sequence)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return sequence, False


def _safe_frame_value(
    frame: pd.DataFrame,
    timestamp: pd.Timestamp,
    column: str,
) -> float:
    if column not in frame.columns:
        return float("nan")
    try:
        value = frame.at[timestamp, column]
    except KeyError:
        return float("nan")
    value = pd.to_numeric(value, errors="coerce")
    return float(value) if pd.notna(value) else float("nan")


def _future_target(
    candidate_frame: pd.DataFrame,
    incumbent_frame: pd.DataFrame,
    timestamp: pd.Timestamp,
    switch_log_penalty: float,
) -> dict[str, float]:
    output: dict[str, float] = {}

    candidate_total = _safe_frame_value(
        candidate_frame,
        timestamp,
        "forward_risk_adjusted_utility",
    )
    incumbent_total = _safe_frame_value(
        incumbent_frame,
        timestamp,
        "forward_risk_adjusted_utility",
    )
    output["target_delta_utility_multi"] = (
        candidate_total
        - incumbent_total
        + switch_log_penalty
        if (
            np.isfinite(candidate_total)
            and np.isfinite(incumbent_total)
        )
        else np.nan
    )

    candidate_log = _safe_frame_value(
        candidate_frame,
        timestamp,
        "forward_net_log_return",
    )
    incumbent_log = _safe_frame_value(
        incumbent_frame,
        timestamp,
        "forward_net_log_return",
    )
    output["target_delta_net_log_return_multi"] = (
        candidate_log
        - incumbent_log
        + switch_log_penalty
        if np.isfinite(candidate_log) and np.isfinite(incumbent_log)
        else np.nan
    )

    for horizon in HORIZONS:
        candidate_utility = _safe_frame_value(
            candidate_frame,
            timestamp,
            f"forward_horizon_utility_{horizon}",
        )
        incumbent_utility = _safe_frame_value(
            incumbent_frame,
            timestamp,
            f"forward_horizon_utility_{horizon}",
        )
        output[
            f"target_delta_utility_{horizon}"
        ] = (
            candidate_utility
            - incumbent_utility
            + switch_log_penalty
            if (
                np.isfinite(candidate_utility)
                and np.isfinite(incumbent_utility)
            )
            else np.nan
        )

    output["target_positive_multi"] = (
        float(output["target_delta_utility_multi"] > 0.0)
        if np.isfinite(
            output["target_delta_utility_multi"]
        )
        else np.nan
    )
    return output


def _build_events(
    sequence: pd.DataFrame,
    frames_all: dict[str, pd.DataFrame],
    frames_u67: dict[str, pd.DataFrame],
    config_all,
) -> pd.DataFrame:
    challenge = sequence.loc[
        sequence["beats_incumbent"].astype(bool)
        & sequence["incumbent_asset"].ne("CASH")
    ].copy()

    switch_cost = _custo_troca_proporcional(
        config_all,
        1,
        2,
    )
    switch_log_penalty = math.log(
        max(1e-12, 1.0 - switch_cost)
    )

    rows = []
    total = len(challenge)
    for count, row in enumerate(
        challenge.itertuples(index=False),
        start=1,
    ):
        if count % 5000 == 0 or count == total:
            print(
                f"[events] {count}/{total}",
                flush=True,
            )

        candidate = str(row.candidate_asset)
        incumbent = str(row.incumbent_asset)
        timestamp = pd.Timestamp(row.decision_date)

        candidate_frame = frames_all.get(candidate)
        incumbent_frame = frames_u67.get(incumbent)
        if candidate_frame is None or incumbent_frame is None:
            continue

        record = row._asdict()

        for feature in RELATIVE_TECHNICAL_FEATURES:
            candidate_value = _safe_frame_value(
                candidate_frame,
                timestamp,
                feature,
            )
            incumbent_value = _safe_frame_value(
                incumbent_frame,
                timestamp,
                feature,
            )
            record[f"candidate_{feature}"] = candidate_value
            record[f"incumbent_{feature}"] = incumbent_value
            record[f"delta_{feature}"] = (
                candidate_value - incumbent_value
                if (
                    np.isfinite(candidate_value)
                    and np.isfinite(incumbent_value)
                )
                else np.nan
            )

        record.update(
            _future_target(
                candidate_frame,
                incumbent_frame,
                timestamp,
                switch_log_penalty,
            )
        )
        rows.append(record)

    events = pd.DataFrame(rows)
    if events.empty:
        raise RuntimeError(
            "Nenhum evento candidato > incumbente foi encontrado."
        )

    return events.sort_values(
        ["decision_date", "candidate_asset"],
        ignore_index=True,
    )


def _strength_bins(
    events: pd.DataFrame,
) -> pd.DataFrame:
    valid = events.dropna(
        subset=[
            "strength_vs_margin",
            "target_delta_utility_multi",
        ]
    ).copy()
    if valid.empty:
        return pd.DataFrame()

    unique_values = valid[
        "strength_vs_margin"
    ].nunique()
    q = min(10, int(unique_values))
    if q < 2:
        valid["strength_bin"] = "all"
    else:
        valid["strength_bin"] = pd.qcut(
            valid["strength_vs_margin"],
            q=q,
            duplicates="drop",
        ).astype(str)

    summary = (
        valid.groupby(
            "strength_bin",
            observed=True,
            as_index=False,
        )
        .agg(
            events=(
                "target_delta_utility_multi",
                "size",
            ),
            strength_median=(
                "strength_vs_margin",
                "median",
            ),
            positive_rate=(
                "target_positive_multi",
                "mean",
            ),
            target_median=(
                "target_delta_utility_multi",
                "median",
            ),
            target_mean=(
                "target_delta_utility_multi",
                "mean",
            ),
        )
        .sort_values(
            "strength_median",
            ignore_index=True,
        )
    )
    return summary


def _candidate_summary(
    sequence: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    total_sessions = (
        sequence.groupby("candidate_asset")
        .size()
        .rename("score_sessions")
    )
    challenges = (
        events.groupby("candidate_asset")
        .size()
        .rename("challenge_events")
    )
    positives = (
        events.groupby("candidate_asset")[
            "target_positive_multi"
        ]
        .mean()
        .rename("future_positive_rate")
    )
    median_target = (
        events.groupby("candidate_asset")[
            "target_delta_utility_multi"
        ]
        .median()
        .rename("median_target_delta_utility")
    )
    median_strength = (
        events.groupby("candidate_asset")[
            "strength_vs_margin"
        ]
        .median()
        .rename("median_strength_vs_margin")
    )
    clears = (
        sequence.groupby("candidate_asset")[
            "clears_switch_margin"
        ]
        .mean()
        .rename("clear_margin_share")
    )
    beats_best = (
        sequence.groupby("candidate_asset")[
            "beats_best_u67"
        ]
        .mean()
        .rename("beats_best_u67_share")
    )

    result = pd.concat(
        [
            total_sessions,
            challenges,
            positives,
            median_target,
            median_strength,
            clears,
            beats_best,
        ],
        axis=1,
    ).fillna(
        {
            "challenge_events": 0,
            "future_positive_rate": np.nan,
            "median_target_delta_utility": np.nan,
            "median_strength_vs_margin": np.nan,
        }
    )
    result["challenge_events"] = (
        result["challenge_events"].astype(int)
    )
    result["challenge_share"] = (
        result["challenge_events"]
        / result["score_sessions"].clip(lower=1)
    )
    return (
        result.reset_index()
        .sort_values(
            [
                "challenge_share",
                "median_target_delta_utility",
                "candidate_asset",
            ],
            ascending=[True, False, True],
            ignore_index=True,
        )
    )


def _plot_strength_positive_rate(
    bins: pd.DataFrame,
) -> dict[str, str] | None:
    if bins.empty:
        return None
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.plot(
        bins["strength_median"],
        100.0 * bins["positive_rate"],
        marker="o",
    )
    ax.axhline(
        50.0,
        linewidth=1.0,
        linestyle="--",
        color="#666666",
    )
    ax.set_xlabel(
        "Forca mediana do desafio "
        "(gap candidato-incumbente / margem)"
    )
    ax.set_ylabel("Eventos com utilidade futura positiva (%)")
    ax.set_title(
        "Calibracao observada da forca de substituicao",
        loc="left",
        pad=14,
        fontweight="bold",
    )
    ax.grid(alpha=0.2)
    return _save_figure(
        fig,
        "01_calibracao_forca_substituicao",
    )


def _plot_frequency_vs_target(
    summary: pd.DataFrame,
) -> dict[str, str] | None:
    valid = summary.dropna(
        subset=["median_target_delta_utility"]
    ).copy()
    if valid.empty:
        return None

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.scatter(
        100.0 * valid["challenge_share"],
        valid["median_target_delta_utility"],
        s=34,
        alpha=0.65,
    )
    ax.axhline(
        0.0,
        linewidth=1.0,
        color="#666666",
    )
    ax.set_xlabel(
        "Sessoes em que o candidato supera o incumbente (%)"
    )
    ax.set_ylabel(
        "Mediana da utilidade futura relativa"
    )
    ax.set_title(
        "Frequencia de desafio versus qualidade futura",
        loc="left",
        pad=14,
        fontweight="bold",
    )
    ax.grid(alpha=0.2)
    return _save_figure(
        fig,
        "02_frequencia_vs_utilidade",
    )


def _package_output() -> Path:
    package = OUT / "pacote_eventos_substituicao_u67.zip"
    with zipfile.ZipFile(
        package,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for file_path in sorted(OUT.iterdir()):
            if (
                file_path == package
                or not file_path.is_file()
            ):
                continue
            archive.write(
                file_path,
                arcname=file_path.name,
            )
    return package


def main() -> None:
    started = time.perf_counter()

    print("=" * 78, flush=True)
    print(
        "TCC - EVENTOS DE SUBSTITUICAO CONTRA U67",
        flush=True,
    )
    print(
        f"version={ANALYSIS_VERSION} "
        f"schema={ANALYSIS_SCHEMA}",
        flush=True,
    )
    print("financial_replay=NO", flush=True)
    print("alpaca_download=NO", flush=True)
    print(
        "lightgbm=ONE_SCORE_PASS_WITH_CACHE",
        flush=True,
    )
    print("=" * 78, flush=True)

    u67_payload = _validate_u67_result()
    baseline = _load_baseline_predictions()
    ranked_assets = _load_ranked_assets()

    manifests = {
        "base": validate_snapshot(BASE),
        "b2": validate_snapshot(B2),
        "smart": validate_snapshot(SMART),
    }
    fingerprint = _cache_fingerprint(manifests)

    (
        frames_u67,
        frames_all,
        candidate_symbols,
        folds,
        config_all,
        context,
    ) = _prepare_frames(ranked_assets)

    if OUT.exists() and not _cache_is_valid(fingerprint):
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    sequence, reused_cache = _load_or_score(
        fingerprint,
        frames_all,
        candidate_symbols,
        folds,
        config_all,
        baseline,
    )

    events = _build_events(
        sequence,
        frames_all,
        frames_u67,
        config_all,
    )
    summary = _candidate_summary(
        sequence,
        events,
    )
    bins = _strength_bins(events)

    events_file = OUT / "candidate_substitution_events.csv.gz"
    events.to_csv(
        events_file,
        index=False,
        compression="gzip",
    )
    summary.to_csv(
        OUT / "candidate_event_summary.csv",
        index=False,
    )
    bins.to_csv(
        OUT / "substitution_strength_bins.csv",
        index=False,
    )

    target_columns = sorted(
        column
        for column in events.columns
        if column.startswith("target_")
    )
    feature_columns = sorted(
        column
        for column in events.columns
        if (
            column not in target_columns
            and column not in {
                "decision_date",
                "execution_date",
                "candidate_asset",
                "incumbent_asset",
                "best_u67_asset",
            }
        )
    )

    graphs = {
        "strength_calibration": (
            _plot_strength_positive_rate(bins)
        ),
        "frequency_vs_target": (
            _plot_frequency_vs_target(summary)
        ),
    }

    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "analysis_schema": ANALYSIS_SCHEMA,
        "baseline": {
            "name": "U67_WORKING_BASELINE",
            "expected_capital": EXPECTED_U67_CAPITAL,
            "observed_capital": float(
                u67_payload[
                    "observed_ending_capital"
                ]
            ),
            "scientific_status": (
                "exploratory_working_baseline"
            ),
        },
        "input_fingerprint": fingerprint,
        "score_cache_reused": bool(reused_cache),
        "candidate_count": int(
            sequence["candidate_asset"].nunique()
        ),
        "score_rows": int(len(sequence)),
        "substitution_events": int(len(events)),
        "event_definition": (
            "candidate_score > incumbent_score and "
            "incumbent_asset != CASH"
        ),
        "target_definition": (
            "candidate forward risk-adjusted utility minus incumbent "
            "forward risk-adjusted utility plus log(1-switch_cost). "
            "Targets use future data and are forbidden as model features."
        ),
        "relative_technical_features": list(
            RELATIVE_TECHNICAL_FEATURES
        ),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "validation_design_for_next_phase": {
            "time_axis": "walk_forward_fold",
            "asset_axis": "asset_holdout_bucket",
            "random_row_split_allowed": False,
        },
        "calendar_source": context[
            "calendar_source"
        ],
        "candidate_exclusions": context[
            "candidate_exclusions"
        ],
        "graphs": graphs,
        "runtime_seconds": float(
            time.perf_counter() - started
        ),
    }

    with (
        OUT / "event_analysis.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    package = _package_output()

    print(
        f"[done] candidates={payload['candidate_count']} "
        f"score_rows={len(sequence)} "
        f"events={len(events)} "
        f"cache_reused={reused_cache}",
        flush=True,
    )
    print(f"[done] package={package}", flush=True)


if __name__ == "__main__":
    main()
