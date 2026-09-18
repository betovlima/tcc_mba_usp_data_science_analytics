"""Meta-ranker contextual sobre os episodios contrafactuais Tiingo v2.

Versao: tiingo-contextual-meta-ranker-v1.0.0

Objetivo
--------
Testar, de forma cronologica e sem reusar o proprio futuro, se o delta
contrafactual por candidato pode ser aprendido antes da decisao.

Entrada principal:
  output/tiingo_forced_candidate_episode_advantage_v2/
    forced_candidate_episode_advantage_all_oos.csv
    baseline_score_matrix.csv

Metodologia
-----------
- somente linhas forced_action_changed=True e target_censored=False;
- features: somente informacoes conhecidas no instante t;
- contexto do baseline enriquecido com estado da politica, regime de mercado,
  dispersao cross-sectional dos scores e desempenho da estrategia ate t;
- validacao expanding-window anual;
- uma linha de treino so e elegivel quando seu reconvergence_date ocorreu antes
  do inicio do ano de teste;
- compara dois objetivos fixos, sem tuning no OOS:
    1) regressao L1 do episode_delta_log_capital;
    2) LightGBM LambdaRank com relevancia por quintil dentro de cada data;
- baseline "nao intervir" tem delta=0 e e usado como referencia.

Esta pesquisa NAO altera a estrategia nem calcula capital de portfolio final.
Ela mede capacidade preditiva/ordenacao de intervencoes one-shot, que e a
condicao necessaria antes de integrar um seletor ao backtest.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from lightgbm import LGBMRanker, LGBMRegressor

from tcc_engine.config import ASSETS, CONFIG

VERSION = "tiingo-contextual-meta-ranker-v1.0.0"
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "output" / "tiingo_forced_candidate_episode_advantage_v2"
OUT = ROOT / "output" / "tiingo_contextual_meta_ranker_v1"

EPISODES_PATH = SOURCE / "forced_candidate_episode_advantage_all_oos.csv"
BASELINE_PATH = SOURCE / "baseline_score_matrix.csv"
SOURCE_MANIFEST_PATH = SOURCE / "experiment_manifest.json"

FOLD_METRICS_PATH = OUT / "fold_metrics.csv"
PREDICTIONS_PATH = OUT / "oos_meta_predictions.csv"
IMPORTANCE_PATH = OUT / "feature_importance.csv"
FEATURES_PATH = OUT / "feature_contract.json"
SUMMARY_PATH = OUT / "summary.json"

FIRST_TEST_YEAR = 2022


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    for path in (EPISODES_PATH, BASELINE_PATH, SOURCE_MANIFEST_PATH):
        if not path.exists():
            raise RuntimeError(
                f"Arquivo ausente: {path}. Execute primeiro "
                "pesquisar_vantagem_forcada_candidato_tiingo_v2.py"
            )

    episodes = pd.read_csv(EPISODES_PATH)
    baseline = pd.read_csv(BASELINE_PATH)
    manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))

    episodes["decision_date"] = pd.to_datetime(episodes["decision_date"], utc=True)
    episodes["reconvergence_date"] = pd.to_datetime(
        episodes["reconvergence_date"], utc=True, errors="coerce"
    )
    baseline["decision_date"] = pd.to_datetime(baseline["decision_date"], utc=True)
    return episodes, baseline, manifest


def finite_scores(raw: str) -> np.ndarray:
    try:
        payload = json.loads(raw)
    except Exception:
        return np.asarray([], dtype=float)
    values = []
    for value in payload.values():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            values.append(number)
    return np.asarray(values, dtype=float)


def enrich_baseline_context(baseline: pd.DataFrame) -> pd.DataFrame:
    frame = baseline.sort_values("decision_date").reset_index(drop=True).copy()

    # strategy_equity desta linha ja contem a execucao posterior a decision_date.
    # Portanto, o estado de capital conhecido em t e a equity da linha anterior.
    frame["strategy_equity_before"] = pd.to_numeric(
        frame["strategy_equity"], errors="coerce"
    ).shift(1)
    frame.loc[0, "strategy_equity_before"] = float(CONFIG.initial_capital)

    for horizon in (5, 20, 60):
        frame[f"strategy_return_{horizon}"] = (
            frame["strategy_equity_before"].pct_change(horizon)
        )

    peak = frame["strategy_equity_before"].cummax()
    frame["strategy_drawdown_before"] = frame["strategy_equity_before"] / peak - 1.0

    score_arrays = frame["asset_scores_json"].map(finite_scores)
    frame["score_count"] = score_arrays.map(len)
    frame["score_q10"] = score_arrays.map(
        lambda x: float(np.quantile(x, 0.10)) if len(x) else np.nan
    )
    frame["score_q25"] = score_arrays.map(
        lambda x: float(np.quantile(x, 0.25)) if len(x) else np.nan
    )
    frame["score_median"] = score_arrays.map(
        lambda x: float(np.quantile(x, 0.50)) if len(x) else np.nan
    )
    frame["score_q75"] = score_arrays.map(
        lambda x: float(np.quantile(x, 0.75)) if len(x) else np.nan
    )
    frame["score_q90"] = score_arrays.map(
        lambda x: float(np.quantile(x, 0.90)) if len(x) else np.nan
    )
    frame["score_iqr"] = frame["score_q75"] - frame["score_q25"]

    frame["top1_minus_top2"] = (
        pd.to_numeric(frame["top_1_score"], errors="coerce")
        - pd.to_numeric(frame["top_2_score"], errors="coerce")
    )
    frame["top2_minus_top3"] = (
        pd.to_numeric(frame["top_2_score"], errors="coerce")
        - pd.to_numeric(frame["top_3_score"], errors="coerce")
    )
    std = pd.to_numeric(frame["universe_score_std"], errors="coerce")
    frame["top1_score_z"] = np.where(
        std.abs() > 1e-12,
        (
            pd.to_numeric(frame["top_1_score"], errors="coerce")
            - pd.to_numeric(frame["universe_score_mean"], errors="coerce")
        )
        / std,
        np.nan,
    )

    context_columns = [
        "decision_date",
        "effective_switch_margin",
        "decision_score",
        "top_2_score",
        "top_3_score",
        "current_score",
        "current_asset_rank",
        "universe_score_mean",
        "universe_score_std",
        "best_vs_second_gap",
        "best_vs_current_gap",
        "spy_return_5",
        "spy_return_20",
        "spy_realized_volatility_20",
        "universe_breadth_5",
        "universe_breadth_20",
        "strategy_return_5",
        "strategy_return_20",
        "strategy_return_60",
        "strategy_drawdown_before",
        "score_count",
        "score_q10",
        "score_q25",
        "score_median",
        "score_q75",
        "score_q90",
        "score_iqr",
        "top1_minus_top2",
        "top2_minus_top3",
        "top1_score_z",
        "top_2_asset",
        "top_3_asset",
    ]
    return frame[context_columns].copy()


def build_dataset(
    episodes: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    rows = episodes.loc[
        episodes["forced_action_changed"].astype(bool)
        & ~episodes["target_censored"].astype(bool)
    ].copy()

    context = enrich_baseline_context(baseline)
    rows = rows.merge(context, on="decision_date", how="left", validate="many_to_one")

    month = rows["decision_date"].dt.month.astype(float)
    rows["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    rows["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)

    score_count = pd.to_numeric(rows["score_count"], errors="coerce")
    candidate_rank = pd.to_numeric(rows["candidate_model_rank"], errors="coerce")
    rows["candidate_score_percentile"] = np.where(
        score_count > 1,
        1.0 - (candidate_rank - 1.0) / (score_count - 1.0),
        np.nan,
    )
    return rows


def feature_columns(rows: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_base = [
        "holding_days_at_decision",
        "candidate_model_score",
        "candidate_model_rank",
        "candidate_has_model_score",
        "top1_score",
        "score_gap_candidate_to_top1",
        "baseline_action_score",
        "score_gap_candidate_to_baseline_action",
        "previous_asset_score",
        "candidate_is_top1",
        "candidate_is_baseline_action",
        "candidate_is_previous_asset",
        "effective_switch_margin",
        "decision_score",
        "top_2_score",
        "top_3_score",
        "current_score",
        "current_asset_rank",
        "universe_score_mean",
        "universe_score_std",
        "best_vs_second_gap",
        "best_vs_current_gap",
        "spy_return_5",
        "spy_return_20",
        "spy_realized_volatility_20",
        "universe_breadth_5",
        "universe_breadth_20",
        "strategy_return_5",
        "strategy_return_20",
        "strategy_return_60",
        "strategy_drawdown_before",
        "score_count",
        "score_q10",
        "score_q25",
        "score_median",
        "score_q75",
        "score_q90",
        "score_iqr",
        "top1_minus_top2",
        "top2_minus_top3",
        "top1_score_z",
        "month_sin",
        "month_cos",
        "candidate_score_percentile",
    ]

    generated = [
        column
        for column in rows.columns
        if column.startswith(
            (
                "candidate__",
                "relative__",
                "candidate_vs_baseline__",
                "candidate_z__",
            )
        )
    ]
    numeric = [column for column in [*numeric_base, *generated] if column in rows.columns]

    categorical = [
        column
        for column in (
            "candidate",
            "previous_asset",
            "baseline_action",
            "top1_asset",
            "top_2_asset",
            "top_3_asset",
        )
        if column in rows.columns
    ]
    return numeric, categorical


def design_matrix(
    rows: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> pd.DataFrame:
    numeric_frame = rows[numeric].copy()
    for column in numeric_frame.columns:
        numeric_frame[column] = pd.to_numeric(numeric_frame[column], errors="coerce")

    categories = [*ASSETS, "CASH", "nan"]
    category_frames = []
    for column in categorical:
        values = rows[column].astype(str).fillna("nan")
        cat = pd.Categorical(values, categories=categories)
        dummies = pd.get_dummies(
            cat,
            prefix=column,
            dtype=np.int8,
        )
        dummies.index = rows.index
        category_frames.append(dummies)

    return pd.concat([numeric_frame, *category_frames], axis=1)


def relevance_labels(rows: pd.DataFrame) -> pd.Series:
    percentile = rows.groupby("decision_date")["episode_delta_log_capital"].rank(
        pct=True,
        method="average",
    )
    labels = np.floor(percentile * 5.0).clip(0, 4).astype(int)
    return labels


def safe_spearman(prediction: np.ndarray, target: pd.Series) -> float:
    value = spearmanr(prediction, target.to_numpy(), nan_policy="omit").statistic
    return float(value) if value is not None and np.isfinite(value) else float("nan")


def top_pick_metrics(
    test: pd.DataFrame,
    prediction_column: str,
    *,
    gated: bool,
) -> dict[str, float]:
    top_indices = test.groupby("decision_date")[prediction_column].idxmax()
    picks = test.loc[top_indices].copy()

    if gated:
        intervene = picks[picks[prediction_column] > 0.0].copy()
        effective = np.where(
            picks[prediction_column].to_numpy() > 0.0,
            picks["episode_delta_log_capital"].to_numpy(),
            0.0,
        )
        return {
            "decision_count": int(len(picks)),
            "intervention_rate": float((picks[prediction_column] > 0.0).mean()),
            "mean_realized_delta_log_all_decisions": float(np.mean(effective)),
            "mean_realized_delta_pct_all_decisions": float(np.mean(np.exp(effective) - 1.0)),
            "intervention_positive_rate": (
                float((intervene["episode_delta_log_capital"] > 0.0).mean())
                if len(intervene)
                else float("nan")
            ),
            "mean_realized_delta_log_when_intervening": (
                float(intervene["episode_delta_log_capital"].mean())
                if len(intervene)
                else float("nan")
            ),
        }

    return {
        "decision_count": int(len(picks)),
        "intervention_rate": 1.0,
        "mean_realized_delta_log_all_decisions": float(
            picks["episode_delta_log_capital"].mean()
        ),
        "mean_realized_delta_pct_all_decisions": float(
            picks["episode_delta_capital_pct"].mean()
        ),
        "intervention_positive_rate": float(
            (picks["episode_delta_log_capital"] > 0.0).mean()
        ),
        "mean_realized_delta_log_when_intervening": float(
            picks["episode_delta_log_capital"].mean()
        ),
    }


def oracle_metrics(test: pd.DataFrame) -> dict[str, float]:
    indices = test.groupby("decision_date")["episode_delta_log_capital"].idxmax()
    oracle = test.loc[indices]
    return {
        "oracle_mean_delta_log": float(oracle["episode_delta_log_capital"].mean()),
        "oracle_median_delta_log": float(oracle["episode_delta_log_capital"].median()),
        "oracle_positive_rate": float(
            (oracle["episode_delta_log_capital"] > 0.0).mean()
        ),
        "oracle_mean_delta_pct": float(oracle["episode_delta_capital_pct"].mean()),
    }


def top5_oracle_hit(test: pd.DataFrame, prediction_column: str) -> float:
    hits = []
    for _, group in test.groupby("decision_date"):
        oracle_idx = group["episode_delta_log_capital"].idxmax()
        predicted_top5 = set(
            group.nlargest(5, prediction_column).index.tolist()
        )
        hits.append(oracle_idx in predicted_top5)
    return float(np.mean(hits)) if hits else float("nan")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    episodes, baseline, source_manifest = load_inputs()
    rows = build_dataset(episodes, baseline)

    numeric, categorical = feature_columns(rows)
    x = design_matrix(rows, numeric, categorical)
    y = pd.to_numeric(rows["episode_delta_log_capital"], errors="coerce")
    relevance = relevance_labels(rows)

    write_json(
        FEATURES_PATH,
        {
            "schema_version": 1,
            "script_version": VERSION,
            "source_script_version": source_manifest.get("script_version"),
            "numeric_features": numeric,
            "categorical_features_one_hot": categorical,
            "design_matrix_columns": list(x.columns),
            "target": "episode_delta_log_capital",
            "label_maturity_rule": "reconvergence_date < test_start",
            "test_scheme": "expanding annual walk-forward",
        },
    )

    years = sorted(rows["decision_date"].dt.year.unique().tolist())
    test_years = [year for year in years if year >= FIRST_TEST_YEAR]

    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []

    for year in test_years:
        test_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        test_end = (
            pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
            if year < max(test_years)
            else rows["decision_date"].max() + pd.Timedelta(days=1)
        )

        train_mask = (
            (rows["decision_date"] < test_start)
            & (rows["reconvergence_date"] < test_start)
        )
        test_mask = (
            (rows["decision_date"] >= test_start)
            & (rows["decision_date"] < test_end)
        )

        if int(train_mask.sum()) < 1000 or int(test_mask.sum()) == 0:
            continue

        train_idx = rows.index[train_mask]
        test_idx = rows.index[test_mask]

        log(
            f"Ano {year} | treino={len(train_idx):,} | "
            f"teste={len(test_idx):,} | "
            f"datas={rows.loc[test_idx, 'decision_date'].nunique()}"
        )

        # Regressao robusta de magnitude.
        reg = LGBMRegressor(
            objective="regression_l1",
            n_estimators=500,
            learning_rate=0.025,
            num_leaves=31,
            min_child_samples=100,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.70,
            reg_alpha=0.10,
            reg_lambda=3.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        reg.fit(x.loc[train_idx], y.loc[train_idx])
        reg_pred = reg.predict(x.loc[test_idx])

        test_reg = rows.loc[test_idx].copy()
        test_reg["meta_prediction"] = reg_pred
        reg_metrics = top_pick_metrics(
            test_reg,
            "meta_prediction",
            gated=True,
        )
        oracle = oracle_metrics(test_reg)
        fold_rows.append(
            {
                "year": year,
                "model": "lightgbm_regression_l1",
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "test_decisions": test_reg["decision_date"].nunique(),
                "row_spearman": safe_spearman(reg_pred, y.loc[test_idx]),
                "top5_oracle_hit_rate": top5_oracle_hit(
                    test_reg, "meta_prediction"
                ),
                **reg_metrics,
                **oracle,
            }
        )

        pred_reg = test_reg[
            [
                "decision_date",
                "candidate",
                "baseline_action",
                "episode_delta_log_capital",
                "episode_delta_capital_pct",
            ]
        ].copy()
        pred_reg["year"] = year
        pred_reg["model"] = "lightgbm_regression_l1"
        pred_reg["prediction"] = reg_pred
        prediction_frames.append(pred_reg)

        gain = reg.booster_.feature_importance(importance_type="gain")
        for feature, value in zip(x.columns, gain):
            importance_rows.append(
                {
                    "year": year,
                    "model": "lightgbm_regression_l1",
                    "feature": feature,
                    "gain": float(value),
                }
            )

        # Ranker pareado por data. Nao possui escala absoluta, logo nao ha gate
        # contra baseline=0 nesta versao; mede somente capacidade de ordenar.
        train_order = train_idx[
            np.argsort(rows.loc[train_idx, "decision_date"].to_numpy())
        ]
        test_order = test_idx[
            np.argsort(rows.loc[test_idx, "decision_date"].to_numpy())
        ]
        train_dates = rows.loc[train_order, "decision_date"]
        groups = train_dates.groupby(train_dates, sort=False).size().to_numpy()

        ranker = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=500,
            learning_rate=0.025,
            num_leaves=31,
            min_child_samples=100,
            colsample_bytree=0.70,
            reg_alpha=0.10,
            reg_lambda=3.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        ranker.fit(
            x.loc[train_order],
            relevance.loc[train_order],
            group=groups,
        )
        rank_pred = ranker.predict(x.loc[test_order])

        test_rank = rows.loc[test_order].copy()
        test_rank["meta_prediction"] = rank_pred
        rank_metrics = top_pick_metrics(
            test_rank,
            "meta_prediction",
            gated=False,
        )
        fold_rows.append(
            {
                "year": year,
                "model": "lightgbm_lambdarank",
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "test_decisions": test_rank["decision_date"].nunique(),
                "row_spearman": safe_spearman(
                    rank_pred, y.loc[test_order]
                ),
                "top5_oracle_hit_rate": top5_oracle_hit(
                    test_rank, "meta_prediction"
                ),
                **rank_metrics,
                **oracle,
            }
        )

        pred_rank = test_rank[
            [
                "decision_date",
                "candidate",
                "baseline_action",
                "episode_delta_log_capital",
                "episode_delta_capital_pct",
            ]
        ].copy()
        pred_rank["year"] = year
        pred_rank["model"] = "lightgbm_lambdarank"
        pred_rank["prediction"] = rank_pred
        prediction_frames.append(pred_rank)

        gain = ranker.booster_.feature_importance(importance_type="gain")
        for feature, value in zip(x.columns, gain):
            importance_rows.append(
                {
                    "year": year,
                    "model": "lightgbm_lambdarank",
                    "feature": feature,
                    "gain": float(value),
                }
            )

    metrics = pd.DataFrame(fold_rows)
    metrics.to_csv(FOLD_METRICS_PATH, index=False)

    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            PREDICTIONS_PATH,
            index=False,
        )

    importance = pd.DataFrame(importance_rows)
    if not importance.empty:
        aggregate_importance = (
            importance.groupby(["model", "feature"], as_index=False)["gain"]
            .mean()
            .sort_values(["model", "gain"], ascending=[True, False])
        )
        aggregate_importance.to_csv(IMPORTANCE_PATH, index=False)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "script_version": VERSION,
        "source_script_version": source_manifest.get("script_version"),
        "rows": int(len(rows)),
        "decision_count": int(rows["decision_date"].nunique()),
        "candidate_count": int(rows["candidate"].nunique()),
        "target_mean": float(y.mean()),
        "target_positive_rate": float((y > 0.0).mean()),
        "folds": metrics.to_dict("records"),
    }

    if not metrics.empty:
        model_summary = (
            metrics.groupby("model", as_index=False)
            .agg(
                mean_row_spearman=("row_spearman", "mean"),
                mean_top5_oracle_hit_rate=("top5_oracle_hit_rate", "mean"),
                mean_realized_delta_log_all_decisions=(
                    "mean_realized_delta_log_all_decisions",
                    "mean",
                ),
                mean_intervention_rate=("intervention_rate", "mean"),
                mean_intervention_positive_rate=(
                    "intervention_positive_rate",
                    "mean",
                ),
                mean_oracle_delta_log=("oracle_mean_delta_log", "mean"),
            )
        )
        summary["model_summary"] = model_summary.to_dict("records")

    write_json(SUMMARY_PATH, summary)

    log(f"Concluido | folds={len(metrics)}")
    log(f"Metricas: {FOLD_METRICS_PATH}")
    log(f"Predicoes: {PREDICTIONS_PATH}")
    log(f"Importancias: {IMPORTANCE_PATH}")
    log(f"Resumo: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
