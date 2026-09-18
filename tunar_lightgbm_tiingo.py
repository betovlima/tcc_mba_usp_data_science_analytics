"""Campanha LHS do LightGBM sobre o baseline oficial Tiingo 56 split-causal.

Versao: tiingo-56-lightgbm-lhs-v1.0.0

Somente hiperparametros do LightGBM variam. Dataset, features, targets, folds,
politica de rotacao e custos permanecem fixos. A referencia historica de 43M
nao participa da funcao de selecao; serve apenas como contexto externo.

A campanha usa Latin Hypercube Sampling (LHS) com seed fixa, inclui o baseline
atual como candidate_000 e e retomavel: candidatos ja concluidos sao lidos do
disco e nao sao reexecutados.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import qmc

from tcc_engine.capital_rotation import run_rotation_models
from tcc_engine.config import ASSETS, CALENDAR_ANCHOR_ASSETS, CONFIG
from tcc_engine.execution import apply_slippage, calculate_reference_fees

VERSION = "tiingo-56-lightgbm-lhs-v1.0.0"
ROOT = Path(__file__).resolve().parent
DIR_TIINGO = ROOT / "dados" / "series_historicas"
DIR_EVENTS = ROOT / "dados" / "eventos_corporativos"
DIR_SPLITS = ROOT / "dados" / "desdobramentos"
DIR_OUT = ROOT / "output" / "tiingo_56_lightgbm_lhs_v1"
MANIFEST_TIINGO = ROOT / "dados" / "manifesto_tiingo.json"
MANIFEST_SPLITS = ROOT / "dados" / "manifesto_desdobramentos_tiingo.json"
OHLCV = ["open", "high", "low", "close", "volume"]
HISTORICAL_REFERENCE_CAPITAL = 43_759_854.82
DEFAULT_LHS_CANDIDATES = 32
SEARCH_SEED = 20260916
BASELINE_MAXDD_TOLERANCE = 0.05
BASELINE_WORST_FOLD_FRACTION = 0.80


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_ohlcv(directory: Path, asset: str) -> pd.DataFrame:
    path = directory / f"{asset}.csv"
    if not path.exists():
        raise RuntimeError(f"Arquivo ausente: {path}")
    df = pd.read_csv(path)
    required = ["timestamp", *OHLCV]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{asset}: colunas ausentes em {path.name}: {', '.join(missing)}")
    df = df[required].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in OHLCV:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).sort_values("timestamp")
    df["session"] = df["timestamp"].dt.normalize()
    df = df.drop_duplicates("session", keep="last")
    df = df.loc[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["volume"] >= 0)
    ].copy()
    if df.empty:
        raise RuntimeError(f"{asset}: serie vazia depois da validacao")
    return df.set_index("session")[OHLCV].sort_index()


def apply_splits(asset: str, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    path = DIR_SPLITS / f"{asset}.csv"
    if not path.exists():
        raise RuntimeError(f"{asset}: desdobramentos ausentes: {path}")
    events = pd.read_csv(path)
    session_factor = pd.Series(1.0, index=raw.index, dtype=float)
    if not events.empty:
        events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
        events["fator_split"] = pd.to_numeric(events["fator_split"], errors="coerce")
        if "status" in events.columns:
            events = events.loc[events["status"].astype(str).str.lower().str.strip() == "a"]
        events = events.dropna(subset=["timestamp", "fator_split"])
        events = events.loc[events["fator_split"] > 0]
        for event in events.itertuples(index=False):
            session = pd.Timestamp(event.timestamp).normalize()
            if session in session_factor.index:
                session_factor.loc[session] *= float(event.fator_split)
    cumulative = session_factor.cumprod()
    out = raw.copy()
    for c in ("open", "high", "low", "close"):
        out[c] = out[c] * cumulative
    out["volume"] = out["volume"] / cumulative
    return out, cumulative


def load_tiingo_split_causal() -> dict[str, pd.DataFrame]:
    """Carrega exatamente o mesmo OHLCV usado pelo baseline oficial.

    Apenas splits confirmados sao normalizados causalmente da data do evento
    para frente. Dividendos permanecem armazenados apenas para auditoria e nao
    alteram preco, features ou targets.
    """
    series: dict[str, pd.DataFrame] = {}
    for pos, asset in enumerate(ASSETS, start=1):
        raw = read_ohlcv(DIR_TIINGO, asset)
        split, _ = apply_splits(asset, raw)
        series[asset] = split
        log(f"[dados] {pos:02d}/{len(ASSETS)} {asset} | sessoes={len(series[asset])}")
    return series
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_signature() -> dict[str, Any]:
    digest = hashlib.sha256()
    file_rows: list[dict[str, str]] = []
    for directory, label in (
        (DIR_TIINGO, "ohlcv_raw"),
        (DIR_EVENTS, "corporate_actions"),
        (DIR_SPLITS, "splits"),
    ):
        for asset in ASSETS:
            path = directory / f"{asset}.csv"
            if not path.exists():
                raise RuntimeError(f"Entrada da campanha ausente: {path}")
            sha = file_sha256(path)
            relative = path.relative_to(ROOT).as_posix()
            digest.update(f"{label}|{asset}|{relative}|{sha}\n".encode("utf-8"))
            file_rows.append({"kind": label, "asset": asset, "path": relative, "sha256": sha})
    return {
        "combined_sha256": digest.hexdigest(),
        "files": file_rows,
        "tiingo_manifest": read_json_optional(MANIFEST_TIINGO),
        "split_manifest": read_json_optional(MANIFEST_SPLITS),
    }


def validate_snapshot_contract() -> None:
    """Garante que a campanha usa o snapshot oficial Tiingo dos mesmos 56 ativos."""
    expected_assets = list(ASSETS)
    if len(expected_assets) != 56:
        raise RuntimeError(
            f"Contrato invalido: esperado universo de 56 ativos; CONFIG possui {len(expected_assets)}."
        )

    tiingo = read_json_optional(MANIFEST_TIINGO)
    splits = read_json_optional(MANIFEST_SPLITS)
    if not tiingo:
        raise RuntimeError("manifesto_tiingo.json ausente.")
    if not splits:
        raise RuntimeError("manifesto_desdobramentos_tiingo.json ausente.")

    manifest_assets = [str(x).upper() for x in (tiingo.get("ativos") or [])]
    if int(tiingo.get("quantidade_ativos", -1)) != 56 or manifest_assets != expected_assets:
        raise RuntimeError(
            "Snapshot Tiingo nao corresponde ao universo oficial de 56 ativos. "
            "Execute novamente congelar_series_tiingo.py nesta branch."
        )
    if int(splits.get("quantidade_ativos", -1)) != 56:
        raise RuntimeError(
            "Snapshot de desdobramentos nao corresponde aos 56 ativos. "
            "Execute novamente congelar_desdobramentos_tiingo.py."
        )

    for label, directory in (
        ("OHLCV", DIR_TIINGO),
        ("eventos", DIR_EVENTS),
        ("splits", DIR_SPLITS),
    ):
        missing = [asset for asset in expected_assets if not (directory / f"{asset}.csv").exists()]
        if missing:
            raise RuntimeError(
                f"Snapshot {label} incompleto; faltam {len(missing)} ativo(s): "
                + ", ".join(missing)
            )


def current_baseline_params() -> dict[str, Any]:
    settings = deepcopy(CONFIG.research_model_settings)
    lightgbm = settings.get("lightgbm")
    if not isinstance(lightgbm, dict):
        raise RuntimeError("CONFIG.research_model_settings.lightgbm ausente")
    required = (
        "n_estimators", "learning_rate", "max_depth", "num_leaves",
        "min_child_samples", "min_child_weight", "subsample", "subsample_freq",
        "colsample_bytree", "reg_alpha", "reg_lambda", "max_bin", "n_jobs",
    )
    missing = [key for key in required if key not in lightgbm]
    if missing:
        raise RuntimeError("Baseline LightGBM incompleto: " + ", ".join(missing))
    return {key: lightgbm[key] for key in required}


def log_uniform(low: float, high: float, u: float) -> float:
    return float(math.exp(math.log(low) + float(u) * (math.log(high) - math.log(low))))


def integer_from_unit(low: int, high: int, u: float) -> int:
    value = low + int(math.floor(float(u) * (high - low + 1)))
    return int(min(high, max(low, value)))


def lhs_candidates(count: int) -> list[dict[str, Any]]:
    if count < 1:
        return []
    sampler = qmc.LatinHypercube(d=12, seed=SEARCH_SEED)
    sample = sampler.random(n=count)
    candidates: list[dict[str, Any]] = []
    max_bins = (63, 127, 255)
    for row in sample:
        max_depth = integer_from_unit(2, 6, row[2])
        max_leaves = max(2, min(48, 2 ** max_depth))
        min_leaves = 2 if max_leaves < 4 else 4
        params = {
            "n_estimators": integer_from_unit(180, 700, row[0]),
            "learning_rate": log_uniform(0.008, 0.080, row[1]),
            "max_depth": max_depth,
            "num_leaves": integer_from_unit(min_leaves, max_leaves, row[3]),
            "min_child_samples": integer_from_unit(10, 80, row[4]),
            "min_child_weight": log_uniform(0.5, 20.0, row[5]),
            "subsample": float(0.65 + row[6] * 0.35),
            "subsample_freq": 0 if row[7] < 0.25 else 1,
            "colsample_bytree": float(0.60 + row[8] * 0.40),
            "reg_alpha": log_uniform(1e-4, 1.5, row[9]),
            "reg_lambda": log_uniform(0.10, 10.0, row[10]),
            "max_bin": max_bins[integer_from_unit(0, len(max_bins) - 1, row[11])],
            "n_jobs": -1,
        }
        candidates.append(params)
    return candidates


def candidate_table(lhs_count: int) -> list[dict[str, Any]]:
    rows = [{"candidate_id": "candidate_000", "kind": "baseline", "params": current_baseline_params()}]
    for index, params in enumerate(lhs_candidates(lhs_count), start=1):
        rows.append({"candidate_id": f"candidate_{index:03d}", "kind": "lhs", "params": params})
    return rows


def campaign_manifest(lhs_count: int, signature: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "script_version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "tiingo_56_split_causal_lightgbm_lhs",
        "lhs_candidate_count": lhs_count,
        "candidate_count_including_baseline": lhs_count + 1,
        "search_seed": SEARCH_SEED,
        "historical_reference_capital_context_only": HISTORICAL_REFERENCE_CAPITAL,
        "selection_uses_43m_reference": False,
        "fixed_components": [
            "Tiingo RAW snapshot 56 assets", "causal split normalization", "dividends audit-only (not applied)",
            "features", "targets", "walk-forward folds", "rotation policy", "fees and slippage",
        ],
        "tuned_component": "LightGBM hyperparameters only",
        "guardrails": {
            "max_drawdown_tolerance_vs_campaign_baseline_absolute": BASELINE_MAXDD_TOLERANCE,
            "minimum_worst_fold_multiple_fraction_vs_campaign_baseline": BASELINE_WORST_FOLD_FRACTION,
        },
        "dataset_signature": signature,
    }


def ensure_campaign(lhs_count: int, signature: dict[str, Any]) -> list[dict[str, Any]]:
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = DIR_OUT / "campaign.json"
    candidates_path = DIR_OUT / "candidates.json"
    expected_candidates = candidate_table(lhs_count)
    existing_manifest = read_json_optional(manifest_path)
    existing_candidates = read_json_optional(candidates_path)
    if existing_manifest is not None:
        if existing_manifest.get("script_version") != VERSION:
            raise RuntimeError("A pasta de campanha existente pertence a outra versao do script")
        if int(existing_manifest.get("lhs_candidate_count", -1)) != lhs_count:
            raise RuntimeError(
                "A campanha existente foi criada com outro numero de candidatos. "
                "Use o mesmo --candidates ou remova output/tiingo_56_lightgbm_lhs_v1 para iniciar outra campanha."
            )
        old_sig = ((existing_manifest.get("dataset_signature") or {}).get("combined_sha256"))
        new_sig = signature.get("combined_sha256")
        if old_sig != new_sig:
            raise RuntimeError("O dataset Tiingo mudou desde o inicio da campanha; campanha abortada para preservar reprodutibilidade")
        if not isinstance(existing_candidates, list):
            raise RuntimeError("candidates.json da campanha esta ausente ou invalido")
        return existing_candidates

    manifest_path.write_text(
        json.dumps(campaign_manifest(lhs_count, signature), indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    candidates_path.write_text(
        json.dumps(expected_candidates, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([
        {"candidate_id": row["candidate_id"], "kind": row["kind"], **row["params"]}
        for row in expected_candidates
    ]).to_csv(DIR_OUT / "candidates.csv", index=False)
    return expected_candidates


def build_candidate_config(candidate_id: str, params: dict[str, Any]):
    settings = deepcopy(CONFIG.research_model_settings)
    settings["schema_version"] = 3
    settings["settings_revision"] = 2
    settings["profile_id"] = f"tcc-tiingo-tuning-{candidate_id}"
    settings["lightgbm"] = deepcopy(params)
    return CONFIG.model_copy(update={
        "assets": tuple(ASSETS),
        "calendar_anchor_assets": tuple(CALENDAR_ANCHOR_ASSETS),
        "research_reference_assets": tuple(ASSETS),
        "research_candidate_assets": (),
        "research_model_settings": settings,
        "research_model_family": "lightgbm_utility",
        "random_state": 42,
        "deterministic_execution": True,
        "numeric_thread_limit": 1,
    })


def fold_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    folds = list(metrics.get("walk_forward_folds") or [])
    if not folds:
        raise RuntimeError("Resultado sem walk_forward_folds")
    multiples: list[float] = []
    log_growth: list[float] = []
    drawdowns: list[float] = []
    for fold in folds:
        start = float(fold["strategy_starting_capital"])
        end = float(fold["strategy_ending_capital"])
        multiple = end / start if start > 0 else float("nan")
        if not np.isfinite(multiple) or multiple <= 0:
            multiple = 1e-12
        multiples.append(float(multiple))
        log_growth.append(float(math.log(multiple)))
        drawdowns.append(abs(float(fold.get("maximum_drawdown") or 0.0)))
    return {
        "fold_count": len(folds),
        "fold_multiples": multiples,
        "worst_fold_multiple": float(min(multiples)),
        "median_fold_multiple": float(np.median(multiples)),
        "geometric_mean_fold_multiple": float(math.exp(float(np.mean(log_growth)))),
        "fold_log_growth_std": float(np.std(log_growth)),
        "all_folds_positive": bool(all(value > 1.0 for value in multiples)),
        "mean_fold_drawdown_abs": float(np.mean(drawdowns)),
        "worst_fold_drawdown_abs": float(max(drawdowns)),
    }


def compact_result(candidate: dict[str, Any], metrics: dict[str, Any], elapsed: float) -> dict[str, Any]:
    fm = fold_metrics(metrics)
    return {
        "schema_version": 1,
        "script_version": VERSION,
        "candidate_id": candidate["candidate_id"],
        "candidate_kind": candidate["kind"],
        "status": "completed",
        "elapsed_seconds": elapsed,
        "params": candidate["params"],
        "metrics": {
            "strategy_ending_capital": float(metrics["strategy_ending_capital"]),
            "strategy_return": float(metrics["strategy_return"]),
            "strategy_cagr": float(metrics["strategy_cagr"]),
            "strategy_sharpe": float(metrics["strategy_sharpe"]),
            "strategy_maximum_drawdown": float(metrics["strategy_maximum_drawdown"]),
            "capital_rotations": int(metrics.get("capital_rotations") or 0),
            "effective_switch_margin": metrics.get("effective_switch_margin"),
            "effective_switch_margin_mean": metrics.get("effective_switch_margin_mean"),
            "walk_forward_folds": metrics.get("walk_forward_folds"),
            **fm,
        },
    }


def run_candidate(candidate: dict[str, Any], series: dict[str, pd.DataFrame], save_curves: bool) -> dict[str, Any]:
    candidate_id = candidate["candidate_id"]
    candidate_dir = DIR_OUT / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    result_path = candidate_dir / "result.json"
    existing = read_json_optional(result_path)
    if existing and existing.get("status") == "completed":
        log(f"[{candidate_id}] ja concluido; reutilizando resultado")
        return existing

    config = build_candidate_config(candidate_id, candidate["params"])
    started = time.perf_counter()
    log(f"[{candidate_id}] iniciando | kind={candidate['kind']}")

    def progress(percent: float, stage: str, completed: int) -> None:
        log(f"[{candidate_id}] {percent:5.1f}% | {stage}")

    try:
        results = run_rotation_models(
            series,
            config,
            calculate_reference_fees,
            apply_slippage,
            progress_callback=progress,
            technical_log_callback=lambda message: None,
        )
        if len(results) != 1:
            raise RuntimeError(f"Esperava uma execucao; recebidas={len(results)}")
        result = results[0]
        metrics = dict(result.metrics)
        payload = compact_result(candidate, metrics, time.perf_counter() - started)
        result_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
            encoding="utf-8",
        )
        pd.DataFrame(list(metrics.get("walk_forward_folds") or [])).to_csv(candidate_dir / "folds.csv", index=False)
        if save_curves:
            predictions = result.predictions.copy()
            cols = [c for c in (
                "strategy_equity", "selected_asset", "selected_score", "decision_score",
                "trade_action", "trade_reason", "walk_forward_fold",
            ) if c in predictions.columns]
            curve = predictions[cols].copy() if cols else predictions
            if curve.index.name is not None or not isinstance(curve.index, pd.RangeIndex):
                curve = curve.reset_index()
            curve.to_csv(candidate_dir / "equity_curve.csv", index=False)
        log(
            f"[{candidate_id}] concluido | capital=US$ {payload['metrics']['strategy_ending_capital']:,.2f} "
            f"| worst_fold={payload['metrics']['worst_fold_multiple']:.3f}x "
            f"| maxDD={payload['metrics']['strategy_maximum_drawdown']:.2%}"
        )
        return payload
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "script_version": VERSION,
            "candidate_id": candidate_id,
            "candidate_kind": candidate["kind"],
            "status": "error",
            "elapsed_seconds": time.perf_counter() - started,
            "params": candidate["params"],
            "error": f"{type(exc).__name__}: {exc}",
        }
        result_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
            encoding="utf-8",
        )
        log(f"[{candidate_id}] ERRO | {payload['error']}")
        return payload


def leaderboard_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.get("status") != "completed":
            continue
        metrics = result["metrics"]
        rows.append({
            "candidate_id": result["candidate_id"],
            "candidate_kind": result["candidate_kind"],
            "ending_capital": metrics["strategy_ending_capital"],
            "cagr": metrics["strategy_cagr"],
            "sharpe": metrics["strategy_sharpe"],
            "max_drawdown": metrics["strategy_maximum_drawdown"],
            "worst_fold_multiple": metrics["worst_fold_multiple"],
            "median_fold_multiple": metrics["median_fold_multiple"],
            "geometric_mean_fold_multiple": metrics["geometric_mean_fold_multiple"],
            "fold_log_growth_std": metrics["fold_log_growth_std"],
            "all_folds_positive": metrics["all_folds_positive"],
            "capital_rotations": metrics["capital_rotations"],
            **result["params"],
        })
    return rows


def rank_and_write(all_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = leaderboard_rows(all_results)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    robust = df.sort_values(
        ["all_folds_positive", "worst_fold_multiple", "geometric_mean_fold_multiple", "ending_capital", "sharpe", "max_drawdown"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    robust.insert(0, "robust_rank", np.arange(1, len(robust) + 1))

    baseline = df.loc[df["candidate_id"] == "candidate_000"]
    best_guardrail: dict[str, Any] | None = None
    guardrail_count = 0
    if not baseline.empty:
        b = baseline.iloc[0]
        max_allowed_dd_abs = abs(float(b["max_drawdown"])) + BASELINE_MAXDD_TOLERANCE
        min_worst_fold = float(b["worst_fold_multiple"]) * BASELINE_WORST_FOLD_FRACTION
        eligible = df.loc[
            df["all_folds_positive"].astype(bool)
            & (df["max_drawdown"].abs() <= max_allowed_dd_abs)
            & (df["worst_fold_multiple"] >= min_worst_fold)
        ].copy()
        guardrail_count = len(eligible)
        if not eligible.empty:
            eligible = eligible.sort_values(
                ["ending_capital", "worst_fold_multiple", "sharpe"],
                ascending=[False, False, False],
            )
            best_guardrail = eligible.iloc[0].to_dict()

    robust.to_csv(DIR_OUT / "leaderboard.csv", index=False)
    best_robust = robust.iloc[0].to_dict()
    summary = {
        "schema_version": 1,
        "script_version": VERSION,
        "completed_candidates": int(len(df)),
        "best_robust": best_robust,
        "best_capital_guardrailed": best_guardrail,
        "guardrail_eligible_candidates": int(guardrail_count),
        "historical_reference_capital_context_only": HISTORICAL_REFERENCE_CAPITAL,
        "selection_uses_43m_reference": False,
    }
    (DIR_OUT / "campaign_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    (DIR_OUT / "best_robust.json").write_text(
        json.dumps(best_robust, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    if best_guardrail is not None:
        (DIR_OUT / "best_capital_guardrailed.json").write_text(
            json.dumps(best_guardrail, indent=2, ensure_ascii=False, default=json_default) + "\n",
            encoding="utf-8",
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LHS LightGBM sobre Tiingo 56 split-causal congelado")
    parser.add_argument("--candidates", type=int, default=DEFAULT_LHS_CANDIDATES, help="Quantidade de candidatos LHS, alem do baseline")
    parser.add_argument("--start", type=int, default=0, help="Indice inicial na lista completa, baseline=0")
    parser.add_argument("--limit", type=int, default=None, help="Maximo de candidatos a processar nesta execucao")
    parser.add_argument("--save-curves", action="store_true", help="Salvar equity_curve.csv de cada candidato")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.candidates < 1:
        raise SystemExit("--candidates deve ser >= 1")
    for directory in (DIR_TIINGO, DIR_EVENTS, DIR_SPLITS):
        if not directory.exists():
            raise RuntimeError(f"Diretorio ausente: {directory}")

    log(f"Campanha: {VERSION}")
    log("Somente hiperparametros LightGBM variam; politica/features/targets permanecem fixos")
    log("Referencia de 43M e apenas contexto; nao entra na selecao dos candidatos")
    validate_snapshot_contract()
    log("Contrato validado: Tiingo-only, 56 ativos, splits causais, dividendos fora do modelo")
    log("Calculando assinatura SHA-256 do dataset congelado")
    signature = dataset_signature()
    log(f"Dataset signature: {signature['combined_sha256']}")
    candidates = ensure_campaign(args.candidates, signature)

    start = max(0, int(args.start))
    selected = candidates[start:]
    if args.limit is not None:
        selected = selected[: max(0, int(args.limit))]
    if not selected:
        log("Nenhum candidato selecionado nesta execucao")
        return 0

    log(f"Carregando baseline Tiingo split-causal uma unica vez | ativos={len(ASSETS)}")
    series = load_tiingo_split_causal()

    results_by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        path = DIR_OUT / candidate["candidate_id"] / "result.json"
        existing = read_json_optional(path)
        if existing is not None:
            results_by_id[candidate["candidate_id"]] = existing

    for position, candidate in enumerate(selected, start=1):
        log(f"Campanha local {position}/{len(selected)} | {candidate['candidate_id']}")
        result = run_candidate(candidate, series, args.save_curves)
        results_by_id[candidate["candidate_id"]] = result
        summary = rank_and_write(list(results_by_id.values()))
        if summary:
            best = summary["best_robust"]
            log(
                f"Lider robusto atual: {best['candidate_id']} | "
                f"capital=US$ {best['ending_capital']:,.2f} | worst_fold={best['worst_fold_multiple']:.3f}x"
            )

    completed = sum(1 for result in results_by_id.values() if result.get("status") == "completed")
    errors = sum(1 for result in results_by_id.values() if result.get("status") == "error")
    log(f"Campanha atualizada | concluidos={completed}/{len(candidates)} | erros={errors}")
    log(f"Resultados: {DIR_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
