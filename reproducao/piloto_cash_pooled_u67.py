# -*- coding: utf-8 -*-
"""Piloto causal de CASH vs posicao: modelo pooled, snapshot historico U67.

Este programa NAO reproduz o Control de producao U75, nao negocia, nao
realiza download, nao consulta credenciais e nao assume liquidez intradiaria.
Usa exclusivamente o snapshot versionado, validado por SHA256.

Informacoes na decisao de fechamento D; ordem hipotetica na abertura D+1;
retorno da posicao de abertura D+1 ate abertura D+2 (sem dividendos).
Os custos sao hipoteticos: 5 bps por lado, NAO custo Alpaca comprovado.
Separacao cronologica de sessoes, purge >= 6 sessoes em cada borda.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_pinball_loss, mean_squared_error

from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.preparacao import prepare_model_frames

RANDOM_SEED = 20261110
TRADE_FEE_PER_SIDE = 0.0005
PURGE_SESSIONS = 6
FEATURES = (
    "ret_1", "ret_2", "ret_5", "ret_10", "ret_20", "ret_60", "ret_120",
    "vol_10", "vol_20", "distance_ema_10", "distance_ema_20",
    "distance_ema_50", "ema20_ema50_gap", "ema20_slope_5",
    "atr_14", "drawdown_60", "volume_ratio_20",
    "spy_return_5", "spy_return_20", "spy_vol_20",
    "breadth_positive_5", "breadth_positive_20",
    "breadth_above_ema20", "dispersion_return_5",
    "relative_strength_20", "rank_return_20",
)
RISK_PREFERENCES = (0.0, 0.1, 0.25, 0.5, 1.0)


def make_training_panels(frames: dict[str, pd.DataFrame], dates: pd.DatetimeIndex):
    """Features EX ANTE computed on closes through decision date, per ticker.

    Missing future bars are NOT used to filter trading candidates. They merely
    make a supervised training label unavailable. The trade replay explicitly
    fails if a chosen action would need a missing next-open price.
    """
    raw = {}
    for symbol, frame in frames.items():
        x = frame.copy()
        x.index = pd.DatetimeIndex(x.index).tz_convert("UTC").normalize()
        x = x.loc[~x.index.duplicated(keep="last")].reindex(dates)
        close, op, high, low, volume = (
            x["close"], x["open"], x["high"], x["low"], x["volume"]
        )
        ret = close.pct_change(fill_method=None)
        r5 = close.pct_change(5, fill_method=None)
        r20 = close.pct_change(20, fill_method=None)
        ema10 = close.ewm(span=10, adjust=False, min_periods=10).mean()
        ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
        ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
        true_range = pd.concat([
            high - low, (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        row = pd.DataFrame(index=dates)
        for period in (1, 2, 5, 10, 20, 60, 120):
            row[f"ret_{period}"] = close.pct_change(period, fill_method=None)
        row["vol_10"] = ret.rolling(10, min_periods=10).std()
        row["vol_20"] = ret.rolling(20, min_periods=20).std()
        row["distance_ema_10"] = close / ema10 - 1
        row["distance_ema_20"] = close / ema20 - 1
        row["distance_ema_50"] = close / ema50 - 1
        row["ema20_ema50_gap"] = ema20 / ema50 - 1
        row["ema20_slope_5"] = ema20.pct_change(5, fill_method=None)
        row["atr_14"] = true_range.rolling(14, min_periods=14).mean() / close
        row["drawdown_60"] = close / close.rolling(60, min_periods=60).max() - 1
        row["volume_ratio_20"] = volume / volume.rolling(20, min_periods=20).mean()
        row["relative_strength_20"] = r20
        row["raw_open_next"] = op.shift(-1)
        row["raw_open_after_next"] = op.shift(-2)
        row["target_open_to_open"] = op.shift(-2) / op.shift(-1) - 1
        row["has_observation"] = (
            close.gt(0) & op.gt(0) & volume.gt(0)
            & close.rolling(120, min_periods=120).count().eq(120)
            & row["ret_120"].notna()
        )
        raw[symbol] = row
    if "SPY" not in raw:
        raise RuntimeError("SPY historico necessario como referencia de mercado.")
    spy = raw["SPY"]
    candidate_symbols = sorted(raw)
    ret5_panel = pd.DataFrame({
        symbol: raw[symbol]["ret_5"].where(raw[symbol]["has_observation"])
        for symbol in candidate_symbols
    }, index=dates)
    ret20_panel = pd.DataFrame({
        symbol: raw[symbol]["ret_20"].where(raw[symbol]["has_observation"])
        for symbol in candidate_symbols
    }, index=dates)
    ema20_panel = pd.DataFrame({
        symbol: raw[symbol]["distance_ema_20"].where(raw[symbol]["has_observation"])
        for symbol in candidate_symbols
    }, index=dates)
    breadth5 = ret5_panel.gt(0).sum(axis=1) / ret5_panel.notna().sum(axis=1).replace(0, np.nan)
    breadth20 = ret20_panel.gt(0).sum(axis=1) / ret20_panel.notna().sum(axis=1).replace(0, np.nan)
    breadth_ema = ema20_panel.gt(0).sum(axis=1) / ema20_panel.notna().sum(axis=1).replace(0, np.nan)
    dispersion = ret5_panel.std(axis=1, skipna=True)
    rank20 = ret20_panel.rank(axis=1, ascending=False, pct=True)
    for symbol, row in raw.items():
        row["spy_return_5"] = spy["ret_5"]
        row["spy_return_20"] = spy["ret_20"]
        row["spy_vol_20"] = spy["vol_20"]
        row["breadth_positive_5"] = breadth5
        row["breadth_positive_20"] = breadth20
        row["breadth_above_ema20"] = breadth_ema
        row["dispersion_return_5"] = dispersion
        row["relative_strength_20"] = row["ret_20"] - spy["ret_20"]
        row["rank_return_20"] = rank20[symbol]
        row["symbol"] = symbol
        row["date"] = dates
    panel = pd.concat(
        [df.loc[df["has_observation"]].copy() for df in raw.values()],
        ignore_index=True,
    )
    for column in FEATURES:
        panel[column] = pd.to_numeric(panel[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    panel["target_open_to_open"] = pd.to_numeric(
        panel["target_open_to_open"], errors="coerce",
    )
    return panel


def fit_models(panel, *, start_decision, end_decision):
    mask = (
        panel["date"].ge(start_decision)
        & panel["date"].le(end_decision)
        & panel["target_open_to_open"].notna()
        & panel["target_open_to_open"].gt(-1)
    )
    x = panel.loc[mask, list(FEATURES)]
    y = panel.loc[mask, "target_open_to_open"]
    if len(x) < 8000:
        raise RuntimeError("Menos de 8.000 linhas de treino causal utilizaveis.")
    fixed = dict(
        n_estimators=150, num_leaves=15, max_depth=5,
        min_child_samples=140, reg_lambda=20.0,
        learning_rate=0.035, random_state=RANDOM_SEED,
        n_jobs=2, verbosity=-1,
    )
    mean = LGBMRegressor(objective="regression", **fixed)
    tail = LGBMRegressor(objective="quantile", alpha=0.10, **fixed)
    mean.fit(x, y)
    tail.fit(x, y)
    return mean, tail, y


def infer(panel, mean, tail, start, end):
    d = panel.loc[panel["date"].ge(start) & panel["date"].le(end)].copy()
    d["predicted_mean"] = mean.predict(d.loc[:, list(FEATURES)])
    d["predicted_q10"] = tail.predict(d.loc[:, list(FEATURES)])
    if not (np.isfinite(d["predicted_mean"]).all()
            and np.isfinite(d["predicted_q10"]).all()):
        raise RuntimeError("Previsoes nao finitas.")
    return d


def cumulative_metrics(returns):
    values = np.cumprod(1 + np.asarray(returns, dtype=float))
    peaks = np.maximum.accumulate(np.r_[1., values])[1:]
    return {
        "growth_multiple": float(values[-1]),
        "maximum_drawdown": float(np.min(values / peaks - 1)),
        "annualized_log_growth": float(np.log(values[-1]) * 252 / len(values)),
    }


def decisions(predicted, dates, aversion):
    """Rebalance at next open; CASH competes with HOLD and ROTATE."""
    by_date = {date: data.set_index("symbol") for date, data in predicted.groupby("date", sort=False)}
    held = None
    rows = []
    returns = []
    for decision_date in dates:
        cross = by_date.get(decision_date)
        if cross is None or len(cross) < 25:
            raise RuntimeError(f"Universo de candidatos sem cobertura: {decision_date}")
        utility = (
            cross["predicted_mean"]
            - aversion * (-cross["predicted_q10"]).clip(lower=0)
        )
        cash_score = 0.0 if held is None else -TRADE_FEE_PER_SIDE
        choice = "CASH"
        score = cash_score
        for symbol in utility.index:
            need_to_buy = symbol != held
            turnover_cost = (
                TRADE_FEE_PER_SIDE * (2 if held is not None else 1)
                if need_to_buy else 0.
            )
            candidate_score = float(utility.loc[symbol]) - turnover_cost
            if candidate_score > score:
                score = candidate_score
                choice = symbol
        new_held = None if choice == "CASH" else choice
        trade_sides = (held != new_held) * (
            2 if held is not None and new_held is not None else 1
        )
        fee = TRADE_FEE_PER_SIDE * trade_sides
        if new_held is None:
            gross_return = 0.
        else:
            values = cross.loc[new_held]
            entry = float(values["raw_open_next"])
            exit_ = float(values["raw_open_after_next"])
            if not (
                np.isfinite(entry) and np.isfinite(exit_)
                and entry > 0 and exit_ > 0
            ):
                raise RuntimeError(
                    f"NEGOCIACAO_NAO_VERIFICAVEL: {new_held} em {decision_date}. "
                    "Nao supor fill em ativo sem cotacao de abertura."
                )
            gross_return = exit_ / entry - 1
        net_return = gross_return - fee
        if net_return <= -1:
            raise RuntimeError("Patrimonio hipotetico invalido.")
        kind = (
            "CASH_STAY" if held is None and new_held is None
            else "CASH_BUY" if held is None
            else "HOLD" if held == new_held
            else "SELL_CASH" if new_held is None
            else "ROTATE"
        )
        rows.append({
            "decision_close": str(decision_date.date()),
            "held_before": held or "CASH",
            "action": kind,
            "held_after": new_held or "CASH",
            "best_relative_symbol": utility.idxmax(),
            "best_relative_utility": float(utility.max()),
            "chosen_utility_net_cost": score,
            "gross_return_open_to_open": gross_return,
            "fee_assumption": fee,
            "net_return": net_return,
        })
        returns.append(net_return)
        held = new_held
    out = pd.DataFrame(rows)
    result = cumulative_metrics(returns)
    result.update({
        "cash_sessions": int((out["held_after"] == "CASH").sum()),
        "cash_fraction": float((out["held_after"] == "CASH").mean()),
        "actions": {
            k: int(v) for k, v in out["action"].value_counts().items()
        },
        "distinct_assets_selected": int(
            out.loc[out["held_after"] != "CASH", "held_after"].nunique()
        ),
    })
    return result, out


def predictive_scores(panel, trained_mean, trained_tail, training_returns):
    valid = panel["target_open_to_open"].notna()
    y = panel.loc[valid, "target_open_to_open"].to_numpy(dtype=float)
    x = panel.loc[valid, list(FEATURES)]
    forecasts_mean = trained_mean.predict(x)
    forecasts_q10 = trained_tail.predict(x)
    return {
        "rows": int(len(y)),
        "mean_squared_error_ml": float(mean_squared_error(y, forecasts_mean)),
        "mean_squared_error_zero": float(mean_squared_error(y, np.zeros_like(y))),
        "pinball_q10_ml": float(mean_pinball_loss(y, forecasts_q10, alpha=.1)),
        "pinball_q10_constant": float(mean_pinball_loss(
            y, np.full_like(y, np.quantile(training_returns, .1)), alpha=.1
        )),
        "q10_coverage_fraction": float(np.mean(y <= forecasts_q10)),
    }


def run(root: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    paths = SnapshotPaths.u67(root)
    manifest = validate_snapshot(paths)
    bars = manifest["bars"]
    if bars.get("adjustment") != "raw" or bars.get("timeframe") != "1Day":
        raise RuntimeError("Snapshot diferente do contrato raw/1Day.")
    symbols = tuple(manifest["assets"])
    frames, exclusions, _, _ = prepare_model_frames(
        paths, assets=symbols, csv_float_precision="round_trip"
    )
    if not frames or "SPY" not in frames or "CAPR" in frames:
        raise RuntimeError("Universo historico divergente de U67.")
    # SPY calendar is the reference session calendar. Missing bars are never
    # forward-filled and no future availability is checked at decision time.
    calendar = frames["SPY"].index.tz_convert("UTC").normalize()
    calendar = calendar.sort_values().unique()
    if not (calendar.is_monotonic_increasing and len(calendar) > 1700):
        raise RuntimeError("Calendario historico invalido.")
    if calendar.max().date().isoformat() != bars["bar_snapshot_as_of_end"]:
        raise RuntimeError("Ultima sessao diverge do snapshot versionado.")
    panel = make_training_panels(frames, calendar)
    # Exactly 3 chronological blocks; no random split / forward label leakage.
    valid_decision_dates = calendar[125:-2]
    chunks = np.array_split(valid_decision_dates, 3)
    train_days, val_days, test_days = chunks
    training_end = train_days[-1 - PURGE_SESSIONS]
    refit_end = val_days[-1 - PURGE_SESSIONS]
    mean, tail, y_first = fit_models(
        panel, start_decision=train_days[0], end_decision=training_end,
    )
    validations = infer(panel, mean, tail, val_days[0], val_days[-1])
    val_scores = predictive_scores(validations, mean, tail, y_first)
    selection = {}
    for aversion in RISK_PREFERENCES:
        metric, _ = decisions(validations, val_days, aversion)
        selection[str(aversion)] = {
            **metric,
            "selection_criterion": float(
                np.log(metric["growth_multiple"])
                + 2.0 * metric["maximum_drawdown"]
            ),
        }
    selected = max(
        RISK_PREFERENCES,
        key=lambda a: selection[str(a)]["selection_criterion"],
    )
    # Train once on folds 1+2, with embargo before unseen fold 3.
    mean_final, tail_final, y_development = fit_models(
        panel, start_decision=train_days[0], end_decision=refit_end,
    )
    testing = infer(panel, mean_final, tail_final, test_days[0], test_days[-1])
    test_scores = predictive_scores(testing, mean_final, tail_final, y_development)
    metrics, audit = decisions(testing, test_days, selected)
    audit.to_csv(output / "cash_ou_posicao_u67_teste_decisoes.csv", index=False)
    # A time-indexed panel with only test-time, nonprivileged indicators for audit.
    testing.loc[:, [
        "date", "symbol", "target_open_to_open",
        "predicted_mean", "predicted_q10",
    ]].to_csv(output / "cash_ou_posicao_u67_previsoes_teste.csv", index=False)
    result = {
        "status": "PILOTO_HISTORICO_NAO_COMPARAVEL_A_WINNER_U75",
        "experiment": "pooled_asset_conditional_cash_reversal_v1",
        "snapshot_sha256": manifest["snapshot_sha256"],
        "source_as_of": bars["bar_snapshot_as_of_end"],
        "source_symbols": len(symbols),
        "eligible_symbols": len(frames),
        "excluded_symbols": exclusions,
        "contains_CAPR": False,
        "prediction_horizon": "next_open_to_following_open",
        "execution_assumption": "next_session_regular_open_prices",
        "cost_per_side_assumption": TRADE_FEE_PER_SIDE,
        "features": list(FEATURES),
        "folds": {
            "train": [str(train_days[0].date()), str(train_days[-1].date())],
            "validation": [str(val_days[0].date()), str(val_days[-1].date())],
            "test": [str(test_days[0].date()), str(test_days[-1].date())],
            "purge_sessions": PURGE_SESSIONS,
        },
        "validation_predictive": val_scores,
        "validation_risk_preferences": selection,
        "risk_preference_selected_on_validation": selected,
        "test_predictive": test_scores,
        "test_policy": metrics,
        "cautions": [
            "Universo 67 historico, snapshot termina 2026-10-06. U75 de producao termina 2026-11-09.",
            "Nao inclui CAPR; nao avalia nem previne a suspensao do ativo.",
            "Fronteira de universo ex post implica vies de sobrevivencia/seleção.",
            "OHLCV normalizado por splits; dividendos nao modelados no retorno executado.",
            "Precos de abertura sao apenas proxy, sem spread, impacto ou cheque de volume.",
            "CASH rende 0 e custos 5bps/lado sao hipoteticos.",
            "Fold final exploratorio nao e uma amostra prospectiva nova.",
            "Nao compara patrimonio com Winner Control e nao autoriza ordens.",
        ],
    }
    (output / "cash_ou_posicao_u67_resumo.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )
    print("PILOTO_CASH_POOLED_RESULT_JSON=" + json.dumps({
        "source_as_of": result["source_as_of"],
        "source_symbols": result["source_symbols"],
        "eligible_symbols": result["eligible_symbols"],
        "folds": result["folds"],
        "selected_aversion": selected,
        "validation_predictive": val_scores,
        "test_predictive": test_scores,
        "test_policy": metrics,
    }, ensure_ascii=False, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("out/piloto_cash_pooled_u67"))
    arguments = parser.parse_args()
    run(arguments.root, arguments.output)
