# -*- coding: utf-8 -*-
"""Teste de ML para CASH x posicao, somente sobre retornos do Control.

F1: treino. F2: selecao de modelo e aversao ao risco. F3: teste retrospectivo.
Esta simulacao sombra NAO e executavel: sem OHLCV dos 75 ativos nao e possivel
reconstruir as posicoes, preços de fills e custos de uma politica alternativa.
Nao interfere na Winner nem executa ordens.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_pinball_loss, mean_squared_error
from sklearn.pipeline import make_pipeline

FEATURES = (
    "spy_return_5", "spy_return_20", "spy_realized_volatility_20",
    "universe_breadth_5", "universe_breadth_20",
    "best_score", "raw_best_score", "best_vs_cash_gap",
    "best_vs_second_gap", "universe_score_mean", "universe_score_std",
    "positive_score_count", "finite_score_count",
    "best_score_zscore", "best_vs_second_zscore",
)
RISK_PREFERENCES = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
COST_BPS = 5  # somente cenario de sensibilidade, nao taxa Alpaca verificada
SEED = 20261110


def _model(name, tail):
    if name == "lightgbm_quant_7":
        return LGBMRegressor(
            objective="quantile" if tail else "regression",
            alpha=0.1, n_estimators=150, num_leaves=7, max_depth=4,
            min_child_samples=70, reg_lambda=10, learning_rate=0.025,
            random_state=SEED, verbosity=-1, n_jobs=2,
        )
    if name == "shallow":
        params = (5, 65, 20, 0.035, 125)
    elif name == "regularized":
        params = (7, 70, 10, 0.04, 130)
    else:
        raise ValueError("Modelo desconhecido " + name)
    nodes, samples, l2, rate, iterations = params
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        HistGradientBoostingRegressor(
            loss="quantile" if tail else "squared_error",
            quantile=0.1 if tail else None,
            max_leaf_nodes=nodes, min_samples_leaf=samples,
            l2_regularization=l2, learning_rate=rate,
            max_iter=iterations, early_stopping=False, random_state=SEED,
        ),
    )


def _score(y, predicted, tail):
    if tail:
        return float(mean_pinball_loss(y, predicted, alpha=0.1))
    return float(mean_squared_error(y, predicted))


def _predict(x, y, train, evaluation, tail):
    reference = float(np.quantile(y[train], 0.1) if tail else y[train].mean())
    candidates = {"constant": np.full(int(evaluation.sum()), reference)}
    for name in ("shallow", "regularized", "lightgbm_quant_7"):
        estimator = _model(name, tail)
        estimator.fit(x[train], y[train])
        candidates[name] = estimator.predict(x[evaluation])
    losses = {name: _score(y[evaluation], forecast, tail)
              for name, forecast in candidates.items()}
    return min(losses, key=losses.get), losses


def _forecast(x, y, develop, test, name, tail):
    if name == "constant":
        reference = np.quantile(y[develop], 0.1) if tail else y[develop].mean()
        return np.full(int(test.sum()), float(reference))
    predictor = _model(name, tail)
    predictor.fit(x[develop], y[develop])
    return predictor.predict(x[test])


def _shadow(mean_log_return, tail_q10, control_return, risk_aversion):
    """Duas decisoes a partir de uma previsao: CASH -> BUY e HOLD -> CASH.

    Usa retorno diario JA OBSERVADO do Control como proxy de exposicao. Nao
    reconstrói o ativo que realmente seria detido, e nao avalia outro ranking.
    """
    exposed = False
    returns, actions, utilities = [], [], []
    fee = COST_BPS / 10000
    for forecast, quantile, realized in zip(
        mean_log_return, tail_q10, control_return
    ):
        utility = float(forecast) - risk_aversion * max(0., -float(quantile))
        next_exposed = utility > (fee if not exposed else -fee)
        action = (
            "BUY" if next_exposed and not exposed else
            "HOLD" if next_exposed else
            "SELL_TO_CASH" if exposed else "STAY_CASH"
        )
        # Dado de retorno e custos nao sao um replay real da corretora.
        period_return = (float(realized) if next_exposed else 0.)
        if next_exposed != exposed:
            period_return -= fee
        returns.append(period_return)
        actions.append(action)
        utilities.append(utility)
        exposed = next_exposed
    return np.asarray(returns), np.asarray(actions), np.asarray(utilities)


def _metrics(returns):
    wealth = np.cumprod(1. + returns)
    peak = np.maximum.accumulate(np.r_[1., wealth])[1:]
    return {
        "growth_multiple": float(wealth[-1]),
        "max_drawdown": float(np.min(wealth / peak - 1.)),
    }


def estudar_cash_posicao_ml(predictions, folder: Path):
    """Executa sem alterar o Control e salva dentro do ZIP fixo do Spyder."""
    folder = Path(folder)
    df = predictions.reset_index(drop=True)
    required = set(FEATURES) | {
        "date", "decision_session", "selected_asset",
        "walk_forward_fold", "daily_return",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Atributos nao encontrados: " + ", ".join(sorted(missing)))
    dates = pd.to_datetime(df["date"], errors="raise")
    decisions = pd.to_datetime(df["decision_session"], errors="raise")
    if (len(df) != 1584 or not dates.is_monotonic_increasing
            or dates.duplicated().any() or not (decisions < dates).all()):
        raise ValueError("Referência de producao/cronologia invalida")
    folds = pd.to_numeric(
        df["walk_forward_fold"], errors="raise"
    ).astype(int).to_numpy()
    if (not np.array_equal(np.unique(folds), [1, 2, 3])
            or (np.diff(folds) < 0).any()):
        raise ValueError("Folds fora da cronologia 1 -> 2 -> 3")
    returns = pd.to_numeric(df["daily_return"], errors="raise").to_numpy(float)
    if not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError("Retornos nao-finitos ou impossiveis")
    x = df.loc[:, list(FEATURES)].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).to_numpy(float)
    train, validate, test = folds == 1, folds == 2, folds == 3
    develop = folds <= 2
    if min(train.sum(), validate.sum(), test.sum()) < 200:
        raise ValueError("Amostra insuficiente em um dos tres folds")
    log_returns = np.log1p(returns)
    chosen_mean, scores_mean = _predict(
        x, log_returns, train, validate, tail=False
    )
    chosen_tail, scores_tail = _predict(
        x, returns, train, validate, tail=True
    )
    forecast_val_mean = _forecast(
        x, log_returns, train, validate, chosen_mean, tail=False
    )
    forecast_val_tail = _forecast(
        x, returns, train, validate, chosen_tail, tail=True
    )
    risk_validation = {}
    for aversion in RISK_PREFERENCES:
        shadow, _, _ = _shadow(
            forecast_val_mean, forecast_val_tail, returns[validate], aversion
        )
        m = _metrics(shadow)
        risk_validation[str(aversion)] = {
            "objective_log_growth_minus_2_maxdd": (
                float(np.log(m["growth_multiple"]) + 2. * m["max_drawdown"])
            ),
            **m,
        }
    selected_risk = max(
        RISK_PREFERENCES,
        key=lambda lam: risk_validation[str(lam)][
            "objective_log_growth_minus_2_maxdd"
        ],
    )
    estimated_mean = _forecast(
        x, log_returns, develop, test, chosen_mean, tail=False
    )
    estimated_tail = _forecast(
        x, returns, develop, test, chosen_tail, tail=True
    )
    actual = returns[test]
    shadow, actions, utilities = _shadow(
        estimated_mean, estimated_tail, actual, selected_risk
    )
    control_metrics = _metrics(actual)
    shadow_metrics = _metrics(shadow)
    results = pd.DataFrame({
        "session": dates.loc[test].dt.strftime("%Y-%m-%d").to_numpy(),
        "winner_asset": df.loc[test, "selected_asset"].to_numpy(),
        "actual_control_return": actual,
        "estimated_log_return": estimated_mean,
        "estimated_q10": estimated_tail,
        "estimated_utility": utilities,
        "shadow_action": actions,
        "shadow_return_proxy": shadow,
    })
    results.to_csv(
        folder / "reversao_ml_cash_posicao_oos.csv", index=False
    )
    capr = results.loc[results["session"] == "2026-07-30"]
    capr_case = (
        capr.iloc[0][[
            "session", "winner_asset", "actual_control_return",
            "estimated_log_return", "estimated_q10", "shadow_action",
        ]].to_dict() if len(capr) == 1 else None
    )
    baseline_tail = np.full(
        int(test.sum()), float(np.quantile(returns[develop], 0.1))
    )
    report = {
        "schema": "tcc_cash_position_ml_shadow_v1",
        "scope": "on_policy_Winner_daily_return_shadow_only",
        "folds": {str(f): int((folds == f).sum()) for f in (1, 2, 3)},
        "features": list(FEATURES),
        "validation_selected_mean": chosen_mean,
        "validation_selected_q10": chosen_tail,
        "validation_mean_losses": scores_mean,
        "validation_q10_pinball": scores_tail,
        "validation_risk_preferences": risk_validation,
        "risk_aversion_chosen_on_validation": selected_risk,
        "test_q10_pinball": _score(actual, estimated_tail, tail=True),
        "test_baseline_q10_pinball": _score(actual, baseline_tail, tail=True),
        "test_q10_empirical_coverage": float(np.mean(actual < estimated_tail)),
        "test_control": control_metrics,
        "test_shadow_nonexecutable": shadow_metrics,
        "test_cash_sessions": int(np.isin(
            actions, ["STAY_CASH", "SELL_TO_CASH"]
        ).sum()),
        "test_actions": pd.Series(actions).value_counts().to_dict(),
        "capr_2026_07_30": capr_case,
        "limitations": [
            "Nao e backtest executavel: faltam OHLCV e fills contrafactuais por ativo.",
            "A carteira Control nao teve dias CASH; estado CASH e hipotetico.",
            "O retorno do Control e uma proxy quando a politica sombra esta exposta.",
            "Custos de 5 bps por transicao hipotetica nao reproduzem a Alpaca.",
            "Suspensoes e iliquidez impossibilitam fills arbitrarios.",
            "Teste historico fold 3 ja foi examinado em outras pesquisas.",
            "Nao concluir generalizacao para os 75 ativos com apenas retornos da Winner.",
        ],
    }
    (folder / "reversao_ml_cash_posicao_resumo.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    ts = pd.to_datetime(results["session"])
    ax.plot(ts, np.cumprod(1. + actual), label="Winner Control")
    ax.plot(
        ts, np.cumprod(1. + shadow),
        label="ML: exposicao sombra, nao executavel",
    )
    ax.set(
        yscale="log", ylabel="Multiplicador desde inicio do fold 3",
        title="CASH versus posicao: primeira avaliacao ML retrospectiva",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(folder / "reversao_ml_cash_posicao.png", dpi=150)
    plt.close(fig)
    print(
        "[risco CASH] modelo:", chosen_mean, "/", chosen_tail,
        "| CASH no teste:", report["test_cash_sessions"],
        "| MaxDD sombra:", shadow_metrics["max_drawdown"],
        flush=True,
    )
    return report
