# -*- coding: utf-8 -*-
"""Pesquisa de risco ML em dados CONGELADOS da Winner; jamais envia ordens.

As linhas do backtest representam a sessao executada em t; os indicadores
de decisao vieram de decision_session (t-1). O alvo e o retorno da carteira
Winner Control em t, nao o futuro do ativo caso ele tivesse sido mantido.

Folds do backtest: 1 desenvolvimento, 2 validacao cronologica, 3 teste OOS.
A selecao do modelo NUNCA consulta o fold 3. Sem stops ou gates de mercado.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, mean_pinball_loss
from sklearn.pipeline import make_pipeline

QUANTILE = 0.10
SEED = 20261110

# Todos estes campos constam do snapshot PREVIO a execucao de cada sessao.
# Nao utilizar close, strategy_equity, daily_return, realized_pnl,
# subsequent_position_return ou qualquer informacao do futuro como atributo.
EX_ANTE_FEATURES = (
    "spy_return_5", "spy_return_20", "spy_realized_volatility_20",
    "universe_breadth_5", "universe_breadth_20",
    "position_return_since_entry", "position_drawdown_from_peak",
    "position_mfe_so_far", "position_mae_so_far", "score_change_from_entry",
    "current_score", "best_score", "best_vs_second_gap",
    "best_vs_current_gap", "best_vs_cash_gap",
    "universe_score_mean", "universe_score_std", "current_asset_rank",
    "holding_days_at_decision", "days_current_not_top1",
    "consecutive_days_current_not_top1", "effective_switch_margin",
    "q_delta_final_vs_current",
)

REFERENCES = (
    ("Gu Kelly Xiu 2020", "https://academic.oup.com/rfs/article/33/5/2223/5758276"),
    ("Moreira Muir", "https://www.nber.org/papers/w22208"),
    ("Adams MacKay", "https://arxiv.org/abs/0710.3742"),
    ("HMM regime studies", "https://www.mdpi.com/1911-8074/13/12/311"),
    ("Quantile random forests 2026", "https://doi.org/10.1002/for.70146"),
)


def _params(leaves, min_leaf, regularization, learning_rate, iterations):
    return dict(
        loss="quantile", quantile=QUANTILE,
        max_leaf_nodes=leaves, min_samples_leaf=min_leaf,
        l2_regularization=regularization,
        learning_rate=learning_rate, max_iter=iterations,
        early_stopping=False, random_state=SEED,
    )


CANDIDATES = {
    "gb7": _params(7, 70, 10.0, .04, 130),
    "gb5": _params(5, 65, 25.0, .035, 120),
}


def _model(params):
    # Median and all missing-value treatment fit ONLY on training data.
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        HistGradientBoostingRegressor(**params),
    )


def _loss(y, q):
    return float(mean_pinball_loss(y, q, alpha=QUANTILE))


def _independent_block_interval(y, baseline, model, block_size=20, count=400):
    """Paired moving-block bootstrap, descriptive interval, no p-value claims."""
    rng = np.random.default_rng(SEED)
    n = len(y)
    if n < block_size * 2:
        return [None, None]
    block_starts = np.arange(n - block_size + 1)
    diffs = []
    for _ in range(count):
        idx = []
        while len(idx) < n:
            start = int(rng.choice(block_starts))
            idx.extend(range(start, start + block_size))
        ix = np.asarray(idx[:n], dtype=int)
        diffs.append(_loss(y[ix], baseline[ix]) - _loss(y[ix], model[ix]))
    return [float(v) for v in np.quantile(diffs, [.025, .975])]


def estudo_risco_ml(predictions, destination: Path):
    """Estima quantil inferior para o proximo pregão, sem alterar o Control.

    O experimento é intencionalmente limitado a previsao de risco da
    CARTEIRA que a Winner selecionou, nao de todos os 75 ativos.
    Exige OHLCV por ativo para um verdadeiro modelo pooled por ativo.
    """
    required = set(EX_ANTE_FEATURES) | {
        "date", "decision_session", "selected_asset",
        "daily_return", "walk_forward_fold",
    }
    absent = required - set(predictions.columns)
    if absent:
        raise ValueError("Risco ML: campos ausentes: " + ", ".join(sorted(absent)))
    d = predictions.reset_index(drop=True).copy()
    dates = pd.to_datetime(d["date"], errors="raise")
    previous = pd.to_datetime(d["decision_session"], errors="raise")
    if len(d) != 1584 or not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError("Risco ML: calendario de producao inconsistente.")
    if not (previous < dates).all():
        raise ValueError("Risco ML: decision_session deve preceder o alvo.")
    folds = pd.to_numeric(d["walk_forward_fold"], errors="raise").astype(int).to_numpy()
    if not np.array_equal(np.unique(folds), np.array([1, 2, 3])):
        raise ValueError("Risco ML: exigidos tres folds congelados.")
    if (np.diff(folds) < 0).any():
        raise ValueError("Risco ML: folds fora da ordem cronologica.")
    y = pd.to_numeric(d["daily_return"], errors="raise").to_numpy(float)
    if not np.isfinite(y).all() or (y <= -1).any():
        raise ValueError("Risco ML: retornos invalidos.")
    features = d.loc[:, EX_ANTE_FEATURES].apply(
        pd.to_numeric, errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    x = features.to_numpy(dtype=float)
    train1, val2, oos3 = folds == 1, folds == 2, folds == 3
    training = folds <= 2
    if min(train1.sum(), val2.sum(), oos3.sum()) < 200:
        raise ValueError("Risco ML: folds sem amostra suficiente.")
    # A referêcia estatistica e escolhida somente usando o fold de treino.
    base_val = float(np.quantile(y[train1], QUANTILE))
    val_scores = {
        "baseline_historical_q10": _loss(
            y[val2], np.full(val2.sum(), base_val),
        )
    }
    for name, params in CANDIDATES.items():
        fit = _model(params)
        fit.fit(x[train1], y[train1])
        val_scores[name] = _loss(y[val2], fit.predict(x[val2]))
    # Se ML nao supera o quantil incondicional na validacao, manter baseline.
    chosen = min(val_scores, key=val_scores.get)
    base_final = float(np.quantile(y[training], QUANTILE))
    q_baseline = np.full(int(oos3.sum()), base_final)
    fitted = None
    if chosen == "baseline_historical_q10":
        q_model = q_baseline.copy()
    else:
        fitted = _model(CANDIDATES[chosen])
        fitted.fit(x[training], y[training])
        q_model = fitted.predict(x[oos3])
    if not np.isfinite(q_model).all():
        raise RuntimeError("Risco ML: previsoes nao finitas.")
    actual = y[oos3]
    improvement = _loss(actual, q_baseline) - _loss(actual, q_model)
    interval = _independent_block_interval(actual, q_baseline, q_model)
    out = pd.DataFrame({
        "execution_session": d.loc[oos3, "date"].astype(str).values,
        "decision_session": d.loc[oos3, "decision_session"].astype(str).values,
        "winner_selected_asset": d.loc[oos3, "selected_asset"].astype(str).values,
        "realized_winner_daily_return": actual,
        "statistical_reference_q10": q_baseline,
        "learned_conditional_q10": q_model,
        "realized_below_conditional_q10": actual <= q_model,
        "extreme_loss_below_training_q10": actual <= base_final,
    })
    out.to_csv(destination / "reversao_ml_risco_teste_oos.csv", index=False)

    if fitted is not None:
        # Associacao preditiva em validacao apenas. NAO inferencia causal.
        # Nao usar o teste OOS para selecionar features ou hiperparametros.
        first_fit = _model(CANDIDATES[chosen])
        first_fit.fit(x[train1], y[train1])
        perm = permutation_importance(
            first_fit, x[val2], y[val2],
            scoring=make_scorer(
                mean_pinball_loss, greater_is_better=False, alpha=QUANTILE,
            ),
            n_repeats=3, random_state=SEED,
        )
        importance = pd.DataFrame({
            "feature": EX_ANTE_FEATURES,
            "validation_pinball_loss_increase": perm.importances_mean,
            "validation_importance_sd": perm.importances_std,
        }).sort_values(
            "validation_pinball_loss_increase", ascending=False,
        )
    else:
        importance = pd.DataFrame({
            "feature": EX_ANTE_FEATURES,
            "validation_pinball_loss_increase": np.nan,
            "validation_importance_sd": np.nan,
        })
    importance.to_csv(
        destination / "reversao_ml_risco_importancia_validacao.csv",
        index=False,
    )

    summary = {
        "schema": "tcc_ml_conditional_tail_risk_v1",
        "status": "predictive_diagnostic_only",
        "reference": "production_backtest_20261110",
        "scope": "next_session_Winner_Control_portfolio_return",
        "NOT": [
            "nao e o risco de um ativo mantido contrafactualmente",
            "nao e a previsao OHLCV dos 75 ativos",
            "nao e estrategia CASH validada",
            "nao e ordem de venda, stop ou gate",
        ],
        "decision_information": "features do fechamento anterior a execution_session",
        "target": "quantil 10% inferior do retorno observado da Winner no pregão seguinte",
        "quantile": QUANTILE,
        "train_fold_1_sessions": int(train1.sum()),
        "validation_fold_2_sessions": int(val2.sum()),
        "untouched_test_fold_3_sessions": int(oos3.sum()),
        "candidate_validation_pinball": val_scores,
        "selected_on_validation_only": chosen,
        "baseline_q10_folds_1_2": base_final,
        "test_baseline_pinball": _loss(actual, q_baseline),
        "test_model_pinball": _loss(actual, q_model),
        "test_absolute_pinball_improvement": improvement,
        "test_relative_pinball_improvement": (
            improvement / _loss(actual, q_baseline)
            if _loss(actual, q_baseline) > 0 else None
        ),
        "test_improvement_95pct_moving_block_interval": interval,
        "test_empirical_tail_coverage": float(np.mean(actual <= q_model)),
        "test_expected_tail_coverage": QUANTILE,
        "test_fraction_below_training_q10": float(np.mean(actual <= base_final)),
        "test_extreme_loss_sessions": int(np.sum(actual <= base_final)),
        "features_used": list(EX_ANTE_FEATURES),
        "feature_coverage_by_fold": {
            str(fold): {
                k: round(float(features.loc[folds == fold, k].notna().mean()), 4)
                for k in EX_ANTE_FEATURES
            } for fold in [1, 2, 3]
        },
        "references": [{"study": n, "url": u} for n, u in REFERENCES],
    }
    event = out.loc[out["execution_session"] == "2026-07-30"]
    if len(event) == 1:
        row = event.iloc[0]
        summary["capr_halt_2026_07_30"] = {
            "actual_winner_portfolio_return": float(
                row["realized_winner_daily_return"]
            ),
            "prior_day_conditional_q10": float(row["learned_conditional_q10"]),
            "reference_q10": float(row["statistical_reference_q10"]),
            "not_foreseen_by_model": bool(
                row["realized_winner_daily_return"]
                < row["learned_conditional_q10"]
            ),
            "caution": (
                "Halt e noticias podem produzir saltos nao antecipaveis "
                "por indicadores tecnicos diários; nenhuma ordem durante halt."
            ),
        }

    fig, ax = plt.subplots(figsize=(13, 5))
    focus = out.loc[
        (out["execution_session"] >= "2026-07-01") &
        (out["execution_session"] <= "2026-09-01")
    ]
    dt = pd.to_datetime(focus["execution_session"])
    ax.plot(dt, focus["realized_winner_daily_return"] * 100,
            label="Retorno realizado da Winner", linewidth=1.5)
    ax.plot(dt, focus["learned_conditional_q10"] * 100,
            label="Quantil inferior estimado antes do pregao", linewidth=1.7)
    ax.plot(dt, focus["statistical_reference_q10"] * 100,
            linestyle=":", label="Quantil historico de referencia")
    ax.axhline(0, color="grey", alpha=.35, linewidth=.8)
    ax.set(title="CAPR 2026: previsao de risco da carteira x queda observada",
           ylabel="Retorno diario (%)")
    ax.grid(alpha=.2)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(destination / "reversao_ml_risco_capr_oos.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    q_bucket = pd.qcut(
        pd.Series(q_model).rank(method="first"), q=5, labels=False,
    )
    calibration = pd.DataFrame({
        "risk_quintile": q_bucket, "predicted": q_model,
        "observed_below_q10": actual <= q_model,
    }).groupby("risk_quintile", as_index=False).agg(
        predicted_q10=("predicted", "mean"),
        observed_coverage=("observed_below_q10", "mean"),
    )
    calibration.to_csv(destination / "reversao_ml_risco_calibracao.csv", index=False)
    ax.bar(
        calibration["risk_quintile"].astype(str),
        calibration["observed_coverage"] * 100,
    )
    ax.axhline(QUANTILE * 100, linestyle="--", color="black",
               label="Cobertura esperada (10%)")
    ax.set(
        xlabel="Quintil do quantil inferior estimado no teste",
        ylabel="Frac. retornos abaixo do quantil previsto (%)",
        title="Risco OOS: calibracao do quantil 10%",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "reversao_ml_risco_calibracao.png", dpi=150)
    plt.close(fig)
    (destination / "reversao_ml_risco_resumo.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        "[risco ML] modelo:", chosen, "| pinball baseline:",
        round(summary["test_baseline_pinball"], 8),
        "| modelo:", round(summary["test_model_pinball"], 8),
        "| cobertura:", round(summary["test_empirical_tail_coverage"], 4),
        flush=True,
    )
    return summary
