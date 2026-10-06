"""MODELAGEM MATEMATICA DA ASSINATURA CONTEXTUAL - v1.18.0-dev.1.

Nao executa novo backtest, nao procura ativos e nao altera o universo.
Consolida apenas resultados ja observados.

Modelo de desenvolvimento:
A = 1[beats_best_share > 0]
S = 1 - mean(rank(beats_best_share),
             rank(abs(score_corr_best)),
             rank(score_std))

Os ranks sao calculados dentro de cada coorte, apenas entre candidatos ativos.
A regressao logistica serve somente para calibrar uma probabilidade de
desenvolvimento. Validacao externa ainda exige uma amostra futura intocada.

Execute no Spyder com F5 ou pelas celulas # %%.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pesquisas.directional_change_lightgbm import (
    criar_pacote_analise,
    sinal_sonoro_conclusao,
)


# %% 0 - Configuracao
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dados" / "assinatura_matematica"
OUT = ROOT / "output" / "assinatura_matematica"
FIG = OUT / "graficos"

VERSION = "1.18.0-dev.1"
SCHEMA = "contextual-marginal-signature-math-v1"
BOOTSTRAP_REPS = 3_000
RANDOM_SEED = 42

B3_FILE = DATA / "contextual_batch3.csv"
SMART20_FILE = DATA / "contextual_smart20.csv"

# Controles auxiliares ja produzidos por campanhas anteriores.
STATIC_FILE = (
    ROOT / "output" / "busca_ativos" / "intelligent_known_training.csv"
)
TEMPORAL_FILE = (
    ROOT
    / "output"
    / "analise_standalone_ativos_58m"
    / "standalone_trade_details.csv"
)

STATIC_FEATURES = [
    "cagr",
    "annual_volatility",
    "maximum_drawdown",
    "median_dollar_volume_log10",
    "positive_day_share",
    "momentum_252_median",
    "volatility_20_median",
    "trend_efficiency_20_median",
    "corr_spy",
    "beta_spy",
]

CORE_RAW_FEATURES = [
    "beats_best_share",
    "abs_corr_best",
    "score_std",
]


# %% 1 - Helpers
def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _safe_auc(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def _safe_ap(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if int(y.sum()) == 0:
        return None
    return float(average_precision_score(y, score))


def _percentiles(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"p025": None, "median": None, "p975": None}
    p = np.percentile(arr, [2.5, 50.0, 97.5])
    return {
        "p025": float(p[0]),
        "median": float(p[1]),
        "p975": float(p[2]),
    }


def _stratified_auc_bootstrap(frame, score_col, reps, rng):
    positives = frame[frame["positive"] == 1]
    negatives = frame[frame["positive"] == 0]
    if positives.empty or negatives.empty:
        return []
    result = []
    for _ in range(int(reps)):
        sample = pd.concat(
            [
                positives.iloc[
                    rng.integers(0, len(positives), len(positives))
                ],
                negatives.iloc[
                    rng.integers(0, len(negatives), len(negatives))
                ],
            ],
            ignore_index=True,
        )
        result.append(
            float(
                roc_auc_score(
                    sample["positive"],
                    sample[score_col],
                )
            )
        )
    return result


# %% 2 - Dados contextuais congelados
started = time.perf_counter()

for path in (B3_FILE, SMART20_FILE):
    if not path.exists():
        raise RuntimeError(
            "Dataset congelado ausente: "
            f"{path}. Atualize a branch antes de executar."
        )

b3 = pd.read_csv(B3_FILE)
smart20 = pd.read_csv(SMART20_FILE)
contextual = pd.concat([b3, smart20], ignore_index=True)

contextual["abs_corr_best"] = pd.to_numeric(
    contextual["score_corr_best"],
    errors="coerce",
).abs()
contextual["beats_best_share"] = pd.to_numeric(
    contextual["beats_best_share"],
    errors="coerce",
)
contextual["score_std"] = pd.to_numeric(
    contextual["score_std"],
    errors="coerce",
)
contextual["capital_effect"] = pd.to_numeric(
    contextual["capital_effect"],
    errors="coerce",
)
contextual["positive"] = (
    contextual["capital_effect"] > 1e-12
).astype(int)
contextual["active"] = contextual["beats_best_share"] > 0.0
contextual["log_capital_effect"] = np.log1p(
    contextual["capital_effect"].clip(lower=-0.999999)
)


# %% 3 - Hurdle e score contextual
for cohort in contextual["cohort"].dropna().unique():
    mask = (
        (contextual["cohort"] == cohort)
        & contextual["active"]
    )
    for feature in CORE_RAW_FEATURES + [
        "positive_score_share",
        "score_mean",
    ]:
        contextual.loc[
            mask,
            f"rank_active_{feature}",
        ] = pd.to_numeric(
            contextual.loc[mask, feature],
            errors="coerce",
        ).rank(method="average", pct=True)

contextual["specialist_score"] = np.nan
active_mask = contextual["active"]
contextual.loc[
    active_mask,
    "specialist_score",
] = 1.0 - contextual.loc[
    active_mask,
    [
        "rank_active_beats_best_share",
        "rank_active_abs_corr_best",
        "rank_active_score_std",
    ],
].mean(axis=1)

active = contextual[contextual["active"]].copy()


# %% 4 - Replicacao das features por coorte
feature_directions = {
    "beats_best_share": -1.0,
    "abs_corr_best": -1.0,
    "score_std": -1.0,
    "positive_score_share": +1.0,
    "score_mean": +1.0,
}
feature_rows = []

for cohort, frame in active.groupby("cohort", sort=False):
    for feature, direction in feature_directions.items():
        score = (
            direction
            * pd.to_numeric(
                frame[f"rank_active_{feature}"],
                errors="coerce",
            )
        )
        rho = spearmanr(
            score,
            frame["capital_effect"],
            nan_policy="omit",
        )
        feature_rows.append(
            {
                "cohort": cohort,
                "feature": feature,
                "expected_direction": (
                    "lower_is_better"
                    if direction < 0
                    else "higher_is_better"
                ),
                "n_active": int(len(frame)),
                "positives": int(frame["positive"].sum()),
                "roc_auc": _safe_auc(
                    frame["positive"], score
                ),
                "average_precision": _safe_ap(
                    frame["positive"], score
                ),
                "spearman_vs_capital_effect": _finite(
                    rho.statistic
                ),
            }
        )

feature_validation = pd.DataFrame(feature_rows)


# %% 5 - Robustez por coorte + bootstrap
rng = np.random.default_rng(RANDOM_SEED)
cohort_rows = []

for cohort, frame in active.groupby("cohort", sort=False):
    rho = spearmanr(
        frame["specialist_score"],
        frame["capital_effect"],
        nan_policy="omit",
    )
    boot = _stratified_auc_bootstrap(
        frame,
        "specialist_score",
        BOOTSTRAP_REPS,
        rng,
    )
    cohort_rows.append(
        {
            "cohort": cohort,
            "n_total": int(
                len(
                    contextual[
                        contextual["cohort"] == cohort
                    ]
                )
            ),
            "n_active": int(len(frame)),
            "n_dormant": int(
                (
                    ~contextual.loc[
                        contextual["cohort"] == cohort,
                        "active",
                    ]
                ).sum()
            ),
            "positives": int(frame["positive"].sum()),
            "specialist_score_auc": _safe_auc(
                frame["positive"],
                frame["specialist_score"],
            ),
            "specialist_score_average_precision": _safe_ap(
                frame["positive"],
                frame["specialist_score"],
            ),
            "specialist_score_spearman_effect": _finite(
                rho.statistic
            ),
            "specialist_score_spearman_pvalue": _finite(
                rho.pvalue
            ),
            **{
                f"auc_bootstrap_{key}": value
                for key, value in _percentiles(boot).items()
            },
        }
    )

cohort_validation = pd.DataFrame(cohort_rows)


# %% 6 - Calibracao logistica de desenvolvimento
calibrator = LogisticRegression(
    C=1_000_000.0,
    solver="lbfgs",
    random_state=RANDOM_SEED,
    max_iter=10_000,
)
calibrator.fit(
    active[["specialist_score"]],
    active["positive"],
)
intercept = float(calibrator.intercept_[0])
slope = float(calibrator.coef_[0, 0])

active["development_probability"] = (
    calibrator.predict_proba(
        active[["specialist_score"]]
    )[:, 1]
)

pooled_auc = _safe_auc(
    active["positive"],
    active["specialist_score"],
)
pooled_ap = _safe_ap(
    active["positive"],
    active["specialist_score"],
)
pooled_rho = spearmanr(
    active["specialist_score"],
    active["capital_effect"],
    nan_policy="omit",
)

pooled_boot_auc = []
pooled_boot_rho = []

for _ in range(BOOTSTRAP_REPS):
    pieces = []
    for cohort in active["cohort"].unique():
        frame = active[active["cohort"] == cohort]
        index = rng.integers(
            0,
            len(frame),
            len(frame),
        )
        pieces.append(frame.iloc[index])

    sample = pd.concat(pieces, ignore_index=True)
    if sample["positive"].nunique() < 2:
        continue

    pooled_boot_auc.append(
        float(
            roc_auc_score(
                sample["positive"],
                sample["specialist_score"],
            )
        )
    )
    pooled_boot_rho.append(
        float(
            spearmanr(
                sample["specialist_score"],
                sample["capital_effect"],
                nan_policy="omit",
            ).statistic
        )
    )


# %% 7 - Leave-one-cohort-out
loco_rows = []

for train_cohort, test_cohort in (
    ("batch3", "smart20"),
    ("smart20", "batch3"),
):
    train = active[
        active["cohort"] == train_cohort
    ]
    test = active[
        active["cohort"] == test_cohort
    ]

    model = LogisticRegression(
        C=1_000_000.0,
        solver="lbfgs",
        random_state=RANDOM_SEED,
        max_iter=10_000,
    )
    model.fit(
        train[["specialist_score"]],
        train["positive"],
    )

    probability = model.predict_proba(
        test[["specialist_score"]]
    )[:, 1]
    prediction = probability >= 0.5

    loco_rows.append(
        {
            "train_cohort": train_cohort,
            "test_cohort": test_cohort,
            "train_intercept": float(
                model.intercept_[0]
            ),
            "train_slope": float(
                model.coef_[0, 0]
            ),
            "test_roc_auc": _safe_auc(
                test["positive"],
                probability,
            ),
            "test_average_precision": _safe_ap(
                test["positive"],
                probability,
            ),
            "test_balanced_accuracy_at_0_5": float(
                balanced_accuracy_score(
                    test["positive"],
                    prediction.astype(int),
                )
            ),
        }
    )

loco = pd.DataFrame(loco_rows)


# %% 8 - Controle negativo de features estaticas, se disponivel
static_rows = []

if STATIC_FILE.exists():
    static = pd.read_csv(STATIC_FILE)

    for train_cohort, test_cohort in (
        ("batch2", "batch3"),
        ("batch3", "batch2"),
    ):
        train = static[
            static["cohort"] == train_cohort
        ].dropna(subset=STATIC_FEATURES)
        test = static[
            static["cohort"] == test_cohort
        ].dropna(subset=STATIC_FEATURES)

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_SEED,
            ),
        )
        model.fit(
            train[STATIC_FEATURES],
            train["positive"].astype(int),
        )

        probability = model.predict_proba(
            test[STATIC_FEATURES]
        )[:, 1]
        prediction = probability >= 0.5

        static_rows.append(
            {
                "train_cohort": train_cohort,
                "test_cohort": test_cohort,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "test_positives": int(
                    test["positive"].sum()
                ),
                "roc_auc": _safe_auc(
                    test["positive"],
                    probability,
                ),
                "average_precision": _safe_ap(
                    test["positive"],
                    probability,
                ),
                "balanced_accuracy_at_0_5": float(
                    balanced_accuracy_score(
                        test["positive"],
                        prediction.astype(int),
                    )
                ),
            }
        )
else:
    print(
        "[warning] controle estatico ausente: "
        f"{STATIC_FILE}",
        flush=True,
    )

static_validation = pd.DataFrame(static_rows)


# %% 9 - Camada temporal explicativa, se disponivel
temporal_summary = None

if TEMPORAL_FILE.exists():
    temporal = pd.read_csv(TEMPORAL_FILE)

    numeric = [
        "entry_return_20",
        "entry_rsi_14",
        "entry_ema_distance_20",
        "entry_channel_position_20",
        "entry_since_confirmed_bottom_02pct_sessions",
        "holding_sessions_local",
        "mfe_from_entry",
        "mae_from_entry",
        "exit_drawdown_from_holding_peak",
        "sessions_peak_to_exit",
    ]
    for column in numeric:
        temporal[column] = pd.to_numeric(
            temporal[column],
            errors="coerce",
        )

    temporal_summary = {
        "trade_count": int(len(temporal)),
        "assets_with_trades": int(
            temporal["asset"].nunique()
        ),
        "entry_return20_median": _finite(
            temporal["entry_return_20"].median()
        ),
        "entry_rsi14_median": _finite(
            temporal["entry_rsi_14"].median()
        ),
        "entry_ema20_distance_median": _finite(
            temporal[
                "entry_ema_distance_20"
            ].median()
        ),
        "entry_channel20_median": _finite(
            temporal[
                "entry_channel_position_20"
            ].median()
        ),
        "entry_since_bottom02_median_sessions": _finite(
            temporal[
                "entry_since_confirmed_bottom_02pct_sessions"
            ].median()
        ),
        "share_return20_negative": _finite(
            (
                temporal["entry_return_20"] < 0
            ).mean()
        ),
        "share_rsi_below_0_5": _finite(
            (
                temporal["entry_rsi_14"] < 0.5
            ).mean()
        ),
        "share_below_ema20": _finite(
            (
                temporal["entry_ema_distance_20"] < 0
            ).mean()
        ),
        "holding_sessions_median": _finite(
            temporal[
                "holding_sessions_local"
            ].median()
        ),
        "mfe_median": _finite(
            temporal["mfe_from_entry"].median()
        ),
        "mae_median": _finite(
            temporal["mae_from_entry"].median()
        ),
        "exit_drawdown_from_peak_median": _finite(
            temporal[
                "exit_drawdown_from_holding_peak"
            ].median()
        ),
        "share_exit_within_2_sessions_of_peak": _finite(
            (
                temporal[
                    "sessions_peak_to_exit"
                ] <= 2
            ).mean()
        ),
    }
else:
    print(
        "[warning] morfologia temporal ausente: "
        f"{TEMPORAL_FILE}",
        flush=True,
    )


# %% 10 - Exportacao
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

contextual.to_csv(
    OUT / "signature_development_dataset.csv",
    index=False,
)
feature_validation.to_csv(
    OUT / "contextual_feature_validation.csv",
    index=False,
)
cohort_validation.to_csv(
    OUT / "contextual_cohort_validation.csv",
    index=False,
)
loco.to_csv(
    OUT / "leave_one_cohort_out.csv",
    index=False,
)
if not static_validation.empty:
    static_validation.to_csv(
        OUT / "static_negative_control_validation.csv",
        index=False,
    )


# %% 11 - Graficos
fig, ax = plt.subplots(figsize=(10.5, 6.0))
for label, group in active.groupby(
    "cohort",
    sort=False,
):
    ax.scatter(
        group["specialist_score"],
        100.0 * group["capital_effect"],
        label=str(label),
        alpha=0.8,
    )
    for _, row in group.iterrows():
        ax.annotate(
            str(row["asset"]),
            (
                row["specialist_score"],
                100.0 * row["capital_effect"],
            ),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
ax.axhline(0.0, linewidth=1.0)
ax.set_xlabel(
    "Score contextual de especialista (0-1)"
)
ax.set_ylabel(
    "Efeito marginal no capital (%)"
)
ax.set_title(
    "Assinatura contextual vs contribuicao marginal"
)
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(
    FIG / "specialist_score_vs_capital_effect.png",
    dpi=180,
)
fig.savefig(
    FIG / "specialist_score_vs_capital_effect.svg"
)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10.0, 5.8))
pivot = feature_validation.pivot(
    index="feature",
    columns="cohort",
    values="roc_auc",
)
pivot.plot(kind="bar", ax=ax)
ax.axhline(0.5, linewidth=1.0)
ax.set_ylim(0.0, 1.05)
ax.set_ylabel(
    "ROC AUC na direcao esperada"
)
ax.set_xlabel("Feature contextual")
ax.set_title(
    "Replicacao direcional das features entre coortes"
)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(
    FIG / "contextual_feature_auc_by_cohort.png",
    dpi=180,
)
fig.savefig(
    FIG / "contextual_feature_auc_by_cohort.svg"
)
plt.close(fig)

if not static_validation.empty:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    labels = (
        static_validation["train_cohort"]
        + " -> "
        + static_validation["test_cohort"]
    )
    ax.bar(
        labels,
        static_validation["roc_auc"],
    )
    ax.axhline(0.5, linewidth=1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("ROC AUC")
    ax.set_title(
        "Controle negativo: features estaticas entre coortes"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        FIG / "static_cross_cohort_auc.png",
        dpi=180,
    )
    fig.savefig(
        FIG / "static_cross_cohort_auc.svg"
    )
    plt.close(fig)


# %% 12 - Resultado matematico
probability_examples = {}
for score in (0.25, 0.50, 0.75, 1.00):
    z = intercept + slope * score
    probability_examples[str(score)] = float(
        1.0 / (1.0 + math.exp(-z))
    )

payload = {
    "research_version": VERSION,
    "execution_schema": SCHEMA,
    "status": (
        "development_not_external_validation"
    ),
    "new_backtest": False,
    "new_asset_search": False,
    "formula": {
        "activation": (
            "A = 1[beats_best_share > 0]"
        ),
        "active_rank_scope": (
            "within_cohort_among_active_candidates"
        ),
        "specialist_score": (
            "S = 1 - mean("
            "rank(beats_best_share), "
            "rank(abs(score_corr_best)), "
            "rank(score_std))"
        ),
        "logistic_active": (
            "P(deltaCapital>0|A=1) = "
            f"logistic({intercept:.12g} "
            f"+ {slope:.12g} * S)"
        ),
        "probability_examples": (
            probability_examples
        ),
    },
    "sample": {
        "contextual_total": int(
            len(contextual)
        ),
        "contextual_active": int(
            len(active)
        ),
        "contextual_dormant": int(
            (~contextual["active"]).sum()
        ),
        "positive_active": int(
            active["positive"].sum()
        ),
        "cohorts": (
            contextual.groupby(
                "cohort"
            ).size().to_dict()
        ),
    },
    "pooled_active": {
        "roc_auc": pooled_auc,
        "average_precision": pooled_ap,
        "spearman_score_vs_capital_effect": _finite(
            pooled_rho.statistic
        ),
        "spearman_pvalue": _finite(
            pooled_rho.pvalue
        ),
        "auc_bootstrap": _percentiles(
            pooled_boot_auc
        ),
        "spearman_bootstrap": _percentiles(
            pooled_boot_rho
        ),
    },
    "static_negative_control": static_rows,
    "cohort_validation": cohort_rows,
    "leave_one_cohort_out": loco_rows,
    "temporal_explanation_positive8": (
        temporal_summary
    ),
    "limitations": [
        (
            "The score was synthesized using "
            "already observed development cohorts."
        ),
        (
            "Smart20 is range-restricted because "
            "all 20 passed the prior signature screen."
        ),
        (
            "Batch3 contextual features are relative "
            "to U56 while its financial outcome is "
            "measured versus U59; within-cohort ranks "
            "reduce but do not eliminate this mismatch."
        ),
        (
            "The trade morphology exists only for "
            "positive8 assets and is explanatory, "
            "not a negative-control classifier."
        ),
        (
            "Prospective untouched assets are still "
            "required for external validation."
        ),
    ],
    "runtime_seconds": float(
        time.perf_counter() - started
    ),
}

with (
    OUT / "signature_math.json"
).open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        payload,
        handle,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

package = criar_pacote_analise(
    OUT,
    comparison_file="signature_math.json",
    execution_schema=SCHEMA,
    archive_name="pacote_assinatura_matematica_v118.zip",
)

print("=" * 78, flush=True)
print(
    "TCC - ASSINATURA MATEMATICA "
    f"{VERSION}",
    flush=True,
)
print(
    f"[sample] total={len(contextual)} "
    f"active={len(active)} "
    f"positive_active="
    f"{int(active['positive'].sum())}",
    flush=True,
)
print(
    "[signature] "
    "S=1-mean(rank(beats),"
    "rank(abs_corr),rank(std)) "
    f"pooled_auc={pooled_auc:.4f} "
    f"rho={float(pooled_rho.statistic):.4f}",
    flush=True,
)
print(
    f"[logistic] intercept={intercept:.6f} "
    f"slope={slope:.6f}",
    flush=True,
)
print("[cohort-validation]", flush=True)
print(
    cohort_validation.to_string(index=False),
    flush=True,
)
print("[leave-one-cohort-out]", flush=True)
print(
    loco.to_string(index=False),
    flush=True,
)
if not static_validation.empty:
    print(
        "[static-negative-control]",
        flush=True,
    )
    print(
        static_validation.to_string(index=False),
        flush=True,
    )
print(f"[package] pronto={package}", flush=True)
print(
    f"[done] seconds="
    f"{time.perf_counter()-started:.3f}",
    flush=True,
)
sinal_sonoro_conclusao()
