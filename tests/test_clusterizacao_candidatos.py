from __future__ import annotations

import numpy as np
import pandas as pd

from pesquisas.clusterizacao_candidatos import (
    FEATURES,
    _quality_table,
    _semantic_name,
)


def test_cluster_features_do_not_include_financial_outcome() -> None:
    assert "capital_pct_vs_u59" not in FEATURES
    assert all("capital" not in feature for feature in FEATURES)


def test_quality_table_selects_one_k() -> None:
    rng = np.random.default_rng(42)
    x = np.vstack(
        [
            rng.normal(loc=(-3.0, -3.0), scale=0.25, size=(20, 2)),
            rng.normal(loc=(0.0, 0.0), scale=0.25, size=(20, 2)),
            rng.normal(loc=(3.0, 3.0), scale=0.25, size=(20, 2)),
        ]
    )
    quality = _quality_table(x)
    assert quality["selected"].sum() == 1
    selected_k = int(
        quality.loc[quality["selected"], "k"].iloc[0]
    )
    assert selected_k in {2, 3, 4, 5, 6}
    assert quality["silhouette"].notna().all()
    assert quality["seed_stability_ari"].between(-1.0, 1.0).all()


def test_semantic_names_are_post_fit_interpretation() -> None:
    harmful = pd.Series(
        {
            "known_outcomes": 4,
            "median_capital_pct_vs_u59": -0.25,
            "positive_rate": 0.0,
            "harm10_rate": 0.75,
        }
    )
    positive = pd.Series(
        {
            "known_outcomes": 4,
            "median_capital_pct_vs_u59": 0.08,
            "positive_rate": 0.75,
            "harm10_rate": 0.0,
        }
    )
    survivor = pd.Series(
        {
            "known_outcomes": 4,
            "median_capital_pct_vs_u59": -0.02,
            "positive_rate": 0.25,
            "harm10_rate": 0.25,
        }
    )

    assert _semantic_name(harmful) == "Prejudiciais"
    assert _semantic_name(positive) == "Impulsionadores"
    assert _semantic_name(survivor) == "Sobreviventes"
