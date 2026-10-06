from __future__ import annotations

import numpy as np
import pandas as pd

from pesquisas.clusterizacao_candidatos import (
    FEATURES,
    _behavior_names,
    _quality_table,
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


def test_behavior_names_do_not_use_financial_outcomes() -> None:
    members = pd.DataFrame(
        {
            "cluster_id": [0, 0, 1, 1, 2, 2],
            "candidate_beats_u59_best_share": [
                0.04, 0.05, 0.22, 0.18, 0.02, 0.03
            ],
            "candidate_positive_score_share": [
                0.52, 0.56, 0.65, 0.70, 0.86, 0.84
            ],
            "candidate_score_std": [
                0.22, 0.20, 0.34, 0.31, 0.11, 0.12
            ],
            "candidate_score_mean": [
                0.02, 0.01, 0.17, 0.16, 0.12, 0.11
            ],
        }
    )
    names = _behavior_names(members)
    assert set(names.values()) == {
        "Oportunistas",
        "Dominantes",
        "Persistentes",
    }
