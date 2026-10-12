"""Contratos locais da pesquisa incremental U67+n."""
import numpy as np
import pandas as pd
import pytest
import pesquisar_expansao_u67_spyder as expansion

from pesquisar_expansao_u67_spyder import (
    _capital_delta,
    _continuity_audit,
    _merge_utility_caches,
    _ranker_dataset,
    _forward_backward_subset_search,
    _fit_quality_row,
    _compound_fold_capital,
    _safe_rank_correlation,
    _deflated_sharpe_ratio,
    _pbo_cscv,
    _dynamic_transition_rows,
)


def test_merge_cache_preserva_scores_u67_e_insere_candidato_na_ordem_combinada():
    timestamp = pd.Timestamp("2026-01-05", tz="UTC")
    baseline_symbols = ["AAA", "CCC"]
    combined_symbols = ["AAA", "BBB", "CCC"]
    baseline = {timestamp: np.array([0.0, 1.5, 3.5])}
    candidates = {"BBB": {timestamp: np.array([0.0, 2.5])}}

    merged = _merge_utility_caches(
        baseline,
        baseline_symbols,
        candidates,
        combined_symbols,
    )

    assert merged[timestamp].tolist() == pytest.approx([0.0, 1.5, 2.5, 3.5])


def test_continuidade_rejeita_sessao_ausente_sem_reconstruir_serie():
    dates = pd.date_range("2016-01-04", periods=8, freq="B", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.0,
            "volume": 1000.0,
        },
        index=dates.delete(4),
    )
    audit = _continuity_audit("NEW", frame, dates)
    assert audit["status"] == "rejected"
    assert audit["reason"] == "discontinuous_history"
    assert audit["missing_required_sessions"] == 1


def test_delta_capital_e_calculado_contra_baseline_fixo():
    delta, rate = _capital_delta(125.0, 100.0)
    assert delta == pytest.approx(25.0)
    assert rate == pytest.approx(0.25)


def test_ranker_dataset_cria_relevancia_cross_sectional_por_data():
    dates = pd.date_range("2020-01-01", periods=180, freq="B", tz="UTC")
    frames = {}
    for index, symbol in enumerate(["AAA", "BBB", "CCC"], start=1):
        close = pd.Series(50.0 + index + np.linspace(0.0, 25.0 * index, len(dates)), index=dates)
        frames[symbol] = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000.0 + index * 10_000.0,
            },
            index=dates,
        )

    dataset = _ranker_dataset(frames)
    assert not dataset.empty
    assert set(dataset["symbol"]) == {"AAA", "BBB", "CCC"}
    assert dataset.groupby("date")["symbol"].count().min() >= 3
    assert dataset["relevance"].between(0, 4).all()



def test_amostra_congelada_mantem_500_sem_gate_do_catalogo_atual(tmp_path, monkeypatch):
    assets = [f"X{index:03d}" for index in range(500)]
    manifest = tmp_path / "sample.csv"
    pd.DataFrame({"asset": assets}).to_csv(manifest, index=False)
    monkeypatch.setattr(expansion, "CANDIDATE_SAMPLE_MANIFEST", manifest)

    catalog = pd.DataFrame(
        {
            "asset": [assets[0]],
            "stable_key": ["known"],
            "name": ["known asset"],
            "exchange": ["NYSE"],
        }
    )
    sample = expansion._load_candidate_sample(catalog)

    assert sample["asset"].tolist() == assets
    assert sample["sample_order"].tolist() == list(range(1, 501))
    assert bool(sample.loc[0, "catalog_currently_eligible"]) is True
    assert bool(sample.loc[1, "catalog_currently_eligible"]) is False



def test_forward_backward_subset_search_encontra_interacao_conjunta_melhor():
    table = {
        (): 100.0,
        ("A",): 130.0,
        ("B",): 125.0,
        ("C",): 110.0,
        ("A", "B"): 180.0,
        ("A", "C"): 135.0,
        ("B", "C"): 140.0,
        ("A", "B", "C"): 170.0,
    }

    def score(assets):
        return table[tuple(assets)]

    result = _forward_backward_subset_search(
        ["A", "B", "C"],
        score,
        baseline_capital=100.0,
        improvement_tol=0.0,
    )

    assert result["forward"]["assets"] == ["A", "B"]
    assert result["forward"]["ending_capital"] == pytest.approx(180.0)
    assert result["backward_from_all"]["assets"] == ["A", "B"]
    assert result["selected"]["assets"] == ["A", "B"]
    assert result["selected"]["ending_capital"] == pytest.approx(180.0)



def test_fit_quality_detecta_overfit_sem_limiar_arbitrario():
    dates = pd.date_range("2020-01-01", periods=8, freq="B", tz="UTC")
    frame = pd.DataFrame(
        {
            "forward_risk_adjusted_utility": [
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0
            ]
        },
        index=dates,
    )
    row = _fit_quality_row(
        universe_role="candidate",
        symbol="AAA",
        fold_id=1,
        frame=frame,
        train_dates=dates[:4],
        decision_dates=dates[4:],
        fit_diagnostics={"train": {"rows": 4, "mae": 0.05, "rmse": 0.10}},
        oos_diagnostics={"rows": 4, "mae": 0.60, "rmse": 0.60},
    )
    assert row["train_skill_vs_naive"] > 0
    assert row["oos_skill_vs_naive"] <= 0
    assert row["overfit_warning"] is True
    assert row["underfit_warning"] is False


def test_fit_quality_detecta_underfit_contra_media_ingenua():
    dates = pd.date_range("2020-01-01", periods=8, freq="B", tz="UTC")
    frame = pd.DataFrame(
        {"forward_risk_adjusted_utility": [0.0, 1.0] * 4},
        index=dates,
    )
    row = _fit_quality_row(
        universe_role="candidate",
        symbol="AAA",
        fold_id=1,
        frame=frame,
        train_dates=dates[:4],
        decision_dates=dates[4:],
        fit_diagnostics={"train": {"rows": 4, "mae": 0.60, "rmse": 0.60}},
        oos_diagnostics={"rows": 4, "mae": 0.60, "rmse": 0.60},
    )
    assert row["train_skill_vs_naive"] <= 0
    assert row["oos_skill_vs_naive"] <= 0
    assert row["underfit_warning"] is True


def test_compound_fold_capital_usa_somente_folds_informados():
    metrics = {
        "folds": [
            {"fold_id": 1, "strategy_return": 1.0},
            {"fold_id": 2, "strategy_return": 2.0},
            {"fold_id": 3, "strategy_return": 99.0},
        ]
    }
    assert _compound_fold_capital(metrics, [1, 2], 100.0) == pytest.approx(600.0)



def test_safe_rank_correlation_identifica_ordem_perfeita_e_invertida():
    left = np.array([1.0, 2.0, 3.0, 4.0])
    assert _safe_rank_correlation(left, left) == pytest.approx(1.0)
    assert _safe_rank_correlation(left, left[::-1]) == pytest.approx(-1.0)


def test_deflated_sharpe_ratio_retorna_probabilidade_valida():
    returns = pd.Series(
        [0.01, 0.02, -0.005, 0.015, 0.01, 0.005, -0.002] * 40
    )
    trials = np.array([0.01, 0.02, 0.03, 0.015, 0.025])
    result = _deflated_sharpe_ratio(returns, trials, 5.0)

    assert 0.0 <= result["probabilistic_sharpe_ratio"] <= 1.0
    assert 0.0 <= result["deflated_sharpe_ratio"] <= 1.0
    assert result["observations"] == len(returns)


def test_pbo_cscv_completa_todas_as_divisoes():
    rng = np.random.default_rng(42)
    matrix = pd.DataFrame(
        rng.normal(0.001, 0.01, size=(100, 4)),
        columns=["S1", "S2", "S3", "S4"],
    )
    summary, rows = _pbo_cscv(
        matrix,
        blocks=4,
        metric="log_growth",
    )

    assert summary["expected_splits"] == 6
    assert summary["completed_splits"] == 6
    assert len(rows) == 6
    assert 0.0 <= summary["pbo"] <= 1.0


def test_dynamic_transition_rows_registra_adicionar_manter_remover():
    stages = [
        {
            "stage": "one",
            "holdout_fold": 2,
            "selected_assets": ["A", "B"],
        },
        {
            "stage": "two",
            "holdout_fold": 3,
            "selected_assets": ["B", "C"],
        },
    ]
    rows = _dynamic_transition_rows(stages)
    second = rows.loc[rows["stage"].eq("two")]
    actions = dict(zip(second["asset"], second["action"]))

    assert actions == {"A": "REMOVER", "B": "MANTER", "C": "ADICIONAR"}
