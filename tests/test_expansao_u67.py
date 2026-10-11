"""Contratos locais da pesquisa incremental U67+n."""
import numpy as np
import pandas as pd
import pytest

from pesquisar_expansao_u67_spyder import (
    _capital_delta,
    _continuity_audit,
    _merge_utility_caches,
    _ranker_dataset,
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