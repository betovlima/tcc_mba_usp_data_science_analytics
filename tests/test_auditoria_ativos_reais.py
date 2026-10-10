from __future__ import annotations

import pandas as pd

from reproducao.auditoria_ativos_reais import (
    construir_movimentos_casos,
    gerar_auditoria_ativos_reais,
    selecionar_casos_cronologicos,
)


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": _ts("2020-01-03"),
                "action": "BUY",
                "asset": "TSLA",
                "entry_timestamp": _ts("2020-01-03"),
                "decision_timestamp": _ts("2020-01-02"),
                "execution_price": 101.0,
                "final_action_score": 0.40,
                "top_1_asset": "TSLA",
                "top_1_score": 0.40,
                "top_2_asset": "NVDA",
                "top_2_score": 0.35,
                "top_3_asset": "VNCE",
                "top_3_score": 0.30,
                "walk_forward_fold": 1,
                "rotation_id": "r1",
            },
            {
                "timestamp": _ts("2020-01-07"),
                "action": "SELL",
                "asset": "TSLA",
                "entry_timestamp": _ts("2020-01-03"),
                "decision_timestamp": _ts("2020-01-06"),
                "entry_price": 101.0,
                "execution_price": 103.0,
                "holding_bars": 2,
                "position_return": 0.0198019802,
                "realized_pnl": 20.0,
                "current_score": 0.22,
                "best_asset": "NVDA",
                "best_score": 0.45,
                "best_vs_current_gap": 0.23,
                "walk_forward_fold": 1,
                "rotation_id": "r2",
            },
            {
                "timestamp": _ts("2020-01-08"),
                "action": "BUY",
                "asset": "NVDA",
                "entry_timestamp": _ts("2020-01-08"),
                "decision_timestamp": _ts("2020-01-07"),
                "execution_price": 51.0,
                "final_action_score": 0.45,
                "top_1_asset": "NVDA",
                "top_1_score": 0.45,
                "top_2_asset": "TSLA",
                "top_2_score": 0.32,
                "top_3_asset": "VNCE",
                "top_3_score": 0.28,
                "walk_forward_fold": 1,
                "rotation_id": "r3",
            },
            {
                "timestamp": _ts("2020-01-10"),
                "action": "SELL",
                "asset": "NVDA",
                "entry_timestamp": _ts("2020-01-08"),
                "decision_timestamp": _ts("2020-01-09"),
                "entry_price": 51.0,
                "execution_price": 52.0,
                "holding_bars": 2,
                "position_return": 0.0196078431,
                "realized_pnl": 19.0,
                "current_score": 0.20,
                "best_asset": "VNCE",
                "best_score": 0.31,
                "best_vs_current_gap": 0.11,
                "walk_forward_fold": 1,
                "rotation_id": "r4",
            },
            {
                "timestamp": _ts("2020-01-13"),
                "action": "BUY",
                "asset": "TSLA",
                "entry_timestamp": _ts("2020-01-13"),
                "decision_timestamp": _ts("2020-01-10"),
                "execution_price": 104.0,
                "final_action_score": 0.60,
                "walk_forward_fold": 1,
                "rotation_id": "r5",
            },
            {
                "timestamp": _ts("2020-01-15"),
                "action": "SELL",
                "asset": "TSLA",
                "entry_timestamp": _ts("2020-01-13"),
                "decision_timestamp": _ts("2020-01-14"),
                "entry_price": 104.0,
                "execution_price": 156.0,
                "holding_bars": 2,
                "position_return": 0.50,
                "realized_pnl": 500.0,
                "current_score": 0.30,
                "best_asset": "NVDA",
                "best_score": 0.50,
                "best_vs_current_gap": 0.20,
                "walk_forward_fold": 1,
                "rotation_id": "r6",
            },
        ]
    )


def _frames() -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    dates = pd.date_range(
        "2020-01-01",
        periods=16,
        freq="D",
        tz="UTC",
    )
    frames = {}
    for offset, asset in enumerate(("TSLA", "NVDA", "VNCE", "SPY")):
        base = 100.0 + offset * 20.0
        closes = [base + index for index in range(len(dates))]
        frames[asset] = pd.DataFrame(
            {
                "open": closes,
                "high": [value + 1 for value in closes],
                "low": [value - 1 for value in closes],
                "close": closes,
                "volume": [1_000_000 + index for index in range(len(dates))],
            },
            index=dates,
        )
    return frames, dates


def test_seleciona_primeira_posicao_sem_usar_resultado() -> None:
    cases = selecionar_casos_cronologicos(
        _trades(),
        case_assets=("TSLA", "NVDA"),
    )

    assert list(cases["asset"]) == ["TSLA", "NVDA"]
    tsla = cases.loc[cases["asset"] == "TSLA"].iloc[0]
    assert tsla["entry_execution_timestamp"] == _ts("2020-01-03")
    assert tsla["exit_execution_timestamp"] == _ts("2020-01-07")
    assert tsla["position_return"] < 0.50
    assert (
        tsla["selection_rule"]
        == "first_completed_position_chronologically"
    )


def test_movimentos_incluem_spy_sem_operacao_e_marcam_futuro() -> None:
    frames, dates = _frames()
    cases = selecionar_casos_cronologicos(
        _trades(),
        case_assets=("TSLA",),
    )

    movements = construir_movimentos_casos(
        cases,
        frames=frames,
        common_dates=dates,
        focus_assets=("TSLA", "NVDA", "VNCE", "SPY"),
        sessions_before=1,
        sessions_after=1,
    )

    assert set(movements["asset"]) == {"TSLA", "NVDA", "VNCE", "SPY"}

    decision = _ts("2020-01-02")
    at_decision = movements.loc[movements["timestamp"] == decision]
    assert set(at_decision["normalized_close"].round(12)) == {100.0}
    assert at_decision["available_at_entry_decision"].all()

    future = movements.loc[movements["timestamp"] > decision]
    assert not future.empty
    assert not future["available_at_entry_decision"].any()


def test_gerador_exporta_casos_reais_sem_ativo_hipotetico(tmp_path) -> None:
    frames, dates = _frames()
    metadata = gerar_auditoria_ativos_reais(
        tmp_path,
        trades=_trades(),
        frames=frames,
        common_dates=dates,
        case_assets=("TSLA", "NVDA"),
        focus_assets=("TSLA", "NVDA", "VNCE", "SPY"),
        sessions_before=1,
        sessions_after=1,
    )

    assert metadata["case_count"] == 2
    assert metadata["outcome_used_to_select_cases"] is False
    assert metadata["hypothetical_asset_used"] is False
    assert (tmp_path / "casos_ativos_reais.csv").exists()
    assert (tmp_path / "movimentos_ativos_reais.csv").exists()
    assert (tmp_path / "auditoria_ativos_reais.md").exists()
    assert len(metadata["files"]["figures"]) == 4
