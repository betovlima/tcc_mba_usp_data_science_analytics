"""v1.25.0 — integração contábil split-aware sem alterar legado."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.configuracao import CONFIG
from engine.rotacao import (
    _benchmark_pesos_iguais, _fator_custodia_entre_datas, _simular_exato,
)
from reproducao.visao_economica_splits import (
    construir_visao_economica,
    extrair_fatores_efetivos,
    fingerprint_frame,
    restaurar_execucao_raw,
)


def bars_with_forward_split(n: int = 90, k: int = 25) -> pd.DataFrame:
    days = pd.bdate_range("2020-01-02", periods=n, tz="UTC")
    close = np.full(n, 100., dtype=float)
    close[k:] /= 10.
    vol = np.full(n, 1000., dtype=float)
    vol[k:] *= 10.
    return pd.DataFrame({
        "open": close.copy(), "high": close.copy(),
        "low": close.copy(), "close": close.copy(),
        "volume": vol,
    }, index=days)


def actions(day: pd.Timestamp) -> list[dict]:
    return [{
        "action_type": "forward_split",
        "ex_date": day.date().isoformat(),
        "old_rate": 1, "new_rate": 10,
    }]


def no_fees(*_args) -> dict:
    return {"total_fee": 0.0}


def identity_slippage(value, *_args):
    return value


def test_economic_view_is_causal_and_does_not_modify_RAW() -> None:
    raw = bars_with_forward_split()
    original = raw.copy(deep=True)
    hash0 = fingerprint_frame(raw)
    k = 25
    events = extrair_fatores_efetivos(raw, actions(raw.index[k]))
    view = construir_visao_economica(raw, events)
    assert view.loc[raw.index[k-1], "close"] == 100.
    assert view.loc[raw.index[k], "close"] == 100.
    assert view.loc[raw.index[k], "volume"] == 1000.
    assert view.loc[raw.index[k-1], "volume"] == 1000.
    for column in ["open", "high", "low", "close", "volume"]:
        pd.testing.assert_series_equal(
            view[column].iloc[:k], raw[column].iloc[:k],
        )
    derived = view.copy(deep=True)
    derived["return_20"] = view["close"].pct_change(20)
    restored = restaurar_execucao_raw(derived, raw)
    assert restored.loc[raw.index[k], "close"] == 10.
    assert restored.loc[raw.index[k], "return_20"] == pytest.approx(0.)
    assert fingerprint_frame(raw) == hash0
    pd.testing.assert_frame_equal(raw, original, check_exact=True)


def test_custody_split_applies_only_if_position_held_before_ex_date() -> None:
    raw = bars_with_forward_split()
    events = tuple((date, factor) for date, factor in
                   extrair_fatores_efetivos(raw, actions(raw.index[25])))
    state = {"AAA": events}
    factor, cases = _fator_custodia_entre_datas(
        state, "AAA", raw.index[24], raw.index[25],
    )
    assert factor == 10.
    assert len(cases) == 1
    factor, cases = _fator_custodia_entre_datas(
        state, "AAA", raw.index[25], raw.index[27],
    )
    assert factor == 1.
    assert cases == []


def test_benchmark_adjusts_quantity_at_split_without_touching_RAW() -> None:
    raw = bars_with_forward_split()
    other = raw.copy()
    other.loc[:, ["open", "high", "low", "close"]] = 100.
    frames = {"AAA": raw, "BBB": other}
    state = {"AAA": extrair_fatores_efetivos(raw, actions(raw.index[25]))}
    corrected = _benchmark_pesos_iguais(
        frames, ["AAA", "BBB"], raw.index, 10000., CONFIG,
        no_fees, identity_slippage,
        split_events_by_symbol=state,
    )
    legacy = _benchmark_pesos_iguais(
        frames, ["AAA", "BBB"], raw.index, 10000., CONFIG,
        no_fees, identity_slippage,
    )
    assert corrected.iloc[0] == pytest.approx(10000.)
    assert corrected.iloc[-1] == pytest.approx(10000.)
    assert legacy.iloc[-1] == pytest.approx(5500.)
    assert corrected.index.equals(legacy.index)


def test_strategy_replay_hold_through_split_preserves_capital() -> None:
    raw = bars_with_forward_split()
    other = raw.copy()
    other.loc[:, ["open", "high", "low", "close"]] = 100.
    frames = {"AAA": raw, "BBB": other}
    state = {"AAA": extrair_fatores_efetivos(raw, actions(raw.index[25]))}
    result = _simular_exato(
        "split-test", lambda _date, _position, _holding: (1, .2),
        frames, ["AAA", "BBB"], raw.index, CONFIG,
        no_fees, identity_slippage,
        split_events_by_symbol=state,
    )
    assert result.metrics["strategy_ending_capital"] == pytest.approx(10000.)
    assert result.metrics["buy_hold_ending_capital"] == pytest.approx(10000.)
    assert result.metrics["split_events_applied_to_strategy"] == 1
    ledger = result.metrics["split_ledger"]
    assert ledger[0]["asset"] == "AAA"
    assert ledger[0]["shares_after"] == pytest.approx(
        ledger[0]["shares_before"] * 10,
    )
    assert result.metrics["split_accounting_mode"] == (
        "RAW_OHLCV_SHARE_QUANTITY_EX_DATE"
    )
    assert float(frames["AAA"].loc[raw.index[25], "close"]) == 10.


def test_invalid_split_fails_instead_of_silent_omission() -> None:
    raw = bars_with_forward_split()
    with pytest.raises(RuntimeError, match="Corporate Action"):
        extrair_fatores_efetivos(raw, [{
            **actions(raw.index[25])[0], "new_rate": 0,
        }])


def test_future_split_is_not_applied_to_old_features() -> None:
    raw = bars_with_forward_split()
    upcoming = [{
        **actions(raw.index[25])[0],
        "ex_date": "2025-12-01",
    }]
    assert extrair_fatores_efetivos(raw, upcoming) == ()
    view = construir_visao_economica(raw, ())
    pd.testing.assert_frame_equal(view, raw)
