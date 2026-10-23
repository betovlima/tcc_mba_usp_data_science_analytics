"""Testes temporais deterministas dos splits, sem Alpaca nem snapshots oficiais."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from auditar_temporalidade_splits_spyder import (
    compare_features_before_future_split,
)
from reproducao.dados import SnapshotPaths
from reproducao.preparacao import prepare_model_frames
from reproducao.splits_temporais import (
    apply_custody_split, known_as_of, recorded_effective_by,
    split_from_action,
)


def action(first_seen: str | None = None, *, kind: str = "forward_split") -> dict:
    event = {
        "action_type": kind, "symbol": "AAA",
        "ex_date": "2020-08-12", "old_rate": 1, "new_rate": 10,
        "process_date": "2020-07-01", "id": "split-1",
    }
    if first_seen is not None:
        event["first_seen_at_utc"] = first_seen
    return event


def synthetic_bars() -> pd.DataFrame:
    index = pd.bdate_range("2019-01-02", periods=440, tz="UTC")
    position = np.arange(len(index), dtype=float)
    close = 80.0 + position * .08 + np.sin(position * .19) * 3.0
    return pd.DataFrame({
        "open": close * 1.001,
        "high": close * 1.02,
        "low": close * .98,
        "close": close,
        "volume": 100000.0 + np.sin(position * .31) * 20000.0,
    }, index=index)


def test_process_date_is_not_proof_of_announcement_availability() -> None:
    # A API pode expor process_date; nao e log first_seen verificado.
    parsed = split_from_action(action())
    assert parsed is not None
    assert parsed.first_seen_at_utc is None
    assert known_as_of([parsed], "2020-07-28T12:00:00Z") == ()


def test_event_announced_after_decision_is_invisible_before_then_visible() -> None:
    event = split_from_action(action("2020-08-01T12:00:00Z"))
    assert event is not None
    assert known_as_of([event], "2020-07-28T20:00:00Z") == ()
    assert known_as_of([event], "2020-08-02T12:00:00Z") == (event,)
    assert recorded_effective_by([event], "2020-08-10T12:00:00Z") == ()
    assert recorded_effective_by([event], "2020-08-12T12:00:00Z") == (event,)


def test_event_announced_early_is_known_but_not_applied_to_custody_early() -> None:
    event = split_from_action(action("2020-07-20T12:00:00Z"))
    assert event is not None
    assert known_as_of([event], "2020-07-28T12:00:00Z") == (event,)
    with pytest.raises(ValueError, match="antes da ex_date"):
        apply_custody_split(
            quantity=Decimal("1"), average_unit_cost=Decimal("100"),
            event=event, valuation_time="2020-07-28T12:00:00Z",
        )


def test_split_changes_position_quantity_not_raw_prices_or_wealth() -> None:
    event = split_from_action(action("2020-08-01T12:00:00Z"))
    assert event is not None
    qty, cost, applied = apply_custody_split(
        quantity=Decimal("1"), average_unit_cost=Decimal("100"),
        event=event, valuation_time="2020-08-12T12:00:00Z",
    )
    assert qty == Decimal("10")
    assert cost == Decimal("10")
    assert qty * Decimal("10") == Decimal("1") * Decimal("100")
    with pytest.raises(ValueError, match="mais de uma vez"):
        apply_custody_split(
            quantity=qty, average_unit_cost=cost,
            event=event, valuation_time="2020-08-13T12:00:00Z",
            applied_event_keys=applied,
        )


def test_reverse_split_preserves_cost_basis_on_synthetic_position() -> None:
    item = action("2020-07-01T12:00:00Z", kind="reverse_split")
    item["old_rate"] = 10
    item["new_rate"] = 1
    event = split_from_action(item)
    assert event is not None
    qty, cost, _ = apply_custody_split(
        quantity=Decimal("20"), average_unit_cost=Decimal("5"),
        event=event, valuation_time="2020-08-12T12:00:00Z",
    )
    assert qty == Decimal("2")
    assert cost == Decimal("50")
    assert qty * cost == Decimal("100")


def test_future_split_proves_legacy_back_adjusts_but_raw_features_do_not() -> None:
    report = compare_features_before_future_split(synthetic_bars())
    assert report["feature_columns_compared"] >= 40
    assert report["feature_rows_compared"] > 10
    assert report["raw_preserved"]
    assert report["features_equal_before_event"]
    assert report["legacy_split_normalization_changes_pre_event_prices"]
    assert not report["score_and_rotation_parity_tested"]


def test_real_preparation_feature_inputs_independent_of_future_action(
    tmp_path: Path,
) -> None:
    """Duas copias, uma com split futuro: OHLCV e features devem ser iguais."""
    from engine.configuracao import CONFIG
    from engine.rotacao import ROTATION_FEATURES, construir_quadro_rotacao
    bars = synthetic_bars()
    source = []
    for variant in ("no_event", "with_unannounced_future"):
        paths = SnapshotPaths.from_root(tmp_path / variant)
        paths.ensure()
        pd.DataFrame({"timestamp": bars.index, **{
            name: bars[name].to_numpy() for name in
            ("open", "high", "low", "close", "volume")
        }}).to_csv(paths.raw_bars / "AAA.csv", index=False,
                    float_format="%.17g")
        if variant == "no_event":
            pd.DataFrame(columns=["action_type", "symbol", "ex_date",
                                  "old_rate", "new_rate"]).to_csv(
                paths.corporate_actions / "AAA.csv", index=False,
            )
        else:
            pd.DataFrame([action("2020-08-01T12:00:00Z")]).to_csv(
                paths.corporate_actions / "AAA.csv", index=False,
            )
        frames, excluded, _, audit = prepare_model_frames(
            paths, assets=("AAA",), strict_raw=True,
            apply_split_adjustments=False, csv_float_precision="round_trip",
        )
        assert excluded == []
        assert audit["actual_total_splits_applied"] == 0
        source.append((
            frames["AAA"], construir_quadro_rotacao(frames["AAA"], CONFIG),
        ))
    pd.testing.assert_frame_equal(source[0][0], source[1][0], check_exact=True)
    pd.testing.assert_frame_equal(
        source[0][1].loc[:, ROTATION_FEATURES],
        source[1][1].loc[:, ROTATION_FEATURES],
        check_exact=True,
    )


@pytest.mark.parametrize("invalid", [
    {**action(), "new_rate": 0},
    {**action(), "old_rate": -1},
    {**action(), "symbol": ""},
    {**action(), "ex_date": None},
])
def test_split_records_with_invalid_identity_or_ratio_rejected(invalid: dict) -> None:
    with pytest.raises(ValueError):
        split_from_action(invalid)
