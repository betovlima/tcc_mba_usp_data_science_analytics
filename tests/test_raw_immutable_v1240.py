"""Testes de preservacao estrita RAW para v1.24.0. Sem rede."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from reproducao.dados import SnapshotPaths, _normalize_alpaca_frame
from reproducao.preparacao import load_raw_bar_file, prepare_model_frames


def make_raw(tmp_path: Path, values: list[dict]) -> Path:
    path = tmp_path / "AAA.csv"
    pd.DataFrame(values).to_csv(
        path, index=False, float_format="%.17g",
    )
    return path


def bar(timestamp: str = "2026-01-02T05:00:00Z", *,
        price: float = 51.378186457421, volume: float = 1234.0) -> dict:
    return {
        "timestamp": timestamp,
        "open": price, "high": price * 1.02,
        "low": price * .98, "close": price,
        "volume": volume,
    }


def test_strict_raw_reader_preserves_float64_exact_values(tmp_path: Path) -> None:
    target = make_raw(tmp_path, [
        bar("2026-01-02T05:00:00Z"),
        bar("2026-01-05T05:00:00Z", price=8.478125),
    ])
    before = target.read_bytes()
    result = load_raw_bar_file(
        target, float_precision="round_trip", strict_raw=True,
    )
    assert len(result) == 2
    assert result.iloc[0]["close"] == 51.378186457421
    assert result.iloc[1]["close"] == 8.478125
    assert target.read_bytes() == before


@pytest.mark.parametrize("rows,error", [
    ([bar(),bar()], "duplicados"),
    ([bar("2026-01-05T05:00:00Z"),bar("2026-01-02T05:00:00Z")],
     "fora de ordem"),
    ([bar(price=-2.0)], "nao positivo"),
    ([bar(volume=-1.0)], "negativo"),
    ([bar(price=float("nan"))], "ausente"),
])
def test_strict_reader_refuses_invalid_rows_without_repair(
    tmp_path: Path, rows: list[dict], error: str,
) -> None:
    path = make_raw(tmp_path, rows)
    unchanged = path.read_bytes()
    with pytest.raises((RuntimeError, ValueError), match=error):
        load_raw_bar_file(path, strict_raw=True)
    assert path.read_bytes() == unchanged


def test_strict_download_frame_never_drops_unexpected_rows() -> None:
    frame = pd.DataFrame([
        bar("2026-01-02T05:00:00Z"),
        bar("2026-01-05T05:00:00Z"),
    ])
    strict = _normalize_alpaca_frame(frame, "AAA", strict_raw=True)
    assert len(strict) == 2
    assert float(strict.loc[0, "close"]) == 51.378186457421
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="duplicado"):
        _normalize_alpaca_frame(duplicate, "AAA", strict_raw=True)
    invalid = frame.copy()
    invalid.loc[1, "volume"] = -11.0
    with pytest.raises(RuntimeError, match="volume"):
        _normalize_alpaca_frame(invalid, "AAA", strict_raw=True)


def test_raw_prepare_keeps_price_and_volume_despite_split_and_dividend(
    tmp_path: Path,
) -> None:
    paths = SnapshotPaths.from_root(tmp_path / "snapshot")
    paths.ensure()
    raw = paths.raw_bars / "AAA.csv"
    pd.DataFrame([
        bar("2026-01-02T05:00:00Z",price=100.,volume=1000),
        bar("2026-01-05T05:00:00Z",price=10.,volume=10000),
    ]).to_csv(raw,index=False)
    actions_path = paths.corporate_actions / "AAA.csv"
    actions = pd.DataFrame([
        {"action_type": "forward_split", "ex_date": "2026-01-05",
         "old_rate": 1, "new_rate": 10, "symbol": "AAA"},
        {"action_type": "cash_dividend", "ex_date": "2026-01-05",
         "old_rate": None, "new_rate": None, "symbol": "AAA"},
    ])
    actions.to_csv(actions_path, index=False)
    before_raw = raw.read_bytes()
    before_actions = actions_path.read_bytes()
    prepared, exclusions, details, audit = prepare_model_frames(
        paths, assets=("AAA",), strict_raw=True,
        apply_split_adjustments=False, csv_float_precision="round_trip",
    )
    assert not exclusions
    assert len(details) == 1
    assert audit["price_transform_policy"] == "alpaca_raw_unchanged"
    assert audit["actual_total_splits_applied"] == 0
    assert prepared["AAA"]["close"].tolist() == [100.,10.]
    assert prepared["AAA"]["volume"].tolist() == [1000.,10000.]
    assert raw.read_bytes() == before_raw
    assert actions_path.read_bytes() == before_actions


def test_historical_default_split_normalization_still_reproducible(
    tmp_path: Path,
) -> None:
    paths = SnapshotPaths.from_root(tmp_path / "snapshot")
    paths.ensure()
    pd.DataFrame([
        bar("2026-01-02T05:00:00Z",price=100.,volume=1000),
        bar("2026-01-05T05:00:00Z",price=10.,volume=10000),
    ]).to_csv(paths.raw_bars / "AAA.csv", index=False)
    pd.DataFrame([{
        "action_type":"forward_split","ex_date":"2026-01-05",
        "old_rate":1,"new_rate":10,
    }]).to_csv(paths.corporate_actions / "AAA.csv", index=False)
    result, _, _, audit = prepare_model_frames(paths, assets=("AAA",))
    assert result["AAA"]["close"].iloc[0] == 10.0
    assert result["AAA"]["volume"].iloc[0] == 10000.0
    assert audit["actual_total_splits_applied"] == 1
