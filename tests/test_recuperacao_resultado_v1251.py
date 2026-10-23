"""Contrato do resultado resumido e recuperação v1.25.1 sem reexecutar modelo."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from validar_backtest_economico_raw_spyder import (
    SNAPSHOT_SHA256, capital_do_resultado,
)
import recuperar_validacao_economica_raw_spyder as recovery


CAPITAL = 42_600_974.32


def fake_result(capital: float = CAPITAL) -> dict:
    return {
        "reproduction_version": "1.25.0",
        "status": "completed",
        "execution_schema": "fixed75-raw-economic-custody-v1",
        "analysis_end_date": "2026-10-06",
        "data_mode": "alpaca_raw_unchanged_split_economic_in_memory",
        "data": {"snapshot_sha256": SNAPSHOT_SHA256},
        "checkpoint": {"observed_capital": capital},
        "universe": {"effective_count": 75, "excluded_assets": []},
        "metrics": {
            "ending_capital": capital,
            "buy_hold_ending_capital": 12345.67,
            "folds": [{"fold": 1, "strategy_return": 0.3}],
        },
    }


def make_run(tmp_path: Path, *, cash: float = CAPITAL) -> Path:
    folder = (
        tmp_path / "output" / "validacao_economica_raw_75" / recovery.RUN_ID
    )
    folder.mkdir(parents=True)
    (folder / "reproducao_universo.json").write_text(
        json.dumps(fake_result()), encoding="utf-8",
    )
    pd.DataFrame({
        "timestamp": ["2026-10-05", "2026-10-06"],
        "strategy_equity": [10000., CAPITAL],
    }).to_csv(folder / "predictions.csv", index=False)
    pd.DataFrame({
        "action": ["BUY", "FINAL_SELL"],
        "cash_after_trade": [0., cash],
    }).to_csv(folder / "trades.csv", index=False)
    with zipfile.ZipFile(
        folder / "pacote_validacao_economica_raw.zip", "w",
    ) as z:
        z.writestr("reproducao_universo.json", json.dumps(fake_result()))
    return folder


def test_capital_from_summarized_result_contract() -> None:
    assert capital_do_resultado(fake_result()) == CAPITAL
    assert "strategy_ending_capital" not in fake_result()["metrics"]


def test_capital_missing_or_checkpoint_mismatch_fails_closed() -> None:
    bad = fake_result()
    bad["metrics"] = {"strategy_ending_capital": CAPITAL}
    with pytest.raises(KeyError, match="ending_capital"):
        capital_do_resultado(bad)
    bad = fake_result()
    bad["checkpoint"]["observed_capital"] += 1.
    with pytest.raises(RuntimeError, match="divergem"):
        capital_do_resultado(bad)


def test_saved_run_can_be_verified_without_model_or_market(tmp_path: Path) -> None:
    folder = make_run(tmp_path)
    result, checks = recovery.validar_resultado_salvo(folder)
    assert capital_do_resultado(result) == CAPITAL
    assert checks["capital"] == CAPITAL
    assert checks["trades_total"] == 2


def test_saved_run_requires_final_cash_equal_to_equity(tmp_path: Path) -> None:
    folder = make_run(tmp_path, cash=42.)
    with pytest.raises(RuntimeError, match="última venda"):
        recovery.validar_resultado_salvo(folder)


def test_recovery_copies_original_zip_immutably(tmp_path: Path, monkeypatch) -> None:
    folder = make_run(tmp_path)
    prices, events = tmp_path / "raw", tmp_path / "events"
    prices.mkdir()
    events.mkdir()
    assets = tuple(f"A{i:02d}" for i in range(75))
    for symbol in assets:
        (prices / f"{symbol}.csv").write_text(
            "timestamp,open,close\n2020-01-02,1,1\n", encoding="utf-8",
        )
        (events / f"{symbol}.csv").write_text(
            "ex_date,action_type\n", encoding="utf-8",
        )
    source = SimpleNamespace(raw_bars=prices, corporate_actions=events)
    monkeypatch.setattr(
        recovery, "_verify_source", lambda _root: (source, assets),
    )
    original_zip = folder / "pacote_validacao_economica_raw.zip"
    old_bytes = original_zip.read_bytes()
    report = recovery.recuperar(root=tmp_path)
    assert report["measured_capital_raw_economic"] == CAPITAL
    assert report["model_retrained"] is False
    assert report["backtest_reexecuted"] is False
    assert report["raw_and_actions_files_checked"] == 150
    assert original_zip.read_bytes() == old_bytes
    archive = folder / "pacote_validacao_economica_raw_v1251_finalizado.zip"
    with zipfile.ZipFile(archive, "r") as z:
        assert "interpretacao_resultado_v1251.json" in z.namelist()
        assert z.testzip() is None
    with pytest.raises(FileExistsError, match="não sobrescrever"):
        recovery.recuperar(root=tmp_path)


def test_saved_run_can_finish_in_cash_with_regular_sell(tmp_path: Path) -> None:
    folder = make_run(tmp_path)
    trades_path = folder / "trades.csv"
    trades = pd.read_csv(trades_path)
    trades.loc[1, "action"] = "SELL"
    trades.to_csv(trades_path, index=False)
    _, checks = recovery.validar_resultado_salvo(folder)
    assert checks["capital"] == CAPITAL
