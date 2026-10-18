"""Testes de contrato do replay contrafactual do Challenger (sem rede)."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import zipfile
import pandas as pd

import reproduzir_u77_paper_20260804_spyder as replay


class U77PaperReplayContractTests(unittest.TestCase):
    def test_universo_e_exclusoes_sao_fixos(self):
        replay.validate_contract()
        self.assertEqual(len(replay.REQUESTED_ASSETS), 77)
        self.assertEqual(len(replay.EFFECTIVE_ASSETS), 75)
        self.assertEqual(len(replay.REFERENCE_ASSETS), 65)
        self.assertEqual(set(replay.REQUESTED_ASSETS) - set(replay.EFFECTIVE_ASSETS), {"DOC", "CLMT"})
        self.assertEqual(replay.EFFECTIVE_ASSETS[-10:], replay.CHALLENGER_ADDITIONS)

    def test_capital_em_caixa_e_janela_inclusiva(self):
        config = replay.make_config()
        self.assertEqual(config.initial_capital, 10_000.0)
        self.assertEqual(config.analysis_start_date, "2026-08-04")
        self.assertEqual(config.analysis_end_date, "2026-10-16")
        self.assertEqual(config.slippage_bps, replay.CONFIG.slippage_bps)
        self.assertEqual(config.rotation_min_holding_days, replay.CONFIG.rotation_min_holding_days)
        self.assertEqual(tuple(config.assets), replay.EFFECTIVE_ASSETS)

    def test_primeira_decisao_e_sessao_anterior(self):
        dates = pd.to_datetime([
            "2026-08-03T04:00:00Z", "2026-08-04T04:00:00Z",
            "2026-10-16T04:00:00Z"
        ])
        replay.assert_window(dates)

    def test_falha_quando_data_inicial_foge_do_contrato(self):
        dates = pd.to_datetime(["2026-08-04T04:00:00Z", "2026-08-05T04:00:00Z",
                                "2026-10-16T04:00:00Z"])
        with self.assertRaisesRegex(RuntimeError, "Janela"):
            replay.assert_window(dates)

    def test_serialize_nan_and_timestamp(self):
        self.assertEqual(
            replay._json_safe({"x": float("nan"), "date": pd.Timestamp("2026-10-16")}),
            {"x": None, "date": "2026-10-16T00:00:00"}
        )


    def test_recover_csv_without_alpaca_or_training(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id = "20261018T061600_abcdef01"
            folder = root / "output" / "u77_paper_replay" / run_id
            folder.mkdir(parents=True)
            snapshot = root / "dados" / ".u77_paper_replay" / run_id
            snapshot.mkdir(parents=True)

            pd.DataFrame({
                "timestamp": ["2026-08-04T04:00:00Z", "2026-10-16T04:00:00Z"],
                "strategy_equity": [10020.0, 10100.0],
                "selected_asset": ["ADEA", "ADEA"],
                "previous_asset": ["CASH", "ADEA"],
                "trade_action": ["BUY", "FINAL_SELL"],
                "buy_hold_equity": [10015.0, 10077.0],
            }).to_csv(folder / "decisoes_e_patrimonio.csv", index=False)
            pd.DataFrame({
                "timestamp": ["2026-08-04T04:00:00Z", "2026-10-16T04:00:00Z"],
                "action": ["BUY", "FINAL_SELL"], "total_fee": [0.0, 0.05],
            }).to_csv(folder / "operacoes.csv", index=False)
            pd.DataFrame({
                "fold_id": [1], "candidate_margin": [0.0005],
                "calibration_score": [0.1], "selected": [True],
            }).to_csv(folder / "margens_calibracao.csv", index=False)
            pd.DataFrame({"symbol": ["ADEA"], "raw_rows": [2100],
                          "splits_applied": [0]}).to_csv(
                folder / "auditoria_dados.csv", index=False
            )
            (folder / "resumo_motor.txt").write_text("Simulacao concluida.", encoding="utf-8")
            original_csv = (folder / "operacoes.csv").read_bytes()
            manifest = {
                "bar_snapshot_as_of_end": replay.EXECUTION_END,
                "assets": list(replay.EFFECTIVE_ASSETS),
                "snapshot_sha256": "abc123",
            }
            with patch.object(replay, "ROOT", root), patch.object(
                replay, "validate_snapshot", return_value=manifest
            ), patch.object(replay, "load_alpaca_credentials", side_effect=AssertionError("NO ALPACA")):
                archive = replay.recover_partial_run(run_id)
            self.assertTrue(archive.is_file())
            self.assertEqual((folder / "operacoes.csv").read_bytes(), original_csv)
            report = json.loads((folder / "resultado.json").read_text(encoding="utf-8"))
            self.assertEqual(report["ending_capital"], 10100.0)
            self.assertEqual(report["profit"], 100.0)
            self.assertEqual(report["first_execution_date"], "2026-08-04")
            self.assertEqual(report["final_execution_date"], "2026-10-16")
            self.assertTrue(report["recovered_from_partial_export"])
            with zipfile.ZipFile(archive) as zf:
                self.assertIn("decisoes_e_patrimonio.csv", zf.namelist())
                self.assertIn("operacoes.csv", zf.namelist())
                self.assertIn("resultado.json", zf.namelist())

    def test_recover_missing_csv_fails_before_snapshot(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "output" / "u77_paper_replay" / "partial"
            folder.mkdir(parents=True)
            with patch.object(replay, "ROOT", root), self.assertRaises(FileNotFoundError):
                replay.recover_partial_run("partial")


if __name__ == "__main__":
    unittest.main()
