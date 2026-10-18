"""Testes de contrato do replay contrafactual do Challenger (sem rede)."""
from __future__ import annotations

import unittest
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


if __name__ == "__main__":
    unittest.main()
