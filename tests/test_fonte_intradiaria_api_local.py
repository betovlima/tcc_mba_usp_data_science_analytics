"""Testes isolados do leitor historico TCC. Sem rede, credenciais ou ordens."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "pesquisar_segunda_camada_intradiaria_spyder.py"
spec = importlib.util.spec_from_file_location("tcc_intraday", SCRIPT)
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)


class HistoricoTccLocalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        folder = root / "dados_intraday"
        folder.mkdir()
        for attribute, value in (
            ("ROOT", root), ("INPUT", folder),
            ("SNAPSHOT_CONTROL", root / "pesquisa_permanencia_minima.zip"),
            ("MAX_GRUPOS_DOWNLOAD", None), ("FORCAR_DOWNLOAD", False)):
            p = patch.object(research, attribute, value)
            p.start()
            self.addCleanup(p.stop)

    def _snapshot(self):
        rows = pd.DataFrame({
            "data": ["2026-07-28 04:00:00+00:00", "2026-07-29 04:00:00+00:00",
                     "2026-07-30 04:00:00+00:00"],
            "ativo_2_pregoes": ["CAPR", "CAPR", "CASH"],
        })
        with ZipFile(research.SNAPSHOT_CONTROL, "w") as archive:
            archive.writestr(
                "pesquisa_permanencia_minima/comparacao_diaria_ativos_e_patrimonio.csv",
                rows.to_csv(index=False),
            )

    def test_snapshot_produces_position_csv_without_cash_or_future_selection(self):
        self._snapshot()
        positions, manifest = research.preparar_posicoes_control()
        self.assertEqual(manifest["status"], "derivado_do_control_congelado")
        self.assertEqual(positions.session.tolist(), ["2026-07-28", "2026-07-29"])
        self.assertEqual(positions.symbol.tolist(), ["CAPR", "CAPR"])
        self.assertTrue((research.INPUT / "winner_positions.csv").is_file())
        again, origin = research.preparar_posicoes_control()
        self.assertEqual(origin["status"], "arquivo_existente_reutilizado")
        pd.testing.assert_frame_equal(positions.reset_index(drop=True), again.reset_index(drop=True))

    def test_local_market_data_is_readonly_cached_and_scoped_to_position(self):
        self._snapshot()
        positions, _ = research.preparar_posicoes_control()
        ix = pd.date_range("2026-07-28T13:30:00Z", periods=3, freq="5min")
        bars = pd.DataFrame(
            {"open": [6.78, 6.7, 6.6],
             "high": [6.8, 6.72, 6.65],
             "low": [6.7, 6.5, 6.4],
             "close": [6.73, 6.61, 6.45],
             "volume": [5000, 3000, 8000]}, index=ix)
        bars.index.name = "timestamp"
        hits = []
        def fake_download(**kwargs):
            hits.append(kwargs)
            return bars.copy()
        with patch.object(research, "_market_reader_local",
                          return_value=(fake_download, {"api_key_id":"hidden",
                                                        "secret_key":"hidden"})):
            first = research.preencher_cache_control(positions)
            second = research.preencher_cache_control(positions)
        self.assertEqual(first["novos_baixados"], 1)
        self.assertEqual(second["novos_baixados"], 0)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["symbol"], "CAPR")
        self.assertEqual(hits[0]["timeframe"], "5Min")
        self.assertEqual(hits[0]["adjustment"], "raw")
        self.assertEqual(hits[0]["feed"], "sip")
        cached = research.INPUT / "cotacoes_5min" / "CAPR_2026-07.csv"
        self.assertTrue(cached.exists())
        saved = pd.read_csv(cached)
        self.assertEqual(len(saved), 3)
        self.assertEqual(set(saved.symbol), {"CAPR"})
        self.assertNotIn("hidden", cached.read_text())
        c = research.medir_cobertura(positions, saved)
        self.assertEqual(c["sessoes_cobertas"], 1)
        self.assertEqual(c["sessoes_faltantes"], 1)

    def test_missing_control_and_missing_local_fail_without_faking_prices(self):
        positions, manifest = research.preparar_posicoes_control()
        self.assertTrue(positions.empty)
        self.assertEqual(manifest["status"], "snapshot_control_ausente")
        positions = pd.DataFrame({"session":["2026-07-28"],"symbol":["CAPR"]})
        with patch.object(research, "_market_reader_local",
                          side_effect=RuntimeError("no local local")):
            result = research.preencher_cache_control(positions)
        self.assertEqual(result["status"], "local_local_indisponivel")
        self.assertFalse(list((research.INPUT / "cotacoes_5min").glob("*.csv")))


if __name__ == "__main__":
    unittest.main()
