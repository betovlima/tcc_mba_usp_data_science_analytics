"""Testes sintéticos da consolidação de ZIPs sem dados de mercado."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from consolidar_contribuicao_externa_u75_spyder import consolidar, _read_package


class ConsolidacaoExternaTests(unittest.TestCase):
    def _package(self, path: Path, symbol: str = "A00", capital: float = 110.0):
        assets = [f"A{i:02d}" for i in range(24)]
        row = {
            "symbol": symbol, "sessions": 3,
            "changed_positions": 2, "direct_candidate_days": 1,
            "indirect_changed_days": 1,
            "candidate_raw_sha256": "raw", "candidate_actions_sha256": "actions",
            "calendar_source": "AAPL", "control_ending_capital": 100.0,
            "challenger_ending_capital": capital,
            "delta_ending_capital": capital - 100.0,
            "delta_compound_log_growth": 0.09531017980432493,
            "strategy_sharpe": 1.0, "strategy_maximum_drawdown": -0.2,
            "capital_rotations": 4.0, "total_transaction_fees": 1.0,
        }
        result = {
            "status": "completed",
            "parity": {"fully_reproduced": True},
            "source": {
                "external_candidates": assets,
                "candidate_snapshot": "hash-candidate",
                "reference_snapshot": "hash-control",
                "control_zip_sha256": "hash-zip",
            },
            "baseline_metrics": {"strategy_ending_capital": 100.0},
            "batch": {"symbols": [symbol]},
            "candidate_results": [row], "candidate_errors": [],
        }
        files = {"resultado.json": json.dumps(result).encode()}
        manifest = {
            "experiment": "u75-external-marginal-capital",
            "files": [
                {"name": name, "size": len(blob), "sha256": sha256(blob).hexdigest()}
                for name, blob in files.items()
            ],
        }
        with ZipFile(path, "w") as z:
            z.writestr("manifesto_pacote.json", json.dumps(manifest))
            for name, blob in files.items():
                z.writestr(name, blob)
        return path

    def test_incompleto_nao_e_encerrado(self):
        with TemporaryDirectory() as t:
            package = self._package(Path(t) / "a.zip")
            result = consolidar([package])
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["measured_candidates"], 1)
            self.assertEqual(len(result["missing_candidates"]), 23)
            self.assertIn("A00", result["incomplete_evidence"])

    def test_repeticao_divergente_reprovada(self):
        with TemporaryDirectory() as t:
            a = self._package(Path(t) / "a.zip", capital=110.0)
            b = self._package(Path(t) / "b.zip", capital=120.0)
            with self.assertRaisesRegex(RuntimeError, "duplicados divergentes"):
                consolidar([a, b])

    def test_hash_divergente_reprovado(self):
        with TemporaryDirectory() as t:
            package = self._package(Path(t) / "a.zip")
            with ZipFile(package, "a") as z:
                z.writestr("resultado.json", '{"alterado": true}')
            with self.assertRaises(RuntimeError):
                _read_package(package)


if __name__ == "__main__":
    unittest.main()
