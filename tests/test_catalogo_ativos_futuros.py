"""Testes sem rede para catalogo prospectivo de novos ativos."""
import unittest
import pandas as pd

from pesquisar_catalogo_ativos_futuros_spyder import (
    _assets_from_response, _new_ids,
)


class CatalogoFuturoTests(unittest.TestCase):
    def setUp(self):
        self.raw = [
            {"id": "a1", "symbol": "AAPL", "class": "us_equity",
             "status": "active", "tradable": True, "exchange": "NASDAQ"},
            {"id": "b1", "symbol": "ABCD", "class": "us_equity",
             "status": "active", "tradable": True, "exchange": "NYSE"},
            {"id": "c1", "symbol": "PEND", "class": "us_equity",
             "status": "inactive", "tradable": False},
            {"id": "d1", "symbol": "CRYPTO", "class": "crypto",
             "status": "active", "tradable": True},
        ]

    def test_nao_altera_u75_e_nao_promove_novos(self):
        current = _assets_from_response(self.raw, {"AAPL"})
        self.assertEqual(set(current["symbol"]), {"AAPL", "ABCD"})
        self.assertTrue(current.loc[current.symbol == "AAPL", "in_frozen_u75"].iloc[0])
        self.assertFalse(current["entry_allowed"].any())
        diff = _new_ids(current, None)
        self.assertEqual(diff["symbol"].tolist(), ["ABCD"])
        self.assertEqual(diff["change"].tolist(), ["catalogo_inicial"])

    def test_detecta_novo_id_sem_fazer_bridge(self):
        current = _assets_from_response(self.raw, {"AAPL"})
        prior = pd.DataFrame({"symbol": ["AAPL", "ABCD"],
                              "alpaca_id": ["a1", "outro"]})
        diff = _new_ids(current, prior)
        self.assertEqual(diff.iloc[0]["change"], "identidade_alterada")
        self.assertFalse(bool(diff.iloc[0]["entry_allowed"]))

    def test_identidade_duplicada_reprovada(self):
        with self.assertRaisesRegex(ValueError, "duplicada"):
            _assets_from_response(self.raw + [self.raw[1]], {"AAPL"})


if __name__ == "__main__":
    unittest.main()
