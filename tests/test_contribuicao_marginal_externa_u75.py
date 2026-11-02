"""Testes deterministas do comparador de contribuicao marginal de candidatos."""
import unittest
import pandas as pd

from pesquisa.contribuicao_marginal_externa_u75 import comparar_trajetorias


def frame(assets, capital, folds=(1, 1, 2), shift=0):
    index = pd.date_range("2026-01-02", periods=len(assets), freq="B", tz="UTC")
    if shift:
        index = index + pd.Timedelta(days=shift)
    return pd.DataFrame({
        "selected_asset": list(assets),
        "walk_forward_fold": list(folds),
        "strategy_equity": list(capital),
    }, index=index)


class TestComparacaoMarginalExterna(unittest.TestCase):
    def test_sem_efeito_marginal(self):
        base = frame(["A", "A", "B"], [100, 110, 130])
        daily, metrics, folds = comparar_trajetorias(base, base.copy(), "X")
        self.assertEqual(metrics["changed_positions"], 0)
        self.assertEqual(metrics["direct_candidate_days"], 0)
        self.assertEqual(metrics["indirect_changed_days"], 0)
        self.assertEqual(metrics["delta_ending_capital"], 0)
        self.assertEqual(len(folds), 2)
        self.assertFalse(daily["mudou_posicao"].any())

    def test_distingue_direta_e_indireta(self):
        base = frame(["A", "A", "B"], [100, 110, 130])
        challenger = frame(["A", "X", "A"], [100, 120, 150])
        daily, metrics, _ = comparar_trajetorias(base, challenger, "X")
        self.assertEqual(metrics["changed_positions"], 2)
        self.assertEqual(metrics["direct_candidate_days"], 1)
        self.assertEqual(metrics["indirect_changed_days"], 1)
        self.assertEqual(metrics["delta_ending_capital"], 20.0)
        self.assertTrue(bool(daily.loc[2, "mudou_sem_selecionar_candidato"]))

    def test_calendario_diferente_reprovado(self):
        base = frame(["A", "B", "A"], [100, 105, 106])
        wrong = frame(["A", "X", "A"], [100, 105, 110], shift=1)
        with self.assertRaisesRegex(ValueError, "Calendario diferente"):
            comparar_trajetorias(base, wrong, "X")

    def test_folds_diferentes_reprovados(self):
        base = frame(["A", "B", "A"], [100, 105, 106])
        wrong = frame(["A", "X", "A"], [100, 105, 110], folds=(1, 2, 2))
        with self.assertRaisesRegex(ValueError, "Folds diferentes"):
            comparar_trajetorias(base, wrong, "X")

    def test_capital_invalido_reprovado(self):
        base = frame(["A", "B", "A"], [100, 105, 106])
        wrong = frame(["A", "X", "A"], [100, 0, 110])
        with self.assertRaisesRegex(ValueError, "patrimonio invalido"):
            comparar_trajetorias(base, wrong, "X")

    def test_duplicata_reprovada(self):
        base = frame(["A", "B", "A"], [100, 105, 106])
        wrong = base.copy().reset_index(names="timestamp")
        wrong.loc[1, "timestamp"] = wrong.loc[0, "timestamp"]
        with self.assertRaisesRegex(ValueError, "duplicadas"):
            comparar_trajetorias(base, wrong, "X")


if __name__ == "__main__":
    unittest.main()
