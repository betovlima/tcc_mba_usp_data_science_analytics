"""Testes sintéticos de causalidade, CASH como último recurso e rotação intacta."""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from pesquisa.risco_reversao_antecipado_u75 import (
    EarlyRiskSettings, FEATURES, adaptar_politica_rotacao,
    preparar_painel_supervisionado, estimar_folds,
)


def _frames(close_values, symbol="A"):
    values = np.asarray(close_values, dtype=float)
    ix = pd.date_range("2020-01-01", periods=len(values), freq="B", tz="UTC")
    data = pd.DataFrame(index=ix)
    data["open"] = values
    data["close"] = values
    data["low"] = values * .98
    data["high"] = values * 1.02
    for i, f in enumerate(FEATURES):
        data[f] = .01 + (i + 1) * .001 + np.arange(len(values)) * .00001
    data["distance_from_high_20"] = -.15
    return {symbol: data}, ix


def _signals(risk_a=.1, risk_b=.1, recovery_b=None):
    return {
        pd.Timestamp("2026-01-05", tz="UTC"): {
            "AAA": {"p_downside": risk_a, "p_recovery": None,
                    "recent_drawdown": False},
            "BBB": {"p_downside": risk_b, "p_recovery": recovery_b,
                    "recent_drawdown": True},
        }
    }


class RiskBottomResearchTests(unittest.TestCase):
    def setUp(self):
        self.settings = EarlyRiskSettings(min_train_samples=30)
        self.date = pd.Timestamp("2026-01-05", tz="UTC")
        self.symbols = ["AAA", "BBB"]
        self.cache = {self.date: np.asarray([0.0, .10, .08])}

    def _policy(self, action, signals, recovery=False):
        audit = []
        diagnostics = {}
        policy = adaptar_politica_rotacao(
            lambda ts, pos, hold: (action, .10),
            self.symbols, self.cache, signals, self.settings,
            enable_recovery=recovery, audit=audit,
            decision_diagnostics=diagnostics,
        )
        return policy, audit, diagnostics

    def test_preserva_rotacao_control_quando_sem_risco(self):
        p, audit, _ = self._policy(2, _signals())
        chosen, _ = p(self.date, 1, 6)
        self.assertEqual(chosen, 2)
        self.assertEqual(audit[0]["reason"], "CONTROL_PRESERVED")

    def test_risco_no_alvo_roda_primeiro_para_outro_ativo(self):
        p, audit, diagnostics = self._policy(
            1, _signals(risk_a=.9, risk_b=.1),
        )
        chosen, score = p(self.date, 1, 5)
        self.assertEqual(chosen, 2)
        self.assertAlmostEqual(score, .08)
        self.assertEqual(audit[0]["reason"], "RISK_ROTATE_TO_ALTERNATIVE")
        self.assertEqual(diagnostics[self.date]["final_action_asset"], "BBB")

    def test_cash_somente_quando_nao_ha_alternativa_segura(self):
        p, audit, _ = self._policy(1, _signals(risk_a=.9, risk_b=.8))
        chosen, score = p(self.date, 1, 1)
        self.assertEqual((chosen, score), (0, 0.0))
        self.assertTrue(audit[0]["cash_last_resort"])

    def test_reversao_nao_substitui_uma_rotacao_com_vantagem_insuficiente(self):
        s = _signals(recovery_b=.99)
        p, audit, _ = self._policy(1, s, recovery=True)
        chosen, _ = p(self.date, 1, 5)
        self.assertEqual(chosen, 1)
        self.assertEqual(audit[0]["reason"], "CONTROL_PRESERVED")

    def test_reversao_pode_produzir_nova_rotacao(self):
        s = _signals(recovery_b=.99)
        self.cache[self.date][2] = .101
        p, audit, _ = self._policy(1, s, recovery=True)
        chosen, _ = p(self.date, 1, 5)
        self.assertEqual(chosen, 2)
        self.assertEqual(audit[0]["reason"], "RECOVERY_OPPORTUNITY_ROTATION")

    def test_reversao_respeita_min_holding_control(self):
        self.cache[self.date][2] = .101
        diagnostic = {}
        def base(ts, current, holding):
            diagnostic[ts] = {"decision_reason": "MIN_HOLD_GUARD"}
            return 1, .10
        audit = []
        p = adaptar_politica_rotacao(
            base, self.symbols, self.cache, _signals(recovery_b=.99),
            self.settings, enable_recovery=True, audit=audit,
            decision_diagnostics=diagnostic,
        )
        chosen, _ = p(self.date, 1, 1)
        self.assertEqual(chosen, 1)
        self.assertEqual(audit[0]["reason"], "CONTROL_PRESERVED")

    def test_modelo_ausente_nao_altera_decisao(self):
        p, audit, _ = self._policy(1, {self.date: {}})
        chosen, _ = p(self.date, 1, 5)
        self.assertEqual(chosen, 1)
        self.assertEqual(audit[0]["reason"], "CONTROL_PRESERVED")

    def test_alvo_invalido_nao_e_reconstruido(self):
        frames, cal = _frames([100.0] * 50)
        frames["A"].iloc[8, frames["A"].columns.get_loc("low")] = np.nan
        panel = preparar_painel_supervisionado(frames, cal, self.settings)
        row = panel[(panel["calendar_position"] == 5)].iloc[0]
        self.assertTrue(np.isnan(row["downside_event"]))

    def test_maturacao_causal_por_fold_sem_futuro(self):
        values = 100 + np.sin(np.arange(100) / 5) * 15
        frames, cal = _frames(values)
        panel = preparar_painel_supervisionado(frames, cal, self.settings)
        fold = {
            "fold_id": 1, "test_start_index": 85,
            "test_end_index": 96, "final_fit_end_index": 70,
        }
        a, stats, predictions = estimar_folds(panel, [fold], cal, self.settings)
        self.assertEqual(stats.iloc[0]["last_train_feature_position"], 69)
        self.assertTrue((predictions["calendar_position"] >= 84).all())
        # Precos posteriores ao primeiro teste nao podem mudar o conjunto
        # de treino (com features historicas preservadas ate a decisao).
        changed = {s: frame.copy() for s, frame in frames.items()}
        changed["A"].loc[cal[85]:, "close"] *= 4
        changed["A"].loc[cal[85]:, "low"] *= .1
        panel2 = preparar_painel_supervisionado(changed, cal, self.settings)
        b, _, predictions2 = estimar_folds(panel2, [fold], cal, self.settings)
        key = cal[84]
        self.assertEqual(a[1][key]["A"]["p_downside"],
                         b[1][key]["A"]["p_downside"])
        self.assertEqual(a[1][key]["A"]["p_recovery"],
                         b[1][key]["A"]["p_recovery"])


if __name__ == "__main__":
    unittest.main()
