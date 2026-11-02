"""Regressão: não confundir pequeno retorno bruto positivo com retorno líquido.

Dados sintéticos representam a contradição econômica observada na auditoria:
melhor alternativa com score +0.0006 antes do custo de 0.003.
Nenhuma série de 2026 é carregada, e a regra não é recalibrada pelo histórico.
"""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from pesquisa.vantagem_economica_rotacao_u75 import (
    EconomicSettings, politica_vantagem_economica,
)


class EconomicNetDecisionTests(unittest.TestCase):
    def setUp(self):
        self.date = pd.Timestamp("2026-01-05", tz="UTC")
        self.symbols = ["INCUMBENT", "CONTROL", "ALT"]
        self.utilities = {
            self.date: np.array([0.0, .4, .5, .3], dtype=float)
        }
        self.settings = EconomicSettings()
        self.log = []
        self.diag = {}

    def _run(self, scores, *, current=1, base=2,
             hold_guard=False, joint=False):
        self.log.clear()
        self.diag.clear()
        def control(ts, cur, holding):
            self.diag[ts] = {
                "decision_reason": ("MIN_HOLD_GUARD" if hold_guard
                                    else "ROTATE_TO_BEST_ASSET")
            }
            return base, .5
        day = {
            name: {
                "risk_score": score, "joint_score": score,
                "p_downside": .1, "p_recovery": None,
                "expected_recovery_log_return_5d": None,
            }
            for name, score in zip(self.symbols, scores)
        }
        policy = politica_vantagem_economica(
            control, self.symbols, self.utilities,
            {self.date: day}, self.settings,
            use_recovery=joint, audit=self.log,
            diagnostics=self.diag, minimum_score=0.0,
        )
        result = policy(self.date, current, 4)
        return result, self.log[-1]

    def test_cash_se_toda_compra_tem_retorno_liquido_nao_positivo(self):
        # Melhor bruto +0.0006, inferior a 0.003 custo; Control -0.0196.
        (chosen, _), audit = self._run([-.0093, -.0196, .0006])
        self.assertEqual(chosen, 0)
        self.assertEqual(audit["reason"], "ECONOMIC_CASH_LAST_RESORT")
        self.assertLess(audit["economic_score_control"], -.01)
        self.assertLessEqual(audit["best_candidate_incremental_net_edge"], .04)
        self.assertGreater(audit["incremental_net_edge"], 0)

    def test_alternativa_realmente_positiva_prioriza_rotacao(self):
        (chosen, _), audit = self._run([-.0093, -.0196, .012])
        self.assertEqual(chosen, 3)
        self.assertEqual(audit["reason"], "RISK_ECONOMIC_ROTATION")
        self.assertGreater(audit["final_net_score"], 0)

    def test_preserva_ativo_atual_se_melhor_que_cash_liquido(self):
        # Incumbente -0.001 e nenhuma alternativa mais lucrativa apos custo.
        (chosen, _), audit = self._run([-.001, -.0196, .0006])
        self.assertEqual(chosen, 1)
        self.assertEqual(audit["reason"], "ECONOMIC_AVOID_NEGATIVE_ROTATION")

    def test_trava_minima_original_e_preservada(self):
        (chosen, _), audit = self._run(
            [-.0093, -.0196, .0006], hold_guard=True,
        )
        self.assertEqual(chosen, 2)
        self.assertEqual(audit["reason"], "CONTROL_MIN_HOLD_GUARD")

    def test_sem_necessidade_economica_preserva_control(self):
        (chosen, _), audit = self._run([.01, .03, .02])
        self.assertEqual(chosen, 2)
        self.assertEqual(audit["reason"], "CONTROL_PRESERVED")
        self.assertAlmostEqual(audit["incremental_net_edge"], 0.0)

    def test_cash_pode_ser_mantido_ao_evitar_entrada_ruim(self):
        (chosen, _), audit = self._run(
            [-.03, -.0196, .0006], current=0,
        )
        self.assertEqual(chosen, 0)
        self.assertEqual(audit["reason"], "ECONOMIC_CASH_LAST_RESORT")


if __name__ == "__main__":
    unittest.main()
