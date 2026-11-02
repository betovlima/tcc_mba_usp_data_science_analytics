"""Testes do ranking econômico: rotação, recuperação, CASH e maturação."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pesquisa.risco_reversao_antecipado_u75 import (
    FEATURES, EarlyRiskSettings, preparar_painel_supervisionado,
)
from pesquisa.vantagem_economica_rotacao_u75 import (
    EconomicSettings, preparar_resultados_economicos,
    estimar_vantagens_por_fold, politica_vantagem_economica,
)


class EconomicRotationTests(unittest.TestCase):
    def setUp(self):
        self.date = pd.Timestamp("2026-01-05", tz="UTC")
        self.settings = EconomicSettings(minimum_train_samples=30)
        self.symbols = ["AAA", "BBB"]
        self.cache = {self.date: np.array([0.0, 0.4, 0.35])}
        self.audit = []
        self.diag = {}

    def _signals(self, a, b, *, a_risk=.2, b_risk=.9):
        return {self.date: {
            "AAA": {
                "risk_score": a[0], "joint_score": a[1],
                "p_downside": a_risk, "p_recovery": None,
                "expected_recovery_log_return_5d": None,
            },
            "BBB": {
                "risk_score": b[0], "joint_score": b[1],
                "p_downside": b_risk, "p_recovery": .90,
                "expected_recovery_log_return_5d": .12,
            },
        }}

    def _policy(self, signals, use_recovery=False, base_target=1, guard=False):
        def control(date, current, holding):
            self.diag[date] = {
                "decision_reason": ("MIN_HOLD_GUARD" if guard
                                    else "HOLD_CURRENT_BEST"),
            }
            return base_target, 0.4
        return politica_vantagem_economica(
            control, self.symbols, self.cache, signals, self.settings,
            use_recovery=use_recovery, audit=self.audit,
            diagnostics=self.diag, minimum_score=0.0,
        )

    def test_risco_so_nao_veta_por_probabilidade_alta(self):
        sig = self._signals((.03, .03), (.01, .05))
        chosen, _ = self._policy(sig)(self.date, 1, 4)
        self.assertEqual(chosen, 1)
        self.assertEqual(self.audit[-1]["reason"], "CONTROL_PRESERVED")

    def test_reversao_com_risco_alto_pode_rotacionar(self):
        sig = self._signals((.03, .03), (.01, .05))
        chosen, score = self._policy(sig, use_recovery=True)(self.date, 1, 4)
        self.assertEqual(chosen, 2)
        self.assertAlmostEqual(score, .35)
        self.assertEqual(self.audit[-1]["reason"], "RECOVERY_ECONOMIC_ROTATION")
        self.assertEqual(self.diag[self.date]["final_action_asset"], "BBB")

    def test_custo_impede_troca_de_margem_pequena(self):
        sig = self._signals((.03, .03), (.035, .035))
        chosen, _ = self._policy(sig)(self.date, 1, 4)
        self.assertEqual(chosen, 1)

    def test_min_holding_preservado(self):
        sig = self._signals((.03, .03), (.01, .25))
        chosen, _ = self._policy(sig, use_recovery=True, guard=True)(
            self.date, 1, 1,
        )
        self.assertEqual(chosen, 1)
        self.assertEqual(self.audit[-1]["reason"], "CONTROL_MIN_HOLD_GUARD")

    def test_cash_quando_todas_opcoes_tem_perspectiva_negativa(self):
        sig = self._signals((-.03, -.03), (-.01, -.01))
        chosen, _ = self._policy(sig)(self.date, 1, 3)
        self.assertEqual(chosen, 0)
        self.assertTrue(self.audit[-1]["cash_last_resort"])

    def test_previsao_faltante_preserva_control(self):
        sig = self._signals((None, None), (.5, .5))
        chosen, _ = self._policy(sig)(self.date, 1, 3)
        self.assertEqual(chosen, 1)
        self.assertEqual(self.audit[-1]["reason"], "CONTROL_MODEL_UNAVAILABLE")

    def test_previsao_faltante_tambem_reduz_cobertura(self):
        # Motivo especifico prevalece mesmo com cobertura < 90%.
        sig = self._signals((None, None), (.5, .5))
        chosen, _ = self._policy(sig)(self.date, 1, 3)
        self.assertEqual(chosen, 1)
        self.assertEqual(self.audit[-1]["reason"], "CONTROL_MODEL_UNAVAILABLE")
        self.assertFalse(self.audit[-1]["model_ready"])
        self.assertAlmostEqual(self.audit[-1]["model_coverage"], 0.5)
        self.assertFalse(self.audit[-1]["changed_control_action"])

    def test_previsao_do_control_disponivel_cobertura_insuficiente(self):
        # Distinto do caso anterior: so outra acao fica sem previsao.
        sig = self._signals((.01, .01), (None, None))
        chosen, _ = self._policy(sig)(self.date, 1, 3)
        self.assertEqual(chosen, 1)
        self.assertEqual(self.audit[-1]["reason"], "CONTROL_INSUFFICIENT_COVERAGE")
        self.assertAlmostEqual(self.audit[-1]["model_coverage"], 0.5)

    def test_trava_minima_tem_prioridade_sobre_previsao_faltante(self):
        sig = self._signals((None, None), (.5, .5))
        chosen, _ = self._policy(sig, guard=True)(self.date, 1, 1)
        self.assertEqual(chosen, 1)
        self.assertEqual(self.audit[-1]["reason"], "CONTROL_MIN_HOLD_GUARD")

    def test_ausencia_preco_intermediario_nao_inventa_alvo(self):
        s = EarlyRiskSettings()
        n = 80
        cal = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
        values = 100 + np.sin(np.arange(n)/8)*10 + np.arange(n)*.2
        f = pd.DataFrame(index=cal, data={
            "open":values, "close":values, "low":values*.97,
            "high":values*1.03,
        })
        for feature in FEATURES:
            f[feature] = np.sin(np.arange(n)/6)*.02 + .2
        f["distance_from_high_20"] = -.12
        frames = {"AAA": f}
        panel = preparar_painel_supervisionado(frames, cal, s)
        f.loc[cal[12], "low"] = np.nan
        extended = preparar_resultados_economicos(panel, frames, cal, 5, 10)
        self.assertTrue(
            pd.isna(extended.loc[extended["calendar_position"].eq(10),
                                "target_log_return"].iloc[0])
        )

    def test_modelo_nunca_usa_rótulo_imaduro(self):
        s = EarlyRiskSettings()
        n = 90
        cal = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
        values = 100 + np.sin(np.arange(n)/4)*9 + np.arange(n)*.2
        f = pd.DataFrame(index=cal, data={
            "open":values, "close":values,
            "low":values*.96, "high":values*1.04,
        })
        for i, feature in enumerate(FEATURES):
            f[feature] = np.sin(np.arange(n) / (i+2))* .05 + .2
        f["distance_from_high_20"] = -.13
        frames = {"AAA":f}
        panel = preparar_painel_supervisionado(frames, cal, s)
        enriched = preparar_resultados_economicos(panel, frames, cal, 5, 10)
        fold = {"fold_id":1,"test_start_index":70,
                "test_end_index":85,"final_fit_end_index":55}
        inherited = {1:{
            date: {"AAA":{
                "p_downside": .2, "p_recovery": .7,
                "recent_drawdown":True,
            }}
            for date in cal[69:85]
        }}
        economic = EconomicSettings(minimum_train_samples=30)
        first, stats, _ = estimar_vantagens_por_fold(
            enriched, inherited, [fold], cal, economic, s.reversal_horizon,
        )
        self.assertEqual(stats.iloc[0]["last_train_feature_position"], 54)
        changed = {"AAA": f.copy()}
        changed["AAA"].loc[cal[70]:,"close"] *= 2
        changed["AAA"].loc[cal[70]:,"low"] *= .3
        enriched2 = preparar_resultados_economicos(
            preparar_painel_supervisionado(changed, cal, s),
            changed, cal, 5, 10,
        )
        second, _, _ = estimar_vantagens_por_fold(
            enriched2, inherited, [fold], cal, economic, s.reversal_horizon,
        )
        key = cal[69]
        self.assertEqual(
            first[1][key]["AAA"]["risk_score"],
            second[1][key]["AAA"]["risk_score"],
        )
        self.assertEqual(
            first[1][key]["AAA"]["joint_score"],
            second[1][key]["AAA"]["joint_score"],
        )


if __name__ == "__main__":
    unittest.main()
