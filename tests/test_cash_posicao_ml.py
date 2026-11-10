"""Regressoes para o estudo ML CASH/comprado; nunca enviar ordens."""
import unittest

import numpy as np
import pandas as pd

from reproducao.teste_cash_posicao_ml import (
    _metrics, _model, _shadow, estudar_cash_posicao_ml, FEATURES,
)


class CashPosicaoMLTests(unittest.TestCase):
    def test_cash_and_exposure_have_separate_entry_exit_decisions(self):
        forecast = np.array([-.01, .03, -.02, .05])
        q10 = np.array([-.02, -.01, -.07, -.01])
        realized = np.array([-.05, .01, -.03, .02])
        returns, actions, _ = _shadow(forecast, q10, realized, 0.)
        self.assertEqual(
            actions.tolist(), ["STAY_CASH", "BUY", "SELL_TO_CASH", "BUY"]
        )
        np.testing.assert_allclose(
            returns, [0., .01-.0005, -.0005, .02-.0005]
        )

    def test_absolute_tail_risk_can_prefer_cash_with_positive_mean(self):
        returns, actions, utility = _shadow(
            np.array([.01]), np.array([-.09]), np.array([.02]), .25
        )
        self.assertEqual(actions.tolist(), ["STAY_CASH"])
        self.assertAlmostEqual(returns[0], 0.)
        self.assertLess(utility[0], 0.)

    def test_shares_no_state_dependent_control_position_features(self):
        for prohibited in (
            "position_return_since_entry", "position_drawdown_from_peak",
            "holding_days_at_decision", "strategy_equity",
            "realized_pnl", "cash_after_trade", "trade_action",
        ):
            self.assertNotIn(prohibited, FEATURES)

    def test_shadow_drawdown_formula(self):
        m = _metrics(np.array([.1, -.2]))
        self.assertAlmostEqual(m["growth_multiple"], .88)
        self.assertAlmostEqual(m["max_drawdown"], -.2)

    def test_learners_are_quantile_and_expected_return_not_stops(self):
        self.assertEqual(
            _model("lightgbm_quant_7", tail=True).get_params()["objective"],
            "quantile",
        )
        self.assertEqual(
            _model("lightgbm_quant_7", tail=False).get_params()["objective"],
            "regression",
        )

    def test_unusable_inputs_fail_before_training(self):
        df = pd.DataFrame({
            "date": ["2026-11-09"],
            "decision_session": ["2026-11-09"],
            "selected_asset": ["CAPR"],
            "walk_forward_fold": [3],
            "daily_return": [-.01],
            **{name: [0.] for name in FEATURES},
        })
        with self.assertRaisesRegex(ValueError, "Referência de producao"):
            estudar_cash_posicao_ml(df, pd.Path(".") if hasattr(pd, "Path") else ".")


if __name__ == "__main__":
    unittest.main()
