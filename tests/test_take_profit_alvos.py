"""Testes locais do replay TP: nao dependem do snapshot Alpaca ou de treino."""
from __future__ import annotations

from dataclasses import dataclass
import unittest

import pandas as pd

from pesquisar_take_profit_alvos_spyder import simular_alvo

DATES = pd.to_datetime(
    ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"],
    utc=True,
)


@dataclass
class Config:
    initial_capital: float = 10_000.


def fees(side, qty, price, config):
    return {"total_fee": 0.0}


def slip(price, side, config):
    return price


def frames(high=None, opening=None, close=None):
    high = high or [100., 100., 100., 100., 100.]
    opening = opening or [100., 100., 100., 100., 100.]
    close = close or [100., 100., 100., 100., 100.]
    return {"AAA": pd.DataFrame({
        "open": opening, "high": high, "low": [100.] * 5, "close": close,
    }, index=DATES)}


class TakeProfitTargetsTests(unittest.TestCase):
    def replay(self, *, target, policy=None, high=None, opening=None, close=None):
        return simular_alvo(
            policy=policy or (lambda day, pos, held: (1, 0.5)),
            frames=frames(high=high, opening=opening, close=close),
            symbols=["AAA"],
            decision_dates=DATES,
            config=Config(),
            fee_calculator=fees,
            slippage=slip,
            target=target,
            fold_map={d: 1 for d in DATES},
        )

    def test_control_without_tp_maintains_position(self):
        result = self.replay(target=None, high=[100., 107., 107., 107., 107.])
        self.assertEqual(result.metrics["tp_exits"], 0)
        self.assertEqual(result.metrics["buys"], 1)
        self.assertAlmostEqual(result.metrics["ending_capital"], 10_000.)

    def test_three_percent_compounds_and_reenters_same_asset_next_day(self):
        result = self.replay(target=0.03, high=[100., 104., 104., 104., 104.])
        self.assertEqual(result.metrics["tp_exits"], 4)
        self.assertEqual(result.metrics["same_asset_reentries_after_tp"], 3)
        self.assertAlmostEqual(result.metrics["ending_capital"], 10_000. * 1.03 ** 4)
        for _, per_day in result.trades.groupby("timestamp"):
            self.assertLessEqual((per_day.action == "BUY").sum(), 1)

    def test_five_percent_only_fills_after_reaching_105(self):
        below = self.replay(target=0.05, high=[100., 104., 104., 104., 104.])
        above = self.replay(target=0.05, high=[100., 106., 106., 106., 106.])
        self.assertEqual(below.metrics["tp_exits"], 0)
        self.assertEqual(above.metrics["tp_exits"], 4)
        self.assertAlmostEqual(above.metrics["ending_capital"], 10_000. * 1.05 ** 4)

    def test_one_point_five_percent(self):
        result = self.replay(target=0.015, high=[100., 102., 102., 102., 102.])
        self.assertAlmostEqual(result.metrics["ending_capital"], 10_000. * 1.015 ** 4)

    def test_gap_sells_at_open_not_at_target(self):
        result = self.replay(
            target=0.03,
            high=[100., 100., 107., 107., 107.],
            opening=[100., 100., 106., 106., 106.],
            close=[100., 100., 106., 106., 106.],
        )
        gap = result.trades[result.trades.reason == "TP_GAP_OPEN"].iloc[0]
        self.assertAlmostEqual(gap.price, 106.)

    def test_model_may_choose_cash_independently(self):
        def policy(day, pos, held):
            return (0, 0.) if day == DATES[1] else (1, 0.5)
        result = self.replay(target=0.05, policy=policy)
        self.assertEqual(result.trades.action.tolist()[:3], ["BUY", "SELL", "BUY"])
        self.assertEqual(result.metrics["tp_exits"], 0)

    def test_no_intraday_information_in_policy(self):
        observed = []
        def policy(day, pos, held):
            observed.append((day, pos, held))
            return 1, 0.5
        self.replay(target=0.03, policy=policy, high=[100., 104., 104., 104., 104.])
        self.assertEqual([x[0] for x in observed], list(DATES[:-1]))
        self.assertEqual([x[1] for x in observed], [0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
