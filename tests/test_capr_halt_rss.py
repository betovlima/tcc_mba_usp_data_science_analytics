# -*- coding: utf-8 -*-
"""Regressao: RSS oficial Nasdaq em 29/07/2026 tinha campo hora preenchido com espacos.

Esse bug fazia parecer que vender CAPR durante halt era 'not_confirmed'.
Nunca baixar dados, enviar ordem ou rodar backtest durante estes testes.
"""
import unittest

import pandas as pd

from pesquisar_protecao_reversao_spyder import (
    _halt_datetime,
    _halt_order_status,
    parse_nasdaq_halt_rss,
)


RSS_CAPR_20260729 = b"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:n="http://www.nasdaqtrader.com/">
  <channel>
    <title>Nasdaq Trade Halts</title>
    <description>Nasdaq trading halt feed</description>
    <item>
      <title>CAPR</title>
      <n:IssueSymbol>CAPR</n:IssueSymbol>
      <n:Mkt>Q</n:Mkt>
      <n:ReasonCode>T3</n:ReasonCode>
      <n:HaltDate>07/29/2026</n:HaltDate>
      <n:HaltTime>06:55:00                      .000</n:HaltTime>
      <n:ResumptionDate>07/30/2026</n:ResumptionDate>
      <n:ResumptionQuoteTime>07:00:00</n:ResumptionQuoteTime>
      <n:ResumptionTradeTime>07:05:00</n:ResumptionTradeTime>
    </item>
  </channel>
</rss>
"""


class CAPRHistoricalTradingHaltTests(unittest.TestCase):
    def setUp(self):
        self.halts = parse_nasdaq_halt_rss(RSS_CAPR_20260729)

    def test_rss_reads_real_padded_halt_time_and_market(self):
        self.assertEqual(len(self.halts), 1)
        self.assertEqual(self.halts[0]["symbol"], "CAPR")
        self.assertEqual(self.halts[0]["market"], "Q")
        self.assertEqual(self.halts[0]["reason"], "T3")
        self.assertEqual(
            _halt_datetime(self.halts[0]["halt_date"], self.halts[0]["halt_time"]),
            pd.Timestamp("2026-07-29T06:55:00", tz="America/New_York"),
        )

    def test_all_five_20260729_0930_hypothetical_sells_invalid(self):
        # 3 daily policies and 2 intraday study simulations used same 09:30 fill.
        orders = [pd.Timestamp("2026-07-29T09:30:00", tz="America/New_York")] * 5
        statuses = [_halt_order_status(ts, self.halts)[0] for ts in orders]
        self.assertEqual(statuses, ["halted_at_order_time"] * 5)

    def test_no_hypothetical_fill_before_actual_resumption_next_day(self):
        before = pd.Timestamp("2026-07-30T07:04:59", tz="America/New_York")
        at = pd.Timestamp("2026-07-30T07:05:00", tz="America/New_York")
        self.assertEqual(_halt_order_status(before, self.halts)[0], "halted_at_order_time")
        self.assertEqual(_halt_order_status(at, self.halts)[0], "resumed_before_order")

    def test_empty_optional_resumption_does_not_mean_trading(self):
        row = {**self.halts[0], "resume_date": "", "resume_trade_time": ""}
        status, _ = _halt_order_status(
            pd.Timestamp("2026-07-29T09:30:00", tz="America/New_York"), [row]
        )
        self.assertEqual(status, "resume_unverified")

    def test_nonempty_unrecognized_time_is_never_silent(self):
        with self.assertRaises(ValueError):
            _halt_datetime("07/29/2026", "unknown .000")
        with self.assertRaises(ValueError):
            _halt_datetime("07/29/2026", "")


if __name__ == "__main__":
    unittest.main()
