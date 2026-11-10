# -*- coding: utf-8 -*-
"""TCC | Pesquisa exploratoria de reversao da posicao para CASH.

Abra este arquivo no Spyder e execute com F5. Nenhum argumento.
Entrada: output/reproducao_universo/predictions.csv ou ZIP oficial.
Fonte de precos: dados/u67 e dados/.u67_expansion_cache (somente leitura).
Saidas: out/ e out/pacote_protecao_reversao.zip (sobrescrito).
Nao modifica a Winner nem treina modelos. Piloto CAPR pode obter historico Alpaca 5Min, com cache.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine.configuracao import CONFIG
from engine.execucao import aplicar_deslizamento, calcular_taxas_referencia
from reproducao.preparacao import load_raw_bar_file, load_actions_file, split_normalize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
ARCHIVE = "pacote_protecao_reversao.zip"
INITIAL = float(CONFIG.initial_capital)
NY = "America/New_York"
PROFILES = ("ema", "atr", "combinada")


def local_day(value):
    x = pd.Timestamp(value)
    if pd.isna(x):
        raise ValueError("Data invalida no replay")
    if x.tzinfo is None:
        x = x.tz_localize("UTC")
    return x.tz_convert(NY).date()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_replay():
    directory = ROOT / "output" / "reproducao_universo"
    candidates = [
        directory / "predictions.csv",
        directory / "pacote_reproducao_universo.zip",
        ROOT / "out" / "pacote_reproducao_universo.zip",
        ROOT / "pacote_reproducao_universo.zip",
    ]
    source = next((x for x in candidates if x.is_file()), None)
    if source is None:
        raise FileNotFoundError(
            "Nao achei o ultimo replay oficial. Execute reproduzir_experimento.py "
            "no Spyder (F5) para gerar output/reproducao_universo/. "
            "Esta pesquisa nao refaz downloads nem modifica o checkpoint."
        )
    meta = {}
    if source.suffix == ".csv":
        frame = pd.read_csv(source, float_precision="round_trip")
        p = source.parent / "reproducao_universo.json"
        if p.exists():
            meta = json.loads(p.read_text(encoding="utf-8"))
    else:
        with zipfile.ZipFile(source) as z:
            def name(target):
                options = [x for x in z.namelist() if x == target or x.endswith("/" + target)]
                return min(options, key=len) if options else None
            p = name("predictions.csv")
            if not p:
                raise RuntimeError("Pacote nao contem predictions.csv")
            frame = pd.read_csv(io.BytesIO(z.read(p)), float_precision="round_trip")
            m = name("reproducao_universo.json")
            if m:
                meta = json.loads(z.read(m))
    if "timestamp" not in frame and "Unnamed: 0" in frame:
        frame.rename(columns={"Unnamed: 0": "timestamp"}, inplace=True)
    required = {"timestamp", "decision_date", "selected_asset", "strategy_equity"}
    if required - set(frame.columns):
        raise RuntimeError("Campos ausentes: " + repr(sorted(required - set(frame.columns))))
    frame["date"] = frame["timestamp"].map(local_day)
    frame["decision"] = frame["decision_date"].map(local_day)
    frame["selected_asset"] = frame["selected_asset"].fillna("").astype(str).str.upper().str.strip()
    frame["strategy_equity"] = pd.to_numeric(frame["strategy_equity"], errors="raise")
    if "walk_forward_fold" not in frame:
        frame["walk_forward_fold"] = "nao_informado"
    frame["walk_forward_fold"] = frame["walk_forward_fold"].fillna("nao_informado").astype(str)
    frame.sort_values("date", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    if frame["date"].duplicated().any() or len(frame) < 100:
        raise RuntimeError("Historico duplicado ou insuficiente")
    if (frame["strategy_equity"] <= 0).any() or (frame["selected_asset"] == "").any():
        raise RuntimeError("Patrimonio ou posicao invalida; interrompido")
    return source, frame, meta


def prepare_indicators(raw):
    x = raw.copy()
    close = x["close"]
    prev = close.shift(1)
    true_range = pd.concat([
        x["high"] - x["low"], (x["high"] - prev).abs(),
        (x["low"] - prev).abs()
    ], axis=1).max(axis=1)
    x["atr14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    x["ema10"] = close.ewm(span=10, adjust=False).mean()
    x["ema20"] = close.ewm(span=20, adjust=False).mean()
    x["ema10_slope3"] = x["ema10"].pct_change(3)
    x["return5"] = close.pct_change(5)
    x["peak20"] = x["high"].rolling(20, min_periods=20).max()
    x["pullback_atr"] = (x["peak20"] - close) / x["atr14"].replace(0, np.nan)
    x["ema_signal"] = (close < x["ema10"]) & (x["ema10"] < x["ema20"]) & (x["ema10_slope3"] < 0)
    x["atr_signal"] = (x["pullback_atr"] >= 2.0) & (close < x["ema10"]) & (x["return5"] < 0)
    x["combinada_signal"] = x["ema_signal"] & x["atr_signal"]
    x["recover"] = (close > x["ema10"]) & (x["ema10_slope3"] > 0) & (close > prev)
    return x


def load_prices(assets):
    roots = [
        ROOT / "dados" / "u67",
        ROOT / "dados" / ".u67_expansion_cache" / "full_raw_20261006",
    ]
    prices, provenance = {}, {}
    for asset in sorted(assets):
        match = next((
            (d / "raw_bars" / (asset + ".csv"), d / "corporate_actions" / (asset + ".csv"))
            for d in roots
            if (d / "raw_bars" / (asset + ".csv")).is_file()
            and (d / "corporate_actions" / (asset + ".csv")).is_file()
        ), None)
        if match is None:
            raise FileNotFoundError("RAW/corporate_actions ausentes para " + asset + ". Cache congelado obrigatorio.")
        bars, actions = match
        raw = load_raw_bar_file(bars, float_precision="round_trip")
        adjusted, _ = split_normalize(raw, load_actions_file(actions))
        adjusted = adjusted.copy()
        adjusted["date"] = [local_day(x) for x in adjusted.index]
        adjusted.set_index("date", inplace=True)
        if adjusted.index.has_duplicates:
            raise RuntimeError("Datas duplicadas nos precos: " + asset)
        if (adjusted[["open", "high", "low", "close"]] <= 0).any().any():
            raise RuntimeError("Precos invalidos: " + asset)
        prices[asset] = prepare_indicators(adjusted)
        provenance[asset] = {
            "bars_sha256": sha256(bars),
            "actions_sha256": sha256(actions),
            "path": str(bars.relative_to(ROOT)),
        }
    return prices, provenance


def bar(prices, asset, date):
    try:
        row = prices[asset].loc[date]
    except KeyError as exc:
        raise RuntimeError("Sem preco de " + asset + " em " + str(date) + "; sem preenchimento futuro") from exc
    if not np.isfinite([row["open"], row["close"]]).all():
        raise RuntimeError("Preco nao-finito: " + asset + " " + str(date))
    return row


def has_signal(prices, asset, day, profile):
    if asset == "CASH":
        return False
    return bool(bar(prices, asset, day)[profile + "_signal"])


def has_recovered(prices, asset, day):
    return bool(bar(prices, asset, day)["recover"])


def execute(cash, quantity, held, target, session, prices):
    operations = []
    if held != "CASH":
        price = float(aplicar_deslizamento(float(bar(prices, held, session)["open"]), "SELL", CONFIG))
        fee = calcular_taxas_referencia("SELL", quantity, price, CONFIG)["total_fee"]
        cash += quantity * price - fee
        operations.append(("SELL", held, quantity, price, fee))
        quantity = 0.0
    if target != "CASH":
        price = float(aplicar_deslizamento(float(bar(prices, target, session)["open"]), "BUY", CONFIG))
        quantity = cash / price
        for _ in range(25):
            fee = calcular_taxas_referencia("BUY", quantity, price, CONFIG)["total_fee"]
            new_quantity = max(0.0, (cash - fee) / price)
            if abs(new_quantity - quantity) < 1e-10:
                quantity = new_quantity
                break
            quantity = new_quantity
        fee = calcular_taxas_referencia("BUY", quantity, price, CONFIG)["total_fee"]
        cash -= quantity * price + fee
        operations.append(("BUY", target, quantity, price, fee))
    return cash, quantity, operations


def simulate(replay, prices, profile):
    cash, quantity, held, blocked = INITIAL, 0.0, "CASH", None
    days, trades = [], []
    for r in replay.itertuples(index=False):
        session, decision, winner = r.date, r.decision, r.selected_asset
        target, reason = winner, "WINNER"
        if profile != "control":
            if blocked:
                if winner not in (blocked, "CASH"):
                    blocked = None
                    reason = "NOVA_ROTACAO_WINNER"
                elif winner == blocked and has_recovered(prices, blocked, decision):
                    blocked = None
                    reason = "REENTRADA"
                else:
                    target = "CASH"
                    reason = "PROTECAO_MANTIDA"
            elif held != "CASH" and winner == held and has_signal(prices, held, decision, profile):
                blocked = held
                target = "CASH"
                reason = "REVERSAO_" + profile.upper()
        if target != held:
            old = held
            cash, quantity, executions = execute(cash, quantity, held, target, session, prices)
            for action, asset, qty, price, fee in executions:
                trades.append({
                    "profile": profile, "session": str(session), "decision_date": str(decision),
                    "action": action, "asset": asset, "quantity": qty,
                    "execution_price": price, "fee": fee, "reason": reason,
                    "from_asset": old, "to_asset": target,
                })
            held = target
        equity = cash if held == "CASH" else cash + quantity * float(bar(prices, held, session)["close"])
        if equity <= 0 or not np.isfinite(equity):
            raise RuntimeError("Capital invalido em " + str(session))
        days.append({
            "session": session, "date": str(session), "fold": r.walk_forward_fold,
            "profile": profile, "asset": held, "control_asset": winner,
            "equity": equity, "cash": held == "CASH", "reason": reason,
        })
    if held != "CASH":
        final = replay["date"].iloc[-1]
        price = float(aplicar_deslizamento(float(bar(prices, held, final)["close"]), "SELL", CONFIG))
        fee = calcular_taxas_referencia("SELL", quantity, price, CONFIG)["total_fee"]
        cash += quantity * price - fee
        days[-1]["equity"] = cash
        trades.append({
            "profile": profile, "session": str(final), "decision_date": str(final),
            "action": "FINAL_SELL", "asset": held, "quantity": quantity,
            "execution_price": price, "fee": fee, "reason": "FINAL_LIQUIDATION",
            "from_asset": held, "to_asset": "CASH",
        })
    result = pd.DataFrame(days)
    result["peak"] = result["equity"].cummax()
    result["drawdown"] = result["equity"] / result["peak"] - 1
    result["daily_return"] = result["equity"].pct_change()
    result.loc[0, "daily_return"] = result["equity"].iloc[0] / INITIAL - 1
    return result, pd.DataFrame(trades)


def metrics(df):
    years = max((df["session"].iloc[-1] - df["session"].iloc[0]).days / 365.25, 1 / 365.25)
    daily = df["daily_return"].to_numpy(dtype=float)
    std = daily.std(ddof=1)
    capital = float(df["equity"].iloc[-1])
    return {
        "capital_final": capital,
        "retorno_total": capital / INITIAL - 1,
        "cagr": (capital / INITIAL) ** (1 / years) - 1,
        "max_drawdown": float(df["drawdown"].min()),
        "sharpe": float(np.sqrt(252) * daily.mean() / std) if std > 0 else None,
        "dias_cash": int(df["cash"].sum()),
        "percentual_cash": float(df["cash"].mean()),
        "entradas_cash": int((df["cash"] & ~df["cash"].shift(fill_value=False)).sum()),
    }


def fold_metrics(df):
    result = []
    for fold, group in df.groupby("fold", sort=False):
        changes = group["daily_return"].to_numpy(dtype=float)
        balance = np.cumprod(1 + changes)
        peaks = np.maximum.accumulate(np.r_[1., balance])[1:]
        result.append({
            "profile": df["profile"].iloc[0],
            "fold": fold, "sessions": len(group),
            "return": float(balance[-1] - 1),
            "drawdown": float((balance / peaks - 1).min()),
            "cash_days": int(group["cash"].sum()),
        })
    return result


def make_charts(frames, trade_log, prices, target):
    paths = []
    base = frames["control"]
    dates = pd.to_datetime(base["date"])
    fig, ax = plt.subplots(figsize=(13, 6))
    for name, d in frames.items():
        ax.plot(dates, d["equity"], label=name, linewidth=1.4)
    ax.set(yscale="log", ylabel="Capital (US$, escala log)", title="Capitalizacao: Control e protecoes por reversao")
    ax.legend()
    ax.grid(alpha=.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    paths.append(("reversao_patrimonio.png", fig))

    fig, ax = plt.subplots(figsize=(13, 5))
    for name, d in frames.items():
        ax.plot(dates, 100 * d["drawdown"], label=name, linewidth=1.3)
    ax.set(ylabel="Drawdown (%)", title="Queda do patrimonio desde o maior valor anterior")
    ax.legend()
    ax.grid(alpha=.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    paths.append(("reversao_drawdown.png", fig))

    fig, axs = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                            gridspec_kw={"height_ratios": [3, 1]})
    for name, d in frames.items():
        axs[0].plot(dates, d["equity"], label=name)
        if name != "control":
            axs[1].step(dates, d["cash"].astype(int), where="post", label=name)
    axs[0].set_yscale("log")
    axs[0].set_title("Capital e sessoes defensivas em CASH")
    axs[0].legend()
    axs[0].grid(alpha=.25)
    axs[1].set_yticks([0, 1])
    axs[1].set_ylabel("CASH")
    axs[1].legend(ncol=3)
    axs[1].grid(alpha=.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    paths.append(("reversao_cash.png", fig))

    if "CAPR" in prices and (base["asset"] == "CAPR").any():
        import datetime
        ds = base.loc[base["asset"] == "CAPR", "session"]
        start, end = min(ds) - datetime.timedelta(days=35), max(ds) + datetime.timedelta(days=7)
        bars = prices["CAPR"]
        bars = bars.loc[(bars.index >= start) & (bars.index <= end)]
        fig, ax = plt.subplots(figsize=(13, 5))
        bd = pd.to_datetime([str(t) for t in bars.index])
        ax.plot(bd, bars["close"], label="Preco CAPR")
        ax.plot(bd, bars["ema10"], label="EMA 10")
        ax.plot(bd, bars["ema20"], label="EMA 20")
        for name, marker in [("ema", "v"), ("atr", "x"), ("combinada", "s")]:
            e = trade_log.loc[
                (trade_log["profile"] == name) &
                (trade_log["asset"] == "CAPR") &
                (trade_log["action"] == "SELL")
            ]
            # Exibir apenas as vendas de protecao para CASH, na janela visivel.
            # A coordenada Y deve ser o preco executado na abertura seguinte,
            # e nao o fechamento do mesmo dia.
            e = e.loc[
                (e["to_asset"] == "CASH") &
                (e["reason"].astype(str).str.startswith("REVERSAO_"))
            ].copy()
            if not e.empty:
                sell_dates = pd.to_datetime(e["session"])
                e = e.loc[sell_dates.dt.date.isin(bars.index)]
                if not e.empty:
                    x = pd.to_datetime(e["session"]).to_numpy()
                    y = pd.to_numeric(e["execution_price"], errors="raise").to_numpy()
                    ax.scatter(x, y, label="Protecao CASH " + name,
                               marker=marker, s=70, zorder=5)
        ax.set(title="CAPR: medias e saidas simuladas na abertura seguinte", ylabel="Preco (ajustado por splits)")
        ax.legend()
        ax.grid(alpha=.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        paths.append(("reversao_capr.png", fig))
    for name, fig in paths:
        fig.savefig(target / name, dpi=145, bbox_inches="tight")
    # Mantem os graficos disponiveis no painel Plots do Spyder.
    plt.show(block=False)
    return [name for name, _ in paths]



# PILOTO INTRADIARIO CAPR | fontes SIP RAW, sem alterar o Control.
PILOT_BEGIN = "2026-07-27"
PILOT_END = "2026-08-22"  # exclusivo
INTRADAY_CSV = ROOT / "dados" / ".intraday_cache" / "CAPR_20260727_20260822_5min_sip_raw.csv"


def load_capr_intraday():
    """Carrega cache congelado; na primeira vez baixa apenas CAPR e esta janela."""
    from reproducao.dados import load_alpaca_credentials
    if not INTRADAY_CSV.is_file():
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import Adjustment, DataFeed
        creds = load_alpaca_credentials(ROOT)
        client = StockHistoricalDataClient(
            api_key=creds.api_key, secret_key=creds.secret_key
        )
        req = StockBarsRequest(
            symbol_or_symbols="CAPR",
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=pd.Timestamp(PILOT_BEGIN, tz=NY).to_pydatetime(),
            end=pd.Timestamp(PILOT_END, tz=NY).to_pydatetime(),
            adjustment=Adjustment.RAW,
            feed=DataFeed.SIP,
            limit=10000,
        )
        print("[CAPR-5Min] baixando historico SIP RAW...", flush=True)
        frame = client.get_stock_bars(req).df
        if isinstance(frame.index, pd.MultiIndex):
            frame = frame.xs("CAPR", level="symbol")
        frame = frame.reset_index()
        frame.columns = [str(x).lower() for x in frame.columns]
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        if frame.empty or not set(required).issubset(frame.columns):
            raise RuntimeError("Sem candles intradiarios OHLCV completos da Alpaca.")
        frame = frame[required].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp")
        INTRADAY_CSV.parent.mkdir(parents=True, exist_ok=True)
        temp = INTRADAY_CSV.with_suffix(".tmp")
        frame.to_csv(temp, index=False, float_format="%.17g")
        os.replace(temp, INTRADAY_CSV)
    else:
        print("[CAPR-5Min] cache existente, sem novo download", flush=True)
    raw = load_raw_bar_file(INTRADAY_CSV, float_precision="round_trip")
    actions = ROOT / "dados" / ".u67_expansion_cache" / "full_raw_20261006" / "corporate_actions" / "CAPR.csv"
    if not actions.is_file():
        raise FileNotFoundError("Falta corporate_actions/CAPR.csv do snapshot TCC")
    bars, _ = split_normalize(raw, load_actions_file(actions))
    local = bars.index.tz_convert(NY)
    bars["date"] = local.date
    bars["clock"] = local.strftime("%H:%M")
    bars["local"] = local.tz_localize(None)
    bars = bars.loc[(bars["clock"] >= "09:30") & (bars["clock"] < "16:00")].copy()
    if bars.empty or bars.index.has_duplicates:
        raise RuntimeError("Barras intradiarias vazias ou com timestamps repetidos")
    return bars, {
        "cache": str(INTRADAY_CSV.relative_to(ROOT)),
        "sha256_raw": sha256(INTRADAY_CSV),
        "sha256_actions": sha256(actions),
        "bars": int(len(bars)),
        "feed": "sip", "timeframe": "5Min", "adjustment": "raw + split_normalize",
    }


def intraday_midpoint(bars, prices, day):
    """12h30 usa barras ate 12h25. Ordem no OPEN da barra 12h35."""
    session = bars.loc[bars["date"] == day]
    known = session.loc[(session["clock"] >= "09:30") & (session["clock"] <= "12:25")]
    execution = session.loc[session["clock"] == "12:35"]
    if len(known) < 30 or "12:25" not in set(known["clock"]) or len(execution) != 1:
        raise RuntimeError("Faltam barras completas 12h25/12h35 em " + str(day))
    recent = prices["CAPR"].loc[prices["CAPR"].index < day,
                                ["open", "high", "low", "close", "volume"]].copy()
    if len(recent) < 25:
        raise RuntimeError("Historico insuficiente para indicador as 12h30")
    recent.loc[day, :] = [
        float(known["open"].iloc[0]), float(known["high"].max()),
        float(known["low"].min()), float(known["close"].iloc[-1]),
        float(known["volume"].sum()),
    ]
    indicators = prepare_indicators(recent).iloc[-1]
    return indicators, float(execution["open"].iloc[0])


def execute_capr(quantity, cash, action, price):
    px = float(aplicar_deslizamento(float(price), action, CONFIG))
    if action == "SELL":
        fee = float(calcular_taxas_referencia("SELL", quantity, px, CONFIG)["total_fee"])
        cash += quantity * px - fee
        traded, quantity = quantity, 0.0
    else:
        quantity = cash / px
        for _ in range(25):
            fee = float(calcular_taxas_referencia("BUY", quantity, px, CONFIG)["total_fee"])
            updated = max(0.0, (cash - fee) / px)
            if abs(updated - quantity) < 1e-10:
                quantity = updated
                break
            quantity = updated
        fee = float(calcular_taxas_referencia("BUY", quantity, px, CONFIG)["total_fee"])
        cash -= quantity * px + fee
        traded = quantity
    return quantity, cash, {"action": action, "quantity": traded, "price": px, "fee": fee}


def study_capr_intraday(replay, frames, trades, prices, folder):
    """Compara politicas em episodios CAPR, nao em toda a carteira."""
    bars, provenance = load_capr_intraday()
    window = replay.loc[
        (replay["date"] >= pd.Timestamp(PILOT_BEGIN).date()) &
        (replay["date"] < pd.Timestamp(PILOT_END).date())
    ].copy()
    window["_part"] = (window["selected_asset"] != window["selected_asset"].shift()).cumsum()
    segments = [part for _, part in window.groupby("_part")
                if part["selected_asset"].iloc[0] == "CAPR" and len(part) >= 2]
    buys = trades.loc[
        (trades["profile"] == "control") & (trades["asset"] == "CAPR") &
        (trades["action"] == "BUY")
    ]
    daily_rows, events, signals = [], [], []
    cases = ("control", "daily", "midday")
    studied = 0
    for episode, segment in enumerate(segments, start=1):
        first = segment["date"].iloc[0]
        purchase = buys.loc[buys["session"] == str(first)]
        if purchase.empty:
            print("[CAPR-5Min] episodio sem compra inicial dentro da janela:", first)
            continue
        first_control_equity = float(
            frames["control"].loc[frames["control"]["session"] == first, "equity"].iloc[0]
        )
        initial_qty = float(purchase["quantity"].iloc[-1])
        initial_cash = first_control_equity - initial_qty * float(bar(prices, "CAPR", first)["close"])
        if initial_qty <= 0 or initial_cash < -0.05:
            raise RuntimeError("Quantidade ou caixa inicial CAPR nao conciliado.")
        intraday = {day: intraday_midpoint(bars, prices, day)
                    for day in segment["date"]}
        studied += 1
        for policy in cases:
            quantity, cash, blocked = initial_qty, initial_cash, False
            for idx, r in enumerate(segment.itertuples(index=False)):
                day = r.date
                action_today = (idx == 0)  # entrada original da Winner
                if idx:
                    previous = segment["date"].iloc[idx - 1]
                    indicator = bar(prices, "CAPR", previous)
                    action = None
                    if blocked and bool(indicator["recover"]):
                        action = "BUY"
                    elif not blocked and policy != "control" and bool(indicator["combinada_signal"]):
                        action = "SELL"
                    if action:
                        quantity, cash, ev = execute_capr(
                            quantity, cash, action, float(bar(prices, "CAPR", day)["open"])
                        )
                        blocked = action == "SELL"
                        events.append({"episode": episode, "policy": policy, "date": str(day),
                                       "time": "09:30", "reason": "fechamento anterior", **ev})
                        action_today = True
                indicator, px = intraday[day]
                signals.append({
                    "episode": episode, "policy": policy, "date": str(day),
                    "observed_through": "12:30 NY, barra iniciada as 12:25",
                    "execution_after": "12:35 NY",
                    "ema": bool(indicator["ema_signal"]),
                    "atr": bool(indicator["atr_signal"]),
                    "combined": bool(indicator["combinada_signal"]),
                    "recovery": bool(indicator["recover"]),
                    "action_at_open": action_today,
                    "in_cash_at_midday": blocked,
                })
                if policy == "midday" and not action_today:
                    action = ("BUY" if blocked and bool(indicator["recover"]) else
                              "SELL" if not blocked and bool(indicator["combinada_signal"]) else None)
                    if action:
                        quantity, cash, ev = execute_capr(quantity, cash, action, px)
                        blocked = action == "SELL"
                        events.append({"episode": episode, "policy": policy, "date": str(day),
                                       "time": "12:35", "reason": "sinal intradiario 12:30", **ev})
                equity = float(cash if blocked else cash + quantity * float(
                    bar(prices, "CAPR", day)["close"]))
                daily_rows.append({
                    "episode": episode, "policy": policy, "date": str(day),
                    "equity": equity, "cash": blocked, "qty": quantity,
                    "control_equity": float(r.strategy_equity),
                })
    if studied == 0:
        return {"status": "no_eligible_entry", "message":
                "Nao foi encontrada uma compra CAPR que inicia episodio dentro da janela."}
    results = pd.DataFrame(daily_rows)
    operations = pd.DataFrame(events, columns=[
        "episode", "policy", "date", "time", "reason", "action", "quantity", "price", "fee"
    ])
    signals = pd.DataFrame(signals)
    comparison = results.groupby(["episode", "policy"], sort=False).agg(
        start=("date", "first"), end=("date", "last"),
        first_equity=("equity", "first"), last_equity=("equity", "last"),
        cash_sessions=("cash", "sum"),
    ).reset_index()
    comparison["return_from_first_close"] = (
        comparison["last_equity"] / comparison["first_equity"] - 1
    )
    for filename, data in [
        ("reversao_intradiario_diario.csv", results),
        ("reversao_intradiario_operacoes.csv", operations),
        ("reversao_intradiario_sinais.csv", signals),
        ("reversao_intradiario_comparacao.csv", comparison),
    ]:
        data.to_csv(folder / filename, index=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    for (episode, policy), group in results.groupby(["episode", "policy"]):
        ax.plot(pd.to_datetime(group["date"]), group["equity"],
                label=str(episode) + " " + policy, linewidth=1.5)
    ax.set(title="CAPR: Control, protecao diaria e protecao as 12h30",
           ylabel="Patrimonio, episodio CAPR (US$)")
    ax.legend()
    ax.grid(alpha=.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(folder / "reversao_intradiario_patrimonio.png", dpi=145)

    fig2, ax2 = plt.subplots(figsize=(13, 5))
    selected_days = {"2026-07-29", "2026-07-30", "2026-08-14",
                     "2026-08-17", "2026-08-18"}
    view = bars.loc[bars["date"].astype(str).isin(selected_days)]
    for day, group in view.groupby("date"):
        ax2.plot(group["local"], group["close"], label=str(day), linewidth=1)
    for _, op in operations.loc[operations["time"] == "12:35"].iterrows():
        marker = "^" if op["action"] == "BUY" else "v"
        ax2.scatter(pd.Timestamp(op["date"] + " 12:35"),
                    float(op["price"]), marker=marker, s=75, zorder=5)
    ax2.set(title="CAPR: candles 5Min e execucoes 12h35 (apos confirmar 12h30)",
            ylabel="Preco ajustado por splits")
    ax2.legend(ncol=3)
    ax2.grid(alpha=.25)
    fig2.autofmt_xdate()
    fig2.tight_layout()
    fig2.savefig(folder / "reversao_intradiario_capr.png", dpi=145)
    plt.show(block=False)
    return {
        "status": "completed",
        "scope": "episodios CAPR apenas; NAO e portfolio completo",
        "episodes": studied,
        "window": [PILOT_BEGIN, PILOT_END],
        "timing": {"preopen": "fechamento anterior -> 09:30",
                   "midpoint": "barra 12:25 encerrada -> ordem 12:35",
                   "close": "16:00 -> proxima sessao"},
        "max_one_protective_action_per_session": True,
        "cache": provenance,
        "comparisons": comparison.to_dict(orient="records"),
        "limitations": [
            "Nao recalcula ranking Winner apos CASH.",
            "Janela curta escolhida por episodio de CAPR, nao validacao fora da amostra.",
            "Intradiario SIP RAW normalizado por splits; avaliar dividendos e qualidade.",
            "Nao incorpora impacto de mercado, latencia ou restricoes regulatórias de day trading.",
            "Paridade do Control diario continua verificada separadamente.",
        ],
    }



def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source, replay, original = load_replay()
    print("[fonte]", source, flush=True)
    assets = set(replay["selected_asset"]) - {"CASH"}
    prices, provenance = load_prices(assets)
    # Falha explicita se faltar preco; nunca completar por interpolacao.
    for r in replay.itertuples(index=False):
        if r.selected_asset != "CASH":
            bar(prices, r.selected_asset, r.date)
            bar(prices, r.selected_asset, r.decision)
    frames, trades = {}, []
    for name in ("control",) + PROFILES:
        print("[replay]", name, flush=True)
        frames[name], events = simulate(replay, prices, name)
        trades.append(events)
    log = pd.concat(trades, ignore_index=True)
    official = replay["strategy_equity"].to_numpy(dtype=float)
    synthetic = frames["control"]["equity"].to_numpy(dtype=float)
    max_gap = float(np.max(np.abs(official - synthetic) / np.maximum(official, 1)))
    parity_ok = max_gap <= 1e-4
    print("[paridade Control] desvio maximo relativo:", max_gap, flush=True)
    summary = pd.DataFrame([{"profile": k, **metrics(v)} for k, v in frames.items()])
    base_capital = float(summary.loc[summary["profile"] == "control", "capital_final"].iloc[0])
    summary["delta_capital_vs_control"] = summary["capital_final"] - base_capital
    summary["paridade_ok"] = parity_ok
    folds = pd.DataFrame([row for d in frames.values() for row in fold_metrics(d)])
    daily = pd.concat(frames.values(), ignore_index=True)
    result = {
        "schema": "mct_cash_reversal_exploratory_v1",
        "status": "comparavel" if parity_ok else "divergencia_control",
        "fonte": str(source.relative_to(ROOT)),
        "fonte_sha256": sha256(source),
        "checkpoint_original": original.get("checkpoint"),
        "capital_original": float(official[-1]),
        "desvio_maximo_relativo_control": max_gap,
        "diferenca_final_control": float(synthetic[-1] - official[-1]),
        "paridade_ok": parity_ok,
        "regras": {
            "ema": "close < ema10 < ema20 e inclinacao EMA10 3d negativa",
            "atr": "distancia ao pico20 >= 2 ATR14 e close < EMA10 e retorno5 negativo",
            "combinada": "ema e atr simultaneos",
            "reentrada": "close > ema10 e inclinacao EMA10 3d positiva e close > fechamento anterior",
            "rotacao": "troca definida pela Winner tem prioridade",
            "causalidade": "sinal no fechamento da data decision_date; ordem na abertura execution_date",
            "custos": "engine.execucao com CONFIG original",
        },
        "limitacoes": [
            "Exploratorio: sem calibracao temporal de parametros em folds de treino.",
            "Universo congelado foi selecionado retrospectivamente; nao ha validacao prospectiva.",
            "Nao recalcula ranking Winner apos CASH; segue as escolhas congeladas.",
            "Nao oferece protecao contra gaps ocorridos na abertura.",
            "A serie e split-normalizada; efeitos de dividendos exigem auditoria adicional.",
            "Resultados nao devem ser interpretados se paridade_ok for false.",
        ],
        "comparacao": summary.to_dict(orient="records"),
        "fonte_precos": provenance,
    }
    with tempfile.TemporaryDirectory(prefix="estudo_", dir=OUT) as tmp:
        target = Path(tmp)
        summary.to_csv(target / "reversao_comparacao.csv", index=False)
        daily.drop(columns="session").to_csv(target / "reversao_diario.csv", index=False)
        folds.to_csv(target / "reversao_folds.csv", index=False)
        log.to_csv(target / "reversao_operacoes.csv", index=False)
        (target / "reversao_resumo.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        plots = make_charts(frames, log, prices, target)
        try:
            intraday = study_capr_intraday(replay, frames, log, prices, target)
        except Exception as exc:
            intraday = {"status": "unavailable", "error_type": type(exc).__name__,
                        "message": str(exc)[:600]}
            print("[CAPR-5Min] piloto indisponivel:", str(exc)[:300], flush=True)
        (target / "reversao_intradiario_status.json").write_text(
            json.dumps(intraday, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        print("[CAPR-5Min] status:", intraday["status"], flush=True)
        (target / "LEIA_ME.txt").write_text(
            "Estudo de reversao CASH, apenas exploratorio. Control original sem mudancas.\n"
            "Conferir paridade_ok antes de comparar patrimonio.\n"
            "Parametros nao foram ajustados nos folds de treinamento.\n", encoding="utf-8"
        )
        package = target / ARCHIVE
        with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(target.iterdir()):
                if p.is_file() and p != package:
                    z.write(p, arcname=p.name)
        for p in sorted(target.iterdir()):
            if p.is_file() and p != package:
                os.replace(p, OUT / p.name)
        os.replace(package, OUT / ARCHIVE)
    print("\nRESULTADOS\n", summary.to_string(index=False), flush=True)
    if not parity_ok:
        print("\nATENCAO: o Control reconstruido divergiu do original. "
              "Nao concluir que houve melhora antes de investigar.", flush=True)
    print("\nZIP fixo:", OUT / ARCHIVE, "\nGraficos:", ", ".join(plots), flush=True)
    return result


if __name__ == "__main__":
    RESULTADO = main()
