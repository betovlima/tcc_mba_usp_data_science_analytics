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
    # Sem negociacoes em algum bloco de 5 minutos, a Alpaca pode omitir
    # aquela barra. Usar apenas o ultimo candle REAL ja encerrado, desde
    # que nao tenha mais de 15 minutos de defasagem.
    execution = session.loc[(session["clock"] >= "12:35") &
                            (session["clock"] <= "12:50")]
    if (len(known) < 15 or known.empty or known["clock"].iloc[-1] < "12:10"
            or execution.empty):
        raise RuntimeError(
            "CAPR " + str(day) + ": nao ha dados recentes/executaveis "
            "ao redor das 12h30. Nenhum preco sera interpolado."
        )
    execution = execution.iloc[:1]
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
    return (indicators, float(execution["open"].iloc[0]),
            str(execution["clock"].iloc[0]))


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



def audit_pilot_liquidity(bars, operations, initial_entries, target):
    """Audita participacao em volume REAL, sem afirmar capacidade de execucao.

    O limite de 5% do volume de um candle e apenas uma hipotese conservadora
    para diagnostico. Nao e uma regra de mercado nem um modelo de fill.
    """
    records = []
    orders = [*initial_entries, *operations.to_dict(orient="records")]
    for operation in orders:
        day = str(operation["date"])
        clock = str(operation["time"])
        shares = float(operation["quantity"])
        price = float(operation["price"])
        same_day = bars.loc[bars["date"].astype(str) == day]
        candle = same_day.loc[same_day["clock"] == clock]
        candle_volume = float(candle["volume"].sum()) if not candle.empty else None
        day_volume = float(same_day["volume"].sum()) if not same_day.empty else None
        frac_candle = (shares / candle_volume
                       if candle_volume is not None and candle_volume > 0 else None)
        frac_day = (shares / day_volume
                    if day_volume is not None and day_volume > 0 else None)
        if candle_volume is None:
            status = "missing_intraday_execution_bar"
        elif candle_volume <= 0:
            status = "zero_observed_volume"
        elif shares > 0.05 * candle_volume:
            status = "exceeds_5pct_candle_volume"
        else:
            status = "within_5pct_candle_volume"
        records.append({
            "episode": operation["episode"],
            "policy": operation["policy"],
            "date": day,
            "time": clock,
            "action": operation["action"],
            "quantity_shares": shares,
            "price_assumed": price,
            "notional_usd": shares * price,
            "observed_5min_volume": candle_volume,
            "observed_day_volume": day_volume,
            "ratio_quantity_to_5min_volume": frac_candle,
            "ratio_quantity_to_daily_volume": frac_day,
            "hypothetical_5pct_candle_capacity_shares": (
                0.05 * candle_volume if candle_volume is not None else None
            ),
            "status": status,
        })
    audited = pd.DataFrame(records)
    audited.to_csv(target / "reversao_intradiario_liquidez.csv", index=False)
    return audited


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
    daily_rows, events, signals, coverage, initial_entries = [], [], [], [], []
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
        initial_entries.append({
            "episode": episode, "policy": "control_initial_entry",
            "date": str(first), "time": "09:30", "action": "BUY",
            "quantity": initial_qty,
            "price": float(purchase["execution_price"].iloc[-1]),
        })
        # Uma sessao sem candles suficientes perto de 12h30 nao deve
        # cancelar todas as outras sessoes. Nao presumir sinal ou preco.
        intraday = {}
        for day in segment["date"]:
            candles = bars.loc[bars["date"] == day]
            before = candles.loc[candles["clock"] <= "12:25"]
            after = candles.loc[(candles["clock"] >= "12:35") &
                                (candles["clock"] <= "12:50")]
            audit = {
                "episode": episode, "date": str(day),
                "bar_count": int(len(candles)),
                "last_known_before_1230": (
                    str(before["clock"].iloc[-1]) if not before.empty else None
                ),
                "first_executable_after_1230": (
                    str(after["clock"].iloc[0]) if not after.empty else None
                ),
            }
            try:
                intraday[day] = intraday_midpoint(bars, prices, day)
                audit.update({"status": "valid", "reason": None})
            except RuntimeError as exc:
                intraday[day] = None
                audit.update({
                    "status": "skipped_missing_5min_bars", "reason": str(exc)
                })
                print("[CAPR-5Min] pulando avaliacao 12h30", str(day),
                      "|", str(exc), flush=True)
            coverage.append(audit)
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
                snapshot = intraday[day]
                indicator = snapshot[0] if snapshot is not None else None
                px = snapshot[1] if snapshot is not None else None
                executable_clock = snapshot[2] if snapshot is not None else None
                signals.append({
                    "episode": episode, "policy": policy, "date": str(day),
                    "status": ("valid" if indicator is not None
                               else "skipped_missing_5min_bars"),
                    "observed_through": (
                        "12:30 NY, ultimo candle completo ate 12:25"
                        if indicator is not None else None
                    ),
                    "execution_after": (
                        executable_clock + " NY" if executable_clock is not None else None
                    ),
                    "ema": bool(indicator["ema_signal"]) if indicator is not None else None,
                    "atr": bool(indicator["atr_signal"]) if indicator is not None else None,
                    "combined": bool(indicator["combinada_signal"]) if indicator is not None else None,
                    "recovery": bool(indicator["recover"]) if indicator is not None else None,
                    "action_at_open": action_today,
                    "in_cash_at_midday": blocked,
                })
                if policy == "midday" and not action_today and indicator is not None:
                    action = ("BUY" if blocked and bool(indicator["recover"]) else
                              "SELL" if not blocked and bool(indicator["combinada_signal"]) else None)
                    if action:
                        quantity, cash, ev = execute_capr(quantity, cash, action, px)
                        blocked = action == "SELL"
                        events.append({"episode": episode, "policy": policy, "date": str(day),
                                       "time": executable_clock, "reason": "sinal intradiario 12:30", **ev})
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
    liquidity = audit_pilot_liquidity(bars, operations, initial_entries, folder)
    liquidity_flags = liquidity.loc[
        liquidity["status"] != "within_5pct_candle_volume"
    ]
    print("[CAPR-5Min] ordens fora do limite diagnostico 5% ou sem barra:",
          len(liquidity_flags), "de", len(liquidity), flush=True)
    signals = pd.DataFrame(signals)
    coverage = pd.DataFrame(coverage).drop_duplicates(["episode", "date"])
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
        ("reversao_intradiario_cobertura.csv", coverage),
    ]:
        data.to_csv(folder / filename, index=False)
    # Inclui no ZIP o historico observado para auditar lacunas e sinais.
    bars[["local", "date", "clock", "open", "high", "low",
          "close", "volume"]].to_csv(
        folder / "reversao_intradiario_barras.csv", index=False,
        float_format="%.17g",
    )
    control_only = results.loc[results["policy"] == "control"]
    parity = float(np.max(
        np.abs(control_only["equity"].to_numpy(dtype=float) -
               control_only["control_equity"].to_numpy(dtype=float)) /
        np.maximum(control_only["control_equity"].to_numpy(dtype=float), 1.0)
    ))
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
    for _, op in operations.loc[
        operations["reason"] == "sinal intradiario 12:30"
    ].iterrows():
        marker = "^" if op["action"] == "BUY" else "v"
        ax2.scatter(pd.Timestamp(op["date"] + " " + op["time"]),
                    float(op["price"]), marker=marker, s=75, zorder=5)
    ax2.set(title="CAPR: candles 5Min e execucoes apos a avaliacao das 12h30",
            ylabel="Preco ajustado por splits")
    ax2.legend(ncol=3)
    ax2.grid(alpha=.25)
    fig2.autofmt_xdate()
    fig2.tight_layout()
    fig2.savefig(folder / "reversao_intradiario_capr.png", dpi=145)

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    audit_plot = liquidity.sort_values(["date", "time"])
    labels = (
        audit_plot["date"].astype(str) + " " + audit_plot["time"].astype(str)
        + " " + audit_plot["action"].astype(str)
        + " " + audit_plot["policy"].astype(str)
    )
    positions = np.arange(len(audit_plot))
    ax3.bar(positions, audit_plot["quantity_shares"].to_numpy(dtype=float),
            label="Quantidade simulada")
    volumes = audit_plot["observed_5min_volume"].fillna(0).to_numpy(dtype=float)
    ax3.plot(positions, volumes, "o-", label="Volume observado no candle 5Min")
    ax3.set_yscale("symlog", linthresh=1)
    ax3.set_xticks(positions)
    ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax3.set_ylabel("Acoes (escala symlog; barra ausente = 0 apenas no grafico)")
    ax3.set_title("CAPR: quantidade presumida x volume de cada candle de execucao")
    ax3.legend()
    ax3.grid(alpha=.25)
    fig3.tight_layout()
    fig3.savefig(folder / "reversao_intradiario_liquidez.png", dpi=145)
    plt.show(block=False)
    return {
        "status": (
            "baseline_mismatch" if parity > 1e-4
            else "exploratory_execution_unverified" if not liquidity_flags.empty
            else "completed"
        ),
        "scope": "episodios CAPR apenas; NAO e portfolio completo",
        "liquidity": {
            "diagnostic_limit_fraction_of_5min_volume": 0.05,
            "total_audited_orders_including_initial_buy": int(len(liquidity)),
            "unverified_or_above_diagnostic_limit": int(len(liquidity_flags)),
            "orders_without_5min_bar": int((
                liquidity["status"] == "missing_intraday_execution_bar"
            ).sum()),
            "highest_ratio_quantity_to_bar_volume": float(
                liquidity["ratio_quantity_to_5min_volume"].max()
            ),
        },
        "control_episode_max_relative_gap": parity,
        "midday_valid_sessions": int((coverage["status"] == "valid").sum()),
        "midday_skipped_sessions": int((coverage["status"] != "valid").sum()),
        "midday_skipped_dates": coverage.loc[
            coverage["status"] != "valid", "date"
        ].astype(str).tolist(),
        "episodes": studied,
        "window": [PILOT_BEGIN, PILOT_END],
        "timing": {"preopen": "fechamento anterior -> 09:30",
                   "midpoint": "ultimo candle completo ate 12:30; primeira barra entre 12:35 e 12:50",
                   "close": "16:00 -> proxima sessao"},
        "max_one_protective_action_per_session": True,
        "cache": provenance,
        "comparisons": comparison.to_dict(orient="records"),
        "limitations": [
            "Nao recalcula ranking Winner apos CASH.",
            "Janela curta escolhida por episodio de CAPR, nao validacao fora da amostra.",
            "Intradiario SIP RAW normalizado por splits; avaliar dividendos e qualidade.",
            "Nao incorpora impacto de mercado, latencia ou restricoes regulatórias de day trading.",
            "Paridade do Control diario e dos episodios CAPR verificadas separadamente.",
            "Sem candles suficientes as 12h30, nenhum sinal ou ordem; sessao auditada.",
            "Liquidez auditada: valores acima de 5% do candle ou sem barra nao sao fills validados.",
            "Os ganhos simulados sao apenas hipoteticos; nao foi feita simulacao de execucao parcial.",
        ],
    }




# Auditoria historica oficial: CAPR, Nasdaq Trader, 29/07/2026.
# Esta etapa NAO modifica o capital nem corrige fills retrospectivamente.
HALT_DAY = "2026-07-29"
HALT_CACHE = ROOT / "dados" / ".halt_cache" / "nasdaq_20260729_halt_resume.xml"


def parse_nasdaq_halt_rss(data):
    """Interpreta o RSS Nasdaq, ignorando prefixos/namespaces no XML."""
    import xml.etree.ElementTree as ET

    def short_name(tag):
        return str(tag).rsplit("}", 1)[-1].rsplit(":", 1)[-1].lower()

    root = ET.fromstring(data)
    if short_name(root.tag) != "rss":
        raise ValueError("Fonte retornou documento diferente de RSS.")
    channel = next((e for e in root if short_name(e.tag) == "channel"), None)
    if channel is None:
        raise ValueError("RSS sem canal.")
    header = {
        short_name(e.tag): str(e.text or "").strip()
        for e in channel if short_name(e.tag) in ("title", "description")
    }
    label = (header.get("title", "") + " " +
             header.get("description", "")).lower()
    if not ("nasdaq" in label and "halt" in label):
        raise ValueError("O XML nao foi identificado como feed Nasdaq Trade Halts.")
    rows = []
    for item in channel:
        if short_name(item.tag) != "item":
            continue
        values = {
            short_name(child.tag): str(child.text or "").strip()
            for child in item
        }
        symbol = values.get("issuesymbol") or values.get("title") or ""
        rows.append({
            "symbol": symbol.strip().upper(),
            "halt_date": values.get("haltdate", ""),
            "halt_time": values.get("halttime", ""),
            "reason": values.get("reasoncode", ""),
            "resume_date": (
                values.get("resumptiondate") or values.get("resumedate") or ""
            ),
            "resume_quote_time": values.get("resumptionquotetime", ""),
            "resume_trade_time": values.get("resumptiontradetime", ""),
            "market": values.get("mkt") or values.get("market", ""),
            "issue_name": values.get("issuename", ""),
        })
    return rows


def fetch_capr_halts_official():
    """Consulta uma vez e mantem XML original em cache; falha nao vira 'sem halt'."""
    url = (
        "https://www.nasdaqtrader.com/rss.aspx?feed=tradehalts"
        "&haltdate=07292026&resumedate=07292026"
    )
    source = "cache"
    if HALT_CACHE.is_file():
        raw = HALT_CACHE.read_bytes()
        records = parse_nasdaq_halt_rss(raw)
    else:
        import requests
        answer = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; TCC-Research/1.0)",
                "Accept": "application/rss+xml, application/xml, text/xml, */*",
            },
        )
        answer.raise_for_status()
        raw = answer.content
        records = parse_nasdaq_halt_rss(raw)
        HALT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        temporary = HALT_CACHE.with_suffix(".tmp")
        temporary.write_bytes(raw)
        os.replace(temporary, HALT_CACHE)
        source = "official_nasdaq_http"
    return records, raw, {
        "source": source,
        "url": url,
        "cached_file": str(HALT_CACHE.relative_to(ROOT)),
        "xml_sha256": sha256(HALT_CACHE),
        "rss_record_count": len(records),
        "query_date": HALT_DAY,
        "note": (
            "Consulta por data de inicio OU retomada; nao comprova "
            "a ausencia de suspensoes iniciadas antes da data pesquisada."
        ),
    }


def _halt_datetime(local_date, local_clock):
    """Horario Nasdaq em America/New_York, inclusive campos com padding.

    O RSS oficial de 29/07/2026 publica o HaltTime de CAPR como:
        '06:55:00                      .000'
    O parser antigo retornava None, ignorava TODO o halt e classificava as
    cinco vendas hipoteticas as 09h30 como 'not_confirmed'.

    Campo vazio e permitido somente quando a retomada nao foi informada.
    Campo NAO vazio e invalido gera erro, nunca 'mercado negociavel'.
    """
    import re
    from datetime import datetime

    day = str(local_date or "").strip()
    clock = str(local_clock or "").strip()
    if not day and not clock:
        return None
    if not day or not clock:
        raise ValueError("Nasdaq halt: data e hora devem ser informadas juntas.")
    match = re.fullmatch(
        r"(\d{1,2}:\d{2}:\d{2})(?:\s*\.\s*(\d{1,9}))?", clock
    )
    if match is None:
        raise ValueError("Nasdaq halt: horario inesperado: " + repr(clock))
    clean_clock = match.group(1)
    if match.group(2) is not None:
        clean_clock += "." + match.group(2)
    if "/" in day:
        date = datetime.strptime(day, "%m/%d/%Y").date()
    else:
        date = datetime.strptime(day, "%Y-%m-%d").date()
    return pd.Timestamp(str(date) + " " + clean_clock).tz_localize(NY)


def _halt_order_status(execution, rows):
    """Classifica com cautela; NAO considera fonte vazia prova de negociabilidade."""
    if rows is None:
        return "source_unavailable", "Sem feed Nasdaq validado."
    candidates = []
    for h in rows:
        start = _halt_datetime(h["halt_date"], h["halt_time"])
        finish = _halt_datetime(h["resume_date"], h["resume_trade_time"])
        if start is None:
            continue
        if start <= execution:
            candidates.append((start, finish, h["reason"]))
    if not candidates:
        return (
            "not_confirmed", "Nenhum intervalo aplicavel localizado; "
            "isso nao prova que o ativo estava negociavel."
        )
    start, finish, reason = max(candidates, key=lambda x: x[0])
    if finish is None:
        return (
            "resume_unverified", "Inicio do halt registrado em " +
            str(start) + " (" + reason + "), sem retomada efetiva confirmada."
        )
    if finish < start:
        return "invalid_feed_interval", "Retomada anterior ao inicio do halt."
    if execution < finish:
        return (
            "halted_at_order_time", "Halt ativo ate " + str(finish)
            + " (codigo " + reason + ")."
        )
    return (
        "resumed_before_order", "Retomada informada em " + str(finish) +
        "; preco, volume e liquidez ainda precisam ser confirmados."
    )


def audit_capr_halts_20260729(daily_trades, folder):
    """Audita vendas do CAPR no dia 29 e exporta a fonte bruta quando disponivel."""
    rows = None
    original_xml = None
    source_meta = {}
    try:
        rows, original_xml, source_meta = fetch_capr_halts_official()
        (folder / "reversao_halts_nasdaq_20260729.xml").write_bytes(original_xml)
    except Exception as exc:
        source_meta = {
            "source": "unavailable", "query_date": HALT_DAY,
            "error_type": type(exc).__name__, "error": str(exc)[:600],
        }
        print("[halts] Fonte Nasdaq indisponivel:", str(exc)[:250], flush=True)

    only_capr = [x for x in (rows or []) if x["symbol"] == "CAPR"]
    columns = [
        "symbol", "halt_date", "halt_time", "reason", "resume_date",
        "resume_quote_time", "resume_trade_time", "market", "issue_name",
    ]
    pd.DataFrame(only_capr, columns=columns).to_csv(
        folder / "reversao_halts_capr_eventos.csv", index=False
    )

    orders = []
    daily = daily_trades.loc[
        (daily_trades["asset"] == "CAPR") &
        (daily_trades["session"].astype(str).str[:10] == HALT_DAY)
    ]
    for _, row in daily.iterrows():
        if row["action"] not in ("BUY", "SELL", "FINAL_SELL"):
            continue
        orders.append({
            "origin": "replay_daily",
            "policy": row["profile"],
            "date": HALT_DAY,
            "clock": "09:30" if row["action"] != "FINAL_SELL" else "16:00",
            "action": row["action"],
            "quantity": float(row["quantity"]),
            "assumed_price": float(row["execution_price"]),
        })
    pilot_file = folder / "reversao_intradiario_operacoes.csv"
    if pilot_file.is_file():
        pilot = pd.read_csv(pilot_file)
        pilot = pilot.loc[pilot["date"].astype(str) == HALT_DAY]
        for _, row in pilot.iterrows():
            orders.append({
                "origin": "capr_5min_pilot",
                "policy": row["policy"],
                "date": HALT_DAY,
                "clock": str(row["time"]),
                "action": row["action"],
                "quantity": float(row["quantity"]),
                "assumed_price": float(row["price"]),
            })

    audited = []
    for order in orders:
        moment = _halt_datetime(order["date"], order["clock"])
        status, detail = _halt_order_status(moment, only_capr if rows is not None else None)
        audited.append({
            **order, "order_timestamp_ny": str(moment),
            "halt_status": status, "explanation": detail,
            "execution_verified": False,
        })
    order_columns = [
        "origin", "policy", "date", "clock", "action", "quantity",
        "assumed_price", "order_timestamp_ny", "halt_status",
        "explanation", "execution_verified",
    ]
    pd.DataFrame(audited, columns=order_columns).to_csv(
        folder / "reversao_halts_capr_ordens.csv", index=False
    )

    if only_capr or audited:
        fig, ax = plt.subplots(figsize=(12, 3.5))
        base = pd.Timestamp(HALT_DAY)
        effective_resumptions = [
            _halt_datetime(h["resume_date"], h["resume_trade_time"])
            for h in only_capr
        ]
        effective_resumptions = [
            ts.tz_localize(None) for ts in effective_resumptions if ts is not None
        ]
        xmax = max([base + pd.Timedelta(hours=16)]
                   + [ts + pd.Timedelta(hours=1) for ts in effective_resumptions])
        ax.set_xlim(base + pd.Timedelta(hours=6), xmax)
        ax.set_ylim(-.4, 1.2)
        ax.set_yticks([])
        ax.axhline(0.35, color="gray", alpha=.4)
        for item in only_capr:
            start = _halt_datetime(item["halt_date"], item["halt_time"])
            resumed = _halt_datetime(item["resume_date"], item["resume_trade_time"])
            if start is not None:
                ax.axvline(start.tz_localize(None), color="tab:red",
                           linestyle="--", label="Inicio do halt")
                if resumed is not None:
                    ax.axvspan(start.tz_localize(None),
                               resumed.tz_localize(None), color="tab:red",
                               alpha=.12, label="Sem negociacoes")
            if resumed is not None:
                ax.axvline(resumed.tz_localize(None), color="tab:green",
                           linestyle="--", label="Retomada efetiva")
        for order in audited:
            when = _halt_datetime(order["date"], order["clock"])
            if when is not None:
                ax.scatter(when.tz_localize(None), 0.35, s=75, marker="x",
                           color="tab:blue")
        ax.set_title(
            "CAPR 29/07/2026: halts oficiais x horarios das ordens simuladas"
        )
        ax.set_xlabel("Horario de Nova York (fonte Nasdaq, hora local)")
        ax.grid(axis="x", alpha=.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(folder / "reversao_halts_capr_linha_tempo.png", dpi=145)
        plt.show(block=False)

    halted_count = sum(
        item["halt_status"] == "halted_at_order_time" for item in audited
    )
    report = {
        "status": (
            "source_unavailable" if rows is None
            else "simulated_executions_during_halt" if halted_count
            else "CAPR_records_found" if only_capr
            else "CAPR_not_found_in_feed_not_proof_of_absence"
        ),
        "source": source_meta,
        "capr_halt_events": len(only_capr),
        "simulated_orders_audited": len(audited),
        "orders_reported_halted": halted_count,
        "invalid_historical_fills": halted_count,
        "baseline_research_conclusion": (
            "INVALID: as ordens hipoteticas executadas durante halt devem "
            "ser retiradas da simulacao ANTES de comparar protecao e Control."
            if halted_count else
            "Nenhuma ordem deste subconjunto foi confirmada durante halt. "
            "Isso nao valida execucoes de outras datas."
        ),
        "orders_resume_unverified": sum(
            r["halt_status"] == "resume_unverified" for r in audited
        ),
        "note": (
            "Auditoria retrospectiva. Nao altera as ordens ou capital "
            "simulados; resultados financeiros nao implicam fill executavel."
        ),
    }
    (folder / "reversao_halts_status.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print("[halts] status:", report["status"],
          "| ordens:", report["simulated_orders_audited"], flush=True)
    return report



def main_historico_anterior():
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
        "schema": "tcc_cash_reversal_exploratory_v1",
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
        halt_audit = audit_capr_halts_20260729(log, target)
        result["nasdaq_halt_audit"] = halt_audit
        (target / "reversao_resumo.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False,
                       default=str),
            encoding="utf-8",
        )
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



# ---------------------------------------------------------------------------
# NOVA REFERENCIA OFICIAL: backtest da producao de 10/11/2026.
#
# Esta referencia substitui o checkpoint historico COMO FONTE DOS NOVOS
# ESTUDOS. O historico anterior permanece isolado em
# main_historico_anterior(), mas nao e executado com F5.
#
# IMPORTANTE: o ZIP de producao NAO contem OHLCV por ativo. Portanto,
# reproduzir resultados com intervencoes CASH exigira historico adicional
# validado antes de gerar qualquer comparacao financeira contrafactual.
# ---------------------------------------------------------------------------
PROD_FILE = "market_cycle_trader_20261110T140411-8f086044.zip"
PROD_SHA256 = "1cdeda67bafccf424a75b675b5f16084b14c282ec1ba1b9599c6e234312dc0f2"
PROD_JOB_ID = "20261110T140411-8f086044"
PROD_CACHE = ROOT / "dados" / ".production_reference" / PROD_FILE


def localizar_backtest_producao():
    """Usa somente o ZIP identificado por nome E SHA-256, sem fallback antigo."""
    import shutil

    possibilities = [
        PROD_CACHE,
        ROOT / PROD_FILE,
        ROOT / "out" / PROD_FILE,
        ROOT / "dados" / PROD_FILE,
        ROOT.parent / PROD_FILE,
        Path.home() / "Downloads" / PROD_FILE,
        Path.home() / "Desktop" / PROD_FILE,
        Path.home() / "Documentos" / PROD_FILE,
        Path.home() / "Documents" / PROD_FILE,
    ]
    found = next((p for p in possibilities if p.is_file()), None)
    if found is None:
        raise FileNotFoundError(
            "Backtest de producao fixado nao localizado: " + PROD_FILE +
            ". Deixe o ZIP original em Downloads ou na raiz do projeto TCC. "
            "O script importa e guarda uma copia local automaticamente. "
            "O replay historico antigo NAO sera usado como fallback."
        )
    observed_sha = sha256(found)
    if observed_sha != PROD_SHA256:
        raise RuntimeError(
            "ZIP de producao com hash diferente do registrado. "
            "Observado=" + observed_sha + "; esperado=" + PROD_SHA256 +
            ". Nao prosseguir com fonte possivelmente alterada."
        )
    if found.resolve() != PROD_CACHE.resolve():
        PROD_CACHE.parent.mkdir(parents=True, exist_ok=True)
        if PROD_CACHE.is_file() and sha256(PROD_CACHE) != PROD_SHA256:
            raise RuntimeError("Cache de producao modificado. Nao sobrescrever silenciosamente.")
        if not PROD_CACHE.is_file():
            temp = PROD_CACHE.with_suffix(".tmp")
            shutil.copyfile(found, temp)
            if sha256(temp) != PROD_SHA256:
                temp.unlink(missing_ok=True)
                raise RuntimeError("Falha de integridade na copia do backtest.")
            os.replace(temp, PROD_CACHE)
    print("[producao] fonte congelada:", PROD_CACHE, flush=True)
    print("[producao] sha256:", PROD_SHA256, flush=True)
    return PROD_CACHE


def ler_backtest_producao(source):
    """Le e valida manifests, 1.584 decisoes e 700 operacoes da producao."""
    with zipfile.ZipFile(source) as package:
        corrupted = package.testzip()
        if corrupted is not None:
            raise RuntimeError("Entrada corrompida no ZIP: " + corrupted)
        names = package.namelist()
        def locate(suffix):
            candidates = [n for n in names if n.endswith(suffix)]
            if len(candidates) != 1:
                raise RuntimeError("Arquivo ausente/ambiguo no ZIP: " + suffix)
            return candidates[0]
        manifest = json.loads(package.read("experiment_manifest.json"))
        strategy = json.loads(package.read("strategy_manifest.json"))
        metrics_data = json.loads(package.read(locate("_metrics.json")))
        predictions = pd.read_csv(io.BytesIO(package.read(locate("_predictions.csv"))))
        trades = pd.read_csv(io.BytesIO(package.read(locate("_trades.csv"))))
        decisions = pd.read_csv(io.BytesIO(package.read(locate("_decision_diagnostics.csv"))))
    if manifest.get("job_id") != PROD_JOB_ID:
        raise RuntimeError("Job de producao diferente da referencia congelada.")
    if manifest.get("job_status") != "completed":
        raise RuntimeError("Backtest nao consta como concluido.")
    if int(manifest.get("eligible_asset_count") or 0) != 75:
        raise RuntimeError("Universo elegivel diferente do backtest escolhido.")
    required = {
        "timestamp", "decision_date", "selected_asset",
        "strategy_equity", "buy_hold_equity", "walk_forward_fold",
    }
    if required - set(predictions.columns):
        raise RuntimeError("Colunas de predictions ausentes: " +
                           ", ".join(sorted(required - set(predictions.columns))))
    predictions["date"] = predictions["timestamp"].map(local_day)
    predictions["decision_session"] = predictions["decision_date"].map(local_day)
    predictions["selected_asset"] = (
        predictions["selected_asset"].fillna("").astype(str).str.upper().str.strip()
    )
    predictions["strategy_equity"] = pd.to_numeric(
        predictions["strategy_equity"], errors="raise"
    )
    predictions["buy_hold_equity"] = pd.to_numeric(
        predictions["buy_hold_equity"], errors="raise"
    )
    predictions = predictions.sort_values("date").reset_index(drop=True)
    if (len(predictions) != 1584 or predictions["date"].duplicated().any()
            or (predictions["strategy_equity"] <= 0).any()
            or not np.isfinite(predictions["strategy_equity"]).all()):
        raise RuntimeError("Calendario ou patrimonio de producao inconsistente.")
    if str(predictions["date"].iloc[-1]) != "2026-11-09":
        raise RuntimeError("Ultima sessao diferente de 09/11/2026.")
    if not np.isclose(
        float(predictions["strategy_equity"].iloc[-1]),
        float(metrics_data["strategy_ending_capital"]),
        rtol=0.0, atol=0.01,
    ):
        raise RuntimeError("Capital final diverge entre predictions e metrics.")
    if "timestamp" not in trades or "asset" not in trades or "action" not in trades:
        raise RuntimeError("Operacoes do backtest incompletas.")
    trades["date"] = trades["timestamp"].map(local_day)
    if len(trades) != 700:
        raise RuntimeError("Quantidade de operacoes diverge da fonte congelada.")
    if "timestamp" not in decisions:
        raise RuntimeError("Diagnosticos sem data de decisao.")
    decisions["date"] = decisions["timestamp"].map(local_day)
    if len(decisions) != len(predictions):
        raise RuntimeError("Decisoes e sessions em quantidade diferente.")
    if int(metrics_data.get("cash_days", -1)) != 0:
        raise RuntimeError("Sessao CASH divergente da execucao selecionada.")
    return predictions, trades, decisions, metrics_data, manifest, strategy


def gerar_figuras_producao(predictions, trades, decisions, target):
    """Graficos da fonte oficial, sem criar retornos contrafactuais."""
    from matplotlib import dates as mdates

    output = []
    dates = pd.to_datetime(predictions["date"].astype(str))
    curve = predictions["strategy_equity"]
    eq_drawdown = curve / pd.Series(
        np.maximum.accumulate(
            np.r_[float(INITIAL), curve.to_numpy(dtype=float)]
        )[1:]
    ) - 1.0
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(dates, curve, label="Winner Control, producao", linewidth=1.7)
    ax.plot(dates, predictions["buy_hold_equity"],
            label="Buy-and-hold de referencia", linewidth=1.2)
    ax.set(yscale="log", ylabel="US$ (escala logaritmica)",
           title="Patrimonio da nova referencia: producao ate 09/11/2026")
    ax.legend()
    ax.grid(alpha=.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    output.append(("producao_patrimonio.png", fig))

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(dates, eq_drawdown * 100, label="Drawdown da Winner")
    trough_idx = int(eq_drawdown.idxmin())
    ax.scatter(dates.iloc[trough_idx], eq_drawdown.iloc[trough_idx] * 100,
               color="tab:red", s=55, zorder=5)
    ax.annotate(
        str(predictions["date"].iloc[trough_idx]),
        (dates.iloc[trough_idx], eq_drawdown.iloc[trough_idx] * 100),
        xytext=(10, 10), textcoords="offset points",
    )
    ax.set(title="Drawdown do patrimonio na producao",
           ylabel="Queda maxima relativa ao pico (%)")
    ax.grid(alpha=.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    output.append(("producao_drawdown.png", fig))

    plot_start = pd.Timestamp("2026-07-01")
    detail = predictions.loc[dates >= plot_start].copy()
    dts = pd.to_datetime(detail["date"].astype(str))
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(dts, detail["strategy_equity"], label="Patrimonio", linewidth=1.6)
    capr = detail["selected_asset"].eq("CAPR").to_numpy()
    label_used = False
    for i, inside in enumerate(capr):
        if inside:
            left = dts.iloc[i]
            right = (dts.iloc[i + 1] if i + 1 < len(dts)
                     else dts.iloc[i] + pd.Timedelta(days=1))
            ax.axvspan(
                left, right, alpha=.13, color="tab:orange",
                label="Posicao CAPR" if not label_used else None,
            )
            label_used = True
    ax.set(title="Nova referencia: patrimonio e exposicao a CAPR",
           ylabel="Patrimonio (US$)")
    ax.legend()
    ax.grid(alpha=.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    output.append(("producao_capr_exposicao.png", fig))

    diagnostic = decisions.loc[
        decisions["date"] >= pd.Timestamp("2026-07-01").date()
    ].copy()
    if {"best_score", "current_score", "date"}.issubset(diagnostic.columns):
        fig, ax = plt.subplots(figsize=(13, 5))
        dx = pd.to_datetime(diagnostic["date"].astype(str))
        ax.plot(dx, pd.to_numeric(diagnostic["best_score"], errors="coerce"),
                label="Maior utilidade estimada", linewidth=1.5)
        ax.plot(dx, pd.to_numeric(diagnostic["current_score"], errors="coerce"),
                label="Utilidade da posicao atual", linewidth=1.2)
        ax.axhline(0, color="gray", linestyle="--", linewidth=.7)
        ax.set(title="Scores conhecidos nas decisoes diarias",
               ylabel="Score da Winner (escala original)")
        ax.legend()
        ax.grid(alpha=.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        output.append(("producao_utilidades_decisao.png", fig))

    counts = trades.loc[trades["action"] == "BUY", "asset"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(counts.index[::-1], counts.to_numpy()[::-1])
    ax.set(title="Ativos com mais entradas na producao",
           xlabel="Quantidade de compras simuladas")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    output.append(("producao_frequencia_ativos.png", fig))

    for name, fig in output:
        fig.savefig(target / name, dpi=150, bbox_inches="tight")
    plt.show(block=False)
    return [name for name, _ in output]


def main():
    """No Spyder (F5), a unica referencia ativa e a nova producao."""
    source = localizar_backtest_producao()
    pred, trades, decisions, metrics_prod, manifest, strategy = ler_backtest_producao(
        source
    )
    print("[producao] sessoes:", len(pred), "| operacoes:", len(trades),
          "| capital final: US$ {:,.2f}".format(
              float(metrics_prod["strategy_ending_capital"])
          ), flush=True)

    # Verificacoes de paridade com metricas do proprio arquivo de producao.
    eq = pred["strategy_equity"].to_numpy(dtype=float)
    peak = np.maximum.accumulate(np.r_[float(INITIAL), eq])[1:]
    dd_observed = float((eq / peak - 1.0).min())
    dd_reported = float(metrics_prod["strategy_maximum_drawdown"])
    dd_error = abs(dd_observed - dd_reported)
    if dd_error > 1e-10:
        raise RuntimeError(
            "Drawdown reproduzido diverge do reportado: " + str(dd_error)
        )
    capr_buy = trades.loc[
        (trades["asset"] == "CAPR") & (trades["action"] == "BUY")
    ]
    july29_capr = trades.loc[
        (trades["asset"] == "CAPR") &
        (trades["date"].astype(str) == "2026-07-29")
    ]
    pred["drawdown"] = eq / peak - 1.0
    pred["daily_return"] = pred["strategy_equity"].pct_change()
    pred.loc[0, "daily_return"] = eq[0] / INITIAL - 1
    pred["episode"] = (
        pred["selected_asset"] != pred["selected_asset"].shift()
    ).cumsum()
    episodes = []
    for _, group in pred.groupby("episode", sort=False):
        if group["selected_asset"].iloc[0] != "CAPR":
            continue
        episodes.append({
            "start": str(group["date"].iloc[0]),
            "end": str(group["date"].iloc[-1]),
            "sessions": int(len(group)),
            "equity_first_close": float(group["strategy_equity"].iloc[0]),
            "equity_last_close": float(group["strategy_equity"].iloc[-1]),
            "equity_change_between_closes": float(
                group["strategy_equity"].iloc[-1] /
                group["strategy_equity"].iloc[0] - 1
            ),
        })
    folds = []
    for fold, group in pred.groupby("walk_forward_fold", sort=False):
        growth = float(np.prod(1.0 + group["daily_return"].to_numpy(dtype=float)))
        folds.append({
            "fold": str(fold),
            "sessions": int(len(group)),
            "return_in_fold": growth - 1.0,
            "cash_days": int(group["selected_asset"].eq("CASH").sum()),
        })
    # Comparacao POR DATA com o checkpoint historico, quando informado
    # no proprio pacote de producao. Nao presume mesma amostra/modelo.
    original_cutoff = pd.Timestamp("2026-10-06").date()
    cutoff_rows = pred.loc[pred["date"] == original_cutoff]
    historical_capital = metrics_prod.get("tcc_reference_checkpoint_capital")
    cutoff_prod = (
        float(cutoff_rows["strategy_equity"].iloc[0])
        if len(cutoff_rows) == 1 else None
    )
    reference_comparison = {
        "reference_date": str(original_cutoff),
        "production_capital_same_date": cutoff_prod,
        "former_tcc_checkpoint_capital": (
            float(historical_capital) if historical_capital is not None else None
        ),
        "delta_production_vs_former": (
            cutoff_prod - float(historical_capital)
            if cutoff_prod is not None and historical_capital is not None
            else None
        ),
        "caution": (
            "Nao concluir que o periodo adicional causou a diferenca. "
            "Duas execucoes podem diferir em dados, configuracao ou treino."
        ),
    }
    print(
        "[producao] mesmo cutoff 06/10/2026:",
        "producao=", cutoff_prod,
        "| checkpoint anterior=", historical_capital,
        flush=True,
    )
    # Exposicao verdadeira da PRODUCAO durante o halt CAPR:
    # patrimonio de fechamento, nao prova de preco executavel intradiario.
    capr_event_window = pred.loc[
        (pred["date"] >= pd.Timestamp("2026-07-27").date()) &
        (pred["date"] <= pd.Timestamp("2026-08-05").date()),
        ["date", "selected_asset", "strategy_equity", "trade_action"],
    ].copy()
    event_rows = {
        str(row["date"]): row for _, row in capr_event_window.iterrows()
    }
    def _event_equity(day):
        row = event_rows.get(day)
        return float(row["strategy_equity"]) if row is not None else None
    before_halt = _event_equity("2026-07-28")
    after_halt = _event_equity("2026-07-30")
    capr_halt_context = {
        "source": "production_predictions_and_trades",
        "halt_date": "2026-07-29",
        "resumption_trade_date": "2026-07-30",
        "position_on_halt_date": (
            str(event_rows["2026-07-29"]["selected_asset"])
            if "2026-07-29" in event_rows else None
        ),
        "equity_2026_07_28": before_halt,
        "equity_2026_07_29": _event_equity("2026-07-29"),
        "equity_2026_07_30": after_halt,
        "equity_change_2026_07_28_to_30": (
            after_halt / before_halt - 1.0
            if before_halt is not None and after_halt is not None
            and before_halt > 0 else None
        ),
        "production_capr_sales_during_2026_07_29": int(len(
            july29_capr.loc[july29_capr["action"] == "SELL"]
        )),
        "note": (
            "Trata-se de patrimonio de FECHAMENTO do Control de producao. "
            "Nao atribui venda a CAPR enquanto o ativo esteve suspenso "
            "nem declara executaveis as compras simuladas."
        ),
    }
    summary = {
        "capr_halt_exposure_production": capr_halt_context,
        "comparison_with_prior_checkpoint": reference_comparison,
        "research_reference": "production_backtest_20261110",
        "status": "imported_and_validated",
        "job_id": PROD_JOB_ID,
        "archive_name": PROD_FILE,
        "archive_sha256": PROD_SHA256,
        "api_version": manifest.get("api_version"),
        "job_status": manifest.get("job_status"),
        "market_data_signature": manifest.get("market_data_signature_sha256"),
        "strategy_configuration_hash": manifest.get(
            "strategy_configuration_hash"
        ),
        "strategy_manifest_profile": strategy.get("strategy_profile_name"),
        "initial_capital": INITIAL,
        "ending_capital": float(metrics_prod["strategy_ending_capital"]),
        "maximum_drawdown": dd_observed,
        "reported_maximum_drawdown": dd_reported,
        "sharpe": float(metrics_prod["strategy_sharpe"]),
        "rotations": int(metrics_prod["capital_rotations"]),
        "cash_days": int(metrics_prod["cash_days"]),
        "assets_eligible": manifest.get("eligible_asset_count"),
        "sessions": len(pred),
        "first_session": str(pred["date"].iloc[0]),
        "last_session": str(pred["date"].iloc[-1]),
        "capr_entries": len(capr_buy),
        "capr_trades_on_2026_07_29": int(len(july29_capr)),
        "source_data_limitations": [
            "O ZIP exportado nao inclui barras OHLCV de todos os ativos.",
            "Nao calcular comparacoes financeiras CASH sem harmonizar a fonte "
            "historica da producao para o periodo ate 09/11/2026.",
            "As decisoes e operacoes sao do Control original, nao de "
            "um contrafactual com protecao recalculada.",
            "Executabilidade exige auditoria separada de halts e liquidez.",
        ],
        "folds": folds,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="referencia_producao_", dir=OUT) as tmp:
        dest = Path(tmp)
        capr_event_window.to_csv(
            dest / "producao_capr_exposicao_durante_halt.csv", index=False
        )
        pred.to_csv(dest / "producao_decisoes.csv", index=False)
        trades.to_csv(dest / "producao_operacoes.csv", index=False)
        decisions.to_csv(dest / "producao_diagnosticos_decisao.csv", index=False)
        pd.DataFrame(episodes).to_csv(
            dest / "producao_episodios_capr.csv", index=False
        )
        pd.DataFrame(folds).to_csv(dest / "producao_folds.csv", index=False)
        gerar_figuras_producao(pred, trades, decisions, dest)

        # Auditar apenas halts do episodio investigado em julho/2026;
        # nao declarar como validas as demais ordens sem fontes adicionais.
        daily_audit = trades.loc[
            trades["asset"].eq("CAPR"),
            ["date", "action", "asset", "quantity", "execution_price"]
        ].copy()
        daily_audit["session"] = daily_audit["date"].astype(str)
        daily_audit["profile"] = "production_control"
        try:
            summary["halt_audit"] = audit_capr_halts_20260729(daily_audit, dest)
        except Exception as exc:
            summary["halt_audit"] = {
                "status": "audit_error", "message": str(exc)[:400]
            }
        (dest / "producao_referencia_resumo.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str,
                       allow_nan=False),
            encoding="utf-8",
        )
        (dest / "LEIA_ME.txt").write_text(
            "REFERENCIA FIXA: backtest de producao 10/11/2026, "
            "job 20261110T140411-8f086044.\n"
            "Dados originais permanecem intactos no cache local.\n"
            "Esta entrega nao implementa novas politicas CASH nem "
            "recalcula a Winner com novo historico.\n"
            "O checkpoint anterior de US$ 587 milhoes nao e a base ativa.\n"
            "Confirme integridade e cobertura antes de futuros estudos.\n",
            encoding="utf-8",
        )
        package = dest / ARCHIVE
        with zipfile.ZipFile(package, mode="w",
                             compression=zipfile.ZIP_DEFLATED) as z:
            for f in sorted(dest.iterdir()):
                if f.is_file() and f != package:
                    z.write(f, arcname=f.name)
        for f in sorted(dest.iterdir()):
            if f.is_file() and f != package:
                os.replace(f, OUT / f.name)
        os.replace(package, OUT / ARCHIVE)
    print("[producao] referencia ativa:", PROD_JOB_ID, flush=True)
    print("[producao] zip unico:", OUT / ARCHIVE, flush=True)
    print("[producao] nao foram estimados cenarios CASH novos", flush=True)
    return summary


if __name__ == "__main__":
    RESULTADO = main()
