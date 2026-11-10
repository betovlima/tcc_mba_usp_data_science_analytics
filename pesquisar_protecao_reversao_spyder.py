# -*- coding: utf-8 -*-
"""TCC | Pesquisa exploratoria de reversao da posicao para CASH.

Abra este arquivo no Spyder e execute com F5. Nenhum argumento.
Entrada: output/reproducao_universo/predictions.csv ou ZIP oficial.
Fonte de precos: dados/u67 e dados/.u67_expansion_cache (somente leitura).
Saidas: out/ e out/pacote_protecao_reversao.zip (sobrescrito).
Nao modifica a Winner, nao consulta APIs e nao treina modelos.
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
            x = pd.to_datetime(e["session"])
            if len(x):
                y = bars["close"].reindex(x.date).to_numpy()
                ax.scatter(x, y, label="Venda " + name, marker=marker, s=60, zorder=5)
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
