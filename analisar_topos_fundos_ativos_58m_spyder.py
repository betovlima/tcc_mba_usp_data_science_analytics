"""ANALISE STANDALONE DOS OITO ATIVOS DO CENARIO DE US$ 58,56M.

Analisa THO, WDAY, EXR, XEL, SBFG, PAYX, MUX e SXC individualmente, sem
reinseri-los no universo de rotacao, sem treinar modelo e sem novo backtest.

A analise usa:
- OHLCV congelado de dados/pesquisa_smart_candidates;
- entradas/saidas ja observadas no replay U59+8;
- Directional Change de 2%, 4% e 8%;
- features locais/causais no momento de entrada e saida;
- extremos ex-post apenas como diagnostico descritivo.

Execute no Spyder com F5 ou pelas celulas # %% em ordem.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine.configuracao import CONFIG
from engine.rotacao import construir_quadro_rotacao
from pesquisas.directional_change_lightgbm import (
    DIRECTIONAL_CHANGE_THRESHOLDS,
    criar_pacote_analise,
    sinal_sonoro_conclusao,
)
from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.preparacao import prepare_model_frames


# %% 0 - Configuracao congelada
ROOT = Path(__file__).resolve().parent
SMART = SnapshotPaths.from_root(
    ROOT / "dados" / "pesquisa_smart_candidates"
)
TRADES_FILE = (
    ROOT
    / "output"
    / "avaliacao_financeira"
    / "u59_plus_positive8_trades.csv"
)
OUT = ROOT / "output" / "analise_standalone_ativos_58m"

SCRIPT_VERSION = "1.17.3-dev.1"
EXECUTION_SCHEMA = "standalone-top-bottom-eight-assets-v1"

ASSETS_58M = (
    "THO", "WDAY", "EXR", "XEL",
    "SBFG", "PAYX", "MUX", "SXC",
)
PRIMARY_DC_THRESHOLD = 0.04

CAUSAL_FEATURES = (
    "return_5", "return_20", "return_60",
    "vol_20", "atr_pct_14", "rsi_14",
    "ema_distance_20", "ema_slope_20_5",
    "distance_from_high_20", "distance_from_low_20",
    "distance_from_high_50", "distance_from_low_50",
    "channel_position_20", "channel_position_50",
    "trend_efficiency_20", "volume_zscore_20",
)


# %% 1 - Helpers
def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _session_position(index: pd.DatetimeIndex, timestamp) -> int | None:
    if timestamp is None or pd.isna(timestamp) or len(index) == 0:
        return None
    ts = _utc(timestamp)
    pos = int(index.searchsorted(ts, side="left"))
    if pos >= len(index):
        return len(index) - 1
    if pos > 0:
        before = index[pos - 1]
        after = index[pos]
        if abs(ts - before) <= abs(after - ts):
            return pos - 1
    return pos


def _sessions_between(index, start, end) -> int | None:
    p0 = _session_position(index, start)
    p1 = _session_position(index, end)
    if p0 is None or p1 is None:
        return None
    return int(p1 - p0)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# %% 2 - Directional Change standalone
def directional_change_events(
    close: pd.Series,
    *,
    symbol: str,
    threshold: float,
) -> pd.DataFrame:
    """Extrai extremos e confirmacoes usando somente a serie do proprio ativo."""
    series = pd.to_numeric(close, errors="coerce").dropna()
    if series.empty:
        return pd.DataFrame()

    mode = 0
    running_high = float(series.iloc[0])
    running_high_ts = pd.Timestamp(series.index[0])
    running_low = float(series.iloc[0])
    running_low_ts = pd.Timestamp(series.index[0])
    events = []

    for ts, raw_price in series.iloc[1:].items():
        price = float(raw_price)
        ts = pd.Timestamp(ts)

        if mode >= 0 and price > running_high:
            running_high = price
            running_high_ts = ts
        if mode <= 0 and price < running_low:
            running_low = price
            running_low_ts = ts

        if mode in {0, 1} and price <= running_high * (1.0 - threshold):
            events.append(
                {
                    "asset": symbol,
                    "threshold": float(threshold),
                    "event_type": "TOP",
                    "extreme_timestamp": running_high_ts,
                    "extreme_price": running_high,
                    "confirmation_timestamp": ts,
                    "confirmation_price": price,
                    "confirmation_move": price / running_high - 1.0,
                }
            )
            mode = -1
            running_low = price
            running_low_ts = ts
        elif mode in {0, -1} and price >= running_low * (1.0 + threshold):
            events.append(
                {
                    "asset": symbol,
                    "threshold": float(threshold),
                    "event_type": "BOTTOM",
                    "extreme_timestamp": running_low_ts,
                    "extreme_price": running_low,
                    "confirmation_timestamp": ts,
                    "confirmation_price": price,
                    "confirmation_move": price / running_low - 1.0,
                }
            )
            mode = 1
            running_high = price
            running_high_ts = ts

    result = pd.DataFrame(events)
    if result.empty:
        return result

    index = pd.DatetimeIndex(series.index)
    result["confirmation_lag_sessions"] = [
        _sessions_between(index, a, b)
        for a, b in zip(
            result["extreme_timestamp"],
            result["confirmation_timestamp"],
        )
    ]
    return result


def pair_directional_swings(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for threshold, group in events.groupby("threshold", sort=True):
        group = group.sort_values(
            ["extreme_timestamp", "confirmation_timestamp"]
        ).reset_index(drop=True)
        for i in range(len(group) - 1):
            a = group.iloc[i]
            b = group.iloc[i + 1]
            if a["event_type"] == b["event_type"]:
                continue
            rows.append(
                {
                    "asset": a["asset"],
                    "threshold": float(threshold),
                    "from_event": a["event_type"],
                    "to_event": b["event_type"],
                    "from_timestamp": a["extreme_timestamp"],
                    "to_timestamp": b["extreme_timestamp"],
                    "from_price": float(a["extreme_price"]),
                    "to_price": float(b["extreme_price"]),
                    "swing_return": (
                        float(b["extreme_price"])
                        / float(a["extreme_price"])
                        - 1.0
                    ),
                }
            )
    return pd.DataFrame(rows)


# %% 3 - Snapshot e features locais
started = time.perf_counter()
manifest_smart = validate_snapshot(SMART)

frames_raw, exclusions, diagnostics, audit = prepare_model_frames(
    SMART,
    assets=ASSETS_58M,
    comparar_snapshot_referencia=False,
)
if exclusions:
    raise RuntimeError(
        "Os oito ativos precisam estar disponiveis sem exclusoes: "
        + json.dumps(exclusions, ensure_ascii=False, default=str)
    )

missing_assets = sorted(set(ASSETS_58M).difference(frames_raw))
if missing_assets:
    raise RuntimeError(
        "Ativos ausentes no snapshot SMART: " + ",".join(missing_assets)
    )

frames = {}
for symbol in ASSETS_58M:
    frame = construir_quadro_rotacao(frames_raw[symbol], CONFIG)
    frames[symbol] = frame.replace([np.inf, -np.inf], np.nan)
    print(
        f"[standalone-data] {symbol} rows={len(frame)} "
        f"first={frame.index.min()} last={frame.index.max()}",
        flush=True,
    )


# %% 4 - Topos, fundos e swings
event_frames = []
swing_frames = []

for symbol in ASSETS_58M:
    per_asset = []
    for threshold in DIRECTIONAL_CHANGE_THRESHOLDS:
        events = directional_change_events(
            frames[symbol]["close"],
            symbol=symbol,
            threshold=float(threshold),
        )
        if not events.empty:
            event_frames.append(events)
            per_asset.append(events)

    if per_asset:
        swings = pair_directional_swings(
            pd.concat(per_asset, ignore_index=True)
        )
        if not swings.empty:
            swing_frames.append(swings)

events_all = (
    pd.concat(event_frames, ignore_index=True)
    if event_frames else pd.DataFrame()
)
swings_all = (
    pd.concat(swing_frames, ignore_index=True)
    if swing_frames else pd.DataFrame()
)


# %% 5 - Entradas e saidas reais ja observadas no U59+8
if not TRADES_FILE.exists():
    raise RuntimeError(
        "Trades do U59+8 ausentes. Execute antes a avaliacao financeira "
        "1.17.2. Arquivo esperado: " + str(TRADES_FILE)
    )

trades = pd.read_csv(TRADES_FILE)
trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
trades["entry_timestamp"] = pd.to_datetime(
    trades.get("entry_timestamp"),
    utc=True,
    errors="coerce",
)
asset_trades = trades[
    trades["asset"].astype(str).str.upper().isin(ASSETS_58M)
].copy()
closed = asset_trades[
    asset_trades["action"].astype(str).str.upper().isin(
        {"SELL", "FINAL_SELL"}
    )
].copy()


# %% 6 - Cada operacao contra o proprio historico do ativo
trade_rows = []

for trade in closed.to_dict(orient="records"):
    symbol = str(trade["asset"]).upper()
    frame = frames[symbol]
    index = pd.DatetimeIndex(frame.index)
    entry_ts = trade.get("entry_timestamp")
    exit_ts = trade.get("timestamp")
    if pd.isna(entry_ts) or pd.isna(exit_ts):
        continue

    entry_pos = _session_position(index, entry_ts)
    exit_pos = _session_position(index, exit_ts)
    if entry_pos is None or exit_pos is None or exit_pos < entry_pos:
        continue

    entry_row = frame.iloc[entry_pos]
    exit_row = frame.iloc[exit_pos]
    window = frame.iloc[entry_pos : exit_pos + 1]

    entry_price = _finite(trade.get("entry_price"))
    entry_price = entry_price or _finite(entry_row.get("open"))
    exit_price = _finite(trade.get("execution_price"))
    exit_price = exit_price or _finite(exit_row.get("open"))
    if entry_price is None or exit_price is None:
        continue

    highs = pd.to_numeric(window["high"], errors="coerce")
    lows = pd.to_numeric(window["low"], errors="coerce")
    peak_price = float(highs.max())
    trough_price = float(lows.min())
    peak_ts = highs.idxmax()
    trough_ts = lows.idxmin()

    row = {
        "asset": symbol,
        "entry_timestamp": _utc(entry_ts),
        "exit_timestamp": _utc(exit_ts),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "position_return": _finite(trade.get("position_return")),
        "realized_pnl": _finite(trade.get("realized_pnl")),
        "holding_bars_reported": _finite(trade.get("holding_bars")),
        "holding_sessions_local": int(exit_pos - entry_pos),
        "holding_peak_timestamp": pd.Timestamp(peak_ts),
        "holding_peak_price": peak_price,
        "holding_trough_timestamp": pd.Timestamp(trough_ts),
        "holding_trough_price": trough_price,
        "mfe_from_entry": peak_price / entry_price - 1.0,
        "mae_from_entry": trough_price / entry_price - 1.0,
        "sessions_entry_to_peak": _sessions_between(
            index, entry_ts, peak_ts
        ),
        "sessions_peak_to_exit": _sessions_between(
            index, peak_ts, exit_ts
        ),
        "exit_drawdown_from_holding_peak": exit_price / peak_price - 1.0,
        "profit_capture_ratio_local": (
            (exit_price / entry_price - 1.0)
            / (peak_price / entry_price - 1.0)
            if peak_price > entry_price else None
        ),
    }

    for feature in CAUSAL_FEATURES:
        row[f"entry_{feature}"] = _finite(entry_row.get(feature))
        row[f"exit_{feature}"] = _finite(exit_row.get(feature))

    for threshold in DIRECTIONAL_CHANGE_THRESHOLDS:
        tag = f"{int(round(float(threshold) * 100)):02d}pct"
        subset = events_all[
            (events_all["asset"] == symbol)
            & (events_all["threshold"] == float(threshold))
        ].copy()

        confirmed_bottoms = subset[
            (subset["event_type"] == "BOTTOM")
            & (subset["confirmation_timestamp"] <= _utc(entry_ts))
        ]
        if not confirmed_bottoms.empty:
            bottom = confirmed_bottoms.sort_values(
                "confirmation_timestamp"
            ).iloc[-1]
            row[f"entry_since_confirmed_bottom_{tag}_sessions"] = (
                _sessions_between(
                    index,
                    bottom["confirmation_timestamp"],
                    entry_ts,
                )
            )
            row[f"entry_above_bottom_extreme_{tag}"] = (
                entry_price / float(bottom["extreme_price"]) - 1.0
            )
            row[f"entry_bottom_confirmation_cost_{tag}"] = (
                float(bottom["confirmation_price"])
                / float(bottom["extreme_price"])
                - 1.0
            )

        future_tops = subset[
            (subset["event_type"] == "TOP")
            & (subset["extreme_timestamp"] >= _utc(entry_ts))
        ]
        if not future_tops.empty:
            top = future_tops.sort_values("extreme_timestamp").iloc[0]
            row[f"entry_to_next_top_{tag}_sessions_expost"] = (
                _sessions_between(
                    index, entry_ts, top["extreme_timestamp"]
                )
            )
            row[f"entry_to_next_top_{tag}_return_expost"] = (
                float(top["extreme_price"]) / entry_price - 1.0
            )

        prior_tops = subset[
            (subset["event_type"] == "TOP")
            & (subset["extreme_timestamp"] <= _utc(exit_ts))
        ]
        if not prior_tops.empty:
            top = prior_tops.sort_values("extreme_timestamp").iloc[-1]
            row[f"exit_since_top_extreme_{tag}_sessions_expost"] = (
                _sessions_between(
                    index, top["extreme_timestamp"], exit_ts
                )
            )
            row[f"exit_below_top_extreme_{tag}"] = (
                exit_price / float(top["extreme_price"]) - 1.0
            )

    trade_rows.append(row)

trade_detail = pd.DataFrame(trade_rows)


# %% 7 - Resumos por ativo
geometry_rows = []
for symbol in ASSETS_58M:
    symbol_events = events_all[events_all["asset"] == symbol]
    symbol_swings = swings_all[swings_all["asset"] == symbol]
    symbol_trades = trade_detail[trade_detail["asset"] == symbol]

    for threshold in DIRECTIONAL_CHANGE_THRESHOLDS:
        ev = symbol_events[
            symbol_events["threshold"] == float(threshold)
        ]
        sw = symbol_swings[
            symbol_swings["threshold"] == float(threshold)
        ]
        up = sw[
            (sw["from_event"] == "BOTTOM")
            & (sw["to_event"] == "TOP")
        ]
        down = sw[
            (sw["from_event"] == "TOP")
            & (sw["to_event"] == "BOTTOM")
        ]
        geometry_rows.append(
            {
                "asset": symbol,
                "threshold": float(threshold),
                "top_count": int((ev["event_type"] == "TOP").sum()),
                "bottom_count": int((ev["event_type"] == "BOTTOM").sum()),
                "median_confirmation_lag_sessions": (
                    _finite(ev["confirmation_lag_sessions"].median())
                    if not ev.empty else None
                ),
                "median_bottom_to_top_return": (
                    _finite(up["swing_return"].median())
                    if not up.empty else None
                ),
                "median_top_to_bottom_return": (
                    _finite(down["swing_return"].median())
                    if not down.empty else None
                ),
                "closed_trades_in_58m_replay": int(len(symbol_trades)),
            }
        )

geometry = pd.DataFrame(geometry_rows)
primary_tag = f"{int(round(PRIMARY_DC_THRESHOLD * 100)):02d}pct"
summary_rows = []

for symbol in ASSETS_58M:
    part = trade_detail[trade_detail["asset"] == symbol].copy()
    row = {"asset": symbol, "closed_trades": int(len(part))}
    if part.empty:
        row.update(
            {
                "win_rate": None,
                "median_position_return": None,
                "median_holding_sessions": None,
                "median_mfe": None,
                "median_mae": None,
                "median_exit_drawdown_from_peak": None,
                "median_profit_capture_ratio": None,
                "median_entry_since_bottom_04_sessions": None,
                "median_entry_above_bottom_04": None,
                "median_entry_to_next_top_04_return_expost": None,
                "share_entries_within_5_sessions_of_confirmed_bottom_04": None,
                "share_exits_within_5_sessions_of_holding_peak": None,
            }
        )
    else:
        returns = pd.to_numeric(part["position_return"], errors="coerce")
        since_bottom = pd.to_numeric(
            part.get(
                f"entry_since_confirmed_bottom_{primary_tag}_sessions"
            ),
            errors="coerce",
        )
        peak_to_exit = pd.to_numeric(
            part["sessions_peak_to_exit"],
            errors="coerce",
        )
        row.update(
            {
                "win_rate": _finite((returns > 0).mean()),
                "median_position_return": _finite(returns.median()),
                "median_holding_sessions": _finite(
                    pd.to_numeric(
                        part["holding_sessions_local"],
                        errors="coerce",
                    ).median()
                ),
                "median_mfe": _finite(
                    pd.to_numeric(
                        part["mfe_from_entry"],
                        errors="coerce",
                    ).median()
                ),
                "median_mae": _finite(
                    pd.to_numeric(
                        part["mae_from_entry"],
                        errors="coerce",
                    ).median()
                ),
                "median_exit_drawdown_from_peak": _finite(
                    pd.to_numeric(
                        part["exit_drawdown_from_holding_peak"],
                        errors="coerce",
                    ).median()
                ),
                "median_profit_capture_ratio": _finite(
                    pd.to_numeric(
                        part["profit_capture_ratio_local"],
                        errors="coerce",
                    ).replace([np.inf, -np.inf], np.nan).median()
                ),
                "median_entry_since_bottom_04_sessions": (
                    _finite(since_bottom.median())
                    if since_bottom is not None else None
                ),
                "median_entry_above_bottom_04": _finite(
                    pd.to_numeric(
                        part.get(
                            f"entry_above_bottom_extreme_{primary_tag}"
                        ),
                        errors="coerce",
                    ).median()
                ),
                "median_entry_to_next_top_04_return_expost": _finite(
                    pd.to_numeric(
                        part.get(
                            f"entry_to_next_top_{primary_tag}_return_expost"
                        ),
                        errors="coerce",
                    ).median()
                ),
                "share_entries_within_5_sessions_of_confirmed_bottom_04": (
                    _finite(since_bottom.between(0, 5).mean())
                    if since_bottom is not None
                    and since_bottom.notna().any()
                    else None
                ),
                "share_exits_within_5_sessions_of_holding_peak": (
                    _finite(peak_to_exit.between(0, 5).mean())
                    if peak_to_exit.notna().any()
                    else None
                ),
            }
        )
    summary_rows.append(row)

asset_summary = pd.DataFrame(summary_rows)


# %% 8 - Graficos por ativo
OUT.mkdir(parents=True, exist_ok=True)
charts_dir = OUT / "graficos"
charts_dir.mkdir(parents=True, exist_ok=True)

for symbol in ASSETS_58M:
    frame = frames[symbol]
    ev = events_all[
        (events_all["asset"] == symbol)
        & (events_all["threshold"] == PRIMARY_DC_THRESHOLD)
    ]
    symbol_trades = asset_trades[
        asset_trades["asset"].astype(str).str.upper() == symbol
    ].copy()

    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.plot(
        frame.index,
        frame["close"],
        linewidth=1.1,
        label="Fechamento",
    )

    tops = ev[ev["event_type"] == "TOP"]
    bottoms = ev[ev["event_type"] == "BOTTOM"]
    if not tops.empty:
        ax.scatter(
            tops["extreme_timestamp"],
            tops["extreme_price"],
            marker="v",
            s=28,
            label="Topos DC 4%",
        )
    if not bottoms.empty:
        ax.scatter(
            bottoms["extreme_timestamp"],
            bottoms["extreme_price"],
            marker="^",
            s=28,
            label="Fundos DC 4%",
        )

    buys = symbol_trades[
        symbol_trades["action"].astype(str).str.upper() == "BUY"
    ]
    sells = symbol_trades[
        symbol_trades["action"].astype(str).str.upper().isin(
            {"SELL", "FINAL_SELL"}
        )
    ]
    if not buys.empty:
        ax.scatter(
            buys["timestamp"],
            pd.to_numeric(buys["execution_price"], errors="coerce"),
            marker="o",
            s=55,
            label="Entrada real U59+8",
        )
    if not sells.empty:
        ax.scatter(
            sells["timestamp"],
            pd.to_numeric(sells["execution_price"], errors="coerce"),
            marker="x",
            s=55,
            label="Saida real U59+8",
        )

    ax.set_title(
        f"{symbol} - standalone: topos, fundos, entradas e saidas"
    )
    ax.set_xlabel("Data")
    ax.set_ylabel("Preco ajustado por splits")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        charts_dir / f"{symbol}_topos_fundos_entradas_saidas.png",
        dpi=180,
        bbox_inches="tight",
    )
    fig.savefig(
        charts_dir / f"{symbol}_topos_fundos_entradas_saidas.svg",
        bbox_inches="tight",
    )
    plt.close(fig)


# %% 9 - Comparativos entre os oito
if not asset_summary.empty:
    plot = asset_summary.set_index("asset")

    fig, ax = plt.subplots(figsize=(11, 6.4))
    values = 100.0 * pd.to_numeric(
        plot["median_position_return"],
        errors="coerce",
    )
    ax.bar(plot.index, values)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_ylabel("Retorno mediano por operacao (%)")
    ax.set_title("Ativos do US$ 58,56M - retorno mediano")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        charts_dir / "comparativo_retorno_mediano_operacoes.png",
        dpi=180,
        bbox_inches="tight",
    )
    fig.savefig(
        charts_dir / "comparativo_retorno_mediano_operacoes.svg",
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6.4))
    values = 100.0 * pd.to_numeric(
        plot["median_exit_drawdown_from_peak"],
        errors="coerce",
    )
    ax.bar(plot.index, values)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_ylabel("Saida abaixo do pico da posicao (%)")
    ax.set_title("Quanto do topo e devolvido antes da saida")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        charts_dir / "comparativo_saida_vs_topo.png",
        dpi=180,
        bbox_inches="tight",
    )
    fig.savefig(
        charts_dir / "comparativo_saida_vs_topo.svg",
        bbox_inches="tight",
    )
    plt.close(fig)


# %% 10 - Exportacao e pacote
asset_summary.to_csv(
    OUT / "standalone_asset_summary.csv",
    index=False,
)
trade_detail.to_csv(
    OUT / "standalone_trade_details.csv",
    index=False,
)
events_all.to_csv(
    OUT / "standalone_directional_change_events.csv",
    index=False,
)
swings_all.to_csv(
    OUT / "standalone_directional_swings.csv",
    index=False,
)
geometry.to_csv(
    OUT / "standalone_turn_geometry.csv",
    index=False,
)

payload = {
    "research_version": SCRIPT_VERSION,
    "execution_schema": EXECUTION_SCHEMA,
    "question": (
        "Caracterizar isoladamente os oito ativos do cenario de US$ 58,56M "
        "quanto a topos, fundos, entradas e saidas, sem novo backtest."
    ),
    "assets": list(ASSETS_58M),
    "primary_directional_change_threshold": PRIMARY_DC_THRESHOLD,
    "directional_change_thresholds": [
        float(x) for x in DIRECTIONAL_CHANGE_THRESHOLDS
    ],
    "protocol": {
        "new_backtest": False,
        "new_model_training": False,
        "changes_rotation_universe": False,
        "uses_existing_58m_entries_and_exits": True,
        "standalone_price_analysis": True,
        "extreme_timestamps_are_ex_post_descriptive": True,
        "confirmation_timestamps_are_causal": True,
        "smart_snapshot_sha256": manifest_smart.get("snapshot_sha256"),
        "trades_source": str(TRADES_FILE),
    },
    "asset_summary": asset_summary.to_dict(orient="records"),
    "runtime_seconds": float(time.perf_counter() - started),
}

with (OUT / "standalone_asset_analysis.json").open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        _json_safe(payload),
        handle,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

package = criar_pacote_analise(
    OUT,
    comparison_file="standalone_asset_analysis.json",
    execution_schema=EXECUTION_SCHEMA,
    archive_name="pacote_analise_ativos_58m_standalone.zip",
)

print("=" * 78, flush=True)
print("[standalone] resumo por ativo", flush=True)
print(asset_summary.to_string(index=False), flush=True)
print(f"[package] pronto={package}", flush=True)
print(
    f"[done] standalone analysis seconds="
    f"{time.perf_counter() - started:.3f}",
    flush=True,
)
sinal_sonoro_conclusao()
