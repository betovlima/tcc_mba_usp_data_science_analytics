"""BUY-AND-HOLD DO UNIVERSO COMPLETO DO CENARIO DE US$ 58,56M.

O cenario exploratorio de US$ 58,56M e:
    U67 = U59 + 8 positivos

Logo, o universo completo tem 67 ativos:
- U56 congelado;
- + COLB, AMS, FOXF;
- + THO, WDAY, EXR, XEL, SBFG, PAYX, MUX, SXC.

Este arquivo NAO roda a estrategia, NAO treina modelo e NAO altera a pesquisa.
Ele apenas calcula o benchmark buy-and-hold de US$ 10.000 por ativo usando
os mesmos snapshots congelados e os mesmos ajustes por splits do pipeline.

Dois resultados sao produzidos:
1. own_history: cada ativo investe US$10 mil no primeiro fechamento disponivel
   da sua propria serie e vende no ultimo fechamento disponivel.
2. common_window: todos os 67 investem na mesma janela comum, para comparacao
   estritamente temporal entre ativos.

Execute no Spyder com F5 ou pelas celulas # %%.
"""

from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from engine.configuracao import CONFIG
from pesquisas.directional_change_lightgbm import (
    criar_pacote_analise,
    sinal_sonoro_conclusao,
)
from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.preparacao import prepare_model_frames


# %% 0 - Configuracao congelada
ROOT = Path(__file__).resolve().parent
BASE = SnapshotPaths.research(ROOT)
B2 = SnapshotPaths.from_root(
    ROOT / "dados" / "pesquisa_expansao_76_b2"
)
SMART = SnapshotPaths.from_root(
    ROOT / "dados" / "pesquisa_smart_candidates"
)

OUT = ROOT / "output" / "buy_hold_universo_58m"

SCRIPT_VERSION = "1.17.4-dev.1"
EXECUTION_SCHEMA = "buy-hold-u67-58m-v1"
INITIAL_PER_ASSET = 10_000.0

U59_ADDITIONS = ("COLB", "AMS", "FOXF")
POSITIVE8 = (
    "THO", "WDAY", "EXR", "XEL",
    "SBFG", "PAYX", "MUX", "SXC",
)


# %% 1 - Carregamento dos tres snapshots
started = time.perf_counter()

manifest_base = validate_snapshot(BASE)
manifest_b2 = validate_snapshot(B2)
manifest_smart = validate_snapshot(SMART)

frames_u56, exc_u56, _, _ = prepare_model_frames(
    BASE,
    assets=CONFIG.assets,
    comparar_snapshot_referencia=False,
    allow_structural_assets=frozenset({"CLMT", "DOC"}),
)
if len(frames_u56) != 56:
    raise RuntimeError(
        f"U56 deveria conter 56 ativos; obtidos {len(frames_u56)}."
    )

frames_b2, exc_b2, _, _ = prepare_model_frames(
    B2,
    assets=U59_ADDITIONS,
    comparar_snapshot_referencia=False,
)
if exc_b2 or len(frames_b2) != 3:
    raise RuntimeError(
        "COLB, AMS e FOXF precisam estar integralmente disponiveis."
    )

frames_pos8, exc_pos8, _, _ = prepare_model_frames(
    SMART,
    assets=POSITIVE8,
    comparar_snapshot_referencia=False,
)
if exc_pos8 or len(frames_pos8) != 8:
    raise RuntimeError(
        "Os oito ativos positivos precisam estar integralmente disponiveis."
    )

frames_u67 = {
    **frames_u56,
    **frames_b2,
    **frames_pos8,
}
symbols = sorted(frames_u67)

if len(symbols) != 67:
    duplicates = (
        len(frames_u56) + len(frames_b2) + len(frames_pos8) - len(symbols)
    )
    raise RuntimeError(
        "O cenario de US$58,56M deveria conter 67 ativos unicos; "
        f"obtidos={len(symbols)} duplicates={duplicates}."
    )

print(
    f"[universe] U67 assets={len(symbols)} "
    f"initial_total={len(symbols) * INITIAL_PER_ASSET:,.2f}",
    flush=True,
)


# %% 2 - Benchmark pela propria historia de cada ativo
own_rows = []

for symbol in symbols:
    frame = frames_u67[symbol].copy().sort_index()
    close = pd.to_numeric(frame["close"], errors="coerce").dropna()
    close = close[close > 0]
    if close.empty:
        raise RuntimeError(f"{symbol}: serie de fechamento vazia.")

    first_date = pd.Timestamp(close.index[0])
    last_date = pd.Timestamp(close.index[-1])
    first_close = float(close.iloc[0])
    last_close = float(close.iloc[-1])
    gross = last_close / first_close
    final_value = INITIAL_PER_ASSET * gross

    own_rows.append(
        {
            "asset": symbol,
            "first_date": first_date,
            "last_date": last_date,
            "first_close": first_close,
            "last_close": last_close,
            "return": gross - 1.0,
            "initial_value": INITIAL_PER_ASSET,
            "final_value": final_value,
            "profit": final_value - INITIAL_PER_ASSET,
        }
    )

own = pd.DataFrame(own_rows).sort_values(
    ["final_value", "asset"],
    ascending=[False, True],
    ignore_index=True,
)

own_initial_total = float(own["initial_value"].sum())
own_final_total = float(own["final_value"].sum())
own_profit = own_final_total - own_initial_total
own_return = own_final_total / own_initial_total - 1.0


# %% 3 - Benchmark em janela temporal comum
common_start = max(
    pd.Timestamp(
        pd.to_numeric(
            frames_u67[symbol]["close"],
            errors="coerce",
        ).dropna().index[0]
    )
    for symbol in symbols
)
common_end = min(
    pd.Timestamp(
        pd.to_numeric(
            frames_u67[symbol]["close"],
            errors="coerce",
        ).dropna().index[-1]
    )
    for symbol in symbols
)

common_rows = []

for symbol in symbols:
    frame = frames_u67[symbol].copy().sort_index()
    close = pd.to_numeric(frame["close"], errors="coerce").dropna()
    close = close[
        (close.index >= common_start)
        & (close.index <= common_end)
        & (close > 0)
    ]
    if close.empty:
        raise RuntimeError(
            f"{symbol}: sem dados dentro da janela comum "
            f"{common_start} -> {common_end}."
        )

    first_date = pd.Timestamp(close.index[0])
    last_date = pd.Timestamp(close.index[-1])
    first_close = float(close.iloc[0])
    last_close = float(close.iloc[-1])
    gross = last_close / first_close
    final_value = INITIAL_PER_ASSET * gross

    common_rows.append(
        {
            "asset": symbol,
            "first_date": first_date,
            "last_date": last_date,
            "first_close": first_close,
            "last_close": last_close,
            "return": gross - 1.0,
            "initial_value": INITIAL_PER_ASSET,
            "final_value": final_value,
            "profit": final_value - INITIAL_PER_ASSET,
        }
    )

common = pd.DataFrame(common_rows).sort_values(
    ["final_value", "asset"],
    ascending=[False, True],
    ignore_index=True,
)

common_initial_total = float(common["initial_value"].sum())
common_final_total = float(common["final_value"].sum())
common_profit = common_final_total - common_initial_total
common_return = common_final_total / common_initial_total - 1.0


# %% 4 - Exportacao
OUT.mkdir(parents=True, exist_ok=True)

own.to_csv(
    OUT / "buy_hold_u67_own_history.csv",
    index=False,
)
common.to_csv(
    OUT / "buy_hold_u67_common_window.csv",
    index=False,
)

summary = {
    "research_version": SCRIPT_VERSION,
    "execution_schema": EXECUTION_SCHEMA,
    "scenario": "U67 = U59 + 8 positivos = cenario US$58.56M",
    "asset_count": len(symbols),
    "assets": symbols,
    "initial_per_asset": INITIAL_PER_ASSET,
    "own_history": {
        "initial_total": own_initial_total,
        "final_total": own_final_total,
        "profit": own_profit,
        "return": own_return,
        "earliest_first_date": str(own["first_date"].min()),
        "latest_first_date": str(own["first_date"].max()),
        "last_date": str(own["last_date"].max()),
    },
    "common_window": {
        "start": str(common_start),
        "end": str(common_end),
        "initial_total": common_initial_total,
        "final_total": common_final_total,
        "profit": common_profit,
        "return": common_return,
    },
    "method": {
        "fractional_shares": True,
        "dividends_reinvested": False,
        "fees": False,
        "taxes": False,
        "prices_split_normalized_by_existing_pipeline": True,
        "new_backtest": False,
        "new_model_training": False,
    },
    "snapshots": {
        "u56": manifest_base.get("snapshot_sha256"),
        "b2": manifest_b2.get("snapshot_sha256"),
        "smart": manifest_smart.get("snapshot_sha256"),
    },
    "runtime_seconds": float(time.perf_counter() - started),
}

with (OUT / "buy_hold_u67_summary.json").open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        summary,
        handle,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

print("=" * 78, flush=True)
print("[buy-hold] CENARIO DE US$58,56M = U67", flush=True)
print(
    f"[own-history] initial={own_initial_total:,.2f} "
    f"final={own_final_total:,.2f} "
    f"profit={own_profit:,.2f} "
    f"return={own_return:+.4%}",
    flush=True,
)
print(
    f"[common-window] {common_start.date()} -> {common_end.date()} "
    f"initial={common_initial_total:,.2f} "
    f"final={common_final_total:,.2f} "
    f"profit={common_profit:,.2f} "
    f"return={common_return:+.4%}",
    flush=True,
)

package = criar_pacote_analise(
    OUT,
    comparison_file="buy_hold_u67_summary.json",
    execution_schema=EXECUTION_SCHEMA,
    archive_name="pacote_buy_hold_universo_58m_u67.zip",
)

print(f"[package] pronto={package}", flush=True)
print(
    f"[done] buy-hold U67 seconds={time.perf_counter() - started:.3f}",
    flush=True,
)
sinal_sonoro_conclusao()
