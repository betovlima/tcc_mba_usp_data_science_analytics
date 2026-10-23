"""TCC v1.24.4 — preços RAW imutáveis, rótulos econômicos com splits.

Auditoria apenas de retorno de PREÇO (sem taxas, dividendos ou penalidades
de risco), para os horizontes de 5,10,20,40,60 sessões do LightGBM.

Os splits alteram somente a QUANTIDADE ECONÔMICA mantida entre entrada e
saída. Os CSVs e DataFrames RAW permanecem intactos. A versão legada
normalizada é usada EXCLUSIVAMENTE como controle matemático em memória.

Não treina modelos, não opera, não modifica a main nem promove backtest.
Ao pressionar F5 no Spyder, grava:
output/retornos_economicos_splits/<ID>/pacote_retornos_economicos_splits.zip
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd

from engine.configuracao import CONFIG
from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.preparacao import (
    load_actions_file, load_raw_bar_file, split_normalize,
)
from reproducao.universo_fixo75 import universo_75


ROOT = Path(__file__).resolve().parent
VERSION = "1.24.4"
SNAPSHOT_END = "2026-10-22"
HORIZONS = (5, 10, 20, 40, 60)
WINDOW_SESSIONS = 60
RTOL = 1e-10
ATOL = 1e-12


def _utc_day(value: object) -> pd.Timestamp:
    day = pd.Timestamp(value)
    if pd.isna(day):
        raise ValueError("Data de split ausente.")
    return (day.tz_localize("UTC") if day.tzinfo is None
            else day.tz_convert("UTC")).normalize()


def split_events(actions: list[dict]) -> tuple[tuple[pd.Timestamp, float], ...]:
    """Retém somente a ex-date e fator de QUANTIDADE, nunca um preço ajustado."""
    found: list[tuple[pd.Timestamp, float]] = []
    for action in actions:
        if action.get("action_type") not in ("forward_split", "reverse_split"):
            continue
        try:
            factor = float(action["new_rate"]) / float(action["old_rate"])
            day = _utc_day(action["ex_date"])
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
            raise ValueError("Split sem ex-date ou proporção válida.") from exc
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Fator de quantidade deve ser positivo.")
        found.append((day, factor))
    return tuple(sorted(found, key=lambda item: item[0]))


def forward_price_return(
    bars: pd.DataFrame,
    event_factors: tuple[tuple[pd.Timestamp, float], ...],
    *,
    decision_index: int,
    horizon: int,
) -> dict:
    """Compra na ABERTURA seguinte à decisão e venda no FECHAMENTO de t+h.

    Apenas splits cuja ex-date é DEPOIS do pregão de compra e até a venda
    alteram quantidade: um comprador na abertura da própria ex-date já
    compra ações pós-split, não tem direito a outro multiplicador.
    """
    if horizon < 1 or decision_index < 0 or decision_index + horizon >= len(bars):
        raise ValueError("Janela futura indisponível ou índice inválido.")
    if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
        raise ValueError("Calendário de barras inconsistente.")
    first = decision_index + 1
    last = decision_index + horizon
    entry_day = _utc_day(bars.index[first])
    exit_day = _utc_day(bars.index[last])
    entry = float(bars.iloc[first]["open"])
    exit_price = float(bars.iloc[last]["close"])
    if not all(np.isfinite(x) and x > 0 for x in (entry, exit_price)):
        raise ValueError("Preço RAW de entrada/saída inválido.")
    factor = 1.0
    crossed = 0
    for effective_day, quantity_multiplier in event_factors:
        if entry_day < effective_day <= exit_day:
            factor *= quantity_multiplier
            crossed += 1
    raw_return = exit_price / entry - 1
    economic_return = exit_price * factor / entry - 1
    return {
        "decision_day": _utc_day(bars.index[decision_index]).date().isoformat(),
        "entry_day": entry_day.date().isoformat(),
        "exit_day": exit_day.date().isoformat(),
        "horizon": horizon,
        "raw_entry_open": entry,
        "raw_exit_close": exit_price,
        "split_quantity_multiplier": factor,
        "crossed_split_count": crossed,
        "naive_raw_return": raw_return,
        "custody_economic_price_return": economic_return,
        # O target só é observável após o fechamento da sessão de saída.
        "label_matures_after_close_on": exit_day.date().isoformat(),
    }


def measure_around_split(
    symbol: str, bars: pd.DataFrame, actions: list[dict],
    *,
    horizon_set: tuple[int, ...] = HORIZONS,
    window_sessions: int = WINDOW_SESSIONS,
) -> list[dict]:
    """Audita retornos de 5 horizontes, sem modificar séries de origem."""
    original_hash = hashlib.sha256(
        pd.util.hash_pandas_object(bars, index=True).to_numpy(
            dtype=np.uint64
        ).tobytes()
    ).hexdigest()
    factors = split_events(actions)
    if not factors:
        return []
    # Controle de equivalência matemática, não é entrada do novo modelo.
    legacy_bars, _ = split_normalize(bars, actions)
    days = pd.DatetimeIndex(bars.index).normalize()
    output: list[dict] = []
    decisions: set[int] = set()
    for event_day, _ in factors:
        k = int(days.searchsorted(event_day, side="left"))
        if not (0 < k < len(days)):
            continue
        for i in range(max(0, k-window_sessions), min(len(days)-1, k+window_sessions)):
            decisions.add(i)

    for i in sorted(decisions):
        for horizon in horizon_set:
            if i + horizon >= len(bars):
                continue
            finding = forward_price_return(
                bars, factors, decision_index=i, horizon=horizon,
            )
            legacy_entry = float(legacy_bars.iloc[i+1]["open"])
            legacy_exit = float(legacy_bars.iloc[i+horizon]["close"])
            legacy_return = legacy_exit / legacy_entry - 1
            finding["legacy_split_normalized_price_return"] = legacy_return
            finding["economic_vs_legacy_abs_error"] = abs(
                finding["custody_economic_price_return"] - legacy_return
            )
            finding["raw_vs_economic_abs_difference"] = abs(
                finding["naive_raw_return"]
                - finding["custody_economic_price_return"]
            )
            finding["economic_equals_legacy"] = bool(np.isclose(
                finding["custody_economic_price_return"], legacy_return,
                atol=ATOL, rtol=RTOL,
            ))
            finding["symbol"] = symbol
            output.append(finding)

    final_hash = hashlib.sha256(
        pd.util.hash_pandas_object(bars, index=True).to_numpy(
            dtype=np.uint64
        ).tobytes()
    ).hexdigest()
    if original_hash != final_hash:
        raise RuntimeError(f"{symbol}: RAW foi alterado.")
    return output


def audit(*, root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    manifest_contract = json.loads(
        (root / "evidencias" / "manifest_universo_congelado.json").read_text(
            encoding="utf-8"
        )
    )
    assets = universo_75(manifest_contract, ())
    source = None
    manifest = None
    for folder in sorted(
        (root / "dados" / ".u75_fresh_runs").glob(
            f"asof_{SNAPSHOT_END}_*/universe_75"
        ), reverse=True,
    ):
        paths = SnapshotPaths.from_root(folder)
        if paths.manifest.is_file():
            current = validate_snapshot(paths)
            if tuple(current.get("assets") or ()) == assets:
                source, manifest = paths, current
                break
    if source is None or manifest is None:
        raise FileNotFoundError(
            "Snapshot único de 75 ativos não encontrado; sem download automático."
        )
    rows: list[dict] = []
    symbols_with_splits = []
    for symbol in assets:
        bars = load_raw_bar_file(
            source.raw_bars / f"{symbol}.csv",
            float_precision="round_trip", strict_raw=True,
        )
        actions = load_actions_file(
            source.corporate_actions / f"{symbol}.csv"
        )
        events = split_events(actions)
        if not events:
            continue
        symbols_with_splits.append(symbol)
        entries = measure_around_split(symbol, bars, actions)
        rows.extend(entries)
        print(f"[economic-label] {symbol}: eventos={len(events)} "
              f"janelas={len(entries)}", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Nenhuma observação de horizonte foi construída.")
    bad = df.loc[~df["economic_equals_legacy"]]
    changed = df.loc[df["raw_vs_economic_abs_difference"] > ATOL]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / "output" / "retornos_economicos_splits" / stamp
    output.mkdir(parents=True, exist_ok=False)
    df.to_csv(output / "retornos_por_decisao_e_horizonte.csv",
              index=False, float_format="%.17g")
    by_symbol = df.groupby("symbol", as_index=False).agg(
        rows=("decision_day", "size"),
        crossed_split_rows=("crossed_split_count", lambda x: int((x>0).sum())),
        max_naive_vs_economic=("raw_vs_economic_abs_difference", "max"),
        max_economic_vs_legacy=("economic_vs_legacy_abs_error", "max"),
    )
    by_symbol.to_csv(output / "comparacao_por_ativo.csv", index=False)
    result = {
        "version": VERSION,
        "branch": "research/v1.23.1-fresh-snapshot-20261022",
        "snapshot_sha256": manifest["snapshot_sha256"],
        "assets": len(assets),
        "assets_with_recorded_split": len(symbols_with_splits),
        "horizons": list(HORIZONS),
        "horizon_comparisons": len(df),
        "horizons_crossing_split": int((df["crossed_split_count"]>0).sum()),
        "naive_raw_vs_custody_economic_different": len(changed),
        "economic_vs_legacy_not_close": len(bad),
        "max_economic_vs_legacy_abs_error": float(
            df["economic_vs_legacy_abs_error"].max()
        ),
        "max_naive_raw_vs_economic_abs_difference": float(
            df["raw_vs_economic_abs_difference"].max()
        ),
        "raw_modified": False,
        "model_retrained": False,
        "portfolio_simulated": False,
        "label_availability_rule": (
            "Label de horizonte h só pode entrar em treino após o "
            "fechamento da sessão de saída t+h."
        ),
        "limitations": (
            "Só retorno bruto de preço, sem taxas, dividendos em dinheiro, "
            "risco intraperíodo, drawdown, movimento capturado, cash-in-lieu "
            "ou execução de ordens; não substitui labels de utilidade."
        ),
    }
    (output / "resumo.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    archive = output / "pacote_retornos_economicos_splits.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipf:
        for path in ("retornos_por_decisao_e_horizonte.csv",
                     "comparacao_por_ativo.csv", "resumo.json"):
            zipf.write(output / path, arcname=path)
    print(f"[economic-label] {len(df)} comparações, "
          f"divergências econômicas/legado={len(bad)}; pacote={archive}")
    return result


if __name__ == "__main__":
    audit()
