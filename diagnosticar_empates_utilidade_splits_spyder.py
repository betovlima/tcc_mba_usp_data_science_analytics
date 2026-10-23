"""v1.24.7 — localizar os 3 empates de preco nas utilidades RAW de splits.

Apenas leitura da mesma fotografia U75; diagnostico deterministico de:
 YANG: 2024-10-16
 CAPR: 2019-05-20, 2019-05-28.

Compara bit a bit, sem tolerancia no criterio >, as 60 cotacoes futuras
contra a abertura de entrada, e os 59 passos entre fechamentos no motor
economico vs LEGADO, calculado somente em memoria. NUNCA altera RAW.
Nao retreina, otimiza ou calcula capital.

No Spyder: F5. ZIP: output/empates_utilidade_splits/<ID>/
  pacote_empates_utilidade_splits.zip
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
from reproducao.preparacao import load_actions_file, load_raw_bar_file, split_normalize
from reproducao.universo_fixo75 import universo_75
from auditar_retornos_economicos_splits_spyder import split_events
from auditar_utilidades_economicas_raw_spyder import economic_utility_at_decision


ROOT = Path(__file__).resolve().parent
VERSION = "1.24.7"
SNAPSHOT_END = "2026-10-22"
CASES = (
    ("YANG", "2024-10-16"),
    ("CAPR", "2019-05-20"),
    ("CAPR", "2019-05-28"),
)


def compare_positive_share(
    economic_entry: float,
    economic_closes: np.ndarray,
    legacy_entry: float,
    legacy_closes: np.ndarray,
    *,
    days: pd.DatetimeIndex,
    symbol: str,
    decision_day: str,
) -> tuple[dict, list[dict]]:
    """Compara os mesmos 60 fechamentos com as duas entradas, float64 exato."""
    eclose = np.asarray(economic_closes, dtype=np.float64)
    lclose = np.asarray(legacy_closes, dtype=np.float64)
    if len(eclose) != len(lclose) or len(eclose) != len(days):
        raise ValueError("Calendario dos 60 pregões não coincide.")
    if len(eclose) != 60:
        raise ValueError("Esperadas exatamente 60 sessões no horizonte máximo.")
    eentry, lentry = np.float64(economic_entry), np.float64(legacy_entry)
    good = np.isfinite(eclose).all() and np.isfinite(lclose).all()
    if not good or not np.isfinite(eentry) or not np.isfinite(lentry):
        raise ValueError("Valores não finitos impedem comparação.")
    eflags = eclose > eentry
    lflags = lclose > lentry
    esteps = np.diff(eclose) > 0.
    lsteps = np.diff(lclose) > 0.
    mismatch = np.flatnonzero(eflags != lflags)
    step_mismatch = np.flatnonzero(esteps != lsteps)
    results = []
    for pos in mismatch:
        a, b = float(eclose[pos]), float(lclose[pos])
        ediff = a - float(eentry)
        ldiff = b - float(lentry)
        results.append({
            "symbol": symbol,
            "decision_day": decision_day,
            "comparison_type": "close_gt_entry",
            "future_session": days[pos].date().isoformat(),
            "future_offset": int(pos+1),
            "economic_close": a,
            "economic_entry": float(eentry),
            "legacy_close": b,
            "legacy_entry": float(lentry),
            "economic_gt": bool(eflags[pos]),
            "legacy_gt": bool(lflags[pos]),
            "economic_close_minus_entry": ediff,
            "legacy_close_minus_entry": ldiff,
            "economic_close_float_hex": a.hex(),
            "economic_entry_float_hex": float(eentry).hex(),
            "legacy_close_float_hex": b.hex(),
            "legacy_entry_float_hex": float(lentry).hex(),
            "economic_abs_diff_in_entry_ulps": float(
                abs(ediff) / abs(np.spacing(eentry))
            ),
            "legacy_abs_diff_in_entry_ulps": float(
                abs(ldiff) / abs(np.spacing(lentry))
            ),
        })
    for pos in step_mismatch:
        a, b = float(eclose[pos+1]), float(lclose[pos+1])
        ediff = a - float(eclose[pos])
        ldiff = b - float(lclose[pos])
        results.append({
            "symbol": symbol, "decision_day": decision_day,
            "comparison_type": "close_t_plus_1_gt_close_t",
            "future_session": days[pos+1].date().isoformat(),
            "future_offset": int(pos+2),
            "economic_close": a, "economic_entry": float(eclose[pos]),
            "legacy_close": b, "legacy_entry": float(lclose[pos]),
            "economic_gt": bool(esteps[pos]),
            "legacy_gt": bool(lsteps[pos]),
            "economic_close_minus_entry": ediff,
            "legacy_close_minus_entry": ldiff,
            "economic_close_float_hex": a.hex(),
            "economic_entry_float_hex": float(eclose[pos]).hex(),
            "legacy_close_float_hex": b.hex(),
            "legacy_entry_float_hex": float(lclose[pos]).hex(),
            "economic_abs_diff_in_entry_ulps": float(
                abs(ediff) / abs(np.spacing(eclose[pos]))
            ),
            "legacy_abs_diff_in_entry_ulps": float(
                abs(ldiff) / abs(np.spacing(lclose[pos]))
            ),
        })
    e_persist = 0.5 * float(eflags.mean()) + 0.5 * float(esteps.mean())
    l_persist = 0.5 * float(lflags.mean()) + 0.5 * float(lsteps.mean())
    summary = {
        "symbol": symbol, "decision_day": decision_day,
        "economic_close_gt_entry_count": int(eflags.sum()),
        "legacy_close_gt_entry_count": int(lflags.sum()),
        "economic_positive_steps_count": int(esteps.sum()),
        "legacy_positive_steps_count": int(lsteps.sum()),
        "positive_share_mismatch_count": int(len(mismatch)),
        "positive_steps_mismatch_count": int(len(step_mismatch)),
        "economic_trend_persistence": e_persist,
        "legacy_trend_persistence": l_persist,
        "persistence_delta": e_persist - l_persist,
        "misclassified_sessions": ";".join(
            item["future_session"] for item in results
        ),
    }
    return summary, results


def analyze_case(
    symbol: str, day: str, raw: pd.DataFrame, actions: list[dict]
) -> tuple[dict, list[dict]]:
    stamp = pd.Timestamp(day, tz="UTC")
    # A rotina original usa timestamps UTC; compara pela data de sessão.
    indices = pd.DatetimeIndex(raw.index).normalize()
    target = np.flatnonzero(indices == stamp)
    if len(target) != 1:
        raise RuntimeError(f"{symbol} {day}: data ausente/duplicada no RAW.")
    i = int(target[0])
    maximum = max(map(int, CONFIG.rotation_target_horizons))
    if i + maximum >= len(raw):
        raise RuntimeError(f"{symbol} {day}: não há rótulo completo.")
    previous_hash = hashlib.sha256(
        pd.util.hash_pandas_object(raw, index=True).to_numpy(
            dtype=np.uint64
        ).tobytes(),
    ).hexdigest()

    adjusted, _ = split_normalize(raw, actions)  # memoria apenas
    events = split_events(actions)
    economic = economic_utility_at_decision(
        raw, events, decision_index=i,
    )
    start = i + 1
    days = pd.DatetimeIndex(raw.index[start:start+maximum])
    entry_day = indices[start]
    quantity = np.ones(maximum, dtype=np.float64)
    for effective_day, ratio in events:
        if effective_day > entry_day:
            quantity[indices[start:start+maximum] >= effective_day] *= ratio

    econ_close = raw.iloc[start:start+maximum]["close"].to_numpy(
        dtype=np.float64
    ) * quantity
    legacy_close = adjusted.iloc[start:start+maximum]["close"].to_numpy(
        dtype=np.float64
    )
    econ_entry = float(raw.iloc[start]["open"])
    legacy_entry = float(adjusted.iloc[start]["open"])
    finding, mismatches = compare_positive_share(
        econ_entry, econ_close, legacy_entry, legacy_close,
        days=days, symbol=symbol, decision_day=day,
    )
    if not np.isclose(
        finding["economic_trend_persistence"],
        economic["forward_trend_persistence"], rtol=0, atol=1e-14,
    ):
        raise RuntimeError("Diagnóstico de comparações não reproduziu a utilidade.")
    finding.update({
        "entry_day": entry_day.date().isoformat(),
        "last_target_close_day": days[-1].date().isoformat(),
        "economic_forward_risk_adjusted_utility":
            economic["forward_risk_adjusted_utility"],
        "raw_preserved": True,
        "label_maturity_verified": False,
        "model_retrained": False,
    })
    if hashlib.sha256(
        pd.util.hash_pandas_object(raw, index=True).to_numpy(
            dtype=np.uint64
        ).tobytes(),
    ).hexdigest() != previous_hash:
        raise RuntimeError(f"{symbol}: RAW alterado durante o diagnóstico.")
    return finding, mismatches


def audit(*, root: Path=ROOT) -> dict:
    root = Path(root).resolve()
    contract = json.loads(
        (root/"evidencias"/"manifest_universo_congelado.json").read_text(
            encoding="utf-8"
        )
    )
    universe = universo_75(contract, ())
    paths, snapshot = None, None
    for folder in sorted(
        (root/"dados"/".u75_fresh_runs").glob(
            f"asof_{SNAPSHOT_END}_*/universe_75"
        ), reverse=True,
    ):
        probe = SnapshotPaths.from_root(folder)
        if probe.manifest.is_file():
            found = validate_snapshot(probe)
            if tuple(found.get("assets") or ()) == universe:
                paths, snapshot = probe, found
                break
    if paths is None or snapshot is None:
        raise FileNotFoundError(
            "Snapshot U75 validado indisponível; nenhuma rede será consultada."
        )
    summaries, mismatches = [], []
    for symbol, day in CASES:
        bars = load_raw_bar_file(
            paths.raw_bars/f"{symbol}.csv",
            float_precision="round_trip", strict_raw=True,
        )
        events = load_actions_file(
            paths.corporate_actions/f"{symbol}.csv"
        )
        summary, details = analyze_case(symbol, day, bars, events)
        summaries.append(summary)
        mismatches.extend(details)
        print(
            f"[precision] {symbol} {day} "
            f"close_mismatches={summary['positive_share_mismatch_count']} "
            f"steps_mismatches={summary['positive_steps_mismatch_count']}",
            flush=True,
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dest = root/"output"/"empates_utilidade_splits"/stamp
    dest.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(summaries).to_csv(dest/"resumo_decisoes.csv", index=False)
    pd.DataFrame(mismatches).to_csv(dest/"comparacoes_divergentes.csv", index=False)
    report = {
        "version": VERSION,
        "branch": "research/v1.23.1-fresh-snapshot-20261022",
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "universe_size_verified": len(universe),
        "cases": len(CASES),
        "close_gt_entry_mismatches": sum(
            r["positive_share_mismatch_count"] for r in summaries
        ),
        "adjacent_close_step_mismatches": sum(
            r["positive_steps_mismatch_count"] for r in summaries
        ),
        "raw_modified": False,
        "model_retrained": False,
        "policy_changed": False,
        "capital_calculated": False,
        "caution": (
            "Diagnóstico pontual: a divergência de comparação near-tie "
            "não autoriza substituir > por uma tolerância ad hoc no "
            "motor oficial nem concluir paridade OOS de score/decisões."
        ),
    }
    (dest/"resumo.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    path = dest/"pacote_empates_utilidade_splits.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for f in ("resumo.json","resumo_decisoes.csv","comparacoes_divergentes.csv"):
            archive.write(dest/f, arcname=f)
    print(f"[precision] pacote={path}", flush=True)
    return report


if __name__ == "__main__":
    audit()
