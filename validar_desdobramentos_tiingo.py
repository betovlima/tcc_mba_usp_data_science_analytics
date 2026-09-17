"""Valida splitFactor do snapshot Tiingo antes da normalizacao causal.

O campo splitFactor do EOD nao e aplicado cegamente: ele tambem pode aparecer
em distribuicoes/especies de corporate action que nao correspondem a um split
mecanico simples. Cada candidato precisa explicar a mudanca de escala entre o
close anterior e o open da data do evento.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tcc_engine.config import ASSETS, END_DATE, START_DATE

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "dados"
SERIES_DIR = DATA_ROOT / "series_historicas"
EVENTS_DIR = DATA_ROOT / "eventos_corporativos"
SPLITS_DIR = DATA_ROOT / "desdobramentos"
MANIFEST = DATA_ROOT / "manifesto_desdobramentos_tiingo.json"
TOLERANCE = 0.10
COLUMNS = ["timestamp", "split_de", "split_para", "fator_split", "status"]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def decompose_factor(factor: float) -> tuple[float, float]:
    if factor >= 1.0:
        return 1.0, factor
    return 1.0 / factor, 1.0


def load_raw(symbol: str) -> pd.DataFrame:
    path = SERIES_DIR / f"{symbol}.csv"
    table = pd.read_csv(path)
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in table.columns]
    if missing:
        raise RuntimeError(f"{symbol}: serie RAW invalida: {', '.join(missing)}")
    table = table[required].copy()
    table["timestamp"] = pd.to_datetime(table["timestamp"], utc=True, errors="coerce")
    for c in ("open", "high", "low", "close", "volume"):
        table[c] = pd.to_numeric(table[c], errors="coerce")
    return table.dropna(subset=required).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def load_events(symbol: str) -> pd.DataFrame:
    path = EVENTS_DIR / f"{symbol}.csv"
    table = pd.read_csv(path)
    required = ["timestamp", "dividendo", "fator_split"]
    missing = [c for c in required if c not in table.columns]
    if missing:
        raise RuntimeError(f"{symbol}: eventos invalidos: {', '.join(missing)}")
    table = table[required].copy()
    table["timestamp"] = pd.to_datetime(table["timestamp"], utc=True, errors="coerce")
    table["dividendo"] = pd.to_numeric(table["dividendo"], errors="coerce").fillna(0.0)
    table["fator_split"] = pd.to_numeric(table["fator_split"], errors="coerce").fillna(1.0)
    return table.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def build_validated_splits() -> dict[str, Any]:
    if not SERIES_DIR.exists() or not EVENTS_DIR.exists():
        raise RuntimeError("Snapshots Tiingo ausentes; rode preparar_snapshots_mongo_tiingo.py")
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(START_DATE, tz="UTC")
    end = pd.Timestamp(END_DATE, tz="UTC")
    rejected: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    total_candidates = 0
    total_accepted = 0

    log("Validando splitFactor Tiingo contra ruptura mecanica de preco")
    for pos, symbol in enumerate(ASSETS, start=1):
        raw = load_raw(symbol)
        events = load_events(symbol)
        candidates = events.loc[
            (events["timestamp"] >= start)
            & (events["timestamp"] <= end)
            & (~np.isclose(events["fator_split"].astype(float), 1.0)),
            ["timestamp", "dividendo", "fator_split"],
        ].copy()
        accepted_rows: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []

        for candidate in candidates.itertuples(index=False):
            total_candidates += 1
            event_date = pd.Timestamp(candidate.timestamp).normalize()
            factor = float(candidate.fator_split)
            indexes = raw.index[raw["timestamp"].dt.normalize() == event_date].tolist()
            diagnostic: dict[str, Any] = {
                "ativo": symbol,
                "data": event_date.date().isoformat(),
                "fator_split_eod": factor,
                "dividendo_eod": float(candidate.dividendo),
            }

            if len(indexes) != 1 or indexes[0] == 0:
                diagnostic.update({"aceito": False, "motivo": "data_sem_sessao_anterior_unica"})
                diagnostics.append(diagnostic)
                rejected.append(diagnostic)
                continue

            i = int(indexes[0])
            previous_close = float(raw.loc[i - 1, "close"])
            event_open = float(raw.loc[i, "open"])
            event_close = float(raw.loc[i, "close"])
            if not all(np.isfinite(v) and v > 0 for v in (previous_close, event_open, event_close, factor)):
                diagnostic.update({"aceito": False, "motivo": "preco_ou_fator_invalido"})
                diagnostics.append(diagnostic)
                rejected.append(diagnostic)
                continue

            observed_open_ratio = previous_close / event_open
            observed_close_ratio = previous_close / event_close
            relative_error = abs(observed_open_ratio / factor - 1.0)
            accepted = relative_error <= TOLERANCE
            diagnostic.update({
                "fechamento_anterior": previous_close,
                "abertura_evento": event_open,
                "fechamento_evento": event_close,
                "razao_observada_abertura": observed_open_ratio,
                "razao_observada_fechamento": observed_close_ratio,
                "desvio_relativo_abertura": relative_error,
                "aceito": bool(accepted),
                "motivo": "ruptura_compativel_com_desdobramento" if accepted else "fator_nao_explica_ruptura_de_preco",
            })
            diagnostics.append(diagnostic)
            if not accepted:
                rejected.append(diagnostic)
                continue

            split_from, split_to = decompose_factor(factor)
            accepted_rows.append({
                "timestamp": pd.Timestamp(candidate.timestamp),
                "split_de": float(split_from),
                "split_para": float(split_to),
                "fator_split": factor,
                "status": "a",
            })

        output = pd.DataFrame(accepted_rows, columns=COLUMNS)
        if not output.empty:
            output = output.sort_values("timestamp").drop_duplicates(["timestamp", "fator_split"], keep="last")
        output.to_csv(SPLITS_DIR / f"{symbol}.csv", index=False)
        total_accepted += len(output)
        summaries.append({
            "ativo": symbol,
            "candidatos_eod": len(candidates),
            "desdobramentos_aceitos": len(output),
            "diagnosticos": diagnostics,
        })
        log(f"{pos:02d}/{len(ASSETS)} {symbol} | candidatos={len(candidates)} | aceitos={len(output)}")

    manifest = {
        "fonte_precos": "Tiingo EOD snapshot local",
        "fonte_candidatos": "splitFactor do Tiingo EOD snapshot local",
        "metodo": "validacao_de_ruptura_mecanica_no_open",
        "tolerancia_relativa_abertura": TOLERANCE,
        "data_geracao_utc": datetime.now(timezone.utc).isoformat(),
        "periodo_inicio": START_DATE,
        "periodo_fim": END_DATE,
        "quantidade_ativos": len(ASSETS),
        "total_candidatos_eod": total_candidates,
        "total_desdobramentos_aceitos": total_accepted,
        "total_candidatos_rejeitados": len(rejected),
        "candidatos_rejeitados": rejected,
        "resumo_ativos": summaries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    log(f"Splits validados | candidatos={total_candidates} | aceitos={total_accepted} | rejeitados={len(rejected)}")
    for row in rejected:
        log(f"Rejeitado: {row['ativo']} {row['data']} fator={row['fator_split_eod']} motivo={row['motivo']}")
    return manifest


def main() -> int:
    build_validated_splits()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
