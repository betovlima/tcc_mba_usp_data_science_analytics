"""Descoberta FUTURA de ativos, separada do U75 e sem alterar o MCT.

Busca o catalogo de acoes listadas na Alpaca (somente GET), retira os
75 do universo historico e compara com o catalogo congelado anterior.
A entrada em ranking ou na politica de capital exige, em outro processo,
historico valido, identidade, liquidez, atributos e verificacao temporal.
Nenhum ticker descoberto e promovido ou negociado automaticamente.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import requests

from pesquisa.control_operacional_strategy30 import ROOT, carregar_contrato
from reproducao.dados import load_alpaca_credentials


VERSION = "future-asset-watchlist-v1.0.0"
URL = "https://paper-api.alpaca.markets/v2/assets"
OUT = ROOT / "output" / "catalogo_ativos_futuros"


def _assets_from_response(raw: list[dict], control: set[str]) -> pd.DataFrame:
    if not isinstance(raw, list) or not raw:
        raise ValueError("Catalogo Alpaca vazio ou invalido")
    rows = []
    seen_symbols = set()
    for item in raw:
        sym = str(item.get("symbol") or "").strip().upper()
        identity = str(item.get("id") or "").strip()
        if (not sym or sym in seen_symbols or not identity):
            raise ValueError(f"Identidade Alpaca ausente ou duplicada: {sym!r}")
        seen_symbols.add(sym)
        if item.get("status") != "active" or not bool(item.get("tradable")):
            continue
        if str(item.get("class") or item.get("asset_class") or "") != "us_equity":
            continue
        rows.append({
            "symbol": sym, "alpaca_id": identity,
            "exchange": str(item.get("exchange") or ""),
            "asset_class": "us_equity",
            "fractionable": bool(item.get("fractionable")),
            "in_frozen_u75": sym in control,
            "entry_allowed": False,
            "reason": (
                "Control U75 congelado"
                if sym in control else
                "Aguardando historico RAW/SIP, Corporate Actions, "
                "identidade, liquidez e validacao temporal"
            ),
        })
    if not rows:
        raise ValueError("Nenhum ativo elegivel identificado no catalogo")
    result = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    return result


def _previous_catalog() -> pd.DataFrame | None:
    files = sorted(OUT.glob("*/ativos_catalogo.csv"))
    if not files:
        return None
    prior = pd.read_csv(files[-1], dtype=str).fillna("")
    if not {"symbol", "alpaca_id"}.issubset(prior):
        raise ValueError("Catalogo anterior sem identidade Alpaca")
    return prior


def _new_ids(current: pd.DataFrame, previous: pd.DataFrame | None) -> pd.DataFrame:
    candidates = current.loc[~current["in_frozen_u75"]].copy()
    if previous is None:
        candidates["change"] = "catalogo_inicial"
    else:
        prev_map = dict(zip(previous["symbol"], previous["alpaca_id"]))
        candidates["change"] = [
            ("novo" if symbol not in prev_map else
             "identidade_alterada" if prev_map[symbol] != alpaca_id else
             "ja_conhecido")
            for symbol, alpaca_id in zip(candidates["symbol"], candidates["alpaca_id"])
        ]
    return candidates


def main() -> dict:
    # %% 1. Apenas GET metadados; nunca mercado/ordens.
    contract = carregar_contrato()
    frozen = set(contract["assets"])
    if len(frozen) != 75:
        raise RuntimeError("U75 nao corresponde ao universo historico")
    credential = load_alpaca_credentials(ROOT)
    response = requests.get(
        URL,
        headers=credential.headers,
        params={"status": "active", "asset_class": "us_equity"},
        timeout=45,
    )
    response.raise_for_status()
    raw = response.json()
    catalog = _assets_from_response(raw, frozen)
    previous = _previous_catalog()
    candidates = _new_ids(catalog, previous)
    # %% 2. Registrar catalogo integral antes de qualquer escolha economica.
    runid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = OUT / runid
    out.mkdir(parents=True, exist_ok=False)
    catalog.to_csv(out / "ativos_catalogo.csv", index=False)
    candidates.to_csv(out / "candidatos_futuros.csv", index=False)
    new = candidates[candidates["change"].isin(["novo", "identidade_alterada"])]
    new.to_csv(out / "novos_ativos_detectados.csv", index=False)
    digest = sha256((out / "ativos_catalogo.csv").read_bytes()).hexdigest()
    payload = {
        "status": "completed", "version": VERSION, "source": "Alpaca GET /v2/assets",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_sha256": digest,
        "frozen_universe_count": len(frozen),
        "active_tradable_us_equity": len(catalog),
        "outside_u75": len(candidates),
        "new_or_identity_changed_since_prior_catalog": len(new),
        "first_catalog": previous is None,
        "auto_promotion": False, "orders_executed": False,
        "warning": (
            "Catalogo de ativos disponiveis agora nao prova existencia "
            "historica em cada data nem elegibilidade financeira. "
            "Mudancas de identidade devem ser excluidas da analise sem bridge."
        ),
    }
    (out / "resultado.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[catalogo] {len(candidates)} ativos fora do U75; novos={len(new)}", flush=True)
    print(f"[arquivo] {out}", flush=True)
    return payload


if __name__ == "__main__":
    main()
