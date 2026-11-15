"""Somente suporte para reproduzir campanhas antigas guardadas no repositorio.

O executor oficial nao importa este modulo. O conjunto da fonte historica
vem do manifesto congelado e nao de uma segunda estrategia configurada.
"""
from __future__ import annotations

import json
from pathlib import Path

from .configuracao import ASSETS

_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1] / "dados" / "u67" / "manifest.json"
)
_snapshot = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
HISTORICAL_SNAPSHOT_ASSETS = tuple(_snapshot["assets"])
HISTORICAL_SNAPSHOT_COUNT = len(HISTORICAL_SNAPSHOT_ASSETS)
NO_EXPECTED_EXCLUSIONS: frozenset[str] = frozenset()

if (
    len(HISTORICAL_SNAPSHOT_ASSETS) != len(set(HISTORICAL_SNAPSHOT_ASSETS))
    or not set(HISTORICAL_SNAPSHOT_ASSETS).issubset(ASSETS)
):
    raise RuntimeError("O manifesto historico nao pertence ao universo oficial.")
