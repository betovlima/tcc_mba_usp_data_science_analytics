"""Contrato obrigatório da pesquisa Winner: universo exato de 77 ativos.

Não reduzir o universo automaticamente por disponibilidade histórica ou por
eventos societários. Falhas de cobertura precisam BLOQUEAR o experimento.
Não altera o Control histórico, cuja execução antiga documenta 75 elegíveis.
"""
from __future__ import annotations

import json
from pathlib import Path

from reproducao.dados import SnapshotPaths, validate_snapshot

UNIVERSO_WINNER_77 = (
    "NVDA",
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AMD",
    "JPM",
    "SPY",
    "AVGO",
    "NFLX",
    "CRM",
    "ORCL",
    "COST",
    "LLY",
    "XOM",
    "CAT",
    "WMT",
    "V",
    "HD",
    "ADC",
    "ADEA",
    "ADI",
    "ADM",
    "DDS",
    "CNQ",
    "VRTS",
    "XSD",
    "CXW",
    "ENS",
    "CORT",
    "CCK",
    "CEF",
    "GKOS",
    "RACE",
    "TX",
    "UNF",
    "BXMT",
    "PXLW",
    "KKR",
    "SCSC",
    "LKFT",
    "DNN",
    "VNCE",
    "UNFI",
    "DOC",
    "CLMT",
    "APD",
    "MGM",
    "MAN",
    "MYE",
    "YANG",
    "MKSI",
    "MCS",
    "ECC",
    "COLB",
    "AMS",
    "FOXF",
    "THO",
    "WDAY",
    "EXR",
    "XEL",
    "SBFG",
    "PAYX",
    "MUX",
    "SXC",
    "CRT",
    "NBR",
    "CAPR",
    "PNBK",
    "PENN",
    "JBL",
    "STAG",
    "MCHB",
    "LAD",
    "TOWN",
)
UNIVERSO_EXIGIDO = 77
DATA_FINAL_EXIGIDA = "2026-11-09"


def validar_contrato_77(assets) -> tuple[str, ...]:
    """Exige identidade e ordem exatas: sem corte silencioso para 75/65."""
    atual = tuple(str(x).strip().upper() for x in assets)
    if len(UNIVERSO_WINNER_77) != UNIVERSO_EXIGIDO:
        raise AssertionError("Contrato interno do universo não contém 77 ativos.")
    if len(set(UNIVERSO_WINNER_77)) != UNIVERSO_EXIGIDO:
        raise AssertionError("Contrato interno tem ativos duplicados.")
    if atual != UNIVERSO_WINNER_77:
        ausentes = sorted(set(UNIVERSO_WINNER_77) - set(atual))
        extras = sorted(set(atual) - set(UNIVERSO_WINNER_77))
        raise RuntimeError(
            "UNIVERSO_77_INCOMPLETO: exigidos 77 ativos, recebidos "
            + str(len(atual)) + ". Ausentes=" + str(ausentes)
            + "; extras=" + str(extras)
            + "; ordem divergente=" + str(
                not ausentes and not extras and atual != UNIVERSO_WINNER_77
            )
            + ". Não executar o estudo com 75/65."
        )
    return atual


def validar_fonte_77(project_root: Path) -> dict:
    """Usa somente o snapshot dedicado, validado por hashes, sem fallback U67.

    Dados corporativos e continuidade de identidade precisam ser examinados
    antes de transformar a fonte em preços negociáveis ou rótulos de ML.
    """
    path = SnapshotPaths.from_root(Path(project_root) / "dados" / "winner_77")
    if not path.manifest.is_file():
        raise RuntimeError(
            "FONTE_77_AUSENTE: não há snapshot auditável em "
            + str(path.root)
            + ". A base U67 não pode substituir o universo de 77."
        )
    manifest = json.loads(path.manifest.read_text(encoding="utf-8"))
    validar_contrato_77(manifest.get("assets") or [])
    bars = manifest.get("bars") or {}
    if bars.get("bar_snapshot_as_of_end") != DATA_FINAL_EXIGIDA:
        raise RuntimeError(
            "FONTE_77_DESATUALIZADA: data final "
            + str(bars.get("bar_snapshot_as_of_end"))
            + ", exigida " + DATA_FINAL_EXIGIDA
        )
    if bars.get("adjustment") != "raw" or bars.get("timeframe") != "1Day":
        raise RuntimeError("FONTE_77_INVALIDA: exige OHLCV raw / 1Day.")
    ausentes = [
        ticker for ticker in UNIVERSO_WINNER_77
        if not (path.raw_bars / (ticker + ".csv")).is_file()
        or not (path.corporate_actions / (ticker + ".csv")).is_file()
    ]
    if ausentes:
        raise RuntimeError(
            "FONTE_77_INCOMPLETA: OHLCV/eventos não disponíveis para "
            + ", ".join(ausentes)
        )
    validate_snapshot(path)  # SHA-256 de todos os arquivos congelados
    return {
        "status": "snapshot_77_hashes_confirmados",
        "assets": list(UNIVERSO_WINNER_77),
        "count": UNIVERSO_EXIGIDO,
        "end_date": DATA_FINAL_EXIGIDA,
        "path": str(path.root),
        "caution": (
            "Validar mudanças societárias de DOC e CLMT antes de construir "
            "features contínuas. Nunca excluir ativos silenciosamente."
        ),
    }
