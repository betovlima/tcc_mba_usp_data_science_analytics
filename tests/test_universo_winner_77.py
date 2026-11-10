"""A pesquisa atual não pode recair silenciosamente para 65/75 ativos."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from reproducao.universo_winner_77 import (
    UNIVERSO_EXIGIDO,
    UNIVERSO_WINNER_77,
    validar_contrato_77,
    validar_fonte_77,
)


def test_lista_exata_77_sem_duplicacao_e_com_todos_os_ativos():
    assert UNIVERSO_EXIGIDO == 77
    assert len(UNIVERSO_WINNER_77) == 77
    assert len(set(UNIVERSO_WINNER_77)) == 77
    assert "CAPR" in UNIVERSO_WINNER_77
    assert "DOC" in UNIVERSO_WINNER_77
    assert "CLMT" in UNIVERSO_WINNER_77
    assert "TOWN" in UNIVERSO_WINNER_77
    assert validar_contrato_77(UNIVERSO_WINNER_77) == UNIVERSO_WINNER_77


@pytest.mark.parametrize("quantity", [65, 67, 75, 76])
def test_nunca_permitir_execucao_com_universo_reduzido(quantity):
    with pytest.raises(RuntimeError, match="UNIVERSO_77_INCOMPLETO"):
        validar_contrato_77(UNIVERSO_WINNER_77[:quantity])


def test_nunca_aceitar_mesmos_77_em_ordem_divergente():
    with pytest.raises(RuntimeError, match="ordem divergente=True"):
        validar_contrato_77(tuple(reversed(UNIVERSO_WINNER_77)))


def test_nao_usar_snapshot_antigo_como_fallback(tmp_path: Path):
    (tmp_path / "dados" / "u67").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="FONTE_77_AUSENTE"):
        validar_fonte_77(tmp_path)


def test_manifesto_antigo_de_75_e_bloqueado(tmp_path: Path):
    root = tmp_path / "dados" / "winner_77"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({
            "assets": UNIVERSO_WINNER_77[:75],
            "bars": {"bar_snapshot_as_of_end": "2026-11-09"},
        }),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="UNIVERSO_77_INCOMPLETO"):
        validar_fonte_77(tmp_path)


def test_manifesto_77_com_data_antiga_e_bloqueado(tmp_path: Path):
    root = tmp_path / "dados" / "winner_77"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({
            "assets": UNIVERSO_WINNER_77,
            "bars": {
                "bar_snapshot_as_of_end": "2026-10-06",
                "adjustment": "raw", "timeframe": "1Day",
            },
        }),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="FONTE_77_DESATUALIZADA"):
        validar_fonte_77(tmp_path)


def test_nao_considerar_77_validos_sem_77_ohlcv_e_eventos(tmp_path: Path):
    root = tmp_path / "dados" / "winner_77"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({
            "assets": UNIVERSO_WINNER_77,
            "bars": {
                "bar_snapshot_as_of_end": "2026-11-09",
                "adjustment": "raw", "timeframe": "1Day",
            },
        }),
        encoding="utf-8",
    )
    with patch("reproducao.universo_winner_77.validate_snapshot") as check:
        with pytest.raises(RuntimeError, match="FONTE_77_INCOMPLETA"):
            validar_fonte_77(tmp_path)
        check.assert_not_called()
