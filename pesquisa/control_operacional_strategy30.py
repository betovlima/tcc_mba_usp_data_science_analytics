"""Control de pesquisa importado da Strategy #30 operacional.

Valida a evidencia historica congelada antes de aceitar qualquer comparacao.
Este modulo NAO simula nem retreina o motor de trading original. O pacote
de resultados nao contem OHLCV RAW de todos os ativos para reproducao.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = ROOT / "pesquisa" / "control_oficial_strategy30.json"
UNIVERSE_PATH = ROOT / "pesquisa" / "universo_control_75.json"


def carregar_contrato(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    assets = list(contract["assets"])
    expected_count = int(contract["effective_asset_count"])
    if expected_count < 1 or len(assets) != expected_count or assets != sorted(set(assets)):
        raise RuntimeError("Quantidade de ativos inconsistente com o contrato.")
    # Uma unica definicao canonica. Alterar o universo invalida o Control.
    official = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    if assets != list(official["assets"]):
        raise RuntimeError("Contrato Control diverge do universo canonico da referencia.")
    if contract["role"] != "official_research_control":
        raise RuntimeError("Pacote nao esta identificado como Control oficial de pesquisa.")
    return contract


def validar_control_strategy30(
    arquivo_zip: Path,
    *,
    contrato: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Confere identidade, periodo, ativos, patrimonio, operacoes e hashes."""
    arquivo_zip = Path(arquivo_zip)
    contract = dict(contrato) if contrato is not None else carregar_contrato()
    if not arquivo_zip.is_file():
        raise FileNotFoundError(f"Pacote Strategy #30 indisponivel: {arquivo_zip}")
    raw = arquivo_zip.read_bytes()
    digest = sha256(raw).hexdigest()
    if digest != contract["source_file_sha256"]:
        raise RuntimeError(
            "Hash do ZIP divergente. Outro backtest nao pode substituir o Control."
        )
    with ZipFile(BytesIO(raw), "r") as archive:
        if archive.testzip() is not None:
            raise RuntimeError("O ZIP de origem apresenta arquivo corrompido.")
        for name, expected_hash in contract["required_archive_entries"].items():
            if name not in archive.namelist():
                raise RuntimeError(f"Evidencia obrigatoria ausente: {name}")
            if sha256(archive.read(name)).hexdigest() != expected_hash:
                raise RuntimeError(f"Evidencia alterada: {name}")

        strategy = json.loads(archive.read("strategy_manifest.json"))
        experiment = json.loads(archive.read("experiment_manifest.json"))
        files = contract["required_archive_entries"]
        metrics_name = next(x for x in files if x.endswith("_metrics.json"))
        predictions_name = next(x for x in files if x.endswith("_predictions.csv"))
        trades_name = next(x for x in files if x.endswith("_trades.csv"))
        metrics = json.loads(archive.read(metrics_name))
        predictions = pd.read_csv(BytesIO(archive.read(predictions_name)))
        trades = pd.read_csv(BytesIO(archive.read(trades_name)))

    if strategy.get("status") != "completed" or experiment.get("job_status") != "completed":
        raise RuntimeError("O resultado do backtest nao foi concluido.")
    if strategy["strategy_profile_id"] != contract["strategy_profile_id"]:
        raise RuntimeError("Perfil de Strategy #30 divergente.")
    if strategy["strategy_configuration_hash"] != contract["strategy_configuration_hash"]:
        raise RuntimeError("Configuracao da Strategy #30 divergente.")
    if metrics["strategy_configuration_sha256"] != contract["strategy_configuration_sha256"]:
        raise RuntimeError("Configuracao efetiva do motor divergente.")
    if metrics["market_data_signature_sha256"] != contract["market_data_signature_sha256"]:
        raise RuntimeError("Assinatura dos dados de mercado divergente.")
    if experiment["eligible_asset_count"] != int(contract["effective_asset_count"]) or sorted(metrics["assets"]) != contract["assets"]:
        raise RuntimeError("Universo do Control nao corresponde aos ativos declarados no manifesto.")
    if len(predictions) != contract["expected_rows"]:
        raise RuntimeError("Total de sessoes do Control divergente.")
    dates = pd.to_datetime(predictions["timestamp"], utc=True)
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise RuntimeError("Calendario do Control apresenta duplicacao ou desordem.")
    if (
        dates.iloc[0].strftime("%Y-%m-%d") != contract["first_session"]
        or dates.iloc[-1].strftime("%Y-%m-%d") != contract["last_session"]
    ):
        raise RuntimeError("Periodo do Control divergente.")
    counts = predictions["walk_forward_fold"].value_counts().sort_index()
    observed_counts = {str(int(k)): int(v) for k, v in counts.items()}
    if observed_counts != contract["expected_fold_counts"]:
        raise RuntimeError("Folds do Control divergentes.")
    if predictions["selected_asset"].isna().any() or not set(
        predictions["selected_asset"]
    ).issubset(set(contract["assets"])):
        raise RuntimeError("Ativos selecionados fora do universo oficial.")
    curve = pd.to_numeric(predictions["strategy_equity"], errors="coerce")
    if not np.isfinite(curve).all() or bool((curve <= 0).any()):
        raise RuntimeError("Patrimonio invalido no arquivo de decisoes.")
    expected = float(contract["ending_capital"])
    if not np.isclose(float(curve.iloc[-1]), expected, rtol=0, atol=0.01):
        raise RuntimeError("Capital final do CSV diverge do Control oficial.")
    if not np.isclose(float(metrics["strategy_ending_capital"]), expected, rtol=0, atol=0.01):
        raise RuntimeError("Capital final do resumo diverge do Control oficial.")
    for name, metric_key in (
        ("initial_capital", "initial_capital"),
        ("sharpe", "strategy_sharpe"),
        ("max_drawdown", "strategy_maximum_drawdown"),
        ("total_transaction_fees", "total_transaction_fees"),
    ):
        if not np.isclose(float(metrics[metric_key]), float(contract[name]), rtol=0, atol=1e-8):
            raise RuntimeError(f"Indicador financeiro divergente: {name}")
    for name, key in (("cash_days", "cash_days"), ("capital_rotations", "capital_rotations")):
        if int(metrics[key]) != int(contract[name]):
            raise RuntimeError(f"Contagem operacional divergente: {name}")
    if int(metrics["cash_days"]) != 0:
        raise RuntimeError("O Control registrado deve manter a exposicao observada.")
    if len(trades) == 0:
        raise RuntimeError("Operacoes do Control ausentes.")

    audit = {
        "status": "validated",
        "role": "official_research_control",
        "id": contract["id"],
        "source_zip_sha256": digest,
        "strategy_profile_id": contract["strategy_profile_id"],
        "strategy_configuration_hash": contract["strategy_configuration_hash"],
        "market_data_signature_sha256": contract["market_data_signature_sha256"],
        "effective_assets": contract["assets"],
        "effective_asset_count": int(contract["effective_asset_count"]),
        "first_session": contract["first_session"],
        "last_session": contract["last_session"],
        "sessions": len(predictions),
        "fold_counts": observed_counts,
        "initial_capital": float(contract["initial_capital"]),
        "ending_capital": float(curve.iloc[-1]),
        "sharpe": float(metrics["strategy_sharpe"]),
        "max_drawdown": float(metrics["strategy_maximum_drawdown"]),
        "cash_days": int(metrics["cash_days"]),
        "capital_rotations": int(metrics["capital_rotations"]),
        "total_transaction_fees": float(metrics["total_transaction_fees"]),
        "calculation_type": "frozen_operational_backtest_import",
        "model_retrained_here": False,
        "market_data_raw_in_zip": False,
        "warning": (
            "A curva e as decisoes foram importadas e verificadas; "
            "nao se trata de reproducao independente das previsoes. "
            "O estudo CASH requer dados RAW completos e politica de rotacao "
            "equivalente para o mesmo periodo antes de comparar retornos."
        ),
    }
    return audit, predictions, trades
