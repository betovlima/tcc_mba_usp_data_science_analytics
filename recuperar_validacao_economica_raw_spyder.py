"""TCC v1.25.1 — RECUPERAR relatório do backtest JÁ CONCLUÍDO, sem novo fit.

Execute F5 no Spyder. Lê apenas o resultado v1.25.0 que imprimiu
US$ 42,600,974.32, mas interrompeu a etapa de relatório por KeyError
em metrics["strategy_ending_capital"] (a chave correta é "ending_capital").

Não baixa mercado, não reprocessa indicadores, não calibra, não reexecuta
LightGBM, não altera RAW nem o pacote ZIP original.

Cria arquivo complementar e NOVA CÓPIA do pacote com manifesto auditável;
preserva o original, inclusive seu SHA-256.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import zipfile

import pandas as pd

from validar_backtest_economico_raw_spyder import (
    ROOT, VERSION, SCHEMA, CUTOFF, SNAPSHOT_SHA256,
    LEGACY_CAPITAL_SAME_CUTOFF, UPDATED_ALPACA_OLD_PIPELINE,
    _file_sha256, _snapshot_file_hashes, _verify_source, capital_do_resultado,
)

RUN_ID = "20261023T152638959386Z"
SOURCE_VERSION = "1.25.0"


def validar_resultado_salvo(
    folder: Path, snapshot_hash: str = SNAPSHOT_SHA256,
) -> tuple[dict, dict]:
    """Confere contrato, curvas e pacote ORIGINAL, sem recalcular estratégia."""
    output = Path(folder)
    json_path = output / "reproducao_universo.json"
    csv_path = output / "predictions.csv"
    trade_path = output / "trades.csv"
    package = output / "pacote_validacao_economica_raw.zip"
    for path in (json_path, csv_path, trade_path, package):
        if not path.is_file():
            raise FileNotFoundError(f"Artefato original ausente: {path}")

    result = json.loads(json_path.read_text(encoding="utf-8"))
    if result.get("reproduction_version") != SOURCE_VERSION:
        raise RuntimeError("Pasta não contém o replay original v1.25.0.")
    if result.get("execution_schema") != SCHEMA:
        raise RuntimeError("Schema da simulação diferente do registrado.")
    if result.get("analysis_end_date") != CUTOFF:
        raise RuntimeError("Cutoff não coincide com 06/10/2026.")
    if result.get("data_mode") != "alpaca_raw_unchanged_split_economic_in_memory":
        raise RuntimeError("Resultado não é o replay RAW econômico.")
    if result.get("status") != "completed":
        raise RuntimeError("Executor não marcou o replay como concluído.")
    if result["data"]["snapshot_sha256"] != snapshot_hash:
        raise RuntimeError("Snapshot do resultado difere do congelado.")
    if int(result["universe"]["effective_count"]) != 75:
        raise RuntimeError("Execução não usou todos os 75 ativos.")
    if result["universe"].get("excluded_assets"):
        raise RuntimeError("Há ativos excluídos no replay.")
    capital = capital_do_resultado(result)
    if not math.isfinite(capital) or capital < 0:
        raise RuntimeError("Capital salvo inválido.")
    if not result["metrics"].get("folds"):
        raise RuntimeError("Métricas de folds não foram registradas.")

    # A última linha da curva é patrimônio com liquidação (ou caixa
    # se a estratégia terminou sem posição).
    final_equity = pd.read_csv(csv_path, usecols=["strategy_equity"])[
        "strategy_equity"
    ].iloc[-1]
    if not math.isclose(float(final_equity), capital, abs_tol=0.01, rel_tol=0):
        raise RuntimeError("Capital final diverge da curva de patrimônio.")

    trades = pd.read_csv(trade_path)
    last_trade = trades.iloc[-1]
    if str(last_trade["action"]) not in ("SELL", "FINAL_SELL"):
        raise RuntimeError(
            "A última operação não encerrou uma posição; "
            "não é possível reconciliar o caixa final."
        )
    if not math.isclose(
        float(last_trade["cash_after_trade"]), capital,
        abs_tol=0.01, rel_tol=0,
    ):
        raise RuntimeError("Capital final diverge do caixa na última venda.")

    with zipfile.ZipFile(package, "r") as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Arquivo corrompido no ZIP original: {bad}")
    return result, {
        "capital": capital,
        "last_curve_equity": float(final_equity),
        "last_trade_cash": float(last_trade["cash_after_trade"]),
        "trades_total": int(len(trades)),
        "original_zip_sha256": _file_sha256(package),
        "source_json_sha256": _file_sha256(json_path),
        "curve_sha256": _file_sha256(csv_path),
        "trades_sha256": _file_sha256(trade_path),
    }


def recuperar(*, root: Path = ROOT, run_id: str = RUN_ID) -> dict:
    root = Path(root).resolve()
    if run_id != RUN_ID:
        raise RuntimeError("Somente o run_id já confirmado na conversa.")
    source, universe = _verify_source(root)
    folder = root / "output" / "validacao_economica_raw_75" / run_id
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Execução já concluída não encontrada: {folder}"
        )
    source_hashes_before = _snapshot_file_hashes(source, universe)
    result, checks = validar_resultado_salvo(folder)
    capital = checks["capital"]
    interpretation = {
        "report_version": VERSION,
        "original_execution_version": SOURCE_VERSION,
        "schema": SCHEMA,
        "run_id": run_id,
        "original_run_timestamp": run_id,
        "report_recovery_timestamp_utc":
            datetime.now(timezone.utc).isoformat(),
        "source_snapshot_sha256": SNAPSHOT_SHA256,
        "cutoff": CUTOFF,
        "assets": 75,
        "measured_capital_raw_economic": capital,
        "initial_capital": 10000.0,
        "capital_from_existing_files_only": True,
        "model_retrained": False,
        "backtest_reexecuted": False,
        "new_parameters_selected": False,
        "historical_legacy_capital_reference": LEGACY_CAPITAL_SAME_CUTOFF,
        "updated_alpaca_legacy_processing_capital_reference":
            UPDATED_ALPACA_OLD_PIPELINE,
        "delta_vs_updated_legacy": capital - UPDATED_ALPACA_OLD_PIPELINE,
        "pct_vs_updated_legacy": (
            (capital / UPDATED_ALPACA_OLD_PIPELINE - 1) * 100
        ),
        "delta_vs_original_vintage": capital - LEGACY_CAPITAL_SAME_CUTOFF,
        "metrics": result["metrics"],
        "evidence_verification": checks,
        "raw_and_actions_sha256": source_hashes_before,
        "split_count_in_policy_replay": (
            "not_in_summarized_metrics; inspect original full engine event "
            "ledger if independently available"
        ),
        "generalization_warning": (
            "Backtest retrospectivo, não validação prospectiva independente. "
            "A auditoria anterior encontrou sinais de overfitting. "
            "Não afirmar retorno real alcançável; sem market impact, "
            "slippage configurado 0 bps e frações de ações permitidas."
        ),
    }

    original_zip = folder / "pacote_validacao_economica_raw.zip"
    result_json = folder / "interpretacao_resultado_v1251.json"
    new_zip = folder / "pacote_validacao_economica_raw_v1251_finalizado.zip"
    if result_json.exists() or new_zip.exists():
        raise FileExistsError(
            "Relatório de recuperação já existe: não sobrescrever evidência."
        )
    source_hashes_after = _snapshot_file_hashes(source, universe)
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("Os 150 CSVs mudaram durante a recuperação.")
    interpretation["raw_and_actions_files_checked"] = len(source_hashes_after)
    result_json.write_text(
        json.dumps(interpretation, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    # Não sobrescrever o ZIP original: criar uma NOVA cópia derivada.
    shutil.copy2(original_zip, new_zip)
    with zipfile.ZipFile(new_zip, mode="a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(result_json, arcname=result_json.name)
    with zipfile.ZipFile(new_zip, "r") as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Falha no ZIP finalizado: {bad}")
    print(
        f"[recovery v{VERSION}] execução de {run_id} preservada, "
        f"capital=US$ {capital:,.2f}",
        flush=True,
    )
    print(f"[recovery v{VERSION}] ZIP ORIGINAL intacto: {original_zip}", flush=True)
    print(f"[recovery v{VERSION}] ZIP FINALIZADO: {new_zip}", flush=True)
    print(
        "[recovery] nenhum novo treinamento. Resultado histórico simulado, "
        "sem prova de ausência de overfitting.",
        flush=True,
    )
    return interpretation


if __name__ == "__main__":
    recuperar()
