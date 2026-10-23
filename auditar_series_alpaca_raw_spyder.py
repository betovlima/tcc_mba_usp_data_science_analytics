"""TCC v1.24.0 — coleta/validacao Alpaca SIP RAW sem alteracoes de OHLCV.

F5 no Spyder. Baixa apenas os 75 ativos efetivos em NOVO snapshot isolado.
Os dados de precos/volumes sao preservados; conversao de tipos somente.
Nao remove duplicatas, nao corrige valores invalidos, nao ajusta splits ou
dividendos, nao treina modelos e nao executa operacoes ficticias.

Nota: backtest economico sobre precos RAW exige contabilizar as quantidades
de acoes que mudam com splits SEM alterar a serie. Ate essa etapa ser
implementada e testada, um replay RAW puro seria financeiramente incorreto.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from executar_atualizacao_series_spyder import (
    assert_completed_session, verify_reference_session,
)
from reproducao.dados import (
    SnapshotPaths, build_snapshot_manifest, download_corporate_actions,
    download_raw_bars, load_alpaca_credentials, validate_snapshot,
)
from reproducao.preparacao import (
    load_actions_file, load_raw_bar_file, prepare_model_frames,
)
from reproducao.universo_fixo75 import universo_75


ROOT = Path(__file__).resolve().parent
CUTOFF = "2026-10-22"
VERSION = "1.24.0"
DATA_ROOT = ROOT / "dados" / ".u75_strict_raw_runs"
OUTPUT_ROOT = ROOT / "output" / "auditoria_raw_sem_ajustes"


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify_unaltered_model_input(
    paths: SnapshotPaths, assets: tuple[str, ...],
) -> tuple[list[dict], dict]:
    """Audita igualdade bit a bit de valores float64 lidos e preparados."""
    prepared, exclusions, diagnostics, audit = prepare_model_frames(
        paths, assets=assets, csv_float_precision="round_trip",
        apply_split_adjustments=False, strict_raw=True,
    )
    if exclusions or tuple(prepared) != assets or len(prepared) != 75:
        raise RuntimeError("Painel RAW nao preservou exatamente os 75 ativos.")
    if audit["actual_total_splits_applied"] != 0:
        raise RuntimeError("Ajustes de splits nao autorizados detectados.")

    evidence = []
    for asset in assets:
        raw_path = paths.raw_bars / f"{asset}.csv"
        actions_path = paths.corporate_actions / f"{asset}.csv"
        source = load_raw_bar_file(
            raw_path, float_precision="round_trip", strict_raw=True,
        )
        model_input = prepared[asset]
        if not source.index.equals(model_input.index):
            raise RuntimeError(f"{asset}: indice do RAW foi modificado.")
        for field in ("open", "high", "low", "close", "volume"):
            raw_values = source[field].to_numpy(dtype=np.float64)
            model_values = model_input[field].to_numpy(dtype=np.float64)
            if not np.array_equal(raw_values, model_values):
                raise RuntimeError(
                    f"{asset}: valores RAW de {field} foram alterados."
                )
        actions = load_actions_file(actions_path)
        splits = sum(
            item.get("action_type") in ("forward_split", "reverse_split")
            for item in actions
        )
        evidence.append({
            "asset": asset,
            "bars": int(len(source)),
            "raw_sha256": _sha256(raw_path),
            "corporate_actions_sha256": _sha256(actions_path),
            "raw_ohlcv_equals_prepared": True,
            "split_events_recorded_only": splits,
            "cash_dividends_recorded_only": sum(
                item.get("action_type") == "cash_dividend"
                for item in actions
            ),
        })
    return evidence, audit


def run(*, root: Path = ROOT, cutoff: str = CUTOFF) -> dict:
    root = Path(root).resolve()
    assert_completed_session(cutoff)
    historical = json.loads(
        (root / "evidencias" / "manifest_universo_congelado.json").read_text(
            encoding="utf-8"
        )
    )
    assets = universo_75(historical, ())
    if len(assets) != 75 or {"DOC", "CLMT"} & set(assets):
        raise RuntimeError("Somente 75 ativos efetivos, sem DOC nem CLMT.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    data_root = root / "dados" / ".u75_strict_raw_runs" / f"asof_{cutoff}_{stamp}"
    output_root = root / "output" / "auditoria_raw_sem_ajustes" / stamp
    if data_root.exists() or output_root.exists():
        raise FileExistsError("A coleta RAW nao pode sobrescrever snapshots.")
    paths = SnapshotPaths.from_root(data_root / "universe_75")
    credentials = load_alpaca_credentials(root)
    print(
        f"[strict-raw] versao={VERSION}, 75 ativos, feed=SIP, "
        "adjustment=RAW; sem filtros, exclusoes de linhas ou ajustes",
        flush=True,
    )
    bars = download_raw_bars(
        credentials, paths, assets=assets,
        bar_snapshot_as_of_end=cutoff, analysis_end_date=cutoff,
        sip_delay_guard_minutes=20, strict_raw=True,
    )
    actions = download_corporate_actions(
        credentials, paths, assets=assets, query_end=cutoff,
    )
    manifest = build_snapshot_manifest(
        paths, bars, actions,
        bar_snapshot_as_of_end=cutoff, analysis_end_date=cutoff,
        assets=assets, snapshot_name=f"strict-unchanged-75-{cutoff}",
    )
    validated = validate_snapshot(paths)
    if tuple(validated.get("assets") or ()) != assets:
        raise RuntimeError("Snapshot nao contem os exatos 75 ativos.")
    if manifest["snapshot_sha256"] != validated["snapshot_sha256"]:
        raise RuntimeError("Identidade do snapshot alterada.")
    verify_reference_session(paths, cutoff)

    checks, audit = verify_unaltered_model_input(paths, assets)
    output_root.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(checks).to_csv(
        output_root / "integridade_75_ativos.csv", index=False,
    )
    report = {
        "version": VERSION,
        "branch": "research/v1.23.1-fresh-snapshot-20261022",
        "source": "alpaca", "feed": "sip", "adjustment": "raw",
        "timeframe": "1Day", "asof_date": cutoff,
        "assets_requested": 75, "assets_effective": 75,
        "snapshot_sha256": validated["snapshot_sha256"],
        "raw_csv_ohlcv_values_modified": False,
        "rows_dropped_or_duplicate_repaired": False,
        "split_price_adjustments_applied": 0,
        "dividend_price_adjustments_applied": 0,
        "dtype_conversion_only": True,
        "corporate_actions_kept_as_metadata": True,
        "model_input_equals_raw_in_all_assets": all(
            row["raw_ohlcv_equals_prepared"] for row in checks
        ),
        "data_audit": audit,
        "economic_backtest_performed": False,
        "backtest_block_reason": (
            "Split em serie RAW muda quantidade de acoes, nao preco. "
            "O motor de treino e simulacao legado assume continuidade "
            "do preco. Sem contabilidade de quantidade por split e "
            "sem teste de retornos ajustados por eventos, resultado "
            "economico com serie RAW nao deve ser reportado."
        ),
    }
    (output_root / "relatorio_raw.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    archive_path = output_root / "pacote_auditoria_raw_sem_ajustes.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(
            output_root / "relatorio_raw.json", arcname="relatorio_raw.json",
        )
        archive.write(
            output_root / "integridade_75_ativos.csv",
            arcname="integridade_75_ativos.csv",
        )
        archive.write(
            paths.manifest, arcname="manifest_75_strict_raw.json",
        )
    print(
        f"[strict-raw] 75/75 OHLCV preservados; splits_aplicados=0 "
        f"arquivo={archive_path}",
        flush=True,
    )
    return report


if __name__ == "__main__":
    run()
