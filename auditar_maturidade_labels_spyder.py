"""TCC v1.24.5 — audita maturidade temporal real dos labels de treino.

Usa os 75 ativos e os folds obtidos pelo motor atual, sobre a fotografia RAW.
Nao treina LightGBM, nao roda politica, nao modifica OHLCV ou Corporate Actions.
Checa, por ATIVO, a data efetiva do rótulo t+60, mesmo se o ativo tiver
sessões ausentes no calendário comum. As janelas de treino são as passadas
ao ajustador dos modelos; o rótulo é calculado com dados futuros do treino,
mas deve estar completo ANTES da primeira sessão avaliada.

F5 no Spyder -> output/maturidade_labels_u75/<ID>/pacote_maturidade_labels_u75.zip
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import zipfile

import numpy as np
import pandas as pd

from engine.configuracao import CONFIG
from engine.rotacao import ROTATION_FEATURES
from engine.modelo_lightgbm import _construir_contexto_execucao
from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.experimento import build_control_config
from reproducao.preparacao import load_raw_bar_file, prepare_model_frames
from reproducao.universo_fixo75 import universo_75


ROOT = Path(__file__).resolve().parent
VERSION = "1.24.5"
SNAPSHOT_END = "2026-10-22"
TARGET = "forward_risk_adjusted_utility"
EXECUTION_MODES = (
    ("calibration_model_fit", "train_end_index", "calibration_start_index"),
    ("final_model_fit", "final_fit_end_index", "test_start_index"),
)


def audit_maturity_for_phase(
    symbol: str,
    raw_sessions: pd.DatetimeIndex,
    eligible_samples: pd.DatetimeIndex,
    *,
    horizon: int,
    evaluation_start: pd.Timestamp,
    fold_id: int,
    phase: str,
) -> dict:
    """Valida maturidade t+h com calendário do ativo, não com dias corridos."""
    samples = pd.DatetimeIndex(eligible_samples)
    sessions = pd.DatetimeIndex(raw_sessions)
    if sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise RuntimeError(f"{symbol}: calendario RAW inconsistente.")
    if samples.has_duplicates:
        raise RuntimeError(f"{symbol}: amostra de treino duplicada.")
    if horizon < 1:
        raise ValueError("Horizonte invalido.")
    indices = sessions.get_indexer(samples)
    if (indices < 0).any():
        raise RuntimeError(f"{symbol}: treino com data sem barra RAW.")
    maturity_indices = indices + horizon
    if (maturity_indices >= len(sessions)).any():
        raise RuntimeError(f"{symbol}: treino com alvo ainda incompleto no snapshot.")
    maturity = sessions.take(maturity_indices)
    evaluation = pd.Timestamp(evaluation_start)
    if evaluation.tzinfo is None:
        evaluation = evaluation.tz_localize("UTC")
    else:
        evaluation = evaluation.tz_convert("UTC")
    if samples.tz is None:
        samples = samples.tz_localize("UTC")
    if (samples >= evaluation).any():
        raise RuntimeError(f"{symbol}: amostra de treino dentro da avaliacao.")
    late = maturity >= evaluation
    row = {
        "symbol": symbol,
        "fold_id": int(fold_id),
        "phase": phase,
        "max_horizon_sessions": int(horizon),
        "used_training_rows": int(len(samples)),
        "last_training_sample": (
            samples[-1].isoformat() if len(samples) else None
        ),
        "last_label_maturity": (
            maturity.max().isoformat() if len(samples) else None
        ),
        "evaluation_start": evaluation.isoformat(),
        "labels_not_mature_before_evaluation": int(np.sum(late)),
    }
    if np.any(late):
        failures = samples[late]
        row["first_late_sample"] = failures[0].isoformat()
    else:
        row["first_late_sample"] = None
    return row


def audit(*, root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    contract = json.loads(
        (root / "evidencias" / "manifest_universo_congelado.json").read_text(
            encoding="utf-8"
        )
    )
    assets = universo_75(contract, ())
    snapshot_root = root / "dados" / ".u75_fresh_runs"
    source = None
    snapshot = None
    for folder in sorted(
        snapshot_root.glob(f"asof_{SNAPSHOT_END}_*/universe_75"),
        reverse=True,
    ):
        paths = SnapshotPaths.from_root(folder)
        if paths.manifest.is_file():
            current = validate_snapshot(paths)
            if tuple(current.get("assets") or ()) == assets:
                source, snapshot = paths, current
                break
    if source is None or snapshot is None:
        raise FileNotFoundError(
            "Snapshot validado dos 75 ativos indisponivel; nao baixar."
        )

    prepared, excluded, details, input_audit = prepare_model_frames(
        source, assets=assets, csv_float_precision="round_trip",
        strict_raw=True, apply_split_adjustments=False,
    )
    if excluded or tuple(prepared) != assets or len(prepared) != 75:
        raise RuntimeError("Somente o universo fixo de 75 ativos e permitido.")
    config = build_control_config(prepared, CONFIG).copiar_modelo(
        update={"analysis_end_date": SNAPSHOT_END}
    )
    (
        aligned, calendar, calendar_symbol, symbols, folds,
        decision_dates, _, _,
    ) = _construir_contexto_execucao(prepared, config)
    horizon = max(map(int, config.rotation_target_horizons))
    if horizon != 60:
        raise RuntimeError(
            f"Esperados cinco horizontes terminando em 60; atual={horizon}."
        )
    if sorted(symbols) != sorted(assets):
        raise RuntimeError("Falha de identidade do universo.")

    # Calendarios especificos RAW sao indispensaveis para maturidade t+60
    # e nunca sao sobrescritos pela construcao de features.
    calendars = {
        symbol: load_raw_bar_file(
            source.raw_bars / f"{symbol}.csv",
            float_precision="round_trip", strict_raw=True,
        ).index for symbol in symbols
    }
    records = []
    for fold in folds:
        for phase, ending, start in EXECUTION_MODES:
            training_dates = calendar[:int(fold[ending])]
            assessment_start = pd.Timestamp(calendar[int(fold[start])])
            for symbol in symbols:
                actual_fit = aligned[symbol].loc[training_dates].dropna(
                    subset=[TARGET, *ROTATION_FEATURES]
                )
                item = audit_maturity_for_phase(
                    symbol, calendars[symbol], actual_fit.index,
                    horizon=horizon, evaluation_start=assessment_start,
                    fold_id=int(fold["fold_id"]), phase=phase,
                )
                records.append(item)
        # Garantir que a avaliacao de calibracao tenha finalizado seus
        # targets maximos antes do primeiro pregao de teste.
        last_calib_idx = int(fold["calibration_end_index"]) - 1
        test_idx = int(fold["test_start_index"])
        if last_calib_idx + horizon >= test_idx:
            raise RuntimeError(
                f"Fold {fold['fold_id']}: ultimo target da calibracao "
                "invade a janela de teste."
            )

    df = pd.DataFrame(records)
    violations = df.loc[df["labels_not_mature_before_evaluation"] > 0]
    summary = {
        "version": VERSION,
        "branch": "research/v1.23.1-fresh-snapshot-20261022",
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "assets": len(symbols),
        "folds": len(folds),
        "model_training_phases": list(x[0] for x in EXECUTION_MODES),
        "max_target_horizon_sessions": horizon,
        "configured_purge_sessions": int(config.rotation_purge_days),
        "audited_fit_groups": int(len(df)),
        "audited_training_rows_across_folds_and_phases": int(
            df["used_training_rows"].sum()
        ),
        "label_maturity_violations": int(
            df["labels_not_mature_before_evaluation"].sum()
        ),
        "groups_with_violations": len(violations),
        "using_strict_raw": True,
        "raw_price_adjustments": False,
        "download_or_retraining": False,
        "economic_backtest_performed": False,
        "scope_warning": (
            "Verifica datas dos labels t+60 elegiveis para fit e limites "
            "temporais das fases. Nao demonstra ausencia de vazamento em "
            "todos os atributos, diagnosticos de calibracao, uso da API "
            "ou a equivalencia de labels de utilidade economica completos."
        ),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = root / "output" / "maturidade_labels_u75" / stamp
    folder.mkdir(parents=True, exist_ok=False)
    df.to_csv(folder / "maturidade_por_ativo_fold.csv", index=False)
    (folder / "resumo.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    archive_path = folder / "pacote_maturidade_labels_u75.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(folder / "resumo.json", arcname="resumo.json")
        z.write(folder / "maturidade_por_ativo_fold.csv",
                arcname="maturidade_por_ativo_fold.csv")
    print(
        f"[maturity] grupos={len(df)}, erros={len(violations)}, "
        f"pacote={archive_path}", flush=True,
    )
    if len(violations):
        raise RuntimeError(
            "LABEL LEAKAGE: pelo menos um rótulo amadurece apenas "
            "apos o inicio do periodo avaliado; ver pacote."
        )
    return summary


if __name__ == "__main__":
    audit()
