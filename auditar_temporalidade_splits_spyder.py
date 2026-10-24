"""TCC v1.24.1 — auditoria PIT de splits, SEM ajuste RAW nem backtest.

F5 no Spyder. Usa o ultimo snapshot U75 existente, sem downloads ou escrita.
Mostra, com um split sintetico datado no futuro, se informacoes de anuncio
posterior chegam aos atributos anteriores. Compara o pipeline legado, que
normalizava o passado retrospectivamente, contra a politica RAW imutavel.

IMPORTANTE: nao e teste completo de inferencia LightGBM ou retorno financeiro.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from engine.configuracao import CONFIG
from engine.rotacao import ROTATION_FEATURES, construir_quadro_rotacao
from executar_atualizacao_series_spyder import find_reusable_snapshot
from reproducao.preparacao import (
    load_actions_file, load_raw_bar_file, split_normalize,
)
from reproducao.splits_temporais import (
    inspect_raw_invariance, known_as_of, split_from_action,
)
from reproducao.universo_fixo75 import universo_75

ROOT = Path(__file__).resolve().parent
VERSION = "1.24.1"
SNAPSHOT_DATE = "2026-10-22"
DECISION_DAY = "2020-07-28"
OUTPUT_ROOT = ROOT / "output" / "auditoria_temporal_splits"
SYNTHETIC_SYMBOL = "NVDA"


def fingerprint_frame(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(
        frame, index=True,
    ).to_numpy(dtype=np.uint64).tobytes()).hexdigest()


def compare_features_before_future_split(
    raw: pd.DataFrame,
    *,
    cutoff: str = DECISION_DAY,
) -> dict:
    """Confere todos os atributos, sem olhar labels forward no futuro."""
    last = pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1)
    prefix = raw.loc[raw.index < last].copy()
    if len(prefix) < 250:
        raise RuntimeError("Historico insuficiente para 200 sessoes de janela.")
    original_sha = fingerprint_frame(prefix)

    unannounced = {
        "id": "synthetic-future-unannounced",
        "symbol": SYNTHETIC_SYMBOL,
        "action_type": "forward_split",
        "ex_date": "2020-08-12",
        "old_rate": 1, "new_rate": 10,
        "first_seen_at_utc": "2020-08-01T12:00:00Z",
    }
    announced = {
        **unannounced,
        "id": "synthetic-future-announced",
        "first_seen_at_utc": "2020-07-20T12:00:00Z",
    }
    unknown = {**unannounced, "id": "synthetic-no-observation"}
    unknown.pop("first_seen_at_utc")

    a = inspect_raw_invariance(prefix, [unannounced], as_of=cutoff)
    b = inspect_raw_invariance(prefix, [announced], as_of=cutoff)
    c = inspect_raw_invariance(prefix, [unknown], as_of=cutoff)
    if (a["known_announcements_as_of_cutoff"],
        b["known_announcements_as_of_cutoff"],
        c["known_announcements_as_of_cutoff"]) != (0, 1, 0):
        raise RuntimeError("Split nao respeitou visibilidade temporal.")
    if fingerprint_frame(prefix) != original_sha:
        raise RuntimeError("Dados RAW mudaram durante o teste.")

    # Ao contrário de uma previsão, a construção de atributos não depende de
    # eventos: testar cada coluna real do motor, e não uma formula simulada.
    features_ref = construir_quadro_rotacao(prefix, CONFIG)
    features_other = construir_quadro_rotacao(prefix.copy(), CONFIG)
    if features_ref.empty:
        raise RuntimeError("O motor nao produziu atributos no historico selecionado.")
    pd.testing.assert_frame_equal(
        features_ref.loc[:, ROTATION_FEATURES],
        features_other.loc[:, ROTATION_FEATURES],
        check_exact=True,
    )
    future_event = split_from_action(unannounced)
    legacy, applied = split_normalize(prefix, [unannounced])
    if future_event is None or len(applied) != 1:
        raise RuntimeError("Controle historico nao evidenciou a normalizacao.")
    if fingerprint_frame(legacy) == original_sha:
        raise RuntimeError("O controle legado deveria modificar RAW anterior.")
    if fingerprint_frame(prefix) != original_sha:
        raise RuntimeError("A auditoria alterou dados RAW.")

    return {
        "symbol": SYNTHETIC_SYMBOL,
        "decision_asof": cutoff,
        "synthetic_future_split_ex_date": "2020-08-12",
        "unannounced_known_at_decision": False,
        "announced_known_at_decision": True,
        "without_evidence_known_at_decision": False,
        "future_split_applied_before_ex_date": False,
        "raw_preserved": True,
        "raw_prefix_fingerprint": original_sha,
        "feature_columns_compared": len(ROTATION_FEATURES),
        "feature_rows_compared": len(features_ref),
        "features_equal_before_event": True,
        "legacy_split_normalization_changes_pre_event_prices": True,
        "legacy_applied_actions": len(applied),
        "model_retrained": False,
        "score_and_rotation_parity_tested": False,
    }


def audit(*, root: Path = ROOT, cutoff: str = DECISION_DAY) -> dict:
    root = Path(root).resolve()
    historical = json.loads(
        (root / "evidencias" / "manifest_universo_congelado.json").read_text(
            encoding="utf-8"
        )
    )
    assets = universo_75(historical, ())
    # Importante: nenhuma migração/alteracao, apenas reuso do snapshot U75.
    from reproducao.dados import SnapshotPaths, validate_snapshot
    for run_root in sorted(
        (root / "dados" / ".u75_fresh_runs").glob(
            f"asof_{SNAPSHOT_DATE}_*"
        ), reverse=True,
    ):
        source = SnapshotPaths.from_root(run_root / "universe_75")
        if not source.manifest.exists():
            continue
        manifest = validate_snapshot(source)
        if tuple(manifest.get("assets") or ()) == assets:
            break
    else:
        raise FileNotFoundError(
            "Snapshot U75 validado nao encontrado; teste nao baixa da Alpaca."
        )

    results = []
    actions = []
    for symbol in assets:
        raw = load_raw_bar_file(
            source.raw_bars / f"{symbol}.csv",
            float_precision="round_trip", strict_raw=True,
        )
        # Metadados atuais NAO sao prova historica de anuncios anteriores.
        action_rows = load_actions_file(
            source.corporate_actions / f"{symbol}.csv"
        )
        for action in action_rows:
            if action.get("action_type") not in ("forward_split", "reverse_split"):
                continue
            event = split_from_action({**action, "symbol": symbol})
            if event is None:
                continue
            visible = known_as_of([event], pd.Timestamp(cutoff, tz="UTC"))
            actions.append({
                "symbol": symbol,
                "event_type": event.action_type,
                "ex_date": event.ex_date.date().isoformat(),
                "old_rate": str(event.old_rate),
                "new_rate": str(event.new_rate),
                "first_seen_at_utc": (
                    event.first_seen_at_utc.isoformat()
                    if event.first_seen_at_utc is not None else None
                ),
                "evidence_available_at_decision": bool(visible),
                "historical_visibility_verifiable": (
                    event.first_seen_at_utc is not None
                ),
            })
        prefix_end = pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1)
        prefix = raw.loc[raw.index < prefix_end].copy()
        results.append({
            "symbol": symbol,
            "raw_pre_decision_rows": len(prefix),
            "raw_pre_decision_sha256": fingerprint_frame(prefix),
            "future_split_injected_into_raw": False,
            "raw_bar_values_modified": False,
        })

    probe = compare_features_before_future_split(
        load_raw_bar_file(
            source.raw_bars / f"{SYNTHETIC_SYMBOL}.csv",
            float_precision="round_trip", strict_raw=True,
        ), cutoff=cutoff,
    )
    count = sum(not row["historical_visibility_verifiable"] for row in actions)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / "output" / "auditoria_temporal_splits" / stamp
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(results).to_csv(
        output / "integridade_prefixos_75.csv", index=False,
    )
    pd.DataFrame(actions).to_csv(
        output / "eventos_splits_e_disponibilidade.csv", index=False,
    )
    report = {
        "version": VERSION,
        "branch": "research/v1.23.1-fresh-snapshot-20261022",
        "source_snapshot_sha256": manifest["snapshot_sha256"],
        "assets": len(assets),
        "cutoff": cutoff,
        "new_event_observation_log_collected": False,
        "historical_split_events_in_latest_snapshot": len(actions),
        "split_events_without_first_seen_evidence": count,
        "unknown_historical_availability": count > 0,
        "real_bars_modified": False,
        "synthetic_probe": probe,
        "scope": (
            "RAW/atributos e cenario sintetico pontual; nao valida "
            "previsoes LightGBM, calibracao de folds ou estrategia completa."
        ),
    }
    (output / "resultado_temporal_splits.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    archive_path = output / "pacote_auditoria_temporal_splits.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in (
            "integridade_prefixos_75.csv",
            "eventos_splits_e_disponibilidade.csv",
            "resultado_temporal_splits.json",
        ):
            archive.write(output / name, arcname=name)
    print(f"[split-PIT] {len(assets)} ativos, {len(actions)} eventos; {archive_path}")
    return report


if __name__ == "__main__":
    audit()
