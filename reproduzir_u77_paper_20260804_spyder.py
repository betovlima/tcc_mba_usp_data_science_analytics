"""Replay historico contrafactual U67+10, exclusivo do TCC, sem broker/MongoDB.

Uma carteira nova, CASH e US$ 10.000 em 04/08/2026, ate 16/10/2026.
Treina no passado via walk-forward e compra/vende na abertura da sessao seguinte.
Atualiza integralmente barras RAW/SIP e corporate actions da Alpaca por execucao.
Nao altera o snapshot oficial U67 nem utiliza artefatos da conta Paper.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys
import uuid
import zipfile

import pandas as pd

from engine.configuracao import CONFIG, U67_REQUESTED_ASSETS, construir_configuracao_controle
from engine.execucao import aplicar_deslizamento, calcular_taxas_referencia
from engine.modelo_lightgbm import (
    _ajustar_modelos_lightgbm,
    _construir_contexto_execucao,
    selecionar_switch_margin,
)
from engine.rotacao import (
    _crescimento_politica_simples,
    _politica_agendada,
    _politica_utilidade,
    _precalcular_utilidades_modelo,
    _simular_exato,
    preparar_painel_rotacao,
)
from reproducao.dados import (
    SnapshotPaths,
    build_snapshot_manifest,
    download_corporate_actions,
    download_raw_bars,
    load_alpaca_credentials,
    validate_snapshot,
)
from reproducao.preparacao import prepare_model_frames

ROOT = Path(__file__).resolve().parent
VERSION = "u77-paper-replay-v1.0.0"
EXECUTION_START = "2026-08-04"
EXECUTION_END = "2026-10-16"
HISTORY_START = "2016-01-01"
INITIAL_CAPITAL = 10_000.0

CHALLENGER_ADDITIONS = (
    "CRT", "NBR", "CAPR", "PNBK", "PENN", "JBL", "STAG", "MCHB", "LAD", "TOWN"
)
# Exclusoes estruturais da pesquisa original. Nunca reparar/ligar historicos.
STRUCTURAL_EXCLUSIONS = frozenset({"DOC", "CLMT"})
REQUESTED_ASSETS = (*U67_REQUESTED_ASSETS, *CHALLENGER_ADDITIONS)
EFFECTIVE_ASSETS = tuple(s for s in REQUESTED_ASSETS if s not in STRUCTURAL_EXCLUSIONS)
REFERENCE_ASSETS = tuple(s for s in U67_REQUESTED_ASSETS if s not in STRUCTURAL_EXCLUSIONS)


def validate_contract() -> None:
    if len(REQUESTED_ASSETS) != 77 or len(set(REQUESTED_ASSETS)) != 77:
        raise ValueError("U77 solicitado deve conter exatamente 77 simbolos unicos.")
    if len(EFFECTIVE_ASSETS) != 75 or len(REFERENCE_ASSETS) != 65:
        raise ValueError("Contrato U77=75 efetivos e U67=65 efetivos violado.")
    if EFFECTIVE_ASSETS[-10:] != CHALLENGER_ADDITIONS:
        raise ValueError("Os dez ativos do Challenger foram modificados.")
    if EXECUTION_START >= EXECUTION_END:
        raise ValueError("Intervalo de execucao invalido.")


def make_config():
    """Mesmo contrato Control, somente janela/caixa e dados atualizados."""
    return construir_configuracao_controle(
        CONFIG, assets=EFFECTIVE_ASSETS
    ).copiar_modelo(update={
        "analysis_start_date": EXECUTION_START,
        "analysis_end_date": EXECUTION_END,
        "initial_capital": INITIAL_CAPITAL,
    })


def _iso_date(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


def assert_window(decision_dates: pd.DatetimeIndex) -> None:
    """A primeira compra e em 04/08; a decisao inicial usa a sessao anterior."""
    if len(decision_dates) < 2:
        raise RuntimeError("Nao ha sessoes de execucao.")
    first, last = _iso_date(decision_dates[1]), _iso_date(decision_dates[-1])
    if (first, last) != (EXECUTION_START, EXECUTION_END):
        raise RuntimeError(
            "Janela nao corresponde ao contrato: "
            f"primeira execucao={first}, ultima={last}"
        )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (pd.Timedelta,)):
        return str(value)
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and (pd.isna(value) or not (-float("inf") < value < float("inf"))):
        return None
    return value


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> Path:
    validate_contract()
    if datetime.now(timezone.utc).date().isoformat() < EXECUTION_END:
        raise RuntimeError("A janela termina no futuro.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    destination = ROOT / "output" / "u77_paper_replay" / run_id
    snapshot = SnapshotPaths.from_root(ROOT / "dados" / ".u77_paper_replay" / run_id)
    # Criar snapshot novo por execucao, nunca reciclar series antigas.
    destination.mkdir(parents=True, exist_ok=False)
    snapshot.ensure()
    print(f"[start] version={VERSION} run_id={run_id} start={EXECUTION_START} end={EXECUTION_END}", flush=True)
    print(f"[contract] requested=77 effective=75 excluded={sorted(STRUCTURAL_EXCLUSIONS)}", flush=True)

    credentials = load_alpaca_credentials(ROOT)
    raw = download_raw_bars(
        credentials, snapshot, assets=EFFECTIVE_ASSETS,
        bar_snapshot_as_of_end=EXECUTION_END, analysis_end_date=EXECUTION_END,
    )
    actions = download_corporate_actions(
        credentials, snapshot, assets=EFFECTIVE_ASSETS, query_end=EXECUTION_END,
    )
    build_snapshot_manifest(
        snapshot, raw, actions, bar_snapshot_as_of_end=EXECUTION_END,
        analysis_end_date=EXECUTION_END, assets=EFFECTIVE_ASSETS,
        snapshot_name="u77_contrafactual_20260804_20261016",
    )
    manifest = validate_snapshot(snapshot)

    raw_frames, unexpected_exclusions, data_diagnostics, data_audit = prepare_model_frames(
        snapshot, assets=EFFECTIVE_ASSETS, csv_float_precision="round_trip"
    )
    if unexpected_exclusions or set(raw_frames) != set(EFFECTIVE_ASSETS):
        raise RuntimeError(
            "Universo divergente, sem reparos artificiais: "
            f"excluidos={unexpected_exclusions} "
            f"faltantes={sorted(set(EFFECTIVE_ASSETS) - set(raw_frames))}"
        )

    config = make_config()
    # O calendario base e o mesmo U67 (65 efetivos) do protocolo Challenger.
    _, reference_calendar, source = preparar_painel_rotacao(
        {s: raw_frames[s] for s in REFERENCE_ASSETS},
        construir_configuracao_controle(config, assets=REFERENCE_ASSETS),
    )
    frames, calendar, calendar_source, symbols, folds, dates, date_to_fold, metadata = (
        _construir_contexto_execucao(
            raw_frames, config, calendar_override=reference_calendar,
            calendar_source_label=f"U67_FIXED:{source}"
        )
    )
    assert_window(dates)
    relevant_folds = {int(date_to_fold[pd.Timestamp(d)]) for d in dates[:-1]}
    print(f"[calendar] source={calendar_source} sessions={len(dates)-1} active_folds={sorted(relevant_folds)}", flush=True)

    policies = {}
    calibration_rows = []
    decision_diagnostics = {}
    for fold in folds:
        fold_id = int(fold["fold_id"])
        if fold_id not in relevant_folds:
            continue
        train_dates = calendar[:int(fold["train_end_index"])]
        calib_dates = calendar[int(fold["calibration_start_index"]):int(fold["calibration_end_index"])]
        final_dates = calendar[:int(fold["final_fit_end_index"])]
        fold_decisions = pd.DatetimeIndex(fold["decision_dates"])
        print(f"[fold] {fold_id} calibration training: {len(symbols)} models", flush=True)
        calib_models = _ajustar_modelos_lightgbm(
            frames, symbols, train_dates, config, phase=f"u77_paper_fold_{fold_id}_calibration"
        )
        calib_cache, _ = _precalcular_utilidades_modelo(
            calib_models, frames, symbols, calib_dates, config
        )
        candidates = []
        for margin in config.rotation_switch_margin_candidates:
            policy = _politica_utilidade(
                {}, frames, symbols, config, float(margin), utility_cache=calib_cache
            )
            score = _crescimento_politica_simples(
                policy, frames, symbols, calib_dates, config
            )
            candidates.append((float(margin), float(score)))
        selected = selecionar_switch_margin(candidates)
        selected_margin = float(selected["selected_candidate_margin"])
        effective_margin = max(float(config.rotation_switch_margin), selected_margin)
        for margin, score in candidates:
            calibration_rows.append({
                "fold_id": fold_id, "candidate_margin": margin,
                "calibration_score": score, "selected": margin == selected_margin,
            })
        del calib_models, calib_cache

        print(f"[fold] {fold_id} final training: {len(symbols)} models", flush=True)
        final_models = _ajustar_modelos_lightgbm(
            frames, symbols, final_dates, config, phase=f"u77_paper_fold_{fold_id}_final"
        )
        fold_cache, _ = _precalcular_utilidades_modelo(
            final_models, frames, symbols, fold_decisions, config
        )
        policies[fold_id] = _politica_utilidade(
            {}, frames, symbols, config, effective_margin,
            fold_id=fold_id, calibrated_switch_margin=selected_margin,
            utility_cache=fold_cache, decision_diagnostics=decision_diagnostics,
        )
        del final_models
        print(f"[fold] {fold_id} margin={effective_margin:.6f} ready", flush=True)

    result = _simular_exato(
        "tcc_u77_paper_replay",
        _politica_agendada(policies, date_to_fold),
        frames, symbols, dates, config, calcular_taxas_referencia, aplicar_deslizamento,
        decision_metadata=metadata, policy_decision_diagnostics=decision_diagnostics,
        model_label="TCC U67+10 fixed Challenger / CASH restart",
        method_line="Frozen U77 list; fresh Alpaca RAW/SIP; walk-forward Control; no orders.",
    )
    first_date = _iso_date(result.predictions.index[0])
    last_date = _iso_date(result.predictions.index[-1])
    if (first_date, last_date) != (EXECUTION_START, EXECUTION_END):
        raise RuntimeError(f"Intervalo de resultados incorreto: {first_date}..{last_date}")
    if float(result.metrics["initial_capital"]) != INITIAL_CAPITAL:
        raise RuntimeError("Capital inicial divergente.")
    trades = result.trades.copy()
    predictions = result.predictions.reset_index()
    trades.to_csv(destination / "operacoes.csv", index=False, float_format="%.17g")
    predictions.to_csv(destination / "decisoes_e_patrimonio.csv", index=False, float_format="%.17g")
    pd.DataFrame(calibration_rows).to_csv(destination / "margens_calibracao.csv", index=False)
    pd.DataFrame(data_diagnostics).to_csv(destination / "auditoria_dados.csv", index=False)
    (destination / "resumo_motor.txt").write_text(result.summary, encoding="utf-8")

    ending = float(result.metrics["strategy_ending_capital"])
    report = {
        "schema_version": 1,
        "version": VERSION,
        "status": "completed",
        "mode": "TCC_ONLY_NO_ORDERS_NO_DATABASE",
        "run_id": run_id,
        "requested_assets": list(REQUESTED_ASSETS),
        "effective_assets": list(EFFECTIVE_ASSETS),
        "structural_exclusions": sorted(STRUCTURAL_EXCLUSIONS),
        "snapshot_path": str(snapshot.relative_to(ROOT)),
        "snapshot_sha256": manifest["snapshot_sha256"],
        "market_data": {"provider": "alpaca", "feed": "sip", "adjustment": "raw",
                        "corporate_actions_end": EXECUTION_END, "historical_start": HISTORY_START},
        "first_execution_date": first_date,
        "final_execution_date": last_date,
        "initial_state": "CASH",
        "initial_capital": INITIAL_CAPITAL,
        "ending_capital": ending,
        "profit": ending - INITIAL_CAPITAL,
        "return_pct": (ending / INITIAL_CAPITAL - 1.0) * 100.0,
        "transactions": len(trades),
        "session_count": len(predictions),
        "final_liquidation": True,
        "metrics": result.metrics,
        "data_audit": data_audit,
        "notes": [
            "Cenario contrafactual iniciado em caixa; nao e carteira de 2020 reescalada.",
            "Treino por folds do TCC; nao reproduz recalibracao intradiaria do Paper.",
            "Venda FINAL_SELL na ultima data e apenas contabil; Paper pode continuar investido.",
            "Novo download Alpaca pode diferir do snapshot congelado de 2026-10-06.",
            "Universo U77 foi selecionado retrospectivamente e nao elimina selection bias.",
        ],
    }
    _write_json(destination / "resultado.json", report)
    _write_json(destination / "snapshot_manifest.json", manifest)
    files = sorted(p for p in destination.iterdir() if p.is_file() and not p.name.endswith(".zip"))
    _write_json(destination / "hashes_saida.json", {p.name: _sha256(p) for p in files})
    bundle = destination / f"pacote_analise_u77_paper_replay_{run_id}.zip"
    with zipfile.ZipFile(bundle, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(destination.iterdir()):
            if file != bundle and file.is_file():
                archive.write(file, file.name)
    print(f"[finished] capital=US$ {ending:,.2f} profit=US$ {ending-INITIAL_CAPITAL:,.2f}", flush=True)
    print(f"[finished] package={bundle}", flush=True)
    print(f"[snapshot] {snapshot.root}", flush=True)
    return bundle


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"[failed] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
