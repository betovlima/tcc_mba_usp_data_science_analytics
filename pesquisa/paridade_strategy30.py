"""Validação de paridade executável para o Control Strategy #30.

Não altera o Control histórico do TCC. A comparação exige 75 CSVs locais e
o ZIP original de evidências; não substitui a falta de dados por resultados
importados. A saída diferencia validação de tape de reprodução do modelo.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from engine.configuracao import CONFIG, construir_configuracao_controle
from reproducao.preparacao import load_raw_bar_file, load_actions_file, split_normalize


def config_from_frozen_manifest(
    strategy: dict[str, Any],
    contract: dict[str, Any],
):
    """Aplica ao motor científico somente parâmetros observados no manifesto."""
    if strategy["strategy_profile_id"] != contract["strategy_profile_id"]:
        raise RuntimeError("Perfil da estratégia não corresponde ao Control.")
    if strategy["strategy_configuration_hash"] != contract["strategy_configuration_hash"]:
        raise RuntimeError("Hash da configuração de estratégia divergente.")
    effective = strategy["effective_configuration"]
    if effective.get("analysis_end_date") != contract["last_session"]:
        raise RuntimeError("O corte temporal deve coincidir com o Control.")
    if effective.get("research_market_data_protocol") != "raw_total_causal_v1":
        raise RuntimeError("Protocolo RAW da referência diverge.")
    if effective.get("research_model_family") != "lightgbm_utility":
        raise RuntimeError("O motor reproduz somente LightGBM Utility.")
    settings = deepcopy(effective["research_model_settings"])
    if settings.get("soft_horizon_consensus", {}).get("enabled"):
        raise RuntimeError("O Control não pode usar Soft Horizon.")
    if len(contract["assets"]) != int(contract["effective_asset_count"]):
        raise RuntimeError("Quantidade de ativos difere do manifesto do Control.")
    source_assets = sorted(str(x) for x in effective["assets"])
    if source_assets != contract["assets"]:
        raise RuntimeError("Universo da configuração não coincide com o Control.")
    base = construir_configuracao_controle(
        CONFIG, assets=tuple(contract["assets"]),
    )
    field_names = set(base.__dataclass_fields__)
    # São valores observados no manifesto, sem ajuste por capital OOS.
    allowed = (
        "rotation_target_horizons", "rotation_target_horizon_weights",
        "rotation_movement_capture_weight", "rotation_trend_persistence_weight",
        "rotation_minimum_training_rows", "rotation_walk_forward_calibration_days",
        "rotation_walk_forward_test_days", "rotation_walk_forward_min_test_days",
        "rotation_purge_days", "rotation_downside_penalty", "rotation_drawdown_penalty",
        "rotation_min_holding_days", "rotation_min_expected_edge",
        "rotation_cash_threshold", "rotation_switch_margin",
        "rotation_switch_margin_candidates", "initial_capital", "slippage_bps",
        "commission_rate", "sec_fee_rate", "taf_fee_per_share", "taf_fee_cap",
        "cat_fee_per_share", "deterministic_execution", "numeric_thread_limit",
        "random_state", "analysis_start_date", "analysis_end_date",
        "walk_forward_fold_count_override",
    )
    changes: dict[str, Any] = {}
    for name in allowed:
        if name in effective and name in field_names:
            value = effective[name]
            if isinstance(getattr(base, name), tuple):
                value = tuple(value)
            changes[name] = value
    changes.update(
        {
            "strategy_mode": "COMPOUND_ROTATION_SWING_LIGHTGBM",
            "assets": tuple(contract["assets"]),
            "rotation_model_repetitions": int(effective["rotation_xgb_repetitions"]),
            "rotation_seed_step": int(effective["rotation_seed_step"]),
            "research_model_settings": settings,
        }
    )
    config = base.copiar_modelo(update=changes)
    # Não reescreve margens por fold: o algoritmo precisa recalibrá-las.
    if tuple(config.rotation_switch_margin_candidates) != (0.0, 0.0025, 0.005, 0.01):
        raise RuntimeError("Grade de calibração da Strategy #30 diverge.")
    return config


@dataclass(frozen=True)
class SnapshotReport:
    frames: dict[str, pd.DataFrame]
    records: pd.DataFrame


def load_snapshot_strategy30(
    root: Path,
    *,
    contract: dict[str, Any],
    experiment_manifest: dict[str, Any],
) -> SnapshotReport:
    """Carrega 75 ativos, falhando em falta/alteração de identidade; não exclui ativos."""
    root = Path(root)
    raw_folder = root / "raw_bars"
    actions_folder = root / "corporate_actions"
    signatures = experiment_manifest["market_data_signatures"]
    expected = contract["assets"]
    if sorted(signatures) != expected or len(expected) != int(contract["effective_asset_count"]):
        raise RuntimeError("Assinaturas de mercado não cobrem os ativos do contrato.")
    missing = [
        str(directory / f"{symbol}.csv")
        for symbol in expected
        for directory in (raw_folder, actions_folder)
        if not (directory / f"{symbol}.csv").is_file()
    ]
    if missing:
        raise RuntimeError(
            "Snapshot histórico 28/10/2026 incompleto. "
            f"Ausentes {len(missing)} arquivos, incluindo: {missing[:4]}. "
            "Não usar snapshot 06/10/2026 no lugar."
        )
    bars = {}
    rows = []
    for index, symbol in enumerate(expected, start=1):
        raw_path = raw_folder / f"{symbol}.csv"
        action_path = actions_folder / f"{symbol}.csv"
        original = load_raw_bar_file(raw_path, float_precision="round_trip")
        actions = load_actions_file(action_path)
        normalized, adjustments = split_normalize(original, actions)
        signature = signatures[symbol]
        first = pd.Timestamp(normalized.index[0]).isoformat()
        last = pd.Timestamp(normalized.index[-1]).isoformat()
        report = {
            "symbol": symbol,
            "raw_rows": int(len(original)),
            "normalized_rows": int(len(normalized)),
            "manifest_rows": int(signature["rows"]),
            "first_timestamp": first,
            "expected_first_timestamp": signature["first_timestamp"],
            "last_timestamp": last,
            "expected_last_timestamp": signature["last_timestamp"],
            "split_events_applied": int(len(adjustments)),
            "original_raw_sha256": signature["raw_sha256"],
            "original_normalized_sha256": signature["normalized_sha256"],
        }
        # Row count and bounds are necessary but not sufficient for equality.
        if (
            report["normalized_rows"] != report["manifest_rows"]
            or pd.Timestamp(first) != pd.Timestamp(signature["first_timestamp"])
            or pd.Timestamp(last) != pd.Timestamp(signature["last_timestamp"])
        ):
            raise RuntimeError(
                f"{symbol}: calendário/histórico incompatível com a Strategy #30: {report}"
            )
        if normalized.index.has_duplicates or not normalized.index.is_monotonic_increasing:
            raise RuntimeError(f"{symbol}: datas duplicadas ou fora de ordem.")
        required = normalized[["open", "high", "low", "close", "volume"]].to_numpy(float)
        if not np.isfinite(required).all() or (required[:, :4] <= 0).any():
            raise RuntimeError(f"{symbol}: preços não finitos ou inválidos.")
        bars[symbol] = normalized
        rows.append(report)
        if index % 15 == 0 or index == len(expected):
            print(f"[snapshot] {index}/75 ativos conferidos", flush=True)
    return SnapshotReport(frames=bars, records=pd.DataFrame(rows))


def compare_control_trajectories(
    reference: pd.DataFrame,
    calculated: pd.DataFrame,
    *,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Impede declarar reprodução se qualquer sessão, posição ou capital divergir."""
    columns = ["selected_asset", "walk_forward_fold", "strategy_equity"]

    def normalizar_tabela(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        """Aceita timestamp como coluna (ZIP) ou DatetimeIndex (motor)."""
        if "timestamp" in frame.columns:
            result = frame[["timestamp", *columns]].copy()
        elif isinstance(frame.index, pd.DatetimeIndex):
            result = frame[columns].copy()
            result.insert(0, "timestamp", frame.index)
        else:
            raise RuntimeError(
                f"{label}: sem timestamp em coluna ou DatetimeIndex."
            )
        result["timestamp"] = pd.to_datetime(
            result["timestamp"], utc=True, errors="coerce",
        )
        if result["timestamp"].isna().any():
            raise RuntimeError(f"{label}: timestamp ausente/invalido.")
        if result["timestamp"].duplicated().any():
            raise RuntimeError(f"{label}: sessoes duplicadas na comparacao.")
        result["strategy_equity"] = pd.to_numeric(
            result["strategy_equity"], errors="coerce",
        )
        if not np.isfinite(result["strategy_equity"]).all():
            raise RuntimeError(f"{label}: patrimonio nao numerico/finito.")
        if result["selected_asset"].isna().any():
            raise RuntimeError(f"{label}: ativo selecionado ausente.")
        # Remove o indice original ANTES da ordenacao: pandas considera
        # ambiguo ter nivel e coluna chamados timestamp ao mesmo tempo.
        result = result.reset_index(drop=True)
        result.sort_values("timestamp", inplace=True)
        return result.reset_index(drop=True)

    left = normalizar_tabela(reference, "Control de referencia")
    right = normalizar_tabela(calculated, "Control reproduzido")
    left = left.rename(columns={
        "selected_asset": "reference_asset", "walk_forward_fold": "reference_fold",
        "strategy_equity": "reference_equity",
    })
    right = right.rename(columns={
        "selected_asset": "replay_asset", "walk_forward_fold": "replay_fold",
        "strategy_equity": "replay_equity",
    })
    comparison = left.merge(right, on="timestamp", how="outer", validate="one_to_one", indicator=True)
    comparison["same_asset"] = comparison["reference_asset"] == comparison["replay_asset"]
    comparison["same_fold"] = comparison["reference_fold"] == comparison["replay_fold"]
    comparison["absolute_equity_difference"] = (
        comparison["replay_equity"] - comparison["reference_equity"]
    ).abs()
    equivalent = bool(
        len(left) == len(right) == int(contract["expected_rows"])
        and (comparison["_merge"] == "both").all()
        and comparison["same_asset"].all()
        and comparison["same_fold"].all()
        and comparison["absolute_equity_difference"].le(0.01).all()
    )
    mismatches = comparison.loc[
        (comparison["_merge"] != "both")
        | ~comparison["same_asset"]
        | ~comparison["same_fold"]
        | comparison["absolute_equity_difference"].gt(0.01)
    ]
    summary = {
        "status": "reproduced" if equivalent else "mismatch",
        "reference_sessions": len(left),
        "replay_sessions": len(right),
        "mismatch_sessions": int(len(mismatches)),
        "first_mismatch_date": (
            pd.Timestamp(mismatches.iloc[0]["timestamp"]).isoformat()
            if not mismatches.empty else None
        ),
        "reference_final_capital": float(left.iloc[-1]["reference_equity"]),
        "replay_final_capital": float(right.iloc[-1]["replay_equity"]),
        "final_capital_difference": (
            float(right.iloc[-1]["replay_equity"])
            - float(left.iloc[-1]["reference_equity"])
        ),
        "fully_reproduced": equivalent,
        "comparison_requirements": [
            "same historical sessions", "same 75-asset decisions",
            "same walk-forward fold per session",
            "equity absolute difference <= US$0.01 on every session",
        ],
    }
    return summary, comparison
