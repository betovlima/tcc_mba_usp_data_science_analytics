"""TCC v1.25.0 — replay financeiro experimental de Control com RAW intacto.

Roteiro de validação, não prova de lucro prospectivo. O universo e a
configuração foram escolhidos retrospectivamente; NÃO interpretar este
backtest como amostra independente de seleção / de overfitting.

Input único: snapshot U75 Alpaca RAW congelado, SHA esperado obrigatório.
Sem Alpaca, Mongo, atualização de CSV ou busca de hiperparâmetros.
No Spyder, F5. Gera pacote em output/validacao_economica_raw_75/.

Corte 06/10/2026, alinhado às duas simulações anteriores: US$587,17mi
(vintage antigo) e US$89,94mi (RAW renovado, pipeline antigo).
Os novos resultados não são conhecidos antes da execução.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.universo_fixo75 import universo_75, reproduzir_universo_fixo75


ROOT = Path(__file__).resolve().parent
VERSION = "1.25.0"
SCHEMA = "fixed75-raw-economic-custody-v1"
AS_OF_SNAPSHOT = "2026-10-22"
CUTOFF = "2026-10-06"
SNAPSHOT_SHA256 = "595b7a03dc2c8978487975cd3554b1ea65608741893f13208938dc624cb273cc"
LEGACY_CAPITAL_SAME_CUTOFF = 587_165_941.13
UPDATED_ALPACA_OLD_PIPELINE = 89_941_496.49


def _verify_source(root: Path) -> tuple[SnapshotPaths, tuple[str, ...]]:
    contract_path = root / "evidencias" / "manifest_universo_congelado.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    symbols = universo_75(contract, ())
    if len(symbols) != 75:
        raise RuntimeError("Somente universo fixo de 75 ativos.")

    candidates = sorted(
        (root / "dados" / ".u75_fresh_runs").glob(
            f"asof_{AS_OF_SNAPSHOT}_*/universe_75"
        ), reverse=True,
    )
    for candidate in candidates:
        paths = SnapshotPaths.from_root(candidate)
        if not paths.manifest.is_file():
            continue
        snapshot = validate_snapshot(paths)
        if (
            tuple(snapshot.get("assets") or ()) == symbols
            and snapshot.get("snapshot_sha256") == SNAPSHOT_SHA256
        ):
            return paths, symbols
    raise FileNotFoundError(
        "O snapshot U75 e hash exigidos nao foram encontrados. "
        "Falha fechada: nao baixar dados, nao trocar de fonte."
    )


def executar(*, root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    source, assets = _verify_source(root)
    versioned_output = (
        root / "output" / "validacao_economica_raw_75"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    versioned_output.mkdir(parents=True, exist_ok=False)
    print(
        f"[RAW v{VERSION}] U75={len(assets)}; cutoff={CUTOFF}; "
        f"snapshot_sha256={SNAPSHOT_SHA256}; no network; no tuning.",
        flush=True,
    )
    result = reproduzir_universo_fixo75(
        root=root,
        output=versioned_output,
        snapshot_root=source.root,
        analysis_end_date=CUTOFF,
        version=VERSION,
        execution_schema=SCHEMA,
        economic_raw_split_mode=True,
    )
    metrics = result["metrics"]
    output_capital = float(metrics["strategy_ending_capital"])
    if not (0. <= output_capital < float("inf")):
        raise RuntimeError("Capital calculado inválido.")

    interpretation = {
        "version": VERSION,
        "schema": SCHEMA,
        "snapshot_sha256": SNAPSHOT_SHA256,
        "cutoff": CUTOFF,
        "assets": len(assets),
        "measured_capital_raw_economic": output_capital,
        "historical_legacy_capital_reference": LEGACY_CAPITAL_SAME_CUTOFF,
        "updated_alpaca_legacy_processing_capital_reference":
            UPDATED_ALPACA_OLD_PIPELINE,
        "delta_vs_updated_legacy_processing": (
            output_capital - UPDATED_ALPACA_OLD_PIPELINE
        ),
        "delta_vs_original_vintage": (
            output_capital - LEGACY_CAPITAL_SAME_CUTOFF
        ),
        "model_parameters_reselected_on_test": False,
        "walk_forward_calibration_as_frozen_protocol": True,
        "historical_universe_selection_bias": True,
        "cash_in_lieu_simulated": False,
        "market_impact_simulated": False,
        "prospective_out_of_selection_validation": False,
        "financial_realism_warning": (
            "Resultados somente simulados: parametros e 75 ativos "
            "selecionados historicamente; execucao sem market impact, "
            "sem slippage configurado e permitindo frações de ações. "
            "O novo capital NÃO prova ausência de overfitting."
        ),
        "full_result_json": "reproducao_universo.json",
        "package_name": "pacote_validacao_economica_raw.zip",
    }
    (versioned_output / "interpretacao_resultado.json").write_text(
        json.dumps(interpretation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[RAW v{VERSION}] capital_simulado=US$ {output_capital:,.2f}; "
        f"versus_89mi=US$ {interpretation['delta_vs_updated_legacy_processing']:,.2f}",
        flush=True,
    )
    print(
        "[RAW v1.25.0] Resultado retrospectivo. NÃO é validação "
        "de overfitting em dados nunca selecionados.",
        flush=True,
    )
    return interpretation


if __name__ == "__main__":
    executar()
