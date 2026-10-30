"""Reprodução independente do Control Strategy #30 (Spyder: F5).

Usa a fonte LightGBM congelada do commit original e 75 CSVs de mercado
com data final 28/10/2026. O ZIP do backtest é SOMENTE referência para
comparação, nunca substitui dados ou resultados gerados pelo motor.
O processo gera arquivo de auditoria mesmo se abortado.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import traceback
from zipfile import ZipFile

import pandas as pd

from engine.configuracao import CONFIG
from pesquisa.control_operacional_strategy30 import (
    ROOT, carregar_contrato, validar_control_strategy30,
)
from pesquisa.paridade_strategy30 import (
    config_from_frozen_manifest, load_snapshot_strategy30,
    compare_control_trajectories,
)
from pesquisa.referencia_strategy30.execucao import (
    calcular_taxas_referencia, aplicar_deslizamento,
)
from pesquisa.referencia_strategy30.modelo_lightgbm import executar_lightgbm
from pesquisa.referencia_strategy30.rotacao import (
    _construir_folds_walk_forward, preparar_painel_rotacao,
)
from pesquisa.persistencia_resultados import (
    salvar_csv, salvar_json, salvar_texto,
)
from pesquisa.pacote_analise_switching import gerar_zip_resultados
from pesquisar_controle_strategy30_spyder import localizar_zip_original
from pesquisar_expansao_u67_spyder import _write_json

SCRIPT_VERSION = "control-strategy30-independent-parity-v1.0.1"
OUTPUT = ROOT / "output" / "paridade_strategy30"
SNAPSHOT = ROOT / "dados" / "strategy30_snapshot"


def _load_original(reference_zip: Path):
    with ZipFile(reference_zip) as archive:
        strategy = json.loads(archive.read("strategy_manifest.json"))
        experiment = json.loads(archive.read("experiment_manifest.json"))
    return strategy, experiment


def _run(output: Path) -> dict:
    # %% 1 - Autenticidade da referência e dos 75 ativos
    reference_zip = localizar_zip_original()
    contract = carregar_contrato()
    evidence, expected_predictions, expected_trades = validar_control_strategy30(
        reference_zip, contrato=contract,
    )
    salvar_json(output / "referencia_validada.json", evidence, _write_json)
    strategy, experiment = _load_original(reference_zip)
    if experiment.get("market_data_signature_sha256") != contract["market_data_signature_sha256"]:
        raise RuntimeError("O manifesto dos dados não coincide com a referência congelada.")

    # %% 2 - Snapshot local independente; proibido usar calendário 06/10
    snapshot_root = Path(
        os.environ.get("STRATEGY30_SNAPSHOT_DIR", str(SNAPSHOT))
    ).expanduser()
    source = load_snapshot_strategy30(
        snapshot_root, contract=contract, experiment_manifest=experiment,
    )
    salvar_csv(source.records, output / "auditoria_snapshot_75.csv", index=False)
    config = config_from_frozen_manifest(strategy, contract)
    if config.analysis_end_date != "2026-10-28":
        raise RuntimeError("O cutoff observado não é o da Strategy #30.")

    # %% 3 - Verificação do calendário ANTES do treino oneroso
    frames, calendar, calendar_source = preparar_painel_rotacao(
        source.frames, config,
    )
    if len(frames) != 75 or set(frames) != set(contract["assets"]):
        raise RuntimeError("Universo processado tem menos de 75 ativos.")
    folds = _construir_folds_walk_forward(calendar, config)
    expected_folds = list(contract["expected_fold_counts"].values())
    found_folds = [
        int(fold["test_end_index"] - fold["test_start_index"])
        for fold in folds
    ]
    if found_folds != expected_folds:
        raise RuntimeError(
            f"Calendário do motor diverge: folds={found_folds} "
            f"esperado={expected_folds}, calendar={len(calendar)} "
            f"source={calendar_source}."
        )
    if pd.Timestamp(folds[0]["test_start"]).date().isoformat() != contract["first_session"]:
        raise RuntimeError("Primeiro pregão de avaliação diverge.")
    if pd.Timestamp(folds[-1]["test_end"]).date().isoformat() != contract["last_session"]:
        raise RuntimeError("Último pregão de avaliação diverge.")
    calendar_log = [
        {
            "fold": int(fold["fold_id"]),
            "train_end": pd.Timestamp(fold["train_end"]).isoformat(),
            "calibration_start": pd.Timestamp(fold["calibration_start"]).isoformat(),
            "calibration_end": pd.Timestamp(fold["calibration_end"]).isoformat(),
            "test_start": pd.Timestamp(fold["test_start"]).isoformat(),
            "test_end": pd.Timestamp(fold["test_end"]).isoformat(),
            "oos_rows": int(fold["test_end_index"]-fold["test_start_index"]),
        }
        for fold in folds
    ]
    salvar_csv(pd.DataFrame(calendar_log), output / "auditoria_calendario.csv", index=False)
    print(
        f"[calendar] source={calendar_source} 75 ativos, "
        f"folds={found_folds}, cutoff={config.analysis_end_date}", flush=True,
    )

    # %% 4 - Fit real dos 75 modelos por fold, sem forçar margens históricas
    print("[fit] iniciando calibracao e reproduçao independente LightGBM", flush=True)
    results = executar_lightgbm(
        source.frames, config,
        calcular_taxas_referencia, aplicar_deslizamento,
        technical_log_callback=lambda text: print(f"[model] {text}", flush=True),
    )
    if len(results) != 1:
        raise RuntimeError(f"Esperava uma repetição, recebeu {len(results)}.")
    run = results[0]
    salvar_csv(
        run.predictions, output / "reproducao_predictions.csv",
        index=False, float_format="%.17g",
    )
    salvar_csv(run.trades, output / "reproducao_trades.csv", index=False)
    salvar_json(output / "reproducao_metricas.json", {
        "ending_capital": run.metrics.get("strategy_ending_capital"),
        "capital_rotations": run.metrics.get("capital_rotations"),
        "walk_forward_folds": run.metrics.get("walk_forward_folds"),
        "lightgbm_version": run.metrics.get("lightgbm_version"),
        "numeric_thread_limit": config.numeric_thread_limit,
        "source_revision": "f78b718ae4a459e34d91cc78a486dd6bbba197b5",
    }, _write_json)
    # %% 5 - Comparação dia a dia contra decisões reais do Control
    parity, detail = compare_control_trajectories(
        expected_predictions, run.predictions, contract=contract,
    )
    margins = [
        {
            "fold": int(item.get("fold_id")),
            "reproduced_effective_margin": item.get("effective_switch_margin"),
            "reproduced_calibrated_candidate": item.get("calibrated_candidate_margin"),
            "source": item.get("margin_selection_source"),
        }
        for item in run.metrics.get("walk_forward_folds", [])
    ]
    salvar_csv(pd.DataFrame(margins), output / "auditoria_margens.csv", index=False)
    salvar_csv(detail, output / "comparacao_sessao_a_sessao.csv", index=False)
    salvar_json(output / "auditoria_paridade.json", parity, _write_json)

    if not parity["fully_reproduced"]:
        raise RuntimeError(
            "Resultado do motor independente diferente do Control oficial. "
            f"Sessões divergentes: {parity['mismatch_sessions']}; "
            f"primeira: {parity['first_mismatch_date']}; "
            f"patrimônio recalculado: US$ {parity['replay_final_capital']:,.2f}. "
            "Veja comparacao_sessao_a_sessao.csv no ZIP."
        )
    result = {
        "status": "completed",
        "version": SCRIPT_VERSION,
        "source_revision": "f78b718ae4a459e34d91cc78a486dd6bbba197b5",
        "source_backtest_sha256": evidence["source_zip_sha256"],
        "calendar": calendar_log,
        "assets": contract["assets"],
        "control_reference": evidence,
        "parity": parity,
        "independent_reproduction": True,
        "research_only": True,
        "new_cash_experiments_enabled": False,
        "next_step": (
            "Integrar o overlay CASH somente sobre esta politica reproduzida "
            "e com o mesmo calendario; nao reutilizar os overlays historicos."
        ),
    }
    salvar_json(output / "resultado.json", result, _write_json)
    salvar_texto(
        output / "resumo_para_analise.md",
        "# Strategy #30: reprodução independente\n\n"
        f"Status: {parity['status']}\n"
        f"75 ativos, calendário: {calendar_source}, folds={found_folds}\n"
        f"Capital referência: US$ {parity['reference_final_capital']:,.2f}\n"
        f"Capital reproduzido: US$ {parity['replay_final_capital']:,.2f}\n"
        f"Decisões divergentes: {parity['mismatch_sessions']}\n"
        "Simulação independente reexecutada com dados RAW locais.\n"
        "Ainda não testa CASH.\n",
    )
    print(
        f"[parity] Control Strategy #30 reproduzido: "
        f"US$ {parity['replay_final_capital']:,.2f} "
        f"em {parity['reference_sessions']} sessoes", flush=True,
    )
    return result


def main() -> dict:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = OUTPUT / run_id
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = _run(output)
        package = gerar_zip_resultados(
            output, versao=SCRIPT_VERSION, status="completed",
        )
        print(f"[pacote] envie este ZIP para analise: {package}", flush=True)
        return result
    except Exception as exc:
        try:
            salvar_json(output / "erro_execucao.json", {
                "version": SCRIPT_VERSION,
                "status": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=25),
            }, _write_json)
            package = gerar_zip_resultados(
                output, versao=SCRIPT_VERSION, status="failed",
            )
            print(f"[pacote] erro capturado, envie o ZIP: {package}", flush=True)
        except Exception as package_error:
            print(f"[pacote] nao foi possivel salvar ZIP: {package_error}", flush=True)
        raise


if __name__ == "__main__":
    main()
