"""Pesquisa pré-especificada de mitigação de overfitting LightGBM — v1.25.8.

Execute com F5 no Spyder. Usa exclusivamente os CSVs congelados já existentes.
Executa controle e 3 intervenções ISOLADAS, mantendo universo, folds, labels,
regras financeiras e margens. Não acessa Alpaca/Mongo nem muda a produção.
O histórico 2020-2026 já foi analisado pela pesquisa: comparações exploratórias,
NÃO uma validação prospectiva independente. NÃO promove um "winner".
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reproducao.dados import validate_snapshot
from reproducao.experimento import build_control_config
from reproducao.preparacao import (load_raw_bar_file, load_actions_file, split_normalize)
from reproducao.universo_fixo75 import universo_75

ROOT = Path(__file__).resolve().parent
VERSION = "1.25.8"
OUTPUT_ROOT = ROOT / "output" / "mitigacao_overfitting_v1258"
EXPECTED_OLD_CAPITAL = 587_165_941.1330212

# Hipóteses isoladas e fixadas ANTES de observar os resultados.
# Não realizar busca de parâmetros em cima dos folds OOS.
VARIANTS = (
    ("controle_original", {}),
    ("bagging_ativo", {"subsample_freq": 1}),
    ("folha_min_50", {"min_child_samples": 50}),
    ("menos_arvores_160", {"n_estimators": 160}),
)

def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2,
                   default=lambda x: x.isoformat() if hasattr(x, "isoformat") else str(x),
                   allow_nan=False),
        encoding="utf-8",
    )


def finite(x):
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def check_inputs():
    """Verifica identidade e SHA de dados ANTES de criar o painel único."""
    import pesquisar_expansao_u67_spyder as p  # Motor matemático compartilhado.
    contract_path = ROOT / "evidencias" / "manifest_universo_congelado.json"
    manifest = json.loads(contract_path.read_text(encoding="utf-8"))
    selected = universo_75(manifest, ())  # Único universo imutável: 75 ativos.
    if len(selected) != 75 or len(set(selected)) != 75:
        raise RuntimeError("Exigidos exatamente 75 ativos fixos desde o fold 1.")
    if not math.isclose(float(manifest["expected_ending_capital"]),
                        EXPECTED_OLD_CAPITAL, rel_tol=0, abs_tol=.01):
        raise RuntimeError("Contrato do checkpoint histórico não corresponde.")
    snapshot = validate_snapshot(p.BASE_DATA)
    if snapshot["snapshot_sha256"] != manifest["snapshot_sha256"]:
        raise RuntimeError("Vintage histórico divergente: não baixar ou ajustar dados.")
    for symbol, hashes in manifest["candidate_file_hashes"].items():
        if symbol not in selected:
            raise RuntimeError(f"Ativo fora do painel fixo: {symbol}")
        for key, filename in (
            ("raw_sha256", p.FULL_DATA.raw_bars / f"{symbol}.csv"),
            ("corporate_actions_sha256", p.FULL_DATA.corporate_actions / f"{symbol}.csv"),
        ):
            if not filename.is_file() or file_sha(filename) != hashes[key]:
                raise RuntimeError(f"Cache congelado ausente/SHA divergente: {symbol} {key}")
    return p, manifest, snapshot, selected


def prepare_once(p, manifest, selected):
    """Uma ÚNICA etapa de carga e painel com os 75 ativos antes dos folds.

    Os CSVs do vintage histórico estão armazenados em duas pastas por
    proveniência documental. ISSO NÃO cria dois universos de modelagem.
    Nenhum treino, calibração ou replay é executado com 65 ativos.
    """
    raw_75 = {}
    sources = set(p.U67_REQUESTED_ASSETS)  # Apenas localização física histórica.
    for index, symbol in enumerate(selected, start=1):
        origin = p.BASE_DATA if symbol in sources else p.FULL_DATA
        path_bars = origin.raw_bars / f"{symbol}.csv"
        path_actions = origin.corporate_actions / f"{symbol}.csv"
        raw = load_raw_bar_file(path_bars, float_precision="round_trip")
        actions = load_actions_file(path_actions)
        issue = p._generic_identity_issue(symbol, actions)
        if issue is not None:
            raise RuntimeError(f"Ativo estruturalmente inconsistente: {symbol}: {issue}")
        normalized, _ = split_normalize(raw, actions)
        raw_75[symbol] = normalized
        print(f"[painel_unico] {index}/75 {symbol}", flush=True)

    if tuple(raw_75) != tuple(selected) or len(raw_75) != 75:
        raise RuntimeError("Um único painel de 75 ativos é obrigatório.")
    config = build_control_config(raw_75, p.CONFIG).copiar_modelo(
        update={"analysis_end_date": manifest["analysis_end_date"]}
    )
    (
        frames, dates, calendar, symbols, folds, decision_dates,
        decision_to_fold, decision_metadata,
    ) = p._construir_contexto_execucao(raw_75, config)

    if set(symbols) != set(selected) or len(symbols) != 75 or len(folds) != 3:
        raise RuntimeError(
            "Violação do contrato fixed75: símbolos/folds de treino divergentes."
        )
    if tuple(config.assets) != tuple(selected):
        raise RuntimeError("O config Control não contém todos os 75 ativos.")
    print("[contrato] 75 ativos juntos em treino, calibração e replay; 3 folds",
          flush=True)
    return (config, frames, dates, symbols, folds, decision_dates,
            decision_to_fold, decision_metadata)

def settings_for(base_config, changes):
    model_settings = deepcopy(base_config.research_model_settings)
    params = model_settings["lightgbm"]
    expected = {"subsample_freq", "min_child_samples", "n_estimators"}
    if not set(changes) <= expected:
        raise RuntimeError("Uma alteração não pré-registrada foi solicitada.")
    if params["subsample"] != .85 or params["subsample_freq"] != 0:
        raise RuntimeError("Bagging original não coincide com protocolo.")
    if params["min_child_samples"] != 18 or params["n_estimators"] != 329:
        raise RuntimeError("Parâmetros originais divergem do controle.")
    params.update(changes)
    # O motor existente ignora early-stopping: não fingir testá-lo aqui.
    return base_config.copiar_modelo(update={"research_model_settings": model_settings})


def diagnostics_for_variant(p, states, symbols, frames):
    """Diagnóstico de 225 ajustes feitos na MESMA chamada de treino fixed75."""
    rows = []
    for state in states:
        by_asset = (state.oos_diagnostics or {}).get("by_asset") or {}
        if set(state.fit_diagnostics) != set(symbols) or set(by_asset) != set(symbols):
            raise RuntimeError(
                f"Fold {state.fold_id}: foram exigidos 75 modelos/diagnósticos."
            )
        for symbol in symbols:
            rows.append(p._fit_quality_row(
                universe_role="fixed75", symbol=symbol, fold_id=state.fold_id,
                frame=frames[symbol], train_dates=state.final_fit_dates,
                decision_dates=state.decision_dates,
                fit_diagnostics=state.fit_diagnostics[symbol],
                oos_diagnostics=by_asset[symbol],
            ))
    df = pd.DataFrame(rows)
    required = ["asset", "fold_id", "train_rmse", "oos_rmse",
                "train_rows", "oos_rows", "naive_oos_rmse"]
    for col in required[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if (
        len(df) != 225 or df["asset"].nunique() != 75
        or set(df["fold_id"]) != {1, 2, 3}
        or df.duplicated(["asset", "fold_id"]).any()
        or df[required[2:]].isna().any().any()
        or (df["train_rmse"] <= 0).any() or (df["oos_rmse"] < 0).any()
        or (df[["train_rows", "oos_rows"]] <= 0).any().any()
    ):
        raise RuntimeError("225 diagnósticos de 75 ativos x 3 folds inválidos.")
    df["gap_percent"] = 100 * (df["oos_rmse"] / df["train_rmse"] - 1)
    return df.sort_values(["asset", "fold_id"])

def pooled_rmse(values, rows):
    return math.sqrt(float(np.sum((values.to_numpy(dtype=float) ** 2)
                                  * rows.to_numpy(dtype=float))
                           / rows.sum()))


def summarize(df, metrics, changes, original_decisions, current_decisions):
    pooled_oos = pooled_rmse(df["oos_rmse"], df["oos_rows"])
    pooled_naive = pooled_rmse(df["naive_oos_rmse"], df["oos_rows"])
    assert len(current_decisions) == 1560
    delta = None
    if original_decisions is not None:
        if not current_decisions.index.equals(original_decisions.index):
            raise RuntimeError("Datas das decisões divergentes entre modelos.")
        delta = int((current_decisions != original_decisions).sum())
    return {
        "parameters_changed": changes,
        "n_models": len(df),
        "gap_median_percent": finite(df["gap_percent"].median()),
        "oos_rmse_pooled": finite(pooled_oos),
        "naive_rmse_pooled": finite(pooled_naive),
        "relative_rmse_to_naive_percent": finite(
            100 * (pooled_oos / pooled_naive - 1)
        ),
        "overfit_warning_count": int(df["overfit_warning"].astype(str)
                                     .str.lower().eq("true").sum()),
        "positive_oos_skill_count": int((df["oos_skill_vs_naive"] > 0).sum()),
        "changed_decisions_from_control": delta,
        "capital": finite(metrics["ending_capital"]),
        "cagr": finite(metrics["cagr"]),
        "sharpe": finite(metrics["sharpe"]),
        "max_drawdown": finite(metrics["maximum_drawdown"]),
        "worst_fold_return": finite(metrics["worst_fold_return"]),
        "capital_rotations": int(metrics["capital_rotations"]),
        "fold_metrics": metrics["folds"],
    }


def charts(summaries, equities, destination):
    labels = list(summaries)
    x = np.arange(len(labels))
    names = [n.replace("_", "\n") for n in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, [summaries[n]["oos_rmse_pooled"] for n in labels])
    ax.set(xticks=x, xticklabels=names, ylabel="RMSE OOS ponderado",
           title="1. Erro preditivo em períodos OOS internos")
    fig.tight_layout()
    fig.savefig(destination / "01_rmse_oos.svg"); fig.savefig(destination / "01_rmse_oos.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, [summaries[n]["gap_median_percent"] for n in labels])
    ax.set(xticks=x, xticklabels=names, ylabel="Gap mediano de RMSE (%)",
           title="2. Diferença entre RMSE no teste e no treino")
    fig.tight_layout()
    fig.savefig(destination / "02_gap.svg"); fig.savefig(destination / "02_gap.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for label, equity in equities.items():
        ax.plot(pd.to_datetime(equity.index, utc=True), equity["strategy_equity"],
                label=label)
    ax.set_yscale("log")
    ax.set(xlabel="Sessão de decisão", ylabel="Capital simulado (US$, escala log)",
           title="3. Capital histórico por regularização — não é novo holdout")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(destination / "03_capital.svg"); fig.savefig(destination / "03_capital.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    vals = [summaries[n]["changed_decisions_from_control"] or 0 for n in labels]
    ax.bar(x, vals)
    ax.set(xticks=x, xticklabels=names, ylabel="Sessões com ativo selecionado diferente",
           title="4. Efeito da regularização nas escolhas (de 1.560 sessões)")
    fig.tight_layout()
    fig.savefig(destination / "04_decisoes.svg"); fig.savefig(destination / "04_decisoes.png", dpi=220)
    plt.close(fig)


def main():
    p, contract, snap, universe = check_inputs()
    original_snapshot_sha = snap["snapshot_sha256"]
    orig_candidate_hashes = contract["candidate_file_hashes"]
    (
        config, frames, common_dates, symbols, folds, all_dates,
        decision_to_fold, decision_metadata,
    ) = prepare_once(p, contract, universe)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = OUTPUT_ROOT / run_id
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    summaries = {}
    equities = {}
    original_decisions = None

    for idx, (label, changes) in enumerate(VARIANTS, start=1):
        print(f"[variant {idx}/{len(VARIANTS)}] {label}: {changes}", flush=True)
        vdir = output / label
        vdir.mkdir()
        cfg = settings_for(config, changes)
        # UMA chamada por fold, contendo TODOS os 75 modelos e a política completa.
        states, policies, calibration = p._train_baseline_states(
            frames, common_dates, symbols, folds, cfg
        )
        result = p._simular_exato(
            "tcc_single_fixed_75",
            p._politica_agendada(policies, decision_to_fold),
            frames, symbols, all_dates, cfg,
            p.calcular_taxas_referencia,
            p.aplicar_deslizamento,
            decision_metadata=decision_metadata,
            model_label="TCC Control - 75 ativos fixos desde o primeiro fold",
            method_line="- 75 ativos ajustados e calibrados conjuntamente; sem expansão.",
        )
        metrics = p._scenario_metrics(result, folds, float(cfg.initial_capital))
        df = diagnostics_for_variant(p, states, symbols, frames)
        decisions = result.predictions["selected_asset"].copy()
        if label == "controle_original":
            capital = float(metrics["ending_capital"])
            if not math.isclose(capital, EXPECTED_OLD_CAPITAL, rel_tol=0, abs_tol=.01):
                raise RuntimeError(
                    "Controle NÃO reproduziu o checkpoint de US$ 587 mi. "
                    f"Observado: {capital:.8f}. Interrompido sem tentar ajustar "
                    "modelo/dados para obter lucro.")
            original_decisions = decisions.copy()
        summary = summarize(df, metrics, changes, original_decisions, decisions)
        summaries[label] = summary
        equities[label] = result.predictions[[
            "strategy_equity", "buy_hold_equity"
        ]].copy()
        df.to_csv(vdir / "model_generalization_225.csv", index=False)
        pd.DataFrame(calibration).to_csv(vdir / "fold_calibration.csv", index=False)
        
        result.predictions.to_csv(vdir / "predictions.csv",
                                  index=True, float_format="%.17g")
        save_json(vdir / "summary.json", summary)
        save_json(output / "estado_parcial.json", {
            "status": "partial", "completed_variants": list(summaries),
            "snapshot_sha256": original_snapshot_sha,
            "results": summaries,
        })
        print(f"[result {label}] RMSE={summary['oos_rmse_pooled']:.6f}; "
              f"gap_mediano={summary['gap_median_percent']:.2f}%; "
              f"capital=US$ {summary['capital']:,.2f}; "
              f"changed={summary['changed_decisions_from_control']}", flush=True)
        del states, policies, result

    if validate_snapshot(p.BASE_DATA)["snapshot_sha256"] != original_snapshot_sha:
        raise RuntimeError("Snapshot base foi modificado durante a execução.")
    for sym, expected in orig_candidate_hashes.items():
        files = {
            "raw_sha256": p.FULL_DATA.raw_bars / f"{sym}.csv",
            "corporate_actions_sha256": p.FULL_DATA.corporate_actions / f"{sym}.csv",
        }
        for key, path in files.items():
            if file_sha(path) != expected[key]:
                raise RuntimeError(f"Arquivo original modificado: {sym} {key}")

    charts(summaries, equities, output)
    report = {
        "status": "completed",
        "version": VERSION,
        "study_type": "retrospective_predefined_interventions",
        "independent_prospective_holdout": False,
        "snapshot_sha256": original_snapshot_sha,
        "historical_checkpoint": EXPECTED_OLD_CAPITAL,
        "requested_assets": 75,
        "effective_assets": 75,
        "research_design": "fixed_75_trained_together_from_first_fold",
        "training_phases": "single_panel_75_only",
        "legacy_65_plus_10_training": False,
        "folds": 3,
        "comparison_rule": (
            "Não escolher o maior capital ou menor gap dos folds históricos "
            "como prova de generalização prospectiva. Sem promoção automática."
        ),
        "variants": summaries,
        "elapsed_seconds": time.perf_counter() - start,
    }
    save_json(output / "relatorio_mitigacao.json", report)
    pd.DataFrame([
        {"variant": name, **{
            k:v for k,v in s.items()
            if k != "fold_metrics" and k != "parameters_changed"
        }} for name,s in summaries.items()
    ]).to_csv(output / "comparacao_variantes.csv", index=False)
    package = output / "pacote_mitigacao_overfitting_v1258.zip"
    with ZipFile(package, "w", compression=ZIP_DEFLATED) as z:
        for file in output.rglob("*"):
            if file.is_file() and file != package:
                z.write(file, file.relative_to(output).as_posix())
    print(f"[done] pacote={package}; sem rede; sem promocao", flush=True)
    return report


if __name__ == "__main__":
    main()
