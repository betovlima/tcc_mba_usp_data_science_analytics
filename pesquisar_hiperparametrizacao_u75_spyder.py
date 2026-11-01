"""TCC / Spyder F5: testar hiperparametros do LightGBM no U75 congelado.

Nao modifica Control, MCT, CSVs ou snapshots; nao faz downloads, nao usa banco.
Valida primeiro a paridade financeira do universo original US$ 587.165.941,13.
Compara uma configuracao congelada e uma configuracao escolhida em cada fold
somente no periodo anterior ao seu teste, com 52 atributos inalterados.
O otimizador e LHS + GaussianProcessRegressor/EI (nao o CARO historico exato).
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
import math
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine.configuracao import CONFIG
from engine.execucao import calcular_taxas_referencia, aplicar_deslizamento
from engine.modelo_lightgbm import (
    _ajustar_modelos_lightgbm, _construir_contexto_execucao, selecionar_switch_margin,
)
from engine.rotacao import (
    ROTATION_FEATURES, _precalcular_utilidades_modelo, _politica_utilidade,
    _crescimento_politica_simples, _politica_agendada, _simular_exato,
)
from reproducao.dados import validate_snapshot
from reproducao.preparacao import prepare_model_frames
from reproducao.experimento import build_control_config, summarize_metrics
from pesquisa.otimizacao_hiperparametros_u75 import (
    VERSION, SearchSettings, choose_best, propose, validate_fold_windows,
)

# %% 0. Contrato fixo da pesquisa
ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output" / "hiperparametrizacao_u75"
CONTRACT_FILE = ROOT / "evidencias" / "manifest_universo_congelado.json"
SEARCH = SearchSettings(budget=8, lhs_proposals=4, seed=42)
REFERENCE_TOLERANCE_USD = 0.01
CALIBRATION_OPTIMIZE_DAYS = 60
CALIBRATION_GAP_DAYS = 2


def _digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, obj: dict | list) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _config_with_params(frozen_config, lightgbm: dict):
    settings = deepcopy(frozen_config.research_model_settings)
    settings["lightgbm"] = deepcopy(lightgbm)
    return frozen_config.copiar_modelo(update={"research_model_settings": settings})


def _load_u75():
    # A importacao do programa legado nao chama seu main nem faz downloads.
    import pesquisar_expansao_u67_spyder as pesquisa
    contract = json.loads(CONTRACT_FILE.read_text(encoding="utf-8"))
    selected = tuple(contract["selected_assets"])
    if selected != tuple(pesquisa.DISCOVERY_SELECTED_CANDIDATES):
        raise RuntimeError("Universo selecionado nao corresponde ao contrato congelado")
    base_manifest = validate_snapshot(pesquisa.BASE_DATA)
    if base_manifest.get("snapshot_sha256") != contract["snapshot_sha256"]:
        raise RuntimeError("Manifesto U67 nao corresponde ao checkpoint historico")

    file_hashes = {}
    for ticker in selected:
        expected = contract["candidate_file_hashes"][ticker]
        files = {
            "raw_sha256": pesquisa.FULL_DATA.raw_bars / f"{ticker}.csv",
            "corporate_actions_sha256": pesquisa.FULL_DATA.corporate_actions / f"{ticker}.csv",
        }
        observed = {}
        for key, path in files.items():
            if not path.is_file():
                raise FileNotFoundError(f"Snapshot externo congelado nao encontrado: {path}")
            observed[key] = _digest(path)
            if observed[key] != expected[key]:
                raise RuntimeError(f"SHA-256 divergente em {ticker}:{key}; nao fazer download ou reparo")
        file_hashes[ticker] = observed

    if tuple(base_manifest["assets"]) != tuple(pesquisa.U67_REQUESTED_ASSETS):
        raise RuntimeError("Universo U67 e ordem divergentes do manifesto")
    raw, exclusions, _, _ = prepare_model_frames(
        pesquisa.BASE_DATA,
        assets=tuple(pesquisa.U67_REQUESTED_ASSETS),
        csv_float_precision="round_trip",
    )
    excluded = {str(item.get("symbol", "")).upper() for item in exclusions}
    if excluded != pesquisa.U67_EXPECTED_EXCLUSIONS:
        raise RuntimeError(f"Exclusoes estruturais inesperadas: {sorted(excluded)}")
    baseline_order = tuple(x for x in pesquisa.U67_REQUESTED_ASSETS if x in raw)
    if len(baseline_order) != 65:
        raise RuntimeError("Universo base deveria ter 65 ativos efetivos")
    config = build_control_config(
        {name: raw[name] for name in baseline_order}, CONFIG
    ).copiar_modelo(update={"analysis_end_date": contract["analysis_end_date"]})
    (base_frames, common_dates, calendar_source, base_symbols, folds,
     decisions, mapping, metadata) = _construir_contexto_execucao(
         {name: raw[name] for name in baseline_order}, config
     )
    candidate_frames, audit, confirmed_hashes = pesquisa._prepare_candidate_frames(
        list(selected), common_dates, decisions, config
    )
    if set(candidate_frames) != set(selected) or confirmed_hashes != file_hashes:
        raise RuntimeError("Identidade ou continuidade de candidatos divergente")
    frames = dict(base_frames)
    frames.update(candidate_frames)
    symbols = sorted(frames)
    if len(symbols) != 75 or len(set(symbols)) != 75:
        raise RuntimeError(f"Experimento exige U75 exato, encontrados={len(symbols)}")
    if len(ROTATION_FEATURES) != 52 or len(folds) != 3 or len(decisions) - 1 != 1560:
        raise RuntimeError("Contrato de features/folds/calendario divergente da referencia")
    config = build_control_config({x: frames[x] for x in symbols}, config)
    return frames, symbols, common_dates, folds, decisions, mapping, metadata, config, {
        "snapshot_sha256": base_manifest["snapshot_sha256"],
        "candidate_file_hashes": file_hashes,
        "calendar_source": calendar_source,
        "candidate_audit": audit.to_dict("records"),
        "features": list(ROTATION_FEATURES),
        "cutoff": contract["analysis_end_date"],
        "expected_capital": contract["expected_ending_capital"],
    }


def _score_margins(models, frames, symbols, dates, config, fold_id, phase):
    cache, _ = _precalcular_utilidades_modelo(models, frames, symbols, dates, config)
    scores = []
    for margin in config.rotation_switch_margin_candidates:
        margin = float(margin)
        policy = _politica_utilidade(
            {}, frames, symbols, config, margin, utility_cache=cache
        )
        score = float(_crescimento_politica_simples(policy, frames, symbols, dates, config))
        if not math.isfinite(score):
            raise RuntimeError(f"Score invalido fold={fold_id} fase={phase}")
        scores.append((margin, score))
    return scores, cache


def _build_fold_policy(models, frames, symbols, test_dates, config, selected_margin, fold_id):
    cache, _ = _precalcular_utilidades_modelo(models, frames, symbols, test_dates, config)
    effective = max(float(config.rotation_switch_margin), float(selected_margin))
    return _politica_utilidade(
        {}, frames, symbols, config, effective,
        fold_id=fold_id,
        calibrated_switch_margin=float(selected_margin),
        utility_cache=cache,
    )


def _simulate(name, policies, frames, symbols, decisions, mapping, metadata, config, folds):
    result = _simular_exato(
        name, _politica_agendada(policies, mapping), frames, symbols, decisions,
        config, calcular_taxas_referencia, aplicar_deslizamento,
        decision_metadata=metadata,
        model_label="LightGBM U75 - teste de hiperparametros",
        method_line="- Comparacao de configuracoes sobre o mesmo universo U75 congelado.",
    )
    metrics = summarize_metrics(result, folds, float(config.initial_capital))
    return result, metrics


def _cut_windows(fold, dates):
    a = int(fold["calibration_start_index"])
    b = int(fold["calibration_end_index"])
    calibration = dates[a:b]
    # 60 dias para buscar, dois de separacao, dias finais so para margem.
    opt_end = CALIBRATION_OPTIMIZE_DAYS
    margin_start = opt_end + CALIBRATION_GAP_DAYS
    if len(calibration) - margin_start < 40:
        raise RuntimeError("Calibracao nao permite separacao entre busca e margem")
    search_dates = calibration[:opt_end]
    margin_dates = calibration[margin_start:]
    train_dates = dates[:int(fold["train_end_index"])]
    final_dates = dates[:int(fold["final_fit_end_index"])]
    test_dates = pd.DatetimeIndex(fold["decision_dates"])
    validate_fold_windows(train_dates, search_dates, margin_dates, test_dates[1:])
    if not (max(train_dates) < min(search_dates) and max(margin_dates) < min(test_dates[1:])):
        raise RuntimeError("Vazamento temporal entre treino/calibracao/OOS")
    return train_dates, search_dates, margin_dates, final_dates, test_dates


def _run(root: Path) -> dict:
    frames, symbols, calendar, folds, decisions, mapping, metadata, config, identity = _load_u75()
    original_params = deepcopy(config.research_model_settings["lightgbm"])
    output = root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    print(f"[universe] U75 congelado; 52 atributos; 3 folds; snapshot={identity['snapshot_sha256']}",flush=True)

    # Políticas independentes, mesma referencia: nenhum dado OOS entra na busca.
    original_policies, matched_policies, optimized_policies = {}, {}, {}
    search_rows, margin_rows, feature_rows, selection_rows = [], [], [], []
    for fold in folds:
        fold_id = int(fold["fold_id"])
        train, search_dates, margin_dates, final_train, test_dates = _cut_windows(fold, calendar)
        print(f"[fold {fold_id}/3] treino={len(train)} busca={len(search_dates)} "
              f"margem={len(margin_dates)} teste={len(test_dates)-1}", flush=True)

        # Trilha original: 126 sessoes de calibracao, margens originais.
        full_calibration = calendar[int(fold['calibration_start_index']):int(fold['calibration_end_index'])]
        frozen_config = _config_with_params(config, original_params)
        frozen_cal = _ajustar_modelos_lightgbm(
            frames, symbols, train, frozen_config, phase=f"u75_f{fold_id}_frozen_cal"
        )
        full_scores, _ = _score_margins(frozen_cal, frames, symbols, full_calibration, frozen_config,
                                         fold_id, "original_126")
        full_selected = selecionar_switch_margin(full_scores)["selected_candidate_margin"]
        match_scores, _ = _score_margins(frozen_cal, frames, symbols, margin_dates, frozen_config,
                                          fold_id, "matched_margin")
        match_selected = selecionar_switch_margin(match_scores)["selected_candidate_margin"]
        del frozen_cal

        # Busca: trial 0 = config congelada, seguido de LHS e GP/EI.
        # Todas as provas de utilidade dos candidatos terminam ANTES do teste.
        history = []
        for trial in range(SEARCH.budget):
            point, params, method = propose(history, original_params, settings=SEARCH, fold_id=fold_id)
            candidate_config = _config_with_params(config, params)
            cal_models = _ajustar_modelos_lightgbm(
                frames, symbols, train, candidate_config,
                phase=f"u75_f{fold_id}_search_{trial+1}of{SEARCH.budget}",
            )
            scores, _ = _score_margins(cal_models, frames, symbols, search_dates,
                                       candidate_config, fold_id, "hyperparameter_search")
            del cal_models
            best_score = max(score for _, score in scores)
            row = {
                "fold_id": fold_id, "trial": trial, "method": method,
                "normalized": point.tolist(), "parameters": params,
                "objective": float(best_score),
                "best_search_margin": float(max(scores, key=lambda x: x[1])[0]),
                "fit_train_end": str(max(train)),
                "search_first_date": str(min(search_dates)),
                "search_last_date": str(max(search_dates)),
            }
            history.append(row)
            search_rows.append(row)
            pd.DataFrame(search_rows).drop(columns=["parameters","normalized"]).to_csv(
                output / "progresso_busca.csv", index=False
            )
            print(f"[busca] fold={fold_id} trial={trial+1}/{SEARCH.budget} "
                  f"metodo={method} utilidade={best_score:.6f}", flush=True)

        winner = choose_best(history)
        selected_params = winner["parameters"]
        chosen_config = _config_with_params(config, selected_params)
        print(f"[selecao] fold={fold_id} trial={winner['trial']} "
              f"utilidade_interna={winner['objective']:.6f}",flush=True)

        # Margem calibrada numa janela interna disjunta da usada na busca.
        chosen_cal = _ajustar_modelos_lightgbm(
            frames, symbols, train, chosen_config, phase=f"u75_f{fold_id}_chosen_margin"
        )
        chosen_scores, _ = _score_margins(chosen_cal, frames, symbols, margin_dates,
                                           chosen_config, fold_id, "chosen_margin")
        chosen_margin = selecionar_switch_margin(chosen_scores)["selected_candidate_margin"]
        del chosen_cal
        selection_rows.append({
            "fold_id": fold_id, "selected_trial": int(winner["trial"]),
            "selected_method": winner["method"],
            "optimized_parameters": selected_params,
            "original_parameters": original_params,
            "selection_objective": float(winner["objective"]),
            "original_full_margin": float(full_selected),
            "frozen_matched_margin": float(match_selected),
            "optimized_matched_margin": float(chosen_margin),
            "train_end": str(max(train)), "search_end": str(max(search_dates)),
            "margin_start": str(min(margin_dates)), "margin_end": str(max(margin_dates)),
            "test_start": str(min(test_dates[1:])), "test_end": str(max(test_dates[1:])),
        })
        for title, scores in (("original_126", full_scores),
                               ("frozen_matched", match_scores),
                               ("optimized_matched", chosen_scores)):
            for margin, value in scores:
                margin_rows.append({"fold_id": fold_id, "policy": title,
                                    "margin": margin, "score": value})

        # Refit ate o ponto final de treino do fold, com embargo temporal.
        for title, candidate_config, margin, dictionary in (
            ("original", frozen_config, full_selected, original_policies),
            ("frozen_matched", frozen_config, match_selected, matched_policies),
            ("optimized_matched", chosen_config, chosen_margin, optimized_policies),
        ):
            final_models = _ajustar_modelos_lightgbm(
                frames, symbols, final_train, candidate_config,
                phase=f"u75_f{fold_id}_{title}_final",
            )
            dictionary[fold_id] = _build_fold_policy(
                final_models, frames, symbols, test_dates, candidate_config, margin, fold_id,
            )
            for ticker, model in final_models.items():
                try:
                    gains = model.booster_.feature_importance(importance_type="gain")
                except (AttributeError, ValueError):
                    continue
                for name, gain in zip(ROTATION_FEATURES, gains):
                    feature_rows.append({"fold_id": fold_id, "policy": title,
                                         "symbol": ticker, "feature": name, "gain": float(gain)})
            del final_models

    print("[replay] mesmas sessoes, custos e regras de execucao para 3 politicas",flush=True)
    out_original, ref_metrics = _simulate(
        "u75_frozen_original", original_policies, frames, symbols, decisions, mapping,
        metadata, config, folds,
    )
    capital = float(ref_metrics["ending_capital"])
    if not math.isclose(capital, float(identity["expected_capital"]),
                        rel_tol=0.0, abs_tol=REFERENCE_TOLERANCE_USD):
        _write_json(output / "paridade_falhou.json", {
            "status": "aborted", "reason": "reference_checkpoint_mismatch",
            "expected": identity["expected_capital"], "observed": capital,
        })
        raise RuntimeError(
            f"Paridade U75 original falhou: {capital:,.8f} "
            f"vs {identity['expected_capital']:,.8f}; resultados experimentais bloqueados."
        )
    print(f"[paridade] Control U75 ORIGINAL exato: US$ {capital:,.2f}",flush=True)
    out_matched, matched_metrics = _simulate(
        "u75_frozen_matched", matched_policies, frames, symbols, decisions, mapping,
        metadata, config, folds,
    )
    out_optimized, optimized_metrics = _simulate(
        "u75_tuned_matched", optimized_policies, frames, symbols, decisions, mapping,
        metadata, config, folds,
    )

    trial_table = []
    for row in search_rows:
        trial_table.append({k:v for k,v in row.items() if k not in ("parameters","normalized")} |
                           {f"param_{k}":row["parameters"][k] for k in row["parameters"]
                            if k in ("n_estimators","learning_rate","max_depth","num_leaves",
                                      "min_child_samples","colsample_bytree","reg_alpha","reg_lambda")})
    pd.DataFrame(trial_table).to_csv(output / "hiperparametros_testados.csv", index=False)
    pd.DataFrame(margin_rows).to_csv(output / "calibracao_margens.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(output / "importancia_52_atributos.csv", index=False)
    pd.DataFrame([{k: (json.dumps(v,ensure_ascii=False) if isinstance(v,dict) else v)
                   for k,v in row.items()} for row in selection_rows]).to_csv(
        output / "hiperparametros_escolhidos_por_fold.csv", index=False,
    )
    for name, result in (("frozen_original",out_original),
                         ("frozen_matched",out_matched),
                         ("optimized_matched",out_optimized)):
        result.predictions.to_csv(output / f"predictions_{name}.csv", index=True,
                                  float_format="%.17g")
        if hasattr(result,"trades"):
            result.trades.to_csv(output / f"trades_{name}.csv",index=False)

    # Gráfico de patrimonio em log; sem construir métricas a partir de previsões.
    fig, ax = plt.subplots(figsize=(13, 5))
    for label, result in (("Congelado original",out_original),
                          ("Congelado calibração pareada",out_matched),
                          ("Reotimizado calibração pareada",out_optimized)):
        eq = result.predictions["strategy_equity"].astype(float)
        ax.plot(pd.to_datetime(result.predictions.index,utc=True), eq, label=label)
    ax.set_yscale("log")
    ax.set_title("U75: efeito da reotimizacao de hiperparametros")
    ax.set_ylabel("Patrimonio (US$, escala logaritmica)")
    ax.set_xlabel("Sessao de execucao")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "comparacao_patrimonio_u75.png", dpi=140)
    plt.close(fig)

    payload = {
        "status":"completed", "script_version": VERSION,
        "data": identity, "search": SEARCH.__dict__,
        "split_design": {
            "optimizer_sessions": CALIBRATION_OPTIMIZE_DAYS,
            "gap_sessions": CALIBRATION_GAP_DAYS,
            "margin_selection": "remaining chronological calibration sessions",
            "outer_test_used_for_search": False,
            "same_52_features": True,
        },
        "fold_selections": selection_rows,
        "results": {
            "frozen_original":ref_metrics,
            "frozen_matched":matched_metrics,
            "optimized_matched":optimized_metrics,
        },
        "paired_ending_capital_delta":float(optimized_metrics["ending_capital"])-float(matched_metrics["ending_capital"]),
        "caveats": [
            "Universo e seus dez novos ativos foram escolhidos retrospectivamente no historico inteiro.",
            "Os hiperparametros historicos foram escolhidos em dados que se sobrepoem a parte da avaliacao atual; resultados sao condicionais, nao validacao prospectiva independente.",
            "Dependencia futura de elegibilidade no motor congelado ainda nao foi eliminada; nao promover estrategia antes de auditoria causal.",
            "LHS+GP/EI reproduz a ideia bayesiana, nao o CARO proprietario historico em detalhe.",
            "Busca e margens usam as janelas anteriores aos seus folds de teste, sem ajuste pelo capital OOS.",
            "Todos os resultados derivam dos mesmos dados historicos selecionados; nao comprova lucratividade futura.",
        ],
    }
    _write_json(output / "resultado.json",payload)
    pack=output/f"pacote_hiperparametrizacao_u75_{output.name}.zip"
    with ZipFile(pack,"w",ZIP_DEFLATED) as archive:
        for file in sorted(output.iterdir()):
            if file.is_file() and file != pack and file.suffix in (".csv",".json",".png",".md"):
                archive.write(file,arcname=file.name)
    print("[CONCLUIDO] ",pack, flush=True)
    print(json.dumps({"frozen":ref_metrics["ending_capital"],
                      "matched":matched_metrics["ending_capital"],
                      "reoptimized":optimized_metrics["ending_capital"]},indent=2),flush=True)
    return payload


def main() -> dict:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        return _run(OUTPUT_ROOT)
    except Exception:
        (OUTPUT_ROOT/"ultimo_erro.txt").write_text(traceback.format_exc(),encoding="utf-8")
        print("[ERRO] relatorio: ",OUTPUT_ROOT/"ultimo_erro.txt",flush=True)
        raise


if __name__ == "__main__":
    main()
