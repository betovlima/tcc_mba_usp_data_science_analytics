"""Descoberta calibration-only da assinatura de contribuicao marginal.

Esta campanha nao usa o OOS para selecionar ativos. Ela mede, em cada janela
de calibracao cronologica, quanto cada ativo altera:
- a superficie de score dos candidatos congelados de switch margin;
- a margem otima escolhida pela politica;
- o score de calibracao do universo;
- a competicao cross-sectional dos scores LightGBM.

CLMT e o caso positivo motivador. DOC permanece apenas como controle negativo
estrutural. Nenhum score composto arbitrario e criado nesta etapa.
"""

# %% 0 - Imports e configuracao
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from engine.configuracao import CONFIG
from engine.modelo_lightgbm import (
    _ajustar_modelos_lightgbm,
    _selecionar_switch_margin_fold,
)
from engine.rotacao import (
    _construir_folds_walk_forward,
    _crescimento_politica_simples,
    _politica_utilidade,
    _precalcular_utilidades_modelo,
    preparar_painel_rotacao,
)
from pesquisas.directional_change_lightgbm import (
    EXPECTED_EXECUTION_SCHEMA,
    RESEARCH_VERSION,
    criar_pacote_analise,
    sinal_sonoro_conclusao,
)
from reproducao.dados import SnapshotPaths, validate_snapshot
from reproducao.experimento import build_variant_configs
from reproducao.preparacao import prepare_model_frames


RAIZ_PROJETO = Path(__file__).resolve().parent
CAMINHOS = SnapshotPaths.research(RAIZ_PROJETO)
DIRETORIO_RESULTADOS = RAIZ_PROJETO / "output" / "directional_change"
EXECUTION_SCHEMA = "rotation-contribution-signature-loo-v1"


# %% 1 - Guards e snapshot
if EXECUTION_SCHEMA != EXPECTED_EXECUTION_SCHEMA:
    raise RuntimeError(
        "Script e modulo de pesquisa incompatíveis antes do replay: "
        f"script={EXECUTION_SCHEMA!r} "
        f"modulo={EXPECTED_EXECUTION_SCHEMA!r}. "
        "Atualize a branch e reinicie o kernel do Spyder."
    )

print("=" * 78, flush=True)
print("TCC - Rotation Contribution Signature LOO", flush=True)
print(f"versao_pesquisa={RESEARCH_VERSION}", flush=True)
print(f"execution_schema={EXECUTION_SCHEMA}", flush=True)
print(f"script_path={Path(__file__).resolve()}", flush=True)
print("dados=SNAPSHOT_CONGELADO_VERSIONADO", flush=True)
print("selecao=CALIBRATION_ONLY | OOS_NOT_USED_FOR_SELECTION", flush=True)
print("=" * 78, flush=True)

manifesto = validate_snapshot(CAMINHOS)


# %% 2 - Universos diagnosticos U55 e U56
raw_u55, excl_u55, diag_u55, audit_u55 = prepare_model_frames(
    CAMINHOS,
    assets=CONFIG.assets,
    comparar_snapshot_referencia=True,
    allow_structural_assets=frozenset({"CLMT"}),
)
raw_u56, excl_u56, diag_u56, audit_u56 = prepare_model_frames(
    CAMINHOS,
    assets=CONFIG.assets,
    comparar_snapshot_referencia=False,
    allow_structural_assets=frozenset({"CLMT", "DOC"}),
)

if len(raw_u55) != 55:
    raise RuntimeError(f"U55 deveria ter 55 ativos; obtidos {len(raw_u55)}.")
if len(raw_u56) != 56:
    raise RuntimeError(f"U56 deveria ter 56 ativos; obtidos {len(raw_u56)}.")

config_u55, _ = build_variant_configs(raw_u55, CONFIG)
config_u56, _ = build_variant_configs(raw_u56, CONFIG)

frames_u55, dates_u55, calendar_u55 = preparar_painel_rotacao(
    raw_u55,
    config_u55,
)
frames_u56, dates_u56, calendar_u56 = preparar_painel_rotacao(
    raw_u56,
    config_u56,
)
folds_u55 = _construir_folds_walk_forward(dates_u55, config_u55)
folds_u56 = _construir_folds_walk_forward(dates_u56, config_u56)

symbols_u55 = sorted(frames_u55)
symbols_u56 = sorted(frames_u56)
candidate_margins = tuple(
    float(value)
    for value in config_u55.rotation_switch_margin_candidates
)

if candidate_margins != tuple(
    float(value)
    for value in config_u56.rotation_switch_margin_candidates
):
    raise RuntimeError("U55 e U56 nao usam o mesmo conjunto de margens.")


# %% 3 - Funcoes de diagnostico calibration-only

def _surface(
    models,
    frames,
    symbols,
    calibration_dates,
    config,
    *,
    fold_id,
    universe,
    excluded_asset,
):
    rows = []
    candidate_scores = []
    for margin in candidate_margins:
        policy = _politica_utilidade(
            models,
            frames,
            symbols,
            config,
            float(margin),
        )
        score = _crescimento_politica_simples(
            policy,
            frames,
            symbols,
            calibration_dates,
            config,
        )
        candidate_scores.append((float(margin), float(score)))
        rows.append(
            {
                "universe": universe,
                "fold_id": int(fold_id),
                "excluded_asset": excluded_asset,
                "switch_margin": float(margin),
                "calibration_risk_adjusted_score": float(score),
            }
        )

    selection = _selecionar_switch_margin_fold(
        config,
        int(fold_id),
        candidate_scores,
    )
    return rows, selection


def _score_lookup(rows):
    return {
        float(row["switch_margin"]): float(
            row["calibration_risk_adjusted_score"]
        )
        for row in rows
    }


def _selected_share(
    models,
    frames,
    symbols,
    calibration_dates,
    config,
    margin,
):
    policy = _politica_utilidade(
        models,
        frames,
        symbols,
        config,
        float(margin),
    )
    counts = {symbol: 0 for symbol in symbols}
    position = 0
    holding = 0
    total = max(0, len(calibration_dates) - 1)

    for timestamp in calibration_dates[:-1]:
        action, _ = policy(timestamp, position, holding)
        if action > 0:
            counts[symbols[action - 1]] += 1
        if action == position:
            holding = holding + 1 if action > 0 else 0
        else:
            position = int(action)
            holding = 1 if action > 0 else 0

    if total <= 0:
        return {symbol: 0.0 for symbol in symbols}
    return {
        symbol: float(count / total)
        for symbol, count in counts.items()
    }


def _score_fingerprints(
    models,
    frames,
    symbols,
    calibration_dates,
    config,
):
    cache, _ = _precalcular_utilidades_modelo(
        models,
        frames,
        symbols,
        calibration_dates,
        config,
    )
    values = {
        symbol: {
            "scores": [],
            "ranks": [],
            "top1": 0,
            "top3": 0,
            "valid": 0,
        }
        for symbol in symbols
    }

    for timestamp in calibration_dates:
        utilities = cache.get(pd.Timestamp(timestamp))
        if utilities is None:
            continue
        asset_scores = np.asarray(utilities[1:], dtype=float)
        finite_positions = [
            index
            for index, value in enumerate(asset_scores)
            if np.isfinite(value)
        ]
        if not finite_positions:
            continue
        ranked = sorted(
            finite_positions,
            key=lambda index: (
                -float(asset_scores[index]),
                symbols[index],
            ),
        )
        rank_by_position = {
            position: rank
            for rank, position in enumerate(ranked, start=1)
        }

        for position in finite_positions:
            symbol = symbols[position]
            value = float(asset_scores[position])
            rank = int(rank_by_position[position])
            item = values[symbol]
            item["scores"].append(value)
            item["ranks"].append(rank)
            item["valid"] += 1
            if rank == 1:
                item["top1"] += 1
            if rank <= 3:
                item["top3"] += 1

    output = {}
    asset_count = max(1, len(symbols))
    for symbol, item in values.items():
        scores = np.asarray(item["scores"], dtype=float)
        ranks = np.asarray(item["ranks"], dtype=float)
        valid = int(item["valid"])
        output[symbol] = {
            "valid_score_sessions": valid,
            "score_mean": (
                float(np.mean(scores)) if valid else None
            ),
            "score_std": (
                float(np.std(scores)) if valid else None
            ),
            "positive_score_share": (
                float(np.mean(scores > 0.0)) if valid else None
            ),
            "top1_share": (
                float(item["top1"] / valid) if valid else None
            ),
            "top3_share": (
                float(item["top3"] / valid) if valid else None
            ),
            "mean_rank": (
                float(np.mean(ranks)) if valid else None
            ),
            "mean_rank_percentile": (
                float(
                    1.0
                    - (np.mean(ranks) - 1.0)
                    / max(1.0, asset_count - 1.0)
                )
                if valid
                else None
            ),
        }
    return output


def _loo_rows_for_fold(
    *,
    universe,
    frames,
    symbols,
    models,
    calibration_dates,
    config,
    fold_id,
    assets_to_ablate,
):
    surface_rows = []
    signature_rows = []

    full_surface, full_selection = _surface(
        models,
        frames,
        symbols,
        calibration_dates,
        config,
        fold_id=fold_id,
        universe=universe,
        excluded_asset=None,
    )
    surface_rows.extend(full_surface)
    full_lookup = _score_lookup(full_surface)
    full_margin = float(
        full_selection["selected_candidate_margin"]
    )
    full_best_score = float(
        full_selection["selected_calibration_score"]
    )

    fingerprints = _score_fingerprints(
        models,
        frames,
        symbols,
        calibration_dates,
        config,
    )
    selected_share = _selected_share(
        models,
        frames,
        symbols,
        calibration_dates,
        config,
        full_margin,
    )

    for asset in assets_to_ablate:
        loo_symbols = [
            symbol for symbol in symbols
            if symbol != asset
        ]
        loo_models = {
            symbol: models[symbol]
            for symbol in loo_symbols
            if symbol in models
        }
        loo_frames = {
            symbol: frames[symbol]
            for symbol in loo_symbols
        }

        loo_surface, loo_selection = _surface(
            loo_models,
            loo_frames,
            loo_symbols,
            calibration_dates,
            config,
            fold_id=fold_id,
            universe=universe,
            excluded_asset=asset,
        )
        surface_rows.extend(loo_surface)
        loo_lookup = _score_lookup(loo_surface)
        loo_margin = float(
            loo_selection["selected_candidate_margin"]
        )
        loo_best_score = float(
            loo_selection["selected_calibration_score"]
        )

        full_at_loo_margin = float(full_lookup[loo_margin])
        loo_at_full_margin = float(loo_lookup[full_margin])
        fixed_margin_contribution = (
            float(full_lookup[full_margin])
            - loo_at_full_margin
        )
        optimized_marginal_score = (
            full_best_score - loo_best_score
        )
        flip_strength = (
            (full_best_score - full_at_loo_margin)
            + (loo_best_score - loo_at_full_margin)
            if full_margin != loo_margin
            else 0.0
        )

        fp = fingerprints.get(asset, {})
        signature_rows.append(
            {
                "universe": universe,
                "fold_id": int(fold_id),
                "asset": asset,
                "role": (
                    "positive_case"
                    if asset == "CLMT"
                    else (
                        "negative_control"
                        if asset == "DOC"
                        else "universe_asset"
                    )
                ),
                "full_best_margin": full_margin,
                "loo_best_margin": loo_margin,
                "margin_changed": bool(
                    abs(full_margin - loo_margin) > 1e-12
                ),
                "full_best_score": full_best_score,
                "loo_best_score": loo_best_score,
                "optimized_marginal_score": float(
                    optimized_marginal_score
                ),
                "fixed_full_margin_contribution": float(
                    fixed_margin_contribution
                ),
                "margin_flip_strength": float(flip_strength),
                "selected_share_at_full_margin": float(
                    selected_share.get(asset, 0.0)
                ),
                **fp,
            }
        )

    return signature_rows, surface_rows


# %% 4 - U55 leave-one-out por fold
signature_rows = []
surface_rows = []
started = time.perf_counter()

for fold_position, fold in enumerate(folds_u55, start=1):
    fold_id = int(fold["fold_id"])
    train_dates = dates_u55[: int(fold["train_end_index"])]
    calibration_dates = dates_u55[
        int(fold["calibration_start_index"]):
        int(fold["calibration_end_index"])
    ]

    print(
        f"[loo] U55 fold={fold_id} "
        f"training_models={len(symbols_u55)}",
        flush=True,
    )
    models = _ajustar_modelos_lightgbm(
        frames_u55,
        symbols_u55,
        train_dates,
        config_u55,
        phase=f"signature_u55_fold_{fold_id}",
        technical_log_callback=lambda message: print(
            f"[technical] {message}",
            flush=True,
        ),
    )

    fold_signature, fold_surface = _loo_rows_for_fold(
        universe="u55_clmt",
        frames=frames_u55,
        symbols=symbols_u55,
        models=models,
        calibration_dates=calibration_dates,
        config=config_u55,
        fold_id=fold_id,
        assets_to_ablate=symbols_u55,
    )
    signature_rows.extend(fold_signature)
    surface_rows.extend(fold_surface)

    clmt_row = next(
        row for row in fold_signature
        if row["asset"] == "CLMT"
    )
    print(
        f"[loo-result] fold={fold_id} CLMT "
        f"margin={clmt_row['full_best_margin']:.4f}->"
        f"{clmt_row['loo_best_margin']:.4f} "
        f"marginal_score="
        f"{clmt_row['optimized_marginal_score']:+.8f} "
        f"selected_share="
        f"{clmt_row['selected_share_at_full_margin']:.2%}",
        flush=True,
    )


# %% 5 - DOC como controle negativo U56
for fold_position, fold in enumerate(folds_u56, start=1):
    fold_id = int(fold["fold_id"])
    train_dates = dates_u56[: int(fold["train_end_index"])]
    calibration_dates = dates_u56[
        int(fold["calibration_start_index"]):
        int(fold["calibration_end_index"])
    ]

    print(
        f"[loo] U56 DOC-control fold={fold_id} "
        f"training_models={len(symbols_u56)}",
        flush=True,
    )
    models = _ajustar_modelos_lightgbm(
        frames_u56,
        symbols_u56,
        train_dates,
        config_u56,
        phase=f"signature_u56_doc_fold_{fold_id}",
        technical_log_callback=lambda message: print(
            f"[technical] {message}",
            flush=True,
        ),
    )

    fold_signature, fold_surface = _loo_rows_for_fold(
        universe="u56_raw_doc_control",
        frames=frames_u56,
        symbols=symbols_u56,
        models=models,
        calibration_dates=calibration_dates,
        config=config_u56,
        fold_id=fold_id,
        assets_to_ablate=["DOC"],
    )
    signature_rows.extend(fold_signature)
    surface_rows.extend(fold_surface)

    doc_row = fold_signature[0]
    print(
        f"[loo-result] fold={fold_id} DOC "
        f"margin={doc_row['full_best_margin']:.4f}->"
        f"{doc_row['loo_best_margin']:.4f} "
        f"marginal_score="
        f"{doc_row['optimized_marginal_score']:+.8f} "
        f"selected_share="
        f"{doc_row['selected_share_at_full_margin']:.2%}",
        flush=True,
    )


# %% 6 - Agregacao sem score composto arbitrario
signature = pd.DataFrame(signature_rows)
surface = pd.DataFrame(surface_rows)

u55_signature = signature.loc[
    signature["universe"] == "u55_clmt"
].copy()

aggregate = (
    u55_signature.groupby(["asset", "role"], as_index=False)
    .agg(
        fold_count=("fold_id", "nunique"),
        margin_flip_count=("margin_changed", "sum"),
        positive_marginal_score_folds=(
            "optimized_marginal_score",
            lambda values: int((pd.Series(values) > 0.0).sum()),
        ),
        negative_marginal_score_folds=(
            "optimized_marginal_score",
            lambda values: int((pd.Series(values) < 0.0).sum()),
        ),
        mean_optimized_marginal_score=(
            "optimized_marginal_score",
            "mean",
        ),
        mean_fixed_full_margin_contribution=(
            "fixed_full_margin_contribution",
            "mean",
        ),
        mean_margin_flip_strength=(
            "margin_flip_strength",
            "mean",
        ),
        mean_selected_share=(
            "selected_share_at_full_margin",
            "mean",
        ),
        mean_top1_share=("top1_share", "mean"),
        mean_top3_share=("top3_share", "mean"),
        mean_positive_score_share=(
            "positive_score_share",
            "mean",
        ),
        mean_rank_percentile=(
            "mean_rank_percentile",
            "mean",
        ),
    )
)

aggregate = aggregate.sort_values(
    [
        "margin_flip_count",
        "mean_optimized_marginal_score",
        "mean_fixed_full_margin_contribution",
    ],
    ascending=[False, False, False],
).reset_index(drop=True)

print(
    "[signature] assets="
    f"{len(aggregate)} "
    f"margin_flippers="
    f"{int((aggregate['margin_flip_count'] > 0).sum())}",
    flush=True,
)
print(
    aggregate.head(15).to_string(index=False),
    flush=True,
)


# %% 7 - Exportacao e pacote
DIRETORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
for antigo in DIRETORIO_RESULTADOS.glob("*.csv"):
    antigo.unlink()
for antigo in DIRETORIO_RESULTADOS.glob("*.json"):
    antigo.unlink()
for antigo in DIRETORIO_RESULTADOS.glob("*.png"):
    antigo.unlink()
graficos = DIRETORIO_RESULTADOS / "graficos"
if graficos.exists():
    for antigo in graficos.glob("*.png"):
        antigo.unlink()

signature.to_csv(
    DIRETORIO_RESULTADOS / "rotation_contribution_signature_loo.csv",
    index=False,
)
aggregate.to_csv(
    DIRETORIO_RESULTADOS / "rotation_contribution_signature_aggregate.csv",
    index=False,
)
surface.to_csv(
    DIRETORIO_RESULTADOS / "rotation_contribution_margin_surface.csv",
    index=False,
)

clmt_rows = signature.loc[signature["asset"] == "CLMT"]
doc_rows = signature.loc[signature["asset"] == "DOC"]

payload = {
    "research_version": RESEARCH_VERSION,
    "execution_schema": EXECUTION_SCHEMA,
    "snapshot_sha256": manifesto.get("snapshot_sha256"),
    "question": (
        "Which assets have a calibration-only marginal contribution profile "
        "capable of changing the rotation policy, without using OOS capital "
        "as an asset-selection criterion?"
    ),
    "protocol": {
        "selection_uses_oos": False,
        "oos_backtest_executed": False,
        "snapshot_unchanged": True,
        "lightgbm_parameters_unchanged": True,
        "fold_method_unchanged": True,
        "switch_margin_candidates": list(candidate_margins),
        "u55_role": (
            "diagnostic universe with CLMT restored; leave-one-out discovery"
        ),
        "u56_role": (
            "DOC negative control only; no promotion to scientific baseline"
        ),
        "model_retraining_for_loo": False,
        "model_retraining_note": (
            "LightGBM models are asset-specific. Each fold is trained once; "
            "leave-one-out removes only the candidate from the calibration "
            "choice set, leaving other asset models unchanged."
        ),
        "no_composite_signature_score": True,
    },
    "u55": {
        "assets": symbols_u55,
        "calendar_source_asset": calendar_u55,
        "fold_count": len(folds_u55),
        "data_audit": audit_u55,
        "structural_exclusions": excl_u55,
    },
    "u56_doc_control": {
        "assets": symbols_u56,
        "calendar_source_asset": calendar_u56,
        "fold_count": len(folds_u56),
        "data_audit": audit_u56,
        "structural_exclusions": excl_u56,
    },
    "special_cases": {
        "CLMT": clmt_rows.to_dict(orient="records"),
        "DOC": doc_rows.to_dict(orient="records"),
    },
    "aggregate_rows": aggregate.to_dict(orient="records"),
    "runtime_seconds": float(time.perf_counter() - started),
    "interpretation_rule": (
        "An asset is policy-influential when its leave-one-out removal changes "
        "the calibration-optimal margin and/or materially reduces the frozen "
        "calibration objective. This campaign is discovery-only. OOS capital "
        "must not be used to rank or tune the signature."
    ),
}

with (
    DIRETORIO_RESULTADOS / "rotation_contribution_signature_loo.json"
).open("w", encoding="utf-8") as arquivo:
    json.dump(
        payload,
        arquivo,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

PACOTE_ANALISE = criar_pacote_analise(DIRETORIO_RESULTADOS)
print(f"[package] pronto={PACOTE_ANALISE}", flush=True)
sinal_sonoro_conclusao()
print(
    "[done] signature LOO calibration-only concluida "
    f"seconds={time.perf_counter() - started:.3f}",
    flush=True,
)
