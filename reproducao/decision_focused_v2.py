"""Pesquisa v1.3.0-dev.3: OOF temporal, estados visitados e guardas calibradas.

Mantem o Control/Soft e o experimento v1 preservados. OOF aqui significa
predicoes da politica-base em datas posteriores ao treino do respectivo
modelo-base, com purge dos rotulos de horizonte 60. Nenhum rotulo do OOS do
fold externo participa de ajuste, augmentacao ou calibracao. O ranking e um
surrogate decision-focused, nao DFL diferenciavel fim a fim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from engine.execucao import aplicar_deslizamento, calcular_taxas_referencia
from engine.modelo_lightgbm import (
    _ajustar_modelos_lightgbm,
    _construir_contexto_execucao,
)
from engine.rotacao import (
    _crescimento_politica_simples,
    _politica_agendada,
    _politica_utilidade,
    _precalcular_utilidades_modelo,
    _simular_exato,
)
from reproducao.decision_focused import (
    FEATURE_COLUMNS,
    HORIZONTE_CONTRAFACTUAL,
    SEMENTE,
    _atributos,
    _candidatos,
    _escolher_acao,
    _gerar_rotulos,
    _treinar,
)
from reproducao.dfl_softmax import escolher_softmax, treinar_softmax_dfl
from reproducao.auditoria_intervencoes import (
    AuditoriaIntervencoes, auditar_intervencoes,
)
from reproducao.experimento import summarize_metrics

RESEARCH_VERSION = "1.3.0-dev.3"
INNER_MIN_TRAIN_SESSIONS = 620
INNER_MIN_MODEL_ROWS = 250
INNER_PURGE_SESSIONS = 60
INNER_BLOCK_SESSIONS = 126
VALIDATION_FRACTION = 0.25
CALIBRATION_QUANTILE = 0.90


@dataclass
class SegmentoOOF:
    fold_id: int
    train_end: pd.Timestamp
    dates: pd.DatetimeIndex
    policy: Callable
    utility_cache: dict[pd.Timestamp, np.ndarray]


@dataclass
class PesquisaV2:
    regression_result: Any
    regression_metrics: dict[str, Any]
    dfl_result: Any
    dfl_metrics: dict[str, Any]
    softmax_result: Any
    softmax_metrics: dict[str, Any]
    labels: pd.DataFrame
    audit: pd.DataFrame
    calibration: pd.DataFrame
    regression_decisions: pd.DataFrame
    dfl_decisions: pd.DataFrame
    softmax_decisions: pd.DataFrame
    auditoria_intervencoes: AuditoriaIntervencoes


def _parametros_internos(config: Any) -> Any:
    """Apenas modelos-base internos; Control externo nao e alterado."""
    return config.copiar_modelo(update={
        "rotation_minimum_training_rows": INNER_MIN_MODEL_ROWS,
    })


def _gerar_oof(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
    fold: dict[str, Any],
    config: Any,
    horizonte: int,
) -> tuple[pd.DataFrame, list[SegmentoOOF], list[dict[str, Any]]]:
    """Treina modelos-base em passado, prediz/blinda rotulos em bloco futuro."""
    fold_id = int(fold["fold_id"])
    limit = int(fold["final_fit_end_index"])
    initial = INNER_MIN_TRAIN_SESSIONS + INNER_PURGE_SESSIONS
    inner_config = _parametros_internos(config)
    labels: list[pd.DataFrame] = []
    segments: list[SegmentoOOF] = []
    audit: list[dict[str, Any]] = []
    start = initial
    while start + horizonte < limit:
        train_end = start - INNER_PURGE_SESSIONS
        # O fim de train_end e exclusivo; labels de 60d nao tocam bloco OOF.
        if train_end < INNER_MIN_TRAIN_SESSIONS:
            raise AssertionError("Treino interno insuficiente")
        end_inclusive = min(start + INNER_BLOCK_SESSIONS, limit - 1)
        chunk = dates[start:end_inclusive + 1]
        if len(chunk) < horizonte + 2:
            break
        models = _ajustar_modelos_lightgbm(
            frames, symbols, dates[:train_end], inner_config,
            phase=f"decision_focused_v2_fold_{fold_id}_train_end_{train_end}",
        )
        if not models:
            raise RuntimeError(
                f"Fold {fold_id}: nenhum modelo interno treinado em {train_end}"
            )
        cache, _ = _precalcular_utilidades_modelo(
            models, frames, symbols, chunk, inner_config,
        )
        # Margem fixa predeclarada; nao escolhe margem olhando o proprio bloco.
        policy = _politica_utilidade(
            models, frames, symbols, inner_config,
            float(config.rotation_switch_margin), utility_cache=cache,
        )
        generated = _gerar_rotulos(
            frames, symbols, chunk, policy, cache, inner_config,
            fold_id, horizonte, state_source="OOF_CONTROL",
        )
        if not (generated["outcome_end"] < fold["test_start"]).all():
            raise AssertionError("Rotulos internos ultrapassam o teste externo")
        if not (generated["outcome_end"] <= dates[limit - 1]).all():
            raise AssertionError("Rotulos internos ultrapassam o treino externo")
        if not (generated["decision_date"] > dates[train_end - 1]).all():
            raise AssertionError("Features OOF nao sao posteriores ao treino")
        labels.append(generated)
        segments.append(SegmentoOOF(
            fold_id, dates[train_end - 1], chunk, policy, cache,
        ))
        audit.append({
            "fold_id": fold_id, "inner_train_end": dates[train_end - 1],
            "inner_oof_start": chunk[0], "inner_oof_end": chunk[-1],
            "inner_purge_sessions": INNER_PURGE_SESSIONS,
            "inner_minimum_training_rows": INNER_MIN_MODEL_ROWS,
            "trained_models": len(models),
            "label_rows": len(generated),
            "label_dates": int(generated["decision_date"].nunique()),
            "outcome_end_max": generated["outcome_end"].max(),
            "outer_final_fit_end": dates[limit - 1],
            "outer_oos_start": fold["test_start"],
        })
        print(
            f"[decision-focused-v2] fold={fold_id} inner_end={dates[train_end - 1]} "
            f"oof={chunk[0]}..{chunk[-1]} labels={len(generated)}",
            flush=True,
        )
        start = end_inclusive + 1
    if not labels:
        raise ValueError(f"Fold {fold_id}: nenhuma janela OOF interna valida")
    full = pd.concat(labels, ignore_index=True)
    if full["decision_date"].duplicated().all():
        raise AssertionError("Janela OOF sem datas distintas")
    return full, segments, audit


def _divisao_temporal(
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Separa ajuste de guarda e purga outcomes que tocariam a validacao."""
    dates = pd.DatetimeIndex(sorted(labels["decision_date"].unique()))
    if len(dates) < 45:
        raise ValueError("OOF insuficiente para calibracao temporal independente")
    validation_index = max(1, int(len(dates) * (1.0 - VALIDATION_FRACTION)))
    start = dates[validation_index]
    train = labels.loc[labels["outcome_end"] < start].copy()
    validation = labels.loc[labels["decision_date"] >= start].copy()
    if train.empty or validation.empty:
        raise ValueError("Divisao temporal deixou conjunto vazio")
    if train["outcome_end"].max() >= validation["decision_date"].min():
        raise AssertionError("Overlap de outcomes com a validacao")
    if train["decision_date"].max() >= validation["decision_date"].min():
        raise AssertionError("Datas treino/validacao sobrepostas")
    return train, validation, start


def _treinar_piloto(prefix: pd.DataFrame) -> Any:
    """Modelo apenas para visitar estados de uma janela posterior, no treino."""
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(
        n_estimators=120, learning_rate=0.035, max_depth=3,
        num_leaves=7, min_child_samples=8, random_state=SEMENTE,
        n_jobs=1, verbosity=-1, deterministic=True, force_col_wise=True,
    )
    model.fit(
        prefix.loc[:, FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan),
        prefix["log_advantage"].to_numpy(dtype=float),
    )
    return model


def _estados_da_politica_aprendida(
    pilot: Any, segment: SegmentoOOF,
    frames: dict[str, pd.DataFrame], symbols: list[str],
    config: Any,
) -> list[tuple[int, int, int]]:
    """Trajetoria sequencial em bloco posterior ao treino do piloto."""
    states = []
    position, holding = 0, 0
    for now in segment.dates[:-1]:
        base, _ = segment.policy(now, position, holding)
        utilities = segment.utility_cache[pd.Timestamp(now)]
        candidates = _candidatos(utilities, position, int(base), holding, config)
        if len(candidates) > 1:
            frame = pd.DataFrame([
                {
                    "candidate_position": action,
                    **_atributos(
                        frames, symbols, now, position, int(base), action,
                        holding, utilities, config,
                    ),
                }
                for action in candidates
            ])
            selected, _ = _escolher_acao("REGRESSION", pilot, frame, int(base))
        else:
            selected = int(base)
        states.append((position, holding, int(base)))
        if selected == position:
            holding = holding + 1 if position > 0 else 0
        else:
            position, holding = int(selected), (1 if selected > 0 else 0)
    return states


def _aumentar_estados(
    original_train: pd.DataFrame,
    segments: list[SegmentoOOF],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    config: Any,
    horizonte: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Piloto cronologico gera estados em futuro do proprio conjunto de treino.

    Rotulos OOS finais e janela de calibracao nunca participam desta etapa.
    Evita tratar alternativas na mesma data como observacoes independentes.
    """
    unique_dates = pd.DatetimeIndex(sorted(original_train["decision_date"].unique()))
    if len(unique_dates) < 70:
        return original_train, {"augmented_rows": 0, "reason": "short_training_history"}
    pilot_cut = unique_dates[len(unique_dates) // 2]
    prefix = original_train.loc[original_train["outcome_end"] < pilot_cut]
    if prefix["decision_date"].nunique() < 20:
        return original_train, {"augmented_rows": 0, "reason": "short_pilot_prefix"}
    pilot = _treinar_piloto(prefix)
    augmentation: list[pd.DataFrame] = []
    last_allowed = unique_dates[-1]
    for segment in segments:
        eligible = segment.dates[
            (segment.dates >= pilot_cut) & (segment.dates <= last_allowed)
        ]
        if len(eligible) < horizonte + 2:
            continue
        # O segmento e cortado ao periodo permitido de ajuste; nenhum
        # outcome de augmentacao ultrapassa a ultima data de treino.
        truncated = SegmentoOOF(
            segment.fold_id, segment.train_end,
            pd.DatetimeIndex(eligible), segment.policy, segment.utility_cache,
        )
        states = _estados_da_politica_aprendida(
            pilot, truncated, frames, symbols, config,
        )
        labels = _gerar_rotulos(
            frames, symbols, truncated.dates, truncated.policy,
            truncated.utility_cache, config, segment.fold_id,
            horizonte, states_override=states,
            state_source="OOF_LEARNED_STATE",
        )
        if not (labels["outcome_end"] <= last_allowed).all():
            raise AssertionError("Augmentacao excedeu ultimo treino")
        augmentation.append(labels)
    if not augmentation:
        return original_train, {"augmented_rows": 0, "reason": "no_valid_later_segment"}
    extra = pd.concat(augmentation, ignore_index=True)
    final = pd.concat([original_train, extra], ignore_index=True)
    return final, {
        "augmented_rows": int(len(extra)),
        "augmented_dates": int(extra["decision_date"].nunique()),
        "pilot_end_exclusive": pilot_cut,
        "pilot_training_dates": int(prefix["decision_date"].nunique()),
        "reason": "chronological_pilot_and_later_on_policy_states",
    }


def _regression_choice(
    model: Any, group: pd.DataFrame, control: int,
) -> tuple[int, float]:
    x = group.loc[:, FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    positions = group["candidate_position"].to_numpy(dtype=int)
    baseline = int(np.flatnonzero(positions == control)[0])
    predicted = np.asarray(model.predict(x), dtype=float)
    best = int(np.argmax(predicted))
    return int(positions[best]), float(predicted[best] - predicted[baseline])


def _pairwise_probability(
    model: Any, group: pd.DataFrame, selected: int, control: int,
) -> float:
    x = group.loc[:, FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    positions = group["candidate_position"].to_numpy(dtype=int)
    baseline = int(np.flatnonzero(positions == control)[0])
    candidate = int(np.flatnonzero(positions == selected)[0])
    difference = x.iloc[[candidate]].to_numpy() - x.iloc[[baseline]].to_numpy()
    return float(model.predict_proba(
        pd.DataFrame(difference, columns=FEATURE_COLUMNS)
    )[0, 1])


def _quantile_higher(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Nao ha observacoes para calibrar")
    return float(np.quantile(np.asarray(values, dtype=float), q, method="higher"))


def _calibrar_guardas(
    regression: Any, dfl: Any, softmax: Any, validation: pd.DataFrame,
) -> dict[str, Any]:
    """Calibra somente na validacao temporal anterior ao OOS externo.

    Regressao usa erro maximo de otimismo por data em relacao ao Control.
    Ranking usa quantil do score dos candidatos que perderam para Control.
    A cobertura nao e garantia conformal sob dependencia temporal.
    """
    optimism: list[float] = []
    unsafe_probabilities: list[float] = []
    unsafe_softmax_margins: list[float] = []
    groups = 0
    for _, group in validation.groupby("decision_date", sort=True):
        control = int(group["control_position"].iloc[0])
        positions = group["candidate_position"].to_numpy(dtype=int)
        if control not in positions or len(group) < 2:
            continue
        chosen, _ = _regression_choice(regression, group, control)
        del chosen
        x = group.loc[:, FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
        predicted = np.asarray(regression.predict(x), dtype=float)
        base_index = int(np.flatnonzero(positions == control)[0])
        true = group["log_advantage"].to_numpy(dtype=float)
        optimism.append(float(np.max(
            (predicted - predicted[base_index]) - true,
        )))
        dfl_candidate, _ = _escolher_acao("DFL", dfl, group, control)
        if dfl_candidate != control:
            true_advantage = float(
                group.loc[group["candidate_position"] == dfl_candidate,
                          "log_advantage"].iloc[0]
            )
            p = _pairwise_probability(
                dfl, group, dfl_candidate, control,
            )
            if true_advantage <= 0:
                unsafe_probabilities.append(p)
        softmax_candidate, softmax_delta = escolher_softmax(
            softmax, group, control,
        )
        if softmax_candidate != control:
            softmax_true = float(
                group.loc[group["candidate_position"] == softmax_candidate,
                          "log_advantage"].iloc[0]
            )
            if softmax_true <= 0:
                unsafe_softmax_margins.append(softmax_delta)
        groups += 1
    if groups < 20:
        raise ValueError(f"Apenas {groups} datas independentes de validacao")
    # Se nenhum erro negativo foi observado, nunca liberar override em
    # base a ausencia de evidencia. Requer nova validacao, nao OOS tuning.
    reg_penalty = max(0.0, _quantile_higher(
        optimism, CALIBRATION_QUANTILE,
    ))
    dfl_threshold = (
        max(0.5, _quantile_higher(
            unsafe_probabilities, CALIBRATION_QUANTILE,
        ))
        if unsafe_probabilities else 1.0
    )
    softmax_threshold = (
        max(0.0, _quantile_higher(
            unsafe_softmax_margins, CALIBRATION_QUANTILE,
        )) if unsafe_softmax_margins else float("inf")
    )
    return {
        "regression_optimism_margin": reg_penalty,
        "dfl_unsafe_probability_threshold": dfl_threshold,
        "softmax_unsafe_margin_threshold": softmax_threshold,
        "validation_unsafe_softmax_candidates": len(unsafe_softmax_margins),
        "validation_decision_dates": groups,
        "validation_unsafe_dfl_candidates": len(unsafe_probabilities),
        "calibration_quantile": CALIBRATION_QUANTILE,
        "calibration_role": "prior_temporal_oof_only",
    }


def _criar_politica_v2(
    *,
    fold_id: int, variant: str, model: Any,
    baseline: Callable, utility_cache: dict[pd.Timestamp, np.ndarray],
    diagnostics: dict[pd.Timestamp, dict[str, Any]],
    calibration: dict[str, Any],
    frames: dict[str, pd.DataFrame], symbols: list[str],
    config: Any,
) -> Callable:
    """Control como fallback obrigatorio, sem ajuste em OOS."""
    def policy(timestamp: pd.Timestamp, position: int, holding: int) -> tuple[int, float]:
        now = pd.Timestamp(timestamp)
        if now not in utility_cache:
            raise KeyError(f"V2 fold={fold_id} variant={variant} data fora do cache: {now}")
        base, base_score = baseline(now, position, holding)
        utilities = utility_cache[now]
        candidates = _candidatos(
            utilities, position, int(base), holding, config,
        )
        selected, score, accepted = int(base), 0.0, False
        if len(candidates) > 1:
            group = pd.DataFrame([
                {
                    "candidate_position": action,
                    **_atributos(
                        frames, symbols, now, position, int(base),
                        action, holding, utilities, config,
                    ),
                }
                for action in candidates
            ])
            if variant == "REGRESSION":
                raw, score = _regression_choice(model, group, int(base))
                accepted = bool(
                    raw != base and score >
                    float(calibration["regression_optimism_margin"])
                )
            elif variant == "DFL":
                raw, _ = _escolher_acao("DFL", model, group, int(base))
                if raw != base:
                    score = _pairwise_probability(
                        model, group, raw, int(base),
                    )
                    accepted = bool(
                        score > max(
                            0.5,
                            float(calibration["dfl_unsafe_probability_threshold"]),
                        )
                    )
            elif variant == "SOFTMAX_DFL":
                raw, score = escolher_softmax(model, group, int(base))
                accepted = bool(
                    raw != base and score >
                    float(calibration["softmax_unsafe_margin_threshold"])
                )
            else:
                raise ValueError(f"Variante nao suportada: {variant}")
            if accepted:
                selected = int(raw)
        diagnostics[now] = {
            "decision_fold_id": fold_id,
            "decision_diagnostics_schema_version": 2,
            "research_variant": variant,
            "current_asset": "CASH" if position == 0 else symbols[position - 1],
            "final_action_asset": "CASH" if selected == 0 else symbols[selected - 1],
            "final_action_score": float(utilities[selected]),
            "decision_reason": "RESEARCH_V2_" + variant,
            "research_base_action": "CASH" if base == 0 else symbols[base - 1],
            "research_changed_base_action": bool(selected != base),
            "research_guard_accepted": bool(accepted),
            "research_model_score": float(score),
        }
        return (
            selected,
            float(base_score if selected == base else utilities[selected]),
        )
    return policy


def executar_pesquisa_v2(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    *, horizonte: int = HORIZONTE_CONTRAFACTUAL,
) -> PesquisaV2:
    if horizonte <= 0 or horizonte > INNER_PURGE_SESSIONS:
        raise ValueError("Horizonte invalido em relacao ao purge OOF")
    (
        frames, dates, _, symbols, folds, oos_dates, date_to_fold, metadata,
    ) = _construir_contexto_execucao(bars_by_symbol, config)
    all_labels, all_audit, all_calibration = [], [], []
    variants = ("REGRESSION", "DFL", "SOFTMAX_DFL")
    policies: dict[str, dict[int, Callable]] = {v: {} for v in variants}
    baseline_policies: dict[int, Callable] = {}
    decisions: dict[str, dict[pd.Timestamp, dict[str, Any]]] = {
        v: {} for v in variants
    }
    for fold in folds:
        fold_id = int(fold["fold_id"])
        original, segments, segment_audit = _gerar_oof(
            frames, symbols, dates, fold, config, horizonte,
        )
        training, validation, validation_start = _divisao_temporal(original)
        augmented, augmentation_audit = _aumentar_estados(
            training, segments, frames, symbols, _parametros_internos(config),
            horizonte,
        )
        if augmented["outcome_end"].max() >= validation_start:
            raise AssertionError("Vazamento treino para validacao")
        reg, dfl = _treinar(augmented)
        softmax = treinar_softmax_dfl(augmented)
        guards = _calibrar_guardas(reg, dfl, softmax, validation)
        if validation["outcome_end"].max() >= fold["test_start"]:
            raise AssertionError("Vazamento da calibracao para OOS")
        all_labels.append(original)
        if len(augmented) > len(training):
            all_labels.append(augmented.loc[
                augmented["state_source"] == "OOF_LEARNED_STATE"
            ])
        all_audit.extend(segment_audit)
        all_calibration.append({
            "fold_id": fold_id,
            "train_original_rows": len(training),
            "train_unique_dates": int(training["decision_date"].nunique()),
            "validation_start": validation_start,
            "validation_rows": len(validation),
            "train_outcome_end_max": training["outcome_end"].max(),
            "validation_outcome_end_max": validation["outcome_end"].max(),
            "outer_test_start": fold["test_start"],
            "oof_total_rows": len(original),
            "oof_unique_dates": int(original["decision_date"].nunique()),
            **augmentation_audit, **guards,
        })
        # Mesma politica-base final (margem escolhida somente em calibration)
        # que o experimento oficial. Sem acesso ao teste OOS.
        train_dates = dates[:int(fold["train_end_index"])]
        cal_dates = dates[
            int(fold["calibration_start_index"]):
            int(fold["calibration_end_index"])
        ]
        cal_models = _ajustar_modelos_lightgbm(
            frames, symbols, train_dates, config,
            phase=f"decision_focused_v2_fold_{fold_id}_control_calibration",
        )
        candidate_margin, score = float(config.rotation_switch_margin), float("-inf")
        for margin in config.rotation_switch_margin_candidates:
            candidate = _politica_utilidade(
                cal_models, frames, symbols, config, float(margin),
            )
            value = _crescimento_politica_simples(
                candidate, frames, symbols, cal_dates, config,
            )
            if value > score:
                candidate_margin, score = float(margin), value
        effective_margin = max(
            float(config.rotation_switch_margin), candidate_margin,
        )
        final_dates = dates[:int(fold["final_fit_end_index"])]
        final_models = _ajustar_modelos_lightgbm(
            frames, symbols, final_dates, config,
            phase=f"decision_focused_v2_fold_{fold_id}_control_final",
        )
        fold_dates = pd.DatetimeIndex(fold["decision_dates"])
        cache, _ = _precalcular_utilidades_modelo(
            final_models, frames, symbols, fold_dates, config,
        )
        if not all(pd.Timestamp(t) in cache for t in fold_dates[:-1]):
            raise AssertionError(f"Fold {fold_id}: cache OOS incompleto")
        # [TCC-DFL:FIX-004] MESMA politica-base por fold para todos
        # os cenarios e replays; nao reestimar modelos ou margens na auditoria.
        baseline = _politica_utilidade(
            final_models, frames, symbols, config, effective_margin,
            utility_cache=cache,
        )
        baseline_policies[fold_id] = baseline
        for variant, model in (
            ("REGRESSION", reg), ("DFL", dfl), ("SOFTMAX_DFL", softmax),
        ):
            policies[variant][fold_id] = _criar_politica_v2(
                fold_id=fold_id, variant=variant, model=model,
                baseline=baseline, utility_cache=cache,
                diagnostics=decisions[variant], calibration=guards,
                frames=frames, symbols=symbols, config=config,
            )
        print(
            f"[decision-focused-v2] fold={fold_id} OOF_dates="
            f"{original['decision_date'].nunique()} train={len(training)} "
            f"validation={guards['validation_decision_dates']} "
            f"augmented={augmentation_audit['augmented_rows']} "
            f"reg_margin={guards['regression_optimism_margin']:.6f} "
            f"dfl_threshold={guards['dfl_unsafe_probability_threshold']:.3f}",
            flush=True,
        )
    results = {}
    for variant in variants:
        scheduled = _politica_agendada(policies[variant], date_to_fold)
        result = _simular_exato(
            "research_v2_" + variant.lower(), scheduled, frames, symbols,
            oos_dates, config, calcular_taxas_referencia, aplicar_deslizamento,
            decision_metadata=metadata,
            policy_decision_diagnostics=decisions[variant],
            model_label="Research OOF " + variant,
            method_line="- Decision learning: chronological OOF advantage and prior calibration.",
        )
        result.metrics["research_changed_base_actions"] = sum(
            int(x["research_changed_base_action"])
            for x in decisions[variant].values()
        )
        results[variant] = (
            result,
            summarize_metrics(result, folds, float(config.initial_capital)),
        )
    # [TCC-DFL:FIX-004] Reconstroi trajetorias completas dos episodios.
    # O script principal verifica identidade adicional contra Control oficial.
    original_regression = {
        pd.Timestamp(key): dict(value)
        for key, value in decisions["REGRESSION"].items()
    }
    intervention_audit = auditar_intervencoes(
        frames=frames, symbols=symbols, decision_dates=oos_dates, config=config,
        baseline_policy=_politica_agendada(baseline_policies, date_to_fold),
        learned_policy=_politica_agendada(policies["REGRESSION"], date_to_fold),
        original_decisions=original_regression,
        original_regression=results["REGRESSION"][0],
        decision_metadata=metadata, horizon=horizonte,
    )
    print(
        "[decision-focused] intervention audit "
        f"episodes={intervention_audit.checks['episodes']} "
        f"marginal_sum={intervention_audit.checks['sum_episode_marginal_usd']:.2f} "
        f"end_to_end={intervention_audit.checks['end_to_end_delta_usd']:.2f}",
        flush=True,
    )
    return PesquisaV2(
        regression_result=results["REGRESSION"][0],
        regression_metrics=results["REGRESSION"][1],
        dfl_result=results["DFL"][0],
        dfl_metrics=results["DFL"][1],
        softmax_result=results["SOFTMAX_DFL"][0],
        softmax_metrics=results["SOFTMAX_DFL"][1],
        labels=pd.concat(all_labels, ignore_index=True),
        audit=pd.DataFrame(all_audit),
        calibration=pd.DataFrame(all_calibration),
        regression_decisions=pd.DataFrame.from_dict(
            decisions["REGRESSION"], orient="index",
        ),
        dfl_decisions=pd.DataFrame.from_dict(
            decisions["DFL"], orient="index",
        ),
        softmax_decisions=pd.DataFrame.from_dict(
            decisions["SOFTMAX_DFL"], orient="index",
        ),
        auditoria_intervencoes=intervention_audit,
    )
