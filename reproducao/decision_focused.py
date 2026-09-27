"""Pesquisa isolada: vantagem contrafactual e ranking sensivel ao regret.

Nao altera o motor oficial. Rotulos sao produzidos em uma janela historica
anterior ao teste, por duas trajetorias emparelhadas que diferem somente na
primeira acao. Este ranking e um surrogate decision-focused, nao DFL
diferenciavel end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from engine.execucao import aplicar_deslizamento, calcular_taxas_referencia
from engine.modelo_lightgbm import (
    _ajustar_modelos_lightgbm,
    _construir_contexto_execucao,
)
from engine.rotacao import (
    ROTATION_FEATURES,
    _crescimento_politica_simples,
    _executar_compra,
    _politica_agendada,
    _politica_utilidade,
    _precalcular_utilidades_modelo,
    _simular_exato,
)
from reproducao.experimento import summarize_metrics


# Os mesmos valores sao usados em todos os folds e ficam congelados na branch.
HORIZONTE_CONTRAFACTUAL = 20
TOP_CANDIDATOS = 3
SEMENTE = 42
ATRIBUTOS_ATIVO = (
    "return_5", "return_20", "return_60", "vol_20", "vol_60",
    "ema_distance_20", "rsi_14", "atr_pct_14", "trend_efficiency_20",
    "volume_zscore_20",
)
FEATURE_COLUMNS = (
    "candidate_is_cash", "incumbent_is_cash", "candidate_is_incumbent",
    "candidate_is_control", "holding_days", "candidate_utility",
    "incumbent_utility", "control_utility", "edge_vs_incumbent",
    "edge_vs_control", "estimated_switch_cost", "candidate_rank",
    *("candidate_" + name for name in ATRIBUTOS_ATIVO),
    *("incumbent_" + name for name in ATRIBUTOS_ATIVO),
    *("relative_" + name for name in ATRIBUTOS_ATIVO),
)


@dataclass
class ResultadoPesquisa:
    regressao: Any
    regressao_metricas: dict[str, Any]
    dfl: Any
    dfl_metricas: dict[str, Any]
    rotulos: pd.DataFrame
    auditoria_folds: pd.DataFrame
    decisoes_regressao: pd.DataFrame
    decisoes_dfl: pd.DataFrame


def _pontuacao(utilidades: np.ndarray, posicao: int) -> float:
    value = float(utilidades[posicao])
    return value if np.isfinite(value) else 0.0


def _candidatos(
    utilidades: np.ndarray, atual: int, controle: int, holding: int,
    config: Any, *, top: int = TOP_CANDIDATOS,
) -> list[int]:
    """Inclui CASH, incumbente e Control; nao usa retorno futuro para rankear."""
    if atual > 0 and holding < int(config.rotation_min_holding_days):
        return [atual]
    elegiveis = sorted(
        (i for i in range(1, len(utilidades)) if np.isfinite(utilidades[i])),
        key=lambda i: (-float(utilidades[i]), i),
    )
    ordem = [controle, atual, 0, *elegiveis[:top]]
    return list(dict.fromkeys(
        pos for pos in ordem
        if pos == 0 or (0 < pos < len(utilidades) and np.isfinite(utilidades[pos]))
    ))


def _atributos(
    frames: dict[str, pd.DataFrame], symbols: list[str],
    data: pd.Timestamp, atual: int, controle: int, candidato: int,
    holding: int, utilidades: np.ndarray, config: Any,
) -> dict[str, float]:
    """Somente informacoes da sessao de decisao e escores ja previstos."""
    from engine.rotacao import _custo_troca_proporcional

    def valores(pos: int) -> np.ndarray:
        if pos <= 0:
            return np.zeros(len(ATRIBUTOS_ATIVO), dtype=float)
        row = frames[symbols[pos - 1]].loc[data, list(ATRIBUTOS_ATIVO)]
        return np.asarray(row, dtype=float)

    c = valores(candidato)
    a = valores(atual)
    ranking = sorted(
        (i for i in range(1, len(utilidades)) if np.isfinite(utilidades[i])),
        key=lambda i: (-float(utilidades[i]), i),
    )
    result = {
        "candidate_is_cash": float(candidato == 0),
        "incumbent_is_cash": float(atual == 0),
        "candidate_is_incumbent": float(candidato == atual),
        "candidate_is_control": float(candidato == controle),
        "holding_days": float(holding),
        "candidate_utility": _pontuacao(utilidades, candidato),
        "incumbent_utility": _pontuacao(utilidades, atual),
        "control_utility": _pontuacao(utilidades, controle),
        "edge_vs_incumbent": _pontuacao(utilidades, candidato) - _pontuacao(utilidades, atual),
        "edge_vs_control": _pontuacao(utilidades, candidato) - _pontuacao(utilidades, controle),
        "estimated_switch_cost": _custo_troca_proporcional(config, atual, candidato),
        "candidate_rank": float(ranking.index(candidato) + 1) if candidato in ranking else 0.0,
    }
    for name, candidate_value, incumbent_value in zip(ATRIBUTOS_ATIVO, c, a):
        result["candidate_" + name] = float(candidate_value)
        result["incumbent_" + name] = float(incumbent_value)
        result["relative_" + name] = float(candidate_value - incumbent_value)
    return result


def _passo_conta(
    cash: float, quantity: float, position: int, action: int,
    execution_date: pd.Timestamp, frames: dict[str, pd.DataFrame],
    symbols: list[str], config: Any,
) -> tuple[float, float]:
    if action == position:
        return cash, quantity
    if position > 0:
        old_open = float(frames[symbols[position - 1]].loc[execution_date, "open"])
        price = aplicar_deslizamento(old_open, "SELL", config)
        fee = calcular_taxas_referencia("SELL", quantity, price, config)["total_fee"]
        cash += quantity * price - fee
        quantity = 0.0
    if action > 0:
        new_open = float(frames[symbols[action - 1]].loc[execution_date, "open"])
        quantity, price, fee = _executar_compra(
            cash, new_open, config, calcular_taxas_referencia, aplicar_deslizamento,
        )
        cash -= quantity * price + float(fee["total_fee"])
    return cash, quantity


def _replay_emparelhado(
    policy: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    frames: dict[str, pd.DataFrame], symbols: list[str],
    dates: pd.DatetimeIndex, start: int, horizonte: int,
    position: int, holding: int, first_action: int, config: Any,
) -> float:
    """Mesmo estado inicial, mesma politica de continuacao e custos do motor.

    Retorna log do capital liquidado. O capital da posicao inicial e marcado
    ao fechamento da data de decisao; taxas de entrada anteriores sao comuns
    as alternativas e ficam fora de ambas as trajetorias.
    """
    capital = float(config.initial_capital)
    if position:
        first_close = float(frames[symbols[position - 1]].loc[dates[start], "close"])
        cash, quantity = 0.0, capital / first_close
    else:
        cash, quantity = capital, 0.0
    for step in range(horizonte):
        now = dates[start + step]
        next_date = dates[start + step + 1]
        action = first_action if step == 0 else int(policy(now, position, holding)[0])
        if action < 0 or action > len(symbols):
            raise ValueError("Posicao contrafactual invalida")
        old_position = position
        cash, quantity = _passo_conta(
            cash, quantity, position, action, next_date, frames, symbols, config,
        )
        position = action
        holding = (holding + 1 if action > 0 else 0) if action == old_position else (1 if action > 0 else 0)
    # Mesmo tratamento da liquidacao final do motor oficial.
    if position > 0:
        close = float(frames[symbols[position - 1]].loc[dates[start + horizonte], "close"])
        price = aplicar_deslizamento(close, "SELL", config)
        fee = calcular_taxas_referencia("SELL", quantity, price, config)["total_fee"]
        cash += quantity * price - fee
    if not np.isfinite(cash) or cash <= 0:
        raise ValueError("Capital contrafactual invalido")
    return float(math.log(cash / capital))


def _estados_calibracao(
    policy: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    dates: pd.DatetimeIndex,
) -> list[tuple[int, int, int]]:
    states = []
    position = 0
    holding = 0
    for now in dates[:-1]:
        action = int(policy(now, position, holding)[0])
        states.append((position, holding, action))
        if action == position:
            holding = holding + 1 if position > 0 else 0
        else:
            position, holding = action, (1 if action > 0 else 0)
    return states


def _gerar_rotulos(
    frames: dict[str, pd.DataFrame], symbols: list[str],
    dates: pd.DatetimeIndex, policy: Callable, utility_cache: dict,
    config: Any, fold_id: int, horizonte: int,
) -> pd.DataFrame:
    states = _estados_calibracao(policy, dates)
    rows: list[dict[str, Any]] = []
    for index in range(max(0, len(states) - horizonte + 1)):
        now = dates[index]
        position, holding, control = states[index]
        utilities = utility_cache[now]
        candidates = _candidatos(utilities, position, control, holding, config)
        if len(candidates) < 2:
            continue
        try:
            control_reward = _replay_emparelhado(
                policy, frames, symbols, dates, index, horizonte,
                position, holding, control, config,
            )
        except (ValueError, KeyError, ZeroDivisionError):
            continue
        group_rows = []
        for candidate in candidates:
            try:
                reward = _replay_emparelhado(
                    policy, frames, symbols, dates, index, horizonte,
                    position, holding, candidate, config,
                )
                features = _atributos(
                    frames, symbols, now, position, control, candidate,
                    holding, utilities, config,
                )
            except (ValueError, KeyError, ZeroDivisionError):
                continue
            group_rows.append({
                "fold_id": fold_id,
                "decision_date": now,
                "outcome_end": dates[index + horizonte],
                "incumbent": "CASH" if position == 0 else symbols[position - 1],
                "candidate": "CASH" if candidate == 0 else symbols[candidate - 1],
                "control_action": "CASH" if control == 0 else symbols[control - 1],
                "candidate_position": candidate,
                "control_position": control,
                "log_wealth": reward,
                "log_advantage": reward - control_reward,
                **features,
            })
        if len(group_rows) >= 2:
            rows.extend(group_rows)
    if not rows:
        raise ValueError(f"Fold {fold_id}: nenhum rotulo contrafactual valido")
    return pd.DataFrame(rows)


def _pares_regret(rotulos: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Surrogate de ranking: peso dos pares proporcional ao regret observado."""
    features, labels, weights = [], [], []
    for _, group in rotulos.groupby(["fold_id", "decision_date"], sort=True):
        x = group.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
        y = group["log_advantage"].to_numpy(dtype=float)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                delta = float(y[i] - y[j])
                if abs(delta) < 1e-10:
                    continue
                diff = x[i] - x[j]
                features.extend((diff, -diff))
                labels.extend((int(delta > 0), int(delta < 0)))
                weight = 0.1 + min(5.0, abs(delta) * 100.0)
                weights.extend((weight, weight))
    if not features:
        raise ValueError("Nao existem pares de decisoes com resultados diferentes")
    return (
        pd.DataFrame(features, columns=FEATURE_COLUMNS).replace([np.inf, -np.inf], np.nan),
        np.asarray(labels, dtype=int),
        np.asarray(weights, dtype=float),
    )


def _treinar(rotulos: pd.DataFrame) -> tuple[Any, Any]:
    from lightgbm import LGBMClassifier, LGBMRegressor

    x = rotulos.loc[:, FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    y = rotulos["log_advantage"].to_numpy(dtype=float)
    regressao = LGBMRegressor(
        n_estimators=120, learning_rate=0.035, max_depth=3, num_leaves=7,
        min_child_samples=8, random_state=SEMENTE, n_jobs=1, verbosity=-1,
        deterministic=True, force_col_wise=True,
    )
    regressao.fit(x, y)
    pair_x, pair_y, pair_w = _pares_regret(rotulos)
    ranking = LGBMClassifier(
        n_estimators=120, learning_rate=0.035, max_depth=3, num_leaves=7,
        min_child_samples=8, random_state=SEMENTE, n_jobs=1, verbosity=-1,
        deterministic=True, force_col_wise=True,
    )
    ranking.fit(pair_x, pair_y, sample_weight=pair_w)
    return regressao, ranking


def _escolher_acao(
    variant: str, model: Any, group: pd.DataFrame, control: int,
) -> tuple[int, float]:
    x = group.loc[:, FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    positions = group["candidate_position"].astype(int).to_numpy()
    if len(positions) < 2 or control not in positions:
        return control, 0.0
    base_idx = int(np.flatnonzero(positions == control)[0])
    if variant == "REGRESSION":
        advantages = np.asarray(model.predict(x), dtype=float)
        best = int(np.argmax(advantages))
        return (
            (int(positions[best]), float(advantages[best]))
            if best != base_idx and advantages[best] > 0.0
            else (control, float(advantages[base_idx]))
        )
    # Avaliacao de todos os pares da mesma sessao, sem acessar o target.
    array = x.to_numpy(dtype=float)
    comparisons = np.asarray(
        [array[i] - array[j] for i in range(len(x)) for j in range(i + 1, len(x))],
        dtype=float,
    )
    probability = model.predict_proba(
        pd.DataFrame(comparisons, columns=FEATURE_COLUMNS)
    )[:, 1]
    points = np.zeros(len(x), dtype=float)
    cursor = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p = float(probability[cursor])
            points[i] += p
            points[j] += 1.0 - p
            cursor += 1
    best = int(np.argmax(points))
    if best == base_idx:
        return control, float(points[base_idx])
    delta = pd.DataFrame(
        [array[best] - array[base_idx]], columns=FEATURE_COLUMNS,
    )
    if float(model.predict_proba(delta)[0, 1]) <= 0.5:
        return control, float(points[base_idx])
    return int(positions[best]), float(points[best])


def executar_pesquisa(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    *, horizonte: int = HORIZONTE_CONTRAFACTUAL,
) -> ResultadoPesquisa:
    if horizonte <= 0 or horizonte > int(config.rotation_purge_days):
        raise ValueError("Horizonte deve ser positivo e nao superar o purge temporal")
    (
        frames, dates, _, symbols, folds, oos_dates, date_to_fold, metadata,
    ) = _construir_contexto_execucao(bars_by_symbol, config)
    policies: dict[str, dict[int, Callable]] = {"REGRESSION": {}, "DFL": {}}
    all_labels, audit = [], []
    decisions: dict[str, dict[pd.Timestamp, dict[str, Any]]] = {
        "REGRESSION": {}, "DFL": {},
    }
    for fold in folds:
        fold_id = int(fold["fold_id"])
        train_dates = dates[:int(fold["train_end_index"])]
        cal_dates = dates[int(fold["calibration_start_index"]):int(fold["calibration_end_index"])]
        final_dates = dates[:int(fold["final_fit_end_index"])]
        cal_models = _ajustar_modelos_lightgbm(
            frames, symbols, train_dates, config,
            phase=f"decision_focused_fold_{fold_id}_calibration",
        )
        if not cal_models:
            raise RuntimeError(f"Fold {fold_id}: sem modelos da politica-base")
        margins = tuple(float(m) for m in config.rotation_switch_margin_candidates)
        best_margin, best_score = margins[0], float("-inf")
        for margin in margins:
            cal_policy = _politica_utilidade(
                cal_models, frames, symbols, config, margin,
            )
            score = _crescimento_politica_simples(
                cal_policy, frames, symbols, cal_dates, config,
            )
            if score > best_score:
                best_margin, best_score = margin, score
        effective_margin = max(float(config.rotation_switch_margin), best_margin)
        cal_cache, _ = _precalcular_utilidades_modelo(
            cal_models, frames, symbols, cal_dates, config,
        )
        cal_policy = _politica_utilidade(
            cal_models, frames, symbols, config, effective_margin,
            utility_cache=cal_cache,
        )
        labels = _gerar_rotulos(
            frames, symbols, cal_dates, cal_policy, cal_cache,
            config, fold_id, horizonte,
        )
        if labels["outcome_end"].max() > fold["calibration_end"]:
            raise AssertionError("Rotulo ultrapassa janela de calibracao")
        if labels["outcome_end"].max() >= fold["test_start"]:
            raise AssertionError("Vazamento temporal para teste OOS")
        regression, dfl = _treinar(labels)
        all_labels.append(labels)
        final_models = _ajustar_modelos_lightgbm(
            frames, symbols, final_dates, config,
            phase=f"decision_focused_fold_{fold_id}_final",
        )
        fold_dates = pd.DatetimeIndex(fold["decision_dates"])
        final_cache, _ = _precalcular_utilidades_modelo(
            final_models, frames, symbols, fold_dates, config,
        )
        audit.append({
            "fold_id": fold_id, "base_train_end": train_dates[-1],
            "calibration_start": cal_dates[0], "calibration_end": cal_dates[-1],
            "label_outcome_end_max": labels["outcome_end"].max(),
            "final_fit_end": final_dates[-1], "oos_start": fold["test_start"],
            "label_decisions": int(labels["decision_date"].nunique()),
            "label_rows": len(labels), "margin": effective_margin,
            "snapshot_used": "dados/pesquisa",
        })
        for variant, fitted in (("REGRESSION", regression), ("DFL", dfl)):
            baseline = _politica_utilidade(
                final_models, frames, symbols, config, effective_margin,
                utility_cache=final_cache,
            )
            diagnostics = decisions[variant]

            def make_policy(variant=variant, fitted=fitted, baseline=baseline, diagnostics=diagnostics):
                def policy(timestamp: pd.Timestamp, position: int, holding: int) -> tuple[int, float]:
                    current = pd.Timestamp(timestamp)
                    base, base_score = baseline(current, position, holding)
                    utilities = final_cache[current]
                    candidates = _candidatos(
                        utilities, position, int(base), holding, config,
                    )
                    if len(candidates) < 2:
                        target, learned = int(base), 0.0
                    else:
                        rows = [
                            {"candidate_position": candidate, **_atributos(
                                frames, symbols, current, position, int(base),
                                candidate, holding, utilities, config,
                            )}
                            for candidate in candidates
                        ]
                        target, learned = _escolher_acao(
                            variant, fitted, pd.DataFrame(rows), int(base),
                        )
                    diagnostics[current] = {
                        "decision_diagnostics_schema_version": 2,
                        "current_asset": "CASH" if position == 0 else symbols[position - 1],
                        "final_action_asset": "CASH" if target == 0 else symbols[target - 1],
                        "final_action_score": _pontuacao(utilities, target),
                        "decision_reason": f"RESEARCH_{variant}",
                        "research_variant": variant,
                        "research_base_action": "CASH" if base == 0 else symbols[base - 1],
                        "research_changed_base_action": bool(target != base),
                        "research_model_score": float(learned),
                    }
                    return int(target), float(base_score if target == base else _pontuacao(utilities, target))
                return policy

            policies[variant][fold_id] = make_policy()
        print(
            f"[decision-focused] fold={fold_id} labels={len(labels)} "
            f"end={labels['outcome_end'].max()} before_oos={fold['test_start']}",
            flush=True,
        )

    results = {}
    for variant in ("REGRESSION", "DFL"):
        scheduled = _politica_agendada(policies[variant], date_to_fold)
        result = _simular_exato(
            "research_" + variant.lower(), scheduled, frames, symbols, oos_dates,
            config, calcular_taxas_referencia, aplicar_deslizamento,
            decision_metadata=metadata,
            policy_decision_diagnostics=decisions[variant],
            model_label="Research " + variant,
            method_line="- Research policy: fold-local counterfactual decision learning.",
        )
        result.metrics["research_changed_base_actions"] = sum(
            int(item["research_changed_base_action"])
            for item in decisions[variant].values()
        )
        results[variant] = (result, summarize_metrics(
            result, folds, float(config.initial_capital),
        ))
    return ResultadoPesquisa(
        regressao=results["REGRESSION"][0],
        regressao_metricas=results["REGRESSION"][1],
        dfl=results["DFL"][0],
        dfl_metricas=results["DFL"][1],
        rotulos=pd.concat(all_labels, ignore_index=True),
        auditoria_folds=pd.DataFrame(audit),
        decisoes_regressao=pd.DataFrame.from_dict(decisions["REGRESSION"], orient="index"),
        decisoes_dfl=pd.DataFrame.from_dict(decisions["DFL"], orient="index"),
    )
