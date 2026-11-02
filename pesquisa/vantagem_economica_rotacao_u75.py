"""TCC: confronto economico prospectivo entre rotacoes, risco e reversao.

O Control decide primeiro. O overlay usa previsoes treinadas somente no
passado e compara retornos esperados em unidades comuns de log-retorno.
O classificador de reversao nao e bloqueado por um limiar de p_downside.
Nao altera os modelos LightGBM, snapshots, U75 ou regras do MCT.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pesquisa.risco_reversao_antecipado_u75 import FEATURES


VERSION = "u75-economic-rotation-recovery-v1.0.2"


@dataclass(frozen=True)
class EconomicSettings:
    horizon_days: int = 5
    risk_weight: float = 0.40
    recovery_blend_weight: float = 0.35
    minimum_incremental_log_edge: float = 0.005
    estimated_switch_cost: float = 0.003
    cash_loss_buffer: float = 0.010
    minimum_model_coverage_fraction: float = 0.90
    minimum_train_samples: int = 500
    ridge_alpha: float = 10.0

    def __post_init__(self):
        if (self.horizon_days < 1 or self.risk_weight < 0
                or not 0 <= self.recovery_blend_weight <= 1
                or self.minimum_incremental_log_edge < 0
                or self.estimated_switch_cost < 0
                or self.cash_loss_buffer <= 0
                or not 0 < self.minimum_model_coverage_fraction <= 1
                or self.minimum_train_samples < 30 or self.ridge_alpha <= 0):
            raise ValueError("Parametros economicos invalidos")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _strict_future_window(frame: pd.DataFrame, column: str, horizon: int):
    return pd.concat(
        [pd.to_numeric(frame[column], errors="coerce").shift(-i)
         for i in range(1, horizon + 1)], axis=1
    )


def preparar_resultados_economicos(
    panel: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    horizon: int,
    recovery_horizon: int,
) -> pd.DataFrame:
    """Alvos estritamente futuros, nunca expostos ao callback de decisao."""
    if horizon < 1 or recovery_horizon < horizon:
        raise ValueError("Horizontes de treinamento invalidos")
    calendar = pd.DatetimeIndex(calendar)
    data = panel.copy()
    data["target_log_return"] = np.nan
    data["target_recovery_log_return"] = np.nan
    data["target_tail_severity"] = np.nan
    for symbol, source in frames.items():
        rows = data["asset"].eq(symbol).to_numpy()
        if not rows.any():
            continue
        frame = source.reindex(calendar)
        if int(rows.sum()) != len(calendar):
            raise ValueError(f"{symbol}: painel de datas incompleto")
        open_next = pd.to_numeric(frame["open"], errors="coerce").shift(-1)
        lo = _strict_future_window(frame, "low", horizon)
        cl = _strict_future_window(frame, "close", horizon)
        cl_recovery = _strict_future_window(frame, "close", recovery_horizon)
        valid_short = (open_next.gt(0)
                       & (lo.gt(0) & np.isfinite(lo)).all(axis=1)
                       & (cl.gt(0) & np.isfinite(cl)).all(axis=1))
        valid_recovery = (
            open_next.gt(0)
            & (cl_recovery.gt(0) & np.isfinite(cl_recovery)).all(axis=1)
        )
        r = np.log(cl.iloc[:, -1] / open_next).where(valid_short)
        r_long = np.log(cl_recovery.iloc[:, -1] / open_next).where(valid_recovery)
        tail = np.maximum(0.0, -np.log(lo.min(axis=1, skipna=False) / open_next))
        tail = tail.where(valid_short)
        for field, values in (
            ("target_log_return", r),
            ("target_recovery_log_return", r_long),
            ("target_tail_severity", tail),
        ):
            data.loc[rows, field] = values.to_numpy(dtype=float)
    if data["target_log_return"].notna().sum() == 0:
        raise RuntimeError("Nenhum retorno economico verificavel")
    return data


def _fit_return_model(train: pd.DataFrame, settings: EconomicSettings):
    valid = train.dropna(subset=[*FEATURES, "target_log_return"])
    if len(valid) < settings.minimum_train_samples:
        return None, len(valid)
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=settings.ridge_alpha),
    )
    model.fit(valid[list(FEATURES)], valid["target_log_return"])
    return model, len(valid)


def estimar_vantagens_por_fold(
    data: pd.DataFrame,
    original_signals: dict[int, dict[pd.Timestamp, dict[str, dict]]],
    folds: list[dict],
    calendar: pd.DatetimeIndex,
    settings: EconomicSettings,
    recovery_horizon: int,
) -> tuple[dict[int, dict[pd.Timestamp, dict[str, dict]]], pd.DataFrame, pd.DataFrame]:
    """Retorno / perda / reversao ajustados em passado maturado por fold."""
    if recovery_horizon < settings.horizon_days:
        raise ValueError("Horizonte de recuperacao inferior ao horizonte economico")
    calendar = pd.DatetimeIndex(calendar)
    signals: dict[int, dict[pd.Timestamp, dict[str, dict]]] = {}
    stats: list[dict] = []
    audit_rows: list[dict] = []
    for fold in folds:
        fid = int(fold["fold_id"])
        first = int(fold["test_start_index"])
        last = int(fold["test_end_index"])
        final_fit = int(fold["final_fit_end_index"])
        if not (0 < final_fit < first < last <= len(calendar)):
            raise ValueError(f"Fold {fid}: fronteira temporal invalida")
        # Ultimo resultado dos treinos precisa ter amadurecido ANTES de
        # iniciar o teste. Manter embargo do proprio Control.
        economic_train = data.loc[
            (data["calendar_position"] < final_fit)
            & (data["calendar_position"] + settings.horizon_days < first)
        ]
        model, n_return = _fit_return_model(economic_train, settings)
        tails = economic_train.loc[
            economic_train["downside_event"].eq(1.0)
            & economic_train["target_tail_severity"].notna(),
            "target_tail_severity",
        ]
        tail_mean = float(tails.mean()) if len(tails) >= 30 else None

        reversal_train = data.loc[
            (data["calendar_position"] < final_fit)
            & (data["calendar_position"] + recovery_horizon < first)
            & data["recent_drawdown"].eq(True)
            & data["target_recovery_log_return"].notna()
            & data["reversal_event"].notna()
        ]
        positives = reversal_train.loc[
            reversal_train["reversal_event"].eq(1.0),
            "target_recovery_log_return",
        ]
        negatives = reversal_train.loc[
            reversal_train["reversal_event"].eq(0.0),
            "target_recovery_log_return",
        ]
        payoff_pos = float(positives.mean()) if len(positives) >= 30 else None
        payoff_neg = float(negatives.mean()) if len(negatives) >= 30 else None
        stats.append({
            "fold": fid, "last_train_feature_position": final_fit - 1,
            "first_test": calendar[first].isoformat(),
            "n_return": n_return, "n_downside_events": len(tails),
            "n_recovery_positive": len(positives),
            "n_recovery_negative": len(negatives),
            "mean_downside_severity_log": tail_mean,
            "mean_recovery_log_positive": payoff_pos,
            "mean_recovery_log_negative": payoff_neg,
            "return_model_available": model is not None,
            "recovery_payoff_available": (
                payoff_pos is not None and payoff_neg is not None
            ),
        })
        per_day: dict[pd.Timestamp, dict[str, dict]] = {}
        test = data.loc[
            data["calendar_position"].between(first - 1, last - 1)
            & data[list(FEATURES)].notna().all(axis=1)
        ].copy()
        if test.empty:
            raise RuntimeError(f"Fold {fid}: sem atributos de inferencia")
        preds = (model.predict(test[list(FEATURES)])
                 if model is not None else np.full(len(test), np.nan))
        inherited = original_signals[fid]
        for row, mu in zip(test.itertuples(index=False), preds):
            date = pd.Timestamp(row.date)
            original = inherited.get(date, {}).get(row.asset, {})
            pr = original.get("p_downside")
            pb = original.get("p_recovery")
            expected_loss = (
                float(pr) * tail_mean
                if pr is not None and tail_mean is not None
                and np.isfinite(float(pr)) else None
            )
            risk_score = (
                float(mu) - settings.risk_weight * expected_loss
                if np.isfinite(mu) and expected_loss is not None else None
            )
            payoff_estimate = None
            if (pb is not None and payoff_pos is not None
                    and payoff_neg is not None and np.isfinite(float(pb))):
                expected_long = (float(pb) * payoff_pos
                                 + (1.0 - float(pb)) * payoff_neg)
                payoff_estimate = (settings.horizon_days / recovery_horizon
                                   * expected_long)
            joint_score = risk_score
            if risk_score is not None and payoff_estimate is not None:
                blend = settings.recovery_blend_weight
                joint_score = (
                    (1.0 - blend) * float(mu)
                    + blend * payoff_estimate
                    - settings.risk_weight * expected_loss
                )
            payload = {
                "risk_score": risk_score,
                "joint_score": joint_score,
                "expected_log_return": (
                    float(mu) if np.isfinite(mu) else None
                ),
                "expected_downside_loss": expected_loss,
                "expected_recovery_log_return_5d": payoff_estimate,
                "p_downside": pr,
                "p_recovery": pb,
                "recent_drawdown": bool(row.recent_drawdown),
            }
            per_day.setdefault(date, {})[row.asset] = payload
            audit_rows.append({
                "fold": fid, "decision_date": date, "asset": row.asset,
                **payload,
                # Alvos apenas para auditoria, nunca passados ao policy.
                "realized_return": (
                    float(row.target_log_return)
                    if np.isfinite(row.target_log_return) else None
                ),
            })
        signals[fid] = per_day
    return signals, pd.DataFrame(stats), pd.DataFrame(audit_rows)


def politica_vantagem_economica(
    control_policy: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    symbols: list[str],
    utility_cache: dict[pd.Timestamp, np.ndarray],
    economic_signals: dict[pd.Timestamp, dict[str, dict]],
    settings: EconomicSettings,
    *,
    use_recovery: bool,
    audit: list[dict],
    diagnostics: dict[pd.Timestamp, dict[str, Any]],
    minimum_score: float,
) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:
    """A decisao Control e ponto de partida. CASH e ultimo recurso.

    A regra compara BENEFICIO INCREMENTAL previsto ja descontando
    custo de mudar de posicao. Nunca veta exclusivamente por p_downside.
    """
    names = ["CASH", *symbols]
    if len(set(symbols)) != len(symbols):
        raise ValueError("Ativos duplicados")

    def policy(date: pd.Timestamp, current: int, holding: int) -> tuple[int, float]:
        ts = pd.Timestamp(date)
        base, base_utility = control_policy(ts, current, holding)
        base = int(base)
        if not (0 <= base < len(names) and 0 <= current < len(names)):
            raise RuntimeError("Posicao do motor fora do universo")
        utilities = utility_cache.get(ts)
        if utilities is None or len(utilities) != len(names):
            raise RuntimeError("Cache LightGBM nao coincide com Control")
        forecast = economic_signals.get(ts, {})
        key = "joint_score" if use_recovery else "risk_score"
        available = {
            i: float(forecast.get(names[i], {}).get(key))
            for i in range(1, len(names))
            if np.isfinite(float(utilities[i]))
            and float(utilities[i]) > minimum_score
            and forecast.get(names[i], {}).get(key) is not None
            and np.isfinite(float(forecast[names[i]][key]))
        }
        coverage = len(available) / max(1, int(sum(
            np.isfinite(float(u)) and float(u) > minimum_score
            for u in utilities[1:]
        )))
        # Uma parte muito pequena do universo seria vies para CASH.
        ready = coverage >= settings.minimum_model_coverage_fraction
        base_value = 0.0 if base == 0 else available.get(base)
        target = base
        decision = "CONTROL_PRESERVED"
        advantage = None
        candidate_advantage = None
        cost = None
        winner = None

        def net_value(position: int) -> float | None:
            # CASH tem retorno esperado zero antes do custo de saida.
            # Manter a posicao atual nao gera custo de troca.
            value = 0.0 if position == 0 else available.get(position)
            if value is None:
                return None
            return float(value) - settings.estimated_switch_cost * int(
                position != current
            )

        base_net = net_value(base)
        cash_net = net_value(0)
        if (ready and base_net is not None
                and diagnostics.get(ts, {}).get("decision_reason") != "MIN_HOLD_GUARD"):
            if available:
                # Comparar oportunidades em unidade identica: log-retorno
                # previsto descontado do custo de trocar a posicao EFETIVA.
                net_candidates = {i: net_value(i) for i in available}
                winner = max(net_candidates, key=lambda i: (net_candidates[i], -i))
                winner_net = net_candidates[winner]
                candidate_advantage = winner_net - base_net

                # Primeiro, uma rotacao que tenha retorno liquido positivo
                # e supere economicamente a escolha do Control.
                if (winner != base and winner_net > 0.0
                        and candidate_advantage
                        >= settings.minimum_incremental_log_edge):
                    target = winner
                    decision = (
                        "RECOVERY_ECONOMIC_ROTATION"
                        if use_recovery and forecast.get(names[winner], {}).get(
                            "expected_recovery_log_return_5d") is not None
                        else "RISK_ECONOMIC_ROTATION"
                    )
                # Se TODAS as alternativas de compra sao negativas apos
                # custo, nunca tratar score bruto ligeiramente positivo
                # como motivo para manter uma compra muito desfavoravel.
                elif (base > 0 and winner_net <= 0.0
                      and base_value <= -settings.cash_loss_buffer):
                    # Se ja estamos em ativo elegivel e mantê-lo custa menos
                    # que sair para CASH, a melhor defesa e nao rotacionar.
                    if (current > 0 and current in net_candidates
                            and net_candidates[current] > cash_net
                            and net_candidates[current] - base_net
                            >= settings.minimum_incremental_log_edge
                            and current != base):
                        target = current
                        decision = "ECONOMIC_AVOID_NEGATIVE_ROTATION"
                    elif (cash_net - base_net
                          >= settings.minimum_incremental_log_edge):
                        target = 0
                        decision = "ECONOMIC_CASH_LAST_RESORT"

        if base > 0 and base not in available:
            # Sem retorno comparavel no ativo escolhido, nao inferir
            # falta de oportunidade nem vetar a escolha original.
            target, decision = base, "CONTROL_MODEL_UNAVAILABLE"
        if not ready:
            target, decision = base, "CONTROL_INSUFFICIENT_COVERAGE"
        if diagnostics.get(ts, {}).get("decision_reason") == "MIN_HOLD_GUARD":
            target, decision = base, "CONTROL_MIN_HOLD_GUARD"
        # Auditoria financeira deve refletir a acao FINAL apos todos
        # os guards de disponibilidade, cobertura e min-holding.
        final_net = net_value(target)
        if base_net is not None and final_net is not None:
            advantage = final_net - base_net
        cost = settings.estimated_switch_cost * (
            int(target != current) - int(base != current)
        )
        final_utility = (float(utilities[target]) if target > 0 else 0.0)
        if target != base:
            row = diagnostics.setdefault(ts, {})
            row.update({
                "final_action_asset": names[target],
                "final_action_score": final_utility,
                "decision_reason": decision,
                "q_final_action": final_utility,
                "decision_is_rotation": (
                    current > 0 and target > 0 and current != target
                ),
                "decision_is_entry": current == 0 and target > 0,
                "decision_is_exit_to_cash": current > 0 and target == 0,
            })
        audit.append({
            "decision_date": ts.isoformat(), "current_asset": names[current],
            "control_target": names[base], "final_target": names[target],
            "reason": decision, "changed_control_action": bool(target != base),
            "economic_score_control": base_value,
            "economic_score_final": (
                available.get(target) if target > 0 else 0.0
            ),
            "incremental_net_edge": advantage, "estimated_cost_delta": cost,
            "best_candidate_incremental_net_edge": candidate_advantage,
            "control_net_score": base_net, "final_net_score": final_net,
            "cash_net_score": cash_net,
            "model_coverage": coverage, "model_ready": ready,
            "p_downside_control": (
                forecast.get(names[base], {}).get("p_downside") if base else None
            ),
            "p_recovery_final": (
                forecast.get(names[target], {}).get("p_recovery") if target else None
            ),
            "recovery_included": use_recovery,
            "cash_last_resort": decision == "ECONOMIC_CASH_LAST_RESORT",
        })
        return target, (final_utility if target != base else float(base_utility))

    return policy
