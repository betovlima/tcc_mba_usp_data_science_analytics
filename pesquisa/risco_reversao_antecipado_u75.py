"""Pesquisa U75: risco antecipado, recuperacao de fundos e rotacao preservada.

SEM alterar o motor Control, o U75, a logica operacional ou os dados RAW.
Os modelos usam SOMENTE atributos presentes no fechamento t. Rotulos t+h
sao usados apenas no treinamento se o desfecho ja amadureceu antes do teste.
Nao se tenta prever o fundo exato. A politica Control sempre e consultada
primeiro com a posicao efetiva e os dias reais de permanencia.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


VERSION = "u75-early-risk-bottom-reversal-v1.0.0"
FEATURES = (
    "return_5", "return_20", "return_60", "vol_20",
    "vol_ratio_5_20", "ema_distance_20", "distance_from_high_20",
    "distance_from_low_20", "trend_efficiency_20",
    "momentum_acceleration_5_20", "rsi_14", "atr_pct_14",
)


@dataclass(frozen=True)
class EarlyRiskSettings:
    downside_horizon: int = 5
    downside_event: float = -0.08
    downside_alert_probability: float = 0.40
    reversal_horizon: int = 10
    recovery_event: float = 0.05
    recent_drawdown_trigger: float = -0.10
    reversal_alert_probability: float = 0.60
    risk_penalty: float = 0.08
    recovery_bonus: float = 0.015
    minimum_adjusted_advantage: float = 0.0025
    min_train_samples: int = 500
    random_state: int = 42

    def __post_init__(self):
        if (self.downside_horizon < 1 or self.reversal_horizon < 1
                or not -1 < self.downside_event < 0
                or not 0 < self.recovery_event < 1
                or not -1 < self.recent_drawdown_trigger < 0
                or not 0 < self.downside_alert_probability < 1
                or not 0 < self.reversal_alert_probability < 1
                or self.risk_penalty < 0 or self.recovery_bonus < 0
                or self.minimum_adjusted_advantage < 0
                or self.min_train_samples < 30):
            raise ValueError("Protocolo de risco e reversao invalido")

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


def _require(value: bool, explanation: str) -> None:
    if not value:
        raise ValueError(explanation)


def preparar_painel_supervisionado(
    frames: dict[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    settings: EarlyRiskSettings,
) -> pd.DataFrame:
    """Cria atributos t e resultados de futuro separados e com maturacao.

    O forward low dos proximos 5 pregoes representa risco de cauda medido
    a partir da abertura seguinte. Recovery mede close t+10 / open t+1.
    Ausencia de QUALQUER preco no intervalo implica alvo indisponivel.
    """
    calendar = pd.DatetimeIndex(calendar)
    _require(calendar.is_monotonic_increasing and not calendar.has_duplicates,
             "Calendario invalido")
    _require(bool(frames), "Nenhum ativo")
    result = []
    for symbol in sorted(frames):
        frame = frames[symbol].reindex(calendar)
        _require(set(FEATURES).issubset(frame.columns),
                 f"{symbol}: atributos do Control ausentes")
        features = frame.loc[:, FEATURES].replace([np.inf, -np.inf], np.nan)
        open_next = pd.to_numeric(frame["open"], errors="coerce").shift(-1)
        close = pd.to_numeric(frame["close"], errors="coerce")
        low = pd.to_numeric(frame["low"], errors="coerce")
        future_lows = pd.concat(
            [low.shift(-n) for n in range(1, settings.downside_horizon + 1)], axis=1,
        )
        # min(skipna=False) nao ignora lacunas e nao reconstrui ativo excluido.
        minimum = future_lows.min(axis=1, skipna=False)
        valid_risk = (
            open_next.gt(0) & minimum.gt(0)
            & np.isfinite(open_next) & np.isfinite(minimum)
        )
        downside = ((minimum / open_next - 1.0) <= settings.downside_event).astype(float)
        downside = downside.where(valid_risk, np.nan)
        future_closes = pd.concat(
            [close.shift(-n) for n in range(1, settings.reversal_horizon + 1)], axis=1,
        )
        valid_reversal = (
            open_next.gt(0)
            & (future_closes.gt(0) & np.isfinite(future_closes)).all(axis=1)
        )
        reversal = (
            close.shift(-settings.reversal_horizon) / open_next - 1.0
            >= settings.recovery_event
        ).astype(float).where(valid_reversal, np.nan)
        recent_drawdown = pd.to_numeric(
            frame["distance_from_high_20"], errors="coerce",
        ).le(settings.recent_drawdown_trigger)
        part = features.copy()
        part["date"] = calendar
        part["asset"] = symbol
        part["calendar_position"] = np.arange(len(calendar), dtype=int)
        part["recent_drawdown"] = recent_drawdown.to_numpy(dtype=bool)
        part["downside_event"] = downside.to_numpy(dtype=float)
        part["reversal_event"] = reversal.to_numpy(dtype=float)
        result.append(part.reset_index(drop=True))
    panel = pd.concat(result, ignore_index=True)
    return panel.replace([np.inf, -np.inf], np.nan)


class _ConstantProbability:
    """Quando o passado contem somente uma classe, usar frequencia observada."""
    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, x):
        p = np.full(len(x), self.probability, dtype=float)
        return np.column_stack((1 - p, p))


def _fit_classification(train: pd.DataFrame, target: str, settings: EarlyRiskSettings):
    valid = train.dropna(subset=[*FEATURES, target])
    if len(valid) < settings.min_train_samples:
        return None, int(len(valid)), None
    y = valid[target].astype(int).to_numpy()
    prevalence = float(y.mean())
    if len(np.unique(y)) < 2:
        return _ConstantProbability(prevalence), len(y), prevalence
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=400, random_state=settings.random_state),
    )
    model.fit(valid[list(FEATURES)], y)
    return model, len(y), prevalence


def estimar_folds(
    panel: pd.DataFrame,
    folds: list[dict],
    calendar: pd.DatetimeIndex,
    settings: EarlyRiskSettings,
) -> tuple[dict[int, dict[pd.Timestamp, dict[str, dict]]], pd.DataFrame, pd.DataFrame]:
    """Treina por fold sem rotulos cujo t+h alcança o primeiro teste.

    Alvos futuros no painel nao sao usados durante inferencia: somente no
    filtro de maturacao e no diagnostico, apos predict_proba.
    """
    calendar = pd.DatetimeIndex(calendar)
    _require(not panel.empty, "Painel sem registros")
    data = panel.copy()
    dates = pd.to_datetime(data["date"], utc=True)
    _require(not dates.isna().any(), "Datas de painel invalidas")
    valid_features = data[list(FEATURES)].notna().all(axis=1)
    all_signals: dict[int, dict[pd.Timestamp, dict[str, dict]]] = {}
    fold_stats = []
    predictions = []
    for fold in folds:
        fid = int(fold["fold_id"])
        first_test = int(fold["test_start_index"])
        last_test = int(fold["test_end_index"])
        final_fit_end = int(fold["final_fit_end_index"])
        _require(0 < final_fit_end < first_test < last_test <= len(calendar),
                 f"Fold {fid}: fronteiras incorretas")
        base_train = valid_features & (
            data["calendar_position"] < final_fit_end
        )
        risk_train = data.loc[
            base_train &
            (data["calendar_position"] + settings.downside_horizon < first_test)
        ]
        recovery_train = data.loc[
            base_train &
            data["recent_drawdown"].astype(bool) &
            (data["calendar_position"] + settings.reversal_horizon < first_test)
        ]
        risk, n_risk, prevalence_risk = _fit_classification(
            risk_train, "downside_event", settings,
        )
        recovery, n_recovery, prevalence_recovery = _fit_classification(
            recovery_train, "reversal_event", settings,
        )
        fold_stats.append({
            "fold": fid,
            "first_test": calendar[first_test].isoformat(),
            "last_train_feature_position": final_fit_end - 1,
            "max_train_risk_maturity_position": first_test - 1,
            "max_train_recovery_maturity_position": first_test - 1,
            "n_risk": n_risk, "n_recovery": n_recovery,
            "risk_frequency_train": prevalence_risk,
            "recovery_frequency_train": prevalence_recovery,
            "risk_model_available": risk is not None,
            "recovery_model_available": recovery is not None,
        })
        # As datas de decisao incluem a sessao imediatamente anterior ao
        # primeiro dia de execucao do fold, conforme o Control oficial.
        test_decision_start = first_test - 1
        test_data = data.loc[
            valid_features &
            data["calendar_position"].between(
                test_decision_start, last_test - 1,
            )
        ].copy()
        if test_data.empty:
            raise ValueError(f"Fold {fid}: nenhuma feature disponivel")
        pr = np.full(len(test_data), np.nan)
        pb = np.full(len(test_data), np.nan)
        if risk is not None:
            pr[:] = risk.predict_proba(test_data[list(FEATURES)])[:, 1]
        if recovery is not None:
            candidate_mask = test_data["recent_drawdown"].astype(bool).to_numpy()
            if candidate_mask.any():
                pb[candidate_mask] = recovery.predict_proba(
                    test_data.loc[candidate_mask, list(FEATURES)],
                )[:, 1]
        test_data["p_downside"] = pr
        test_data["p_recovery"] = pb
        test_data["fold"] = fid
        # Rotulos aparecem no CSV apenas para auditoria ex post.
        # Regras de escolha so recebem os campos de previsao.
        predictions.append(test_data.loc[:, [
            "fold", "date", "asset", "calendar_position",
            "recent_drawdown", "p_downside", "p_recovery",
            "downside_event", "reversal_event",
        ]])
        per_day: dict[pd.Timestamp, dict[str, dict]] = {}
        for row in test_data.itertuples(index=False):
            day = pd.Timestamp(row.date)
            per_day.setdefault(day, {})[row.asset] = {
                "p_downside": float(row.p_downside) if np.isfinite(row.p_downside) else None,
                "p_recovery": float(row.p_recovery) if np.isfinite(row.p_recovery) else None,
                "recent_drawdown": bool(row.recent_drawdown),
            }
        all_signals[fid] = per_day
    return all_signals, pd.DataFrame(fold_stats), pd.concat(predictions, ignore_index=True)


def adaptar_politica_rotacao(
    base_policy: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    symbols: list[str],
    utility_cache: dict[pd.Timestamp, np.ndarray],
    signals: dict[pd.Timestamp, dict[str, dict]],
    settings: EarlyRiskSettings,
    *,
    enable_recovery: bool,
    audit: list[dict],
    decision_diagnostics: dict[pd.Timestamp, dict[str, Any]] | None = None,
) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:
    """A rotação do Control e SEMPRE a primeira avaliacao.

    Se a opcao do Control nao estiver em risco, mantem-se sua decisao,
    exceto na variante com reversao e vantagem ajustada demonstrada.
    CASH e a ultima alternativa quando o alvo permanece sob alto risco
    e nenhum outro ativo investivel atende ao criterio.
    """
    _require(len(symbols) == len(set(symbols)), "Simbolos duplicados")
    names = ["CASH", *symbols]

    def policy(date: pd.Timestamp, current: int, holding: int) -> tuple[int, float]:
        ts = pd.Timestamp(date)
        base_action, base_score = base_policy(ts, current, holding)
        utilities = utility_cache.get(ts)
        _require(utilities is not None and len(utilities) == len(names),
                 "Utilidades nao correspondem ao calendario Control")
        by_asset = signals.get(ts, {})
        p_risk: dict[int, float | None] = {}
        p_recovery: dict[int, float | None] = {}
        revised: dict[int, float] = {}
        eligible: list[int] = []
        for j in range(1, len(names)):
            u = float(utilities[j])
            if not np.isfinite(u):
                continue
            item = by_asset.get(names[j], {})
            risk = item.get("p_downside")
            recover = item.get("p_recovery") if enable_recovery else None
            p_risk[j] = risk
            p_recovery[j] = recover
            if risk is None or not np.isfinite(float(risk)):
                continue
            adjusted = u - settings.risk_penalty * max(
                float(risk) - settings.downside_alert_probability, 0.0,
            )
            if (enable_recovery and recover is not None
                    and np.isfinite(float(recover))
                    and bool(item.get("recent_drawdown"))
                    and float(recover) >= settings.reversal_alert_probability
                    and float(risk) < settings.downside_alert_probability):
                adjusted += settings.recovery_bonus * (
                    float(recover) - settings.reversal_alert_probability
                )
            revised[j] = float(adjusted)
            if (float(risk) < settings.downside_alert_probability
                    and u > 0 and adjusted > 0):
                eligible.append(j)

        target = int(base_action)
        score = float(base_score)
        reason = "CONTROL_PRESERVED"
        base_risk = p_risk.get(target) if target > 0 else None
        base_high = (base_risk is not None
                     and base_risk >= settings.downside_alert_probability)
        current_risk = p_risk.get(current) if current > 0 else None
        current_high = (current_risk is not None
                        and current_risk >= settings.downside_alert_probability)

        # Sempre permitir que o Control efetue sua rotacao original.
        # Intervir quando o alvo seria arriscado ou quando uma reversao
        # oferece vantagem AJUSTADA sobre o alvo sem risco elevado.
        if base_high:
            if eligible:
                winner = max(eligible, key=lambda j: (revised[j], -j))
                target = winner
                score = float(utilities[winner])
                reason = ("RISK_ROTATE_TO_ALTERNATIVE"
                          if winner != base_action else "CONTROL_PRESERVED")
            else:
                # Nenhum candidato seguro: CASH e alternativa de ultima instancia.
                target, score = 0, 0.0
                reason = "RISK_NO_ACCEPTABLE_ROTATION"
        elif (enable_recovery and eligible and target > 0 and not base_high):
            candidates = [
                j for j in eligible
                if p_recovery.get(j) is not None
                and p_recovery[j] >= settings.reversal_alert_probability
                and by_asset.get(names[j], {}).get("recent_drawdown")
            ]
            if candidates:
                winner = max(candidates, key=lambda j: (revised[j], -j))
                base_adjusted = revised.get(target, float(utilities[target]))
                if (winner != target
                        and revised[winner] >= base_adjusted
                        + settings.minimum_adjusted_advantage):
                    target = winner
                    score = float(utilities[winner])
                    reason = "RECOVERY_OPPORTUNITY_ROTATION"
        # Se nao ha score de risco (warmup/modelo indisponivel) preservar Control,
        # inclusive holding, switch-margin e reentrada original.
        if decision_diagnostics is not None:
            diagnostic = decision_diagnostics.setdefault(ts, {})
            diagnostic["risk_reversal_control_target"] = names[int(base_action)]
            diagnostic["risk_reversal_final_target"] = names[int(target)]
            diagnostic["risk_reversal_reason"] = reason
            diagnostic["risk_reversal_changed_target"] = bool(target != base_action)
            diagnostic["risk_reversal_p_downside_target"] = base_risk
            if target != base_action:
                # A decisao no replay deve corresponder ao ativo efetivamente
                # executado, nao ao alvo original do Control.
                diagnostic["final_action_asset"] = names[int(target)]
                diagnostic["final_action_score"] = float(score)
                diagnostic["decision_reason"] = reason
                diagnostic["q_final_action"] = float(score)
                diagnostic["decision_is_exit_to_cash"] = bool(current > 0 and target == 0)
                diagnostic["decision_is_rotation"] = bool(
                    current > 0 and target > 0 and current != target
                )
        audit.append({
            "decision_date": ts.isoformat(),
            "current_asset": names[current],
            "control_target": names[int(base_action)],
            "final_target": names[int(target)],
            "control_score": float(base_score),
            "p_downside_control_target": base_risk,
            "p_downside_current": current_risk,
            "control_target_high_risk": bool(base_high),
            "current_position_high_risk": bool(current_high),
            "recovery_enabled": bool(enable_recovery),
            "reason": reason,
            "changed_control_action": bool(target != base_action),
            "cash_last_resort": bool(reason == "RISK_NO_ACCEPTABLE_ROTATION"),
        })
        return target, score

    return policy
