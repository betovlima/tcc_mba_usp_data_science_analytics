"""Vantagem contrafactual de CASH contra continuar seguindo o Control.

Os rótulos historicos usam uma politica Control treinada antes da janela de
calibracao e duas trajetorias distintas: acao Control ou CASH uma vez, depois
retorno a mesma regra Control, com estado de posicao proprio em cada ramo.
Aprendizado Ridge recebe somente indicadores conhecidos no fechamento t.
Nenhum retorno futuro e consultado ao prever datas do teste OOS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from pesquisa.referencia_strategy30.rotacao import _retorno_log_transicao_treinamento

Policy = Callable[[pd.Timestamp, int, int], tuple[int, float]]
FEATURE_COLUMNS = (
    "return_5", "return_20", "vol_20", "distance_from_high_20",
    "trend_efficiency_20", "control_score", "cash_position", "holding_days",
)


@dataclass(frozen=True)
class CounterfactualSettings:
    horizon: int = 10
    train_fraction: float = 0.75
    ridge_alpha: float = 10.0
    min_train_samples: int = 45
    min_validation_samples: int = 12

    def __post_init__(self) -> None:
        if self.horizon < 1 or not 0.5 <= self.train_fraction <= 0.9:
            raise ValueError("Horizonte/fracao cronologica invalidos.")
        if self.ridge_alpha <= 0 or not np.isfinite(self.ridge_alpha):
            raise ValueError("Regularizacao Ridge invalida.")
        if self.min_train_samples < 10 or self.min_validation_samples < 5:
            raise ValueError("Amostras minimas invalidas.")


def observable_features(
    frame: pd.DataFrame,
    timestamp: pd.Timestamp,
    *,
    control_score: float,
    current_position: int,
    holding_days: int,
) -> dict[str, float]:
    """Somente atributos backward-looking do candidato selecionado."""
    row = frame.loc[pd.Timestamp(timestamp)]
    result = {
        "return_5": float(row["return_5"]),
        "return_20": float(row["return_20"]),
        "vol_20": float(row["vol_20"]),
        "distance_from_high_20": float(row["distance_from_high_20"]),
        "trend_efficiency_20": float(row["trend_efficiency_20"]),
        "control_score": float(control_score),
        "cash_position": float(current_position == 0),
        "holding_days": float(min(max(holding_days, 0), 20) / 20.0),
    }
    if not np.isfinite(list(result.values())).all():
        raise ValueError("Atributos contextuais ausentes ou nao finitos.")
    return result


def _next_state(
    position: int,
    holding: int,
    target: int,
) -> tuple[int, int]:
    if position == target:
        return position, (holding + 1 if target > 0 else 0)
    return target, (1 if target > 0 else 0)


def _rollout(
    policy: Policy,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
    start: int,
    position: int,
    holding: int,
    config: Any,
    horizon: int,
    *,
    initial_forced_cash: bool,
) -> tuple[float, int]:
    """Retorno bruto logaritmico de h transicoes, incluindo custos proporcionais."""
    total = 0.0
    cash_steps = 0
    for step in range(horizon):
        date = pd.Timestamp(dates[start + step])
        following = pd.Timestamp(dates[start + step + 1])
        action, _ = policy(date, position, holding)
        if step == 0 and initial_forced_cash:
            action = 0
        if action == 0:
            cash_steps += 1
        total += _retorno_log_transicao_treinamento(
            frames, symbols, date, following,
            position, action, config,
        )
        position, holding = _next_state(position, holding, action)
    return float(total), cash_steps


def counterfactual_cash_advantage(
    policy: Policy,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
    *,
    start: int,
    current_position: int,
    holding_days: int,
    config: Any,
    settings: CounterfactualSettings,
) -> tuple[float, float, float, int]:
    """Rotulo CASH menos Control, comparando ramificacoes completas.

    O primeiro ramo segue Control em t e em todas as etapas seguintes.
    O segundo faz CASH na primeira etapa e segue Control nos demais passos.
    Cada ramo recalcula a politica usando sua propria posicao e holding.
    Os preços t+1,...,t+h sao conhecidos APENAS ao criar rotulos historicos.
    """
    if start < 0 or start + settings.horizon >= len(dates):
        raise ValueError("Rotulo ultrapassaria janela historica autorizada.")
    control_log, _ = _rollout(
        policy, frames, symbols, dates, start,
        current_position, holding_days, config, settings.horizon,
        initial_forced_cash=False,
    )
    cash_log, cash_steps = _rollout(
        policy, frames, symbols, dates, start,
        current_position, holding_days, config, settings.horizon,
        initial_forced_cash=True,
    )
    return cash_log - control_log, cash_log, control_log, cash_steps


def build_calibration_samples(
    policy: Policy,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    dates: pd.DatetimeIndex,
    *,
    config: Any,
    settings: CounterfactualSettings,
    fold_id: int,
) -> pd.DataFrame:
    """Rotulos off-policy em trajetorias historicas com CASH persistente.

    O estado na data t depende apenas das escolhas deterministicas anteriores
    e da politica Control treinada antes de calibration; nao usa futuros.
    Cada rotulo simula CASH por UMA decisao vs executar Control, e deixa
    ambos os ramos voltarem a politica normal nas decisoes subsequentes.
    Todos os valores futuros usados nos rotulos ficam em calibration.
    Os periodos de exploracao foram predefinidos, nunca otimizados no OOS.
    """
    dates = pd.DatetimeIndex(dates)
    if len(dates) <= settings.horizon or not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("Calendario de calibracao insuficiente/invalidado.")
    # A primeira trajetoria segue Control. As demais permanecem em CASH
    # por janelas de tamanhos variados, independentemente dos retornos.
    scenarios = (
        ("control_shadow", 0, 1),
        ("cash_1_of_7", 1, 7),
        ("cash_2_of_11", 2, 11),
        ("cash_4_of_15", 4, 15),
        ("cash_8_of_20", 8, 20),
        ("cash_13_of_31", 13, 31),
    )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[pd.Timestamp, int, int, int]] = set()
    for scenario, forced_days, cycle in scenarios:
        position = 0
        holding = 0
        for i, current_date in enumerate(dates[:-1]):
            date = pd.Timestamp(current_date)
            control_target, score = policy(date, position, holding)
            if control_target > 0 and i + settings.horizon < len(dates):
                key = (date, int(position), int(holding), int(control_target))
                if key not in seen:
                    seen.add(key)
                    candidate = symbols[control_target - 1]
                    features = observable_features(
                        frames[candidate], date, control_score=score,
                        current_position=position, holding_days=holding,
                    )
                    advantage, cash_log, control_log, cash_steps = (
                        counterfactual_cash_advantage(
                            policy, frames, symbols, dates,
                            start=i, current_position=position, holding_days=holding,
                            config=config, settings=settings,
                        )
                    )
                    rows.append({
                        "fold_id": int(fold_id),
                        "scenario": scenario,
                        "decision_date": date,
                        "label_last_date": pd.Timestamp(dates[i + settings.horizon]),
                        "current_asset": symbols[position - 1] if position else "CASH",
                        "control_asset": candidate,
                        "holding_before": int(holding),
                        "label_cash_minus_control_log_return": float(advantage),
                        "counterfactual_cash_log_return": float(cash_log),
                        "counterfactual_control_log_return": float(control_log),
                        "counterfactual_cash_steps": int(cash_steps),
                        **features,
                    })
            forced_cash = forced_days > 0 and (i % cycle) < forced_days
            action = 0 if forced_cash else control_target
            position, holding = _next_state(position, holding, int(action))
    samples = pd.DataFrame(rows)
    if samples.empty:
        raise RuntimeError("Nenhum exemplo de calibracao com politica congelada.")
    if int((samples["cash_position"] > 0).sum()) < 20:
        raise RuntimeError(
            "Amostras em estado CASH insuficientes; nao substituir por "
            "dados de outro periodo ou por treino no conjunto OOS."
        )
    if samples["label_last_date"].max() > dates[-1]:
        raise RuntimeError("Rotulo ultrapassou a ultima data de calibracao.")
    return samples.sort_values(
        ["decision_date", "scenario", "current_asset", "holding_before"],
    ).reset_index(drop=True)


@dataclass(frozen=True)
class CounterfactualFit:
    model: Any
    training_rows: int
    validation_rows: int
    calibration_rows: int
    fit_last_label_date: pd.Timestamp
    validation_first_date: pd.Timestamp
    validation_mae: float | None
    validation_direction_accuracy: float | None

    def predict(self, features: dict[str, float]) -> float:
        vector = pd.DataFrame(
            [{key: features[key] for key in FEATURE_COLUMNS}],
            columns=list(FEATURE_COLUMNS),
        )
        prediction = float(self.model.predict(vector)[0])
        if not np.isfinite(prediction):
            raise RuntimeError("Estimativa contrafactual nao finita.")
        return prediction


def fit_counterfactual_advantage(
    samples: pd.DataFrame,
    calibration_dates: pd.DatetimeIndex,
    *,
    settings: CounterfactualSettings,
) -> tuple[CounterfactualFit, pd.DataFrame]:
    """Treino/validacao com embargo >= horizonte, sem dados do teste OOS."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if samples.empty or not pd.DatetimeIndex(calibration_dates).is_monotonic_increasing:
        raise ValueError("Calibracao contrafactual vazia/desordenada.")
    required = [
        "decision_date", "label_last_date",
        "label_cash_minus_control_log_return", *FEATURE_COLUMNS,
    ]
    if samples[required].isna().any().any():
        raise ValueError("Amostra contrafactual incompleta.")
    cutoff_index = int(len(calibration_dates) * settings.train_fraction)
    if cutoff_index <= settings.horizon or cutoff_index >= len(calibration_dates):
        raise ValueError("Separacao de treino insuficiente.")
    split_date = pd.Timestamp(calibration_dates[cutoff_index])
    safe_training = samples.loc[samples["label_last_date"] < split_date].copy()
    validation = samples.loc[samples["decision_date"] >= split_date].copy()
    if len(safe_training) < settings.min_train_samples:
        raise RuntimeError(
            f"Treino contrafactual insuficiente: {len(safe_training)} "
            f"< {settings.min_train_samples}."
        )
    if len(validation) < settings.min_validation_samples:
        raise RuntimeError(
            f"Validacao contrafactual insuficiente: {len(validation)} "
            f"< {settings.min_validation_samples}."
        )
    train_end = pd.Timestamp(safe_training["label_last_date"].max())
    valid_start = pd.Timestamp(validation["decision_date"].min())
    if train_end >= valid_start:
        raise RuntimeError("Embargo cronologico violado nos rotulos.")
    def create_model() -> Any:
        return make_pipeline(
            StandardScaler(),
            Ridge(alpha=settings.ridge_alpha),
        )
    target = "label_cash_minus_control_log_return"
    model_for_validation = create_model()
    model_for_validation.fit(safe_training[list(FEATURE_COLUMNS)], safe_training[target])
    validation_predictions = model_for_validation.predict(validation[list(FEATURE_COLUMNS)])
    actual = validation[target].to_numpy(dtype=float)
    predictions = np.asarray(validation_predictions, dtype=float)
    evaluation = validation[[
        "fold_id", "decision_date", "label_last_date", "control_asset", target,
    ]].copy()
    evaluation["predicted_cash_minus_control_log_return"] = predictions
    evaluation["prediction_error"] = predictions - actual
    evaluation["source"] = "inner_chronological_validation"
    mae = float(np.mean(np.abs(predictions - actual)))
    direction_accuracy = float(np.mean((predictions > 0) == (actual > 0)))
    # Avaliacao dos dados reservados e apenas diagnostica, sem selecao de
    # hiperparametros; modelo final usa todos os rotulos de calibracao.
    final = create_model()
    final.fit(samples[list(FEATURE_COLUMNS)], samples[target])
    return CounterfactualFit(
        model=final,
        training_rows=int(len(safe_training)),
        validation_rows=int(len(validation)),
        calibration_rows=int(len(samples)),
        fit_last_label_date=pd.Timestamp(samples["label_last_date"].max()),
        validation_first_date=valid_start,
        validation_mae=mae,
        validation_direction_accuracy=direction_accuracy,
    ), evaluation


def build_counterfactual_overlay(
    base_policy: Policy,
    *,
    fits_by_fold: dict[int, CounterfactualFit],
    decision_to_fold: dict[pd.Timestamp, int],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    diagnostics: dict[pd.Timestamp, dict[str, Any]],
) -> Policy:
    """CASH se E[cash - Control | contexto] > 0. Empate segue Control."""
    def policy(date: pd.Timestamp, position: int, holding: int) -> tuple[int, float]:
        dt = pd.Timestamp(date)
        fold_id = int(decision_to_fold[dt])
        control_target, control_score = base_policy(dt, position, holding)
        if control_target <= 0:
            diagnostics[dt] = {
                "fold_id": fold_id,
                "current_asset": symbols[position - 1] if position else "CASH",
                "control_target": "CASH",
                "final_target": "CASH",
                "reason": "control_selected_cash",
                "predicted_cash_minus_control": None,
            }
            return 0, 0.0
        candidate = symbols[control_target - 1]
        features = observable_features(
            frames[candidate], dt, control_score=control_score,
            current_position=position, holding_days=holding,
        )
        fit = fits_by_fold[fold_id]
        advantage = fit.predict(features)
        prefer_cash = advantage > 0
        target = 0 if prefer_cash else control_target
        diagnostics[dt] = {
            "fold_id": fold_id,
            "current_asset": symbols[position - 1] if position else "CASH",
            "control_target": candidate,
            "final_target": symbols[target - 1] if target else "CASH",
            "reason": "predicted_cash_advantage" if prefer_cash else "continue_control",
            "predicted_cash_minus_control": advantage,
            "control_score": float(control_score),
            "train_rows": fit.training_rows,
            "calibration_rows": fit.calibration_rows,
            "last_calibration_label_date": fit.fit_last_label_date.isoformat(),
            **{f"feature_{k}": v for k, v in features.items()},
        }
        return (0, 0.0) if prefer_cash else (control_target, control_score)
    return policy
