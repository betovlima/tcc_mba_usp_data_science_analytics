"""Loss orientada a decisao com softmax diferenciavel por grupo de alternativas.

Otimizamos o regret esperado de uma escolha softmax, nao MAE/RMSE. O gradiente
e analitico e a Hessiana e aproximacao positiva para o boosting de arvores.
A decisao OOS real usa argmax; o capital sequencial e avaliado separadamente
pelo simulador. Nao alegar backprop atraves do replay multiperiodo.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from reproducao.decision_focused import FEATURE_COLUMNS
from reproducao.decision_focused_v2 import SEMENTE

TEMPERATURA = 1.0
ESCALA_RECOMPENSA = 100.0


def gradiente_regret_softmax(
    prediction: np.ndarray,
    rewards: np.ndarray,
    groups: list[tuple[int, int]],
    *,
    temperature: float = TEMPERATURA,
    scale: float = ESCALA_RECOMPENSA,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradiente da perda max(r) - E_softmax[r], hessiana diagonal positiva.

    Loss por grupo: L = max_i(r_i) - sum_i softmax(z_i/T) r_i.
    dL/dz_i = p_i * (E[r]-r_i)/T. A Hessiana nao e PSD em geral;
    uma aproximacao Fisher positiva estabiliza o boosting.
    """
    z = np.asarray(prediction, dtype=float)
    r = np.asarray(rewards, dtype=float)
    if z.shape != r.shape or z.ndim != 1:
        raise ValueError("Predicoes e vantagens devem ser vetores iguais")
    if not np.isfinite(z).all() or not np.isfinite(r).all():
        raise ValueError("Valores nao finitos na loss de decisao")
    if temperature <= 0 or scale <= 0:
        raise ValueError("Temperatura e escala devem ser positivas")
    grad = np.zeros(len(z), dtype=float)
    hess = np.zeros(len(z), dtype=float)
    covered = np.zeros(len(z), dtype=bool)
    for start, end in groups:
        if not (0 <= start < end <= len(z)):
            raise ValueError("Intervalo de grupo invalido")
        if covered[start:end].any():
            raise ValueError("Grupos sobrepostos")
        covered[start:end] = True
        centered = (z[start:end] - float(np.max(z[start:end]))) / temperature
        probability = np.exp(np.clip(centered, -50, 0))
        probability /= float(np.sum(probability))
        expectation = float(np.dot(probability, r[start:end]))
        grad[start:end] = (
            scale * probability * (expectation - r[start:end]) / temperature
        )
        # Hessiana diagonal de Fisher (positiva), nao a Hessiana exata,
        # pois a loss esperada de decisao nao e convexa.
        hess[start:end] = (
            scale * np.maximum(probability * (1 - probability), 1e-4)
            / (temperature * temperature)
        )
    if not covered.all():
        raise ValueError("Ha candidatos fora dos grupos")
    return grad, hess


def treinar_softmax_dfl(labels: pd.DataFrame) -> Any:
    from lightgbm import LGBMRegressor

    # Construir grupos contiguos sem misturar datas nem folds.
    groups: list[tuple[int, int]] = []
    ordered = []
    cursor = 0
    for _, group in labels.groupby(["fold_id", "decision_date"], sort=True):
        if len(group) < 2:
            continue
        ordered.append(group)
        groups.append((cursor, cursor + len(group)))
        cursor += len(group)
    if not ordered:
        raise ValueError("Nenhum grupo para treinamento orientado a decisao")
    data = pd.concat(ordered, ignore_index=True)
    x = data.loc[:, FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    target = data["log_advantage"].to_numpy(dtype=float)

    def objective(y_true: np.ndarray, y_pred: np.ndarray):
        return gradiente_regret_softmax(y_pred, y_true, groups)

    model = LGBMRegressor(
        objective=objective, n_estimators=120, learning_rate=0.035,
        max_depth=3, num_leaves=7, min_child_samples=8,
        random_state=SEMENTE, n_jobs=1, verbosity=-1,
        deterministic=True, force_col_wise=True,
    )
    model.fit(x, target)
    return model


def escolher_softmax(
    model: Any, group: pd.DataFrame, control: int,
) -> tuple[int, float]:
    x = group.loc[:, FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    pos = group["candidate_position"].to_numpy(dtype=int)
    base_matches = np.flatnonzero(pos == control)
    if len(base_matches) != 1:
        raise ValueError("Grupo deve conter exatamente uma acao Control")
    score = np.asarray(model.predict(x), dtype=float)
    if not np.isfinite(score).all():
        return int(control), 0.0
    baseline = int(base_matches[0])
    best = int(np.argmax(score))
    return int(pos[best]), float(score[best] - score[baseline])
