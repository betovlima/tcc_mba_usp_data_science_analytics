"""Pesquisa TCC: propostas LHS + processo gaussiano no universo congelado.

Este modulo e puro: nao carrega mercado, nao executa ordens, nao toca no Control.
A pontuacao economica deve ser calculada SOMENTE na janela interna anterior
à avaliação OOS, pelo executor Spyder.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import numpy as np
import warnings
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

VERSION = "u75-nested-hyperparameter-research-v1.0.0"

# Oito domínios documentados na campanha histórica de 13/08/2026.
BOUNDS = (
    (220, 380),       # n_estimators
    (0.020, 0.050),   # learning_rate
    (2, 4),           # max_depth
    (4, 12),          # num_leaves; condicionado pela profundidade
    (15, 30),         # min_child_samples
    (0.75, 0.95),     # colsample_bytree
    (0.0, 0.50),      # reg_alpha
    (1.0, 4.0),       # reg_lambda
)
PARAMETERS = (
    "n_estimators", "learning_rate", "max_depth", "num_leaves",
    "min_child_samples", "colsample_bytree", "reg_alpha", "reg_lambda",
)
INTEGER_KEYS = frozenset({"n_estimators", "max_depth", "num_leaves", "min_child_samples"})


@dataclass(frozen=True)
class SearchSettings:
    budget: int = 8
    lhs_proposals: int = 4
    seed: int = 42
    acquisition_candidates: int = 1024
    xi: float = 0.01

    def __post_init__(self) -> None:
        if self.budget < 2 or self.lhs_proposals < 2 or self.lhs_proposals > self.budget:
            raise ValueError("Orcamento LHS/adaptativo invalido")
        if self.acquisition_candidates < 16:
            raise ValueError("Aquisicao precisa de pelo menos 16 propostas")


def decode_point(point: np.ndarray, original: Mapping[str, object]) -> dict[str, object]:
    """Ajusta só oito parametros; demais opções continuam do Control."""
    x = np.asarray(point, dtype=float)
    if x.shape != (len(PARAMETERS),) or not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
        raise ValueError("Ponto da busca fora do dominio normalizado")
    params = dict(original)
    for i, key in enumerate(PARAMETERS):
        a, b = BOUNDS[i]
        raw = a + (b - a) * float(x[i])
        params[key] = int(round(raw)) if key in INTEGER_KEYS else float(raw)
    params["num_leaves"] = min(int(params["num_leaves"]), 2 ** int(params["max_depth"]))
    return params


def parameter_signature(params: Mapping[str, object]) -> tuple:
    return tuple(round(float(params[key]), 9) if key not in INTEGER_KEYS else int(params[key]) for key in PARAMETERS)


def _lhs(count: int, dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.empty((count, dimension), dtype=float)
    for col in range(dimension):
        strata = (np.arange(count) + rng.random(count)) / count
        x[:, col] = rng.permutation(strata)
    return x


def propose(
    history: list[dict],
    original: Mapping[str, object],
    *,
    settings: SearchSettings,
    fold_id: int,
) -> tuple[np.ndarray, dict, str]:
    """Proxima proposta deterministica baseada apenas nos scores internos.

    Registro 0 é a configuracao historica (âncora). Depois LHS; demais GP/EI.
    Se o GP for mal condicionado, usa candidato aleatorio deterministico.
    """
    n = len(history)
    if n >= settings.budget:
        raise ValueError("Orcamento de pesquisa esgotado")
    used = {parameter_signature(row["parameters"]) for row in history}
    if n == 0:
        keys = []
        for i, key in enumerate(PARAMETERS):
            low, high = BOUNDS[i]
            keys.append(np.clip((float(original[key]) - low) / (high - low), 0.0, 1.0))
        x = np.asarray(keys, dtype=float)
        return x, dict(original), "frozen_anchor"
    seed = settings.seed + 1009 * int(fold_id)
    initial = _lhs(settings.lhs_proposals - 1, len(PARAMETERS), seed)
    if n < settings.lhs_proposals:
        x = initial[n - 1]
        candidate = decode_point(x, original)
        if parameter_signature(candidate) not in used:
            return x, candidate, "lhs"
    rng = np.random.default_rng(seed + 97 * n)
    candidates = rng.uniform(0, 1, (settings.acquisition_candidates, len(PARAMETERS)))
    search_method = "gaussian_process_ei"
    scores = np.asarray([float(row["objective"]) for row in history], dtype=float)
    if (n >= settings.lhs_proposals and np.isfinite(scores).all() and np.std(scores) > 1e-12):
        try:
            x_train = np.asarray([row["normalized"] for row in history], dtype=float)
            mean, deviation = float(scores.mean()), float(scores.std())
            y = (scores - mean) / deviation
            gp = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0, constant_value_bounds="fixed")
                * Matern(length_scale=np.ones(len(PARAMETERS)), length_scale_bounds=(0.1, 10), nu=2.5)
                + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 0.1)),
                normalize_y=False,
                random_state=seed,
                n_restarts_optimizer=0,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                gp.fit(x_train, y)
            mu, sd = gp.predict(candidates, return_std=True)
            improv = mu - float(y.max()) - settings.xi
            z = np.divide(improv, sd, out=np.zeros_like(improv), where=sd > 1e-12)
            ei = improv * norm.cdf(z) + sd * norm.pdf(z)
            ei = np.where(sd > 1e-12, ei, 0.0)
            candidates = candidates[np.argsort(-ei, kind="stable")]
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            search_method = "random_fallback"
    else:
        search_method = "random_fallback"
    for x in candidates:
        params = decode_point(x, original)
        if parameter_signature(params) not in used:
            return x, params, search_method
    raise RuntimeError("Nao existem mais propostas distintas sob os limites")


def validate_fold_windows(train, optimize, calibrate, outer_test) -> None:
    """Defende a ordem temporal no corte e impede treino do futuro."""
    windows = [list(w) for w in (train, optimize, calibrate, outer_test)]
    if any(len(w) < 2 for w in windows):
        raise ValueError("Janelas cronologicas insuficientes")
    if not (max(windows[0]) < min(windows[1]) < max(windows[1]) < min(windows[2])
            < max(windows[2]) < min(windows[3])):
        raise ValueError("Vazamento temporal ou sobreposicao entre janelas")


def choose_best(history: list[dict]) -> dict:
    """Empate favorece parametros historicos (trial 0), nao mais complexidade."""
    if not history or not all(isfinite(float(r["objective"])) for r in history):
        raise ValueError("Nenhum score interno valido")
    return max(history, key=lambda row: (float(row["objective"]), -int(row["trial"])))
