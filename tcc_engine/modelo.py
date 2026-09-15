"""Treinamento e inferência do LightGBM usado no TCC."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .caracteristicas import CARACTERISTICAS_ROTACAO


def _contexto_threads(configuracao: Any):
    if not bool(configuracao.execucao_deterministica):
        return nullcontext()
    return threadpool_limits(limits=int(configuracao.limite_threads_numericas))


def treinar_modelos_lightgbm(
    quadros: dict[str, pd.DataFrame],
    ativos: list[str],
    datas_treinamento: pd.DatetimeIndex,
    configuracao: Any,
    *,
    progresso: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Treina um regressor LightGBM independente para cada ativo."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError as erro:
        raise RuntimeError("LightGBM não está instalado. Execute: pip install -r requirements.txt") from erro

    hiper = dict(configuracao.hiperparametros_lightgbm)
    modelos: dict[str, Any] = {}
    minimo = int(configuracao.linhas_minimas_treinamento)

    with _contexto_threads(configuracao):
        for posicao, ativo in enumerate(ativos, start=1):
            quadro = quadros[ativo].loc[datas_treinamento].dropna(
                subset=["utilidade_ajustada_risco_futura", *CARACTERISTICAS_ROTACAO]
            )
            if len(quadro) < minimo:
                if ativo in set(configuracao.ativos_ancora_calendario):
                    raise ValueError(
                        f"{ativo}: {len(quadro)} linhas de treinamento válidas; mínimo={minimo}."
                    )
                continue

            modelo = LGBMRegressor(
                objective="regression",
                boosting_type="gbdt",
                n_estimators=int(hiper["n_estimators"]),
                learning_rate=float(hiper["learning_rate"]),
                max_depth=int(hiper["max_depth"]),
                num_leaves=int(hiper["num_leaves"]),
                min_child_samples=int(hiper["min_child_samples"]),
                min_child_weight=float(hiper["min_child_weight"]),
                subsample=float(hiper["subsample"]),
                subsample_freq=int(hiper["subsample_freq"]),
                colsample_bytree=float(hiper["colsample_bytree"]),
                reg_alpha=float(hiper["reg_alpha"]),
                reg_lambda=float(hiper["reg_lambda"]),
                max_bin=int(hiper["max_bin"]),
                random_state=int(configuracao.semente_aleatoria),
                n_jobs=int(hiper["n_jobs"]),
                deterministic=bool(configuracao.execucao_deterministica),
                force_col_wise=bool(configuracao.execucao_deterministica),
                verbosity=-1,
            )
            modelo.fit(quadro[CARACTERISTICAS_ROTACAO], quadro["utilidade_ajustada_risco_futura"])
            modelos[ativo] = modelo
            if progresso is not None:
                progresso(posicao, len(ativos), ativo)
    return modelos


def prever_utilidades(
    modelos: dict[str, Any],
    quadros: dict[str, pd.DataFrame],
    ativos: list[str],
    data: pd.Timestamp,
) -> np.ndarray:
    """Calcula a utilidade prevista de cada ativo na data informada."""
    valores = [0.0]
    for ativo in ativos:
        modelo = modelos.get(ativo)
        quadro = quadros[ativo]
        if modelo is None or data not in quadro.index:
            valores.append(float("-inf"))
            continue
        linha = quadro.loc[[data], CARACTERISTICAS_ROTACAO]
        if linha.empty or linha.isna().any(axis=None):
            valores.append(float("-inf"))
            continue
        local = quadro.index.get_loc(data)
        if not isinstance(local, (int, np.integer)) or local + 1 >= len(quadro.index):
            valores.append(float("-inf"))
            continue
        proxima = quadro.iloc[int(local) + 1]
        abertura = float(proxima.get("open", float("nan")))
        fechamento = float(proxima.get("close", float("nan")))
        if not (np.isfinite(abertura) and abertura > 0 and np.isfinite(fechamento) and fechamento > 0):
            valores.append(float("-inf"))
            continue
        valores.append(float(modelo.predict(linha)[0]))
    return np.asarray(valores, dtype=np.float64)
