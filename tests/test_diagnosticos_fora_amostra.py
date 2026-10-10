from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from engine.modelo_lightgbm import diagnosticos_fora_amostra


def test_erros_agregados_usam_rotulos_validos_de_teste_sem_confundir_com_treino():
    dates = pd.date_range("2026-01-05", periods=2, freq="B", tz="UTC")
    frames = {
        "AAA": pd.DataFrame({"forward_risk_adjusted_utility": [1.0, np.nan]}, index=dates),
        "BBB": pd.DataFrame({"forward_risk_adjusted_utility": [3.0, 100.0]}, index=dates),
    }
    models = {
        s: SimpleNamespace(_fit_diagnostics={"train": {"mae": 999.0}, "feature_importance_gain": {"return_1": 1.0}})
        for s in frames
    }
    cache = {dates[0]: np.array([0.0, 2.0, 1.0]), dates[1]: np.array([0.0, 200.0, -np.inf])}
    result = diagnosticos_fora_amostra(models, frames, list(frames), dates, cache, fold_id=2)
    assert result["rows"] == 2
    assert result["mae"] == pytest.approx(1.5)
    assert result["rmse"] == pytest.approx(np.sqrt(2.5))
    assert result["fold_id"] == 2
    assert result["by_asset"]["AAA"]["final_fit_feature_importance_gain"] == {"return_1": 1.0}
    assert "train" not in result["by_asset"]["AAA"]


def test_rotulos_incompletos_nao_sao_preenchidos_com_zero():
    dates = pd.date_range("2026-01-05", periods=1, tz="UTC")
    frames = {"AAA": pd.DataFrame({"forward_risk_adjusted_utility": [np.nan]}, index=dates)}
    result = diagnosticos_fora_amostra(
        {"AAA": SimpleNamespace()}, frames, ["AAA"], dates,
        {dates[0]: np.array([0.0, 1.0])}, fold_id=3,
    )
    assert result["rows"] == 0
    assert result["mae"] is None
    assert result["rmse"] is None
