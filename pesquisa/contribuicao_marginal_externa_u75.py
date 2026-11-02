"""Comparacao pareada do patrimonio e das decisoes U75 vs U75 + candidato.

Funcoes puras: nao consultam mercado, nao selecionam candidatos, nao alteram
a estrategia, nao usam resultados futuros para decidir uma rotacao.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


COLUMNS = ("selected_asset", "walk_forward_fold", "strategy_equity")


def normalizar_decisoes(frame: pd.DataFrame, descricao: str) -> pd.DataFrame:
    """Preserva a data de execucao; reprova duplicatas, faltas e capital invalido."""
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{descricao}: colunas ausentes {sorted(missing)}")
    data = frame.copy()
    if "timestamp" not in data.columns:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(f"{descricao}: timestamp ausente")
        data = data.reset_index()
        data.rename(columns={data.columns[0]: "timestamp"}, inplace=True)
    optional = ["raw_best_asset"] if "raw_best_asset" in data.columns else []
    data = data[["timestamp", *COLUMNS, *optional]].reset_index(drop=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    data["strategy_equity"] = pd.to_numeric(data["strategy_equity"], errors="coerce")
    if data["timestamp"].isna().any() or data["timestamp"].duplicated().any():
        raise ValueError(f"{descricao}: datas invalidas ou duplicadas")
    if (not np.isfinite(data["strategy_equity"].to_numpy(dtype=float)).all()
            or (data["strategy_equity"] <= 0).any()):
        raise ValueError(f"{descricao}: patrimonio invalido")
    if data["selected_asset"].isna().any() or data["walk_forward_fold"].isna().any():
        raise ValueError(f"{descricao}: decisao/fold faltante")
    return data.sort_values("timestamp").reset_index(drop=True)


def comparar_trajetorias(base: pd.DataFrame, challenger: pd.DataFrame, candidato: str):
    """Usa EXATAMENTE os mesmos pregões e folds; nunca alinha por aproximação."""
    left = normalizar_decisoes(base, "Control")
    right = normalizar_decisoes(challenger, "U75+candidato")
    if not left["timestamp"].equals(right["timestamp"]):
        raise ValueError("Calendario diferente: a comparacao de capital foi bloqueada")
    if not left["walk_forward_fold"].equals(right["walk_forward_fold"]):
        raise ValueError("Folds diferentes: a comparacao de capital foi bloqueada")
    out = pd.DataFrame({
        "timestamp": left["timestamp"],
        "fold": left["walk_forward_fold"],
        "asset_control": left["selected_asset"],
        "asset_challenger": right["selected_asset"],
        "capital_control": left["strategy_equity"],
        "capital_challenger": right["strategy_equity"],
    })
    # O Top-1 (maior score bruto) nao equivale necessariamente a posicao
    # efetiva, pois a politica respeita margem de troca e tempo de holding.
    has_top1 = "raw_best_asset" in left.columns and "raw_best_asset" in right.columns
    if has_top1:
        out["top1_control"] = left["raw_best_asset"].to_numpy()
        out["top1_challenger"] = right["raw_best_asset"].to_numpy()
        out["candidato_top1"] = out["top1_challenger"].eq(candidato)
        out["mudou_top1"] = out["top1_control"] != out["top1_challenger"]
    out["mudou_posicao"] = out["asset_control"] != out["asset_challenger"]
    out["candidato_selecionado"] = out["asset_challenger"] == candidato
    out["mudou_sem_selecionar_candidato"] = (
        out["mudou_posicao"] & ~out["candidato_selecionado"]
    )
    out["delta_capital"] = out["capital_challenger"] - out["capital_control"]
    out["razao_capital"] = out["capital_challenger"] / out["capital_control"]
    out["log_razao_capital"] = np.log(out["razao_capital"])
    out["log_retorno_control"] = np.log(out["capital_control"]).diff()
    out["log_retorno_challenger"] = np.log(out["capital_challenger"]).diff()
    out["delta_log_retorno"] = (
        out["log_retorno_challenger"] - out["log_retorno_control"]
    )
    changed = out.loc[out["mudou_posicao"], "timestamp"]
    initial = float(out["capital_control"].iloc[-1])
    final = float(out["capital_challenger"].iloc[-1])
    summary = {
        "symbol": str(candidato),
        "sessions": int(len(out)),
        "changed_positions": int(out["mudou_posicao"].sum()),
        "direct_candidate_days": int(out["candidato_selecionado"].sum()),
        "indirect_changed_days": int(out["mudou_sem_selecionar_candidato"].sum()),
        "top1_diagnostic_available": bool(has_top1),
        "external_top1_days": int(out["candidato_top1"].sum()) if has_top1 else None,
        "changed_top1_days": int(out["mudou_top1"].sum()) if has_top1 else None,
        "external_top1_not_held_days": (
            int((out["candidato_top1"] & ~out["candidato_selecionado"]).sum())
            if has_top1 else None
        ),
        "first_changed_session": changed.iloc[0].isoformat() if len(changed) else None,
        "last_changed_session": changed.iloc[-1].isoformat() if len(changed) else None,
        "control_ending_capital": initial,
        "challenger_ending_capital": final,
        "delta_ending_capital": final - initial,
        "relative_ending_capital_change": final / initial - 1.0,
        "delta_compound_log_growth": math.log(final / initial),
        "selection_type": "exploratory_full_period_marginal_not_nested_discovery",
    }
    folds = []
    for fold_id, group in out.groupby("fold", sort=True):
        # Resultado do fold: diferenca de retornos internos, sem reinicializar
        # o capital; o primeiro pregão global não possui retorno anterior.
        d = group["delta_log_retorno"].dropna()
        folds.append({
            "symbol": str(candidato),
            "fold": int(fold_id),
            "sessions": int(len(group)),
            "changed_positions": int(group["mudou_posicao"].sum()),
            "candidate_days": int(group["candidato_selecionado"].sum()),
            "delta_log_growth": float(d.sum()),
        })
    return out, summary, pd.DataFrame(folds)
