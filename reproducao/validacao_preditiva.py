"""Auditoria independente de previsoes OOS, sem otimizar o modelo nem o capital.

Contrato de entrada: data_decisao, ativo, fold, previsao, observado,
                    label_maturado, split_validado.
A utilidade observada deve ser exatamente o alvo usado pelo LightGBM.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

COLUNAS = ("data_decisao", "ativo", "fold", "previsao", "observado",
           "label_maturado", "split_validado")


def exportar_previsoes_oos(*, states, baseline_frames, baseline_symbols,
                           common_dates, horizon_maximo, data_corte,
                           candidate_frames=None, candidate_caches=None, splits_auditados=False):
    """Exporta caches OOS existentes; nao ajusta modelos e nao faz downloads.

    states: sequencia de BaselineFoldState da campanha historica.
    O ultimo elemento de decision_dates e sentinela (nao e decisao).
    Frames devem ter o alvo forward_risk_adjusted_utility e precos corrigidos
    por eventos corporativos ANTES de construir esse alvo.
    """
    candidate_frames = candidate_frames or {}
    candidate_caches = candidate_caches or {}
    frames = {**baseline_frames, **candidate_frames}
    calendar = pd.DatetimeIndex(common_dates)
    if calendar.tz is None:
        raise ValueError("Calendario deve ter timezone explicito")
    if horizon_maximo <= 0:
        raise ValueError("horizon_maximo deve ser positivo")
    cutoff = pd.Timestamp(data_corte)
    # Cutoff YYYY-MM-DD inclui toda a sessao daquele dia em UTC.
    if isinstance(data_corte, str) and len(data_corte) == 10:
        cutoff += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    calendar = calendar.tz_convert("UTC")
    cutoff_index = int(calendar.searchsorted(cutoff, side="right")) - 1
    baseline_positions = {s: j + 1 for j, s in enumerate(baseline_symbols)}
    out = []
    for state in states:
        fold = int(state.fold_id)
        dates = pd.DatetimeIndex(state.decision_dates)[:-1]
        for date in dates:
            t = pd.Timestamp(date).tz_convert("UTC")
            ix = int(calendar.searchsorted(t, side="left"))
            if ix >= len(calendar) or calendar[ix] != t:
                raise ValueError(f"Data fora do calendario: {t}")
            maturity = ix + int(horizon_maximo) <= cutoff_index
            base_vector = state.decision_cache.get(pd.Timestamp(date))
            if base_vector is None:
                raise ValueError(f"Cache base ausente no fold {fold}: {t}")
            for symbol, frame in frames.items():
                if symbol in baseline_positions:
                    position = baseline_positions[symbol]
                    score = float(base_vector[position])
                else:
                    one = candidate_caches[symbol][fold].decision_cache.get(pd.Timestamp(date))
                    if one is None:
                        raise ValueError(f"Cache candidato ausente: {symbol} {t}")
                    score = float(one[1])
                observed = frame["forward_risk_adjusted_utility"].reindex([t]).iloc[0]
                out.append({
                    "data_decisao": t.isoformat(), "ativo": symbol, "fold": fold,
                    "previsao": score if math.isfinite(score) else np.nan,
                    "observado": float(observed) if pd.notna(observed) else np.nan,
                    "label_maturado": bool(maturity),
                    # Nao assumir eventos revisados so porque o CSV e RAW.
                    "split_validado": bool(splits_auditados),
                })
    return pd.DataFrame.from_records(out, columns=list(COLUNAS))


def _bool_strict(s: pd.Series, coluna: str) -> pd.Series:
    mapped = s.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise ValueError(f"Coluna {coluna} contem valor booleano ambiguo")
    return mapped.astype(bool)


def _metrics(g: pd.DataFrame) -> dict:
    err = g["previsao"].to_numpy(float) - g["observado"].to_numpy(float)
    base = g["observado"].to_numpy(float)
    return {"n": int(len(g)), "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae_base_zero": float(np.mean(np.abs(base))),
            "rmse_base_zero": float(np.sqrt(np.mean(base ** 2)))}


def _ndcg_top_k(y, p, k=5):
    # Converte a utilidade em relevancia nao negativa APENAS para o NDCG.
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    relevance = np.maximum(0.0, y - y.min())
    if not relevance.any():
        return None
    order = np.argsort(-p, kind="stable")[:k]
    ideal = np.argsort(-relevance, kind="stable")[:k]
    discounts = np.log2(np.arange(2, len(order) + 2))
    dcg = float(np.sum(relevance[order] / discounts))
    idcg = float(np.sum(relevance[ideal] / discounts))
    return dcg / idcg if idcg else None


def avaliar_previsoes_oos(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = sorted(set(COLUNAS) - set(df.columns))
    if missing:
        raise ValueError("Colunas obrigatorias ausentes: " + ", ".join(missing))
    x = df.copy()
    x["data_decisao"] = pd.to_datetime(x["data_decisao"], utc=True, errors="raise")
    if x["data_decisao"].isna().any():
        raise ValueError("Datas nulas")
    x["previsao"] = pd.to_numeric(x["previsao"], errors="coerce")
    x["observado"] = pd.to_numeric(x["observado"], errors="coerce")
    x["fold"] = pd.to_numeric(x["fold"], errors="raise")
    for c in ("label_maturado", "split_validado"):
        x[c] = _bool_strict(x[c], c)
    if x.duplicated(["data_decisao", "ativo", "fold"]).any():
        raise ValueError("Previsoes duplicadas por sessao, ativo e fold")
    available = np.isfinite(x["previsao"]) & np.isfinite(x["observado"])
    valid = available & x["label_maturado"] & x["split_validado"]
    x["avaliavel"] = valid
    reasons = np.select(
        [~x["label_maturado"], ~x["split_validado"], ~available],
        ["alvo_nao_maturado", "eventos_nao_validados", "previsao_ou_alvo_ausente"],
        default="ok",
    )
    x["motivo"] = reasons
    good = x[x["avaliavel"]]
    summary = {
        "schema": "tcc-predictive-audit-v1", "rows": int(len(x)),
        "valid_rows": int(len(good)), "excluded_reasons":
            {str(k): int(v) for k, v in x.loc[~valid, "motivo"].value_counts().items()},
        "status": "completed" if not good.empty else "insufficient_evidence",
        "metric_target": "forward_risk_adjusted_utility",
        "note": "OOS pontual; selecao retrospectiva de ativos nao e validacao nested.",
    }
    folds = []
    ranks = []
    assets = []
    if not good.empty:
        summary["overall"] = _metrics(good)
        for fid, g in good.groupby("fold"):
            folds.append({"fold": int(fid), **_metrics(g)})
        for (fid, asset), g in good.groupby(["fold", "ativo"]):
            assets.append({"fold": int(fid), "ativo": str(asset), **_metrics(g)})
        for (fid, t), g in good.groupby(["fold", "data_decisao"]):
            if len(g) < 3 or g["previsao"].nunique() < 2 or g["observado"].nunique() < 2:
                continue
            ranks.append({
                "fold": int(fid), "data_decisao": t.isoformat(), "ativos": int(len(g)),
                "spearman": float(g["previsao"].corr(g["observado"], method="spearman")),
                "ndcg_5": _ndcg_top_k(g["observado"].to_numpy(float),
                                      g["previsao"].to_numpy(float)),
            })
        if ranks:
            summary["ranking_por_sessao"] = {
                "n_sessoes": len(ranks),
                "spearman_medio": float(pd.DataFrame(ranks)["spearman"].mean()),
                "ndcg_5_medio": float(pd.DataFrame(ranks)["ndcg_5"].mean()),
            }
    return summary, pd.DataFrame(folds), pd.DataFrame(ranks), pd.DataFrame(assets)


def gerar_relatorios(df: pd.DataFrame, out: Path):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report, folds, ranks, assets = avaliar_previsoes_oos(df)
    folds.to_csv(out / "metricas_por_fold.csv", index=False)
    ranks.to_csv(out / "ranking_por_sessao.csv", index=False)
    assets.to_csv(out / "metricas_por_ativo_fold.csv", index=False)
    (out / "diagnostico_preditivo.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report
