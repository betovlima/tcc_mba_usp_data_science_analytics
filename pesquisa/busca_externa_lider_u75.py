"""Pesquisa: substituição contextual do líder U75 por candidato externo.

A função prepara um painel de (data, líder U75, candidato externo). Só os
atributos observáveis até o fechamento t entram na seleção. O retorno na
sessão t+1 é usado EXCLUSIVAMENTE para rotular/avaliar a hipótese.
Nenhum portfólio é simulado nem nenhuma operação é transmitida.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_VERSION = "u75-external-leader-search-v1.0.2"
FEATURES = (
    "return_5", "return_20", "vol_20", "ema_distance_20",
    "ema_distance_50", "rsi_14", "atr_pct_14", "distance_from_high_20",
    "channel_position_20", "trend_efficiency_20", "volume_zscore_20",
    "momentum_acceleration_5_20",
)
RELATIVE_COLUMNS = tuple(f"relative_{name}" for name in FEATURES)
MODEL_COLUMNS = RELATIVE_COLUMNS + ("distance_u75_leader",)


@dataclass(frozen=True)
class ResearchSettings:
    # Parâmetros metodológicos predefinidos, sem calibrar no OOS.
    minimum_reference_assets: int = 60
    min_training_dates: int = 250
    min_training_years: int = 2
    ridge_alpha: float = 1.0
    # Sensibilidade de custos como limite estimado, não taxas reais de execução.
    reference_cost_bps: float = 0.0


def _date(value: object) -> str:
    return pd.Timestamp(value).tz_localize("UTC").strftime("%Y-%m-%d") if pd.Timestamp(value).tzinfo is None else pd.Timestamp(value).tz_convert("UTC").strftime("%Y-%m-%d")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_leaders(leaders: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    required = {"decision_date", "execution_date", "leader"}
    _require(required <= set(leaders.columns), f"Líder: campos ausentes {required - set(leaders.columns)}")
    rows = leaders[list(required)].copy()
    rows["decision_date"] = rows["decision_date"].map(_date)
    rows["execution_date"] = rows["execution_date"].map(_date)
    rows["leader"] = rows["leader"].astype(str).str.upper().str.strip()
    rows = rows.sort_values("decision_date").reset_index(drop=True)
    _require(not rows["decision_date"].duplicated().any(), "Duas decisões no mesmo dia")
    _require(rows["leader"].isin(universe).all(), "Líder fora do U75 congelado")
    _require((rows["decision_date"] < rows["execution_date"]).all(), "Execução deve suceder decisão")
    return rows


def make_panel(
    leaders: pd.DataFrame,
    feature_rows: pd.DataFrame,
    universe: set[str],
    *,
    settings: ResearchSettings = ResearchSettings(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gera distância no fechamento e retornos t+1 isolados de toda feature.

    feature_rows: colunas decision_date, execution_date, symbol, FEATURES,
    execution_open, execution_close. O painel externo inclui todo candidato
    válido em t, inclusive quando falta preço futuro (rotulado sem outcome).
    """
    required = {"decision_date", "execution_date", "symbol", "execution_open", "execution_close", *FEATURES}
    _require(required <= set(feature_rows.columns), f"Painel: faltam {required - set(feature_rows.columns)}")
    _require(len(universe) == 75, "Universo deve conter exatamente 75 ativos distintos")
    decisions = validate_leaders(leaders, universe)
    all_rows = feature_rows[list(required)].copy()
    all_rows["decision_date"] = all_rows["decision_date"].map(_date)
    all_rows["execution_date"] = all_rows["execution_date"].map(_date)
    all_rows["symbol"] = all_rows["symbol"].astype(str).str.upper().str.strip()
    _require(not all_rows.duplicated(["decision_date", "symbol"]).any(), "Feature duplicada por dia/ativo")
    _require((all_rows["decision_date"] < all_rows["execution_date"]).all(), "Preço futuro não deve ter data de decisão")
    all_rows = all_rows.merge(decisions, on=["decision_date", "execution_date"], how="inner", validate="many_to_one")
    all_rows[list(FEATURES)] = all_rows[list(FEATURES)].apply(pd.to_numeric, errors="coerce")
    all_rows["execution_open"] = pd.to_numeric(all_rows["execution_open"], errors="coerce")
    all_rows["execution_close"] = pd.to_numeric(all_rows["execution_close"], errors="coerce")
    out: list[dict] = []
    coverage: list[dict] = []
    for day, group in all_rows.groupby("decision_date", sort=True):
        u = group[group["symbol"].isin(universe)].set_index("symbol")
        c = group[~group["symbol"].isin(universe)].set_index("symbol")
        valid_u = u[list(FEATURES)].replace([np.inf,-np.inf],np.nan).notna().all(axis=1)
        valid_c = c[list(FEATURES)].replace([np.inf,-np.inf],np.nan).notna().all(axis=1)
        leader_name = group["leader"].iloc[0]
        audit = {"decision_date": day, "reference_valid": int(valid_u.sum()),
                 "external_total": len(c), "external_valid": int(valid_c.sum()),
                 "leader": leader_name, "status": "ok"}
        if int(valid_u.sum()) < settings.minimum_reference_assets or leader_name not in u.index or not valid_u.get(leader_name, False):
            audit["status"] = "insufficient_u75_reference"; coverage.append(audit);continue
        if not valid_c.any():
            audit["status"] = "no_external_candidates"; coverage.append(audit);continue
        ref = u.loc[valid_u, list(FEATURES)]
        med = ref.median(axis=0)
        # Escala apenas com a distribuição do U75 em t; não é ajustada em t+1.
        scale = (ref.quantile(.75)-ref.quantile(.25)).where(lambda x: x > 1e-12, ref.std(ddof=0)).fillna(1.0)
        scale = scale.where(scale > 1e-12,1.0)
        leader = u.loc[leader_name]
        l_future = (float(leader["execution_open"]), float(leader["execution_close"]))
        l_return = np.log(l_future[1] / l_future[0]) if min(l_future) > 0 and np.isfinite(l_future).all() else np.nan
        if not np.isfinite(l_return):
            audit["status"] = "missing_leader_execution";coverage.append(audit);continue
        for symbol, candidate in c.loc[valid_c].iterrows():
            diff = (candidate[list(FEATURES)].astype(float)-leader[list(FEATURES)].astype(float))/scale
            v = (float(candidate["execution_open"]), float(candidate["execution_close"]))
            ret = np.log(v[1]/v[0]) if min(v) > 0 and np.isfinite(v).all() else np.nan
            item = {"decision_date":day, "execution_date":group["execution_date"].iloc[0],
                    "leader":leader_name,"candidate":symbol,
                    "distance_u75_leader":float(np.linalg.norm(diff.to_numpy(dtype=float))),
                    "leader_log_open_close":float(l_return),
                    "candidate_log_open_close":float(ret) if np.isfinite(ret) else np.nan,
                    "delta_log_open_close":float(ret-l_return) if np.isfinite(ret) else np.nan,
                    "target_available":bool(np.isfinite(ret))}
            item.update({f"relative_{f}":float(diff[f]) for f in FEATURES})
            out.append(item)
        coverage.append(audit)
    panel = pd.DataFrame(out)
    audit = pd.DataFrame(coverage)
    _require(not panel.empty,"Nenhuma sessão comparável: confira os snapshots e calendário")
    return panel.sort_values(["decision_date", "distance_u75_leader", "candidate"]).reset_index(drop=True),audit


def evaluate_expanding_years(panel: pd.DataFrame, *, settings: ResearchSettings=ResearchSettings()) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Ridge causal, treino por datas anteriores com rótulos já maturados.

    Há uma decisão por data. Candidatos sem cotação de execução NÃO são
    filtrados antes do ranking; nesse caso a avaliação registra resultado
    indisponível e a taxa de cobertura deve ser reportada.
    """
    required={"decision_date","execution_date","candidate","distance_u75_leader","delta_log_open_close",*MODEL_COLUMNS}
    _require(required <= set(panel.columns),"Painel incompleto")
    p=panel.copy()
    p["decision_date"]=p["decision_date"].map(_date)
    p["execution_date"]=p["execution_date"].map(_date)
    p["test_year"]=p["decision_date"].str[:4].astype(int)
    picks=[];folds=[]
    for year in sorted(p["test_year"].unique()):
        test=p[p["test_year"]==year].copy()
        first=test["decision_date"].min()
        train=p[(p["execution_date"]<first)&p["delta_log_open_close"].notna()].copy()
        train_dates=train["decision_date"].nunique()
        train_years=train["test_year"].nunique()
        if train_dates < settings.min_training_dates or train_years < settings.min_training_years:
            continue
        _require(max(train["execution_date"]) < min(test["decision_date"]), "Vazamento de rótulo entre treino e teste")
        model=make_pipeline(StandardScaler(),Ridge(alpha=settings.ridge_alpha))
        model.fit(train[list(MODEL_COLUMNS)],train["delta_log_open_close"])
        test["estimated_delta_log"] = model.predict(test[list(MODEL_COLUMNS)])
        threshold=np.log1p(settings.reference_cost_bps/10000.0)
        for date,group in test.groupby("decision_date",sort=True):
            closest=group.sort_values(["distance_u75_leader","candidate"]).iloc[0]
            predicted=group.sort_values(["estimated_delta_log","candidate"],ascending=[False,True]).iloc[0]
            choose=predicted if float(predicted["estimated_delta_log"])>threshold else None
            chosen=choose if choose is not None else None
            picks.append({"decision_date":date,"execution_date":group["execution_date"].iloc[0],
                          "year":year,"leader":group["leader"].iloc[0],"external_available":len(group),
                          "nearest":closest["candidate"],"nearest_distance":float(closest["distance_u75_leader"]),
                          "nearest_delta":float(closest["delta_log_open_close"]) if pd.notna(closest["delta_log_open_close"]) else np.nan,
                          "nearest_target_available":bool(closest["target_available"]),
                          "model_selected":chosen["candidate"] if chosen is not None else "U75_LEADER",
                          "model_score":float(predicted["estimated_delta_log"]),
                          "model_delta":(float(chosen["delta_log_open_close"]) if chosen is not None and pd.notna(chosen["delta_log_open_close"]) else np.nan if chosen is not None else 0.0),
                          "model_target_available":bool(chosen["target_available"]) if chosen is not None else True,
                          "model_external":chosen is not None,
                          "oracle_delta":float(group["delta_log_open_close"].max(skipna=True)) if group["target_available"].any() else np.nan})
        folds.append({"year":year,"train_last_execution":max(train["execution_date"]),
                      "first_test_decision":first,"train_dates":int(train_dates),
                      "train_rows":len(train),"test_dates":test["decision_date"].nunique(),
                      "test_candidates":test["candidate"].nunique()})
    return pd.DataFrame(picks),pd.DataFrame(folds)


def bootstrap_monthly_ci(picks: pd.DataFrame, column: str, availability: str,
                         *, repetitions: int = 1000, seed: int = 42) -> tuple[float | None, float | None]:
    """IC exploratório de média diária com reamostragem de meses inteiros."""
    observed = picks.loc[picks[availability] & picks[column].notna(),
                         ["decision_date", column]].copy()
    if observed.empty:
        return None, None
    observed["month"] = observed["decision_date"].str.slice(0, 7)
    groups = [group[column].to_numpy(float) for _, group in observed.groupby("month", sort=True)]
    if len(groups) < 2:
        return None, None
    random = np.random.default_rng(seed)
    means = np.empty(repetitions)
    for i in range(repetitions):
        sampled = random.integers(0, len(groups), size=len(groups))
        values = np.concatenate([groups[j] for j in sampled])
        means[i] = values.mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def summarize(picks:pd.DataFrame)->dict:
    _require(not picks.empty,"Sem folds OOS válidos")
    result={"oos_sessions":int(len(picks)),"years":sorted(picks.year.unique().tolist()),
            "external_candidate_dates":int(picks["model_external"].sum()),
            "model_unmeasurable":int((~picks["model_target_available"]).sum()),
            "nearest_unmeasurable":int((~picks["nearest_target_available"]).sum())}
    for label,col,flag in [("nearest","nearest_delta","nearest_target_available"),("model","model_delta","model_target_available")]:
        v=picks.loc[picks[flag],col].dropna()
        ci=bootstrap_monthly_ci(picks,col,flag)
        result[label]={"measured_sessions":len(v),"mean_daily_excess_log":float(v.mean()) if len(v) else None,
                       "median_daily_excess_log":float(v.median()) if len(v) else None,
                       "fraction_beating_leader":float((v>0).mean()) if len(v) else None,
                       "bootstrap_monthly_ci95_mean_log":[ci[0],ci[1]]}
    return result
