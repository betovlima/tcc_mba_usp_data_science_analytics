"""Diagnostico causal descritivo de estresse e posicoes do universo U75.

Nao substitui o Control congelado nem treina um classificador em OOS.
Sinais usam SOMENTE historico de preco ate a data do fechamento t.
Os limites sao PREDECLARADOS para analise exploratoria, nao calibrados
contra o patrimonio final do teste. Um sinal tecnico positivo nao
prova retorno futuro nem confiabilidade da previsao LightGBM.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StressSettings:
    min_valid_assets: int = 68
    max_positive_share_5: float = 0.35
    max_positive_share_20: float = 0.45
    min_below_ma50_share: float = 0.55
    max_median_return_20: float = -0.015
    # Horizonte exclusivo para comparar resultados, NUNCA permanencia minima em CASH.
    audit_sessions: int = 10

    def __post_init__(self) -> None:
        if self.min_valid_assets < 1 or self.audit_sessions < 1:
            raise ValueError("Parametros de estresse ou auditoria invalidos.")


def _valid_date_index(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    result = pd.DatetimeIndex(pd.to_datetime(dates, utc=True))
    if result.empty or result.has_duplicates or not result.is_monotonic_increasing:
        raise ValueError("Datas de decisao devem ser ordenadas e unicas.")
    return result


def calcular_indicadores_u75(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    decision_dates: pd.DatetimeIndex,
    *,
    settings: StressSettings = StressSettings(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula U75 inteiro e observacoes por ativo sem usar barras futuras.

    A unidade e cada ativo do universo, pesos iguais para diagnostico.
    A disponibilidade exige indicadores passados suficientes. Nao
    elimina simbolos do universo; cobertura ausente e informada.
    """
    dates = _valid_date_index(decision_dates)
    if len(symbols) != 75 or set(symbols) != set(frames) or len(set(symbols)) != 75:
        raise RuntimeError("Diagnostico requer os mesmos 75 ativos da referencia.")
    parts: list[pd.DataFrame] = []
    for symbol in symbols:
        raw = frames[symbol]
        close = pd.to_numeric(raw["close"], errors="coerce")
        if not isinstance(close.index, pd.DatetimeIndex):
            raise RuntimeError(f"{symbol}: indice de precos invalido.")
        close = close.sort_index()
        if close.index.has_duplicates:
            raise RuntimeError(f"{symbol}: barras repetidas.")
        ret5 = close.pct_change(periods=5, fill_method=None)
        ret20 = close.pct_change(periods=20, fill_method=None)
        ma20 = close.rolling(20, min_periods=20).mean()
        ma50 = close.rolling(50, min_periods=50).mean()
        # Reindexar DEPOIS de calcular indicadores com historico completo.
        metrics = pd.DataFrame({
            "close": close, "return_5": ret5, "return_20": ret20,
            "average_20": ma20, "average_50": ma50,
        }).reindex(dates)
        metrics["symbol"] = symbol
        metrics["valid"] = (
            np.isfinite(metrics["close"])
            & np.isfinite(metrics["return_5"])
            & np.isfinite(metrics["return_20"])
            & np.isfinite(metrics["average_20"])
            & np.isfinite(metrics["average_50"])
            & (metrics["close"] > 0)
        )
        metrics["positive_5"] = metrics["return_5"] > 0
        metrics["positive_20"] = metrics["return_20"] > 0
        metrics["below_ma50"] = metrics["close"] < metrics["average_50"]
        # Proxy retrospectiva de momentum, NAO previsao de retorno.
        metrics["technical_candidate"] = (
            metrics["valid"] & metrics["positive_5"]
            & metrics["positive_20"]
            & (metrics["close"] > metrics["average_20"])
            & (metrics["close"] > metrics["average_50"])
        )
        metrics.index.name = "decision_date"
        parts.append(metrics.reset_index())
    detail = pd.concat(parts, ignore_index=True)
    valid = detail.loc[detail["valid"]]
    by = valid.groupby("decision_date", sort=True)
    agg = by.agg(
        valid_assets=("symbol", "nunique"),
        positive_5_count=("positive_5", "sum"),
        positive_20_count=("positive_20", "sum"),
        below_ma50_count=("below_ma50", "sum"),
        technical_candidates_count=("technical_candidate", "sum"),
        median_return_5=("return_5", "median"),
        median_return_20=("return_20", "median"),
        mean_return_20=("return_20", "mean"),
    ).reindex(dates)
    agg.index.name = "decision_date"
    agg["valid_assets"] = agg["valid_assets"].fillna(0).astype(int)
    for name in ("positive_5_count", "positive_20_count", "below_ma50_count",
                 "technical_candidates_count"):
        agg[name] = agg[name].fillna(0).astype(int)
    agg["coverage"] = agg["valid_assets"] / len(symbols)
    denominator = agg["valid_assets"].replace(0, np.nan)
    for num, name in [
        ("positive_5_count", "positive_share_5"),
        ("positive_20_count", "positive_share_20"),
        ("below_ma50_count", "below_ma50_share"),
    ]:
        agg[name] = agg[num] / denominator
    agg["coverage_ok"] = agg["valid_assets"] >= settings.min_valid_assets
    agg["universe_stress"] = (
        agg["coverage_ok"]
        & (agg["positive_share_5"] <= settings.max_positive_share_5)
        & (agg["positive_share_20"] <= settings.max_positive_share_20)
        & (agg["below_ma50_share"] >= settings.min_below_ma50_share)
        & (agg["median_return_20"] <= settings.max_median_return_20)
    )
    agg["no_technical_candidate"] = (
        agg["coverage_ok"] & (agg["technical_candidates_count"] == 0)
    )
    agg["stress_without_candidate"] = (
        agg["universe_stress"] & agg["no_technical_candidate"]
    )
    # Resultado ignorado se cobertura incompleta, nao imputar melhora.
    return agg.reset_index(), detail


def juntar_decisoes(
    indicators: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Relaciona sinal no fechamento t a posicao EFETIVA no pregão t+1."""
    required = {"decision_date", "timestamp", "selected_asset", "strategy_equity"}
    if not required.issubset(predictions.columns):
        raise RuntimeError("Previsoes oficiais nao contem estado/tempo/capital.")
    decisions = predictions[list(required)].copy()
    decisions["decision_date"] = pd.to_datetime(
        decisions["decision_date"], utc=True,
    )
    decisions["timestamp"] = pd.to_datetime(
        decisions["timestamp"], utc=True,
    )
    if decisions["timestamp"].duplicated().any() or decisions["decision_date"].duplicated().any():
        raise RuntimeError("Calendario oficial contem decisoes duplicadas.")
    stats = indicators.copy()
    stats["decision_date"] = pd.to_datetime(stats["decision_date"], utc=True)
    joint = decisions.merge(
        stats, how="left", on="decision_date",
        validate="one_to_one", indicator=True,
    )
    if len(joint) != len(predictions) or not joint["_merge"].eq("both").all():
        raise RuntimeError("Nem todas as sessoes possuem indicadores anteriores.")
    joint = joint.drop(columns="_merge").sort_values("timestamp").reset_index(drop=True)
    joint["effective_cash"] = joint["selected_asset"].eq("CASH")
    joint["invested_during_stress"] = (
        joint["universe_stress"] & ~joint["effective_cash"]
    )
    joint["invested_without_candidate"] = (
        joint["stress_without_candidate"] & ~joint["effective_cash"]
    )
    return joint


def extrair_episodios(
    joined: pd.DataFrame,
    *,
    signal: str = "invested_without_candidate",
    settings: StressSettings = StressSettings(),
) -> pd.DataFrame:
    """Agrupa episodios contiguos; seleciona inicios disjuntos por 10 sessoes."""
    if signal not in joined.columns:
        raise ValueError(f"Indicador inexistente: {signal}")
    s = joined[signal].astype(bool).to_numpy()
    dates = pd.to_datetime(joined["decision_date"], utc=True)
    starts: list[dict] = []
    previous_selected = -settings.audit_sessions
    i = 0
    while i < len(s):
        if not s[i]:
            i += 1
            continue
        start = i
        while i + 1 < len(s) and s[i + 1]:
            i += 1
        end = i
        room = start + settings.audit_sessions < len(s)
        disjoint = (start - previous_selected) >= settings.audit_sessions
        usable = bool(room and disjoint)
        if usable:
            previous_selected = start
        starts.append({
            "decision_date": dates.iloc[start],
            "episode_end": dates.iloc[end],
            "episode_sessions": end - start + 1,
            "start_index": start,
            "end_index": end,
            "audit_eligible": usable,
            "exclusion_reason": (
                "" if usable else ("end_of_sample" if not room else "overlapping_horizon")
            ),
        })
        i += 1
    return pd.DataFrame(starts, columns=[
        "decision_date", "episode_end", "episode_sessions",
        "start_index", "end_index", "audit_eligible", "exclusion_reason",
    ])


def posicoes_mensais(
    predictions: pd.DataFrame,
    trades: pd.DataFrame,
    indicators: pd.DataFrame,
    *,
    initial_capital: float = 10000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mapa e detalhes de ocupacao EFETIVA por sessao, nao ticker previsto."""
    p = predictions.copy()
    p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True)
    if p["timestamp"].isna().any() or p["timestamp"].duplicated().any():
        raise RuntimeError("Datas de carteira invalidas.")
    p = p.sort_values("timestamp")
    p["month"] = p["timestamp"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m")
    grouped = (
        p.groupby(["month", "selected_asset"], sort=True)
        .size().rename("sessions").reset_index()
    )
    if grouped["sessions"].sum() != len(p):
        raise RuntimeError("Mapa mensal nao conserva todas as sessoes.")
    t = trades.copy()
    if not t.empty:
        t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True)
        t["month"] = t["timestamp"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m")
    events = {}
    for m in sorted(p["month"].unique()):
        tp = t[t["month"] == m] if not t.empty else t
        events[m] = {
            "buys": int(tp["action"].eq("BUY").sum()) if not tp.empty else 0,
            "sells": int(tp["action"].isin(["SELL", "FINAL_SELL"]).sum()) if not tp.empty else 0,
            "rotations": int(tp["action"].eq("BUY").sum()) if not tp.empty else 0,
        }
    period = p.groupby("month", sort=True)
    details = []
    for month, frame in period:
        asset_days = grouped[grouped["month"] == month].sort_values(
            ["sessions", "selected_asset"], ascending=[False, True],
        )
        first_date = frame.iloc[0]["timestamp"]
        last_date = frame.iloc[-1]["timestamp"]
        first_idx = int(p.index.get_loc(frame.index[0])) if p.index.is_unique else -1
        before = float(initial_capital if first_idx <= 0 else p.iloc[first_idx - 1]["strategy_equity"])
        ending = float(frame.iloc[-1]["strategy_equity"])
        signal_dates = pd.to_datetime(frame["decision_date"], utc=True)
        market = indicators.set_index("decision_date").reindex(signal_dates)
        details.append({
            "month": month, "year": int(month[:4]), "month_number": int(month[5:]),
            "sessions": len(frame), "dominant_asset": asset_days.iloc[0]["selected_asset"],
            "dominant_sessions": int(asset_days.iloc[0]["sessions"]),
            "dominant_share": float(asset_days.iloc[0]["sessions"] / len(frame)),
            "distinct_assets": asset_days["selected_asset"].nunique(),
            "cash_sessions": int(frame["selected_asset"].eq("CASH").sum()),
            "buys": events[month]["buys"], "sells": events[month]["sells"],
            "monthly_return": float(ending / before - 1),
            "stress_sessions": int(market["universe_stress"].fillna(False).sum()),
            "stress_no_candidate_sessions": int(
                market["stress_without_candidate"].fillna(False).sum()
            ),
            "first_execution": first_date.isoformat(),
            "last_execution": last_date.isoformat(),
        })
    return pd.DataFrame(details), grouped


def serializar_mapa_html(months: pd.DataFrame, assets: pd.DataFrame, *, title: str) -> str:
    """Gera pagina local independente com detalhes ao clicar no mes."""
    records = []
    for item in months.to_dict("records"):
        month = item["month"]
        holdings = assets[assets["month"] == month][["selected_asset", "sessions"]]
        records.append({
            **item,
            "positions": holdings.sort_values(
                ["sessions", "selected_asset"], ascending=[False, True]
            ).to_dict("records"),
        })
    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    return """<!doctype html><html lang="pt-BR"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mapa mensal de posicoes</title>
<style>
body{font:15px system-ui,sans-serif;background:#f7f8fa;color:#17212f;margin:24px}
h1{font-size:22px}p{max-width:940px}
#map{display:grid;grid-template-columns:55px repeat(12,minmax(65px,1fr));gap:5px;max-width:1180px}
.cell{border:1px solid #d7dce5;border-radius:6px;padding:9px 4px;min-height:48px;
 text-align:center;background:rgba(31,105,155,var(--density,.08));font:inherit;color:#122333}
button.cell{cursor:pointer}button.cell:focus{outline:3px solid #115599}
.cell small{display:block;font-size:11px;margin-top:4px}
.heading{background:#e7ecf3;font-weight:600}
#detail{border:1px solid #d7dce5;background:white;border-radius:8px;
 padding:15px;margin-top:18px;max-width:1100px}
table{border-collapse:collapse}td,th{padding:7px 14px 7px 0;text-align:left;
 border-bottom:1px solid #e8eaf0} .hint{color:#465466}
</style>
<h1 id="title"></h1><p class="hint">Cada mes mostra o ativo com mais pregoes
depos do fechamento e a quantidade de sessoes. A intensidade representa a
concentracao no ativo dominante, nao lucro ou risco. Clique em um mes
para consultar TODOS os ativos, dias, operacoes e condicao do universo.</p>
<div id="map" role="group" aria-label="Mapa mensal de ativos"></div>
<div id="detail" aria-live="polite">Selecione um mes para os detalhes.</div>
<script>
"use strict";
const rows=__PAYLOAD__;
const title=__TITLE__;
document.getElementById("title").textContent=title;
const root=document.getElementById("map"),info=document.getElementById("detail");
const months=["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"];
function node(tag, text, parent, cls){
 const x=document.createElement(tag);if(text!==null)x.textContent=text;
 if(cls)x.className=cls;parent.appendChild(x);return x;
}
node("div","Ano",root,"cell heading");
months.forEach(m=>node("div",m,root,"cell heading"));
const lookup=new Map(rows.map(r=>[r.month,r]));
for(const year of [...new Set(rows.map(r=>r.year))].sort()){
 node("div",String(year),root,"cell heading");
 for(let i=1;i<=12;i++){
  const key=year+"-"+String(i).padStart(2,"0"),r=lookup.get(key);
  if(!r){node("div","",root,"cell");continue;}
  const b=node("button",r.dominant_asset,root,"cell");
  b.type="button";
  b.style.setProperty("--density",String((0.12+r.dominant_share*0.52).toFixed(3)));
  node("small",r.dominant_sessions+"/"+r.sessions+" pregoes",b);
  b.addEventListener("click",()=>show(r));
 }
}
function show(r){
 info.replaceChildren();
 node("h2",r.month+": "+r.dominant_asset+" dominante",info);
 node("p","Sessoes "+r.sessions+" | ativos "+r.distinct_assets+
 " | CASH "+r.cash_sessions+" | compras "+r.buys+" | vendas "+r.sells+
 " | retorno mensal "+(100*r.monthly_return).toFixed(2)+"%"+
 " | pregoes sob estresse "+r.stress_sessions+
 " | estresse sem candidato tecnico "+r.stress_no_candidate_sessions,info);
 const table=node("table",null,info);
 const hdr=node("tr",null,table);
 node("th","Ativo",hdr);node("th","Pregoes",hdr);
 for(const p of r.positions){
  const tr=node("tr",null,table);
  node("td",p.selected_asset,tr);node("td",String(p.sessions),tr);
 }
}
if(rows.length)show(rows[0]);
</script></html>""".replace("__PAYLOAD__", payload).replace(
        "__TITLE__", json.dumps(title, ensure_ascii=False)
    )


def construir_politica_protecao_diaria(
    base_policy,
    daily: pd.DataFrame,
    *,
    active_dates: pd.DatetimeIndex | None = None,
):
    """Reavaliar a condicao do U75 a CADA pregao, sem prazo fixo em CASH.

    O fechamento t determina a posicao efetiva em t+1.
    Quando o sinal cessa, a politica original e chamada imediatamente
    com a posicao/tempo de permanencia REAIS, inclusive CASH.
    O intervalo active_dates serve apenas a simulacoes locais de
    atribuicao, nao define nem prolonga o estado CASH.
    """
    required = {"decision_date", "stress_without_candidate"}
    if not required.issubset(daily.columns):
        raise ValueError("Tabela diaria incompleta para protecao de mercado.")
    rows = daily[list(required)].copy()
    rows["decision_date"] = pd.to_datetime(rows["decision_date"], utc=True)
    if rows["decision_date"].isna().any() or rows["decision_date"].duplicated().any():
        raise ValueError("Datas de protecao invalidas ou duplicadas.")
    if rows["stress_without_candidate"].isna().any():
        raise ValueError("Nao tratar ausencia de sinal como decisao segura.")
    signals = {
        pd.Timestamp(dt): bool(s)
        for dt, s in rows.itertuples(index=False, name=None)
    }
    allowed = (
        None if active_dates is None
        else set(_valid_date_index(active_dates))
    )

    def policy(date, position, holding):
        dt = pd.Timestamp(date)
        if dt not in signals:
            raise KeyError(f"Sem avaliacao do U75 para {dt}.")
        if (allowed is None or dt in allowed) and signals[dt]:
            return 0, 0.0
        return base_policy(dt, position, holding)

    return policy
