"""Compara Alpaca atual SIP adjustment=all com Tiingo total-causal.

Versao: alpaca-tiingo-audit-v1.0.0

Gera diagnostico por ativo/data e, quando os dois backtests ja existem,
compara tambem a politica de rotacao e localiza a primeira decisao divergente.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tcc_engine.config import ASSETS as ATIVOS

VERSAO = "alpaca-tiingo-audit-v1.0.0"
RAIZ = Path(__file__).resolve().parent
DIR_ALPACA = RAIZ / "dados" / "referencia_alpaca_atual"
DIR_TIINGO = RAIZ / "dados" / "series_historicas"
DIR_EVENTOS = RAIZ / "dados" / "eventos_corporativos"
DIR_SPLITS = RAIZ / "dados" / "desdobramentos"
DIR_OUT = RAIZ / "output" / "auditoria_alpaca_vs_tiingo"
CURVA_ALPACA = RAIZ / "output" / "alpaca_atual_22m_control" / "equity_curve.csv"
CURVA_TIINGO = RAIZ / "output" / "tiingo_total_causal_v1" / "equity_curve.csv"
COLUNAS = ["open", "high", "low", "close", "volume"]
LIMIAR = 0.001


def registrar(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def ler_serie(diretorio: Path, ativo: str) -> pd.DataFrame:
    arquivo = diretorio / f"{ativo}.csv"
    if not arquivo.exists():
        raise RuntimeError(f"Arquivo ausente: {arquivo}")
    tabela = pd.read_csv(arquivo)
    obrigatorias = ["timestamp", *COLUNAS]
    ausentes = [c for c in obrigatorias if c not in tabela.columns]
    if ausentes:
        raise RuntimeError(f"{ativo}: colunas ausentes em {arquivo}: {', '.join(ausentes)}")
    tabela = tabela[obrigatorias].copy()
    tabela["timestamp"] = pd.to_datetime(tabela["timestamp"], utc=True, errors="coerce")
    for c in COLUNAS:
        tabela[c] = pd.to_numeric(tabela[c], errors="coerce")
    tabela = tabela.dropna(subset=obrigatorias).sort_values("timestamp")
    tabela["data_sessao"] = tabela["timestamp"].dt.normalize()
    tabela = tabela.drop_duplicates(subset=["data_sessao"], keep="last")
    return tabela.set_index("data_sessao")[COLUNAS].sort_index()


def aplicar_splits(ativo: str, serie_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    arquivo = DIR_SPLITS / f"{ativo}.csv"
    if not arquivo.exists():
        raise RuntimeError(f"{ativo}: splits ausentes: {arquivo}")
    eventos = pd.read_csv(arquivo)
    fator_sessao = pd.Series(1.0, index=serie_raw.index, dtype=float)
    if not eventos.empty:
        eventos["timestamp"] = pd.to_datetime(eventos["timestamp"], utc=True, errors="coerce")
        eventos["fator_split"] = pd.to_numeric(eventos["fator_split"], errors="coerce")
        if "status" in eventos.columns:
            eventos = eventos.loc[eventos["status"].astype(str).str.lower().str.strip() == "a"]
        eventos = eventos.dropna(subset=["timestamp", "fator_split"])
        eventos = eventos.loc[eventos["fator_split"] > 0]
        for evento in eventos.itertuples(index=False):
            data = pd.Timestamp(evento.timestamp).normalize()
            if data not in fator_sessao.index:
                continue
            fator_sessao.loc[data] *= float(evento.fator_split)
    acumulado = fator_sessao.cumprod()
    serie = serie_raw.copy()
    for c in ("open", "high", "low", "close"):
        serie[c] = serie[c] * acumulado
    serie["volume"] = serie["volume"] / acumulado
    return serie, acumulado


def aplicar_dividendos(ativo: str, serie_split: pd.DataFrame, fator_split: pd.Series) -> pd.DataFrame:
    arquivo = DIR_EVENTOS / f"{ativo}.csv"
    if not arquivo.exists():
        raise RuntimeError(f"{ativo}: eventos ausentes: {arquivo}")
    eventos = pd.read_csv(arquivo)
    fator_sessao = pd.Series(1.0, index=serie_split.index, dtype=float)
    if not eventos.empty:
        eventos["timestamp"] = pd.to_datetime(eventos["timestamp"], utc=True, errors="coerce")
        eventos["dividendo"] = pd.to_numeric(eventos["dividendo"], errors="coerce").fillna(0.0)
        eventos = eventos.dropna(subset=["timestamp"])
        eventos = eventos.loc[eventos["dividendo"] != 0.0].sort_values("timestamp")
        for evento in eventos.itertuples(index=False):
            data = pd.Timestamp(evento.timestamp).normalize()
            if data not in serie_split.index:
                continue
            pos = int(serie_split.index.get_loc(data))
            if pos == 0:
                continue
            anterior = float(serie_split.iloc[pos - 1]["close"])
            dividendo = float(evento.dividendo) * float(fator_split.loc[data])
            denominador = anterior - dividendo
            if not np.isfinite(denominador) or denominador <= 0:
                raise RuntimeError(f"{ativo}: dividendo invalido em {data.date()}")
            fator_sessao.loc[data] *= anterior / denominador
    acumulado = fator_sessao.cumprod()
    serie = serie_split.copy()
    for c in ("open", "high", "low", "close"):
        serie[c] = serie[c] * acumulado
    return serie


def data_iso(indice: Any) -> str | None:
    if indice is None or pd.isna(indice):
        return None
    return pd.Timestamp(indice).date().isoformat()


for diretorio in (DIR_ALPACA, DIR_TIINGO, DIR_EVENTOS, DIR_SPLITS):
    if not diretorio.exists():
        raise RuntimeError(f"Diretorio ausente: {diretorio}")
DIR_OUT.mkdir(parents=True, exist_ok=True)
registrar("Comparando Alpaca atual SIP adjustment=all com Tiingo total-causal")

resumos: list[dict[str, Any]] = []
detalhes: list[pd.DataFrame] = []
for posicao, ativo in enumerate(ATIVOS, start=1):
    alpaca = ler_serie(DIR_ALPACA, ativo)
    tiingo_raw = ler_serie(DIR_TIINGO, ativo)
    tiingo_split, fator_split = aplicar_splits(ativo, tiingo_raw)
    tiingo = aplicar_dividendos(ativo, tiingo_split, fator_split)
    datas = alpaca.index.intersection(tiingo.index)
    if len(datas) < 2:
        raise RuntimeError(f"{ativo}: datas comuns insuficientes")
    c = pd.DataFrame(index=datas)
    c["alpaca_close"] = alpaca.loc[datas, "close"].astype(float)
    c["tiingo_total_causal_close"] = tiingo.loc[datas, "close"].astype(float)
    c["retorno_alpaca"] = c["alpaca_close"].pct_change()
    c["retorno_tiingo"] = c["tiingo_total_causal_close"].pct_change()
    c["dif_retorno"] = c["retorno_tiingo"] - c["retorno_alpaca"]
    validos = c[["retorno_alpaca", "retorno_tiingo"]].dropna()
    correlacao = float(validos["retorno_alpaca"].corr(validos["retorno_tiingo"])) if len(validos) >= 2 else float("nan")
    erros = c["dif_retorno"].abs().dropna()
    divergentes = c.loc[c["dif_retorno"].abs() > LIMIAR]
    max_idx = erros.idxmax() if not erros.empty else None
    resumo = {
        "ativo": ativo,
        "datas_alpaca": len(alpaca),
        "datas_tiingo": len(tiingo),
        "datas_comuns": len(c),
        "inicio_comum": data_iso(c.index.min()),
        "fim_comum": data_iso(c.index.max()),
        "correlacao_retornos": correlacao,
        "erro_medio_abs_retorno_bps": float(erros.mean() * 10_000),
        "erro_p95_abs_retorno_bps": float(erros.quantile(0.95) * 10_000),
        "erro_max_abs_retorno_bps": float(erros.max() * 10_000),
        "data_erro_max": data_iso(max_idx),
        "primeira_divergencia_retorno_maior_0_1pct": data_iso(divergentes.index.min()) if not divergentes.empty else None,
        "dias_dif_maior_0_1pct": int((erros > 0.001).sum()),
        "dias_dif_maior_0_5pct": int((erros > 0.005).sum()),
        "dias_dif_maior_1pct": int((erros > 0.01).sum()),
    }
    resumos.append(resumo)
    d = c.reset_index().rename(columns={"data_sessao": "timestamp", "index": "timestamp"})
    d.insert(0, "ativo", ativo)
    detalhes.append(d)
    registrar(f"{posicao:02d}/{len(ATIVOS)} {ativo} | erro={resumo['erro_medio_abs_retorno_bps']:.3f} bps | corr={correlacao:.6f}")

resumo_df = pd.DataFrame(resumos).sort_values("erro_medio_abs_retorno_bps", ascending=False)
detalhe_df = pd.concat(detalhes, ignore_index=True)
resumo_df.to_csv(DIR_OUT / "comparacao_precos_resumo.csv", index=False)
detalhe_df.to_csv(DIR_OUT / "comparacao_precos_detalhe.csv", index=False)

politica: dict[str, Any] = {"disponivel": False}
if CURVA_ALPACA.exists() and CURVA_TIINGO.exists():
    a = pd.read_csv(CURVA_ALPACA)
    t = pd.read_csv(CURVA_TIINGO)
    def preparar_curva(df: pd.DataFrame, prefixo: str) -> pd.DataFrame:
        data_col = next((c for c in ("timestamp", "date", "data_sessao", "index") if c in df.columns), None)
        if data_col is None:
            raise RuntimeError(f"Curva {prefixo}: coluna de data nao encontrada")
        if "selected_asset" not in df.columns:
            raise RuntimeError(f"Curva {prefixo}: selected_asset ausente")
        saida = df.copy()
        saida["data_sessao"] = pd.to_datetime(saida[data_col], utc=True, errors="coerce").dt.normalize()
        colunas = ["data_sessao", "selected_asset"]
        if "strategy_equity" in saida.columns:
            colunas.append("strategy_equity")
        saida = saida[colunas].dropna(subset=["data_sessao"]).drop_duplicates("data_sessao", keep="last")
        ren = {"selected_asset": f"selected_asset_{prefixo}"}
        if "strategy_equity" in saida.columns:
            ren["strategy_equity"] = f"strategy_equity_{prefixo}"
        return saida.rename(columns=ren)
    pa = preparar_curva(a, "alpaca")
    pt = preparar_curva(t, "tiingo")
    p = pa.merge(pt, on="data_sessao", how="inner").sort_values("data_sessao")
    p["decisao_divergente"] = p["selected_asset_alpaca"].astype(str) != p["selected_asset_tiingo"].astype(str)
    if "strategy_equity_alpaca" in p.columns and "strategy_equity_tiingo" in p.columns:
        p["razao_equity_tiingo_alpaca"] = p["strategy_equity_tiingo"] / p["strategy_equity_alpaca"]
    diverg = p.loc[p["decisao_divergente"]]
    primeira = diverg.iloc[0] if not diverg.empty else None
    politica = {
        "disponivel": True,
        "sessoes_comuns": int(len(p)),
        "sessoes_decisao_divergente": int(p["decisao_divergente"].sum()),
        "percentual_decisao_divergente": float(p["decisao_divergente"].mean()) if len(p) else 0.0,
        "primeira_decisao_divergente": data_iso(primeira["data_sessao"]) if primeira is not None else None,
        "ativo_alpaca_primeira_divergencia": str(primeira["selected_asset_alpaca"]) if primeira is not None else None,
        "ativo_tiingo_primeira_divergencia": str(primeira["selected_asset_tiingo"]) if primeira is not None else None,
    }
    p.to_csv(DIR_OUT / "comparacao_politica.csv", index=False)
    registrar(
        "Politica: primeira divergencia=" + str(politica["primeira_decisao_divergente"])
        + f" | sessoes divergentes={politica['sessoes_decisao_divergente']}/{politica['sessoes_comuns']}"
    )
else:
    registrar("Comparacao de politica ainda indisponivel: execute os dois backtests primeiro.")

relatorio = {
    "schema_version": 1,
    "script_version": VERSAO,
    "comparison": "alpaca_current_sip_adjustment_all_vs_tiingo_total_causal",
    "assets": len(ATIVOS),
    "erro_medio_abs_bps_ponderado_por_observacao": float(detalhe_df["dif_retorno"].abs().dropna().mean() * 10_000),
    "politica": politica,
    "top_10_ativos_por_erro_medio_bps": resumo_df.head(10).to_dict(orient="records"),
}
(DIR_OUT / "relatorio.json").write_text(json.dumps(relatorio, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
registrar(f"Resultados: {DIR_OUT}")
