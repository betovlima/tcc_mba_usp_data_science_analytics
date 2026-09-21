"""Auditoria de features e targets: Alpaca atual versus Tiingo total-causal.

Versao: alpaca-tiingo-feature-audit-v1.0.0

Objetivo
--------
Comparar as duas fontes no nivel que realmente entra no modelo:
OHLCV normalizado -> features -> targets. O script nao treina modelos e nao
altera os snapshots. Ele apenas reconstrui a mesma transformacao usada pelo TCC
para localizar quais variaveis diferem antes da primeira decisao divergente.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tcc_engine.capital_rotation import ROTATION_FEATURES, build_rotation_frame
from tcc_engine.config import ASSETS as ATIVOS
from tcc_engine.config import CONFIG as CONFIGURACAO

VERSAO = "alpaca-tiingo-feature-audit-v1.0.0"
RAIZ = Path(__file__).resolve().parent
DIR_ALPACA = RAIZ / "dados" / "referencia_alpaca_atual"
DIR_TIINGO = RAIZ / "dados" / "series_historicas"
DIR_EVENTOS = RAIZ / "dados" / "eventos_corporativos"
DIR_SPLITS = RAIZ / "dados" / "desdobramentos"
DIR_OUT = RAIZ / "output" / "auditoria_features_alpaca_vs_tiingo"
COLUNAS = ["open", "high", "low", "close", "volume"]
TARGETS = [
    "forward_net_log_return",
    "forward_cash_edge",
    "forward_movement_capture",
    "forward_trend_persistence",
    "forward_risk_adjusted_utility",
]
PRIMEIRA_DIVERGENCIA = pd.Timestamp("2020-08-03", tz="UTC")
JANELAS = {
    "fold1_treino": (None, pd.Timestamp("2019-07-30", tz="UTC")),
    "fold1_calibracao": (
        pd.Timestamp("2019-10-24", tz="UTC"),
        pd.Timestamp("2020-04-24", tz="UTC"),
    ),
    "ate_primeira_divergencia": (None, PRIMEIRA_DIVERGENCIA),
    "historico_completo": (None, None),
}


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
        raise RuntimeError(f"{ativo}: colunas ausentes: {', '.join(ausentes)}")
    tabela = tabela[obrigatorias].copy()
    tabela["timestamp"] = pd.to_datetime(
        tabela["timestamp"], utc=True, errors="coerce"
    )
    for c in COLUNAS:
        tabela[c] = pd.to_numeric(tabela[c], errors="coerce")
    tabela = tabela.dropna(subset=obrigatorias).sort_values("timestamp")
    tabela["data_sessao"] = tabela["timestamp"].dt.normalize()
    tabela = tabela.drop_duplicates("data_sessao", keep="last")
    return tabela.set_index("data_sessao")[COLUNAS].sort_index()


def aplicar_splits(
    ativo: str, serie_raw: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    arquivo = DIR_SPLITS / f"{ativo}.csv"
    eventos = pd.read_csv(arquivo)
    fator_sessao = pd.Series(1.0, index=serie_raw.index, dtype=float)
    if not eventos.empty:
        eventos["timestamp"] = pd.to_datetime(
            eventos["timestamp"], utc=True, errors="coerce"
        )
        eventos["fator_split"] = pd.to_numeric(
            eventos["fator_split"], errors="coerce"
        )
        if "status" in eventos.columns:
            eventos = eventos.loc[
                eventos["status"].astype(str).str.lower().str.strip() == "a"
            ]
        eventos = eventos.dropna(subset=["timestamp", "fator_split"])
        eventos = eventos.loc[eventos["fator_split"] > 0]
        for evento in eventos.itertuples(index=False):
            data = pd.Timestamp(evento.timestamp).normalize()
            if data in fator_sessao.index:
                fator_sessao.loc[data] *= float(evento.fator_split)

    acumulado = fator_sessao.cumprod()
    serie = serie_raw.copy()
    for c in ("open", "high", "low", "close"):
        serie[c] = serie[c] * acumulado
    serie["volume"] = serie["volume"] / acumulado
    return serie, acumulado


def aplicar_dividendos(
    ativo: str, serie_split: pd.DataFrame, fator_split: pd.Series
) -> pd.DataFrame:
    arquivo = DIR_EVENTOS / f"{ativo}.csv"
    eventos = pd.read_csv(arquivo)
    fator_sessao = pd.Series(1.0, index=serie_split.index, dtype=float)
    if not eventos.empty:
        eventos["timestamp"] = pd.to_datetime(
            eventos["timestamp"], utc=True, errors="coerce"
        )
        eventos["dividendo"] = pd.to_numeric(
            eventos["dividendo"], errors="coerce"
        ).fillna(0.0)
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


def recortar(
    tabela: pd.DataFrame,
    inicio: pd.Timestamp | None,
    fim: pd.Timestamp | None,
) -> pd.DataFrame:
    saida = tabela
    if inicio is not None:
        saida = saida.loc[saida.index >= inicio]
    if fim is not None:
        saida = saida.loc[saida.index <= fim]
    return saida


def resumo_diferenca(
    ativo: str,
    janela: str,
    variavel: str,
    classe: str,
    alpaca: pd.Series,
    tiingo: pd.Series,
) -> dict[str, Any] | None:
    dados = pd.concat(
        [alpaca.rename("alpaca"), tiingo.rename("tiingo")], axis=1, join="inner"
    ).dropna()
    if dados.empty:
        return None
    diferenca = dados["tiingo"] - dados["alpaca"]
    abs_dif = diferenca.abs()
    escala = dados["alpaca"].abs().median()
    normalizador = float(escala) if np.isfinite(escala) and escala > 1e-12 else 1.0
    max_idx = abs_dif.idxmax()
    return {
        "ativo": ativo,
        "janela": janela,
        "classe": classe,
        "variavel": variavel,
        "observacoes": int(len(dados)),
        "erro_medio_abs": float(abs_dif.mean()),
        "erro_p95_abs": float(abs_dif.quantile(0.95)),
        "erro_max_abs": float(abs_dif.max()),
        "erro_medio_abs_normalizado": float(abs_dif.mean() / normalizador),
        "correlacao": float(dados["alpaca"].corr(dados["tiingo"]))
        if len(dados) >= 2
        else float("nan"),
        "data_erro_max": pd.Timestamp(max_idx).date().isoformat(),
        "valor_alpaca_erro_max": float(dados.loc[max_idx, "alpaca"]),
        "valor_tiingo_erro_max": float(dados.loc[max_idx, "tiingo"]),
    }


for diretorio in (DIR_ALPACA, DIR_TIINGO, DIR_EVENTOS, DIR_SPLITS):
    if not diretorio.exists():
        raise RuntimeError(f"Diretorio ausente: {diretorio}")
DIR_OUT.mkdir(parents=True, exist_ok=True)

registrar("Auditoria OHLCV -> features -> targets")
linhas_resumo: list[dict[str, Any]] = []
linhas_contexto: list[dict[str, Any]] = []
linhas_ohlcv: list[dict[str, Any]] = []

for posicao, ativo in enumerate(ATIVOS, start=1):
    alpaca = ler_serie(DIR_ALPACA, ativo)
    tiingo_raw = ler_serie(DIR_TIINGO, ativo)
    tiingo_split, fator_split = aplicar_splits(ativo, tiingo_raw)
    tiingo = aplicar_dividendos(ativo, tiingo_split, fator_split)

    datas = alpaca.index.intersection(tiingo.index)
    alpaca = alpaca.loc[datas].copy()
    tiingo = tiingo.loc[datas].copy()

    # Comparacoes OHLCV invariantes a escala de preco sempre que possivel.
    forma_alpaca = pd.DataFrame(index=datas)
    forma_tiingo = pd.DataFrame(index=datas)
    for nome in ("open", "high", "low"):
        forma_alpaca[f"{nome}_sobre_close"] = alpaca[nome] / alpaca["close"] - 1.0
        forma_tiingo[f"{nome}_sobre_close"] = tiingo[nome] / tiingo["close"] - 1.0
    forma_alpaca["retorno_close"] = alpaca["close"].pct_change()
    forma_tiingo["retorno_close"] = tiingo["close"].pct_change()
    forma_alpaca["volume_variacao"] = alpaca["volume"].pct_change()
    forma_tiingo["volume_variacao"] = tiingo["volume"].pct_change()

    for variavel in forma_alpaca.columns:
        item = resumo_diferenca(
            ativo,
            "historico_completo",
            variavel,
            "ohlcv",
            forma_alpaca[variavel],
            forma_tiingo[variavel],
        )
        if item:
            linhas_ohlcv.append(item)

    frame_a = build_rotation_frame(alpaca, CONFIGURACAO)
    frame_t = build_rotation_frame(tiingo, CONFIGURACAO)
    comuns = frame_a.index.intersection(frame_t.index)
    frame_a = frame_a.loc[comuns]
    frame_t = frame_t.loc[comuns]

    for janela, (inicio, fim) in JANELAS.items():
        a = recortar(frame_a, inicio, fim)
        t = recortar(frame_t, inicio, fim)
        idx = a.index.intersection(t.index)
        a = a.loc[idx]
        t = t.loc[idx]
        for variavel in [*ROTATION_FEATURES, *TARGETS]:
            if variavel not in a.columns or variavel not in t.columns:
                continue
            classe = "target" if variavel in TARGETS else "feature"
            item = resumo_diferenca(
                ativo, janela, variavel, classe, a[variavel], t[variavel]
            )
            if item:
                linhas_resumo.append(item)

    if ativo in {"AMD", "GKOS", "CORT"} and PRIMEIRA_DIVERGENCIA in frame_a.index and PRIMEIRA_DIVERGENCIA in frame_t.index:
        for variavel in [*ROTATION_FEATURES, *TARGETS]:
            if variavel not in frame_a.columns or variavel not in frame_t.columns:
                continue
            va = float(frame_a.loc[PRIMEIRA_DIVERGENCIA, variavel])
            vt = float(frame_t.loc[PRIMEIRA_DIVERGENCIA, variavel])
            linhas_contexto.append(
                {
                    "data": PRIMEIRA_DIVERGENCIA.date().isoformat(),
                    "ativo": ativo,
                    "variavel": variavel,
                    "classe": "target" if variavel in TARGETS else "feature",
                    "alpaca": va,
                    "tiingo": vt,
                    "diferenca": vt - va,
                    "abs_diferenca": abs(vt - va),
                }
            )

    registrar(
        f"{posicao:02d}/{len(ATIVOS)} {ativo} | "
        f"frames comuns={len(comuns)}"
    )

resumo = pd.DataFrame(linhas_resumo)
ohlcv = pd.DataFrame(linhas_ohlcv)
contexto = pd.DataFrame(linhas_contexto)

if not resumo.empty:
    resumo = resumo.sort_values(
        ["janela", "erro_medio_abs_normalizado"], ascending=[True, False]
    )
    resumo.to_csv(DIR_OUT / "comparacao_features_targets_resumo.csv", index=False)
if not ohlcv.empty:
    ohlcv = ohlcv.sort_values("erro_medio_abs_normalizado", ascending=False)
    ohlcv.to_csv(DIR_OUT / "comparacao_ohlcv_resumo.csv", index=False)
if not contexto.empty:
    contexto = contexto.sort_values("abs_diferenca", ascending=False)
    contexto.to_csv(DIR_OUT / "contexto_primeira_divergencia_2020_08_03.csv", index=False)

relatorio = {
    "schema_version": 1,
    "script_version": VERSAO,
    "primeira_decisao_divergente": "2020-08-03",
    "ativos_contexto_primeira_divergencia": ["AMD", "GKOS", "CORT"],
    "top_features_fold1_treino": (
        resumo.loc[
            (resumo["janela"] == "fold1_treino")
            & (resumo["classe"] == "feature")
        ]
        .head(25)
        .to_dict(orient="records")
        if not resumo.empty
        else []
    ),
    "top_targets_fold1_treino": (
        resumo.loc[
            (resumo["janela"] == "fold1_treino")
            & (resumo["classe"] == "target")
        ]
        .head(25)
        .to_dict(orient="records")
        if not resumo.empty
        else []
    ),
    "top_features_fold1_calibracao": (
        resumo.loc[
            (resumo["janela"] == "fold1_calibracao")
            & (resumo["classe"] == "feature")
        ]
        .head(25)
        .to_dict(orient="records")
        if not resumo.empty
        else []
    ),
    "top_ohlcv": ohlcv.head(25).to_dict(orient="records") if not ohlcv.empty else [],
}
(DIR_OUT / "relatorio_features.json").write_text(
    json.dumps(relatorio, indent=2, ensure_ascii=False, default=str) + "\n",
    encoding="utf-8",
)
registrar(f"Resultados: {DIR_OUT}")
