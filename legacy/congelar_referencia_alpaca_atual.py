"""Congela a referencia Alpaca atual usada como controle do TCC.

Versao: alpaca-tiingo-audit-v1.0.0

Fonte: Alpaca Stock Bars, feed SIP, adjustment=all, timeframe=1Day.
O snapshot e gravado separadamente da base Tiingo e nunca sobrescreve
``dados/series_historicas``.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import dotenv_values, load_dotenv

from tcc_engine.config import ASSETS as ATIVOS
from tcc_engine.config import END_DATE as DATA_FIM
from tcc_engine.config import START_DATE as DATA_INICIO

VERSAO = "alpaca-tiingo-audit-v1.0.0"
RAIZ_PROJETO = Path(__file__).resolve().parent
DIRETORIO_SNAPSHOT = RAIZ_PROJETO / "dados" / "referencia_alpaca_atual"
ARQUIVO_MANIFESTO = RAIZ_PROJETO / "dados" / "manifesto_referencia_alpaca_atual.json"
COLUNAS = ["timestamp", "open", "high", "low", "close", "volume"]


def sha256_arquivo(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def localizar_env() -> Path:
    candidatos = [RAIZ_PROJETO / ".env", Path.cwd() / ".env", RAIZ_PROJETO.parent / ".env"]
    unicos: list[Path] = []
    for candidato in candidatos:
        resolvido = candidato.resolve()
        if resolvido not in unicos:
            unicos.append(resolvido)
    encontrado = next((caminho for caminho in unicos if caminho.exists()), None)
    if encontrado is None:
        caminhos = "\n".join(f"- {caminho}" for caminho in unicos)
        raise RuntimeError("Arquivo .env nao encontrado. Verificados:\n" + caminhos)
    return encontrado


def ler_credencial(valores: dict[str, str], *nomes: str) -> tuple[str, str | None]:
    for nome in nomes:
        valor = str(valores.get(nome) or os.getenv(nome) or "").strip()
        if valor:
            return valor, nome
    return "", None


arquivo_env = localizar_env()
load_dotenv(arquivo_env, override=True)
valores_env = {str(k): str(v or "").strip() for k, v in dotenv_values(arquivo_env).items()}
chave, nome_chave = ler_credencial(valores_env, "ALPACA_API_KEY", "ALPACA_API_KEY_ID", "APCA_API_KEY_ID")
segredo, nome_segredo = ler_credencial(valores_env, "ALPACA_SECRET_KEY", "ALPACA_API_SECRET_KEY", "APCA_API_SECRET_KEY")
if not chave or not segredo:
    raise RuntimeError("Credenciais Alpaca nao encontradas no .env.")

cliente = StockHistoricalDataClient(api_key=chave, secret_key=segredo)
inicio = pd.Timestamp(DATA_INICIO, tz="UTC").to_pydatetime()
fim = (pd.Timestamp(DATA_FIM, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)).to_pydatetime()
DIRETORIO_SNAPSHOT.mkdir(parents=True, exist_ok=True)

print("TCC MBA USP - congelamento da referencia Alpaca atual")
print(f"Versao: {VERSAO}")
print(f"Periodo: {DATA_INICIO} -> {DATA_FIM}")
print("Fonte: Alpaca | timeframe=1Day | feed=SIP | adjustment=all")
print(f"Ativos: {len(ATIVOS)}")
print(f"Credenciais: {nome_chave} + {nome_segredo}")

arquivos: dict[str, dict[str, object]] = {}
for posicao, ativo in enumerate(ATIVOS, start=1):
    print(f"[{posicao:02d}/{len(ATIVOS)}] {ativo}", flush=True)
    solicitacao = StockBarsRequest(
        symbol_or_symbols=ativo,
        timeframe=TimeFrame.Day,
        start=inicio,
        end=fim,
        adjustment=Adjustment.ALL,
        feed=DataFeed.SIP,
        limit=10_000,
    )
    resposta = cliente.get_stock_bars(solicitacao)
    serie = resposta.df.copy()
    if serie.empty:
        raise RuntimeError(f"Alpaca nao retornou dados para {ativo}.")
    if isinstance(serie.index, pd.MultiIndex) and "symbol" in list(serie.index.names):
        serie = serie.xs(ativo, level="symbol")
    serie = serie.reset_index()
    serie.columns = [str(coluna).lower() for coluna in serie.columns]
    if "symbol" in serie.columns:
        serie = serie.drop(columns=["symbol"])
    ausentes = [coluna for coluna in COLUNAS if coluna not in serie.columns]
    if ausentes:
        raise RuntimeError(f"{ativo}: colunas ausentes: {', '.join(ausentes)}")
    serie = serie[COLUNAS].copy()
    serie["timestamp"] = pd.to_datetime(serie["timestamp"], utc=True, errors="coerce")
    for coluna in COLUNAS[1:]:
        serie[coluna] = pd.to_numeric(serie[coluna], errors="coerce")
    serie = serie.dropna(subset=COLUNAS).sort_values("timestamp")
    serie = serie.drop_duplicates(subset=["timestamp"], keep="last")
    valida = (
        (serie["open"] > 0) & (serie["high"] > 0) & (serie["low"] > 0)
        & (serie["close"] > 0) & (serie["volume"] >= 0)
    )
    serie = serie.loc[valida].copy()
    if serie.empty:
        raise RuntimeError(f"{ativo}: serie vazia depois da validacao.")
    arquivo = DIRETORIO_SNAPSHOT / f"{ativo}.csv"
    serie.to_csv(arquivo, index=False)
    arquivos[ativo] = {
        "arquivo": str(arquivo.relative_to(RAIZ_PROJETO)),
        "linhas": int(len(serie)),
        "inicio": serie["timestamp"].min().isoformat(),
        "fim": serie["timestamp"].max().isoformat(),
        "sha256": sha256_arquivo(arquivo),
    }

manifesto = {
    "schema_version": 1,
    "script_version": VERSAO,
    "data_congelamento_utc": datetime.now(timezone.utc).isoformat(),
    "source": "alpaca_stock_bars",
    "feed": "sip",
    "adjustment": "all",
    "timeframe": "1Day",
    "history_start": DATA_INICIO,
    "history_end": DATA_FIM,
    "asset_count": len(ATIVOS),
    "assets": list(ATIVOS),
    "files": arquivos,
}
ARQUIVO_MANIFESTO.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Snapshot concluido: {DIRETORIO_SNAPSHOT}")
print(f"Manifesto: {ARQUIVO_MANIFESTO}")
