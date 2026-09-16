"""Compara a referencia Mongo 43M com o snapshot Tiingo congelado.

O script nao treina modelos e nao altera dados. Ele compara, por ativo e data:

1. serie armazenada no Mongo que reproduz o backtest historico;
2. serie RAW congelada da Tiingo;
3. serie Tiingo com normalizacao causal apenas de desdobramentos.

A finalidade e localizar exatamente onde e como as fontes divergem antes de
qualquer nova calibracao de hiperparametros ou politica de rotacao.
"""

# %% 0 - Imports e configuracao
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tcc_engine.config import ASSETS as ATIVOS

RAIZ_PROJETO = Path(__file__).resolve().parent
DIRETORIO_MONGO = RAIZ_PROJETO / "dados" / "referencia_mongo_43m"
DIRETORIO_TIINGO = RAIZ_PROJETO / "dados" / "series_historicas"
DIRETORIO_DESDOBRAMENTOS = RAIZ_PROJETO / "dados" / "desdobramentos"
DIRETORIO_RESULTADOS = RAIZ_PROJETO / "output"

COLUNAS_OHLCV = ["open", "high", "low", "close", "volume"]
LIMIAR_DIVERGENCIA = 0.001  # 0,10%


def registrar(mensagem: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {mensagem}", flush=True)


def ler_serie(caminho: Path, ativo: str) -> pd.DataFrame:
    arquivo = caminho / f"{ativo}.csv"
    if not arquivo.exists():
        raise RuntimeError(f"Arquivo ausente para {ativo}: {arquivo}")

    tabela = pd.read_csv(arquivo)
    obrigatorias = ["timestamp", *COLUNAS_OHLCV]
    ausentes = [coluna for coluna in obrigatorias if coluna not in tabela.columns]
    if ausentes:
        raise RuntimeError(
            f"{ativo}: colunas ausentes em {arquivo}: " + ", ".join(ausentes)
        )

    tabela = tabela[obrigatorias].copy()
    tabela["timestamp"] = pd.to_datetime(tabela["timestamp"], utc=True, errors="coerce")
    for coluna in COLUNAS_OHLCV:
        tabela[coluna] = pd.to_numeric(tabela[coluna], errors="coerce")
    tabela = tabela.dropna(subset=obrigatorias)
    tabela = tabela.sort_values("timestamp")
    tabela = tabela.drop_duplicates(subset=["timestamp"], keep="last")
    return tabela.set_index("timestamp")


def normalizar_desdobramentos_causalmente(
    ativo: str,
    serie_bruta: pd.DataFrame,
) -> pd.DataFrame:
    arquivo = DIRETORIO_DESDOBRAMENTOS / f"{ativo}.csv"
    if not arquivo.exists():
        raise RuntimeError(
            f"{ativo}: desdobramentos ausentes. Execute congelar_desdobramentos_tiingo.py."
        )

    eventos = pd.read_csv(arquivo)
    serie = serie_bruta.copy()
    fator_na_sessao = pd.Series(1.0, index=serie.index, dtype=float)

    if not eventos.empty:
        eventos["timestamp"] = pd.to_datetime(
            eventos["timestamp"], utc=True, errors="coerce"
        )
        eventos["fator_split"] = pd.to_numeric(
            eventos["fator_split"], errors="coerce"
        )
        eventos = eventos.dropna(subset=["timestamp", "fator_split"])
        eventos = eventos.loc[eventos["fator_split"] > 0].copy()

        for evento in eventos.itertuples(index=False):
            data_evento = pd.Timestamp(evento.timestamp).normalize()
            sessoes = serie.index[serie.index.normalize() == data_evento]
            if len(sessoes) != 1:
                raise RuntimeError(
                    f"{ativo}: evento de {data_evento.date()} sem sessao unica correspondente."
                )
            fator_na_sessao.loc[sessoes[0]] *= float(evento.fator_split)

    fator_acumulado = fator_na_sessao.cumprod()
    for coluna in ("open", "high", "low", "close"):
        serie[coluna] = serie[coluna] * fator_acumulado
    serie["volume"] = serie["volume"] / fator_acumulado
    return serie


# %% 1 - Validacao das entradas
for diretorio, descricao in (
    (DIRETORIO_MONGO, "referencia Mongo 43M"),
    (DIRETORIO_TIINGO, "snapshot Tiingo RAW"),
    (DIRETORIO_DESDOBRAMENTOS, "desdobramentos"),
):
    if not diretorio.exists():
        raise RuntimeError(f"Diretorio de {descricao} nao encontrado: {diretorio}")

registrar("Comparando referencia Mongo 43M com Tiingo RAW e Tiingo causal")
registrar(f"Ativos: {len(ATIVOS)}")
registrar(f"Primeira divergencia relevante: abs(diferenca) > {LIMIAR_DIVERGENCIA:.2%}")

DIRETORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
resumos: list[dict[str, object]] = []
detalhes: list[pd.DataFrame] = []


# %% 2 - Comparacao ativo a ativo
for posicao, ativo in enumerate(ATIVOS, start=1):
    mongo = ler_serie(DIRETORIO_MONGO, ativo)
    tiingo_raw = ler_serie(DIRETORIO_TIINGO, ativo)
    tiingo_causal = normalizar_desdobramentos_causalmente(ativo, tiingo_raw)

    datas = mongo.index.intersection(tiingo_raw.index).intersection(tiingo_causal.index)
    if len(datas) < 2:
        raise RuntimeError(f"{ativo}: poucas datas comuns para comparacao.")

    comparacao = pd.DataFrame(index=datas)
    comparacao["mongo_close"] = mongo.loc[datas, "close"].astype(float)
    comparacao["tiingo_raw_close"] = tiingo_raw.loc[datas, "close"].astype(float)
    comparacao["tiingo_causal_close"] = tiingo_causal.loc[datas, "close"].astype(float)

    comparacao["dif_preco_raw"] = (
        comparacao["tiingo_raw_close"] / comparacao["mongo_close"] - 1.0
    )
    comparacao["dif_preco_causal"] = (
        comparacao["tiingo_causal_close"] / comparacao["mongo_close"] - 1.0
    )

    comparacao["retorno_mongo"] = comparacao["mongo_close"].pct_change()
    comparacao["retorno_tiingo_raw"] = comparacao["tiingo_raw_close"].pct_change()
    comparacao["retorno_tiingo_causal"] = comparacao["tiingo_causal_close"].pct_change()
    comparacao["dif_retorno_causal"] = (
        comparacao["retorno_tiingo_causal"] - comparacao["retorno_mongo"]
    )

    validos_retorno = comparacao[["retorno_mongo", "retorno_tiingo_causal"]].dropna()
    correlacao = (
        float(validos_retorno["retorno_mongo"].corr(validos_retorno["retorno_tiingo_causal"]))
        if len(validos_retorno) >= 2
        else float("nan")
    )

    divergencias = comparacao.loc[
        comparacao["dif_preco_causal"].abs() > LIMIAR_DIVERGENCIA
    ]
    primeira_divergencia = (
        divergencias.index.min().date().isoformat() if not divergencias.empty else None
    )

    resumo = {
        "ativo": ativo,
        "datas_comuns": len(comparacao),
        "inicio_comum": comparacao.index.min().date().isoformat(),
        "fim_comum": comparacao.index.max().date().isoformat(),
        "primeira_divergencia_causal_maior_0_1pct": primeira_divergencia,
        "mediana_abs_dif_preco_raw": float(comparacao["dif_preco_raw"].abs().median()),
        "mediana_abs_dif_preco_causal": float(comparacao["dif_preco_causal"].abs().median()),
        "max_abs_dif_preco_causal": float(comparacao["dif_preco_causal"].abs().max()),
        "correlacao_retornos_mongo_tiingo_causal": correlacao,
        "erro_medio_abs_retorno_causal_bps": float(
            comparacao["dif_retorno_causal"].abs().dropna().mean() * 10_000
        ),
        "dias_dif_preco_causal_maior_0_1pct": int(
            (comparacao["dif_preco_causal"].abs() > 0.001).sum()
        ),
        "dias_dif_preco_causal_maior_0_5pct": int(
            (comparacao["dif_preco_causal"].abs() > 0.005).sum()
        ),
        "dias_dif_preco_causal_maior_1pct": int(
            (comparacao["dif_preco_causal"].abs() > 0.01).sum()
        ),
    }
    resumos.append(resumo)

    detalhe = comparacao.reset_index().rename(columns={"index": "timestamp"})
    detalhe.insert(0, "ativo", ativo)
    detalhes.append(detalhe)

    registrar(
        f"{posicao:02d}/{len(ATIVOS)} {ativo} | "
        f"correlacao retornos={correlacao:.6f} | "
        f"erro retorno={resumo['erro_medio_abs_retorno_causal_bps']:.2f} bps | "
        f"1a divergencia={primeira_divergencia or 'nenhuma'}"
    )


# %% 3 - Artefatos de auditoria
resumo_df = pd.DataFrame(resumos)
detalhe_df = pd.concat(detalhes, ignore_index=True)

resumo_df = resumo_df.sort_values(
    ["erro_medio_abs_retorno_causal_bps", "ativo"],
    ascending=[False, True],
)
detalhe_df["abs_dif_retorno_causal"] = detalhe_df["dif_retorno_causal"].abs()

resumo_df.to_csv(
    DIRETORIO_RESULTADOS / "comparacao_mongo_tiingo_resumo.csv",
    index=False,
)
detalhe_df.to_csv(
    DIRETORIO_RESULTADOS / "comparacao_mongo_tiingo_detalhe.csv",
    index=False,
)

top_divergencias = detalhe_df.sort_values(
    ["abs_dif_retorno_causal", "ativo", "timestamp"],
    ascending=[False, True, True],
).head(500)
top_divergencias.to_csv(
    DIRETORIO_RESULTADOS / "comparacao_mongo_tiingo_top_divergencias.csv",
    index=False,
)

registrar("Comparacao concluida.")
registrar(
    "Arquivos: comparacao_mongo_tiingo_resumo.csv, "
    "comparacao_mongo_tiingo_detalhe.csv e "
    "comparacao_mongo_tiingo_top_divergencias.csv"
)
