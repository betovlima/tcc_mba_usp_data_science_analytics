"""Compara a referencia Mongo 43M com o snapshot Tiingo congelado.

O script nao treina modelos e nao altera dados. Ele compara, por ativo e data:

1. serie armazenada no Mongo que reproduz o backtest historico;
2. serie RAW congelada da Tiingo;
3. serie Tiingo com normalizacao causal apenas de desdobramentos;
4. serie Tiingo com desdobramentos e neutralizacao causal do efeito mecanico
   de dividendos sobre o preco.

As fontes podem representar o mesmo pregao com horarios UTC diferentes. Por
isso, a comparacao e feita pela data da sessao. A serie RAW permanece intocada.
A neutralizacao de dividendos usa somente informacoes conhecidas na data do
evento e atua dali em diante, sem reescrever o passado.
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
DIRETORIO_EVENTOS = RAIZ_PROJETO / "dados" / "eventos_corporativos"
DIRETORIO_DESDOBRAMENTOS = RAIZ_PROJETO / "dados" / "desdobramentos"
DIRETORIO_RESULTADOS = RAIZ_PROJETO / "output"

COLUNAS_OHLCV = ["open", "high", "low", "close", "volume"]
LIMIAR_DIVERGENCIA_RETORNO = 0.001  # 0,10 ponto percentual


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
    tabela["timestamp"] = pd.to_datetime(
        tabela["timestamp"], utc=True, errors="coerce"
    )
    for coluna in COLUNAS_OHLCV:
        tabela[coluna] = pd.to_numeric(tabela[coluna], errors="coerce")

    tabela = tabela.dropna(subset=obrigatorias)
    tabela = tabela.sort_values("timestamp")
    tabela["data_sessao"] = tabela["timestamp"].dt.normalize()
    tabela = tabela.drop_duplicates(subset=["data_sessao"], keep="last")
    tabela = tabela.set_index("data_sessao")[COLUNAS_OHLCV]
    tabela.index.name = "timestamp"
    return tabela.sort_index()


def normalizar_desdobramentos_causalmente(
    ativo: str,
    serie_bruta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
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
            if data_evento not in serie.index:
                raise RuntimeError(
                    f"{ativo}: evento de {data_evento.date()} sem sessao correspondente."
                )
            fator_na_sessao.loc[data_evento] *= float(evento.fator_split)

    fator_acumulado = fator_na_sessao.cumprod()
    for coluna in ("open", "high", "low", "close"):
        serie[coluna] = serie[coluna] * fator_acumulado
    serie["volume"] = serie["volume"] / fator_acumulado
    return serie, fator_acumulado


def neutralizar_dividendos_causalmente(
    ativo: str,
    serie_desdobrada: pd.DataFrame,
    fator_desdobramento: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Neutraliza a ruptura mecanica do ex-dividendo, da sessao para frente.

    O preco RAW nao e alterado. Para um dividendo conhecido na sessao t, o
    multiplicador passa a valer somente em t e nas sessoes seguintes:

        fator = fechamento_anterior / (fechamento_anterior - dividendo)

    O dividendo e primeiro convertido para a mesma unidade da serie ja
    normalizada por desdobramentos. Volume nao e ajustado por dividendos.
    """

    arquivo = DIRETORIO_EVENTOS / f"{ativo}.csv"
    if not arquivo.exists():
        raise RuntimeError(f"{ativo}: eventos corporativos ausentes: {arquivo}")

    eventos = pd.read_csv(arquivo)
    serie = serie_desdobrada.copy()
    fator_na_sessao = pd.Series(1.0, index=serie.index, dtype=float)

    if not eventos.empty:
        obrigatorias = {"timestamp", "dividendo"}
        if not obrigatorias.issubset(eventos.columns):
            raise RuntimeError(
                f"{ativo}: arquivo de eventos precisa conter timestamp e dividendo."
            )

        eventos["timestamp"] = pd.to_datetime(
            eventos["timestamp"], utc=True, errors="coerce"
        )
        eventos["dividendo"] = pd.to_numeric(
            eventos["dividendo"], errors="coerce"
        ).fillna(0.0)
        eventos = eventos.dropna(subset=["timestamp"])
        eventos = eventos.loc[eventos["dividendo"] != 0.0].copy()
        eventos = eventos.sort_values("timestamp")

        for evento in eventos.itertuples(index=False):
            data_evento = pd.Timestamp(evento.timestamp).normalize()
            if data_evento not in serie.index:
                continue

            posicao = int(serie.index.get_loc(data_evento))
            if posicao == 0:
                continue

            fechamento_anterior = float(serie.iloc[posicao - 1]["close"])
            dividendo_normalizado = (
                float(evento.dividendo) * float(fator_desdobramento.loc[data_evento])
            )
            denominador = fechamento_anterior - dividendo_normalizado

            if not np.isfinite(denominador) or denominador <= 0:
                raise RuntimeError(
                    f"{ativo}: dividendo invalido para neutralizacao em "
                    f"{data_evento.date()} | fechamento anterior={fechamento_anterior} | "
                    f"dividendo normalizado={dividendo_normalizado}"
                )

            fator_na_sessao.loc[data_evento] *= fechamento_anterior / denominador

    fator_acumulado = fator_na_sessao.cumprod()
    for coluna in ("open", "high", "low", "close"):
        serie[coluna] = serie[coluna] * fator_acumulado
    return serie, fator_acumulado


# %% 1 - Validacao das entradas
for diretorio, descricao in (
    (DIRETORIO_MONGO, "referencia Mongo 43M"),
    (DIRETORIO_TIINGO, "snapshot Tiingo RAW"),
    (DIRETORIO_EVENTOS, "eventos corporativos"),
    (DIRETORIO_DESDOBRAMENTOS, "desdobramentos"),
):
    if not diretorio.exists():
        raise RuntimeError(f"Diretorio de {descricao} nao encontrado: {diretorio}")

registrar("Comparando referencia Mongo 43M com Tiingo RAW e series causais")
registrar(f"Ativos: {len(ATIVOS)}")
registrar("Alinhamento: data da sessao diaria, ignorando diferencas de horario UTC")
registrar(
    "Primeira divergencia relevante de retorno: "
    f"abs(diferenca) > {LIMIAR_DIVERGENCIA_RETORNO:.2%}"
)
registrar("Dividendos: RAW intocado; neutralizacao causal avaliada separadamente")

DIRETORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
resumos: list[dict[str, object]] = []
detalhes: list[pd.DataFrame] = []


# %% 2 - Comparacao ativo a ativo
for posicao, ativo in enumerate(ATIVOS, start=1):
    mongo = ler_serie(DIRETORIO_MONGO, ativo)
    tiingo_raw = ler_serie(DIRETORIO_TIINGO, ativo)
    tiingo_desdobramentos, fator_desdobramento = normalizar_desdobramentos_causalmente(
        ativo,
        tiingo_raw,
    )
    tiingo_total_causal, _ = neutralizar_dividendos_causalmente(
        ativo,
        tiingo_desdobramentos,
        fator_desdobramento,
    )

    datas = (
        mongo.index.intersection(tiingo_raw.index)
        .intersection(tiingo_desdobramentos.index)
        .intersection(tiingo_total_causal.index)
    )
    if len(datas) < 2:
        raise RuntimeError(
            f"{ativo}: poucas datas comuns para comparacao | "
            f"Mongo={len(mongo)} | Tiingo={len(tiingo_raw)} | comuns={len(datas)}"
        )

    comparacao = pd.DataFrame(index=datas)
    comparacao["mongo_close"] = mongo.loc[datas, "close"].astype(float)
    comparacao["tiingo_raw_close"] = tiingo_raw.loc[datas, "close"].astype(float)
    comparacao["tiingo_desdobramentos_close"] = tiingo_desdobramentos.loc[
        datas, "close"
    ].astype(float)
    comparacao["tiingo_total_causal_close"] = tiingo_total_causal.loc[
        datas, "close"
    ].astype(float)

    comparacao["retorno_mongo"] = comparacao["mongo_close"].pct_change()
    comparacao["retorno_tiingo_raw"] = comparacao["tiingo_raw_close"].pct_change()
    comparacao["retorno_tiingo_desdobramentos"] = comparacao[
        "tiingo_desdobramentos_close"
    ].pct_change()
    comparacao["retorno_tiingo_total_causal"] = comparacao[
        "tiingo_total_causal_close"
    ].pct_change()
    comparacao["dif_retorno_raw"] = (
        comparacao["retorno_tiingo_raw"] - comparacao["retorno_mongo"]
    )
    comparacao["dif_retorno_desdobramentos"] = (
        comparacao["retorno_tiingo_desdobramentos"] - comparacao["retorno_mongo"]
    )
    comparacao["dif_retorno_total_causal"] = (
        comparacao["retorno_tiingo_total_causal"] - comparacao["retorno_mongo"]
    )

    validos_desdobramentos = comparacao[
        ["retorno_mongo", "retorno_tiingo_desdobramentos"]
    ].dropna()
    validos_total = comparacao[
        ["retorno_mongo", "retorno_tiingo_total_causal"]
    ].dropna()
    correlacao_desdobramentos = (
        float(
            validos_desdobramentos["retorno_mongo"].corr(
                validos_desdobramentos["retorno_tiingo_desdobramentos"]
            )
        )
        if len(validos_desdobramentos) >= 2
        else float("nan")
    )
    correlacao_total = (
        float(
            validos_total["retorno_mongo"].corr(
                validos_total["retorno_tiingo_total_causal"]
            )
        )
        if len(validos_total) >= 2
        else float("nan")
    )

    divergencias_retorno = comparacao.loc[
        comparacao["dif_retorno_total_causal"].abs() > LIMIAR_DIVERGENCIA_RETORNO
    ]
    primeira_divergencia = (
        divergencias_retorno.index.min().date().isoformat()
        if not divergencias_retorno.empty
        else None
    )

    erro_desdobramentos = float(
        comparacao["dif_retorno_desdobramentos"].abs().dropna().mean() * 10_000
    )
    erro_total = float(
        comparacao["dif_retorno_total_causal"].abs().dropna().mean() * 10_000
    )

    resumo = {
        "ativo": ativo,
        "datas_mongo": len(mongo),
        "datas_tiingo": len(tiingo_raw),
        "datas_comuns": len(comparacao),
        "inicio_comum": comparacao.index.min().date().isoformat(),
        "fim_comum": comparacao.index.max().date().isoformat(),
        "primeira_divergencia_retorno_total_causal_maior_0_1pct": primeira_divergencia,
        "correlacao_retornos_mongo_tiingo_desdobramentos": correlacao_desdobramentos,
        "correlacao_retornos_mongo_tiingo_total_causal": correlacao_total,
        "erro_medio_abs_retorno_raw_bps": float(
            comparacao["dif_retorno_raw"].abs().dropna().mean() * 10_000
        ),
        "erro_medio_abs_retorno_desdobramentos_bps": erro_desdobramentos,
        "erro_medio_abs_retorno_total_causal_bps": erro_total,
        "reducao_erro_total_causal_bps": erro_desdobramentos - erro_total,
        "reducao_erro_total_causal_pct": (
            1.0 - erro_total / erro_desdobramentos
            if erro_desdobramentos > 0
            else 0.0
        ),
        "dias_dif_retorno_total_causal_maior_0_1pct": int(
            (comparacao["dif_retorno_total_causal"].abs() > 0.001).sum()
        ),
        "dias_dif_retorno_total_causal_maior_0_5pct": int(
            (comparacao["dif_retorno_total_causal"].abs() > 0.005).sum()
        ),
        "dias_dif_retorno_total_causal_maior_1pct": int(
            (comparacao["dif_retorno_total_causal"].abs() > 0.01).sum()
        ),
    }
    resumos.append(resumo)

    detalhe = comparacao.reset_index()
    detalhe.insert(0, "ativo", ativo)
    detalhes.append(detalhe)

    registrar(
        f"{posicao:02d}/{len(ATIVOS)} {ativo} | "
        f"Mongo={len(mongo)} | Tiingo={len(tiingo_raw)} | comuns={len(datas)} | "
        f"erro splits={erro_desdobramentos:.2f} bps -> "
        f"splits+div={erro_total:.2f} bps | "
        f"1a divergencia retorno={primeira_divergencia or 'nenhuma'}"
    )


# %% 3 - Artefatos de auditoria
resumo_df = pd.DataFrame(resumos)
detalhe_df = pd.concat(detalhes, ignore_index=True)

resumo_df = resumo_df.sort_values(
    ["erro_medio_abs_retorno_total_causal_bps", "ativo"],
    ascending=[False, True],
)
detalhe_df["abs_dif_retorno_total_causal"] = detalhe_df[
    "dif_retorno_total_causal"
].abs()

resumo_df.to_csv(
    DIRETORIO_RESULTADOS / "comparacao_mongo_tiingo_resumo.csv",
    index=False,
)
detalhe_df.to_csv(
    DIRETORIO_RESULTADOS / "comparacao_mongo_tiingo_detalhe.csv",
    index=False,
)

top_divergencias = detalhe_df.sort_values(
    ["abs_dif_retorno_total_causal", "ativo", "timestamp"],
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
