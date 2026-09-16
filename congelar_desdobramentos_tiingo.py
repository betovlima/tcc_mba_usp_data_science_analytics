"""Extrai desdobramentos a partir do snapshot Tiingo EOD ja congelado.

O endpoint especifico de Corporate Actions da Tiingo pode exigir acesso beta e
retornar HTTP 403. Para nao depender desse endpoint, este script trabalha
somente com os arquivos locais ja congelados.

O campo ``splitFactor`` do EOD nao e aplicado cegamente, porque a documentacao
da Tiingo informa que ele tambem pode aparecer em distribuicoes. Cada evento
candidato e validado contra a ruptura mecanica observada entre o fechamento da
sessao anterior e a abertura da data do evento.

Nenhum preco bruto e alterado neste script. O resultado e uma lista separada de
desdobramentos aceita pelo backtest para normalizacao causal em memoria.
"""

# %% 0 - Imports e configuracao
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tcc_engine.config import ASSETS as ATIVOS
from tcc_engine.config import END_DATE as DATA_FIM
from tcc_engine.config import START_DATE as DATA_INICIO

RAIZ_PROJETO = Path(__file__).resolve().parent
DIRETORIO_SERIES = RAIZ_PROJETO / "dados" / "series_historicas"
DIRETORIO_EVENTOS = RAIZ_PROJETO / "dados" / "eventos_corporativos"
DIRETORIO_DESDOBRAMENTOS = RAIZ_PROJETO / "dados" / "desdobramentos"
ARQUIVO_MANIFESTO = RAIZ_PROJETO / "dados" / "manifesto_desdobramentos_tiingo.json"

COLUNAS_DESDOBRAMENTOS = [
    "timestamp",
    "split_de",
    "split_para",
    "fator_split",
    "status",
]

# Um desdobramento real deve produzir uma mudanca mecanica de escala muito
# proxima do fator informado. A tolerancia existe apenas para permitir o gap
# normal de mercado entre o fechamento anterior e a abertura da sessao.
TOLERANCIA_RELATIVA_ABERTURA = 0.10


def registrar(mensagem: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {mensagem}", flush=True)


def ler_serie_bruta(ativo: str) -> pd.DataFrame:
    arquivo = DIRETORIO_SERIES / f"{ativo}.csv"
    if not arquivo.exists():
        raise RuntimeError(
            f"{ativo}: serie RAW nao encontrada. "
            "Nao baixe novamente; restaure o snapshot congelado em dados/series_historicas."
        )

    tabela = pd.read_csv(arquivo)
    obrigatorias = ["timestamp", "open", "high", "low", "close", "volume"]
    ausentes = [coluna for coluna in obrigatorias if coluna not in tabela.columns]
    if ausentes:
        raise RuntimeError(
            f"{ativo}: serie RAW invalida; colunas ausentes: " + ", ".join(ausentes)
        )

    tabela = tabela[obrigatorias].copy()
    tabela["timestamp"] = pd.to_datetime(
        tabela["timestamp"], utc=True, errors="coerce"
    )
    for coluna in ("open", "high", "low", "close", "volume"):
        tabela[coluna] = pd.to_numeric(tabela[coluna], errors="coerce")

    tabela = tabela.dropna(subset=obrigatorias)
    tabela = tabela.sort_values("timestamp")
    tabela = tabela.drop_duplicates(subset=["timestamp"], keep="last")
    return tabela.reset_index(drop=True)


def ler_eventos_eod(ativo: str) -> pd.DataFrame:
    arquivo = DIRETORIO_EVENTOS / f"{ativo}.csv"
    if not arquivo.exists():
        raise RuntimeError(
            f"{ativo}: eventos EOD nao encontrados. "
            "O snapshot congelado precisa conter dados/eventos_corporativos."
        )

    tabela = pd.read_csv(arquivo)
    obrigatorias = ["timestamp", "dividendo", "fator_split"]
    ausentes = [coluna for coluna in obrigatorias if coluna not in tabela.columns]
    if ausentes:
        raise RuntimeError(
            f"{ativo}: eventos EOD invalidos; colunas ausentes: "
            + ", ".join(ausentes)
        )

    if tabela.empty:
        return pd.DataFrame(columns=obrigatorias)

    tabela = tabela[obrigatorias].copy()
    tabela["timestamp"] = pd.to_datetime(
        tabela["timestamp"], utc=True, errors="coerce"
    )
    tabela["dividendo"] = pd.to_numeric(
        tabela["dividendo"], errors="coerce"
    ).fillna(0.0)
    tabela["fator_split"] = pd.to_numeric(
        tabela["fator_split"], errors="coerce"
    ).fillna(1.0)
    tabela = tabela.dropna(subset=["timestamp"])
    tabela = tabela.sort_values("timestamp")
    tabela = tabela.drop_duplicates(subset=["timestamp"], keep="last")
    return tabela.reset_index(drop=True)


def decompor_fator(fator: float) -> tuple[float, float]:
    if fator >= 1.0:
        return 1.0, fator
    return 1.0 / fator, 1.0


# %% 1 - Validacao da materia-prima local
if not DIRETORIO_SERIES.exists():
    raise RuntimeError(
        "Diretorio dados/series_historicas nao encontrado. "
        "Use o snapshot Tiingo RAW que ja foi congelado."
    )
if not DIRETORIO_EVENTOS.exists():
    raise RuntimeError(
        "Diretorio dados/eventos_corporativos nao encontrado. "
        "Use o snapshot Tiingo que ja foi congelado."
    )

DIRETORIO_DESDOBRAMENTOS.mkdir(parents=True, exist_ok=True)

registrar("Extraindo desdobramentos do snapshot Tiingo EOD congelado")
registrar("Nenhuma chamada a API sera realizada")
registrar(f"Periodo: {DATA_INICIO} -> {DATA_FIM} | ativos={len(ATIVOS)}")
registrar(
    "Criterio: splitFactor candidato precisa explicar a mudanca de escala "
    f"no open com desvio relativo <= {TOLERANCIA_RELATIVA_ABERTURA:.0%}"
)

inicio = pd.Timestamp(DATA_INICIO, tz="UTC")
fim = pd.Timestamp(DATA_FIM, tz="UTC")
resumo_ativos: list[dict[str, object]] = []
candidatos_rejeitados: list[dict[str, object]] = []
total_desdobramentos = 0
total_candidatos = 0


# %% 2 - Classificacao dos eventos splitFactor do EOD
for posicao, ativo in enumerate(ATIVOS, start=1):
    serie = ler_serie_bruta(ativo)
    eventos = ler_eventos_eod(ativo)

    candidatos = eventos.loc[
        (eventos["timestamp"] >= inicio)
        & (eventos["timestamp"] <= fim)
        & (~np.isclose(eventos["fator_split"].astype(float), 1.0)),
        ["timestamp", "dividendo", "fator_split"],
    ].copy()

    linhas_aceitas: list[dict[str, object]] = []
    diagnosticos_ativo: list[dict[str, object]] = []

    for candidato in candidatos.itertuples(index=False):
        total_candidatos += 1
        data_evento = pd.Timestamp(candidato.timestamp).normalize()
        fator = float(candidato.fator_split)

        indices = serie.index[
            serie["timestamp"].dt.normalize() == data_evento
        ].tolist()

        diagnostico: dict[str, object] = {
            "ativo": ativo,
            "data": data_evento.date().isoformat(),
            "fator_split_eod": fator,
            "dividendo_eod": float(candidato.dividendo),
        }

        if len(indices) != 1:
            diagnostico.update(
                {
                    "aceito": False,
                    "motivo": "data_sem_sessao_unica",
                }
            )
            candidatos_rejeitados.append(diagnostico)
            diagnosticos_ativo.append(diagnostico)
            continue

        indice = int(indices[0])
        if indice == 0:
            diagnostico.update(
                {
                    "aceito": False,
                    "motivo": "sem_sessao_anterior",
                }
            )
            candidatos_rejeitados.append(diagnostico)
            diagnosticos_ativo.append(diagnostico)
            continue

        fechamento_anterior = float(serie.loc[indice - 1, "close"])
        abertura_evento = float(serie.loc[indice, "open"])
        fechamento_evento = float(serie.loc[indice, "close"])

        if not all(
            np.isfinite(valor) and valor > 0
            for valor in (fechamento_anterior, abertura_evento, fechamento_evento, fator)
        ):
            diagnostico.update(
                {
                    "aceito": False,
                    "motivo": "preco_ou_fator_invalido",
                }
            )
            candidatos_rejeitados.append(diagnostico)
            diagnosticos_ativo.append(diagnostico)
            continue

        razao_observada_abertura = fechamento_anterior / abertura_evento
        razao_observada_fechamento = fechamento_anterior / fechamento_evento
        desvio_relativo_abertura = abs(razao_observada_abertura / fator - 1.0)

        aceito = desvio_relativo_abertura <= TOLERANCIA_RELATIVA_ABERTURA
        diagnostico.update(
            {
                "fechamento_anterior": fechamento_anterior,
                "abertura_evento": abertura_evento,
                "fechamento_evento": fechamento_evento,
                "razao_observada_abertura": razao_observada_abertura,
                "razao_observada_fechamento": razao_observada_fechamento,
                "desvio_relativo_abertura": desvio_relativo_abertura,
                "aceito": bool(aceito),
                "motivo": (
                    "ruptura_compativel_com_desdobramento"
                    if aceito
                    else "fator_nao_explica_ruptura_de_preco"
                ),
            }
        )
        diagnosticos_ativo.append(diagnostico)

        if not aceito:
            candidatos_rejeitados.append(diagnostico)
            continue

        split_de, split_para = decompor_fator(fator)
        linhas_aceitas.append(
            {
                "timestamp": pd.Timestamp(candidato.timestamp),
                "split_de": float(split_de),
                "split_para": float(split_para),
                "fator_split": fator,
                "status": "a",
            }
        )

    tabela_saida = pd.DataFrame(
        linhas_aceitas,
        columns=COLUNAS_DESDOBRAMENTOS,
    )
    if not tabela_saida.empty:
        tabela_saida = tabela_saida.sort_values("timestamp")
        tabela_saida = tabela_saida.drop_duplicates(
            subset=["timestamp", "fator_split"], keep="last"
        )

    tabela_saida.to_csv(
        DIRETORIO_DESDOBRAMENTOS / f"{ativo}.csv",
        index=False,
    )

    total_desdobramentos += len(tabela_saida)
    resumo_ativos.append(
        {
            "ativo": ativo,
            "candidatos_eod": len(candidatos),
            "desdobramentos_aceitos": len(tabela_saida),
            "diagnosticos": diagnosticos_ativo,
        }
    )

    registrar(
        f"{posicao:02d}/{len(ATIVOS)} {ativo} | "
        f"candidatos={len(candidatos)} | aceitos={len(tabela_saida)}"
    )


# %% 3 - Manifesto auditavel
manifesto = {
    "fonte_precos": "Tiingo EOD snapshot local",
    "fonte_candidatos": "splitFactor do Tiingo EOD snapshot local",
    "endpoint_corporate_actions_usado": False,
    "motivo_sem_endpoint_corporate_actions": (
        "O endpoint especifico pode exigir habilitacao beta e retornar HTTP 403."
    ),
    "metodo": "validacao_de_ruptura_mecanica_no_open",
    "tolerancia_relativa_abertura": TOLERANCIA_RELATIVA_ABERTURA,
    "data_geracao_utc": datetime.now(timezone.utc).isoformat(),
    "periodo_inicio": DATA_INICIO,
    "periodo_fim": DATA_FIM,
    "quantidade_ativos": len(ATIVOS),
    "total_candidatos_eod": total_candidatos,
    "total_desdobramentos_aceitos": total_desdobramentos,
    "total_candidatos_rejeitados": len(candidatos_rejeitados),
    "candidatos_rejeitados": candidatos_rejeitados,
    "resumo_ativos": resumo_ativos,
}

ARQUIVO_MANIFESTO.write_text(
    json.dumps(manifesto, indent=2, ensure_ascii=False, default=str) + "\n",
    encoding="utf-8",
)

registrar(
    f"Concluido: candidatos={total_candidatos} | "
    f"desdobramentos aceitos={total_desdobramentos} | "
    f"rejeitados={len(candidatos_rejeitados)}"
)
for rejeitado in candidatos_rejeitados:
    registrar(
        "Rejeitado: "
        f"{rejeitado['ativo']} {rejeitado['data']} "
        f"fator={rejeitado['fator_split_eod']} "
        f"motivo={rejeitado['motivo']}"
    )
registrar(f"Diretorio: {DIRETORIO_DESDOBRAMENTOS}")
registrar("Agora execute: python backtest.py")
