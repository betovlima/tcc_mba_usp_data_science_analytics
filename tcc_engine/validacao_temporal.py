"""Construção das janelas walk-forward do experimento."""
from __future__ import annotations

from typing import Any

import pandas as pd


def construir_janelas_walk_forward(
    datas_comuns: pd.DatetimeIndex,
    configuracao: Any,
) -> list[dict[str, Any]]:
    """Cria janelas expansivas com calibração, separação temporal e teste."""
    separacao = max(int(configuracao.dias_separacao), max(int(v) for v in configuracao.horizontes_alvo))
    dias_calibracao = int(configuracao.dias_calibracao)
    dias_teste = int(configuracao.dias_teste)
    dias_minimos_teste = int(configuracao.dias_minimos_teste)
    minimo_treinamento = int(configuracao.linhas_minimas_treinamento)
    primeiro_teste = minimo_treinamento + separacao + dias_calibracao + separacao

    if primeiro_teste >= len(datas_comuns) - dias_minimos_teste:
        disponiveis = max(0, len(datas_comuns) - primeiro_teste)
        raise ValueError(
            "Histórico insuficiente para o protocolo walk-forward: "
            f"linhas_teste_disponiveis={disponiveis}, mínimo={dias_minimos_teste}."
        )

    intervalos: list[tuple[int, int]] = []
    quantidade_forcada = configuracao.numero_janelas_forcado
    if quantidade_forcada is not None:
        quantidade = int(quantidade_forcada)
        disponiveis = int(len(datas_comuns) - primeiro_teste)
        if disponiveis < quantidade * dias_minimos_teste:
            raise ValueError("Histórico insuficiente para o número de janelas solicitado.")
        tamanho_base, resto = divmod(disponiveis, quantidade)
        cursor = primeiro_teste
        for indice in range(quantidade):
            tamanho = tamanho_base + (1 if indice < resto else 0)
            fim = cursor + tamanho
            intervalos.append((cursor, fim))
            cursor = fim
    else:
        inicio = primeiro_teste
        while inicio < len(datas_comuns):
            fim = min(len(datas_comuns), inicio + dias_teste)
            if fim - inicio < dias_minimos_teste:
                if intervalos:
                    inicio_anterior, _ = intervalos[-1]
                    intervalos[-1] = (inicio_anterior, len(datas_comuns))
                break
            intervalos.append((inicio, fim))
            inicio = fim

    janelas: list[dict[str, Any]] = []
    for identificador, (inicio_teste, fim_teste) in enumerate(intervalos, start=1):
        fim_calibracao = inicio_teste - separacao
        inicio_calibracao = fim_calibracao - dias_calibracao
        fim_treinamento = inicio_calibracao - separacao
        fim_ajuste_final = inicio_teste - separacao
        if fim_treinamento < minimo_treinamento:
            raise ValueError(
                f"Janela {identificador}: apenas {fim_treinamento} linhas de treinamento; "
                f"mínimo={minimo_treinamento}."
            )
        janelas.append(
            {
                "janela_id": identificador,
                "fim_treinamento_indice": fim_treinamento,
                "inicio_calibracao_indice": inicio_calibracao,
                "fim_calibracao_indice": fim_calibracao,
                "fim_ajuste_final_indice": fim_ajuste_final,
                "inicio_teste_indice": inicio_teste,
                "fim_teste_indice": fim_teste,
                "inicio_treinamento": datas_comuns[0],
                "fim_treinamento": datas_comuns[fim_treinamento - 1],
                "inicio_calibracao": datas_comuns[inicio_calibracao],
                "fim_calibracao": datas_comuns[fim_calibracao - 1],
                "inicio_separacao": datas_comuns[fim_calibracao],
                "fim_separacao": datas_comuns[inicio_teste - 1],
                "inicio_teste": datas_comuns[inicio_teste],
                "fim_teste": datas_comuns[fim_teste - 1],
                "datas_decisao": datas_comuns[inicio_teste - 1:fim_teste],
            }
        )
    if not janelas:
        raise ValueError("Nenhuma janela walk-forward válida foi criada.")
    return janelas


def datas_decisao_analise(
    datas_comuns: pd.DatetimeIndex,
    janelas: list[dict[str, Any]],
) -> pd.DatetimeIndex:
    """Retorna o intervalo OOS contínuo coberto pelas janelas."""
    if not janelas:
        raise ValueError("Não existem janelas de validação disponíveis.")
    inicio = int(janelas[0]["inicio_teste_indice"])
    fim = int(janelas[-1]["fim_teste_indice"])
    return datas_comuns[inicio - 1:fim]


def mapear_datas_para_janelas(
    janelas: list[dict[str, Any]],
) -> tuple[dict[pd.Timestamp, int], dict[pd.Timestamp, dict[str, Any]]]:
    """Associa cada data de decisão à janela responsável por ela."""
    mapa: dict[pd.Timestamp, int] = {}
    metadados: dict[pd.Timestamp, dict[str, Any]] = {}
    for janela in janelas:
        for data in janela["datas_decisao"][:-1]:
            chave = pd.Timestamp(data)
            identificador = int(janela["janela_id"])
            mapa[chave] = identificador
            metadados[chave] = {
                "janela_id": identificador,
                "inicio_teste": janela["inicio_teste"],
                "fim_teste": janela["fim_teste"],
            }
    return mapa, metadados
