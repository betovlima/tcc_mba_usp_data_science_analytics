"""Simulação causal da estratégia e cálculo das métricas financeiras."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from .diagnosticos_rotacao import enriquecer_diagnosticos_operacoes
from .metricas import cagr, drawdown_maximo, retorno_geometrico_operacoes, sharpe_anualizado


@dataclass
class ResultadoRotacao:
    """Resultado de uma execução fora da amostra da estratégia."""

    backend: str
    predictions: pd.DataFrame
    trades: pd.DataFrame
    summary: str
    metrics: dict[str, Any]


def _custo_proporcional_troca(configuracao: Any, origem: int, destino: int) -> float:
    if origem == destino:
        return 0.0
    um_lado = max(0.0, float(configuracao.deslizamento_bps)) / 10_000.0
    um_lado += max(0.0, float(configuracao.comissao))
    lados = int(origem != 0) + int(destino != 0)
    return min(0.25, um_lado * lados)


def retorno_log_transicao(
    quadros: dict[str, pd.DataFrame],
    ativos: list[str],
    data_atual: pd.Timestamp,
    proxima_data: pd.Timestamp,
    origem: int,
    destino: int,
    configuracao: Any,
) -> float:
    """Retorno logarítmico causal usado na calibração da política."""
    bruto = 1.0
    if origem > 0:
        ativo = ativos[origem - 1]
        fechamento_atual = float(quadros[ativo].loc[data_atual, "close"])
        abertura_seguinte = float(quadros[ativo].loc[proxima_data, "open"])
        if fechamento_atual > 0:
            bruto *= abertura_seguinte / fechamento_atual
    bruto *= max(1e-8, 1.0 - _custo_proporcional_troca(configuracao, origem, destino))
    if destino > 0:
        ativo = ativos[destino - 1]
        abertura = float(quadros[ativo].loc[proxima_data, "open"])
        fechamento = float(quadros[ativo].loc[proxima_data, "close"])
        if abertura > 0:
            bruto *= fechamento / abertura
    return float(np.log(max(bruto, 1e-12)))


def recompensa_ajustada_risco(
    retorno_log: float,
    patrimonio_anterior: float,
    pico_anterior: float,
    configuracao: Any,
) -> tuple[float, float, float]:
    """Aplica as penalidades de baixa e aumento de drawdown."""
    patrimonio_atual = patrimonio_anterior * math.exp(retorno_log)
    pico_atual = max(pico_anterior, patrimonio_atual)
    drawdown_anterior = max(0.0, 1.0 - patrimonio_anterior / max(pico_anterior, 1e-12))
    drawdown_atual = max(0.0, 1.0 - patrimonio_atual / max(pico_atual, 1e-12))
    aumento_drawdown = max(0.0, drawdown_atual - drawdown_anterior)
    baixa = max(0.0, -retorno_log)
    recompensa = (
        retorno_log
        - float(configuracao.penalidade_baixa) * baixa
        - float(configuracao.penalidade_drawdown) * aumento_drawdown
    )
    return float(recompensa), float(patrimonio_atual), float(pico_atual)


def avaliar_politica_calibracao(
    politica: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    quadros: dict[str, pd.DataFrame],
    ativos: list[str],
    datas: pd.DatetimeIndex,
    configuracao: Any,
) -> float:
    """Avalia uma margem candidata apenas no bloco de calibração."""
    if len(datas) < 2:
        return float("-inf")
    patrimonio = 1.0
    pico = 1.0
    posicao = 0
    dias_posicao = 0
    utilidade = 0.0
    for indice in range(len(datas) - 1):
        atual = pd.Timestamp(datas[indice])
        seguinte = pd.Timestamp(datas[indice + 1])
        destino, _ = politica(atual, posicao, dias_posicao)
        retorno = retorno_log_transicao(
            quadros, ativos, atual, seguinte, posicao, destino, configuracao
        )
        recompensa, patrimonio, pico = recompensa_ajustada_risco(
            retorno, patrimonio, pico, configuracao
        )
        utilidade += recompensa
        if destino == posicao:
            dias_posicao = dias_posicao + 1 if destino > 0 else 0
        else:
            posicao = destino
            dias_posicao = 1 if destino > 0 else 0
    return float(utilidade)


def _executar_compra(
    caixa: float,
    preco: float,
    configuracao: Any,
    calcular_taxas: Callable,
    aplicar_deslizamento: Callable,
) -> tuple[float, float, dict[str, float]]:
    preco_execucao = float(aplicar_deslizamento(preco, "BUY", configuracao))
    quantidade = caixa / preco_execucao
    for _ in range(25):
        taxas = calcular_taxas("BUY", quantidade, preco_execucao, configuracao)
        nova_quantidade = max(0.0, (caixa - float(taxas["total_fee"])) / preco_execucao)
        if not bool(configuracao.acoes_fracionarias):
            nova_quantidade = float(math.floor(nova_quantidade))
        if abs(nova_quantidade - quantidade) < 1e-10:
            quantidade = nova_quantidade
            break
        quantidade = nova_quantidade
    taxas = calcular_taxas("BUY", quantidade, preco_execucao, configuracao)
    return float(quantidade), preco_execucao, taxas


def _benchmark_pesos_iguais(
    quadros: dict[str, pd.DataFrame],
    ativos: list[str],
    datas_execucao: pd.DatetimeIndex,
    capital_inicial: float,
    configuracao: Any,
    calcular_taxas: Callable,
    aplicar_deslizamento: Callable,
) -> pd.Series:
    if len(datas_execucao) < 2:
        return pd.Series(dtype=float)
    primeira = pd.Timestamp(datas_execucao[0])
    ultima = pd.Timestamp(datas_execucao[-1])
    elegiveis: list[str] = []
    for ativo in ativos:
        janela = quadros[ativo].reindex(datas_execucao)
        primeira_abertura = float(janela.iloc[0].get("open", float("nan")))
        fechamentos = pd.to_numeric(janela["close"], errors="coerce")
        if (
            np.isfinite(primeira_abertura)
            and primeira_abertura > 0
            and fechamentos.notna().all()
            and (fechamentos > 0).all()
        ):
            elegiveis.append(ativo)
    if not elegiveis:
        raise ValueError("Nenhum ativo possui preços completos para o benchmark.")

    capital_por_ativo = float(capital_inicial) / len(elegiveis)
    quantidades: dict[str, float] = {}
    residual = 0.0
    for ativo in elegiveis:
        preco = float(aplicar_deslizamento(float(quadros[ativo].loc[primeira, "open"]), "BUY", configuracao))
        quantidade = capital_por_ativo / preco
        for _ in range(20):
            taxas = calcular_taxas("BUY", quantidade, preco, configuracao)
            nova = max(0.0, (capital_por_ativo - float(taxas["total_fee"])) / preco)
            if not bool(configuracao.acoes_fracionarias):
                nova = float(math.floor(nova))
            if abs(nova - quantidade) < 1e-10:
                quantidade = nova
                break
            quantidade = nova
        taxas = calcular_taxas("BUY", quantidade, preco, configuracao)
        quantidades[ativo] = quantidade
        residual += capital_por_ativo - (quantidade * preco + float(taxas["total_fee"]))

    valores: list[float] = []
    for data in datas_execucao:
        patrimonio = residual
        for ativo, quantidade in quantidades.items():
            patrimonio += quantidade * float(quadros[ativo].loc[data, "close"])
        valores.append(patrimonio)
    serie = pd.Series(valores, index=datas_execucao, dtype=float)

    caixa_final = residual
    for ativo, quantidade in quantidades.items():
        preco = float(aplicar_deslizamento(float(quadros[ativo].loc[ultima, "close"]), "SELL", configuracao))
        taxas = calcular_taxas("SELL", quantidade, preco, configuracao)
        caixa_final += quantidade * preco - float(taxas["total_fee"])
    serie.iloc[-1] = caixa_final
    return serie


def simular_rotacao(
    politica: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    quadros: dict[str, pd.DataFrame],
    ativos: list[str],
    datas_decisao: pd.DatetimeIndex,
    configuracao: Any,
    calcular_taxas: Callable,
    aplicar_deslizamento: Callable,
    *,
    metadados_decisao: dict[pd.Timestamp, dict[str, Any]] | None = None,
    diagnosticos_decisao: dict[pd.Timestamp, dict[str, Any]] | None = None,
) -> ResultadoRotacao:
    """Executa decisões no pregão seguinte e liquida a posição no final."""
    if len(datas_decisao) < 2:
        raise ValueError("O intervalo final de teste é curto demais.")

    datas_execucao = pd.DatetimeIndex(datas_decisao[1:])
    benchmark = _benchmark_pesos_iguais(
        quadros,
        ativos,
        datas_execucao,
        float(configuracao.capital_inicial),
        configuracao,
        calcular_taxas,
        aplicar_deslizamento,
    )

    caixa = float(configuracao.capital_inicial)
    posicao = 0
    quantidade = 0.0
    preco_entrada = float("nan")
    data_entrada: pd.Timestamp | None = None
    dias_posicao = 0
    taxas_totais = 0.0
    giro_bruto = 0.0
    rotacoes = 0
    registros: list[dict[str, Any]] = []
    linhas_curva: list[dict[str, Any]] = []

    for indice in range(len(datas_decisao) - 1):
        data_decisao = pd.Timestamp(datas_decisao[indice])
        data_execucao = pd.Timestamp(datas_decisao[indice + 1])
        posicao_anterior = posicao
        destino, pontuacao = politica(data_decisao, posicao, dias_posicao)
        metadados = (metadados_decisao or {}).get(data_decisao, {})
        diagnostico = dict((diagnosticos_decisao or {}).get(data_decisao, {}))
        janela_id = metadados.get("janela_id")
        operacoes_dia: list[dict[str, Any]] = []

        if destino != posicao:
            ativo_antigo = ativos[posicao - 1] if posicao > 0 else None
            ativo_novo = ativos[destino - 1] if destino > 0 else None
            identificador_rotacao = (
                f"{data_execucao.isoformat()}::{ativo_antigo or 'CASH'}->{ativo_novo or 'CASH'}"
            )

            if posicao > 0:
                ativo = ativos[posicao - 1]
                preco = float(aplicar_deslizamento(float(quadros[ativo].loc[data_execucao, "open"]), "SELL", configuracao))
                taxas = calcular_taxas("SELL", quantidade, preco, configuracao)
                bruto = quantidade * preco
                pnl = quantidade * (preco - preco_entrada) - float(taxas["total_fee"])
                caixa += bruto - float(taxas["total_fee"])
                taxas_totais += float(taxas["total_fee"])
                giro_bruto += bruto
                retorno_posicao = preco / preco_entrada - 1.0 if np.isfinite(preco_entrada) and preco_entrada > 0 else 0.0
                operacoes_dia.append(
                    {
                        "timestamp": data_execucao,
                        "decision_timestamp": data_decisao,
                        "rotation_id": identificador_rotacao,
                        "rotation_from_asset": ativo_antigo or "CASH",
                        "rotation_to_asset": ativo_novo or "CASH",
                        "action": "SELL",
                        "asset": ativo,
                        "reason": f"ROTATE_TO_{ativo_novo}" if ativo_novo else "MOVE_TO_CASH",
                        "execution_price": preco,
                        "quantity": quantidade,
                        "gross_trade_value": bruto,
                        **taxas,
                        "realized_pnl": pnl,
                        "position_return": retorno_posicao,
                        "holding_bars": dias_posicao,
                        "entry_timestamp": data_entrada,
                        "entry_price": preco_entrada,
                        "cash_after_trade": caixa,
                        "shares_after_trade": 0.0,
                        "walk_forward_fold": janela_id,
                        **diagnostico,
                    }
                )
                quantidade = 0.0
                preco_entrada = float("nan")
                data_entrada = None
                dias_posicao = 0

            posicao = int(destino)
            if posicao > 0:
                ativo = ativos[posicao - 1]
                quantidade, preco, taxas = _executar_compra(
                    caixa,
                    float(quadros[ativo].loc[data_execucao, "open"]),
                    configuracao,
                    calcular_taxas,
                    aplicar_deslizamento,
                )
                bruto = quantidade * preco
                caixa -= bruto + float(taxas["total_fee"])
                taxas_totais += float(taxas["total_fee"])
                giro_bruto += bruto
                preco_entrada = preco
                data_entrada = data_execucao
                dias_posicao = 1
                operacoes_dia.append(
                    {
                        "timestamp": data_execucao,
                        "decision_timestamp": data_decisao,
                        "rotation_id": identificador_rotacao,
                        "rotation_from_asset": ativo_antigo or "CASH",
                        "rotation_to_asset": ativo_novo or "CASH",
                        "action": "BUY",
                        "asset": ativo,
                        "reason": f"ROTATE_FROM_{ativo_antigo}" if ativo_antigo else "BEST_CAPITAL_UTILITY",
                        "execution_price": preco,
                        "quantity": quantidade,
                        "gross_trade_value": bruto,
                        **taxas,
                        "realized_pnl": 0.0,
                        "position_return": 0.0,
                        "holding_bars": 0,
                        "entry_timestamp": data_execucao,
                        "entry_price": preco,
                        "cash_after_trade": caixa,
                        "shares_after_trade": quantidade,
                        "walk_forward_fold": janela_id,
                        **diagnostico,
                    }
                )
            if posicao_anterior > 0 and destino > 0:
                rotacoes += 1
        elif posicao > 0:
            dias_posicao += 1

        registros.extend(operacoes_dia)
        if posicao > 0:
            ativo_selecionado = ativos[posicao - 1]
            patrimonio = caixa + quantidade * float(quadros[ativo_selecionado].loc[data_execucao, "close"])
        else:
            ativo_selecionado = "CASH"
            patrimonio = caixa

        acoes = [operacao["action"] for operacao in operacoes_dia]
        acao_dia = "ROTATE" if "SELL" in acoes and "BUY" in acoes else (acoes[-1] if acoes else "")
        linhas_curva.append(
            {
                "timestamp": data_execucao,
                "strategy_equity": float(patrimonio),
                "buy_hold_equity": float(benchmark.loc[data_execucao]),
                "selected_asset": ativo_selecionado,
                "decision_score": float(pontuacao),
                "trade_action": acao_dia,
                "trade_reason": "COMPOUND_CAPITAL_ROTATION" if acao_dia else "",
                "walk_forward_fold": janela_id,
                **diagnostico,
            }
        )

    if posicao > 0 and linhas_curva:
        data_final = datas_execucao[-1]
        ativo = ativos[posicao - 1]
        preco = float(aplicar_deslizamento(float(quadros[ativo].loc[data_final, "close"]), "SELL", configuracao))
        taxas = calcular_taxas("SELL", quantidade, preco, configuracao)
        bruto = quantidade * preco
        pnl = quantidade * (preco - preco_entrada) - float(taxas["total_fee"])
        caixa += bruto - float(taxas["total_fee"])
        taxas_totais += float(taxas["total_fee"])
        giro_bruto += bruto
        retorno_posicao = preco / preco_entrada - 1.0 if np.isfinite(preco_entrada) and preco_entrada > 0 else 0.0
        registros.append(
            {
                "timestamp": data_final,
                "decision_timestamp": data_final,
                "action": "FINAL_SELL",
                "asset": ativo,
                "reason": "FINAL_LIQUIDATION",
                "execution_price": preco,
                "quantity": quantidade,
                "gross_trade_value": bruto,
                **taxas,
                "realized_pnl": pnl,
                "position_return": retorno_posicao,
                "holding_bars": dias_posicao,
                "entry_timestamp": data_entrada,
                "entry_price": preco_entrada,
                "cash_after_trade": caixa,
                "shares_after_trade": 0.0,
                "walk_forward_fold": linhas_curva[-1].get("walk_forward_fold"),
            }
        )
        linhas_curva[-1]["strategy_equity"] = float(caixa)
        if not linhas_curva[-1]["trade_action"]:
            linhas_curva[-1]["trade_action"] = "FINAL_SELL"
            linhas_curva[-1]["trade_reason"] = "FINAL_LIQUIDATION"

    registros = enriquecer_diagnosticos_operacoes(registros, quadros, ativos)
    previsoes = pd.DataFrame(linhas_curva).set_index("timestamp")
    previsoes.index = pd.to_datetime(previsoes.index, utc=True)
    previsoes.index.name = "timestamp"
    operacoes = pd.DataFrame(registros)
    if not operacoes.empty:
        operacoes["timestamp"] = pd.to_datetime(operacoes["timestamp"], utc=True)
        operacoes = operacoes.sort_values("timestamp").reset_index(drop=True)

    curva = previsoes["strategy_equity"].astype(float)
    curva_benchmark = previsoes["buy_hold_equity"].astype(float)
    inicial = float(configuracao.capital_inicial)
    final = float(curva.iloc[-1])
    final_benchmark = float(curva_benchmark.iloc[-1])
    compras = int((operacoes["action"] == "BUY").sum()) if not operacoes.empty else 0
    vendas = int(operacoes["action"].isin(["SELL", "FINAL_SELL"]).sum()) if not operacoes.empty else 0
    dias_caixa = int((previsoes["selected_asset"] == "CASH").sum())
    exposicao = 1.0 - dias_caixa / max(1, len(previsoes))
    vendas_fechadas = operacoes.loc[operacoes["action"].isin(["SELL", "FINAL_SELL"])] if not operacoes.empty else pd.DataFrame()
    media_dias = float(pd.to_numeric(vendas_fechadas["holding_bars"]).mean()) if not vendas_fechadas.empty else float("nan")
    dias = max(1, (previsoes.index[-1] - previsoes.index[0]).days)
    anos = max(dias / 365.25, 1 / 365.25)

    metricas = {
        "portfolio_rotation": True,
        "strategy_label": "LightGBM Utility",
        "backend": "lightgbm_utility",
        "assets": ativos,
        "timeframe": "1Day",
        "initial_capital": inicial,
        "strategy_ending_capital": final,
        "strategy_return": final / inicial - 1.0,
        "strategy_cagr": cagr(curva, inicial),
        "strategy_sharpe": sharpe_anualizado(curva),
        "strategy_maximum_drawdown": drawdown_maximo(curva),
        "buy_hold_ending_capital": final_benchmark,
        "buy_hold_return": final_benchmark / inicial - 1.0,
        "buy_hold_cagr": cagr(curva_benchmark, inicial),
        "buy_hold_sharpe": sharpe_anualizado(curva_benchmark),
        "buy_hold_maximum_drawdown": drawdown_maximo(curva_benchmark),
        "market_exposure": float(exposicao),
        "cash_days": dias_caixa,
        "simulated_buys": compras,
        "simulated_sells": vendas,
        "capital_rotations": int(rotacoes),
        "cycles_per_year": float(compras / anos),
        "average_holding_days": media_dias,
        "average_holding_bars": media_dias,
        "geometric_trade_return": retorno_geometrico_operacoes(operacoes),
        "total_transaction_fees": float(taxas_totais),
        "turnover_ratio": float(giro_bruto / max(inicial, 1e-9)),
        "test_start": previsoes.index[0],
        "test_end": previsoes.index[-1],
        "test_calendar_years": anos,
    }

    resumo = "\n".join(
        [
            "ROTAÇÃO DE CAPITAL COMPOSTO",
            "",
            "Modelo: LightGBM Utility",
            f"Ativos: {', '.join(ativos)}",
            "Decisões: fechamento diário; execução: abertura do pregão seguinte",
            "",
            f"Capital inicial: US$ {inicial:,.2f}",
            f"Capital final: US$ {final:,.2f}",
            f"Retorno total: {metricas['strategy_return']:.2%}",
            f"CAGR: {metricas['strategy_cagr']:.2%}",
            f"Drawdown máximo: {metricas['strategy_maximum_drawdown']:.2%}",
            f"Sharpe: {metricas['strategy_sharpe']:.3f}",
            f"Rotações: {rotacoes}",
            f"Dias em caixa: {dias_caixa}",
        ]
    )
    return ResultadoRotacao(
        backend="lightgbm_utility",
        predictions=previsoes,
        trades=operacoes,
        summary=resumo,
        metrics=metricas,
    )


def desempenho_por_janela(
    previsoes: pd.DataFrame,
    janelas: list[dict[str, Any]],
    capital_inicial: float,
) -> list[dict[str, Any]]:
    """Resume o desempenho financeiro de cada janela walk-forward."""
    if previsoes.empty:
        return []
    linhas = previsoes.reset_index().sort_values("timestamp").reset_index(drop=True)
    saida: list[dict[str, Any]] = []
    for janela in janelas:
        identificador = int(janela["janela_id"])
        subconjunto = linhas.loc[linhas["walk_forward_fold"] == identificador]
        if subconjunto.empty:
            continue
        primeiro_indice = int(subconjunto.index[0])
        inicio_estrategia = (
            float(capital_inicial)
            if primeiro_indice == 0
            else float(linhas.loc[primeiro_indice - 1, "strategy_equity"])
        )
        inicio_benchmark = (
            float(capital_inicial)
            if primeiro_indice == 0
            else float(linhas.loc[primeiro_indice - 1, "buy_hold_equity"])
        )
        fim_estrategia = float(subconjunto.iloc[-1]["strategy_equity"])
        fim_benchmark = float(subconjunto.iloc[-1]["buy_hold_equity"])
        curva = pd.Series([inicio_estrategia, *subconjunto["strategy_equity"].astype(float).tolist()])
        saida.append(
            {
                "fold_id": identificador,
                "train_end": janela["fim_treinamento"],
                "calibration_start": janela["inicio_calibracao"],
                "calibration_end": janela["fim_calibracao"],
                "purge_start": janela["inicio_separacao"],
                "purge_end": janela["fim_separacao"],
                "model_test_start": janela["inicio_teste"],
                "model_test_end": janela["fim_teste"],
                "test_start": pd.Timestamp(subconjunto.iloc[0]["timestamp"]),
                "test_end": pd.Timestamp(subconjunto.iloc[-1]["timestamp"]),
                "strategy_starting_capital": inicio_estrategia,
                "strategy_ending_capital": fim_estrategia,
                "strategy_return": fim_estrategia / inicio_estrategia - 1.0,
                "benchmark_return": fim_benchmark / inicio_benchmark - 1.0,
                "excess_return": fim_estrategia / inicio_estrategia - fim_benchmark / inicio_benchmark,
                "maximum_drawdown": drawdown_maximo(curva),
                "sessions": int(len(subconjunto)),
            }
        )
    return saida
