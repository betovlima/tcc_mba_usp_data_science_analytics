"""Execução acadêmica reproduzível para o TCC MBA USP.

A entrada do experimento é um arquivo CSV com o histórico diário OHLCV
certificado da Alpaca. O motor reconstrói atributos, alvos, janelas de
validação temporal, modelos LightGBM, decisões, operações, curva de capital e
métricas a cada execução.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tcc_engine.configuracao import ATIVOS, CONFIGURACAO, DATA_FIM, DATA_INICIO
from tcc_engine.execucao import aplicar_deslizamento, calcular_taxas_referencia
from tcc_engine.rotacao_capital import executar_modelos_rotacao

RAIZ_PROJETO = Path(__file__).resolve().parent
DIRETORIO_DADOS = RAIZ_PROJETO / "dados"
ARQUIVO_DADOS_CERTIFICADOS = DIRETORIO_DADOS / "mercado_certificado.csv"
DIRETORIO_SAIDA = RAIZ_PROJETO / "output"
COLUNAS_MERCADO = ["open", "high", "low", "close", "volume"]


def serializar_json(valor: Any) -> Any:
    """Converte objetos NumPy, pandas e Path para tipos compatíveis com JSON."""
    if isinstance(valor, (pd.Timestamp, datetime)):
        return pd.Timestamp(valor).isoformat()
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, np.floating):
        numero = float(valor)
        return numero if np.isfinite(numero) else None
    if isinstance(valor, np.bool_):
        return bool(valor)
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    if isinstance(valor, Path):
        return str(valor)
    return str(valor)


def carregar_dados_mercado() -> dict[str, pd.DataFrame]:
    """Carrega do CSV as séries OHLCV usadas pelo experimento certificado."""
    if not ARQUIVO_DADOS_CERTIFICADOS.exists():
        raise FileNotFoundError(
            "O arquivo dados/mercado_certificado.csv ainda não existe. "
            "Execute primeiro: python exportar_dados_certificados.py"
        )

    dados = pd.read_csv(ARQUIVO_DADOS_CERTIFICADOS)
    colunas_necessarias = ["symbol", "timestamp", *COLUNAS_MERCADO]
    ausentes = [coluna for coluna in colunas_necessarias if coluna not in dados.columns]
    if ausentes:
        raise RuntimeError(
            "O arquivo de mercado não possui as colunas necessárias: "
            + ", ".join(ausentes)
        )

    dados = dados[colunas_necessarias].copy()
    dados["symbol"] = dados["symbol"].astype(str).str.upper().str.strip()
    dados["timestamp"] = pd.to_datetime(dados["timestamp"], utc=True)

    inicio = pd.Timestamp(DATA_INICIO, tz="UTC")
    fim = pd.Timestamp(DATA_FIM, tz="UTC") + pd.Timedelta(days=1)
    dados = dados.loc[
        dados["symbol"].isin(ATIVOS)
        & dados["timestamp"].ge(inicio)
        & dados["timestamp"].lt(fim)
    ].copy()

    quadros: dict[str, pd.DataFrame] = {}
    for posicao, ativo in enumerate(ATIVOS, start=1):
        quadro = dados.loc[dados["symbol"].eq(ativo), ["timestamp", *COLUNAS_MERCADO]].copy()
        if quadro.empty:
            raise RuntimeError(f"Não existem dados certificados para {ativo}.")

        quadro = quadro.set_index("timestamp").sort_index()
        quadro = quadro[~quadro.index.duplicated(keep="last")]
        for coluna in COLUNAS_MERCADO:
            quadro[coluna] = pd.to_numeric(quadro[coluna], errors="coerce")
        quadro = quadro.dropna(subset=COLUNAS_MERCADO)

        validas = (
            (quadro["open"] > 0)
            & (quadro["high"] > 0)
            & (quadro["low"] > 0)
            & (quadro["close"] > 0)
            & (quadro["volume"] >= 0)
        )
        quadro = quadro.loc[validas].copy()
        if quadro.empty:
            raise RuntimeError(f"{ativo}: histórico vazio após validação.")

        quadros[ativo] = quadro
        print(
            f"[{posicao:02d}/{len(ATIVOS)}] {ativo}: "
            f"{len(quadro)} sessões | "
            f"{quadro.index.min().date()} -> {quadro.index.max().date()}"
        )

    return quadros


def salvar_dados_mercado(quadros: dict[str, pd.DataFrame]) -> int:
    """Salva em output o mesmo conjunto de dados usado pelo backtest e pelo Excel."""
    partes: list[pd.DataFrame] = []
    for ativo in ATIVOS:
        parte = quadros[ativo].reset_index().copy()
        parte.insert(0, "symbol", ativo)
        partes.append(parte)

    dados = pd.concat(partes, ignore_index=True)
    dados = dados[["symbol", "timestamp", *COLUNAS_MERCADO]]

    DIRETORIO_SAIDA.mkdir(parents=True, exist_ok=True)
    dados.to_csv(DIRETORIO_SAIDA / "market_data.csv", index=False)
    return len(dados)


def salvar_resultados(resultado: Any, contexto: dict[str, Any], tempo: float) -> None:
    """Grava os artefatos do backtest."""
    previsoes = resultado.predictions.copy()
    operacoes = resultado.trades.copy()
    janelas = list(resultado.metrics.get("walk_forward_folds") or [])
    metricas = dict(resultado.metrics)

    colunas_curva = [
        coluna
        for coluna in (
            "strategy_equity",
            "buy_hold_equity",
            "selected_asset",
            "decision_score",
            "trade_action",
            "trade_reason",
            "walk_forward_fold",
        )
        if coluna in previsoes.columns
    ]
    curva = previsoes[colunas_curva].copy() if colunas_curva else previsoes.copy()

    curva_csv = curva.reset_index()
    operacoes_csv = operacoes.reset_index() if operacoes.index.name is not None else operacoes.copy()
    for tabela in (curva_csv, operacoes_csv):
        for coluna in tabela.columns:
            if tabela[coluna].map(lambda valor: isinstance(valor, (dict, list, tuple))).any():
                tabela[coluna] = tabela[coluna].map(
                    lambda valor: json.dumps(valor, ensure_ascii=False, default=serializar_json)
                    if isinstance(valor, (dict, list, tuple))
                    else valor
                )

    curva_csv.to_csv(DIRETORIO_SAIDA / "equity_curve.csv", index=False)
    operacoes_csv.to_csv(DIRETORIO_SAIDA / "trades.csv", index=False)
    pd.DataFrame(janelas).to_csv(DIRETORIO_SAIDA / "folds.csv", index=False)

    dados_resultado = {
        "experiment": "mba_usp_backtest_certificado",
        "input_source": "arquivo_csv_alpaca_certificado",
        **contexto,
        "walk_forward": {
            "minimum_training_rows": CONFIGURACAO.linhas_minimas_treinamento,
            "calibration_days": CONFIGURACAO.dias_calibracao,
            "test_days": CONFIGURACAO.dias_teste,
            "minimum_test_days": CONFIGURACAO.dias_minimos_teste,
            "purge_days": CONFIGURACAO.dias_separacao,
        },
        "elapsed_seconds": float(tempo),
        "metrics": metricas,
    }
    (DIRETORIO_SAIDA / "backtest_result.json").write_text(
        json.dumps(dados_resultado, indent=2, ensure_ascii=False, default=serializar_json) + "\n",
        encoding="utf-8",
    )
    (DIRETORIO_SAIDA / "summary.txt").write_text(
        str(resultado.summary).rstrip() + "\n",
        encoding="utf-8",
    )


def principal() -> None:
    """Executa todas as etapas do experimento certificado."""
    inicio = time.perf_counter()
    print("TCC MBA USP — execução reconstruída passo a passo")
    print("Fonte de mercado: arquivo CSV certificado da Alpaca")
    print(f"Período solicitado: {DATA_INICIO} -> {DATA_FIM} | ativos={len(ATIVOS)}")

    quadros = carregar_dados_mercado()
    quantidade_linhas = salvar_dados_mercado(quadros)
    print(f"Dados de mercado carregados: {quantidade_linhas:,} linhas")
    print("Construindo atributos, alvos e janelas de validação temporal")
    print("Treinando LightGBM do zero em cada janela e ativo")

    resultados = executar_modelos_rotacao(
        quadros,
        CONFIGURACAO,
        calcular_taxas_referencia,
        aplicar_deslizamento,
    )
    if len(resultados) != 1:
        raise RuntimeError(f"Era esperada uma execução; foram retornadas {len(resultados)}.")
    resultado = resultados[0]
    tempo = time.perf_counter() - inicio

    contexto = {
        "assets": list(ATIVOS),
        "asset_count": len(ATIVOS),
        "history_start": DATA_INICIO,
        "history_end": DATA_FIM,
        "market_data": {
            "provider": "alpaca",
            "feed": "sip",
            "adjustment": "all",
            "file": "dados/mercado_certificado.csv",
            "rows": quantidade_linhas,
        },
        "model_family": "lightgbm_utility",
        "target_horizons": list(CONFIGURACAO.horizontes_alvo),
    }
    salvar_resultados(resultado, contexto, tempo)

    metricas = resultado.metrics
    print(f"Capital inicial : US$ {CONFIGURACAO.capital_inicial:,.2f}")
    print(f"Capital final   : US$ {float(metricas['strategy_ending_capital']):,.2f}")
    print(f"Retorno         : {float(metricas['strategy_return']):.2%}")
    print(f"CAGR            : {float(metricas['strategy_cagr']):.2%}")
    print(f"Sharpe          : {float(metricas['strategy_sharpe']):.3f}")
    print(f"Queda máxima    : {float(metricas['strategy_maximum_drawdown']):.2%}")
    print(f"Rotações        : {int(metricas.get('capital_rotations') or 0)}")
    print(f"Dias em caixa   : {int(metricas.get('cash_days') or 0)}")
    print(f"Tempo total     : {tempo:.2f}s")
    print(f"Resultados      : {DIRETORIO_SAIDA}")


if __name__ == "__main__":
    principal()
