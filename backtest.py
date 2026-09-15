"""Execução acadêmica reproduzível para o TCC MBA USP.

A entrada externa é o histórico diário OHLCV obtido do Yahoo Finance. O motor
reconstrói atributos, alvos, janelas walk-forward, modelos LightGBM, decisões,
operações, curva de capital e métricas a cada execução.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from tcc_engine.configuracao import ATIVOS, CONFIGURACAO, DATA_FIM, DATA_INICIO
from tcc_engine.execucao import aplicar_deslizamento, calcular_taxas_referencia
from tcc_engine.rotacao_capital import executar_modelos_rotacao

RAIZ_PROJETO = Path(__file__).resolve().parent
DIRETORIO_SAIDA = RAIZ_PROJETO / "output"
INTERVALO_YAHOO = "1d"
AJUSTE_AUTOMATICO_YAHOO = True


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


def baixar_dados_mercado() -> dict[str, pd.DataFrame]:
    """Baixa e valida as séries OHLCV usadas pelo experimento."""
    data_fim_yahoo = (pd.Timestamp(DATA_FIM) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    quadros: dict[str, pd.DataFrame] = {}

    for posicao, ativo in enumerate(ATIVOS, start=1):
        print(f"[{posicao:02d}/{len(ATIVOS)}] Yahoo Finance: {ativo}")
        quadro = yf.download(
            ativo,
            start=DATA_INICIO,
            end=data_fim_yahoo,
            interval=INTERVALO_YAHOO,
            auto_adjust=AJUSTE_AUTOMATICO_YAHOO,
            actions=False,
            progress=False,
            threads=False,
            repair=False,
            keepna=False,
            multi_level_index=False,
        )
        if quadro is None or quadro.empty:
            raise RuntimeError(f"Yahoo Finance não retornou histórico para {ativo}.")

        quadro = quadro.rename(columns={str(c): str(c).lower() for c in quadro.columns})
        colunas = ["open", "high", "low", "close", "volume"]
        ausentes = [coluna for coluna in colunas if coluna not in quadro.columns]
        if ausentes:
            raise RuntimeError(f"{ativo}: colunas ausentes: {', '.join(ausentes)}")

        quadro = quadro[colunas].copy()
        quadro.index = pd.to_datetime(quadro.index)
        quadro.index = (
            quadro.index.tz_localize("UTC")
            if quadro.index.tz is None
            else quadro.index.tz_convert("UTC")
        )
        quadro.index.name = "timestamp"
        quadro = quadro.sort_index()
        quadro = quadro[~quadro.index.duplicated(keep="last")]
        for coluna in colunas:
            quadro[coluna] = pd.to_numeric(quadro[coluna], errors="coerce")
        quadro = quadro.dropna(subset=colunas)
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
        print(f"    {len(quadro)} sessões | {quadro.index.min().date()} -> {quadro.index.max().date()}")
    return quadros


def salvar_snapshot_mercado(quadros: dict[str, pd.DataFrame]) -> tuple[int, str]:
    """Persiste o conjunto bruto utilizado e retorna linhas e SHA-256."""
    partes: list[pd.DataFrame] = []
    for ativo in ATIVOS:
        parte = quadros[ativo].reset_index().copy()
        parte.insert(0, "symbol", ativo)
        partes.append(parte)
    dados = pd.concat(partes, ignore_index=True)
    dados = dados[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]
    csv = dados.to_csv(index=False)
    DIRETORIO_SAIDA.mkdir(parents=True, exist_ok=True)
    (DIRETORIO_SAIDA / "market_data.csv").write_text(csv, encoding="utf-8")
    resumo = hashlib.sha256(csv.encode("utf-8")).hexdigest()
    return len(dados), resumo


def salvar_resultados(resultado: Any, contexto: dict[str, Any], tempo: float) -> None:
    """Grava os artefatos canônicos do backtest."""
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
            if tabela[coluna].map(lambda v: isinstance(v, (dict, list, tuple))).any():
                tabela[coluna] = tabela[coluna].map(
                    lambda v: json.dumps(v, ensure_ascii=False, default=serializar_json)
                    if isinstance(v, (dict, list, tuple))
                    else v
                )

    curva_csv.to_csv(DIRETORIO_SAIDA / "equity_curve.csv", index=False)
    operacoes_csv.to_csv(DIRETORIO_SAIDA / "trades.csv", index=False)
    pd.DataFrame(janelas).to_csv(DIRETORIO_SAIDA / "folds.csv", index=False)

    dados_resultado = {
        "experiment": "mba_usp_yahoo_backtest",
        "input_source": "yahoo_finance_ohlcv",
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
    """Executa todas as etapas do experimento."""
    inicio = time.perf_counter()
    print("TCC MBA USP — execução reconstruída passo a passo")
    print("Fonte de mercado: Yahoo Finance por meio de yfinance")
    print(f"Período solicitado: {DATA_INICIO} -> {DATA_FIM} | ativos={len(ATIVOS)}")

    quadros = baixar_dados_mercado()
    linhas_snapshot, sha_snapshot = salvar_snapshot_mercado(quadros)
    print(f"Snapshot: {linhas_snapshot:,} linhas")
    print(f"SHA-256: {sha_snapshot}")
    print("Construindo atributos, alvos e janelas walk-forward")
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
            "provider": "yahoo_finance",
            "library": "yfinance",
            "interval": INTERVALO_YAHOO,
            "auto_adjust": AJUSTE_AUTOMATICO_YAHOO,
            "snapshot_file": "market_data.csv",
            "snapshot_rows": linhas_snapshot,
            "snapshot_sha256": sha_snapshot,
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
