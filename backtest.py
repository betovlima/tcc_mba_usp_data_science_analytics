"""Execução acadêmica reproduzível para o TCC MBA USP.

A entrada externa do experimento é o histórico diário OHLCV baixado do Yahoo
Finance por meio da biblioteca yfinance. Atributos técnicos, alvos, janelas de
validação temporal, treinamento LightGBM, política de rotação, operações, curva
de capital e métricas são reconstruídos a cada execução.

O arquivo é organizado em células Spyder ``# %%``. F5 executa o arquivo completo;
Ctrl+Enter executa somente a célula atual e mantém as variáveis disponíveis no
explorador de variáveis.
"""

# %% 0 - Importações e configuração
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

from tcc_engine.capital_rotation import run_rotation_models as executar_modelos_rotacao
from tcc_engine.config import (
    ASSETS as ATIVOS,
    CONFIG as CONFIGURACAO,
    END_DATE as DATA_FIM,
    START_DATE as DATA_INICIO,
)
from tcc_engine.execution import (
    apply_slippage as aplicar_deslizamento,
    calculate_reference_fees as calcular_taxas_referencia,
)

RAIZ_PROJETO = Path(__file__).resolve().parent
DIRETORIO_SAIDA = RAIZ_PROJETO / "output"

INTERVALO_YAHOO = "1d"
AJUSTE_AUTOMATICO_YAHOO = True


def serializar_json(valor: Any) -> Any:
    """Converte valores NumPy, pandas e Path para tipos compatíveis com JSON."""
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


# %% 1 - Início da execução
inicio_execucao = time.perf_counter()

print("TCC MBA USP — execução reconstruída passo a passo")
print("Fonte de mercado: Yahoo Finance por meio de yfinance")
print(f"Período solicitado: {DATA_INICIO} -> {DATA_FIM} | ativos={len(ATIVOS)}")

# No yfinance, o parâmetro `end` não inclui a data final.
data_fim_yahoo = (pd.Timestamp(DATA_FIM) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


# %% 2 - Download e validação das séries temporais OHLCV
quadros_por_ativo: dict[str, pd.DataFrame] = {}

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

    quadro = quadro.rename(
        columns={str(coluna): str(coluna).lower() for coluna in quadro.columns}
    )
    colunas_necessarias = ["open", "high", "low", "close", "volume"]
    colunas_ausentes = [
        coluna for coluna in colunas_necessarias if coluna not in quadro.columns
    ]
    if colunas_ausentes:
        raise RuntimeError(
            f"{ativo}: colunas ausentes no Yahoo Finance: {', '.join(colunas_ausentes)}"
        )

    quadro = quadro[colunas_necessarias].copy()
    quadro.index = pd.to_datetime(quadro.index)
    if quadro.index.tz is None:
        quadro.index = quadro.index.tz_localize("UTC")
    else:
        quadro.index = quadro.index.tz_convert("UTC")
    quadro.index.name = "timestamp"
    quadro = quadro.sort_index()
    quadro = quadro[~quadro.index.duplicated(keep="last")]

    for coluna in colunas_necessarias:
        quadro[coluna] = pd.to_numeric(quadro[coluna], errors="coerce")

    quadro = quadro.dropna(subset=colunas_necessarias)
    linhas_validas = (
        (quadro["open"] > 0)
        & (quadro["high"] > 0)
        & (quadro["low"] > 0)
        & (quadro["close"] > 0)
        & (quadro["volume"] >= 0)
    )
    quadro = quadro.loc[linhas_validas].copy()

    if quadro.empty:
        raise RuntimeError(f"{ativo}: histórico vazio após validação OHLCV.")

    quadros_por_ativo[ativo] = quadro
    print(
        f"    {len(quadro)} sessões | "
        f"{quadro.index.min().date()} -> {quadro.index.max().date()}"
    )

# Snapshot exato das séries usadas pelo motor.
partes_dados_mercado: list[pd.DataFrame] = []
for ativo in ATIVOS:
    parte = quadros_por_ativo[ativo].reset_index().copy()
    parte.insert(0, "symbol", ativo)
    partes_dados_mercado.append(parte)

dados_mercado = pd.concat(partes_dados_mercado, ignore_index=True)
dados_mercado = dados_mercado[
    ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
]

DIRETORIO_SAIDA.mkdir(parents=True, exist_ok=True)
csv_dados_mercado = dados_mercado.to_csv(index=False)
(DIRETORIO_SAIDA / "market_data.csv").write_text(
    csv_dados_mercado,
    encoding="utf-8",
)
sha256_dados_mercado = hashlib.sha256(
    csv_dados_mercado.encode("utf-8")
).hexdigest()

print(f"Snapshot: {len(dados_mercado):,} linhas")
print(f"SHA-256: {sha256_dados_mercado}")


# %% 3 - Preparação metodológica
print("Construindo atributos técnicos a partir do OHLCV")
print(
    "Horizontes do alvo: "
    + ", ".join(str(valor) for valor in CONFIGURACAO.rotation_target_horizons)
)
print("Criando janelas de validação temporal com período de separação")

contexto_experimento = {
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
        "snapshot_rows": len(dados_mercado),
        "snapshot_sha256": sha256_dados_mercado,
    },
    "model_family": CONFIGURACAO.research_model_family,
    "target_horizons": list(CONFIGURACAO.rotation_target_horizons),
    "minimum_training_rows": CONFIGURACAO.rotation_minimum_training_rows,
    "calibration_days": CONFIGURACAO.rotation_walk_forward_calibration_days,
    "test_days": CONFIGURACAO.rotation_walk_forward_test_days,
    "minimum_test_days": CONFIGURACAO.rotation_walk_forward_min_test_days,
    "purge_days": CONFIGURACAO.rotation_purge_days,
}


# %% 4 - Treinamento LightGBM, previsões fora da amostra e política de rotação
print("Treinando LightGBM do zero em cada janela e ativo")

resultados = executar_modelos_rotacao(
    quadros_por_ativo,
    CONFIGURACAO,
    calcular_taxas_referencia,
    aplicar_deslizamento,
)

if not resultados:
    raise RuntimeError("O motor não retornou resultado.")
if len(resultados) != 1:
    raise RuntimeError(f"Era esperada uma execução; foram retornadas {len(resultados)}.")

resultado = resultados[0]


# %% 5 - Objetos de resultado para inspeção no Spyder
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
        "selected_score",
        "decision_score",
        "trade_action",
        "trade_reason",
        "walk_forward_fold",
        "cash_weight",
        "market_exposure_weight",
        "assets_held",
    )
    if coluna in previsoes.columns
]
curva_capital = (
    previsoes[colunas_curva].copy() if colunas_curva else previsoes.copy()
)

tempo_decorrido = time.perf_counter() - inicio_execucao

dados_resultado = {
    "experiment": "mba_usp_yahoo_backtest",
    "input_source": "yahoo_finance_ohlcv",
    **contexto_experimento,
    "walk_forward": {
        "minimum_training_rows": CONFIGURACAO.rotation_minimum_training_rows,
        "calibration_days": CONFIGURACAO.rotation_walk_forward_calibration_days,
        "test_days": CONFIGURACAO.rotation_walk_forward_test_days,
        "minimum_test_days": CONFIGURACAO.rotation_walk_forward_min_test_days,
        "purge_days": CONFIGURACAO.rotation_purge_days,
    },
    "elapsed_seconds": float(tempo_decorrido),
    "metrics": metricas,
}


# %% 6 - Gravação dos artefatos
curva_para_csv = curva_capital.copy()
if curva_para_csv.index.name is not None or not isinstance(
    curva_para_csv.index,
    pd.RangeIndex,
):
    curva_para_csv = curva_para_csv.reset_index()

operacoes_para_csv = operacoes.copy()
if operacoes_para_csv.index.name is not None or not isinstance(
    operacoes_para_csv.index,
    pd.RangeIndex,
):
    operacoes_para_csv = operacoes_para_csv.reset_index()

for tabela in (curva_para_csv, operacoes_para_csv):
    for coluna in tabela.columns:
        possui_valores_compostos = tabela[coluna].map(
            lambda valor: isinstance(valor, (dict, list, tuple))
        ).any()
        if possui_valores_compostos:
            tabela[coluna] = tabela[coluna].map(
                lambda valor: json.dumps(
                    valor,
                    ensure_ascii=False,
                    default=serializar_json,
                )
                if isinstance(valor, (dict, list, tuple))
                else valor
            )

curva_para_csv.to_csv(DIRETORIO_SAIDA / "equity_curve.csv", index=False)
operacoes_para_csv.to_csv(DIRETORIO_SAIDA / "trades.csv", index=False)
pd.DataFrame(janelas).to_csv(DIRETORIO_SAIDA / "folds.csv", index=False)

(DIRETORIO_SAIDA / "backtest_result.json").write_text(
    json.dumps(
        dados_resultado,
        indent=2,
        ensure_ascii=False,
        default=serializar_json,
    )
    + "\n",
    encoding="utf-8",
)
(DIRETORIO_SAIDA / "summary.txt").write_text(
    str(resultado.summary).rstrip() + "\n",
    encoding="utf-8",
)


# %% 7 - Métricas finais
print(f"Capital inicial : US$ {CONFIGURACAO.initial_capital:,.2f}")
print(f"Capital final   : US$ {float(metricas['strategy_ending_capital']):,.2f}")
print(f"Retorno         : {float(metricas['strategy_return']):.2%}")
print(f"CAGR            : {float(metricas['strategy_cagr']):.2%}")
print(f"Sharpe          : {float(metricas['strategy_sharpe']):.3f}")
print(f"Queda máxima    : {float(metricas['strategy_maximum_drawdown']):.2%}")
print(f"Rotações        : {int(metricas.get('capital_rotations') or 0)}")
print(f"Dias em caixa   : {int(metricas.get('cash_days') or 0)}")
print(f"Tempo total     : {tempo_decorrido:.2f}s")
print(f"Resultados      : {DIRETORIO_SAIDA}")
