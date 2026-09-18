"""Backtest academico reproduzivel para o TCC MBA USP.

A entrada do experimento e uma fotografia local e congelada da Tiingo. Os
arquivos de mercado contem somente OHLCV bruto. Dividendos e desdobramentos
ficam separados da serie de precos.

Antes da criacao das variaveis do modelo, somente os desdobramentos reais
obtidos pelo endpoint especifico de Corporate Actions da Tiingo sao tratados.
O tratamento e causal: o fator passa a valer na data do evento e somente dali
em diante. Nenhum evento futuro reescreve observacoes passadas.

Dividendos nao ajustam open, high, low, close ou volume, nao entram nas
variaveis e nao alteram os alvos de rotacao nesta etapa.
"""

# %% 0 - Imports e configuracao
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tcc_engine.capital_rotation import run_rotation_models as executar_modelos_rotacao
from tcc_engine.config import ASSETS as ATIVOS
from tcc_engine.config import CONFIG as CONFIGURACAO
from tcc_engine.config import END_DATE as DATA_FIM
from tcc_engine.config import START_DATE as DATA_INICIO
from tcc_engine.execution import apply_slippage as aplicar_deslizamento
from tcc_engine.execution import calculate_reference_fees as calcular_taxas_referencia

RAIZ_PROJETO = Path(__file__).resolve().parent
DIRETORIO_SERIES = RAIZ_PROJETO / "dados" / "series_historicas"
DIRETORIO_EVENTOS = RAIZ_PROJETO / "dados" / "eventos_corporativos"
DIRETORIO_DESDOBRAMENTOS = RAIZ_PROJETO / "dados" / "desdobramentos"
ARQUIVO_MANIFESTO = RAIZ_PROJETO / "dados" / "manifesto_tiingo.json"
ARQUIVO_MANIFESTO_DESDOBRAMENTOS = (
    RAIZ_PROJETO / "dados" / "manifesto_desdobramentos_tiingo.json"
)
DIRETORIO_RESULTADOS = RAIZ_PROJETO / "output"

COLUNAS_OHLCV = ["open", "high", "low", "close", "volume"]
COLUNAS_EVENTOS = ["timestamp", "dividendo"]
COLUNAS_DESDOBRAMENTOS = [
    "timestamp",
    "split_de",
    "split_para",
    "fator_split",
    "status",
]


def registrar(mensagem: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {mensagem}", flush=True)


def converter_json(valor: Any) -> Any:
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


def registrar_progresso(percentual: float, etapa: str, execucoes_concluidas: int) -> None:
    registrar(f"[5/8] {percentual:5.1f}% | {etapa}")


def registrar_detalhe_tecnico(mensagem: str) -> None:
    registrar(f"[motor] {mensagem}")


# %% 1 - Inicio da execucao e identificacao dos snapshots
inicio_execucao = time.perf_counter()

registrar("TCC MBA USP - backtest reconstruido passo a passo")
registrar("Entrada: Tiingo RAW congelado + desdobramentos causais -> motor")
registrar(f"Periodo: {DATA_INICIO} -> {DATA_FIM} | ativos={len(ATIVOS)}")
registrar("Dividendos: preservados para auditoria, sem ajuste de preco")

if not DIRETORIO_SERIES.exists():
    raise RuntimeError(
        "Diretorio de series historicas nao encontrado. "
        "Execute primeiro: python congelar_series_tiingo.py"
    )
if not DIRETORIO_EVENTOS.exists():
    raise RuntimeError(
        "Diretorio de eventos corporativos nao encontrado. "
        "Execute novamente: python congelar_series_tiingo.py"
    )
if not DIRETORIO_DESDOBRAMENTOS.exists():
    raise RuntimeError(
        "Diretorio de desdobramentos nao encontrado. "
        "Execute primeiro: python congelar_desdobramentos_tiingo.py"
    )

manifesto: dict[str, Any] = {}
if ARQUIVO_MANIFESTO.exists():
    manifesto = json.loads(ARQUIVO_MANIFESTO.read_text(encoding="utf-8"))
    registrar(
        "Snapshot OHLCV: "
        + str(manifesto.get("data_congelamento_utc") or "data nao informada")
    )
else:
    registrar("Aviso: manifesto_tiingo.json nao encontrado; usando os CSVs locais.")

manifesto_desdobramentos: dict[str, Any] = {}
if ARQUIVO_MANIFESTO_DESDOBRAMENTOS.exists():
    manifesto_desdobramentos = json.loads(
        ARQUIVO_MANIFESTO_DESDOBRAMENTOS.read_text(encoding="utf-8")
    )
    registrar(
        "Snapshot de desdobramentos: "
        + str(
            manifesto_desdobramentos.get("data_congelamento_utc")
            or "data nao informada"
        )
    )
else:
    registrar(
        "Aviso: manifesto_desdobramentos_tiingo.json nao encontrado; "
        "validando os CSVs individualmente."
    )


# %% 2 - Carregamento da materia-prima e normalizacao causal de desdobramentos
registrar(f"[1/8] Carregando {len(ATIVOS)} series OHLCV RAW congeladas")
# %% 1B - Contrato estrito do baseline Tiingo 56
ativos_esperados = list(ATIVOS)
if len(ativos_esperados) != 56:
    raise RuntimeError(
        f"Baseline invalido: esperado universo de 56 ativos; CONFIG possui {len(ativos_esperados)}."
    )

if not manifesto:
    raise RuntimeError(
        "manifesto_tiingo.json e obrigatorio para o baseline certificado."
    )
ativos_manifesto = [str(valor).upper() for valor in (manifesto.get("ativos") or [])]
if (
    int(manifesto.get("quantidade_ativos", -1)) != 56
    or ativos_manifesto != ativos_esperados
):
    raise RuntimeError(
        "Snapshot Tiingo nao corresponde exatamente aos 56 ativos configurados. "
        "Execute novamente congelar_series_tiingo.py."
    )

if not manifesto_desdobramentos:
    raise RuntimeError(
        "manifesto_desdobramentos_tiingo.json e obrigatorio para o baseline certificado."
    )
if int(manifesto_desdobramentos.get("quantidade_ativos", -1)) != 56:
    raise RuntimeError(
        "Snapshot de desdobramentos nao corresponde aos 56 ativos. "
        "Execute novamente congelar_desdobramentos_tiingo.py."
    )

if tuple(CONFIGURACAO.calendar_anchor_assets) != tuple(ATIVOS):
    raise RuntimeError("calendar_anchor_assets deve conter exatamente os 56 ativos.")
if tuple(CONFIGURACAO.research_reference_assets) != tuple(ATIVOS):
    raise RuntimeError("research_reference_assets deve conter exatamente os 56 ativos.")
if tuple(CONFIGURACAO.research_candidate_assets):
    raise RuntimeError("research_candidate_assets deve estar vazio no baseline oficial.")

registrar("[0/8] Contrato validado: Tiingo-only | 56 ativos | anchors=56 | references=56 | candidates=0")

registrar("[1/8] OHLCV bruto permanece imutavel em series_historicas_brutas")
registrar("[1/8] Dividendos nao serao incorporados aos precos")
registrar("[1/8] splitFactor do endpoint EOD nao sera usado como split")
registrar("[1/8] Nenhuma consulta externa sera realizada")

series_historicas_brutas: dict[str, pd.DataFrame] = {}
eventos_corporativos: dict[str, pd.DataFrame] = {}
desdobramentos_corporativos: dict[str, pd.DataFrame] = {}

colunas_proibidas_serie = {
    "adjopen",
    "adjhigh",
    "adjlow",
    "adjclose",
    "adjvolume",
    "dividendo",
    "fator_split",
    "divcash",
    "splitfactor",
}

for posicao, ativo in enumerate(ATIVOS, start=1):
    arquivo_serie = DIRETORIO_SERIES / f"{ativo}.csv"
    arquivo_eventos = DIRETORIO_EVENTOS / f"{ativo}.csv"
    arquivo_desdobramentos = DIRETORIO_DESDOBRAMENTOS / f"{ativo}.csv"

    if not arquivo_serie.exists():
        raise RuntimeError(
            f"Snapshot incompleto: serie bruta ausente para {ativo}: {arquivo_serie}"
        )
    if not arquivo_eventos.exists():
        raise RuntimeError(
            f"Snapshot incompleto: eventos ausentes para {ativo}: {arquivo_eventos}"
        )
    if not arquivo_desdobramentos.exists():
        raise RuntimeError(
            f"Snapshot incompleto: desdobramentos ausentes para {ativo}: "
            f"{arquivo_desdobramentos}"
        )

    tabela = pd.read_csv(arquivo_serie)
    colunas_obrigatorias = ["timestamp", *COLUNAS_OHLCV]
    ausentes = [coluna for coluna in colunas_obrigatorias if coluna not in tabela.columns]
    if ausentes:
        raise RuntimeError(
            f"{ativo}: colunas ausentes na serie bruta: " + ", ".join(ausentes)
        )

    proibidas_encontradas = [
        coluna
        for coluna in tabela.columns
        if str(coluna).lower() in colunas_proibidas_serie
    ]
    if proibidas_encontradas:
        raise RuntimeError(
            f"{ativo}: serie bruta contaminada por colunas de ajuste/evento: "
            + ", ".join(proibidas_encontradas)
        )

    tabela = tabela[colunas_obrigatorias].copy()
    tabela["timestamp"] = pd.to_datetime(tabela["timestamp"], utc=True, errors="coerce")
    for coluna in COLUNAS_OHLCV:
        tabela[coluna] = pd.to_numeric(tabela[coluna], errors="coerce")

    tabela = tabela.dropna(subset=colunas_obrigatorias)
    tabela = tabela.sort_values("timestamp")
    tabela = tabela.drop_duplicates(subset=["timestamp"], keep="last")

    valores_validos = (
        (tabela["open"] > 0)
        & (tabela["high"] > 0)
        & (tabela["low"] > 0)
        & (tabela["close"] > 0)
        & (tabela["volume"] >= 0)
    )
    tabela = tabela.loc[valores_validos].copy()
    if tabela.empty:
        raise RuntimeError(f"{ativo}: serie bruta vazia depois da validacao.")

    serie_bruta = tabela.set_index("timestamp")[COLUNAS_OHLCV].copy()
    series_historicas_brutas[ativo] = serie_bruta

    eventos = pd.read_csv(arquivo_eventos)
    if "timestamp" not in eventos.columns or "dividendo" not in eventos.columns:
        raise RuntimeError(
            f"{ativo}: arquivo de eventos precisa conter timestamp e dividendo."
        )
    if not eventos.empty:
        eventos = eventos[["timestamp", "dividendo"]].copy()
        eventos["timestamp"] = pd.to_datetime(
            eventos["timestamp"], utc=True, errors="coerce"
        )
        eventos["dividendo"] = pd.to_numeric(
            eventos["dividendo"], errors="coerce"
        ).fillna(0.0)
        eventos = eventos.dropna(subset=["timestamp"])
        eventos = eventos.sort_values("timestamp")
        eventos = eventos.drop_duplicates(subset=["timestamp"], keep="last")
        eventos = eventos.loc[eventos["dividendo"] != 0.0].copy()
    else:
        eventos = pd.DataFrame(columns=COLUNAS_EVENTOS)
    eventos_corporativos[ativo] = eventos.reset_index(drop=True)

    desdobramentos = pd.read_csv(arquivo_desdobramentos)
    ausentes_desdobramentos = [
        coluna
        for coluna in COLUNAS_DESDOBRAMENTOS
        if coluna not in desdobramentos.columns
    ]
    if ausentes_desdobramentos:
        raise RuntimeError(
            f"{ativo}: colunas ausentes no arquivo de desdobramentos: "
            + ", ".join(ausentes_desdobramentos)
        )

    if not desdobramentos.empty:
        desdobramentos = desdobramentos[COLUNAS_DESDOBRAMENTOS].copy()
        desdobramentos["timestamp"] = pd.to_datetime(
            desdobramentos["timestamp"], utc=True, errors="coerce"
        )
        for coluna in ("split_de", "split_para", "fator_split"):
            desdobramentos[coluna] = pd.to_numeric(
                desdobramentos[coluna], errors="coerce"
            )
        desdobramentos["status"] = (
            desdobramentos["status"].astype(str).str.lower().str.strip()
        )
        desdobramentos = desdobramentos.dropna(
            subset=["timestamp", "split_de", "split_para", "fator_split"]
        )
        desdobramentos = desdobramentos.loc[
            (desdobramentos["split_de"] > 0)
            & (desdobramentos["split_para"] > 0)
            & (desdobramentos["fator_split"] > 0)
            & (desdobramentos["status"] == "a")
        ].copy()
        desdobramentos = desdobramentos.sort_values("timestamp")
        desdobramentos = desdobramentos.drop_duplicates(
            subset=["timestamp", "split_de", "split_para"], keep="last"
        )
    else:
        desdobramentos = pd.DataFrame(columns=COLUNAS_DESDOBRAMENTOS)

    desdobramentos_corporativos[ativo] = desdobramentos.reset_index(drop=True)

    registrar(
        f"[1/8] {posicao:02d}/{len(ATIVOS)} {ativo} | "
        f"{len(serie_bruta)} candles | "
        f"{serie_bruta.index.min().date()} -> {serie_bruta.index.max().date()} | "
        f"desdobramentos={len(desdobramentos)} | dividendos={len(eventos)}"
    )

if len(series_historicas_brutas) != len(ATIVOS):
    raise RuntimeError(
        f"Esperava {len(ATIVOS)} series; foram carregadas {len(series_historicas_brutas)}."
    )

total_desdobramentos = sum(
    len(tabela) for tabela in desdobramentos_corporativos.values()
)
total_dividendos = sum(len(tabela) for tabela in eventos_corporativos.values())

registrar(
    f"[1/8] Materia-prima carregada: {len(series_historicas_brutas)} series RAW | "
    f"desdobramentos={total_desdobramentos} | dividendos={total_dividendos}"
)

registrar("[2/8] Normalizando somente desdobramentos, da data do evento para frente")
registrar("[2/8] Dividendos permanecem fora do OHLCV, das features e dos targets")

series_historicas: dict[str, pd.DataFrame] = {}
fatores_desdobramentos: dict[str, pd.DataFrame] = {}
linhas_diagnostico_desdobramentos: list[dict[str, Any]] = []

for ativo in ATIVOS:
    serie_bruta = series_historicas_brutas[ativo]
    desdobramentos = desdobramentos_corporativos[ativo]

    fator_na_sessao = pd.Series(1.0, index=serie_bruta.index, dtype=float)

    for evento in desdobramentos.itertuples(index=False):
        data_evento = pd.Timestamp(evento.timestamp).normalize()
        sessoes = serie_bruta.index[serie_bruta.index.normalize() == data_evento]
        if len(sessoes) != 1:
            raise RuntimeError(
                f"{ativo}: desdobramento de {data_evento.date()} nao corresponde "
                "a exatamente uma sessao da serie RAW."
            )

        sessao = sessoes[0]
        fator = float(evento.fator_split)
        fator_calculado = float(evento.split_para) / float(evento.split_de)
        if not np.isclose(fator, fator_calculado, rtol=1e-8, atol=1e-10):
            raise RuntimeError(
                f"{ativo}: fator inconsistente em {data_evento.date()}: "
                f"fonte={fator} calculado={fator_calculado}"
            )
        fator_na_sessao.loc[sessao] *= fator

    fator_acumulado = fator_na_sessao.cumprod()
    serie_modelo = serie_bruta.copy()

    for coluna in ("open", "high", "low", "close"):
        serie_modelo[coluna] = serie_modelo[coluna] * fator_acumulado
    serie_modelo["volume"] = serie_modelo["volume"] / fator_acumulado

    if not np.isfinite(serie_modelo[COLUNAS_OHLCV].to_numpy(dtype=float)).all():
        raise RuntimeError(f"{ativo}: normalizacao causal gerou valor nao finito.")
    if (serie_modelo[["open", "high", "low", "close"]] <= 0).any().any():
        raise RuntimeError(f"{ativo}: normalizacao causal gerou preco nao positivo.")
    if (serie_modelo["volume"] < 0).any():
        raise RuntimeError(f"{ativo}: normalizacao causal gerou volume negativo.")

    series_historicas[ativo] = serie_modelo
    fatores_desdobramentos[ativo] = pd.DataFrame(
        {
            "fator_na_sessao": fator_na_sessao,
            "fator_acumulado": fator_acumulado,
        }
    )

    for evento in desdobramentos.itertuples(index=False):
        data_evento = pd.Timestamp(evento.timestamp).normalize()
        sessao = serie_bruta.index[serie_bruta.index.normalize() == data_evento][0]
        posicao = int(serie_bruta.index.get_loc(sessao))
        if posicao == 0:
            continue
        sessao_anterior = serie_bruta.index[posicao - 1]
        fechamento_bruto_anterior = float(serie_bruta.loc[sessao_anterior, "close"])
        fechamento_bruto_evento = float(serie_bruta.loc[sessao, "close"])
        fechamento_modelo_anterior = float(serie_modelo.loc[sessao_anterior, "close"])
        fechamento_modelo_evento = float(serie_modelo.loc[sessao, "close"])

        linhas_diagnostico_desdobramentos.append(
            {
                "ativo": ativo,
                "data": sessao,
                "split_de": float(evento.split_de),
                "split_para": float(evento.split_para),
                "fator_split": float(evento.fator_split),
                "fechamento_bruto_anterior": fechamento_bruto_anterior,
                "fechamento_bruto_evento": fechamento_bruto_evento,
                "retorno_bruto": fechamento_bruto_evento / fechamento_bruto_anterior - 1.0,
                "fechamento_modelo_anterior": fechamento_modelo_anterior,
                "fechamento_modelo_evento": fechamento_modelo_evento,
                "retorno_apos_normalizacao": (
                    fechamento_modelo_evento / fechamento_modelo_anterior - 1.0
                ),
                "fator_acumulado": float(fator_acumulado.loc[sessao]),
            }
        )

    if not desdobramentos.empty:
        registrar(
            f"[2/8] {ativo} | desdobramentos={len(desdobramentos)} | "
            f"fator acumulado final={float(fator_acumulado.iloc[-1]):g}"
        )

diagnostico_desdobramentos = pd.DataFrame(linhas_diagnostico_desdobramentos)
registrar(
    f"[2/8] Normalizacao concluida: {total_desdobramentos} desdobramento(s) "
    "aplicado(s) causalmente"
)


# %% 3 - Preparacao metodologica
registrar("[3/8] Construindo features e targets sobre a serie causal")
registrar(
    "[3/8] Horizontes do target: "
    + ", ".join(str(valor) for valor in CONFIGURACAO.rotation_target_horizons)
)
registrar("[4/8] Criando folds temporais walk-forward com purge")

contexto_experimento = {
    "assets": list(ATIVOS),
    "asset_count": len(ATIVOS),
    "history_start": DATA_INICIO,
    "history_end": DATA_FIM,
    "market_data_source": "tiingo_eod_raw_snapshot_local",
    "market_data_feed": "eod",
    "market_data_input_pure_raw": True,
    "market_data_snapshot_frozen": True,
    "market_data_snapshot_created_at": manifesto.get("data_congelamento_utc"),
    "model_price_basis": "raw_with_forward_causal_split_normalization",
    "split_source": "tiingo_corporate_actions_splits_snapshot",
    "split_snapshot_created_at": manifesto_desdobramentos.get("data_congelamento_utc"),
    "split_normalization_applied": True,
    "split_normalization_direction": "event_date_forward",
    "split_normalization_uses_future_events": False,
    "split_event_count": total_desdobramentos,
    "dividend_event_count": total_dividendos,
    "dividend_adjustment_applied": False,
    "dividend_events_used_by_model": False,
    "corporate_actions_stored_separately": True,
    "model_family": CONFIGURACAO.research_model_family,
    "target_horizons": list(CONFIGURACAO.rotation_target_horizons),
    "minimum_training_rows": CONFIGURACAO.rotation_minimum_training_rows,
    "calibration_days": CONFIGURACAO.rotation_walk_forward_calibration_days,
    "test_days": CONFIGURACAO.rotation_walk_forward_test_days,
    "minimum_test_days": CONFIGURACAO.rotation_walk_forward_min_test_days,
    "purge_days": CONFIGURACAO.rotation_purge_days,
}


# %% 4 - Treinamento LightGBM, previsoes fora da amostra e politica de rotacao
registrar("[5/8] Treinando LightGBM do zero em cada fold e ativo")

resultados = executar_modelos_rotacao(
    series_historicas,
    CONFIGURACAO,
    calcular_taxas_referencia,
    aplicar_deslizamento,
    progress_callback=registrar_progresso,
    technical_log_callback=registrar_detalhe_tecnico,
)

if not resultados:
    raise RuntimeError("O motor nao retornou resultado.")
if len(resultados) != 1:
    raise RuntimeError(
        f"Esperava uma execucao; foram retornadas {len(resultados)}."
    )

resultado = resultados[0]

registrar("[6/8] Aplicando a politica de rotacao somente nas sessoes fora da amostra")
registrar("[7/8] Reconstruindo operacoes, custos e curva de capital")


# %% 5 - Objetos de resultado para inspecao no Spyder
previsoes = resultado.predictions.copy()
operacoes = resultado.trades.copy()
folds = list(resultado.metrics.get("walk_forward_folds") or [])
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
    previsoes[colunas_curva].copy()
    if colunas_curva
    else previsoes.copy()
)

tempo_total = time.perf_counter() - inicio_execucao

resultado_serializado = {
    "experiment": "mba_usp_reproducible_backtest",
    "input_source": "tiingo_eod_raw_snapshot_plus_causal_splits",
    **contexto_experimento,
    "walk_forward": {
        "minimum_training_rows": CONFIGURACAO.rotation_minimum_training_rows,
        "calibration_days": CONFIGURACAO.rotation_walk_forward_calibration_days,
        "test_days": CONFIGURACAO.rotation_walk_forward_test_days,
        "minimum_test_days": CONFIGURACAO.rotation_walk_forward_min_test_days,
        "purge_days": CONFIGURACAO.rotation_purge_days,
    },
    "elapsed_seconds": float(tempo_total),
    "metrics": metricas,
}


# %% 6 - Gravacao somente dos artefatos finais
DIRETORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)

arquivo_residual = DIRETORIO_RESULTADOS / "market_data.csv"
if arquivo_residual.exists():
    arquivo_residual.unlink()

curva_csv = curva_capital.copy()
if curva_csv.index.name is not None or not isinstance(curva_csv.index, pd.RangeIndex):
    curva_csv = curva_csv.reset_index()

operacoes_csv = operacoes.copy()
if operacoes_csv.index.name is not None or not isinstance(operacoes_csv.index, pd.RangeIndex):
    operacoes_csv = operacoes_csv.reset_index()

for tabela in (curva_csv, operacoes_csv):
    for coluna in tabela.columns:
        possui_valores_aninhados = tabela[coluna].map(
            lambda valor: isinstance(valor, (dict, list, tuple))
        ).any()
        if possui_valores_aninhados:
            tabela[coluna] = tabela[coluna].map(
                lambda valor: json.dumps(
                    valor,
                    ensure_ascii=False,
                    default=converter_json,
                )
                if isinstance(valor, (dict, list, tuple))
                else valor
            )

curva_csv.to_csv(DIRETORIO_RESULTADOS / "equity_curve.csv", index=False)
operacoes_csv.to_csv(DIRETORIO_RESULTADOS / "trades.csv", index=False)
pd.DataFrame(folds).to_csv(DIRETORIO_RESULTADOS / "folds.csv", index=False)
diagnostico_desdobramentos.to_csv(
    DIRETORIO_RESULTADOS / "diagnostico_desdobramentos.csv",
    index=False,
)

(DIRETORIO_RESULTADOS / "backtest_result.json").write_text(
    json.dumps(
        resultado_serializado,
        indent=2,
        ensure_ascii=False,
        default=converter_json,
    )
    + "\n",
    encoding="utf-8",
)
(DIRETORIO_RESULTADOS / "summary.txt").write_text(
    str(resultado.summary).rstrip() + "\n",
    encoding="utf-8",
)


# %% 7 - Metricas finais
registrar("[8/8] Calculando e salvando metricas finais")
registrar(f"Capital inicial : US$ {CONFIGURACAO.initial_capital:,.2f}")
registrar(f"Capital final   : US$ {float(metricas['strategy_ending_capital']):,.2f}")
registrar(f"Retorno         : {float(metricas['strategy_return']):.2%}")
registrar(f"CAGR            : {float(metricas['strategy_cagr']):.2%}")
registrar(f"Sharpe          : {float(metricas['strategy_sharpe']):.3f}")
registrar(f"Max Drawdown    : {float(metricas['strategy_maximum_drawdown']):.2%}")
registrar(f"Rotacoes        : {int(metricas.get('capital_rotations') or 0)}")
registrar(f"CASH days       : {int(metricas.get('cash_days') or 0)}")
registrar(f"Series RAW      : {len(series_historicas_brutas)}")
registrar(f"Desdobramentos  : {total_desdobramentos} aplicados causalmente")
registrar("Dividendos      : 0 ajustes no preco/modelo")
registrar(f"Tempo total     : {tempo_total:.2f}s")
registrar(f"Resultados      : {DIRETORIO_RESULTADOS}")
