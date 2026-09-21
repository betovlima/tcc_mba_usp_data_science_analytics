"""Experimento Tiingo total-causal para o TCC MBA USP.

Versao: tiingo-total-causal-v1.0.0

Objetivo experimental
---------------------
Alterar uma unica familia de processamento em relacao ao backtest Tiingo atual:
alem da normalizacao causal dos desdobramentos, neutralizar causalmente o efeito
mecanico dos dividendos no preco a partir da data do evento.

O OHLCV RAW congelado permanece imutavel. Nenhum evento futuro reescreve o
passado. O backtest continua sem consultas externas e sem dependencia de dados
ou processamento do Market Cycle Trader em tempo de execucao.

A serie entregue ao modelo e construida assim:

    Tiingo RAW
      -> normalizacao causal de splits
      -> neutralizacao causal de dividendos
      -> features / targets / LightGBM / walk-forward / rotacao

O experimento grava seus resultados em output/tiingo_total_causal_v1 para nao
sobrescrever o controle atual baseado apenas em splits.
"""

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

VERSAO_EXPERIMENTO = "tiingo-total-causal-v1.0.0"
CAPITAL_REFERENCIA_43M = 43_759_854.82

RAIZ_PROJETO = Path(__file__).resolve().parent
DIRETORIO_SERIES = RAIZ_PROJETO / "dados" / "series_historicas"
DIRETORIO_EVENTOS = RAIZ_PROJETO / "dados" / "eventos_corporativos"
DIRETORIO_DESDOBRAMENTOS = RAIZ_PROJETO / "dados" / "desdobramentos"
ARQUIVO_MANIFESTO = RAIZ_PROJETO / "dados" / "manifesto_tiingo.json"
ARQUIVO_MANIFESTO_DESDOBRAMENTOS = (
    RAIZ_PROJETO / "dados" / "manifesto_desdobramentos_tiingo.json"
)
DIRETORIO_RESULTADOS = RAIZ_PROJETO / "output" / "tiingo_total_causal_v1"

COLUNAS_OHLCV = ["open", "high", "low", "close", "volume"]
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


def ler_json_opcional(caminho: Path) -> dict[str, Any]:
    if not caminho.exists():
        return {}
    return json.loads(caminho.read_text(encoding="utf-8"))


def carregar_serie_raw(ativo: str) -> pd.DataFrame:
    arquivo = DIRETORIO_SERIES / f"{ativo}.csv"
    if not arquivo.exists():
        raise RuntimeError(f"Serie RAW ausente para {ativo}: {arquivo}")

    tabela = pd.read_csv(arquivo)
    obrigatorias = ["timestamp", *COLUNAS_OHLCV]
    ausentes = [coluna for coluna in obrigatorias if coluna not in tabela.columns]
    if ausentes:
        raise RuntimeError(
            f"{ativo}: colunas ausentes na serie RAW: " + ", ".join(ausentes)
        )

    proibidas = {
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
    contaminadas = [
        coluna for coluna in tabela.columns if str(coluna).lower() in proibidas
    ]
    if contaminadas:
        raise RuntimeError(
            f"{ativo}: serie RAW contaminada por ajuste/evento: "
            + ", ".join(contaminadas)
        )

    tabela = tabela[obrigatorias].copy()
    tabela["timestamp"] = pd.to_datetime(tabela["timestamp"], utc=True, errors="coerce")
    for coluna in COLUNAS_OHLCV:
        tabela[coluna] = pd.to_numeric(tabela[coluna], errors="coerce")

    tabela = tabela.dropna(subset=obrigatorias)
    tabela = tabela.sort_values("timestamp")
    tabela = tabela.drop_duplicates(subset=["timestamp"], keep="last")
    tabela = tabela.loc[
        (tabela["open"] > 0)
        & (tabela["high"] > 0)
        & (tabela["low"] > 0)
        & (tabela["close"] > 0)
        & (tabela["volume"] >= 0)
    ].copy()

    if tabela.empty:
        raise RuntimeError(f"{ativo}: serie RAW vazia depois da validacao.")

    return tabela.set_index("timestamp")[COLUNAS_OHLCV].copy()


def carregar_desdobramentos(ativo: str) -> pd.DataFrame:
    arquivo = DIRETORIO_DESDOBRAMENTOS / f"{ativo}.csv"
    if not arquivo.exists():
        raise RuntimeError(f"Desdobramentos ausentes para {ativo}: {arquivo}")

    tabela = pd.read_csv(arquivo)
    ausentes = [coluna for coluna in COLUNAS_DESDOBRAMENTOS if coluna not in tabela.columns]
    if ausentes:
        raise RuntimeError(
            f"{ativo}: colunas ausentes em desdobramentos: " + ", ".join(ausentes)
        )

    if tabela.empty:
        return pd.DataFrame(columns=COLUNAS_DESDOBRAMENTOS)

    tabela = tabela[COLUNAS_DESDOBRAMENTOS].copy()
    tabela["timestamp"] = pd.to_datetime(tabela["timestamp"], utc=True, errors="coerce")
    for coluna in ("split_de", "split_para", "fator_split"):
        tabela[coluna] = pd.to_numeric(tabela[coluna], errors="coerce")
    tabela["status"] = tabela["status"].astype(str).str.lower().str.strip()
    tabela = tabela.dropna(subset=["timestamp", "split_de", "split_para", "fator_split"])
    tabela = tabela.loc[
        (tabela["split_de"] > 0)
        & (tabela["split_para"] > 0)
        & (tabela["fator_split"] > 0)
        & (tabela["status"] == "a")
    ].copy()
    tabela = tabela.sort_values("timestamp")
    tabela = tabela.drop_duplicates(
        subset=["timestamp", "split_de", "split_para"], keep="last"
    )
    return tabela.reset_index(drop=True)


def carregar_dividendos(ativo: str) -> pd.DataFrame:
    arquivo = DIRETORIO_EVENTOS / f"{ativo}.csv"
    if not arquivo.exists():
        raise RuntimeError(f"Eventos corporativos ausentes para {ativo}: {arquivo}")

    tabela = pd.read_csv(arquivo)
    if "timestamp" not in tabela.columns or "dividendo" not in tabela.columns:
        raise RuntimeError(
            f"{ativo}: arquivo de eventos precisa conter timestamp e dividendo."
        )

    if tabela.empty:
        return pd.DataFrame(columns=["timestamp", "dividendo"])

    tabela = tabela[["timestamp", "dividendo"]].copy()
    tabela["timestamp"] = pd.to_datetime(tabela["timestamp"], utc=True, errors="coerce")
    tabela["dividendo"] = pd.to_numeric(tabela["dividendo"], errors="coerce").fillna(0.0)
    tabela = tabela.dropna(subset=["timestamp"])
    tabela = tabela.loc[tabela["dividendo"] != 0.0].copy()
    tabela = tabela.sort_values("timestamp")
    tabela = tabela.drop_duplicates(subset=["timestamp"], keep="last")
    return tabela.reset_index(drop=True)


def normalizar_splits_causalmente(
    ativo: str,
    serie_raw: pd.DataFrame,
    desdobramentos: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[dict[str, Any]]]:
    fator_na_sessao = pd.Series(1.0, index=serie_raw.index, dtype=float)
    diagnostico: list[dict[str, Any]] = []

    for evento in desdobramentos.itertuples(index=False):
        data_evento = pd.Timestamp(evento.timestamp).normalize()
        sessoes = serie_raw.index[serie_raw.index.normalize() == data_evento]
        if len(sessoes) != 1:
            raise RuntimeError(
                f"{ativo}: split de {data_evento.date()} nao corresponde a uma unica sessao."
            )

        sessao = sessoes[0]
        fator = float(evento.fator_split)
        esperado = float(evento.split_para) / float(evento.split_de)
        if not np.isclose(fator, esperado, rtol=1e-8, atol=1e-10):
            raise RuntimeError(
                f"{ativo}: fator de split inconsistente em {data_evento.date()}: "
                f"fonte={fator} calculado={esperado}"
            )
        fator_na_sessao.loc[sessao] *= fator

    fator_acumulado = fator_na_sessao.cumprod()
    serie = serie_raw.copy()
    for coluna in ("open", "high", "low", "close"):
        serie[coluna] = serie[coluna] * fator_acumulado
    serie["volume"] = serie["volume"] / fator_acumulado

    for evento in desdobramentos.itertuples(index=False):
        data_evento = pd.Timestamp(evento.timestamp).normalize()
        sessao = serie_raw.index[serie_raw.index.normalize() == data_evento][0]
        posicao = int(serie_raw.index.get_loc(sessao))
        if posicao == 0:
            continue
        anterior = serie_raw.index[posicao - 1]
        diagnostico.append(
            {
                "ativo": ativo,
                "data": sessao,
                "split_de": float(evento.split_de),
                "split_para": float(evento.split_para),
                "fator_split": float(evento.fator_split),
                "retorno_raw": float(serie_raw.loc[sessao, "close"] / serie_raw.loc[anterior, "close"] - 1.0),
                "retorno_apos_split": float(serie.loc[sessao, "close"] / serie.loc[anterior, "close"] - 1.0),
                "fator_split_acumulado": float(fator_acumulado.loc[sessao]),
            }
        )

    return serie, fator_acumulado, diagnostico


def neutralizar_dividendos_causalmente(
    ativo: str,
    serie_split: pd.DataFrame,
    fator_split_acumulado: pd.Series,
    dividendos: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[dict[str, Any]]]:
    """Neutraliza a ruptura mecanica do ex-dividendo sem olhar o futuro.

    Para um dividendo conhecido na sessao t, o fator passa a valer em t e nas
    sessoes seguintes:

        fator_t = close_(t-1) / (close_(t-1) - dividendo_normalizado_t)

    O dividendo e convertido para a mesma unidade economica da serie ja
    normalizada por splits. Volume nao e ajustado por dividendos.
    """

    fator_na_sessao = pd.Series(1.0, index=serie_split.index, dtype=float)
    diagnostico: list[dict[str, Any]] = []

    for evento in dividendos.itertuples(index=False):
        data_evento = pd.Timestamp(evento.timestamp).normalize()
        sessoes = serie_split.index[serie_split.index.normalize() == data_evento]
        if len(sessoes) == 0:
            continue
        if len(sessoes) != 1:
            raise RuntimeError(
                f"{ativo}: dividendo de {data_evento.date()} corresponde a multiplas sessoes."
            )

        sessao = sessoes[0]
        posicao = int(serie_split.index.get_loc(sessao))
        if posicao == 0:
            continue

        anterior = serie_split.index[posicao - 1]
        fechamento_anterior = float(serie_split.loc[anterior, "close"])
        dividendo_raw = float(evento.dividendo)
        dividendo_normalizado = dividendo_raw * float(fator_split_acumulado.loc[sessao])
        denominador = fechamento_anterior - dividendo_normalizado

        if not np.isfinite(denominador) or denominador <= 0:
            raise RuntimeError(
                f"{ativo}: dividendo invalido em {data_evento.date()} | "
                f"close anterior={fechamento_anterior} | "
                f"dividendo normalizado={dividendo_normalizado}"
            )

        fator_evento = fechamento_anterior / denominador
        fator_na_sessao.loc[sessao] *= fator_evento
        diagnostico.append(
            {
                "ativo": ativo,
                "data": sessao,
                "dividendo_raw": dividendo_raw,
                "fator_split_acumulado": float(fator_split_acumulado.loc[sessao]),
                "dividendo_normalizado": dividendo_normalizado,
                "fechamento_anterior_split": fechamento_anterior,
                "fator_dividendo_na_sessao": fator_evento,
            }
        )

    fator_dividendo_acumulado = fator_na_sessao.cumprod()
    serie = serie_split.copy()
    for coluna in ("open", "high", "low", "close"):
        serie[coluna] = serie[coluna] * fator_dividendo_acumulado

    for linha in diagnostico:
        sessao = pd.Timestamp(linha["data"])
        posicao = int(serie.index.get_loc(sessao))
        if posicao == 0:
            continue
        anterior = serie.index[posicao - 1]
        linha["retorno_apos_split"] = float(
            serie_split.loc[sessao, "close"] / serie_split.loc[anterior, "close"] - 1.0
        )
        linha["retorno_total_causal"] = float(
            serie.loc[sessao, "close"] / serie.loc[anterior, "close"] - 1.0
        )
        linha["fator_dividendo_acumulado"] = float(
            fator_dividendo_acumulado.loc[sessao]
        )

    return serie, fator_dividendo_acumulado, diagnostico


def validar_serie_modelo(ativo: str, serie: pd.DataFrame) -> None:
    valores = serie[COLUNAS_OHLCV].to_numpy(dtype=float)
    if not np.isfinite(valores).all():
        raise RuntimeError(f"{ativo}: serie causal contem valor nao finito.")
    if (serie[["open", "high", "low", "close"]] <= 0).any().any():
        raise RuntimeError(f"{ativo}: serie causal contem preco nao positivo.")
    if (serie["volume"] < 0).any():
        raise RuntimeError(f"{ativo}: serie causal contem volume negativo.")


def preparar_csv(tabela: pd.DataFrame) -> pd.DataFrame:
    saida = tabela.copy()
    if saida.index.name is not None or not isinstance(saida.index, pd.RangeIndex):
        saida = saida.reset_index()
    for coluna in saida.columns:
        possui_aninhado = saida[coluna].map(
            lambda valor: isinstance(valor, (dict, list, tuple))
        ).any()
        if possui_aninhado:
            saida[coluna] = saida[coluna].map(
                lambda valor: json.dumps(
                    valor,
                    ensure_ascii=False,
                    default=converter_json,
                )
                if isinstance(valor, (dict, list, tuple))
                else valor
            )
    return saida


def main() -> int:
    inicio = time.perf_counter()

    registrar(f"TCC MBA USP - {VERSAO_EXPERIMENTO}")
    registrar("Hipotese: splits + dividendos causais aproximam a referencia adjustment=all")
    registrar(f"Periodo: {DATA_INICIO} -> {DATA_FIM} | ativos={len(ATIVOS)}")
    registrar("RAW permanece imutavel; nenhuma consulta externa sera realizada")

    for diretorio, descricao in (
        (DIRETORIO_SERIES, "series RAW"),
        (DIRETORIO_EVENTOS, "eventos corporativos"),
        (DIRETORIO_DESDOBRAMENTOS, "desdobramentos"),
    ):
        if not diretorio.exists():
            raise RuntimeError(f"Diretorio ausente ({descricao}): {diretorio}")

    manifesto = ler_json_opcional(ARQUIVO_MANIFESTO)
    manifesto_splits = ler_json_opcional(ARQUIVO_MANIFESTO_DESDOBRAMENTOS)

    series_raw: dict[str, pd.DataFrame] = {}
    series_modelo: dict[str, pd.DataFrame] = {}
    diagnostico_splits: list[dict[str, Any]] = []
    diagnostico_dividendos: list[dict[str, Any]] = []
    total_splits = 0
    total_dividendos = 0

    registrar("[1/8] Carregando snapshots locais")
    for posicao, ativo in enumerate(ATIVOS, start=1):
        serie_raw = carregar_serie_raw(ativo)
        splits = carregar_desdobramentos(ativo)
        dividendos = carregar_dividendos(ativo)

        serie_split, fator_split, diag_split = normalizar_splits_causalmente(
            ativo,
            serie_raw,
            splits,
        )
        serie_total, _, diag_div = neutralizar_dividendos_causalmente(
            ativo,
            serie_split,
            fator_split,
            dividendos,
        )
        validar_serie_modelo(ativo, serie_total)

        series_raw[ativo] = serie_raw
        series_modelo[ativo] = serie_total
        diagnostico_splits.extend(diag_split)
        diagnostico_dividendos.extend(diag_div)
        total_splits += len(splits)
        total_dividendos += len(dividendos)

        registrar(
            f"[1/8] {posicao:02d}/{len(ATIVOS)} {ativo} | "
            f"candles={len(serie_raw)} | splits={len(splits)} | dividendos={len(dividendos)}"
        )

    registrar(
        f"[2/8] Series causais prontas | splits={total_splits} | dividendos={total_dividendos}"
    )
    registrar("[3/8] Features e targets serao construidos sobre a serie total-causal")
    registrar("[4/8] Mantendo exatamente a mesma configuracao walk-forward e LightGBM")

    resultados = executar_modelos_rotacao(
        series_modelo,
        CONFIGURACAO,
        calcular_taxas_referencia,
        aplicar_deslizamento,
        progress_callback=registrar_progresso,
        technical_log_callback=registrar_detalhe_tecnico,
    )

    if not resultados:
        raise RuntimeError("O motor nao retornou resultado.")
    if len(resultados) != 1:
        raise RuntimeError(f"Esperava uma execucao; retornaram {len(resultados)}.")

    resultado = resultados[0]
    previsoes = resultado.predictions.copy()
    operacoes = resultado.trades.copy()
    metricas = dict(resultado.metrics)
    folds = list(metricas.get("walk_forward_folds") or [])

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
    curva = previsoes[colunas_curva].copy() if colunas_curva else previsoes.copy()

    capital_final = float(metricas["strategy_ending_capital"])
    tempo_total = time.perf_counter() - inicio
    diferenca_referencia = capital_final - CAPITAL_REFERENCIA_43M
    razao_referencia = capital_final / CAPITAL_REFERENCIA_43M

    resultado_serializado = {
        "experiment": "mba_usp_tiingo_total_causal_dividends_v1",
        "experiment_version": VERSAO_EXPERIMENTO,
        "single_change_vs_tiingo_split_control": "causal_dividend_neutralization",
        "input_source": "tiingo_eod_raw_snapshot_plus_causal_splits_plus_causal_dividends",
        "assets": list(ATIVOS),
        "asset_count": len(ATIVOS),
        "history_start": DATA_INICIO,
        "history_end": DATA_FIM,
        "market_data_source": "tiingo_eod_raw_snapshot_local",
        "market_data_snapshot_frozen": True,
        "market_data_snapshot_created_at": manifesto.get("data_congelamento_utc"),
        "raw_ohlcv_mutated_on_disk": False,
        "split_source": "tiingo_corporate_actions_splits_snapshot",
        "split_snapshot_created_at": manifesto_splits.get("data_congelamento_utc"),
        "split_normalization_applied": True,
        "split_normalization_direction": "event_date_forward",
        "split_normalization_uses_future_events": False,
        "split_event_count": total_splits,
        "dividend_adjustment_applied": True,
        "dividend_adjustment_direction": "event_date_forward",
        "dividend_adjustment_uses_future_events": False,
        "dividend_events_used_by_model": True,
        "dividend_event_count": total_dividendos,
        "model_price_basis": "raw_with_forward_causal_split_and_dividend_normalization",
        "model_family": CONFIGURACAO.research_model_family,
        "target_horizons": list(CONFIGURACAO.rotation_target_horizons),
        "walk_forward": {
            "minimum_training_rows": CONFIGURACAO.rotation_minimum_training_rows,
            "calibration_days": CONFIGURACAO.rotation_walk_forward_calibration_days,
            "test_days": CONFIGURACAO.rotation_walk_forward_test_days,
            "minimum_test_days": CONFIGURACAO.rotation_walk_forward_min_test_days,
            "purge_days": CONFIGURACAO.rotation_purge_days,
        },
        "historical_reference": {
            "label": "certified-43m-standalone",
            "ending_capital": CAPITAL_REFERENCIA_43M,
            "difference_usd": diferenca_referencia,
            "ratio_to_reference": razao_referencia,
        },
        "elapsed_seconds": tempo_total,
        "metrics": metricas,
    }

    DIRETORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
    preparar_csv(curva).to_csv(DIRETORIO_RESULTADOS / "equity_curve.csv", index=False)
    preparar_csv(operacoes).to_csv(DIRETORIO_RESULTADOS / "trades.csv", index=False)
    pd.DataFrame(folds).to_csv(DIRETORIO_RESULTADOS / "folds.csv", index=False)
    pd.DataFrame(diagnostico_splits).to_csv(
        DIRETORIO_RESULTADOS / "diagnostico_splits.csv", index=False
    )
    pd.DataFrame(diagnostico_dividendos).to_csv(
        DIRETORIO_RESULTADOS / "diagnostico_dividendos.csv", index=False
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

    registrar("[8/8] Resultado concluido")
    registrar(f"Capital inicial : US$ {CONFIGURACAO.initial_capital:,.2f}")
    registrar(f"Capital final   : US$ {capital_final:,.2f}")
    registrar(f"Referencia 43M  : US$ {CAPITAL_REFERENCIA_43M:,.2f}")
    registrar(f"Razao referencia: {razao_referencia:.4f}x")
    registrar(f"Diferenca       : US$ {diferenca_referencia:,.2f}")
    registrar(f"Retorno         : {float(metricas['strategy_return']):.2%}")
    registrar(f"CAGR            : {float(metricas['strategy_cagr']):.2%}")
    registrar(f"Sharpe          : {float(metricas['strategy_sharpe']):.3f}")
    registrar(f"Max Drawdown    : {float(metricas['strategy_maximum_drawdown']):.2%}")
    registrar(f"Rotacoes        : {int(metricas.get('capital_rotations') or 0)}")
    registrar(f"Splits causais  : {total_splits}")
    registrar(f"Dividendos causais: {total_dividendos}")
    registrar(f"Tempo total     : {tempo_total:.2f}s")
    registrar(f"Resultados      : {DIRETORIO_RESULTADOS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
