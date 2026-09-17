"""Atribuicao causal da diferenca de capital Mongo 43M vs Tiingo.

Versao: mongo-tiingo-exact-capital-attribution-v1.0.0

Este experimento nasce diretamente de ``certified-43m-spyder`` e mantem o
``tcc_engine`` certificado intacto. A unica variavel experimental e o conjunto
de frames OHLCV entregue ao mesmo motor.

Objetivos:
1. reproduzir o controle certificado usando Mongo local;
2. executar o MESMO codigo com todas as series Tiingo total-causais;
3. substituir um ativo por vez Mongo -> Tiingo;
4. substituir um ativo por vez Tiingo -> Mongo;
5. medir capital final e a primeira decisao OOS divergente do controle.

Os snapshots Tiingo sao lidos apenas do diretorio local ``dados``. Nao ha
consulta a Alpaca, Tiingo, MCT runtime ou qualquer API externa durante o teste.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

from tcc_engine.capital_rotation import run_rotation_models
from tcc_engine.config import ASSETS, CONFIG, END_DATE, START_DATE
from tcc_engine.execution import apply_slippage, calculate_reference_fees

VERSION = "mongo-tiingo-exact-capital-attribution-v1.0.0"
CERTIFIED_TAG = "certified-43m-spyder"
CERTIFIED_COMMIT = "726cf883489a403fae5699cce715e59863e4a368"
CERTIFIED_CAPITAL = 43_759_854.82

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output" / "mongo_tiingo_exact_capital_attribution_v1"
TIINGO_ROOT_DEFAULT = ROOT / "dados"
TIINGO_SERIES_DIRNAME = "series_historicas"
TIINGO_EVENTS_DIRNAME = "eventos_corporativos"
TIINGO_SPLITS_DIRNAME = "desdobramentos"

DEFAULT_MONGO_URI = "mongodb://localhost:27017"
DEFAULT_MONGO_DATABASE = "extrema_backtest"
MARKET_COLLECTION = "alpaca_market_bars"
OHLCV = ["open", "high", "low", "close", "volume"]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def progress_callback(percent: float, stage: str, completed_runs: int) -> None:
    log(f"{percent:5.1f}% | {stage}")


def technical_log_callback(message: str) -> None:
    log(f"[motor] {message}")


def _validate_frame(symbol: str, frame: pd.DataFrame, source: str) -> pd.DataFrame:
    missing = [column for column in OHLCV if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{symbol}/{source}: colunas ausentes: {', '.join(missing)}")

    result = frame[OHLCV].copy()
    result.index = pd.to_datetime(result.index, utc=True)
    result = result.sort_index()
    result = result[~result.index.duplicated(keep="last")]
    for column in OHLCV:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=OHLCV)
    valid = (
        (result["open"] > 0)
        & (result["high"] > 0)
        & (result["low"] > 0)
        & (result["close"] > 0)
        & (result["volume"] >= 0)
    )
    result = result.loc[valid].copy()
    if result.empty:
        raise RuntimeError(f"{symbol}/{source}: serie vazia depois da validacao")
    return result


def load_mongo_frames() -> dict[str, pd.DataFrame]:
    load_dotenv(ROOT / ".env", override=False)
    mongo_uri = str(os.getenv("TCC_MONGO_URI") or DEFAULT_MONGO_URI).strip()
    database_name = str(os.getenv("TCC_MONGO_DATABASE") or DEFAULT_MONGO_DATABASE).strip()

    lowered = mongo_uri.lower()
    if not any(host in lowered for host in ("localhost", "127.0.0.1", "::1")):
        raise RuntimeError("O experimento aceita somente MongoDB local")

    start_timestamp = pd.Timestamp(START_DATE, tz="UTC").to_pydatetime()
    end_timestamp = (pd.Timestamp(END_DATE, tz="UTC") + pd.Timedelta(days=1)).to_pydatetime()

    log(f"Carregando controle Mongo: {database_name}.{MARKET_COLLECTION}")
    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=3_000,
        connectTimeoutMS=3_000,
        retryWrites=False,
    )
    try:
        client.admin.command("ping")
        rows = list(
            client[database_name][MARKET_COLLECTION].find(
                {
                    "symbol": {"$in": list(ASSETS)},
                    "interval": CONFIG.timeframe,
                    "feed": CONFIG.alpaca_historical_feed,
                    "adjustment": CONFIG.alpaca_adjustment,
                    "timestamp": {"$gte": start_timestamp, "$lt": end_timestamp},
                },
                {
                    "_id": 0,
                    "symbol": 1,
                    "timestamp": 1,
                    "open": 1,
                    "high": 1,
                    "low": 1,
                    "close": 1,
                    "volume": 1,
                },
            ).sort([("symbol", 1), ("timestamp", 1)])
        )
    finally:
        client.close()

    if not rows:
        raise RuntimeError("Nenhum candle foi encontrado no MongoDB local")

    raw = pd.DataFrame(rows)
    raw["symbol"] = raw["symbol"].astype(str).str.upper()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)

    frames: dict[str, pd.DataFrame] = {}
    for symbol, group in raw.groupby("symbol", sort=False):
        frame = group.drop(columns=["symbol"]).set_index("timestamp")
        frames[str(symbol)] = _validate_frame(str(symbol), frame, "mongo")

    missing = [symbol for symbol in ASSETS if symbol not in frames]
    if missing:
        raise RuntimeError("Mongo sem historico para: " + ", ".join(missing))
    return {symbol: frames[symbol] for symbol in ASSETS}


def load_tiingo_raw(symbol: str, tiingo_root: Path) -> pd.DataFrame:
    path = tiingo_root / TIINGO_SERIES_DIRNAME / f"{symbol}.csv"
    if not path.exists():
        raise RuntimeError(f"Serie Tiingo RAW ausente: {path}")
    table = pd.read_csv(path)
    required = ["timestamp", *OHLCV]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise RuntimeError(f"{symbol}/tiingo: colunas ausentes: {', '.join(missing)}")
    table = table[required].copy()
    table["timestamp"] = pd.to_datetime(table["timestamp"], utc=True, errors="coerce")
    table = table.dropna(subset=["timestamp"])
    return _validate_frame(symbol, table.set_index("timestamp"), "tiingo_raw")


def load_splits(symbol: str, tiingo_root: Path) -> pd.DataFrame:
    path = tiingo_root / TIINGO_SPLITS_DIRNAME / f"{symbol}.csv"
    if not path.exists():
        raise RuntimeError(f"Arquivo de splits ausente: {path}")
    table = pd.read_csv(path)
    columns = ["timestamp", "split_de", "split_para", "fator_split", "status"]
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise RuntimeError(f"{symbol}/splits: colunas ausentes: {', '.join(missing)}")
    if table.empty:
        return pd.DataFrame(columns=columns)
    table = table[columns].copy()
    table["timestamp"] = pd.to_datetime(table["timestamp"], utc=True, errors="coerce")
    for column in ("split_de", "split_para", "fator_split"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table["status"] = table["status"].astype(str).str.lower().str.strip()
    table = table.dropna(subset=["timestamp", "split_de", "split_para", "fator_split"])
    table = table.loc[
        (table["split_de"] > 0)
        & (table["split_para"] > 0)
        & (table["fator_split"] > 0)
        & (table["status"] == "a")
    ].copy()
    return table.sort_values("timestamp").drop_duplicates(
        subset=["timestamp", "split_de", "split_para"], keep="last"
    ).reset_index(drop=True)


def load_dividends(symbol: str, tiingo_root: Path) -> pd.DataFrame:
    path = tiingo_root / TIINGO_EVENTS_DIRNAME / f"{symbol}.csv"
    if not path.exists():
        raise RuntimeError(f"Arquivo de eventos ausente: {path}")
    table = pd.read_csv(path)
    if "timestamp" not in table.columns or "dividendo" not in table.columns:
        raise RuntimeError(f"{symbol}/eventos: timestamp/dividendo ausentes")
    if table.empty:
        return pd.DataFrame(columns=["timestamp", "dividendo"])
    table = table[["timestamp", "dividendo"]].copy()
    table["timestamp"] = pd.to_datetime(table["timestamp"], utc=True, errors="coerce")
    table["dividendo"] = pd.to_numeric(table["dividendo"], errors="coerce").fillna(0.0)
    table = table.dropna(subset=["timestamp"])
    table = table.loc[table["dividendo"] != 0.0].copy()
    return table.sort_values("timestamp").drop_duplicates(
        subset=["timestamp"], keep="last"
    ).reset_index(drop=True)


def normalize_splits_causally(
    symbol: str,
    raw: pd.DataFrame,
    splits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    factor_on_session = pd.Series(1.0, index=raw.index, dtype=float)
    for event in splits.itertuples(index=False):
        event_date = pd.Timestamp(event.timestamp).normalize()
        sessions = raw.index[raw.index.normalize() == event_date]
        if len(sessions) != 1:
            raise RuntimeError(f"{symbol}: split {event_date.date()} sem sessao unica")
        session = sessions[0]
        factor = float(event.fator_split)
        expected = float(event.split_para) / float(event.split_de)
        if not np.isclose(factor, expected, rtol=1e-8, atol=1e-10):
            raise RuntimeError(f"{symbol}: fator split inconsistente em {event_date.date()}")
        factor_on_session.loc[session] *= factor

    cumulative = factor_on_session.cumprod()
    result = raw.copy()
    for column in ("open", "high", "low", "close"):
        result[column] = result[column] * cumulative
    result["volume"] = result["volume"] / cumulative
    return result, cumulative


def normalize_dividends_causally(
    symbol: str,
    split_series: pd.DataFrame,
    split_factor: pd.Series,
    dividends: pd.DataFrame,
) -> pd.DataFrame:
    factor_on_session = pd.Series(1.0, index=split_series.index, dtype=float)
    for event in dividends.itertuples(index=False):
        event_date = pd.Timestamp(event.timestamp).normalize()
        sessions = split_series.index[split_series.index.normalize() == event_date]
        if not len(sessions):
            continue
        if len(sessions) != 1:
            raise RuntimeError(f"{symbol}: dividendo {event_date.date()} sem sessao unica")
        session = sessions[0]
        position = int(split_series.index.get_loc(session))
        if position == 0:
            continue
        previous = split_series.index[position - 1]
        previous_close = float(split_series.loc[previous, "close"])
        normalized_dividend = float(event.dividendo) * float(split_factor.loc[session])
        denominator = previous_close - normalized_dividend
        if not np.isfinite(denominator) or denominator <= 0:
            raise RuntimeError(f"{symbol}: dividendo invalido em {event_date.date()}")
        factor_on_session.loc[session] *= previous_close / denominator

    cumulative = factor_on_session.cumprod()
    result = split_series.copy()
    for column in ("open", "high", "low", "close"):
        result[column] = result[column] * cumulative
    return result


def load_tiingo_frames(tiingo_root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    required_dirs = [
        tiingo_root / TIINGO_SERIES_DIRNAME,
        tiingo_root / TIINGO_EVENTS_DIRNAME,
        tiingo_root / TIINGO_SPLITS_DIRNAME,
    ]
    for path in required_dirs:
        if not path.exists():
            raise RuntimeError(
                f"Snapshot Tiingo ausente: {path}. "
                "Mantenha a pasta dados local ao trocar de branch."
            )

    frames: dict[str, pd.DataFrame] = {}
    file_hashes: dict[str, dict[str, str]] = {}
    log(f"Carregando Tiingo total-causal de {tiingo_root}")
    for index, symbol in enumerate(ASSETS, start=1):
        raw_path = tiingo_root / TIINGO_SERIES_DIRNAME / f"{symbol}.csv"
        splits_path = tiingo_root / TIINGO_SPLITS_DIRNAME / f"{symbol}.csv"
        events_path = tiingo_root / TIINGO_EVENTS_DIRNAME / f"{symbol}.csv"

        raw = load_tiingo_raw(symbol, tiingo_root)
        splits = load_splits(symbol, tiingo_root)
        dividends = load_dividends(symbol, tiingo_root)
        split_series, split_factor = normalize_splits_causally(symbol, raw, splits)
        total = normalize_dividends_causally(symbol, split_series, split_factor, dividends)
        frames[symbol] = _validate_frame(symbol, total, "tiingo_total_causal")
        file_hashes[symbol] = {
            "raw": sha256_file(raw_path),
            "splits": sha256_file(splits_path),
            "events": sha256_file(events_path),
        }
        if index == 1 or index % 5 == 0 or index == len(ASSETS):
            log(f"Tiingo {index:02d}/{len(ASSETS)} {symbol} | candles={len(total)}")

    signature_payload = json.dumps(file_hashes, sort_keys=True).encode("utf-8")
    signature = hashlib.sha256(signature_payload).hexdigest()
    return frames, {"combined_sha256": signature, "files": file_hashes}


def run_engine(frames: dict[str, pd.DataFrame]) -> Any:
    results = run_rotation_models(
        frames,
        CONFIG,
        calculate_reference_fees,
        apply_slippage,
        progress_callback=progress_callback,
        technical_log_callback=technical_log_callback,
    )
    if not results or len(results) != 1:
        raise RuntimeError(f"Motor retornou {len(results) if results else 0} resultados")
    return results[0]


def predictions_for_compare(result: Any) -> pd.DataFrame:
    predictions = result.predictions.copy()
    if not isinstance(predictions.index, pd.DatetimeIndex):
        predictions.index = pd.to_datetime(predictions.index, utc=True, errors="coerce")
    elif predictions.index.tz is None:
        predictions.index = predictions.index.tz_localize("UTC")
    else:
        predictions.index = predictions.index.tz_convert("UTC")
    return predictions.sort_index()


def compare_decisions(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> dict[str, Any]:
    if "selected_asset" not in reference.columns or "selected_asset" not in candidate.columns:
        return {"available": False}
    common = reference.index.intersection(candidate.index)
    if common.empty:
        return {"available": False, "reason": "no_common_sessions"}
    left = reference.loc[common, "selected_asset"].astype(str)
    right = candidate.loc[common, "selected_asset"].astype(str)
    different = left != right
    count = int(different.sum())
    first = common[different.to_numpy()][0] if count else None
    payload: dict[str, Any] = {
        "available": True,
        "common_sessions": int(len(common)),
        "divergent_sessions": count,
        "divergence_rate": float(count / len(common)),
        "first_divergence": first,
    }
    if first is not None:
        payload["reference_selected_asset"] = str(left.loc[first])
        payload["candidate_selected_asset"] = str(right.loc[first])
        for column in ("selected_score", "decision_score", "walk_forward_fold"):
            if column in reference.columns:
                payload[f"reference_{column}"] = json_default(reference.loc[first, column])
            if column in candidate.columns:
                payload[f"candidate_{column}"] = json_default(candidate.loc[first, column])
    return payload


def scenario_frames(
    mongo: dict[str, pd.DataFrame],
    tiingo: dict[str, pd.DataFrame],
    direction: str,
    symbol: str | None,
) -> dict[str, pd.DataFrame]:
    if direction == "mongo":
        return {asset: mongo[asset].copy() for asset in ASSETS}
    if direction == "tiingo":
        return {asset: tiingo[asset].copy() for asset in ASSETS}
    if symbol is None:
        raise ValueError("symbol e obrigatorio para substituicao individual")
    if direction == "mongo-to-tiingo":
        result = {asset: mongo[asset].copy() for asset in ASSETS}
        result[symbol] = tiingo[symbol].copy()
        return result
    if direction == "tiingo-to-mongo":
        result = {asset: tiingo[asset].copy() for asset in ASSETS}
        result[symbol] = mongo[symbol].copy()
        return result
    raise ValueError(direction)


def scenario_name(direction: str, symbol: str | None) -> str:
    if direction in ("mongo", "tiingo"):
        return f"all_{direction}"
    return f"{direction}_{symbol}"


def save_result(
    name: str,
    direction: str,
    symbol: str | None,
    result: Any,
    elapsed_seconds: float,
    tiingo_signature: dict[str, Any],
    baseline_predictions: pd.DataFrame | None,
) -> dict[str, Any]:
    metrics = dict(result.metrics)
    predictions = predictions_for_compare(result)
    capital = float(metrics["strategy_ending_capital"])
    comparison = (
        compare_decisions(baseline_predictions, predictions)
        if baseline_predictions is not None and direction != "mongo"
        else None
    )
    payload = {
        "schema_version": 1,
        "script_version": VERSION,
        "certified_tag": CERTIFIED_TAG,
        "certified_commit": CERTIFIED_COMMIT,
        "certified_reference_capital": CERTIFIED_CAPITAL,
        "scenario": name,
        "direction": direction,
        "replaced_symbol": symbol,
        "history_start": START_DATE,
        "history_end": END_DATE,
        "asset_count": len(ASSETS),
        "tiingo_snapshot_signature": tiingo_signature["combined_sha256"],
        "strategy_ending_capital": capital,
        "delta_vs_certified_capital": capital - CERTIFIED_CAPITAL,
        "ratio_vs_certified_capital": capital / CERTIFIED_CAPITAL,
        "elapsed_seconds": elapsed_seconds,
        "decision_divergence_vs_mongo_control": comparison,
        "metrics": metrics,
    }
    path = OUTPUT_DIR / f"{name}.json"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    predictions.to_csv(OUTPUT_DIR / f"{name}_predictions.csv", index=True)
    return payload


def load_baseline_predictions() -> pd.DataFrame | None:
    path = OUTPUT_DIR / "all_mongo_predictions.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path, index_col=0)
    table.index = pd.to_datetime(table.index, utc=True, errors="coerce")
    return table.sort_index()


def execute_scenario(
    mongo: dict[str, pd.DataFrame],
    tiingo: dict[str, pd.DataFrame],
    tiingo_signature: dict[str, Any],
    direction: str,
    symbol: str | None,
    *,
    force: bool,
) -> dict[str, Any]:
    name = scenario_name(direction, symbol)
    result_path = OUTPUT_DIR / f"{name}.json"
    if result_path.exists() and not force:
        log(f"[resume] {name} ja existe; pulando")
        return json.loads(result_path.read_text(encoding="utf-8"))

    baseline_predictions = load_baseline_predictions()
    frames = scenario_frames(mongo, tiingo, direction, symbol)
    log(f"=== CENARIO {name} ===")
    started = time.perf_counter()
    result = run_engine(frames)
    elapsed = time.perf_counter() - started
    payload = save_result(
        name,
        direction,
        symbol,
        result,
        elapsed,
        tiingo_signature,
        baseline_predictions,
    )
    log(
        f"{name} concluido | capital=US$ {payload['strategy_ending_capital']:,.2f} | "
        f"delta_43m=US$ {payload['delta_vs_certified_capital']:,.2f}"
    )
    divergence = payload.get("decision_divergence_vs_mongo_control") or {}
    if divergence.get("available"):
        log(
            f"divergencia vs Mongo: {divergence['divergent_sessions']}/"
            f"{divergence['common_sessions']} | primeira={divergence['first_divergence']}"
        )
    return payload


def write_leaderboard() -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(OUTPUT_DIR.glob("*.json")):
        if path.name in {"campaign.json", "summary.json"}:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "scenario" not in payload:
            continue
        divergence = payload.get("decision_divergence_vs_mongo_control") or {}
        rows.append(
            {
                "scenario": payload["scenario"],
                "direction": payload["direction"],
                "symbol": payload.get("replaced_symbol"),
                "capital": payload["strategy_ending_capital"],
                "delta_vs_43m": payload["delta_vs_certified_capital"],
                "ratio_vs_43m": payload["ratio_vs_certified_capital"],
                "divergent_sessions": divergence.get("divergent_sessions"),
                "divergence_rate": divergence.get("divergence_rate"),
                "first_divergence": divergence.get("first_divergence"),
                "elapsed_seconds": payload.get("elapsed_seconds"),
            }
        )
    if rows:
        pd.DataFrame(rows).sort_values("capital", ascending=False).to_csv(
            OUTPUT_DIR / "leaderboard.csv", index=False
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("baseline", "pair", "mongo-to-tiingo", "tiingo-to-mongo", "both"),
        default="pair",
        help=(
            "baseline=Mongo; pair=Mongo+Tiingo; *_to_*=um ativo por vez; "
            "both=duas direcoes por ativo"
        ),
    )
    parser.add_argument("--symbol", type=str, default=None, help="Executa apenas um ativo")
    parser.add_argument("--limit", type=int, default=0, help="Limita quantidade de ativos")
    parser.add_argument("--force", action="store_true", help="Refaz cenarios ja persistidos")
    parser.add_argument(
        "--tiingo-root",
        type=Path,
        default=TIINGO_ROOT_DEFAULT,
        help="Diretorio local contendo series_historicas/eventos/desdobramentos",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log(f"{VERSION}")
    log(f"Base de codigo certificada: {CERTIFIED_TAG} @ {CERTIFIED_COMMIT[:12]}")
    log("Variavel experimental: apenas frames OHLCV Mongo/Tiingo")
    log("Sem internet e sem runtime/API do Market Cycle Trader")

    mongo = load_mongo_frames()
    tiingo, tiingo_signature = load_tiingo_frames(args.tiingo_root.resolve())

    campaign = {
        "script_version": VERSION,
        "certified_tag": CERTIFIED_TAG,
        "certified_commit": CERTIFIED_COMMIT,
        "certified_reference_capital": CERTIFIED_CAPITAL,
        "tiingo_root": str(args.tiingo_root.resolve()),
        "tiingo_snapshot_signature": tiingo_signature["combined_sha256"],
        "assets": list(ASSETS),
        "selection_uses_43m_reference": False,
        "purpose": "diagnostic_attribution_only",
    }
    (OUTPUT_DIR / "campaign.json").write_text(
        json.dumps(campaign, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )

    # O controle Mongo sempre vem primeiro quando necessario, pois sua curva de
    # decisoes e a referencia para localizar a primeira bifurcacao.
    if args.mode in ("baseline", "pair", "mongo-to-tiingo", "tiingo-to-mongo", "both"):
        execute_scenario(mongo, tiingo, tiingo_signature, "mongo", None, force=args.force)

    if args.mode == "baseline":
        write_leaderboard()
        return 0

    if args.mode == "pair":
        execute_scenario(mongo, tiingo, tiingo_signature, "tiingo", None, force=args.force)
        write_leaderboard()
        return 0

    symbols = [str(args.symbol).upper()] if args.symbol else list(ASSETS)
    invalid = [symbol for symbol in symbols if symbol not in ASSETS]
    if invalid:
        raise RuntimeError("Ativo fora do universo certificado: " + ", ".join(invalid))
    if args.limit > 0:
        symbols = symbols[: args.limit]

    if args.mode in ("mongo-to-tiingo", "both"):
        for symbol in symbols:
            execute_scenario(
                mongo,
                tiingo,
                tiingo_signature,
                "mongo-to-tiingo",
                symbol,
                force=args.force,
            )

    if args.mode in ("tiingo-to-mongo", "both"):
        # O all-Tiingo e util como controle da direcao inversa.
        execute_scenario(mongo, tiingo, tiingo_signature, "tiingo", None, force=args.force)
        for symbol in symbols:
            execute_scenario(
                mongo,
                tiingo,
                tiingo_signature,
                "tiingo-to-mongo",
                symbol,
                force=args.force,
            )

    write_leaderboard()
    log(f"Resultados: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
