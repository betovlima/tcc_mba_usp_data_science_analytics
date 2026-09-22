"""Aquisicao e congelamento do snapshot Alpaca em CSV por ativo."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import pandas as pd
import requests
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import dotenv_values, load_dotenv

from tcc_engine.config import (
    ANALYSIS_END_DATE,
    ASSETS,
    EXPERIMENT_VERSION,
    BAR_SNAPSHOT_AS_OF_END,
    START_DATE,
)

CORPORATE_ACTIONS_ENDPOINT = "https://data.alpaca.markets/v1/corporate-actions"
REQUEST_TYPES = (
    "forward_split",
    "reverse_split",
    "unit_split",
    "cash_dividend",
    "stock_dividend",
    "spin_off",
    "cash_merger",
    "stock_merger",
    "stock_and_cash_merger",
    "redemption",
    "name_change",
    "worthless_removal",
    "rights_distribution",
)
ARRAY_TO_TYPE = {
    "forward_splits": "forward_split",
    "reverse_splits": "reverse_split",
    "unit_splits": "unit_split",
    "cash_dividends": "cash_dividend",
    "stock_dividends": "stock_dividend",
    "spin_offs": "spin_off",
    "cash_mergers": "cash_merger",
    "stock_mergers": "stock_merger",
    "stock_and_cash_mergers": "stock_and_cash_merger",
    "redemptions": "redemption",
    "name_changes": "name_change",
    "worthless_removals": "worthless_removal",
    "rights_distributions": "rights_distribution",
}


@dataclass(frozen=True)
class SnapshotPaths:
    root: Path
    raw_bars: Path
    corporate_actions: Path
    manifest: Path

    @classmethod
    def under(cls, project_root: Path) -> "SnapshotPaths":
        root = project_root / "dados" / "reproducao_v1"
        return cls(
            root=root,
            raw_bars=root / "raw_bars",
            corporate_actions=root / "corporate_actions",
            manifest=root / "manifest.json",
        )

    def ensure(self) -> None:
        self.raw_bars.mkdir(parents=True, exist_ok=True)
        self.corporate_actions.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AlpacaCredentials:
    api_key: str
    secret_key: str
    source_file: str

    @property
    def headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_alpaca_credentials(project_root: Path) -> AlpacaCredentials:
    candidates = [
        project_root / ".env",
        Path.cwd() / ".env",
        project_root.parent / ".env",
    ]
    unique: list[Path] = []
    for item in candidates:
        resolved = item.resolve()
        if resolved not in unique:
            unique.append(resolved)
    env_file = next((path for path in unique if path.exists()), None)
    if env_file is None:
        checked = "\n".join(f"- {path}" for path in unique)
        raise RuntimeError("Arquivo .env nao encontrado. Caminhos verificados:\n" + checked)

    load_dotenv(env_file, override=True)
    values = {
        str(name): str(value or "").strip()
        for name, value in dotenv_values(env_file).items()
    }

    def resolve(*names: str) -> tuple[str, str | None]:
        for name in names:
            value = values.get(name) or os.getenv(name) or ""
            value = str(value).strip()
            if value:
                return value, name
        return "", None

    api_key, key_name = resolve(
        "ALPACA_API_KEY",
        "ALPACA_API_KEY_ID",
        "APCA_API_KEY_ID",
    )
    secret_key, secret_name = resolve(
        "ALPACA_SECRET_KEY",
        "ALPACA_API_SECRET_KEY",
        "APCA_API_SECRET_KEY",
    )
    print(f"[credentials] env={env_file}", flush=True)
    print(
        f"[credentials] key={key_name or 'missing'} secret={secret_name or 'missing'}",
        flush=True,
    )
    if not api_key or not secret_key:
        raise RuntimeError(
            "Credenciais Alpaca ausentes. Configure ALPACA_API_KEY + "
            "ALPACA_SECRET_KEY (ou equivalentes APCA_*)."
        )
    return AlpacaCredentials(api_key, secret_key, str(env_file))


def _normalize_alpaca_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    if isinstance(result.index, pd.MultiIndex):
        names = list(result.index.names)
        if "symbol" in names:
            result = result.xs(symbol, level="symbol")
    result = result.reset_index()
    result.columns = [str(col).lower() for col in result.columns]
    if "symbol" in result.columns:
        result = result.drop(columns=["symbol"])
    if "timestamp" not in result.columns:
        raise RuntimeError(f"{symbol}: timestamp ausente na resposta Alpaca.")
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result = result.sort_values("timestamp")
    result = result.drop_duplicates(subset=["timestamp"], keep="last")
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in result.columns]
    if missing:
        raise RuntimeError(f"{symbol}: colunas ausentes: {', '.join(missing)}")
    optional = [col for col in ("trade_count", "vwap") if col in result.columns]
    result = result[["timestamp", *required, *optional]].copy()
    for column in [*required, *optional]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["timestamp", *required])
    valid = (
        (result["open"] > 0)
        & (result["high"] > 0)
        & (result["low"] > 0)
        & (result["close"] > 0)
        & (result["volume"] >= 0)
    )
    return result.loc[valid].copy()


def download_raw_bars(
    credentials: AlpacaCredentials,
    paths: SnapshotPaths,
    *,
    assets: tuple[str, ...] = ASSETS,
    replace: bool = False,
) -> dict[str, Path]:
    """Baixa SIP/1Day/RAW, um ativo por requisicao, e grava um CSV por ativo."""
    paths.ensure()
    client = StockHistoricalDataClient(
        api_key=credentials.api_key,
        secret_key=credentials.secret_key,
    )
    start = pd.Timestamp(START_DATE, tz="UTC").to_pydatetime()
    bar_end = pd.Timestamp(BAR_SNAPSHOT_AS_OF_END, tz="UTC")
    api_end = (bar_end + pd.Timedelta(days=1)).to_pydatetime()

    print(
        "[alpaca-bars] loader=10.8.74-download_stock_bars "
        f"assets={len(assets)} feed=sip adjustment=raw timeframe=1Day "
        f"start={START_DATE} bar_asof_end={BAR_SNAPSHOT_AS_OF_END} "
        f"research_end={ANALYSIS_END_DATE}",
        flush=True,
    )

    files: dict[str, Path] = {}
    for position, symbol in enumerate(assets, start=1):
        target = paths.raw_bars / f"{symbol}.csv"
        if target.exists() and not replace:
            frame = pd.read_csv(target)
            if frame.empty:
                raise RuntimeError(f"{symbol}: CSV RAW existente esta vazio.")
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            files[symbol] = target
            print(
                f"[alpaca-bars] {position}/{len(assets)} reusing {symbol} "
                f"rows={len(frame)} first={frame['timestamp'].min().date()} "
                f"last={frame['timestamp'].max().date()}",
                flush=True,
            )
            continue

        print(
            f"[alpaca-bars] {position}/{len(assets)} downloading {symbol}...",
            flush=True,
        )
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=api_end,
            adjustment=Adjustment.RAW,
            feed=DataFeed.SIP,
            limit=10_000,
        )
        frame = _normalize_alpaca_frame(client.get_stock_bars(request).df, symbol)
        if frame.empty:
            raise RuntimeError(f"Fresh Alpaca RAW snapshot returned no bars for {symbol}.")
        frame.to_csv(target, index=False, float_format="%.17g")
        files[symbol] = target
        print(
            f"[alpaca-bars] {symbol} rows={len(frame)} "
            f"first={frame['timestamp'].min().date()} "
            f"last={frame['timestamp'].max().date()}",
            flush=True,
        )
    return files


def _request_json(
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, Any],
    max_attempts: int = 6,
) -> dict[str, Any]:
    delay = 2.0
    for attempt in range(1, max_attempts + 1):
        response = requests.get(url, headers=headers, params=params, timeout=60)
        if response.status_code == 200:
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("Resposta de Corporate Actions nao e um objeto JSON.")
            return payload
        if response.status_code == 429 or 500 <= response.status_code < 600:
            if attempt == max_attempts:
                break
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else delay
            print(
                f"[corporate-actions] retry status={response.status_code} "
                f"attempt={attempt}/{max_attempts} wait={wait:.1f}s",
                flush=True,
            )
            time.sleep(wait)
            delay = min(delay * 2.0, 30.0)
            continue
        raise RuntimeError(
            f"Alpaca Corporate Actions HTTP {response.status_code}: {response.text[:500]}"
        )
    raise RuntimeError("Falha ao consultar Corporate Actions apos varias tentativas.")


def _flatten_corporate_actions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    groups = payload.get("corporate_actions") or {}
    if not isinstance(groups, dict):
        return documents
    for array_name, values in groups.items():
        if not isinstance(values, list):
            continue
        action_type = ARRAY_TO_TYPE.get(str(array_name), str(array_name))
        for value in values:
            if isinstance(value, dict):
                item = dict(value)
                item["action_type"] = action_type
                item["source_array"] = str(array_name)
                documents.append(item)
    return documents


def _action_matches_symbol(item: dict[str, Any], symbol: str) -> bool:
    normalized = symbol.strip().upper()
    fields = (
        "symbol",
        "source_symbol",
        "old_symbol",
        "new_symbol",
        "acquirer_symbol",
        "acquiree_symbol",
    )
    return any(
        str(item.get(field) or "").strip().upper() == normalized
        for field in fields
    )


def download_corporate_actions(
    credentials: AlpacaCredentials,
    paths: SnapshotPaths,
    *,
    assets: tuple[str, ...] = ASSETS,
    chunk_size: int = 40,
    replace: bool = False,
) -> dict[str, Path]:
    """Consulta Corporate Actions e persiste um CSV independente por ativo."""
    paths.ensure()
    targets = {symbol: paths.corporate_actions / f"{symbol}.csv" for symbol in assets}
    if not replace and all(path.exists() for path in targets.values()):
        for position, symbol in enumerate(assets, start=1):
            rows = len(pd.read_csv(targets[symbol]))
            print(
                f"[corporate-actions] {position}/{len(assets)} reusing {symbol} rows={rows}",
                flush=True,
            )
        return targets

    research_start = date.fromisoformat(START_DATE)
    query_start = (research_start - timedelta(days=366)).isoformat()
    query_end = ANALYSIS_END_DATE
    symbols = sorted(set(assets))
    resolved_chunk = max(1, min(100, int(chunk_size)))
    documents: list[dict[str, Any]] = []

    for offset in range(0, len(symbols), resolved_chunk):
        chunk = symbols[offset : offset + resolved_chunk]
        page_token: str | None = None
        pages = 0
        while True:
            params: dict[str, Any] = {
                "symbols": ",".join(chunk),
                "types": ",".join(REQUEST_TYPES),
                "start": query_start,
                "end": query_end,
                "region": "us",
                "data_quality": "complete",
                "limit": 1000,
                "sort": "asc",
            }
            if page_token:
                params["page_token"] = page_token
            payload = _request_json(
                CORPORATE_ACTIONS_ENDPOINT,
                headers=credentials.headers,
                params=params,
            )
            pages += 1
            documents.extend(_flatten_corporate_actions(payload))
            page_token = payload.get("next_page_token")
            if not page_token:
                break
        print(
            f"[corporate-actions] progress={min(offset + len(chunk), len(symbols))}/"
            f"{len(symbols)} chunk={chunk[0]}..{chunk[-1]} pages={pages}",
            flush=True,
        )

    sort_fields = (
        "action_type",
        "symbol",
        "source_symbol",
        "process_date",
        "ex_date",
        "effective_date",
        "id",
    )
    documents = sorted(
        documents,
        key=lambda item: tuple(str(item.get(field) or "") for field in sort_fields),
    )

    for symbol in assets:
        rows = [item for item in documents if _action_matches_symbol(item, symbol)]
        target = targets[symbol]
        if rows:
            frame = pd.DataFrame(rows)
            stable_columns = sorted(frame.columns)
            frame = frame[stable_columns]
        else:
            frame = pd.DataFrame(
                columns=[
                    "action_type",
                    "source_array",
                    "symbol",
                    "process_date",
                    "ex_date",
                    "effective_date",
                    "old_rate",
                    "new_rate",
                    "acquiree_symbol",
                    "acquirer_symbol",
                ]
            )
        frame.to_csv(target, index=False)
    return targets


def build_snapshot_manifest(
    paths: SnapshotPaths,
    raw_files: dict[str, Path],
    action_files: dict[str, Path],
    *,
    credentials: AlpacaCredentials | None = None,
) -> dict[str, Any]:
    """Calcula hashes do snapshot; credenciais nunca sao persistidas."""
    file_hashes: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    for symbol, path in sorted(raw_files.items()):
        relative = path.relative_to(paths.root).as_posix()
        file_hashes[relative] = _sha256_file(path)
        row_counts[symbol] = len(pd.read_csv(path))
    action_counts: dict[str, int] = {}
    for symbol, path in sorted(action_files.items()):
        relative = path.relative_to(paths.root).as_posix()
        file_hashes[relative] = _sha256_file(path)
        action_counts[symbol] = len(pd.read_csv(path))

    identity = {
        "schema_version": 1,
        "experiment_version": EXPERIMENT_VERSION,
        "source": "alpaca",
        "bars": {
            "feed": "sip",
            "timeframe": "1Day",
            "adjustment": "raw",
            "start": START_DATE,
            "bar_snapshot_as_of_end": BAR_SNAPSHOT_AS_OF_END,
        },
        "corporate_actions": {
            "query_start": (date.fromisoformat(START_DATE) - timedelta(days=366)).isoformat(),
            "query_end": ANALYSIS_END_DATE,
            "types": list(REQUEST_TYPES),
        },
        "assets": list(ASSETS),
        "row_counts": row_counts,
        "corporate_action_counts": action_counts,
        "file_hashes": file_hashes,
    }
    manifest = dict(identity)
    manifest["snapshot_sha256"] = _canonical_sha256(identity)
    manifest["credential_source"] = credentials.source_file if credentials else None
    paths.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(f"[snapshot] id={manifest['snapshot_sha256']}", flush=True)
    return manifest


def validate_snapshot(paths: SnapshotPaths) -> dict[str, Any]:
    if not paths.manifest.exists():
        raise RuntimeError("manifest.json nao encontrado; crie o snapshot primeiro.")
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))

    expected_snapshot_sha = str(manifest.get("snapshot_sha256") or "")
    identity = {
        key: value
        for key, value in manifest.items()
        if key not in {"snapshot_sha256", "credential_source"}
    }
    actual_snapshot_sha = _canonical_sha256(identity)
    if not expected_snapshot_sha or actual_snapshot_sha != expected_snapshot_sha:
        raise RuntimeError(
            "Snapshot manifest identity mismatch: "
            f"expected={expected_snapshot_sha or 'missing'} "
            f"actual={actual_snapshot_sha}"
        )

    for relative, expected in (manifest.get("file_hashes") or {}).items():
        path = paths.root / relative
        if not path.exists():
            raise RuntimeError(f"Snapshot incompleto: {relative}")
        actual = _sha256_file(path)
        if actual != str(expected):
            raise RuntimeError(
                f"Snapshot integrity mismatch: {relative} expected={expected} actual={actual}"
            )
    print(f"[snapshot] validated id={manifest.get('snapshot_sha256')}", flush=True)
    return manifest
