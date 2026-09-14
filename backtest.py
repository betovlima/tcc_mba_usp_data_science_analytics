"""Reproducible USP MBA capstone backtest.

Open this file in Spyder and press F5, or run:

    python backtest.py

The experiment is intentionally frozen. It does not search for assets and does
not tune the strategy. It replays the certified 37-asset portfolio using the
exact historical Market Cycle Trader source commit that produced the reference
result and the local MongoDB market snapshot.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULT_FILE = OUTPUT_DIR / "backtest_result.json"

# ---------------------------------------------------------------------------
# Frozen academic reference
# ---------------------------------------------------------------------------
REFERENCE_REPOSITORY = "https://github.com/betovlima/market_cycle_trader_api.git"
REFERENCE_COMMIT = "17019d95bfce6f0fbcd153e097b1968d9cfce1ca"
REFERENCE_STRATEGY_SEQUENCE = 10
REFERENCE_STRATEGY_ID = "strategy-87713a05860748719ec18d0a086dcce7"
REFERENCE_STRATEGY_REVISION = 15
REFERENCE_STRATEGY_CONFIGURATION_SHA256 = (
    "509b940659a89a7348be3690882213c839ce1a43b7e44057656074f5b2517a6e"
)
REFERENCE_HISTORY_START = "2016-01-01"
REFERENCE_SNAPSHOT_END = "2026-09-04"
REFERENCE_INITIAL_CAPITAL = 10_000.0
REFERENCE_ENDING_CAPITAL = 43_759_854.82
REFERENCE_CAGR = 2.9382
REFERENCE_SHARPE = 2.557
REFERENCE_MAX_DRAWDOWN = -0.2819

# Final 37-asset universe obtained by the already-finished historical pruning
# experiment. This project never searches or changes this list.
REFERENCE_ASSETS = (
    "NVDA", "MSFT", "META", "TSLA", "AMD", "JPM", "SPY", "AVGO", "NFLX",
    "ORCL", "COST", "LLY", "XOM", "CAT", "WMT", "V", "HD", "ADC", "ADEA",
    "ADI", "ADM", "GKOS", "VNCE", "CORT", "UNFI", "DNN", "MKSI", "APD",
    "DDS", "RACE", "UNF", "TX", "CEF", "YANG", "KKR", "BXMT", "SCSC",
)

# Ending capital is the primary certification metric. The historical result has
# reproduced to near machine precision; the tolerance permits only tiny runtime
# numeric differences, not a materially different backtest.
ENDING_CAPITAL_RELATIVE_TOLERANCE = 1e-6

# Local source resolution. Set MCT_REFERENCE_SOURCE only if auto-detection does
# not find the existing historical worktree. If nothing is found, the script
# clones the exact reference commit into .mct_reference (git + internet needed
# only for that bootstrap path).
LOCAL_REFERENCE_DIR = PROJECT_ROOT / ".mct_reference"

# MongoDB is local by default. Values can be overridden in a local .env without
# changing this source file.
DEFAULT_MONGO_URI = "mongodb://localhost:27017"
DEFAULT_MONGO_DATABASE = "extrema_backtest"


PACKAGE_NAMES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "threadpoolctl",
    "exchange-calendars",
    "lightgbm",
    "xgboost",
    "pymongo",
    "python-dotenv",
    "pydantic",
    "alpaca-py",
)


def _log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _canonical_sha256(value: Any) -> str:
    # This exactly mirrors the canonicalization used when Strategy #10 was
    # checked before the certified reproduction run.
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_head(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip().lower() or None


def _looks_like_reference_source(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "src" / "market_cycle_trader_api").is_dir()
        and (path / "scripts" / "research_sequential_exact_marginal_search.py").is_file()
        and (path / "scripts" / "research_sequential_exact_marginal_search_v110.py").is_file()
    )


def _candidate_source_paths() -> list[Path]:
    candidates: list[Path] = []
    explicit = str(os.getenv("MCT_REFERENCE_SOURCE") or "").strip()
    if explicit:
        candidates.append(Path(explicit).expanduser())

    candidates.append(LOCAL_REFERENCE_DIR)

    # Common layouts used by this project and by the historical validation
    # worktree. No user-specific absolute path is committed.
    roots = [PROJECT_ROOT, *list(PROJECT_ROOT.parents)[:5]]
    for root in roots:
        candidates.extend(
            [
                root / "mct_exact_pruning_validation",
                root / "market_cycle_trader" / "mct_exact_pruning_validation",
                root / "market_cycle_trader_api",
            ]
        )

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate.absolute())
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _bootstrap_reference_source() -> Path:
    if LOCAL_REFERENCE_DIR.exists() and not _looks_like_reference_source(LOCAL_REFERENCE_DIR):
        raise RuntimeError(
            f"{LOCAL_REFERENCE_DIR} exists but is not a valid MCT source checkout. "
            "Remove or rename it before retrying."
        )

    if not LOCAL_REFERENCE_DIR.exists():
        _log("Frozen MCT source not found locally; cloning the pinned reference commit.")
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                REFERENCE_REPOSITORY,
                str(LOCAL_REFERENCE_DIR),
            ],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(LOCAL_REFERENCE_DIR), "checkout", "--detach", REFERENCE_COMMIT],
        check=True,
    )
    return LOCAL_REFERENCE_DIR


def resolve_reference_source() -> Path:
    wrong_commit: list[tuple[Path, str | None]] = []
    for candidate in _candidate_source_paths():
        if not _looks_like_reference_source(candidate):
            continue
        head = _git_head(candidate)
        if head == REFERENCE_COMMIT:
            _log(f"Frozen MCT source: {candidate}")
            return candidate.resolve()
        wrong_commit.append((candidate, head))

    try:
        source = _bootstrap_reference_source()
    except Exception as exc:
        details = "; ".join(
            f"{path} @ {head or 'unknown'}" for path, head in wrong_commit[:5]
        )
        raise RuntimeError(
            "Could not resolve the exact historical MCT source commit. "
            f"Expected {REFERENCE_COMMIT}. Nearby checkouts: {details or 'none'}. "
            "Set MCT_REFERENCE_SOURCE to the detached historical worktree if needed."
        ) from exc

    head = _git_head(source)
    if head != REFERENCE_COMMIT:
        raise RuntimeError(
            f"Reference source checkout mismatch: expected {REFERENCE_COMMIT}, got {head}."
        )
    _log(f"Frozen MCT source bootstrapped: {source}")
    return source.resolve()


def install_reference_imports(source_root: Path) -> None:
    for path in (source_root / "src", source_root / "scripts"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def market_frames_sha256(frames: dict[str, pd.DataFrame]) -> str:
    """Create a deterministic fingerprint of the exact OHLCV input frames."""
    digest = hashlib.sha256()
    for symbol in sorted(frames):
        frame = frames[symbol].sort_index()
        digest.update(symbol.encode("utf-8"))
        digest.update(b"\n")
        for timestamp, row in frame.iterrows():
            ts = pd.Timestamp(timestamp)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            values = [
                ts.isoformat(),
                *[
                    "" if pd.isna(row.get(column)) else format(float(row.get(column)), ".17g")
                    for column in ("open", "high", "low", "close", "volume")
                ],
            ]
            digest.update(("|".join(values) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _write_result(payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = RESULT_FILE.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(RESULT_FILE)


def _success_sound() -> None:
    if platform.system().lower() != "windows":
        return
    try:
        import winsound

        winsound.MessageBeep(winsound.MB_OK)
    except Exception:
        pass


def run_backtest() -> dict[str, Any]:
    started = time.perf_counter()
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    _log("USP MBA Data Science & Analytics — reproducible backtest")
    _log("Mode: frozen replay only; asset search and tuning are disabled")

    source_root = resolve_reference_source()
    install_reference_imports(source_root)

    # Imports happen only after the historical source tree has been pinned in
    # sys.path. This prevents accidentally executing code from another MCT
    # version installed elsewhere on the machine.
    import research_asset_signature_leave_one_out as common
    import research_sequential_exact_marginal_search as sequential
    import research_sequential_exact_marginal_search_v110 as accelerator
    from market_cycle_trader_api.schemas.requests import BacktestRequest

    mongo_uri = str(
        os.getenv("TCC_MONGO_URI")
        or os.getenv("MONGO_URL")
        or os.getenv("MONGO_URI")
        or DEFAULT_MONGO_URI
    ).strip()
    database_name = str(
        os.getenv("TCC_MONGO_DATABASE")
        or os.getenv("MONGO_DATABASE")
        or DEFAULT_MONGO_DATABASE
    ).strip()

    common._assert_local_mongo(mongo_uri, allow_remote=False)
    _log(f"MongoDB database: {database_name}")

    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=3_000,
        connectTimeoutMS=3_000,
        retryWrites=False,
    )
    try:
        client.admin.command("ping")
        db = client[database_name]
        strategy = common._strategy_document(
            db,
            REFERENCE_STRATEGY_SEQUENCE,
            REFERENCE_STRATEGY_ID,
        )
        strategy_id = str(strategy.get("_id") or "")
        strategy_revision = int(strategy.get("revision") or 0)
        stored_configuration = common._configuration(strategy)
        configuration_hash = _canonical_sha256(stored_configuration)

        _log(
            f"Strategy: #{REFERENCE_STRATEGY_SEQUENCE} | id={strategy_id} | "
            f"revision={strategy_revision}"
        )
        _log(f"Strategy configuration SHA-256: {configuration_hash}")

        if strategy_id != REFERENCE_STRATEGY_ID:
            raise RuntimeError(
                f"Strategy id mismatch: expected {REFERENCE_STRATEGY_ID}, got {strategy_id}."
            )
        if configuration_hash != REFERENCE_STRATEGY_CONFIGURATION_SHA256:
            raise RuntimeError(
                "Strategy configuration changed since certification. "
                f"Expected {REFERENCE_STRATEGY_CONFIGURATION_SHA256}, got {configuration_hash}."
            )
        if str(stored_configuration.get("start_date")) != REFERENCE_HISTORY_START:
            raise RuntimeError(
                "Strategy start date changed since certification: "
                f"expected {REFERENCE_HISTORY_START}, got {stored_configuration.get('start_date')}."
            )

        configured_assets = {
            str(value).strip().upper()
            for value in list(stored_configuration.get("assets") or [])
            if str(value).strip()
        }
        missing_assets = sorted(set(REFERENCE_ASSETS).difference(configured_assets))
        if missing_assets:
            raise RuntimeError(
                "Frozen 37-asset universe is not contained in Strategy #10: "
                + ", ".join(missing_assets)
            )

        config = BacktestRequest.model_validate(stored_configuration).model_copy(
            update={
                "assets": list(REFERENCE_ASSETS),
                "end_date": REFERENCE_SNAPSHOT_END,
            }
        )
        history_start = common._normalize_date(REFERENCE_HISTORY_START)
        snapshot_end = common._normalize_date(REFERENCE_SNAPSHOT_END)
        identity = common._market_identity(stored_configuration)
        expected_sessions = common._expected_sessions(history_start, snapshot_end)
        collection = db[common.ALPACA_MARKET_BARS_COLLECTION]

        # The historical exact-search accelerator changes performance only; its
        # parity guard was previously validated against the uncached replay.
        accelerator.install_v110(parity_mode=False)

        frames: dict[str, pd.DataFrame] = {}
        _log(
            f"Loading {len(REFERENCE_ASSETS)} frozen assets from local MongoDB "
            f"({REFERENCE_HISTORY_START} -> {REFERENCE_SNAPSHOT_END})"
        )
        for position, symbol in enumerate(REFERENCE_ASSETS, start=1):
            frame, coverage = sequential._load_local_frame(
                collection,
                symbol,
                identity,
                history_start,
                snapshot_end,
                config,
                expected_sessions,
            )
            frames[symbol] = frame
            if position == 1 or position % 5 == 0 or position == len(REFERENCE_ASSETS):
                _log(
                    f"Market data {position:02d}/{len(REFERENCE_ASSETS)} | {symbol} | "
                    f"rows={len(frame)} | complete={bool(coverage.get('history_complete', True))}"
                )

        market_hash = market_frames_sha256(frames)
        _log(f"Market OHLCV SHA-256: {market_hash}")

        request = sequential._exact_request(
            db,
            config,
            strategy_id,
            list(REFERENCE_ASSETS),
            list(REFERENCE_ASSETS),
            [],
            history_start,
            snapshot_end,
        )
        request_hash = _canonical_sha256(request.model_dump(mode="json"))
        _log(f"Execution request SHA-256: {request_hash}")

        _log("Starting exact frozen walk-forward replay")
        replay_started = time.perf_counter()
        metrics, decision_sessions = sequential.discovery._run_rotation_replay(
            frames,
            request,
        )
        replay_seconds = time.perf_counter() - replay_started

        ending_capital = float(metrics["ending_capital"])
        cagr = float(metrics["cagr"])
        sharpe = float(metrics["sharpe"])
        maximum_drawdown = float(metrics["maximum_drawdown"])
        capital_relative_error = abs(
            ending_capital - REFERENCE_ENDING_CAPITAL
        ) / REFERENCE_ENDING_CAPITAL
        certified = bool(capital_relative_error <= ENDING_CAPITAL_RELATIVE_TOLERANCE)

        _log("Backtest completed")
        _log(f"Ending capital : US$ {ending_capital:,.2f}")
        _log(f"CAGR           : {cagr:.4%}")
        _log(f"Sharpe         : {sharpe:.6f}")
        _log(f"Max Drawdown   : {maximum_drawdown:.4%}")
        _log(f"Switches       : {metrics.get('switches')}")
        _log(f"Cash days      : {metrics.get('cash_days')}")
        _log(f"Worst fold     : {metrics.get('worst_fold_return')}")
        _log(f"Replay time    : {replay_seconds:.1f}s")
        _log(
            "Certification  : "
            + ("PASS" if certified else "FAIL")
            + f" | capital relative error={capital_relative_error:.3e}"
        )

        payload = {
            "schema_version": 1,
            "status": "certified" if certified else "mismatch",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "project": "tcc_mba_usp_data_science_analytics",
            "reference": {
                "repository": REFERENCE_REPOSITORY,
                "source_commit": REFERENCE_COMMIT,
                "strategy_id": REFERENCE_STRATEGY_ID,
                "strategy_sequence": REFERENCE_STRATEGY_SEQUENCE,
                "strategy_revision_at_certification": REFERENCE_STRATEGY_REVISION,
                "strategy_configuration_sha256": REFERENCE_STRATEGY_CONFIGURATION_SHA256,
                "history_start": REFERENCE_HISTORY_START,
                "snapshot_end": REFERENCE_SNAPSHOT_END,
                "assets": list(REFERENCE_ASSETS),
                "asset_count": len(REFERENCE_ASSETS),
                "initial_capital": REFERENCE_INITIAL_CAPITAL,
                "expected_ending_capital": REFERENCE_ENDING_CAPITAL,
                "expected_cagr_approx": REFERENCE_CAGR,
                "expected_sharpe_approx": REFERENCE_SHARPE,
                "expected_maximum_drawdown_approx": REFERENCE_MAX_DRAWDOWN,
            },
            "runtime_fingerprint": {
                "source_path": str(source_root),
                "source_commit": _git_head(source_root),
                "strategy_revision": strategy_revision,
                "strategy_configuration_sha256": configuration_hash,
                "market_ohlcv_sha256": market_hash,
                "execution_request_sha256": request_hash,
                "python_executable": sys.executable,
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "packages": package_versions(),
            },
            "result": {
                **metrics,
                "ending_capital_relative_error_vs_reference": capital_relative_error,
                "certification_tolerance": ENDING_CAPITAL_RELATIVE_TOLERANCE,
                "certification_passed": certified,
                "decision_session_count": int(len(decision_sessions)),
                "decision_session_start": (
                    pd.Timestamp(decision_sessions[0]).date().isoformat()
                    if len(decision_sessions)
                    else None
                ),
                "decision_session_end": (
                    pd.Timestamp(decision_sessions[-1]).date().isoformat()
                    if len(decision_sessions)
                    else None
                ),
                "replay_seconds": replay_seconds,
                "total_seconds": time.perf_counter() - started,
            },
        }
        _write_result(payload)
        _log(f"Stable result artifact: {RESULT_FILE}")

        if not certified:
            raise RuntimeError(
                "Backtest did not reproduce the certified ending capital. "
                "Do not use this run in the TCC until the source/config/data fingerprints are reconciled."
            )

        _success_sound()
        return payload
    finally:
        client.close()


def main() -> int:
    run_backtest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
