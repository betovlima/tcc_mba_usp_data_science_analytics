"""v1.23.9 — diagnostico offline dos 11 pregoes de outubro e integridade RAW.

No Spyder: F5. Le o pacote U75 existente e gera CSV/JSON de diagnostico;
NAO baixa novas series, nao reotimiza, nao altera snapshots ou main.
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parent
VERSION = "1.23.9"
START = "2026-10-06"
END = "2026-10-21"
OUTPUT_DIR = ROOT / "output" / "diagnostico_decisoes_outubro_2026"


def select_11_sessions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "strategy_equity", "selected_asset", "decision_score"}
    if missing := (required - set(predictions)):
        raise RuntimeError(f"Predicoes sem campos obrigatorios: {sorted(missing)}")
    series = predictions.copy()
    series["day"] = pd.to_datetime(series["timestamp"], utc=True).dt.strftime("%Y-%m-%d")
    series = series.loc[series["day"].between(START, END)].copy()
    if len(series) != 12 or series["day"].nunique() != 12:
        raise RuntimeError("Esperada a sessao-base de 06/10 e 11 novas sessoes.")
    series["portfolio_return_pct"] = series["strategy_equity"].pct_change() * 100
    series["daily_pnl_usd"] = series["strategy_equity"].diff()
    series["next_session_equity_return_pct"] = series["portfolio_return_pct"].shift(-1)
    return series


def paired_completed_positions(trades: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """Usa somente BUY em outubro e sua venda efetiva, sem alvos futuros."""
    required = {"timestamp", "action", "asset", "execution_price", "position_return", "realized_pnl"}
    if missing := (required - set(trades)):
        raise RuntimeError(f"Negociacoes sem campos obrigatorios: {sorted(missing)}")
    series = trades.copy()
    series["day"] = pd.to_datetime(series["timestamp"], utc=True).dt.strftime("%Y-%m-%d")
    buys = series[(series["day"].between(START, END)) & (series["action"] == "BUY")]
    sells = series[
        (series["day"].between(START, END))
        & (series["action"].isin(("SELL", "FINAL_SELL")))
    ]
    rows = []
    for _, buy in buys.sort_values("timestamp").iterrows():
        subsequent = sells.loc[
            (sells["asset"] == buy["asset"])
            & (pd.to_datetime(sells["timestamp"], utc=True)
               > pd.Timestamp(buy["timestamp"]))
        ].sort_values("timestamp")
        if subsequent.empty:
            raise RuntimeError(f"Posicao sem saida observavel: {buy['asset']} em {buy['day']}.")
        sold = subsequent.iloc[0]
        chosen = sessions.loc[sessions["day"] == buy["day"]]
        if len(chosen) != 1 or chosen.iloc[0]["selected_asset"] != buy["asset"]:
            raise RuntimeError(f"Score da entrada inconsistente em {buy['day']}.")
        rows.append({
            "entry_day": buy["day"], "exit_day": sold["day"],
            "asset": buy["asset"],
            "score_at_entry": float(chosen.iloc[0]["decision_score"]),
            "entry_price": float(buy["execution_price"]),
            "exit_price": float(sold["execution_price"]),
            "realized_position_return_pct": 100 * float(sold["position_return"]),
            "realized_pnl_usd": float(sold["realized_pnl"]),
            "rotation_value_added_pct": (
                100 * float(buy["rotation_value_added"])
                if "rotation_value_added" in series and pd.notna(buy["rotation_value_added"])
                else None
            ),
        })
    return pd.DataFrame(rows)


def run(*, package: Path | None = None, output_dir: Path = OUTPUT_DIR) -> dict:
    if package is None:
        packages = sorted(
            (ROOT / "output" / "fresh_universe_75").glob(
                "asof_*/pacote_comparacao_cutoffs_75.zip"
            ), reverse=True,
        )
        if not packages:
            raise FileNotFoundError(
                "Pacote U75 nao encontrado; execute antes "
                "executar_atualizacao_series_spyder.py."
            )
        package = packages[0]
    package, output_dir = Path(package).resolve(), Path(output_dir).resolve()
    with zipfile.ZipFile(package) as archive:
        predictions = pd.read_csv(BytesIO(
            archive.read("cutoff_2026-10-21/predictions.csv")
        ))
        trades = pd.read_csv(BytesIO(
            archive.read("cutoff_2026-10-21/trades.csv")
        ))
        result = json.loads(archive.read(
            "cutoff_2026-10-21/reproducao_universo.json"
        ))
    data = result["data"]
    if (data.get("feed"), data.get("adjustment"), data.get("timeframe")) != (
        "sip", "raw", "1Day"
    ):
        raise RuntimeError("Comparacao exige Alpaca SIP, adjustment raw e 1Day.")
    if result["universe"]["effective_count"] != 75:
        raise RuntimeError("Universo atual deve conter exatamente 75 ativos.")
    sessions = select_11_sessions(predictions)
    positions = paired_completed_positions(trades, sessions)
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(output_dir / "sessoes.csv", index=False)
    positions.to_csv(output_dir / "posicoes_encerradas.csv", index=False)
    first = float(sessions.iloc[0]["strategy_equity"])
    last = float(sessions.iloc[-1]["strategy_equity"])
    report = {
        "schema": "tcc-recent-decision-diagnostics-v1",
        "version": VERSION,
        "branch": "research/v1.23.1-fresh-snapshot-20261022",
        "source_package": package.name,
        "source_package_sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
        "source_snapshot_sha256": data["snapshot_sha256"],
        "feed": "sip", "adjustment": "raw",
        "splits_applied_by_pipeline": data.get("data_audit", {}).get("actual_total_splits_applied"),
        "dividend_price_adjustment": False,
        "universe": 75,
        "start_date": START, "end_date": END,
        "additional_sessions": len(sessions)-1,
        "initial_marked_capital": first, "ending_capital": last,
        "delta_usd": last-first,
        "delta_pct": 100*(last/first-1),
        "worst_session": sessions.loc[
            sessions["portfolio_return_pct"].idxmin(), "day"
        ],
        "worst_session_pct": float(sessions["portfolio_return_pct"].min()),
        "completed_positions_opened_inside_window": len(positions),
        "negative_completed_positions": int(
            (positions["realized_position_return_pct"] < 0).sum()
        ),
        "warning": (
            "decision_score e utilidade multihorizonte, nao probabilidade "
            "calibrada ou retorno futuro garantido; sem vetor completo "
            "de scores dos 75 ativos nao medir ranking das alternativas."
        ),
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "resumo.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    archive_path = output_dir / "pacote_diagnostico_decisoes_outubro_2026.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in ("sessoes.csv", "posicoes_encerradas.csv", "resumo.json"):
            archive.write(output_dir / item, arcname=item)
    print(
        f"[october] capital inicial=US$ {first:,.2f} final=US$ {last:,.2f} "
        f"variacao={report['delta_pct']:+.2f}% | {archive_path}",
        flush=True,
    )
    return report


if __name__ == "__main__":
    run()
