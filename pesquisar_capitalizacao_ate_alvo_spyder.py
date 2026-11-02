"""Pesquisa: manter cada posicao ate um lucro-alvo e reinvestir.

Experimento independente (TCC, sem novos downloads):
  1. Control original, com rotacoes normais;
  2. TP 1,5%, 3%, 5% com rotacoes normais, para comparacao;
  3. TP 1,5%, 3%, 5% sem rotacoes, segurando ate o alvo.

Requer os arquivos locais congelados da reproducao oficial do TCC.
No Spyder: reiniciar o kernel e executar com F5.
"""
from __future__ import annotations

# %% 0 - Parametros
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import json
import math

import numpy as np
import pandas as pd

from pesquisar_take_profit_alvos_spyder import (
    BASELINE_EXPECTED,
    _buy,
    simular_alvo,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "pesquisa_capitalizacao_ate_alvo"
TARGETS = (0.015, 0.03, 0.05)
VERSION = "compound-hold-until-target-v1"


@dataclass
class HoldResult:
    curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, Any]


# %% 1 - Simulador independente: nao vende por troca de ranking
def simular_ate_alvo(
    policy: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    decision_dates: pd.DatetimeIndex,
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    target: float,
    fold_map: dict[pd.Timestamp, int] | None = None,
) -> HoldResult:
    """Politica de manter ate o alvo, com reaplicacao integral dos recursos.

    Decisoes usam SOMENTE fechamentos D. Se em CASH, a politica seleciona
    entrada para a abertura D+1, sem restricao ao ticker vendido antes.
    Se posicionado, a escolha da politica nao provoca rotacao; a ordem
    limite de venda fica vigente ate ser atingida. Quando a ordem executa,
    CASH permanece ate o encerramento do pregao (sem recompra intradia).
    Caso nunca atinja o alvo, a posicao so e liquidada no fechamento final
    para calcular patrimonio e taxas de modo comparavel ao Control.

    A ordem limite nao promete liquidez: OHLC diario revela apenas que
    o nivel foi alcancado, nao que toda a carteira poderia ser vendida.
    """
    target = float(target)
    if not math.isfinite(target) or not (0.0 < target < 1.0):
        raise ValueError("target deve estar entre zero e um")
    dates = pd.DatetimeIndex(decision_dates)
    if len(dates) < 2 or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Calendario invalido")
    if len(symbols) == 0 or len(symbols) != len(set(symbols)):
        raise ValueError("Universo de ativos invalido")

    cash = float(config.initial_capital)
    if not math.isfinite(cash) or cash <= 0:
        raise ValueError("Capital inicial invalido")
    pos = 0
    qty = 0.0
    entry_price = math.nan
    entry_date = None
    holding = 0
    previous_tp_symbol = None
    previous_tp_date = None
    fees_paid = 0.0
    tp_count = 0
    max_hold = 0
    same_symbol_reentries = 0
    total_hold_sessions = 0
    total_equity_trough = math.inf
    trades: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    def quote(sym: str, day: pd.Timestamp, field: str) -> float:
        try:
            v = float(frames[sym].loc[day, field])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Historico indisponivel: {sym} {day} {field}") from exc
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError(f"Preco invalido: {sym} {day} {field}={v}")
        return v

    def sell(day: pd.Timestamp, raw: float, reason: str, fold: int | None):
        nonlocal cash, pos, qty, entry_price, entry_date, holding, fees_paid
        nonlocal max_hold, total_hold_sessions
        if pos <= 0:
            raise RuntimeError("Venda sem posicao")
        sym = symbols[pos - 1]
        execution = float(slippage(raw, "SELL", config))
        fee = float(fee_calculator("SELL", qty, execution, config)["total_fee"])
        gross_return = execution / entry_price - 1.0
        cash += qty * execution - fee
        fees_paid += fee
        if cash <= 0:
            raise RuntimeError("Capital zerado apos venda")
        max_hold = max(max_hold, holding)
        total_hold_sessions += holding
        trades.append({
            "timestamp": day, "fold": fold, "action": "SELL", "asset": sym,
            "reason": reason, "price": execution, "shares": qty,
            "entry_price": entry_price, "entry_date": entry_date,
            "position_return": gross_return, "holding_sessions": holding,
            "fees": fee, "cash_after_trade": cash,
        })
        pos, qty, holding = 0, 0.0, 0
        entry_price, entry_date = math.nan, None

    def buy(day: pd.Timestamp, target_pos: int, fold: int | None):
        nonlocal cash, pos, qty, entry_price, entry_date, holding, fees_paid
        nonlocal same_symbol_reentries
        sym = symbols[target_pos - 1]
        units, execution, fee = _buy(
            cash, quote(sym, day, "open"), config, fee_calculator, slippage
        )
        if units <= 0 or not math.isfinite(units):
            raise RuntimeError("Quantidade de compra invalida")
        cash -= units * execution + fee
        if cash < -0.02:
            raise RuntimeError("Saldo negativo")
        fees_paid += fee
        pos, qty, holding = target_pos, units, 1
        entry_price, entry_date = execution, day
        if previous_tp_symbol == sym and previous_tp_date is not None and day > previous_tp_date:
            same_symbol_reentries += 1
        trades.append({
            "timestamp": day, "fold": fold, "action": "BUY", "asset": sym,
            "reason": "MODEL_FROM_CASH", "price": execution,
            "shares": units, "entry_price": execution, "entry_date": day,
            "position_return": 0., "holding_sessions": 0,
            "fees": fee, "cash_after_trade": cash,
        })

    for i in range(len(dates) - 1):
        signal_day = dates[i]
        trade_day = dates[i + 1]
        fold = (fold_map or {}).get(pd.Timestamp(signal_day))
        was_invested = pos > 0
        target_fired = False
        selected_by_model = None

        # A politica so e consultada quando ha capital em CASH.
        # O preco de trade_day nao interfere nesta decisao.
        if not was_invested:
            candidate, score = policy(signal_day, 0, 0)
            candidate = int(candidate)
            if not 0 <= candidate <= len(symbols):
                raise RuntimeError("Posicao indicada fora do universo")
            selected_by_model = "CASH" if candidate == 0 else symbols[candidate - 1]
            if candidate > 0:
                buy(trade_day, candidate, fold)
        else:
            holding += 1

        if pos > 0:
            sym = symbols[pos - 1]
            threshold = entry_price * (1.0 + target)
            opening = quote(sym, trade_day, "open")
            max_price = quote(sym, trade_day, "high")
            # Ordem limitada hipotetica ja presente na abertura da sessao.
            # A abertura pode ser mais favoravel que o preco-limite.
            if opening >= threshold:
                sell(trade_day, opening, "TP_GAP_OPEN", fold)
                target_fired = True
            elif max_price >= threshold:
                sell(trade_day, threshold, "TP_LIMIT", fold)
                target_fired = True
            if target_fired:
                tp_count += 1
                previous_tp_symbol = sym
                previous_tp_date = trade_day

        if pos > 0:
            sym = symbols[pos - 1]
            capital = cash + qty * quote(sym, trade_day, "close")
            selected = sym
        else:
            capital = cash
            selected = "CASH"
        if not math.isfinite(capital) or capital <= 0:
            raise RuntimeError(f"Patrimonio nao positivo em {trade_day}")

        total_equity_trough = min(total_equity_trough, capital)
        rows.append({
            "timestamp": trade_day, "decision_date": signal_day,
            "fold": fold, "strategy_equity": capital, "selected_asset": selected,
            "in_cash": pos == 0, "tp_fired": target_fired,
            "model_choice_from_cash": selected_by_model,
            "holding_sessions": holding,
            "unrealized_return": (
                quote(sym, trade_day, "close") / entry_price - 1.
                if pos > 0 else 0.
            ),
        })

    final_liquidation = False
    if pos > 0:
        last = dates[-1]
        sell(last, quote(symbols[pos - 1], last, "close"),
             "FINAL_LIQUIDATION", rows[-1]["fold"])
        final_liquidation = True
        rows[-1]["strategy_equity"] = cash
        rows[-1]["selected_asset"] = "CASH_FINAL"
        rows[-1]["in_cash"] = False

    curve = pd.DataFrame(rows).set_index("timestamp")
    trade_df = pd.DataFrame(trades)
    equity = curve["strategy_equity"].to_numpy(dtype=float)
    initial = float(config.initial_capital)
    returns = equity / np.r_[initial, equity[:-1]] - 1.
    vol = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.
    sharpe = float(np.mean(returns) / vol * np.sqrt(252.)) if vol else None
    path = np.r_[initial, equity]
    drawdown = path / np.maximum.accumulate(path) - 1.
    fold_rows = []
    for fold_id, group in curve.groupby("fold", sort=True):
        if fold_id is None or pd.isna(fold_id):
            continue
        first = curve.index.get_indexer([group.index[0]])[0]
        start = initial if first == 0 else float(equity[first - 1])
        fold_path = np.r_[start, group["strategy_equity"].to_numpy(float)]
        fold_rows.append({
            "fold": int(fold_id),
            "sessions": len(group),
            "starting_capital": start,
            "ending_capital": float(fold_path[-1]),
            "fold_return": float(fold_path[-1] / start - 1.),
            "maximum_drawdown": float(
                (fold_path / np.maximum.accumulate(fold_path) - 1.).min()
            ),
        })

    sell_df = trade_df[trade_df["action"] == "SELL"]
    losing = sell_df[sell_df["position_return"] < 0.0]
    metrics = {
        "target_pct": 100. * target,
        "initial_capital": initial,
        "ending_capital": float(equity[-1]),
        "total_return_pct": 100. * (float(equity[-1]) / initial - 1.),
        "maximum_drawdown_pct": 100. * float(drawdown.min()),
        "sharpe": sharpe, "sessions": len(curve),
        "buys": int((trade_df["action"] == "BUY").sum()),
        "sells": int(len(sell_df)), "target_exits": tp_count,
        "other_exits": int(len(sell_df) - tp_count),
        "losing_exits": int(len(losing)),
        "final_liquidation": final_liquidation,
        "cash_sessions": int(curve["in_cash"].sum()),
        "exposure_sessions": int((~curve["in_cash"]).sum()),
        "max_hold_sessions": int(max_hold),
        "mean_completed_hold_sessions": (
            float(total_hold_sessions / len(sell_df)) if len(sell_df) else None
        ),
        "same_asset_reentries_after_tp": same_symbol_reentries,
        "total_transaction_fees": float(fees_paid),
        "folds": fold_rows, "no_forced_rotation": True,
        "no_stop_loss": True, "cash_yield": 0.,
        "intraday_fill_assumption": "daily_OHLC_limit_with_open_gap",
        "no_same_session_reentry_after_tp": True,
        "compounding_identity": "all_realized_proceeds_reinvested",
        "theoretical_all_wins_same_exit_count": (
            initial * (1. + target) ** tp_count if tp_count < 10000 else None
        ),
    }
    return HoldResult(curve=curve, trades=trade_df, metrics=metrics)


# %% 2 - Reutilizar o treinamento e os resultados da reproducao original
def executar_pesquisa() -> dict[str, Any]:
    from reproducao.experimento import reproduzir_universo_congelado
    import pesquisar_expansao_u67_spyder as pesquisa

    expected = json.loads(
        (ROOT / "evidencias" / "manifest_universo_congelado.json")
        .read_text(encoding="utf-8")
    )
    if not math.isclose(
        float(expected["expected_ending_capital"]),
        BASELINE_EXPECTED, rel_tol=0., abs_tol=.01
    ):
        raise RuntimeError("Manifesto nao corresponde ao checkpoint")
    original_simulator = pesquisa._simular_exato
    scenarios: dict[str, Any] = {}
    provenance: dict[str, Any] = {}

    def intercept(backend, policy, frames, symbols, decision_dates, config,
                  fee_calculator, slippage, *args, **kwargs):
        native = original_simulator(
            backend, policy, frames, symbols, decision_dates, config,
            fee_calculator, slippage, *args, **kwargs
        )
        if backend != "tcc_u67_asset_expansion":
            return native
        fold_map = {
            pd.Timestamp(k): int(v["fold_id"])
            for k, v in (kwargs.get("decision_metadata") or {}).items()
            if v.get("fold_id") is not None
        }
        common = dict(
            policy=policy, frames=frames, symbols=symbols,
            decision_dates=decision_dates, config=config,
            fee_calculator=fee_calculator, slippage=slippage,
            fold_map=fold_map,
        )
        replay_control = simular_alvo(target=None, **common)
        native_curve = native.predictions["strategy_equity"]
        if not native_curve.index.equals(replay_control.curve.index):
            raise RuntimeError("Calendario de replay diferente do Control")
        parity = float(np.max(np.abs(
            native_curve.to_numpy(float) -
            replay_control.curve["strategy_equity"].to_numpy(float)
        )))
        if parity > .02:
            raise RuntimeError(f"Replay Control nao passou: {parity:.8f}")
        native_final = float(native.metrics["strategy_ending_capital"])
        if not math.isclose(
            native_final, BASELINE_EXPECTED, rel_tol=0., abs_tol=.01
        ):
            raise RuntimeError("Checkpoint original nao foi reproduzido")

        scenarios["control"] = replay_control
        for pct in TARGETS:
            name = f"tp_{int(round(pct * 1000)):03d}"
            scenarios[f"{name}_com_rotacao"] = simular_alvo(
                target=pct, **common
            )
            scenarios[f"{name}_ate_alvo"] = simular_ate_alvo(
                target=pct, **common
            )
        provenance.update({
            "version": VERSION, "original_capital": native_final,
            "maximum_control_parity_difference": parity,
            "snapshot_sha256": expected["snapshot_sha256"],
            "selected_universe": list(symbols),
            "analysis_end_date": expected["analysis_end_date"],
            "fixed_historical_folds": len(set(fold_map.values())),
            "models_retrained_for_target": False,
        })
        return native

    pesquisa._simular_exato = intercept
    try:
        official = reproduzir_universo_congelado(
            root=ROOT, output=OUT / "reproducao_controle",
            version="1.23.0",
            execution_schema="universe-reproduction-v3",
        )
    finally:
        pesquisa._simular_exato = original_simulator
    if len(scenarios) != 1 + len(TARGETS) * 2:
        raise RuntimeError("Faltam cenarios")

    # %% 3 - Saidas e auditoria de capitalizacao
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for key, result in scenarios.items():
        result.curve.to_csv(
            OUT / f"{key}_patrimonio.csv", index_label="timestamp"
        )
        result.trades.to_csv(
            OUT / f"{key}_operacoes.csv", index=False
        )
        m = result.metrics
        summary.append({
            "scenario": key,
            "ending_capital": float(m["ending_capital"]),
            "maximum_drawdown_pct": float(m["maximum_drawdown_pct"]),
            "sharpe": m.get("sharpe"),
            "cash_sessions": int(m["cash_sessions"]),
            "target_exits": int(m.get("target_exits", m.get("tp_exits", 0))),
            "other_exits": m.get("other_exits"),
            "losing_exits": m.get("losing_exits"),
            "max_hold_sessions": m.get("max_hold_sessions"),
            "mean_completed_hold_sessions": m.get("mean_completed_hold_sessions"),
            "theoretical_all_wins_same_exit_count": m.get(
                "theoretical_all_wins_same_exit_count"
            ),
            "total_transaction_fees": m.get(
                "total_transaction_fees", m.get("total_fees")
            ),
        })
    pd.DataFrame(summary).to_csv(OUT / "comparacao.csv", index=False)
    payload = {
        "provenance": provenance,
        "official_reproduction": {
            "checkpoint": official.get("checkpoint"),
            "reproduced_exact_to_cent": official.get(
                "reproduced_exact_to_cent"
            ),
        },
        "scenarios": {k: v.metrics for k, v in scenarios.items()},
    }
    (OUT / "resultado.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    import matplotlib.pyplot as plt
    for metric in ("capital", "drawdown"):
        fig, ax = plt.subplots(figsize=(11, 6))
        for key, result in scenarios.items():
            values = result.curve["strategy_equity"].astype(float)
            if metric == "drawdown":
                values = 100. * (values / values.cummax() - 1.)
            ax.plot(values.index, values, label=key, linewidth=1.1)
        if metric == "capital":
            ax.set_yscale("log")
            ax.set_ylabel("Patrimonio US$ (escala logaritmica)")
        else:
            ax.set_ylabel("Drawdown (%)")
        ax.set_title("Capitalizacao: Control, TP com rotacao e manter ate alvo")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(OUT / f"comparacao_{metric}.png", dpi=160)
        plt.close(fig)

    print(
        pd.DataFrame(summary)[[
            "scenario", "ending_capital", "maximum_drawdown_pct",
            "cash_sessions", "target_exits", "max_hold_sessions",
        ]].to_string(index=False), flush=True
    )
    print(f"[concluido] resultados em {OUT}", flush=True)
    return payload


# %% 4 - Execucao Spyder
if __name__ == "__main__":
    executar_pesquisa()
