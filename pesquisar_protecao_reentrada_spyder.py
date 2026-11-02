"""Pesquisa exploratoria: take-profit 1,5%, 3% e 5% com CASH e reentrada D+1.

Usa a reproducao v1.23.0, com dados e parametros congelados.
NAO altera o simulador original, o treinamento, a calibracao ou o TCC entregue.
Execute no Spyder (F5) ou: python pesquisar_take_profit_alvos_spyder.py
"""
from __future__ import annotations

# %% 0. Dependencias e parametros da pesquisa
from dataclasses import dataclass
from pathlib import Path
import json
import math
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "pesquisa_protecao_reentrada"
TARGETS = (0.015, 0.03, 0.05)
BASELINE_EXPECTED = 587_165_941.1330212
RESEARCH_VERSION = "reentry-confirmation-independent-stop-v1"
ALLOW_SAME_DAY_TP_ON_ENTRY = True


@dataclass
class Replay:
    curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, Any]


def _buy(cash: float, raw_open: float, cfg: Any, fees_fn: Callable, slip: Callable):
    price = float(slip(raw_open, "BUY", cfg))
    if not math.isfinite(price) or price <= 0:
        raise ValueError("Preco de compra invalido")
    quantity = cash / price
    for _ in range(25):  # Mesma iteracao de taxas do Control
        fees = fees_fn("BUY", quantity, price, cfg)
        nxt = max(0., (cash - float(fees["total_fee"])) / price)
        if abs(nxt - quantity) < 1e-10:
            quantity = nxt
            break
        quantity = nxt
    fees = fees_fn("BUY", quantity, price, cfg)
    return quantity, price, float(fees["total_fee"])


def simular_alvo(
    policy: Callable,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    decision_dates: pd.DatetimeIndex,
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    target: float | None,
    fold_map: dict[pd.Timestamp, int] | None = None,
    *,
    reentry_confirmations: int = 1,
    stop_loss: float | None = None,
) -> Replay:
    """Replay integral com decisões originais e estado real da nova carteira.

    Score na data D (fechamento), ordem de entrada/rotacao na abertura D+1.
    Ordem limite hipotetica: abertura >= alvo -> venda na abertura; caso
    contrario, maxima >= alvo -> venda no alvo. Sem compra no dia da venda TP.
    No fechamento do mesmo dia, a politica usa o novo estado CASH. A entrada
    seguinte e na abertura D+1, podendo comprar exatamente o mesmo ativo.
    """
    if target is not None and not 0 < float(target) < 1:
        raise ValueError("Alvo precisa estar entre zero e um")
    if type(reentry_confirmations) is not int or reentry_confirmations < 1:
        raise ValueError("reentry_confirmations precisa ser >= 1")
    if stop_loss is not None and not 0 < float(stop_loss) < 1:
        raise ValueError("stop_loss precisa estar entre zero e um")
    dates = pd.DatetimeIndex(decision_dates)
    if len(dates) < 2 or not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("Calendario temporal invalido")
    cash, position, qty, held = float(config.initial_capital), 0, 0., 0
    entry_price, entry_date = math.nan, None
    total_fees, targets, rotations = 0., 0, 0
    last_tp_symbol, last_tp_date, same_symbol_reentries = None, None, 0
    last_forced_symbol, last_forced_date = None, None
    consecutive_reentry_signals, stop_exits, ambiguous_bars, blocked_reentries = 0, 0, 0, 0
    trades, rows = [], []

    def price(symbol: str, timestamp: pd.Timestamp, key: str) -> float:
        try:
            value = float(frames[symbol].loc[timestamp, key])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Barra ausente: {symbol} {timestamp} {key}") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Preco nao positivo: {symbol} {timestamp} {key}")
        return value

    def sell(timestamp, raw_price, reason, fold):
        nonlocal cash, position, qty, held, entry_price, entry_date, total_fees
        symbol = symbols[position - 1]
        executed = float(slippage(raw_price, "SELL", config))
        fees = float(fee_calculator("SELL", qty, executed, config)["total_fee"])
        proceeds = qty * executed - fees
        cash += proceeds
        total_fees += fees
        trades.append(dict(timestamp=timestamp, fold=fold, action="SELL", asset=symbol,
                           reason=reason, price=executed, shares=qty,
                           entry_date=entry_date, entry_price=entry_price,
                           gross_return=executed / entry_price - 1.,
                           fees=fees, cash_after=cash))
        position, qty, held = 0, 0., 0
        entry_price, entry_date = math.nan, None

    def buy(timestamp, target_position, fold):
        nonlocal cash, position, qty, held, entry_price, entry_date
        nonlocal total_fees, same_symbol_reentries
        symbol = symbols[target_position - 1]
        quantity, executed, fees = _buy(
            cash, price(symbol, timestamp, "open"),
            config, fee_calculator, slippage,
        )
        if quantity <= 0:
            raise RuntimeError("Quantidade de compra nao positiva")
        cash -= quantity * executed + fees
        total_fees += fees
        position, qty, held = target_position, quantity, 1
        entry_price, entry_date = executed, timestamp
        if last_tp_symbol == symbol and last_tp_date is not None and timestamp > last_tp_date:
            same_symbol_reentries += 1
        trades.append(dict(timestamp=timestamp, fold=fold, action="BUY", asset=symbol,
                           reason="MODEL_ENTRY", price=executed, shares=qty,
                           entry_date=timestamp, entry_price=executed,
                           gross_return=0., fees=fees, cash_after=cash))

    for i in range(len(dates) - 1):
        decision_day = dates[i]
        session = dates[i + 1]
        fold = (fold_map or {}).get(pd.Timestamp(decision_day))
        previous_position = position

        # Este callback sempre consulta os mesmos scores de cada fold;
        # seu estado depende da carteira realmente produzida no replay.
        target_position, score = policy(decision_day, position, held)
        target_position = int(target_position)
        if not 0 <= target_position <= len(symbols):
            raise RuntimeError("Indice de ativo fora do universo")
        original_target_position = target_position

        # Reentrada: nova confirmacao com sinais de fechamentos sucessivos.
        # O sinal continua vindo do modelo existente, sem olhar o pregão seguinte.
        # Alternativas sem bloqueio podem entrar imediatamente.
        if position == 0 and last_forced_symbol is not None:
            proposed = symbols[target_position - 1] if target_position else "CASH"
            if proposed == last_forced_symbol and decision_day >= last_forced_date:
                consecutive_reentry_signals += 1
                if consecutive_reentry_signals < reentry_confirmations:
                    target_position = 0
                    blocked_reentries += 1
            else:
                consecutive_reentry_signals = 0

        tp_today, entered_today, stop_today = False, False, False
        # Ordem limite ja ativa na abertura, inclusive quando havia rotacao
        # programada. Um gap positivo fecha a posicao e cancela nova compra D.
        if position and (target is not None or stop_loss is not None):
            symbol = symbols[position - 1]
            opening = price(symbol, session, "open")
            stop_line = entry_price * (1. - stop_loss) if stop_loss is not None else None
            tp_line = entry_price * (1. + target) if target is not None else None
            if stop_line is not None and opening <= stop_line:
                sell(session, opening, "STOP_GAP_OPEN", fold)
                stop_exits += 1
                stop_today = True
                last_forced_symbol, last_forced_date = symbol, session
                consecutive_reentry_signals = 0
            elif tp_line is not None and opening >= tp_line:
                sell(session, opening, "TP_GAP_OPEN", fold)
                targets += 1
                tp_today = True
                last_tp_symbol, last_tp_date = symbol, session
                last_forced_symbol, last_forced_date = symbol, session
                consecutive_reentry_signals = 0

        if not tp_today and not stop_today and position != target_position:
            if position:
                symbol = symbols[position - 1]
                sell(session, price(symbol, session, "open"), "MODEL_ROTATE_OR_CASH", fold)
            if target_position:
                buy(session, target_position, fold)
                entered_today = True
            if previous_position and target_position:
                rotations += 1
        elif not tp_today and not stop_today and position:
            held += 1

        # O preco maximo so executa a ordem ja existente. Nao decide ativo
        # com informacao intradia, nem reabre posicao no mesmo pregao.
        if position and not tp_today and not stop_today:
            symbol = symbols[position - 1]
            stop_line = entry_price * (1. - stop_loss) if stop_loss is not None else None
            tp_line = entry_price * (1. + target) if target is not None else None
            low_hit = (stop_line is not None
                       and price(symbol, session, "low") <= stop_line)
            high_hit = (tp_line is not None and
                        (ALLOW_SAME_DAY_TP_ON_ENTRY or not entered_today) and
                        price(symbol, session, "high") >= tp_line)
            # Sem ticks, nao sabemos qual nivel foi atingido primeiro.
            # Assumir STOP primeiro e a escolha conservadora.
            if low_hit and high_hit:
                ambiguous_bars += 1
            if low_hit:
                sell(session, stop_line, "STOP_INTRADAY", fold)
                stop_exits += 1
                stop_today = True
                last_forced_symbol, last_forced_date = symbol, session
                consecutive_reentry_signals = 0
            elif high_hit:
                sell(session, tp_line, "TP_LIMIT", fold)
                targets += 1
                tp_today = True
                last_tp_symbol, last_tp_date = symbol, session
                last_forced_symbol, last_forced_date = symbol, session
                consecutive_reentry_signals = 0

        if position:
            selected = symbols[position - 1]
            equity = cash + qty * price(selected, session, "close")
        else:
            selected, equity = "CASH", cash
        if not math.isfinite(equity) or equity <= 0:
            raise RuntimeError(f"Patrimonio invalido em {session}")
        rows.append(dict(timestamp=session, decision_date=decision_day, fold=fold,
                         strategy_equity=equity, selected_asset=selected,
                         in_cash=not bool(position), tp_fired=tp_today,
                         stop_fired=stop_today,
                         proposed_model_choice=("CASH" if original_target_position == 0
                                                else symbols[original_target_position-1]),
                         model_choice="CASH" if target_position == 0 else symbols[target_position-1],
                         score=float(score)))

    if position:
        sell(dates[-1], price(symbols[position-1], dates[-1], "close"),
             "FINAL_LIQUIDATION", rows[-1]["fold"])
        rows[-1]["strategy_equity"] = cash

    curve = pd.DataFrame(rows).set_index("timestamp")
    trade_frame = pd.DataFrame(trades)
    wealth = curve["strategy_equity"].to_numpy(float)
    ret = wealth / np.r_[float(config.initial_capital), wealth[:-1]] - 1.
    sigma = float(ret.std(ddof=1)) if len(ret) > 1 else 0.
    sharpe = float(ret.mean() / sigma * np.sqrt(252.)) if sigma else None
    path = np.r_[float(config.initial_capital), wealth]
    drawdown = path / np.maximum.accumulate(path) - 1.
    fold_rows = []
    for fold_id, frame in curve.groupby("fold", sort=True):
        idx = curve.index.get_indexer([frame.index[0]])[0]
        starting = float(config.initial_capital) if idx == 0 else float(wealth[idx-1])
        ending = float(frame["strategy_equity"].iloc[-1])
        trajectory = np.r_[starting, frame["strategy_equity"].to_numpy(float)]
        fold_rows.append(dict(fold=int(fold_id), start=starting, end=ending,
                              return_pct=100 * (ending / starting - 1),
                              maximum_drawdown_pct=100 * float(
                                  (trajectory / np.maximum.accumulate(trajectory) - 1.).min()
                              ), sessions=len(frame)))
    metrics = dict(target_pct=None if target is None else 100 * target,
                   stop_loss_pct=None if stop_loss is None else 100 * stop_loss,
                   reentry_confirmations=reentry_confirmations,
                   stop_exits=stop_exits, ambiguous_intraday_bars=ambiguous_bars,
                   blocked_same_asset_reentries=blocked_reentries,
                   initial_capital=float(config.initial_capital),
                   ending_capital=float(wealth[-1]),
                   return_pct=100 * (wealth[-1] / float(config.initial_capital) - 1.),
                   maximum_drawdown_pct=100 * float(drawdown.min()), sharpe=sharpe,
                   cash_sessions=int(curve["in_cash"].sum()),
                   tp_exits=targets, buys=int((trade_frame["action"] == "BUY").sum()),
                   sells=int((trade_frame["action"] == "SELL").sum()),
                   rotations=rotations, total_fees=float(total_fees),
                   same_asset_reentries_after_tp=same_symbol_reentries,
                   sessions=len(curve), folds=fold_rows,
                   intraday_fill_model="daily_OHLC_stop_first_if_both_levels_hit",
                   same_day_reentry=False, next_day_same_asset_reentry=True,
                   cash_yield=0.0)
    return Replay(curve=curve, trades=trade_frame, metrics=metrics)


# %% 1. Validacao da base original e comparacao cientifica
def executar_pesquisa() -> dict[str, Any]:
    from reproducao.experimento import reproduzir_universo_congelado
    import pesquisar_expansao_u67_spyder as pesquisa

    source_simulator = pesquisa._simular_exato
    scenarios: dict[str, Replay] = {}
    provenance: dict[str, Any] = {}
    expected = json.loads((ROOT / "evidencias" / "manifest_universo_congelado.json")
                          .read_text(encoding="utf-8"))
    if not math.isclose(float(expected["expected_ending_capital"]),
                        BASELINE_EXPECTED, rel_tol=0, abs_tol=0.01):
        raise RuntimeError("Identidade do checkpoint mudou. Auditar manifesto")

    def intercept(backend, policy, frames, symbols, decision_dates, config,
                  fee_calculator, slippage, *args, **kwargs):
        native = source_simulator(
            backend, policy, frames, symbols, decision_dates, config,
            fee_calculator, slippage, *args, **kwargs,
        )
        if backend != "tcc_u67_asset_expansion":
            return native
        fold_map = {
            pd.Timestamp(k): int(v["fold_id"])
            for k, v in (kwargs.get("decision_metadata") or {}).items()
            if v.get("fold_id") is not None
        }
        common = dict(policy=policy, frames=frames, symbols=symbols,
                      decision_dates=decision_dates, config=config,
                      fee_calculator=fee_calculator, slippage=slippage,
                      fold_map=fold_map)
        control = simular_alvo(target=None, **common)
        observed_native = native.predictions["strategy_equity"]
        observed = control.curve["strategy_equity"]
        if not observed_native.index.equals(observed.index):
            raise RuntimeError("Indice de datas nao coincide com o controle")
        max_delta = float(np.max(np.abs(observed_native.to_numpy(float) -
                                         observed.to_numpy(float))))
        final_native = float(native.metrics["strategy_ending_capital"])
        if abs(final_native - BASELINE_EXPECTED) > 0.01:
            raise RuntimeError("Checkpoint original nao reproduzido")
        if max_delta > 0.02:
            raise RuntimeError(f"Replay sem TP nao reproduz engine: max_diff={max_delta:.8f}")
        scenarios["control"] = control
        for pct in TARGETS:
            key = f"tp_{int(round(pct * 1000)):03d}"
            scenarios[key] = simular_alvo(target=pct, **common)
            scenarios[f"{key}_confirm2"] = simular_alvo(
                target=pct, reentry_confirmations=2, **common,
            )
            scenarios[f"{key}_stop10"] = simular_alvo(
                target=pct, stop_loss=0.10, **common,
            )
            scenarios[f"{key}_confirm2_stop10"] = simular_alvo(
                target=pct, reentry_confirmations=2, stop_loss=0.10, **common,
            )
        scenarios["stop10_only"] = simular_alvo(
            target=None, stop_loss=0.10, **common,
        )
        provenance.update(dict(native_capital=final_native,
                               maximum_equity_parity_error=max_delta,
                               data_signature=expected["snapshot_sha256"],
                               universe_assets=list(symbols),
                               historical_end=expected["analysis_end_date"],
                               source_reproduction="TCC v1.23.0 U67 + selected 10",
                               research_version=RESEARCH_VERSION))
        return native

    # Troca temporaria da referencia apenas neste processo; sempre restaura.
    pesquisa._simular_exato = intercept
    try:
        original_payload = reproduzir_universo_congelado(
            root=ROOT, output=OUT / "reproducao_controle",
            version="1.23.0", execution_schema="universe-reproduction-v3",
        )
    finally:
        pesquisa._simular_exato = source_simulator
    if len(scenarios) != len(TARGETS) * 4 + 2:
        raise RuntimeError("Cenarios nao foram capturados")

    # %% 2. Exportacao e graficos, somente depois de reproduzir o controle
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for key, result in scenarios.items():
        result.curve.to_csv(OUT / f"{key}_patrimonio.csv", index_label="timestamp")
        result.trades.to_csv(OUT / f"{key}_operacoes.csv", index=False)
        summary.append(dict(scenario=key, **{
            k: v for k, v in result.metrics.items() if k != "folds"
        }))
    pd.DataFrame(summary).to_csv(OUT / "comparacao.csv", index=False)
    with (OUT / "resultado.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(provenance=provenance, original_reproduction={
            "checkpoint": original_payload.get("checkpoint"),
            "reproduced_exact_to_cent": original_payload.get("reproduced_exact_to_cent"),
        }, scenarios={key: r.metrics for key, r in scenarios.items()}),
                  handle, ensure_ascii=False, indent=2, default=str)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 6))
    for key, r in scenarios.items():
        ax.plot(r.curve.index, r.curve["strategy_equity"],
                label=key.replace("_", " "), linewidth=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("Patrimonio (US$, escala log)")
    ax.set_title("TCC: confirmar reentrada e controlar perda por posicao")
    ax.legend()
    ax.grid(alpha=.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "comparacao_patrimonio.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for key, r in scenarios.items():
        values = r.curve["strategy_equity"].astype(float)
        dd = values.div(values.cummax()).sub(1.0).mul(100)
        ax.plot(r.curve.index, dd, label=key.replace("_", " "), linewidth=1.1)
    ax.set_ylabel("Queda desde o pico (%)")
    ax.set_title("Drawdown por variante: realizacao e protecao")
    ax.legend()
    ax.grid(alpha=.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "comparacao_drawdown.png", dpi=160)
    plt.close(fig)
    print(pd.DataFrame(summary)[[
        "scenario", "ending_capital", "maximum_drawdown_pct",
        "cash_sessions", "tp_exits", "stop_exits", "total_fees",
    ]].to_string(index=False), flush=True)
    print(f"[PESQUISA] Resultados em {OUT}", flush=True)
    return dict(provenance=provenance, scenarios=scenarios)


# %% 3. Execucao Spyder
if __name__ == "__main__":
    executar_pesquisa()
