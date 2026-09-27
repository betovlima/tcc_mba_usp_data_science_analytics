"""Auditoria de contribuicao marginal dos overrides OOF, sem tuning.

[TCC-DFL:FIX-004] Replays exatos por episodios sequenciais com MESMO passado,
MESMA politica Control e MESMOS custos. Compara a trajetoria sem episodio
contra a trajetoria com episodio, depois ambas voltam ao Control. Os efeitos
marginais sao condicionais aos episodios anteriores; nao sao causalidade de
mercado nem somas independentes de intervenções. Nenhum modelo e retreinado.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from engine.execucao import aplicar_deslizamento, calcular_taxas_referencia
from engine.rotacao import RotationRunResult, _simular_exato


@dataclass
class AuditoriaIntervencoes:
    baseline_result: RotationRunResult
    episodios: pd.DataFrame
    curvas: pd.DataFrame
    pares_diarios: pd.DataFrame
    operacoes: pd.DataFrame
    checks: dict[str, Any]


def _episodios(
    decision_dates: pd.DatetimeIndex,
    original: dict[pd.Timestamp, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Agrupa overrides em sessoes contiguas, nao por diferenca calendaria."""
    dates = pd.DatetimeIndex(decision_dates)
    index = {pd.Timestamp(d): i for i, d in enumerate(dates[:-1])}
    entries = []
    for key, diag in original.items():
        now = pd.Timestamp(key)
        if not bool(diag["research_changed_base_action"]):
            continue
        if now not in index:
            raise ValueError(f"Override fora das datas OOS: {now}")
        if diag["final_action_asset"] == diag["research_base_action"]:
            raise AssertionError(f"Override inconsistente em {now}")
        entries.append((index[now], now, diag))
    entries.sort(key=lambda row: row[0])
    groups: list[list[tuple[int, pd.Timestamp, dict[str, Any]]]] = []
    for item in entries:
        if not groups or item[0] != groups[-1][-1][0] + 1:
            groups.append([])
        groups[-1].append(item)
    return [
        {
            "episode_id": f"EP{group_id:03d}",
            "start": group[0][1],
            "end": group[-1][1],
            "dates": tuple(item[1] for item in group),
            "fold_id": int(group[0][2]["decision_fold_id"]),
            "initial_state": str(group[0][2]["current_asset"]),
            "control_action": str(group[0][2]["research_base_action"]),
            "learned_action": str(group[0][2]["final_action_asset"]),
            "scores": tuple(float(item[2]["research_model_score"]) for item in group),
        }
        for group_id, group in enumerate(groups, start=1)
    ]


def _validar_equivalencia(
    a: RotationRunResult,
    b: RotationRunResult,
    *,
    context: str,
    rtol: float = 1e-10,
    atol: float = 1e-5,
) -> float:
    """Confere curva COMPLETA e estado escolhido, nao apenas capital final."""
    left = a.predictions
    right = b.predictions
    if not left.index.equals(right.index):
        raise AssertionError(f"{context}: calendarios de replay diferentes")
    equity_a = left["strategy_equity"].to_numpy(dtype=float)
    equity_b = right["strategy_equity"].to_numpy(dtype=float)
    max_abs = float(np.max(np.abs(equity_a - equity_b)))
    if not np.allclose(equity_a, equity_b, rtol=rtol, atol=atol):
        pos = int(np.argmax(np.abs(equity_a - equity_b)))
        raise AssertionError(
            f"{context}: curvas diferem em {left.index[pos]} "
            f"({equity_a[pos]} vs {equity_b[pos]})"
        )
    if not left["selected_asset"].fillna("").equals(
        right["selected_asset"].fillna("")
    ):
        raise AssertionError(f"{context}: ativos selecionados divergem")
    return max_abs


def auditar_intervencoes(
    *,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    decision_dates: pd.DatetimeIndex,
    config: Any,
    baseline_policy: Callable,
    learned_policy: Callable,
    original_decisions: dict[pd.Timestamp, dict[str, Any]],
    original_regression: RotationRunResult,
    official_control: RotationRunResult | None = None,
    decision_metadata: dict[pd.Timestamp, dict[str, Any]] | None = None,
    horizon: int = 20,
) -> AuditoriaIntervencoes:
    """Bridge cronologica Control -> EP001 -> EP002 -> ... -> Regression.

    Cada par inclui os mesmos episodios PREVIOS, diverge somente no episodio
    avaliado e usa Control para todas as sessoes futuras. Assim, diferencas
    finais telescopam exatamente; episodios nao sao tratados independentes.
    """
    dates = pd.DatetimeIndex(decision_dates)
    if horizon < 1 or len(dates) < horizon + 2:
        raise ValueError("Historico insuficiente para auditoria")
    episodes = _episodios(dates, original_decisions)
    if not episodes:
        raise ValueError("Regressao nao fez overrides: nenhum episodio")
    all_override_dates = {
        pd.Timestamp(d) for ep in episodes for d in ep["dates"]
    }
    if len(all_override_dates) != sum(len(ep["dates"]) for ep in episodes):
        raise AssertionError("Um override consta em mais de um episodio")
    results: list[tuple[str, RotationRunResult]] = []
    gate: set[pd.Timestamp] = set()

    def simulate(label: str, permitted: set[pd.Timestamp]) -> RotationRunResult:
        seen: set[pd.Timestamp] = set()

        def policy(now: pd.Timestamp, position: int, holding: int):
            key = pd.Timestamp(now)
            if key not in permitted:
                return baseline_policy(now, position, holding)
            state = "CASH" if position == 0 else symbols[position - 1]
            diagnostic = original_decisions[key]
            if state != str(diagnostic["current_asset"]):
                raise AssertionError(
                    f"{label}: estado nao reproduzido em {key}: "
                    f"{state} vs {diagnostic['current_asset']}"
                )
            base, _ = baseline_policy(now, position, holding)
            base_asset = "CASH" if base == 0 else symbols[base - 1]
            if base_asset != str(diagnostic["research_base_action"]):
                raise AssertionError(
                    f"{label}: acao base difere em {key}: "
                    f"{base_asset} vs {diagnostic['research_base_action']}"
                )
            learned, score = learned_policy(now, position, holding)
            learned_asset = "CASH" if learned == 0 else symbols[learned - 1]
            if learned_asset != str(diagnostic["final_action_asset"]):
                raise AssertionError(
                    f"{label}: acao aprendida difere em {key}: "
                    f"{learned_asset} vs {diagnostic['final_action_asset']}"
                )
            seen.add(key)
            return learned, score

        output = _simular_exato(
            "research_override_audit_" + label.lower(),
            policy, frames, symbols, dates, config,
            calcular_taxas_referencia, aplicar_deslizamento,
            decision_metadata=decision_metadata,
            model_label="Counterfactual audit " + label,
            method_line=(
                "- Retain earlier overrides, toggle current episode, "
                "then follow Control. No model retraining."
            ),
        )
        if seen != permitted:
            raise AssertionError(
                f"{label}: datas de intervencao nao executadas: "
                f"{sorted(permitted - seen)}"
            )
        return output

    initial = simulate("BASELINE", gate)
    results.append(("BASELINE", initial))
    control_delta = None
    if official_control is not None:
        control_delta = _validar_equivalencia(
            initial, official_control, context="baseline vs Control oficial",
            rtol=1e-8, atol=1e-4,
        )
    for ep in episodes:
        gate.update(ep["dates"])
        results.append((ep["episode_id"], simulate(ep["episode_id"], set(gate))))
    final_delta = _validar_equivalencia(
        results[-1][1], original_regression,
        context="bridge final vs regressao OOF original",
    )

    curves: list[pd.DataFrame] = []
    trades: list[pd.DataFrame] = []
    for label, result in results:
        p = result.predictions.reset_index()
        cols = [
            "timestamp", "strategy_equity", "selected_asset",
            "trade_action", "walk_forward_fold",
        ]
        if not set(cols).issubset(p.columns):
            raise ValueError(f"{label}: campos ausentes na curva exata")
        part = p[cols].copy()
        part.insert(0, "scenario", label)
        curves.append(part)
        t = result.trades.copy()
        t.insert(0, "scenario", label)
        trades.append(t)

    summaries: list[dict[str, Any]] = []
    daily: list[pd.DataFrame] = []
    for i, ep in enumerate(episodes, start=1):
        prev_label, prev = results[i - 1]
        next_label, after = results[i]
        left = prev.predictions
        right = after.predictions
        start = pd.Timestamp(ep["start"])
        if not left.loc[left.index <= start, "strategy_equity"].equals(
            right.loc[right.index <= start, "strategy_equity"]
        ):
            raise AssertionError(
                f"{ep['episode_id']}: historico anterior ao episodio divergiu"
            )
        future = left.index[left.index > start]
        if len(future) < 1:
            raise AssertionError(f"{ep['episode_id']}: nenhum dia apos decisao")
        initial_date = future[0]
        horizon_date = future[min(len(future), horizon) - 1]
        horizon_without = float(left.loc[horizon_date, "strategy_equity"])
        horizon_with = float(right.loc[horizon_date, "strategy_equity"])
        ending_without = float(left["strategy_equity"].iloc[-1])
        ending_with = float(right["strategy_equity"].iloc[-1])
        post_a = left.loc[future, "selected_asset"].fillna("")
        post_b = right.loc[future, "selected_asset"].fillna("")
        differs = post_a.ne(post_b)
        reconverged = post_a.index[~differs]
        first_rejoin = reconverged[0] if len(reconverged) else pd.NaT
        row = {
            "episode_id": ep["episode_id"],
            "scenario_without": prev_label,
            "scenario_with": next_label,
            "decision_start": ep["start"],
            "decision_end": ep["end"],
            "fold_id": ep["fold_id"],
            "overrides": len(ep["dates"]),
            "dates": ";".join(str(d) for d in ep["dates"]),
            "state_at_start": ep["initial_state"],
            "control_action_at_start": ep["control_action"],
            "learned_action_at_start": ep["learned_action"],
            "learned_model_scores": ";".join(str(v) for v in ep["scores"]),
            "first_execution_date": initial_date,
            "horizon_execution_date": horizon_date,
            "horizon_sessions": int(min(len(future), horizon)),
            "equity_horizon_without": horizon_without,
            "equity_horizon_with": horizon_with,
            "delta_horizon_usd": horizon_with - horizon_without,
            "equity_final_without": ending_without,
            "equity_final_with": ending_with,
            "delta_final_usd": ending_with - ending_without,
            "delta_final_pct_of_without": (
                ending_with / ending_without - 1.0
            ),
            "log_capital_ratio": float(np.log(ending_with / ending_without)),
            "different_asset_sessions_post": int(differs.sum()),
            "first_same_asset_after_start": first_rejoin,
            "fees_without_usd": float(prev.metrics["total_transaction_fees"]),
            "fees_with_usd": float(after.metrics["total_transaction_fees"]),
            "delta_fees_usd": (
                float(after.metrics["total_transaction_fees"]) -
                float(prev.metrics["total_transaction_fees"])
            ),
            "rotations_without": int(prev.metrics["capital_rotations"]),
            "rotations_with": int(after.metrics["capital_rotations"]),
            "cash_days_without": int(prev.metrics["cash_days"]),
            "cash_days_with": int(after.metrics["cash_days"]),
        }
        summaries.append(row)
        pair = pd.DataFrame({
            "timestamp": future,
            "scenario_without": prev_label,
            "scenario_with": next_label,
            "capital_without": left.loc[future, "strategy_equity"].to_numpy(),
            "capital_with": right.loc[future, "strategy_equity"].to_numpy(),
            "asset_without": post_a.to_numpy(),
            "asset_with": post_b.to_numpy(),
        })
        pair.insert(0, "episode_id", ep["episode_id"])
        pair["delta_capital_usd"] = (
            pair["capital_with"] - pair["capital_without"]
        )
        pair["asset_differs"] = pair["asset_without"].ne(pair["asset_with"])
        daily.append(pair)

    baseline_capital = float(initial.metrics["strategy_ending_capital"])
    actual_capital = float(original_regression.metrics["strategy_ending_capital"])
    summed = sum(float(item["delta_final_usd"]) for item in summaries)
    if not np.isclose(
        summed, actual_capital - baseline_capital, rtol=1e-10, atol=1e-4,
    ):
        raise AssertionError("Contribuicoes telescopicas nao reconciliam")
    return AuditoriaIntervencoes(
        baseline_result=initial,
        episodios=pd.DataFrame(summaries),
        curvas=pd.concat(curves, ignore_index=True),
        pares_diarios=pd.concat(daily, ignore_index=True),
        operacoes=pd.concat(trades, ignore_index=True),
        checks={
            "schema_version": 1,
            "method": "chronological cumulative episode replay",
            "episodes": len(episodes),
            "override_sessions": len(all_override_dates),
            "baseline_vs_official_control_max_abs": control_delta,
            "final_bridge_vs_regression_max_abs": final_delta,
            "baseline_ending_capital": baseline_capital,
            "reproduced_regression_ending_capital": actual_capital,
            "sum_episode_marginal_usd": summed,
            "end_to_end_delta_usd": actual_capital - baseline_capital,
            "marginal_reconciliation_abs_error": abs(
                summed - (actual_capital - baseline_capital)
            ),
            "note": (
                "Effects depend on chronological inclusion order and prior "
                "episodes. Matching asset does not mean matching equity. "
                "No future decisions were used to train a model."
            ),
        },
    )
