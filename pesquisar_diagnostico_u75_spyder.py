"""Diagnostico U75: estresse coletivo, oportunidade tecnica e posicoes.

Spyder: F5. Nao opera, nao recalibra limiares usando resultado OOS
e nao altera o Control congelado. Reproduz primeiro a Strategy #30 e
compara todas as decisoes antes de iniciar qualquer simulacao alternativa.

V1 = pesquisa exploratoria. "Sem candidato tecnico" nao equivale a
"nenhum ativo tera retorno positivo". O filtro financeiro precisa
ser validado em novas janelas temporais antes de qualquer utilizacao.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import traceback
from zipfile import ZipFile

import numpy as np
import pandas as pd

from pesquisa.control_operacional_strategy30 import (
    ROOT, carregar_contrato, validar_control_strategy30,
)
from pesquisa.paridade_strategy30 import (
    config_from_frozen_manifest, load_snapshot_strategy30,
    compare_control_trajectories,
)
from pesquisa.referencia_strategy30.modelo_lightgbm import (
    _ajustar_modelos_lightgbm, _construir_contexto_execucao,
    _selecionar_switch_margin_fold,
)
from pesquisa.referencia_strategy30.rotacao import (
    _politica_utilidade, _politica_agendada, _crescimento_politica_simples,
    _precalcular_utilidades_modelo, _simular_exato,
)
from pesquisa.referencia_strategy30.execucao import (
    calcular_taxas_referencia, aplicar_deslizamento,
)
from pesquisa.diagnostico_u75 import (
    StressSettings, calcular_indicadores_u75, juntar_decisoes,
    extrair_episodios, posicoes_mensais, serializar_mapa_html,
    construir_politica_protecao_diaria,
)
from pesquisa.figuras_u75 import (
    grafico_forca_universo, grafico_capital_e_estresse,
    grafico_mapa_mensal,
)
from pesquisa.pacote_visual_u75 import gerar_zip_visual
from pesquisa.persistencia_resultados import salvar_csv, salvar_json, salvar_texto
from pesquisar_controle_strategy30_spyder import localizar_zip_original
from pesquisar_expansao_u67_spyder import _write_json

SCRIPT_VERSION = "u75-universe-stress-diagnostics-v1.0.1"
OUTPUT = ROOT / "output" / "diagnostico_u75"
SNAPSHOT = ROOT / "dados" / "strategy30_snapshot"
SETTINGS = StressSettings(
    min_valid_assets=68,
    max_positive_share_5=0.35,
    max_positive_share_20=0.45,
    min_below_ma50_share=0.55,
    max_median_return_20=-0.015,
    audit_sessions=10,
)


def _simulacao(
    name: str, policy, frames, symbols, dates, config, metadata,
    *,
    benchmark: pd.Series | None = None,
):
    return _simular_exato(
        name, policy, frames, symbols, dates, config,
        calcular_taxas_referencia, aplicar_deslizamento,
        decision_metadata=metadata, model_label=name,
        benchmark_override=benchmark,
        benchmark_override_name=("Control original" if benchmark is not None else None),
    )


def _run(output: Path) -> dict:
    # %% 1 - Evidencias, configuracao e snapshot congelados
    contract = carregar_contrato()
    source_zip = localizar_zip_original()
    reference_report, reference, _ = validar_control_strategy30(
        source_zip, contrato=contract,
    )
    with ZipFile(source_zip) as archive:
        strategy = json.loads(archive.read("strategy_manifest.json"))
        experiment = json.loads(archive.read("experiment_manifest.json"))
    if experiment.get("market_data_signature_sha256") != contract["market_data_signature_sha256"]:
        raise RuntimeError("Assinatura do historico divergente.")
    snapshot_dir = Path(os.environ.get(
        "STRATEGY30_SNAPSHOT_DIR", str(SNAPSHOT),
    )).expanduser()
    snapshot = load_snapshot_strategy30(
        snapshot_dir, contract=contract, experiment_manifest=experiment,
    )
    config = config_from_frozen_manifest(strategy, contract)
    salvar_csv(snapshot.records, output / "auditoria_snapshot_75.csv", index=False)
    (
        frames, calendar, source, symbols, folds, decisions,
        decision_to_fold, decision_metadata,
    ) = _construir_contexto_execucao(snapshot.frames, config)
    if len(symbols) != 75 or len(decisions) != 1577 or len(folds) != 3:
        raise RuntimeError("Universo/calendario diferem da Strategy #30.")
    print("[fonte] 75 ativos, tres periodos temporais, CSVs RAW/SIP", flush=True)

    # %% 2 - Reproduzir o mesmo treinamento e calibracao cronologicos
    policies = {}
    diagnostic: dict[pd.Timestamp, dict] = {}
    margins = []
    model_positive_counts = []
    for fold in folds:
        fid = int(fold["fold_id"])
        train = calendar[:int(fold["train_end_index"])]
        calibration = calendar[
            int(fold["calibration_start_index"]):int(fold["calibration_end_index"])
        ]
        final_train = calendar[:int(fold["final_fit_end_index"])]
        if not (
            pd.Timestamp(train[-1]) < pd.Timestamp(calibration[0])
            and pd.Timestamp(calibration[-1]) < pd.Timestamp(fold["test_start"])
        ):
            raise RuntimeError("Separacao temporal da pesquisa foi violada.")
        print(f"[modelo] periodo={fid}/3 treinando calibracao (75 ativos)", flush=True)
        cal_models = _ajustar_modelos_lightgbm(
            frames, symbols, train, config, phase=f"u75_stress_fold_{fid}_calibration",
        )
        candidate_scores = [
            (float(m), _crescimento_politica_simples(
                _politica_utilidade(cal_models, frames, symbols, config, float(m)),
                frames, symbols, calibration, config,
            ))
            for m in config.rotation_switch_margin_candidates
        ]
        choice = _selecionar_switch_margin_fold(config, fid, candidate_scores)
        if choice["selection_source"] != "calibration":
            raise RuntimeError("Override de margem nao permitido.")
        margin = max(
            float(config.rotation_switch_margin),
            float(choice["selected_candidate_margin"]),
        )
        if pd.Timestamp(final_train[-1]) >= pd.Timestamp(fold["test_start"]):
            raise RuntimeError("Treinamento entrou no conjunto de avaliacao.")
        print(f"[modelo] periodo={fid}/3 treinando modelos finais (75 ativos)", flush=True)
        final_models = _ajustar_modelos_lightgbm(
            frames, symbols, final_train, config,
            phase=f"u75_stress_fold_{fid}_final",
        )
        fold_dates = pd.DatetimeIndex(fold["decision_dates"])
        cache, _ = _precalcular_utilidades_modelo(
            final_models, frames, symbols, fold_dates, config,
        )
        policies[fid] = _politica_utilidade(
            final_models, frames, symbols, config, margin,
            decision_diagnostics=diagnostic, fold_id=fid,
            calibrated_switch_margin=float(choice["selected_candidate_margin"]),
            utility_cache=cache,
        )
        margins.append({
            "fold_id": fid,
            "calibration_start": pd.Timestamp(calibration[0]).isoformat(),
            "calibration_end": pd.Timestamp(calibration[-1]).isoformat(),
            "oos_start": pd.Timestamp(fold["test_start"]).isoformat(),
            "oos_end": pd.Timestamp(fold["test_end"]).isoformat(),
            "calibrated_candidate_margin": float(choice["selected_candidate_margin"]),
            "effective_margin": margin,
        })
    salvar_csv(pd.DataFrame(margins), output / "auditoria_margens.csv", index=False)

    # %% 3 - Auditoria completa de paridade antes de qualquer alternativa
    base = _politica_agendada(policies, decision_to_fold)
    control = _simulacao(
        "Control Strategy #30", base, frames, symbols,
        decisions, config, decision_metadata,
    )
    parity, diff = compare_control_trajectories(
        reference, control.predictions, contract=contract,
    )
    salvar_csv(diff, output / "control_paridade_sessao.csv", index=False)
    salvar_json(output / "control_paridade.json", parity, _write_json)
    if not parity["fully_reproduced"] or not math.isclose(
        float(control.metrics["strategy_ending_capital"]),
        float(contract["ending_capital"]), rel_tol=0, abs_tol=0.01,
    ):
        raise RuntimeError(
            "Control nao reproduzido integralmente: diagnostico bloqueado. "
            f"Primeira divergencia: {parity['first_mismatch_date']}"
        )
    print(
        f"[control] paridade OK; patrimonio final "
        f"US$ {control.metrics['strategy_ending_capital']:,.2f}",
        flush=True,
    )
    control_predictions = control.predictions.reset_index()
    salvar_csv(control_predictions, output / "control_posicoes_diarias.csv",
               index=False, float_format="%.17g")
    salvar_csv(control.trades, output / "control_operacoes.csv", index=False)

    # %% 4 - Indicadores exclusivos do U75 com dados conhecidos em t
    indicator_dates = pd.DatetimeIndex(decisions[:-1])
    daily, per_asset = calcular_indicadores_u75(
        snapshot.frames, symbols, indicator_dates, settings=SETTINGS,
    )
    joined = juntar_decisoes(daily, control_predictions)
    # As pontuacoes originais sao auditoria, NAO entram no sinal de
    # estresse: a elegibilidade na fonte congelada olha o pregão seguinte.
    if len(diagnostic) != len(joined):
        raise RuntimeError("Faltam diagnosticos de selecao do Control.")
    joined["control_positive_score_count_observational"] = [
        int(diagnostic[pd.Timestamp(x)]["positive_score_count"])
        for x in joined["decision_date"]
    ]
    joined["control_best_score_observational"] = [
        diagnostic[pd.Timestamp(x)].get("best_score")
        for x in joined["decision_date"]
    ]
    salvar_csv(daily, output / "indicadores_diarios_u75.csv",
               index=False, float_format="%.17g")
    salvar_csv(per_asset, output / "indicadores_por_ativo.csv",
               index=False, float_format="%.17g")
    salvar_csv(joined, output / "decisoes_e_estresse.csv",
               index=False, float_format="%.17g")
    print(
        f"[estresse] dias={joined['universe_stress'].sum()} "
        f"investido_sob_estresse={joined['invested_during_stress'].sum()} "
        f"sem_candidato_tecnico={joined['invested_without_candidate'].sum()}",
        flush=True,
    )

    # %% 5 - Mapa mensal de posicoes efetivas, nao de previsoes
    monthly, holdings = posicoes_mensais(
        control_predictions, control.trades, daily,
        initial_capital=float(config.initial_capital),
    )
    salvar_csv(monthly, output / "mapa_mensal_resumo.csv", index=False)
    salvar_csv(holdings, output / "mapa_mensal_todos_ativos.csv", index=False)
    (output / "mapa_mensal_interativo.html").write_text(
        serializar_mapa_html(
            monthly, holdings,
            title="U75: ativos mantidos em cada mes (Control Strategy #30)",
        ), encoding="utf-8",
    )
    grafico_forca_universo(daily, output / "forca_diaria_u75.png")
    grafico_capital_e_estresse(joined, output / "patrimonio_e_estresse.png")
    grafico_mapa_mensal(monthly, output / "mapa_mensal_ativos.png")

    # %% 6 - Contrafactuais: mesmo estado, mesma data, taxas do motor
    episodes = extrair_episodios(joined, settings=SETTINGS)
    salvar_csv(episodes, output / "episodios_de_estresse.csv", index=False)
    audited = []
    selected_starts = [
        int(r["start_index"]) for r in episodes.to_dict("records")
        if r["audit_eligible"]
    ]
    benchmark = control.predictions["buy_hold_equity"]
    for k, i in enumerate(selected_starts, start=1):
        if pd.Timestamp(decisions[i]) != pd.Timestamp(joined.iloc[i]["decision_date"]):
            raise RuntimeError("Decisao do episodio nao coincide com calendario.")
        alternative = _simulacao(
            f"cash_episodio_{k}",
            construir_politica_protecao_diaria(
                base, joined,
                active_dates=pd.DatetimeIndex(decisions[i:i + SETTINGS.audit_sessions]),
            ),
            frames, symbols, decisions, config, decision_metadata,
            benchmark=benchmark,
        )
        prior = (
            float(config.initial_capital) if i == 0
            else float(control.predictions.iloc[i - 1]["strategy_equity"])
        )
        before_alt = (
            float(config.initial_capital) if i == 0
            else float(alternative.predictions.iloc[i - 1]["strategy_equity"])
        )
        if abs(prior - before_alt) > 0.01:
            raise RuntimeError(f"Episodio {k}: estados iniciais diferentes.")
        until = i + SETTINGS.audit_sessions - 1
        actual_end = float(control.predictions.iloc[until]["strategy_equity"])
        other_end = float(alternative.predictions.iloc[until]["strategy_equity"])
        local_decisions = alternative.predictions.iloc[i:until + 1]
        local_cash = local_decisions["selected_asset"].eq("CASH")
        local_reentry = (
            ~local_cash & local_cash.shift(fill_value=False)
        )
        audited.append({
            "decision_date": pd.Timestamp(decisions[i]).isoformat(),
            "evaluation_end": pd.Timestamp(decisions[i + SETTINGS.audit_sessions]).isoformat(),
            "original_asset": joined.iloc[i]["selected_asset"],
            "cash_sessions_observed": int(local_cash.sum()),
            "reentries_observed": int(local_reentry.sum()),
            "starting_capital": prior,
            "control_capital_at_horizon": actual_end,
            "cash_capital_at_horizon": other_end,
            "difference_usd_at_horizon": other_end - actual_end,
            "difference_return_points": 100.0 * (other_end - actual_end) / prior,
            "avoided_loss": bool(other_end > actual_end + 0.01),
            "comparison": "one_ex_ante_intervention_followed_by_Control",
        })
        print(
            f"[episodio] {k}/{len(selected_starts)} "
            f"data={joined.iloc[i]['decision_date']} "
            f"delta_USD={other_end - actual_end:+,.2f}", flush=True,
        )
    episode_finance = pd.DataFrame(audited, columns=[
        "decision_date", "evaluation_end", "original_asset",
        "cash_sessions_observed", "reentries_observed", "starting_capital",
        "control_capital_at_horizon", "cash_capital_at_horizon",
        "difference_usd_at_horizon", "difference_return_points",
        "avoided_loss", "comparison",
    ])
    salvar_csv(episode_finance, output / "contrafactual_por_episodio.csv",
               index=False, float_format="%.17g")

    # %% 7 - Protecao do universo reavaliada TODOS os pregoes, sem trava
    # Se o sinal deixar de existir no fechamento t, o Control ja pode
    # escolher a recompra para a execucao seguinte, usando posicao CASH.
    protective_policy = construir_politica_protecao_diaria(base, joined)
    protected = _simulacao(
        "cash_protecao_u75_diaria", protective_policy,
        frames, symbols, decisions, config, decision_metadata,
        benchmark=benchmark,
    )
    salvar_csv(
        protected.predictions.reset_index(),
        output / "protecao_posicoes_diarias.csv",
        index=False, float_format="%.17g",
    )
    salvar_csv(protected.trades, output / "protecao_operacoes.csv", index=False)
    protected_daily = protected.predictions.reset_index().merge(
        joined[[
            "decision_date", "universe_stress",
            "stress_without_candidate", "technical_candidates_count",
        ]], on="decision_date", validate="one_to_one", how="left", indicator=True,
    )
    if len(protected_daily) != len(joined) or not protected_daily["_merge"].eq("both").all():
        raise RuntimeError("Protecao diaria sem correspondencia completa de datas.")
    protected_daily = protected_daily.drop(columns="_merge")
    cash_flags = protected_daily["selected_asset"].eq("CASH")
    previous_cash = cash_flags.shift(fill_value=False)
    reentries = ~cash_flags & previous_cash
    cash_entries = cash_flags & ~previous_cash
    protected_daily["entry_to_cash"] = cash_entries
    protected_daily["reentry_from_cash"] = reentries
    protected_daily["decision_reason"] = np.where(
        protected_daily["stress_without_candidate"],
        "protecao_do_universo", "seguir_control",
    )
    salvar_csv(
        protected_daily, output / "decisoes_protecao_diaria.csv",
        index=False, float_format="%.17g",
    )

    # %% 8 - Resultado e limites da pesquisa
    metrics = ("strategy_ending_capital", "strategy_sharpe",
               "strategy_maximum_drawdown", "cash_days",
               "capital_rotations", "total_transaction_fees")
    result = {
        "status": "completed",
        "version": SCRIPT_VERSION,
        "research_only": True,
        "market_gate_promoted": False,
        "reference_zip_sha256": reference_report["source_zip_sha256"],
        "control_parity": parity,
        "settings_predeclared": SETTINGS.__dict__,
        "selected_universe_count": len(symbols),
        "sessions": len(joined),
        "coverage_insufficient_days": int((~joined["coverage_ok"]).sum()),
        "universe_stress_days": int(joined["universe_stress"].sum()),
        "invested_during_universe_stress": int(joined["invested_during_stress"].sum()),
        "invested_under_stress_without_technical_candidate": int(
            joined["invested_without_candidate"].sum()
        ),
        "stress_episodes": len(episodes),
        "auditable_disjoint_episodes": len(selected_starts),
        "counterfactual_evaluation_horizon_sessions": SETTINGS.audit_sessions,
        "cash_minimum_holding_sessions": 0,
        "cash_maximum_holding_sessions": None,
        "market_condition_recomputed_each_session": True,
        "protection_signal_sessions": int(joined["stress_without_candidate"].sum()),
        "actual_protected_cash_sessions": int(cash_flags.sum()),
        "cash_periods": int(cash_entries.sum()),
        "cash_reentries": int(reentries.sum()),
        "episodes_where_cash_protected": int(
            episode_finance["avoided_loss"].sum()
        ) if len(episode_finance) else 0,
        "control_metrics": {k: control.metrics.get(k) for k in metrics},
        "exploratory_protection_metrics": {
            k: protected.metrics.get(k) for k in metrics
        },
        "exploratory_final_difference_usd": float(
            protected.metrics["strategy_ending_capital"]
            - control.metrics["strategy_ending_capital"]
        ),
        "cautions": [
            "Estresse e oportunidade tecnica sao indicadores retrospectivos do universo; nao estimativas calibradas de probabilidade futura.",
            "Limites de estresse foram predeclarados; o estado CASH e reavaliado por dia, sem duracao fixa. Horizonte de 10 sessoes e apenas avaliacao financeira.",
            "Pontuacoes de modelo originais nao entram no filtro: a fonte congelada consulta precos futuros para checar elegibilidade.",
            "Episodios com horizontes de comparacao sobrepostos nao sao auditados duas vezes; politica completa usa todos os sinais diarios.",
            "Deltas locais de patrimonio nao sao aditivos entre episodios.",
            "Resultados OOS deste periodo ja foram estudados anteriormente e nao sao teste prospectivo virgem.",
            "Nao usar o replay exploratorio para operar ou afirmar ausencia de sobreajuste.",
        ],
    }
    salvar_json(output / "resultado.json", result, _write_json)
    salvar_texto(
        output / "resumo_para_analise.md",
        "# Diagnostico U75: estresse, posicoes e CASH\n\n"
        f"Versao: {SCRIPT_VERSION}\n"
        f"Control reproduzido: US$ {control.metrics['strategy_ending_capital']:,.2f}\n"
        f"Estresse coletivo: {result['universe_stress_days']} sessoes; "
        f"investido sob estresse: {result['invested_during_universe_stress']}\n"
        f"Estresse sem candidato tecnico: "
        f"{result['invested_under_stress_without_technical_candidate']}\n"
        f"Episodios auditados: {len(selected_starts)}\n"
        f"Protecao reavaliada por pregao, SEM trava de duracao: "
        f"{int(cash_flags.sum())} sessoes CASH em {int(cash_entries.sum())} periodos; "
        f"{int(reentries.sum())} reentradas.\n"
        f"Replay exploratorio: US$ "
        f"{protected.metrics['strategy_ending_capital']:,.2f}; "
        f"diferenca: US$ {result['exploratory_final_difference_usd']:+,.2f}\n\n"
        "Todos os 75 ativos foram mantidos no universo e nos mapas mensais.\n"
        "O mapa HTML permite clicar nos meses para consultar todas as posicoes.\n"
        "A contagem de candidatos e TECNICA, nao comprovacao de oportunidades "
        "futuras. A fonte do Control inclui uma dependencia de elegibilidade "
        "na barra seguinte; esse problema segue pendente de revisao causal.\n",
    )
    print(
        f"[resumo] Control=US$ {control.metrics['strategy_ending_capital']:,.2f}; "
        f"hipotese=US$ {protected.metrics['strategy_ending_capital']:,.2f}",
        flush=True,
    )
    return result


def main() -> dict:
    output = OUTPUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = _run(output)
        package = gerar_zip_visual(output, versao=SCRIPT_VERSION, status="completed")
        print(f"[pacote] Envie este ZIP para analise: {package}", flush=True)
        return result
    except Exception as exc:
        try:
            salvar_json(output / "erro_execucao.json", {
                "status": "failed", "version": SCRIPT_VERSION,
                "type": type(exc).__name__, "message": str(exc),
                "traceback": traceback.format_exc(limit=30),
            }, _write_json)
            package = gerar_zip_visual(output, versao=SCRIPT_VERSION, status="failed")
            print(f"[pacote] Falha auditavel: {package}", flush=True)
        except Exception as package_error:
            print(f"[pacote] Nao foi possivel criar ZIP: {package_error}", flush=True)
        raise


if __name__ == "__main__":
    main()
