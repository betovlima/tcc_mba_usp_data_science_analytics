"""TCC - Spyder F5: comparacao unica dos modelos de risco sistemico U75.

Sem operacoes em CASH, sem selecionar ativo, sem retreinar Control e sem banco.
Executa todos os metodos no mesmo historico e exibe graficos no Spyder.
"""
# %% 0 - Configuracao cientifica e importacoes
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pesquisa.control_operacional_strategy30 import (
    ROOT, carregar_contrato, validar_control_strategy30,
)
from pesquisa.paridade_strategy30 import load_snapshot_strategy30
from pesquisa.benchmark_risco_sistemico_u75 import (
    VERSION, METHODS, Settings, make_panel, evaluate,
)
from pesquisa.andamento_motivos_u75 import record_progress
from pesquisar_controle_strategy30_spyder import localizar_zip_original

OUTPUT_ROOT = Path(os.environ.get(
    "TCC_SYSTEMIC_OUTPUT_DIR",
    str(ROOT / "output" / "comparacao_risco_sistemico_u75"),
)).expanduser()
SNAPSHOT_ROOT = Path(os.environ.get(
    "STRATEGY30_SNAPSHOT_DIR", str(ROOT / "dados" / "strategy30_snapshot")
)).expanduser()
CFG = Settings()


def _csv(df, output):
    df.to_csv(output, index=False, float_format="%.17g")


def _save_fig(fig, out, filename, figures):
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150, bbox_inches="tight")
    figures.append(fig)


# %% 1 - Graficos comparativos: gerados e exibidos no Spyder
def _charts(features, labels, predictions, summary, out):
    figures = []
    for metric, higher_better, filename in [
        ("brier", False, "01_brier_por_metodo_horizonte.png"),
        ("average_precision", True, "02_precision_recall_area.png"),
        ("roc_auc", True, "03_auc_roc.png"),
    ]:
        matrix = summary.pivot(index="method", columns="horizon", values=metric)
        matrix = matrix.reindex(list(METHODS))
        fig, ax = plt.subplots(figsize=(12, 5.5))
        x = np.arange(len(matrix))
        width = .78 / max(1, len(matrix.columns))
        for j, horizon in enumerate(matrix.columns):
            ax.bar(x - .39 + width * (j + .5),
                   matrix[horizon].values, width, label=f"h={horizon}")
        ax.set_xticks(x, matrix.index, rotation=38, ha="right")
        ax.set_title({
            "brier": "Qualidade das probabilidades (Brier; menor e melhor)",
            "average_precision": "Precisao media de eventos sistemicos (maior e melhor)",
            "roc_auc": "Discriminacao de eventos (AUC ROC; maior e melhor)",
        }[metric])
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=.2)
        ax.legend(title="Pregoes a frente", ncols=min(4, len(matrix.columns)))
        _save_fig(fig, out, filename, figures)

    # O h principal e o primeiro horizonte multi-pregao disponivel.
    selected_h = next((h for h in CFG.horizons if h >= 5), CFG.horizons[0])
    p = predictions.loc[predictions.horizon.eq(selected_h)].copy()
    fig, ax = plt.subplots(figsize=(13.5, 5))
    sample = p.loc[p.method.eq("Base_historica")]
    if len(sample):
        ax.plot(pd.to_datetime(sample.date), sample.probability, label="Frequencia historica",
                lw=1.1, ls="--", color="black")
        ax.fill_between(pd.to_datetime(sample.date), 0, 1,
                        where=sample.realized_event.astype(bool).to_numpy(),
                        color="gray", alpha=.12, label="Evento coletivo (pos-fato)")
    methods = ["CUSUM", "BOCPD", "Markov_filtrado", "PCA_Absorption",
               "EVT_POT", "LightGBM_multivariado"]
    for method in methods:
        m = p.loc[p.method.eq(method)]
        if len(m):
            ax.plot(pd.to_datetime(m.date), m.probability,
                    linewidth=.8, alpha=.9, label=method)
    ax.set_title(f"Probabilidade cronologica de queda generalizada nos proximos {selected_h} pregoes")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probabilidade prevista")
    ax.grid(alpha=.2)
    ax.legend(ncols=3, fontsize=8)
    _save_fig(fig, out, "04_risco_sistemico_temporal.png", figures)

    fig, ax = plt.subplots(figsize=(9, 6))
    boundaries = np.linspace(0, 1, 11)
    for method in METHODS:
        frame = p.loc[p.method.eq(method)].copy()
        if len(frame) < 10:
            continue
        frame["bucket"] = pd.cut(frame.probability, bins=boundaries, include_lowest=True)
        calib = frame.groupby("bucket", observed=True).agg(
            predicted=("probability", "mean"),
            observed=("realized_event", "mean"),
            n=("date", "size"),
        )
        calib = calib.loc[calib.n >= 5]
        if not calib.empty:
            ax.plot(calib.predicted, calib.observed, marker=".", alpha=.8, label=method)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Calibracao perfeita")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Probabilidade prevista")
    ax.set_ylabel("Frequencia observada")
    ax.set_title(f"Calibracao anual OOS combinada, horizonte {selected_h}")
    ax.legend(loc="best", fontsize=8, ncols=2)
    ax.grid(alpha=.2)
    _save_fig(fig, out, "05_calibracao_probabilidades.png", figures)

    # Contexto do mercado, apenas observacoes contemporaneas.
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 9), sharex=True)
    axes[0].plot(features.index, features.breadth_down20, lw=.8)
    axes[0].set_ylabel("Fracao em queda, 20d")
    axes[1].plot(features.index, features.absorption_ratio, lw=.8)
    axes[1].set_ylabel("Absorption ratio")
    axes[2].plot(features.index, features.dcc_mean_corr, lw=.8)
    axes[2].set_ylabel("Correlacao dinamica")
    axes[2].set_xlabel("Data")
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle("Indicadores coletivos causais do universo U75")
    _save_fig(fig, out, "06_indicadores_risco_coletivo.png", figures)

    fig, ax = plt.subplots(figsize=(12, 5))
    matrix = p.groupby(["year", "method"]).apply(
        lambda g: float(np.mean((g.probability-g.realized_event.astype(int))**2)),
        include_groups=False,
    ).unstack("method")
    matrix = matrix.reindex(columns=list(METHODS))
    visual = matrix.to_numpy(float)
    im = ax.imshow(np.ma.masked_invalid(visual), aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index.astype(str))
    ax.set_title(f"Brier por ano, horizonte {selected_h} (menor e melhor)")
    fig.colorbar(im, ax=ax)
    _save_fig(fig, out, "07_brier_por_ano.png", figures)

    # In Spyder, backend Inline or Plots keeps figures visible. No forced Agg.
    if os.environ.get("TCC_SHOW_PLOTS", "1") != "0":
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)
    return len(figures)


# %% 2 - Fonte e calculo unico de atributos para os metodos
def _run(out):
    contract = carregar_contrato()
    original = localizar_zip_original()
    identity, _, _ = validar_control_strategy30(original, contrato=contract)
    with ZipFile(original) as archive:
        manifest = json.loads(archive.read("experiment_manifest.json"))
    if manifest["market_data_signature_sha256"] != contract["market_data_signature_sha256"]:
        raise RuntimeError("Assinatura do historico incompatível com o Control.")
    snapshot = load_snapshot_strategy30(
        SNAPSHOT_ROOT, contract=contract, experiment_manifest=manifest)
    _csv(snapshot.records, out / "snapshot_identidade.csv")
    record_progress(out, "snapshot_validado", details={
        "assets":len(contract["assets"]),"reference":identity["id"]})
    print(f"[fonte] {len(contract['assets'])} ativos do contrato; sem alteracao do universo",
          flush=True)

    features, labels = make_panel(snapshot.frames, contract["assets"], CFG)
    if len(features) < CFG.min_training_dates + max(CFG.horizons):
        raise RuntimeError("Historico insuficiente para periodos cronologicos.")
    _csv(features.rename_axis("date").reset_index(), out / "atributos_coletivos.csv")
    _csv(labels.rename_axis("date").reset_index(), out / "rotulos_futuros_poshoc.csv")
    record_progress(out, "atributos_coletivos_concluidos", details={
        "rows":len(features),"methods":len(METHODS),"horizons":CFG.horizons})

    # %% 3 - Todos os metodos/horizontes com avaliacoes pareadas OOS
    def notify(year, horizon):
        record_progress(out, "avaliacao_ano_horizonte", details={
            "year":int(year),"horizon":int(horizon)})
        print(f"[avaliacao] ano={year} h={horizon}", flush=True)
    predictions, metrics, folds = evaluate(features, labels, CFG, progress=notify)
    _csv(predictions, out / "previsoes_por_modelo.csv")
    _csv(metrics, out / "comparacao_metricas.csv")
    _csv(folds, out / "auditoria_folds.csv")
    count = predictions.groupby(["horizon","method"]).size().unstack("method")
    if not count.nunique(axis=1).eq(1).all():
        raise RuntimeError("Datas de teste diferentes entre algoritmos; comparacao rejeitada.")
    record_progress(out, "previsoes_comparadas", details={
        "predictions":len(predictions), "folds":len(folds)})

    # %% 4 - Eventos e alarmes por ano, para diagnostico de lead time e falhas
    yearly = predictions.groupby(["horizon","year","method"], as_index=False).agg(
        sessions=("date","size"),events=("realized_event","sum"),
        alerts=("alert","sum"))
    yearly["hits"] = [
        int((g.alert & g.realized_event).sum())
        for _, g in predictions.groupby(["horizon","year","method"], sort=True)
    ]
    _csv(yearly, out / "comparacao_por_ano.csv")
    # Each h-day label overlaps adjacent labels. Event starts are explicitly
    # first TRUE dates, not daily TP counts as independent crashes.
    onset=[]
    for (h, method), group in predictions.groupby(["horizon","method"]):
        group=group.sort_values("date")
        current=group.realized_event.astype(bool).to_numpy()
        started=current & ~np.r_[False, current[:-1]]
        for pos in np.flatnonzero(started):
            beginning=max(0, pos-h)
            earliest=group.iloc[beginning:pos+1]
            early=earliest.loc[earliest.alert]
            onset.append({
                "horizon":int(h),"method":method,
                "event_start_decision_date":str(group.iloc[pos].date),
                "early_signal_in_previous_h_sessions":bool(len(early)),
                "earliest_alert_in_window":str(early.iloc[0].date) if len(early) else None,
                "lead_sessions_in_decision_calendar":int(pos-early.index[0]) if False else (
                    int(pos - int(np.flatnonzero(
                        earliest.alert.to_numpy(bool))[0] + beginning)) if len(early) else None
                ),
            })
    _csv(pd.DataFrame(onset), out / "eventos_e_antecedencia.csv")
    record_progress(out, "eventos_e_antecedencia_calculados")

    # %% 5 - Graficos Spyder, analise e pacote reproduzivel
    graphics=_charts(features, labels, predictions, metrics, out)
    report={
        "version":VERSION,"status":"completed",
        "study":"Risco de queda generalizada do universo; pesquisa shadow",
        "settings":CFG.__dict__,
        "methods":list(METHODS),
        "comparison":metrics.to_dict(orient="records"),
        "source_identity":identity,
        "universe_size":len(contract["assets"]),
        "graphs":graphics,
        "claim_financial_protection":False,
        "limitations":[
            "Eventos de h pregoes se sobrepoem; sessoes nao sao crises independentes.",
            "Limiares de perda definidos por quantis apenas do treino maturado, por ano e horizonte.",
            "Indice equal-weight e apenas diagnostico coletivo; nao representa carteira Control.",
            "Probabilidade de evento coletivo nao comprova vantagem de CASH perante todos os ativos.",
            "Nenhum modelo foi escolhido ou ajustado para CAPR, AMS, 2026 ou capital final.",
            "Nao representa holdout inteiramente inedito; episodios historicos ja foram inspecionados.",
            "DCC_proxy usa EWMA com coeficientes fixos e nao e uma estimacao completa de DCC-GARCH.",
            "Estados de Markov ajustados apenas com dados passados do fold e filtrados ate a data t.",
            "O Control permanece congelado; a limitacao historica de elegibilidade t+1 persiste.",
            "Nenhuma alteracao de politica, nenhum trade ou protecao CASH executados.",
        ],
    }
    (out / "resultado.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str, allow_nan=False),
        encoding="utf-8")
    best=metrics.sort_values(["horizon","brier"]).groupby("horizon").first().reset_index()
    lines=["# Comparacao de risco sistemico U75",
           "", "Resultado exclusivo de previsao, sem simulacao financeira.",
           "", "## Menor Brier por horizonte", "",
           best[["horizon","method","brier","average_precision","roc_auc","observations"]].to_string(index=False),
           "", "## Metricas completas", "", metrics.to_string(index=False),
           "", "## Limites de interpretacao", "",
           *("- "+s for s in report["limitations"])]
    (out / "resumo.md").write_text("\n".join(lines), encoding="utf-8")
    return report


# %% 6 - Entrada F5, progresso e ZIP de evidencias
def main():
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out=OUTPUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=False)
    try:
        report=_run(out)
        record_progress(out, "execucao_concluida", status="completed")
        status="completed"
    except Exception as exc:
        status="failed"
        (out/"erro_execucao.txt").write_text(traceback.format_exc(), encoding="utf-8")
        record_progress(out, "falha_execucao", status="failed",
                        details={"error":str(exc),"type":type(exc).__name__})
        report=None
    package=out / f"pacote_comparacao_risco_sistemico_u75_{run_id}.zip"
    with ZipFile(package,"w",ZIP_DEFLATED) as archive:
        for f in sorted(out.iterdir()):
            if f.is_file() and f != package and f.suffix in (".json",".jsonl",".csv",".png",".md",".txt"):
                archive.write(f, arcname=f.name)
    print(f"[resultado] {status}; arquivo={package}",flush=True)
    if status!="completed":
        raise RuntimeError(f"Execucao interrompida. Consultar {out/'erro_execucao.txt'}")
    print("[conclusao] Comparacao de todos os metodos gravada e graficos exibidos.", flush=True)
    return report


if __name__=="__main__":
    main()
