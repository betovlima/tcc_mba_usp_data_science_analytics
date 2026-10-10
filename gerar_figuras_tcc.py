"""Figuras e dados derivados dos registros preservados, sem treinar modelos."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figuras"
COLORS = ["#1764A2", "#C57719", "#397B4B"]
GROUPS = [
    ("Retornos", 0, 9),
    ("Volatilidade", 9, 17),
    ("Médias exponenciais", 17, 28),
    ("Oscilação e amplitude", 28, 30),
    ("Extremos e canais", 30, 42),
    ("Eficiência de tendência", 42, 46),
    ("Aceleração e expansão", 46, 49),
    ("Volume", 49, 52),
]


def pt(value: float, decimals: int = 0) -> str:
    return f"{value:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def features() -> list[str]:
    tree = ast.parse((ROOT / "engine/rotacao.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "ROTATION_FEATURES"
            for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("Lista de atributos ausente.")


def style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    for name in ("bottom", "left"):
        ax.spines[name].set_linewidth(1.5)
        ax.spines[name].set_color("black")
    ax.tick_params(labelsize=10, colors="black")
    ax.set_facecolor("none")


def save(fig: plt.Figure, number: int, name: str) -> None:
    stem = OUT / f"figura_{number:02d}_{name}"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    svg = stem.with_suffix(".svg")
    fig.savefig(svg, facecolor="white", metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def flow() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 6.5))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str) -> None:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="black", linewidth=1))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=11)

    def arrow(a: tuple, b: tuple, dashed: bool = False) -> None:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10,
                                    color="black", linewidth=1,
                                    linestyle="--" if dashed else "-"))

    box(.35, 5.86, 5.5, .53, "Séries diárias e eventos corporativos\nCópia congelada e identidade dos arquivos")
    box(.35, 5.05, 5.5, .54, "Preparação por ativo\nDatas, duplicatas, valores ausentes e splits")
    arrow((3.1, 5.86), (3.1, 5.59))
    box(.35, 4.15, 2.55, .64, "X(i,t): 52 atributos\nPreços e volumes\ndisponíveis até t")
    box(3.3, 4.15, 2.55, .64, "y(i,t): utilidade\nTrajetória futura de preços\nRótulo até t + 60 sessões")
    arrow((1.625, 5.05), (1.625, 4.79))
    arrow((4.575, 5.05), (4.575, 4.79))
    box(.35, 3.23, 5.5, .64, "Treino cronológico e calibração da margem\nModelos por ativo e intervalos de separação\nReajuste antes de cada bloco de teste")
    arrow((1.625, 4.15), (1.625, 3.87))
    arrow((4.575, 4.15), (4.575, 3.87))
    box(.35, 2.32, 2.55, .66, "Previsão em cada decisão\nModelo ajustado + X(i,t)\nUtilidades estimadas")
    box(3.3, 2.32, 2.55, .66, "Diagnósticos\nMAE e RMSE no teste\nGanho no ajuste final")
    arrow((1.625, 3.23), (1.625, 2.98))
    arrow((4.575, 3.23), (4.575, 2.98), True)
    arrow((2.90, 2.65), (3.30, 2.65), True)
    box(.35, 1.46, 2.55, .66, "Regra de alocação\nManter, trocar ou caixa\nComparação das utilidades")
    box(3.3, 1.46, 2.55, .66, "Compra e manutenção\nMesmo universo fixo,\ncapital e datas")
    arrow((1.625, 2.32), (1.625, 2.12))
    # A referência usa as séries preparadas, sem depender do modelo.
    ax.plot([5.85, 6.05, 6.05], [5.32, 5.32, 1.79], color="black", linewidth=1)
    arrow((6.05, 1.79), (5.85, 1.79))
    box(.35, .62, 2.55, .66, "Execução e contabilização\nAbertura seguinte, posição,\ncaixa e taxas")
    arrow((1.625, 1.46), (1.625, 1.28))
    box(.35, .03, 5.5, .43, "Curvas diárias e análise financeira\nCapital, retornos, Sharpe e drawdown")
    arrow((1.625, .62), (1.625, .46))
    arrow((4.575, 1.46), (4.575, .46))
    save(fig, 1, "fluxo_dados")


def timeline(folds: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 2.95))
    fig.subplots_adjust(left=.10, right=.98, bottom=.35, top=.94)
    palette = ["#AAC7DE", "#D6D6D6", "#EDC896", "#A8C6AA"]
    start = pd.Timestamp("2016-10-17")
    for j, fold in enumerate(folds):
        segments = [
            (start, pd.Timestamp(fold["train_end"]), 0),
            (pd.Timestamp(fold["train_end"]), pd.Timestamp(fold["calibration_start"]), 1),
            (pd.Timestamp(fold["calibration_start"]), pd.Timestamp(fold["calibration_end"]), 2),
            (pd.Timestamp(fold["calibration_end"]), pd.Timestamp(fold["test_start"]), 1),
            (pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]), 3),
        ]
        for a, b, color in segments:
            left, right = mdates.date2num(a), mdates.date2num(b)
            ax.barh(2-j, right-left, left=left, height=.38, color=palette[color], linewidth=0)
    ax.set_yticks([2, 1, 0], ["Fold 1", "Fold 2", "Fold 3"])
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("Calendário da pesquisa", fontsize=11)
    ax.set_xlim(mdates.date2num(start)-70, mdates.date2num(pd.Timestamp("2026-10-06"))+90)
    ax.set_ylim(-.5, 2.5)
    style_axis(ax)
    labels = ["Treino inicial", "Separação", "Calibração", "Teste financeiro"]
    handles = [Rectangle((0, 0), 1, 1, facecolor=c, linewidth=0) for c in palette]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .005),
               ncol=2, frameon=False, fontsize=10)
    save(fig, 2, "janelas_temporais")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "Nimbus Sans", "font.size": 11,
                         "svg.fonttype": "path",
                         "axes.labelsize": 11})
    curve_path = ROOT / "evidencias/series_financeiras_v1.22.0-dev.10.csv"
    evidence_path = ROOT / "evidencias/eligibilidade_v1.22.0-dev.10.json"
    evidence = json.loads(evidence_path.read_text())
    curve = pd.read_csv(curve_path, float_precision="round_trip")
    curve["timestamp"] = pd.to_datetime(curve["timestamp"], utc=True)
    assert len(curve) == evidence["sessions"] == 1560
    assert curve["timestamp"].is_monotonic_increasing and curve["timestamp"].is_unique
    assert np.isfinite(curve[["strategy_equity", "buy_hold_equity"]]).all().all()
    assert (curve[["strategy_equity", "buy_hold_equity"]] > 0).all().all()
    assert abs(curve.strategy_equity.iloc[-1] - evidence["official_reproduction"]["ending_capital"]) < 1e-7
    assert abs(curve.buy_hold_equity.iloc[-1] - evidence["official_reproduction"]["buy_hold_ending_capital"]) < 1e-7
    folds = evidence["metrics"]["corrigido"]["folds"]
    names = features()
    assert len(names) == 52
    diagnostics = evidence["predictive_diagnostics"]["folds"]
    errors = pd.DataFrame([{k: f[k] for k in ["fold_id", "rows", "mae", "rmse"]} for f in diagnostics])
    gain_rows = []
    for f in diagnostics:
        assets = f["by_asset"]
        assert len(assets) == 65
        for label, begin, end in GROUPS:
            gain = np.mean([sum(a["final_fit_feature_importance_gain"][k]
                                for k in names[begin:end]) for a in assets.values()])
            gain_rows.append({"fold_id": f["fold_id"], "grupo": label,
                              "atributos": end-begin, "modelos": len(assets), "ganho_medio": float(gain)})
    gains = pd.DataFrame(gain_rows)
    assert np.allclose(gains.groupby("fold_id").ganho_medio.sum(), 1., atol=1e-12)
    annual_rows, prior = [], np.array([10000., 10000.])
    for year, part in curve.groupby(curve.timestamp.dt.year):
        ending = part[["strategy_equity", "buy_hold_equity"]].iloc[-1].to_numpy()
        returns = ending / prior - 1
        annual_rows.append({"ano": int(year), "rotacao": float(returns[0]),
                            "compra_manutencao": float(returns[1]), "parcial": year in (2020, 2026)})
        prior = ending
    annual = pd.DataFrame(annual_rows)
    assert np.allclose(np.prod(1 + annual[["rotacao", "compra_manutencao"]], axis=0), prior/10000., rtol=1e-12)
    for name, data in [("retornos_anuais", annual), ("erros_por_fold", errors), ("ganho_por_grupo_fold", gains)]:
        data.to_csv(OUT/f"{name}.csv", index=False, float_format="%.17g")
    flow()
    timeline(folds)
    dates = curve.timestamp.dt.tz_localize(None)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    fig.subplots_adjust(left=.17, right=.98, bottom=.20, top=.87)
    for i, (column, label) in enumerate([("strategy_equity", "Rotação"), ("buy_hold_equity", "Compra e manutenção")]):
        ax.plot(dates, curve[column], color=COLORS[i], linewidth=1.2, label=label)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: pt(x)))
    ax.set_ylabel("Capital (US$)")
    ax.set_xlabel("Data")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, fontsize=10)
    style_axis(ax)
    save(fig, 3, "capital_logaritmico")
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    fig.subplots_adjust(left=.14, right=.98, bottom=.22, top=.87)
    for i, (column, label) in enumerate([("strategy_equity", "Rotação"), ("buy_hold_equity", "Compra e manutenção")]):
        dd = curve[column] / curve[column].cummax() - 1
        ax.plot(dates, dd, color=COLORS[i], linewidth=.8, label=label)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: pt(100*x)+"%"))
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Data")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, fontsize=10)
    style_axis(ax)
    save(fig, 4, "drawdown")
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    fig.subplots_adjust(left=.14, right=.98, bottom=.21, top=.86)
    x = np.arange(len(annual))
    for i, (column, label) in enumerate([("rotacao", "Rotação"), ("compra_manutencao", "Compra e manutenção")]):
        ax.bar(x+(i-.5)*.32, annual[column], width=.32, color=COLORS[i], label=label)
    ax.set_xticks(x, [str(y)+( "*" if p else "") for y,p in zip(annual.ano,annual.parcial)])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: pt(v*100)+"%"))
    ax.set_ylabel("Retorno no período (%)")
    ax.set_xlabel("Ano (* período parcial)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, fontsize=10)
    style_axis(ax)
    save(fig, 5, "retornos_anuais")
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    fig.subplots_adjust(left=.13, right=.98, bottom=.23, top=.86)
    x = np.arange(len(errors))
    for i, column in enumerate(["mae", "rmse"]):
        ax.bar(x+(i-.5)*.27, errors[column], width=.27, color=COLORS[i], label=column.upper())
    ax.set_xticks(x, [f"Fold {i}" for i in errors.fold_id])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: pt(v,2)))
    ax.set_ylabel("Erro na escala da utilidade")
    ax.set_xlabel("Bloco de teste")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, fontsize=10)
    style_axis(ax)
    save(fig, 6, "erros_fora_amostra")
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    fig.subplots_adjust(left=.35, right=.98, bottom=.16, top=.91)
    y = np.arange(len(GROUPS))
    for i in range(3):
        vals = gains.loc[gains.fold_id == i+1, "ganho_medio"].to_numpy()
        ax.barh(y+(i-1)*.22, vals, height=.22, color=COLORS[i], label=f"Fold {i+1}")
    ax.set_yticks(y, [label for label, _, _ in GROUPS], fontsize=10)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: pt(v*100)+"%"))
    ax.set_xlabel("Participação média no ganho do ajuste (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, fontsize=10)
    style_axis(ax)
    save(fig, 7, "ganho_grupos")
    provenance = {
        "natureza": "Derivações descritivas dos registros preservados; nenhum modelo foi reajustado.",
        "execucao_cientifica_commit": "3f8d1b1a2784721201938071471e91fd4282f628",
        "fontes": [
            str(curve_path.relative_to(ROOT)),
            str(evidence_path.relative_to(ROOT)),
        ],
        "retornos_anuais": "Capital final de cada ano / capital da última sessão do ano anterior - 1; primeiro intervalo começa em US$ 10.000; 2020 e 2026 parciais; sem anualização.",
        "importancias": "Soma dos ganhos normalizados de atributos do grupo em cada modelo; média simples entre 65 modelos finais em cada fold. Soma dos oito grupos igual a um.",
        "limite_importancias": "Ganhos do treinamento; não são efeitos causais nem importâncias medidas no teste. Grupos têm números diferentes de atributos.",
        "figuras": [p.name for p in sorted(OUT.glob("figura_*"))],
    }
    (OUT/"procedencia_figuras.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+"\n")
    print(f"Sete figuras e três tabelas de dados verificadas em {OUT}.")


if __name__ == "__main__":
    main()
