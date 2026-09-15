from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output"
DEFAULT_DEST = ROOT / "analysis" / "tcc_backtest_output_analysis.xlsx"


def args():
    parser = argparse.ArgumentParser(description="Build the Excel audit workbook from output/.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DEST)
    return parser.parse_args()


def load_data(output_dir: Path):
    required = ["backtest_result.json", "equity_curve.csv", "folds.csv", "trades.csv", "summary.txt"]
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError("Missing files: " + ", ".join(missing))
    result = json.loads((output_dir / "backtest_result.json").read_text(encoding="utf-8"))
    equity = pd.read_csv(output_dir / "equity_curve.csv")
    folds = pd.read_csv(output_dir / "folds.csv")
    trades = pd.read_csv(output_dir / "trades.csv")
    summary = (output_dir / "summary.txt").read_text(encoding="utf-8")
    return result, equity, folds, trades, summary


def build_cycles(trades: pd.DataFrame) -> pd.DataFrame:
    buys = trades.loc[trades["action"].eq("BUY"), [
        "asset", "timestamp", "best_alternative_asset", "best_alternative_return", "opportunity_cost"
    ]].rename(columns={"timestamp": "entry_timestamp"})
    sells = trades.loc[trades["action"].isin(["SELL", "FINAL_SELL"]), [
        "entry_timestamp", "timestamp", "asset", "entry_price", "execution_price", "quantity",
        "realized_pnl", "position_return", "holding_bars", "total_fee", "walk_forward_fold",
        "rotation_id", "position_entry_score", "current_score", "best_score", "best_vs_current_gap",
        "effective_switch_margin", "current_asset_rank", "maximum_favorable_excursion",
        "maximum_adverse_excursion", "profit_capture_ratio"
    ]].copy()
    cycles = sells.merge(buys, on=["asset", "entry_timestamp"], how="left")
    cycles = cycles.rename(columns={
        "timestamp": "exit_timestamp", "execution_price": "exit_price", "total_fee": "exit_fee",
        "walk_forward_fold": "fold", "position_entry_score": "entry_score",
        "maximum_favorable_excursion": "mfe", "maximum_adverse_excursion": "mae"
    })
    cycles["result"] = cycles["position_return"].map(lambda x: "WIN" if x > 0 else "LOSS" if x < 0 else "FLAT")
    return cycles


def flatten(value, prefix=""):
    rows = []
    if isinstance(value, dict):
        for key, child in value.items():
            rows.extend(flatten(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        rows.append((prefix, "list", json.dumps(value, ensure_ascii=False, default=str)))
    else:
        rows.append((prefix, type(value).__name__, value))
    return rows


def style_workbook(path: Path, result: dict, summary: str):
    wb = load_workbook(path)
    blue, navy, white = "2F75B5", "17365D", "FFFFFF"
    for ws in wb.worksheets:
        ws.sheet_view.showGridLines = False
        ws.freeze_panes = "A2"
        for cell in ws[1]:
            cell.fill = PatternFill("solid", fgColor=blue)
            cell.font = Font(color=white, bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for col in range(1, min(ws.max_column, 20) + 1):
            width = max(11, min(28, max(len(str(ws.cell(r, col).value or "")) for r in range(1, min(ws.max_row, 40) + 1)) + 2))
            ws.column_dimensions[get_column_letter(col)].width = width

    guide = wb["00_Guia"]
    guide.insert_rows(1, 1)
    guide.merge_cells("A1:F1")
    guide["A1"] = "Auditoria do backtest no Excel"
    guide["A1"].fill = PatternFill("solid", fgColor=navy)
    guide["A1"].font = Font(color=white, bold=True, size=15)

    equity = wb["02_Equity"]
    last_eq = equity.max_row
    base_headers = equity.max_column
    derived = ["Strategy Daily Return", "Benchmark Daily Return", "Strategy Peak", "Strategy Drawdown", "Strategy Log Return", "Strategy Growth x", "Benchmark Growth x", "Excess Daily Return", "Benchmark Peak", "Benchmark Drawdown"]
    for offset, name in enumerate(derived, start=1):
        equity.cell(1, base_headers + offset, name)
    for r in range(2, last_eq + 1):
        h = base_headers + 1
        if r > 2:
            equity.cell(r, h, f"=B{r}/B{r-1}-1")
            equity.cell(r, h + 1, f"=C{r}/C{r-1}-1")
            equity.cell(r, h + 4, f"=LN(B{r}/B{r-1})")
            equity.cell(r, h + 7, f"={get_column_letter(h)}{r}-{get_column_letter(h+1)}{r}")
        equity.cell(r, h + 2, f"=MAX(10000,B{r})" if r == 2 else f"=MAX({get_column_letter(h+2)}{r-1},B{r})")
        equity.cell(r, h + 3, f"=B{r}/{get_column_letter(h+2)}{r}-1")
        equity.cell(r, h + 5, f"=B{r}/10000")
        equity.cell(r, h + 6, f"=C{r}/10000")
        equity.cell(r, h + 8, f"=MAX(10000,C{r})" if r == 2 else f"=MAX({get_column_letter(h+8)}{r-1},C{r})")
        equity.cell(r, h + 9, f"=C{r}/{get_column_letter(h+8)}{r}-1")
    dd_col = get_column_letter(base_headers + 4)
    equity.conditional_formatting.add(f"{dd_col}2:{dd_col}{last_eq}", ColorScaleRule(start_type="min", start_color="F8696B", mid_type="percentile", mid_value=50, mid_color="FFEB84", end_type="max", end_color="63BE7B"))

    cycles = wb["05_Cycles"]
    last_cycle = cycles.max_row
    cycles["Z1"] = "LN(1+Return)"
    for r in range(2, last_cycle + 1):
        cycles[f"Z{r}"] = f"=LN(1+H{r})"

    kpi = wb["01_KPIs"]
    kpi.insert_rows(1, 2)
    kpi.merge_cells("A1:F1")
    kpi["A1"] = "Motor x fórmulas do Excel"
    kpi["A1"].fill = PatternFill("solid", fgColor=navy)
    kpi["A1"].font = Font(color=white, bold=True, size=15)
    last_trades = wb["04_Trades"].max_row
    last_folds = wb["03_Folds"].max_row
    metrics = result["metrics"]
    worst_fold = min(float(x["strategy_return"]) for x in metrics["walk_forward_folds"])
    rows = [
        ("Initial Capital", metrics["initial_capital"], "='03_Folds'!K2", "Capital inicial."),
        ("Final Capital", metrics["strategy_ending_capital"], f"='02_Equity'!B{last_eq}", "Última equity."),
        ("Strategy Return", metrics["strategy_return"], "=C5/C4-1", "Final/inicial - 1."),
        ("Test Calendar Years", metrics["test_calendar_years"], f"=('02_Equity'!A{last_eq}-'02_Equity'!A2)/365.25", "Período OOS."),
        ("CAGR", metrics["strategy_cagr"], "=(C5/C4)^(1/C7)-1", "Compound anualizado."),
        ("Sharpe", metrics["strategy_sharpe"], f"=AVERAGE('02_Equity'!H3:H{last_eq})/STDEV.S('02_Equity'!H3:H{last_eq})*SQRT(252)", "Retorno diário anualizado."),
        ("Max Drawdown", metrics["strategy_maximum_drawdown"], f"=MIN('02_Equity'!K2:K{last_eq})", "Pior pico→vale."),
        ("Buys", metrics["simulated_buys"], f'=COUNTIF(\'04_Trades\'!B2:B{last_trades},"BUY")', "Compras."),
        ("Sells", metrics["simulated_sells"], f'=COUNTIF(\'04_Trades\'!B2:B{last_trades},"SELL")+COUNTIF(\'04_Trades\'!B2:B{last_trades},"FINAL_SELL")', "Vendas."),
        ("Rotations", metrics["capital_rotations"], f'=COUNTIF(\'04_Trades\'!B2:B{last_trades},"SELL")', "FINAL_SELL não é rotação."),
        ("Avg Holding Bars", metrics["average_holding_bars"], f"=AVERAGE('05_Cycles'!I2:I{last_cycle})", "Holding médio."),
        ("Geometric Trade Return", metrics["geometric_trade_return"], f"=EXP(AVERAGE('05_Cycles'!Z2:Z{last_cycle}))-1", "Média geométrica."),
        ("Total Fees", metrics["total_transaction_fees"], f"=SUM('04_Trades'!L2:L{last_trades})", "Custos."),
        ("Cash Days", metrics["cash_days"], f'=COUNTIF(\'02_Equity\'!D2:D{last_eq},"CASH")', "Dias em CASH."),
        ("Decision Sessions", metrics["decision_diagnostics_rows"], f"=COUNTA('02_Equity'!A2:A{last_eq})", "Sessões OOS."),
        ("Worst Fold Return", worst_fold, f"=MIN('03_Folds'!M2:M{last_folds})", "Pior fold."),
    ]
    kpi.append(["Métrica", "Motor / JSON", "Excel", "Diferença", "Status", "Reconstrução"])
    for i, (name, engine, formula, note) in enumerate(rows, start=4):
        kpi.append([name, engine, formula, f"=C{i}-B{i}", f'=IF(ABS(D{i})<1E-8,"OK","CHECK")', note])
    kpi.append([])
    kpi.append(["Ciclos", "Valor", "Uso"])
    kpi.append(["Wins", f'=COUNTIF(\'05_Cycles\'!Y2:Y{last_cycle},"WIN")', "Ciclos positivos"])
    kpi.append(["Losses", f'=COUNTIF(\'05_Cycles\'!Y2:Y{last_cycle},"LOSS")', "Ciclos negativos"])
    kpi.append(["Win Rate", f"=B{kpi.max_row-1}/(B{kpi.max_row-1}+B{kpi.max_row})", "Taxa de acerto"])

    chart = LineChart()
    chart.title = "Capital: Strategy vs Benchmark"
    chart.add_data(Reference(equity, min_col=2, max_col=3, min_row=1, max_row=last_eq), titles_from_data=True)
    chart.height, chart.width = 8, 16
    kpi.add_chart(chart, "H3")

    rec = wb["10_Reconciliation"]
    rec.append(["Capital inicial", "='01_KPIs'!B4"])
    rec.append(["PnL realizado", f'=SUMIF(\'04_Trades\'!B2:B{last_trades},"SELL",\'04_Trades\'!M2:M{last_trades})+SUMIF(\'04_Trades\'!B2:B{last_trades},"FINAL_SELL",\'04_Trades\'!M2:M{last_trades})'])
    rec.append(["BUY fees", f'=SUMIF(\'04_Trades\'!B2:B{last_trades},"BUY",\'04_Trades\'!L2:L{last_trades})'])
    rec.append(["Capital reconciliado", "=B2+B3-B4"])
    rec.append(["Capital equity", f"='02_Equity'!B{last_eq}"])
    rec.append(["Diferença", "=B5-B6"])

    raw = wb["09_JSON"]
    raw[raw.max_row + 2][0].value = "summary.txt"
    raw.cell(raw.max_row + 1, 1, summary)

    if hasattr(wb, "calculation"):
        wb.calculation.fullCalcOnLoad = True
        wb.calculation.forceFullCalc = True
        wb.calculation.calcMode = "auto"
    wb.save(path)


def main():
    config = args()
    result, equity, folds, trades, summary = load_data(config.output_dir)
    cycles = build_cycles(trades)

    guide = pd.DataFrame([
        ["backtest_result.json", "Resultado canônico", "Métricas e metadados", "Parcial"],
        ["equity_curve.csv", "Curva de capital", "CAGR, Sharpe, drawdown", "Sim"],
        ["folds.csv", "Walk-forward", "Retorno e compound por fold", "Sim"],
        ["trades.csv", "Livro-razão", "PnL, custos e diagnósticos", "Sim após decisão"],
        ["summary.txt", "Resumo", "Conferência", "Não necessário"],
    ], columns=["Arquivo", "Papel", "Auditoria", "Excel"])
    dictionary = pd.DataFrame([
        ["strategy_equity", "Capital ao fim da sessão"],
        ["effective_switch_margin", "Margem mínima para rotação"],
        ["current_score", "Utility da posição atual"],
        ["best_score", "Utility do melhor candidato"],
        ["best_vs_current_gap", "Diferença entre candidato e posição"],
        ["current_asset_rank", "Ranking da posição atual"],
        ["maximum_favorable_excursion", "MFE: maior excursão favorável"],
        ["maximum_adverse_excursion", "MAE: maior excursão adversa"],
        ["profit_capture_ratio", "Fração do MFE capturada"],
        ["opportunity_cost", "Melhor alternativa ex post - escolha"],
    ], columns=["Campo", "Significado"])
    flat = pd.DataFrame(flatten(result), columns=["JSON Path", "Type", "Value"])

    config.destination.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(config.destination, engine="openpyxl") as writer:
        guide.to_excel(writer, sheet_name="00_Guia", index=False)
        pd.DataFrame().to_excel(writer, sheet_name="01_KPIs", index=False)
        equity.to_excel(writer, sheet_name="02_Equity", index=False)
        folds.to_excel(writer, sheet_name="03_Folds", index=False)
        trades.to_excel(writer, sheet_name="04_Trades", index=False)
        cycles.to_excel(writer, sheet_name="05_Cycles", index=False)
        pd.DataFrame({"Asset": result.get("assets", [])}).to_excel(writer, sheet_name="06_Assets", index=False)
        equity.assign(Month=equity["timestamp"].str[:7]).groupby("Month", as_index=False).tail(1)[["Month", "strategy_equity", "buy_hold_equity"]].to_excel(writer, sheet_name="07_Monthly", index=False)
        dictionary.to_excel(writer, sheet_name="08_Dictionary", index=False)
        flat.to_excel(writer, sheet_name="09_JSON", index=False)
        pd.DataFrame(columns=["Passo", "Valor"]).to_excel(writer, sheet_name="10_Reconciliation", index=False)
    style_workbook(config.destination, result, summary)
    print(f"Created: {config.destination}")


if __name__ == "__main__":
    main()
