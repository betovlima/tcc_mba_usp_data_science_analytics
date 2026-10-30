"""Importacao auditavel do Control oficial de pesquisa: Strategy #30.

Spyder F5. Procura o ZIP original no diretorio do projeto, em
evidencias/ ou Downloads; CONTROL_STRATEGY30_ZIP pode definir outro caminho.
Nao retreina o modelo nem altera a estrategia operacional.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import traceback

from pesquisa.control_operacional_strategy30 import (
    ROOT, carregar_contrato, validar_control_strategy30,
)
from pesquisa.persistencia_resultados import salvar_csv, salvar_json, salvar_texto
from pesquisa.pacote_analise_switching import gerar_zip_resultados
from pesquisar_expansao_u67_spyder import _write_json

SCRIPT_VERSION = "control-strategy30-reference-v1.0.0"
OUTPUT = ROOT / "output" / "controle_strategy30"
FILE_GLOB = "market_cycle_trader_20261029T010353-f60d8486*.zip"


def localizar_zip_original() -> Path:
    explicit = os.environ.get("CONTROL_STRATEGY30_ZIP")
    if explicit:
        return Path(explicit).expanduser().resolve()

    folders = [
        ROOT / "evidencias",
        ROOT,
        ROOT / "dados",
        Path.home() / "Downloads",
    ]
    choices: list[Path] = []
    for folder in folders:
        if folder.is_dir():
            choices.extend(sorted(folder.glob(FILE_GLOB)))
    for path in choices:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Pacote Strategy #30 nao encontrado. Copie o ZIP original para "
        "evidencias/ ou Downloads, ou defina CONTROL_STRATEGY30_ZIP."
    )


def _run(output: Path) -> dict:
    archive = localizar_zip_original()
    contract = carregar_contrato()
    audit, predictions, trades = validar_control_strategy30(
        archive, contrato=contract,
    )
    salvar_csv(
        predictions, output / "control_predictions.csv", index=False,
        float_format="%.17g",
    )
    salvar_csv(trades, output / "control_trades.csv", index=False)
    result = {
        "version": SCRIPT_VERSION,
        "status": "completed",
        "current_control_id": contract["id"],
        "control_metrics": audit,
        "source_file": archive.name,
        "historical_checkpoint": contract["archival_reference"],
        "comparison_is_allowed": False,
        "comparison_block_reason": (
            "A referencia original foi importada e validada, mas o snapshot "
            "RAW e a politica executavel equivalentes devem ser reproduzidos "
            "antes de comparar novos algoritmos no mesmo calendario."
        ),
    }
    salvar_json(output / "resultado.json", result, _write_json)
    salvar_texto(
        output / "resumo_para_analise.md",
        "# Control de pesquisa: Strategy #30\n\n"
        f"Versao: {SCRIPT_VERSION}\n\n"
        "Universo oficial: 75 ativos, sem exclusoes nesta referencia.\n"
        f"Periodo: {audit['first_session']} ate {audit['last_session']}.\n"
        f"Capital inicial: US$ {audit['initial_capital']:,.2f}.\n"
        f"Capital final: US$ {audit['ending_capital']:,.2f}.\n"
        f"Sharpe: {audit['sharpe']:.6f}; "
        f"Drawdown maximo: {audit['max_drawdown']:.4%}.\n"
        f"Sessoes: {audit['sessions']}; folds: {audit['fold_counts']}.\n"
        f"Assinatura do ZIP: {audit['source_zip_sha256']}.\n\n"
        "Este resultado foi importado e verificado, nao reproduzido "
        "com RAW/Corporate Actions no motor independente. A integridade "
        "do pacote nao demonstra equivalencia dos pipelines.\n"
        "Comparacoes de CASH devem aguardar dados e politica iguais "
        "ao backtest congelado. A serie antiga do TCC nao constitui "
        "uma comparacao valida com esta referencia.\n",
    )
    print(
        f"[control] Strategy #30 validada: 75 ativos, "
        f"{audit['sessions']} sessoes, US$ {audit['ending_capital']:,.2f}",
        flush=True,
    )
    print("[control] modo=importacao_verificada; pesquisa comparativa bloqueada ate paridade", flush=True)
    return result


def main() -> dict:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = OUTPUT / run_id
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = _run(output)
        package = gerar_zip_resultados(
            output, versao=SCRIPT_VERSION, status="completed",
        )
        print(f"[pacote] resultados para analise: {package}", flush=True)
        return result
    except Exception as exc:
        try:
            salvar_json(output / "erro_execucao.json", {
                "status": "failed", "version": SCRIPT_VERSION,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=20),
            }, _write_json)
            package = gerar_zip_resultados(
                output, versao=SCRIPT_VERSION, status="failed",
            )
            print(f"[pacote] diagnostico: {package}", flush=True)
        except Exception as archive_error:
            print(f"[pacote] nao foi possivel compactar: {archive_error}", flush=True)
        raise


if __name__ == "__main__":
    main()
