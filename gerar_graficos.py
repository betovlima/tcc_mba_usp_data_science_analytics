from __future__ import annotations

import argparse
from pathlib import Path

from visualizacao.heatmap_mensal import (
    generate_monthly_realized_pnl_artifacts,
    generate_monthly_return_artifacts,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = ROOT / "output" / "reproducao_v1"
DEFAULT_CHARTS_DIR = DEFAULT_RESULTS_DIR / "graficos"

VARIANT_FILES = {
    "control": {
        "predictions": "control_predictions.csv",
        "trades": "control_trades.csv",
    },
    "soft": {
        "predictions": "soft_horizon_consensus_predictions.csv",
        "trades": "soft_horizon_consensus_trades.csv",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gera graficos reproduziveis a partir dos CSVs "
            "da reproducao do TCC."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Diretorio que contem predictions e trades CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CHARTS_DIR,
        help="Diretorio de destino dos graficos.",
    )
    parser.add_argument(
        "--chart",
        choices=("monthly-return", "realized-pnl", "all"),
        default="monthly-return",
        help="Grafico que sera gerado.",
    )
    parser.add_argument(
        "--variant",
        choices=("control", "soft", "all"),
        default="control",
        help="Variante da estrategia.",
    )
    parser.add_argument(
        "--mode",
        choices=("simulation", "reference", "excess", "all"),
        default="simulation",
        help="Serie usada no heatmap de retorno mensal.",
    )
    return parser.parse_args()


def _print_artifacts(
    variant: str,
    chart: str,
    artifacts: dict[str, Path],
) -> None:
    print(f"[graficos] variante={variant} grafico={chart}")
    for key, path in artifacts.items():
        print(f"[graficos] {key}={path}")


def main() -> int:
    args = _parse_args()
    variants = (
        tuple(VARIANT_FILES)
        if args.variant == "all"
        else (args.variant,)
    )
    modes = (
        ("simulation", "reference", "excess")
        if args.mode == "all"
        else (args.mode,)
    )

    for variant in variants:
        files = VARIANT_FILES[variant]
        predictions = args.results_dir / files["predictions"]
        trades = args.results_dir / files["trades"]

        if args.chart in {"monthly-return", "all"}:
            artifacts = generate_monthly_return_artifacts(
                predictions,
                args.output_dir,
                variant=variant,
                modes=modes,
            )
            _print_artifacts(variant, "monthly-return", artifacts)

        if args.chart in {"realized-pnl", "all"}:
            artifacts = generate_monthly_realized_pnl_artifacts(
                predictions,
                trades,
                args.output_dir,
                variant=variant,
            )
            _print_artifacts(variant, "realized-pnl", artifacts)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
