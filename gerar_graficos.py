from __future__ import annotations

import argparse
from pathlib import Path

from visualizacao.heatmap_mensal import generate_monthly_return_artifacts


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = ROOT / "output" / "reproducao_v1"
DEFAULT_CHARTS_DIR = DEFAULT_RESULTS_DIR / "graficos"

VARIANT_FILES = {
    "control": "control_predictions.csv",
    "soft": "soft_horizon_consensus_predictions.csv",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gera heatmaps mensais reproduziveis a partir dos CSVs "
            "da reproducao do TCC."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Diretorio que contem os predictions CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CHARTS_DIR,
        help="Diretorio de destino dos graficos.",
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
        help="Serie mostrada no heatmap.",
    )
    return parser.parse_args()


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
        predictions = args.results_dir / VARIANT_FILES[variant]
        artifacts = generate_monthly_return_artifacts(
            predictions,
            args.output_dir,
            variant=variant,
            modes=modes,
        )
        print(f"[graficos] variante={variant}")
        for key, path in artifacts.items():
            print(f"[graficos] {key}={path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
