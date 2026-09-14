"""Stable Spyder entrypoint for the MBA capstone backtest.

Open this file in Spyder and press F5.

The first project milestone is environment validation. The certified backtest
engine will be migrated into this same entrypoint in the next project stage,
without depending on the Market Cycle Trader web API.
"""

from __future__ import annotations

import os
import platform
import random
import sys
from importlib import metadata

import numpy as np


PROJECT_NAME = "USP MBA Data Science & Analytics — Reproducible Backtest"
RANDOM_SEED = 42
NUMERIC_THREAD_LIMIT = 1


REQUIRED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "exchange-calendars",
    "matplotlib",
    "lightgbm",
    "xgboost",
    "pymongo",
    "python-dotenv",
    "pydantic",
)


def configure_determinism() -> None:
    """Configure the deterministic defaults used by the academic backtest."""
    os.environ.setdefault("OMP_NUM_THREADS", str(NUMERIC_THREAD_LIMIT))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(NUMERIC_THREAD_LIMIT))
    os.environ.setdefault("MKL_NUM_THREADS", str(NUMERIC_THREAD_LIMIT))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(NUMERIC_THREAD_LIMIT))
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in REQUIRED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "NOT INSTALLED"
    return versions


def validate_environment() -> None:
    print("=" * 72)
    print(PROJECT_NAME)
    print("=" * 72)
    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {sys.version.split()[0]}")
    print(f"Operating system  : {platform.platform()}")
    print(f"Random seed       : {RANDOM_SEED}")
    print(f"Numeric threads   : {NUMERIC_THREAD_LIMIT}")
    print()
    print("Dependencies:")

    versions = package_versions()
    missing: list[str] = []
    for package, version in versions.items():
        print(f"  {package:<20} {version}")
        if version == "NOT INSTALLED":
            missing.append(package)

    if missing:
        raise RuntimeError(
            "Missing project dependencies: "
            + ", ".join(missing)
            + ". Install them with: python -m pip install -r requirements.txt"
        )

    print()
    print("Environment validation: OK")
    print("Backtest engine migration: pending next project stage")


def main() -> int:
    configure_determinism()
    validate_environment()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
