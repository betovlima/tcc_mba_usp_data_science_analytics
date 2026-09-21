"""Compatibilidade com o nome historico backtest.py.

A reproducao oficial do TCC esta em reproduzir_experimento_spyder.py.
Este arquivo apenas executa esse roteiro sequencial.
"""
from pathlib import Path
import runpy

SCRIPT = Path(__file__).resolve().with_name("reproduzir_experimento_spyder.py")
runpy.run_path(str(SCRIPT), run_name="__main__")
