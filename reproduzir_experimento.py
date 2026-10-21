"""Entrada única da reprodução científica de rotação, sem cardinalidade fixa.

A seleção dos ativos é lida dos manifestos e dos dados disponíveis; as
regras de validação cronológica e a estratégia LightGBM Control são mantidas.
O resultado histórico é referência descritiva, nunca uma trava.
"""
from __future__ import annotations

from reproducao.control import executar_control

if __name__ == "__main__":
    executar_control()
