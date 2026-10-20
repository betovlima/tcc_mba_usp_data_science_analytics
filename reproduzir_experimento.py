"""Ponto de entrada do TCC — reproduz o Control de 75 ativos.

Campanha científica U67+10 / Control / LightGBM / swing trade.
Referência histórica: capital simulado US$ 587.165.941,13.
A reprodução usa os arquivos de mercado locais disponíveis e mostra
eventuais diferenças numéricas sem abortar por hash, checkpoint ou
versão de branch.

Execute no Spyder:
    %run reproduzir_experimento.py
"""

from __future__ import annotations

from reproducao.control_u77 import executar_control_u77

if __name__ == "__main__":
    executar_control_u77()
