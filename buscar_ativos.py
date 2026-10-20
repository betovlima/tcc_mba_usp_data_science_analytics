"""Busca exploratória de ativos para o TCC, independente da reprodução oficial.

Este é um dos dois únicos scripts executáveis na raiz. A pesquisa não altera
automaticamente a composição do universo da reprodução.

No Spyder: abra buscar_ativos.py e execute com F5. A busca por agrupamentos
emprega atributos técnicos, PCA e clusterização; em modo alternativo preserva
a pesquisa histórica U67+n e os checkpoints financeiros da campanha.
"""
from __future__ import annotations

# %% Configuração explícita da pesquisa
# "agrupamentos": busca exploratória e shortlist, sem promoção de ativos.
# "expansao_u67": campanha histórica U67+n (não altera o Control oficial).
MODO_BUSCA = "agrupamentos"


# %% Execução
def main() -> None:
    if MODO_BUSCA == "agrupamentos":
        from reproducao.busca_agrupamentos import main as executar_agrupamentos
        executar_agrupamentos()
    elif MODO_BUSCA == "expansao_u67":
        from reproducao.expansao_u67 import main as executar_expansao
        executar_expansao()
    else:
        raise ValueError(
            "MODO_BUSCA inválido. Use 'agrupamentos' ou 'expansao_u67'."
        )


if __name__ == "__main__":
    main()
