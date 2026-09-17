"""Bootstrap autocontido dos snapshots Mongo/Tiingo para atribuicao de capital.

Versao: mongo-tiingo-snapshot-bootstrap-v2.0.0

Mantem a coleta v1 e acrescenta uma etapa obrigatoria de validacao dos
splitFactor da Tiingo. Somente eventos cuja ruptura mecanica de preco e
compativel com um split sao gravados em dados/desdobramentos.
"""
from __future__ import annotations

import preparar_snapshots_mongo_tiingo_v1 as v1
from validar_desdobramentos_tiingo import build_validated_splits

VERSION = "mongo-tiingo-snapshot-bootstrap-v2.0.0"


def main() -> int:
    args = v1.parse_args()
    v1.VERSION = VERSION
    v1.log(VERSION)
    v1.log(f"Destino: {v1.DATA_ROOT}")

    if args.only in ("all", "mongo"):
        v1.export_mongo(args.force_mongo)

    if args.only in ("all", "tiingo"):
        v1.download_tiingo(args.force_tiingo)
        manifest = build_validated_splits()
        v1.log(
            "Validacao de splits concluida | "
            f"candidatos={manifest['total_candidatos_eod']} | "
            f"aceitos={manifest['total_desdobramentos_aceitos']} | "
            f"rejeitados={manifest['total_candidatos_rejeitados']}"
        )

    v1.log(
        "Snapshots prontos. Agora rode "
        "diagnosticar_atribuicao_capital_mongo_tiingo.py --mode pair"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
