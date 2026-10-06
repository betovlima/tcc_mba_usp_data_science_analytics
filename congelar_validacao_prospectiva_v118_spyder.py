"""CONGELAMENTO DA VALIDACAO PROSPECTIVA DA ASSINATURA v1.18.

Este arquivo NAO executa backtest financeiro.

Ele aplica a assinatura matematica ja congelada aos candidatos cujo capital
ainda nao foi consultado e congela, antes de qualquer replay, uma coorte de
validacao com cobertura deliberada do espectro do score.

Desenho one-shot:
- 8 ativos do tercil alto de S;
- 8 ativos do tercil medio de S;
- 8 ativos do tercil baixo de S;
- 8 candidatos dormant (A=0).

A amostragem dentro de cada estrato e pseudoaleatoria deterministica via
SHA-256, com salt congelado. Nao usa stage1_rank, nome do ativo ou qualquer
resultado financeiro para favorecer exemplos.

Depois desta execucao, NAO altere a lista com base em preferencia visual,
noticias, ticker, setor ou qualquer outra informacao.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import shutil
import time

import numpy as np
import pandas as pd

from pesquisas.directional_change_lightgbm import (
    criar_pacote_analise,
    sinal_sonoro_conclusao,
)


# %% 0 - Configuracao congelada
ROOT = Path(__file__).resolve().parent
SEARCH_OUT = ROOT / "output" / "busca_ativos"
RANKED_FILE = SEARCH_OUT / "intelligent_candidates_ranked.csv"
SEARCH_JSON = SEARCH_OUT / "asset_search.json"
FROZEN_SIGNATURE = (
    ROOT
    / "dados"
    / "assinatura_matematica"
    / "signature_v1_18_frozen.json"
)

OUT = ROOT / "output" / "validacao_prospectiva_v118"
LOCAL_FREEZE = (
    ROOT
    / "dados"
    / "assinatura_matematica"
    / "prospective_validation_cohort_v118.csv"
)

VERSION = "1.18.1-dev.1"
SCHEMA = "prospective-signature-validation-freeze-v1"
EXPECTED_SEARCH_VERSION = "1.17.0-dev.1"
EXPECTED_SEARCH_SCHEMA = "intelligent-asset-search-u59-v1"
EXPECTED_REFERENCE = "U59_WINNER"
EXPECTED_RANKED_ROWS = 446
EXPECTED_KNOWN_SMART20 = 20

PER_STRATUM = 8
SALT = "tcc-v118-prospective-u59-one-shot-v1"

KNOWN_SMART20 = {
    "SGA", "THO", "XNTK", "CIVB", "WDAY",
    "EXR", "PAYX", "FMBH", "SBFG", "ALNY",
    "SXC", "ICCC", "XEL", "EBMT", "VLRS",
    "PDFS", "SITC", "FDX", "FNWB", "MUX",
}


# %% 1 - Helpers
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_key(asset: str, stratum: str) -> str:
    payload = f"{SALT}|{stratum}|{asset}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise RuntimeError(
            "intelligent_candidates_ranked.csv incompleto. "
            "Colunas ausentes: " + ",".join(missing)
        )


# %% 2 - Guards de origem
started = time.perf_counter()

for path in (RANKED_FILE, SEARCH_JSON, FROZEN_SIGNATURE):
    if not path.exists():
        raise RuntimeError(
            "Arquivo necessario ausente: "
            f"{path}. Nao rode nova busca. Recupere o output congelado "
            "da busca 1.17.0."
        )

search_meta = json.loads(
    SEARCH_JSON.read_text(encoding="utf-8")
)
signature_meta = json.loads(
    FROZEN_SIGNATURE.read_text(encoding="utf-8")
)

if search_meta.get("research_version") != EXPECTED_SEARCH_VERSION:
    raise RuntimeError(
        "Versao da busca inesperada: "
        f"{search_meta.get('research_version')!r}"
    )
if search_meta.get("execution_schema") != EXPECTED_SEARCH_SCHEMA:
    raise RuntimeError(
        "Schema da busca inesperado: "
        f"{search_meta.get('execution_schema')!r}"
    )
if (
    (search_meta.get("stage2") or {}).get("score_reference")
    != EXPECTED_REFERENCE
):
    raise RuntimeError(
        "A busca de origem nao usa U59_WINNER como referencia."
    )
if (
    signature_meta.get("status")
    != "frozen_development_signature_pending_external_validation"
):
    raise RuntimeError(
        "A assinatura v1.18 nao esta marcada como congelada."
    )
if signature_meta.get("validation_reference") != EXPECTED_REFERENCE:
    raise RuntimeError(
        "Referencia da assinatura congelada nao e U59_WINNER."
    )


# %% 3 - Leitura do universo intocado
ranked = pd.read_csv(RANKED_FILE)
required = {
    "asset",
    "candidate_beats_u59_best_share",
    "candidate_score_std",
    "model_score_corr_u59_best",
    "model_score_sessions",
}
require_columns(ranked, required)

ranked["asset"] = (
    ranked["asset"].astype(str).str.strip().str.upper()
)
ranked = ranked.drop_duplicates(
    subset=["asset"],
    keep="first",
).copy()

if len(ranked) != EXPECTED_RANKED_ROWS:
    raise RuntimeError(
        "A coorte de busca mudou. "
        f"Esperado={EXPECTED_RANKED_ROWS}, observado={len(ranked)}."
    )

known_found = set(ranked["asset"]).intersection(KNOWN_SMART20)
if len(known_found) != EXPECTED_KNOWN_SMART20:
    raise RuntimeError(
        "Os 20 candidatos financeiros conhecidos nao foram encontrados "
        "integralmente no ranked original. "
        f"Encontrados={len(known_found)}."
    )

untouched = ranked[
    ~ranked["asset"].isin(KNOWN_SMART20)
].copy()

if len(untouched) != 426:
    raise RuntimeError(
        "Pool prospectivo deveria conter 426 candidatos intocados; "
        f"observado={len(untouched)}."
    )

untouched["beats_best_share"] = numeric(
    untouched["candidate_beats_u59_best_share"]
)
untouched["score_std"] = numeric(
    untouched["candidate_score_std"]
)
untouched["score_corr_best"] = numeric(
    untouched["model_score_corr_u59_best"]
)
untouched["abs_corr_best"] = (
    untouched["score_corr_best"].abs()
)
untouched["score_sessions"] = numeric(
    untouched["model_score_sessions"]
)

minimum_sessions = int(
    (
        (search_meta.get("signature") or {})
        .get("minimum_score_sessions")
        or 0
    )
)
if minimum_sessions <= 0:
    raise RuntimeError(
        "minimum_score_sessions ausente no asset_search.json."
    )

untouched["score_data_eligible"] = (
    untouched["beats_best_share"].notna()
    & untouched["score_std"].notna()
    & untouched["abs_corr_best"].notna()
    & (untouched["score_sessions"] >= minimum_sessions)
)
untouched["activation"] = (
    untouched["score_data_eligible"]
    & (untouched["beats_best_share"] > 0.0)
)
untouched["dormant"] = (
    untouched["score_data_eligible"]
    & (untouched["beats_best_share"] <= 0.0)
)

active_mask = untouched["activation"]
for feature in (
    "beats_best_share",
    "abs_corr_best",
    "score_std",
):
    untouched.loc[
        active_mask,
        f"rank_active_{feature}",
    ] = untouched.loc[
        active_mask,
        feature,
    ].rank(
        method="average",
        pct=True,
    )

untouched["specialist_score"] = np.nan
untouched.loc[
    active_mask,
    "specialist_score",
] = 1.0 - untouched.loc[
    active_mask,
    [
        "rank_active_beats_best_share",
        "rank_active_abs_corr_best",
        "rank_active_score_std",
    ],
].mean(axis=1)

# Score global apenas para auditoria do hurdle:
# dormant = 0; ativos = 1 + S.
untouched["validation_score"] = 0.0
untouched.loc[
    active_mask,
    "validation_score",
] = (
    1.0
    + untouched.loc[
        active_mask,
        "specialist_score",
    ]
)


# %% 4 - Estratos predeclarados
active = untouched[untouched["activation"]].copy()
if len(active) < 3 * PER_STRATUM:
    raise RuntimeError(
        "Poucos candidatos ativos para os tres estratos."
    )

active["active_percentile"] = active[
    "specialist_score"
].rank(
    method="average",
    pct=True,
)

active["validation_stratum"] = np.select(
    [
        active["active_percentile"] > (2.0 / 3.0),
        active["active_percentile"] > (1.0 / 3.0),
    ],
    [
        "active_high_S",
        "active_mid_S",
    ],
    default="active_low_S",
)

dormant = untouched[untouched["dormant"]].copy()
dormant["active_percentile"] = np.nan
dormant["validation_stratum"] = "dormant_A0"

candidate_pool = pd.concat(
    [active, dormant],
    ignore_index=True,
)

selection_parts = []
stratum_counts = {}

for stratum in (
    "active_high_S",
    "active_mid_S",
    "active_low_S",
    "dormant_A0",
):
    part = candidate_pool[
        candidate_pool["validation_stratum"] == stratum
    ].copy()
    stratum_counts[stratum] = int(len(part))
    if len(part) < PER_STRATUM:
        raise RuntimeError(
            f"Estrato {stratum} possui somente {len(part)} candidatos; "
            f"necessarios={PER_STRATUM}."
        )

    part["deterministic_sample_key"] = [
        deterministic_key(asset, stratum)
        for asset in part["asset"]
    ]
    chosen = part.sort_values(
        ["deterministic_sample_key", "asset"],
        ascending=[True, True],
    ).head(PER_STRATUM).copy()
    selection_parts.append(chosen)

selected = pd.concat(
    selection_parts,
    ignore_index=True,
)

if len(selected) != 4 * PER_STRATUM:
    raise RuntimeError(
        "A coorte congelada deveria conter 32 candidatos."
    )
if selected["asset"].duplicated().any():
    raise RuntimeError(
        "Duplicata detectada na coorte prospectiva."
    )
if set(selected["asset"]).intersection(KNOWN_SMART20):
    raise RuntimeError(
        "Leakage: candidato do Smart20 apareceu na validacao."
    )

stratum_order = {
    "active_high_S": 0,
    "active_mid_S": 1,
    "active_low_S": 2,
    "dormant_A0": 3,
}
selected["_stratum_order"] = selected[
    "validation_stratum"
].map(stratum_order)
selected = selected.sort_values(
    [
        "_stratum_order",
        "specialist_score",
        "asset",
    ],
    ascending=[True, False, True],
).drop(columns=["_stratum_order"])
selected["prospective_validation_rank"] = (
    np.arange(1, len(selected) + 1)
)
selected["research_version"] = VERSION
selected["execution_schema"] = SCHEMA
selected["financial_reference"] = EXPECTED_REFERENCE
selected["capital_observed_at_freeze"] = False
selected["selection_salt"] = SALT
selected["source_search_sha256"] = sha256_file(RANKED_FILE)
selected["frozen_signature_sha256"] = sha256_file(
    FROZEN_SIGNATURE
)


# %% 5 - Congelamento local auditavel
OUT.mkdir(parents=True, exist_ok=True)
LOCAL_FREEZE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

selection_columns = [
    "prospective_validation_rank",
    "asset",
    "validation_stratum",
    "activation",
    "dormant",
    "specialist_score",
    "validation_score",
    "active_percentile",
    "beats_best_share",
    "abs_corr_best",
    "score_std",
    "score_sessions",
    "deterministic_sample_key",
    "research_version",
    "execution_schema",
    "financial_reference",
    "capital_observed_at_freeze",
    "selection_salt",
    "source_search_sha256",
    "frozen_signature_sha256",
]

selection_path = (
    OUT / "prospective_validation_cohort_v118.csv"
)
selected[selection_columns].to_csv(
    selection_path,
    index=False,
)
shutil.copy2(
    selection_path,
    LOCAL_FREEZE,
)

pool_export = untouched[
    [
        "asset",
        "score_data_eligible",
        "activation",
        "dormant",
        "specialist_score",
        "validation_score",
        "beats_best_share",
        "abs_corr_best",
        "score_std",
        "score_sessions",
    ]
].copy()
pool_export.to_csv(
    OUT / "prospective_untouched_pool_scored.csv",
    index=False,
)

selection_sha = sha256_file(selection_path)

payload = {
    "research_version": VERSION,
    "execution_schema": SCHEMA,
    "status": "frozen_before_financial_replay",
    "financial_reference": EXPECTED_REFERENCE,
    "source": {
        "search_version": search_meta.get(
            "research_version"
        ),
        "search_schema": search_meta.get(
            "execution_schema"
        ),
        "ranked_rows": int(len(ranked)),
        "untouched_rows": int(len(untouched)),
        "source_ranked_sha256": sha256_file(
            RANKED_FILE
        ),
        "frozen_signature_sha256": sha256_file(
            FROZEN_SIGNATURE
        ),
    },
    "protocol": {
        "financial_outcomes_used": False,
        "capital_replays": 0,
        "known_smart20_excluded": sorted(
            KNOWN_SMART20
        ),
        "score_formula": (
            "A=1[beats_best_share>0]; "
            "S=1-mean(rank_active(beats),"
            "rank_active(abs_corr_best),"
            "rank_active(score_std))"
        ),
        "rank_scope": (
            "all eligible active candidates in the "
            "untouched prospective cohort"
        ),
        "sampling": (
            "8 deterministic SHA256-sampled candidates "
            "from each active S tercile plus 8 dormant"
        ),
        "per_stratum": PER_STRATUM,
        "selection_salt": SALT,
        "post_outcome_retuning_allowed": False,
    },
    "pool": {
        "untouched": int(len(untouched)),
        "score_data_eligible": int(
            untouched["score_data_eligible"].sum()
        ),
        "active": int(
            untouched["activation"].sum()
        ),
        "dormant": int(
            untouched["dormant"].sum()
        ),
        "insufficient": int(
            (~untouched["score_data_eligible"]).sum()
        ),
        "stratum_counts": stratum_counts,
    },
    "selection": {
        "count": int(len(selected)),
        "assets": selected[
            "asset"
        ].tolist(),
        "selection_sha256": selection_sha,
        "local_freeze_path": str(
            LOCAL_FREEZE.relative_to(ROOT)
        ),
    },
    "next_step": (
        "Do not change this cohort. The next runner "
        "may reveal financial outcomes versus U59."
    ),
    "runtime_seconds": float(
        time.perf_counter() - started
    ),
}

with (
    OUT / "prospective_validation_freeze.json"
).open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        payload,
        handle,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

package = criar_pacote_analise(
    OUT,
    comparison_file="prospective_validation_freeze.json",
    execution_schema=SCHEMA,
    archive_name=(
        "pacote_congelamento_validacao_prospectiva_v118.zip"
    ),
)

print("=" * 78, flush=True)
print(
    "TCC - CONGELAMENTO PROSPECTIVO v1.18",
    flush=True,
)
print(
    f"[pool] untouched={len(untouched)} "
    f"eligible={int(untouched['score_data_eligible'].sum())} "
    f"active={int(untouched['activation'].sum())} "
    f"dormant={int(untouched['dormant'].sum())} "
    f"insufficient="
    f"{int((~untouched['score_data_eligible']).sum())}",
    flush=True,
)
print("[frozen-cohort]", flush=True)
print(
    selected[
        [
            "prospective_validation_rank",
            "asset",
            "validation_stratum",
            "specialist_score",
            "beats_best_share",
            "abs_corr_best",
            "score_std",
        ]
    ].to_string(index=False),
    flush=True,
)
print(
    f"[freeze] sha256={selection_sha}",
    flush=True,
)
print(f"[package] pronto={package}", flush=True)
print(
    "[guard] NAO execute replay financeiro antes de "
    "preservar este pacote.",
    flush=True,
)
sinal_sonoro_conclusao()
