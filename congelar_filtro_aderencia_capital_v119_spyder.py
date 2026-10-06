"""CONGELAMENTO DO FILTRO DE ADERENCIA DE CAPITAL v1.19.

Nova hipotese de pesquisa, criada APOS a validacao prospectiva v1.18.

Objetivo:
usar o que a v1.18 realmente demonstrou melhor: evitar candidatos com baixa
aderencia ao processo de crescimento de capital. A v1.19 NAO altera nem
reinterpreta retroativamente a validacao v1.18.

Este runner:
1. reproduz a evidencia de desenvolvimento usando somente resultados ja
   conhecidos (Smart20 + 24 ativos da validacao prospectiva v1.18);
2. congela o novo risco de baixa aderencia:
       R = mean(rank(beats_best_share), rank(score_std))
   em que menor R e melhor;
3. exclui todos os 52 ativos cujo capital ja foi observado;
4. aplica R aos candidatos ainda intocados;
5. seleciona deterministicamente os 8 menores R;
6. NAO executa nenhum replay financeiro novo.

Depois desta execucao, preserve o pacote antes de abrir qualquer resultado
financeiro dos oito selecionados.
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
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

from pesquisas.directional_change_lightgbm import (
    criar_pacote_analise,
    sinal_sonoro_conclusao,
)


# %% 0 - Configuracao
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dados" / "assinatura_matematica"
SEARCH_OUT = ROOT / "output" / "busca_ativos"
PROSPECTIVE_OUT = ROOT / "output" / "validacao_prospectiva_financeira_v118"

RANKED_FILE = SEARCH_OUT / "intelligent_candidates_ranked.csv"
SEARCH_JSON = SEARCH_OUT / "asset_search.json"
SMART20_FILE = DATA / "contextual_smart20.csv"
PROSPECTIVE_FILE = PROSPECTIVE_OUT / "prospective_individual_outcomes.csv"
MODEL_FILE = DATA / "capital_adherence_filter_v1_19_frozen.json"

OUT = ROOT / "output" / "filtro_aderencia_capital_v119"
LOCAL_FREEZE = DATA / "capital_adherence_cohort_v119.csv"

VERSION = "1.19.0-dev.1"
SCHEMA = "capital-adherence-freeze-v1"
EXPECTED_SEARCH_VERSION = "1.17.0-dev.1"
EXPECTED_SEARCH_SCHEMA = "intelligent-asset-search-u59-v1"
EXPECTED_REFERENCE = "U59_WINNER"
EXPECTED_RANKED_ROWS = 446
EXPECTED_KNOWN_COUNT = 52
EXPECTED_REMAINING_ROWS = 394
SELECTION_SIZE = 8
HARM_THRESHOLD = -0.10

KNOWN_SMART20 = {
    "SGA", "THO", "XNTK", "CIVB", "WDAY",
    "EXR", "PAYX", "FMBH", "SBFG", "ALNY",
    "SXC", "ICCC", "XEL", "EBMT", "VLRS",
    "PDFS", "SITC", "FDX", "FNWB", "MUX",
}

KNOWN_PROSPECTIVE32 = {
    "HMN", "LOCO", "ACU", "HBCP", "LYV", "PNFP", "TCBI", "OVLY",
    "WDC", "PROV", "EDU", "RARE", "RDCM", "RELL", "NUE", "JOUT",
    "NSIT", "ENTG", "PLUG", "GOGO", "STRA", "CETX", "IOVA", "TANH",
    "EQIX", "FTQI", "ILF", "IPAC", "MTG", "QUAL", "VAW", "XSLV",
}

KNOWN52 = KNOWN_SMART20 | KNOWN_PROSPECTIVE32


# %% 1 - Helpers
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _risk_within_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["rank_beats"] = pd.to_numeric(
        result["beats_best_share"],
        errors="coerce",
    ).rank(method="average", pct=True)
    result["rank_score_std"] = pd.to_numeric(
        result["score_std"],
        errors="coerce",
    ).rank(method="average", pct=True)
    result["capital_adherence_risk"] = (
        result["rank_beats"] + result["rank_score_std"]
    ) / 2.0
    result["capital_adherence_score"] = (
        1.0 - result["capital_adherence_risk"]
    )
    return result


def _auc_harm(frame: pd.DataFrame) -> float:
    y = (
        pd.to_numeric(
            frame["capital_effect"],
            errors="coerce",
        ) <= HARM_THRESHOLD
    ).astype(int)
    return float(
        roc_auc_score(
            y,
            frame["capital_adherence_risk"],
        )
    )


# %% 2 - Guardas
started = time.perf_counter()

for path in (
    RANKED_FILE,
    SEARCH_JSON,
    SMART20_FILE,
    PROSPECTIVE_FILE,
    MODEL_FILE,
):
    if not path.exists():
        raise RuntimeError(
            "Arquivo necessario ausente: "
            f"{path}. Nao execute nova busca para substituir este artefato."
        )

if len(KNOWN52) != EXPECTED_KNOWN_COUNT:
    raise RuntimeError("A lista de 52 resultados conhecidos esta inconsistente.")

search_meta = json.loads(
    SEARCH_JSON.read_text(encoding="utf-8")
)
model_meta = json.loads(
    MODEL_FILE.read_text(encoding="utf-8")
)

if search_meta.get("research_version") != EXPECTED_SEARCH_VERSION:
    raise RuntimeError("Versao da busca original inesperada.")
if search_meta.get("execution_schema") != EXPECTED_SEARCH_SCHEMA:
    raise RuntimeError("Schema da busca original inesperado.")
if (
    (search_meta.get("stage2") or {}).get("score_reference")
    != EXPECTED_REFERENCE
):
    raise RuntimeError("A busca original nao usa U59_WINNER.")
if (
    model_meta.get("status")
    != "new_hypothesis_frozen_before_new_candidate_capital"
):
    raise RuntimeError("A hipotese v1.19 nao esta congelada corretamente.")


# %% 3 - Reproduzir desenvolvimento conhecido
smart = pd.read_csv(SMART20_FILE)
smart["asset"] = smart["asset"].astype(str).str.upper()
smart_dev = pd.DataFrame(
    {
        "asset": smart["asset"],
        "cohort": "smart20",
        "beats_best_share": pd.to_numeric(
            smart["beats_best_share"],
            errors="coerce",
        ),
        "score_std": pd.to_numeric(
            smart["score_std"],
            errors="coerce",
        ),
        "capital_effect": pd.to_numeric(
            smart["capital_effect"],
            errors="coerce",
        ),
    }
)

prospective = pd.read_csv(PROSPECTIVE_FILE)
prospective["asset"] = (
    prospective["asset"].astype(str).str.upper()
)
active_mask = (
    prospective["activation"]
    .astype(str)
    .str.lower()
    .isin({"true", "1", "yes"})
)
prospective_active = prospective[active_mask].copy()

pros_dev = pd.DataFrame(
    {
        "asset": prospective_active["asset"],
        "cohort": "prospective24",
        "beats_best_share": pd.to_numeric(
            prospective_active["beats_best_share"],
            errors="coerce",
        ),
        "score_std": pd.to_numeric(
            prospective_active["score_std"],
            errors="coerce",
        ),
        "capital_effect": pd.to_numeric(
            prospective_active["capital_pct_vs_u59"],
            errors="coerce",
        ),
    }
)

if len(smart_dev) != 20 or len(pros_dev) != 24:
    raise RuntimeError(
        "Amostra de desenvolvimento deveria ser 20 Smart20 + 24 prospectivos ativos."
    )

smart_dev = _risk_within_cohort(smart_dev)
pros_dev = _risk_within_cohort(pros_dev)
development = pd.concat(
    [smart_dev, pros_dev],
    ignore_index=True,
)
development["harm10"] = (
    development["capital_effect"] <= HARM_THRESHOLD
)

auc_smart = _auc_harm(smart_dev)
auc_pros = _auc_harm(pros_dev)
auc_pooled = _auc_harm(development)

rho = spearmanr(
    development["capital_adherence_risk"],
    development["capital_effect"],
)
harm = development[
    development["harm10"]
]["capital_adherence_risk"]
safe = development[
    ~development["harm10"]
]["capital_adherence_risk"]
mw = mannwhitneyu(
    harm,
    safe,
    alternative="greater",
)

expected = model_meta["development_evidence"]
checks = {
    "smart20_auc_harm10": (auc_smart, expected["smart20_auc_harm10"]),
    "prospective24_auc_harm10": (
        auc_pros,
        expected["prospective24_auc_harm10"],
    ),
    "pooled_auc_harm10": (
        auc_pooled,
        expected["pooled_auc_harm10"],
    ),
    "pooled_spearman_risk_vs_capital_effect": (
        float(rho.statistic),
        expected["pooled_spearman_risk_vs_capital_effect"],
    ),
}

for name, (observed, expected_value) in checks.items():
    if not math.isclose(
        float(observed),
        float(expected_value),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            f"Falha ao reproduzir {name}: "
            f"esperado={expected_value} observado={observed}"
        )

print("=" * 78, flush=True)
print("TCC - FILTRO DE ADERENCIA DE CAPITAL v1.19", flush=True)
print(
    f"[development] active=44 harm10={int(development['harm10'].sum())} "
    f"auc_smart={auc_smart:.4f} auc_pros={auc_pros:.4f} "
    f"auc_pooled={auc_pooled:.4f}",
    flush=True,
)
print(
    f"[development] spearman_R_vs_delta={float(rho.statistic):.4f} "
    f"p={float(rho.pvalue):.6g} "
    f"MW_one_sided_p={float(mw.pvalue):.6g}",
    flush=True,
)


# %% 4 - Aplicar ao pool ainda intocado
ranked = pd.read_csv(RANKED_FILE)
required = {
    "asset",
    "candidate_beats_u59_best_share",
    "candidate_score_std",
    "model_score_sessions",
}
missing = sorted(required.difference(ranked.columns))
if missing:
    raise RuntimeError(
        "Ranked original incompleto: " + ",".join(missing)
    )

ranked["asset"] = ranked["asset"].astype(str).str.strip().str.upper()
ranked = ranked.drop_duplicates("asset", keep="first").copy()

if len(ranked) != EXPECTED_RANKED_ROWS:
    raise RuntimeError(
        f"Ranked deveria ter {EXPECTED_RANKED_ROWS}; observado={len(ranked)}."
    )

known_found = set(ranked["asset"]).intersection(KNOWN52)
if known_found != KNOWN52:
    raise RuntimeError(
        "Nem todos os 52 resultados conhecidos estao no ranked original."
    )

untouched = ranked[
    ~ranked["asset"].isin(KNOWN52)
].copy()

if len(untouched) != EXPECTED_REMAINING_ROWS:
    raise RuntimeError(
        f"Pool novo deveria ter {EXPECTED_REMAINING_ROWS}; "
        f"observado={len(untouched)}."
    )

untouched["beats_best_share"] = pd.to_numeric(
    untouched["candidate_beats_u59_best_share"],
    errors="coerce",
)
untouched["score_std"] = pd.to_numeric(
    untouched["candidate_score_std"],
    errors="coerce",
)
untouched["score_sessions"] = pd.to_numeric(
    untouched["model_score_sessions"],
    errors="coerce",
)

minimum_sessions = int(
    ((search_meta.get("signature") or {}).get("minimum_score_sessions"))
    or 0
)
if minimum_sessions <= 0:
    raise RuntimeError("minimum_score_sessions ausente no asset_search.json.")

untouched["score_data_eligible"] = (
    untouched["beats_best_share"].notna()
    & untouched["score_std"].notna()
    & (untouched["score_sessions"] >= minimum_sessions)
)
untouched["activation"] = (
    untouched["score_data_eligible"]
    & (untouched["beats_best_share"] > 0.0)
)

active_pool = untouched[
    untouched["activation"]
].copy()

if len(active_pool) < SELECTION_SIZE:
    raise RuntimeError("Poucos ativos elegiveis no pool intocado.")

active_pool = _risk_within_cohort(
    active_pool.rename(
        columns={
            "candidate_beats_u59_best_share": "_source_beats",
            "candidate_score_std": "_source_std",
        }
    )
)
# _risk_within_cohort usa as colunas normalizadas beats_best_share e score_std.

active_pool["risk_percentile"] = (
    active_pool["capital_adherence_risk"]
    .rank(method="average", pct=True)
)

selected = active_pool.sort_values(
    [
        "capital_adherence_risk",
        "beats_best_share",
        "score_std",
        "asset",
    ],
    ascending=[True, True, True, True],
).head(SELECTION_SIZE).copy()

selected["selection_rank"] = np.arange(
    1,
    len(selected) + 1,
)
selected["research_version"] = VERSION
selected["execution_schema"] = SCHEMA
selected["financial_reference"] = EXPECTED_REFERENCE
selected["capital_observed_at_freeze"] = False
selected["model_sha256"] = sha256_file(MODEL_FILE)
selected["source_ranked_sha256"] = sha256_file(RANKED_FILE)


# %% 5 - Exportacao e congelamento
OUT.mkdir(parents=True, exist_ok=True)
LOCAL_FREEZE.parent.mkdir(parents=True, exist_ok=True)

pool_columns = [
    "asset",
    "score_data_eligible",
    "activation",
    "beats_best_share",
    "score_std",
    "score_sessions",
]
for optional in (
    "stage1_rank",
    "raw_winner_probability",
    "candidate_score_mean",
    "candidate_positive_score_share",
    "model_score_corr_u59_best",
):
    if optional in untouched.columns:
        pool_columns.append(optional)

untouched[pool_columns].to_csv(
    OUT / "remaining_untouched_pool_v119.csv",
    index=False,
)

scored_columns = [
    "asset",
    "capital_adherence_risk",
    "capital_adherence_score",
    "risk_percentile",
    "rank_beats",
    "rank_score_std",
    "beats_best_share",
    "score_std",
    "score_sessions",
]
active_pool[scored_columns].sort_values(
    "capital_adherence_risk"
).to_csv(
    OUT / "active_pool_scored_v119.csv",
    index=False,
)

selection_columns = [
    "selection_rank",
    "asset",
    "capital_adherence_risk",
    "capital_adherence_score",
    "risk_percentile",
    "rank_beats",
    "rank_score_std",
    "beats_best_share",
    "score_std",
    "score_sessions",
    "research_version",
    "execution_schema",
    "financial_reference",
    "capital_observed_at_freeze",
    "model_sha256",
    "source_ranked_sha256",
]

selection_path = OUT / "capital_adherence_cohort_v119.csv"
selected[selection_columns].to_csv(
    selection_path,
    index=False,
)
shutil.copy2(selection_path, LOCAL_FREEZE)

development.to_csv(
    OUT / "capital_adherence_development_44.csv",
    index=False,
)

selection_sha = sha256_file(selection_path)

payload = {
    "research_version": VERSION,
    "execution_schema": SCHEMA,
    "status": "frozen_before_new_candidate_capital",
    "relationship_to_v118": (
        "NEW hypothesis. v1.18 remains unchanged and already completed."
    ),
    "financial_reference": EXPECTED_REFERENCE,
    "development": {
        "n_active": int(len(development)),
        "harm10_count": int(development["harm10"].sum()),
        "harm_threshold": HARM_THRESHOLD,
        "auc_smart20": auc_smart,
        "auc_prospective24": auc_pros,
        "auc_pooled": auc_pooled,
        "spearman_risk_vs_capital_effect": float(rho.statistic),
        "spearman_pvalue": float(rho.pvalue),
        "mann_whitney_one_sided_pvalue": float(mw.pvalue),
    },
    "formula": {
        "activation": "A = 1[beats_best_share > 0]",
        "risk": (
            "R = mean(rank_active(beats_best_share), "
            "rank_active(score_std))"
        ),
        "direction": "lower_R_is_better",
        "rank_scope": (
            "remaining financially untouched active candidates only"
        ),
    },
    "pool": {
        "ranked_original": int(len(ranked)),
        "known_financial_excluded": len(KNOWN52),
        "remaining_untouched": int(len(untouched)),
        "score_data_eligible": int(
            untouched["score_data_eligible"].sum()
        ),
        "active": int(untouched["activation"].sum()),
        "dormant_or_ineligible": int(
            len(untouched) - untouched["activation"].sum()
        ),
    },
    "selection": {
        "count": int(len(selected)),
        "assets": selected["asset"].tolist(),
        "selection_sha256": selection_sha,
        "local_freeze_path": str(
            LOCAL_FREEZE.relative_to(ROOT)
        ),
        "rule": "eight lowest R; no financial outcome used",
    },
    "next_test": {
        "primary_economic_endpoint": (
            "ending_capital(U59 + frozen_adherence8) "
            "> ending_capital(U59)"
        ),
        "individual_safety_diagnostic": (
            "count capital_pct_vs_u59 <= -0.10"
        ),
        "stretch_benchmark": 58_557_157.67496595,
        "stretch_is_confirmatory": False,
        "post_outcome_retuning_allowed": False,
        "second_selection_attempt_allowed": False,
    },
    "runtime_seconds": float(
        time.perf_counter() - started
    ),
}

with (
    OUT / "capital_adherence_freeze_v119.json"
).open("w", encoding="utf-8") as handle:
    json.dump(
        payload,
        handle,
        ensure_ascii=False,
        indent=2,
        default=str,
    )

package = criar_pacote_analise(
    OUT,
    comparison_file="capital_adherence_freeze_v119.json",
    execution_schema=SCHEMA,
    archive_name="pacote_congelamento_aderencia_capital_v119.zip",
)

print("[remaining-pool]", flush=True)
print(
    f"untouched={len(untouched)} "
    f"eligible={int(untouched['score_data_eligible'].sum())} "
    f"active={int(untouched['activation'].sum())}",
    flush=True,
)
print("[frozen-adherence8]", flush=True)
print(
    selected[
        [
            "selection_rank",
            "asset",
            "capital_adherence_risk",
            "beats_best_share",
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
    "[guard] preserve este pacote antes de qualquer replay financeiro v1.19.",
    flush=True,
)
sinal_sonoro_conclusao()
