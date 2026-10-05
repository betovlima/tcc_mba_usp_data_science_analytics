"""BUSCA DE ATIVOS - runner independente para Spyder.

Este arquivo NAO executa backtest financeiro dos novos candidatos.

Fluxo:
1. usa somente dados ja conhecidos para uma triagem inicial;
2. varre o catalogo Alpaca sem sorteio e exige historico integral;
3. leva ate 500 candidatos para avaliacao por LightGBM;
4. mede o comportamento de score contra o U59 vencedor (~US$ 30 milhoes);
5. congela apenas os candidatos que passam a assinatura, sem preencher vagas.

Execucao no Spyder:
- executar o arquivo inteiro com F5; ou
- executar as celulas "# %%" em ordem, de cima para baixo.
"""
from __future__ import annotations

from pathlib import Path
import json
import math
import re
import time

import numpy as np
import pandas as pd
import requests
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from engine.configuracao import (
    ANALYSIS_END_DATE,
    BAR_SNAPSHOT_AS_OF_END,
    CONFIG,
    START_DATE,
)
from engine.modelo_lightgbm import (
    _ajustar_modelos_lightgbm,
    _construir_contexto_execucao,
)
from engine.rotacao import (
    _precalcular_utilidades_modelo,
    preparar_painel_rotacao,
)
from pesquisas.directional_change_lightgbm import (
    EXPECTED_EXECUTION_SCHEMA,
    RESEARCH_VERSION,
    criar_pacote_analise,
    sinal_sonoro_conclusao,
)
from reproducao.dados import (
    SnapshotPaths,
    _normalize_alpaca_frame,
    build_snapshot_manifest,
    download_corporate_actions,
    download_raw_bars,
    load_alpaca_credentials,
    validate_snapshot,
)
from reproducao.experimento import build_variant_configs
from reproducao.preparacao import prepare_model_frames


# %% 0 - Configuracao da busca\nROOT = Path(__file__).resolve().parent
BASE = SnapshotPaths.research(ROOT)
B2 = SnapshotPaths.from_root(ROOT / "dados" / "pesquisa_expansao_76_b2")
B3 = SnapshotPaths.from_root(ROOT / "dados" / "pesquisa_expansao_76_b3")
SMART = SnapshotPaths.from_root(ROOT / "dados" / "pesquisa_smart_candidates")
OUT = ROOT / "output" / "busca_ativos"

SCRIPT_RESEARCH_VERSION = "1.17.0-dev.1"
EXECUTION_SCHEMA = "intelligent-asset-search-u59-v1"
CHECKPOINT_SHA = "4b6b71414ce6f7047dc465679e57629ba8a4c453"

B2_ASSETS = (
    "VIOV","MBSD","MVIS","EWD","OPHC","CASY","COLB","EES","GNK","VUZI",
    "FOXF","AMS","IQLT","ISCF","FUTY","KB","PRN","CE","XTNT","UEC",
)
B3_ASSETS = (
    "MG","VSTM","HEWJ","GBAB","CRESY","BGT","DBJP","UBND","PPLT","RXL",
    "JPIN","REXR","QVAL","REM","CEVA","TRC","SCHA","CALM","DRN","EVH",
)
B1_ASSETS = (
    "FAF","IJR","GAB","ELS","AEIS","VWOB","BDJ","DGX","ESP","BWZ",
    "PSF","DBA","HEEM","NPKI","MHK","BLKB","ARCO","AGM","NWFL","SKOR",
)
B2_POS = frozenset({"COLB","AMS","FOXF"})
B3_POS = frozenset({"MG","REXR","CALM"})
KNOWN_POS = tuple(sorted(B2_POS | B3_POS))
EXCLUDED = (
    set(CONFIG.assets)
    | set(B1_ASSETS)
    | set(B2_ASSETS)
    | set(B3_ASSETS)
    | {"ONTO","FLG","LBTYA","DCOY","BLOX","VISN","COR","ECON"}
)

ALLOWED_EXCHANGES = {"NYSE","NASDAQ","AMEX","ARCA","BATS"}
SYMBOL_RE = re.compile(r"^[A-Z]{1,5}$")
SCOUT_START = START_DATE
SCOUT_MIN_ROWS = 2600
SCOUT_MAX_GAP = 10.0
SCOUT_CHUNK = 50
MODEL_POOL_SIZE = 500
SELECTED_COUNT = 20

RAW_FEATURES = (
    "cagr",
    "annual_volatility",
    "maximum_drawdown",
    "median_dollar_volume_log10",
    "positive_day_share",
    "momentum_252_median",
    "volatility_20_median",
    "trend_efficiency_20_median",
    "corr_spy",
    "beta_spy",
)

SIGNATURE_NAME = "selective-specialist-u59-v0.1"
BEATS_MAX = 0.05
SCORE_STD_MAX = 0.15
SCORE_MEAN_MAX = 0.16
ABS_CORR_BEST_MAX = 0.10
MIN_SESSION_SHARE = 0.85
BEATS_TARGET = float(np.median([0.0045248869,0.0193923723,0.0129282482]))

STAGE1_CSV = SMART.root / "stage1_ranked.csv"
STAGE1_JSON = SMART.root / "stage1_selection.json"
CATALOG_JSON = SMART.root / "alpaca_asset_catalog.json"


# %% 1 - Guards, snapshots e identificacao da execucao\nif EXECUTION_SCHEMA != EXPECTED_EXECUTION_SCHEMA:
    raise RuntimeError(
        f"Schema incompatível: script={EXECUTION_SCHEMA} "
        f"modulo={EXPECTED_EXECUTION_SCHEMA}."
    )
if SCRIPT_RESEARCH_VERSION != RESEARCH_VERSION:
    raise RuntimeError(
        f"Versao incompatível: script={SCRIPT_RESEARCH_VERSION} "
        f"modulo={RESEARCH_VERSION}."
    )

print("=" * 78, flush=True)
print("TCC - Intelligent Candidate Screen", flush=True)
print(f"version={RESEARCH_VERSION}", flush=True)
print("random_sampling=False", flush=True)
print("candidate_strategy_replays=0", flush=True)
print("score_reference=U59_WINNER", flush=True)
print("known_positive=" + ",".join(KNOWN_POS), flush=True)
print("=" * 78, flush=True)

t0 = time.perf_counter()
base_manifest = validate_snapshot(BASE)
validate_snapshot(B2)
validate_snapshot(B3)


# %% 2 - Funcoes utilitarias de dados e propriedades\ndef read_raw(paths: SnapshotPaths, symbol: str) -> pd.DataFrame:
    path = paths.raw_bars / f"{symbol}.csv"
    frame = pd.read_csv(path)
    frame.columns = [str(c).lower() for c in frame.columns]
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    for col in ("open","high","low","close","volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["timestamp","close","volume"]).copy()


def scout_window(frame: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp(SCOUT_START, tz="UTC")
    end = pd.Timestamp(BAR_SNAPSHOT_AS_OF_END, tz="UTC") + pd.Timedelta(days=1)
    return frame.loc[
        (frame["timestamp"] >= start) & (frame["timestamp"] < end)
    ].copy()


def max_gap_days(frame: pd.DataFrame) -> float:
    d = pd.DatetimeIndex(frame["timestamp"]).to_series().diff()
    if d.dropna().empty:
        return float("inf")
    return float(d.dt.total_seconds().div(86400.0).dropna().max())


def corr(left: pd.Series, right: pd.Series, minimum: int = 20):
    pair = pd.concat(
        [pd.Series(left, dtype=float), pd.Series(right, dtype=float)],
        axis=1,
    ).replace([np.inf,-np.inf], np.nan).dropna()
    if len(pair) < minimum or pair.iloc[:,0].nunique() < 2 or pair.iloc[:,1].nunique() < 2:
        return None
    return float(pair.iloc[:,0].corr(pair.iloc[:,1]))


def raw_features(frame: pd.DataFrame, spy_returns: pd.Series) -> dict:
    w = scout_window(frame)
    close = pd.to_numeric(w["close"], errors="coerce")
    volume = pd.to_numeric(w["volume"], errors="coerce")
    valid = close.notna() & volume.notna() & (close > 0) & (volume >= 0)
    w = w.loc[valid].copy()
    close = close.loc[valid]
    volume = volume.loc[valid]
    if len(w) < 2:
        return {"rows": int(len(w)), "max_gap_days": float("inf")}

    r = close.pct_change(fill_method=None).replace([np.inf,-np.inf], np.nan)
    years = max((w["timestamp"].iloc[-1] - w["timestamp"].iloc[0]).days / 365.25, 1/365.25)
    ratio = float(close.iloc[-1] / close.iloc[0])
    cagr = float(ratio ** (1/years) - 1) if ratio > 0 else None
    dd = close / close.cummax() - 1
    m252 = close / close.shift(252) - 1
    vol20 = r.rolling(20).std(ddof=0) * math.sqrt(252)
    eff = (close - close.shift(20)).abs() / close.diff().abs().rolling(20).sum()
    candidate_returns = pd.Series(
        r.to_numpy(), index=pd.DatetimeIndex(w["timestamp"]), dtype=float
    )
    aligned = pd.concat(
        [candidate_returns.rename("c"), spy_returns.rename("s")], axis=1
    ).dropna()
    if len(aligned) >= 20 and aligned["s"].var() > 0:
        corr_spy = float(aligned["c"].corr(aligned["s"]))
        beta_spy = float(aligned["c"].cov(aligned["s"]) / aligned["s"].var())
    else:
        corr_spy = None
        beta_spy = None

    dollar = float((close * volume).median())
    return {
        "rows": int(len(w)),
        "first_timestamp": str(w["timestamp"].iloc[0]),
        "last_timestamp": str(w["timestamp"].iloc[-1]),
        "max_gap_days": max_gap_days(w),
        "cagr": cagr,
        "annual_volatility": float(r.std(ddof=0) * math.sqrt(252)),
        "maximum_drawdown": float(dd.min()),
        "median_dollar_volume_log10": (
            float(math.log10(dollar)) if dollar > 0 else None
        ),
        "positive_day_share": float((r > 0).mean()),
        "momentum_252_median": (
            float(m252.dropna().median()) if not m252.dropna().empty else None
        ),
        "volatility_20_median": (
            float(vol20.dropna().median()) if not vol20.dropna().empty else None
        ),
        "trend_efficiency_20_median": (
            float(eff.replace([np.inf,-np.inf],np.nan).dropna().median())
            if not eff.replace([np.inf,-np.inf],np.nan).dropna().empty
            else None
        ),
        "corr_spy": corr_spy,
        "beta_spy": beta_spy,
    }


# %% 3 - Base de aprendizado ja conhecida\nspy = scout_window(read_raw(BASE, "SPY"))
spy_returns = pd.Series(
    pd.to_numeric(spy["close"], errors="coerce").pct_change(fill_method=None).to_numpy(),
    index=pd.DatetimeIndex(spy["timestamp"]),
    dtype=float,
)

label_rows = []
for cohort, paths, assets, positives in (
    ("batch2", B2, B2_ASSETS, B2_POS),
    ("batch3", B3, B3_ASSETS, B3_POS),
):
    for symbol in assets:
        label_rows.append({
            "asset": symbol,
            "cohort": cohort,
            "positive": int(symbol in positives),
            **raw_features(read_raw(paths, symbol), spy_returns),
        })
labels = pd.DataFrame(label_rows)
selector = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("model", LogisticRegression(
        class_weight="balanced",
        C=0.5,
        max_iter=3000,
        solver="lbfgs",
        random_state=20261005,
    )),
])
selector.fit(labels.loc[:, RAW_FEATURES], labels["positive"])


def load_catalog(credentials) -> list[dict]:
    errors = []
    for endpoint in (
        "https://paper-api.alpaca.markets/v2/assets",
        "https://api.alpaca.markets/v2/assets",
    ):
        try:
            response = requests.get(
                endpoint,
                headers=credentials.headers,
                params={"status":"active","asset_class":"us_equity"},
                timeout=60,
            )
        except Exception as exc:
            errors.append(str(exc))
            continue
        if response.status_code != 200:
            errors.append(f"{response.status_code}:{response.text[:120]}")
            continue
        payload = response.json()
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
    raise RuntimeError("Falha no catalogo Alpaca: " + " | ".join(errors))


def eligible_catalog(catalog: list[dict]) -> list[dict]:
    rows = []
    seen = set()
    for item in catalog:
        symbol = str(item.get("symbol") or "").strip().upper()
        exchange = str(item.get("exchange") or "").strip().upper()
        status = str(item.get("status") or "").strip().lower()
        if (
            not symbol
            or symbol in seen
            or symbol in EXCLUDED
            or SYMBOL_RE.fullmatch(symbol) is None
            or exchange not in ALLOWED_EXCHANGES
            or (status and status != "active")
            or not bool(item.get("tradable", False))
            or not bool(item.get("marginable", False))
        ):
            continue
        seen.add(symbol)
        rows.append({
            "symbol": symbol,
            "name": str(item.get("name") or ""),
            "exchange": exchange,
            "shortable": bool(item.get("shortable", False)),
            "easy_to_borrow": bool(item.get("easy_to_borrow", False)),
            "fractionable": bool(item.get("fractionable", False)),
        })
    return sorted(rows, key=lambda x: x["symbol"])


def get_scout_bars(client, symbols: list[str]) -> dict[str, pd.DataFrame]:
    start = pd.Timestamp(SCOUT_START, tz="UTC").to_pydatetime()
    end = (
        pd.Timestamp(BAR_SNAPSHOT_AS_OF_END, tz="UTC") + pd.Timedelta(days=1)
    ).to_pydatetime()
    for attempt in range(4):
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                adjustment=Adjustment.RAW,
                feed=DataFeed.SIP,
                limit=10_000,
            )
            raw = client.get_stock_bars(req).df
            out = {}
            for symbol in symbols:
                try:
                    f = _normalize_alpaca_frame(raw, symbol)
                except Exception:
                    f = pd.DataFrame()
                if not f.empty:
                    out[symbol] = f
            return out
        except Exception as exc:
            if attempt == 3:
                print(f"[scout] chunk failed size={len(symbols)} error={exc}", flush=True)
                break
            wait = 2 ** attempt
            print(f"[scout] retry wait={wait}s error={exc}", flush=True)
            time.sleep(wait)
    if len(symbols) == 1:
        return {}
    out = {}
    for symbol in symbols:
        out.update(get_scout_bars(client, [symbol]))
    return out


def smart_snapshot_reusable() -> bool:
    if not (
        SMART.manifest.exists()
        and STAGE1_CSV.exists()
        and STAGE1_JSON.exists()
        and CATALOG_JSON.exists()
    ):
        return False
    try:
        manifest = validate_snapshot(SMART)
        meta = json.loads(STAGE1_JSON.read_text(encoding="utf-8"))
    except Exception:
        return False
    pool = tuple(meta.get("selected_model_pool") or ())
    return (
        meta.get("research_version") == RESEARCH_VERSION
        and meta.get("execution_schema") == EXECUTION_SCHEMA
        and int(meta.get("model_pool_size") or 0) == len(pool)
        and SELECTED_COUNT <= len(pool) <= MODEL_POOL_SIZE
        and tuple(manifest.get("assets") or ()) == pool
    )


# %% 4 - Busca no catalogo Alpaca e congelamento do pool de ate 500\ncredentials = load_alpaca_credentials(ROOT)
if smart_snapshot_reusable():
    stage1 = pd.read_csv(STAGE1_CSV)
    stage1_meta = json.loads(STAGE1_JSON.read_text(encoding="utf-8"))
    pool = tuple(stage1_meta["selected_model_pool"])
    smart_manifest = validate_snapshot(SMART)
    print(f"[stage1] reusing frozen pool={len(pool)}", flush=True)
else:
    catalog = eligible_catalog(load_catalog(credentials))
    SMART.root.mkdir(parents=True, exist_ok=True)
    CATALOG_JSON.write_text(
        json.dumps(catalog, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    client = StockHistoricalDataClient(
        api_key=credentials.api_key,
        secret_key=credentials.secret_key,
    )
    rows = []
    for offset in range(0, len(catalog), SCOUT_CHUNK):
        chunk = catalog[offset:offset + SCOUT_CHUNK]
        symbols = [x["symbol"] for x in chunk]
        frames = get_scout_bars(client, symbols)
        metadata = {x["symbol"]: x for x in chunk}
        for symbol in symbols:
            frame = frames.get(symbol)
            if frame is None or frame.empty:
                continue
            feats = raw_features(frame, spy_returns)
            if (
                int(feats.get("rows") or 0) < SCOUT_MIN_ROWS
                or float(feats.get("max_gap_days") or float("inf")) > SCOUT_MAX_GAP
            ):
                continue
            x = pd.DataFrame([{k: feats.get(k) for k in RAW_FEATURES}])
            probability = float(selector.predict_proba(x)[0,1])
            rows.append({
                **metadata[symbol],
                **feats,
                "raw_winner_probability": probability,
            })
        print(
            f"[stage1] scanned={min(offset+len(chunk),len(catalog))}/{len(catalog)} "
            f"history_ok={len(rows)}",
            flush=True,
        )

    stage1 = pd.DataFrame(rows).sort_values(
        ["raw_winner_probability","symbol"],
        ascending=[False,True],
    ).reset_index(drop=True)
    if len(stage1) < SELECTED_COUNT:
        raise RuntimeError(
            f"Somente {len(stage1)} candidatos passaram o stage1."
        )
    stage1["stage1_rank"] = np.arange(1, len(stage1)+1)
    resolved_pool_size = min(MODEL_POOL_SIZE, len(stage1))
    pool = tuple(stage1.head(resolved_pool_size)["symbol"].astype(str))

    SMART.clear_generated()
    raw_files = download_raw_bars(
        credentials,
        SMART,
        assets=pool,
        replace=True,
        bar_snapshot_as_of_end=BAR_SNAPSHOT_AS_OF_END,
        analysis_end_date=ANALYSIS_END_DATE,
    )
    action_files = download_corporate_actions(
        credentials,
        SMART,
        assets=pool,
        replace=True,
        query_end=ANALYSIS_END_DATE,
    )
    smart_manifest = build_snapshot_manifest(
        SMART,
        raw_files,
        action_files,
        credentials=credentials,
        bar_snapshot_as_of_end=BAR_SNAPSHOT_AS_OF_END,
        analysis_end_date=ANALYSIS_END_DATE,
        assets=pool,
        snapshot_name="tcc-intelligent-candidate-pool-u59-v1",
        parent_snapshot_sha256=str(base_manifest.get("snapshot_sha256") or ""),
    )
    stage1.to_csv(STAGE1_CSV, index=False)
    STAGE1_JSON.write_text(
        json.dumps({
            "research_version": RESEARCH_VERSION,
            "execution_schema": EXECUTION_SCHEMA,
            "random_sampling": False,
            "selected_model_pool": list(pool),
            "model_pool_size": len(pool),
            "model_pool_target": MODEL_POOL_SIZE,
            "catalog_eligible_assets": len(catalog),
            "stage1_history_eligible_assets": len(stage1),
            "selection_uses_new_candidate_capital": False,
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# %% 5 - Qualidade integral e exclusoes estruturais\nquality = []
full_ok = []
for symbol in pool:
    f = read_raw(SMART, symbol)
    first = pd.Timestamp(f["timestamp"].min())
    last = pd.Timestamp(f["timestamp"].max())
    ok = (
        len(f) >= 2600
        and max_gap_days(f) <= 10.0
        and first <= pd.Timestamp(START_DATE, tz="UTC") + pd.Timedelta(days=45)
        and last >= pd.Timestamp(BAR_SNAPSHOT_AS_OF_END, tz="UTC") - pd.Timedelta(days=10)
    )
    quality.append({
        "asset": symbol,
        "rows": len(f),
        "first": str(first),
        "last": str(last),
        "max_gap_days": max_gap_days(f),
        "full_history_eligible": bool(ok),
    })
    if ok:
        full_ok.append(symbol)

frames_candidates, structural_exclusions, candidate_diagnostics, candidate_audit = (
    prepare_model_frames(
        SMART,
        assets=tuple(full_ok),
        comparar_snapshot_referencia=False,
    )
)
candidates = sorted(frames_candidates)
if len(candidates) < SELECTED_COUNT:
    raise RuntimeError(
        f"Apenas {len(candidates)} candidatos sobraram apos filtros."
    )

# %% 6 - Contexto U59 vencedor e treinamento LightGBM sem backtest\nframes_u56, u56_exclusions, u56_diagnostics, u56_audit = prepare_model_frames(
    BASE,
    assets=CONFIG.assets,
    comparar_snapshot_referencia=False,
    allow_structural_assets=frozenset({"CLMT","DOC"}),
)
if len(frames_u56) != 56:
    raise RuntimeError(f"U56 deveria ter 56 ativos; obtidos {len(frames_u56)}.")

frames_b2_pos, b2_pos_exclusions, b2_pos_diagnostics, b2_pos_audit = (
    prepare_model_frames(
        B2,
        assets=tuple(sorted(B2_POS)),
        comparar_snapshot_referencia=False,
    )
)
if b2_pos_exclusions:
    raise RuntimeError(
        "COLB, AMS ou FOXF foi excluido estruturalmente; "
        "nao e permitido montar silenciosamente um U59 incompleto."
    )
frames_u59 = {**frames_u56, **frames_b2_pos}
if len(frames_u59) != 59:
    raise RuntimeError(
        f"U59 vencedor deveria ter 59 ativos; obtidos {len(frames_u59)}."
    )

frames_raw = {**frames_u59, **frames_candidates}
config_u56, _ = build_variant_configs(frames_u56, CONFIG)
_, reference_calendar, reference_source = preparar_painel_rotacao(
    frames_u56, config_u56
)
config_all, _ = build_variant_configs(frames_raw, CONFIG)
(
    frames_all,
    common_dates,
    calendar_source,
    symbols_all,
    folds,
    all_decision_dates,
    decision_to_fold,
    decision_metadata,
) = _construir_contexto_execucao(
    frames_raw,
    config_all,
    calendar_override=reference_calendar,
    calendar_source_label=f"U56_FIXED:{reference_source}",
)

position = {symbol:i+1 for i,symbol in enumerate(symbols_all)}
u59_symbols = sorted(frames_u59)
u59_idx = [position[s] for s in u59_symbols]
artifacts = {}

for i, fold in enumerate(folds, start=1):
    fold_id = int(fold["fold_id"])
    fit_dates = common_dates[:int(fold["final_fit_end_index"])]
    decision_dates = pd.DatetimeIndex(fold["decision_dates"])
    print(
        f"[stage2-train] fold={fold_id} {i}/{len(folds)} "
        f"models={len(symbols_all)}",
        flush=True,
    )
    models = _ajustar_modelos_lightgbm(
        frames_all,
        symbols_all,
        fit_dates,
        config_all,
        phase=f"smart_screen_fold_{fold_id}_final",
        technical_log_callback=lambda m: print(f"[technical] {m}", flush=True),
    )
    cache, _ = _precalcular_utilidades_modelo(
        models, frames_all, symbols_all, decision_dates, config_all
    )
    artifacts[fold_id] = {"dates":decision_dates, "cache":cache}

expected_sessions = sum(max(0, len(x["dates"])-1) for x in artifacts.values())
min_sessions = math.ceil(expected_sessions * MIN_SESSION_SHARE)


# %% 7 - Perfil de score de cada candidato contra U59\ndef score_profile(symbol: str) -> dict:
    idx = position[symbol]
    rows = []
    for artifact in artifacts.values():
        for ts in artifact["dates"][:-1]:
            u = artifact["cache"].get(pd.Timestamp(ts))
            if u is None:
                continue
            u = np.asarray(u, dtype=float)
            if idx >= len(u):
                continue
            score = float(u[idx])
            ref = u[u59_idx]
            finite = ref[np.isfinite(ref)]
            if not np.isfinite(score) or len(finite) == 0:
                continue
            best = float(np.max(finite))
            rows.append((score, float(np.mean(finite)), best, score > best))
    if not rows:
        return {
            "model_score_corr_u59_mean":None,
            "model_score_corr_u59_best":None,
            "candidate_beats_u59_best_share":None,
            "candidate_score_mean":None,
            "candidate_score_std":None,
            "candidate_positive_score_share":None,
            "model_score_sessions":0,
        }
    f = pd.DataFrame(rows, columns=["score","mean","best","beats"])
    return {
        "model_score_corr_u59_mean":corr(f["score"],f["mean"],5),
        "model_score_corr_u59_best":corr(f["score"],f["best"],5),
        "candidate_beats_u59_best_share":float(f["beats"].mean()),
        "candidate_score_mean":float(f["score"].mean()),
        "candidate_score_std":float(f["score"].std(ddof=0)),
        "candidate_positive_score_share":float((f["score"]>0).mean()),
        "model_score_sessions":int(len(f)),
    }


def classify(p: dict) -> tuple[str,bool]:
    beats = p["candidate_beats_u59_best_share"]
    std = p["candidate_score_std"]
    mean = p["candidate_score_mean"]
    corr_best = p["model_score_corr_u59_best"]
    sessions = p["model_score_sessions"]
    if (
        beats is None or std is None or mean is None or corr_best is None
        or sessions < min_sessions
    ):
        return "insufficient_score_data", False
    if beats <= 0:
        return "dormant", False
    if beats > BEATS_MAX or std > SCORE_STD_MAX:
        return "invasive_or_unstable", False
    if mean <= SCORE_MEAN_MAX and abs(corr_best) <= ABS_CORR_BEST_MAX:
        return "selective_specialist", True
    return "uncertain", False


# %% 8 - Ranking e congelamento dos candidatos aprovados\nstage1_map = stage1.set_index("symbol")
rank_rows = []
for i, symbol in enumerate(candidates, start=1):
    print(f"[stage2-score] {i}/{len(candidates)} {symbol}", flush=True)
    p = score_profile(symbol)
    cls, predicted = classify(p)
    meta = stage1_map.loc[symbol].to_dict() if symbol in stage1_map.index else {}
    cb = p["model_score_corr_u59_best"]
    beats = p["candidate_beats_u59_best_share"]
    rank_rows.append({
        "asset":symbol,
        "name":meta.get("name"),
        "exchange":meta.get("exchange"),
        "stage1_rank":meta.get("stage1_rank"),
        "raw_winner_probability":meta.get("raw_winner_probability"),
        **p,
        "signature_name":SIGNATURE_NAME,
        "signature_class":cls,
        "signature_predicted_positive":bool(predicted),
        "abs_score_corr_u59_best":abs(float(cb)) if cb is not None else None,
        "beats_target_distance":abs(float(beats)-BEATS_TARGET) if beats is not None else None,
    })

ranked = pd.DataFrame(rank_rows)
priority = {
    "selective_specialist":0,
    "uncertain":1,
    "invasive_or_unstable":2,
    "dormant":3,
    "insufficient_score_data":4,
}
ranked["_priority"] = ranked["signature_class"].map(priority).fillna(9)
ranked["_corr"] = pd.to_numeric(
    ranked["abs_score_corr_u59_best"], errors="coerce"
).fillna(999.0)
ranked["_beats"] = pd.to_numeric(
    ranked["beats_target_distance"], errors="coerce"
).fillna(999.0)
ranked["_prob"] = pd.to_numeric(
    ranked["raw_winner_probability"], errors="coerce"
).fillna(-1.0)
ranked = ranked.sort_values(
    ["_priority","_corr","_beats","_prob","asset"],
    ascending=[True,True,True,False,True],
).reset_index(drop=True)
ranked["selection_rank"] = np.arange(1, len(ranked)+1)
ranked["selection_tier"] = np.where(
    ranked["signature_predicted_positive"].astype(bool),
    "strong_signature_match",
    "best_remaining_without_capital_replay",
)
ranked = ranked.drop(columns=["_priority","_corr","_beats","_prob"])
selected = ranked.loc[
    ranked["signature_predicted_positive"].astype(bool)
].head(SELECTED_COUNT).copy()
selected_symbols = tuple(selected["asset"].astype(str))
strong = int(len(selected))

print(
    f"[selection] selected={len(selected_symbols)}/{SELECTED_COUNT} "
    f"strong={strong} capital_replays=0 forced_fill=0",
    flush=True,
)
print(
    selected[
        [
            "selection_rank","asset","selection_tier","raw_winner_probability",
            "signature_class","candidate_beats_u59_best_share",
            "candidate_score_std","candidate_score_mean",
            "model_score_corr_u59_best",
        ]
    ].to_string(index=False),
    flush=True,
)

# %% 9 - Exportacao da busca; ainda sem resultado financeiro\nOUT.mkdir(parents=True, exist_ok=True)
for old in OUT.rglob("*"):
    if old.is_file():
        old.unlink()

labels.to_csv(OUT / "intelligent_known_training.csv", index=False)
stage1.to_csv(OUT / "intelligent_stage1_ranked.csv", index=False)
pd.DataFrame(quality).to_csv(
    OUT / "intelligent_model_pool_quality.csv", index=False
)
ranked.to_csv(OUT / "intelligent_candidates_ranked.csv", index=False)
selected.to_csv(OUT / "intelligent_selected_candidates.csv", index=False)

payload = {
    "research_version":RESEARCH_VERSION,
    "execution_schema":EXECUTION_SCHEMA,
    "checkpoint_positive_only_sha":CHECKPOINT_SHA,
    "known_positive_additions":list(KNOWN_POS),
        "financial_reference_additions":sorted(B2_POS),
    "protocol":{
        "random_sampling":False,
        "candidate_strategy_replays":0,
        "selection_uses_new_candidate_capital":False,
        "stage1_model_pool_target":MODEL_POOL_SIZE,
        "stage1_model_pool_actual":len(pool),
        "stage2_score_reference":"U59_WINNER",
        "target_selected_count":SELECTED_COUNT,
        "actual_selected_count":len(selected_symbols),
        "forced_fill":False,
        "if_strong_under_20":"freeze fewer than 20; never fill with rejected classes",
    },
    "signature":{
        "name":SIGNATURE_NAME,
        "beats_max":BEATS_MAX,
        "score_std_max":SCORE_STD_MAX,
        "score_mean_max":SCORE_MEAN_MAX,
        "abs_corr_u59_best_max":ABS_CORR_BEST_MAX,
        "minimum_score_session_share":MIN_SESSION_SHARE,
        "expected_score_sessions":expected_sessions,
        "minimum_score_sessions":min_sessions,
    },
    "stage1":{
        "snapshot_sha256":smart_manifest.get("snapshot_sha256"),
        "pool":list(pool),
        "history_eligible_count":len(stage1),
    },
    "stage2":{
        "score_reference":"U59_WINNER",
        "u59_assets":list(u59_symbols),
        "financial_reference_additions":sorted(B2_POS),
        "known_positive_additions":list(KNOWN_POS),
        "model_eligible_count":len(candidates),
        "strong_signature_selected":strong,
        "selected_assets":list(selected_symbols),
        "structural_exclusions":structural_exclusions,
    },
    "runtime_seconds":float(time.perf_counter()-t0),
    "interpretation_rule":(
        "Selected assets are hypotheses only. Their capital was not observed "
        "during selection. They were scored against the U59 winner context. "
        "The financial runner must test this frozen list without changing it."
    ),
}
with (OUT / "asset_search.json").open(
    "w", encoding="utf-8"
) as f:
    json.dump(payload, f, indent=2, sort_keys=True, default=str)

# A mesma lista e salva junto ao snapshot para o runner financeiro.
selected.to_csv(SMART.root / "selected_candidates.csv", index=False)

package = criar_pacote_analise(
    OUT,
    comparison_file="asset_search.json",
    execution_schema=EXECUTION_SCHEMA,
    archive_name="pacote_busca_ativos.zip",
)
print(f"[package] pronto={package}", flush=True)
sinal_sonoro_conclusao()
print(
    f"[done] intelligent screen seconds={time.perf_counter()-t0:.3f}",
    flush=True,
)
