"""TCC: benchmark causal de risco sistemico, apenas pesquisa shadow.

Toda feature termina no fechamento t; evento futuro e apenas rotulo posterior.
Nao e modelo da posicao, nem substitui Control, nem executa CASH.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import genpareto, t as student_t
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler

VERSION = "u75-systemic-risk-benchmark-v1.0.0"
METHOD_FEATURES = {
    "CUSUM": ("cusum_down",),
    "BOCPD": ("bocpd_change_probability", "bocpd_down_change"),
    "Markov_filtrado": ("markov_stress", "markov_transition_stress"),
    "PCA_Absorption": ("pc1_share", "absorption_ratio", "breadth_down20"),
    "DCC_proxy": ("dcc_mean_corr", "dcc_tail_dependence_proxy"),
    "EVT_POT": ("evt_es95", "evt_tail_scale"),
    "LedoitWolf": ("mahalanobis", "coexceedance"),
    "Logistica_multivariada": (
        "market_r5", "market_r20", "market_vol20", "breadth_down5",
        "breadth_down20", "dispersion20", "coexceedance", "cusum_down",
        "bocpd_change_probability", "pc1_share", "absorption_ratio",
        "dcc_mean_corr", "evt_es95", "mahalanobis",
    ),
}
METHODS = ("Base_historica", *METHOD_FEATURES, "LightGBM_multivariado")


@dataclass(frozen=True)
class Settings:
    horizons: tuple[int, ...] = (1, 5, 10, 20)
    min_coverage_fraction: float = .90
    future_breadth_fraction: float = .60
    tail_quantile: float = .15
    min_training_dates: int = 252
    covariance_window: int = 60
    covariance_step: int = 5
    evt_window: int = 252
    evt_step: int = 20
    bocpd_hazard: float = 1 / 125
    bocpd_max_run_length: int = 128
    alert_training_quantile: float = .90
    min_positive_events: int = 10
    random_state: int = 42

    def __post_init__(self):
        if (not self.horizons or min(self.horizons) < 1
                or not 0 < self.min_coverage_fraction <= 1
                or not .5 <= self.future_breadth_fraction <= 1
                or not 0 < self.tail_quantile < .5
                or self.min_training_dates < 100
                or self.covariance_window < 20
                or self.covariance_step < 1
                or self.evt_window < 100
                or not 0 < self.bocpd_hazard < 1):
            raise ValueError("Configuracao cientifica invalida")


def _positive_log_series(frame, field, index):
    s = pd.to_numeric(frame[field], errors="coerce").copy()
    s.index = pd.to_datetime(s.index, utc=True).normalize()
    if s.index.has_duplicates or not s.index.is_monotonic_increasing:
        raise ValueError("Serie diaria duplicada ou desordenada")
    return np.log(s.where(s > 0).reindex(index))


def _bocpd_series(standardized, cfg):
    """Student-t conjugado, run-length filtrado; somente amostras ate t."""
    x = np.asarray(standardized, dtype=float)
    n = len(x)
    cp, down = np.full(n, np.nan), np.full(n, np.nan)
    posterior = np.array([1.])
    mus = np.array([0.])
    kappas = np.array([1.])
    alphas = np.array([2.])
    betas = np.array([2.])
    for i, obs in enumerate(x):
        if not np.isfinite(obs):
            posterior, mus, kappas, alphas, betas = (
                np.array([1.]), np.array([0.]), np.array([1.]),
                np.array([2.]), np.array([2.]))
            continue
        df = 2 * alphas
        scale = np.sqrt(betas * (kappas + 1) / (alphas * kappas))
        likelihood = np.maximum(student_t.pdf(obs, df, loc=mus, scale=scale), 1e-200)
        growth = posterior * likelihood * (1 - cfg.bocpd_hazard)
        change = np.sum(posterior * likelihood) * cfg.bocpd_hazard
        raw = np.r_[change, growth][:cfg.bocpd_max_run_length]
        posterior = raw / max(raw.sum(), 1e-200)
        old_mu, old_k, old_a, old_b = mus, kappas, alphas, betas
        new_k = old_k + 1
        new_mu = (old_k * old_mu + obs) / new_k
        new_a = old_a + .5
        new_b = old_b + old_k * (obs - old_mu) ** 2 / (2 * new_k)
        mus = np.r_[0., new_mu][:len(posterior)]
        kappas = np.r_[1., new_k][:len(posterior)]
        alphas = np.r_[2., new_a][:len(posterior)]
        betas = np.r_[2., new_b][:len(posterior)]
        cp[i] = posterior[0]
        # Change-point probability * current negative level, not only any shift.
        down[i] = cp[i] * max(0., -obs)
    return cp, down


def _cusum_series(ret, volatility):
    acc = np.zeros(len(ret), dtype=float)
    score = np.full(len(ret), np.nan)
    for i, (r, v) in enumerate(zip(ret, volatility)):
        if not np.isfinite(r) or not np.isfinite(v) or v <= 0:
            acc[i] = 0.
            continue
        acc[i] = max(0., (acc[i - 1] if i else 0.) + (-r / v) - .5)
        score[i] = acc[i]
    return score


def _covariance_features(asset_returns, index, cfg):
    """Ledoit-Wolf + PCA covariance shrunk; outputs last-seen t, never future."""
    n, m = asset_returns.shape
    pc1 = np.full(n, np.nan)
    absorption = np.full(n, np.nan)
    corr_mean = np.full(n, np.nan)
    mahal = np.full(n, np.nan)
    dcc_mean = np.full(n, np.nan)
    dcc_tail = np.full(n, np.nan)
    required = ceil(cfg.min_coverage_fraction * m)
    step = cfg.covariance_step
    for i in range(cfg.covariance_window, n):
        # Corr/PCA recalculada a cada step e carregada apenas para frente.
        if i % step != 0:
            for vector in (pc1, absorption, corr_mean, mahal, dcc_mean, dcc_tail):
                vector[i] = vector[i - 1]
            continue
        window = asset_returns[i - cfg.covariance_window + 1:i + 1]
        valid = np.isfinite(window).all(axis=0)
        if valid.sum() < required:
            continue
        sample = window[:, valid]
        lw = LedoitWolf().fit(sample[:-1])
        covariance = lw.covariance_
        sd = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
        cor = covariance / np.outer(sd, sd)
        eigenvalues = np.maximum(np.linalg.eigvalsh(cor), 0.)[::-1]
        k = max(1, ceil(.20 * len(eigenvalues)))
        pc1[i] = eigenvalues[0] / max(eigenvalues.sum(), 1e-12)
        absorption[i] = eigenvalues[:k].sum() / max(eigenvalues.sum(), 1e-12)
        corr_mean[i] = (cor.sum() - np.trace(cor)) / max(len(cor) * (len(cor) - 1), 1)
        residual = sample[-1] - lw.location_
        mahal[i] = np.log1p(max(0., float(residual @ np.linalg.solve(covariance, residual))))
        # Fixed-parameter DCC(1,1)-like recursion, EWMA-standardized shocks;
        # label as proxy, not a fitted multivariate DCC-GARCH.
        vol = pd.DataFrame(sample).ewm(span=20, adjust=False).std().to_numpy()
        z = sample / np.maximum(vol, 1e-6)
        z = np.nan_to_num(z, nan=0., posinf=0., neginf=0.)
        S = np.cov(z[:-1], rowvar=False)
        q = S.copy()
        a, b = .02, .97
        for obs in z[1:]:
            q = (1 - a - b) * S + a * np.outer(obs, obs) + b * q
        d = np.sqrt(np.maximum(np.diag(q), 1e-12))
        dyn_corr = q / np.outer(d, d)
        upper = dyn_corr[np.triu_indices_from(dyn_corr, 1)]
        dcc_mean[i] = float(np.nanmean(upper))
        dcc_tail[i] = float(np.nanmean(upper > .6))
    return pd.DataFrame({
        "pc1_share": pc1, "absorption_ratio": absorption, "corr_mean": corr_mean,
        "mahalanobis": mahal, "dcc_mean_corr": dcc_mean,
        "dcc_tail_dependence_proxy": dcc_tail,
    }, index=index)


def _evt_features(market_returns, cfg, index):
    """POT/GPD no passado, estimativa da cauda de perda diaria."""
    n = len(market_returns)
    es = np.full(n, np.nan)
    scale_out = np.full(n, np.nan)
    for i in range(cfg.evt_window, n):
        if i % cfg.evt_step and i:
            es[i], scale_out[i] = es[i - 1], scale_out[i - 1]
            continue
        history = market_returns[i - cfg.evt_window:i]
        history = history[np.isfinite(history)]
        if len(history) < cfg.evt_window * cfg.min_coverage_fraction:
            continue
        loss = -history
        threshold = np.quantile(loss, .90)
        excess = loss[loss > threshold] - threshold
        if len(excess) < 15 or np.min(excess) < 0:
            continue
        try:
            shape, _, scale = genpareto.fit(excess, floc=0)
            if not np.isfinite(shape + scale) or scale <= 0 or shape >= 1:
                continue
            # ES above empirical 95%, GPD conditional tail survival.
            tail_prob = len(excess) / len(loss)
            q95 = threshold + float(genpareto.ppf(
                1 - .05 / tail_prob, shape, loc=0, scale=scale))
            es[i] = q95 + (scale + shape * (q95 - threshold)) / (1 - shape)
            scale_out[i] = scale
        except (ValueError, FloatingPointError):
            continue
    return pd.DataFrame({"evt_es95": es, "evt_tail_scale": scale_out}, index=index)


def make_panel(raw: dict[str, pd.DataFrame], symbols: list[str],
               cfg: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Universo declarado pelo manifesto; nao completa barras ausentes."""
    if not symbols or len(set(symbols)) != len(symbols) or set(raw) != set(symbols):
        raise ValueError("Ativos devem coincidir com universo declarado")
    indices = [pd.DatetimeIndex(pd.to_datetime(f.index, utc=True)).normalize()
               for f in raw.values()]
    index = pd.DatetimeIndex(sorted(set().union(*(set(v) for v in indices))), tz="UTC")
    close = pd.DataFrame({s: _positive_log_series(raw[s], "close", index) for s in symbols})
    returns = close.diff()
    valid_count = returns.notna().sum(axis=1)
    minimum = ceil(len(symbols) * cfg.min_coverage_fraction)
    simple = np.expm1(returns)
    market = np.log1p(simple.mean(axis=1).where(valid_count >= minimum))
    # Continuous h-step future labels require no missing market-return session.
    r5 = market.rolling(5, min_periods=5).sum()
    r20 = market.rolling(20, min_periods=20).sum()
    vol20 = market.rolling(20, min_periods=20).std()
    vol_lag = market.shift(1).rolling(60, min_periods=30).std()
    standardized = (market / vol_lag.where(vol_lag > 1e-8)).clip(-8, 8)
    cp, down_cp = _bocpd_series(standardized.to_numpy(float), cfg)
    daily_vol = returns.rolling(20, min_periods=20).std()
    extreme_down = (returns < (-1.645 * daily_vol.shift(1))).where(
        returns.notna() & daily_vol.shift(1).notna())
    complete20 = returns.notna().rolling(20, min_periods=20).sum().eq(20)
    complete5 = returns.notna().rolling(5, min_periods=5).sum().eq(5)
    breadth20_return = close.diff(20).where(complete20)
    breadth5_return = close.diff(5).where(complete5)
    breadth_down20 = (breadth20_return < 0).sum(axis=1) / breadth20_return.notna().sum(axis=1).replace(0, np.nan)
    breadth_down5 = (breadth5_return < 0).sum(axis=1) / breadth5_return.notna().sum(axis=1).replace(0, np.nan)
    coexceed = extreme_down.sum(axis=1) / extreme_down.notna().sum(axis=1).replace(0, np.nan)
    dispersion = breadth20_return.std(axis=1)
    features = pd.DataFrame({
        "market_r1": market, "market_r5": r5, "market_r20": r20,
        "market_vol20": vol20, "breadth_down5": breadth_down5,
        "breadth_down20": breadth_down20, "dispersion20": dispersion,
        "coexceedance": coexceed,
        "cusum_down": _cusum_series(market.to_numpy(float), vol_lag.to_numpy(float)),
        "bocpd_change_probability": cp, "bocpd_down_change": down_cp,
        "valid_asset_returns": valid_count,
    }, index=index)
    # Do not allow weak coverage to be interpreted as no systemic loss.
    for name, h in (("breadth_down5", 5), ("breadth_down20", 20)):
        coverage = close.diff(h).notna().sum(axis=1)
        features[name] = features[name].where(coverage >= minimum)
    features["coexceedance"] = features["coexceedance"].where(
        extreme_down.notna().sum(axis=1) >= minimum)
    features = features.join(_covariance_features(returns.to_numpy(float), index, cfg))
    features = features.join(_evt_features(market.to_numpy(float), cfg, index))
    labels = pd.DataFrame(index=index)
    for h in cfg.horizons:
        future = pd.concat([market.shift(-lag) for lag in range(1, h + 1)], axis=1)
        future_market = future.sum(axis=1, min_count=h)
        future_contiguous = returns.notna().rolling(h, min_periods=h).sum().eq(h).shift(-h).fillna(False)
        future_asset = (close.shift(-h) - close).where(future_contiguous)
        count = future_asset.notna().sum(axis=1)
        down = ((future_asset < 0).sum(axis=1) / count.replace(0, np.nan)).where(count >= minimum)
        labels[f"future_market_{h}"] = future_market
        labels[f"future_breadth_down_{h}"] = down
        # Matured is calendar index + h, and only used once it is past train boundary.
        labels[f"maturity_{h}"] = pd.Series(index, index=index).shift(-h)
    return features, labels


def _markov_features(frame: pd.DataFrame, train_rows: np.ndarray,
                     rows: np.ndarray, seed: int) -> np.ndarray:
    """Gaussian HMM com 3 estados, EM diagonal, filtro forward somente ate t.

    Aprende parametros no treino e devolve P(estresse_t), P(estresse_t+1).
    O estado 'estresse' e definido APENAS no treino pela menor media de retorno.
    """
    from sklearn.cluster import KMeans
    sample = frame[["market_r1", "breadth_down5"]].to_numpy(float)
    valid = np.isfinite(sample).all(axis=1)
    train = np.asarray(train_rows, int)
    train = train[valid[train]]
    if len(train) < 90:
        return np.full((len(rows), 2), np.nan)
    scaler = StandardScaler().fit(sample[train])
    z = np.zeros_like(sample)
    z[valid] = np.clip(scaler.transform(sample[valid]), -8, 8)
    x = z[train]
    k = 3
    km = KMeans(n_clusters=k, random_state=seed, n_init=5).fit(x)
    mu = km.cluster_centers_.copy()
    var = np.vstack([np.var(x[km.labels_ == j], axis=0) if (km.labels_ == j).sum() > 5
                     else np.ones(2) for j in range(k)])
    var = np.maximum(var, .1)
    A = np.full((k, k), .025)
    np.fill_diagonal(A, .95)
    pi = np.full(k, 1 / k)
    def emissions(data):
        return -0.5 * (np.log(2 * np.pi * var).sum(axis=1)[None, :]
                + (((data[:, None, :] - mu[None, :, :]) ** 2) / var[None, :, :]).sum(axis=2))
    # Baum-Welch, past-only training, stable scaled forward-backward.
    for _ in range(12):
        ll = emissions(x)
        logA = np.log(np.maximum(A, 1e-12))
        fwd = np.empty((len(x), k))
        fwd[0] = np.log(pi) + ll[0]
        for i in range(1, len(x)):
            fwd[i] = ll[i] + logsumexp(fwd[i-1][:, None] + logA, axis=0)
        bwd = np.zeros((len(x), k))
        for i in range(len(x)-2, -1, -1):
            bwd[i] = logsumexp(logA + ll[i+1][None, :] + bwd[i+1][None, :], axis=1)
        gamma = np.exp(fwd + bwd - logsumexp(fwd[-1]))
        weight = gamma.sum(axis=0) + 1e-10
        mu = (gamma.T @ x) / weight[:, None]
        var = np.maximum((gamma.T @ (x**2)) / weight[:, None] - mu**2, .05)
        xi = np.zeros((k, k))
        for i in range(len(x)-1):
            zz = (fwd[i][:, None] + logA + ll[i+1][None, :] + bwd[i+1][None, :])
            xi += np.exp(zz - logsumexp(zz))
        A = (xi + .5) / (xi.sum(axis=1, keepdims=True) + .5 * k)
        pi = (gamma[0] + .5) / (gamma[0].sum() + .5 * k)
    stress = int(np.argmin(mu[:, 0]))
    output = np.full((len(rows), 2), np.nan)
    wanted = {int(t): p for p, t in enumerate(rows)}
    alpha = pi.copy()
    for i in range(max(0, int(train.min())), int(rows.max()) + 1):
        if not valid[i]:
            continue
        logp = np.log(np.maximum(alpha @ A, 1e-200)) + emissions(z[i:i+1])[0]
        alpha = np.exp(logp - logsumexp(logp))
        if i in wanted:
            output[wanted[i]] = (alpha[stress], (alpha @ A)[stress])
    return output


def _fit_predict(method, train: pd.DataFrame, ytrain: np.ndarray,
                 test: pd.DataFrame, cfg: Settings):
    """Calibration is fit only on already matured historical labels."""
    if method == "Base_historica":
        p = float((ytrain.sum() + 1) / (len(ytrain) + 2))
        return np.full(len(test), p), np.full(len(train), p)
    cols = METHOD_FEATURES.get(method, METHOD_FEATURES["Logistica_multivariada"])
    x = train[list(cols)].to_numpy(float)
    xt = test[list(cols)].to_numpy(float)
    scaler = StandardScaler().fit(x)
    x = np.clip(scaler.transform(x), -8, 8)
    xt = np.clip(scaler.transform(xt), -8, 8)
    if method == "LightGBM_multivariado":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=100, max_depth=3, num_leaves=7,
                               learning_rate=.035, min_child_samples=35,
                               reg_lambda=10., verbosity=-1,
                               random_state=cfg.random_state, n_jobs=2)
    else:
        model = LogisticRegression(max_iter=400, C=.2, class_weight=None,
                                   random_state=cfg.random_state)
    model.fit(x, ytrain)
    return model.predict_proba(xt)[:, 1], model.predict_proba(x)[:, 1]


def evaluate(features: pd.DataFrame, labels: pd.DataFrame, cfg: Settings,
             *, progress=None):
    """Nested temporal maturity: outcome maturity < first test date.

    Annual expanding folds. Same fold and threshold for every method.
    Returns all decisions, metrics and audit, no posthoc tuning.
    """
    index = features.index
    preds, audit, summaries = [], [], []
    methods = list(METHODS)
    for year in sorted(index.year.unique()):
        test_positions = np.flatnonzero(index.year == year)
        if not len(test_positions):
            continue
        first = test_positions[0]
        if first < cfg.min_training_dates + max(cfg.horizons):
            continue
        # The HMM emissions and transitions depend on the training-year
        # boundary, not on the forecast horizon: fit only ONCE per year.
        markov_positions = np.arange(int(test_positions[-1]) + 1)
        markov_result = _markov_features(
            features, np.arange(first), markov_positions, cfg.random_state)
        extra = features.copy()
        extra["markov_stress"] = np.nan
        extra["markov_transition_stress"] = np.nan
        extra.iloc[markov_positions, extra.columns.get_loc("markov_stress")] = markov_result[:, 0]
        extra.iloc[markov_positions, extra.columns.get_loc("markov_transition_stress")] = markov_result[:, 1]
        shared_columns = sorted({x for cols in METHOD_FEATURES.values() for x in cols})
        for h in cfg.horizons:
            if progress:
                progress(year, h)
            outcome = labels[f"future_market_{h}"]
            breadth = labels[f"future_breadth_down_{h}"]
            matured = labels[f"maturity_{h}"]
            historical = ((np.arange(len(index)) < first)
                          & ((matured < index[first]).fillna(False).to_numpy())
                          & outcome.notna().to_numpy() & breadth.notna().to_numpy())
            idx_train_all = np.flatnonzero(historical)
            if len(idx_train_all) < cfg.min_training_dates:
                continue
            threshold = float(np.quantile(outcome.iloc[idx_train_all], cfg.tail_quantile))
            event = ((outcome <= threshold) &
                     (breadth >= cfg.future_breadth_fraction))
            # Shared sample guarantees identical dates for all models.
            shared_train = idx_train_all[np.isfinite(extra.iloc[idx_train_all][shared_columns].to_numpy(float)).all(axis=1)]
            shared_test = test_positions[np.isfinite(extra.iloc[test_positions][shared_columns].to_numpy(float)).all(axis=1)]
            shared_test = shared_test[(outcome.iloc[shared_test].notna() & breadth.iloc[shared_test].notna()).to_numpy()]
            for method in methods:
                tr = shared_train
                te = shared_test
                positive = int(event.iloc[tr].sum()) if len(tr) else 0
                if len(tr) < cfg.min_training_dates or positive < cfg.min_positive_events or not len(te):
                    audit.append({"year":int(year), "horizon":h,"method":method,
                                  "status":"insufficient_training", "training":int(len(tr)),
                                  "events_training":positive,"testing":int(len(te))})
                    continue
                p, ptr = _fit_predict(method, extra.iloc[tr], event.iloc[tr].to_numpy(int),
                                      extra.iloc[te], cfg)
                if not np.isfinite(p).all():
                    raise RuntimeError(f"Probabilidades invalidas: {method} {year} h={h}")
                p = np.clip(p, 1e-6, 1-1e-6)
                cutoff = float(np.quantile(ptr, cfg.alert_training_quantile))
                if method == "Base_historica":
                    alerts = np.zeros(len(p), dtype=bool)
                    cutoff = 1.
                else:
                    alerts = p > cutoff  # strict > avoids all-constant alert.
                for position, pr, flag in zip(te, p, alerts):
                    preds.append({"date":index[position].date().isoformat(),
                                  "year":int(year), "horizon":h, "method":method,
                                  "probability":float(pr), "alert":bool(flag),
                                  "realized_event":bool(event.iloc[position]),
                                  "market_future_log_return":float(outcome.iloc[position]),
                                  "future_breadth_down":float(breadth.iloc[position]),
                                  "tail_threshold_train":threshold,
                                  "train_max_maturity":str(matured.iloc[tr].max().date())})
                audit.append({"year":int(year),"horizon":h,"method":method,
                              "status":"ok","training":int(len(tr)),
                              "events_training":positive,"testing":int(len(te)),
                              "first_test_date":str(index[te[0]].date()),
                              "last_training_maturity":str(matured.iloc[tr].max().date()),
                              "tail_threshold_train":threshold,
                              "alert_cutoff_train":cutoff})
    result = pd.DataFrame(preds)
    checks = pd.DataFrame(audit)
    if result.empty:
        raise RuntimeError("Nenhuma previsao cronologica OOS; verificar cobertura.")
    for (h, method), group in result.groupby(["horizon", "method"], sort=False):
        y = group.realized_event.to_numpy(dtype=int)
        p = group.probability.to_numpy(float)
        a = group.alert.to_numpy(bool)
        tp = int(np.sum(a & (y == 1)))
        fp = int(np.sum(a & (y == 0)))
        fn = int(np.sum((~a) & (y == 1)))
        summaries.append({
            "horizon":int(h),"method":method,"observations":len(y),
            "events":int(y.sum()),"base_rate":float(y.mean()),
            "alerts":int(a.sum()),"tp":tp,"fp":fp,"fn":fn,
            "precision":tp / (tp+fp) if tp+fp else np.nan,
            "recall":tp / (tp+fn) if tp+fn else np.nan,
            "brier":float(brier_score_loss(y,p)),
            "logloss":float(log_loss(y,p,labels=[0,1])),
            "roc_auc":float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,
            "average_precision":float(average_precision_score(y,p)) if y.sum() else np.nan,
        })
    return result, pd.DataFrame(summaries), checks
