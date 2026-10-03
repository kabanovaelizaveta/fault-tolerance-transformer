"""Archimedean copula analysis of the fault-severity / degradation dependence.

Paper B, Section 3.5.  Families: Clayton, Gumbel, Frank and the 180-degree
rotated (survival) Clayton and Gumbel.  Parameters by maximum likelihood,
family choice by AIC, adequacy by the Cramer-von Mises statistic with a
parametric bootstrap.

The notebooks obtain the survival families from pyvinecopulib; that dependency
is replaced here by the analytic rotation

    C_survival(u, v) = u + v - 1 + C(1 - u, 1 - v)
    c_survival(u, v) = c(1 - u, 1 - v)

which is exact and keeps the toolkit dependency-free.  The empirical copula is
evaluated with a Fenwick tree (O(n log n)) instead of the O(n^2) pairwise
comparison, with explicit tie handling -- ties matter here because N_fail is
integer valued.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import kendalltau, pearsonr, spearmanr

EPS = 1e-10

BOUNDS = {
    "clayton": (0.01, 30.0),
    "gumbel": (1.001, 30.0),
    "frank": (0.01, 60.0),
    "survival_clayton": (0.01, 30.0),
    "survival_gumbel": (1.001, 30.0),
}

FAMILIES = tuple(BOUNDS)


# --------------------------------------------------------------------------
# margins
# --------------------------------------------------------------------------
def pseudo_observations(x: np.ndarray) -> np.ndarray:
    """Average-rank pseudo-observations u_i = R_i / (n + 1)."""
    from scipy.stats import rankdata

    x = np.asarray(x, dtype=float)
    return rankdata(x, method="average") / (len(x) + 1.0)


def clip_uv(u: np.ndarray, v: np.ndarray, eps: float = EPS):
    return np.clip(u, eps, 1 - eps), np.clip(v, eps, 1 - eps)


def correlations(x: np.ndarray, y: np.ndarray) -> dict:
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    kr, kp = kendalltau(x, y)
    return {
        "pearson": float(pr), "pearson_p": float(pp),
        "spearman": float(sr), "spearman_p": float(sp),
        "kendall": float(kr), "kendall_p": float(kp),
    }


# --------------------------------------------------------------------------
# base family CDFs and log densities
# --------------------------------------------------------------------------
def _clayton_cdf(u, v, theta):
    return np.power(np.power(u, -theta) + np.power(v, -theta) - 1.0, -1.0 / theta)


def _clayton_logpdf(u, v, theta):
    s = np.power(u, -theta) + np.power(v, -theta) - 1.0
    s = np.maximum(s, EPS)
    return (
        np.log1p(theta)
        - (theta + 1.0) * (np.log(u) + np.log(v))
        - (2.0 + 1.0 / theta) * np.log(s)
    )


def _gumbel_parts(u, v, theta):
    x = -np.log(u)
    y = -np.log(v)
    a = np.power(np.power(x, theta) + np.power(y, theta), 1.0 / theta)
    return x, y, a


def _gumbel_cdf(u, v, theta):
    _, _, a = _gumbel_parts(u, v, theta)
    return np.exp(-a)


def _gumbel_logpdf(u, v, theta):
    x, y, a = _gumbel_parts(u, v, theta)
    a = np.maximum(a, EPS)
    return (
        -a
        + (1.0 - 2.0 * theta) * np.log(a)
        + (theta - 1.0) * (np.log(x) + np.log(y))
        + np.log(a + theta - 1.0)
        - np.log(u)
        - np.log(v)
    )


def _frank_cdf(u, v, theta):
    em = np.expm1(-theta)                       # e^-theta - 1
    num = np.expm1(-theta * u) * np.expm1(-theta * v)
    return -np.log1p(num / em) / theta


def _frank_logpdf(u, v, theta):
    e = -np.expm1(-theta)                       # 1 - e^-theta
    bracket = e - (-np.expm1(-theta * u)) * (-np.expm1(-theta * v))
    bracket = np.maximum(bracket, EPS)
    return np.log(theta) + np.log(e) - theta * (u + v) - 2.0 * np.log(bracket)


_BASE = {
    "clayton": (_clayton_cdf, _clayton_logpdf),
    "gumbel": (_gumbel_cdf, _gumbel_logpdf),
    "frank": (_frank_cdf, _frank_logpdf),
}


def copula_cdf(u, v, family: str, theta: float):
    u, v = clip_uv(np.asarray(u, float), np.asarray(v, float))
    if family in _BASE:
        return _BASE[family][0](u, v, theta)
    if family.startswith("survival_"):
        base = family[len("survival_"):]
        uu, vv = clip_uv(1.0 - u, 1.0 - v)
        return u + v - 1.0 + _BASE[base][0](uu, vv, theta)
    raise ValueError("Unknown copula family: %s" % family)


def copula_logpdf(u, v, family: str, theta: float):
    u, v = clip_uv(np.asarray(u, float), np.asarray(v, float))
    if family in _BASE:
        return _BASE[family][1](u, v, theta)
    if family.startswith("survival_"):
        base = family[len("survival_"):]
        uu, vv = clip_uv(1.0 - u, 1.0 - v)
        return _BASE[base][1](uu, vv, theta)
    raise ValueError("Unknown copula family: %s" % family)


def tail_dependence(family: str, theta: float) -> dict:
    if family == "clayton":
        return {"lambda_L": float(2.0 ** (-1.0 / theta)), "lambda_U": 0.0}
    if family == "gumbel":
        return {"lambda_L": 0.0, "lambda_U": float(2.0 - 2.0 ** (1.0 / theta))}
    if family == "frank":
        return {"lambda_L": 0.0, "lambda_U": 0.0}
    if family == "survival_clayton":
        return {"lambda_L": 0.0, "lambda_U": float(2.0 ** (-1.0 / theta))}
    if family == "survival_gumbel":
        return {"lambda_L": float(2.0 - 2.0 ** (1.0 / theta)), "lambda_U": 0.0}
    raise ValueError("Unknown copula family: %s" % family)


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------
def fit_copula(u: np.ndarray, v: np.ndarray, family: str) -> dict:
    u, v = clip_uv(np.asarray(u, float), np.asarray(v, float))

    def nll(theta: float) -> float:
        with np.errstate(all="ignore"):
            ll = copula_logpdf(u, v, family, float(theta))
        if not np.all(np.isfinite(ll)):
            return 1e12
        return float(-np.sum(ll))

    lo, hi = BOUNDS[family]
    res = minimize_scalar(nll, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-6})
    theta = float(res.x)
    loglik = -float(res.fun)
    out = {
        "family": family,
        "theta": theta,
        "loglik": loglik,
        "aic": float(2.0 * 1.0 - 2.0 * loglik),
        "n": int(len(u)),
        "at_bound": bool(abs(theta - lo) < 1e-4 or abs(theta - hi) < 1e-4),
    }
    out.update(tail_dependence(family, theta))
    return out


# --------------------------------------------------------------------------
# empirical copula (Fenwick tree, exact with ties)
# --------------------------------------------------------------------------
class _Fenwick:
    __slots__ = ("n", "t")

    def __init__(self, n: int):
        self.n = n
        self.t = [0] * (n + 1)

    def add(self, i: int) -> None:
        while i <= self.n:
            self.t[i] += 1
            i += i & (-i)

    def prefix(self, i: int) -> int:
        s = 0
        while i > 0:
            s += self.t[i]
            i -= i & (-i)
        return s


def empirical_copula_at_points(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """C_n(u_i, v_i) = (1/n) #{j : u_j <= u_i and v_j <= v_i}, ties included."""
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    n = len(u)

    v_ranks = np.searchsorted(np.unique(v), v) + 1        # 1..m, ties share a rank
    order = np.argsort(u, kind="mergesort")
    bit = _Fenwick(int(v_ranks.max()))
    out = np.empty(n, dtype=float)

    i = 0
    u_sorted = u[order]
    while i < n:
        j = i
        while j + 1 < n and u_sorted[j + 1] == u_sorted[i]:
            j += 1
        group = order[i : j + 1]
        for idx in group:                                  # insert the whole tie group
            bit.add(int(v_ranks[idx]))
        for idx in group:                                  # then query it
            out[idx] = bit.prefix(int(v_ranks[idx]))
        i = j + 1

    return out / n


def cvm_statistic(u: np.ndarray, v: np.ndarray, family: str, theta: float) -> float:
    cn = empirical_copula_at_points(u, v)
    ct = copula_cdf(u, v, family, theta)
    return float(np.sum((cn - ct) ** 2))


# --------------------------------------------------------------------------
# sampling (for the parametric bootstrap)
# --------------------------------------------------------------------------
def _sample_clayton(theta, n, rng):
    u1 = rng.random(n)
    t = rng.random(n)
    w = np.power(t, -theta / (theta + 1.0)) - 1.0
    u2 = np.power(w * np.power(u1, -theta) + 1.0, -1.0 / theta)
    return u1, u2


def _sample_frank(theta, n, rng):
    u1 = rng.random(n)
    t = rng.random(n)
    em = np.expm1(-theta)
    denom = np.exp(-theta * u1) * (1.0 - t) + t
    u2 = -np.log1p(t * em / denom) / theta
    return u1, np.clip(u2, EPS, 1 - EPS)


def _sample_gumbel(theta, n, rng):
    """Conditional inversion with vectorised bisection on the h-function."""
    u1 = rng.random(n)
    t = rng.random(n)
    u1 = np.clip(u1, EPS, 1 - EPS)

    def h(v):
        x, y, a = _gumbel_parts(u1, v, theta)
        return np.exp(-a) * np.power(a, 1.0 - theta) * np.power(x, theta - 1.0) / u1

    lo = np.full(n, EPS)
    hi = np.full(n, 1.0 - EPS)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        with np.errstate(all="ignore"):
            hv = h(mid)
        hv = np.where(np.isfinite(hv), hv, 0.0)
        go_up = hv < t
        lo = np.where(go_up, mid, lo)
        hi = np.where(go_up, hi, mid)
    return u1, 0.5 * (lo + hi)


def sample_copula(family: str, theta: float, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    if family == "clayton":
        return _sample_clayton(theta, n, rng)
    if family == "frank":
        return _sample_frank(theta, n, rng)
    if family == "gumbel":
        return _sample_gumbel(theta, n, rng)
    if family.startswith("survival_"):
        u, v = sample_copula(family[len("survival_"):], theta, n, rng)
        return 1.0 - u, 1.0 - v
    raise ValueError("Unknown copula family: %s" % family)


def cvm_bootstrap(u: np.ndarray, v: np.ndarray, family: str, theta: float,
                  reps: int = 500, seed: int = 2026) -> dict:
    """Parametric bootstrap p-value for H0: C = C_theta."""
    rng = np.random.default_rng(seed)
    observed = cvm_statistic(u, v, family, theta)
    n = len(u)

    stats = np.empty(reps, dtype=float)
    for b in range(reps):
        ub, vb = sample_copula(family, theta, n, rng)
        ub = pseudo_observations(ub)
        vb = pseudo_observations(vb)
        fit_b = fit_copula(ub, vb, family)
        stats[b] = cvm_statistic(ub, vb, family, fit_b["theta"])

    p = (1.0 + float(np.sum(stats >= observed))) / (reps + 1.0)
    return {
        "cvm_observed": observed,
        "bootstrap_reps": reps,
        "bootstrap_p": p,
        "bootstrap_mean_cvm": float(stats.mean()),
        "bootstrap_q95_cvm": float(np.quantile(stats, 0.95)),
        "reject_at_005": bool(p < 0.05),
        # Kept so the notebook can save the complete bootstrap distribution,
        # matching the reproducibility outputs of the source study.
        "bootstrap_statistics": stats,
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def analyse_pair(x: np.ndarray, y: np.ndarray, families=FAMILIES,
                 bootstrap_reps: int = 500, seed: int = 2026) -> dict:
    """Fit every family to one (severity, degradation) pair and select by AIC."""
    u = pseudo_observations(x)
    v = pseudo_observations(y)

    fits = [fit_copula(u, v, f) for f in families]
    for fit in fits:
        fit["cvm"] = cvm_statistic(u, v, fit["family"], fit["theta"])
    fits.sort(key=lambda d: d["aic"])
    best = fits[0]

    gof = cvm_bootstrap(u, v, best["family"], best["theta"],
                        reps=bootstrap_reps, seed=seed)
    return {
        "correlations": correlations(x, y),
        "fits": fits,
        "selected": best,
        "gof": gof,
        "u": u,
        "v": v,
    }
