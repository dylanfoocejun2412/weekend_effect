"""Econometric core for the conditional weekend-effect study.

The maintained hypothesis: the classic negative-Monday "weekend effect" has
largely disappeared on average post-2015, but *survives in high-volatility
regimes*. We test this with day-of-week dummies interacted with a high-VIX
indicator, using Newey-West (HAC) standard errors to handle the serial
correlation and heteroskedasticity that daily return series exhibit.

Specification (estimated per asset, no intercept so every weekday gets its own
coefficient):

    ret_t = sum_d  beta_d  * D[weekday=d]_t
          + sum_d  gamma_d * D[weekday=d]_t * HighVIX_t
          + e_t

  beta_Mon   = average Monday return in the LOW-VIX regime
  gamma_Mon  = ADDITIONAL Monday return when VIX is high
  beta_Mon + gamma_Mon = average Monday return in the HIGH-VIX regime

A surviving, regime-conditional weekend effect implies gamma_Mon < 0 and
(beta_Mon + gamma_Mon) significantly negative, even when beta_Mon alone is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
HAC_LAGS = 5  # one trading week


def assign_regime(
    df: pd.DataFrame, method: str = "median", threshold: float | None = None
) -> pd.Series:
    """Return a boolean HighVIX series for a single-asset frame.

    method='median'    : split on the in-sample median VIX (balanced regimes).
    method='threshold' : HighVIX when vix > `threshold` (default 20, a common
                         "elevated volatility" line).
    """
    if method == "median":
        cut = df["vix"].median()
    elif method == "threshold":
        cut = 20.0 if threshold is None else threshold
    else:
        raise ValueError(f"unknown method {method!r}")
    return df["vix"] > cut


def _design(df: pd.DataFrame, high: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Build the dummy + interaction design matrix X and target y."""
    wd = pd.get_dummies(df["weekday"], prefix="", prefix_sep="")
    wd = wd.reindex(columns=WEEKDAYS, fill_value=0).astype(float)

    inter = wd.mul(high.astype(float).values, axis=0)
    inter.columns = [f"{c}_HV" for c in inter.columns]

    X = pd.concat([wd, inter], axis=1)
    y = df["ret"].astype(float)
    return X, y


@dataclass
class FitResult:
    asset: str
    sector: str
    n: int
    n_high: int
    params: pd.Series
    bse: pd.Series
    pvalues: pd.Series
    # Per-weekday high-VIX mean (beta_d + gamma_d) and its exact HAC SE,
    # computed as a linear contrast so the covariance between the weekday and
    # interaction terms is accounted for (not the naive sqrt(se^2 + se^2)).
    high_coef: pd.Series
    high_se: pd.Series
    # Convenience scalars for the Monday story.
    beta_mon: float
    beta_mon_p: float
    gamma_mon: float
    gamma_mon_p: float
    mon_high: float  # beta_Mon + gamma_Mon
    mon_high_p: float  # p-value of that linear combination


def fit_asset(
    df: pd.DataFrame, regime_method: str = "median", threshold: float | None = None
) -> FitResult:
    """Estimate the interacted weekday model for one asset with HAC errors."""
    df = df.sort_values("date")
    high = assign_regime(df, regime_method, threshold)
    X, y = _design(df, high)

    model = sm.OLS(y.values, X.values)
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})

    names = list(X.columns)
    params = pd.Series(res.params, index=names)
    bse = pd.Series(res.bse, index=names)
    pvals = pd.Series(res.pvalues, index=names)

    # Per-weekday high-VIX mean beta_d + gamma_d, with the exact contrast SE
    # (uses the full covariance matrix, so the weekday/interaction correlation
    # is handled correctly). Done for every weekday for the regime plot.
    high_coef = {}
    high_se = {}
    for d in WEEKDAYS:
        c = np.zeros(len(names))
        c[names.index(d)] = 1.0
        c[names.index(f"{d}_HV")] = 1.0
        tt_d = res.t_test(c)
        high_coef[d] = float(np.ravel(tt_d.effect)[0])
        high_se[d] = float(np.ravel(tt_d.sd)[0])
    high_coef = pd.Series(high_coef)
    high_se = pd.Series(high_se)

    mon_high = float(high_coef["Mon"])
    contrast = np.zeros(len(names))
    contrast[names.index("Mon")] = 1.0
    contrast[names.index("Mon_HV")] = 1.0
    mon_high_p = float(np.ravel(res.t_test(contrast).pvalue)[0])

    return FitResult(
        asset=df["asset"].iloc[0],
        sector=df["sector"].iloc[0],
        n=len(df),
        n_high=int(high.sum()),
        params=params,
        bse=bse,
        pvalues=pvals,
        high_coef=high_coef,
        high_se=high_se,
        beta_mon=float(params["Mon"]),
        beta_mon_p=float(pvals["Mon"]),
        gamma_mon=float(params["Mon_HV"]),
        gamma_mon_p=float(pvals["Mon_HV"]),
        mon_high=mon_high,
        mon_high_p=mon_high_p,
    )


def fit_all(
    panel: pd.DataFrame, regime_method: str = "median", threshold: float | None = None
) -> tuple[FitResult, pd.DataFrame]:
    """Fit the market plus every sector. Return (market_result, sector_table)."""
    results: list[FitResult] = []
    for asset, df in panel.groupby("asset"):
        if len(df) < 250:  # need ~1y of data for a stable fit
            continue
        results.append(fit_asset(df, regime_method, threshold))

    market = next(r for r in results if r.asset == "SPY")
    sectors = [r for r in results if r.asset != "SPY"]

    rows = []
    for r in sorted(sectors, key=lambda x: x.mon_high):
        rows.append(
            {
                "asset": r.asset,
                "sector": r.sector,
                "n": r.n,
                "Mon_low": r.beta_mon,
                "Mon_low_p": r.beta_mon_p,
                "Mon_interact": r.gamma_mon,
                "Mon_interact_p": r.gamma_mon_p,
                "Mon_high": r.mon_high,
                "Mon_high_p": r.mon_high_p,
            }
        )
    return market, pd.DataFrame(rows)


def weekday_table(df: pd.DataFrame, ycol: str = "ret") -> pd.DataFrame:
    """Per-weekday mean of `ycol` with HAC errors (one coefficient per weekday).

    Returns a frame indexed by weekday with columns: coef, se, pvalue, ci_low,
    ci_high. Used by the overnight/intraday gap decomposition.
    """
    df = df.sort_values("date")
    wd = pd.get_dummies(df["weekday"], prefix="", prefix_sep="")
    wd = wd.reindex(columns=WEEKDAYS, fill_value=0).astype(float)
    res = sm.OLS(df[ycol].astype(float).values, wd.values).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}
    )
    out = pd.DataFrame(
        {
            "coef": res.params,
            "se": res.bse,
            "pvalue": res.pvalues,
        },
        index=WEEKDAYS,
    )
    out["ci_low"] = out["coef"] - 1.96 * out["se"]
    out["ci_high"] = out["coef"] + 1.96 * out["se"]
    return out


def fit_unconditional_monday(df: pd.DataFrame) -> tuple[float, float, float]:
    """Plain weekday-dummy model (no VIX). Return (beta_Mon, se, p) with HAC errors.

    Used by the rolling-window decay analysis, where the question is simply
    "is the *unconditional* Monday effect alive in this window?"
    """
    df = df.sort_values("date")
    wd = pd.get_dummies(df["weekday"], prefix="", prefix_sep="")
    wd = wd.reindex(columns=WEEKDAYS, fill_value=0).astype(float)
    res = sm.OLS(df["ret"].astype(float).values, wd.values).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}
    )
    i = WEEKDAYS.index("Mon")
    return float(res.params[i]), float(res.bse[i]), float(res.pvalues[i])


def rolling_monday(
    df: pd.DataFrame, window_years: int = 3, step_days: int = 21
) -> pd.DataFrame:
    """Estimate the unconditional Monday effect in sliding windows over time.

    Returns one row per window with the window-end date, beta_Mon, its HAC
    standard error, p-value, and 95% CI bounds. This is what lets us *date* the
    death of the anomaly: find where the CI first stops excluding zero.
    """
    df = df.sort_values("date").reset_index(drop=True)
    win = pd.Timedelta(days=365 * window_years)
    dates = df["date"]

    rows = []
    end = dates.iloc[0] + win
    last = dates.iloc[-1]
    while end <= last:
        start = end - win
        chunk = df[(dates >= start) & (dates < end)]
        # Require a reasonably full window and both regimes of weekdays present.
        if len(chunk) >= 250 and chunk["weekday"].nunique() == 5:
            b, se, p = fit_unconditional_monday(chunk)
            rows.append(
                {
                    "window_end": end,
                    "beta_mon": b,
                    "se": se,
                    "pvalue": p,
                    "ci_low": b - 1.96 * se,
                    "ci_high": b + 1.96 * se,
                    "n": len(chunk),
                }
            )
        end += pd.Timedelta(days=step_days)
    return pd.DataFrame(rows)


def placebo_test(
    df: pd.DataFrame,
    regime_method: str = "median",
    threshold: float | None = None,
    n_iter: int = 2000,
    seed: int = 7,
) -> dict:
    """Permutation placebo for the Monday-in-high-VIX effect.

    Randomly shuffles the weekday labels (breaking any true day-of-week
    structure) and re-estimates `mon_high` each time. The empirical p-value is
    the share of shuffles whose |mon_high| is at least the observed |mon_high|.
    If the real effect is just a multiple-comparisons artifact, this p-value
    will be large.
    """
    rng = np.random.default_rng(seed)
    df = df.sort_values("date").reset_index(drop=True)

    observed = fit_asset(df, regime_method, threshold).mon_high

    null = np.empty(n_iter)
    shuffled = df.copy()
    for i in range(n_iter):
        shuffled["weekday"] = rng.permutation(df["weekday"].values)
        null[i] = fit_asset(shuffled, regime_method, threshold).mon_high

    emp_p = float(np.mean(np.abs(null) >= abs(observed)))
    return {
        "observed": observed,
        "null": null,  # the actual permutation draws, for an honest histogram
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "empirical_p": emp_p,
        "n_iter": n_iter,
    }
