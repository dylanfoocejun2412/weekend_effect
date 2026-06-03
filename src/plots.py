"""Figures for the conditional weekend-effect study."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import WEEKDAYS, FitResult

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def _save(fig, name: str) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_weekday_by_regime(market: FitResult) -> str:
    """Bar chart of average return per weekday, low- vs high-VIX regime, for SPY."""
    low = [market.params[d] for d in WEEKDAYS]
    high = [market.high_coef[d] for d in WEEKDAYS]
    low_err = [1.96 * market.bse[d] for d in WEEKDAYS]
    high_err = [1.96 * market.high_se[d] for d in WEEKDAYS]

    x = np.arange(len(WEEKDAYS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, low, w, yerr=low_err, capsize=3, label="Low VIX", color="#4C78A8")
    ax.bar(x + w / 2, high, w, yerr=high_err, capsize=3, label="High VIX", color="#E45756")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(WEEKDAYS)
    ax.set_ylabel("Mean daily return (%)")
    ax.set_title("SPY day-of-week return by volatility regime\n(95% CI; VIX split at in-sample median)")
    ax.legend()
    return _save(fig, "spy_weekday_by_regime.png")


def plot_sector_monday(sectors: pd.DataFrame) -> str:
    """Horizontal bars of the high-VIX Monday return across sectors."""
    df = sectors.sort_values("Mon_high")
    colors = ["#E45756" if p < 0.05 else "#BAB0AC" for p in df["Mon_high_p"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["sector"], df["Mon_high"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Mean Monday return in high-VIX regime (%)")
    ax.set_title("Conditional Monday effect by sector\n(red = significant at 5% with HAC errors)")
    return _save(fig, "sector_monday_high_vix.png")


def plot_rolling_monday(roll: pd.DataFrame, window_years: int = 3) -> str:
    """Time series of the rolling unconditional Monday coefficient with 95% CI.

    Windows whose entire 95% CI is below zero (a live negative weekend effect)
    are marked in red; this dates when the anomaly was last significant. If no
    window qualifies, a caption says so rather than leaving an unused legend key.
    """
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhline(0, color="black", lw=0.8)
    ax.fill_between(roll["window_end"], roll["ci_low"], roll["ci_high"],
                    color="#4C78A8", alpha=0.20, label="95% CI")
    ax.plot(roll["window_end"], roll["beta_mon"], color="#4C78A8", lw=1.6,
            label="Monday coefficient")

    alive = roll["ci_high"] < 0  # significantly negative
    # Only draw the red marker (and its legend entry) when such windows exist;
    # otherwise the legend would advertise a point that never appears.
    if alive.any():
        ax.scatter(roll.loc[alive, "window_end"], roll.loc[alive, "beta_mon"],
                   color="#E45756", s=14, zorder=5,
                   label="significantly negative (effect alive)")
        subtitle = "red points = window where the classic negative weekend effect is significant"
    else:
        ax.text(0.02, 0.04,
                "no window is significantly negative: the classic weekend\n"
                "effect is never alive in SPY's sample (CI always includes 0)",
                transform=ax.transAxes, fontsize=8, color="#E45756",
                va="bottom", ha="left")
        subtitle = "no window is significantly negative (effect never alive in this sample)"

    ax.set_ylabel("Mean Monday return (%)")
    ax.set_title(f"Rolling {window_years}-year unconditional Monday effect (SPY, HAC errors)\n"
                 + subtitle)
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, "rolling_monday_decay.png")


def plot_gap_decomposition(overnight: pd.DataFrame, intraday: pd.DataFrame) -> str:
    """Grouped bars: overnight vs intraday return for each weekday, plus total.

    `overnight` and `intraday` are weekday_table outputs (indexed by weekday).
    The weekend gap (Fri-close -> Mon-open) is the overnight component on Monday.
    Grouped (not stacked) so days where the two components have opposite signs,
    such as Monday and Tuesday, read correctly. The total is marked separately
    rather than implied by a bar height, which a stacked chart would distort.
    """
    on = overnight.reindex(WEEKDAYS)["coef"]
    on_se = overnight.reindex(WEEKDAYS)["se"]
    on_p = overnight.reindex(WEEKDAYS)["pvalue"]
    ind = intraday.reindex(WEEKDAYS)["coef"]
    ind_se = intraday.reindex(WEEKDAYS)["se"]
    total = on.values + ind.values

    x = np.arange(len(WEEKDAYS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - w / 2, on, w, yerr=1.96 * on_se, capsize=3,
           color="#4C78A8", label="overnight (incl. weekend gap)")
    ax.bar(x + w / 2, ind, w, yerr=1.96 * ind_se, capsize=3,
           color="#F58518", label="intraday")
    ax.scatter(x, total, color="black", zorder=5, marker="D", s=28,
               label="total (overnight + intraday)")
    ax.axhline(0, color="black", lw=0.8)

    for xi, (o, p) in enumerate(zip(on, on_p)):
        if p < 0.05:
            ax.annotate("significant", (xi - w / 2, o), ha="center", va="bottom",
                        fontsize=8, color="#E45756")

    ax.set_xticks(x)
    ax.set_xticklabels(WEEKDAYS)
    ax.set_ylabel("Mean return (%)")
    ax.set_title("Decomposing day-of-week returns: overnight vs intraday (SPY)\n"
                 "95% CI bars; Monday overnight is the Friday-to-Monday weekend gap")
    ax.legend(fontsize=8)
    return _save(fig, "gap_decomposition.png")


def plot_placebo(placebo: dict) -> str:
    """Histogram of the actual permutation null vs the observed effect."""
    null = np.asarray(placebo["null"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(null, bins=60, color="#BAB0AC", alpha=0.85,
            label=f"{len(null)} shuffled-label estimates")
    ax.axvline(placebo["observed"], color="#E45756", lw=2,
               label=f"observed = {placebo['observed']:.3f}%")
    ax.set_xlabel("Monday high-VIX return under shuffled weekday labels (%)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Permutation placebo (empirical p = {placebo['empirical_p']:.3f})")
    ax.legend()
    return _save(fig, "placebo_null.png")
