"""End-to-end pipeline: download data, fit models, run placebo, write figures.

Usage:
    python src/run.py                      # median VIX split, 2015-2025
    python src/run.py --regime threshold   # split at VIX > 20
    python src/run.py --start 2010-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse

import pandas as pd

from analysis import fit_all, placebo_test
from data import build_panel
from plots import plot_placebo, plot_sector_monday, plot_weekday_by_regime

pd.set_option("display.float_format", lambda v: f"{v:8.4f}")
pd.set_option("display.width", 120)


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2025-01-01")
    ap.add_argument("--regime", choices=["median", "threshold"], default="median")
    ap.add_argument("--threshold", type=float, default=20.0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--placebo-iter", type=int, default=2000)
    args = ap.parse_args()

    print(f"\nDownloading panel {args.start} .. {args.end} ...")
    panel = build_panel(args.start, args.end, use_cache=not args.no_cache)
    print(f"  rows={len(panel):,}  assets={panel['asset'].nunique()}  "
          f"dates={panel['date'].nunique():,}")

    market, sectors = fit_all(panel, args.regime, args.threshold)

    print("\n" + "=" * 72)
    print(f"MARKET (SPY) -- regime split: {args.regime}"
          + (f" (>{args.threshold})" if args.regime == "threshold" else ""))
    print("=" * 72)
    print(f"  n = {market.n:,}   high-VIX days = {market.n_high:,} "
          f"({market.n_high / market.n:.0%})")
    print(f"  Monday, LOW  VIX : {market.beta_mon:+.4f}%  "
          f"(p={market.beta_mon_p:.3f}) {stars(market.beta_mon_p)}")
    print(f"  Monday interact  : {market.gamma_mon:+.4f}%  "
          f"(p={market.gamma_mon_p:.3f}) {stars(market.gamma_mon_p)}")
    print(f"  Monday, HIGH VIX : {market.mon_high:+.4f}%  "
          f"(p={market.mon_high_p:.3f}) {stars(market.mon_high_p)}")

    print("\n" + "-" * 72)
    print("SECTORS -- sorted by high-VIX Monday return (most negative first)")
    print("-" * 72)
    show = sectors.copy()
    show["sig"] = show["Mon_high_p"].map(stars)
    print(show[["sector", "Mon_low", "Mon_low_p", "Mon_interact",
                "Mon_interact_p", "Mon_high", "Mon_high_p", "sig"]]
          .to_string(index=False))

    print("\n" + "-" * 72)
    print("ROBUSTNESS -- SPY Monday effect under two regime definitions")
    print("-" * 72)
    m_med, _ = fit_all(panel, "median")
    m_thr, _ = fit_all(panel, "threshold", args.threshold)
    cmp = pd.DataFrame(
        [
            {"regime": "median split",
             "high_share": m_med.n_high / m_med.n,
             "Mon_low": m_med.beta_mon, "Mon_low_p": m_med.beta_mon_p,
             "Mon_interact": m_med.gamma_mon, "Mon_interact_p": m_med.gamma_mon_p,
             "Mon_high": m_med.mon_high, "Mon_high_p": m_med.mon_high_p},
            {"regime": f"VIX > {args.threshold:g}",
             "high_share": m_thr.n_high / m_thr.n,
             "Mon_low": m_thr.beta_mon, "Mon_low_p": m_thr.beta_mon_p,
             "Mon_interact": m_thr.gamma_mon, "Mon_interact_p": m_thr.gamma_mon_p,
             "Mon_high": m_thr.mon_high, "Mon_high_p": m_thr.mon_high_p},
        ]
    )
    print(cmp.to_string(index=False))

    print("\n" + "-" * 72)
    print(f"PLACEBO -- permuting weekday labels for SPY ({args.placebo_iter} draws)")
    print("-" * 72)
    spy = panel[panel["asset"] == "SPY"]
    pb = placebo_test(spy, args.regime, args.threshold, n_iter=args.placebo_iter)
    print(f"  observed Monday high-VIX  : {pb['observed']:+.4f}%")
    print(f"  null mean / std           : {pb['null_mean']:+.4f}% / {pb['null_std']:.4f}")
    print(f"  empirical p-value         : {pb['empirical_p']:.4f}")

    f1 = plot_weekday_by_regime(market)
    f2 = plot_sector_monday(sectors)
    f3 = plot_placebo(pb)
    print(f"\nFigures written:\n  {f1}\n  {f2}\n  {f3}\n")


if __name__ == "__main__":
    main()
