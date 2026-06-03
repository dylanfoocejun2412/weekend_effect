"""Date the death of the weekend effect.

Pulls SPY back to 1993 (its inception) and estimates the *unconditional* Monday
return in rolling multi-year windows, so we can see when the classic negative
effect stopped being statistically significant.

Usage:
    python src/history.py                       # 3y rolling window, SPY since 1993
    python src/history.py --window 5            # 5y windows
    python src/history.py --start 1993-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from analysis import rolling_monday
from data import build_panel
from plots import plot_rolling_monday

HIST_CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "panel_history.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1993-01-01")
    ap.add_argument("--end", default="2025-01-01")
    ap.add_argument("--window", type=int, default=3, help="rolling window in years")
    ap.add_argument("--step", type=int, default=21, help="step between windows in trading days")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    print(f"\nDownloading SPY history {args.start} .. {args.end} ...")
    panel = build_panel(args.start, args.end, use_cache=not args.no_cache,
                        cache_path=HIST_CACHE)
    spy = panel[panel["asset"] == "SPY"].copy()
    print(f"  SPY days = {len(spy):,}  "
          f"range = {spy['date'].min().date()} .. {spy['date'].max().date()}")

    roll = rolling_monday(spy, window_years=args.window, step_days=args.step)

    alive = roll[roll["ci_high"] < 0]  # windows with a significant negative Monday
    print("\n" + "=" * 72)
    print(f"ROLLING {args.window}-YEAR UNCONDITIONAL MONDAY EFFECT (SPY)")
    print("=" * 72)
    print(f"  windows estimated            : {len(roll)}")
    print(f"  windows with sig. neg. Monday: {len(alive)} "
          f"({len(alive) / max(len(roll), 1):.0%})")
    if not alive.empty:
        first = alive["window_end"].min().date()
        last = alive["window_end"].max().date()
        print(f"  effect significant from windows ending {first} .. {last}")
        print(f"  -> last window where the classic effect was alive ends ~{last}")
    else:
        print("  the unconditional Monday effect is never significant in this sample")

    print("\n  selected windows (every ~2 years):")
    show = roll.iloc[::24][["window_end", "beta_mon", "ci_low", "ci_high", "pvalue"]]
    show = show.assign(window_end=show["window_end"].dt.date)
    print(show.to_string(index=False))

    path = plot_rolling_monday(roll, window_years=args.window)
    print(f"\nFigure written:\n  {path}\n")


if __name__ == "__main__":
    main()
