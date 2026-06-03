"""Overnight vs intraday decomposition of the day-of-week return.

The classic weekend-effect story is about *news accumulating over the weekend* —
so any Monday effect should live in the **weekend gap** (Friday close -> Monday
open), not in Monday's intraday session. This script splits each weekday's mean
return into its overnight and intraday components and tests them separately with
HAC errors, with special attention to Monday's overnight = the weekend gap.

Usage:
    python src/gap.py                 # SPY, 2015-2025
    python src/gap.py --start 2010-01-01
"""

from __future__ import annotations

import argparse

import pandas as pd

from analysis import weekday_table
from data import build_gap_panel
from plots import plot_gap_decomposition

pd.set_option("display.float_format", lambda v: f"{v:8.4f}")
pd.set_option("display.width", 120)


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2025-01-01")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    print(f"\nDownloading OHLC gap panel {args.start} .. {args.end} ...")
    panel = build_gap_panel(args.start, args.end, use_cache=not args.no_cache)
    spy = panel[panel["asset"] == "SPY"].copy()
    print(f"  SPY days = {len(spy):,}  "
          f"range = {spy['date'].min().date()} .. {spy['date'].max().date()}")

    overnight = weekday_table(spy, "overnight")
    intraday = weekday_table(spy, "intraday")
    total = weekday_table(spy, "total")

    table = pd.DataFrame(
        {
            "overnight": overnight["coef"],
            "overnight_p": overnight["pvalue"],
            "intraday": intraday["coef"],
            "intraday_p": intraday["pvalue"],
            "total": total["coef"],
            "total_p": total["pvalue"],
        }
    )

    print("\n" + "=" * 72)
    print("SPY DAY-OF-WEEK RETURN DECOMPOSITION (overnight + intraday = total)")
    print("=" * 72)
    print(table.to_string())

    mo = overnight.loc["Mon"]
    mi = intraday.loc["Mon"]
    print("\n" + "-" * 72)
    print("THE WEEKEND GAP (Friday close -> Monday open)")
    print("-" * 72)
    print(f"  Monday overnight (weekend gap): {mo['coef']:+.4f}%  "
          f"(p={mo['pvalue']:.3f}) {stars(mo['pvalue'])}")
    print(f"  Monday intraday               : {mi['coef']:+.4f}%  "
          f"(p={mi['pvalue']:.3f}) {stars(mi['pvalue'])}")
    verdict = ("the weekend gap, not intraday trading"
               if mo["pvalue"] < mi["pvalue"] else "Monday intraday, not the weekend gap")
    print(f"  -> Monday's signal lives mostly in: {verdict}")

    path = plot_gap_decomposition(overnight, intraday)
    print(f"\nFigure written:\n  {path}\n")


if __name__ == "__main__":
    main()
