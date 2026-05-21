"""
baseline_change_report.py
=========================
EpiHack — Symptom × Area Baseline Change Report: Reproducibility Script
------------------------------------------------------------------------
Replicates the analytical pipeline of baseline_change_report.html using
Python, pandas, and matplotlib.

Epidemiological Design
----------------------
For each of the 14 individual symptom flags in fakeData.csv, this script
computes the prevalence rate (YES / valid responses) in two non-overlapping
temporal windows and quantifies the relative change:

  Baseline window : the 30 calendar days immediately preceding the recent window.
  Recent window   : the last N calendar days in the dataset (default N=7).
  % Change        : (recent_rate − baseline_rate) / baseline_rate × 100

UNKNOWN responses are excluded from both numerator and denominator,
preserving epidemiological validity under partial response.

Four surveillance figures are produced:
  Fig 1 — Diverging horizontal bar: % change from baseline per symptom
  Fig 2 — Grouped horizontal bar: baseline vs recent prevalence rates
  Fig 3 — Daily symptom burden trend with window annotations
  Fig 4 — Postal Code × Symptom heatmap (recent window only)

Usage
-----
    python baseline_change_report.py [--csv PATH] [--n DAYS] [--postal CODE]
                                     [--out DIR]

Arguments
---------
  --csv     Path to fakeData.csv          (default: fakeData.csv)
  --n       Recent window in days         (default: 7)
  --postal  Filter to one postal code     (default: ALL — aggregate all)
  --out     Output directory for figures  (default: figures_baseline/)

Requirements
------------
    pip install pandas matplotlib seaborn numpy

Author  : EpiHack Project
Date    : 2025-07-12
Version : 1.0.0
"""

# ──────────────────────────────────────────────────────────────────────────────
import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.patches as mpatches
    import numpy as np
    import seaborn as sns
except ImportError as e:
    sys.exit(f"[ERROR] Missing dependency: {e}\n"
             "Install with:  pip install pandas matplotlib seaborn numpy")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SYMPTOM_COLS = [
    "Cough/Congestion",
    "Nauseas/Vomiting",
    "Difficulty Breathing",
    "Sore Throat",
    "Rash",
    "Fever",
    "Chills",
    "Diarrhea",
    "Bleeding from Body Openings",
    "Red Eyes",
    "Muscle or Body Aches and Pains",
    "Discolored or Bloody Urine",
    "Loss of Smell or Taste",
    "Yellow Skin/Yellow Eyes",
]

SHORT_NAMES = {
    "Cough/Congestion":              "Cough/Cong.",
    "Nauseas/Vomiting":              "Nausea/Vomit.",
    "Difficulty Breathing":          "Diff. Breathing",
    "Sore Throat":                   "Sore Throat",
    "Rash":                          "Rash",
    "Fever":                         "Fever",
    "Chills":                        "Chills",
    "Diarrhea":                      "Diarrhea",
    "Bleeding from Body Openings":   "Bleeding",
    "Red Eyes":                      "Red Eyes",
    "Muscle or Body Aches and Pains":"Body Aches",
    "Discolored or Bloody Urine":    "Discolored Urine",
    "Loss of Smell or Taste":        "Loss Smell/Taste",
    "Yellow Skin/Yellow Eyes":       "Yellow Skin/Eyes",
}

# Dark-mode palette
CLR_BG     = "#0f1117"
CLR_SFC    = "#1a1d27"
CLR_BORDER = "#2e3148"
CLR_TEXT   = "#e8eaf6"
CLR_MUTED  = "#8b8fa8"
CLR_UP     = "#ff6584"    # increasing
CLR_DN     = "#43e97b"    # decreasing
CLR_NC     = "#8b8fa8"    # no change / stable
CLR_NEW    = "#fd9644"    # newly reported
CLR_BAS    = "#6c63ff"    # baseline series
CLR_REC    = "#ff6584"    # recent series


# ──────────────────────────────────────────────────────────────────────────────
# I.  DATA INGESTION
# ──────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Ingest fakeData.csv and prepare a tidy DataFrame.

    Each symptom column is coerced to: 1 (YES), 0 (NO), NaN (UNKNOWN).
    This encoding allows pandas to ignore UNKNOWN values in mean() and
    count() operations by default (skipna=True).

    Parameters
    ----------
    csv_path : str

    Returns
    -------
    pd.DataFrame with columns: date_report, postal_code, *SYMPTOM_COLS
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # Parse date
    df["date_report"] = pd.to_datetime(df["Date of Report"], format="%m/%d/%y")
    df["postal_code"] = pd.to_numeric(df["Postal Code"], errors="coerce").astype("Int64")

    # Coerce symptom flags: YES→1, NO→0, UNKNOWN→NaN
    for col in SYMPTOM_COLS:
        df[col] = df[col].map({"YES": 1, "NO": 0, "UNKNOWN": None}).astype("Float64")

    keep = ["date_report", "postal_code"] + SYMPTOM_COLS
    df   = df[keep].copy()

    print(f"[INFO] Loaded {len(df)} records from '{csv_path}'.")
    print(f"[INFO] Date range: {df['date_report'].min().date()} → "
          f"{df['date_report'].max().date()}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# II.  WINDOW COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_windows(df: pd.DataFrame, n_days: int, postal: str
                    ) -> tuple[pd.DataFrame, pd.DataFrame,
                               pd.Timestamp, pd.Timestamp,
                               pd.Timestamp, pd.Timestamp]:
    """
    Partition records into baseline (30 days) and recent (N days) windows.

    Temporal layout (anchored to dataset maximum date):
        [baseline_start ← 30 days → baseline_end] [recent_start ← N → max_date]

    UNKNOWN-exclusion: each symptom is independently excluded per record
    (handled downstream by skipna=True in aggregation).

    Parameters
    ----------
    df     : full DataFrame
    n_days : length of recent observation window
    postal : postal code string or 'ALL'

    Returns
    -------
    (df_recent, df_baseline, rec_start, rec_end, bas_start, bas_end)
    """
    max_date  = df["date_report"].max()
    rec_end   = max_date
    rec_start = rec_end   - pd.Timedelta(days=n_days - 1)
    bas_end   = rec_start - pd.Timedelta(days=1)
    bas_start = bas_end   - pd.Timedelta(days=29)

    if postal != "ALL":
        df = df[df["postal_code"] == int(postal)]

    df_rec = df[(df["date_report"] >= rec_start) & (df["date_report"] <= rec_end)].copy()
    df_bas = df[(df["date_report"] >= bas_start) & (df["date_report"] <= bas_end)].copy()

    print(f"\n[INFO] N={n_days}  |  Postal filter: {postal}")
    print(f"       Recent  : {rec_start.date()} → {rec_end.date()}  ({len(df_rec)} records)")
    print(f"       Baseline: {bas_start.date()} → {bas_end.date()}  ({len(df_bas)} records)")

    return df_rec, df_bas, rec_start, rec_end, bas_start, bas_end


# ──────────────────────────────────────────────────────────────────────────────
# III.  SYMPTOM PREVALENCE & CHANGE COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_stats(df_rec: pd.DataFrame,
                  df_bas: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-symptom prevalence rates and % change from baseline.

    Prevalence rate = mean(symptom_col, skipna=True) = YES / (YES + NO)
    UNKNOWN values are excluded from both numerator and denominator.

    % Change classification:
        'up'    : positive % change > 5%
        'dn'    : negative % change < -5%
        'nc'    : |% change| ≤ 5% (stable)
        'new'   : baseline rate = 0, recent rate > 0  (emerging signal)
        'indef' : insufficient data in one or both windows

    Parameters
    ----------
    df_rec : recent window DataFrame
    df_bas : baseline window DataFrame

    Returns
    -------
    pd.DataFrame with one row per symptom.
    """
    rows = []
    for col in SYMPTOM_COLS:
        bas_valid = df_bas[col].dropna()
        rec_valid = df_rec[col].dropna()

        bas_rate = float(bas_valid.mean()) if len(bas_valid) > 0 else None
        rec_rate = float(rec_valid.mean()) if len(rec_valid) > 0 else None
        bas_n    = int(len(bas_valid))
        rec_n    = int(len(rec_valid))

        if bas_rate is None or rec_rate is None:
            pct_chg = None
            signal  = "indef"
        elif bas_rate == 0 and rec_rate == 0:
            pct_chg = 0.0
            signal  = "nc"
        elif bas_rate == 0 and rec_rate > 0:
            pct_chg = None
            signal  = "new"
        else:
            pct_chg = (rec_rate - bas_rate) / bas_rate * 100
            signal  = "up" if pct_chg > 5 else "dn" if pct_chg < -5 else "nc"

        rows.append({
            "symptom":    col,
            "short":      SHORT_NAMES[col],
            "bas_rate":   bas_rate,
            "bas_n":      bas_n,
            "rec_rate":   rec_rate,
            "rec_n":      rec_n,
            "pct_change": pct_chg,
            "signal":     signal,
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# IV.  CONSOLE REPORT
# ──────────────────────────────────────────────────────────────────────────────

def print_report(stats: pd.DataFrame, n_days: int,
                 rec_start, rec_end, bas_start, bas_end) -> None:
    """Print a structured surveillance summary to stdout."""
    sep = "─" * 80
    print(f"\n{sep}")
    print(f"  EpiHack — Symptom Baseline Change Report  |  N={n_days} days")
    print(sep)
    print(f"  Recent window   : {rec_start.date()} → {rec_end.date()}")
    print(f"  Baseline window : {bas_start.date()} → {bas_end.date()}")
    print(sep)
    hdr = f"  {'Symptom':<32} {'Bas%':>6}  {'n':>4}  {'Rec%':>6}  {'n':>4}  {'Δ%':>8}  {'Signal':<12}"
    print(hdr)
    print(f"  {'─'*32} {'─'*6}  {'─'*4}  {'─'*6}  {'─'*4}  {'─'*8}  {'─'*12}")
    df_sorted = stats.sort_values("pct_change", ascending=False, na_position="last")
    for _, r in df_sorted.iterrows():
        bas_s = f"{r.bas_rate*100:.1f}%" if r.bas_rate is not None else "—"
        rec_s = f"{r.rec_rate*100:.1f}%" if r.rec_rate is not None else "—"
        pc = r["pct_change"]
        pct_s = (f"+{pc:.1f}%" if pc is not None and pc > 0
                 else (f"{pc:.1f}%" if pc is not None
                       else ("NEW" if r["signal"] == "new" else "—")))
        sig_s = {"up":"↑ Increasing","dn":"↓ Decreasing","nc":"→ Stable",
                 "new":"🆕 Emerging","indef":"— No data"}.get(r.signal,"—")
        print(f"  {r.symptom:<32} {bas_s:>6}  {r.bas_n:>4}  {rec_s:>6}  "
              f"{r.rec_n:>4}  {pct_s:>8}  {sig_s}")
    print(f"\n{sep}\n")


# ──────────────────────────────────────────────────────────────────────────────
# V.  FIGURE FACTORY
# ──────────────────────────────────────────────────────────────────────────────

def _dark_ax(ax):
    """Apply dark-mode theme to a matplotlib Axes."""
    ax.set_facecolor(CLR_SFC)
    for spine in ax.spines.values():
        spine.set_edgecolor(CLR_BORDER)
    ax.tick_params(colors=CLR_MUTED, labelsize=9)
    ax.xaxis.label.set_color(CLR_MUTED)
    ax.yaxis.label.set_color(CLR_MUTED)
    ax.title.set_color(CLR_TEXT)
    ax.grid(color=CLR_BORDER, linewidth=0.6)
    ax.set_axisbelow(True)


def fig1_diverging(stats: pd.DataFrame, n_days: int, out_dir: Path) -> Path:
    """
    Figure 1: Horizontal diverging bar chart of % change from baseline.

    Bars are colour-coded by epidemiological signal:
        Red    → increasing (> +5%)
        Green  → decreasing (< -5%)
        Grey   → stable (within ±5%)
        Orange → newly emerging (0% baseline → >0% recent)

    Parameters
    ----------
    stats   : output of compute_stats()
    n_days  : lookback window for title
    out_dir : save directory

    Returns
    -------
    Path to saved PNG.
    """
    df = stats.sort_values("pct_change", ascending=True, na_position="first").copy()
    df["pct_plot"] = df["pct_change"].fillna(0)   # place NEW/indef at 0 for bar

    color_map = {"up": CLR_UP, "dn": CLR_DN, "nc": CLR_NC,
                 "new": CLR_NEW, "indef": "#4a4f6a"}
    colors = [color_map[s] for s in df["signal"]]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=CLR_BG)
    _dark_ax(ax)

    bars = ax.barh(df["short"], df["pct_plot"], color=colors,
                   edgecolor="none", height=0.68)

    # Annotate bars
    for bar, (_, row) in zip(bars, df.iterrows()):
        w   = bar.get_width()
        pc  = row["pct_change"]
        sig = row["signal"]
        if sig == "new":
            label = "NEW"
        elif pc is None or (isinstance(pc, float) and np.isnan(pc)):
            label = "—"
        else:
            label = f"{float(pc):+.1f}%"
        ax.text(w + (0.5 if w >= 0 else -0.5), bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left" if w >= 0 else "right",
                fontsize=8, color=CLR_TEXT)

    ax.axvline(0, color=CLR_MUTED, linewidth=1)
    ax.set_xlabel("% Change from 30-Day Baseline", color=CLR_MUTED)
    ax.set_title(f"Symptom % Change from 30-Day Baseline  (N={n_days}-day Recent Window)",
                 fontsize=11, color=CLR_TEXT, pad=10)
    ax.tick_params(axis="y", colors=CLR_TEXT)

    # Legend patches
    legend_items = [
        mpatches.Patch(color=CLR_UP,  label="↑ Increasing (>+5%)"),
        mpatches.Patch(color=CLR_DN,  label="↓ Decreasing (<-5%)"),
        mpatches.Patch(color=CLR_NC,  label="→ Stable (±5%)"),
        mpatches.Patch(color=CLR_NEW, label="🆕 Newly reported"),
    ]
    ax.legend(handles=legend_items, facecolor=CLR_SFC, edgecolor=CLR_BORDER,
              labelcolor=CLR_TEXT, fontsize=8.5, loc="lower right")

    fig.tight_layout()
    out = out_dir / f"fig1_diverging_N{n_days}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=CLR_BG)
    plt.close(fig)
    print(f"[SAVED] {out}")
    return out


def fig2_comparison(stats: pd.DataFrame, n_days: int, out_dir: Path) -> Path:
    """
    Figure 2: Grouped horizontal bar chart — baseline rate vs recent rate
    for every symptom, sorted by baseline rate descending.

    Parameters
    ----------
    stats   : output of compute_stats()
    n_days  : lookback window for title
    out_dir : save directory

    Returns
    -------
    Path to saved PNG.
    """
    df = stats.copy()
    df["bas_pct"] = df["bas_rate"].apply(lambda v: v * 100 if v is not None else 0)
    df["rec_pct"] = df["rec_rate"].apply(lambda v: v * 100 if v is not None else 0)
    df = df.sort_values("bas_pct", ascending=True)

    y    = np.arange(len(df))
    h    = 0.35
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=CLR_BG)
    _dark_ax(ax)

    ax.barh(y - h/2, df["bas_pct"], h, color=CLR_BAS, alpha=0.85, label="Baseline Rate (%)")
    ax.barh(y + h/2, df["rec_pct"], h, color=CLR_REC, alpha=0.85, label="Recent Rate (%)")

    ax.set_yticks(y)
    ax.set_yticklabels(df["short"].tolist(), fontsize=9, color=CLR_TEXT)
    ax.set_xlabel("Prevalence Rate (%)", color=CLR_MUTED)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.set_title(f"Symptom Prevalence: Baseline (30d) vs Recent (N={n_days}d)",
                 fontsize=11, color=CLR_TEXT, pad=10)
    ax.legend(facecolor=CLR_SFC, edgecolor=CLR_BORDER,
              labelcolor=CLR_TEXT, fontsize=9)

    fig.tight_layout()
    out = out_dir / f"fig2_comparison_N{n_days}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=CLR_BG)
    plt.close(fig)
    print(f"[SAVED] {out}")
    return out


def fig3_trend(df_full: pd.DataFrame,
               rec_start: pd.Timestamp, rec_end: pd.Timestamp,
               bas_start: pd.Timestamp, bas_end: pd.Timestamp,
               n_days: int, out_dir: Path) -> Path:
    """
    Figure 3: Daily aggregate symptom burden over the full dataset,
    with vertical lines marking the baseline and recent windows.

    Symptom burden = total symptom-YES count across all 14 columns per day.
    UNKNOWN values are excluded (treated as 0 in sum, after filling NaN→0
    only for burden counting, not for rate computation elsewhere).

    Parameters
    ----------
    df_full    : full DataFrame (pre-filtered by postal if applicable)
    rec_start  : start of recent window
    rec_end    : end of recent window
    bas_start  : start of baseline window
    bas_end    : end of baseline window
    n_days     : lookback window (for title)
    out_dir    : save directory

    Returns
    -------
    Path to saved PNG.
    """
    daily = (
        df_full.groupby("date_report")[SYMPTOM_COLS]
        .apply(lambda g: g.fillna(0).values.sum())
        .reset_index(name="burden")
    )

    fig, ax = plt.subplots(figsize=(13, 4), facecolor=CLR_BG)
    _dark_ax(ax)

    ax.fill_between(daily["date_report"], daily["burden"],
                    color=CLR_BAS, alpha=0.15)
    ax.plot(daily["date_report"], daily["burden"],
            color=CLR_BAS, linewidth=2, marker="o", markersize=3)

    # Shade baseline window
    ax.axvspan(bas_start, bas_end, color=CLR_BAS, alpha=0.07,
               label=f"Baseline window (30d)")
    # Shade recent window
    ax.axvspan(rec_start, rec_end, color=CLR_REC, alpha=0.10,
               label=f"Recent window (N={n_days}d)")

    ax.axvline(bas_start, color=CLR_BAS, linewidth=1, linestyle="--")
    ax.axvline(rec_start, color=CLR_REC, linewidth=1, linestyle="--")

    ax.set_ylabel("Daily Symptom Reports", color=CLR_MUTED)
    ax.set_title("Daily Aggregate Symptom Burden — Full Dataset",
                 fontsize=11, color=CLR_TEXT, pad=10)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             fontsize=8, color=CLR_TEXT)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(facecolor=CLR_SFC, edgecolor=CLR_BORDER,
              labelcolor=CLR_TEXT, fontsize=9)

    fig.tight_layout()
    out = out_dir / f"fig3_trend_N{n_days}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=CLR_BG)
    plt.close(fig)
    print(f"[SAVED] {out}")
    return out


def fig4_heatmap(df_rec: pd.DataFrame, n_days: int, out_dir: Path) -> Path:
    """
    Figure 4: Postal Code × Symptom heatmap for the recent window.

    Cell colour encodes the symptom value (1=YES, 0=NO, NaN=UNKNOWN).
    Only records present in the recent window are shown.

    Parameters
    ----------
    df_rec  : recent-window DataFrame
    n_days  : lookback window (for filename)
    out_dir : save directory

    Returns
    -------
    Path to saved PNG, or None if recent window is empty.
    """
    if df_rec.empty:
        print("[WARN] Fig 4 skipped — no records in recent window.")
        return None

    mat = df_rec.set_index("postal_code")[SYMPTOM_COLS].sort_index()
    mat.columns = [SHORT_NAMES[c] for c in mat.columns]
    # Encode: 1=YES (red), 0=NO (green), NaN=UNKNOWN (yellow)
    mat_num = mat.astype(float)    # NaN preserved

    fig, ax = plt.subplots(
        figsize=(max(10, len(mat.columns) * 0.75), max(4, len(mat) * 0.55)),
        facecolor=CLR_BG
    )
    ax.set_facecolor(CLR_SFC)

    # Custom 3-colour discrete palette: 0→green, 0.5→yellow, 1→red
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "epihack", [(0, "#43e97b"), (0.5, "#f9ca24"), (1, "#ff6584")]
    )

    sns.heatmap(
        mat_num, ax=ax,
        cmap=cmap, vmin=0, vmax=1,
        annot=True, fmt=".0f", annot_kws={"size": 9, "color": CLR_BG},
        linewidths=0.5, linecolor=CLR_BORDER,
        cbar_kws={"label": "0=No  /  1=Yes  /  NaN=Unknown", "shrink": 0.6}
    )
    ax.set_title(f"Postal Code × Symptom Matrix — Recent Window (N={n_days}d)",
                 fontsize=10, color=CLR_TEXT, pad=10)
    ax.set_xlabel("Symptom", color=CLR_MUTED)
    ax.set_ylabel("Postal Code", color=CLR_MUTED)
    ax.tick_params(axis="x", colors=CLR_TEXT, labelsize=8, rotation=30)
    ax.tick_params(axis="y", colors=CLR_TEXT, labelsize=9)

    cbar = ax.collections[0].colorbar
    cbar.set_label("0 = No  |  1 = Yes  |  NaN = Unknown", color=CLR_MUTED)
    cbar.ax.yaxis.set_tick_params(color=CLR_MUTED)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=CLR_MUTED)

    fig.tight_layout()
    out = out_dir / f"fig4_heatmap_N{n_days}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=CLR_BG)
    plt.close(fig)
    print(f"[SAVED] {out}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# VI.  CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EpiHack — Symptom Baseline Change Report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--csv",    default="fakeData.csv")
    parser.add_argument("--n",      type=int, default=7,
                        help="Recent observation window (days)")
    parser.add_argument("--postal", default="ALL",
                        help="Filter to a single postal code, or 'ALL'")
    parser.add_argument("--out",    default="figures_baseline",
                        help="Output directory for figures")
    return parser.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 — Ingest
    df = load_data(args.csv)

    # Step 2 — Partition windows
    df_rec, df_bas, rec_start, rec_end, bas_start, bas_end = \
        compute_windows(df, args.n, args.postal)

    # Step 3 — Compute per-symptom stats
    stats = compute_stats(df_rec, df_bas)

    # Step 4 — Console report
    print_report(stats, args.n, rec_start, rec_end, bas_start, bas_end)

    # Step 5 — Figures
    # For trend, use the full dataset filtered only by postal
    df_postal = df if args.postal == "ALL" else df[df["postal_code"] == int(args.postal)]
    fig1_diverging(stats,             args.n, out_dir)
    fig2_comparison(stats,            args.n, out_dir)
    fig3_trend(df_postal, rec_start, rec_end, bas_start, bas_end, args.n, out_dir)
    fig4_heatmap(df_rec,              args.n, out_dir)

    print(f"[DONE] All outputs saved to '{out_dir.resolve()}'.\n")


if __name__ == "__main__":
    main()
