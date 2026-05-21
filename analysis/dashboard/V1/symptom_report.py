"""
symptom_report.py
=================
EpiHack — Symptom Surveillance Report: Reproducibility Script
-------------------------------------------------------------
Replicates the full analytical pipeline underpinning the interactive
dashboard (symptom_report.html), using Python/pandas and matplotlib.

Purpose
-------
Given a cohort CSV (fakeData.csv), this script:
  1. Ingests and validates the raw data.
  2. Filters records to a configurable lookback window of N days,
     anchored to the latest Date of Report present in the dataset.
  3. Groups records by demographic strata: Age Bin × Sex × Postal Area.
  4. Computes, for each stratum, the percentage of individuals
     reporting Symptoms vs. No Symptoms vs. Unknown/Pending.
  5. Produces four publication-ready figures:
       Fig 1 — Grouped bar chart: % Symptomatic vs % No Symptoms by stratum
       Fig 2 — Donut chart: global symptom status distribution in window
       Fig 3 — Daily case count time series within the window
       Fig 4 — Heatmap: symptom rate by Age Bin × Sex

Usage
-----
    python symptom_report.py [--csv PATH] [--n DAYS] [--out DIR]

Default values:
    --csv  fakeData.csv   (same directory as this script)
    --n    7              (lookback window in days)
    --out  ./figures/     (output directory for saved figures)

Requirements
------------
    pip install pandas matplotlib seaborn

Author  : EpiHack Project
Date    : 2025-07-12
Version : 1.0.0
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard library
# ──────────────────────────────────────────────────────────────────────────────
import argparse
import os
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Third-party
# ──────────────────────────────────────────────────────────────────────────────
try:
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    import numpy as np
except ImportError as e:
    sys.exit(
        f"[ERROR] Missing dependency: {e}\n"
        "Install with:  pip install pandas matplotlib seaborn numpy"
    )

# ──────────────────────────────────────────────────────────────────────────────
# Configuration constants
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_CSV = "fakeData.csv"
DEFAULT_N   = 7          # lookback window (days)
DEFAULT_OUT = "figures"  # output directory

# Colour palette (mirrors the dashboard's dark-mode tokens)
CLR_SYMP    = "#ff6584"   # symptomatic
CLR_NOSYMP  = "#43e97b"   # no symptoms
CLR_UNK     = "#f9ca24"   # unknown / pending
CLR_BG      = "#1a1d27"   # card background
CLR_TEXT    = "#e8eaf6"   # primary text
CLR_MUTED   = "#8b8fa8"   # secondary text

# Age bin definitions (closed on the right)
AGE_BINS   = [0, 30, 45, 60, 120]
AGE_LABELS = ["18–30", "31–45", "46–60", "61+"]


# ──────────────────────────────────────────────────────────────────────────────
# I.  DATA INGESTION & VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the surveillance CSV into a pandas DataFrame and apply
    type coercions and derived columns.

    Parameters
    ----------
    csv_path : str
        Filesystem path to fakeData.csv.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns:
            no_symptoms, symptoms, date_illness, age, sex,
            date_report, postal_code, occupation,
            age_group, postal_area, status
    """
    required_cols = {
        "No Symptoms", "Symptoms", "Date of Illness",
        "Age", "Sex", "Date of Report", "Postal Code", "Occupation"
    }

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    # ── Rename for idiomatic Python snake_case ────────────────────────────────
    df = df.rename(columns={
        "No Symptoms":    "no_symptoms",
        "Symptoms":       "symptoms",
        "Date of Illness":"date_illness",
        "Age":            "age",
        "Sex":            "sex",
        "Date of Report": "date_report",
        "Postal Code":    "postal_code",
        "Occupation":     "occupation",
    })

    # ── Parse dates ───────────────────────────────────────────────────────────
    df["date_report"]  = pd.to_datetime(df["date_report"],  format="%m/%d/%y")
    df["date_illness"] = pd.to_datetime(df["date_illness"], format="%m/%d/%y")

    # ── Coerce types ──────────────────────────────────────────────────────────
    df["age"]         = pd.to_numeric(df["age"], errors="coerce").astype("Int64")
    df["postal_code"] = pd.to_numeric(df["postal_code"], errors="coerce").astype("Int64")
    df["sex"]         = df["sex"].str.upper().str.strip()
    df["symptoms"]    = df["symptoms"].str.upper().str.strip()
    df["no_symptoms"] = df["no_symptoms"].str.upper().str.strip()

    # ── Derived: age group ────────────────────────────────────────────────────
    df["age_group"] = pd.cut(
        df["age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=True
    )

    # ── Derived: postal area (coarse) ─────────────────────────────────────────
    df["postal_area"] = df["postal_code"].apply(
        lambda p: "85001–85049" if pd.notna(p) and p <= 85049 else "85050–85100"
    )

    # ── Derived: unified status label ─────────────────────────────────────────
    def resolve_status(row):
        if row["symptoms"] == "YES":
            return "Symptomatic"
        if row["no_symptoms"] == "YES":
            return "No Symptoms"
        return "Unknown"

    df["status"] = df.apply(resolve_status, axis=1)

    print(f"[INFO] Loaded {len(df)} records from '{csv_path}'.")
    print(f"[INFO] Date range: {df['date_report'].min().date()} → "
          f"{df['date_report'].max().date()}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# II.  TEMPORAL FILTERING
# ──────────────────────────────────────────────────────────────────────────────

def apply_window(df: pd.DataFrame, n_days: int) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Retain only records whose Date of Report falls within the last
    `n_days` days relative to the dataset's maximum Date of Report.

    Parameters
    ----------
    df     : pd.DataFrame
    n_days : int  — lookback window length (default = 7)

    Returns
    -------
    (filtered_df, window_start, window_end)
    """
    window_end   = df["date_report"].max()
    window_start = window_end - pd.Timedelta(days=n_days - 1)

    mask     = (df["date_report"] >= window_start) & (df["date_report"] <= window_end)
    filtered = df.loc[mask].copy()

    print(f"\n[INFO] Lookback window (N={n_days}): "
          f"{window_start.date()} → {window_end.date()} "
          f"({len(filtered)} records)")
    return filtered, window_start, window_end


# ──────────────────────────────────────────────────────────────────────────────
# III.  DEMOGRAPHIC AGGREGATION
# ──────────────────────────────────────────────────────────────────────────────

def build_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate filtered records by [Age Group × Sex] and compute
    absolute counts and row-wise percentages for each symptom status.

    Parameters
    ----------
    df : pd.DataFrame  — already filtered to the active time window

    Returns
    -------
    pd.DataFrame with columns:
        age_group, sex, Symptomatic, No Symptoms, Unknown,
        total, pct_symp, pct_nosymp, pct_unk
    """
    pivot = (
        df.groupby(["age_group", "sex", "status"], observed=True)
          .size()
          .unstack(fill_value=0)
          .reset_index()
    )

    # Ensure all three status columns exist even if absent in the window
    for col in ["Symptomatic", "No Symptoms", "Unknown"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["total"]      = pivot[["Symptomatic", "No Symptoms", "Unknown"]].sum(axis=1)
    pivot["pct_symp"]   = (pivot["Symptomatic"]  / pivot["total"] * 100).round(1)
    pivot["pct_nosymp"] = (pivot["No Symptoms"]  / pivot["total"] * 100).round(1)
    pivot["pct_unk"]    = (pivot["Unknown"]       / pivot["total"] * 100).round(1)

    # Readable stratum label
    sex_sym = {"MALE": "♂ Male", "FEMALE": "♀ Female"}
    pivot["stratum"] = (
        pivot["age_group"].astype(str)
        + " · "
        + pivot["sex"].map(sex_sym).fillna(pivot["sex"])
    )

    return pivot.sort_values(["age_group", "sex"]).reset_index(drop=True)


def build_daily_counts(df: pd.DataFrame,
                        window_start: pd.Timestamp,
                        window_end: pd.Timestamp) -> pd.DataFrame:
    """
    Produce a complete daily time series of symptom status counts
    over the active window, including zero-count days.

    Parameters
    ----------
    df           : filtered DataFrame
    window_start : first day of the window
    window_end   : last day (= dataset max date)

    Returns
    -------
    pd.DataFrame indexed by date with columns:
        Symptomatic, No Symptoms, Unknown
    """
    full_index = pd.date_range(start=window_start, end=window_end, freq="D")

    daily = (
        df.groupby(["date_report", "status"])
          .size()
          .unstack(fill_value=0)
          .reindex(full_index, fill_value=0)
    )

    for col in ["Symptomatic", "No Symptoms", "Unknown"]:
        if col not in daily.columns:
            daily[col] = 0

    daily.index.name = "date"
    return daily[["Symptomatic", "No Symptoms", "Unknown"]]


# ──────────────────────────────────────────────────────────────────────────────
# IV.  FIGURE FACTORY
# ──────────────────────────────────────────────────────────────────────────────

def _dark_fig(nrows=1, ncols=1, figsize=(10, 5)):
    """Return a matplotlib Figure with the dark-mode style applied."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             facecolor=CLR_BG)
    if not hasattr(axes, "__len__"):
        axes = [axes]
    for ax in (axes if nrows * ncols > 1 else axes):
        ax.set_facecolor(CLR_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2e3148")
        ax.tick_params(colors=CLR_MUTED)
        ax.xaxis.label.set_color(CLR_MUTED)
        ax.yaxis.label.set_color(CLR_MUTED)
        ax.title.set_color(CLR_TEXT)
    return fig, axes


def fig1_grouped_bar(summary: pd.DataFrame, n_days: int, out_dir: Path) -> Path:
    """
    Figure 1: Grouped bar chart showing % Symptomatic and % No Symptoms
    side-by-side for each [Age Group × Sex] stratum.

    Parameters
    ----------
    summary : output of build_group_summary()
    n_days  : N-day window (for title annotation)
    out_dir : directory where the PNG will be saved

    Returns
    -------
    Path to the saved figure.
    """
    if summary.empty:
        print("[WARN] Fig 1 skipped — no data in window.")
        return None

    strata = summary["stratum"].tolist()
    x      = np.arange(len(strata))
    width  = 0.28

    fig, (ax,) = _dark_fig(figsize=(max(10, len(strata) * 1.4), 5))

    bars_s = ax.bar(x - width, summary["pct_symp"],   width, label="% Symptomatic",
                    color=CLR_SYMP,   alpha=0.88, linewidth=0)
    bars_n = ax.bar(x,          summary["pct_nosymp"], width, label="% No Symptoms",
                    color=CLR_NOSYMP, alpha=0.88, linewidth=0)
    bars_u = ax.bar(x + width, summary["pct_unk"],    width, label="% Unknown",
                    color=CLR_UNK,    alpha=0.80, linewidth=0)

    # Annotate bars with percentage labels
    for bars in (bars_s, bars_n, bars_u):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2,
                        f"{h:.0f}%", ha="center", va="bottom",
                        fontsize=7.5, color=CLR_TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(strata, rotation=30, ha="right", fontsize=9, color=CLR_TEXT)
    ax.set_ylabel("Percentage (%)", color=CLR_MUTED)
    ax.set_ylim(0, 115)
    ax.set_title(f"Symptom Status by Age Group × Sex  (Last {n_days} Days)",
                 fontsize=12, color=CLR_TEXT, pad=12)
    ax.yaxis.grid(True, color="#2e3148", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(facecolor=CLR_BG, edgecolor="#2e3148", labelcolor=CLR_TEXT, fontsize=9)

    # Sample size annotations below x-axis
    for i, row in summary.iterrows():
        ax.text(i, -8, f"n={row['total']}", ha="center", va="top",
                fontsize=7, color=CLR_MUTED,
                transform=ax.get_xaxis_transform())

    fig.tight_layout()
    out = out_dir / f"fig1_grouped_bar_N{n_days}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=CLR_BG)
    plt.close(fig)
    print(f"[SAVED] {out}")
    return out


def fig2_donut(df_window: pd.DataFrame, n_days: int, out_dir: Path) -> Path:
    """
    Figure 2: Donut chart of global symptom status distribution
    within the active time window.

    Parameters
    ----------
    df_window : filtered DataFrame
    n_days    : lookback window size
    out_dir   : output directory

    Returns
    -------
    Path to the saved figure.
    """
    counts = df_window["status"].value_counts().reindex(
        ["Symptomatic", "No Symptoms", "Unknown"], fill_value=0
    )
    total  = counts.sum()
    if total == 0:
        print("[WARN] Fig 2 skipped — no data in window.")
        return None

    labels = [f"{k}\n{v} ({v/total*100:.1f}%)" for k, v in counts.items()]
    colors = [CLR_SYMP, CLR_NOSYMP, CLR_UNK]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor=CLR_BG)
    ax.set_facecolor(CLR_BG)

    wedges, _ = ax.pie(
        counts.values,
        labels=None,
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.52, edgecolor=CLR_BG, linewidth=2),
        pctdistance=0.80
    )

    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12),
              ncol=1, facecolor=CLR_BG, edgecolor="#2e3148",
              labelcolor=CLR_TEXT, fontsize=9.5)

    ax.text(0, 0, f"{total}\nrecords", ha="center", va="center",
            fontsize=13, fontweight="bold", color=CLR_TEXT)

    ax.set_title(f"Symptom Status Distribution  (Last {n_days} Days)",
                 fontsize=11, color=CLR_TEXT, pad=14)

    fig.tight_layout()
    out = out_dir / f"fig2_donut_N{n_days}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=CLR_BG)
    plt.close(fig)
    print(f"[SAVED] {out}")
    return out


def fig3_timeseries(daily: pd.DataFrame, n_days: int, out_dir: Path) -> Path:
    """
    Figure 3: Filled area line chart — daily count of Symptomatic vs
    No Symptoms cases over the active time window.

    Parameters
    ----------
    daily   : output of build_daily_counts()
    n_days  : lookback window size
    out_dir : output directory

    Returns
    -------
    Path to the saved figure.
    """
    fig, (ax,) = _dark_fig(figsize=(12, 4))

    ax.fill_between(daily.index, daily["Symptomatic"],
                    color=CLR_SYMP,   alpha=0.20)
    ax.fill_between(daily.index, daily["No Symptoms"],
                    color=CLR_NOSYMP, alpha=0.18)
    ax.fill_between(daily.index, daily["Unknown"],
                    color=CLR_UNK,    alpha=0.15)

    ax.plot(daily.index, daily["Symptomatic"],
            color=CLR_SYMP,   linewidth=2,   marker="o", markersize=4, label="Symptomatic")
    ax.plot(daily.index, daily["No Symptoms"],
            color=CLR_NOSYMP, linewidth=2,   marker="s", markersize=4, label="No Symptoms")
    ax.plot(daily.index, daily["Unknown"],
            color=CLR_UNK,    linewidth=1.5, marker="^", markersize=4, label="Unknown")

    ax.set_ylabel("Daily Count", color=CLR_MUTED)
    ax.set_title(f"Daily Case Count — Symptomatic vs No Symptoms  (Last {n_days} Days)",
                 fontsize=11, color=CLR_TEXT, pad=10)
    ax.yaxis.grid(True, color="#2e3148", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             fontsize=8.5, color=CLR_TEXT)
    ax.legend(facecolor=CLR_BG, edgecolor="#2e3148",
              labelcolor=CLR_TEXT, fontsize=9)

    fig.tight_layout()
    out = out_dir / f"fig3_timeseries_N{n_days}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=CLR_BG)
    plt.close(fig)
    print(f"[SAVED] {out}")
    return out


def fig4_heatmap(df_window: pd.DataFrame, n_days: int, out_dir: Path) -> Path:
    """
    Figure 4: Heatmap of symptom rate (%) by Age Group × Sex.
    Cell values represent % Symptomatic within each stratum.

    Parameters
    ----------
    df_window : filtered DataFrame
    n_days    : lookback window size
    out_dir   : output directory

    Returns
    -------
    Path to the saved figure.
    """
    if df_window.empty:
        print("[WARN] Fig 4 skipped — no data in window.")
        return None

    pivot = (
        df_window.groupby(["age_group", "sex"], observed=True)["status"]
        .apply(lambda s: (s == "Symptomatic").sum() / len(s) * 100)
        .unstack(fill_value=0)
        .round(1)
    )

    # Ensure both sexes appear
    for col in ["MALE", "FEMALE"]:
        if col not in pivot.columns:
            pivot[col] = 0.0

    pivot = pivot[["FEMALE", "MALE"]]

    fig, ax = plt.subplots(figsize=(5, 4), facecolor=CLR_BG)
    ax.set_facecolor(CLR_BG)

    cmap = sns.color_palette(
        ["#1a1d27", "#ff3a5c", "#ff6584", "#ff94a8"], as_cmap=True
    )
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True, fmt=".1f", annot_kws={"size": 11, "color": CLR_TEXT},
        cmap="RdYlGn_r",
        vmin=0, vmax=100,
        linewidths=0.4, linecolor="#2e3148",
        cbar_kws={"label": "% Symptomatic", "shrink": 0.85}
    )

    ax.set_title(f"Symptomatic Rate (%)  by Age × Sex  (Last {n_days} Days)",
                 fontsize=10, color=CLR_TEXT, pad=10)
    ax.set_xlabel("Sex",       color=CLR_MUTED)
    ax.set_ylabel("Age Group", color=CLR_MUTED)
    ax.tick_params(colors=CLR_TEXT, labelsize=9)

    cbar = ax.collections[0].colorbar
    cbar.set_label("% Symptomatic", color=CLR_MUTED)
    cbar.ax.yaxis.set_tick_params(color=CLR_MUTED)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=CLR_MUTED)

    fig.tight_layout()
    out = out_dir / f"fig4_heatmap_N{n_days}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=CLR_BG)
    plt.close(fig)
    print(f"[SAVED] {out}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# V.  SUMMARY STATISTICS (console report)
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(df_full: pd.DataFrame,
                  df_window: pd.DataFrame,
                  summary: pd.DataFrame,
                  n_days: int,
                  window_start: pd.Timestamp,
                  window_end: pd.Timestamp) -> None:
    """
    Print a structured summary table to stdout, mirroring the
    dashboard's KPI cards and group breakdown table.
    """
    total  = len(df_window)
    symp   = (df_window["status"] == "Symptomatic").sum()
    nosymp = (df_window["status"] == "No Symptoms").sum()
    unk    = (df_window["status"] == "Unknown").sum()
    pct    = lambda v: f"{v/total*100:.1f}%" if total else "—"

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  EpiHack Symptom Surveillance — N={n_days}-Day Window Report")
    print(sep)
    print(f"  Window   : {window_start.date()} → {window_end.date()}")
    print(f"  Records  : {total} of {len(df_full)} total")
    print(f"  Symptom+ : {symp:>3}  ({pct(symp)})")
    print(f"  No Symp  : {nosymp:>3}  ({pct(nosymp)})")
    print(f"  Unknown  : {unk:>3}  ({pct(unk)})")
    print(sep)

    if not summary.empty:
        print(f"\n  {'Stratum':<26} {'n':>4}  {'Symp%':>6}  {'NoSymp%':>8}  {'Unk%':>6}")
        print(f"  {'─'*26} {'─'*4}  {'─'*6}  {'─'*8}  {'─'*6}")
        for _, row in summary.iterrows():
            print(f"  {row['stratum']:<26} {int(row['total']):>4}  "
                  f"{row['pct_symp']:>5.1f}%  {row['pct_nosymp']:>7.1f}%  "
                  f"{row['pct_unk']:>5.1f}%")
    print(f"\n{sep}\n")


# ──────────────────────────────────────────────────────────────────────────────
# VI.  CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EpiHack Symptom Surveillance — Reproducibility Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--csv", default=DEFAULT_CSV,
                        help="Path to the input CSV file.")
    parser.add_argument("--n",   type=int, default=DEFAULT_N,
                        help="Lookback window in days (N).")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="Directory to save generated figures.")
    return parser.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    df = load_data(args.csv)

    # ── Step 2: Filter to window ──────────────────────────────────────────────
    df_window, window_start, window_end = apply_window(df, args.n)

    # ── Step 3: Aggregate by stratum ──────────────────────────────────────────
    summary = build_group_summary(df_window)
    daily   = build_daily_counts(df_window, window_start, window_end)

    # ── Step 4: Console report ────────────────────────────────────────────────
    print_summary(df, df_window, summary, args.n, window_start, window_end)

    # ── Step 5: Generate figures ──────────────────────────────────────────────
    fig1_grouped_bar(summary,   args.n, out_dir)
    fig2_donut(df_window,       args.n, out_dir)
    fig3_timeseries(daily,      args.n, out_dir)
    fig4_heatmap(df_window,     args.n, out_dir)

    print(f"[DONE] All figures saved to '{out_dir.resolve()}'.\n")


if __name__ == "__main__":
    main()
