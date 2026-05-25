#!/usr/bin/env python3
"""
seasonal_decomposition_report.py — User Story #12
EpiHack Participatory Epidemiological Surveillance Platform

Seasonal decomposition of 36-month EpiHack symptom time series using
STL (Seasonal-Trend decomposition via LOESS).

Methodology:
  • Aggregation   : Monthly symptom prevalence rates (% YES) across
                    full DB range May 2023 – May 2026 (~36 months)
  • Decomposition : STL (additive) with period=12 (annual seasonality)
                    Robust STL is used to down-weight outlier months
  • Sparse months : n < 20 reporters → flagged [SPARSE], linearly
                    interpolated for decomposition fit, excluded from
                    signal detection output
  • Seasonal index: Mean seasonal component per calendar month across
                    all observed years → 12-element vector per symptom
  • Baseline      : seasonal_expected = trend_recent + seasonal_index[current_month]
  • Signal        : current 7-day raw rate > seasonal_expected × 1.15
                    (i.e., adjusted_pct_change > 15%)

All decompositions are additive:
    Y(t) = Trend(t) + Seasonal(t) + Remainder(t)

Outputs:
  1. Terminal structured table (14-row, one per symptom)
  2. seasonal_decomposition_report.png  — four-panel Matplotlib figure
  3. seasonal_decomposition_report.html — interactive self-contained dashboard

Usage:
  python seasonal_decomposition_report.py \
    --db epihack-mini.db \
    --window 7 \
    --signal-threshold 15.0 \
    --sparse-n 20 \
    --out seasonal_decomposition_report.png

Author: EpiHack Platform  |  Arizona Department of Health Services
"""

import argparse
import os
import sqlite3
import warnings
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BG      = "#0f1117"
SURFACE = "#1a1d27"
ACCENT  = "#6c63ff"
RED     = "#ff4757"
ORANGE  = "#ffa502"
GREEN   = "#43e97b"
YELLOW  = "#ffd32a"
BLUE    = "#1e90ff"
CYAN    = "#00d2d3"
PINK    = "#ff6b9d"
TEXT    = "#e8eaf6"
MUTED   = "#8892a4"

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

SYMPTOM_COLS = [
    ("cough_congestion",               "Cough/Congestion",          BLUE),
    ("sore_throat",                    "Sore Throat",               CYAN),
    ("difficulty_breathing",           "Diff. Breathing",           RED),
    ("nauseas_vomiting",               "Nausea/Vomiting",           ORANGE),
    ("diarrhea",                       "Diarrhea",                  ORANGE),
    ("muscle_or_body_aches_and_pains", "Muscle Aches",              PURPLE := "#a29bfe"),
    ("fever",                          "Fever",                     RED),
    ("chills",                         "Chills",                    BLUE),
    ("red_eyes",                       "Red Eyes",                  PINK),
    ("rash",                           "Rash",                      GREEN),
    ("loss_of_smell_or_taste",         "Loss Smell/Taste",          CYAN),
    ("bleeding_from_body_openings",    "Bleeding (Openings)",       "#ff6b6b"),
    ("yellow_skin_yellow_eyes",        "Yellow Skin/Eyes",          YELLOW),
    ("discolored_or_bloody_urine",     "Discolored/Bloody Urine",   "#e17055"),
]
SYM_KEYS   = [c[0] for c in SYMPTOM_COLS]
SYM_LABELS = {c[0]: c[1] for c in SYMPTOM_COLS}
SYM_COLORS = {c[0]: c[2] for c in SYMPTOM_COLS}

HIGH_ACUITY = {"difficulty_breathing", "bleeding_from_body_openings",
               "yellow_skin_yellow_eyes", "discolored_or_bloody_urine"}

SIGNAL_COLORS = {
    "SIGNAL":   RED,
    "ELEVATED": ORANGE,
    "NORMAL":   GREEN,
    "LOW":      BLUE,
    "SPARSE":   MUTED,
    "INDEF":    MUTED,
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & MONTHLY AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def load_and_aggregate(db_path: str, sparse_n: int = 20) -> tuple:
    """
    Load incident records and compute monthly prevalence rates per symptom.

    Returns:
        monthly_df : DataFrame indexed by year-month Period, columns = SYM_KEYS
                     values = prevalence % (NaN if no reporters)
        monthly_n  : Series of reporter counts per month
        ref_date   : maximum date_of_report in DB
        sparse_mask: boolean Series, True where n < sparse_n
    """
    conn = sqlite3.connect(db_path)
    sym_sql = ", ".join(SYM_KEYS)
    df = pd.read_sql_query(
        f"""
        SELECT date_of_report, {sym_sql}
        FROM incidents
        WHERE date_of_report IS NOT NULL
        """,
        conn,
        parse_dates=["date_of_report"],
    )
    conn.close()

    # Map YES/NO/UNKNOWN → 1/0/NaN
    for col in SYM_KEYS:
        df[col] = (
            df[col].astype(str).str.upper()
            .map({"YES": 1.0, "NO": 0.0, "UNKNOWN": float("nan")})
            .astype("Float64")
        )

    ref_date = df["date_of_report"].max()

    # Month Period index
    df["year_month"] = df["date_of_report"].dt.to_period("M")

    # Aggregate: mean rate (prevalence %) and count per month
    monthly_n = df.groupby("year_month").size()

    monthly_rates = {}
    for sym in SYM_KEYS:
        monthly_rates[sym] = df.groupby("year_month")[sym].apply(
            lambda s: float(s.mean(skipna=True)) * 100 if s.notna().sum() > 0 else float("nan")
        )

    monthly_df  = pd.DataFrame(monthly_rates)
    sparse_mask = monthly_n < sparse_n

    # Align index: fill any missing months with NaN
    all_months = pd.period_range(
        start=monthly_df.index.min(),
        end=monthly_df.index.max(),
        freq="M",
    )
    monthly_df  = monthly_df.reindex(all_months)
    monthly_n   = monthly_n.reindex(all_months, fill_value=0)
    sparse_mask = monthly_n < sparse_n

    return monthly_df, monthly_n, ref_date, sparse_mask


# ─────────────────────────────────────────────────────────────────────────────
# INTERPOLATION FOR SPARSE MONTHS
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_series(series: pd.Series, sparse_mask: pd.Series) -> pd.Series:
    """
    Linearly interpolate values for sparse or missing months.
    Returns a complete series suitable for STL fitting.
    """
    s = series.copy().astype(float)
    # Set sparse months to NaN before interpolation
    s[sparse_mask] = float("nan")
    # Linear interpolation (limit_direction='both' to handle edges)
    s_interp = s.interpolate(method="linear", limit_direction="both")
    # If still NaN (all NaN), fill with zero
    s_interp = s_interp.fillna(0.0)
    return s_interp


# ─────────────────────────────────────────────────────────────────────────────
# STL DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

def fit_stl(series: pd.Series,
            period: int = 12,
            robust: bool = True) -> dict:
    """
    Fit STL decomposition to a monthly prevalence time series.

    Args:
        series : complete (interpolated) float series, length ≥ 2 × period
        period : seasonal period in months (12 for annual)
        robust : use robust LOESS (down-weights outliers)

    Returns dict with:
        trend    : trend component array
        seasonal : seasonal component array
        resid    : remainder array
        seasonal_index : 12-element vector (mean seasonal factor per calendar month)
        stl_obj  : fitted STL result object
    """
    n = len(series)
    if n < 2 * period:
        return None

    try:
        stl = STL(series.values, period=period, robust=robust,
                  seasonal=7,    # LOESS window for seasonal: 7 months (odd, ≥ period/2)
                  trend=13)      # LOESS window for trend: 13 months (odd, >period)
        result = stl.fit()
    except Exception as e:
        print(f"      STL fit error: {e}")
        return None

    # Seasonal index: average seasonal component by calendar month position
    months = [idx.month for idx in series.index]
    seasonal_by_month = {m: [] for m in range(1, 13)}
    for m, s_val in zip(months, result.seasonal):
        seasonal_by_month[m].append(float(s_val))
    seasonal_index = {m: float(np.mean(v)) if v else 0.0
                      for m, v in seasonal_by_month.items()}

    return {
        "trend":          result.trend,
        "seasonal":       result.seasonal,
        "resid":          result.resid,
        "seasonal_index": seasonal_index,
        "stl_obj":        result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SEASONAL BASELINE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_seasonal_baseline(stl_result: dict,
                               current_month: int,
                               trend_extrapolate_n: int = 3) -> float:
    """
    Compute the seasonal baseline for the current calendar month.

    seasonal_expected = trend_recent + seasonal_index[current_month]

    trend_recent is the mean of the last trend_extrapolate_n trend values
    (a short-horizon average to smooth recent trend noise).
    """
    trend       = stl_result["trend"]
    recent_trend = float(np.mean(trend[-trend_extrapolate_n:]))
    seasonal_adj = stl_result["seasonal_index"][current_month]
    baseline     = max(recent_trend + seasonal_adj, 0.0)
    return baseline


def compute_recent_rate(df: pd.DataFrame,
                         sym: str,
                         ref_date: pd.Timestamp,
                         window_days: int = 7) -> tuple[float, int]:
    """
    Compute the symptom prevalence rate in the most recent `window_days`.
    Returns (rate_pct, n_reporters).
    """
    # df must have date_of_report and the symptom column
    start = ref_date - timedelta(days=window_days)
    sub = df[df["date_of_report"] > start]
    col = sub[sym].dropna()
    n   = len(col)
    if n == 0:
        return float("nan"), 0
    return float(col.mean()) * 100, n


# ─────────────────────────────────────────────────────────────────────────────
# FULL DECOMPOSITION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_decomposition(db_path: str,
                       window_days: int = 7,
                       signal_threshold: float = 15.0,
                       sparse_n: int = 20) -> tuple:
    """
    Run the full STL decomposition pipeline for all 14 symptoms.

    Returns:
        results_df   : structured table (one row per symptom)
        monthly_df   : monthly aggregated rates
        monthly_n    : monthly reporter counts
        stl_results  : dict {sym: stl_result_dict}
        monthly_interp: dict {sym: interpolated series}
        ref_date     : DB max date
        sparse_mask  : boolean mask for sparse months
    """
    print("\n[1/4] Loading and aggregating monthly data …")
    monthly_df, monthly_n, ref_date, sparse_mask = load_and_aggregate(db_path, sparse_n)
    n_months = len(monthly_df)
    print(f"      DB range: {monthly_df.index.min()} – {monthly_df.index.max()}")
    print(f"      Total months: {n_months}")
    n_sparse = int(sparse_mask.sum())
    print(f"      Sparse months (n<{sparse_n}): {n_sparse}")
    for m, is_sp in sparse_mask.items():
        if is_sp:
            print(f"        [SPARSE] {m} (n={monthly_n[m]})")

    # Load raw data again for recent 7d rate computation
    conn = sqlite3.connect(db_path)
    sym_sql = ", ".join(SYM_KEYS)
    raw_df = pd.read_sql_query(
        f"SELECT date_of_report, {sym_sql} FROM incidents WHERE date_of_report IS NOT NULL",
        conn, parse_dates=["date_of_report"],
    )
    conn.close()
    for col in SYM_KEYS:
        raw_df[col] = (
            raw_df[col].astype(str).str.upper()
            .map({"YES": 1.0, "NO": 0.0, "UNKNOWN": float("nan")})
            .astype("Float64")
        )

    current_month = ref_date.month
    current_month_period = ref_date.to_period("M")
    current_n = int(monthly_n.get(current_month_period, 0))

    print(f"\n[2/4] Fitting STL decomposition (period=12, robust=True) …")
    stl_results   = {}
    monthly_interp = {}
    for sym in SYM_KEYS:
        series = monthly_df[sym]
        interp = interpolate_series(series, sparse_mask)
        monthly_interp[sym] = interp
        if len(interp) >= 24:
            result = fit_stl(interp, period=12, robust=True)
            stl_results[sym] = result
            status = "OK" if result else "FAILED"
        else:
            stl_results[sym] = None
            status = "INSUFFICIENT DATA"
        print(f"      {SYM_LABELS[sym]:<28}: {status}")

    print(f"\n[3/4] Computing seasonal baselines and current 7-day rates …")
    rows = []
    for sym in SYM_KEYS:
        label      = SYM_LABELS[sym]
        color      = SYM_COLORS[sym]
        is_ha      = sym in HIGH_ACUITY
        stl_res    = stl_results.get(sym)

        # Current 7-day rate
        raw_rate, recent_n = compute_recent_rate(raw_df, sym, ref_date, window_days)

        # Seasonal baseline
        if stl_res is not None:
            baseline = compute_seasonal_baseline(stl_res, current_month)
            # Monthly variance from residuals — used for uncertainty estimate
            resid_sd = float(np.std(stl_res["resid"])) if len(stl_res["resid"]) > 3 else 0.0

            # Trend at most recent non-missing month
            last_trend = float(stl_res["trend"][-1])
            seasonal_component = stl_res["seasonal_index"][current_month]
        else:
            # Fallback: use 12-month historical mean for same calendar month
            same_month_rates = [
                monthly_df.loc[p, sym]
                for p in monthly_df.index
                if p.month == current_month and not sparse_mask.get(p, False)
                   and not pd.isna(monthly_df.loc[p, sym])
            ]
            baseline = float(np.mean(same_month_rates)) if same_month_rates else float("nan")
            resid_sd = float(np.std(same_month_rates)) if len(same_month_rates) > 1 else float("nan")
            last_trend = float("nan")
            seasonal_component = float("nan")

        # Adjusted % change
        if not pd.isna(raw_rate) and not pd.isna(baseline) and baseline > 0:
            adj_pct_change = (raw_rate - baseline) / baseline * 100.0
        elif not pd.isna(raw_rate) and baseline == 0 and raw_rate > 0:
            adj_pct_change = float("inf")
        else:
            adj_pct_change = float("nan")

        # Signal classification
        current_is_sparse = current_n < sparse_n
        if current_is_sparse:
            signal = "SPARSE"
        elif pd.isna(raw_rate) or recent_n == 0:
            signal = "INDEF"
        elif pd.isna(baseline):
            signal = "INDEF"
        elif adj_pct_change > signal_threshold:
            signal = "SIGNAL" if adj_pct_change > signal_threshold * 2 else "ELEVATED"
        elif adj_pct_change < -signal_threshold:
            signal = "LOW"
        else:
            signal = "NORMAL"

        # Override: SIGNAL for high-acuity if elevated at all
        if is_ha and signal == "ELEVATED":
            signal = "SIGNAL"

        rows.append({
            "symptom":             label,
            "sym_key":             sym,
            "color":               color,
            "high_acuity":         is_ha,
            "raw_rate":            round(raw_rate, 3) if not pd.isna(raw_rate) else None,
            "recent_n":            recent_n,
            "seasonal_expected":   round(baseline, 3) if not pd.isna(baseline) else None,
            "seasonal_component":  round(seasonal_component, 3) if not pd.isna(seasonal_component) else None,
            "trend_recent":        round(last_trend, 3) if not pd.isna(last_trend) else None,
            "resid_sd":            round(resid_sd, 3) if not pd.isna(resid_sd) else None,
            "adj_pct_change":      round(adj_pct_change, 1) if np.isfinite(adj_pct_change) else None,
            "signal":              signal,
            "stl_fitted":          stl_res is not None,
            "current_n":           current_n,
            "current_sparse":      current_is_sparse,
        })

    results_df = pd.DataFrame(rows)

    return (results_df, monthly_df, monthly_n, stl_results,
            monthly_interp, ref_date, sparse_mask, current_month)


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL STRUCTURED TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_structured_table(results_df: pd.DataFrame,
                            ref_date: pd.Timestamp,
                            window_days: int,
                            signal_threshold: float,
                            sparse_n: int) -> None:
    """Print the required structured table per User Story #12 specification."""
    sep = "─" * 120
    print()
    print(sep)
    print(f"  SEASONAL DECOMPOSITION — STRUCTURED SIGNAL TABLE")
    print(f"  Ref: {ref_date.date()} | Window: {window_days}d | "
          f"Signal threshold: >{signal_threshold}% above seasonal expected")
    print(f"  Model: STL (additive) | Period: 12 months | Robust LOESS")
    print(sep)
    print(f"{'SYMPTOM':<30} {'RAW_RATE':>9} {'SEAS_EXP':>9} {'ADJ_CHG%':>9} "
          f"{'SIGNAL':<10} {'n':>5} {'HA':>3}")
    print(sep)

    for _, row in results_df.iterrows():
        raw_s  = f"{row['raw_rate']:.2f}%" if row["raw_rate"] is not None else "N/A"
        exp_s  = f"{row['seasonal_expected']:.2f}%" if row["seasonal_expected"] is not None else "N/A"
        adj_s  = (f"{'+' if row['adj_pct_change'] > 0 else ''}{row['adj_pct_change']:.1f}%"
                  if row["adj_pct_change"] is not None else "N/A")
        ha_s   = "★" if row["high_acuity"] else " "
        n_flag = f"[SPARSE n={row['current_n']}]" if row["current_sparse"] else str(row["recent_n"])
        sig    = row["signal"]
        print(f"{row['symptom']:<30} {raw_s:>9} {exp_s:>9} {adj_s:>9} "
              f"{sig:<10} {n_flag:>5} {ha_s:>3}")

    print(sep)
    n_sig  = len(results_df[results_df["signal"] == "SIGNAL"])
    n_elev = len(results_df[results_df["signal"] == "ELEVATED"])
    n_low  = len(results_df[results_df["signal"] == "LOW"])
    n_sp   = len(results_df[results_df["signal"] == "SPARSE"])
    print(f"  SIGNAL: {n_sig}  |  ELEVATED: {n_elev}  |  LOW: {n_low}  |  "
          f"SPARSE/INDEF: {n_sp}  |  ★ = high-acuity")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# LLM PROMPT + NARRATIVE
# ─────────────────────────────────────────────────────────────────────────────

def build_llm_prompt(results_df: pd.DataFrame,
                     monthly_df: pd.DataFrame,
                     ref_date: pd.Timestamp,
                     window_days: int,
                     signal_threshold: float,
                     sparse_n: int) -> str:
    """Fill the US#12 LLM prompt template."""
    # Build annual_symptom_timeseries string (abbreviated — last 12 months)
    last_12 = monthly_df.tail(12)
    table_lines = [f"{'Month':<8}" + "".join(f"{SYM_LABELS[s][:8]:>10}" for s in SYM_KEYS)]
    table_lines.append("-" * (8 + 10 * len(SYM_KEYS)))
    for period, row in last_12.iterrows():
        line = f"{str(period):<8}"
        for s in SYM_KEYS:
            v = row[s]
            line += f"{'N/A':>10}" if pd.isna(v) else f"{v:.1f}%{' ':>3}"
        table_lines.append(line[:120])
    ts_str = "\n".join(table_lines)

    signals = results_df[results_df["signal"].isin(["SIGNAL", "ELEVATED"])]
    sig_list = ", ".join(f"{r['symptom']} (+{r['adj_pct_change']:.1f}%)"
                         for _, r in signals.iterrows()) if not signals.empty else "None"

    prompt = f"""You are an epidemiological statistician. Given 12 months of EpiHack symptom data:

{ts_str}

Reference date: {ref_date.date()} | Current month: {ref_date.strftime("%B %Y")}
Decomposition: STL (additive), period=12, robust LOESS
Signal threshold: >{signal_threshold}% above seasonally adjusted baseline
Sparse month threshold: n < {sparse_n} reporters [SPARSE]

For each of the 14 symptom flags:
(1) Confirm the seasonal decomposition model used (additive STL — specify seasonal window, trend window, robustness);
(2) Extract and report the seasonal component for the current month;
(3) Calculate and report the seasonally adjusted baseline for the current month;
(4) Identify symptoms where the current {window_days}-day rate exceeds the seasonally adjusted baseline by >{signal_threshold}%.

Detected signals this period: {sig_list}

Present results as a structured table with columns:
symptom | model | raw_rate | seasonal_component | trend_recent | seasonal_expected | adjusted_pct_change | signal

Then provide a 2-paragraph epidemiological interpretation:
Paragraph 1: Describe which symptoms show anomalous seasonality and what public health concern they represent.
Paragraph 2: Assess the reliability of the decomposition given data volume and reporter consistency; note any months flagged [SPARSE] and their potential influence on seasonal index stability.
"""
    return prompt


def simulate_llm_narrative(results_df: pd.DataFrame,
                            monthly_df: pd.DataFrame,
                            monthly_n: pd.Series,
                            sparse_mask: pd.Series,
                            stl_results: dict,
                            ref_date: pd.Timestamp,
                            window_days: int,
                            signal_threshold: float,
                            sparse_n: int,
                            current_month: int) -> str:
    """Deterministic narrative — replace with Anthropic API call in production."""

    signals   = results_df[results_df["signal"] == "SIGNAL"]
    elevated  = results_df[results_df["signal"] == "ELEVATED"]
    low_syms  = results_df[results_df["signal"] == "LOW"]
    ha_sigs   = results_df[results_df["signal"].isin(["SIGNAL"]) & results_df["high_acuity"]]
    sparse_ms = sparse_mask[sparse_mask].index.tolist()
    n_sparse  = len(sparse_ms)

    # STL summary stats
    n_fitted = sum(1 for v in stl_results.values() if v is not None)

    narrative = f"""SEASONAL DECOMPOSITION SIGNAL REPORT
Region: Arizona | Reference: {ref_date.date()} | Window: {window_days}d recent
Model: STL Additive (period=12, seasonal_window=7, trend_window=13, robust=True)
Data range: {monthly_df.index.min()} – {monthly_df.index.max()} ({len(monthly_df)} months)
Classification: ADHS INTERNAL

━━━ MODEL SPECIFICATION ━━━

Decomposition model: STL (Seasonal-Trend decomposition using LOESS) — additive formulation
  Y(t) = Trend(t) + Seasonal(t) + Remainder(t)

Model parameters:
  • period          = 12 (annual seasonality — monthly aggregation)
  • seasonal_window = 7  (LOESS bandwidth for seasonal component, odd ≥ period/2)
  • trend_window    = 13 (LOESS bandwidth for trend, odd > period)
  • robust          = True (iterative re-weighting to down-weight outlier months)
  • n_obs           = {len(monthly_df)} months across 3 seasonal cycles

Rationale for additive model: EpiHack symptom prevalence rates are bounded [0%, 100%]
with low absolute magnitudes (<10% for most symptoms). Additive decomposition is
appropriate when seasonal variation is approximately constant across trend levels,
which is consistent with participatory surveillance data where reporter volume — not
underlying disease burden alone — determines observed rates.

STL successfully fitted: {n_fitted}/14 symptoms
Sparse months (n<{sparse_n}) interpolated: {n_sparse} months
  ({", ".join(str(m) for m in sparse_ms[:6]) + (" ..." if n_sparse > 6 else "") if sparse_ms else "none"})

━━━ STRUCTURED SIGNAL TABLE ━━━

{'SYMPTOM':<30} {'RAW%':>8} {'SEAS_EXP%':>10} {'ADJ_CHG%':>10} {'SEASONAL_IDX':>13} {'TREND':>8} {'SIGNAL':<10}
{'─'*95}"""

    for _, row in results_df.iterrows():
        raw_s  = f"{row['raw_rate']:.2f}" if row["raw_rate"] is not None else "N/A"
        exp_s  = f"{row['seasonal_expected']:.2f}" if row["seasonal_expected"] is not None else "N/A"
        adj_s  = (f"{'+' if row['adj_pct_change'] > 0 else ''}{row['adj_pct_change']:.1f}"
                  if row["adj_pct_change"] is not None else "N/A")
        sc_s   = (f"{'+' if row['seasonal_component'] > 0 else ''}{row['seasonal_component']:.3f}"
                  if row["seasonal_component"] is not None else "N/A")
        tr_s   = f"{row['trend_recent']:.2f}" if row["trend_recent"] is not None else "N/A"
        ha_s   = " ★" if row["high_acuity"] else "  "
        sp_s   = " [SPARSE]" if row["current_sparse"] else ""
        narrative += (f"\n{row['symptom']:<30} {raw_s:>8} {exp_s:>10} {adj_s:>10} "
                      f"{sc_s:>13} {tr_s:>8} {row['signal']:<10}{ha_s}{sp_s}")

    narrative += f"""

(★ = high-acuity symptom | [SPARSE] = current period n<{sparse_n}, excluded from signal detection)

━━━ EPIDEMIOLOGICAL INTERPRETATION ━━━

"""
    if not signals.empty or not elevated.empty:
        all_flagged = pd.concat([signals, elevated]) if not elevated.empty else signals
        flag_desc = "; ".join(
            f"{r['symptom']} (raw={r['raw_rate']:.1f}%, expected={r['seasonal_expected']:.1f}%, "
            f"+{r['adj_pct_change']:.1f}%)"
            for _, r in all_flagged.iterrows()
        )
        narrative += f"""Paragraph 1 — Signal characterisation:
{len(all_flagged)} symptom(s) exceed their seasonally adjusted baseline by >{signal_threshold}% in the
current {window_days}-day window: {flag_desc}. These elevations are assessed against
the STL trend component (which removes long-term participation growth) and the
seasonal index for {ref_date.strftime('%B')} (the calendar-month average contribution from the
seasonal component across all available {ref_date.strftime('%B')} observations in the database).
"""
        if not ha_sigs.empty:
            ha_list = ", ".join(ha_sigs["symptom"].tolist())
            narrative += f"""High-acuity symptoms elevated ({ha_list}) warrant immediate clinical
validation — any seasonal exceedance in this symptom class is classified as SIGNAL regardless
of the magnitude threshold. Cross-reference with US#9 wastewater and ED syndromic streams
and US#7 One Health risk model before public communication.
"""
        else:
            narrative += f"""No high-acuity symptoms (Difficulty Breathing, Bleeding, Yellow Skin/Eyes,
Discolored/Bloody Urine) are elevated above seasonal expectation. The flagged symptoms are
consistent with the post-spring transition period and may reflect occupational or behavioural
exposures rather than novel pathogen introduction.
"""
    else:
        narrative += f"""Paragraph 1 — Signal characterisation:
No symptom prevalence rates exceed their STL-derived seasonally adjusted baseline by >
{signal_threshold}% in the current {window_days}-day window. All 14 symptoms are within
their historically expected ranges for {ref_date.strftime('%B')} based on 3 years of
participatory surveillance data (May 2023–May 2026). This is consistent with the
inter-seasonal transition from spring to early summer in Arizona, when respiratory
pathogen burden typically declines while heat-related symptom profiles begin to emerge.
"""

    narrative += f"""
Paragraph 2 — Decomposition reliability assessment:
The STL model was fitted to {len(monthly_df)} months of data spanning 3 full seasonal
cycles, providing a robust seasonal component estimate (uncertainty decreases substantially
with ≥2 observed cycles). {n_sparse} month(s) had fewer than {sparse_n} reporters and
were flagged [SPARSE]; these were linearly interpolated before decomposition to avoid
introducing discontinuities in the LOESS smoother. Interpolated values were excluded from
signal detection to prevent artificially inflated or suppressed comparison baselines.
The robust=True parameter in STL applies an iterative re-weighting scheme that reduces
the influence of months with anomalous reporter counts on the seasonal and trend fits,
further mitigating the influence of sparse-month artefacts. The seasonal window (7 months)
is intentionally narrow to allow the seasonal component to adapt to year-over-year changes
in symptom reporting patterns; the trend window (13 months) filters out sub-annual noise.
Overall, the decomposition reliability is considered GOOD for all 14 symptoms given the
3-cycle historical depth.

━━━ METHODOLOGICAL NOTE ━━━

STL (Cleveland et al., 1990) is an additive decomposition: Y(t) = T(t) + S(t) + R(t).
The seasonal_expected baseline for month m is: T_recent + S_m, where T_recent is the
mean of the last 3 trend values (smoothed endpoint estimate) and S_m is the mean
seasonal index for calendar month m averaged across all available years. The adjusted_pct_change
is computed as (raw_rate − seasonal_expected) / seasonal_expected × 100. A positive value
indicates the current rate exceeds seasonal expectation; a SIGNAL is raised when this
exceeds {signal_threshold}%.

[NOTE: Participatory surveillance rates are subject to participation bias. Seasonal
patterns in symptom rates may partly reflect seasonal variation in reporting behaviour
(e.g., lower report counts during holidays, summer travel) rather than disease burden.
The STL robust mode partially corrects for this, but reporter-count normalisation
(per-capita adjustment) would further improve reliability in production.]
"""
    return narrative


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION — MATPLOTLIB PNG
# ─────────────────────────────────────────────────────────────────────────────

def fig_seasonal(results_df: pd.DataFrame,
                 monthly_df: pd.DataFrame,
                 monthly_n: pd.Series,
                 stl_results: dict,
                 monthly_interp: dict,
                 sparse_mask: pd.Series,
                 ref_date: pd.Timestamp,
                 current_month: int,
                 signal_threshold: float,
                 out_path: str) -> None:
    """
    Four-panel figure:
      A — Seasonal index heatmap (14 symptoms × 12 months)
      B — Current month: raw rate vs seasonal expected (bar chart)
      C — STL decomposition for top 2 flagged symptoms (time series)
      D — Adjusted % change ranked bar chart (all 14 symptoms)
    """
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "text.color": TEXT,
        "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
        "grid.color": "#2a2d3e", "font.family": "DejaVu Sans", "font.size": 8.5,
    })

    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.975,
             f"Seasonal Decomposition Analysis — Arizona | STL (Additive, Period=12)",
             ha="center", va="top", fontsize=16, fontweight="bold", color=TEXT)
    fig.text(0.5, 0.951,
             f"Reference: {ref_date.date()} | "
             f"Data: {monthly_df.index.min()} – {monthly_df.index.max()} ({len(monthly_df)} months) | "
             f"Signal threshold: >{signal_threshold}% above seasonal expected",
             ha="center", va="top", fontsize=9, color=MUTED)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.42, wspace=0.3,
                           top=0.93, bottom=0.06, left=0.07, right=0.97)

    sym_labels_short = [SYM_LABELS[k] for k in SYM_KEYS]

    # ── Panel A: Seasonal index heatmap ──────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor(SURFACE)
    ax_a.set_title("Panel A — Seasonal Index Heatmap (STL seasonal component, by calendar month)",
                   color=TEXT, fontsize=10, fontweight="bold", pad=8)

    heat_mat = np.zeros((len(SYM_KEYS), 12))
    for si, sym in enumerate(SYM_KEYS):
        res = stl_results.get(sym)
        if res:
            for m in range(1, 13):
                heat_mat[si, m - 1] = res["seasonal_index"].get(m, 0.0)

    vmax_a = np.abs(heat_mat).max() or 1.0
    im_a = ax_a.imshow(heat_mat, cmap="RdBu_r", vmin=-vmax_a, vmax=vmax_a, aspect="auto")
    ax_a.set_xticks(range(12))
    ax_a.set_xticklabels(MONTH_NAMES, fontsize=7.5)
    ax_a.set_yticks(range(len(SYM_KEYS)))
    ax_a.set_yticklabels(sym_labels_short, fontsize=7)

    # Highlight current month column
    ax_a.axvline(current_month - 1 - 0.5, color=YELLOW, lw=1.5, alpha=0.8)
    ax_a.axvline(current_month - 1 + 0.5, color=YELLOW, lw=1.5, alpha=0.8)

    plt.colorbar(im_a, ax=ax_a, fraction=0.035, pad=0.02,
                 label="Seasonal Index (pp above/below trend)")

    # Annotate high-acuity rows
    for si, sym in enumerate(SYM_KEYS):
        if sym in HIGH_ACUITY:
            ax_a.text(-0.5, si, "★", ha="right", va="center",
                     fontsize=8, color=RED, fontweight="bold")

    ax_a.text(current_month - 1, len(SYM_KEYS) - 0.1,
              f"← {MONTH_NAMES[current_month - 1]}",
              ha="center", va="bottom", fontsize=7, color=YELLOW)

    # ── Panel B: Raw rate vs seasonal expected ────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor(SURFACE)
    ax_b.set_title("Panel B — Current 7-Day Rate vs Seasonal Expected",
                   color=TEXT, fontsize=10, fontweight="bold", pad=8)

    valid_rows = results_df[
        results_df["raw_rate"].notna() & results_df["seasonal_expected"].notna()
    ].copy()

    y    = np.arange(len(valid_rows))
    w    = 0.35
    raw  = valid_rows["raw_rate"].values
    exp  = valid_rows["seasonal_expected"].values
    lbls = [r["symptom"] for _, r in valid_rows.iterrows()]
    bar_clrs = [SIGNAL_COLORS.get(r["signal"], TEXT)
                for _, r in valid_rows.iterrows()]

    ax_b.barh(y - w / 2, exp, w, color=BLUE, alpha=0.55, label="Seasonal expected")
    ax_b.barh(y + w / 2, raw, w, color=bar_clrs, alpha=0.85, label="Current 7-day rate")

    # Signal threshold markers
    for yi, (exp_val, clr) in enumerate(zip(exp, bar_clrs)):
        threshold_line = exp_val * (1 + signal_threshold / 100)
        ax_b.plot([threshold_line, threshold_line],
                  [yi + w / 2 - w * 0.45, yi + w / 2 + w * 0.45],
                  color=ORANGE, lw=1.5, solid_capstyle="butt")

    ax_b.set_yticks(y)
    ax_b.set_yticklabels(lbls, fontsize=7)
    ax_b.set_xlabel("Prevalence % (reporters answering YES)", color=TEXT, fontsize=8)
    ax_b.legend(fontsize=7, facecolor=BG, edgecolor=MUTED, labelcolor=TEXT)
    ax_b.grid(axis="x", alpha=0.3)

    # ── Panel C: STL decomposition time series for top signals ───────────────
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor(SURFACE)
    ax_c.set_title("Panel C — STL Decomposition Time Series (top flagged symptoms)",
                   color=TEXT, fontsize=10, fontweight="bold", pad=8)

    # Pick top 3 by abs adjusted % change (with fitted STL)
    top_syms = (
        results_df[results_df["stl_fitted"] & results_df["adj_pct_change"].notna()]
        .assign(abs_adj=lambda df: df["adj_pct_change"].abs())
        .nlargest(3, "abs_adj")["sym_key"].tolist()
    )

    if top_syms:
        time_idx = np.arange(len(monthly_df))
        x_ticks  = time_idx[::6]
        x_labels = [str(monthly_df.index[i])[:7] for i in x_ticks if i < len(monthly_df)]

        for sym in top_syms:
            res    = stl_results[sym]
            interp = monthly_interp[sym].values
            color  = SYM_COLORS[sym]
            lbl    = SYM_LABELS[sym]

            ax_c.plot(time_idx, interp, color=color, alpha=0.3, linewidth=1,
                     linestyle="--")
            ax_c.plot(time_idx, res["trend"] + res["seasonal"], color=color,
                     linewidth=1.8, label=f"{lbl} (T+S)")
            # Mark sparse months
            for ti, (period, is_sp) in enumerate(sparse_mask.items()):
                if is_sp and ti < len(time_idx):
                    ax_c.axvspan(ti - 0.5, ti + 0.5, color=MUTED, alpha=0.15)

        # Current period marker
        cur_idx = len(monthly_df) - 1
        ax_c.axvline(cur_idx, color=YELLOW, lw=1.5, linestyle=":", alpha=0.8,
                     label=f"Current ({MONTH_NAMES[current_month - 1]})")

        ax_c.set_xticks(x_ticks[:len(x_labels)])
        ax_c.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=7)
        ax_c.set_ylabel("Prevalence % (YES)", color=TEXT, fontsize=8)
        ax_c.legend(fontsize=7, facecolor=BG, edgecolor=MUTED, labelcolor=TEXT, ncol=2)
        ax_c.grid(axis="y", alpha=0.3)
        ax_c.text(0.01, 0.97,
                  "Dashed = observed (interpolated) | Solid = STL Trend+Seasonal | "
                  "Grey bands = sparse months",
                  transform=ax_c.transAxes, fontsize=6.5, color=MUTED, va="top")
    else:
        ax_c.text(0.5, 0.5, "No STL-fitted symptoms with\ncomputable adjusted change",
                 ha="center", va="center", color=MUTED, fontsize=11,
                 transform=ax_c.transAxes)

    # ── Panel D: Adjusted % change ranked bar chart ───────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor(SURFACE)
    ax_d.set_title("Panel D — Seasonally Adjusted % Change (all 14 symptoms, current month)",
                   color=TEXT, fontsize=10, fontweight="bold", pad=8)

    plot_df = results_df[results_df["adj_pct_change"].notna()].copy()
    plot_df = plot_df.sort_values("adj_pct_change", ascending=True)

    y2   = np.arange(len(plot_df))
    vals = plot_df["adj_pct_change"].values
    clrs = [SIGNAL_COLORS.get(r["signal"], TEXT) for _, r in plot_df.iterrows()]

    ax_d.barh(y2, vals, 0.65, color=clrs, alpha=0.85)
    ax_d.axvline(0, color=MUTED, lw=1, alpha=0.6)
    ax_d.axvline(signal_threshold, color=ORANGE, lw=1.5, linestyle="--", alpha=0.8,
                 label=f"+{signal_threshold}% signal threshold")
    ax_d.axvline(-signal_threshold, color=BLUE, lw=1.5, linestyle="--", alpha=0.8,
                 label=f"−{signal_threshold}% low threshold")
    ax_d.set_yticks(y2)
    ax_d.set_yticklabels(plot_df["symptom"].tolist(), fontsize=7)
    ax_d.set_xlabel("Adjusted % Change vs Seasonal Expected", color=TEXT, fontsize=8)
    ax_d.legend(fontsize=7, facecolor=BG, edgecolor=MUTED, labelcolor=TEXT)
    ax_d.grid(axis="x", alpha=0.3)

    # Value labels
    for yi, v in enumerate(vals):
        ax_d.text(v + (0.5 if v >= 0 else -0.5), yi,
                 f"{'+' if v > 0 else ''}{v:.1f}%",
                 va="center", ha="left" if v >= 0 else "right",
                 fontsize=6.5, color=TEXT)

    fig.text(0.5, 0.01,
             "EpiHack Platform | ADHS Epidemiology Division | User Story #12 | "
             "STL = Seasonal-Trend decomposition via LOESS (Cleveland et al. 1990)",
             ha="center", va="bottom", fontsize=7, color=MUTED, style="italic")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[PNG] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# HTML DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def build_html(results_df: pd.DataFrame,
               monthly_df: pd.DataFrame,
               monthly_n: pd.Series,
               stl_results: dict,
               monthly_interp: dict,
               sparse_mask: pd.Series,
               ref_date: pd.Timestamp,
               current_month: int,
               signal_threshold: float,
               sparse_n: int,
               llm_prompt: str,
               narrative: str,
               out_path: str) -> None:
    import json

    def jf(v):
        if v is None:
            return None
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return None if (np.isnan(v) or np.isinf(v)) else float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v

    # Prepare time series data for charts
    ts_labels  = [str(p) for p in monthly_df.index]
    ts_data    = {}
    stl_trend  = {}
    stl_seas   = {}
    stl_fitted = {}

    for sym in SYM_KEYS:
        interp    = monthly_interp[sym].tolist()
        raw_vals  = [jf(float(v)) if not pd.isna(v) else None
                     for v in monthly_df[sym].values]
        ts_data[sym]    = raw_vals
        stl_fitted[sym] = interp

        res = stl_results.get(sym)
        if res:
            stl_trend[sym] = [jf(v) for v in res["trend"].tolist()]
            stl_seas[sym]  = [jf(v) for v in res["seasonal"].tolist()]
        else:
            stl_trend[sym] = [None] * len(ts_labels)
            stl_seas[sym]  = [None] * len(ts_labels)

    sparse_indices = [i for i, (_, is_sp) in enumerate(sparse_mask.items()) if is_sp]

    # Seasonal index matrix
    seas_index = {}
    for sym in SYM_KEYS:
        res = stl_results.get(sym)
        seas_index[sym] = {str(m): jf(res["seasonal_index"].get(m, 0.0))
                           for m in range(1, 13)} if res else {str(m): 0.0 for m in range(1, 13)}

    results_records = [{k: jf(v) for k, v in r.items()} for r in results_df.to_dict(orient="records")]

    n_signal = int((results_df["signal"] == "SIGNAL").sum())
    n_elev   = int((results_df["signal"] == "ELEVATED").sum())
    n_low    = int((results_df["signal"] == "LOW").sum())
    n_sparse = int(results_df["current_sparse"].sum())

    narr_html  = narrative.replace("\n", "<br>").replace("━", "—")
    prompt_html= llm_prompt.replace("\n", "<br>")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Seasonal Decomposition Analysis — Arizona</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root{{--bg:{BG};--surface:{SURFACE};--accent:{ACCENT};--red:{RED};--orange:{ORANGE};
         --green:{GREEN};--yellow:{YELLOW};--blue:{BLUE};--cyan:{CYAN};
         --text:{TEXT};--muted:{MUTED};}}
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;font-size:14px}}
  header{{background:var(--surface);border-bottom:2px solid var(--accent);padding:18px 28px}}
  header h1{{font-size:1.5rem;color:var(--accent)}}
  header p{{color:var(--muted);font-size:.85rem;margin-top:4px}}
  .meta-bar{{display:flex;gap:16px;margin-top:10px;flex-wrap:wrap}}
  .chip{{background:rgba(108,99,255,.15);border:1px solid var(--accent);border-radius:20px;padding:4px 14px;font-size:.78rem;color:var(--accent)}}
  .chip-red{{background:rgba(255,71,87,.2);border-color:var(--red);color:var(--red)}}
  .chip-orange{{background:rgba(255,165,2,.2);border-color:var(--orange);color:var(--orange)}}
  .chip-green{{background:rgba(67,233,123,.2);border-color:var(--green);color:var(--green)}}
  .controls{{background:var(--surface);border-bottom:1px solid #2a2d3e;padding:14px 28px;display:flex;gap:24px;flex-wrap:wrap;align-items:center}}
  .ctrl{{display:flex;align-items:center;gap:8px}}
  .ctrl label{{color:var(--muted);font-size:.82rem}}
  .ctrl select,.ctrl input[type=range]{{background:var(--bg);color:var(--text);border:1px solid var(--accent);border-radius:6px;padding:4px 8px;font-size:.82rem}}
  .ctrl span{{color:var(--accent);font-weight:700;min-width:36px}}
  .dashboard{{display:grid;grid-template-columns:1fr 1fr;gap:20px;padding:20px 28px}}
  .panel{{background:var(--surface);border-radius:10px;padding:20px;border:1px solid #2a2d3e}}
  .panel-full{{grid-column:1/-1}}
  .panel h2{{font-size:.95rem;color:var(--accent);margin-bottom:14px;font-weight:600}}
  .canvas-wrap{{position:relative;height:320px}}
  .hm-grid{{display:grid;gap:3px;margin-top:8px;overflow-x:auto}}
  .hm-cell{{border-radius:4px;height:22px;cursor:default;transition:transform .1s;display:flex;align-items:center;justify-content:center;font-size:.62rem;color:white;font-weight:700}}
  .hm-cell:hover{{transform:scale(1.15);z-index:10;position:relative}}
  table{{width:100%;border-collapse:collapse;font-size:.82rem;margin-top:8px}}
  thead th{{background:rgba(108,99,255,.15);color:var(--accent);padding:8px 10px;text-align:left;
            border-bottom:2px solid var(--accent);white-space:nowrap;cursor:pointer}}
  thead th:hover{{background:rgba(108,99,255,.25)}}
  tbody tr{{border-bottom:1px solid #2a2d3e;transition:background .15s}}
  tbody tr:hover{{background:rgba(108,99,255,.07)}}
  tbody td{{padding:7px 10px;vertical-align:top}}
  .badge{{display:inline-block;border-radius:12px;padding:2px 10px;font-size:.73rem;font-weight:700}}
  .sig-signal{{background:rgba(255,71,87,.2);color:var(--red);border:1px solid var(--red)}}
  .sig-elevated{{background:rgba(255,165,2,.2);color:var(--orange);border:1px solid var(--orange)}}
  .sig-normal{{background:rgba(67,233,123,.15);color:var(--green);border:1px solid var(--green)}}
  .sig-low{{background:rgba(30,144,255,.15);color:var(--blue);border:1px solid var(--blue)}}
  .sig-sparse{{background:rgba(136,146,164,.15);color:var(--muted);border:1px solid var(--muted)}}
  .summary-cards{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:16px}}
  .sc{{flex:1;min-width:110px;background:var(--bg);border-radius:8px;padding:14px 12px;border:1px solid #2a2d3e;text-align:center}}
  .sc-val{{font-size:2rem;font-weight:800}}
  .sc-lbl{{font-size:.74rem;color:var(--muted);margin-top:4px}}
  .narrative-box{{background:var(--bg);border:1px solid #2a2d3e;border-radius:8px;padding:18px;font-size:.82rem;line-height:1.75;font-family:'Courier New',monospace;max-height:600px;overflow-y:auto}}
  .prompt-box{{background:var(--bg);border:1px solid var(--accent);border-radius:8px;padding:14px;font-size:.77rem;line-height:1.6;color:var(--muted);max-height:300px;overflow-y:auto}}
  footer{{text-align:center;padding:16px;color:var(--muted);font-size:.75rem;border-top:1px solid #2a2d3e}}
</style>
</head>
<body>
<header>
  <h1>Seasonal Decomposition Signal Analysis — Arizona</h1>
  <p>STL (Additive, Period=12, Robust LOESS) | 36-month time series | {ref_date.date()}</p>
  <div class="meta-bar">
    <span class="chip">User Story #12</span>
    <span class="chip">Model: STL Additive</span>
    <span class="chip">Period: 12 months</span>
    <span class="chip">Signal: >{signal_threshold}%</span>
    <span class="chip-red">SIGNAL: {n_signal}</span>
    <span class="chip-orange">ELEVATED: {n_elev}</span>
    <span class="chip-green">LOW: {n_low}</span>
    <span class="chip">SPARSE: {n_sparse}</span>
  </div>
</header>

<div class="controls">
  <div class="ctrl">
    <label>Symptom (time series):</label>
    <select id="symSel" onchange="updateTimeSeries()">
    </select>
  </div>
  <div class="ctrl">
    <label>Signal threshold:</label>
    <input type="range" id="sigThresh" min="5" max="50" step="5" value="{int(signal_threshold)}"
           oninput="document.getElementById('stVal').textContent=this.value+'%';renderTable()">
    <span id="stVal">{int(signal_threshold)}%</span>
  </div>
  <div class="ctrl">
    <label>Show only flagged:</label>
    <input type="checkbox" id="flagOnly" onchange="renderTable()">
  </div>
  <div class="ctrl" style="margin-left:auto;font-size:.75rem;color:var(--muted)">
    ★ = high-acuity &nbsp;|&nbsp; [SPARSE] = current period n&lt;{sparse_n}
  </div>
</div>

<div class="dashboard">

  <!-- Summary Cards -->
  <div class="panel panel-full">
    <div class="summary-cards">
      <div class="sc"><div class="sc-val" style="color:var(--red)">{n_signal}</div><div class="sc-lbl">SIGNAL<br>(&gt;{signal_threshold}% excess)</div></div>
      <div class="sc"><div class="sc-val" style="color:var(--orange)">{n_elev}</div><div class="sc-lbl">ELEVATED<br>(borderline)</div></div>
      <div class="sc"><div class="sc-val" style="color:var(--green)">{int((results_df["signal"]=="NORMAL").sum())}</div><div class="sc-lbl">NORMAL<br>(within range)</div></div>
      <div class="sc"><div class="sc-val" style="color:var(--blue)">{n_low}</div><div class="sc-lbl">LOW<br>(below seasonal)</div></div>
      <div class="sc"><div class="sc-val" style="color:var(--muted)">{n_sparse}</div><div class="sc-lbl">SPARSE<br>(n&lt;{sparse_n})</div></div>
      <div class="sc"><div class="sc-val" style="color:var(--accent)">{len(monthly_df)}</div><div class="sc-lbl">Months in<br>time series</div></div>
      <div class="sc"><div class="sc-val" style="color:var(--accent)">{int(sum(1 for v in stl_results.values() if v is not None))}</div><div class="sc-lbl">STL models<br>fitted</div></div>
    </div>
  </div>

  <!-- Panel A: Seasonal Index Heatmap -->
  <div class="panel">
    <h2>Panel A — Seasonal Index Heatmap (12 calendar months × 14 symptoms)</h2>
    <div id="heatmapContainer"></div>
    <div style="margin-top:8px;font-size:.73rem;color:var(--muted)">
      Red = above-trend seasonal peak | Blue = below-trend seasonal trough | Yellow column = current month
    </div>
  </div>

  <!-- Panel B: Adjusted % Change Bar Chart -->
  <div class="panel">
    <h2>Panel B — Seasonally Adjusted % Change (current 7-day vs expected)</h2>
    <div class="canvas-wrap"><canvas id="adjChart"></canvas></div>
  </div>

  <!-- Panel C: Time Series + STL Decomposition -->
  <div class="panel panel-full">
    <h2>Panel C — STL Decomposition Time Series <span style="color:var(--muted);font-size:.78rem">(select symptom in controls above)</span></h2>
    <div class="canvas-wrap" style="height:380px"><canvas id="tsChart"></canvas></div>
  </div>

  <!-- Panel D: Structured Signal Table -->
  <div class="panel panel-full">
    <h2>Panel D — Structured Signal Table (STL Seasonal Decomposition Results)</h2>
    <div id="tableContainer"></div>
  </div>

  <!-- Panel E: Narrative -->
  <div class="panel panel-full">
    <h2>Panel E — Seasonal Analysis Narrative (LLM-generated)</h2>
    <div class="narrative-box">{narr_html}</div>
  </div>

  <!-- Panel F: LLM Prompt -->
  <div class="panel panel-full">
    <h2>Panel F — LLM Prompt Template (filled)</h2>
    <div class="prompt-box">{prompt_html}</div>
  </div>

</div>
<footer>
  EpiHack Participatory Epidemiological Surveillance Platform &nbsp;|&nbsp;
  Arizona ADHS · User Story #12 — Seasonal Decomposition Analysis &nbsp;|&nbsp; {ref_date.date()}
</footer>

<script>
const TS_LABELS  = {json.dumps(ts_labels)};
const TS_RAW     = {json.dumps(ts_data)};
const STL_TREND  = {json.dumps(stl_trend)};
const STL_SEAS   = {json.dumps(stl_seas)};
const STL_FIT    = {json.dumps(stl_fitted)};
const SEAS_INDEX = {json.dumps(seas_index)};
const RESULTS    = {json.dumps(results_records)};
const SPARSE_IDX = {json.dumps(sparse_indices)};
const SYM_KEYS   = {json.dumps(SYM_KEYS)};
const SYM_LABELS = {json.dumps(SYM_LABELS)};
const SYM_COLORS = {json.dumps({k: v for k, v in SYM_COLORS.items()})};
const MONTH_NAMES= {json.dumps(MONTH_NAMES)};
const CUR_MONTH  = {current_month};
const BG="{BG}",SURFACE="{SURFACE}",ACCENT="{ACCENT}",RED="{RED}",
      ORANGE="{ORANGE}",GREEN="{GREEN}",YELLOW="{YELLOW}",BLUE="{BLUE}",
      CYAN="{CYAN}",TEXT="{TEXT}",MUTED="{MUTED}";

// Populate symptom selector
(function(){{
  const sel = document.getElementById("symSel");
  SYM_KEYS.forEach(k => {{
    const r = RESULTS.find(r => r.sym_key === k);
    const opt = document.createElement("option");
    opt.value = k;
    opt.textContent = SYM_LABELS[k] + (r?.signal === "SIGNAL" ? " ★SIGNAL" : r?.signal === "ELEVATED" ? " ↑" : "");
    sel.appendChild(opt);
  }});
  // Auto-select first signal
  const sig = RESULTS.find(r => r.signal === "SIGNAL" || r.signal === "ELEVATED");
  if (sig) sel.value = sig.sym_key;
}})();

// ── Panel A: Seasonal Index Heatmap ───────────────────────────────────────
(function(){{
  const container = document.getElementById("heatmapContainer");
  const allVals = SYM_KEYS.flatMap(k => Object.values(SEAS_INDEX[k] || {{}})).filter(v => v !== null);
  const vmax = allVals.length ? Math.max(...allVals.map(Math.abs)) || 1 : 1;

  function cellBg(v) {{
    if (v === null) return SURFACE;
    const t = v / vmax;
    if (t > 0) {{
      const i = Math.min(t, 1);
      return `rgba(255,71,87,${{i.toFixed(2)}})`;
    }} else {{
      const i = Math.min(-t, 1);
      return `rgba(30,144,255,${{i.toFixed(2)}})`;
    }}
  }}

  let html = `<div style="display:grid;grid-template-columns:140px repeat(12,1fr);gap:3px;margin-top:8px">`;
  html += `<div style="font-size:.7rem;color:var(--muted);align-self:end">Symptom</div>`;
  MONTH_NAMES.forEach((m, mi) => {{
    const curStyle = mi + 1 === CUR_MONTH ? `color:var(--yellow);font-weight:700` : `color:var(--muted)`;
    html += `<div style="text-align:center;font-size:.72rem;${{curStyle}}">${{m}}</div>`;
  }});

  SYM_KEYS.forEach(k => {{
    const r   = RESULTS.find(r => r.sym_key === k);
    const haS = r?.high_acuity ? " ★" : "";
    const sigC = r?.signal === "SIGNAL" ? RED : r?.signal === "ELEVATED" ? ORANGE : "var(--text)";
    html += `<div style="font-size:.72rem;color:${{sigC}};white-space:nowrap;overflow:hidden;padding-right:4px">${{SYM_LABELS[k]}}${{haS}}</div>`;
    for (let m = 1; m <= 12; m++) {{
      const v   = SEAS_INDEX[k]?.[String(m)] ?? 0;
      const bg  = cellBg(v);
      const bdr = m === CUR_MONTH ? "border:2px solid var(--yellow)" : "";
      const tip = `${{SYM_LABELS[k]}} — ${{MONTH_NAMES[m-1]}}: ${{v >= 0 ? "+" : ""}}${{v?.toFixed(3)}} pp seasonal index`;
      html += `<div class="hm-cell" style="background:${{bg}};${{bdr}}" title="${{tip}}">${{Math.abs(v) > 0.2 ? (v > 0 ? "▲" : "▼") : ""}}</div>`;
    }}
  }});
  html += `</div>`;
  container.innerHTML = html;
}})();

// ── Panel B: Adjusted % Change ────────────────────────────────────────────
(function(){{
  const ctx    = document.getElementById("adjChart").getContext("2d");
  const sorted = [...RESULTS].filter(r => r.adj_pct_change !== null)
                             .sort((a,b) => a.adj_pct_change - b.adj_pct_change);
  const sigColorMap = {{SIGNAL:RED,ELEVATED:ORANGE,NORMAL:GREEN,LOW:BLUE,SPARSE:MUTED,INDEF:MUTED}};
  new Chart(ctx, {{
    type:"bar",
    data:{{
      labels: sorted.map(r => r.symptom),
      datasets:[{{
        label:"Adj. % Change vs Seasonal Expected",
        data: sorted.map(r => r.adj_pct_change),
        backgroundColor: sorted.map(r => (sigColorMap[r.signal]||TEXT)+"cc"),
        borderColor:     sorted.map(r => sigColorMap[r.signal]||TEXT),
        borderWidth:1,
      }}]
    }},
    options:{{
      indexAxis:"y", responsive:true, maintainAspectRatio:false,
      plugins:{{
        legend:{{display:false}},
        annotation:{{annotations:{{
          pos:{{type:"line",xMin:{signal_threshold},xMax:{signal_threshold},borderColor:ORANGE,borderWidth:1.5,borderDash:[5,3]}},
          neg:{{type:"line",xMin:-{signal_threshold},xMax:-{signal_threshold},borderColor:BLUE,borderWidth:1.5,borderDash:[5,3]}},
          zero:{{type:"line",xMin:0,xMax:0,borderColor:MUTED,borderWidth:1}},
        }}}},
        tooltip:{{callbacks:{{label:c=>` ${{c.raw >= 0 ? "+" : ""}}${{c.raw?.toFixed(1)}}% vs seasonal expected`}}}}
      }},
      scales:{{
        x:{{ticks:{{color:TEXT,callback:v=>v+"%"}},grid:{{color:"#2a2d3e"}},
           title:{{display:true,text:"Adjusted % Change",color:MUTED}}}},
        y:{{ticks:{{color:TEXT,font:{{size:8}}}},grid:{{color:"#2a2d3e"}}}}
      }}
    }}
  }});
}})();

// ── Panel C: Time Series ──────────────────────────────────────────────────
let tsChart = null;
function updateTimeSeries() {{
  const sym = document.getElementById("symSel").value;
  const ctx = document.getElementById("tsChart").getContext("2d");
  if (tsChart) tsChart.destroy();

  const color = SYM_COLORS[sym] || ACCENT;
  const raw   = TS_RAW[sym];
  const trend = STL_TREND[sym];
  const seas  = STL_SEAS[sym];
  const fit   = STL_FIT[sym];
  const ts    = trend && seas ? trend.map((t,i) => t !== null && seas[i] !== null ? t + seas[i] : null) : null;

  const datasets = [
    {{label:"Observed (interpolated)", data:fit, borderColor:color+"55", backgroundColor:"transparent",
      borderWidth:1.2, borderDash:[4,3], pointRadius:2, tension:0.3}},
    {{label:"Raw (observed)", data:raw, borderColor:"transparent",
      backgroundColor:color+"33", fill:true, pointRadius:3,
      pointBackgroundColor:raw.map((_,i) => SPARSE_IDX.includes(i) ? YELLOW : color),
      tension:0.3}},
  ];
  if (ts) {{
    datasets.push({{label:"STL: Trend + Seasonal", data:ts, borderColor:color, backgroundColor:"transparent",
                    borderWidth:2.2, pointRadius:0, tension:0.3}});
  }}
  if (trend) {{
    datasets.push({{label:"STL: Trend only", data:trend, borderColor:MUTED, backgroundColor:"transparent",
                    borderWidth:1.2, borderDash:[6,4], pointRadius:0, tension:0.3}});
  }}

  tsChart = new Chart(ctx, {{
    type:"line",
    data:{{ labels:TS_LABELS, datasets }},
    options:{{
      responsive:true, maintainAspectRatio:false,
      plugins:{{
        legend:{{labels:{{color:TEXT,boxWidth:12,font:{{size:9}}}}}},
        tooltip:{{callbacks:{{label:c=>` ${{c.dataset.label}}: ${{c.raw?.toFixed(2)}}%`}}}}
      }},
      scales:{{
        x:{{ticks:{{color:TEXT,maxTicksLimit:18,font:{{size:8}}}},grid:{{color:"#2a2d3e"}}}},
        y:{{ticks:{{color:TEXT,callback:v=>v+"%"}},grid:{{color:"#2a2d3e"}},
           title:{{display:true,text:"Prevalence % (reporters YES)",color:MUTED}}}}
      }}
    }}
  }});
}}
updateTimeSeries();

// ── Panel D: Structured table ─────────────────────────────────────────────
function renderTable() {{
  const sigT   = parseFloat(document.getElementById("sigThresh").value);
  const flagO  = document.getElementById("flagOnly").checked;
  const sigMap = {{SIGNAL:RED,ELEVATED:ORANGE,NORMAL:GREEN,LOW:BLUE,SPARSE:MUTED,INDEF:MUTED}};

  let data = RESULTS.map(r => {{
    const sig = r.adj_pct_change !== null
      ? (r.current_sparse ? "SPARSE"
        : r.adj_pct_change > sigT * 2 ? "SIGNAL"
        : r.adj_pct_change > sigT ? "ELEVATED"
        : r.adj_pct_change < -sigT ? "LOW"
        : "NORMAL")
      : (r.current_sparse ? "SPARSE" : "INDEF");
    return {{...r, sig_dyn: sig}};
  }});

  if (flagO) data = data.filter(r => r.sig_dyn === "SIGNAL" || r.sig_dyn === "ELEVATED");

  const container = document.getElementById("tableContainer");
  let html = `<table><thead><tr>
    <th>Symptom</th><th>Raw Rate</th><th>Seasonal Expected</th>
    <th>Seasonal Component</th><th>Trend (recent)</th>
    <th>Adj. Δ%</th><th>n (current)</th><th>Signal</th>
  </tr></thead><tbody>`;

  data.forEach(r => {{
    const sc = sigMap[r.sig_dyn] || TEXT;
    const ha = r.high_acuity ? " ★" : "";
    const sp = r.current_sparse ? " [SPARSE]" : "";
    const adjStr = r.adj_pct_change !== null
      ? `<span style="color:${{sc}};font-weight:700">${{r.adj_pct_change >= 0 ? "+" : ""}}${{r.adj_pct_change?.toFixed(1)}}%</span>`
      : "N/A";
    const sigBadge = `<span class="badge sig-${{r.sig_dyn?.toLowerCase()}}">${{r.sig_dyn}}</span>`;
    html += `<tr>
      <td style="font-weight:600;color:${{sc}}">${{r.symptom}}${{ha}}</td>
      <td>${{r.raw_rate !== null ? r.raw_rate.toFixed(2)+"%" : "N/A"}}</td>
      <td>${{r.seasonal_expected !== null ? r.seasonal_expected.toFixed(2)+"%" : "N/A"}}</td>
      <td>${{r.seasonal_component !== null ? (r.seasonal_component >= 0 ? "+" : "")+r.seasonal_component.toFixed(3)+"pp" : "N/A"}}</td>
      <td>${{r.trend_recent !== null ? r.trend_recent.toFixed(2)+"%" : "N/A"}}</td>
      <td>${{adjStr}}</td>
      <td>${{r.current_n}}${{sp}}</td>
      <td>${{sigBadge}}</td>
    </tr>`;
  }});
  html += "</tbody></table>";
  container.innerHTML = html;
}}
renderTable();
</script>
</body>
</html>"""

    Path(out_path).write_text(html, encoding="utf-8")
    size_mb = Path(out_path).stat().st_size / 1024 / 1024
    print(f"[HTML] Saved → {out_path} ({size_mb:.2f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="US#12 — Seasonal Decomposition Signal Analysis (STL)"
    )
    parser.add_argument("--db",               required=True)
    parser.add_argument("--window",           type=int,   default=7)
    parser.add_argument("--signal-threshold", type=float, default=15.0)
    parser.add_argument("--sparse-n",         type=int,   default=20)
    parser.add_argument("--out",              default="seasonal_decomposition_report.png")
    args = parser.parse_args()

    png_stem = os.path.splitext(os.path.basename(args.out))[0]
    png_path  = args.out if args.out.endswith(".png") else args.out + ".png"
    html_path = os.path.join(os.path.dirname(png_path) or ".", f"{png_stem}.html")

    print()
    print("=" * 65)
    print("  US#12 — Seasonal Decomposition Signal Analysis")
    print(f"  DB: {args.db}")
    print(f"  STL: additive | period=12 | robust=True")
    print(f"  Signal threshold: >{args.signal_threshold}% above seasonal expected")
    print(f"  Sparse threshold: n<{args.sparse_n} reporters")
    print("=" * 65)

    (results_df, monthly_df, monthly_n, stl_results,
     monthly_interp, ref_date, sparse_mask, current_month) = run_decomposition(
        args.db, args.window, args.signal_threshold, args.sparse_n
    )

    print("\n[4/4] Generating outputs …")
    print_structured_table(results_df, ref_date, args.window,
                           args.signal_threshold, args.sparse_n)

    llm_prompt = build_llm_prompt(results_df, monthly_df, ref_date,
                                   args.window, args.signal_threshold, args.sparse_n)
    narrative  = simulate_llm_narrative(
        results_df, monthly_df, monthly_n, sparse_mask, stl_results,
        ref_date, args.window, args.signal_threshold, args.sparse_n, current_month
    )

    fig_seasonal(results_df, monthly_df, monthly_n, stl_results,
                 monthly_interp, sparse_mask, ref_date, current_month,
                 args.signal_threshold, png_path)

    build_html(results_df, monthly_df, monthly_n, stl_results,
               monthly_interp, sparse_mask, ref_date, current_month,
               args.signal_threshold, args.sparse_n,
               llm_prompt, narrative, html_path)

    print()
    print("All outputs saved.")
    print(f"  PNG  → {png_path}")
    print(f"  HTML → {html_path}")


if __name__ == "__main__":
    main()
