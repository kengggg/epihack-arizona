#!/usr/bin/env python3
"""
demographic_surveillance_report.py — User Story #10
EpiHack Participatory Epidemiological Surveillance Platform

Demographic surveillance analyst:
  Compares symptom prevalence by age group × sex for:
    • Recent window : last 7 days (relative to DB max date)
    • Baseline      : prior 30-day period

  Detection thresholds:
    1. Symptom rate increased >20% RELATIVE to baseline per age×sex stratum
    2. Stratum's share of total reporters shifted >10 percentage points

  For each identified shift:
    • Describes the demographic pattern
    • Assesses true epidemiological signal vs reporting artifact
    • Recommends follow-up actions

  Age bands (5-band ADHS/CDC standard):
    Pediatric (0–17) | Young Adult (18–34) | Middle Adult (35–54)
    Pre-Senior (55–64) | Senior (65+)

  Minimum reporters per stratum: 10 (sparse strata flagged and excluded)

Outputs:
  1. Terminal structured report
  2. demographic_surveillance_report.png  — four-panel Matplotlib figure
  3. demographic_surveillance_report.html — interactive self-contained dashboard

Usage:
  python demographic_surveillance_report.py \
    --db epihack-mini.db \
    --recent 7 \
    --baseline 30 \
    --min-n 10 \
    --rate-threshold 20.0 \
    --share-threshold 10.0 \
    --out demographic_surveillance_report.png

Author: EpiHack Platform  |  Arizona Department of Health Services
"""

import argparse
import os
import sqlite3
import sys
import warnings
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — design system
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

# Age band definitions (label, low, high inclusive)
AGE_BANDS = [
    ("Pediatric (0–17)",    0,  17),
    ("Young Adult (18–34)", 18, 34),
    ("Middle Adult (35–54)", 35, 54),
    ("Pre-Senior (55–64)",  55, 64),
    ("Senior (65+)",        65, 200),
]
AGE_BAND_LABELS  = [b[0] for b in AGE_BANDS]
AGE_BAND_COLORS  = [CYAN, GREEN, ACCENT, ORANGE, RED]

SEX_CATEGORIES = ["Female", "Male", "Other/Unknown"]
SEX_COLORS     = [PINK, BLUE, MUTED]

# Canonical symptom columns and display labels
SYMPTOM_COLS = [
    ("cough_congestion",               "Cough/Congestion"),
    ("sore_throat",                    "Sore Throat"),
    ("difficulty_breathing",           "Difficulty Breathing"),
    ("nauseas_vomiting",               "Nausea/Vomiting"),
    ("diarrhea",                       "Diarrhea"),
    ("muscle_or_body_aches_and_pains", "Muscle Aches"),
    ("fever",                          "Fever"),
    ("chills",                         "Chills"),
    ("red_eyes",                       "Red Eyes"),
    ("rash",                           "Rash"),
    ("loss_of_smell_or_taste",         "Loss of Smell/Taste"),
    ("bleeding_from_body_openings",    "Bleeding (Openings)"),
    ("yellow_skin_yellow_eyes",        "Yellow Skin/Eyes"),
    ("discolored_or_bloody_urine",     "Discolored/Bloody Urine"),
]
SYM_KEYS   = [c[0] for c in SYMPTOM_COLS]
SYM_LABELS = {c[0]: c[1] for c in SYMPTOM_COLS}

# High-acuity symptom set (escalation priority)
HIGH_ACUITY = {"difficulty_breathing", "bleeding_from_body_openings",
               "yellow_skin_yellow_eyes", "discolored_or_bloody_urine"}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(db_path: str) -> pd.DataFrame:
    """Load incident records from epihack-mini.db, mapping symptoms to numeric."""
    conn = sqlite3.connect(db_path)
    sym_cols_sql = ", ".join(SYM_KEYS)
    df = pd.read_sql_query(
        f"""
        SELECT date_of_report, age, sex, postal_code, county,
               {sym_cols_sql}
        FROM incidents
        WHERE date_of_report IS NOT NULL
        """,
        conn,
        parse_dates=["date_of_report"],
    )
    conn.close()

    # Map YES/NO/UNKNOWN → 1/0/NaN
    for col in SYM_KEYS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.upper()
                .map({"YES": 1.0, "NO": 0.0, "UNKNOWN": float("nan")})
                .astype("Float64")
            )

    # Age → numeric, coerce non-numeric to NaN
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # Assign age band
    def assign_band(age):
        if pd.isna(age):
            return "Unknown Age"
        for label, lo, hi in AGE_BANDS:
            if lo <= age <= hi:
                return label
        return "Unknown Age"
    df["age_band"] = df["age"].apply(assign_band)

    # Normalise sex
    def normalise_sex(s):
        s = str(s).strip().upper()
        if s in ("F", "FEMALE", "WOMAN"):
            return "Female"
        if s in ("M", "MALE", "MAN"):
            return "Male"
        return "Other/Unknown"
    df["sex_norm"] = df["sex"].apply(normalise_sex)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STRATA COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_strata(df: pd.DataFrame,
                   recent_days: int,
                   baseline_days: int,
                   min_n: int) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Compute per-stratum (age_band × sex) symptom rates for recent and baseline windows.

    Returns:
        strata_df — long-format DataFrame with rate columns per window
        ref_date  — reference date (DB max)
    """
    ref_date  = df["date_of_report"].max()
    rec_start = ref_date - timedelta(days=recent_days)
    bas_start = ref_date - timedelta(days=recent_days + baseline_days)
    bas_end   = rec_start

    df_rec = df[df["date_of_report"] > rec_start].copy()
    df_bas = df[(df["date_of_report"] > bas_start) &
                (df["date_of_report"] <= bas_end)].copy()

    total_rec_n = len(df_rec)
    total_bas_n = len(df_bas)

    rows = []
    for age_band in AGE_BAND_LABELS:
        for sex in SEX_CATEGORIES:
            sub_rec = df_rec[(df_rec["age_band"] == age_band) &
                             (df_rec["sex_norm"] == sex)]
            sub_bas = df_bas[(df_bas["age_band"] == age_band) &
                             (df_bas["sex_norm"] == sex)]

            rec_n = len(sub_rec)
            bas_n = len(sub_bas)

            # Reporter-share (% of total reporters in each window)
            rec_share = (rec_n / total_rec_n * 100) if total_rec_n > 0 else 0.0
            bas_share = (bas_n / total_bas_n * 100) if total_bas_n > 0 else 0.0
            share_shift = rec_share - bas_share

            for sym_key in SYM_KEYS:
                if sym_key not in sub_rec.columns:
                    continue

                # Prevalence rate = % of reporters who answered YES
                def rate(sub):
                    valid = sub[sym_key].dropna()
                    if len(valid) == 0:
                        return float("nan"), 0
                    return float(valid.mean() * 100), int(len(valid))

                rec_rate, rec_sym_n = rate(sub_rec)
                bas_rate, bas_sym_n = rate(sub_bas)

                # Relative rate increase (%)
                if bas_rate > 0:
                    rel_increase = (rec_rate - bas_rate) / bas_rate * 100
                else:
                    rel_increase = float("nan") if rec_rate == 0 else float("inf")

                # Sparse flag
                sparse = rec_n < min_n or bas_n < min_n

                rows.append({
                    "age_band":    age_band,
                    "sex":         sex,
                    "symptom":     sym_key,
                    "sym_label":   SYM_LABELS[sym_key],
                    "rec_n":       rec_n,
                    "bas_n":       bas_n,
                    "rec_share":   round(rec_share, 2),
                    "bas_share":   round(bas_share, 2),
                    "share_shift": round(share_shift, 2),
                    "rec_rate":    round(rec_rate, 3) if not np.isnan(rec_rate) else None,
                    "bas_rate":    round(bas_rate, 3) if not np.isnan(bas_rate) else None,
                    "rel_increase": round(rel_increase, 2) if np.isfinite(rel_increase) else None,
                    "abs_change":  (round(rec_rate - bas_rate, 3)
                                    if (rec_rate is not None and bas_rate is not None
                                        and not np.isnan(rec_rate) and not np.isnan(bas_rate))
                                    else None),
                    "sparse":      sparse,
                    "high_acuity": sym_key in HIGH_ACUITY,
                    "total_rec_n": total_rec_n,
                    "total_bas_n": total_bas_n,
                })

    return pd.DataFrame(rows), ref_date, total_rec_n, total_bas_n


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_signals(strata_df: pd.DataFrame,
                   rate_threshold: float,
                   share_threshold: float) -> pd.DataFrame:
    """
    Identify age×sex strata with:
      (A) symptom rate increased >rate_threshold% relative to baseline
      (B) stratum reporter-share shifted >share_threshold pp

    Excludes sparse strata (flagged). Returns only rows with at least one flag.
    """
    df = strata_df[~strata_df["sparse"]].copy()

    df["flag_rate"]  = df["rel_increase"].apply(
        lambda x: bool(x is not None and x > rate_threshold)
    )
    df["flag_share"] = df["share_shift"].apply(
        lambda x: abs(x) > share_threshold
    )
    df["flagged"]    = df["flag_rate"] | df["flag_share"]

    flagged = df[df["flagged"]].copy()
    return flagged.sort_values(
        ["flag_rate", "rel_increase"],
        ascending=[False, False]
    ).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL INTERPRETATION
# ─────────────────────────────────────────────────────────────────────────────

def assess_signal(row: pd.Series, all_flagged: pd.DataFrame) -> tuple[str, str]:
    """
    Heuristic assessment: true epidemiological signal vs reporting artifact.

    Decision logic:
      ARTIFACT signals:
        • Stratum reporter-share itself shifted >10pp → denominator change
        • Only one symptom elevated in stratum → isolated, less coherent
        • rec_n < 30 → small sample, high variance
        • rel_increase driven by base_rate < 2% → floor effect

      TRUE SIGNAL signals:
        • Multiple symptoms co-elevated in same stratum → syndromic coherence
        • High-acuity symptom elevated → severity marker
        • Stratum share stable but rates diverged → real burden increase
        • Consistent sex-differentiated pattern (e.g., fever elevated in Males
          but not Females in same age band) → possible occupational/behavioural

    Returns (assessment: str, recommendation: str)
    """
    age_band = row["age_band"]
    sex      = row["sex"]
    sym      = row["symptom"]

    # How many symptoms co-elevated in this stratum?
    same_stratum = all_flagged[
        (all_flagged["age_band"] == age_band) &
        (all_flagged["sex"] == sex) &
        (all_flagged["flag_rate"] == True)
    ]
    n_co_elevated = len(same_stratum)

    # Opposite-sex pattern check
    opposite_sex = all_flagged[
        (all_flagged["age_band"] == age_band) &
        (all_flagged["sex"] != sex) &
        (all_flagged["symptom"] == sym) &
        (all_flagged["flag_rate"] == True)
    ]
    sex_differential = len(opposite_sex) == 0   # True → only this sex shows elevation

    # Artifact indicators
    large_share_shift = abs(row["share_shift"]) > 10
    small_n           = row["rec_n"] < 30
    low_base_rate     = (row["bas_rate"] is not None and row["bas_rate"] < 2.0)
    single_symptom    = (n_co_elevated == 1 and row["flag_rate"])

    # Signal quality score (0–4)
    score = 0
    score += min(n_co_elevated - 1, 2)          # syndromic coherence (0–2)
    score += int(row["high_acuity"])             # high-acuity boost
    score += int(not large_share_shift)          # stable denominator
    score -= int(small_n)                        # penalise small n
    score -= int(low_base_rate)                  # penalise floor effect

    if score >= 2:
        assessment = "LIKELY TRUE SIGNAL"
        if row["high_acuity"]:
            rec = (
                f"[HIGH PRIORITY] Investigate {row['sym_label']} elevation in "
                f"{age_band} {sex} — high-acuity symptom with syndromic co-elevation. "
                f"Alert county health departments; consider active case-finding; "
                f"review ED and hospital discharge data for this demographic."
            )
        elif sex_differential:
            rec = (
                f"Investigate sex-differentiated {row['sym_label']} pattern in {age_band}. "
                f"Assess occupational exposure, healthcare-seeking behaviour, and "
                f"biological susceptibility differences. Cross-reference with "
                f"occupational health databases."
            )
        else:
            rec = (
                f"Increase surveillance frequency for {age_band} {sex}. "
                f"Prepare targeted public health messaging. Notify primary care "
                f"providers in high-reporter postal codes."
            )
    elif score >= 0:
        assessment = "UNCERTAIN — REQUIRES VALIDATION"
        rec = (
            f"Monitor {row['sym_label']} in {age_band} {sex} over next 7 days. "
            f"Collect additional data to distinguish true signal from random variation. "
            f"{'Small sample (n=' + str(row['rec_n']) + ') limits confidence. ' if small_n else ''}"
            f"{'Low baseline rate (<2%) may produce floor-effect artefacts. ' if low_base_rate else ''}"
            f"Cross-validate with wastewater and ED syndromic streams (see US#9)."
        )
    else:
        assessment = "LIKELY REPORTING ARTIFACT"
        reasons = []
        if large_share_shift:
            reasons.append(
                f"reporter-share shifted {row['share_shift']:+.1f}pp "
                f"(denominator change, not burden change)"
            )
        if small_n:
            reasons.append(f"small recent reporter count (n={row['rec_n']})")
        if low_base_rate:
            reasons.append(f"near-zero baseline rate ({row['bas_rate']:.1f}%) → floor effect")
        if single_symptom:
            reasons.append("isolated single-symptom elevation — no syndromic coherence")
        rec = (
            f"Interpret with caution: probable artifact due to {'; '.join(reasons)}. "
            f"No immediate public health action required. "
            f"Improve recruitment in {age_band} {sex} stratum to reduce reporting bias."
        )

    return assessment, rec


def annotate_signals(flagged_df: pd.DataFrame) -> pd.DataFrame:
    """Apply assess_signal() to every flagged row."""
    assessments = []
    recommendations = []
    for _, row in flagged_df.iterrows():
        a, r = assess_signal(row, flagged_df)
        assessments.append(a)
        recommendations.append(r)
    flagged_df = flagged_df.copy()
    flagged_df["assessment"]     = assessments
    flagged_df["recommendation"] = recommendations
    return flagged_df


# ─────────────────────────────────────────────────────────────────────────────
# LLM PROMPT TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

def build_llm_prompt(flagged_df: pd.DataFrame,
                     strata_df: pd.DataFrame,
                     ref_date: pd.Timestamp,
                     recent_days: int,
                     baseline_days: int,
                     rate_threshold: float,
                     share_threshold: float,
                     region: str = "Arizona") -> str:
    """
    Fill the LLM prompt template from User Story #10.
    Replace stub narrative with Anthropic API call in production:

        import anthropic
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2048,
            messages=[{"role": "user", "content": filled_prompt}]
        )
        narrative = message.content[0].text
    """
    # Build demographic_comparison_table for prompt injection
    if flagged_df.empty:
        table_str = "No strata exceeded detection thresholds."
    else:
        lines = [f"{'Age Band':<25} {'Sex':<15} {'Symptom':<28} "
                 f"{'Baseline%':>10} {'Recent%':>10} "
                 f"{'Rel.Increase%':>14} {'Share Shift pp':>15}"]
        lines.append("-" * 120)
        for _, row in flagged_df.head(20).iterrows():
            lines.append(
                f"{row['age_band']:<25} {row['sex']:<15} {row['sym_label']:<28} "
                f"{(str(round(row['bas_rate'], 1)) + '%') if row['bas_rate'] is not None else 'N/A':>10} "
                f"{(str(round(row['rec_rate'], 1)) + '%') if row['rec_rate'] is not None else 'N/A':>10} "
                f"{('+' if row['rel_increase'] and row['rel_increase'] > 0 else '') + str(round(row['rel_increase'], 1)) + '%' if row['rel_increase'] is not None else 'N/A':>14} "
                f"{('+' if row['share_shift'] > 0 else '') + str(round(row['share_shift'], 1)) + 'pp':>15}"
            )
        table_str = "\n".join(lines)

    prompt = f"""You are a demographic surveillance analyst. Compare symptom prevalence by age group and sex for the current {recent_days}-day window vs. the prior {baseline_days}-day baseline:

{table_str}

Reference date: {ref_date.date()} | Region: {region}
Detection thresholds: rate increase >{rate_threshold}% relative | reporter-share shift >{share_threshold}pp

Identify any age×sex strata where:
(1) symptom rates have increased by >{rate_threshold}% relative to baseline;
(2) the stratum's share of total reporters has shifted by >{share_threshold} percentage points.

For each identified shift:
- Describe the demographic pattern in epidemiological terms.
- Assess whether it likely reflects a TRUE EPIDEMIOLOGICAL SIGNAL or a REPORTING ARTIFACT.
- Recommend appropriate follow-up actions (public health response, data quality improvement, or further investigation).

Structure your output as:
SECTION 1: Strata with Rate Increases >{rate_threshold}% (sorted by relative increase)
SECTION 2: Strata with Reporter-Share Shifts >{share_threshold}pp
SECTION 3: High-Acuity Signals (if any)
SECTION 4: Recommended Actions Matrix (stratum | signal type | priority | action)
"""
    return prompt


def simulate_llm_narrative(flagged_df: pd.DataFrame,
                            strata_df: pd.DataFrame,
                            ref_date: pd.Timestamp,
                            recent_days: int,
                            baseline_days: int,
                            rate_threshold: float,
                            share_threshold: float,
                            total_rec_n: int,
                            total_bas_n: int) -> str:
    """Deterministic narrative — replace with Anthropic API call in production."""

    rate_flags   = flagged_df[flagged_df["flag_rate"]].copy()
    share_flags  = flagged_df[flagged_df["flag_share"]].copy()
    ha_flags     = flagged_df[flagged_df["high_acuity"] & flagged_df["flag_rate"]].copy()
    true_sigs    = flagged_df[flagged_df["assessment"] == "LIKELY TRUE SIGNAL"].copy()
    artifacts    = flagged_df[flagged_df["assessment"] == "LIKELY REPORTING ARTIFACT"].copy()
    uncertain    = flagged_df[flagged_df["assessment"] == "UNCERTAIN — REQUIRES VALIDATION"].copy()

    top_rate_row = rate_flags.iloc[0] if not rate_flags.empty else None
    top_share_row = share_flags.sort_values("share_shift", key=abs, ascending=False).iloc[0] \
                    if not share_flags.empty else None

    narrative = f"""DEMOGRAPHIC SURVEILLANCE ANALYSIS REPORT
Region: Arizona | Reference: {ref_date.date()} | Window: {recent_days}d recent / {baseline_days}d baseline
Total reporters: {total_rec_n} (recent) vs {total_bas_n} (baseline)
Generated: {ref_date.date()} | Classification: ADHS INTERNAL

━━━ EXECUTIVE SUMMARY ━━━

Stratified analysis of all 14 EpiHack symptom columns across {len(AGE_BAND_LABELS)} age bands
(Pediatric through Senior) and 3 sex categories identified:

  • {len(rate_flags)} stratum×symptom combinations with rate increase >{rate_threshold}% relative to 30d baseline
  • {len(share_flags)} strata with reporter-share shift >{share_threshold}pp between windows
  • {len(ha_flags)} high-acuity symptom elevations
  • {len(true_sigs)} assessed as LIKELY TRUE SIGNAL
  • {len(artifacts)} assessed as LIKELY REPORTING ARTIFACT
  • {len(uncertain)} assessed as UNCERTAIN — requires validation

━━━ SECTION 1: RATE INCREASE SIGNALS (>{rate_threshold}% relative) ━━━
"""
    if rate_flags.empty:
        narrative += "\nNo strata exceeded the relative rate increase threshold.\n"
    else:
        for i, (_, row) in enumerate(rate_flags.head(10).iterrows()):
            narrative += f"""
[{i + 1}] {row['age_band']} × {row['sex']} — {row['sym_label']}
    Baseline: {row['bas_rate']:.1f}% | Recent: {row['rec_rate']:.1f}% | Relative increase: +{row['rel_increase']:.1f}%
    Reporters: n={row['rec_n']} (recent), n={row['bas_n']} (baseline)
    Assessment: {row['assessment']}
    Action: {row['recommendation']}
"""

    narrative += f"""
━━━ SECTION 2: REPORTER-SHARE SHIFTS (>{share_threshold}pp) ━━━
"""
    if share_flags.empty:
        narrative += "\nNo strata exceeded the reporter-share shift threshold.\n"
    else:
        share_summary = (share_flags[["age_band", "sex", "share_shift"]]
                         .drop_duplicates(subset=["age_band", "sex"])
                         .sort_values("share_shift", key=abs, ascending=False))
        for _, row in share_summary.head(8).iterrows():
            direction = "INCREASED" if row["share_shift"] > 0 else "DECREASED"
            narrative += (
                f"  {row['age_band']} × {row['sex']}: share {direction} by "
                f"{row['share_shift']:+.1f}pp between baseline and recent window. "
                f"{'Denominator artifact likely — stratum participation changed, not disease burden. ' if abs(row['share_shift']) > 15 else ''}\n"
            )

    narrative += f"""
━━━ SECTION 3: HIGH-ACUITY SIGNALS ━━━
"""
    if ha_flags.empty:
        narrative += "\nNo high-acuity symptom elevations detected above threshold. Routine monitoring continues.\n"
    else:
        for _, row in ha_flags.iterrows():
            narrative += (
                f"  [ALERT] {row['sym_label']} in {row['age_band']} × {row['sex']}: "
                f"+{row['rel_increase']:.1f}% relative increase (n={row['rec_n']}). "
                f"Assessment: {row['assessment']}. {row['recommendation']}\n"
            )

    narrative += f"""
━━━ SECTION 4: RECOMMENDED ACTIONS MATRIX ━━━

Priority | Stratum                         | Signal              | Action
"""
    narrative += "-" * 100 + "\n"
    priority_order = {"LIKELY TRUE SIGNAL": 0,
                      "UNCERTAIN — REQUIRES VALIDATION": 1,
                      "LIKELY REPORTING ARTIFACT": 2}
    for _, row in flagged_df.sort_values(
        "assessment", key=lambda s: s.map(priority_order)
    ).drop_duplicates(subset=["age_band", "sex", "symptom"]).head(12).iterrows():
        pri_map = {"LIKELY TRUE SIGNAL": "HIGH",
                   "UNCERTAIN — REQUIRES VALIDATION": "MEDIUM",
                   "LIKELY REPORTING ARTIFACT": "LOW"}
        pri = pri_map.get(row["assessment"], "MEDIUM")
        action_short = row["recommendation"][:70] + "…" if len(row["recommendation"]) > 70 else row["recommendation"]
        narrative += (
            f"{pri:<8} | {row['age_band'][:20]:<20} × {row['sex']:<10} | "
            f"{row['sym_label']:<20} | {action_short}\n"
        )

    narrative += """
━━━ METHODOLOGICAL NOTE ━━━

Relative rate increases (%), not absolute percentage-point differences, are used as the
primary detection criterion. This avoids suppressing signals in low-prevalence symptoms
where a 2pp increase may represent a 100% relative rise. Reporter-share shifts are evaluated
at the stratum level — a 10pp shift in a stratum's proportion of total reporters typically
indicates participation change (recruitment artifact) rather than a genuine burden change,
unless corroborated by stable raw counts and elevated symptom rates.

Sparse strata (n<10 reporters in either window) are excluded from signal detection to
prevent floor-effect artifacts. All signals flagged as UNCERTAIN should be cross-validated
against wastewater surveillance (ADHS BEACON) and ED syndromic data (BioSense/NSSP ESSENCE)
as described in User Story #9 triangulation methodology.

[NOTE: EpiHack participatory surveillance is subject to digital-access bias: Pediatric
reporters are typically proxy-reported by parents/guardians; Senior strata may be
structurally under-represented due to lower digital literacy. Demographic shifts in
reporter composition should always be evaluated as potential participation artifacts before
public health escalation.]
"""
    return narrative


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(flagged_df: pd.DataFrame,
                 ref_date: pd.Timestamp,
                 recent_days: int,
                 rate_threshold: float,
                 share_threshold: float,
                 total_rec_n: int,
                 total_bas_n: int) -> None:
    sep = "─" * 120
    print()
    print(sep)
    print(f"  DEMOGRAPHIC SURVEILLANCE REPORT | Ref: {ref_date.date()} | "
          f"Window: {recent_days}d recent / 30d baseline")
    print(f"  Reporters: {total_rec_n} (recent) vs {total_bas_n} (baseline)")
    print(sep)
    print(f"{'STRATUM':<42} {'SYMPTOM':<28} {'BAS%':>6} {'REC%':>6} "
          f"{'REL%':>8} {'SHFT':>6} {'FLAG':>5} {'ASSESSMENT':<30}")
    print(sep)
    if flagged_df.empty:
        print("  No strata exceeded detection thresholds.")
    else:
        for _, row in flagged_df.iterrows():
            stratum = f"{row['age_band']} × {row['sex']}"[:41]
            flag_str = ("R" if row["flag_rate"] else " ") + ("S" if row["flag_share"] else " ")
            print(
                f"{stratum:<42} {row['sym_label']:<28} "
                f"{str(round(row['bas_rate'], 1)) + '%':>6} "
                f"{str(round(row['rec_rate'], 1)) + '%':>6} "
                f"{('+' if row['rel_increase'] > 0 else '') + str(round(row['rel_increase'], 1)) + '%':>8} "
                f"{('+' if row['share_shift'] > 0 else '') + str(round(row['share_shift'], 1)):>6} "
                f"{flag_str:>5} "
                f"{row['assessment'][:29]:<30}"
            )
    print(sep)
    rate_n  = int(flagged_df["flag_rate"].sum())  if not flagged_df.empty else 0
    share_n = int(flagged_df["flag_share"].sum()) if not flagged_df.empty else 0
    print(f"  Flagged: {rate_n} rate signals (R) | {share_n} share shifts (S) | "
          f"Total: {len(flagged_df)} rows")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION — MATPLOTLIB PNG
# ─────────────────────────────────────────────────────────────────────────────

def fig_demographic(strata_df: pd.DataFrame,
                    flagged_df: pd.DataFrame,
                    ref_date: pd.Timestamp,
                    recent_days: int,
                    baseline_days: int,
                    total_rec_n: int,
                    total_bas_n: int,
                    out_path: str) -> None:
    """
    Four-panel Matplotlib figure:
      A — Stacked reporter-share by age band (recent vs baseline)
      B — Heatmap: relative rate increase by age band × sex (top flagged symptom)
      C — Symptom prevalence by age band for top 6 symptoms (grouped bar)
      D — Sex-stratified comparison for top signals
    """
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "text.color": TEXT,
        "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
        "grid.color": "#2a2d3e", "font.family": "DejaVu Sans", "font.size": 9,
    })

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.97,
             f"Demographic Surveillance Analysis — Arizona",
             ha="center", va="top", fontsize=16, fontweight="bold", color=TEXT)
    fig.text(0.5, 0.945,
             f"Reference: {ref_date.date()} | Recent: {recent_days}d (n={total_rec_n}) | "
             f"Baseline: {baseline_days}d (n={total_bas_n}) | "
             f"Flagged strata: {len(flagged_df)}",
             ha="center", va="top", fontsize=9, color=MUTED)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.42, wspace=0.32,
                           top=0.91, bottom=0.07, left=0.07, right=0.97)

    # ── Panel A: Reporter-share stacked bar (age bands, recent vs baseline) ──
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor(SURFACE)
    ax_a.set_title("Panel A — Reporter Share by Age Band (Recent vs Baseline)",
                   color=TEXT, fontsize=10, fontweight="bold", pad=8)

    share_rec = []
    share_bas = []
    for age_band in AGE_BAND_LABELS:
        sub = strata_df[strata_df["age_band"] == age_band]
        # Sum across sex categories (share is per-row; need unique stratum share)
        rec_s = sub[sub["sex"] == "Female"]["rec_share"].values
        bas_s = sub[sub["sex"] == "Female"]["bas_share"].values
        share_rec.append(float(rec_s[0]) if len(rec_s) > 0 else 0.0)
        share_bas.append(float(bas_s[0]) if len(bas_s) > 0 else 0.0)

    # Sum across sex to get full-band share
    full_rec = []
    full_bas = []
    for age_band in AGE_BAND_LABELS:
        sub = strata_df[(strata_df["age_band"] == age_band) &
                        (strata_df["symptom"] == SYM_KEYS[0])]
        r = sub["rec_share"].sum()
        b = sub["bas_share"].sum()
        full_rec.append(r)
        full_bas.append(b)

    x    = np.arange(len(AGE_BAND_LABELS))
    w    = 0.35
    bars_rec = ax_a.bar(x - w / 2, full_rec, w, color=AGE_BAND_COLORS, alpha=0.85,
                        label="Recent")
    bars_bas = ax_a.bar(x + w / 2, full_bas, w, color=AGE_BAND_COLORS, alpha=0.40,
                        label="Baseline")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([b.split(" (")[0] for b in AGE_BAND_LABELS], fontsize=8,
                         rotation=20, ha="right")
    ax_a.set_ylabel("% of Total Reporters", color=TEXT, fontsize=8)
    ax_a.set_ylim(0, max(max(full_rec), max(full_bas)) * 1.25 if full_rec else 10)
    ax_a.legend(fontsize=7, facecolor=BG, edgecolor=MUTED, labelcolor=TEXT)
    ax_a.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars_rec, full_rec):
        ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=7, color=TEXT)

    # ── Panel B: Heatmap — relative rate increase by age×sex ─────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor(SURFACE)
    ax_b.set_title("Panel B — Relative Rate Increase Heatmap by Age×Sex\n(top flagged symptom per stratum)",
                   color=TEXT, fontsize=10, fontweight="bold", pad=8)

    # Pick top symptom per age×sex stratum by rel_increase
    if not flagged_df.empty:
        top_per_stratum = (
            flagged_df[flagged_df["flag_rate"]]
            .groupby(["age_band", "sex"])["rel_increase"]
            .max()
            .reset_index()
        )
    else:
        top_per_stratum = pd.DataFrame(columns=["age_band", "sex", "rel_increase"])

    mat = np.full((len(AGE_BAND_LABELS), len(SEX_CATEGORIES)), np.nan)
    for _, row in top_per_stratum.iterrows():
        ri = AGE_BAND_LABELS.index(row["age_band"]) if row["age_band"] in AGE_BAND_LABELS else -1
        ci = SEX_CATEGORIES.index(row["sex"]) if row["sex"] in SEX_CATEGORIES else -1
        if ri >= 0 and ci >= 0 and row["rel_increase"] is not None:
            mat[ri, ci] = row["rel_increase"]

    # Draw heatmap manually
    vmax = max(np.nanmax(mat), 1) if not np.all(np.isnan(mat)) else 100
    for ri in range(len(AGE_BAND_LABELS)):
        for ci in range(len(SEX_CATEGORIES)):
            val = mat[ri, ci]
            if np.isnan(val):
                colour = "#1a1d27"
                txt    = "—"
                txc    = MUTED
            else:
                intensity = min(val / vmax, 1.0)
                r_comp = int(255 * intensity)
                g_comp = int(71 * (1 - intensity))
                b_comp = int(87 * (1 - intensity))
                colour = f"#{r_comp:02x}{g_comp:02x}{b_comp:02x}"
                txt    = f"+{val:.0f}%"
                txc    = "white"
            rect = plt.Rectangle([ci - 0.5, ri - 0.5], 1, 1,
                                  fc=colour, ec=MUTED, linewidth=0.5)
            ax_b.add_patch(rect)
            ax_b.text(ci, ri, txt, ha="center", va="center", fontsize=8,
                     color=txc, fontweight="bold")

    ax_b.set_xlim(-0.5, len(SEX_CATEGORIES) - 0.5)
    ax_b.set_ylim(-0.5, len(AGE_BAND_LABELS) - 0.5)
    ax_b.set_xticks(range(len(SEX_CATEGORIES)))
    ax_b.set_xticklabels(SEX_CATEGORIES, fontsize=8)
    ax_b.set_yticks(range(len(AGE_BAND_LABELS)))
    ax_b.set_yticklabels([b.split(" (")[0] for b in AGE_BAND_LABELS], fontsize=8)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Sex", color=TEXT, fontsize=8)
    ax_b.set_ylabel("Age Band", color=TEXT, fontsize=8)

    # Colour bar proxy
    from matplotlib.colors import LinearSegmentedColormap
    cmap_proxy = LinearSegmentedColormap.from_list("rr", ["#1a1d27", RED])
    sm = plt.cm.ScalarMappable(cmap=cmap_proxy,
                                norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_b, fraction=0.04, pad=0.02)
    cbar.set_label("Max relative rate increase (%)", color=MUTED, fontsize=7)
    cbar.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED, fontsize=7)

    # ── Panel C: Top 6 symptoms, prevalence by age band ──────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor(SURFACE)
    ax_c.set_title("Panel C — Symptom Prevalence by Age Band\n(recent window, top 6 symptoms)",
                   color=TEXT, fontsize=10, fontweight="bold", pad=8)

    # Top 6 symptoms by statewide recent mean rate
    top6_syms = (
        strata_df.groupby("symptom")["rec_rate"]
        .mean()
        .dropna()
        .sort_values(ascending=False)
        .head(6)
        .index.tolist()
    )

    x3   = np.arange(len(AGE_BAND_LABELS))
    n6   = len(top6_syms)
    w6   = 0.13
    offsets = np.linspace(-(n6 - 1) * w6 / 2, (n6 - 1) * w6 / 2, n6)
    sym_clrs6 = [CYAN, GREEN, ACCENT, ORANGE, RED, PINK]

    for si, sym_key in enumerate(top6_syms):
        rates = []
        for age_band in AGE_BAND_LABELS:
            sub = strata_df[(strata_df["age_band"] == age_band) &
                            (strata_df["symptom"] == sym_key)]
            rates.append(float(sub["rec_rate"].mean(skipna=True))
                         if not sub["rec_rate"].isna().all() else 0.0)
        ax_c.bar(x3 + offsets[si], rates, w6,
                 color=sym_clrs6[si % len(sym_clrs6)],
                 alpha=0.85, label=SYM_LABELS.get(sym_key, sym_key))

    ax_c.set_xticks(x3)
    ax_c.set_xticklabels([b.split(" (")[0] for b in AGE_BAND_LABELS],
                         fontsize=8, rotation=20, ha="right")
    ax_c.set_ylabel("Prevalence % (reporters answering YES)", color=TEXT, fontsize=8)
    ax_c.legend(fontsize=6.5, facecolor=BG, edgecolor=MUTED, labelcolor=TEXT,
               ncol=2, loc="upper right")
    ax_c.grid(axis="y", alpha=0.3)

    # ── Panel D: Sex-stratified comparison for top 3 flagged strata ──────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor(SURFACE)
    ax_d.set_title("Panel D — Sex-Stratified Symptom Rates\n(baseline vs recent, top flagged strata)",
                   color=TEXT, fontsize=10, fontweight="bold", pad=8)

    if not flagged_df.empty and "flag_rate" in flagged_df.columns:
        top3 = flagged_df[flagged_df["flag_rate"]].head(6)
        if not top3.empty:
            cats = [f"{r['age_band'].split(' (')[0]}\n{r['sex']}\n{r['sym_label']}"
                    for _, r in top3.iterrows()]
            bas_rates = [r["bas_rate"] for _, r in top3.iterrows()]
            rec_rates = [r["rec_rate"] for _, r in top3.iterrows()]
            x4 = np.arange(len(cats))
            w4 = 0.35
            ax_d.bar(x4 - w4 / 2, bas_rates, w4, color=BLUE, alpha=0.7, label="Baseline (30d)")
            ax_d.bar(x4 + w4 / 2, rec_rates, w4, color=RED, alpha=0.85, label="Recent (7d)")

            for xi, (bas, rec) in enumerate(zip(bas_rates, rec_rates)):
                if bas and rec:
                    ax_d.annotate("",
                                  xy=(xi + w4 / 2, rec + 0.5),
                                  xytext=(xi - w4 / 2, bas + 0.5),
                                  arrowprops=dict(arrowstyle="->", color=ORANGE,
                                                  lw=1.2, connectionstyle="arc3,rad=0.3"))

            ax_d.set_xticks(x4)
            ax_d.set_xticklabels(cats, fontsize=6.5)
            ax_d.set_ylabel("Symptom Prevalence (%)", color=TEXT, fontsize=8)
            ax_d.legend(fontsize=7, facecolor=BG, edgecolor=MUTED, labelcolor=TEXT)
            ax_d.grid(axis="y", alpha=0.3)
        else:
            ax_d.text(0.5, 0.5, "No flagged rate signals\nin current window",
                     ha="center", va="center", color=MUTED, fontsize=11,
                     transform=ax_d.transAxes)
    else:
        ax_d.text(0.5, 0.5, "No flagged rate signals\nin current window",
                 ha="center", va="center", color=MUTED, fontsize=11,
                 transform=ax_d.transAxes)

    # Watermark
    fig.text(0.5, 0.01,
             "EpiHack Platform | ADHS Epidemiology Division | User Story #10",
             ha="center", va="bottom", fontsize=7, color=MUTED, style="italic")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[PNG] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# HTML DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def build_html(strata_df: pd.DataFrame,
               flagged_df: pd.DataFrame,
               ref_date: pd.Timestamp,
               recent_days: int,
               baseline_days: int,
               rate_threshold: float,
               share_threshold: float,
               total_rec_n: int,
               total_bas_n: int,
               llm_prompt: str,
               narrative: str,
               out_path: str) -> None:
    """Build self-contained interactive HTML dashboard."""
    import json

    def to_json_safe(obj):
        if isinstance(obj, dict):
            return {k: to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_json_safe(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, pd.NA.__class__) or obj is pd.NA:
            return None
        if obj is None:
            return None
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    # Build per-age-band reporter share data
    share_data = {}
    for age_band in AGE_BAND_LABELS:
        sub = strata_df[(strata_df["age_band"] == age_band) &
                        (strata_df["symptom"] == SYM_KEYS[0])]
        share_data[age_band] = {
            "rec_share": to_json_safe(float(sub["rec_share"].sum())),
            "bas_share": to_json_safe(float(sub["bas_share"].sum())),
        }

    # Build symptom prevalence by age band (recent window)
    prevalence_data = {}
    for sym_key in SYM_KEYS:
        prevalence_data[sym_key] = {}
        for age_band in AGE_BAND_LABELS:
            sub = strata_df[(strata_df["age_band"] == age_band) &
                            (strata_df["symptom"] == sym_key)]
            for sex in SEX_CATEGORIES:
                row = sub[sub["sex"] == sex]
                key = f"{age_band}||{sex}"
                if prevalence_data[sym_key].get(key) is None:
                    prevalence_data[sym_key][key] = {}
                prevalence_data[sym_key][key] = {
                    "rec_rate": to_json_safe(row["rec_rate"].values[0] if len(row) > 0 and row["rec_rate"].values[0] is not None else None),
                    "bas_rate": to_json_safe(row["bas_rate"].values[0] if len(row) > 0 and row["bas_rate"].values[0] is not None else None),
                    "rec_n":    int(row["rec_n"].values[0]) if len(row) > 0 else 0,
                    "sparse":   bool(row["sparse"].values[0]) if len(row) > 0 else True,
                }

    # Build heatmap matrix data (rel_increase per age×sex, max across symptoms)
    heatmap_data = {}
    for age_band in AGE_BAND_LABELS:
        for sex in SEX_CATEGORIES:
            key = f"{age_band}||{sex}"
            if flagged_df.empty:
                heatmap_data[key] = None
            else:
                sub = flagged_df[(flagged_df["age_band"] == age_band) &
                                 (flagged_df["sex"] == sex) &
                                 (flagged_df["flag_rate"] == True)]
                if sub.empty:
                    heatmap_data[key] = None
                else:
                    best = sub.loc[sub["rel_increase"].idxmax()]
                    heatmap_data[key] = {
                        "rel_increase": to_json_safe(best["rel_increase"]),
                        "symptom":      SYM_LABELS.get(best["symptom"], best["symptom"]),
                        "rec_rate":     to_json_safe(best["rec_rate"]),
                        "bas_rate":     to_json_safe(best["bas_rate"]),
                        "assessment":   best["assessment"],
                    }

    flagged_records = [to_json_safe(r) for r in flagged_df.to_dict(orient="records")] \
                      if not flagged_df.empty else []

    share_json      = json.dumps(share_data)
    prevalence_json = json.dumps(prevalence_data)
    heatmap_json    = json.dumps(heatmap_data)
    flagged_json    = json.dumps(flagged_records)
    narrative_html  = narrative.replace("\n", "<br>").replace("━", "—")
    prompt_html     = llm_prompt.replace("\n", "<br>")

    n_rate    = int(flagged_df["flag_rate"].sum())   if not flagged_df.empty else 0
    n_share   = int(flagged_df["flag_share"].sum())  if not flagged_df.empty else 0
    n_ha      = int((flagged_df["high_acuity"] & flagged_df["flag_rate"]).sum()) \
                if not flagged_df.empty else 0
    n_true    = int((flagged_df["assessment"] == "LIKELY TRUE SIGNAL").sum()) \
                if not flagged_df.empty else 0
    n_artifact= int((flagged_df["assessment"] == "LIKELY REPORTING ARTIFACT").sum()) \
                if not flagged_df.empty else 0

    age_band_colors_js = json.dumps(AGE_BAND_COLORS)
    sex_colors_js      = json.dumps(SEX_COLORS)
    age_band_labels_js = json.dumps(AGE_BAND_LABELS)
    sex_cats_js        = json.dumps(SEX_CATEGORIES)
    sym_keys_js        = json.dumps(SYM_KEYS)
    sym_labels_js      = json.dumps(SYM_LABELS)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Demographic Surveillance Analysis — Arizona</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:{BG}; --surface:{SURFACE}; --accent:{ACCENT};
    --red:{RED}; --orange:{ORANGE}; --green:{GREEN};
    --blue:{BLUE}; --cyan:{CYAN}; --pink:{PINK};
    --text:{TEXT}; --muted:{MUTED};
  }}
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
  .ctrl span{{color:var(--accent);font-weight:700;min-width:40px}}
  .dashboard{{display:grid;grid-template-columns:1fr 1fr;gap:20px;padding:20px 28px}}
  .panel{{background:var(--surface);border-radius:10px;padding:20px;border:1px solid #2a2d3e}}
  .panel-full{{grid-column:1/-1}}
  .panel h2{{font-size:.95rem;color:var(--accent);margin-bottom:14px;font-weight:600}}
  .canvas-wrap{{position:relative;height:300px}}
  .heatmap-grid{{display:grid;gap:4px;margin-top:8px}}
  .hm-cell{{border-radius:6px;padding:8px 4px;text-align:center;font-size:.75rem;font-weight:700;cursor:default;transition:transform .15s}}
  .hm-cell:hover{{transform:scale(1.05);z-index:10;position:relative}}
  table{{width:100%;border-collapse:collapse;font-size:.81rem;margin-top:8px}}
  thead th{{background:rgba(108,99,255,.15);color:var(--accent);padding:8px 10px;text-align:left;border-bottom:2px solid var(--accent);cursor:pointer;white-space:nowrap}}
  thead th:hover{{background:rgba(108,99,255,.25)}}
  tbody tr{{border-bottom:1px solid #2a2d3e;transition:background .15s}}
  tbody tr:hover{{background:rgba(108,99,255,.07)}}
  tbody td{{padding:7px 10px;vertical-align:top}}
  .badge{{display:inline-block;border-radius:12px;padding:2px 10px;font-size:.73rem;font-weight:700}}
  .sig-true{{background:rgba(255,71,87,.2);color:var(--red);border:1px solid var(--red)}}
  .sig-artifact{{background:rgba(67,233,123,.2);color:var(--green);border:1px solid var(--green)}}
  .sig-uncertain{{background:rgba(255,165,2,.2);color:var(--orange);border:1px solid var(--orange)}}
  .flag-r{{background:rgba(255,71,87,.15);color:var(--red);border-radius:4px;padding:1px 6px;font-size:.72rem}}
  .flag-s{{background:rgba(255,165,2,.15);color:var(--orange);border-radius:4px;padding:1px 6px;font-size:.72rem}}
  .narrative-box{{background:var(--bg);border:1px solid #2a2d3e;border-radius:8px;padding:18px;font-size:.83rem;line-height:1.75;font-family:'Courier New',monospace;max-height:500px;overflow-y:auto}}
  .prompt-box{{background:var(--bg);border:1px solid var(--accent);border-radius:8px;padding:14px;font-size:.77rem;line-height:1.6;color:var(--muted);max-height:300px;overflow-y:auto}}
  .summary-cards{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:16px}}
  .sc{{flex:1;min-width:120px;background:var(--bg);border-radius:8px;padding:14px 12px;border:1px solid #2a2d3e;text-align:center}}
  .sc-val{{font-size:2rem;font-weight:800}}
  .sc-lbl{{font-size:.76rem;color:var(--muted);margin-top:4px}}
  .filter-btn{{background:var(--bg);color:var(--text);border:1px solid #2a2d3e;border-radius:20px;padding:4px 14px;cursor:pointer;font-size:.78rem;transition:all .2s}}
  .filter-btn.active{{background:var(--accent);border-color:var(--accent);color:white}}
  footer{{text-align:center;padding:16px;color:var(--muted);font-size:.75rem;border-top:1px solid #2a2d3e}}
</style>
</head>
<body>
<header>
  <h1>Demographic Surveillance Analysis — Arizona</h1>
  <p>Age × Sex stratification | EpiHack Participatory Surveillance | {ref_date.date()}</p>
  <div class="meta-bar">
    <span class="chip">User Story #10</span>
    <span class="chip">Recent: {recent_days}d (n={total_rec_n})</span>
    <span class="chip">Baseline: {baseline_days}d (n={total_bas_n})</span>
    <span class="chip-red">Rate flags: {n_rate}</span>
    <span class="chip-orange">Share shifts: {n_share}</span>
    <span class="chip-red">High-acuity: {n_ha}</span>
    <span class="chip-green">True signals: {n_true}</span>
  </div>
</header>

<div class="controls">
  <div class="ctrl">
    <label>Rate threshold:</label>
    <input type="range" id="rateThresh" min="5" max="100" step="5" value="{int(rate_threshold)}"
           oninput="document.getElementById('rateVal').textContent=this.value+'%';filterTable()">
    <span id="rateVal">{int(rate_threshold)}%</span>
  </div>
  <div class="ctrl">
    <label>Share threshold:</label>
    <input type="range" id="shareThresh" min="2" max="30" step="1" value="{int(share_threshold)}"
           oninput="document.getElementById('shareVal').textContent=this.value+'pp';filterTable()">
    <span id="shareVal">{int(share_threshold)}pp</span>
  </div>
  <div class="ctrl">
    <label>Sex:</label>
    <select id="sexFilter" onchange="filterTable(); updatePrevalenceChart()">
      <option value="all">All</option>
      <option value="Female">Female</option>
      <option value="Male">Male</option>
      <option value="Other/Unknown">Other/Unknown</option>
    </select>
  </div>
  <div class="ctrl">
    <label>Symptom:</label>
    <select id="symFilter" onchange="filterTable(); updatePrevalenceChart()">
      <option value="all">All symptoms</option>
    </select>
  </div>
  <div class="ctrl" style="margin-left:auto">
    <button class="filter-btn active" onclick="setAssessFilter('all',this)">All</button>
    <button class="filter-btn" onclick="setAssessFilter('TRUE',this)">True Signals</button>
    <button class="filter-btn" onclick="setAssessFilter('ARTIFACT',this)">Artifacts</button>
    <button class="filter-btn" onclick="setAssessFilter('UNCERTAIN',this)">Uncertain</button>
  </div>
</div>

<div class="dashboard">

  <!-- Summary Cards -->
  <div class="panel panel-full">
    <div class="summary-cards">
      <div class="sc"><div class="sc-val" style="color:var(--accent)">{len(flagged_df)}</div><div class="sc-lbl">Flagged<br>Stratum×Symptoms</div></div>
      <div class="sc"><div class="sc-val" style="color:var(--red)">{n_rate}</div><div class="sc-lbl">Rate Increase<br>Signals</div></div>
      <div class="sc"><div class="sc-val" style="color:var(--orange)">{n_share}</div><div class="sc-lbl">Reporter-Share<br>Shifts</div></div>
      <div class="sc"><div class="sc-val" style="color:var(--red)">{n_ha}</div><div class="sc-lbl">High-Acuity<br>Elevations</div></div>
      <div class="sc"><div class="sc-val" style="color:var(--red)">{n_true}</div><div class="sc-lbl">Likely True<br>Signals</div></div>
      <div class="sc"><div class="sc-val" style="color:var(--green)">{n_artifact}</div><div class="sc-lbl">Likely<br>Artifacts</div></div>
      <div class="sc"><div class="sc-val" style="color:var(--orange)">{total_rec_n}</div><div class="sc-lbl">Recent<br>Reporters</div></div>
      <div class="sc"><div class="sc-val" style="color:var(--muted)">{total_bas_n}</div><div class="sc-lbl">Baseline<br>Reporters</div></div>
    </div>
  </div>

  <!-- Panel A: Reporter Share Chart -->
  <div class="panel">
    <h2>Panel A — Reporter Share by Age Band</h2>
    <div class="canvas-wrap"><canvas id="shareChart"></canvas></div>
  </div>

  <!-- Panel B: Heatmap -->
  <div class="panel">
    <h2>Panel B — Max Relative Rate Increase Heatmap (Age × Sex)</h2>
    <div id="heatmapGrid"></div>
  </div>

  <!-- Panel C: Prevalence by Age Band -->
  <div class="panel">
    <h2>Panel C — Symptom Prevalence by Age Band <span style="color:var(--muted);font-size:.75rem">(recent window)</span></h2>
    <div class="canvas-wrap"><canvas id="prevalenceChart"></canvas></div>
  </div>

  <!-- Panel D: Sex-Stratified Flagged Signals -->
  <div class="panel">
    <h2>Panel D — Baseline vs Recent: Top Flagged Strata</h2>
    <div class="canvas-wrap"><canvas id="sigChart"></canvas></div>
  </div>

  <!-- Panel E: Flagged Signal Table -->
  <div class="panel panel-full">
    <h2>Panel E — Flagged Strata Decision Table <span id="flagCount" style="color:var(--muted);font-size:.82rem"></span></h2>
    <div id="flagTableContainer"></div>
  </div>

  <!-- Panel F: Narrative -->
  <div class="panel panel-full">
    <h2>Panel F — Demographic Surveillance Narrative (LLM-generated)</h2>
    <div class="narrative-box">{narrative_html}</div>
  </div>

  <!-- Panel G: LLM Prompt -->
  <div class="panel panel-full">
    <h2>Panel G — LLM Prompt Template (filled)</h2>
    <div class="prompt-box">{prompt_html}</div>
  </div>

</div>
<footer>
  EpiHack Participatory Epidemiological Surveillance Platform &nbsp;|&nbsp;
  Arizona ADHS · User Story #10 — Demographic Surveillance Analysis &nbsp;|&nbsp; {ref_date.date()}
</footer>

<script>
// ── Embedded data ──────────────────────────────────────────────────────────
const SHARE_DATA     = {share_json};
const PREVALENCE_DATA= {prevalence_json};
const HEATMAP_DATA   = {heatmap_json};
const FLAGGED_DATA   = {flagged_json};
const AGE_BANDS      = {age_band_labels_js};
const SEX_CATS       = {sex_cats_js};
const SYM_KEYS       = {sym_keys_js};
const SYM_LABELS     = {sym_labels_js};
const AGE_COLORS     = {age_band_colors_js};
const SEX_COLORS     = {sex_colors_js};
const BG="{BG}",SURFACE="{SURFACE}",ACCENT="{ACCENT}",RED="{RED}",
      ORANGE="{ORANGE}",GREEN="{GREEN}",BLUE="{BLUE}",CYAN="{CYAN}",
      PINK="{PINK}",TEXT="{TEXT}",MUTED="{MUTED}";

let assessFilter = "all";
let prevChart = null;

// Populate symptom filter
(function() {{
  const sel = document.getElementById("symFilter");
  SYM_KEYS.forEach(k => {{
    const opt = document.createElement("option");
    opt.value = k; opt.textContent = SYM_LABELS[k];
    sel.appendChild(opt);
  }});
}})();

// ── Panel A: Reporter Share ────────────────────────────────────────────────
(function() {{
  const ctx  = document.getElementById("shareChart").getContext("2d");
  const rec  = AGE_BANDS.map(b => SHARE_DATA[b]?.rec_share ?? 0);
  const bas  = AGE_BANDS.map(b => SHARE_DATA[b]?.bas_share ?? 0);
  const lbls = AGE_BANDS.map(b => b.split(" (")[0]);
  new Chart(ctx, {{
    type:"bar",
    data:{{
      labels: lbls,
      datasets:[
        {{label:"Recent ({recent_days}d)",  data:rec, backgroundColor:AGE_COLORS.map(c=>c+"cc"), borderColor:AGE_COLORS, borderWidth:1}},
        {{label:"Baseline ({baseline_days}d)", data:bas, backgroundColor:AGE_COLORS.map(c=>c+"44"), borderColor:AGE_COLORS, borderWidth:1, borderDash:[4,2]}},
      ]
    }},
    options:{{
      responsive:true, maintainAspectRatio:false,
      plugins:{{ legend:{{labels:{{color:TEXT,boxWidth:12}}}}, tooltip:{{callbacks:{{label:c=>` ${{c.dataset.label}}: ${{c.raw?.toFixed(1)}}%`}}}} }},
      scales:{{
        x:{{ticks:{{color:TEXT,font:{{size:9}}}},grid:{{color:"#2a2d3e"}}}},
        y:{{ticks:{{color:TEXT,callback:v=>v+"%"}},grid:{{color:"#2a2d3e"}},
           title:{{display:true,text:"% of Total Reporters",color:MUTED}}}}
      }}
    }}
  }});
}})();

// ── Panel B: Heatmap ───────────────────────────────────────────────────────
(function() {{
  const container = document.getElementById("heatmapGrid");
  const nAge = AGE_BANDS.length;
  const nSex = SEX_CATS.length;

  // Compute vmax
  let vmax = 0;
  Object.values(HEATMAP_DATA).forEach(v => {{
    if (v && v.rel_increase) vmax = Math.max(vmax, v.rel_increase);
  }});
  if (vmax === 0) vmax = 100;

  // Header row
  let html = `<div style="display:grid;grid-template-columns:160px repeat(${{nSex}},1fr);gap:4px;margin-top:8px">`;
  html += `<div style="color:var(--muted);font-size:.75rem;align-self:end">Age Band</div>`;
  SEX_CATS.forEach((s,i) => {{
    html += `<div style="text-align:center;color:${{SEX_COLORS[i]}};font-size:.8rem;font-weight:700">${{s}}</div>`;
  }});

  AGE_BANDS.forEach((band, ri) => {{
    html += `<div style="color:var(--text);font-size:.78rem;padding:4px 0">${{band.split(" (")[0]}}</div>`;
    SEX_CATS.forEach((sex, ci) => {{
      const key = `${{band}}||${{sex}}`;
      const d   = HEATMAP_DATA[key];
      let bg, txt, tip;
      if (!d || d.rel_increase === null) {{
        bg = "#1a1d27"; txt = "—"; tip = "No signal";
      }} else {{
        const intensity = Math.min(d.rel_increase / vmax, 1);
        const r = Math.round(255 * intensity);
        const g = Math.round(71 * (1 - intensity));
        const b = Math.round(87 * (1 - intensity));
        bg  = `rgb(${{r}},${{g}},${{b}})`;
        txt = `+${{d.rel_increase?.toFixed(0)}}%`;
        tip = `${{band}} × ${{sex}}: ${{d.symptom}} — +${{d.rel_increase?.toFixed(1)}}% rel. increase\\nBaseline: ${{d.bas_rate?.toFixed(1)}}% → Recent: ${{d.rec_rate?.toFixed(1)}}%\\n${{d.assessment}}`;
      }}
      html += `<div class="hm-cell" style="background:${{bg}};color:white" title="${{tip}}">${{txt}}</div>`;
    }});
  }});
  html += `</div>`;

  // Legend
  html += `<div style="margin-top:10px;font-size:.73rem;color:var(--muted)">
    Colour intensity → relative rate increase magnitude. "—" = no rate signal above threshold or sparse stratum.<br>
    Hover cells for detail: symptom, rates, and signal assessment.
  </div>`;
  container.innerHTML = html;
}})();

// ── Panel C: Prevalence Chart ──────────────────────────────────────────────
function updatePrevalenceChart() {{
  const ctx   = document.getElementById("prevalenceChart");
  const sex   = document.getElementById("sexFilter").value;
  const symK  = document.getElementById("symFilter").value;

  // Compute top 6 symptoms by mean recent rate across all strata
  const symMeans = SYM_KEYS.map(k => {{
    let sum = 0, cnt = 0;
    AGE_BANDS.forEach(b => SEX_CATS.forEach(s => {{
      const key = `${{b}}||${{s}}`;
      const d = PREVALENCE_DATA[k]?.[key];
      if (d && d.rec_rate !== null && !d.sparse) {{ sum += d.rec_rate; cnt++; }}
    }}));
    return {{key: k, mean: cnt > 0 ? sum / cnt : 0}};
  }}).sort((a,b) => b.mean - a.mean).slice(0,6).map(x => x.key);

  const targetSyms = (symK !== "all") ? [symK] : symMeans;
  const targetSexes = (sex !== "all") ? [sex] : SEX_CATS;
  const lbls = AGE_BANDS.map(b => b.split(" (")[0]);

  const datasets = [];
  const clrs = [CYAN, GREEN, ACCENT, ORANGE, RED, PINK, BLUE];
  targetSyms.forEach((k, si) => {{
    targetSexes.forEach((s, xi) => {{
      const data = AGE_BANDS.map(b => {{
        const key = `${{b}}||${{s}}`;
        const d   = PREVALENCE_DATA[k]?.[key];
        return (d && !d.sparse) ? d.rec_rate : null;
      }});
      datasets.push({{
        label: `${{SYM_LABELS[k]}} (${{s}})`,
        data,
        backgroundColor: clrs[si % clrs.length] + (targetSexes.length > 1 && xi === 1 ? "55" : "cc"),
        borderColor:     clrs[si % clrs.length],
        borderWidth: 1,
        borderDash: xi === 1 ? [4, 2] : [],
      }});
    }});
  }});

  if (prevChart) prevChart.destroy();
  prevChart = new Chart(ctx.getContext("2d"), {{
    type:"bar",
    data:{{ labels: lbls, datasets }},
    options:{{
      responsive:true, maintainAspectRatio:false,
      plugins:{{ legend:{{labels:{{color:TEXT,boxWidth:10,font:{{size:9}}}}}}, tooltip:{{callbacks:{{label:c=>` ${{c.dataset.label}}: ${{c.raw?.toFixed(1)}}%`}}}} }},
      scales:{{
        x:{{ticks:{{color:TEXT,font:{{size:8}}}},grid:{{color:"#2a2d3e"}}}},
        y:{{ticks:{{color:TEXT,callback:v=>v+"%"}},grid:{{color:"#2a2d3e"}},
           title:{{display:true,text:"Prevalence % (YES reporters)",color:MUTED}}}}
      }}
    }}
  }});
}}
updatePrevalenceChart();

// ── Panel D: Flagged signal comparison ────────────────────────────────────
(function() {{
  const ctx = document.getElementById("sigChart").getContext("2d");
  const top = FLAGGED_DATA.filter(r => r.flag_rate).slice(0, 6);
  if (top.length === 0) {{
    ctx.canvas.parentElement.innerHTML = "<p style='color:var(--muted);text-align:center;padding:40px'>No rate-flagged signals in current window.</p>";
    return;
  }}
  const lbls  = top.map(r => `${{r.age_band?.split(" (")[0]}}\\n${{r.sex}}\\n${{r.sym_label}}`);
  const bas   = top.map(r => r.bas_rate);
  const rec   = top.map(r => r.rec_rate);
  new Chart(ctx, {{
    type:"bar",
    data:{{
      labels: lbls,
      datasets:[
        {{label:"Baseline ({baseline_days}d)", data:bas, backgroundColor:BLUE+"99", borderColor:BLUE, borderWidth:1}},
        {{label:"Recent ({recent_days}d)",  data:rec, backgroundColor:RED+"bb",  borderColor:RED,  borderWidth:1}},
      ]
    }},
    options:{{
      responsive:true, maintainAspectRatio:false,
      plugins:{{ legend:{{labels:{{color:TEXT,boxWidth:12}}}}, tooltip:{{callbacks:{{label:c=>` ${{c.dataset.label}}: ${{c.raw?.toFixed(1)}}%`}}}} }},
      scales:{{
        x:{{ticks:{{color:TEXT,font:{{size:8,lineHeight:1.2}}}},grid:{{color:"#2a2d3e"}}}},
        y:{{ticks:{{color:TEXT,callback:v=>v+"%"}},grid:{{color:"#2a2d3e"}},
           title:{{display:true,text:"Symptom Prevalence %",color:MUTED}}}}
      }}
    }}
  }});
}})();

// ── Panel E: Decision Table ────────────────────────────────────────────────
function setAssessFilter(val, btn) {{
  assessFilter = val;
  document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  filterTable();
}}

function filterTable() {{
  const rateT  = parseFloat(document.getElementById("rateThresh").value);
  const shareT = parseFloat(document.getElementById("shareThresh").value);
  const sex    = document.getElementById("sexFilter").value;
  const sym    = document.getElementById("symFilter").value;

  let data = FLAGGED_DATA.filter(r => {{
    const matchRate  = r.flag_rate && r.rel_increase > rateT;
    const matchShare = r.flag_share && Math.abs(r.share_shift) > shareT;
    if (!matchRate && !matchShare) return false;
    if (sex !== "all" && r.sex !== sex) return false;
    if (sym !== "all" && r.symptom !== sym) return false;
    if (assessFilter === "TRUE"     && !r.assessment?.includes("TRUE")) return false;
    if (assessFilter === "ARTIFACT" && !r.assessment?.includes("ARTIFACT")) return false;
    if (assessFilter === "UNCERTAIN"&& !r.assessment?.includes("UNCERTAIN")) return false;
    return true;
  }});

  document.getElementById("flagCount").textContent = `(${{data.length}} rows)`;

  const container = document.getElementById("flagTableContainer");
  if (data.length === 0) {{
    container.innerHTML = "<p style='color:var(--muted);padding:20px;text-align:center'>No results match current filters.</p>";
    return;
  }}

  function assessClass(a) {{
    if (!a) return "";
    if (a.includes("TRUE"))     return "sig-true";
    if (a.includes("ARTIFACT")) return "sig-artifact";
    return "sig-uncertain";
  }}

  let html = `<table><thead><tr>
    <th>Age Band</th><th>Sex</th><th>Symptom</th>
    <th>Baseline %</th><th>Recent %</th><th>Rel. Increase</th>
    <th>Share Shift</th><th>n (recent)</th><th>Flags</th><th>Assessment</th>
  </tr></thead><tbody>`;

  data.forEach(r => {{
    const relStr = r.rel_increase !== null
      ? `<span style="color:${{r.rel_increase > rateT ? RED : MUTED}}">${{r.rel_increase > 0 ? "+" : ""}}${{r.rel_increase?.toFixed(1)}}%</span>`
      : "N/A";
    const shareStr = `<span style="color:${{Math.abs(r.share_shift) > shareT ? ORANGE : MUTED}}">${{r.share_shift > 0 ? "+" : ""}}${{r.share_shift?.toFixed(1)}}pp</span>`;
    const flags = (r.flag_rate ? '<span class="flag-r">R↑</span> ' : '') +
                  (r.flag_share ? '<span class="flag-s">S⇔</span>' : '');
    html += `<tr>
      <td style="font-weight:600">${{r.age_band?.split(" (")[0]}}</td>
      <td>${{r.sex}}</td>
      <td>${{r.sym_label}}</td>
      <td>${{r.bas_rate?.toFixed(1)}}%</td>
      <td>${{r.rec_rate?.toFixed(1)}}%</td>
      <td>${{relStr}}</td>
      <td>${{shareStr}}</td>
      <td>${{r.rec_n}}</td>
      <td>${{flags}}</td>
      <td><span class="badge ${{assessClass(r.assessment)}}">${{r.assessment?.split(" — ")[0]}}</span></td>
    </tr>`;
  }});
  html += "</tbody></table>";
  container.innerHTML = html;
}}
filterTable();
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
        description="US#10 — Demographic Surveillance Report (Age × Sex stratification)"
    )
    parser.add_argument("--db",             required=True)
    parser.add_argument("--recent",         type=int,   default=7)
    parser.add_argument("--baseline",       type=int,   default=30)
    parser.add_argument("--min-n",          type=int,   default=10)
    parser.add_argument("--rate-threshold", type=float, default=20.0)
    parser.add_argument("--share-threshold",type=float, default=10.0)
    parser.add_argument("--out",            default="demographic_surveillance_report.png")
    args = parser.parse_args()

    png_stem = os.path.splitext(os.path.basename(args.out))[0]
    if os.path.isdir(args.out):
        png_path  = os.path.join(args.out, f"{png_stem}.png")
        html_path = os.path.join(args.out, f"{png_stem}.html")
    else:
        png_path  = args.out if args.out.endswith(".png") else args.out + ".png"
        base_dir  = os.path.dirname(png_path) or "."
        html_path = os.path.join(base_dir, f"{png_stem}.html")

    print()
    print("=" * 60)
    print("  US#10 — Demographic Surveillance Analysis")
    print(f"  DB: {args.db}")
    print(f"  Window: {args.recent}d recent / {args.baseline}d baseline")
    print(f"  Min reporters: {args.min_n}")
    print(f"  Thresholds: rate >{args.rate_threshold}% rel | share >{args.share_threshold}pp")
    print("=" * 60)

    print("\n[1/5] Loading and preprocessing data …")
    df = load_data(args.db)
    print(f"      Total records: {len(df):,} | "
          f"Date range: {df['date_of_report'].min().date()} – "
          f"{df['date_of_report'].max().date()}")

    print("\n[2/5] Computing age×sex strata …")
    strata_df, ref_date, total_rec_n, total_bas_n = compute_strata(
        df, args.recent, args.baseline, args.min_n
    )
    print(f"      Ref date: {ref_date.date()}")
    print(f"      Strata rows: {len(strata_df):,} "
          f"({len(AGE_BAND_LABELS)} age bands × {len(SEX_CATEGORIES)} sex × "
          f"{len(SYM_KEYS)} symptoms)")
    print(f"      Recent reporters: {total_rec_n} | Baseline: {total_bas_n}")
    sparse_n = strata_df["sparse"].sum()
    print(f"      Sparse strata excluded: {sparse_n:,} rows "
          f"({sparse_n / len(strata_df) * 100:.1f}%)")

    print("\n[3/5] Detecting signals …")
    flagged_df = detect_signals(strata_df, args.rate_threshold, args.share_threshold)
    if not flagged_df.empty:
        flagged_df = annotate_signals(flagged_df)
    rate_n  = int(flagged_df["flag_rate"].sum())  if not flagged_df.empty else 0
    share_n = int(flagged_df["flag_share"].sum()) if not flagged_df.empty else 0
    ha_n    = int((flagged_df["high_acuity"] & flagged_df["flag_rate"]).sum()) \
              if not flagged_df.empty else 0
    print(f"      Rate signals (>{args.rate_threshold}% rel): {rate_n}")
    print(f"      Share shifts (>{args.share_threshold}pp):   {share_n}")
    print(f"      High-acuity elevations:          {ha_n}")

    print_report(flagged_df, ref_date, args.recent,
                 args.rate_threshold, args.share_threshold,
                 total_rec_n, total_bas_n)

    print("\n[4/5] Building LLM prompt and narrative …")
    llm_prompt = build_llm_prompt(
        flagged_df, strata_df, ref_date,
        args.recent, args.baseline,
        args.rate_threshold, args.share_threshold
    )
    narrative = simulate_llm_narrative(
        flagged_df, strata_df, ref_date,
        args.recent, args.baseline,
        args.rate_threshold, args.share_threshold,
        total_rec_n, total_bas_n
    )

    print("\n[5/5] Generating outputs …")
    fig_demographic(strata_df, flagged_df, ref_date,
                    args.recent, args.baseline,
                    total_rec_n, total_bas_n, png_path)

    build_html(strata_df, flagged_df, ref_date,
               args.recent, args.baseline,
               args.rate_threshold, args.share_threshold,
               total_rec_n, total_bas_n,
               llm_prompt, narrative, html_path)

    print()
    print("All outputs saved.")
    print(f"  PNG  → {png_path}")
    print(f"  HTML → {html_path}")


if __name__ == "__main__":
    main()
