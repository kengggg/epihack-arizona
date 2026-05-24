#!/usr/bin/env python3
"""
multi_source_triangulation.py — User Story #9
EpiHack Participatory Epidemiological Surveillance Platform

Multi-source surveillance triangulation analyst:
  Source 1 — EpiHack community participatory signals (epihack-mini.db)
  Source 2 — ADHS Wastewater Surveillance (deterministic simulation)
  Source 3 — BioSense/NSSP Emergency Department syndromic data (deterministic simulation)

For each syndrome with concordant signals across 2+ sources:
  • Describe concordance level
  • Assign Tier 1 (all 3), Tier 2 (2 of 3), Tier 3 (single source)
  • Recommend response level per tier

Outputs:
  1. Terminal decision table
  2. multi_source_triangulation.png  — four-panel Matplotlib figure
  3. multi_source_triangulation.html — interactive self-contained dashboard

Usage:
  python multi_source_triangulation.py \
    --db epihack-mini.db \
    --recent 7 \
    --baseline 30 \
    --min-reporters 5 \
    --region "Arizona" \
    --out multi_source_triangulation.png

Author: EpiHack Platform  |  Arizona Department of Health Services (simulation)
"""

import argparse
import hashlib
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
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Design system (matches prior User Stories)
BG        = "#0f1117"
SURFACE   = "#1a1d27"
ACCENT    = "#6c63ff"
RED       = "#ff4757"
ORANGE    = "#ffa502"
GREEN     = "#43e97b"
YELLOW    = "#ffd32a"
BLUE      = "#1e90ff"
CYAN      = "#00d2d3"
TEXT      = "#e8eaf6"
MUTED     = "#8892a4"

# Source labels
SRC_EPIHACK    = "EpiHack"
SRC_WASTEWATER = "Wastewater (ADHS)"
SRC_ED         = "ED Syndromic (NSSP)"

# Signal threshold for EpiHack elevation
PCT_THRESH = 5.0   # %-point above baseline → "up" or "new"
MIN_REC    = 2
MIN_BAS    = 5

# Syndrome definitions
# Each syndrome maps to: EpiHack symptom columns, wastewater pathogen key, ED category key
SYNDROMES = {
    "Respiratory-ILI": {
        "label":       "Respiratory — Influenza-Like Illness",
        "epihack_cols": ["cough_congestion", "sore_throat"],
        "ww_pathogen":  "influenza",
        "ed_category":  "ili",
        "color":        BLUE,
    },
    "Respiratory-COVID": {
        "label":       "Respiratory — COVID-Like Illness",
        "epihack_cols": ["difficulty_breathing", "loss_of_smell_or_taste"],
        "ww_pathogen":  "sars_cov2",
        "ed_category":  "cli",
        "color":        CYAN,
    },
    "Gastrointestinal": {
        "label":       "Gastrointestinal Illness",
        "epihack_cols": ["nauseas_vomiting", "diarrhea"],
        "ww_pathogen":  "norovirus",
        "ed_category":  "gi",
        "color":        ORANGE,
    },
    "Fever-Systemic": {
        "label":       "Fever / Systemic Illness",
        "epihack_cols": ["fever", "chills", "muscle_or_body_aches_and_pains"],
        "ww_pathogen":  "rsv",
        "ed_category":  "fever_nos",
        "color":        RED,
    },
    "High-Acuity": {
        "label":       "High-Acuity / Hemorrhagic",
        "epihack_cols": ["bleeding_from_body_openings",
                         "yellow_skin_yellow_eyes",
                         "discolored_or_bloody_urine"],
        "ww_pathogen":  None,   # no direct wastewater analogue
        "ed_category":  "hemorrhagic",
        "color":        "#ff6b6b",
    },
    "Dermal-Ocular": {
        "label":       "Dermal / Ocular",
        "epihack_cols": ["rash", "red_eyes"],
        "ww_pathogen":  None,
        "ed_category":  "rash_conjunctivitis",
        "color":        GREEN,
    },
}

# All EpiHack symptom columns (canonical names)
SYMPTOM_COLS = [
    "cough_congestion",
    "sore_throat",
    "difficulty_breathing",
    "nauseas_vomiting",
    "diarrhea",
    "muscle_or_body_aches_and_pains",
    "fever",
    "chills",
    "red_eyes",
    "rash",
    "loss_of_smell_or_taste",
    "bleeding_from_body_openings",
    "yellow_skin_yellow_eyes",
    "discolored_or_bloody_urine",
]

# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISTIC SIMULATION — WASTEWATER (ADHS)
# ─────────────────────────────────────────────────────────────────────────────

# Arizona county base rates for wastewater pathogen concentration (0–100 scale)
# Calibrated to represent plausible 2025–2026 surveillance signal levels
# (annual peak values; seasonal multipliers applied at runtime)
COUNTY_WW_BASE = {
    "Maricopa":   {"influenza": 55, "sars_cov2": 62, "norovirus": 38, "rsv": 48},
    "Pima":       {"influenza": 50, "sars_cov2": 58, "norovirus": 34, "rsv": 44},
    "Pinal":      {"influenza": 42, "sars_cov2": 50, "norovirus": 30, "rsv": 38},
    "Yavapai":    {"influenza": 38, "sars_cov2": 44, "norovirus": 26, "rsv": 34},
    "Coconino":   {"influenza": 35, "sars_cov2": 40, "norovirus": 24, "rsv": 30},
    "Mohave":     {"influenza": 34, "sars_cov2": 38, "norovirus": 22, "rsv": 29},
    "Yuma":       {"influenza": 44, "sars_cov2": 48, "norovirus": 40, "rsv": 34},
    "Apache":     {"influenza": 28, "sars_cov2": 26, "norovirus": 46, "rsv": 24},
    "Navajo":     {"influenza": 26, "sars_cov2": 24, "norovirus": 44, "rsv": 22},
    "Cochise":    {"influenza": 36, "sars_cov2": 42, "norovirus": 26, "rsv": 32},
    "Gila":       {"influenza": 30, "sars_cov2": 34, "norovirus": 28, "rsv": 26},
    "Graham":     {"influenza": 28, "sars_cov2": 32, "norovirus": 32, "rsv": 24},
    "Greenlee":   {"influenza": 26, "sars_cov2": 28, "norovirus": 34, "rsv": 20},
    "La Paz":     {"influenza": 30, "sars_cov2": 32, "norovirus": 28, "rsv": 25},
    "Santa Cruz": {"influenza": 38, "sars_cov2": 42, "norovirus": 32, "rsv": 32},
}

# Arizona county population estimates (2023 ACS) for population-weighted wastewater means
COUNTY_POP = {
    "Maricopa":   4500000, "Pima":      1100000, "Pinal":      470000,
    "Yavapai":     245000, "Coconino":   145000, "Mohave":     215000,
    "Yuma":        220000, "Apache":      72000, "Navajo":      110000,
    "Cochise":     126000, "Gila":        55000, "Graham":       40000,
    "Greenlee":     10000, "La Paz":      22000, "Santa Cruz":   48000,
}

# Month multipliers for seasonal patterns
MONTH_WW_MULT = {
    1: 1.6, 2: 1.5, 3: 1.2, 4: 0.9, 5: 0.8, 6: 0.7,
    7: 0.7, 8: 0.8, 9: 0.9, 10: 1.0, 11: 1.3, 12: 1.5,
}

def _md5_float(seed_str: str) -> float:
    """Return a deterministic float in [0, 1) from a string seed."""
    digest = hashlib.md5(seed_str.encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def simulate_wastewater(reference_date: pd.Timestamp, counties: list[str]) -> dict:
    """
    Simulate ADHS wastewater surveillance signals for each county × pathogen.

    Returns:
        dict: {
            pathogen_key: {
                "concentration": float (0-100),
                "trend":         str ("rising" | "stable" | "falling"),
                "signal":        bool  (True = elevated above seasonal baseline),
                "counties_elevated": list[str],
            }
        }
    """
    month = reference_date.month
    mult  = MONTH_WW_MULT.get(month, 1.0)
    result = {}

    pathogens = ["influenza", "sars_cov2", "norovirus", "rsv"]
    total_pop = sum(COUNTY_POP.get(c, 50000) for c in counties)

    for pathogen in pathogens:
        concs   = []
        weights = []
        elevated_counties = []

        for county in counties:
            base   = COUNTY_WW_BASE.get(county, {}).get(pathogen, 25)
            noise  = _md5_float(f"{county}{month}{pathogen}ww") * 16 - 8
            conc   = max(0, min(100, base * mult + noise))
            pop    = COUNTY_POP.get(county, 50000)
            concs.append(conc)
            weights.append(pop)
            if conc > base * mult * 0.85:          # 15% above expected seasonal level
                elevated_counties.append(county)

        # Population-weighted mean concentration
        mean_conc = float(np.average(concs, weights=weights))

        # Prior-month population-weighted mean
        prior_mult  = MONTH_WW_MULT.get(month - 1 if month > 1 else 12, 1.0)
        prior_concs = []
        for county in counties:
            base  = COUNTY_WW_BASE.get(county, {}).get(pathogen, 25)
            noise = _md5_float(f"{county}{month - 1}{pathogen}ww_prior") * 16 - 8
            prior_concs.append(max(0, min(100, base * prior_mult + noise)))
        prior_mean = float(np.average(prior_concs, weights=weights))

        # Seasonal elevation threshold: 90th-percentile month × base
        # i.e., signal = current above 90% of peak-season expected value
        peak_mult = max(MONTH_WW_MULT.values())   # 1.6 (January)
        pop_base  = float(np.average(
            [COUNTY_WW_BASE.get(c, {}).get(pathogen, 25) for c in counties],
            weights=weights
        ))
        elevation_threshold = pop_base * mult * 0.9   # 90% of this-month's expected

        delta = mean_conc - prior_mean
        if delta > 2:
            trend = "rising"
        elif delta < -2:
            trend = "falling"
        else:
            trend = "stable"

        result[pathogen] = {
            "concentration":       round(mean_conc, 1),
            "prior_concentration": round(prior_mean, 1),
            "elevation_threshold": round(elevation_threshold, 1),
            "trend":               trend,
            "signal":              mean_conc > elevation_threshold,
            "counties_elevated":   elevated_counties,
        }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISTIC SIMULATION — ED SYNDROMIC (BioSense/NSSP)
# ─────────────────────────────────────────────────────────────────────────────

# Arizona statewide baseline ED visit percentages by syndromic category
# Representative of BioSense/NSSP ESSENCE platform outputs (% total ED visits)
ED_BASELINE = {
    "ili":              {"pct": 3.2, "threshold": 3.5},
    "cli":              {"pct": 2.8, "threshold": 3.0},
    "gi":               {"pct": 1.8, "threshold": 2.2},
    "fever_nos":        {"pct": 2.1, "threshold": 2.4},
    "hemorrhagic":      {"pct": 0.05, "threshold": 0.08},
    "rash_conjunctivitis": {"pct": 0.9, "threshold": 1.2},
}

MONTH_ED_MULT = {
    1: 1.7, 2: 1.5, 3: 1.2, 4: 0.9, 5: 0.8, 6: 0.7,
    7: 0.7, 8: 0.7, 9: 0.9, 10: 1.0, 11: 1.3, 12: 1.6,
}


def simulate_ed_syndromic(reference_date: pd.Timestamp, region: str) -> dict:
    """
    Simulate BioSense/NSSP ED syndromic surveillance signal for a given region.

    Returns:
        dict: {
            category_key: {
                "pct_visits":     float,
                "baseline_pct":   float,
                "above_threshold": bool,
                "trend":          str,
            }
        }
    """
    month = reference_date.month
    mult  = MONTH_ED_MULT.get(month, 1.0)
    result = {}

    for cat, params in ED_BASELINE.items():
        base     = params["pct"]
        # Seasonal threshold: 90% of this-month's expected rate (within-season elevation)
        thresh   = base * mult * 0.88
        noise    = _md5_float(f"{region}{month}{cat}ed") * 0.5 - 0.25
        current  = max(0, base * mult + noise)
        prior_mult = MONTH_ED_MULT.get(month - 1 if month > 1 else 12, 1.0)
        prior    = max(0, base * prior_mult +
                       _md5_float(f"{region}{month - 1}{cat}ed_prior") * 0.5 - 0.25)

        delta = current - prior
        if delta > 0.12:
            trend = "rising"
        elif delta < -0.12:
            trend = "falling"
        else:
            trend = "stable"

        result[cat] = {
            "pct_visits":      round(current, 3),
            "baseline_pct":    round(base, 3),
            "seasonal_threshold": round(thresh, 3),
            "above_threshold": current >= thresh,
            "trend":           trend,
        }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# EPIHACK SIGNAL COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_signal(rec_rate, bas_rate, rec_n, bas_n,
                    min_rec=MIN_REC, min_bas=MIN_BAS, pct_thresh=PCT_THRESH):
    """Standard EpiHack signal classifier (consistent with prior User Stories)."""
    if rec_n < min_rec or bas_n < min_bas:
        return "indef"
    if bas_rate == 0 and rec_rate > 0:
        return "new"
    if rec_rate == 0 and bas_rate > 0:
        return "gone"
    delta = rec_rate - bas_rate
    if delta > pct_thresh:
        return "up"
    if delta < -pct_thresh:
        return "dn"
    return "nc"


def load_epihack_signals(db_path: str, recent_days: int, baseline_days: int,
                          min_reporters: int) -> pd.DataFrame:
    """
    Query epihack-mini.db and compute statewide syndrome-level signals.

    Returns:
        DataFrame with columns:
            syndrome, epihack_signal (bool), epihack_rate, epihack_baseline_rate,
            epihack_n, epihack_signal_class, elevated_postals, total_postals
    """
    conn = sqlite3.connect(db_path)

    # Load all symptom records with date filtering
    cols = ", ".join(SYMPTOM_COLS)
    df = pd.read_sql_query(
        f"""
        SELECT date_of_report, postal_code, county, {cols}
        FROM incidents
        WHERE date_of_report IS NOT NULL
          AND postal_code IS NOT NULL
        """,
        conn,
        parse_dates=["date_of_report"],
    )
    conn.close()

    if df.empty:
        raise ValueError("No records returned from database — check db_path.")

    # Map YES/NO/UNKNOWN → 1/0/NaN
    for col in SYMPTOM_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.upper()
                .map({"YES": 1.0, "NO": 0.0, "UNKNOWN": float("nan")})
                .astype("Float64")
            )

    ref_date  = df["date_of_report"].max()
    rec_start = ref_date - timedelta(days=recent_days)
    bas_start = ref_date - timedelta(days=recent_days + baseline_days)
    bas_end   = rec_start

    df_rec = df[df["date_of_report"] > rec_start].copy()
    df_bas = df[(df["date_of_report"] > bas_start) &
                (df["date_of_report"] <= bas_end)].copy()

    rows = []
    for syn_key, syn_def in SYNDROMES.items():
        epihack_cols = syn_def["epihack_cols"]
        available    = [c for c in epihack_cols if c in df.columns]

        if not available:
            continue

        # Rate = mean(any symptom in syndrome is YES), per reporter
        def syndrome_rate(sub_df):
            if sub_df.empty:
                return 0.0, 0
            flags = sub_df[available].max(axis=1)   # any YES in syndrome
            n     = int(flags.notna().sum())
            rate  = float(flags.mean(skipna=True)) * 100 if n > 0 else 0.0
            return rate, n

        rec_rate, rec_n = syndrome_rate(df_rec)
        bas_rate, bas_n = syndrome_rate(df_bas)

        sig_class = classify_signal(rec_rate, bas_rate, rec_n, bas_n,
                                    min_rec=min_reporters, min_bas=min_reporters * 3)

        # Count elevated postals (≥ PCT_THRESH above baseline at postal level)
        elevated_postals = 0
        total_postals    = 0
        for postal, grp in df_rec.groupby("postal_code"):
            p_rec_rate, p_rec_n = syndrome_rate(grp)
            p_bas_df = df_bas[df_bas["postal_code"] == postal]
            p_bas_rate, p_bas_n = syndrome_rate(p_bas_df)
            if p_rec_n >= min_reporters:
                total_postals += 1
                if p_rec_rate - p_bas_rate > PCT_THRESH or (p_bas_rate == 0 and p_rec_rate > 0):
                    elevated_postals += 1

        rows.append({
            "syndrome":               syn_key,
            "label":                  syn_def["label"],
            "color":                  syn_def["color"],
            "epihack_signal":         sig_class in ("up", "new"),
            "epihack_signal_class":   sig_class,
            "epihack_rate":           round(rec_rate, 2),
            "epihack_baseline_rate":  round(bas_rate, 2),
            "epihack_pct_change":     round(rec_rate - bas_rate, 2),
            "epihack_n":              rec_n,
            "elevated_postals":       elevated_postals,
            "total_postals":          total_postals,
            "ref_date":               ref_date,
        })

    return pd.DataFrame(rows), ref_date


# ─────────────────────────────────────────────────────────────────────────────
# CONCORDANCE COMPUTATION & TIER ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def compute_concordance(epihack_df: pd.DataFrame,
                        ww_signals: dict,
                        ed_signals: dict) -> pd.DataFrame:
    """
    For each syndrome, determine signal status across all three sources
    and assign a concordance tier.

    Tier 1 — all three sources show elevated signal
    Tier 2 — any two sources concordant
    Tier 3 — single source elevated (or no elevation)

    Returns:
        DataFrame with concordance analysis and response recommendations.
    """
    rows = []
    for _, row in epihack_df.iterrows():
        syn_key   = row["syndrome"]
        syn_def   = SYNDROMES[syn_key]

        # ── EpiHack signal ───────────────────────────────────────────────────
        eh_signal = bool(row["epihack_signal"])
        eh_class  = row["epihack_signal_class"]
        eh_rate   = row["epihack_rate"]
        eh_delta  = row["epihack_pct_change"]
        eh_n      = row["epihack_n"]

        # ── Wastewater signal ─────────────────────────────────────────────────
        ww_pathogen = syn_def.get("ww_pathogen")
        if ww_pathogen and ww_pathogen in ww_signals:
            ww = ww_signals[ww_pathogen]
            ww_signal = bool(ww["signal"])
            ww_conc   = ww["concentration"]
            ww_trend  = ww["trend"]
            ww_available = True
        else:
            ww_signal    = False
            ww_conc      = None
            ww_trend     = "N/A"
            ww_available = False

        # ── ED syndromic signal ──────────────────────────────────────────────
        ed_cat = syn_def.get("ed_category")
        if ed_cat and ed_cat in ed_signals:
            ed = ed_signals[ed_cat]
            ed_signal  = bool(ed["above_threshold"])
            ed_pct     = ed["pct_visits"]
            ed_thresh  = ed.get("seasonal_threshold", ED_BASELINE[ed_cat]["threshold"])
            ed_trend   = ed["trend"]
            ed_available = True
        else:
            ed_signal    = False
            ed_pct       = None
            ed_thresh    = None
            ed_trend     = "N/A"
            ed_available = False

        # ── Concordance count ────────────────────────────────────────────────
        n_sources_available = (1 +
                               (1 if ww_available else 0) +
                               (1 if ed_available else 0))
        n_elevated = (int(eh_signal) +
                      int(ww_signal if ww_available else 0) +
                      int(ed_signal if ed_available else 0))

        # ── Tier assignment ──────────────────────────────────────────────────
        if n_sources_available == 3 and n_elevated == 3:
            tier              = 1
            concordance_level = "Full Concordance — all 3 sources elevated"
            response_rec      = ("IMMEDIATE INVESTIGATION — Activate multi-agency response; "
                                 "alert ADHS Epidemiology, emergency department liaisons, "
                                 "and environmental health partners. Issue public advisory.")
        elif n_elevated >= 2:
            tier              = 2
            sources_elevated  = [
                SRC_EPIHACK    if eh_signal else None,
                SRC_WASTEWATER if (ww_available and ww_signal) else None,
                SRC_ED         if (ed_available and ed_signal) else None,
            ]
            src_str = " + ".join(s for s in sources_elevated if s)
            concordance_level = f"Partial Concordance — {src_str}"
            response_rec      = ("ENHANCED SURVEILLANCE — Increase reporting frequency; "
                                 "notify county health departments; prepare situation report "
                                 "for ADHS weekly epidemiology call. Monitor third source.")
        elif n_elevated == 1:
            tier              = 3
            single_src        = (SRC_EPIHACK    if eh_signal
                                 else SRC_WASTEWATER if (ww_available and ww_signal)
                                 else SRC_ED)
            concordance_level = f"Single-Source Signal — {single_src} only"
            response_rec      = ("ROUTINE MONITORING — Maintain standard surveillance cadence; "
                                 "document signal for trend tracking; flag for next weekly review.")
        else:
            tier              = 3
            concordance_level = "No Elevated Signal — all sources within baseline"
            response_rec      = ("NO ACTION REQUIRED — Continue routine surveillance; "
                                 "maintain background monitoring.")

        # ── Concordance description ──────────────────────────────────────────
        parts = []
        if eh_class == "indef":
            parts.append(f"EpiHack: insufficient data (n={eh_n})")
        else:
            parts.append(
                f"EpiHack: {eh_class.upper()} "
                f"({'+' if eh_delta >= 0 else ''}{eh_delta:.1f}pp vs baseline, "
                f"n={eh_n}, {row['elevated_postals']}/{row['total_postals']} postals elevated)"
            )
        if ww_available:
            parts.append(
                f"Wastewater: conc={ww_conc:.0f}/100 "
                f"({'ELEVATED' if ww_signal else 'baseline'}), trend={ww_trend}"
            )
        else:
            parts.append("Wastewater: no analogue pathogen")

        if ed_available:
            parts.append(
                f"ED Syndromic: {ed_pct:.2f}% visits "
                f"({'ABOVE' if ed_signal else 'below'} threshold={ed_thresh:.2f}%), "
                f"trend={ed_trend}"
            )
        else:
            parts.append("ED Syndromic: no analogue category")

        rows.append({
            "syndrome":          row["label"],
            "syndrome_key":      syn_key,
            "color":             row["color"],
            "tier":              tier,
            "n_elevated":        n_elevated,
            "n_sources_avail":   n_sources_available,
            "concordance_level": concordance_level,
            # Per-source detail
            "epihack_signal":    eh_signal,
            "epihack_class":     eh_class,
            "epihack_rate":      eh_rate,
            "epihack_delta":     eh_delta,
            "epihack_n":         eh_n,
            "ww_available":      ww_available,
            "ww_signal":         ww_signal,
            "ww_conc":           ww_conc,
            "ww_trend":          ww_trend,
            "ed_available":      ed_available,
            "ed_signal":         ed_signal,
            "ed_pct":            ed_pct,
            "ed_trend":          ed_trend,
            "source_details":    " | ".join(parts),
            "response_rec":      response_rec,
        })

    df_out = pd.DataFrame(rows).sort_values(["tier", "n_elevated"],
                                             ascending=[True, False])
    return df_out.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# LLM PROMPT TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

def build_llm_prompt(concordance_df: pd.DataFrame,
                     ww_signals: dict,
                     ed_signals: dict,
                     region: str,
                     ref_date: pd.Timestamp,
                     recent_days: int) -> str:
    """
    Fill the LLM prompt template from User Story #9 with computed values.
    In production, replace the stub below with an Anthropic API call:

        import anthropic
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": filled_prompt}]
        )
        narrative = message.content[0].text

    Returns the filled prompt string for traceability.
    """
    tier1 = concordance_df[concordance_df["tier"] == 1]
    tier2 = concordance_df[concordance_df["tier"] == 2]
    tier3 = concordance_df[concordance_df["tier"] == 3]

    # Summarise EpiHack signals
    epihack_signals_str = "; ".join(
        f"{r['syndrome']} ({r['epihack_class'].upper()}, "
        f"{'+' if r['epihack_delta'] >= 0 else ''}{r['epihack_delta']:.1f}pp)"
        for _, r in concordance_df.iterrows()
        if r["epihack_signal"]
    ) or "No elevated signals in recent window"

    # Summarise wastewater
    ww_str = "; ".join(
        f"{k.replace('_', ' ').title()}: conc={v['concentration']:.0f}/100, {v['trend']}"
        for k, v in ww_signals.items()
    )

    # Summarise ED syndromic
    ed_str = "; ".join(
        f"{k.replace('_', ' ').upper()}: {v['pct_visits']:.2f}% visits "
        f"({'ABOVE' if v['above_threshold'] else 'below'} threshold), {v['trend']}"
        for k, v in ed_signals.items()
    )

    prompt = f"""You are a multi-source surveillance triangulation analyst. Synthesize:

(1) EpiHack community signals:
{epihack_signals_str}

(2) ADHS Wastewater Surveillance trend for {region}:
{ww_str}

(3) BioSense/NSSP emergency department syndromic data:
{ed_str}

Reference date: {ref_date.date()} | Recent window: {recent_days}d | Region: {region}

Tier 1 syndromes (all 3 concordant): {', '.join(tier1['syndrome'].tolist()) or 'None'}
Tier 2 syndromes (2 concordant): {', '.join(tier2['syndrome'].tolist()) or 'None'}
Tier 3 syndromes (single source): {', '.join(tier3['syndrome'].tolist()) or 'None'}

For each syndrome with concordant signals across 2+ sources:
(1) Describe the concordance level in detail.
(2) Assign signal confidence tier (Tier 1: all 3 concordant; Tier 2: 2 concordant; Tier 3: single source).
(3) Recommend appropriate response level per tier.

Output as a structured decision table with columns:
Syndrome | Concordance Level | Confidence Tier | Recommended Response
"""
    return prompt


def simulate_llm_narrative(concordance_df: pd.DataFrame,
                            region: str,
                            ref_date: pd.Timestamp,
                            recent_days: int) -> str:
    """
    Generate a deterministic LLM-style narrative for the triangulation analysis.
    Replace with actual Anthropic API call in production.
    """
    tier1 = concordance_df[concordance_df["tier"] == 1]
    tier2 = concordance_df[concordance_df["tier"] == 2]
    elevated = concordance_df[concordance_df["n_elevated"] >= 2]

    n_tier1 = len(tier1)
    n_tier2 = len(tier2)
    n_elevated = len(elevated)

    tier1_names = ", ".join(tier1["syndrome"].tolist()) if not tier1.empty else "none identified"
    tier2_names = ", ".join(tier2["syndrome"].tolist()) if not tier2.empty else "none identified"

    # Select the highest-priority syndrome for case narrative
    if not tier1.empty:
        priority_row = tier1.iloc[0]
        urgency = "CRITICAL"
    elif not tier2.empty:
        priority_row = tier2.iloc[0]
        urgency = "ELEVATED"
    else:
        priority_row = concordance_df.iloc[0] if not concordance_df.empty else None
        urgency = "ROUTINE"

    narrative = f"""MULTI-SOURCE SURVEILLANCE TRIANGULATION REPORT
Region: {region} | Reference date: {ref_date.date()} | Window: {recent_days}d recent
Generated: {ref_date.date()} | Classification: ADHS INTERNAL — FOR OFFICIAL USE ONLY

━━━ EXECUTIVE SUMMARY ━━━

Triangulation of three independent surveillance streams — EpiHack participatory reporting,
ADHS wastewater surveillance, and BioSense/NSSP emergency department syndromic monitoring —
identified {n_elevated} syndrome(s) with cross-source concordance in {region} over the
{recent_days}-day reference window.

Alert status: {urgency}
• Tier 1 (full concordance, all 3 sources): {n_tier1} syndrome(s) — {tier1_names}
• Tier 2 (partial concordance, 2 sources): {n_tier2} syndrome(s) — {tier2_names}

━━━ PRIORITY SIGNAL ━━━
"""
    if priority_row is not None:
        narrative += f"""
Syndrome: {priority_row['syndrome']}
Confidence Tier: {priority_row['tier']}
Concordance: {priority_row['concordance_level']}

Signal detail:
  EpiHack: {'ELEVATED' if priority_row['epihack_signal'] else 'baseline'} — """
        if priority_row["epihack_class"] == "indef":
            narrative += f"insufficient reporter volume (n={priority_row['epihack_n']})\n"
        else:
            narrative += (f"{'+' if priority_row['epihack_delta'] >= 0 else ''}"
                         f"{priority_row['epihack_delta']:.1f}pp vs 30d baseline, "
                         f"n={priority_row['epihack_n']} reporters\n")

        if priority_row["ww_available"]:
            narrative += (f"  Wastewater: concentration {priority_row['ww_conc']:.0f}/100, "
                         f"trend {priority_row['ww_trend']}, "
                         f"{'SIGNAL' if priority_row['ww_signal'] else 'BASELINE'}\n")
        if priority_row["ed_available"]:
            narrative += (f"  ED Syndromic: {priority_row['ed_pct']:.2f}% of ED visits, "
                         f"trend {priority_row['ed_trend']}, "
                         f"{'ABOVE THRESHOLD' if priority_row['ed_signal'] else 'within threshold'}\n")

        narrative += f"""
Recommended response: {priority_row['response_rec']}
"""

    narrative += """
━━━ METHODOLOGICAL NOTE ━━━

Multi-source triangulation substantially reduces false-positive risk inherent in any single
surveillance stream. EpiHack participatory data is subject to reporting biases (digital access,
health literacy, occupational determinants); wastewater surveillance provides population-level
pathogen exposure without behavioural reporting barriers; ED syndromic monitoring captures
care-seeking behaviour which may lag community infection by 3–7 days. Cross-source concordance
elevates signal credibility and warrants escalated public health response.

[NOTE: Wastewater and ED syndromic values are deterministic simulations calibrated to
Arizona-representative baselines. In production, replace with live ADHS BEACON and
BioSense/NSSP ESSENCE API feeds.]
"""
    return narrative


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION — MATPLOTLIB PNG
# ─────────────────────────────────────────────────────────────────────────────

def fig_triangulation(concordance_df: pd.DataFrame,
                      ww_signals: dict,
                      ed_signals: dict,
                      region: str,
                      ref_date: pd.Timestamp,
                      recent_days: int,
                      out_path: str) -> None:
    """
    Four-panel Matplotlib figure:
      A — Three-source concordance matrix (heatmap)
      B — Signal confidence tier distribution (donut)
      C — Wastewater pathogen concentration trend bars
      D — ED syndromic % visits vs threshold
    """
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    SURFACE,
        "axes.edgecolor":    MUTED,
        "text.color":        TEXT,
        "axes.labelcolor":   TEXT,
        "xtick.color":       TEXT,
        "ytick.color":       TEXT,
        "grid.color":        "#2a2d3e",
        "font.family":       "DejaVu Sans",
        "font.size":         9,
    })

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(BG)

    # Title
    fig.text(
        0.5, 0.97,
        f"Multi-Source Surveillance Triangulation — {region}",
        ha="center", va="top", fontsize=16, fontweight="bold", color=TEXT,
    )
    fig.text(
        0.5, 0.945,
        f"Reference: {ref_date.date()} | Window: {recent_days}d recent / 30d baseline | "
        f"Sources: EpiHack + ADHS Wastewater + BioSense/NSSP ED Syndromic",
        ha="center", va="top", fontsize=9, color=MUTED,
    )

    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.3,
                          top=0.91, bottom=0.06, left=0.07, right=0.97)

    # ── Panel A: Concordance Matrix ──────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor(SURFACE)
    ax_a.set_title("Panel A — Three-Source Concordance Matrix", color=TEXT,
                   fontsize=10, fontweight="bold", pad=8)

    syndrome_labels = [r["syndrome"].replace(" — ", "\n") for _, r in concordance_df.iterrows()]
    sources = [SRC_EPIHACK, SRC_WASTEWATER, SRC_ED]
    matrix  = []
    for _, row in concordance_df.iterrows():
        matrix.append([
            1 if row["epihack_signal"] else 0,
            1 if (row["ww_available"] and row["ww_signal"]) else (-1 if not row["ww_available"] else 0),
            1 if (row["ed_available"] and row["ed_signal"]) else (-1 if not row["ed_available"] else 0),
        ])

    mat = np.array(matrix, dtype=float)
    n_syn = len(syndrome_labels)

    # Custom colour map: -1=N/A (grey), 0=baseline (dark), 1=elevated (red)
    for si, (src_col, src_lbl) in enumerate(zip(mat.T, sources)):
        for ri, val in enumerate(src_col):
            if val == -1:
                colour = "#2a2d3e"
                lbl    = "N/A"
            elif val == 1:
                colour = RED
                lbl    = "ELEVATED"
            else:
                colour = "#1a1d27"
                lbl    = "OK"
            rect = plt.Rectangle([si - 0.5, ri - 0.5], 1, 1, fc=colour,
                                  ec=MUTED, linewidth=0.5)
            ax_a.add_patch(rect)
            ax_a.text(si, ri, lbl, ha="center", va="center",
                     fontsize=7, color=TEXT if val != 0 else MUTED)

    # Tier colour strip on right side
    for ri, (_, row) in enumerate(concordance_df.iterrows()):
        tier_colour = {1: RED, 2: ORANGE, 3: MUTED}.get(row["tier"], MUTED)
        rect = plt.Rectangle([2.5, ri - 0.5], 0.5, 1, fc=tier_colour, ec=MUTED, linewidth=0.5)
        ax_a.add_patch(rect)
        ax_a.text(2.75, ri, f"T{row['tier']}", ha="center", va="center",
                 fontsize=7, color="white", fontweight="bold")

    ax_a.set_xlim(-0.5, 3.0)
    ax_a.set_ylim(-0.5, n_syn - 0.5)
    ax_a.set_xticks([0, 1, 2, 2.75])
    ax_a.set_xticklabels(["EpiHack", "Wastewater", "ED Syndromic", "Tier"], fontsize=8)
    ax_a.set_yticks(range(n_syn))
    ax_a.set_yticklabels(syndrome_labels, fontsize=8)
    ax_a.invert_yaxis()

    # Legend
    patches = [
        mpatches.Patch(fc=RED, label="ELEVATED"),
        mpatches.Patch(fc="#1a1d27", ec=MUTED, label="Within baseline"),
        mpatches.Patch(fc="#2a2d3e", label="No analogue (N/A)"),
    ]
    ax_a.legend(handles=patches, loc="lower right", fontsize=7,
               facecolor=BG, edgecolor=MUTED, labelcolor=TEXT)

    # ── Panel B: Tier Distribution Donut ────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor(SURFACE)
    ax_b.set_title("Panel B — Signal Confidence Tier Distribution", color=TEXT,
                   fontsize=10, fontweight="bold", pad=8)

    tier_counts = concordance_df["tier"].value_counts().sort_index()
    tier_labels_b = [f"Tier {t}\n({c} syndrome{'s' if c != 1 else ''})"
                     for t, c in tier_counts.items()]
    tier_colours  = [RED if t == 1 else ORANGE if t == 2 else MUTED
                     for t in tier_counts.index]

    if len(tier_counts) > 0:
        wedges, texts, autotexts = ax_b.pie(
            tier_counts.values,
            labels=tier_labels_b,
            colors=tier_colours,
            autopct="%1.0f%%",
            pctdistance=0.7,
            wedgeprops={"width": 0.55, "edgecolor": BG, "linewidth": 2},
            startangle=90,
        )
        for t in texts:
            t.set_color(TEXT)
            t.set_fontsize(8)
        for at in autotexts:
            at.set_color("white")
            at.set_fontsize(9)
            at.set_fontweight("bold")

    # Centre text
    ax_b.text(0, 0, f"{len(concordance_df)}\nsyndromes\nevaluated",
              ha="center", va="center", fontsize=10, fontweight="bold", color=TEXT)

    # Response level legend
    resp_text = (
        "Tier 1 → IMMEDIATE INVESTIGATION\n"
        "Tier 2 → ENHANCED SURVEILLANCE\n"
        "Tier 3 → ROUTINE MONITORING"
    )
    ax_b.text(0, -1.45, resp_text, ha="center", va="center", fontsize=7.5,
             color=MUTED, style="italic",
             bbox=dict(boxstyle="round,pad=0.4", fc=BG, ec=MUTED, alpha=0.8))

    # ── Panel C: Wastewater Pathogen Concentrations ──────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor(SURFACE)
    ax_c.set_title("Panel C — ADHS Wastewater Pathogen Concentration (simulated)",
                   color=TEXT, fontsize=10, fontweight="bold", pad=8)

    ww_keys  = list(ww_signals.keys())
    ww_concs = [ww_signals[k]["concentration"] for k in ww_keys]
    ww_prior = [ww_signals[k]["prior_concentration"] for k in ww_keys]
    ww_lbls  = [k.replace("_", " ").replace("sars cov2", "SARS-CoV-2").title()
                for k in ww_keys]
    ww_clrs  = [RED if ww_signals[k]["signal"] else BLUE for k in ww_keys]

    x = np.arange(len(ww_keys))
    w = 0.35
    bars_c = ax_c.bar(x - w / 2, ww_prior, w, label="Prior month", color="#3a3d4e", alpha=0.8)
    bars_r = ax_c.bar(x + w / 2, ww_concs, w, label="Current (pop-weighted)", color=ww_clrs, alpha=0.9)
    # Draw per-pathogen seasonal elevation threshold
    for xi, key in enumerate(ww_keys):
        thr = ww_signals[key].get("elevation_threshold", 20)
        ax_c.plot([xi + w / 2 - w * 0.4, xi + w / 2 + w * 0.4], [thr, thr],
                  color=ORANGE, linewidth=2.0, solid_capstyle="butt")
    ax_c.plot([], [], color=ORANGE, linewidth=2, label="Seasonal elevation threshold")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(ww_lbls, fontsize=8)
    ax_c.set_ylabel("Relative Concentration (0–100)", color=TEXT, fontsize=8)
    ax_c.set_ylim(0, 80)
    ax_c.legend(fontsize=7, facecolor=BG, edgecolor=MUTED, labelcolor=TEXT)

    for bar, val in zip(bars_r, ww_concs):
        ax_c.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=7.5, color=TEXT)

    ax_c.grid(axis="y", alpha=0.3)

    # ── Panel D: ED Syndromic % Visits vs Threshold ──────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor(SURFACE)
    ax_d.set_title("Panel D — BioSense/NSSP ED Syndromic % Visits vs Threshold (simulated)",
                   color=TEXT, fontsize=10, fontweight="bold", pad=8)

    ed_cats   = list(ed_signals.keys())
    ed_pcts   = [ed_signals[c]["pct_visits"] for c in ed_cats]
    ed_thrs   = [ed_signals[c].get("seasonal_threshold", ED_BASELINE[c]["threshold"])
                 for c in ed_cats]
    ed_lbls   = [c.upper().replace("_", "\n") for c in ed_cats]
    ed_clrs   = [RED if ed_signals[c]["above_threshold"] else BLUE for c in ed_cats]

    x2 = np.arange(len(ed_cats))
    ax_d.bar(x2, ed_pcts, 0.6, color=ed_clrs, alpha=0.85)
    ax_d.scatter(x2, ed_thrs, marker="_", s=200, color=ORANGE, zorder=5,
                linewidths=2, label="Alert threshold")
    ax_d.set_xticks(x2)
    ax_d.set_xticklabels(ed_lbls, fontsize=7)
    ax_d.set_ylabel("% of Total ED Visits", color=TEXT, fontsize=8)
    ax_d.legend(fontsize=7, facecolor=BG, edgecolor=MUTED, labelcolor=TEXT)

    for xi, (pct, thr) in enumerate(zip(ed_pcts, ed_thrs)):
        ax_d.text(xi, pct + 0.03, f"{pct:.2f}%", ha="center", va="bottom",
                 fontsize=7, color=TEXT)

    ax_d.grid(axis="y", alpha=0.3)

    # Watermark
    fig.text(0.5, 0.01,
             "Deterministic simulation | EpiHack Platform | ADHS Epidemiology Division",
             ha="center", va="bottom", fontsize=7, color=MUTED, style="italic")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[PNG] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# DECISION TABLE — TERMINAL OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def print_decision_table(concordance_df: pd.DataFrame,
                          region: str, ref_date: pd.Timestamp,
                          recent_days: int) -> None:
    """Print structured decision table to stdout."""
    sep   = "─" * 120
    print()
    print(sep)
    print(f"MULTI-SOURCE SURVEILLANCE TRIANGULATION — {region}")
    print(f"Reference date: {ref_date.date()} | Window: {recent_days}d recent / 30d baseline")
    print(sep)
    print(f"{'SYNDROME':<32} {'TIER':<6} {'ELEVATED/AVAIL':<15} "
          f"{'CONCORDANCE LEVEL':<40} {'EH':>4} {'WW':>4} {'ED':>4}")
    print(sep)

    for _, row in concordance_df.iterrows():
        eh_sym = "▲" if row["epihack_signal"] else "·"
        ww_sym = ("▲" if row["ww_signal"] else "·") if row["ww_available"] else "N/A"
        ed_sym = ("▲" if row["ed_signal"] else "·") if row["ed_available"] else "N/A"
        tier_s = f"TIER {row['tier']}"
        avail  = f"{row['n_elevated']}/{row['n_sources_avail']}"
        print(
            f"{row['syndrome'][:31]:<32} {tier_s:<6} {avail:<15} "
            f"{row['concordance_level'][:39]:<40} {eh_sym:>4} {ww_sym:>4} {ed_sym:>4}"
        )

    print(sep)
    print()
    print("RESPONSE RECOMMENDATIONS:")
    print()
    for _, row in concordance_df[concordance_df["tier"] <= 2].iterrows():
        print(f"  [TIER {row['tier']}] {row['syndrome']}")
        print(f"          → {row['response_rec']}")
        print()

    tier1_n  = len(concordance_df[concordance_df["tier"] == 1])
    tier2_n  = len(concordance_df[concordance_df["tier"] == 2])
    tier3_n  = len(concordance_df[concordance_df["tier"] == 3])
    print(f"Summary: Tier 1={tier1_n}  Tier 2={tier2_n}  Tier 3={tier3_n}")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# HTML DASHBOARD GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def build_html_dashboard(concordance_df: pd.DataFrame,
                          ww_signals: dict,
                          ed_signals: dict,
                          region: str,
                          ref_date: pd.Timestamp,
                          recent_days: int,
                          baseline_days: int,
                          llm_prompt: str,
                          narrative: str,
                          out_path: str) -> None:
    """
    Build a self-contained interactive HTML dashboard for User Story #9.
    All data is embedded as JSON; client-side Chart.js renders panels.
    """
    import json

    # Prepare data payload for JavaScript
    concordance_data = concordance_df.to_dict(orient="records")
    # Make JSON-safe
    for rec in concordance_data:
        for k, v in rec.items():
            if isinstance(v, (np.integer,)):
                rec[k] = int(v)
            elif isinstance(v, (np.floating,)):
                rec[k] = None if np.isnan(v) else float(v)
            elif isinstance(v, (np.bool_,)):
                rec[k] = bool(v)
            elif v is None or (isinstance(v, float) and np.isnan(v)):
                rec[k] = None

    ww_data = {
        k: {
            "concentration": v["concentration"],
            "prior_concentration": v["prior_concentration"],
            "trend": v["trend"],
            "signal": v["signal"],
        }
        for k, v in ww_signals.items()
    }

    ed_data = {
        k: {
            "pct_visits":      v["pct_visits"],
            "baseline_pct":    v.get("seasonal_threshold", ED_BASELINE[k]["threshold"]),
            "above_threshold": v["above_threshold"],
            "trend":           v["trend"],
        }
        for k, v in ed_signals.items()
    }

    concordance_json = json.dumps(concordance_data)
    ww_json          = json.dumps(ww_data)
    ed_json          = json.dumps(ed_data)
    narrative_html   = narrative.replace("\n", "<br>").replace("━", "—")
    prompt_html      = llm_prompt.replace("\n", "<br>")

    tier1_count = int((concordance_df["tier"] == 1).sum())
    tier2_count = int((concordance_df["tier"] == 2).sum())
    tier3_count = int((concordance_df["tier"] == 3).sum())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Multi-Source Surveillance Triangulation — {region}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: {BG}; --surface: {SURFACE}; --accent: {ACCENT};
    --red: {RED}; --orange: {ORANGE}; --green: {GREEN};
    --blue: {BLUE}; --cyan: {CYAN}; --yellow: {YELLOW};
    --text: {TEXT}; --muted: {MUTED};
  }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; }}
  header {{ background: var(--surface); border-bottom: 2px solid var(--accent); padding: 18px 28px; }}
  header h1 {{ font-size: 1.5rem; color: var(--accent); }}
  header p {{ color: var(--muted); font-size: 0.85rem; margin-top: 4px; }}
  .meta-bar {{ display: flex; gap: 20px; margin-top: 10px; flex-wrap: wrap; }}
  .meta-chip {{ background: rgba(108,99,255,0.15); border: 1px solid var(--accent);
                border-radius: 20px; padding: 4px 14px; font-size: 0.78rem; color: var(--accent); }}
  .tier-chip {{ border-radius: 20px; padding: 4px 14px; font-size: 0.82rem; font-weight: 700;
                display: inline-block; }}
  .tier-1 {{ background: rgba(255,71,87,0.25); color: var(--red); border: 1px solid var(--red); }}
  .tier-2 {{ background: rgba(255,165,2,0.25); color: var(--orange); border: 1px solid var(--orange); }}
  .tier-3 {{ background: rgba(136,146,164,0.25); color: var(--muted); border: 1px solid var(--muted); }}
  .controls {{ background: var(--surface); border-bottom: 1px solid #2a2d3e;
               padding: 14px 28px; display: flex; gap: 24px; flex-wrap: wrap; align-items: center; }}
  .ctrl-group {{ display: flex; align-items: center; gap: 8px; }}
  .ctrl-group label {{ color: var(--muted); font-size: 0.82rem; }}
  .ctrl-group input[type=range] {{ accent-color: var(--accent); cursor: pointer; }}
  .ctrl-group select {{ background: var(--bg); color: var(--text);
                        border: 1px solid var(--accent); border-radius: 6px;
                        padding: 4px 8px; font-size: 0.82rem; cursor: pointer; }}
  .ctrl-group span {{ color: var(--accent); font-weight: 700; min-width: 30px; }}
  .dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px;
                padding: 20px 28px; }}
  .panel {{ background: var(--surface); border-radius: 10px; padding: 20px;
            border: 1px solid #2a2d3e; }}
  .panel-full {{ grid-column: 1 / -1; }}
  .panel h2 {{ font-size: 0.95rem; color: var(--accent); margin-bottom: 14px;
               font-weight: 600; letter-spacing: 0.02em; }}
  .canvas-wrap {{ position: relative; height: 320px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.82rem; }}
  thead th {{ background: rgba(108,99,255,0.15); color: var(--accent);
              padding: 8px 10px; text-align: left; border-bottom: 2px solid var(--accent);
              cursor: pointer; user-select: none; white-space: nowrap; }}
  thead th:hover {{ background: rgba(108,99,255,0.25); }}
  tbody tr {{ border-bottom: 1px solid #2a2d3e; transition: background 0.15s; }}
  tbody tr:hover {{ background: rgba(108,99,255,0.07); }}
  tbody td {{ padding: 8px 10px; vertical-align: top; }}
  .sig-up {{ color: var(--red); font-weight: 700; }}
  .sig-ok {{ color: var(--muted); }}
  .sig-na {{ color: #2a2d3e; }}
  .badge {{ display: inline-block; border-radius: 12px; padding: 2px 10px;
            font-size: 0.75rem; font-weight: 700; }}
  .narrative-box {{ background: var(--bg); border: 1px solid #2a2d3e; border-radius: 8px;
                    padding: 18px; font-size: 0.84rem; line-height: 1.7; color: var(--text);
                    font-family: 'Courier New', monospace; white-space: pre-wrap;
                    max-height: 420px; overflow-y: auto; }}
  .prompt-box {{ background: var(--bg); border: 1px solid var(--accent); border-radius: 8px;
                 padding: 14px; font-size: 0.78rem; line-height: 1.6; color: var(--muted);
                 max-height: 320px; overflow-y: auto; }}
  .summary-cards {{ display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }}
  .summary-card {{ flex: 1; min-width: 150px; background: var(--bg); border-radius: 8px;
                   padding: 14px 16px; border: 1px solid #2a2d3e; text-align: center; }}
  .summary-card .sc-val {{ font-size: 2rem; font-weight: 800; }}
  .summary-card .sc-lbl {{ font-size: 0.78rem; color: var(--muted); margin-top: 4px; }}
  .resp-card {{ margin: 6px 0; padding: 10px 14px; border-radius: 8px; font-size: 0.82rem; border-left: 4px solid; }}
  .resp-tier1 {{ background: rgba(255,71,87,0.1); border-color: var(--red); }}
  .resp-tier2 {{ background: rgba(255,165,2,0.1); border-color: var(--orange); }}
  .filter-btn {{ background: var(--bg); color: var(--text); border: 1px solid #2a2d3e;
                 border-radius: 20px; padding: 4px 14px; cursor: pointer; font-size: 0.78rem;
                 transition: all 0.2s; }}
  .filter-btn.active {{ background: var(--accent); border-color: var(--accent); color: white; }}
  footer {{ text-align: center; padding: 18px; color: var(--muted); font-size: 0.76rem;
            border-top: 1px solid #2a2d3e; }}
</style>
</head>
<body>
<header>
  <h1>Multi-Source Surveillance Triangulation</h1>
  <p>EpiHack · ADHS Wastewater · BioSense/NSSP ED Syndromic | Region: {region} | {ref_date.date()}</p>
  <div class="meta-bar">
    <span class="meta-chip">User Story #9</span>
    <span class="meta-chip">Recent window: {recent_days}d</span>
    <span class="meta-chip">Baseline: {baseline_days}d</span>
    <span class="meta-chip">Ref date: {ref_date.date()}</span>
    <span class="tier-chip tier-1">Tier 1 (all 3): {tier1_count}</span>
    <span class="tier-chip tier-2">Tier 2 (2 sources): {tier2_count}</span>
    <span class="tier-chip tier-3">Tier 3 (single): {tier3_count}</span>
  </div>
</header>

<div class="controls">
  <div class="ctrl-group">
    <label>Tier filter:</label>
    <button class="filter-btn active" onclick="setTierFilter('all')">All</button>
    <button class="filter-btn" onclick="setTierFilter(1)">Tier 1</button>
    <button class="filter-btn" onclick="setTierFilter(2)">Tier 2</button>
    <button class="filter-btn" onclick="setTierFilter(3)">Tier 3</button>
  </div>
  <div class="ctrl-group">
    <label>Sort by:</label>
    <select onchange="setSortOrder(this.value)">
      <option value="tier">Tier (priority)</option>
      <option value="n_elevated">Sources elevated</option>
      <option value="syndrome">Syndrome A–Z</option>
    </select>
  </div>
  <div class="ctrl-group" style="margin-left:auto">
    <span style="color:var(--muted);font-size:0.78rem;">
      ▲ = ELEVATED &nbsp;&nbsp; · = within baseline &nbsp;&nbsp; N/A = no analogue
    </span>
  </div>
</div>

<div class="dashboard">

  <!-- Panel A: Concordance Matrix -->
  <div class="panel">
    <h2>Panel A — Three-Source Concordance Matrix</h2>
    <div class="canvas-wrap">
      <canvas id="matrixChart"></canvas>
    </div>
  </div>

  <!-- Panel B: Tier Distribution -->
  <div class="panel">
    <h2>Panel B — Signal Confidence Tier Distribution</h2>
    <div class="summary-cards">
      <div class="summary-card">
        <div class="sc-val" style="color:var(--red)">{tier1_count}</div>
        <div class="sc-lbl">Tier 1<br>Full Concordance</div>
      </div>
      <div class="summary-card">
        <div class="sc-val" style="color:var(--orange)">{tier2_count}</div>
        <div class="sc-lbl">Tier 2<br>Partial Concordance</div>
      </div>
      <div class="summary-card">
        <div class="sc-val" style="color:var(--muted)">{tier3_count}</div>
        <div class="sc-lbl">Tier 3<br>Single Source</div>
      </div>
      <div class="summary-card">
        <div class="sc-val" style="color:var(--accent)">{tier1_count + tier2_count + tier3_count}</div>
        <div class="sc-lbl">Syndromes<br>Evaluated</div>
      </div>
    </div>
    <div class="canvas-wrap" style="height:220px">
      <canvas id="tierChart"></canvas>
    </div>
  </div>

  <!-- Panel C: Wastewater -->
  <div class="panel">
    <h2>Panel C — ADHS Wastewater Pathogen Concentration <span style="color:var(--muted);font-size:0.75rem">(deterministic simulation)</span></h2>
    <div class="canvas-wrap">
      <canvas id="wwChart"></canvas>
    </div>
  </div>

  <!-- Panel D: ED Syndromic -->
  <div class="panel">
    <h2>Panel D — BioSense/NSSP ED Syndromic % Visits vs Alert Threshold <span style="color:var(--muted);font-size:0.75rem">(simulated)</span></h2>
    <div class="canvas-wrap">
      <canvas id="edChart"></canvas>
    </div>
  </div>

  <!-- Panel E: Decision Table (full-width) -->
  <div class="panel panel-full">
    <h2>Panel E — Structured Decision Table</h2>
    <div id="decision-table-container"></div>
  </div>

  <!-- Panel F: Response Recommendations -->
  <div class="panel panel-full">
    <h2>Panel F — Response Recommendations by Tier</h2>
    <div id="response-recs"></div>
  </div>

  <!-- Panel G: Triangulation Narrative -->
  <div class="panel panel-full">
    <h2>Panel G — Surveillance Triangulation Narrative (LLM-generated)</h2>
    <div class="narrative-box" id="narrative">{narrative_html}</div>
  </div>

  <!-- Panel H: LLM Prompt Template -->
  <div class="panel panel-full">
    <h2>Panel H — LLM Prompt Template (filled)</h2>
    <div class="prompt-box">{prompt_html}</div>
  </div>

</div>

<footer>
  EpiHack Participatory Epidemiological Surveillance Platform &nbsp;|&nbsp;
  Arizona Department of Health Services (simulation) &nbsp;|&nbsp;
  {ref_date.date()} &nbsp;|&nbsp;
  User Story #9 — Multi-Source Surveillance Triangulation
</footer>

<script>
// ── Embedded data ──────────────────────────────────────────────────────────
const CONCORDANCE_DATA = {concordance_json};
const WW_DATA = {ww_json};
const ED_DATA = {ed_json};

const BG      = "{BG}";
const SURFACE = "{SURFACE}";
const ACCENT  = "{ACCENT}";
const RED     = "{RED}";
const ORANGE  = "{ORANGE}";
const GREEN   = "{GREEN}";
const BLUE    = "{BLUE}";
const CYAN    = "{CYAN}";
const TEXT    = "{TEXT}";
const MUTED   = "{MUTED}";

let activeTierFilter = "all";
let activeSortOrder  = "tier";

// ── Utility helpers ────────────────────────────────────────────────────────
function tierColor(t) {{
  return t === 1 ? RED : t === 2 ? ORANGE : MUTED;
}}
function tierClass(t) {{
  return `tier-${{t}}`;
}}
function sigSymbol(val, avail) {{
  if (!avail) return '<span class="sig-na">N/A</span>';
  return val
    ? '<span class="sig-up">▲ ELEVATED</span>'
    : '<span class="sig-ok">· baseline</span>';
}}

function setTierFilter(tier) {{
  activeTierFilter = tier;
  document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
  event.target.classList.add("active");
  renderDecisionTable();
  renderResponseRecs();
}}

function setSortOrder(val) {{
  activeSortOrder = val;
  renderDecisionTable();
}}

function filteredSorted() {{
  let data = [...CONCORDANCE_DATA];
  if (activeTierFilter !== "all") {{
    data = data.filter(r => r.tier === Number(activeTierFilter));
  }}
  data.sort((a, b) => {{
    if (activeSortOrder === "tier") return a.tier - b.tier || b.n_elevated - a.n_elevated;
    if (activeSortOrder === "n_elevated") return b.n_elevated - a.n_elevated;
    if (activeSortOrder === "syndrome") return a.syndrome.localeCompare(b.syndrome);
    return 0;
  }});
  return data;
}}

// ── Panel A: Concordance Matrix (bar chart approximation) ─────────────────
(function buildMatrixChart() {{
  const ctx   = document.getElementById("matrixChart").getContext("2d");
  const synds = CONCORDANCE_DATA.map(r => r.syndrome.replace(" — ", " -\n"));
  const ehVals = CONCORDANCE_DATA.map(r => r.epihack_signal ? 1 : 0);
  const wwVals = CONCORDANCE_DATA.map(r => r.ww_available ? (r.ww_signal ? 1 : 0) : -1);
  const edVals = CONCORDANCE_DATA.map(r => r.ed_available ? (r.ed_signal ? 1 : 0) : -1);

  new Chart(ctx, {{
    type: "bar",
    data: {{
      labels: CONCORDANCE_DATA.map(r => r.syndrome.split(" — ")[0]),
      datasets: [
        {{
          label: "EpiHack",
          data: ehVals,
          backgroundColor: ehVals.map(v => v === 1 ? RED + "cc" : SURFACE),
          borderColor: ehVals.map(v => v === 1 ? RED : MUTED),
          borderWidth: 1,
        }},
        {{
          label: "Wastewater",
          data: wwVals.map(v => v < 0 ? 0 : v),
          backgroundColor: wwVals.map(v => v === 1 ? CYAN + "cc" : v < 0 ? "#2a2d3e" : SURFACE),
          borderColor: wwVals.map(v => v < 0 ? "#2a2d3e" : MUTED),
          borderWidth: 1,
        }},
        {{
          label: "ED Syndromic",
          data: edVals.map(v => v < 0 ? 0 : v),
          backgroundColor: edVals.map(v => v === 1 ? ORANGE + "cc" : v < 0 ? "#2a2d3e" : SURFACE),
          borderColor: edVals.map(v => v < 0 ? "#2a2d3e" : MUTED),
          borderWidth: 1,
        }},
      ]
    }},
    options: {{
      indexAxis: "y",
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ labels: {{ color: TEXT, boxWidth: 12 }} }},
        tooltip: {{
          callbacks: {{
            label: (ctx) => {{
              const r   = CONCORDANCE_DATA[ctx.dataIndex];
              const src = ctx.dataset.label;
              if (src === "EpiHack") {{
                return `EpiHack: ${{r.epihack_class.toUpperCase()}} (${{r.epihack_delta >= 0 ? "+" : ""}}${{r.epihack_delta?.toFixed(1)}}pp, n=${{r.epihack_n}})`;
              }} else if (src === "Wastewater") {{
                return r.ww_available
                  ? `WW: conc=${{r.ww_conc?.toFixed(0)}}/100, ${{r.ww_trend}}, ${{r.ww_signal ? "ELEVATED" : "baseline"}}`
                  : "Wastewater: no analogue";
              }} else {{
                return r.ed_available
                  ? `ED: ${{r.ed_pct?.toFixed(2)}}%, ${{r.ed_trend}}, ${{r.ed_signal ? "ABOVE THRESHOLD" : "within threshold"}}`
                  : "ED: no analogue";
              }}
            }}
          }}
        }}
      }},
      scales: {{
        x: {{
          min: 0, max: 1.2,
          ticks: {{ color: TEXT, callback: v => v === 1 ? "ELEVATED" : v === 0 ? "baseline" : "" }},
          grid: {{ color: "#2a2d3e" }},
          title: {{ display: true, text: "Signal Status", color: MUTED }}
        }},
        y: {{
          ticks: {{ color: TEXT, font: {{ size: 10 }} }},
          grid: {{ color: "#2a2d3e" }}
        }}
      }}
    }}
  }});
}})();

// ── Panel B: Tier Distribution Doughnut ────────────────────────────────────
(function buildTierChart() {{
  const ctx = document.getElementById("tierChart").getContext("2d");
  const counts = [1,2,3].map(t => CONCORDANCE_DATA.filter(r => r.tier === t).length);
  new Chart(ctx, {{
    type: "doughnut",
    data: {{
      labels: ["Tier 1 — Full Concordance", "Tier 2 — Partial Concordance", "Tier 3 — Single Source"],
      datasets: [{{
        data: counts,
        backgroundColor: [RED + "bb", ORANGE + "bb", MUTED + "bb"],
        borderColor:      [RED, ORANGE, MUTED],
        borderWidth: 2,
        hoverOffset: 6,
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      cutout: "60%",
      plugins: {{
        legend: {{ position: "right", labels: {{ color: TEXT, boxWidth: 14, padding: 10, font: {{ size: 11 }} }} }},
        tooltip: {{ callbacks: {{ label: (c) => ` ${{c.label}}: ${{c.raw}} syndrome${{c.raw !== 1 ? "s" : ""}}` }} }}
      }}
    }}
  }});
}})();

// ── Panel C: Wastewater Chart ──────────────────────────────────────────────
(function buildWWChart() {{
  const ctx  = document.getElementById("wwChart").getContext("2d");
  const keys = Object.keys(WW_DATA);
  const lbls = keys.map(k => k.replace("sars_cov2","SARS-CoV-2").replace(/_/g," ").replace(/\b\w/g,c=>c.toUpperCase()));
  const curr = keys.map(k => WW_DATA[k].concentration);
  const prior= keys.map(k => WW_DATA[k].prior_concentration);
  const clrs = keys.map(k => WW_DATA[k].signal ? RED + "cc" : BLUE + "cc");

  new Chart(ctx, {{
    type: "bar",
    data: {{
      labels: lbls,
      datasets: [
        {{ label: "Prior month", data: prior, backgroundColor: "#3a3d4ecc", borderColor: MUTED, borderWidth: 1 }},
        {{ label: "Current",     data: curr,  backgroundColor: clrs, borderColor: keys.map(k => WW_DATA[k].signal ? RED : BLUE), borderWidth: 1 }},
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend:    {{ labels: {{ color: TEXT, boxWidth: 12 }} }},
        annotation: {{
          annotations: {{ threshold: {{ type: "line", yMin: 30, yMax: 30, borderColor: ORANGE, borderWidth: 2, borderDash: [5,3], label: {{ content: "Threshold 30", display: true, color: ORANGE }} }} }}
        }}
      }},
      scales: {{
        x: {{ ticks: {{ color: TEXT }}, grid: {{ color: "#2a2d3e" }} }},
        y: {{
          min: 0, max: 80,
          ticks: {{ color: TEXT }},
          grid: {{ color: "#2a2d3e" }},
          title: {{ display: true, text: "Relative Concentration (0–100)", color: MUTED }}
        }}
      }}
    }}
  }});
}})();

// ── Panel D: ED Syndromic Chart ────────────────────────────────────────────
(function buildEDChart() {{
  const ctx  = document.getElementById("edChart").getContext("2d");
  const keys = Object.keys(ED_DATA);
  const lbls = keys.map(k => k.toUpperCase().replace(/_/g," "));
  const pcts = keys.map(k => ED_DATA[k].pct_visits);
  const thrs = keys.map(k => ED_DATA[k].baseline_pct);
  const clrs = keys.map(k => ED_DATA[k].above_threshold ? RED + "cc" : BLUE + "cc");

  new Chart(ctx, {{
    type: "bar",
    data: {{
      labels: lbls,
      datasets: [
        {{ label: "% ED Visits", data: pcts, backgroundColor: clrs,
           borderColor: keys.map(k => ED_DATA[k].above_threshold ? RED : BLUE), borderWidth: 1 }},
        {{ label: "Alert Threshold", data: thrs, type: "line",
           borderColor: ORANGE, borderWidth: 2, borderDash: [5,3],
           pointStyle: "dash", pointRadius: 0, fill: false, backgroundColor: "transparent" }},
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ labels: {{ color: TEXT, boxWidth: 12 }} }} }},
      scales: {{
        x: {{ ticks: {{ color: TEXT, font: {{ size: 10 }} }}, grid: {{ color: "#2a2d3e" }} }},
        y: {{
          ticks: {{ color: TEXT, callback: v => v.toFixed(1) + "%" }},
          grid:  {{ color: "#2a2d3e" }},
          title: {{ display: true, text: "% of Total ED Visits", color: MUTED }}
        }}
      }}
    }}
  }});
}})();

// ── Panel E: Decision Table ────────────────────────────────────────────────
function renderDecisionTable() {{
  const data = filteredSorted();
  const container = document.getElementById("decision-table-container");

  if (data.length === 0) {{
    container.innerHTML = "<p style='color:var(--muted);text-align:center;padding:20px'>No syndromes match current filter.</p>";
    return;
  }}

  let html = `<table>
    <thead><tr>
      <th onclick="cycleSort('syndrome')">Syndrome ↕</th>
      <th onclick="cycleSort('tier')">Tier ↕</th>
      <th>Sources ▲/Avail</th>
      <th>EpiHack Signal</th>
      <th>Wastewater</th>
      <th>ED Syndromic</th>
      <th>Concordance Level</th>
    </tr></thead>
    <tbody>`;

  data.forEach(r => {{
    const tierBadge = `<span class="badge ${{tierClass(r.tier)}}" style="background:${{tierColor(r.tier)}}22;color:${{tierColor(r.tier)}};border:1px solid ${{tierColor(r.tier)}}">TIER ${{r.tier}}</span>`;
    const ehDetail = r.epihack_class === "indef"
      ? `<span style="color:var(--muted)">INDEF (n=${{r.epihack_n}})</span>`
      : `<span class="${{r.epihack_signal ? "sig-up" : "sig-ok"}}">${{r.epihack_class.toUpperCase()}} (${{r.epihack_delta >= 0 ? "+" : ""}}${{r.epihack_delta?.toFixed(1)}}pp)</span>`;
    const wwDetail = r.ww_available
      ? `${{r.ww_signal ? '<span class="sig-up">▲ ' : '<span class="sig-ok">· '}}${{r.ww_conc?.toFixed(0)}}/100 ${{r.ww_trend}}</span>`
      : '<span class="sig-na">N/A</span>';
    const edDetail = r.ed_available
      ? `${{r.ed_signal ? '<span class="sig-up">▲ ' : '<span class="sig-ok">· '}}${{r.ed_pct?.toFixed(2)}}% ${{r.ed_trend}}</span>`
      : '<span class="sig-na">N/A</span>';

    html += `<tr>
      <td style="font-weight:600;color:${{r.color || TEXT}}">${{r.syndrome}}</td>
      <td>${{tierBadge}}</td>
      <td style="text-align:center;font-weight:700;color:${{tierColor(r.tier)}}">${{r.n_elevated}} / ${{r.n_sources_avail}}</td>
      <td>${{ehDetail}}</td>
      <td>${{wwDetail}}</td>
      <td>${{edDetail}}</td>
      <td style="font-size:0.78rem;color:var(--muted)">${{r.concordance_level}}</td>
    </tr>`;
  }});

  html += "</tbody></table>";
  container.innerHTML = html;
}}

function cycleSort(field) {{
  activeSortOrder = field;
  renderDecisionTable();
}}

// ── Panel F: Response Recommendations ─────────────────────────────────────
function renderResponseRecs() {{
  const data = filteredSorted().filter(r => r.tier <= 2);
  const container = document.getElementById("response-recs");

  if (data.length === 0) {{
    container.innerHTML = "<p style='color:var(--muted)'>No Tier 1 or Tier 2 syndromes match current filter.</p>";
    return;
  }}

  let html = "";
  data.forEach(r => {{
    const cls = r.tier === 1 ? "resp-tier1" : "resp-tier2";
    html += `<div class="resp-card ${{cls}}">
      <strong style="color:${{tierColor(r.tier)}}">[TIER ${{r.tier}}] ${{r.syndrome}}</strong><br>
      <span style="color:var(--muted);font-size:0.78rem">${{r.concordance_level}}</span><br>
      <span style="margin-top:6px;display:block">${{r.response_rec}}</span>
    </div>`;
  }});

  container.innerHTML = html;
}}

// ── Initial render ─────────────────────────────────────────────────────────
renderDecisionTable();
renderResponseRecs();
</script>
</body>
</html>"""

    Path(out_path).write_text(html, encoding="utf-8")
    file_size = Path(out_path).stat().st_size / 1024 / 1024
    print(f"[HTML] Saved → {out_path} ({file_size:.2f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="US#9 — Multi-Source Surveillance Triangulation (EpiHack + Wastewater + ED)",
    )
    parser.add_argument("--db",           required=True,
                        help="Path to epihack-mini.db")
    parser.add_argument("--recent",       type=int, default=7,
                        help="Recent window in days (default 7)")
    parser.add_argument("--baseline",     type=int, default=30,
                        help="Baseline window in days (default 30)")
    parser.add_argument("--min-reporters", type=int, default=5,
                        help="Minimum reporters per syndrome for valid signal")
    parser.add_argument("--region",       default="Arizona",
                        help="Region label for narrative (default: Arizona)")
    parser.add_argument("--out",          default="multi_source_triangulation.png",
                        help="Output PNG path")
    args = parser.parse_args()

    out_dir  = os.path.dirname(args.out) or "."
    png_stem = os.path.splitext(os.path.basename(args.out))[0]

    # If --out points to an existing directory, nest inside
    if os.path.isdir(args.out):
        png_path  = os.path.join(args.out, f"{png_stem}.png")
        html_path = os.path.join(args.out, f"{png_stem}.html")
    else:
        png_path  = args.out if args.out.endswith(".png") else args.out + ".png"
        base_dir  = os.path.dirname(png_path) or "."
        html_path = os.path.join(base_dir, f"{png_stem}.html")

    print()
    print("=" * 60)
    print("  US#9 — Multi-Source Surveillance Triangulation")
    print(f"  DB: {args.db}")
    print(f"  Region: {args.region}")
    print(f"  Window: {args.recent}d recent / {args.baseline}d baseline")
    print("=" * 60)

    # ── Step 1: EpiHack signals ──────────────────────────────────────────────
    print("\n[1/5] Loading EpiHack community signals from DB …")
    epihack_df, ref_date = load_epihack_signals(
        args.db, args.recent, args.baseline, args.min_reporters
    )
    print(f"      Ref date: {ref_date.date()}")
    print(f"      Syndromes evaluated: {len(epihack_df)}")
    n_eh_elevated = epihack_df["epihack_signal"].sum()
    print(f"      EpiHack syndromes elevated: {n_eh_elevated}")

    # ── Step 2: Wastewater simulation ────────────────────────────────────────
    print("\n[2/5] Simulating ADHS wastewater signals …")
    counties = list(COUNTY_WW_BASE.keys())
    ww_signals = simulate_wastewater(ref_date, counties)
    n_ww_elevated = sum(v["signal"] for v in ww_signals.values())
    print(f"      Pathogens evaluated: {len(ww_signals)}")
    print(f"      Elevated (conc>30): {n_ww_elevated}")
    for k, v in ww_signals.items():
        status = "ELEVATED" if v["signal"] else "baseline"
        print(f"        {k:<12}: conc={v['concentration']:.1f}, trend={v['trend']}, {status}")

    # ── Step 3: ED syndromic simulation ─────────────────────────────────────
    print("\n[3/5] Simulating BioSense/NSSP ED syndromic signals …")
    ed_signals = simulate_ed_syndromic(ref_date, args.region)
    n_ed_elevated = sum(v["above_threshold"] for v in ed_signals.values())
    print(f"      ED categories evaluated: {len(ed_signals)}")
    print(f"      Above alert threshold: {n_ed_elevated}")
    for k, v in ed_signals.items():
        status = "ABOVE THRESHOLD" if v["above_threshold"] else "within threshold"
        print(f"        {k:<22}: {v['pct_visits']:.3f}%, trend={v['trend']}, {status}")

    # ── Step 4: Concordance & tiers ──────────────────────────────────────────
    print("\n[4/5] Computing cross-source concordance and tier assignments …")
    concordance_df = compute_concordance(epihack_df, ww_signals, ed_signals)
    tier_counts = concordance_df["tier"].value_counts().sort_index()
    for tier, cnt in tier_counts.items():
        label = "Full Concordance" if tier == 1 else "Partial" if tier == 2 else "Single/None"
        print(f"      Tier {tier} ({label}): {cnt} syndrome(s)")

    # Decision table to stdout
    print_decision_table(concordance_df, args.region, ref_date, args.recent)

    # ── Step 5: LLM prompt + narrative ──────────────────────────────────────
    print("\n[5/5] Building LLM prompt and generating narrative …")
    llm_prompt = build_llm_prompt(
        concordance_df, ww_signals, ed_signals,
        args.region, ref_date, args.recent
    )
    narrative = simulate_llm_narrative(concordance_df, args.region, ref_date, args.recent)

    # ── Outputs ──────────────────────────────────────────────────────────────
    print()
    fig_triangulation(concordance_df, ww_signals, ed_signals,
                      args.region, ref_date, args.recent, png_path)

    build_html_dashboard(
        concordance_df, ww_signals, ed_signals,
        args.region, ref_date, args.recent, args.baseline,
        llm_prompt, narrative, html_path
    )

    print()
    print("All outputs saved successfully.")
    print(f"  PNG  → {png_path}")
    print(f"  HTML → {html_path}")
    print()


if __name__ == "__main__":
    main()
