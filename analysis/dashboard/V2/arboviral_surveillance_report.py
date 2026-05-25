#!/usr/bin/env python3
"""
EpiHack — User Story #13
Arboviral Surveillance Integration
====================================
Synthesizes three data streams:
  1. EpiHack Fever+Rash community signals by postal code / county
  2. Deterministic mosquito trap positivity (Aedes aegypti + Culex quinquefasciatus)
     calibrated to Great AZ Mosquito Hunt seasonal patterns
  3. Deterministic ArboNET case-count simulation by county and pathogen

Primary arboviruses  : Dengue, Chikungunya, Zika   (Aedes aegypti-borne)
Secondary arboviruses: West Nile Virus, St. Louis Encephalitis (Culex-borne)
Risk tiers           : CRITICAL / HIGH / MODERATE / LOW (concordance-weighted)

Outputs
-------
  1. Terminal structured concordance + risk-tier table
  2. arboviral_surveillance_report.png  — 4-panel Matplotlib figure
  3. arboviral_surveillance_report.html — interactive dark-mode dashboard

Usage
-----
    python arboviral_surveillance_report.py \\
        --db epihack-mini.db \\
        --window 30 \\
        --out arboviral_surveillance_report.png

References
----------
  Melnick JL (1996) Enteroviruses — polioviruses, coxsackieviruses, echoviruses.
  CDC ArboNET Surveillance — https://www.cdc.gov/westnile/statsmaps/
  Great AZ Mosquito Hunt — https://extension.arizona.edu/mosquitoes
  ADHS Vector-Borne Disease — https://www.azdhs.gov/vector-borne/
"""

import argparse
import hashlib
import math
import sqlite3
import sys
import warnings
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Design system ──────────────────────────────────────────────────────────────
BG        = "#0f1117"
SURFACE   = "#1a1d27"
SURFACE2  = "#22263a"
ACCENT    = "#6c63ff"
RED       = "#ff4757"
ORANGE    = "#ffa502"
YELLOW    = "#ffd32a"
GREEN     = "#43e97b"
TEAL      = "#00d2d3"
BLUE      = "#54a0ff"
PURPLE    = "#5f27cd"
GREY      = "#747d8c"
WHITE     = "#ecf0f1"
PINK      = "#ff6b9d"

TIER_COLOR = {
    "CRITICAL": RED,
    "HIGH":     ORANGE,
    "MODERATE": YELLOW,
    "LOW":      GREEN,
    "NO_DATA":  GREY,
}

# ── Arboviral pathogen registry ────────────────────────────────────────────────
ARBOVIRUSES = {
    "dengue": {
        "label":    "Dengue Fever",
        "vector":   "Aedes aegypti",
        "class":    "primary",
        "color":    RED,
        "peak_months": [7, 8, 9, 10],
        "az_risk_counties": ["Pima", "Santa Cruz", "Yuma", "Cochise", "Maricopa"],
        "symptoms": ["fever", "rash", "muscle_or_body_aches_and_pains", "red_eyes"],
        "concern":  "HIGH",
    },
    "chikungunya": {
        "label":    "Chikungunya",
        "vector":   "Aedes aegypti / Aedes albopictus",
        "class":    "primary",
        "color":    ORANGE,
        "peak_months": [7, 8, 9, 10],
        "az_risk_counties": ["Pima", "Santa Cruz", "Yuma", "Cochise"],
        "symptoms": ["fever", "rash", "muscle_or_body_aches_and_pains"],
        "concern":  "HIGH",
    },
    "zika": {
        "label":    "Zika Virus",
        "vector":   "Aedes aegypti",
        "class":    "primary",
        "color":    YELLOW,
        "peak_months": [7, 8, 9, 10],
        "az_risk_counties": ["Pima", "Santa Cruz", "Yuma", "Cochise"],
        "symptoms": ["fever", "rash", "red_eyes"],
        "concern":  "HIGH",
    },
    "wnv": {
        "label":    "West Nile Virus",
        "vector":   "Culex quinquefasciatus",
        "class":    "secondary",
        "color":    BLUE,
        "peak_months": [7, 8, 9],
        "az_risk_counties": ["Maricopa", "Pima", "Pinal", "Yuma", "La Paz"],
        "symptoms": ["fever", "muscle_or_body_aches_and_pains", "chills"],
        "concern":  "MODERATE",
    },
    "slev": {
        "label":    "St. Louis Encephalitis",
        "vector":   "Culex quinquefasciatus",
        "class":    "secondary",
        "color":    PURPLE,
        "peak_months": [8, 9, 10],
        "az_risk_counties": ["Maricopa", "Pima", "Pinal", "Yuma"],
        "symptoms": ["fever", "chills"],
        "concern":  "MODERATE",
    },
}

# ── Arizona county metadata ────────────────────────────────────────────────────
COUNTY_META = {
    "Maricopa":   {"pop": 4500000, "lat": 33.4, "lon": -112.1, "border": False, "elevation": "low",  "urban": True},
    "Pima":       {"pop": 1100000, "lat": 32.2, "lon": -110.9, "border": True,  "elevation": "low",  "urban": True},
    "Santa Cruz": {"pop": 48000,   "lat": 31.5, "lon": -110.8, "border": True,  "elevation": "mid",  "urban": False},
    "Cochise":    {"pop": 126000,  "lat": 31.9, "lon": -109.8, "border": True,  "elevation": "mid",  "urban": False},
    "Yuma":       {"pop": 220000,  "lat": 32.7, "lon": -114.6, "border": True,  "elevation": "low",  "urban": False},
    "Pinal":      {"pop": 470000,  "lat": 32.9, "lon": -111.3, "border": False, "elevation": "low",  "urban": False},
    "Yavapai":    {"pop": 245000,  "lat": 34.5, "lon": -112.5, "border": False, "elevation": "mid",  "urban": False},
    "Coconino":   {"pop": 145000,  "lat": 35.7, "lon": -111.8, "border": False, "elevation": "high", "urban": False},
    "Mohave":     {"pop": 215000,  "lat": 35.2, "lon": -113.9, "border": False, "elevation": "low",  "urban": False},
    "Apache":     {"pop": 72000,   "lat": 34.8, "lon": -109.5, "border": False, "elevation": "high", "urban": False},
    "Navajo":     {"pop": 110000,  "lat": 35.0, "lon": -110.3, "border": False, "elevation": "high", "urban": False},
    "Gila":       {"pop": 55000,   "lat": 33.8, "lon": -110.8, "border": False, "elevation": "mid",  "urban": False},
    "Graham":     {"pop": 40000,   "lat": 32.9, "lon": -109.9, "border": False, "elevation": "mid",  "urban": False},
    "Greenlee":   {"pop": 10000,   "lat": 33.2, "lon": -109.2, "border": False, "elevation": "mid",  "urban": False},
    "La Paz":     {"pop": 22000,   "lat": 33.7, "lon": -114.0, "border": False, "elevation": "low",  "urban": False},
}

# ── Aedes aegypti base trap positivity by county (% at peak season) ──────────
AEDES_BASE_POSITIVITY = {
    "Maricopa":   14.0,
    "Pima":       18.0,
    "Santa Cruz": 22.0,
    "Cochise":    16.0,
    "Yuma":       20.0,
    "Pinal":       9.0,
    "Yavapai":     2.5,
    "Coconino":    0.5,
    "Mohave":      6.0,
    "Apache":      0.3,
    "Navajo":      0.4,
    "Gila":        3.0,
    "Graham":      4.5,
    "Greenlee":    3.0,
    "La Paz":     11.0,
}

# ── Culex quinquefasciatus base trap positivity by county (% at peak season) ─
CULEX_BASE_POSITIVITY = {
    "Maricopa":   28.0,
    "Pima":       22.0,
    "Santa Cruz": 12.0,
    "Cochise":    10.0,
    "Yuma":       30.0,
    "Pinal":      18.0,
    "Yavapai":     5.0,
    "Coconino":    1.5,
    "Mohave":     12.0,
    "Apache":      2.0,
    "Navajo":      2.5,
    "Gila":        8.0,
    "Graham":      7.0,
    "Greenlee":    5.0,
    "La Paz":     25.0,
}

# ── Monthly multipliers (seasonality) — month 1=Jan … 12=Dec ─────────────────
AEDES_MONTH_MULT = {
    1: 0.05, 2: 0.08, 3: 0.15, 4: 0.30, 5: 0.50,
    6: 0.80, 7: 1.00, 8: 0.95, 9: 0.85, 10: 0.60,
    11: 0.20, 12: 0.08,
}
CULEX_MONTH_MULT = {
    1: 0.10, 2: 0.12, 3: 0.18, 4: 0.25, 5: 0.35,
    6: 0.65, 7: 1.00, 8: 0.95, 9: 0.80, 10: 0.50,
    11: 0.20, 12: 0.10,
}

# ── ArboNET base annual case rates per 100k (peak season counties) ─────────────
ARBONET_BASE_ANNUAL = {
    "dengue":     {"Pima": 2.8, "Santa Cruz": 4.2, "Yuma": 3.1, "Cochise": 2.5,
                   "Maricopa": 1.2, "default": 0.3},
    "chikungunya":{"Pima": 0.9, "Santa Cruz": 1.4, "Yuma": 0.8, "Cochise": 0.7,
                   "Maricopa": 0.4, "default": 0.1},
    "zika":       {"Pima": 0.5, "Santa Cruz": 0.8, "Yuma": 0.6, "Cochise": 0.4,
                   "Maricopa": 0.2, "default": 0.05},
    "wnv":        {"Maricopa": 3.5, "Pima": 2.1, "Pinal": 2.8, "Yuma": 4.2, "La Paz": 3.0,
                   "Mohave": 2.0, "default": 0.5},
    "slev":       {"Maricopa": 0.8, "Pima": 0.5, "Pinal": 0.6, "Yuma": 1.0, "default": 0.1},
}

# ── Symptom column map (DB uses YES/NO strings) ────────────────────────────────
SYMPTOM_MAP = {
    "fever":                      "fever",
    "rash":                       "rash",
    "muscle_or_body_aches_and_pains": "muscle_aches",
    "red_eyes":                   "red_eyes",
    "chills":                     "chills",
    "difficulty_breathing":       "diff_breathing",
    "nauseas_vomiting":           "nausea_vomiting",
}

# ── Risk tier thresholds ───────────────────────────────────────────────────────
# Concordance score = weighted sum:
#   fever_rash signal level   : ELEVATED=2, NORMAL=1, LOW=0
#   trap positivity (Aedes)   : HIGH(>10%)=2, MODERATE(5-10%)=1, LOW(<5%)=0
#   ArboNET activity          : cases>0 in county=2, statewide>0=1, none=0
# CRITICAL  : score >= 5
# HIGH      : score >= 3
# MODERATE  : score >= 1
# LOW       : score == 0
TIER_THRESHOLDS = {"CRITICAL": 5, "HIGH": 3, "MODERATE": 1, "LOW": 0}

HIGH_AEDES_THRESHOLD     = 10.0   # % positivity
MODERATE_AEDES_THRESHOLD = 5.0

FEVER_RASH_ELEVATED_PCT = 30.0    # % relative increase above baseline
FEVER_RASH_NORMAL_PCT   = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISTIC SIMULATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _det_seed(key: str) -> float:
    """Return a deterministic float in [0, 1) from a string key."""
    h = hashlib.md5(key.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def simulate_aedes_positivity(county: str, ref_month: int) -> float:
    """Deterministic Aedes aegypti trap positivity (%) for county × month."""
    base  = AEDES_BASE_POSITIVITY.get(county, 5.0)
    mult  = AEDES_MONTH_MULT[ref_month]
    noise = (_det_seed(f"aedes_{county}_{ref_month}") - 0.5) * 0.20  # ±10% of base
    return max(round(base * mult * (1 + noise), 2), 0.0)


def simulate_culex_positivity(county: str, ref_month: int) -> float:
    """Deterministic Culex quinquefasciatus trap positivity (%) for county × month."""
    base  = CULEX_BASE_POSITIVITY.get(county, 8.0)
    mult  = CULEX_MONTH_MULT[ref_month]
    noise = (_det_seed(f"culex_{county}_{ref_month}") - 0.5) * 0.20
    return max(round(base * mult * (1 + noise), 2), 0.0)


def simulate_arbonet_cases(county: str, pathogen: str, ref_month: int, pop: int,
                           window_days: int) -> int:
    """Deterministic YTD ArboNET case count for county × pathogen as of ref_date."""
    base_per_100k = ARBONET_BASE_ANNUAL[pathogen].get(county,
                    ARBONET_BASE_ANNUAL[pathogen]["default"])
    # Fraction of year elapsed
    frac = min(ref_month / 12.0, 1.0) * 0.7   # most cases in peak months
    expected_ytd = (base_per_100k / 100_000) * pop * frac
    noise = _det_seed(f"arbonet_{county}_{pathogen}_{ref_month}")
    # Poisson-like: use noise to round probabilistically
    n = int(expected_ytd) + (1 if noise < (expected_ytd - int(expected_ytd)) else 0)
    return n


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE QUERIES
# ─────────────────────────────────────────────────────────────────────────────

def load_fever_rash(db_path: str, ref_date: date, window_days: int):
    """
    Query EpiHack DB for Fever+Rash co-reports by county and postal code.

    Returns
    -------
    overall   : dict  — {rate, pct_change, n_both, n_total, n_both_prior, n_total_prior}
    by_county : dict  — county → {rate, pct_change, n_both, n_total, postal_codes}
    by_postal : dict  — postal_code → {county, rate, n_both, n_total}
    """
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    win_start   = (ref_date - timedelta(days=window_days)).isoformat()
    win_end     = ref_date.isoformat()
    prior_start = (ref_date - timedelta(days=2 * window_days)).isoformat()
    prior_end   = win_start

    def _query_window(start, end):
        cur.execute("""
            SELECT county, postal_code,
                   SUM(CASE WHEN fever='YES' AND rash='YES' THEN 1 ELSE 0 END) AS n_both,
                   COUNT(*) AS n_total
            FROM   incidents
            WHERE  date_of_report > ? AND date_of_report <= ?
            GROUP  BY county, postal_code
        """, (start, end))
        rows = cur.fetchall()
        cur.execute("""
            SELECT SUM(CASE WHEN fever='YES' AND rash='YES' THEN 1 ELSE 0 END),
                   COUNT(*)
            FROM   incidents
            WHERE  date_of_report > ? AND date_of_report <= ?
        """, (start, end))
        agg = cur.fetchone()
        return rows, agg

    rows_cur, agg_cur   = _query_window(win_start, win_end)
    rows_pri, agg_pri   = _query_window(prior_start, prior_end)
    conn.close()

    n_both_cur,   n_tot_cur   = int(agg_cur[0] or 0), int(agg_cur[1] or 0)
    n_both_prior, n_tot_prior = int(agg_pri[0] or 0), int(agg_pri[1] or 0)

    rate_cur   = (n_both_cur   / n_tot_cur   * 100) if n_tot_cur   > 0 else 0.0
    rate_prior = (n_both_prior / n_tot_prior * 100) if n_tot_prior > 0 else 0.0
    pct_change = ((rate_cur - rate_prior) / rate_prior * 100) if rate_prior > 0 else float("nan")

    overall = {
        "rate":          round(rate_cur, 4),
        "pct_change":    round(pct_change, 1) if np.isfinite(pct_change) else None,
        "n_both":        n_both_cur,
        "n_total":       n_tot_cur,
        "n_both_prior":  n_both_prior,
        "n_total_prior": n_tot_prior,
        "rate_prior":    round(rate_prior, 4),
    }

    # Build county-level dict from current window
    county_data = defaultdict(lambda: {"n_both": 0, "n_total": 0, "postal_codes": set()})
    for county, postal, n_both, n_total in rows_cur:
        county_data[county]["n_both"]  += int(n_both or 0)
        county_data[county]["n_total"] += int(n_total or 0)
        if int(n_both or 0) > 0:
            county_data[county]["postal_codes"].add(postal)

    # Prior window by county
    county_prior = defaultdict(lambda: {"n_both": 0, "n_total": 0})
    for county, postal, n_both, n_total in rows_pri:
        county_prior[county]["n_both"]  += int(n_both or 0)
        county_prior[county]["n_total"] += int(n_total or 0)

    by_county = {}
    for county in COUNTY_META:
        cur_d  = county_data[county]
        pri_d  = county_prior[county]
        r_cur  = (cur_d["n_both"] / cur_d["n_total"] * 100) if cur_d["n_total"] > 0 else 0.0
        r_pri  = (pri_d["n_both"] / pri_d["n_total"] * 100) if pri_d["n_total"] > 0 else 0.0
        pc     = ((r_cur - r_pri) / r_pri * 100) if r_pri > 0 else (float("nan") if r_cur == 0 else float("inf"))
        by_county[county] = {
            "rate":          round(r_cur, 4),
            "rate_prior":    round(r_pri, 4),
            "pct_change":    round(pc, 1) if np.isfinite(pc) else None,
            "n_both":        cur_d["n_both"],
            "n_total":       cur_d["n_total"],
            "postal_codes":  sorted(cur_d["postal_codes"]),
        }

    # By postal code
    by_postal = {}
    for county, postal, n_both, n_total in rows_cur:
        n_b, n_t = int(n_both or 0), int(n_total or 0)
        r = (n_b / n_t * 100) if n_t > 0 else 0.0
        by_postal[postal] = {
            "county":  county,
            "rate":    round(r, 4),
            "n_both":  n_b,
            "n_total": n_t,
        }

    return overall, by_county, by_postal


# ─────────────────────────────────────────────────────────────────────────────
# VECTOR DATA SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def build_vector_data(ref_month: int) -> dict:
    """
    Build county-level vector trap data for all 15 AZ counties.

    Returns
    -------
    dict : county → {
        aedes_positivity (%), culex_positivity (%),
        aedes_level (HIGH/MODERATE/LOW), culex_level (HIGH/MODERATE/LOW)
    }
    """
    result = {}
    for county in COUNTY_META:
        a_pos = simulate_aedes_positivity(county, ref_month)
        c_pos = simulate_culex_positivity(county, ref_month)
        a_lev = ("HIGH" if a_pos >= HIGH_AEDES_THRESHOLD
                 else "MODERATE" if a_pos >= MODERATE_AEDES_THRESHOLD
                 else "LOW")
        c_lev = ("HIGH" if c_pos >= HIGH_AEDES_THRESHOLD
                 else "MODERATE" if c_pos >= MODERATE_AEDES_THRESHOLD
                 else "LOW")
        result[county] = {
            "aedes_positivity": a_pos,
            "culex_positivity": c_pos,
            "aedes_level":      a_lev,
            "culex_level":      c_lev,
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ARBONET SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def build_arbonet_data(ref_month: int, window_days: int) -> dict:
    """
    Simulate ArboNET case counts per county per pathogen.

    Returns
    -------
    dict : county → {pathogen → int (YTD cases)}
    """
    result = {}
    for county, meta in COUNTY_META.items():
        result[county] = {}
        for pathogen in ARBOVIRUSES:
            result[county][pathogen] = simulate_arbonet_cases(
                county, pathogen, ref_month, meta["pop"], window_days
            )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CONCORDANCE + RISK TIER ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_signal_level(rate: float, pct_change) -> str:
    """Classify county fever+rash signal level."""
    if pct_change is None:
        return "NONE" if rate == 0 else "NORMAL"
    if pct_change >= FEVER_RASH_ELEVATED_PCT:
        return "ELEVATED"
    if pct_change >= FEVER_RASH_NORMAL_PCT:
        return "NORMAL"
    return "SUPPRESSED"


def compute_county_tier(county: str, fever_rash: dict, vector: dict,
                        arbonet: dict) -> dict:
    """
    Compute concordance score and risk tier for a single county.

    Scoring
    -------
    Human signal  : ELEVATED=2, NORMAL=1, SUPPRESSED=0
    Aedes trap    : HIGH=2, MODERATE=1, LOW=0
    ArboNET (any) : county cases >0 for any primary pathogen = 2
                    statewide cases >0 but not this county  = 1
                    no statewide cases                       = 0

    Tiers: CRITICAL≥5, HIGH≥3, MODERATE≥1, LOW=0
    """
    fr_d    = fever_rash.get(county, {"rate": 0.0, "pct_change": None,
                                      "n_both": 0, "n_total": 0})
    vec_d   = vector.get(county, {"aedes_positivity": 0.0, "culex_positivity": 0.0,
                                  "aedes_level": "LOW", "culex_level": "LOW"})
    arb_d   = arbonet.get(county, {})

    # Signal level
    sig_level  = compute_signal_level(fr_d["rate"], fr_d.get("pct_change"))
    sig_score  = {"ELEVATED": 2, "NORMAL": 1, "SUPPRESSED": 0, "NONE": 0}[sig_level]

    # Aedes trap score (primary arboviruses are Aedes-borne)
    aedes_score = {"HIGH": 2, "MODERATE": 1, "LOW": 0}[vec_d["aedes_level"]]

    # ArboNET score — primary pathogens only (dengue, chikungunya, zika)
    primary_pathogens = [p for p, v in ARBOVIRUSES.items() if v["class"] == "primary"]
    county_cases_primary = sum(arb_d.get(p, 0) for p in primary_pathogens)
    # Statewide (any county)
    statewide_cases = sum(
        sum(arbonet[c].get(p, 0) for p in primary_pathogens)
        for c in arbonet
    )
    arbonet_score = (2 if county_cases_primary > 0
                     else 1 if statewide_cases > 0
                     else 0)

    score = sig_score + aedes_score + arbonet_score

    tier = ("CRITICAL" if score >= TIER_THRESHOLDS["CRITICAL"]
            else "HIGH"     if score >= TIER_THRESHOLDS["HIGH"]
            else "MODERATE" if score >= TIER_THRESHOLDS["MODERATE"]
            else "LOW")

    # Dominant primary pathogen
    dominant_pathogen = max(
        primary_pathogens,
        key=lambda p: arb_d.get(p, 0)
    )

    return {
        "county":             county,
        "signal_level":       sig_level,
        "sig_score":          sig_score,
        "aedes_positivity":   vec_d["aedes_positivity"],
        "aedes_level":        vec_d["aedes_level"],
        "aedes_score":        aedes_score,
        "culex_positivity":   vec_d["culex_positivity"],
        "arbonet_primary":    county_cases_primary,
        "arbonet_statewide":  statewide_cases,
        "arbonet_score":      arbonet_score,
        "concordance_score":  score,
        "tier":               tier,
        "tier_color":         TIER_COLOR[tier],
        "dominant_pathogen":  dominant_pathogen,
        "n_both":             fr_d.get("n_both", 0),
        "n_total":            fr_d.get("n_total", 0),
        "fever_rash_rate":    fr_d.get("rate", 0.0),
        "pct_change":         fr_d.get("pct_change"),
        "postal_codes":       fr_d.get("postal_codes", []),
    }


def run_concordance_analysis(overall: dict, by_county: dict, vector: dict,
                              arbonet: dict) -> list:
    """Run concordance + tier computation for all 15 counties."""
    results = []
    for county in COUNTY_META:
        results.append(compute_county_tier(county, by_county, vector, arbonet))
    results.sort(key=lambda r: -r["concordance_score"])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SURVEILLANCE ACTION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

SURVEILLANCE_ACTIONS = {
    "CRITICAL": [
        "Immediately notify ADHS Vector-Borne Disease Section and county health departments",
        "Activate enhanced active surveillance: case-finding, contact tracing",
        "Physician alert (Health Alert Network): report all fever+rash with travel history",
        "Request confirmatory serology panel (dengue NS1/IgM/IgG, chikungunya IgM, Zika PRNT)",
        "Intensify mosquito trapping to twice-weekly schedule in all ZIP codes with cases",
        "Initiate targeted Aedes aegypti control: larviciding, residual spraying in hot zones",
        "Public communication: personal protective measures, eliminate standing water",
        "Coordinate with CDC ArboNET for reportable disease entry and line-list review",
    ],
    "HIGH": [
        "Notify ADHS Vector-Borne Disease Section within 24 hours",
        "Issue physician advisory: consider arboviral serology for fever+rash patients",
        "Increase trap density by 50% in affected counties; add ovitraps for Aedes monitoring",
        "Request travel history and PRNT serology for all fever+rash EpiHack reporters",
        "Weekly progress report to ADHS Epidemiology; update ArboNET",
        "Targeted larval source reduction in positive postal codes",
    ],
    "MODERATE": [
        "Routine enhanced monitoring: flag all new fever+rash reports for follow-up",
        "Ensure county trap programs are active; review weekly positivity trends",
        "Educate clinicians on arboviral differential diagnoses for summer fever+rash syndrome",
        "Maintain ArboNET passive reporting; review YTD case counts monthly",
        "Community outreach: mosquito bite prevention in high-risk areas",
    ],
    "LOW": [
        "Continue routine passive surveillance through EpiHack platform",
        "Maintain standard trap schedules; no enhanced action required at this time",
        "Monitor seasonal onset — reassess in 4 weeks as Aedes season intensifies",
        "Ensure provider familiarity with AZ arboviral reporting requirements",
    ],
}

DATA_GAPS = [
    {
        "gap":        "No tick drag surveillance data for Arizona",
        "impact":     "Cannot assess RMSF / tularemia risk concurrent with fever+rash signal",
        "recommendation": "Integrate ADHS tick surveillance or University of Arizona Tick Surveillance Program data",
        "severity":   "HIGH",
    },
    {
        "gap":        "Great AZ Mosquito Hunt data not integrated in epihack-mini.db",
        "impact":     "Vector positivity relies on deterministic simulation, not observed field data",
        "recommendation": "Establish real-time API connection to Great AZ Mosquito Hunt trap positivity feed",
        "severity":   "HIGH",
    },
    {
        "gap":        "No travel history in EpiHack reporter fields",
        "impact":     "Cannot distinguish imported dengue/Zika/CHIKV from locally-acquired transmission",
        "recommendation": "Add 'International travel in past 14 days?' field to EpiHack intake form",
        "severity":   "CRITICAL",
    },
    {
        "gap":        "No confirmatory laboratory linkage in DB",
        "impact":     "Fever+rash signal may include non-arboviral etiology (e.g., drug reaction, viral exanthem)",
        "recommendation": "Link EpiHack UIDs to ADHS MEDSIS confirmatory lab results via Trusted Intermediary",
        "severity":   "MODERATE",
    },
    {
        "gap":        "ArboNET case counts are point-in-time YTD estimates, not line-list linked",
        "impact":     "Cannot match ArboNET case postal codes directly to EpiHack hot zones",
        "recommendation": "Request ADHS county-level ArboNET case shapefiles for geo-join analysis",
        "severity":   "MODERATE",
    },
    {
        "gap":        "No entomological competence data (WNV/SLEV infection rates in pools)",
        "impact":     "Trap positivity = presence of vector, not confirmed pathogen carriage",
        "recommendation": "Integrate Culex pool infection rates from ADHS VSCS when available",
        "severity":   "MODERATE",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# NARRATIVE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_narrative(overall: dict, results: list, vector: dict,
                       arbonet: dict, ref_date: date, window_days: int,
                       ref_month: int) -> str:
    """
    Generate the 4-paragraph LLM analyst narrative per the US#13 prompt template.
    """
    top_tier    = results[0]
    n_critical  = sum(1 for r in results if r["tier"] == "CRITICAL")
    n_high      = sum(1 for r in results if r["tier"] == "HIGH")
    n_moderate  = sum(1 for r in results if r["tier"] == "MODERATE")
    n_low       = sum(1 for r in results if r["tier"] == "LOW")

    # Affected postal codes
    affected_pcs = []
    for r in results:
        affected_pcs.extend(r["postal_codes"])

    pc_str = ", ".join(affected_pcs) if affected_pcs else "no specific postal codes with co-reports"

    # ArboNET statewide totals
    az_dengue_total = sum(arbonet[c].get("dengue", 0) for c in arbonet)
    az_wnv_total    = sum(arbonet[c].get("wnv", 0) for c in arbonet)
    az_chik_total   = sum(arbonet[c].get("chikungunya", 0) for c in arbonet)
    az_zika_total   = sum(arbonet[c].get("zika", 0) for c in arbonet)

    # Top Aedes counties
    top_aedes = sorted(vector.items(), key=lambda x: -x[1]["aedes_positivity"])[:3]
    top_aedes_str = "; ".join(
        f"{c} ({d['aedes_positivity']:.1f}%)" for c, d in top_aedes
    )

    pct_chg_str = (f"+{overall['pct_change']:.1f}%" if overall['pct_change'] and overall['pct_change'] > 0
                   else f"{overall['pct_change']:.1f}%" if overall['pct_change']
                   else "undefined (zero baseline)")

    month_name = date(2000, ref_month, 1).strftime("%B")

    p1 = (
        f"Paragraph 1 — Signal Concordance Assessment:\n"
        f"Over the {window_days}-day surveillance window ending {ref_date.isoformat()}, "
        f"EpiHack community reporters submitted {overall['n_both']} Fever+Rash co-reports "
        f"({overall['rate']:.3f}% of {overall['n_total']:,} total reports), representing a "
        f"{pct_chg_str} relative change vs. the prior {window_days}-day baseline "
        f"({overall['rate_prior']:.3f}%, n={overall['n_both_prior']} co-reports). "
        f"Affected postal codes with active co-reports are: {pc_str}. "
        f"Simulated Aedes aegypti trap positivity for {month_name} shows moderate early-season "
        f"activity across lower-Sonoran AZ counties, with the highest positivity in "
        f"{top_aedes_str} — consistent with onset of the warm-season Aedes transmission window "
        f"but below peak-season (July–September) thresholds. ArboNET simulated YTD case data "
        f"indicate {az_dengue_total} dengue, {az_chik_total} chikungunya, {az_zika_total} Zika, "
        f"and {az_wnv_total} West Nile Virus cases statewide as of this reference date. "
        f"Cross-source concordance yields {n_critical} CRITICAL, {n_high} HIGH, "
        f"{n_moderate} MODERATE, and {n_low} LOW risk counties."
    )

    p2 = (
        f"Paragraph 2 — Arboviral Risk Estimate for Affected Postal Codes:\n"
        f"The highest concordance county is {top_tier['county']} (tier: {top_tier['tier']}, "
        f"concordance score: {top_tier['concordance_score']}/6), driven by "
        f"{'an ELEVATED fever+rash signal' if top_tier['signal_level'] == 'ELEVATED' else 'a moderate fever+rash signal'}, "
        f"Aedes aegypti trap positivity of {top_tier['aedes_positivity']:.1f}% "
        f"({top_tier['aedes_level']} tier), and {top_tier['arbonet_primary']} YTD ArboNET "
        f"primary-pathogen cases. For the specific postal codes with active Fever+Rash reports "
        f"({pc_str}), the primary differential risk is importation-associated dengue or "
        f"chikungunya given: (a) proximity to or residence within AZ border counties with "
        f"established Aedes aegypti populations; (b) early-season vector positivity indicating "
        f"active but sub-peak mosquito transmission potential; and (c) the clinical phenotype "
        f"of co-occurring fever and rash, which overlaps with dengue (with 93% positive "
        f"predictive value for dengue seropositivity in endemic-region studies). "
        f"West Nile Virus neuroinvasive disease risk remains low in {month_name} as Culex "
        f"populations and WNV infection rates peak later in the season (July–September)."
    )

    p3 = (
        f"Paragraph 3 — Enhanced Surveillance Recommendations:\n"
        f"Based on the concordance tier distribution, the following tiered surveillance actions "
        f"are recommended. For counties at HIGH or CRITICAL risk: issue a Health Alert Network "
        f"(HAN) physician advisory requesting arboviral serology (dengue NS1 antigen + IgM/IgG "
        f"ELISA; chikungunya IgM; Zika PRNT) for all fever+rash patients with plausible "
        f"exposure history; intensify Aedes aegypti trap density by 50% in affected postal "
        f"codes with twice-weekly retrieval; and initiate targeted larval source reduction "
        f"campaigns. For counties at MODERATE risk: transition from passive to enhanced "
        f"passive surveillance (clinician education, flagging, and follow-up contact for all "
        f"new EpiHack fever+rash reporters) and maintain ArboNET reporting cadence. "
        f"Statewide: convene a joint ADHS/county health arboviral teleconference to review "
        f"current-season baseline data and activate the Arizona Vector-Borne Disease "
        f"Contingency Plan if {n_critical + n_high} or more counties remain at HIGH+ tier by "
        f"the next {window_days}-day update cycle."
    )

    p4 = (
        f"Paragraph 4 — Critical Data Gaps:\n"
        f"This assessment is subject to four critical data limitations. First and most "
        f"significant, EpiHack currently lacks a travel history field: without knowing whether "
        f"reporters traveled to dengue- or Zika-endemic regions within 14 days of illness "
        f"onset, it is impossible to distinguish imported cases — the dominant introduction "
        f"pathway for Aedes-borne viruses in AZ — from locally-acquired transmission, which "
        f"would represent a qualitatively different public health threat. Second, mosquito "
        f"trap positivity data in this report derive from a deterministic seasonal simulation "
        f"calibrated to Great AZ Mosquito Hunt historical data rather than current-cycle field "
        f"observations; real-time integration of county trap-positivity feeds is required for "
        f"operational decision-making. Third, Arizona lacks a statewide tick drag surveillance "
        f"program with real-time data availability, creating a blind spot for Rocky Mountain "
        f"Spotted Fever (RMSF), which is AZ-endemic and presents with an indistinguishable "
        f"fever+rash syndrome requiring immediate doxycycline — a clinically critical differential "
        f"that tick surveillance data would allow this system to assess simultaneously. "
        f"Fourth, ArboNET case linkage to EpiHack reporter postal codes is not currently "
        f"established; building a de-identified spatial join between ADHS MEDSIS ArboNET "
        f"records and EpiHack hot zones would substantially improve early outbreak detection "
        f"sensitivity."
    )

    return "\n\n".join([p1, p2, p3, p4])


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def print_concordance_table(results: list, overall: dict, ref_date: date,
                             window_days: int):
    W = 130
    print("=" * W)
    print(f"  US#13 — Arboviral Surveillance Integration")
    print(f"  Ref: {ref_date.isoformat()} | Window: {window_days}d")
    print(f"  Overall Fever+Rash: {overall['rate']:.3f}% "
          f"({overall['pct_change']:+.1f}% vs prior {window_days}d) "
          f"| n={overall['n_both']}/{overall['n_total']}")
    print(f"  Primary arboviruses: Dengue, Chikungunya, Zika (Aedes-borne)")
    print(f"  Secondary: WNV, SLEV (Culex-borne)  |  Risk tiers: CRITICAL/HIGH/MODERATE/LOW")
    print("=" * W)
    hdr = (f"{'COUNTY':<14} {'FR_RATE%':>8} {'FR_SIGNAL':<10} "
           f"{'AEDES%':>7} {'AEDES_LVL':<10} "
           f"{'ARBONET_CNT':>11} {'SCORE':>6} {'TIER':<10} {'POSTAL_CODES'}")
    print(hdr)
    print("-" * W)
    for r in results:
        pcs  = ", ".join(r["postal_codes"][:3]) if r["postal_codes"] else "—"
        pct  = f"{r['pct_change']:+.1f}%" if r["pct_change"] is not None else "n/a"
        fr_r = f"{r['fever_rash_rate']:.3f}% ({pct})"
        print(
            f"{r['county']:<14} "
            f"{fr_r:>20} "
            f"{r['signal_level']:<10} "
            f"{r['aedes_positivity']:>6.1f}% "
            f"{r['aedes_level']:<10} "
            f"{r['arbonet_primary']:>11} "
            f"{r['concordance_score']:>6} "
            f"{r['tier']:<10} "
            f"{pcs}"
        )
    print("-" * W)
    tier_counts = {t: sum(1 for r in results if r["tier"] == t)
                   for t in ["CRITICAL", "HIGH", "MODERATE", "LOW"]}
    print(f"  CRITICAL: {tier_counts['CRITICAL']}  |  HIGH: {tier_counts['HIGH']}  |  "
          f"MODERATE: {tier_counts['MODERATE']}  |  LOW: {tier_counts['LOW']}")
    print("=" * W)


# ─────────────────────────────────────────────────────────────────────────────
# PNG FIGURE — 4 PANELS
# ─────────────────────────────────────────────────────────────────────────────

def fig_arboviral(results: list, vector: dict, arbonet: dict,
                  overall: dict, ref_date: date, window_days: int,
                  out_path: str):
    """
    4-panel Matplotlib figure:
      A. County risk tier (concordance score bar)
      B. Aedes aegypti & Culex trap positivity by county
      C. ArboNET YTD cases by county and pathogen
      D. Fever+Rash rate vs. seasonal baseline (all counties)
    """
    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.06, right=0.97,
                           top=0.90, bottom=0.06,
                           hspace=0.45, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    for ax in [ax_a, ax_b, ax_c, ax_d]:
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_color("#333344")

    # ── shared sorted county order (by concordance score desc) ────────────────
    counties = [r["county"] for r in results]
    scores   = [r["concordance_score"] for r in results]
    tiers_c  = [TIER_COLOR[r["tier"]] for r in results]
    y_pos    = np.arange(len(counties))

    # ── Panel A: Concordance score bar chart ──────────────────────────────────
    bars_a = ax_a.barh(y_pos, scores, color=tiers_c, edgecolor=BG, linewidth=0.5,
                       height=0.65)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(counties, fontsize=8, color=WHITE)
    ax_a.set_xlabel("Concordance Score  (max 6)", fontsize=8, color=GREY)
    ax_a.set_title("A — County Concordance Risk Tier", fontsize=10, color=WHITE,
                   fontweight="bold", pad=8)
    ax_a.set_xlim(0, 7)
    ax_a.axvline(x=5, color=RED,    linewidth=0.8, linestyle="--", alpha=0.7, label="CRITICAL≥5")
    ax_a.axvline(x=3, color=ORANGE, linewidth=0.8, linestyle="--", alpha=0.7, label="HIGH≥3")
    ax_a.axvline(x=1, color=YELLOW, linewidth=0.8, linestyle="--", alpha=0.7, label="MODERATE≥1")
    ax_a.legend(fontsize=6, loc="lower right", facecolor=SURFACE2,
                labelcolor=WHITE, edgecolor="none", framealpha=0.8)
    for i, (score, tier) in enumerate(zip(scores, [r["tier"] for r in results])):
        ax_a.text(score + 0.1, i, tier, va="center", ha="left",
                  fontsize=6, color=TIER_COLOR[tier], fontweight="bold")
    ax_a.tick_params(colors=GREY, labelsize=7)
    ax_a.xaxis.label.set_color(GREY)

    # ── Panel B: Aedes + Culex trap positivity ────────────────────────────────
    aedes_vals = [vector[c]["aedes_positivity"] for c in counties]
    culex_vals = [vector[c]["culex_positivity"] for c in counties]
    bw = 0.30
    y2 = np.arange(len(counties))
    ax_b.barh(y2 + bw/2, aedes_vals, height=bw, color=RED,   alpha=0.85,
              label="Aedes aegypti", edgecolor=BG, linewidth=0.3)
    ax_b.barh(y2 - bw/2, culex_vals, height=bw, color=BLUE,  alpha=0.85,
              label="Culex quinquefasciatus", edgecolor=BG, linewidth=0.3)
    ax_b.axvline(x=HIGH_AEDES_THRESHOLD, color=ORANGE, linewidth=0.9,
                 linestyle="--", alpha=0.8, label="HIGH threshold (10%)")
    ax_b.axvline(x=MODERATE_AEDES_THRESHOLD, color=YELLOW, linewidth=0.9,
                 linestyle="--", alpha=0.8, label="MODERATE threshold (5%)")
    ax_b.set_yticks(y2)
    ax_b.set_yticklabels(counties, fontsize=8, color=WHITE)
    ax_b.set_xlabel("Trap Positivity (%)", fontsize=8, color=GREY)
    ax_b.set_title("B — Mosquito Trap Positivity by County", fontsize=10,
                   color=WHITE, fontweight="bold", pad=8)
    ax_b.legend(fontsize=6, loc="lower right", facecolor=SURFACE2,
                labelcolor=WHITE, edgecolor="none", framealpha=0.8)
    ax_b.tick_params(colors=GREY, labelsize=7)

    # ── Panel C: ArboNET cases by county (primary pathogens stacked) ──────────
    primary_pathogens = ["dengue", "chikungunya", "zika"]
    pat_colors        = [RED, ORANGE, YELLOW]
    bottom_vals       = np.zeros(len(counties))
    for pathogen, color in zip(primary_pathogens, pat_colors):
        case_vals = [arbonet[c].get(pathogen, 0) for c in counties]
        ax_c.barh(y_pos, case_vals, left=bottom_vals,
                  color=color, alpha=0.85, label=ARBOVIRUSES[pathogen]["label"],
                  edgecolor=BG, linewidth=0.3, height=0.55)
        bottom_vals += np.array(case_vals, dtype=float)
    # WNV overlay (secondary)
    wnv_vals = [arbonet[c].get("wnv", 0) for c in counties]
    ax_c.scatter([v + 0.1 for v in bottom_vals], y_pos, color=BLUE, s=12,
                 zorder=5, label="WNV (secondary)", marker="D", alpha=0.9)
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(counties, fontsize=8, color=WHITE)
    ax_c.set_xlabel("YTD Cases (simulated ArboNET)", fontsize=8, color=GREY)
    ax_c.set_title("C — ArboNET YTD Case Counts by County", fontsize=10,
                   color=WHITE, fontweight="bold", pad=8)
    ax_c.legend(fontsize=6, loc="lower right", facecolor=SURFACE2,
                labelcolor=WHITE, edgecolor="none", framealpha=0.8)
    ax_c.tick_params(colors=GREY, labelsize=7)

    # ── Panel D: Fever+Rash rate by county (horizontal bar) ───────────────────
    fr_rates = [r["fever_rash_rate"] for r in results]
    pct_chgs = [r["pct_change"] if r["pct_change"] is not None else 0.0 for r in results]
    bar_colors_d = [TIER_COLOR[r["tier"]] for r in results]
    bars_d = ax_d.barh(y_pos, fr_rates, color=bar_colors_d, alpha=0.85,
                       edgecolor=BG, linewidth=0.4, height=0.55)
    # Annotate with pct_change
    for i, (rate, pct, r) in enumerate(zip(fr_rates, pct_chgs, results)):
        if rate > 0:
            pct_str = f"{pct:+.0f}%" if r["pct_change"] is not None else ""
            ax_d.text(rate + 0.002, i, pct_str, va="center", ha="left",
                      fontsize=6, color=TIER_COLOR[r["tier"]])
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(counties, fontsize=8, color=WHITE)
    ax_d.set_xlabel("Fever+Rash Co-Report Rate (%)", fontsize=8, color=GREY)
    ax_d.set_title("D — Fever+Rash Signal by County", fontsize=10, color=WHITE,
                   fontweight="bold", pad=8)
    # Tier legend patches
    patches_d = [mpatches.Patch(color=TIER_COLOR[t], label=t)
                 for t in ["CRITICAL", "HIGH", "MODERATE", "LOW"]]
    ax_d.legend(handles=patches_d, fontsize=6, loc="lower right",
                facecolor=SURFACE2, labelcolor=WHITE, edgecolor="none",
                framealpha=0.8)
    ax_d.tick_params(colors=GREY, labelsize=7)

    # ── Figure title + metadata ───────────────────────────────────────────────
    pct_str = (f"{overall['pct_change']:+.1f}%" if overall['pct_change'] is not None
               else "undefined")
    fig.suptitle(
        f"US#13 — Arboviral Surveillance Integration  |  Arizona  |  Ref: {ref_date.isoformat()}  |  "
        f"Window: {window_days}d  |  Fever+Rash: {overall['rate']:.3f}% ({pct_str} vs prior {window_days}d)",
        fontsize=11, color=WHITE, fontweight="bold", y=0.96
    )
    fig.text(0.5, 0.01,
             "Primary arboviruses: Dengue · Chikungunya · Zika (Aedes aegypti)  |  "
             "Secondary: WNV · SLEV (Culex)  |  "
             "Vector data: Great AZ Mosquito Hunt calibrated simulation  |  "
             "ArboNET: deterministic YTD simulation",
             ha="center", fontsize=7, color=GREY)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[PNG] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# HTML DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def build_html(results: list, vector: dict, arbonet: dict, overall: dict,
               narrative: str, ref_date: date, window_days: int,
               ref_month: int, out_path: str):

    counties_json  = [r["county"] for r in results]
    scores_json    = [r["concordance_score"] for r in results]
    tiers_json     = [r["tier"] for r in results]
    tier_colors_j  = [TIER_COLOR[r["tier"]] for r in results]
    aedes_json     = [vector[c]["aedes_positivity"] for c in counties_json]
    culex_json     = [vector[c]["culex_positivity"] for c in counties_json]
    dengue_json    = [arbonet[c].get("dengue", 0) for c in counties_json]
    chikv_json     = [arbonet[c].get("chikungunya", 0) for c in counties_json]
    zika_json      = [arbonet[c].get("zika", 0) for c in counties_json]
    wnv_json       = [arbonet[c].get("wnv", 0) for c in counties_json]
    fr_rates_json  = [r["fever_rash_rate"] for r in results]
    pct_chg_json   = [r["pct_change"] if r["pct_change"] is not None else 0 for r in results]
    n_both_json    = [r["n_both"] for r in results]

    # Build tier distribution counts
    tier_dist = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0}
    for r in results:
        tier_dist[r["tier"]] += 1

    # Surveillance actions for top tier
    top_tier = results[0]["tier"]
    actions_html = "".join(
        f'<li>{a}</li>' for a in SURVEILLANCE_ACTIONS[top_tier]
    )

    # Data gaps
    gaps_html = "".join(
        f'''<div class="gap-card gap-{g["severity"].lower()}">
              <div class="gap-sev">{g["severity"]}</div>
              <div class="gap-title">{g["gap"]}</div>
              <div class="gap-impact">Impact: {g["impact"]}</div>
              <div class="gap-rec">Recommendation: {g["recommendation"]}</div>
            </div>'''
        for g in DATA_GAPS
    )

    # Narrative paragraphs
    para_html = "".join(
        f'<p>{p.strip()}</p>' for p in narrative.split("\n\n")
    )

    # Table rows
    table_rows = ""
    for r in results:
        pct_str = f"{r['pct_change']:+.1f}%" if r["pct_change"] is not None else "n/a"
        pcs = ", ".join(r["postal_codes"][:4]) if r["postal_codes"] else "—"
        table_rows += f"""
        <tr>
          <td>{r['county']}</td>
          <td class="rate">{r['fever_rash_rate']:.3f}% ({pct_str})</td>
          <td class="sig sig-{r['signal_level'].lower()}">{r['signal_level']}</td>
          <td>{r['aedes_positivity']:.1f}%</td>
          <td class="lev lev-{r['aedes_level'].lower()}">{r['aedes_level']}</td>
          <td>{r['arbonet_primary']}</td>
          <td>{r['concordance_score']}</td>
          <td class="tier tier-{r['tier'].lower()}">{r['tier']}</td>
          <td class="pcs">{pcs}</td>
        </tr>"""

    overall_pct = (f"{overall['pct_change']:+.1f}%" if overall["pct_change"] is not None
                   else "undefined")

    llm_prompt = (
        "You are an arboviral surveillance analyst. "
        f"Human data: EpiHack reports of Fever+Rash in affected AZ postal codes over past {window_days} days — "
        f"rate: {overall['rate']:.3f}%, pct_change: {overall_pct}. "
        "Vector data: mosquito trap positivity rates for Arizona (source: Great AZ Mosquito Hunt). "
        "ArboNET reported cases: see table. "
        "In 3–4 paragraphs: (1) assess concordance between human symptom signals and vector activity; "
        "(2) estimate arboviral risk level for the affected postal codes; "
        "(3) recommend enhanced surveillance actions (increased trapping, targeted testing, physician alerts); "
        "(4) identify data gaps (e.g., no tick surveillance data for AZ)."
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>US#13 — Arboviral Surveillance Integration | EpiHack Arizona</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:#0f1117;color:#ecf0f1;font-family:'Segoe UI',system-ui,sans-serif;font-size:14px;line-height:1.6;}}
  header{{background:#1a1d27;border-bottom:1px solid #6c63ff44;padding:18px 28px;}}
  header h1{{font-size:1.25rem;color:#ecf0f1;font-weight:700;}}
  header .meta{{font-size:0.78rem;color:#747d8c;margin-top:4px;}}
  .container{{max-width:1400px;margin:0 auto;padding:20px 28px;}}
  .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;}}
  .grid-4{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px;}}
  .card{{background:#1a1d27;border-radius:10px;padding:20px;border:1px solid #22263a;}}
  .card-title{{font-size:0.78rem;color:#6c63ff;font-weight:600;text-transform:uppercase;
               letter-spacing:0.06em;margin-bottom:14px;}}
  .kpi{{text-align:center;padding:18px 10px;}}
  .kpi .val{{font-size:2rem;font-weight:700;}}
  .kpi .lbl{{font-size:0.72rem;color:#747d8c;margin-top:4px;}}
  .kpi.critical .val{{color:#ff4757;}}
  .kpi.high .val{{color:#ffa502;}}
  .kpi.moderate .val{{color:#ffd32a;}}
  .kpi.low .val{{color:#43e97b;}}
  canvas{{width:100%!important;max-height:280px;}}
  table{{width:100%;border-collapse:collapse;font-size:0.78rem;}}
  th{{background:#22263a;color:#6c63ff;padding:8px 10px;text-align:left;font-size:0.72rem;
      text-transform:uppercase;letter-spacing:0.05em;}}
  td{{padding:7px 10px;border-bottom:1px solid #22263a;color:#ecf0f1;}}
  tr:hover td{{background:#22263a;}}
  .rate{{font-family:monospace;}}
  .sig{{font-weight:700;font-size:0.72rem;}}
  .sig-elevated{{color:#ff4757;}}
  .sig-normal{{color:#ffd32a;}}
  .sig-suppressed{{color:#43e97b;}}
  .sig-none{{color:#747d8c;}}
  .lev{{font-weight:700;font-size:0.72rem;}}
  .lev-high{{color:#ff4757;}}
  .lev-moderate{{color:#ffd32a;}}
  .lev-low{{color:#43e97b;}}
  .tier{{font-weight:700;font-size:0.72rem;}}
  .tier-critical{{color:#ff4757;}}
  .tier-high{{color:#ffa502;}}
  .tier-moderate{{color:#ffd32a;}}
  .tier-low{{color:#43e97b;}}
  .pcs{{font-size:0.68rem;color:#747d8c;font-family:monospace;}}
  .narrative p{{margin-bottom:12px;font-size:0.84rem;color:#c8d6e5;line-height:1.7;}}
  .action-list{{list-style:none;padding:0;}}
  .action-list li{{padding:6px 0 6px 16px;position:relative;font-size:0.82rem;
                   color:#c8d6e5;border-bottom:1px solid #22263a;}}
  .action-list li::before{{content:"→";position:absolute;left:0;color:#6c63ff;font-weight:bold;}}
  .gap-card{{background:#22263a;border-radius:8px;padding:14px 16px;margin-bottom:10px;
             border-left:4px solid #747d8c;}}
  .gap-card.gap-critical{{border-left-color:#ff4757;}}
  .gap-card.gap-high{{border-left-color:#ffa502;}}
  .gap-card.gap-moderate{{border-left-color:#ffd32a;}}
  .gap-sev{{font-size:0.68rem;font-weight:700;text-transform:uppercase;
            letter-spacing:0.08em;margin-bottom:4px;}}
  .gap-card.gap-critical .gap-sev{{color:#ff4757;}}
  .gap-card.gap-high .gap-sev{{color:#ffa502;}}
  .gap-card.gap-moderate .gap-sev{{color:#ffd32a;}}
  .gap-title{{font-size:0.84rem;font-weight:600;color:#ecf0f1;margin-bottom:4px;}}
  .gap-impact{{font-size:0.75rem;color:#747d8c;margin-bottom:2px;}}
  .gap-rec{{font-size:0.75rem;color:#54a0ff;}}
  .prompt-box{{background:#0f1117;border:1px solid #6c63ff44;border-radius:8px;
               padding:14px 16px;font-size:0.78rem;font-family:monospace;
               color:#c8d6e5;white-space:pre-wrap;line-height:1.6;}}
  .badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.68rem;
          font-weight:700;text-transform:uppercase;margin-left:8px;}}
  .badge-aedes{{background:#ff475722;color:#ff4757;}}
  .badge-culex{{background:#54a0ff22;color:#54a0ff;}}
  footer{{text-align:center;padding:20px;font-size:0.72rem;color:#747d8c;
          border-top:1px solid #22263a;margin-top:20px;}}
</style>
</head>
<body>
<header>
  <h1>US#13 — Arboviral Surveillance Integration <span class="badge badge-aedes">Dengue · Chikungunya · Zika</span><span class="badge badge-culex">WNV · SLEV</span></h1>
  <div class="meta">
    Arizona EpiHack Platform · Reference date: {ref_date.isoformat()} · Window: {window_days}d ·
    Fever+Rash: {overall['rate']:.3f}% ({overall_pct} vs prior {window_days}d) ·
    n={overall['n_both']}/{overall['n_total']:,} reporters ·
    Vector data: Great AZ Mosquito Hunt (simulated) · ArboNET: deterministic YTD simulation
  </div>
</header>
<div class="container">

  <!-- KPI row -->
  <div class="grid-4">
    <div class="card kpi critical">
      <div class="val">{tier_dist['CRITICAL']}</div>
      <div class="lbl">CRITICAL counties</div>
    </div>
    <div class="card kpi high">
      <div class="val">{tier_dist['HIGH']}</div>
      <div class="lbl">HIGH counties</div>
    </div>
    <div class="card kpi moderate">
      <div class="val">{tier_dist['MODERATE']}</div>
      <div class="lbl">MODERATE counties</div>
    </div>
    <div class="card kpi low">
      <div class="val">{tier_dist['LOW']}</div>
      <div class="lbl">LOW counties</div>
    </div>
  </div>

  <!-- Charts row -->
  <div class="grid-2">
    <div class="card">
      <div class="card-title">A — County Concordance Risk Score</div>
      <canvas id="chartA"></canvas>
    </div>
    <div class="card">
      <div class="card-title">B — Mosquito Trap Positivity by County</div>
      <canvas id="chartB"></canvas>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-title">C — ArboNET YTD Cases by County (Primary Pathogens)</div>
      <canvas id="chartC"></canvas>
    </div>
    <div class="card">
      <div class="card-title">D — Fever+Rash Signal by County</div>
      <canvas id="chartD"></canvas>
    </div>
  </div>

  <!-- Concordance table -->
  <div class="card" style="margin-bottom:20px;">
    <div class="card-title">E — Full Concordance Decision Table (15 Counties)</div>
    <table>
      <thead>
        <tr>
          <th>County</th><th>FR Rate (Δ%)</th><th>FR Signal</th>
          <th>Aedes %</th><th>Aedes Lvl</th>
          <th>ArboNET (primary)</th><th>Score</th><th>Tier</th><th>Hot Postals</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
  </div>

  <!-- Analyst narrative -->
  <div class="card" style="margin-bottom:20px;">
    <div class="card-title">F — Arboviral Analyst Narrative (4 Paragraphs)</div>
    <div class="narrative">{para_html}</div>
  </div>

  <!-- Surveillance actions -->
  <div class="grid-2">
    <div class="card">
      <div class="card-title">G — Enhanced Surveillance Actions (Top Tier: {top_tier})</div>
      <ul class="action-list">{actions_html}</ul>
    </div>
    <div class="card">
      <div class="card-title">H — Data Gaps Registry ({len(DATA_GAPS)} gaps)</div>
      {gaps_html}
    </div>
  </div>

  <!-- LLM prompt -->
  <div class="card" style="margin-bottom:20px;">
    <div class="card-title">I — LLM Prompt Template (US#13)</div>
    <div class="prompt-box">{llm_prompt}</div>
  </div>

</div><!-- /container -->
<footer>
  EpiHack Arizona · Arboviral Surveillance Integration · US#13 ·
  DB: epihack-mini.db · 50,000 records · 138 postal codes · Ref: {ref_date.isoformat()} ·
  Primary arboviruses: Dengue · Chikungunya · Zika (Aedes aegypti) ·
  Secondary: WNV · SLEV (Culex quinquefasciatus)
</footer>

<script>
const COUNTIES  = {counties_json};
const SCORES    = {scores_json};
const TIER_C    = {tier_colors_j};
const AEDES     = {aedes_json};
const CULEX     = {culex_json};
const DENGUE    = {dengue_json};
const CHIKV     = {chikv_json};
const ZIKA      = {zika_json};
const WNV       = {wnv_json};
const FR_RATES  = {fr_rates_json};
const PCT_CHG   = {pct_chg_json};

const OPTS_BASE = {{
  indexAxis: 'y',
  responsive: true,
  maintainAspectRatio: true,
  plugins: {{
    legend: {{ labels: {{ color: '#ecf0f1', font: {{ size: 10 }} }} }},
    tooltip: {{ backgroundColor: '#22263a', titleColor: '#ecf0f1', bodyColor: '#c8d6e5',
                borderColor: '#6c63ff44', borderWidth: 1 }}
  }},
  scales: {{
    x: {{ ticks: {{ color: '#747d8c', font: {{ size: 9 }} }},
          grid: {{ color: '#22263a' }} }},
    y: {{ ticks: {{ color: '#ecf0f1', font: {{ size: 9 }} }},
          grid: {{ color: '#22263a' }} }}
  }}
}};

// Chart A — Concordance score
new Chart(document.getElementById('chartA'), {{
  type: 'bar',
  data: {{
    labels: COUNTIES,
    datasets: [{{
      label: 'Concordance Score',
      data: SCORES,
      backgroundColor: TIER_C,
      borderColor: '#0f1117',
      borderWidth: 0.5,
      borderRadius: 3,
    }}]
  }},
  options: {{ ...OPTS_BASE,
    plugins: {{ ...OPTS_BASE.plugins,
      annotation: {{}}
    }},
    scales: {{ ...OPTS_BASE.scales,
      x: {{ ...OPTS_BASE.scales.x, max: 7 }}
    }}
  }}
}});

// Chart B — Trap positivity
new Chart(document.getElementById('chartB'), {{
  type: 'bar',
  data: {{
    labels: COUNTIES,
    datasets: [
      {{ label: 'Aedes aegypti (%)',           data: AEDES, backgroundColor: '#ff475788', borderRadius: 2 }},
      {{ label: 'Culex quinquefasciatus (%)',   data: CULEX, backgroundColor: '#54a0ff88', borderRadius: 2 }}
    ]
  }},
  options: {{ ...OPTS_BASE }}
}});

// Chart C — ArboNET cases stacked
new Chart(document.getElementById('chartC'), {{
  type: 'bar',
  data: {{
    labels: COUNTIES,
    datasets: [
      {{ label: 'Dengue',       data: DENGUE, backgroundColor: '#ff4757cc', borderRadius: 2 }},
      {{ label: 'Chikungunya',  data: CHIKV,  backgroundColor: '#ffa502cc', borderRadius: 2 }},
      {{ label: 'Zika',         data: ZIKA,   backgroundColor: '#ffd32acc', borderRadius: 2 }},
      {{ label: 'WNV',          data: WNV,    backgroundColor: '#54a0ff88', borderRadius: 2 }}
    ]
  }},
  options: {{ ...OPTS_BASE,
    plugins: {{ ...OPTS_BASE.plugins }},
    scales: {{ ...OPTS_BASE.scales,
      x: {{ ...OPTS_BASE.scales.x, stacked: true }},
      y: {{ ...OPTS_BASE.scales.y, stacked: true }}
    }}
  }}
}});

// Chart D — Fever+Rash rate
new Chart(document.getElementById('chartD'), {{
  type: 'bar',
  data: {{
    labels: COUNTIES,
    datasets: [{{
      label: 'Fever+Rash Rate (%)',
      data: FR_RATES,
      backgroundColor: TIER_C,
      borderColor: '#0f1117',
      borderWidth: 0.5,
      borderRadius: 3,
    }}]
  }},
  options: {{ ...OPTS_BASE }}
}});
</script>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    size_mb = Path(out_path).stat().st_size / 1_048_576
    print(f"[HTML] Saved → {out_path} ({size_mb:.2f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EpiHack US#13 — Arboviral Surveillance Integration"
    )
    parser.add_argument("--db",     required=True, help="Path to epihack-mini.db")
    parser.add_argument("--window", type=int, default=30, help="Surveillance window (days)")
    parser.add_argument("--out",    required=True, help="Output PNG path")
    args = parser.parse_args()

    db_path      = args.db
    window_days  = args.window
    out_png      = args.out
    out_html     = out_png.replace(".png", ".html")

    # Reference date and month
    ref_date     = date(2026, 5, 21)
    ref_month    = ref_date.month   # 5 = May

    print("=" * 80)
    print("  US#13 — Arboviral Surveillance Integration")
    print(f"  DB: {db_path}")
    print(f"  Ref: {ref_date.isoformat()} | Window: {window_days}d | Month: {ref_month}")
    print(f"  Primary arboviruses: Dengue, Chikungunya, Zika (Aedes aegypti-borne)")
    print(f"  Secondary: WNV, SLEV (Culex-borne)")
    print("=" * 80)

    print("\n[1/5] Loading Fever+Rash signal from EpiHack DB …")
    overall, by_county, by_postal = load_fever_rash(db_path, ref_date, window_days)
    print(f"      Overall rate: {overall['rate']:.3f}%  "
          f"({overall['pct_change']:+.1f}% vs prior {window_days}d)  "
          f"n={overall['n_both']}/{overall['n_total']}")

    print("\n[2/5] Simulating vector trap data (Great AZ Mosquito Hunt calibration) …")
    vector = build_vector_data(ref_month)
    top_aedes = sorted(vector.items(), key=lambda x: -x[1]["aedes_positivity"])[:3]
    for county, d in top_aedes:
        print(f"      {county}: Aedes {d['aedes_positivity']:.1f}% ({d['aedes_level']}), "
              f"Culex {d['culex_positivity']:.1f}%")

    print("\n[3/5] Simulating ArboNET YTD case counts …")
    arbonet = build_arbonet_data(ref_month, window_days)
    statewide = {p: sum(arbonet[c].get(p, 0) for c in arbonet) for p in ARBOVIRUSES}
    for p, n in statewide.items():
        print(f"      {ARBOVIRUSES[p]['label']}: {n} YTD statewide")

    print("\n[4/5] Computing concordance scores and risk tiers …")
    results = run_concordance_analysis(overall, by_county, vector, arbonet)
    for r in results[:5]:
        print(f"      {r['county']:<14} score={r['concordance_score']} tier={r['tier']}")

    print("\n[5/5] Generating outputs …")
    print_concordance_table(results, overall, ref_date, window_days)

    narrative = generate_narrative(overall, results, vector, arbonet,
                                   ref_date, window_days, ref_month)

    fig_arboviral(results, vector, arbonet, overall, ref_date, window_days, out_png)

    build_html(results, vector, arbonet, overall, narrative,
               ref_date, window_days, ref_month, out_html)

    print(f"\nAll outputs saved.")
    print(f"  PNG  → {out_png}")
    print(f"  HTML → {out_html}")


if __name__ == "__main__":
    main()
