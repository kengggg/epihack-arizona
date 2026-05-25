#!/usr/bin/env python3
"""
rare_event_cooccurrence_report.py — User Story #11
EpiHack Participatory Epidemiological Surveillance Platform

Rare event detection via symptom co-occurrence analysis.

Methodology:
  • Co-occurrence metric : joint prevalence rate — % of reporters in window
                           answering YES to BOTH symptoms A and B simultaneously
  • Baseline distribution: 90-day period preceding the recent window, split into
                           12 non-overlapping 7-day blocks; co-occurrence rate
                           computed per block → mean ± SD per symptom pair
  • Flagging criteria     :
      (A) Current 7-day rate > (baseline_mean + 2 × baseline_SD)  [z-score > 2]
      (B) Combination includes any high-concern symptom:
          bleeding_from_body_openings | yellow_skin_yellow_eyes |
          discolored_or_bloody_urine
      (C) Co-reported cases cluster in ≤ 2 postal codes

  Any pair meeting ≥ 1 criterion is flagged.
  Pairs meeting 2–3 criteria are escalated.

  For each flagged combination:
    • Top-5 differential diagnosis (Arizona-endemic-prioritised)
    • Confirmatory laboratory and epidemiological investigations recommended

Outputs:
  1. Terminal flagged-pair report
  2. rare_event_cooccurrence_report.png  — four-panel Matplotlib figure
  3. rare_event_cooccurrence_report.html — interactive self-contained dashboard

Usage:
  python rare_event_cooccurrence_report.py \
    --db epihack-mini.db \
    --window 7 \
    --baseline-days 90 \
    --out rare_event_cooccurrence_report.png

Author: EpiHack Platform  |  Arizona Department of Health Services
"""

import argparse
import itertools
import os
import sqlite3
import warnings
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

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
PURPLE  = "#a29bfe"
TEXT    = "#e8eaf6"
MUTED   = "#8892a4"

SYMPTOM_COLS = [
    ("cough_congestion",               "Cough/Congestion"),
    ("sore_throat",                    "Sore Throat"),
    ("difficulty_breathing",           "Diff. Breathing"),
    ("nauseas_vomiting",               "Nausea/Vomiting"),
    ("diarrhea",                       "Diarrhea"),
    ("muscle_or_body_aches_and_pains", "Muscle Aches"),
    ("fever",                          "Fever"),
    ("chills",                         "Chills"),
    ("red_eyes",                       "Red Eyes"),
    ("rash",                           "Rash"),
    ("loss_of_smell_or_taste",         "Loss Smell/Taste"),
    ("bleeding_from_body_openings",    "Bleeding (Openings)"),
    ("yellow_skin_yellow_eyes",        "Yellow Skin/Eyes"),
    ("discolored_or_bloody_urine",     "Discolored/Bloody Urine"),
]
SYM_KEYS   = [c[0] for c in SYMPTOM_COLS]
SYM_SHORT  = {c[0]: c[1] for c in SYMPTOM_COLS}
N_SYM      = len(SYM_KEYS)

HIGH_CONCERN = {
    "bleeding_from_body_openings",
    "yellow_skin_yellow_eyes",
    "discolored_or_bloody_urine",
}

# ─────────────────────────────────────────────────────────────────────────────
# DIFFERENTIAL DIAGNOSIS KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: list of symptom keys (any subset match triggers inclusion)
# Ranked by concern level; Arizona-endemic pathogens prioritised

DDX_KB = [
    # ── Hemorrhagic / Multi-organ ──────────────────────────────────────────
    {
        "syndrome": "Rocky Mountain Spotted Fever (RMSF)",
        "pathogen": "Rickettsia rickettsii (tick-borne)",
        "az_endemic": True,
        "concern": "CRITICAL",
        "trigger_any": {"rash", "fever", "bleeding_from_body_openings", "red_eyes"},
        "trigger_require_one": {"rash", "fever"},
        "confirmatory_lab": [
            "Paired serology (IgG/IgM): 4-fold rise — Indirect Fluorescent Antibody (IFA)",
            "Whole-blood PCR (Rickettsia spp.) — highest yield days 3–5 of illness",
            "Immunohistochemistry on skin biopsy (rash site)",
            "CBC: thrombocytopenia, hyponatremia",
            "LFTs: transaminase elevation",
        ],
        "confirmatory_epi": [
            "Tick exposure history within 14 days (American dog tick, Rocky Mountain wood tick)",
            "Case mapping: compare to Amblyomma americanum distribution in AZ",
            "Veterinary partner reports of tick-borne illness in livestock/companion animals",
            "Notify ADHS immediately — RMSF is a nationally notifiable condition",
        ],
        "notes": "Arizona has among the highest RMSF incidence rates in the U.S. "
                 "Treatment with doxycycline is life-saving — do not wait for confirmation.",
    },
    {
        "syndrome": "Hantavirus Pulmonary Syndrome (HPS)",
        "pathogen": "Sin Nombre hantavirus (rodent-borne aerosol)",
        "az_endemic": True,
        "concern": "CRITICAL",
        "trigger_any": {"difficulty_breathing", "fever", "muscle_or_body_aches_and_pains",
                        "discolored_or_bloody_urine"},
        "trigger_require_one": {"difficulty_breathing", "fever"},
        "confirmatory_lab": [
            "Hantavirus IgM/IgG serology (CDC Hantavirus panel) — ELISA",
            "RT-PCR on peripheral blood mononuclear cells (early illness only)",
            "CBC: thrombocytopenia, left-shifted neutrophilia, hemoconcentration",
            "Chest X-ray / CT: bilateral interstitial infiltrates (early pulmonary edema)",
            "BUN/Creatinine: renal involvement indicates hemorrhagic fever with renal syndrome",
        ],
        "confirmatory_epi": [
            "Rodent exposure history: cleaning outbuildings, rural/agricultural settings",
            "Peromyscus maniculatus (deer mouse) seroprevalence surveys in local area",
            "Home/workplace peridomestic rodent infestation assessment",
            "Navajo Nation / Four Corners surveillance network notification",
        ],
        "notes": "Case fatality rate ~36%. Arizona Four Corners region is highest-risk area. "
                 "Rare but rapidly fatal — escalate immediately.",
    },
    {
        "syndrome": "Leptospirosis (Weil's Disease — severe form)",
        "pathogen": "Leptospira interrogans (water/soil contact, zoonotic)",
        "az_endemic": True,
        "concern": "HIGH",
        "trigger_any": {"fever", "yellow_skin_yellow_eyes", "discolored_or_bloody_urine",
                        "red_eyes", "muscle_or_body_aches_and_pains"},
        "trigger_require_one": {"yellow_skin_yellow_eyes", "discolored_or_bloody_urine"},
        "confirmatory_lab": [
            "Microscopic Agglutination Test (MAT) — gold standard serology, paired sera",
            "Blood culture (first week of illness) — Leptospira spp., BSL-2 lab required",
            "PCR on blood/urine during acute phase",
            "Urinalysis: pyuria, hematuria, casts",
            "LFTs/Bilirubin: jaundice confirms Weil's disease",
            "Renal function: BUN, creatinine (acute kidney injury)",
        ],
        "confirmatory_epi": [
            "Flood/irrigation water exposure in 2–30 days prior to onset",
            "Livestock, dog, or rodent contact (cattle, swine, rats primary reservoirs)",
            "Occupational risk: farm workers, irrigation workers, veterinarians",
            "Agriculture worker housing / canal irrigation exposure mapping",
        ],
        "notes": "Arizona agricultural communities (Yuma, Pinal counties) at elevated risk. "
                 "Report to ADHS — reportable condition.",
    },
    {
        "syndrome": "Acute Viral Hepatitis (A, B, or E)",
        "pathogen": "HAV / HBV / HEV",
        "az_endemic": False,
        "concern": "HIGH",
        "trigger_any": {"yellow_skin_yellow_eyes", "nauseas_vomiting", "diarrhea",
                        "fever", "discolored_or_bloody_urine"},
        "trigger_require_one": {"yellow_skin_yellow_eyes"},
        "confirmatory_lab": [
            "Hepatitis A IgM antibody (anti-HAV IgM) — confirms acute HAV",
            "Hepatitis B surface antigen (HBsAg), anti-HBc IgM — confirms acute HBV",
            "Hepatitis E IgM (anti-HEV IgM) — consider if travel or pork exposure",
            "LFTs: AST/ALT (markedly elevated, >10× ULN in acute hepatitis)",
            "Total and direct bilirubin, PT/INR (severity assessment)",
            "Urinalysis: bilirubin in urine (dark urine = bilirubinuria)",
        ],
        "confirmatory_epi": [
            "Shellfish or contaminated water consumption (HAV)",
            "Arizona HAV outbreak tracking — contact MARICOPA/PIMA county health",
            "Shared injection equipment or high-risk sexual contact (HBV)",
            "Recent international travel to endemic regions (HEV)",
            "Foodborne outbreak investigation if ≥2 cases share a common exposure",
        ],
        "notes": "HAV outbreaks have occurred in Arizona among unhoused populations. "
                 "All hepatitis cases are nationally notifiable.",
    },
    {
        "syndrome": "Coccidioidomycosis (Valley Fever)",
        "pathogen": "Coccidioides immitis/posadasii (soil-borne fungal)",
        "az_endemic": True,
        "concern": "MODERATE",
        "trigger_any": {"fever", "cough_congestion", "rash", "muscle_or_body_aches_and_pains",
                        "difficulty_breathing"},
        "trigger_require_one": {"fever", "cough_congestion"},
        "confirmatory_lab": [
            "Coccidioides serology: IgM (EIA or IDTP) — acute infection",
            "Coccidioides IgG (CF titer) — severity and dissemination marker",
            "Chest X-ray: nodules, infiltrates, hilar adenopathy",
            "Serum complement fixation titer ≥1:16 → disseminated disease",
            "BAL or sputum culture (BSL-3 lab required — notify lab before sending)",
            "Urine antigen test (Mira Vista Diagnostics)",
        ],
        "confirmatory_epi": [
            "Soil disturbance activities: construction, farming, archaeology, military exercises",
            "Residence or travel in Sonoran Desert (Maricopa, Pima, Pinal, Yavapai counties)",
            "Immunocompromised status assessment (HIV, transplant, TNF-alpha inhibitors)",
            "Cluster mapping: compare postal codes to Coccidioides endemic soil zones",
            "Report disseminated or meningeal cases to ADHS (reportable condition in AZ)",
        ],
        "notes": "Arizona accounts for ~66% of U.S. Valley Fever cases. Most are self-limiting; "
                 "dissemination risk is elevated in pregnancy, immunocompromised, and Black/Filipino populations.",
    },
    {
        "syndrome": "West Nile Virus (Neuroinvasive or Fever)",
        "pathogen": "West Nile virus (Culex mosquito-borne)",
        "az_endemic": True,
        "concern": "HIGH",
        "trigger_any": {"fever", "rash", "muscle_or_body_aches_and_pains",
                        "red_eyes", "chills"},
        "trigger_require_one": {"fever"},
        "confirmatory_lab": [
            "WNV IgM antibody in serum or CSF — plaque-reduction neutralization (PRNT)",
            "WNV PCR (blood or CSF) — highest yield during viremia (week 1)",
            "CBC: leukopenia, lymphopenia typical in WNV fever",
            "CSF analysis if neurological symptoms (pleocytosis, elevated protein)",
            "Report to ArboNET (CDC arboviral surveillance system)",
        ],
        "confirmatory_epi": [
            "Mosquito exposure history (unprotected outdoor activity, no repellent)",
            "Culex quinquefasciatus trap counts — local mosquito abatement district data",
            "Dead bird surveillance (corvid mortality is sentinel event for WNV activity)",
            "Blood donor WNV screening positivity rates in affected counties",
            "Link to Arizona Department of Agriculture Vector-Borne Disease map",
        ],
        "notes": "Peak season: July–September in Arizona. Most infections subclinical; "
                 "1% develop neuroinvasive disease. Report all confirmed cases to ADHS.",
    },
    {
        "syndrome": "Hemolytic Uremic Syndrome (HUS) / Shiga-toxin E. coli (STEC)",
        "pathogen": "E. coli O157:H7 or other STEC",
        "az_endemic": False,
        "concern": "HIGH",
        "trigger_any": {"diarrhea", "discolored_or_bloody_urine", "nauseas_vomiting",
                        "fever", "yellow_skin_yellow_eyes"},
        "trigger_require_one": {"diarrhea", "discolored_or_bloody_urine"},
        "confirmatory_lab": [
            "Stool culture for STEC O157:H7 and non-O157 STEC (within 7 days of onset)",
            "Shiga toxin EIA on stool (Stx1, Stx2)",
            "CBC: microangiopathic hemolytic anemia (schistocytes on smear)",
            "Renal function: BUN, creatinine (AKI in HUS)",
            "Platelet count: thrombocytopenia in HUS triad",
            "Urinalysis: hematuria, proteinuria, casts",
        ],
        "confirmatory_epi": [
            "Food exposure history: ground beef, leafy greens, unpasteurized products",
            "Waterborne exposure: recreational water, irrigation water",
            "Case clustering: school lunch, restaurant, farm produce — trace-back investigation",
            "Contact with farm animals or agricultural fairs",
            "ADHS Enteric Disease unit notification (STEC is reportable in AZ)",
        ],
        "notes": "Antibiotics contraindicated in STEC — worsen HUS risk. "
                 "Cluster of ≥2 cases requires immediate foodborne outbreak investigation.",
    },
    {
        "syndrome": "Dengue Fever / Hemorrhagic Dengue",
        "pathogen": "Dengue virus 1–4 (Aedes mosquito-borne, travel-associated)",
        "az_endemic": False,
        "concern": "HIGH",
        "trigger_any": {"fever", "rash", "muscle_or_body_aches_and_pains",
                        "red_eyes", "bleeding_from_body_openings", "chills"},
        "trigger_require_one": {"fever", "rash"},
        "confirmatory_lab": [
            "Dengue NS1 antigen EIA (days 1–5 of fever)",
            "Dengue IgM/IgG EIA (day 5+ — seroconversion)",
            "RT-PCR: dengue DENV 1–4 typing (best in first 5 days)",
            "CBC: leukopenia, thrombocytopenia (platelet <100K/μL → DHF warning)",
            "Hematocrit: hemoconcentration ≥20% above baseline",
            "Liver enzymes (ALT/AST elevation common)",
        ],
        "confirmatory_epi": [
            "Travel history to Mexico, Central/South America, Caribbean, Southeast Asia within 14 days",
            "Arizona border-region exposure (dengue has been detected in northern Sonora)",
            "Aedes aegypti surveillance near case residential address",
            "Contact tracing for imported cases to prevent local transmission",
            "Report to ADHS — dengue is nationally notifiable",
        ],
        "notes": "Local Aedes aegypti are present in Maricopa/Pima counties — imported cases "
                 "carry vectorborne transmission risk. Travel history is essential.",
    },
    {
        "syndrome": "Meningococcemia / Bacterial Meningitis",
        "pathogen": "Neisseria meningitidis",
        "az_endemic": False,
        "concern": "CRITICAL",
        "trigger_any": {"fever", "rash", "chills", "bleeding_from_body_openings",
                        "muscle_or_body_aches_and_pains"},
        "trigger_require_one": {"fever", "rash"},
        "confirmatory_lab": [
            "Blood culture × 2 (before antibiotics) — Neisseria meningitidis",
            "CSF analysis: opening pressure, Gram stain, culture, PCR",
            "CBC: leukocytosis, bandemia; PT/INR (DIC in severe cases)",
            "Petechial skin biopsy — Gram stain, culture, PCR",
            "Meningococcal serogroup PCR (to guide outbreak response)",
        ],
        "confirmatory_epi": [
            "Close contacts within 7 days (household, dormitory, kissing contacts)",
            "Vaccination status: meningococcal ACWY and MenB vaccines",
            "College dormitory, military barracks, homeless shelter exposure",
            "Notify ADHS within 4 hours — immediate chemoprophylaxis for contacts",
            "Serogroup determination to assess outbreak potential",
        ],
        "notes": "MEDICAL EMERGENCY — case fatality 10–15%. Immediate notification of ADHS "
                 "required. Rifampin/ciprofloxacin prophylaxis for close contacts within 24h.",
    },
    {
        "syndrome": "Rhabdomyolysis (Exertional or Toxic)",
        "pathogen": "Non-infectious: exertional, drug/toxin, heat stroke",
        "az_endemic": True,  # Arizona summer heat context
        "concern": "MODERATE",
        "trigger_any": {"muscle_or_body_aches_and_pains", "discolored_or_bloody_urine",
                        "fever", "chills"},
        "trigger_require_one": {"muscle_or_body_aches_and_pains", "discolored_or_bloody_urine"},
        "confirmatory_lab": [
            "Serum CK (creatine kinase): >5× ULN (~1,000 U/L) confirms rhabdomyolysis",
            "Urinalysis: myoglobinuria (brown/red urine, dipstick positive for 'blood', "
            "microscopy negative for RBCs)",
            "BUN/Creatinine: acute kidney injury risk when CK >15,000 U/L",
            "Electrolytes: hyperkalemia, hyperphosphatemia, hypocalcemia",
            "LDH, ALT/AST: elevated due to muscle necrosis",
        ],
        "confirmatory_epi": [
            "Strenuous physical exertion in high-heat environment (NWS HeatRisk ≥3)",
            "Medication/drug history: statins, fibrates, antipsychotics, alcohol, illicit drugs",
            "Occupational exposure: construction, agricultural, military physical training in AZ summer",
            "Cross-reference NWS HeatRisk data for case postal codes (see US#4 heat risk model)",
            "No person-to-person transmission — no contact-tracing required",
        ],
        "notes": "Arizona summer heat context: exertional rhabdomyolysis peaks July–September. "
                 "Co-occurrence with HeatRisk index ≥3 is a strong contextual signal.",
    },
    {
        "syndrome": "Salmonella Typhi (Typhoid Fever)",
        "pathogen": "Salmonella Typhi (foodborne / waterborne)",
        "az_endemic": False,
        "concern": "HIGH",
        "trigger_any": {"fever", "diarrhea", "rash", "yellow_skin_yellow_eyes",
                        "chills", "nauseas_vomiting"},
        "trigger_require_one": {"fever"},
        "confirmatory_lab": [
            "Blood culture (weeks 1–2): S. Typhi — 60–80% sensitivity",
            "Stool culture (weeks 2–4): highest yield later in illness",
            "Widal test (agglutination): limited specificity — use with caution",
            "CBC: leukopenia with relative lymphocytosis (rose-spot period)",
            "LFTs: mild transaminase elevation common",
            "Bone marrow culture: 90% sensitivity, reserved for difficult cases",
        ],
        "confirmatory_epi": [
            "International travel history to South Asia, Southeast Asia, Sub-Saharan Africa",
            "Consumption of food prepared by a known carrier",
            "Food handler exposure investigation if no travel history",
            "Contact with recently returned international traveler",
            "Report to ADHS and CDC — typhoid is nationally notifiable",
        ],
        "notes": "Most Arizona cases are travel-associated. Local transmission suggests "
                 "food handler/water supply investigation is needed urgently.",
    },
]


def get_ddx(sym_set: frozenset, max_results: int = 5) -> list[dict]:
    """
    Retrieve ranked differential diagnoses for a given symptom combination.

    Ranking: concern level (CRITICAL > HIGH > MODERATE) + overlap score
             (n symptoms matching trigger_any).

    Args:
        sym_set      : frozenset of symptom column keys present in the combination
        max_results  : number of DDx entries to return

    Returns:
        list of DDx dicts, sorted by priority
    """
    concern_rank = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}
    scored = []
    for entry in DDX_KB:
        trigger_any = entry.get("trigger_any", set())
        trigger_req = entry.get("trigger_require_one", set())
        overlap     = len(sym_set & trigger_any)
        req_met     = len(sym_set & trigger_req) > 0
        if overlap > 0 and req_met:
            score = (concern_rank.get(entry["concern"], 3) * 10) - overlap
            scored.append((score, entry))
    scored.sort(key=lambda x: x[0])
    return [e for _, e in scored[:max_results]]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    sym_sql = ", ".join(SYM_KEYS)
    df = pd.read_sql_query(
        f"""
        SELECT date_of_report, postal_code, age, sex, {sym_sql}
        FROM incidents
        WHERE date_of_report IS NOT NULL AND postal_code IS NOT NULL
        """,
        conn,
        parse_dates=["date_of_report"],
    )
    conn.close()

    for col in SYM_KEYS:
        df[col] = (
            df[col].astype(str).str.upper()
            .map({"YES": 1.0, "NO": 0.0, "UNKNOWN": float("nan")})
            .astype("Float64")
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CO-OCCURRENCE MATRIX COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def joint_prevalence(df: pd.DataFrame, sym_a: str, sym_b: str) -> tuple[float, int]:
    """
    Compute the joint prevalence rate (%) of symptoms A and B in a DataFrame.
    Reporters who answered YES to BOTH symptoms simultaneously.

    Returns: (rate_pct, n_reporters_with_valid_answers_to_both)
    """
    both = df[[sym_a, sym_b]].dropna()
    n    = len(both)
    if n == 0:
        return 0.0, 0
    rate = float(((both[sym_a] == 1.0) & (both[sym_b] == 1.0)).mean()) * 100
    return rate, n


def build_cooccurrence_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build N_SYM × N_SYM joint prevalence matrix (lower triangle filled)."""
    mat = np.zeros((N_SYM, N_SYM))
    for i in range(N_SYM):
        for j in range(i + 1, N_SYM):
            rate, _ = joint_prevalence(df, SYM_KEYS[i], SYM_KEYS[j])
            mat[i, j] = rate
            mat[j, i] = rate
    return mat


def build_baseline_distribution(df: pd.DataFrame,
                                 ref_date: pd.Timestamp,
                                 window_days: int,
                                 baseline_days: int,
                                 n_blocks: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """
    Divide the 90-day baseline into n_blocks non-overlapping 7-day blocks.
    Compute co-occurrence matrix for each block.

    Returns:
        baseline_mean : N_SYM × N_SYM mean co-occurrence rate
        baseline_sd   : N_SYM × N_SYM std dev of co-occurrence rate across blocks
    """
    # Baseline spans [ref_date - window_days - baseline_days,
    #                 ref_date - window_days)
    bas_end   = ref_date - timedelta(days=window_days)
    bas_start = bas_end  - timedelta(days=baseline_days)

    block_mats = []
    for b in range(n_blocks):
        block_end   = bas_end   - timedelta(days=b * window_days)
        block_start = block_end - timedelta(days=window_days)
        block_df = df[(df["date_of_report"] > block_start) &
                      (df["date_of_report"] <= block_end)]
        if len(block_df) >= 5:   # at least 5 reporters to count
            block_mats.append(build_cooccurrence_matrix(block_df))

    if not block_mats:
        return np.zeros((N_SYM, N_SYM)), np.ones((N_SYM, N_SYM)) * 0.01

    stack         = np.stack(block_mats, axis=0)
    baseline_mean = stack.mean(axis=0)
    baseline_sd   = stack.std(axis=0, ddof=1)
    # Floor SD to prevent division-by-zero for zero-variance pairs
    baseline_sd   = np.where(baseline_sd < 0.001, 0.001, baseline_sd)
    return baseline_mean, baseline_sd


# ─────────────────────────────────────────────────────────────────────────────
# FLAGGING LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def count_cluster_postals(df: pd.DataFrame, sym_a: str, sym_b: str) -> int:
    """Count distinct postal codes where BOTH symptoms co-occur."""
    both = df[(df[sym_a].fillna(0) == 1.0) & (df[sym_b].fillna(0) == 1.0)]
    return both["postal_code"].nunique()


def flag_pairs(recent_df: pd.DataFrame,
               recent_mat: np.ndarray,
               baseline_mean: np.ndarray,
               baseline_sd: np.ndarray,
               z_thresh: float = 2.0,
               cluster_thresh: int = 3) -> pd.DataFrame:
    """
    Evaluate all 91 unique symptom pairs and return flagged rows.

    Flag criteria:
      A — z-score > z_thresh (>2 SD above baseline mean)
      B — pair includes ≥ 1 high-concern symptom
      C — co-reported cases cluster in < cluster_thresh postal codes
    """
    rows = []
    for i in range(N_SYM):
        for j in range(i + 1, N_SYM):
            sym_a    = SYM_KEYS[i]
            sym_b    = SYM_KEYS[j]
            rec_rate = recent_mat[i, j]
            b_mean   = baseline_mean[i, j]
            b_sd     = baseline_sd[i, j]
            z_score  = (rec_rate - b_mean) / b_sd

            n_postals      = count_cluster_postals(recent_df, sym_a, sym_b)
            has_hc         = bool({sym_a, sym_b} & HIGH_CONCERN)
            flag_z         = z_score > z_thresh
            flag_hc        = has_hc and rec_rate > 0
            flag_cluster   = (n_postals > 0 and n_postals < cluster_thresh)
            flagged        = flag_z or flag_hc or flag_cluster
            n_flags        = int(flag_z) + int(flag_hc) + int(flag_cluster)

            # Joint reporter count
            both_df  = recent_df[[sym_a, sym_b, "postal_code"]].dropna()
            n_joint  = int(((both_df[sym_a] == 1.0) & (both_df[sym_b] == 1.0)).sum())

            if flagged or has_hc:   # always include high-concern, even if rate=0
                sym_set = frozenset({sym_a, sym_b})
                ddx     = get_ddx(sym_set)
                rows.append({
                    "sym_a":         sym_a,
                    "sym_b":         sym_b,
                    "label_a":       SYM_SHORT[sym_a],
                    "label_b":       SYM_SHORT[sym_b],
                    "rec_rate":      round(rec_rate, 4),
                    "baseline_mean": round(b_mean, 4),
                    "baseline_sd":   round(b_sd, 4),
                    "z_score":       round(z_score, 2),
                    "n_joint":       n_joint,
                    "n_postals":     n_postals,
                    "has_hc":        has_hc,
                    "flag_z":        flag_z,
                    "flag_hc":       flag_hc,
                    "flag_cluster":  flag_cluster,
                    "n_flags":       n_flags,
                    "flagged":       flagged,
                    "ddx":           ddx,
                })

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out
    return df_out.sort_values(["n_flags", "z_score"], ascending=[False, False]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# LLM PROMPT + NARRATIVE
# ─────────────────────────────────────────────────────────────────────────────

def build_llm_prompt(flagged_df: pd.DataFrame,
                     recent_mat: np.ndarray,
                     window_days: int,
                     ref_date: pd.Timestamp) -> str:
    """Fill the US#11 LLM prompt template."""
    # Build co-occurrence_matrix string (top pairs only)
    top_pairs = (
        flagged_df[flagged_df["flagged"] == True]
        .head(10)[["label_a", "label_b", "rec_rate", "z_score", "n_postals"]]
    )
    if top_pairs.empty:
        matrix_str = "No pairs exceeded flagging thresholds."
    else:
        lines = [f"{'Symptom A':<28} {'Symptom B':<28} "
                 f"{'Joint Rate%':>12} {'Z-score':>8} {'N Postals':>10}"]
        lines.append("-" * 90)
        for _, r in top_pairs.iterrows():
            lines.append(
                f"{r['label_a']:<28} {r['label_b']:<28} "
                f"{r['rec_rate']:.3f}%{' ':>5} {r['z_score']:>8.2f} {r['n_postals']:>10}"
            )
        matrix_str = "\n".join(lines)

    hc_pairs   = flagged_df[flagged_df["flag_hc"] == True]
    z_pairs    = flagged_df[flagged_df["flag_z"]  == True]
    clus_pairs = flagged_df[flagged_df["flag_cluster"] == True]

    prompt = f"""You are an epidemiological signal analyst specializing in rare event detection. Review the following co-occurrence matrix of symptoms reported in the past {window_days} days:

{matrix_str}

Reference date: {ref_date.date()} | Window: {window_days} days
Total pairs evaluated: 91 | Flagging criteria:
  (1) Z-score >2.0: {len(z_pairs)} pairs
  (2) High-concern symptom involvement: {len(hc_pairs)} pairs
  (3) Geographic clustering ≤2 postal codes: {len(clus_pairs)} pairs

Flag any symptom combinations that:
(1) have a prevalence rate >2 standard deviations above the 90-day historical mean;
(2) include high-concern symptoms (Bleeding from Body Openings, Yellow Skin/Eyes, Discolored/Bloody Urine);
(3) cluster in fewer than 3 postal codes.

For each flagged combination:
(a) Generate a differential diagnosis list (top 5 candidate syndromes, Arizona-endemic prioritised).
(b) Recommend confirmatory laboratory investigations.
(c) Recommend confirmatory epidemiological investigations.
(d) Assess whether the pattern is consistent with a true rare event or a reporting artifact.

Structure output as: FLAGGED PAIR | CRITERIA MET | DIFFERENTIAL DIAGNOSES | LAB INVESTIGATIONS | EPI INVESTIGATIONS | ASSESSMENT
"""
    return prompt


def simulate_llm_narrative(flagged_df: pd.DataFrame,
                            window_days: int,
                            ref_date: pd.Timestamp,
                            baseline_days: int) -> str:
    """Deterministic narrative — replace with Anthropic API call in production."""

    truly_flagged = flagged_df[flagged_df["flagged"] == True] if not flagged_df.empty else pd.DataFrame()
    hc_pairs      = flagged_df[flagged_df["flag_hc"]] if not flagged_df.empty else pd.DataFrame()
    z_pairs       = flagged_df[flagged_df["flag_z"]]  if not flagged_df.empty else pd.DataFrame()
    clus_pairs    = flagged_df[flagged_df["flag_cluster"]] if not flagged_df.empty else pd.DataFrame()
    critical_pairs= truly_flagged[truly_flagged["n_flags"] >= 2] if not truly_flagged.empty else pd.DataFrame()

    narrative = f"""RARE EVENT CO-OCCURRENCE SIGNAL REPORT
Region: Arizona | Reference: {ref_date.date()} | Window: {window_days}d recent / {baseline_days}d baseline
Classification: ADHS INTERNAL — SENSITIVE SURVEILLANCE DATA

━━━ EXECUTIVE SUMMARY ━━━

Systematic evaluation of all 91 unique 2-symptom pairs across {len(SYM_KEYS)} EpiHack
symptom dimensions for the {window_days}-day window ending {ref_date.date()}:

  • Pairs with z-score >2 SD above 90-day baseline: {len(z_pairs)}
  • Pairs involving high-concern symptoms:           {len(hc_pairs)}
  • Pairs clustering in ≤2 postal codes:            {len(clus_pairs)}
  • Pairs meeting ≥2 criteria (ESCALATED):          {len(critical_pairs)}

━━━ FLAGGED PAIRS — DETAILED ANALYSIS ━━━
"""

    if truly_flagged.empty:
        narrative += "\nNo pairs exceeded flagging thresholds in the current window. Routine monitoring continues.\n"
    else:
        for rank, (_, row) in enumerate(truly_flagged.head(8).iterrows(), 1):
            flags_str = []
            if row["flag_z"]:
                flags_str.append(f"Z-SCORE={row['z_score']:.2f}")
            if row["flag_hc"]:
                flags_str.append("HIGH-CONCERN SYMPTOM")
            if row["flag_cluster"]:
                flags_str.append(f"GEOGRAPHIC CLUSTER (n={row['n_postals']} postals)")

            # Severity assessment
            if row["n_flags"] >= 2:
                severity = "ESCALATED"
                action   = "Immediate ADHS notification; activate rare disease investigation protocol."
            elif row["flag_z"] and row["z_score"] > 3:
                severity = "HIGH ALERT"
                action   = "Notify county health department; request enhanced data collection; cross-validate with wastewater and ED syndromic streams."
            elif row["flag_hc"]:
                severity = "HIGH ALERT"
                action   = "Clinical referral recommended for co-reporters; activate healthcare provider alert."
            else:
                severity = "MONITOR"
                action   = "Continue enhanced surveillance; collect additional data over next 7 days."

            # Artifact vs true signal
            if row["n_joint"] <= 1:
                interpretation = "POSSIBLE ARTIFACT — single co-reporter; rates may be inflated by coincidental reporting."
            elif row["n_postals"] <= 2 and row["n_joint"] >= 3:
                interpretation = "GEOGRAPHIC CLUSTER — multiple co-reporters in ≤2 postal codes; consistent with point-source exposure."
            elif row["z_score"] > 3:
                interpretation = "STATISTICALLY SIGNIFICANT — rate exceeds 3 SD above baseline; unlikely to be random variation."
            else:
                interpretation = "BORDERLINE — meets one criterion; interpret with caution pending additional data."

            narrative += f"""
[{rank}] {row['label_a']} + {row['label_b']}
    Severity: {severity} | Criteria: {" | ".join(flags_str)}
    Joint prevalence: {row['rec_rate']:.3f}% (baseline: {row['baseline_mean']:.3f}% ± {row['baseline_sd']:.3f}%)
    Co-reporters: n={row['n_joint']} across {row['n_postals']} postal code(s)
    Interpretation: {interpretation}
    Action: {action}

    DIFFERENTIAL DIAGNOSES (top 5, Arizona-prioritised):"""

            if row["ddx"]:
                for di, d in enumerate(row["ddx"][:5], 1):
                    az_flag = " [AZ ENDEMIC]" if d.get("az_endemic") else ""
                    narrative += f"\n      {di}. {d['syndrome']} ({d['concern']}){az_flag}"
                    narrative += f"\n         Pathogen: {d['pathogen']}"
                    narrative += f"\n         Key lab: {d['confirmatory_lab'][0]}"
            else:
                narrative += "\n      No matching DDx entries for this combination."

            narrative += "\n"

    narrative += """
━━━ HIGH-CONCERN SYMPTOM SURVEILLANCE STATUS ━━━

High-concern symptoms (Bleeding from Body Openings, Yellow Skin/Eyes, Discolored/Bloody Urine)
are monitored regardless of statistical threshold, as even isolated reports may represent:
  • Hemorrhagic fever (RMSF, Hantavirus, Dengue) — immediately notifiable
  • Severe hepatic disease (acute viral hepatitis, Weil's disease)
  • Renal emergency (HUS, Hantavirus HFRS, rhabdomyolysis)

Any report of high-concern symptom co-occurrence with any other symptom should trigger:
  1. Clinical alert to reporting postal code's primary care network
  2. Cross-validation with ED syndromic and wastewater data (see US#9)
  3. One Health risk assessment (see US#7)
  4. ADHS Epidemiology duty officer notification within 24 hours

━━━ METHODOLOGICAL NOTE ━━━

The 90-day baseline is divided into 12 non-overlapping 7-day blocks; mean and SD of
joint prevalence are computed across blocks. This block-bootstrap approach ensures
variance estimation reflects genuine week-to-week fluctuation rather than
aggregation-window artefacts. Geographic clustering (≤2 postal codes) serves as an
independent signal of point-source exposure, orthogonal to the statistical threshold.

[NOTE: EpiHack participatory data reflects self-reported symptoms. Co-occurrence of
two rare symptoms in a small reporter pool may represent coincidental co-reporting.
Statistical significance alone does not constitute a public health emergency — always
interpret in clinical and epidemiological context.]
"""
    return narrative


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(flagged_df: pd.DataFrame, ref_date: pd.Timestamp,
                 window_days: int, recent_n: int) -> None:
    sep = "─" * 130
    print()
    print(sep)
    print(f"  RARE EVENT CO-OCCURRENCE REPORT | Ref: {ref_date.date()} | "
          f"Window: {window_days}d | Reporters: {recent_n}")
    print(sep)
    truly = flagged_df[flagged_df["flagged"] == True] if not flagged_df.empty else pd.DataFrame()
    if truly.empty:
        print("  No pairs exceeded flagging thresholds.")
    else:
        print(f"{'PAIR':<52} {'REC%':>7} {'Z':>7} {'N_JOINT':>8} {'N_POST':>7} "
              f"{'FLAGS':>5} {'TOP DDX':<35}")
        print(sep)
        for _, row in truly.iterrows():
            pair = f"{row['label_a']} + {row['label_b']}"[:51]
            flags = ("Z" if row["flag_z"] else " ") + \
                    ("H" if row["flag_hc"] else " ") + \
                    ("C" if row["flag_cluster"] else " ")
            top_ddx = row["ddx"][0]["syndrome"][:34] if row["ddx"] else "—"
            print(f"{pair:<52} {row['rec_rate']:.3f}% {row['z_score']:>7.2f} "
                  f"{row['n_joint']:>8} {row['n_postals']:>7} {flags:>5}  {top_ddx}")
    print(sep)
    z_n = int(flagged_df["flag_z"].sum())   if not flagged_df.empty else 0
    h_n = int(flagged_df["flag_hc"].sum())  if not flagged_df.empty else 0
    c_n = int(flagged_df["flag_cluster"].sum()) if not flagged_df.empty else 0
    print(f"  Z>2: {z_n}  |  High-Concern: {h_n}  |  Cluster≤2: {c_n}  "
          f"|  Escalated (≥2 flags): "
          f"{int((flagged_df['n_flags'] >= 2).sum()) if not flagged_df.empty else 0}")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION — MATPLOTLIB PNG
# ─────────────────────────────────────────────────────────────────────────────

def fig_cooccurrence(recent_mat: np.ndarray,
                     baseline_mean: np.ndarray,
                     baseline_sd: np.ndarray,
                     flagged_df: pd.DataFrame,
                     ref_date: pd.Timestamp,
                     window_days: int,
                     recent_n: int,
                     out_path: str) -> None:
    """
    Four-panel figure:
      A — Co-occurrence heatmap (joint prevalence %, recent window)
      B — Z-score heatmap (SD above baseline mean)
      C — Top flagged pairs bar chart (z-score ranked)
      D — DDx priority table for top escalated pair
    """
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "text.color": TEXT,
        "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
        "grid.color": "#2a2d3e", "font.family": "DejaVu Sans", "font.size": 8.5,
    })

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.patch.set_facecolor(BG)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)

    short_labels = [SYM_SHORT[k] for k in SYM_KEYS]

    fig.text(0.5, 0.97,
             f"Rare Event Co-Occurrence Analysis — Arizona",
             ha="center", va="top", fontsize=16, fontweight="bold", color=TEXT)
    fig.text(0.5, 0.945,
             f"Reference: {ref_date.date()} | Window: {window_days}d | "
             f"Reporters: {recent_n} | 91 symptom pairs evaluated",
             ha="center", va="top", fontsize=9, color=MUTED)

    # ── Panel A: Co-occurrence heatmap ─────────────────────────────────────
    ax_a = axes[0, 0]
    ax_a.set_title("Panel A — Joint Prevalence Heatmap (recent window %)",
                   color=TEXT, fontsize=10, fontweight="bold", pad=8)

    # Mask diagonal (self-co-occurrence not meaningful)
    mat_display = recent_mat.copy()
    np.fill_diagonal(mat_display, np.nan)
    vmax_a = np.nanpercentile(mat_display, 95) if not np.all(np.isnan(mat_display)) else 1.0

    cmap_a = plt.cm.get_cmap("YlOrRd")
    cmap_a.set_bad(color=BG)

    im_a = ax_a.imshow(mat_display, cmap=cmap_a, vmin=0, vmax=vmax_a, aspect="auto")
    ax_a.set_xticks(range(N_SYM))
    ax_a.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=6.5)
    ax_a.set_yticks(range(N_SYM))
    ax_a.set_yticklabels(short_labels, fontsize=6.5)
    plt.colorbar(im_a, ax=ax_a, fraction=0.04, pad=0.02, label="Joint Prevalence %")

    # Highlight high-concern rows/cols
    hc_indices = [SYM_KEYS.index(k) for k in HIGH_CONCERN if k in SYM_KEYS]
    for hci in hc_indices:
        ax_a.axhline(hci - 0.5, color=RED, lw=0.8, alpha=0.6)
        ax_a.axhline(hci + 0.5, color=RED, lw=0.8, alpha=0.6)
        ax_a.axvline(hci - 0.5, color=RED, lw=0.8, alpha=0.6)
        ax_a.axvline(hci + 0.5, color=RED, lw=0.8, alpha=0.6)

    # ── Panel B: Z-score heatmap ────────────────────────────────────────────
    ax_b = axes[0, 1]
    ax_b.set_title("Panel B — Z-Score Heatmap (SD above 90-day baseline mean)",
                   color=TEXT, fontsize=10, fontweight="bold", pad=8)

    z_mat = (recent_mat - baseline_mean) / baseline_sd
    np.fill_diagonal(z_mat, np.nan)

    # Diverging colormap centered at 0; red = elevated, blue = suppressed
    cmap_b = plt.cm.get_cmap("RdBu_r")
    cmap_b.set_bad(color=BG)
    vmax_b = max(abs(np.nanpercentile(z_mat, 2)), abs(np.nanpercentile(z_mat, 98)))
    vmax_b = max(vmax_b, 2.5)

    im_b = ax_b.imshow(z_mat, cmap=cmap_b, vmin=-vmax_b, vmax=vmax_b, aspect="auto")
    ax_b.set_xticks(range(N_SYM))
    ax_b.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=6.5)
    ax_b.set_yticks(range(N_SYM))
    ax_b.set_yticklabels(short_labels, fontsize=6.5)
    cbar_b = plt.colorbar(im_b, ax=ax_b, fraction=0.04, pad=0.02, label="Z-Score")
    cbar_b.ax.axhline(2.0, color=RED, lw=1.5)  # threshold line

    # Overlay flag markers on z-score heatmap
    if not flagged_df.empty:
        for _, row in flagged_df[flagged_df["flag_z"] == True].iterrows():
            xi = SYM_KEYS.index(row["sym_b"])
            yi = SYM_KEYS.index(row["sym_a"])
            ax_b.plot(xi, yi, marker="*", color=YELLOW, markersize=8, alpha=0.9)
            ax_b.plot(yi, xi, marker="*", color=YELLOW, markersize=8, alpha=0.9)

    # ── Panel C: Top flagged pairs bar chart ────────────────────────────────
    ax_c = axes[1, 0]
    ax_c.set_title("Panel C — Top Flagged Pairs by Z-Score",
                   color=TEXT, fontsize=10, fontweight="bold", pad=8)

    truly_flagged = (flagged_df[flagged_df["flagged"] == True]
                     .sort_values("z_score", ascending=True)
                     .tail(12)) if not flagged_df.empty else pd.DataFrame()

    if truly_flagged.empty:
        ax_c.text(0.5, 0.5, "No flagged pairs in current window",
                 ha="center", va="center", color=MUTED, fontsize=11,
                 transform=ax_c.transAxes)
    else:
        pair_labels = [f"{r['label_a'][:14]}\n+{r['label_b'][:14]}"
                       for _, r in truly_flagged.iterrows()]
        z_scores    = truly_flagged["z_score"].values
        bar_colors  = [RED if z > 3 else ORANGE if z > 2 else BLUE
                       for z in z_scores]
        y = np.arange(len(pair_labels))
        ax_c.barh(y, z_scores, color=bar_colors, alpha=0.85, height=0.65)
        ax_c.axvline(2.0, color=ORANGE, linestyle="--", lw=1.2, alpha=0.7,
                     label="Z=2 threshold")
        ax_c.axvline(3.0, color=RED, linestyle="--", lw=1.2, alpha=0.7,
                     label="Z=3 threshold")
        ax_c.set_yticks(y)
        ax_c.set_yticklabels(pair_labels, fontsize=7)
        ax_c.set_xlabel("Z-Score (SD above 90-day baseline)", color=TEXT, fontsize=8)
        ax_c.legend(fontsize=7, facecolor=BG, edgecolor=MUTED, labelcolor=TEXT)
        ax_c.grid(axis="x", alpha=0.3)

        # Flag indicator strip
        for yi, (_, row) in enumerate(truly_flagged.iterrows()):
            flag_txt = ("Z" if row["flag_z"] else "·") + \
                       ("H" if row["flag_hc"] else "·") + \
                       ("C" if row["flag_cluster"] else "·")
            ax_c.text(ax_c.get_xlim()[1] * 0.98, yi, flag_txt,
                     ha="right", va="center", fontsize=6.5, color=YELLOW,
                     fontfamily="monospace")

    # ── Panel D: DDx table for top pair ────────────────────────────────────
    ax_d = axes[1, 1]
    ax_d.set_title("Panel D — Differential Diagnosis: Top Escalated Pair",
                   color=TEXT, fontsize=10, fontweight="bold", pad=8)
    ax_d.axis("off")

    top_row = (flagged_df[(flagged_df["flagged"] == True)]
               .sort_values("n_flags", ascending=False)
               .iloc[0] if not (flagged_df.empty or flagged_df[flagged_df["flagged"]].empty)
               else None)

    if top_row is not None and top_row["ddx"]:
        ax_d.text(0.01, 0.97,
                  f"Pair: {top_row['label_a']} + {top_row['label_b']}",
                  transform=ax_d.transAxes, fontsize=9, fontweight="bold",
                  color=YELLOW, va="top")
        ax_d.text(0.01, 0.90,
                  f"Z={top_row['z_score']:.2f} | "
                  f"Rate={top_row['rec_rate']:.3f}% | "
                  f"n={top_row['n_joint']} cases | "
                  f"{top_row['n_postals']} postal(s)",
                  transform=ax_d.transAxes, fontsize=7.5, color=MUTED, va="top")

        y_pos = 0.82
        concern_colors = {"CRITICAL": RED, "HIGH": ORANGE, "MODERATE": YELLOW, "LOW": GREEN}
        for i, ddx_entry in enumerate(top_row["ddx"][:5]):
            cc = concern_colors.get(ddx_entry["concern"], TEXT)
            ax_d.text(0.01, y_pos,
                      f"{i + 1}. [{ddx_entry['concern']}] {ddx_entry['syndrome']}",
                      transform=ax_d.transAxes, fontsize=8, color=cc,
                      fontweight="bold", va="top")
            ax_d.text(0.01, y_pos - 0.055,
                      f"   Pathogen: {ddx_entry['pathogen'][:65]}",
                      transform=ax_d.transAxes, fontsize=6.5, color=MUTED, va="top")
            ax_d.text(0.01, y_pos - 0.1,
                      f"   Key lab: {ddx_entry['confirmatory_lab'][0][:70]}",
                      transform=ax_d.transAxes, fontsize=6.5, color=TEXT, va="top")
            ax_d.text(0.01, y_pos - 0.145,
                      f"   Key epi: {ddx_entry['confirmatory_epi'][0][:70]}",
                      transform=ax_d.transAxes, fontsize=6.5, color=CYAN, va="top")
            y_pos -= 0.19
    else:
        ax_d.text(0.5, 0.5, "No escalated pairs\nin current window",
                 ha="center", va="center", color=MUTED, fontsize=11,
                 transform=ax_d.transAxes)

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.text(0.5, 0.01,
             "EpiHack Platform | ADHS Epidemiology Division | User Story #11 | "
             "Z>2: Statistical flag | H: High-concern symptom | C: Geographic cluster",
             ha="center", va="bottom", fontsize=7, color=MUTED, style="italic")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[PNG] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# HTML DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def build_html(recent_mat: np.ndarray,
               baseline_mean: np.ndarray,
               baseline_sd: np.ndarray,
               flagged_df: pd.DataFrame,
               ref_date: pd.Timestamp,
               window_days: int,
               baseline_days: int,
               recent_n: int,
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

    # Z-score matrix for heatmap
    z_mat = (recent_mat - baseline_mean) / baseline_sd
    np.fill_diagonal(z_mat, 0)
    np.fill_diagonal(recent_mat, 0)

    mat_rows = [[jf(recent_mat[i, j]) for j in range(N_SYM)] for i in range(N_SYM)]
    z_rows   = [[jf(float(z_mat[i, j])) for j in range(N_SYM)] for i in range(N_SYM)]

    # Flagged data (strip ddx to avoid huge nested JSON; store top 5 summary)
    flagged_records = []
    if not flagged_df.empty:
        for _, row in flagged_df[flagged_df["flagged"] == True].iterrows():
            rec = {k: jf(v) for k, v in row.items() if k != "ddx"}
            rec["ddx_summary"] = [
                {"syndrome": d["syndrome"],
                 "concern":  d["concern"],
                 "pathogen": d["pathogen"],
                 "az_endemic": d["az_endemic"],
                 "lab0":    d["confirmatory_lab"][0] if d["confirmatory_lab"] else "",
                 "epi0":    d["confirmatory_epi"][0] if d["confirmatory_epi"] else "",
                 }
                for d in row["ddx"][:5]
            ]
            flagged_records.append(rec)

    mat_json     = json.dumps(mat_rows)
    z_json       = json.dumps(z_rows)
    flagged_json = json.dumps(flagged_records)
    sym_short_js = json.dumps([SYM_SHORT[k] for k in SYM_KEYS])
    hc_indices   = json.dumps([SYM_KEYS.index(k) for k in HIGH_CONCERN if k in SYM_KEYS])
    narr_html    = narrative.replace("\n", "<br>").replace("━", "—")
    prompt_html  = llm_prompt.replace("\n", "<br>")

    n_z = int(flagged_df["flag_z"].sum())    if not flagged_df.empty else 0
    n_h = int(flagged_df["flag_hc"].sum())   if not flagged_df.empty else 0
    n_c = int(flagged_df["flag_cluster"].sum()) if not flagged_df.empty else 0
    n_esc = int((flagged_df["n_flags"] >= 2).sum()) if not flagged_df.empty else 0
    n_truly = len(flagged_records)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Rare Event Co-Occurrence Analysis — Arizona</title>
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
  .chip-yellow{{background:rgba(255,211,42,.2);border-color:var(--yellow);color:var(--yellow)}}
  .dashboard{{display:grid;grid-template-columns:1fr 1fr;gap:20px;padding:20px 28px}}
  .panel{{background:var(--surface);border-radius:10px;padding:20px;border:1px solid #2a2d3e}}
  .panel-full{{grid-column:1/-1}}
  .panel h2{{font-size:.95rem;color:var(--accent);margin-bottom:14px;font-weight:600}}
  .canvas-wrap{{position:relative;height:340px}}
  .heatmap-wrap{{overflow-x:auto}}
  .hm-table{{border-collapse:separate;border-spacing:2px;font-size:.62rem}}
  .hm-cell{{width:28px;height:28px;border-radius:3px;cursor:default;transition:transform .1s}}
  .hm-cell:hover{{transform:scale(1.5);z-index:10;position:relative}}
  table.signal-table{{width:100%;border-collapse:collapse;font-size:.81rem;margin-top:8px}}
  thead th{{background:rgba(108,99,255,.15);color:var(--accent);padding:8px 10px;text-align:left;
            border-bottom:2px solid var(--accent);white-space:nowrap;cursor:pointer}}
  thead th:hover{{background:rgba(108,99,255,.25)}}
  tbody tr{{border-bottom:1px solid #2a2d3e;transition:background .15s}}
  tbody tr:hover{{background:rgba(108,99,255,.07)}}
  tbody td{{padding:7px 10px;vertical-align:top}}
  .badge{{display:inline-block;border-radius:12px;padding:2px 10px;font-size:.73rem;font-weight:700}}
  .crit{{background:rgba(255,71,87,.2);color:var(--red);border:1px solid var(--red)}}
  .high{{background:rgba(255,165,2,.2);color:var(--orange);border:1px solid var(--orange)}}
  .mod{{background:rgba(255,211,42,.2);color:var(--yellow);border:1px solid var(--yellow)}}
  .flag-z{{color:var(--orange);font-weight:700;font-size:.73rem}}
  .flag-h{{color:var(--red);font-weight:700;font-size:.73rem}}
  .flag-c{{color:var(--cyan);font-weight:700;font-size:.73rem}}
  .ddx-list{{margin-top:6px;padding:8px 12px;background:var(--bg);border-radius:6px;font-size:.78rem;line-height:1.7}}
  .narrative-box{{background:var(--bg);border:1px solid #2a2d3e;border-radius:8px;padding:18px;font-size:.82rem;line-height:1.75;font-family:'Courier New',monospace;max-height:520px;overflow-y:auto}}
  .prompt-box{{background:var(--bg);border:1px solid var(--accent);border-radius:8px;padding:14px;font-size:.77rem;line-height:1.6;color:var(--muted);max-height:300px;overflow-y:auto}}
  .summary-cards{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:16px}}
  .sc{{flex:1;min-width:110px;background:var(--bg);border-radius:8px;padding:14px 12px;border:1px solid #2a2d3e;text-align:center}}
  .sc-val{{font-size:2rem;font-weight:800}}
  .sc-lbl{{font-size:.74rem;color:var(--muted);margin-top:4px}}
  .controls{{background:var(--surface);border-bottom:1px solid #2a2d3e;padding:14px 28px;display:flex;gap:24px;flex-wrap:wrap;align-items:center}}
  .ctrl{{display:flex;align-items:center;gap:8px}}
  .ctrl label{{color:var(--muted);font-size:.82rem}}
  .ctrl input[type=range]{{accent-color:var(--accent)}}
  .ctrl span{{color:var(--accent);font-weight:700;min-width:40px}}
  footer{{text-align:center;padding:16px;color:var(--muted);font-size:.75rem;border-top:1px solid #2a2d3e}}
  .legend-dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}}
</style>
</head>
<body>
<header>
  <h1>Rare Event Co-Occurrence Signal Analysis — Arizona</h1>
  <p>90-day block-bootstrap baseline | 91 symptom pairs | Full Arizona-relevant DDx | {ref_date.date()}</p>
  <div class="meta-bar">
    <span class="chip">User Story #11</span>
    <span class="chip">Window: {window_days}d (n={recent_n})</span>
    <span class="chip">Baseline: {baseline_days}d</span>
    <span class="chip-orange">Z>2 flags: {n_z}</span>
    <span class="chip-red">High-concern: {n_h}</span>
    <span class="chip">Cluster≤2: {n_c}</span>
    <span class="chip-yellow">Escalated (≥2): {n_esc}</span>
  </div>
</header>

<div class="controls">
  <div class="ctrl">
    <label>Z-score threshold:</label>
    <input type="range" id="zThresh" min="1" max="5" step="0.5" value="2"
           oninput="document.getElementById('zVal').textContent=this.value;renderTable()">
    <span id="zVal">2.0</span>
  </div>
  <div class="ctrl">
    <label>Show high-concern only:</label>
    <input type="checkbox" id="hcOnly" onchange="renderTable()">
  </div>
  <div class="ctrl">
    <label>Heatmap mode:</label>
    <select id="hmMode" onchange="renderHeatmap()">
      <option value="joint">Joint Prevalence %</option>
      <option value="zscore">Z-Score</option>
    </select>
  </div>
  <div class="ctrl" style="margin-left:auto;font-size:.78rem;color:var(--muted)">
    <span class="legend-dot" style="background:var(--orange)"></span>Z>2 &nbsp;
    <span class="legend-dot" style="background:var(--red)"></span>High-concern &nbsp;
    <span class="legend-dot" style="background:var(--cyan)"></span>Cluster≤2
  </div>
</div>

<div class="dashboard">

  <!-- Summary Cards -->
  <div class="panel panel-full">
    <div class="summary-cards">
      <div class="sc"><div class="sc-val" style="color:var(--accent)">91</div><div class="sc-lbl">Pairs<br>Evaluated</div></div>
      <div class="sc"><div class="sc-val" style="color:var(--orange)">{n_z}</div><div class="sc-lbl">Z>2<br>Signals</div></div>
      <div class="sc"><div class="sc-val" style="color:var(--red)">{n_h}</div><div class="sc-lbl">High-Concern<br>Pairs</div></div>
      <div class="sc"><div class="sc-val" style="color:var(--cyan)">{n_c}</div><div class="sc-lbl">Geographic<br>Clusters</div></div>
      <div class="sc"><div class="sc-val" style="color:var(--yellow)">{n_esc}</div><div class="sc-lbl">Escalated<br>(≥2 flags)</div></div>
      <div class="sc"><div class="sc-val" style="color:var(--accent)">{n_truly}</div><div class="sc-lbl">Total<br>Flagged</div></div>
      <div class="sc"><div class="sc-val" style="color:var(--green)">{recent_n}</div><div class="sc-lbl">Recent<br>Reporters</div></div>
      <div class="sc"><div class="sc-val" style="color:var(--muted)">12</div><div class="sc-lbl">Baseline<br>Blocks</div></div>
    </div>
  </div>

  <!-- Panel A/B: Heatmap -->
  <div class="panel panel-full">
    <h2 id="hmTitle">Panel A — Joint Prevalence Co-Occurrence Heatmap (%)</h2>
    <div class="heatmap-wrap"><div id="heatmapContainer"></div></div>
    <div style="margin-top:8px;font-size:.73rem;color:var(--muted)">
      Red borders mark high-concern symptom rows/columns. ★ = Z-score flagged cell. Hover cells for detail.
    </div>
  </div>

  <!-- Panel C: Flagged pairs chart -->
  <div class="panel">
    <h2>Panel C — Flagged Pairs by Z-Score</h2>
    <div class="canvas-wrap"><canvas id="zChart"></canvas></div>
  </div>

  <!-- Panel D: DDx summary -->
  <div class="panel">
    <h2>Panel D — Differential Diagnosis Explorer</h2>
    <div id="ddxExplorer" style="font-size:.82rem">
      <p style="color:var(--muted)">Select a flagged pair from Panel E to view its differential diagnosis.</p>
    </div>
  </div>

  <!-- Panel E: Flagged signal table -->
  <div class="panel panel-full">
    <h2>Panel E — Flagged Signal Decision Table <span id="flagCount" style="color:var(--muted);font-size:.82rem"></span></h2>
    <div id="flagTableContainer"></div>
  </div>

  <!-- Panel F: Narrative -->
  <div class="panel panel-full">
    <h2>Panel F — Rare Event Surveillance Narrative (LLM-generated)</h2>
    <div class="narrative-box">{narr_html}</div>
  </div>

  <!-- Panel G: LLM Prompt -->
  <div class="panel panel-full">
    <h2>Panel G — LLM Prompt Template (filled)</h2>
    <div class="prompt-box">{prompt_html}</div>
  </div>

</div>
<footer>
  EpiHack Participatory Epidemiological Surveillance Platform &nbsp;|&nbsp;
  Arizona ADHS · User Story #11 — Rare Event Co-Occurrence Analysis &nbsp;|&nbsp; {ref_date.date()}
</footer>

<script>
const JOINT_MAT  = {mat_json};
const Z_MAT      = {z_json};
const FLAGGED    = {flagged_json};
const SYM_SHORT  = {sym_short_js};
const HC_INDICES = {hc_indices};
const N_SYM = {N_SYM};
const BG="{BG}",SURFACE="{SURFACE}",ACCENT="{ACCENT}",RED="{RED}",
      ORANGE="{ORANGE}",GREEN="{GREEN}",YELLOW="{YELLOW}",BLUE="{BLUE}",
      CYAN="{CYAN}",TEXT="{TEXT}",MUTED="{MUTED}";

// ── Build flagged lookup for quick z-star overlay ─────────────────────────
const flaggedPairs = new Set(FLAGGED.map(r => `${{r.sym_a}}||${{r.sym_b}}`));

// ── Heatmap rendering ─────────────────────────────────────────────────────
function renderHeatmap() {{
  const mode = document.getElementById("hmMode").value;
  const mat  = mode === "joint" ? JOINT_MAT : Z_MAT;
  document.getElementById("hmTitle").textContent = mode === "joint"
    ? "Panel A — Joint Prevalence Co-Occurrence Heatmap (%)"
    : "Panel B — Z-Score Heatmap (SD above 90-day baseline)";

  // Compute colour scale
  const vals = mat.flat().filter(v => v !== null && v !== 0);
  if (!vals.length) return;
  const vmax = mode === "joint" ? Math.max(...vals) : Math.max(Math.abs(Math.min(...vals)), Math.max(...vals));
  const vmin = mode === "joint" ? 0 : -vmax;

  function cellColor(v) {{
    if (v === null) return BG;
    const t = Math.max(0, Math.min(1, (v - vmin) / (vmax - vmin)));
    if (mode === "joint") {{
      const r = Math.round(255 * t);
      const g = Math.round(255 * (1 - t) * 0.6);
      return `rgb(${{r}},${{g}},0)`;
    }} else {{
      if (t > 0.5) {{
        const i = (t - 0.5) * 2;
        return `rgb(${{Math.round(255*i)}},0,0)`;
      }} else {{
        const i = (0.5 - t) * 2;
        return `rgb(0,${{Math.round(100*i)}},${{Math.round(255*i)}})`;
      }}
    }}
  }}

  const container = document.getElementById("heatmapContainer");
  // Header row
  let html = `<table class="hm-table"><tr><td style="width:80px"></td>`;
  SYM_SHORT.forEach((s,j) => {{
    const hcStyle = HC_INDICES.includes(j) ? "color:var(--red);font-weight:700" : "color:var(--muted)";
    html += `<th style="writing-mode:vertical-rl;transform:rotate(180deg);padding:2px 0;font-size:.6rem;${{hcStyle}}">${{s}}</th>`;
  }});
  html += `</tr>`;

  for (let i = 0; i < N_SYM; i++) {{
    const rowHcStyle = HC_INDICES.includes(i) ? "color:var(--red);font-weight:700" : "";
    html += `<tr><td style="font-size:.65rem;white-space:nowrap;padding-right:6px;${{rowHcStyle}}">${{SYM_SHORT[i]}}</td>`;
    for (let j = 0; j < N_SYM; j++) {{
      if (i === j) {{
        html += `<td class="hm-cell" style="background:#2a2d3e"></td>`;
        continue;
      }}
      const v   = mat[i][j];
      const bg  = cellColor(v);
      const isPair = flaggedPairs.has(`${{FLAGGED.map(r=>r.sym_a)[0]||""}}||${{FLAGGED.map(r=>r.sym_b)[0]||""}}`)
                    ? "" : "";  // placeholder
      const star = FLAGGED.some(r =>
        (r.sym_a_idx === i && r.sym_b_idx === j) ||
        (r.sym_a_idx === j && r.sym_b_idx === i)) ? "★" : "";
      const hcBorder = (HC_INDICES.includes(i) || HC_INDICES.includes(j))
        ? "border:1px solid rgba(255,71,87,0.5)" : "";
      const dispV = mode === "joint"
        ? (v !== null ? v.toFixed(2) : "")
        : (v !== null ? v.toFixed(1) : "");
      const tip = `${{SYM_SHORT[i]}} + ${{SYM_SHORT[j]}}: ${{dispV}}${{mode==="joint"?"%":" SD"}}`;
      html += `<td class="hm-cell" style="background:${{bg}};${{hcBorder}}" title="${{tip}}">${{star}}</td>`;
    }}
    html += `</tr>`;
  }}
  html += `</table>`;
  container.innerHTML = html;
}}
renderHeatmap();

// ── Panel C: Z-score bar chart ─────────────────────────────────────────────
(function() {{
  const ctx = document.getElementById("zChart").getContext("2d");
  if (!FLAGGED.length) {{
    ctx.canvas.parentElement.innerHTML = "<p style='color:var(--muted);padding:40px;text-align:center'>No flagged pairs in current window.</p>";
    return;
  }}
  const sorted = [...FLAGGED].sort((a,b) => (b.z_score??0)-(a.z_score??0)).slice(0,12);
  const lbls   = sorted.map(r => `${{r.label_a?.slice(0,12)}}\\n+${{r.label_b?.slice(0,12)}}`);
  const zs     = sorted.map(r => r.z_score ?? 0);
  const clrs   = zs.map(z => z > 3 ? RED+"cc" : z > 2 ? ORANGE+"cc" : BLUE+"cc");
  new Chart(ctx, {{
    type:"bar",
    data:{{
      labels: lbls,
      datasets:[{{label:"Z-Score",data:zs,backgroundColor:clrs,borderColor:clrs.map(c=>c.slice(0,-2)),borderWidth:1}}]
    }},
    options:{{
      indexAxis:"y", responsive:true, maintainAspectRatio:false,
      onClick:(e,el) => {{
        if (el.length) showDDX(sorted[el[0].index]);
      }},
      plugins:{{
        legend:{{display:false}},
        annotation:{{annotations:{{
          z2:{{type:"line",xMin:2,xMax:2,borderColor:ORANGE,borderWidth:1.5,borderDash:[4,3]}},
          z3:{{type:"line",xMin:3,xMax:3,borderColor:RED,borderWidth:1.5,borderDash:[4,3]}},
        }}}},
        tooltip:{{callbacks:{{label:c=>` Z=${{c.raw?.toFixed(2)}} | click to view DDx`}}}}
      }},
      scales:{{
        x:{{ticks:{{color:TEXT}},grid:{{color:"#2a2d3e"}},title:{{display:true,text:"Z-Score",color:MUTED}}}},
        y:{{ticks:{{color:TEXT,font:{{size:8}}}},grid:{{color:"#2a2d3e"}}}}
      }}
    }}
  }});
}})();

// ── DDx Explorer ──────────────────────────────────────────────────────────
function showDDX(row) {{
  const container = document.getElementById("ddxExplorer");
  if (!row || !row.ddx_summary?.length) {{
    container.innerHTML = "<p style='color:var(--muted)'>No DDx available for this pair.</p>";
    return;
  }}
  const conclr = {{"CRITICAL":RED,"HIGH":ORANGE,"MODERATE":YELLOW,"LOW":GREEN}};
  let html = `<div style="margin-bottom:10px">
    <strong style="color:var(--yellow)">${{row.label_a}} + ${{row.label_b}}</strong>
    <span style="color:var(--muted);font-size:.75rem;margin-left:10px">
      Z=${{row.z_score?.toFixed(2)}} | Rate=${{row.rec_rate?.toFixed(3)}}% | n=${{row.n_joint}} | ${{row.n_postals}} postal(s)
    </span>
  </div>`;
  row.ddx_summary.forEach((d,i) => {{
    const c = conclr[d.concern] || TEXT;
    const az = d.az_endemic ? "🌵 AZ Endemic" : "";
    html += `<div style="margin:8px 0;padding:10px;background:var(--bg);border-radius:6px;border-left:3px solid ${{c}}">
      <strong style="color:${{c}}">${{i+1}}. ${{d.syndrome}}</strong>
      <span class="badge ${{d.concern==="CRITICAL"?"crit":d.concern==="HIGH"?"high":"mod"}}" style="margin-left:8px">${{d.concern}}</span>
      ${{az ? `<span style="color:var(--green);font-size:.73rem;margin-left:6px">${{az}}</span>` : ""}}
      <div style="color:var(--muted);font-size:.75rem;margin-top:4px">Pathogen: ${{d.pathogen}}</div>
      <div style="font-size:.75rem;margin-top:3px">🔬 Lab: ${{d.lab0}}</div>
      <div style="font-size:.75rem;color:var(--cyan);margin-top:2px">🔍 Epi: ${{d.epi0}}</div>
    </div>`;
  }});
  container.innerHTML = html;
}}

// ── Panel E: Decision Table ────────────────────────────────────────────────
function renderTable() {{
  const zT  = parseFloat(document.getElementById("zThresh").value);
  const hcO = document.getElementById("hcOnly").checked;

  let data = FLAGGED.filter(r => {{
    if (hcO && !r.flag_hc) return false;
    const meetZ = r.flag_z && (r.z_score ?? 0) > zT;
    const meetH = r.flag_hc;
    const meetC = r.flag_cluster;
    return meetZ || meetH || meetC;
  }});

  document.getElementById("flagCount").textContent = `(${{data.length}} pairs)`;
  const container = document.getElementById("flagTableContainer");

  if (!data.length) {{
    container.innerHTML = "<p style='color:var(--muted);padding:20px;text-align:center'>No pairs match current filters.</p>";
    return;
  }}

  const conclr = {{"CRITICAL":RED,"HIGH":ORANGE,"MODERATE":YELLOW,"LOW":GREEN}};

  let html = `<table class="signal-table"><thead><tr>
    <th>Pair</th><th>Joint %</th><th>Z-Score</th><th>n Cases</th>
    <th>Postals</th><th>Flags</th><th>Top DDx (concern)</th>
  </tr></thead><tbody>`;

  data.forEach(r => {{
    const flags = [
      r.flag_z   ? `<span class="flag-z">Z=${{r.z_score?.toFixed(2)}}</span>` : "",
      r.flag_hc  ? `<span class="flag-h">HIGH-CONCERN</span>` : "",
      r.flag_cluster ? `<span class="flag-c">CLUSTER(${{r.n_postals}})</span>` : "",
    ].filter(Boolean).join(" ");

    const top3ddx = (r.ddx_summary || []).slice(0,3).map(d => {{
      const c = conclr[d.concern] || TEXT;
      const az = d.az_endemic ? " 🌵" : "";
      return `<span style="color:${{c}};font-size:.72rem">${{d.syndrome.split("(")[0].trim()}}${{az}}</span>`;
    }}).join("<br>");

    html += `<tr onclick="showDDX(${{JSON.stringify(r).replace(/"/g,"&quot;")}});" style="cursor:pointer">
      <td><strong style="color:var(--yellow)">${{r.label_a}}</strong><br>
          <span style="color:var(--muted)">+</span> <strong style="color:var(--cyan)">${{r.label_b}}</strong></td>
      <td>${{r.rec_rate?.toFixed(3)}}%<br><span style="color:var(--muted);font-size:.72rem">base: ${{r.baseline_mean?.toFixed(3)}}%</span></td>
      <td style="color:${{(r.z_score??0)>3?RED:(r.z_score??0)>2?ORANGE:TEXT}};font-weight:700">${{r.z_score?.toFixed(2)}}</td>
      <td>${{r.n_joint}}</td>
      <td>${{r.n_postals}}</td>
      <td>${{flags}}</td>
      <td class="ddx-list">${{top3ddx}}</td>
    </tr>`;
  }});
  html += "</tbody></table><p style='color:var(--muted);font-size:.75rem;margin-top:8px'>Click any row to view full differential diagnosis in Panel D.</p>";
  container.innerHTML = html;

  // Auto-show DDx for first row
  if (data.length) showDDX(data[0]);
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
        description="US#11 — Rare Event Co-Occurrence Analysis"
    )
    parser.add_argument("--db",            required=True)
    parser.add_argument("--window",        type=int, default=7)
    parser.add_argument("--baseline-days", type=int, default=90)
    parser.add_argument("--z-threshold",   type=float, default=2.0)
    parser.add_argument("--cluster-max",   type=int, default=3)
    parser.add_argument("--out",           default="rare_event_cooccurrence_report.png")
    args = parser.parse_args()

    png_stem = os.path.splitext(os.path.basename(args.out))[0]
    if os.path.isdir(args.out):
        png_path  = os.path.join(args.out, f"{png_stem}.png")
        html_path = os.path.join(args.out, f"{png_stem}.html")
    else:
        png_path  = args.out if args.out.endswith(".png") else args.out + ".png"
        html_path = os.path.join(os.path.dirname(png_path) or ".", f"{png_stem}.html")

    print()
    print("=" * 65)
    print("  US#11 — Rare Event Co-Occurrence Analysis")
    print(f"  DB: {args.db}")
    print(f"  Window: {args.window}d | Baseline: {args.baseline_days}d")
    print(f"  Z-threshold: {args.z_threshold} | Cluster max: {args.cluster_max}")
    print("=" * 65)

    print("\n[1/6] Loading data …")
    df = load_data(args.db)
    ref_date  = df["date_of_report"].max()
    rec_start = ref_date - pd.Timedelta(days=args.window)
    recent_df = df[df["date_of_report"] > rec_start].copy()
    print(f"      Total records: {len(df):,} | Ref: {ref_date.date()}")
    print(f"      Recent reporters (last {args.window}d): {len(recent_df):,}")

    print("\n[2/6] Building recent co-occurrence matrix …")
    recent_mat = build_cooccurrence_matrix(recent_df)
    nonzero = int((recent_mat > 0).sum()) // 2
    print(f"      Non-zero pairs: {nonzero} / 91")

    print("\n[3/6] Building 90-day baseline distribution (12 blocks) …")
    baseline_mean, baseline_sd = build_baseline_distribution(
        df, ref_date, args.window, args.baseline_days, n_blocks=12
    )
    print(f"      Baseline blocks computed: 12 × {args.window}-day windows")
    print(f"      Max baseline mean: {baseline_mean.max():.3f}%")

    print("\n[4/6] Flagging pairs …")
    flagged_df = flag_pairs(
        recent_df, recent_mat, baseline_mean, baseline_sd,
        z_thresh=args.z_threshold, cluster_thresh=args.cluster_max
    )
    n_z  = int(flagged_df["flag_z"].sum())     if not flagged_df.empty else 0
    n_h  = int(flagged_df["flag_hc"].sum())    if not flagged_df.empty else 0
    n_c  = int(flagged_df["flag_cluster"].sum()) if not flagged_df.empty else 0
    n_esc= int((flagged_df["n_flags"] >= 2).sum()) if not flagged_df.empty else 0
    print(f"      Z>{args.z_threshold}: {n_z} | High-concern: {n_h} | "
          f"Cluster≤{args.cluster_max}: {n_c} | Escalated: {n_esc}")

    print_report(flagged_df, ref_date, args.window, len(recent_df))

    print("\n[5/6] Building LLM prompt and narrative …")
    llm_prompt = build_llm_prompt(flagged_df, recent_mat, args.window, ref_date)
    narrative  = simulate_llm_narrative(
        flagged_df, args.window, ref_date, args.baseline_days
    )

    print("\n[6/6] Generating outputs …")
    fig_cooccurrence(
        recent_mat, baseline_mean, baseline_sd, flagged_df,
        ref_date, args.window, len(recent_df), png_path
    )
    build_html(
        recent_mat, baseline_mean, baseline_sd, flagged_df,
        ref_date, args.window, args.baseline_days, len(recent_df),
        llm_prompt, narrative, html_path
    )

    print()
    print("All outputs saved.")
    print(f"  PNG  → {png_path}")
    print(f"  HTML → {html_path}")


if __name__ == "__main__":
    main()
