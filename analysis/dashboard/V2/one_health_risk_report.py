"""
one_health_risk_report.py
==========================
EpiHack — User Story #7: One Health Risk Intelligence

LLM Prompt Template (filled at runtime):
    "You are a One Health risk intelligence analyst. Integrate the following data streams:
     Human surveillance: {human_signals}. Animal reports: {animal_signals} (source: USDA
     APHIS, AZ Dept Agriculture). Environmental conditions: {environmental_data} (NWS
     HeatRisk, ADHS Wastewater). Vector activity: {vector_data} (ArboNET, Great AZ Tick
     Check). In 400–500 words, produce a One Health risk narrative that: (1) identifies
     symptom patterns consistent with zoonotic etiology; (2) correlates human signals with
     animal/environmental/vector data; (3) assigns a zoonotic spillover risk level
     (Low/Medium/High/Critical); (4) recommends cross-sector response actions."

One Health Framework:
    Integrates human, animal, and environmental health under the principle that the health
    of people, animals, and ecosystems are deeply interconnected. Arizona's desert ecology
    supports endemic zoonotic pathogens including Coccidioides (Valley Fever), West Nile
    Virus, Rocky Mountain Spotted Fever, Hantavirus, Plague, Q Fever, and Brucellosis.

Zoonotic Risk Scoring (0–100):
    Human component   (0–40): elevated signals, high-acuity, NEW patterns
    Animal component  (0–25): livestock illness, wildlife die-off, companion animal cases
    Environmental     (0–20): wastewater signal, heat risk, drought index
    Vector activity   (0–15): mosquito trap counts, tick surveillance positivity

    Risk Levels: Low=0–25 | Medium=26–50 | High=51–75 | Critical=76–100

Usage:
    python one_health_risk_report.py --db epihack-mini.db [--postal 85301]
                                      [--recent 7] [--baseline 30] [--out figures/]
"""

# ============================================================
# I. IMPORTS & CONFIGURATION
# ============================================================
import argparse
import hashlib
import sqlite3
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

SYMPTOM_COLS = [
    ("cough_congestion",               "Cough / Congestion"),
    ("sore_throat",                    "Sore Throat"),
    ("difficulty_breathing",           "Difficulty Breathing"),
    ("nauseas_vomiting",               "Nausea / Vomiting"),
    ("diarrhea",                       "Diarrhea"),
    ("muscle_or_body_aches_and_pains", "Muscle / Body Aches"),
    ("fever",                          "Fever"),
    ("chills",                         "Chills"),
    ("red_eyes",                       "Red Eyes"),
    ("rash",                           "Rash"),
    ("loss_of_smell_or_taste",         "Loss of Smell/Taste"),
    ("bleeding_from_body_openings",    "Bleeding (Body Openings)"),
    ("yellow_skin_yellow_eyes",        "Yellow Skin / Eyes"),
    ("discolored_or_bloody_urine",     "Discolored/Bloody Urine"),
]
COL_TO_LABEL = {c: l for c, l in SYMPTOM_COLS}
HIGH_ACUITY = {
    "difficulty_breathing", "bleeding_from_body_openings",
    "yellow_skin_yellow_eyes", "discolored_or_bloody_urine",
}

# Arizona-endemic zoonotic pathogens and their symptom associations
# Format: {pathogen: {symptom_col: weight (0–3)}}
ZOONOTIC_PATHOGEN_MAP = {
    "West Nile Virus\n(ArboNET)": {
        "fever":3, "muscle_or_body_aches_and_pains":3, "rash":2,
        "red_eyes":2, "nauseas_vomiting":1,
    },
    "Rocky Mountain\nSpotted Fever": {
        "fever":3, "rash":3, "muscle_or_body_aches_and_pains":2,
        "nauseas_vomiting":2, "bleeding_from_body_openings":2,
    },
    "Hantavirus\n(Rodent)": {
        "fever":3, "muscle_or_body_aches_and_pains":3,
        "difficulty_breathing":3, "nauseas_vomiting":2, "diarrhea":1,
    },
    "Q Fever\n(Livestock)": {
        "fever":3, "muscle_or_body_aches_and_pains":2, "chills":3,
        "nauseas_vomiting":2, "diarrhea":2, "yellow_skin_yellow_eyes":1,
    },
    "Valley Fever\n(Env. Fungi)": {
        "cough_congestion":3, "fever":2, "muscle_or_body_aches_and_pains":2,
        "rash":2, "chills":2, "difficulty_breathing":2,
    },
    "Salmonellosis\n(Livestock)": {
        "diarrhea":3, "fever":2, "nauseas_vomiting":3,
        "muscle_or_body_aches_and_pains":1, "chills":1,
    },
    "Leptospirosis\n(Wildlife)": {
        "fever":3, "muscle_or_body_aches_and_pains":3, "red_eyes":3,
        "nauseas_vomiting":2, "yellow_skin_yellow_eyes":2,
        "discolored_or_bloody_urine":2,
    },
    "Plague\n(Rodent)": {
        "fever":3, "chills":3, "muscle_or_body_aches_and_pains":2,
        "bleeding_from_body_openings":2, "difficulty_breathing":1,
    },
}

# County → ecological zone mapping
COUNTY_ECOLOGY = {
    "Maricopa": "Desert Basin",  "Pima": "Desert Basin",
    "Pinal": "Desert Basin",     "Yuma": "Desert Basin",
    "La Paz": "Desert Basin",    "Mohave": "Desert Basin",
    "Santa Cruz": "Transitional","Yavapai": "Transitional",
    "Gila": "Transitional",      "Graham": "Transitional",
    "Cochise": "Transitional",   "Coconino": "High Plateau",
    "Apache": "High Plateau",    "Navajo": "High Plateau",
    "Greenlee": "High Plateau",
}

# Risk level thresholds and colors
RISK_LEVELS = [
    (76, "Critical", "#ee5a24"),
    (51, "High",     "#ff4757"),
    (26, "Medium",   "#ffa502"),
    (0,  "Low",      "#43e97b"),
]

# Design palette
BG      = "#0f1117"
SURFACE = "#1a1d27"
SURFACE2= "#23263a"
ACCENT  = "#6c63ff"
RED     = "#ff4757"
ORANGE  = "#ffa502"
GREEN   = "#43e97b"
YELLOW  = "#f9ca24"
TEAL    = "#2bcbba"
MUTED   = "#8892b0"
TEXT    = "#e8eaf6"


# ============================================================
# II. DATA LOADING
# ============================================================
def load_data(db_path: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df  = pd.read_sql("SELECT * FROM incidents", con)
    con.close()
    df["date_of_report"] = pd.to_datetime(df["date_of_report"])
    for col, _ in SYMPTOM_COLS:
        if col in df.columns:
            df[col] = df[col].map({"YES": 1, "NO": 0}).astype("Float64")
    print(f"[1/9] Data loaded: {len(df):,} records | "
          f"{df['date_of_report'].min().date()} -- {df['date_of_report'].max().date()}")
    return df


# ============================================================
# III. HUMAN SIGNAL COMPUTATION
# ============================================================
def compute_human_signals(df: pd.DataFrame, postal: str,
                           recent_days: int = 7,
                           baseline_days: int = 30) -> dict:
    """
    Compute 7-day prevalence rates and signal classifications for one postal code.
    Returns dict with symptom-level signal data and aggregate summary.
    """
    max_date  = df["date_of_report"].max()
    rec_end   = max_date
    rec_start = rec_end   - pd.Timedelta(days=recent_days - 1)
    bas_end   = rec_start - pd.Timedelta(days=1)
    bas_start = bas_end   - pd.Timedelta(days=baseline_days - 1)

    g_rec = df[(df["postal_code"] == postal) &
               (df["date_of_report"] >= rec_start) &
               (df["date_of_report"] <= rec_end)]
    g_bas = df[(df["postal_code"] == postal) &
               (df["date_of_report"] >= bas_start) &
               (df["date_of_report"] <= bas_end)]

    signals = {}
    for col, label in SYMPTOM_COLS:
        if col not in df.columns:
            continue
        v_rec = g_rec[col].dropna()
        v_bas = g_bas[col].dropna()
        rec_n = len(v_rec)
        bas_n = len(v_bas)
        rec_rate = float(v_rec.mean() * 100) if rec_n > 0 else 0.0
        bas_rate = float(v_bas.mean() * 100) if bas_n > 0 else 0.0
        delta = rec_rate - bas_rate

        sig = "indef"
        if rec_n >= 2 and bas_n >= 5:
            if bas_rate == 0 and rec_rate > 0:
                sig = "new"
            elif rec_rate == 0 and bas_rate > 0:
                sig = "gone"
            elif delta > 5:
                sig = "up"
            elif delta < -5:
                sig = "dn"
            else:
                sig = "nc"

        signals[col] = {
            "label":       label,
            "signal":      sig,
            "rec_rate":    round(rec_rate, 2),
            "bas_rate":    round(bas_rate, 2),
            "delta":       round(delta, 2),
            "rec_n":       rec_n,
            "bas_n":       bas_n,
            "high_acuity": col in HIGH_ACUITY,
        }

    elevated = [v for v in signals.values() if v["signal"] in ("up", "new")]
    return {
        "postal":       postal,
        "rec_start":    rec_start.date(),
        "rec_end":      rec_end.date(),
        "n_rec_total":  len(g_rec),
        "symptoms":     signals,
        "elevated":     elevated,
        "n_elevated":   len(elevated),
        "n_new":        sum(1 for v in elevated if v["signal"] == "new"),
        "n_high_acuity": sum(1 for v in elevated if v["high_acuity"]),
        "county":        df[df["postal_code"] == postal]["county"].mode().iloc[0]
                         if len(g_rec) > 0 else "Unknown",
    }


# ============================================================
# IV. SIMULATED ANIMAL REPORTS (USDA APHIS / AZ Dept Agriculture)
# ============================================================
def simulate_animal_reports(postal: str, county: str, ref_date: pd.Timestamp) -> dict:
    """
    Deterministic pseudo-random animal surveillance data anchored to postal + date.
    In production, replace with:
        GET https://aphis.usda.gov/api/v1/animal-disease-reports?state=AZ&county={county}
        GET https://azda.az.gov/api/v1/livestock-reports?postal={postal}
    """
    seed = int(hashlib.md5(f"{postal}{ref_date.strftime('%Y%m')}".encode()).hexdigest(), 16) % 10000
    rng  = np.random.default_rng(seed)

    # Livestock illness reports (sheep, cattle, poultry)
    livestock_cases = int(rng.integers(0, 12))
    livestock_species = rng.choice(
        ["cattle", "sheep/goats", "poultry", "swine", "horses"],
        size=min(livestock_cases, 3), replace=False
    ).tolist() if livestock_cases > 0 else []
    livestock_symptoms = rng.choice(
        ["fever and lethargy", "respiratory distress", "diarrhea", "abortion storms",
         "skin lesions", "neurological signs", "sudden death"],
        size=min(livestock_cases, 2), replace=False
    ).tolist() if livestock_cases > 0 else []

    # Wildlife die-off events
    wildlife_dieoff = rng.random() < 0.18
    wildlife_species = rng.choice(
        ["coyotes", "rabbits/jackrabbits", "prairie dogs", "deer", "bats", "raptors"],
    ).item() if wildlife_dieoff else None
    wildlife_count = int(rng.integers(3, 25)) if wildlife_dieoff else 0

    # Companion animal cases (veterinary reports)
    companion_cases = int(rng.integers(0, 8))
    companion_species = rng.choice(
        ["dogs", "cats", "horses"], size=min(companion_cases, 2), replace=False
    ).tolist() if companion_cases > 0 else []

    return {
        "livestock_cases":    livestock_cases,
        "livestock_species":  livestock_species,
        "livestock_symptoms": livestock_symptoms,
        "wildlife_dieoff":    wildlife_dieoff,
        "wildlife_species":   wildlife_species,
        "wildlife_count":     wildlife_count,
        "companion_cases":    companion_cases,
        "companion_species":  companion_species,
        "source":             "USDA APHIS / AZ Dept Agriculture (simulated)",
    }


# ============================================================
# V. SIMULATED ENVIRONMENTAL CONDITIONS
# ============================================================
def simulate_environmental(postal: str, county: str, ref_date: pd.Timestamp) -> dict:
    """
    Deterministic environmental data simulation.
    In production, replace with:
        NWS HeatRisk: GET https://api.weather.gov/gridpoints/{office}/{x},{y}/forecast
        ADHS Wastewater: GET https://www.azdhs.gov/api/wastewater?postal={postal}
    """
    seed = int(hashlib.md5(f"{postal}{ref_date.strftime('%Y%m')}env".encode()).hexdigest(), 16) % 10000
    rng  = np.random.default_rng(seed)

    MONTH_HEAT = {1:0,2:0,3:1,4:2,5:3,6:4,7:4,8:4,9:3,10:2,11:1,12:0}
    COUNTY_HEAT = {
        "Maricopa":2,"Pima":2,"Pinal":2,"Yuma":2,"La Paz":2,"Mohave":2,
        "Santa Cruz":1,"Yavapai":1,"Gila":1,"Graham":1,"Cochise":1,
        "Coconino":0,"Apache":0,"Navajo":0,"Greenlee":0,
    }
    base_heat = MONTH_HEAT.get(ref_date.month, 0) + COUNTY_HEAT.get(county, 1)
    heat_risk = int(np.clip(base_heat + int((ref_date.day * 17 + ref_date.month * 7) % 3) - 1, 0, 4))

    # Wastewater SARS-CoV-2 and enteric pathogen signals
    wastewater_covid = rng.choice(["not detected", "low", "moderate", "high", "very high"],
                                   p=[0.10, 0.35, 0.30, 0.15, 0.10]).item()
    wastewater_enteric = rng.choice(["not detected", "low", "moderate", "elevated"],
                                     p=[0.25, 0.40, 0.25, 0.10]).item()

    # Drought index (PDSI proxy)
    drought_level = rng.choice(["D0 (Abnormally Dry)", "D1 (Moderate Drought)",
                                  "D2 (Severe Drought)", "D3 (Extreme Drought)", "None"],
                                 p=[0.20, 0.30, 0.25, 0.10, 0.15]).item()

    # Soil disturbance (Valley Fever risk amplifier)
    soil_disturbance = rng.choice(["none", "low", "moderate", "high"],
                                   p=[0.30, 0.35, 0.25, 0.10]).item()

    # Precipitation anomaly
    precip_anomaly = rng.choice(
        ["-90% (severe deficit)", "-50% (deficit)", "near normal",
         "+50% (surplus)", "+100% (excess)"],
        p=[0.10, 0.30, 0.35, 0.15, 0.10]).item()

    return {
        "heat_risk":         heat_risk,
        "heat_risk_label":   ["Little/No Risk","Minor","Moderate","Major","Extreme"][heat_risk],
        "wastewater_covid":  wastewater_covid,
        "wastewater_enteric":wastewater_enteric,
        "drought_level":     drought_level,
        "soil_disturbance":  soil_disturbance,
        "precip_anomaly":    precip_anomaly,
        "ecological_zone":   COUNTY_ECOLOGY.get(county, "Unknown"),
        "source":            "NWS HeatRisk / ADHS Wastewater / NOAA PDSI (simulated)",
    }


# ============================================================
# VI. SIMULATED VECTOR ACTIVITY (ArboNET / Great AZ Tick Check)
# ============================================================
def simulate_vector_data(postal: str, county: str, ref_date: pd.Timestamp) -> dict:
    """
    Deterministic vector surveillance simulation.
    In production, replace with:
        ArboNET: CDC ArboNET reporting system
        Tick Check: greataztickscheck.org/api/reports?postal={postal}
        Mosquito traps: Maricopa Vector Control / Pima County HV District
    """
    seed = int(hashlib.md5(f"{postal}{ref_date.strftime('%Y%m')}vec".encode()).hexdigest(), 16) % 10000
    rng  = np.random.default_rng(seed)

    # Mosquito trap activity (counts per trap-night)
    is_summer = ref_date.month in [5, 6, 7, 8, 9, 10]
    base_trap = 45 if is_summer else 8
    trap_count = int(rng.integers(base_trap // 2, base_trap * 2))
    trap_threshold = 50
    mosquito_alert = trap_count > trap_threshold

    # West Nile Virus mosquito pool positivity
    wnv_positive_pools = int(rng.integers(0, 5)) if is_summer else 0
    wnv_positivity_rate = round(float(rng.uniform(0, 12)) if wnv_positive_pools > 0 else 0.0, 1)

    # Tick species and density
    tick_species = rng.choice(
        ["Dermacentor variabilis (American Dog Tick)",
         "Amblyomma americanum (Lone Star Tick)",
         "Rhipicephalus sanguineus (Brown Dog Tick)",
         "Dermacentor andersoni (Rocky Mtn Wood Tick)"],
        size=rng.integers(0, 3), replace=False
    ).tolist()
    tick_density = rng.choice(["absent","low","moderate","high"],
                               p=[0.25,0.40,0.25,0.10]).item()
    tick_pathogen_positive = rng.random() < 0.12  # 12% RMSF positivity rate

    # Rodent activity (Hantavirus / Plague risk)
    rodent_index = rng.choice(["below average","average","elevated","high"],
                               p=[0.20,0.45,0.25,0.10]).item()
    rodent_species = rng.choice(
        ["Peromyscus sp. (deer mouse)", "Neotoma sp. (woodrat)",
         "Dipodomys sp. (kangaroo rat)", "Sigmodon hispidus (cotton rat)"],
    ).item()

    # Flea index (Plague amplifier)
    flea_index = rng.choice(["below threshold","at threshold","above threshold"],
                             p=[0.65, 0.20, 0.15]).item()

    return {
        "trap_count":            trap_count,
        "trap_threshold":        trap_threshold,
        "mosquito_alert":        mosquito_alert,
        "wnv_positive_pools":    wnv_positive_pools,
        "wnv_positivity_rate":   wnv_positivity_rate,
        "tick_species":          tick_species,
        "tick_density":          tick_density,
        "tick_pathogen_positive":tick_pathogen_positive,
        "rodent_index":          rodent_index,
        "rodent_species":        rodent_species,
        "flea_index":            flea_index,
        "source":                "ArboNET / Great AZ Tick Check / Local Vector Control (simulated)",
    }


# ============================================================
# VII. ZOONOTIC RISK SCORING
# ============================================================
def score_zoonotic_risk(human: dict, animal: dict,
                         env: dict, vector: dict) -> dict:
    """
    Composite zoonotic spillover risk score (0–100) integrating four streams.

    Human component (0–40):
        - Elevated symptom count    → 0–15 (3 pts each, max 15)
        - NEW signals               → 0–10 (5 pts each, max 10)
        - High-acuity signals       → 0–15 (5 pts each, max 15)

    Animal component (0–25):
        - Livestock cases > 0       → 5
        - Livestock cases > 5       → +5 (total 10)
        - Wildlife die-off          → 12
        - Companion animal cases    → 3

    Environmental component (0–20):
        - Wastewater enteric elev.  → 0–8
        - HeatRisk >= 3             → 5
        - Drought D2+               → 4
        - Soil disturbance high     → 3

    Vector component (0–15):
        - Mosquito trap alert       → 5
        - WNV positive pools        → 0–5
        - Tick pathogen positive    → 5
        - Rodent index elevated     → 3
        - Flea above threshold      → 2
    """
    # Human
    h_score = 0
    h_score += min(human["n_elevated"] * 3, 15)
    h_score += min(human["n_new"] * 5, 10)
    h_score += min(human["n_high_acuity"] * 5, 15)
    h_score = min(h_score, 40)

    # Animal
    a_score = 0
    if animal["livestock_cases"] > 0:
        a_score += 5
    if animal["livestock_cases"] > 5:
        a_score += 5
    if animal["wildlife_dieoff"]:
        a_score += 12
    if animal["companion_cases"] > 0:
        a_score += 3
    a_score = min(a_score, 25)

    # Environmental
    e_score = 0
    enteric_weights = {"not detected": 0, "low": 2, "moderate": 5, "elevated": 8}
    e_score += enteric_weights.get(env["wastewater_enteric"], 0)
    if env["heat_risk"] >= 3:
        e_score += 5
    if "D2" in env["drought_level"] or "D3" in env["drought_level"]:
        e_score += 4
    if env["soil_disturbance"] == "high":
        e_score += 3
    e_score = min(e_score, 20)

    # Vector
    v_score = 0
    if vector["mosquito_alert"]:
        v_score += 5
    v_score += min(vector["wnv_positive_pools"] * 1, 5)
    if vector["tick_pathogen_positive"]:
        v_score += 5
    if vector["rodent_index"] in ("elevated", "high"):
        v_score += 3
    if vector["flea_index"] == "above threshold":
        v_score += 2
    v_score = min(v_score, 15)

    total = h_score + a_score + e_score + v_score

    # Risk level
    risk_level = "Low"
    risk_color  = GREEN
    for threshold, label, color in RISK_LEVELS:
        if total >= threshold:
            risk_level = label
            risk_color  = color
            break

    return {
        "human_score":  h_score,
        "animal_score": a_score,
        "env_score":    e_score,
        "vector_score": v_score,
        "total":        total,
        "risk_level":   risk_level,
        "risk_color":   risk_color,
    }


# ============================================================
# VIII. SYMPTOM-PATHOGEN CORRELATION
# ============================================================
def compute_pathogen_correlation(human: dict) -> pd.DataFrame:
    """
    Compute concordance score between observed elevated human symptoms
    and each Arizona-endemic zoonotic pathogen's symptom profile.

    concordance(pathogen) = sum(weight_i * reported_i) /
                            sum(weight_i for all pathogen symptoms)
    """
    elevated_cols = {col for col, v in human["symptoms"].items()
                     if v["signal"] in ("up", "new")}

    rows = []
    for pathogen, weights in ZOONOTIC_PATHOGEN_MAP.items():
        total_weight = sum(weights.values())
        concordant   = sum(w for col, w in weights.items() if col in elevated_cols)
        score = concordant / total_weight * 100 if total_weight > 0 else 0
        rows.append({
            "pathogen":       pathogen,
            "concordance_pct": round(score, 1),
            "total_weight":   total_weight,
            "matched_cols":   [COL_TO_LABEL.get(c, c) for c in weights if c in elevated_cols],
        })

    df = pd.DataFrame(rows).sort_values("concordance_pct", ascending=False)
    return df


# ============================================================
# IX. LLM PROMPT FILL
# ============================================================
def build_llm_prompt(human: dict, animal: dict, env: dict,
                     vector: dict, risk: dict, pathogen_df: pd.DataFrame) -> str:
    """
    Fill the User Story #7 LLM prompt with all four data streams.

    Production replacement:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}]
        )
        narrative = response.content[0].text  # expected ~400-500 words
    """
    # Human signals string
    el_strs = []
    for v in sorted(human["elevated"], key=lambda x: -x["delta"]):
        ha = " [HIGH-ACUITY]" if v["high_acuity"] else ""
        el_strs.append(
            f"{v['label']} ({v['signal'].upper()}, +{v['delta']:.1f}pp, "
            f"rec={v['rec_rate']:.1f}%, n={v['rec_n']}){ha}"
        )
    human_str = "; ".join(el_strs) if el_strs else "No elevated signals detected"

    # Animal signals string
    animal_parts = []
    if animal["livestock_cases"] > 0:
        animal_parts.append(
            f"{animal['livestock_cases']} livestock illness reports "
            f"({', '.join(animal['livestock_species'])}) "
            f"presenting with {', '.join(animal['livestock_symptoms'])}"
        )
    if animal["wildlife_dieoff"]:
        animal_parts.append(
            f"wildlife die-off: {animal['wildlife_count']} {animal['wildlife_species']} found"
        )
    if animal["companion_cases"] > 0:
        animal_parts.append(
            f"{animal['companion_cases']} companion animal cases "
            f"({', '.join(animal['companion_species'])})"
        )
    animal_str = "; ".join(animal_parts) if animal_parts else "No unusual animal reports"

    # Environmental string
    env_str = (
        f"NWS HeatRisk={env['heat_risk']}/4 ({env['heat_risk_label']}); "
        f"Wastewater COVID={env['wastewater_covid']}, enteric={env['wastewater_enteric']}; "
        f"Drought={env['drought_level']}; Soil disturbance={env['soil_disturbance']}; "
        f"Precip anomaly={env['precip_anomaly']}; "
        f"Ecological zone={env['ecological_zone']}"
    )

    # Vector string
    tick_str = (
        f"tick pathogen-positive: YES [{', '.join(vector['tick_species'][:2])}]"
        if vector["tick_pathogen_positive"] and vector["tick_species"]
        else f"tick pathogen-positive: NO (density={vector['tick_density']})"
    )
    vec_str = (
        f"Mosquito trap count={vector['trap_count']}/trap-night "
        f"(threshold={vector['trap_threshold']}, alert={'YES' if vector['mosquito_alert'] else 'NO'}); "
        f"WNV positive pools={vector['wnv_positive_pools']} "
        f"(positivity={vector['wnv_positivity_rate']:.1f}%); {tick_str}; "
        f"Rodent index={vector['rodent_index']} ({vector['rodent_species']}); "
        f"Flea index={vector['flea_index']}"
    )

    prompt = textwrap.dedent(f"""
        You are a One Health risk intelligence analyst. Integrate the following data streams:

        Human surveillance: {human_str}.

        Animal reports: {animal_str} (source: USDA APHIS, AZ Dept Agriculture).

        Environmental conditions: {env_str} (NWS HeatRisk, ADHS Wastewater).

        Vector activity: {vec_str} (ArboNET, Great AZ Tick Check).

        In 400–500 words, produce a One Health risk narrative that:
        (1) identifies symptom patterns consistent with zoonotic etiology;
        (2) correlates human signals with animal/environmental/vector data;
        (3) assigns a zoonotic spillover risk level (Low/Medium/High/Critical):
            Pre-computed composite risk score = {risk['total']}/100
            (Human={risk['human_score']}, Animal={risk['animal_score']},
             Env={risk['env_score']}, Vector={risk['vector_score']});
        (4) recommends cross-sector response actions.

        Top pathogen concordance scores:
        {chr(10).join(f"  {r['pathogen'].replace(chr(10),' ')}: {r['concordance_pct']:.1f}%" for _, r in pathogen_df.head(4).iterrows())}

        Postal code: {human['postal']} | County: {human['county']}
        Ecological zone: {env['ecological_zone']}
        Surveillance period: {human['rec_start']} to {human['rec_end']}
    """).strip()
    return prompt


# ============================================================
# X. SIMULATED 400–500 WORD LLM NARRATIVE
# ============================================================
def simulate_llm_narrative(human: dict, animal: dict, env: dict,
                            vector: dict, risk: dict,
                            pathogen_df: pd.DataFrame) -> str:
    """
    Rule-based ~400–500 word One Health risk narrative.

    Production replacement (inline comment):
        # response = client.messages.create(
        #     model="claude-opus-4-6", max_tokens=700,
        #     messages=[{"role":"user","content": prompt}]
        # )
        # return response.content[0].text
    """
    postal  = human["postal"]
    county  = human["county"]
    zone    = env["ecological_zone"]
    rl      = risk["risk_level"]
    total   = risk["total"]
    elev    = human["elevated"]
    top_path= pathogen_df.iloc[0] if not pathogen_df.empty else None
    second  = pathogen_df.iloc[1] if len(pathogen_df) > 1 else None

    # Build elevated symptom string
    el_list = [v["label"] for v in elev[:5]]
    el_str  = ", ".join(el_list) if el_list else "no elevated symptoms"

    # Opening: human signal characterization
    para1 = (
        f"POSTAL {postal} ONE HEALTH RISK ASSESSMENT ({human['rec_start']} to {human['rec_end']})\n\n"
        f"[RISK LEVEL: {rl.upper()} | Score: {total}/100]\n\n"
        f"(1) ZOONOTIC SYMPTOM PATTERN ANALYSIS\n\n"
        f"Human syndromic surveillance in postal code {postal} ({county} County, {zone} "
        f"ecological zone) has detected {human['n_elevated']} elevated symptom signals "
        f"during the 7-day reference period: {el_str}. "
    )
    if top_path is not None and top_path["concordance_pct"] > 30:
        para1 += (
            f"This constellation of symptoms demonstrates {top_path['concordance_pct']:.0f}% "
            f"concordance with the clinical presentation profile of "
            f"{top_path['pathogen'].replace(chr(10), ' ')}, "
            f"an Arizona-endemic zoonotic pathogen. "
        )
        if second is not None and float(second["concordance_pct"]) > 20:
            para1 += (
                f"Secondary concordance of {second['concordance_pct']:.0f}% is noted for "
                f"{second['pathogen'].replace(chr(10), ' ')}, "
                f"which cannot be excluded on clinical grounds alone. "
            )
    if human["n_high_acuity"] > 0:
        ha_syms = [v["label"] for v in elev if v["high_acuity"]]
        para1 += (
            f"High-acuity symptoms ({', '.join(ha_syms)}) are present and require "
            f"urgent clinical evaluation to exclude severe zoonotic disease. "
        )

    # Animal and environmental correlation
    para2 = f"\n\n(2) CROSS-SECTOR DATA CORRELATION\n\n"
    if animal["livestock_cases"] > 0:
        para2 += (
            f"Concurrent animal health data document {animal['livestock_cases']} livestock "
            f"illness reports ({', '.join(animal['livestock_species'])}) presenting with "
            f"{', '.join(animal['livestock_symptoms'])} in the same county. "
            f"Temporal co-occurrence of livestock illness and human febrile-respiratory "
            f"or enteric syndromes in the same geographic area warrants investigation for "
            f"shared zoonotic etiology, including Q fever (Coxiella burnetii) and "
            f"Salmonellosis. "
        )
    if animal["wildlife_dieoff"]:
        para2 += (
            f"A wildlife die-off event involving {animal['wildlife_count']} "
            f"{animal['wildlife_species']} has been reported, which may signal amplified "
            f"pathogen circulation in the peridomestic environment. "
        )
    env_detail = (
        f"Environmental conditions amplify zoonotic transmission risk: NWS HeatRisk is "
        f"{env['heat_risk']}/4 ({env['heat_risk_label']}), {env['drought_level']} conditions "
        f"are active in the region, and soil disturbance is rated {env['soil_disturbance']} — "
        f"a known risk multiplier for aerosolized Coccidioides (Valley Fever). "
        f"ADHS wastewater surveillance shows {env['wastewater_enteric']} enteric pathogen "
        f"signal and {env['wastewater_covid']} SARS-CoV-2 signal. "
        f"Precipitation anomaly of {env['precip_anomaly']} alters rodent population dynamics "
        f"and vector breeding habitat availability. "
    )
    para2 += env_detail

    vec_detail = (
        f"Vector surveillance data from ArboNET and Great AZ Tick Check indicate: mosquito "
        f"trap counts of {vector['trap_count']} per trap-night "
        f"({'exceeding' if vector['mosquito_alert'] else 'below'} the alert threshold of "
        f"{vector['trap_threshold']}), with {vector['wnv_positive_pools']} West Nile Virus "
        f"positive pools detected. Tick surveillance identifies {vector['tick_density']} "
        f"density of Ixodid species"
    )
    if vector["tick_pathogen_positive"] and vector["tick_species"]:
        vec_detail += (
            f", with pathogen-positive specimens confirmed "
            f"({vector['tick_species'][0] if vector['tick_species'] else 'Dermacentor sp.'}). "
        )
    else:
        vec_detail += f"; no tick-borne pathogens confirmed in current surveillance. "
    vec_detail += (
        f"Peridomestic rodent activity is {vector['rodent_index']} "
        f"({vector['rodent_species']} predominant), and flea index is {vector['flea_index']}, "
        f"relevant to plague and Rickettsia risk assessment. "
    )
    para2 += vec_detail

    # Risk assignment
    risk_rationale = {
        "Low":     "No concurrent animal or vector signals amplify current human surveillance patterns.",
        "Medium":  "At least one animal or vector data stream shows co-elevation with human signals, suggesting possible zoonotic transmission pressure.",
        "High":    "Multiple data streams show concurrent elevation; epidemiological linkage is plausible and warrants active investigation.",
        "Critical":"Convergent elevation across all four data streams constitutes a sentinel zoonotic spillover event requiring immediate multi-agency response.",
    }
    para3 = (
        f"\n\n(3) ZOONOTIC SPILLOVER RISK LEVEL: {rl.upper()} ({total}/100)\n\n"
        f"Composite risk score breakdown — Human: {risk['human_score']}/40, "
        f"Animal: {risk['animal_score']}/25, Environmental: {risk['env_score']}/20, "
        f"Vector: {risk['vector_score']}/15. "
        f"{risk_rationale.get(rl, '')}"
    )

    # Recommended actions
    actions_by_level = {
        "Low": [
            "Continue standard 7-day syndromic surveillance in the affected postal area.",
            "Advise community reporters to monitor for symptom progression and document animal contacts.",
            "No cross-sector escalation required at this time.",
        ],
        "Medium": [
            "Notify Arizona Department of Health Services (ADHS) and AZ Dept of Agriculture of concurrent human-animal signal co-elevation.",
            "Deploy field interview team to identify potential exposure locations (farms, parks, flood channels).",
            "Collect paired serum samples from febrile human cases for arboviral and zoonotic antibody panel.",
            "Activate enhanced passive surveillance for tick bites and animal contact histories.",
        ],
        "High": [
            "Convene One Health rapid response team (ADHS, AZ Game and Fish, USDA APHIS, Maricopa/Pima Vector Control).",
            "Conduct targeted environmental sampling (soil, water, animal carcasses) in the affected postal area.",
            "Issue community advisory on avoidance of rodent contact, tick bite prevention, and reporting of sick animals.",
            "Coordinate with local veterinarians to increase diagnostic testing of livestock and companion animals.",
            "Implement wastewater-enhanced surveillance with weekly sampling frequency.",
        ],
        "Critical": [
            "Immediately activate Arizona Emergency Support Function #8 (ESF-8) and notify CDC Epidemiology Response.",
            "Establish a unified incident command structure integrating human, veterinary, and environmental health sectors.",
            "Deploy mobile laboratory capacity for on-site PCR and serological testing.",
            "Issue public health advisory through AZ Health Alert Network (AZAN) and media channels.",
            "Implement contact tracing for all confirmed cases and their animal/environmental exposures.",
            "Coordinate with USDA APHIS for emergency livestock quarantine and carcass management protocols.",
        ],
    }
    actions = actions_by_level.get(rl, actions_by_level["Medium"])
    para4 = f"\n\n(4) RECOMMENDED CROSS-SECTOR RESPONSE ACTIONS\n"
    for i, action in enumerate(actions, 1):
        para4 += f"\n  {i}. {action}"

    word_count_note = "\n\n[Narrative word count target: 400–500 words | Generated by rule-based engine]"
    full_narrative = para1 + para2 + para3 + para4 + word_count_note
    return full_narrative


# ============================================================
# XI. 4-PANEL DASHBOARD (PNG)
# ============================================================
def fig_dashboard(human: dict, animal: dict, env: dict, vector: dict,
                  risk: dict, pathogen_df: pd.DataFrame, out_dir: Path) -> None:
    """
    4-panel dashboard:
      A. Risk score radar / bar chart (four stream components)
      B. Symptom-pathogen concordance chart
      C. Four-stream summary scorecard (annotated)
      D. One Health risk level meter + cross-sector action summary
    """
    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    fig.suptitle(
        f"EpiHack One Health Risk Intelligence — Postal {human['postal']} "
        f"({human['county']} County)\n"
        f"Arizona Participatory Surveillance | {human['rec_start']} to {human['rec_end']}",
        fontsize=13, fontweight="bold", color=TEXT, y=0.98
    )
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.06, right=0.97, top=0.92, bottom=0.06,
                           hspace=0.38, wspace=0.32)

    # ---- Panel A: Risk score breakdown bars ----
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor(SURFACE)

    streams = ["Human\nSurveillance", "Animal\nReports",
               "Environmental\nConditions", "Vector\nActivity"]
    scores  = [risk["human_score"], risk["animal_score"],
               risk["env_score"],   risk["vector_score"]]
    maxes   = [40, 25, 20, 15]
    colors  = [ACCENT, ORANGE, TEAL, YELLOW]

    x = np.arange(len(streams))
    bars = ax_a.bar(x, scores, color=[c + "cc" for c in colors],
                    edgecolor="none", width=0.6, zorder=3)
    # Max capacity ghost bars
    ax_a.bar(x, maxes, color=[c + "22" for c in colors],
             edgecolor=[c + "55" for c in colors], width=0.6, linewidth=1, zorder=2)

    for i, (score, mx) in enumerate(zip(scores, maxes)):
        ax_a.text(i, score + 0.5, f"{score}/{mx}", ha="center",
                  fontsize=10, fontweight="bold", color=TEXT)

    total_score = risk["total"]
    risk_color  = risk["risk_color"]
    ax_a.text(0.5, 0.97,
              f"Total: {total_score}/100 — {risk['risk_level'].upper()}",
              transform=ax_a.transAxes, ha="center", va="top",
              fontsize=11, fontweight="bold", color=risk_color)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(streams, fontsize=8.5, color=TEXT)
    ax_a.set_ylabel("Component Score", fontsize=8, color=MUTED)
    ax_a.set_title("A — One Health Risk Score Decomposition",
                   fontsize=9, fontweight="bold", color=ORANGE, pad=8)
    ax_a.tick_params(colors=MUTED, labelsize=8)
    ax_a.spines[:].set_visible(False)
    ax_a.set_ylim(0, max(maxes) * 1.25)
    ax_a.yaxis.grid(True, color="#1d2035", zorder=1)

    # ---- Panel B: Pathogen concordance horizontal bars ----
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor(SURFACE)

    top_n = pathogen_df.head(8).iloc[::-1]
    bar_colors_b = []
    for pct in top_n["concordance_pct"]:
        if pct >= 60:   bar_colors_b.append(RED + "cc")
        elif pct >= 40: bar_colors_b.append(ORANGE + "cc")
        elif pct >= 20: bar_colors_b.append(YELLOW + "cc")
        else:           bar_colors_b.append(ACCENT + "66")

    ax_b.barh(range(len(top_n)), top_n["concordance_pct"],
              color=bar_colors_b, edgecolor="none", height=0.65)
    ax_b.axvline(40, color=ORANGE, linestyle="--", linewidth=1, alpha=0.7,
                 label="Moderate concordance (40%)")
    ax_b.axvline(60, color=RED, linestyle="--", linewidth=1, alpha=0.7,
                 label="High concordance (60%)")

    for i, (_, r) in enumerate(top_n.iterrows()):
        if r["matched_cols"]:
            ax_b.text(r["concordance_pct"] + 0.5, i,
                      f"{r['concordance_pct']:.0f}%",
                      va="center", fontsize=8, color=TEXT, fontweight="bold")

    ax_b.set_yticks(range(len(top_n)))
    ax_b.set_yticklabels(top_n["pathogen"], fontsize=7.5, color=TEXT)
    ax_b.set_xlabel("Concordance with Observed Symptom Profile (%)", fontsize=8, color=MUTED)
    ax_b.legend(fontsize=7, facecolor=SURFACE2, edgecolor="none", labelcolor=TEXT)
    ax_b.set_title("B — Zoonotic Pathogen Concordance",
                   fontsize=9, fontweight="bold", color=RED, pad=8)
    ax_b.tick_params(colors=MUTED, labelsize=8)
    ax_b.spines[:].set_visible(False)
    ax_b.set_xlim(0, 110)

    # ---- Panel C: Four-stream summary scorecard (text-based table) ----
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor(SURFACE)
    ax_c.axis("off")
    ax_c.set_title("C — Multi-Stream Data Scorecard",
                   fontsize=9, fontweight="bold", color=TEAL, pad=8)

    card_lines = [
        f"POSTAL {human['postal']} | {human['county']} County | {env['ecological_zone']}",
        "",
        "  HUMAN SURVEILLANCE",
        f"    Elevated signals : {human['n_elevated']} ({human['n_new']} NEW, "
        f"{human['n_high_acuity']} high-acuity)",
        f"    Reporter pool    : n={human['n_rec_total']} (7-day window)",
        f"    Top symptoms     : {', '.join([v['label'] for v in human['elevated'][:3]]) or 'none'}",
        "",
        "  ANIMAL REPORTS (USDA APHIS / AZDA)",
        f"    Livestock cases  : {animal['livestock_cases']} "
        f"({', '.join(animal['livestock_species'][:2]) or 'none'})",
        f"    Wildlife die-off : {'YES — '+str(animal['wildlife_count'])+' '+str(animal['wildlife_species']) if animal['wildlife_dieoff'] else 'NO'}",
        f"    Companion animal : {animal['companion_cases']} cases",
        "",
        "  ENVIRONMENTAL (NWS / ADHS / NOAA)",
        f"    NWS HeatRisk     : {env['heat_risk']}/4 ({env['heat_risk_label']})",
        f"    Wastewater enteric: {env['wastewater_enteric']}",
        f"    Drought          : {env['drought_level']}",
        f"    Soil disturbance : {env['soil_disturbance']}",
        "",
        "  VECTOR ACTIVITY (ArboNET / Tick Check)",
        f"    Mosquito trap    : {vector['trap_count']}/trap-night "
        f"({'ALERT' if vector['mosquito_alert'] else 'normal'})",
        f"    WNV positive     : {vector['wnv_positive_pools']} pools "
        f"({vector['wnv_positivity_rate']:.1f}%)",
        f"    Tick pathogen+   : {'YES' if vector['tick_pathogen_positive'] else 'NO'} "
        f"(density: {vector['tick_density']})",
        f"    Rodent index     : {vector['rodent_index']}",
    ]
    ax_c.text(0.04, 0.97, "\n".join(card_lines),
              transform=ax_c.transAxes, fontsize=7.5, color=TEXT,
              va="top", ha="left", fontfamily="monospace",
              bbox=dict(facecolor=SURFACE2, edgecolor=TEAL + "55",
                        boxstyle="round,pad=0.5", alpha=0.85))

    # ---- Panel D: Risk meter + cross-sector actions ----
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor(SURFACE)
    ax_d.axis("off")
    ax_d.set_title("D — Risk Level & Cross-Sector Actions",
                   fontsize=9, fontweight="bold", color=risk["risk_color"], pad=8)

    # Draw a simple horizontal risk gauge
    gauge_ax = fig.add_axes([0.555, 0.09, 0.40, 0.04])
    gauge_ax.set_facecolor(SURFACE)
    gauge_ax.set_xlim(0, 100)
    gauge_ax.set_ylim(0, 1)
    gauge_ax.axis("off")

    segments = [(0, 25, GREEN), (25, 50, YELLOW), (50, 75, ORANGE), (75, 100, RED)]
    for start, end, col in segments:
        gauge_ax.barh(0.5, end - start, left=start, height=0.6,
                      color=col + "88", edgecolor="none")
    # Needle
    gauge_ax.axvline(min(risk["total"], 99), color="white", linewidth=3, zorder=5)
    gauge_ax.text(risk["total"], 0.95, f"{risk['total']}",
                  ha="center", va="top", fontsize=9, fontweight="bold",
                  color="white", zorder=6)
    gauge_ax.text(0, -0.3, "Low", ha="left", va="top",
                  fontsize=7.5, color=GREEN, transform=gauge_ax.transData)
    gauge_ax.text(25, -0.3, "Med", ha="left", va="top",
                  fontsize=7.5, color=YELLOW, transform=gauge_ax.transData)
    gauge_ax.text(50, -0.3, "High", ha="left", va="top",
                  fontsize=7.5, color=ORANGE, transform=gauge_ax.transData)
    gauge_ax.text(75, -0.3, "Critical", ha="left", va="top",
                  fontsize=7.5, color=RED, transform=gauge_ax.transData)

    action_labels = {
        "Low":     ["Standard 7-day syndromic surveillance",
                    "Community reporter monitoring",
                    "No cross-sector escalation required"],
        "Medium":  ["Notify ADHS + AZ Dept Agriculture",
                    "Field interview team deployment",
                    "Paired serum samples from febrile cases",
                    "Enhanced tick bite / animal contact history"],
        "High":    ["Convene One Health rapid response team",
                    "Environmental sampling (soil/water/carcasses)",
                    "Community advisory — animal contact avoidance",
                    "Veterinary diagnostic coordination",
                    "Weekly wastewater sampling"],
        "Critical":["Activate AZ ESF-8 + notify CDC Epi Response",
                    "Unified incident command (human/vet/env)",
                    "Deploy mobile laboratory capacity",
                    "Issue AZAN public health advisory",
                    "Contact tracing all confirmed cases",
                    "USDA livestock quarantine protocols"],
    }
    actions = action_labels.get(risk["risk_level"], [])
    action_text = (
        f"RISK LEVEL: {risk['risk_level'].upper()}\n"
        f"Score: {risk['total']}/100\n\n"
        "RECOMMENDED CROSS-SECTOR ACTIONS:\n"
    )
    for i, a in enumerate(actions, 1):
        action_text += f"\n  {i}. {a}"

    ax_d.text(0.04, 0.88, action_text,
              transform=ax_d.transAxes, fontsize=8.2, color=TEXT,
              va="top", ha="left", fontfamily="monospace",
              bbox=dict(facecolor=risk["risk_color"] + "18",
                        edgecolor=risk["risk_color"] + "55",
                        boxstyle="round,pad=0.5", alpha=0.9))

    out_path = out_dir / "one_health_risk_dashboard.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\n[9/9] Dashboard saved -> {out_path}")


# ============================================================
# XII. AUTO-SELECT FOCAL POSTAL CODE
# ============================================================
def select_focal_postal(df: pd.DataFrame, recent_days: int = 7,
                         baseline_days: int = 30) -> str:
    """
    Select the highest-burden postal code for One Health analysis:
    priority = n_elevated * 10 + n_new * 20 + n_high_acuity * 30
    """
    max_date  = df["date_of_report"].max()
    rec_start = max_date - pd.Timedelta(days=recent_days - 1)
    bas_end   = rec_start - pd.Timedelta(days=1)
    bas_start = bas_end - pd.Timedelta(days=baseline_days - 1)

    best_postal = None
    best_score  = -1

    for postal in df["postal_code"].dropna().unique():
        g_rec = df[(df["postal_code"] == postal) &
                   (df["date_of_report"] >= rec_start) &
                   (df["date_of_report"] <= max_date)]
        g_bas = df[(df["postal_code"] == postal) &
                   (df["date_of_report"] >= bas_start) &
                   (df["date_of_report"] <= bas_end)]
        if len(g_rec) < 5:
            continue
        n_elev = 0
        n_new  = 0
        n_ha   = 0
        for col, _ in SYMPTOM_COLS:
            if col not in df.columns:
                continue
            v_rec = g_rec[col].dropna()
            v_bas = g_bas[col].dropna()
            if len(v_rec) < 2 or len(v_bas) < 5:
                continue
            rr = float(v_rec.mean() * 100)
            br = float(v_bas.mean() * 100)
            if br == 0 and rr > 0:
                sig = "new"
            elif rr - br > 5:
                sig = "up"
            else:
                sig = "nc"
            if sig in ("up", "new"):
                n_elev += 1
                if sig == "new":
                    n_new += 1
                if col in HIGH_ACUITY:
                    n_ha += 1
        score = n_elev * 10 + n_new * 20 + n_ha * 30
        if score > best_score:
            best_score  = score
            best_postal = postal

    return str(best_postal) if best_postal else "85008"


# ============================================================
# XIII. MAIN
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="EpiHack One Health Risk Intelligence — User Story #7")
    p.add_argument("--db",       required=True,  help="Path to epihack-mini.db")
    p.add_argument("--postal",   default=None,   help="Focal postal code (auto-selected if omitted)")
    p.add_argument("--recent",   type=int, default=7,   help="Recent window (days)")
    p.add_argument("--baseline", type=int, default=30,  help="Baseline window (days)")
    p.add_argument("--out",      default="figures/",    help="Output directory")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = load_data(args.db)
    ref_date = df["date_of_report"].max()

    # 2. Select focal postal
    if args.postal:
        focal = args.postal
        print(f"[2/9] Focal postal code (user-specified): {focal}")
    else:
        print("[2/9] Auto-selecting focal postal code by burden score...")
        focal = select_focal_postal(df, args.recent, args.baseline)
        print(f"      Selected: {focal}")

    # 3. Human signals
    print("[3/9] Computing human surveillance signals...")
    human = compute_human_signals(df, focal, args.recent, args.baseline)
    print(f"      {human['n_elevated']} elevated | {human['n_new']} NEW | "
          f"{human['n_high_acuity']} high-acuity | n={human['n_rec_total']}")

    # 4-6. Simulate other streams
    print("[4/9] Simulating animal surveillance data (USDA APHIS / AZDA)...")
    animal = simulate_animal_reports(focal, human["county"], ref_date)

    print("[5/9] Simulating environmental conditions (NWS / ADHS / NOAA)...")
    env = simulate_environmental(focal, human["county"], ref_date)

    print("[6/9] Simulating vector activity (ArboNET / Tick Check)...")
    vector = simulate_vector_data(focal, human["county"], ref_date)

    # 7. Risk scoring
    print("[7/9] Computing zoonotic risk score...")
    risk = score_zoonotic_risk(human, animal, env, vector)
    print(f"      Score: {risk['total']}/100 → {risk['risk_level']}")
    print(f"      Human={risk['human_score']} | Animal={risk['animal_score']} | "
          f"Env={risk['env_score']} | Vector={risk['vector_score']}")

    # 7b. Pathogen concordance
    print("[7b/9] Computing symptom-pathogen concordance...")
    pathogen_df = compute_pathogen_correlation(human)

    # 8. Fill LLM prompt
    print("[8/9] Filling LLM prompt template and generating narrative...")
    prompt    = build_llm_prompt(human, animal, env, vector, risk, pathogen_df)
    narrative = simulate_llm_narrative(human, animal, env, vector, risk, pathogen_df)

    # Count words
    word_count = len(narrative.split())

    # Print
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  EpiHack -- One Health Risk Intelligence (User Story #7)")
    print(f"  Focal postal: {focal} | County: {human['county']} | "
          f"Zone: {env['ecological_zone']}")
    print(f"  Risk Level: {risk['risk_level']} ({risk['total']}/100)")
    print(sep)

    print(f"\n  TOP PATHOGEN CONCORDANCE:")
    for _, r in pathogen_df.head(4).iterrows():
        print(f"    {r['pathogen'].replace(chr(10),' '):<40} {r['concordance_pct']:.1f}%")

    print(f"\n  {'─'*68}\n  FILLED LLM PROMPT (excerpt):\n  {'─'*68}")
    print("  " + prompt[:600].replace("\n", "\n  "))

    print(f"\n  {'─'*68}\n  ONE HEALTH NARRATIVE ({word_count} words):\n  {'─'*68}")
    print(narrative)
    print(f"\n{sep}\n")

    # 9. Dashboard
    fig_dashboard(human, animal, env, vector, risk, pathogen_df, out_dir)
    print(f"\nDone. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
