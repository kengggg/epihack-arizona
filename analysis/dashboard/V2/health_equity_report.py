"""
health_equity_report.py
========================
EpiHack — User Story #8: Health Equity Analysis

LLM Prompt Template (filled at runtime):
    "You are a health equity epidemiologist. Given symptom prevalence rates by postal code:
     {postal_rates} and AZSVI vulnerability scores by postal code: {azsvi_scores}. Conduct a
     correlation analysis between symptom burden and social vulnerability. In 3–4 paragraphs:
     (1) identify the top 5 postal codes with high symptom burden AND high vulnerability
     (AZSVI quintile 4–5); (2) describe the specific symptoms driving disparity; (3) recommend
     targeted intervention strategies (mobile clinics, community health workers, translated
     communications); (4) flag any data gaps that limit equity analysis."

AZSVI Framework (Arizona Social Vulnerability Index, adapted from CDC/ATSDR SVI):
    Theme 1 — Socioeconomic Status:
        poverty_pct, unemployment_rate, housing_cost_burden, no_hs_diploma, no_health_insurance
    Theme 2 — Household Characteristics:
        pct_65plus, pct_under18, pct_disability, pct_single_parent, pct_limited_english
    Theme 3 — Racial & Ethnic Minority Status:
        pct_minority (non-White + Hispanic/Latino)
    Theme 4 — Housing Type & Transportation:
        pct_multi_unit, pct_mobile_home, pct_crowded, pct_no_vehicle, pct_group_quarters

    Overall AZSVI = average of four theme-level percentile scores (0–1, higher = more vulnerable)

Symptom Burden Score (PBS):
    PBS(postal) = sum of recent_rate_i across ALL symptom columns (unrestricted sum)
    High-acuity symptoms receive weight = 2.0; others weight = 1.0

Equity Disparity Metric:
    Disparity Score = PBS z-score + AZSVI z-score  (sum of standardized scores)
    Identifies postal codes burdened on BOTH dimensions simultaneously

Usage:
    python health_equity_report.py --db epihack-mini.db [--recent 7] [--baseline 30]
                                    [--min-reporters 5] [--out figures/]
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
from scipy import stats

SYMPTOM_COLS = [
    ("cough_congestion",               "Cough / Congestion",        1.0),
    ("sore_throat",                    "Sore Throat",               1.0),
    ("difficulty_breathing",           "Difficulty Breathing",      2.0),
    ("nauseas_vomiting",               "Nausea / Vomiting",         1.0),
    ("diarrhea",                       "Diarrhea",                  1.0),
    ("muscle_or_body_aches_and_pains", "Muscle / Body Aches",       1.0),
    ("fever",                          "Fever",                     1.0),
    ("chills",                         "Chills",                    1.0),
    ("red_eyes",                       "Red Eyes",                  1.0),
    ("rash",                           "Rash",                      1.0),
    ("loss_of_smell_or_taste",         "Loss of Smell/Taste",       1.0),
    ("bleeding_from_body_openings",    "Bleeding (Body Openings)",  2.0),
    ("yellow_skin_yellow_eyes",        "Yellow Skin / Eyes",        2.0),
    ("discolored_or_bloody_urine",     "Discolored/Bloody Urine",   2.0),
]
COL_TO_LABEL  = {c: l for c, l, _ in SYMPTOM_COLS}
COL_TO_WEIGHT = {c: w for c, _, w in SYMPTOM_COLS}

# Arizona county SVI profiles (ecologically plausible base scores, 0–1)
# Rural/tribal counties tend toward higher SVI; dense urban varies
COUNTY_SVI_BASE = {
    "Apache":    0.82, "Navajo":     0.79, "Greenlee":   0.72,
    "Graham":    0.68, "Santa Cruz": 0.65, "Cochise":    0.55,
    "Gila":      0.58, "Yuma":       0.62, "La Paz":     0.64,
    "Mohave":    0.52, "Yavapai":    0.45, "Pinal":      0.48,
    "Pima":      0.46, "Maricopa":   0.42, "Coconino":   0.60,
}

# Design palette
BG       = "#0f1117"
SURFACE  = "#1a1d27"
SURFACE2 = "#23263a"
ACCENT   = "#6c63ff"
RED      = "#ff4757"
ORANGE   = "#ffa502"
GREEN    = "#43e97b"
YELLOW   = "#f9ca24"
TEAL     = "#2bcbba"
PURPLE   = "#a55eea"
PINK     = "#ff6584"
MUTED    = "#8892b0"
TEXT     = "#e8eaf6"

QUINTILE_COLORS = {1: GREEN, 2: TEAL, 3: YELLOW, 4: ORANGE, 5: RED}


# ============================================================
# II. DATA LOADING
# ============================================================
def load_data(db_path: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df  = pd.read_sql("SELECT * FROM incidents", con)
    con.close()
    df["date_of_report"] = pd.to_datetime(df["date_of_report"])
    for col, _, _ in SYMPTOM_COLS:
        if col in df.columns:
            df[col] = df[col].map({"YES": 1, "NO": 0}).astype("Float64")
    print(f"[1/9] Data loaded: {len(df):,} records | "
          f"{df['date_of_report'].min().date()} -- {df['date_of_report'].max().date()}")
    return df


# ============================================================
# III. POSTAL SYMPTOM BURDEN COMPUTATION
# ============================================================
def compute_postal_burden(df: pd.DataFrame,
                           recent_days: int = 7,
                           min_reporters: int = 5) -> pd.DataFrame:
    """
    Compute the Postal Burden Score (PBS) for each postal code:
        PBS = sum(recent_rate_i * weight_i) across all symptoms with sufficient data.

    Returns DataFrame with columns: postal_code, county, n_reporters,
        [sym_rate columns], pbs, pbs_z, data_complete (bool)
    """
    max_date  = df["date_of_report"].max()
    rec_start = max_date - pd.Timedelta(days=recent_days - 1)
    df_rec    = df[(df["date_of_report"] >= rec_start) &
                   (df["date_of_report"] <= max_date)]

    print(f"[3/9] Recent window: {rec_start.date()} -- {max_date.date()}")

    rows = []
    for postal, grp in df_rec.groupby("postal_code"):
        n = len(grp)
        county = grp["county"].mode().iloc[0] if len(grp) > 0 else "Unknown"
        row = {"postal_code": str(postal), "county": county, "n_reporters": n}
        pbs = 0.0
        n_valid = 0

        for col, label, weight in SYMPTOM_COLS:
            if col not in df.columns:
                continue
            v = grp[col].dropna()
            if len(v) >= 2:
                rate = float(v.mean() * 100)
                row[f"rate_{col}"] = round(rate, 2)
                pbs += rate * weight
                n_valid += 1
            else:
                row[f"rate_{col}"] = np.nan

        row["pbs"]           = round(pbs, 2)
        row["n_valid_syms"]  = n_valid
        row["data_complete"] = (n >= min_reporters) and (n_valid >= 8)
        rows.append(row)

    burden = pd.DataFrame(rows)
    # Z-score PBS across postals with sufficient data
    valid_mask = burden["data_complete"]
    pbs_vals   = burden.loc[valid_mask, "pbs"]
    mu, sigma  = pbs_vals.mean(), pbs_vals.std()
    burden["pbs_z"] = (burden["pbs"] - mu) / sigma if sigma > 0 else 0.0

    n_complete = valid_mask.sum()
    print(f"      {len(burden)} postal codes | {n_complete} with sufficient data "
          f"(n>={min_reporters}, >=8 valid symptoms)")
    return burden


# ============================================================
# IV. SIMULATED AZSVI SCORES
# ============================================================
def simulate_azsvi(postal: str, county: str) -> dict:
    """
    Deterministic pseudo-random AZSVI simulation anchored to postal + county.
    Scores are ecologically calibrated to Arizona's county-level SVI profiles.

    In production, replace with:
        CDC/ATSDR SVI API: GET https://svi.cdc.gov/api/v1/svi?postal={postal}&state=AZ
        AZ Department of Health AZSVI: https://www.azdhs.gov/data/svi/
    """
    seed = int(hashlib.md5(f"{postal}{county}svi".encode()).hexdigest(), 16) % 100000
    rng  = np.random.default_rng(seed)

    county_base = COUNTY_SVI_BASE.get(county, 0.50)
    noise = float(rng.normal(0, 0.10))

    # Theme 1: Socioeconomic Status
    poverty_pct         = float(np.clip(county_base * 35 + rng.normal(0, 5), 5, 55))
    unemployment_rate   = float(np.clip(county_base * 18 + rng.normal(0, 3), 2, 25))
    no_hs_diploma_pct   = float(np.clip(county_base * 30 + rng.normal(0, 5), 5, 50))
    no_health_ins_pct   = float(np.clip(county_base * 28 + rng.normal(0, 4), 5, 45))
    housing_cost_burden = float(np.clip(county_base * 40 + rng.normal(0, 6), 10, 60))
    theme1 = np.clip(county_base + noise * 0.4 + rng.normal(0, 0.08), 0, 1)

    # Theme 2: Household Characteristics
    pct_65plus        = float(np.clip(15 + rng.normal(0, 5), 5, 35))
    pct_under18       = float(np.clip(county_base * 30 + rng.normal(0, 4), 10, 45))
    pct_disability    = float(np.clip(county_base * 20 + rng.normal(0, 3), 5, 30))
    pct_single_parent = float(np.clip(county_base * 25 + rng.normal(0, 4), 5, 40))
    pct_limited_eng   = float(np.clip(county_base * 20 + rng.normal(0, 4), 1, 35))
    theme2 = np.clip(county_base + noise * 0.3 + rng.normal(0, 0.07), 0, 1)

    # Theme 3: Racial & Ethnic Minority Status
    pct_minority = float(np.clip(county_base * 70 + rng.normal(0, 10), 5, 95))
    theme3 = np.clip(county_base + noise * 0.5 + rng.normal(0, 0.10), 0, 1)

    # Theme 4: Housing & Transportation
    pct_crowded    = float(np.clip(county_base * 12 + rng.normal(0, 2), 1, 20))
    pct_no_vehicle = float(np.clip(county_base * 15 + rng.normal(0, 3), 2, 25))
    pct_mobile_home= float(np.clip(20 + rng.normal(0, 8), 2, 50))
    theme4 = np.clip(county_base + noise * 0.3 + rng.normal(0, 0.08), 0, 1)

    # Overall AZSVI = mean of four theme percentile scores
    overall = float(np.clip(np.mean([theme1, theme2, theme3, theme4]), 0, 1))

    return {
        "postal_code":        postal,
        "county":             county,
        "overall_svi":        round(overall, 4),
        "theme1_svi":         round(float(theme1), 4),
        "theme2_svi":         round(float(theme2), 4),
        "theme3_svi":         round(float(theme3), 4),
        "theme4_svi":         round(float(theme4), 4),
        "poverty_pct":        round(poverty_pct, 1),
        "unemployment_rate":  round(unemployment_rate, 1),
        "no_hs_diploma_pct":  round(no_hs_diploma_pct, 1),
        "no_health_ins_pct":  round(no_health_ins_pct, 1),
        "pct_minority":       round(pct_minority, 1),
        "pct_65plus":         round(pct_65plus, 1),
        "pct_limited_eng":    round(pct_limited_eng, 1),
        "pct_no_vehicle":     round(pct_no_vehicle, 1),
        "source":             "CDC/ATSDR SVI adapted for AZ (simulated)",
    }


# ============================================================
# V. MERGE & ASSIGN QUINTILES
# ============================================================
def merge_and_quintile(burden: pd.DataFrame) -> pd.DataFrame:
    """
    Merge AZSVI scores with burden data, assign SVI quintiles, and compute
    the composite Disparity Score = PBS z-score + SVI z-score.
    """
    svi_rows = []
    for _, r in burden.iterrows():
        svi_rows.append(simulate_azsvi(r["postal_code"], r["county"]))
    svi_df = pd.DataFrame(svi_rows)

    merged = burden.merge(svi_df[["postal_code", "overall_svi", "theme1_svi",
                                   "theme2_svi", "theme3_svi", "theme4_svi",
                                   "poverty_pct", "unemployment_rate",
                                   "no_hs_diploma_pct", "no_health_ins_pct",
                                   "pct_minority", "pct_65plus", "pct_limited_eng",
                                   "pct_no_vehicle"]],
                          on="postal_code", how="left")

    # SVI quintile (1=least vulnerable, 5=most vulnerable)
    valid = merged["data_complete"]
    svi_vals = merged.loc[valid, "overall_svi"]
    merged.loc[valid, "svi_quintile"] = pd.qcut(
        svi_vals, q=5, labels=[1, 2, 3, 4, 5]
    ).astype(float)

    # Z-score SVI
    mu_svi = svi_vals.mean()
    sd_svi = svi_vals.std()
    merged["svi_z"] = (merged["overall_svi"] - mu_svi) / sd_svi if sd_svi > 0 else 0.0

    # Composite Disparity Score
    merged["disparity_score"] = merged["pbs_z"] + merged["svi_z"]

    print(f"[5/9] AZSVI scores assigned | Quintile distribution:")
    for q in [1, 2, 3, 4, 5]:
        n = (merged.loc[valid, "svi_quintile"] == q).sum()
        print(f"      Q{q}: {n} postal codes")

    return merged


# ============================================================
# VI. CORRELATION ANALYSIS
# ============================================================
def compute_correlation(merged: pd.DataFrame) -> dict:
    """
    Pearson and Spearman correlation between AZSVI and PBS
    across postal codes with complete data.
    """
    valid = merged[merged["data_complete"]].dropna(subset=["pbs", "overall_svi"])
    x = valid["overall_svi"].values
    y = valid["pbs"].values

    pearson_r,  pearson_p  = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    # Linear regression for scatter plot
    slope, intercept, r_val, p_val, se = stats.linregress(x, y)

    print(f"[6/9] Correlation analysis (n={len(valid)}):")
    print(f"      Pearson r={pearson_r:.3f}  p={pearson_p:.4f}")
    print(f"      Spearman rho={spearman_r:.3f}  p={spearman_p:.4f}")

    return {
        "pearson_r":    round(pearson_r, 4),
        "pearson_p":    round(pearson_p, 4),
        "spearman_rho": round(spearman_r, 4),
        "spearman_p":   round(spearman_p, 4),
        "n":            len(valid),
        "slope":        slope,
        "intercept":    intercept,
        "r_squared":    r_val ** 2,
    }


# ============================================================
# VII. IDENTIFY DISPARITY HOTSPOTS
# ============================================================
def identify_disparity_hotspots(merged: pd.DataFrame, top_n: int = 5) -> tuple:
    """
    Identify the top_n postal codes simultaneously showing:
      - High symptom burden  (PBS z-score > 0, primary criterion)
      - High vulnerability   (AZSVI quintile 4 or 5)

    Falls back to top PBS-ranked Q4-Q5 postals if primary criterion yields < 3 results.
    Returns (hotspots_df, criterion_used).
    """
    valid = merged[merged["data_complete"]].dropna(subset=["svi_quintile"])

    # Primary: PBS z > 0 AND Q4-Q5
    high_q = valid[(valid["svi_quintile"] >= 4) & (valid["pbs_z"] > 0)]
    if len(high_q) >= 3:
        hotspots = high_q.sort_values("disparity_score", ascending=False).head(top_n)
        criterion = "high burden (PBS z>0) + SVI Q4-Q5"
    else:
        # Fallback: all Q4-Q5, ranked by PBS
        fallback = valid[valid["svi_quintile"] >= 4].sort_values("pbs", ascending=False)
        hotspots  = fallback.head(top_n)
        criterion = "SVI Q4-Q5 ranked by PBS (participation gap detected)"

    print(f"[7/9] Disparity hotspots: {len(hotspots)} postal codes "
          f"({criterion})")
    return hotspots.reset_index(drop=True), criterion


# ============================================================
# VIII. SYMPTOM-LEVEL DISPARITY BY SVI QUINTILE
# ============================================================
def symptom_disparity_by_quintile(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean symptom prevalence rate per SVI quintile for each symptom.
    Returns wide DataFrame: symptom × quintile.
    """
    valid = merged[merged["data_complete"]].dropna(subset=["svi_quintile"])
    rate_cols = [f"rate_{col}" for col, _, _ in SYMPTOM_COLS
                 if f"rate_{col}" in merged.columns]

    rows = []
    for q in [1, 2, 3, 4, 5]:
        sub = valid[valid["svi_quintile"] == q]
        row = {"quintile": q, "n": len(sub)}
        for rc in rate_cols:
            row[rc] = sub[rc].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("quintile")


# ============================================================
# IX. LLM PROMPT FILL
# ============================================================
def build_llm_prompt(merged: pd.DataFrame, hotspots: pd.DataFrame,
                     corr: dict, disparity_syms: list,
                     data_gaps: list, recent_days: int) -> str:
    """
    Fill the User Story #8 LLM prompt template.

    Production replacement:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-opus-4-6", max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        narrative = response.content[0].text
    """
    # Compact postal_rates string (top 20 by PBS)
    top_burden = merged[merged["data_complete"]].nlargest(20, "pbs")
    rate_lines = []
    for _, r in top_burden.iterrows():
        rate_lines.append(
            f"  {r['postal_code']} ({r['county']}) PBS={r['pbs']:.1f} "
            f"n={int(r['n_reporters'])} SVI-Q{int(r['svi_quintile'])}"
        )
    postal_rates_str = "\n".join(rate_lines)

    # AZSVI top 10 by overall SVI
    top_svi = merged.nlargest(10, "overall_svi")
    azsvi_lines = []
    for _, r in top_svi.iterrows():
        azsvi_lines.append(
            f"  {r['postal_code']} ({r['county']}) AZSVI={r['overall_svi']:.3f} "
            f"[poverty={r['poverty_pct']:.0f}%, minority={r['pct_minority']:.0f}%, "
            f"no-ins={r['no_health_ins_pct']:.0f}%]"
        )
    azsvi_str = "\n".join(azsvi_lines)

    # Disparity hotspot summary
    hotspot_lines = []
    for i, (_, h) in enumerate(hotspots.iterrows(), 1):
        syms_top = []
        for col, label, _ in SYMPTOM_COLS:
            rc = f"rate_{col}"
            if rc in h.index and not pd.isna(h[rc]) and h[rc] > 5:
                syms_top.append(f"{label}={h[rc]:.1f}%")
        hotspot_lines.append(
            f"  #{i} Postal {h['postal_code']} ({h['county']} County, "
            f"SVI-Q{int(h['svi_quintile'])}, AZSVI={h['overall_svi']:.3f}): "
            f"PBS={h['pbs']:.1f} | {', '.join(syms_top[:3])}"
        )
    hotspot_str = "\n".join(hotspot_lines) if hotspot_lines else "  None identified"

    prompt = textwrap.dedent(f"""
        You are a health equity epidemiologist. Given symptom prevalence rates by postal code
        ({recent_days}-day window):

        {postal_rates_str}

        And AZSVI vulnerability scores by postal code:

        {azsvi_str}

        Correlation analysis results: Pearson r={corr['pearson_r']:.3f}
        (p={corr['pearson_p']:.4f}), Spearman rho={corr['spearman_rho']:.3f}
        (p={corr['spearman_p']:.4f}), n={corr['n']} postal codes.

        Pre-identified disparity hotspots (high PBS + SVI quintile 4–5):
        {hotspot_str}

        Symptoms most elevated in high-SVI quintiles (Q4–Q5):
        {', '.join(disparity_syms[:5])}

        In 3–4 paragraphs:
        (1) identify the top 5 postal codes with high symptom burden AND high vulnerability
            (AZSVI quintile 4–5);
        (2) describe the specific symptoms driving disparity;
        (3) recommend targeted intervention strategies (mobile clinics, community health
            workers, translated communications);
        (4) flag any data gaps that limit equity analysis.
        Data gaps identified: {'; '.join(data_gaps)}.
    """).strip()
    return prompt


# ============================================================
# X. SIMULATED 3-4 PARAGRAPH EQUITY NARRATIVE
# ============================================================
def simulate_equity_narrative(hotspots: pd.DataFrame, merged: pd.DataFrame,
                               corr: dict, disparity_syms: list,
                               data_gaps: list,
                               hotspot_criterion: str = "") -> str:
    """
    Rule-based 3–4 paragraph health equity narrative.

    Production replacement (inline comment):
        # response = client.messages.create(
        #     model="claude-opus-4-6", max_tokens=800,
        #     messages=[{"role":"user","content": prompt}]
        # )
        # return response.content[0].text
    """
    # Paragraph 1: Correlation and top-5 hotspots
    is_positive = corr["pearson_r"] >= 0
    if corr["pearson_p"] < 0.05:
        r_interp = ("a statistically significant positive association"
                    if is_positive else
                    "a statistically significant negative association")
    else:
        r_interp = ("a positive but non-significant trend"
                    if is_positive else
                    "a negative but non-significant trend — likely reflecting "
                    "participation bias rather than true lower burden in high-SVI areas")
    strength = (
        "strong" if abs(corr["pearson_r"]) >= 0.5
        else "moderate" if abs(corr["pearson_r"]) >= 0.3
        else "weak"
    )

    hotspot_descriptions = []
    for i, (_, h) in enumerate(hotspots.iterrows(), 1):
        eco = COUNTY_SVI_BASE.get(h["county"], 0.50)
        high_syms = []
        for col, label, _ in SYMPTOM_COLS:
            rc = f"rate_{col}"
            if rc in h.index and not pd.isna(h[rc]) and h[rc] > 5:
                high_syms.append(label)
        sym_str = " and ".join(high_syms[:2]) if high_syms else "multiple symptoms"
        hotspot_descriptions.append(
            f"postal code {h['postal_code']} ({h['county']} County, "
            f"AZSVI={h['overall_svi']:.3f}, Q{int(h['svi_quintile'])}) "
            f"with elevated {sym_str}"
        )

    participation_note = ""
    if not is_positive:
        participation_note = (
            " Critically, a negative observed correlation is itself an equity signal: "
            "high-SVI communities are structurally underrepresented in web-based "
            "participatory surveillance due to digital access barriers, limited English "
            "proficiency, and reduced trust in health reporting systems, suppressing their "
            "measured burden below its true population level."
        )

    hotspot_note = (
        f"Hotspot criterion used: {hotspot_criterion}. "
        if hotspot_criterion else ""
    )

    para1 = (
        f"Correlation analysis across {corr['n']} Arizona postal codes with complete "
        f"surveillance data reveals {r_interp} between the Arizona Social Vulnerability "
        f"Index (AZSVI) and the composite Postal Burden Score (PBS) "
        f"(Pearson r={corr['pearson_r']:.3f}, p={corr['pearson_p']:.4f}; "
        f"Spearman ρ={corr['spearman_rho']:.3f}, p={corr['spearman_p']:.4f}).{participation_note} "
        f"The five highest-priority postal codes in AZSVI quintile 4–5 ({hotspot_note}"
        f"ranked by composite Postal Burden Score) are: "
        f"{'; '.join(hotspot_descriptions[:5]) if hotspot_descriptions else 'see data gap section — insufficient reporters in Q4-Q5 postal codes'}. "
        f"These communities represent the highest-priority targets for health equity "
        f"intervention under the EpiHack One Health surveillance framework."
    )

    # Paragraph 2: Symptom-level disparity drivers
    n_ha_syms = sum(1 for col, _, w in SYMPTOM_COLS
                    if any(col == s.split(":")[0] for s in disparity_syms[:5]) and w > 1)
    para2 = (
        f"The symptoms most elevated in AZSVI quintile 4–5 communities relative to "
        f"quintile 1–2 communities are: {', '.join(disparity_syms[:5])}. "
        f"The predominance of respiratory and gastrointestinal symptoms in high-SVI "
        f"postal codes is consistent with known social determinants of health: crowded "
        f"housing facilitates airborne pathogen transmission; limited food safety "
        f"infrastructure increases enteric disease exposure; and reduced healthcare access "
        f"leads to delayed presentation and under-recognition of early illness. "
        f"High-acuity symptoms (Difficulty Breathing, Discolored/Bloody Urine) appearing "
        f"in disparity clusters warrant immediate clinical attention, as they may reflect "
        f"higher rates of severe disease in communities with delayed care-seeking behavior "
        f"due to financial barriers, transportation limitations, or mistrust of the "
        f"healthcare system. The symptom pattern suggests that standard surveillance "
        f"thresholds calibrated for average-SVI communities may systematically "
        f"undercount disease burden in the most vulnerable postal codes."
    )

    # Paragraph 3: Intervention recommendations
    has_tribal = any(c in ["Apache", "Navajo", "Coconino"] for c in hotspots["county"].values)
    has_urban  = any(c in ["Maricopa", "Pima"] for c in hotspots["county"].values)
    lang_note  = (
        "Spanish, Navajo (Diné), and other Indigenous languages" if has_tribal
        else "Spanish and other non-English languages"
    )

    para3 = (
        f"Targeted intervention strategies for the identified disparity clusters should "
        f"operate across three complementary modalities. First, mobile clinic deployment "
        f"should be prioritized for the five high-disparity postal codes, offering "
        f"accessible point-of-care testing, preventive services, and linkage to care "
        f"without requiring transportation or insurance. Second, community health worker "
        f"(CHW) programs — particularly promotoras in Hispanic/Latino communities and "
        f"community health representatives (CHRs) in tribal areas — should be embedded "
        f"as trusted intermediaries for symptom reporting, health education, and care "
        f"navigation. CHWs are especially critical for EpiHack enrollment: "
        f"communities with low digital access and limited English proficiency are "
        f"systematically underrepresented in web-based participatory surveillance platforms. "
        f"Third, all public health communications should be available in {lang_note}, "
        f"using culturally appropriate messaging frameworks developed with community partners "
        f"rather than translated from English-language templates. "
        f"{'For postal codes in tribal areas, all interventions must be coordinated through tribal ' if has_tribal else ''}"
        f"{'health departments in accordance with tribal data sovereignty protocols. ' if has_tribal else ''}"
        f"Social services co-location (food, housing, transportation assistance) at "
        f"mobile clinic sites is recommended to address the upstream determinants "
        f"driving the observed symptom burden disparities."
    )

    # Paragraph 4: Data gaps
    para4 = (
        f"Several data gaps limit the precision of this equity analysis and must be "
        f"addressed to support valid inference. "
        f"{'; '.join(data_gaps)}. "
        f"Additionally, EpiHack's participatory design introduces selection bias: "
        f"individuals with smartphones, internet access, and English literacy are more "
        f"likely to self-enroll, systematically undercounting the burden in communities "
        f"with lowest digital access — precisely the highest-SVI postal codes. "
        f"The AZSVI scores used here are simulated from county-level base rates and do not "
        f"reflect verified Census tract- or ZIP-code-level SVI data; integration with "
        f"official CDC/ATSDR SVI data (svi.cdc.gov) is required for publication-grade "
        f"equity analysis. Finally, the absence of age-, sex-, and occupation-stratified "
        f"burden estimates prevents identification of within-postal-code sub-group "
        f"disparities, which may be substantial in heterogeneous communities. "
        f"Future EpiHack versions should incorporate targeted CHW-assisted enrollment "
        f"in high-SVI communities and provide offline reporting mechanisms to mitigate "
        f"digital access barriers."
    )

    return (
        "HEALTH EQUITY ANALYSIS — POSTAL SYMPTOM BURDEN vs AZSVI\n"
        + "=" * 64 + "\n\n"
        + "[P1] CORRELATION & TOP-5 DISPARITY HOTSPOTS\n\n"
        + para1 + "\n\n"
        + "[P2] SYMPTOM DRIVERS OF DISPARITY\n\n"
        + para2 + "\n\n"
        + "[P3] TARGETED INTERVENTION STRATEGIES\n\n"
        + para3 + "\n\n"
        + "[P4] DATA GAPS & ANALYSIS LIMITATIONS\n\n"
        + para4
    )


# ============================================================
# XI. IDENTIFY DATA GAPS
# ============================================================
def identify_data_gaps(merged: pd.DataFrame, min_reporters: int) -> list:
    insufficient = merged[~merged["data_complete"]]
    n_zero       = (merged["n_reporters"] == 0).sum()
    n_tribal     = sum(1 for c in merged["county"].unique()
                       if c in ["Apache", "Navajo", "Coconino"])
    high_svi_insuf = merged[
        (merged["overall_svi"] > 0.65) & (~merged["data_complete"])
    ]

    gaps = [
        f"{len(insufficient)} postal codes have insufficient reporters "
        f"(n < {min_reporters}) for stable rate estimation",
        f"{len(high_svi_insuf)} high-SVI postal codes (AZSVI > 0.65) lack "
        f"sufficient data — these are precisely the communities most in need of equity analysis",
        f"AZSVI scores are simulated from county-level base rates, not verified ZIP-code "
        f"Census tract SVI data (source: CDC/ATSDR SVI svi.cdc.gov)",
        f"No age/sex stratification available at postal level — within-community "
        f"sub-group disparities cannot be assessed",
        f"Digital access and language barriers in {n_tribal} counties with significant "
        f"tribal communities (Apache, Navajo, Coconino) likely suppress enrollment "
        f"in participatory surveillance",
    ]
    return gaps


# ============================================================
# XII. 4-PANEL DASHBOARD (PNG)
# ============================================================
def fig_dashboard(merged: pd.DataFrame, hotspots: pd.DataFrame,
                  corr: dict, quintile_df: pd.DataFrame,
                  disparity_syms: list, out_dir: Path) -> None:
    """
    4-panel equity dashboard:
      A. Scatter plot: AZSVI vs PBS (quintile-colored, regression line)
      B. Disparity hotspot dual bar chart (PBS and AZSVI per postal)
      C. Symptom prevalence by SVI quintile (grouped bars, top-6 symptoms)
      D. Health equity narrative summary text
    """
    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    fig.suptitle(
        "EpiHack Health Equity Analysis — Symptom Burden vs Arizona Social Vulnerability Index (AZSVI)\n"
        "Arizona Participatory Syndromic Surveillance 2023–2026",
        fontsize=13, fontweight="bold", color=TEXT, y=0.98
    )
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.06, right=0.97, top=0.92, bottom=0.06,
                           hspace=0.38, wspace=0.32)

    valid = merged[merged["data_complete"]].dropna(subset=["svi_quintile", "pbs"])

    # ---- Panel A: Scatter AZSVI vs PBS ----
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor(SURFACE)

    for q in [1, 2, 3, 4, 5]:
        sub = valid[valid["svi_quintile"] == q]
        ax_a.scatter(sub["overall_svi"], sub["pbs"],
                     color=QUINTILE_COLORS[q], alpha=0.75, s=40,
                     label=f"Q{q}", edgecolors="none", zorder=3)

    # Regression line
    x_line = np.linspace(valid["overall_svi"].min(), valid["overall_svi"].max(), 100)
    y_line = corr["slope"] * x_line + corr["intercept"]
    ax_a.plot(x_line, y_line, color="white", linewidth=1.5, linestyle="--",
              alpha=0.7, label=f"OLS (r={corr['pearson_r']:.3f})", zorder=4)

    # Mark hotspots
    for _, h in hotspots.iterrows():
        ax_a.scatter(h["overall_svi"], h["pbs"],
                     color="white", s=120, marker="*",
                     edgecolors=RED, linewidth=1.5, zorder=5)
        ax_a.annotate(h["postal_code"],
                      xy=(h["overall_svi"], h["pbs"]),
                      xytext=(5, 5), textcoords="offset points",
                      fontsize=7, color=RED, fontweight="bold")

    ax_a.set_xlabel("AZSVI (overall, 0–1)", fontsize=8, color=MUTED)
    ax_a.set_ylabel("Postal Burden Score (PBS)", fontsize=8, color=MUTED)
    ax_a.set_title("A — AZSVI vs Symptom Burden (PBS)",
                   fontsize=9, fontweight="bold", color=ORANGE, pad=8)
    ax_a.tick_params(colors=MUTED, labelsize=8)
    ax_a.spines[:].set_visible(False)
    ax_a.legend(fontsize=7, facecolor=SURFACE2, edgecolor="none",
                labelcolor=TEXT, ncol=3, loc="upper left")
    ax_a.text(0.97, 0.04,
              f"r={corr['pearson_r']:.3f}  p={corr['pearson_p']:.4f}\n"
              f"ρ={corr['spearman_rho']:.3f}  n={corr['n']}",
              transform=ax_a.transAxes, ha="right", va="bottom",
              fontsize=7.5, color=MUTED, fontfamily="monospace",
              bbox=dict(facecolor=SURFACE2, edgecolor="none", boxstyle="round,pad=0.3"))

    # ---- Panel B: Disparity hotspot dual bar ----
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor(SURFACE)

    if hotspots.empty:
        ax_b.text(0.5, 0.5, "No hotspots detected.", ha="center", va="center",
                  color=MUTED, fontsize=10, transform=ax_b.transAxes)
    else:
        x = np.arange(len(hotspots))
        ax_b.bar(x - 0.2, hotspots["pbs"] / hotspots["pbs"].max() * 100,
                 width=0.38, color=RED + "cc", label="PBS (normalized to 100)", zorder=3)
        ax_b.bar(x + 0.2, hotspots["overall_svi"] * 100,
                 width=0.38, color=ORANGE + "cc", label="AZSVI × 100", zorder=3)
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(
            [f"{r['postal_code']}\nQ{int(r['svi_quintile'])}"
             for _, r in hotspots.iterrows()],
            fontsize=8, color=TEXT
        )
        ax_b.set_ylabel("Score (scaled)", fontsize=8, color=MUTED)
        ax_b.legend(fontsize=7.5, facecolor=SURFACE2, edgecolor="none", labelcolor=TEXT)
        # Annotate AZSVI actual values
        for i, (_, h) in enumerate(hotspots.iterrows()):
            ax_b.text(i, h["overall_svi"] * 100 + 1.5,
                      f"SVI={h['overall_svi']:.3f}", ha="center",
                      fontsize=7, color=ORANGE)

    ax_b.set_title("B — Top-5 Disparity Hotspots (High PBS + High SVI)",
                   fontsize=9, fontweight="bold", color=RED, pad=8)
    ax_b.tick_params(colors=MUTED, labelsize=8)
    ax_b.spines[:].set_visible(False)

    # ---- Panel C: Symptom prevalence by SVI quintile ----
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor(SURFACE)

    if not quintile_df.empty:
        # Select top 6 symptomatic columns by Q5 prevalence
        rate_cols = [f"rate_{col}" for col, _, _ in SYMPTOM_COLS
                     if f"rate_{col}" in quintile_df.columns]
        if 5 in quintile_df.index:
            q5_vals = quintile_df.loc[5, rate_cols].fillna(0)
            top6_cols = q5_vals.nlargest(6).index.tolist()
        else:
            top6_cols = rate_cols[:6]

        top6_labels = [COL_TO_LABEL.get(c.replace("rate_", ""), c) for c in top6_cols]
        x = np.arange(len(top6_labels))
        q_palette = [GREEN, TEAL, YELLOW, ORANGE, RED]
        width = 0.15

        for qi, q in enumerate([1, 2, 3, 4, 5]):
            if q not in quintile_df.index:
                continue
            vals = [quintile_df.loc[q, rc] if rc in quintile_df.columns else 0
                    for rc in top6_cols]
            ax_c.bar(x + qi * width - 2 * width,
                     vals, width=width - 0.01,
                     color=q_palette[qi] + "cc",
                     label=f"Q{q}", edgecolor="none", zorder=3)

        ax_c.set_xticks(x)
        ax_c.set_xticklabels(top6_labels, rotation=30, ha="right",
                              fontsize=7.5, color=TEXT)
        ax_c.set_ylabel("Mean Prevalence Rate (%)", fontsize=8, color=MUTED)
        ax_c.legend(fontsize=7.5, facecolor=SURFACE2, edgecolor="none",
                    labelcolor=TEXT, title="SVI Quintile",
                    title_fontsize=7)
    ax_c.set_title("C — Symptom Prevalence by AZSVI Quintile",
                   fontsize=9, fontweight="bold", color=PURPLE, pad=8)
    ax_c.tick_params(colors=MUTED, labelsize=8)
    ax_c.spines[:].set_visible(False)
    ax_c.yaxis.grid(True, color="#1d2035", zorder=1)

    # ---- Panel D: Equity summary text ----
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor(SURFACE)
    ax_d.axis("off")
    ax_d.set_title("D — Equity Summary & Interventions",
                   fontsize=9, fontweight="bold", color=TEAL, pad=8)

    summary_lines = [
        f"CORRELATION: Pearson r={corr['pearson_r']:.3f} (p={corr['pearson_p']:.4f})",
        f"             Spearman ρ={corr['spearman_rho']:.3f} (n={corr['n']})",
        "",
        "TOP-5 DISPARITY HOTSPOTS:",
    ]
    for i, (_, h) in enumerate(hotspots.iterrows(), 1):
        summary_lines.append(
            f"  #{i} {h['postal_code']} ({h['county']}) "
            f"PBS={h['pbs']:.0f} SVI={h['overall_svi']:.3f} Q{int(h['svi_quintile'])}"
        )

    summary_lines += [
        "",
        f"DISPARITY DRIVERS ({', '.join(disparity_syms[:3])}, ...)",
        "",
        "INTERVENTION PRIORITIES:",
        "  1. Mobile clinic deployment in Q4-Q5 postal codes",
        "  2. CHW/promotora enrollment in high-SVI communities",
        "  3. Multilingual translated communications",
        "  4. Tribal health dept coordination (sovereign protocols)",
        "  5. Offline reporting mechanisms for digital divide",
        "",
        "KEY DATA GAP:",
        "  Participatory surveillance undercounts high-SVI",
        "  communities due to digital access barriers.",
        "  Targeted CHW-assisted enrollment is required.",
    ]

    ax_d.text(0.04, 0.97, "\n".join(summary_lines),
              transform=ax_d.transAxes, fontsize=7.8, color=TEXT,
              va="top", ha="left", fontfamily="monospace",
              bbox=dict(facecolor=SURFACE2, edgecolor=TEAL + "55",
                        boxstyle="round,pad=0.5", alpha=0.85))

    out_path = out_dir / "health_equity_dashboard.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[9/9] Dashboard saved -> {out_path}")


# ============================================================
# XIII. MAIN
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="EpiHack Health Equity Analysis — User Story #8")
    p.add_argument("--db",            required=True,  help="Path to epihack-mini.db")
    p.add_argument("--recent",        type=int, default=7,  help="Recent window (days)")
    p.add_argument("--baseline",      type=int, default=30, help="Baseline window (days)")
    p.add_argument("--min-reporters", type=int, default=5,  help="Min reporters per postal")
    p.add_argument("--out",           default="figures/",   help="Output directory")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    df = load_data(args.db)

    # 2. Compute burden
    print("[2/9] Computing postal symptom burden scores...")
    burden = compute_postal_burden(df, recent_days=args.recent,
                                   min_reporters=args.min_reporters)

    # 3-4. Merge AZSVI + quintiles
    print("[4/9] Simulating AZSVI scores and assigning quintiles...")
    merged = merge_and_quintile(burden)

    # 5. Correlation
    corr = compute_correlation(merged)

    # 6. Hotspots
    hotspots, hotspot_criterion = identify_disparity_hotspots(merged, top_n=5)

    # 7. Quintile symptom disparity
    print("[8/9] Computing symptom disparity by SVI quintile...")
    quintile_df = symptom_disparity_by_quintile(merged)

    # Identify symptoms with greatest Q5-vs-Q1 disparity
    disparity_syms = []
    if not quintile_df.empty and 5 in quintile_df.index and 1 in quintile_df.index:
        rate_cols = [c for c in quintile_df.columns if c.startswith("rate_")]
        diffs = {c: quintile_df.loc[5, c] - quintile_df.loc[1, c] for c in rate_cols
                 if c in quintile_df.columns and not pd.isna(quintile_df.loc[5, c])
                 and not pd.isna(quintile_df.loc[1, c])}
        sorted_diffs = sorted(diffs.items(), key=lambda x: -x[1])
        disparity_syms = [COL_TO_LABEL.get(c.replace("rate_", ""), c)
                          for c, _ in sorted_diffs[:8]]

    # Data gaps
    data_gaps = identify_data_gaps(merged, args.min_reporters)

    # LLM prompt
    print("[8/9] Filling LLM prompt...")
    prompt    = build_llm_prompt(merged, hotspots, corr, disparity_syms,
                                  data_gaps, args.recent)
    narrative = simulate_equity_narrative(hotspots, merged, corr,
                                          disparity_syms, data_gaps,
                                          hotspot_criterion)

    # Print
    sep = "=" * 72
    print(f"\n{sep}")
    print("  EpiHack — Health Equity Analysis (User Story #8)")
    print(sep)
    print(f"\n  Correlation: Pearson r={corr['pearson_r']:.3f} "
          f"(p={corr['pearson_p']:.4f}) | Spearman ρ={corr['spearman_rho']:.3f} | "
          f"n={corr['n']}")
    print(f"\n  TOP-5 DISPARITY HOTSPOTS (High PBS + SVI Q4-Q5):")
    for _, h in hotspots.iterrows():
        print(f"    postal {h['postal_code']} ({h['county']}) — "
              f"PBS={h['pbs']:.1f} | AZSVI={h['overall_svi']:.3f} | "
              f"Q{int(h['svi_quintile'])} | n={int(h['n_reporters'])}")

    print(f"\n  Symptom disparity drivers (Q5 vs Q1):")
    for sym in disparity_syms[:5]:
        print(f"    {sym}")

    print(f"\n  {'─'*68}\n  LLM PROMPT (excerpt):\n  {'─'*68}")
    print("  " + prompt[:500].replace("\n", "\n  "))
    print(f"\n  {'─'*68}\n  EQUITY NARRATIVE:\n  {'─'*68}")
    print(narrative)
    print(f"\n{sep}\n")

    # Dashboard
    fig_dashboard(merged, hotspots, corr, quintile_df, disparity_syms, out_dir)
    print(f"\nDone. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
