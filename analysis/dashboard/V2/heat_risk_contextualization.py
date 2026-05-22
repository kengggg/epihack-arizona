"""
heat_risk_contextualization.py
================================
EpiHack — Heat-Related Risk Contextualization (User Story #4)
--------------------------------------------------------------
Pipeline:
  1. Load epihack-mini.db (SQLite)
  2. Simulate NWS HeatRisk index (0–4) for each record by date + Arizona county
       (production: replace with live NWS HeatRisk API or HEAT.AZ.gov endpoint)
  3. Select a participant with heat-relevant symptoms on a high-risk day
  4. Score each reported symptom for heat clinical relevance
  5. Fill the LLM Prompt Template:
       "The participant reported {symptoms} on {report_date}. The NWS HeatRisk
        index for {postal_code} on that date was {heat_risk_level} (scale: 0–4).
        In 2–3 sentences, explain the relationship between heat exposure and the
        reported symptoms, describe signs of heat illness requiring emergency care,
        and recommend protective behaviors. Cite the heat risk level explicitly."
  6. Simulate LLM narrative (rule-based; swap for API call in production)
  7. Render 4-panel dashboard → PNG

NWS HeatRisk scale
------------------
  0 = Little to No Risk    1 = Minor Risk
  2 = Moderate Risk        3 = Major Risk
  4 = Extreme Risk (dangerous heat conditions)

Production API integration
--------------------------
  Replace simulate_heat_risk() with:
      GET https://heat.gov/api/v1/heatrisk?lat={lat}&lon={lon}&date={date}
  or:
      GET https://api.weather.gov/gridpoints/{office}/{x},{y}/forecast
      (NWS gridpoint HeatRisk product)

Usage
-----
    python heat_risk_contextualization.py \\
        --db epihack-mini.db [--uid 195640] [--out figures/]

Requirements
------------
    pip install pandas matplotlib numpy
"""

import argparse
import sqlite3
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Column definitions ────────────────────────────────────────────────────────
SYMPTOM_COLS = [
    "cough_congestion", "nauseas_vomiting", "difficulty_breathing",
    "sore_throat", "rash", "fever", "chills", "diarrhea",
    "bleeding_from_body_openings", "red_eyes",
    "muscle_or_body_aches_and_pains", "discolored_or_bloody_urine",
    "loss_of_smell_or_taste", "yellow_skin_yellow_eyes",
]
SYMPTOM_LABELS = {
    "cough_congestion":              "Cough / Congestion",
    "nauseas_vomiting":              "Nausea / Vomiting",
    "difficulty_breathing":          "Difficulty Breathing",
    "sore_throat":                   "Sore Throat",
    "rash":                          "Rash",
    "fever":                         "Fever",
    "chills":                        "Chills",
    "diarrhea":                      "Diarrhea",
    "bleeding_from_body_openings":   "Bleeding (Body Openings)",
    "red_eyes":                      "Red Eyes",
    "muscle_or_body_aches_and_pains":"Muscle / Body Aches",
    "discolored_or_bloody_urine":    "Discolored/Bloody Urine",
    "loss_of_smell_or_taste":        "Loss of Smell / Taste",
    "yellow_skin_yellow_eyes":       "Yellow Skin / Eyes",
}

# Heat clinical relevance scores (0=none, 1=indirect, 2=moderate, 3=direct heat indicator)
HEAT_RELEVANCE = {
    "fever":                         3,  # core sign of heat exhaustion/stroke
    "nauseas_vomiting":              3,  # hallmark of heat exhaustion
    "muscle_or_body_aches_and_pains":2,  # heat cramps / exertional heat illness
    "diarrhea":                      2,  # dehydration consequence
    "difficulty_breathing":          2,  # severe heat stroke / exertional
    "chills":                        2,  # paradoxical chills in heat exhaustion
    "discolored_or_bloody_urine":    2,  # exertional rhabdomyolysis
    "red_eyes":                      1,  # UV / heat exposure
    "rash":                          1,  # heat rash (miliaria)
    "cough_congestion":              0,  # not heat-related
    "sore_throat":                   0,
    "bleeding_from_body_openings":   0,
    "loss_of_smell_or_taste":        0,
    "yellow_skin_yellow_eyes":       0,
}
HEAT_RELEVANCE_LABELS = {0: "Not heat-related", 1: "Possible heat link",
                          2: "Moderate heat association", 3: "Direct heat illness sign"}

# NWS HeatRisk levels
HEAT_RISK_LABELS  = {0:"Little to No Risk", 1:"Minor Risk", 2:"Moderate Risk",
                     3:"Major Risk",         4:"Extreme Risk"}
HEAT_RISK_COLORS  = {0:"#43e97b", 1:"#f9ca24", 2:"#ffa502", 3:"#ff6b35", 4:"#ff4757"}

# Arizona county thermal profiles (elevation-adjusted base heat index, 0–2 addend)
COUNTY_BASE_HEAT = {
    "Maricopa":2, "Pima":2, "Pinal":2, "Yuma":2, "La Paz":2, "Mohave":2,
    "Santa Cruz":1, "Yavapai":1, "Gila":1, "Graham":1, "Cochise":1,
    "Coconino":0, "Apache":0, "Navajo":0, "Greenlee":0,
}
# Monthly ambient risk contribution (Arizona climatology, 0–2)
MONTH_BASE_RISK = {1:0, 2:0, 3:1, 4:2, 5:3, 6:4, 7:4, 8:4, 9:3, 10:2, 11:1, 12:0}

# Emergency signs of heat illness (for narrative)
HEAT_EMERGENCY_SIGNS = [
    "body temperature above 104 °F (40 °C)",
    "hot, red, and dry or damp skin",
    "rapid, strong pulse",
    "confusion, slurred speech, or loss of consciousness",
    "difficulty breathing or seizures",
]

# Protective behaviors (tiered by risk level)
HEAT_BEHAVIORS = {
    0: ["Stay hydrated throughout the day",
        "Limit strenuous outdoor activity during peak afternoon hours",
        "Wear lightweight, light-colored, loose-fitting clothing"],
    1: ["Drink water regularly — do not wait until thirsty",
        "Seek shade or air-conditioned spaces during midday (10 am–4 pm)",
        "Wear a hat and apply sunscreen when outdoors"],
    2: ["Avoid outdoor exertion between 10 am and 6 pm",
        "Stay in air-conditioned spaces as much as possible",
        "Check on elderly neighbors, young children, and pets",
        "Keep cool with wet towels or fans if AC is unavailable"],
    3: ["Avoid all non-essential outdoor activity during daylight hours",
        "Stay in air-conditioned spaces — visit a cooling center if needed",
        "Wear moisture-wicking clothing and cool down with wet cloths",
        "Drink at least 8 oz of water or electrolyte fluid every 30 minutes during activity"],
    4: ["STAY INDOORS in air conditioning — outdoor exposure is dangerous",
        "If you must go outside, limit exposure to under 15 minutes and rest frequently",
        "Wear a moisture-wicking shirt, hat, and apply SPF 50+ sunscreen",
        "Drink 1 cup (8 oz) of water every 15–20 minutes during any outdoor activity",
        "Call 911 immediately if anyone shows signs of heat stroke"],
}

# ── Design tokens (heat-mode palette) ────────────────────────────────────────
BG      = "#0d0e14"
SURFACE = "#161820"
SURFACE2= "#0f1118"
HEAT4   = "#ff4757"   # extreme
HEAT3   = "#ff6b35"   # major
HEAT2   = "#ffa502"   # moderate
HEAT1   = "#f9ca24"   # minor
HEAT0   = "#43e97b"   # little/none
ACCENT  = "#6c63ff"
TEXT    = "#e8eaf6"
SUBTEXT = "#9e9eb0"


# ════════════════════════════════════════════════════════════════════════════
# I.  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_data(db_path: str) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    df  = pd.read_sql("SELECT * FROM incidents", con,
                      parse_dates=["date_of_report", "date_of_illness"])
    con.close()
    for col in SYMPTOM_COLS + ["no_symptoms", "symptoms"]:
        df[col] = df[col].map({"YES": 1, "NO": 0}).astype("Float64")
    return df


# ════════════════════════════════════════════════════════════════════════════
# II.  NWS HEATRISK SIMULATION
# ════════════════════════════════════════════════════════════════════════════

def simulate_heat_risk(date: pd.Timestamp, county: str) -> int:
    """
    Deterministic simulation of NWS HeatRisk (0–4) for an Arizona location.

    Logic:
      base = MONTH_BASE_RISK[month] + COUNTY_BASE_HEAT[county]
      daily_variation = deterministic ±1 derived from day-of-year (no randomness)
      result = clip(base + variation, 0, 4)

    Production replacement:
        import requests
        # NWS HeatRisk API (when available):
        r = requests.get(
            "https://heat.gov/api/v1/heatrisk",
            params={"lat": lat, "lon": lon, "date": date.strftime("%Y-%m-%d")}
        )
        return r.json()["heatrisk_level"]
        # Alternative: NWS gridpoint forecast API + parse HeatRisk field
    """
    month_risk   = MONTH_BASE_RISK.get(date.month, 0)
    county_add   = COUNTY_BASE_HEAT.get(county, 1)
    base         = month_risk + county_add
    # Deterministic daily variation: cycles ±1 with period ~3 days
    day_of_year  = date.timetuple().tm_yday
    variation    = (day_of_year * 17 + date.month * 7) % 3 - 1
    return int(np.clip(base + variation, 0, 4))


def add_heat_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized application of simulate_heat_risk to all rows."""
    df = df.copy()
    df["heat_risk"] = df.apply(
        lambda r: simulate_heat_risk(r["date_of_report"],
                                     r.get("county", "Maricopa")),
        axis=1
    )
    return df


# ════════════════════════════════════════════════════════════════════════════
# III.  PARTICIPANT SELECTION
# ════════════════════════════════════════════════════════════════════════════

def get_participant(df: pd.DataFrame, uid: int) -> pd.Series:
    m = df[df["unique_id"] == uid]
    if m.empty:
        raise ValueError(f"unique_id {uid} not found.")
    return m.iloc[0]


def default_participant(df: pd.DataFrame) -> pd.Series:
    """
    Auto-select: participant with highest heat risk + most heat-relevant symptoms,
    preferring summer months (June–September, Arizona heat season).
    """
    df = df.copy()
    df["heat_score"] = df.apply(
        lambda r: sum(HEAT_RELEVANCE.get(c, 0) * (1 if r[c] == 1 else 0)
                      for c in SYMPTOM_COLS),
        axis=1
    )
    # Weight heat risk AND number of heat-relevant symptoms
    df["priority"] = df["heat_risk"] * 10 + df["heat_score"]
    summer = df[df["date_of_report"].dt.month.isin([6, 7, 8, 9])]
    pool   = summer if not summer.empty else df
    return pool.sort_values("priority", ascending=False).iloc[0]


# ════════════════════════════════════════════════════════════════════════════
# IV.  HEAT RELEVANCE SCORING
# ════════════════════════════════════════════════════════════════════════════

def score_symptoms(participant: pd.Series) -> pd.DataFrame:
    """
    Return a DataFrame with heat relevance score for each reported symptom.
    """
    rows = []
    for col in SYMPTOM_COLS:
        val = participant.get(col, None)
        rows.append({
            "symptom"    : col,
            "label"      : SYMPTOM_LABELS[col],
            "reported"   : int(val) if pd.notna(val) else None,
            "relevance"  : HEAT_RELEVANCE.get(col, 0),
            "rel_label"  : HEAT_RELEVANCE_LABELS[HEAT_RELEVANCE.get(col, 0)],
        })
    return pd.DataFrame(rows).sort_values(
        ["reported", "relevance"], ascending=False
    )


# ════════════════════════════════════════════════════════════════════════════
# V.  AGGREGATE ANALYTICS
# ════════════════════════════════════════════════════════════════════════════

def monthly_heat_symptom_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly aggregate: average heat risk and prevalence of heat-relevant symptoms.
    """
    df = df.copy()
    df["month"] = df["date_of_report"].dt.month

    heat_syms = [c for c in SYMPTOM_COLS if HEAT_RELEVANCE[c] >= 2]
    rows = []
    for month in range(1, 13):
        sub = df[df["month"] == month]
        if len(sub) == 0:
            continue
        avg_risk = sub["heat_risk"].mean()
        # Composite heat symptom prevalence (mean across heat-relevant symptoms)
        rates = []
        for col in heat_syms:
            valid = sub[col].dropna()
            if len(valid):
                rates.append(float(valid.sum() / len(valid) * 100))
        avg_sym_rate = np.mean(rates) if rates else 0.0
        rows.append({"month": month, "heat_risk": avg_risk,
                     "heat_symptom_rate": avg_sym_rate})
    return pd.DataFrame(rows)


def symptom_rate_by_risk_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each heat risk level (0–4): compute prevalence of heat-relevant symptoms.
    """
    heat_syms = [c for c in SYMPTOM_COLS if HEAT_RELEVANCE[c] >= 2]
    rows = []
    for hr in range(5):
        sub = df[df["heat_risk"] == hr]
        row = {"heat_risk": hr, "n": len(sub),
               "label": HEAT_RISK_LABELS[hr]}
        for col in heat_syms:
            valid = sub[col].dropna()
            row[col] = float(valid.sum() / len(valid) * 100) if len(valid) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# VI.  LLM PROMPT TEMPLATE BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_llm_prompt(participant: pd.Series, heat_risk: int,
                     scored: pd.DataFrame) -> str:
    """
    Fill in User Story #4 LLM Prompt Template with real values.

    Template:
    "The participant reported {symptoms} on {report_date}. The NWS HeatRisk
     index for {postal_code} on that date was {heat_risk_level} (scale: 0–4).
     In 2–3 sentences, explain the relationship between heat exposure and the
     reported symptoms, describe signs of heat illness requiring emergency care,
     and recommend protective behaviors. Cite the heat risk level explicitly."
    """
    # {symptoms}
    reported_syms = scored[scored["reported"] == 1]["label"].tolist()
    symptoms_str  = (", ".join(reported_syms) if reported_syms else "no symptoms")

    # {report_date}
    report_date_str = str(participant["date_of_report"].date())

    # {postal_code}
    postal_code = participant.get("postal_code", "unknown")

    # {heat_risk_level}
    heat_risk_str = (f"{heat_risk} ({HEAT_RISK_LABELS[heat_risk]}) — "
                     f"NWS HeatRisk index for {postal_code} "
                     f"({participant.get('county','?')} County, Arizona)")

    prompt = (
        "You are a One Health environmental health advisor.\n\n"
        f"The participant reported {symptoms_str} on {report_date_str}. "
        f"The NWS HeatRisk index for {postal_code} on that date was "
        f"{heat_risk_str} (scale: 0–4). "
        "In 2–3 sentences, explain the relationship between heat exposure "
        "and the reported symptoms, describe signs of heat illness requiring "
        "emergency care, and recommend protective behaviors. "
        "Cite the heat risk level explicitly.\n\n"
        f"[Context: participant age {participant.get('age','?')}, "
        f"sex {participant.get('sex','?')}, "
        f"occupation {participant.get('occupation','?')}. "
        f"Heat-relevant symptoms present: "
        f"{', '.join(scored[scored['reported']==1][scored['relevance']>=2]['label'].tolist()) or 'none'}"
        f"]"
    )
    return prompt


# ════════════════════════════════════════════════════════════════════════════
# VII.  SIMULATED LLM NARRATIVE
# ════════════════════════════════════════════════════════════════════════════

def simulate_narrative(participant: pd.Series, heat_risk: int,
                       scored: pd.DataFrame) -> str:
    """
    Rule-based 2–3 sentence narrative approximating a real LLM response.

    Production replacement:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        msg = client.messages.create(
            model="claude-opus-4-6", max_tokens=200,
            system="You are a One Health environmental health advisor. "
                   "Be specific, cite the HeatRisk level explicitly, and "
                   "do not overstate risk. Describe emergency signs clearly.",
            messages=[{"role":"user","content": prompt}]
        )
        return msg.content[0].text
    """
    pc          = participant.get("postal_code", "your area")
    county      = participant.get("county", "Maricopa")
    date_str    = str(participant["date_of_report"].date())
    risk_label  = HEAT_RISK_LABELS[heat_risk]
    risk_color_word = {0:"low", 1:"low", 2:"moderate", 3:"elevated", 4:"extreme"}[heat_risk]

    heat_sym_reported = scored[(scored["reported"] == 1) & (scored["relevance"] >= 2)]
    all_reported      = scored[scored["reported"] == 1]["label"].tolist()
    heat_labels       = heat_sym_reported["label"].tolist()
    has_direct        = (scored[scored["reported"] == 1]["relevance"] == 3).any()
    has_moderate      = (scored[scored["reported"] == 1]["relevance"] == 2).any()

    # Sentence 1: Heat risk + symptom context
    sym_str = (", ".join(all_reported[:-1]) + " and " + all_reported[-1]
               if len(all_reported) > 1 else all_reported[0]) if all_reported else "no symptoms"

    if heat_risk >= 3:
        if heat_labels:
            s1 = (f"On {date_str}, the NWS HeatRisk index for your postal area ({pc}) "
                  f"was {heat_risk} out of 4 ({risk_label}), indicating {risk_color_word} "
                  f"heat conditions in {county} County; your reported symptoms — "
                  f"{', '.join(heat_labels)} — are recognized signs of heat-related "
                  f"illness and may be exacerbated by prolonged exposure to these conditions.")
        else:
            s1 = (f"On {date_str}, the NWS HeatRisk index for postal area {pc} was "
                  f"{heat_risk} ({risk_label}), indicating {risk_color_word} heat "
                  f"conditions; while your reported symptoms ({sym_str}) are not "
                  f"primarily heat-related, heat exposure at this risk level can worsen "
                  f"many underlying health conditions and increase overall symptom burden.")
    elif heat_risk >= 2:
        if heat_labels:
            s1 = (f"The NWS HeatRisk index for {pc} on {date_str} was {heat_risk} "
                  f"({risk_label}); your reported symptoms of {', '.join(heat_labels)} "
                  f"can be associated with heat exposure during moderate-risk conditions, "
                  f"particularly without adequate hydration or shade.")
        else:
            s1 = (f"The NWS HeatRisk index for {pc} on {date_str} was {heat_risk} "
                  f"({risk_label}); while your reported symptoms ({sym_str}) are "
                  f"not strongly linked to heat exposure, moderate heat conditions "
                  f"can amplify general illness discomfort and dehydration risk.")
    else:
        s1 = (f"The NWS HeatRisk index for {pc} on {date_str} was {heat_risk} "
              f"({risk_label}), representing low ambient heat risk; your reported "
              f"symptoms ({sym_str}) are unlikely to be primarily heat-related "
              f"under these conditions.")

    # Sentence 2: Emergency signs of heat illness
    top2_signs = HEAT_EMERGENCY_SIGNS[:2]
    s2 = (f"Signs of heat illness requiring emergency care include "
          f"{top2_signs[0]}, {top2_signs[1]}, rapid strong pulse, "
          f"and sudden confusion or loss of consciousness — call 911 immediately "
          f"if any of these occur.")

    # Sentence 3: Protective behaviors (tiered)
    behaviors = HEAT_BEHAVIORS.get(heat_risk, HEAT_BEHAVIORS[2])
    b1, b2 = behaviors[0], behaviors[1]
    if heat_risk >= 3:
        s3 = (f"Given HeatRisk level {heat_risk}, we strongly recommend {b1.lower()} "
              f"and {b2.lower()}; avoid outdoor activity between 10 am and 6 pm and "
              f"drink water or electrolyte fluids regularly.")
    else:
        s3 = (f"As protective measures at HeatRisk {heat_risk}, we recommend "
              f"{b1.lower()} and {b2.lower()}, and continue monitoring your "
              f"symptoms over the next 24–48 hours.")

    return " ".join([s1, s2, s3])


# ════════════════════════════════════════════════════════════════════════════
# VIII.  VISUALIZATION — 4-PANEL DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

def fig_dashboard(participant: pd.Series, heat_risk: int,
                  scored: pd.DataFrame, monthly: pd.DataFrame,
                  by_risk: pd.DataFrame, narrative: str,
                  out_path: str) -> None:
    """
    4-panel heat-mode dashboard:
      Panel A — Symptom card with heat-relevance color coding
      Panel B — Monthly heat risk vs heat symptom rate (dual-axis)
      Panel C — Symptom prevalence by heat risk level (grouped bar)
      Panel D — LLM narrative with protective behaviors
    """
    risk_col = HEAT_RISK_COLORS[heat_risk]
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    plt.rcParams.update({
        "figure.facecolor": BG,    "axes.facecolor": SURFACE,
        "axes.edgecolor":   "#252840", "text.color":  TEXT,
        "axes.labelcolor":  TEXT,  "xtick.color":   SUBTEXT,
        "ytick.color":      SUBTEXT, "grid.color":   "#252840",
        "font.family":      "DejaVu Sans", "font.size": 9,
    })

    fig = plt.figure(figsize=(18, 13), facecolor=BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                            left=0.06, right=0.97, top=0.89, bottom=0.05)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    uid    = participant.get("unique_id", "?")
    age    = int(participant.get("age", 0)) if pd.notna(participant.get("age")) else "?"
    sex    = participant.get("sex", "?")
    pc     = participant.get("postal_code", "?")
    county = participant.get("county", "?")
    occ    = participant.get("occupation", "?")
    date_s = str(participant["date_of_report"].date())

    # Heat risk banner
    fig.text(0.5, 0.970,
             f"EpiHack — Heat-Related Risk Contextualization (User Story #4)",
             ha="center", fontsize=13, fontweight="bold", color=TEXT)
    fig.text(0.5, 0.952,
             f"Participant: ID {uid}  |  {sex}, Age {age}  |  Postal {pc}  |  "
             f"{county} County  |  Report Date: {date_s}  |  "
             f"NWS HeatRisk: {heat_risk}/4  ({HEAT_RISK_LABELS[heat_risk]})",
             ha="center", fontsize=10.5, color=risk_col, fontweight="bold")

    # ── Panel A: Symptom heat-relevance card ─────────────────────────────────
    rel_colors = {0: "#2a2a3a", 1: HEAT1 + "55", 2: HEAT2 + "66", 3: HEAT3 + "88"}
    dot_colors = {0: SUBTEXT, 1: HEAT1, 2: HEAT2, 3: HEAT3}

    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.axis("off")
    ax_a.set_facecolor(SURFACE)

    n_syms = len(SYMPTOM_COLS)
    row_h  = 0.92 / n_syms
    for i, (_, row) in enumerate(scored.iterrows()):
        y     = 0.95 - (i + 1) * row_h
        rep   = row["reported"]
        rel   = row["relevance"]
        bg_c  = rel_colors[rel] if rep == 1 else "#1a1a2a"
        dot_c = dot_colors[rel] if rep == 1 else ("#43e97b" if rep == 0 else SUBTEXT)
        alpha = 1.0 if rep == 1 else 0.45

        rect = plt.Rectangle((0.01, y), 0.98, row_h * 0.88,
                              transform=ax_a.transAxes, clip_on=False,
                              facecolor=bg_c, edgecolor="#2e3150", linewidth=0.4,
                              alpha=alpha)
        ax_a.add_patch(rect)

        # Dot indicator
        dot = plt.Circle((0.06, y + row_h * 0.44), 0.022,
                          transform=ax_a.transAxes, clip_on=False,
                          facecolor=dot_c, edgecolor="none")
        ax_a.add_patch(dot)

        # Symptom label
        ax_a.text(0.11, y + row_h * 0.44, row["label"],
                  va="center", ha="left", fontsize=7.5,
                  color=TEXT if rep == 1 else SUBTEXT,
                  fontweight="bold" if rep == 1 else "normal",
                  transform=ax_a.transAxes, alpha=alpha)

        # Relevance badge (right side)
        if rep == 1 and rel > 0:
            badge_color = dot_colors[rel]
            ax_a.text(0.98, y + row_h * 0.44, HEAT_RELEVANCE_LABELS[rel],
                      va="center", ha="right", fontsize=6,
                      color=badge_color, fontweight="bold",
                      transform=ax_a.transAxes)

    ax_a.set_title("Panel A — Symptom Heat-Relevance Card",
                   color=risk_col, fontsize=10, pad=8)
    # Legend
    leg_items = [
        mpatches.Patch(color=HEAT3 + "88", label="Direct heat illness sign"),
        mpatches.Patch(color=HEAT2 + "66", label="Moderate heat association"),
        mpatches.Patch(color=HEAT1 + "55", label="Possible heat link"),
        mpatches.Patch(color="#2a2a3a",     label="Not heat-related"),
    ]
    ax_a.legend(handles=leg_items, fontsize=6.5, loc="lower right",
                facecolor=SURFACE, edgecolor=risk_col, labelcolor=TEXT)

    # ── Panel B: Monthly trend — heat risk vs heat symptoms ───────────────────
    months     = monthly["month"].values
    m_labels   = [month_names[m] for m in months]
    risk_vals  = monthly["heat_risk"].values
    sym_vals   = monthly["heat_symptom_rate"].values

    ax_b2 = ax_b.twinx()

    ax_b.fill_between(range(len(months)), risk_vals, alpha=0.25, color=HEAT3, zorder=2)
    ax_b.plot(range(len(months)), risk_vals, color=HEAT3, lw=2.5, zorder=3,
               marker="o", ms=4, label="Avg HeatRisk level (left axis)")

    ax_b2.plot(range(len(months)), sym_vals, color=ACCENT, lw=2,
                zorder=4, marker="s", ms=4, ls="--",
                label="Heat-symptom prevalence % (right axis)")
    ax_b2.fill_between(range(len(months)), sym_vals, alpha=0.12, color=ACCENT, zorder=1)

    ax_b.set_xticks(range(len(months)))
    ax_b.set_xticklabels(m_labels, fontsize=8)
    ax_b.set_ylabel("Avg NWS HeatRisk Index", color=HEAT3, fontsize=8)
    ax_b.tick_params(axis="y", colors=HEAT3)
    ax_b2.set_ylabel("Heat Symptom Prevalence (%)", color=ACCENT, fontsize=8)
    ax_b2.tick_params(axis="y", colors=ACCENT)
    ax_b.set_ylim(0, 5.5)
    ax_b2.set_ylim(0, max(sym_vals) * 1.6 if sym_vals.max() > 0 else 5)
    ax_b.grid(alpha=0.3, zorder=0)
    ax_b.set_title("Panel B — Monthly Heat Risk vs Heat-Symptom Prevalence\n"
                   "(epidemiological paradox: winter fevers despite low heat risk)",
                   color=risk_col, fontsize=9.5, pad=8)

    lines1, labs1 = ax_b.get_legend_handles_labels()
    lines2, labs2 = ax_b2.get_legend_handles_labels()
    ax_b.legend(lines1 + lines2, labs1 + labs2, fontsize=7,
                facecolor=SURFACE, edgecolor=risk_col, labelcolor=TEXT, loc="upper left")

    # Annotate report date month
    rep_month = participant["date_of_report"].month
    m_idx = list(months).index(rep_month) if rep_month in months else None
    if m_idx is not None:
        ax_b.axvline(m_idx, color=HEAT4, lw=1.5, ls=":", alpha=0.8, zorder=5)
        ax_b.text(m_idx + 0.1, 5.0, "Report\nmonth", fontsize=6.5,
                  color=HEAT4, va="top")

    # ── Panel C: Symptom prevalence by heat risk level ────────────────────────
    heat_syms  = [c for c in SYMPTOM_COLS if HEAT_RELEVANCE[c] >= 2]
    heat_lbls  = [SYMPTOM_LABELS[c] for c in heat_syms]
    x          = np.arange(len(HEAT_RISK_LABELS))
    bar_w      = 0.12
    colors_bar = [HEAT_RISK_COLORS[i] for i in range(5)]

    for j, (col, lbl) in enumerate(zip(heat_syms, heat_lbls)):
        vals = [by_risk[by_risk["heat_risk"] == hr][col].values[0]
                if hr in by_risk["heat_risk"].values else 0
                for hr in range(5)]
        offset = (j - len(heat_syms) / 2 + 0.5) * bar_w
        ax_c.bar(x + offset, vals, bar_w * 0.9,
                 color=colors_bar, alpha=0.75, label=lbl,
                 edgecolor="none")

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(
        [f"Risk {i}\n{HEAT_RISK_LABELS[i]}\n(n={by_risk[by_risk['heat_risk']==i]['n'].values[0]:,})"
         for i in range(5)],
        fontsize=7
    )
    ax_c.set_ylabel("Symptom Prevalence (%)", fontsize=8)
    ax_c.set_title("Panel C — Heat-Relevant Symptom Prevalence by Heat Risk Level",
                   color=risk_col, fontsize=10, pad=8)
    ax_c.grid(axis="y", alpha=0.3)
    ax_c.legend(fontsize=7, facecolor=SURFACE, edgecolor=risk_col,
                labelcolor=TEXT, ncol=2, loc="upper right")

    # Highlight participant's risk level
    ax_c.axvspan(heat_risk - 0.45, heat_risk + 0.45,
                 alpha=0.08, color=risk_col, zorder=0)
    ax_c.text(heat_risk, ax_c.get_ylim()[1] * 0.95,
              f"Participant\nHeatRisk {heat_risk}", ha="center", fontsize=6.5,
              color=risk_col, fontweight="bold")

    # ── Panel D: LLM narrative + protective behaviors ─────────────────────────
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.axis("off")

    # Heat risk badge header
    badge = FancyBboxPatch((0.01, 0.88), 0.98, 0.10,
                           boxstyle="round,pad=0.02", linewidth=0,
                           facecolor=risk_col + "44", edgecolor="none")
    ax_d.add_patch(badge)
    ax_d.text(0.5, 0.935,
              f"NWS HeatRisk {heat_risk}/4 — {HEAT_RISK_LABELS[heat_risk]}  "
              f"|  Postal {pc}  |  {date_s}",
              ha="center", va="center", fontsize=9.5, fontweight="bold",
              color=risk_col, transform=ax_d.transAxes)

    card = FancyBboxPatch((0.01, 0.01), 0.98, 0.87,
                          boxstyle="round,pad=0.02", linewidth=1.5,
                          edgecolor=risk_col, facecolor=SURFACE2)
    ax_d.add_patch(card)

    ax_d.text(0.06, 0.84,
              "Panel D — LLM Health Narrative (2–3 sentences):",
              ha="left", va="top", fontsize=9, fontweight="bold",
              color=risk_col, transform=ax_d.transAxes)

    wrapped = textwrap.fill(narrative, width=78)
    ax_d.text(0.06, 0.78, wrapped,
              ha="left", va="top", fontsize=8, color=TEXT,
              transform=ax_d.transAxes, linespacing=1.6)

    # Protective behaviors
    behaviors = HEAT_BEHAVIORS.get(heat_risk, HEAT_BEHAVIORS[2])
    ax_d.text(0.06, 0.37,
              f"Protective Behaviors (HeatRisk {heat_risk}):",
              ha="left", va="top", fontsize=8.5, fontweight="bold",
              color=HEAT1, transform=ax_d.transAxes)
    for k, b in enumerate(behaviors[:4]):
        ax_d.text(0.06, 0.30 - k * 0.065, f"  {k+1}. {b}",
                  ha="left", va="top", fontsize=7.5, color=TEXT,
                  transform=ax_d.transAxes)

    # Emergency signs strip
    ax_d.text(0.06, 0.06,
              "EMERGENCY: Seek immediate care for — " +
              " · ".join(HEAT_EMERGENCY_SIGNS[:3]),
              ha="left", va="top", fontsize=6.5, color=HEAT4,
              fontweight="bold", transform=ax_d.transAxes,
              wrap=True)

    ax_d.set_title("Panel D — LLM Narrative & Protective Behaviors",
                   color=risk_col, fontsize=10, pad=8)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Dashboard saved -> {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# IX.  CONSOLE REPORT
# ════════════════════════════════════════════════════════════════════════════

def print_report(participant: pd.Series, heat_risk: int,
                 scored: pd.DataFrame, prompt: str, narrative: str) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print("  EpiHack -- Heat-Related Risk Contextualization  (User Story #4)")
    print(sep)
    print(f"  Participant ID  : {participant['unique_id']}")
    print(f"  Age / Sex       : {participant.get('age','?')} / {participant.get('sex','?')}")
    print(f"  Occupation      : {participant.get('occupation','?')}")
    print(f"  Postal Code     : {participant.get('postal_code','?')}")
    print(f"  County          : {participant.get('county','?')}")
    print(f"  Report Date     : {participant.get('date_of_report','?')}")
    print(f"  NWS HeatRisk    : {heat_risk}/4 -- {HEAT_RISK_LABELS[heat_risk]}")
    print()
    print(f"  {'SYMPTOM':<35} {'REPORTED':>10}  {'HEAT RELEVANCE'}")
    print(f"  {'-'*35} {'-'*10}  {'-'*25}")
    for _, row in scored.iterrows():
        rep_s = "YES" if row["reported"] == 1 else ("NO" if row["reported"] == 0 else "UNK")
        print(f"  {row['label']:<35} {rep_s:>10}  {row['rel_label']}")
    print()
    heat_score = scored[scored["reported"]==1]["relevance"].sum()
    print(f"  Composite heat relevance score: {heat_score}")
    print()
    print(f"  {'─'*68}")
    print("  LLM PROMPT TEMPLATE (filled):")
    print(f"  {'─'*68}")
    for line in prompt.split("\n"):
        print(f"  {line}")
    print()
    print(f"  {'─'*68}")
    print("  SIMULATED LLM NARRATIVE (2--3 sentences):")
    print(f"  {'─'*68}")
    for line in textwrap.wrap(narrative, 68):
        print(f"  {line}")
    print(f"\n{sep}\n")


# ════════════════════════════════════════════════════════════════════════════
# X.  CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EpiHack -- Heat-Related Risk Contextualization (User Story #4)"
    )
    p.add_argument("--db",  default="epihack-mini.db",
                   help="Path to epihack SQLite database")
    p.add_argument("--uid", type=int, default=None,
                   help="Participant unique_id (default: auto-select)")
    p.add_argument("--out", default="figures/",
                   help="Output directory for PNG figures")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/7] Loading data from: {args.db}")
    df = load_data(args.db)
    print(f"      {len(df):,} records | "
          f"{df['date_of_report'].min().date()} -- {df['date_of_report'].max().date()}")

    print("[2/7] Simulating NWS HeatRisk indices...")
    df = add_heat_risk(df)
    dist = df["heat_risk"].value_counts().sort_index()
    print("      Distribution:", dict(dist))

    print("[3/7] Selecting participant...")
    if args.uid:
        participant = get_participant(df, args.uid)
        print(f"      uid={args.uid} found.")
    else:
        participant = default_participant(df)
        print(f"      Auto-selected uid={participant['unique_id']} "
              f"(postal {participant['postal_code']}, "
              f"date {participant['date_of_report'].date()}).")
    heat_risk_val = int(participant["heat_risk"])
    print(f"      NWS HeatRisk: {heat_risk_val} ({HEAT_RISK_LABELS[heat_risk_val]})")

    print("[4/7] Scoring symptom heat relevance...")
    scored = score_symptoms(participant)

    print("[5/7] Computing aggregate analytics...")
    monthly = monthly_heat_symptom_trend(df)
    by_risk = symptom_rate_by_risk_level(df)

    print("[6/7] Filling LLM prompt template and generating narrative...")
    prompt    = build_llm_prompt(participant, heat_risk_val, scored)
    narrative = simulate_narrative(participant, heat_risk_val, scored)
    print_report(participant, heat_risk_val, scored, prompt, narrative)

    print("[7/7] Rendering 4-panel dashboard...")
    out_path = str(out_dir / "heat_risk_dashboard.png")
    fig_dashboard(participant, heat_risk_val, scored,
                  monthly, by_risk, narrative, out_path)

    print(f"\nDone. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
