"""
community_comparison_report.py
================================
EpiHack — Community Comparison Report (User Story #2)
------------------------------------------------------
Pipeline:
  1. Load epihack-mini.db (SQLite)
  2. Identify a participant by unique_id (default: auto-select with ≥2 symptoms)
  3. Compute community symptom prevalence rates for the participant's postal code
     over the past N days (default N=7)
  4. Build the LLM prompt template with real values:
       "The participant reports {participant_symptoms} and lives in postal area
        {postal_code}. In their community, {community_symptom_summary}
        (% of reporters with each symptom in the past 7 days). In 2–3 sentences,
        explain whether the participant's symptoms are consistent with community
        trends, and advise them on any precautions. Avoid stigmatizing language.
        Do not speculate about specific diseases."
  5. Compute a symptom consistency score (participant vs community)
  6. Simulate an LLM narrative (rule-based; swap for API call in production)
  7. Render a 4-panel matplotlib dashboard → PNG

Usage
-----
    python community_comparison_report.py \\
        --db epihack-mini.db [--uid 389174] [--n 7] [--out figures/]

Requirements
------------
    pip install pandas matplotlib numpy
"""

import argparse
import sqlite3
import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

# ── Column definitions ────────────────────────────────────────────────────────
SYMPTOM_COLS = [
    "cough_congestion",
    "nauseas_vomiting",
    "difficulty_breathing",
    "sore_throat",
    "rash",
    "fever",
    "chills",
    "diarrhea",
    "bleeding_from_body_openings",
    "red_eyes",
    "muscle_or_body_aches_and_pains",
    "discolored_or_bloody_urine",
    "loss_of_smell_or_taste",
    "yellow_skin_yellow_eyes",
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

HIGH_ACUITY = {
    "difficulty_breathing", "bleeding_from_body_openings",
    "yellow_skin_yellow_eyes", "discolored_or_bloody_urine",
}

# Precaution library keyed by symptom (for narrative generation)
PRECAUTIONS = {
    "cough_congestion":              "cover coughs, ventilate indoor spaces, and use hand hygiene",
    "nauseas_vomiting":              "stay hydrated, eat bland foods, and rest",
    "difficulty_breathing":          "seek medical evaluation promptly",
    "sore_throat":                   "stay hydrated, rest your voice, and avoid sharing utensils",
    "rash":                          "avoid scratching, keep the area clean, and consult a provider if it spreads",
    "fever":                         "rest, stay hydrated, and monitor temperature — seek care if fever exceeds 103 °F",
    "chills":                        "rest, keep warm, and monitor for fever development",
    "diarrhea":                      "prioritize oral rehydration and hand hygiene to prevent spread",
    "bleeding_from_body_openings":   "seek immediate clinical evaluation",
    "red_eyes":                      "avoid touching your eyes, practice hand hygiene, and avoid sharing towels",
    "muscle_or_body_aches_and_pains":"rest, stay hydrated, and use appropriate OTC pain relief as directed",
    "discolored_or_bloody_urine":    "increase fluid intake and consult a healthcare provider promptly",
    "loss_of_smell_or_taste":        "consider self-isolation, rest, and inform household members",
    "yellow_skin_yellow_eyes":       "seek prompt medical evaluation — this symptom warrants clinical attention",
}

# ── Design tokens ─────────────────────────────────────────────────────────────
BG      = "#0f1117"
SURFACE = "#1a1d27"
ACCENT  = "#6c63ff"
ACCENT2 = "#ff6584"
GREEN   = "#43e97b"
YELLOW  = "#f9ca24"
TEAL    = "#1dd1a1"
TEXT    = "#e8eaf6"
SUBTEXT = "#9e9eb0"


# ════════════════════════════════════════════════════════════════════════════
# I.  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_data(db_path: str) -> pd.DataFrame:
    """Load incidents table with symptom columns mapped to 0/1/NaN."""
    con = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT * FROM incidents", con,
        parse_dates=["date_of_report", "date_of_illness"]
    )
    con.close()
    for col in SYMPTOM_COLS + ["no_symptoms", "symptoms"]:
        df[col] = df[col].map({"YES": 1, "NO": 0}).astype("Float64")
    return df


# ════════════════════════════════════════════════════════════════════════════
# II.  PARTICIPANT SELECTION
# ════════════════════════════════════════════════════════════════════════════

def get_participant(df: pd.DataFrame, uid: int) -> pd.Series:
    matches = df[df["unique_id"] == uid]
    if matches.empty:
        raise ValueError(f"unique_id {uid} not found.")
    return matches.iloc[0]


def default_participant(df: pd.DataFrame, N: int) -> pd.Series:
    """
    Auto-select: participant with the highest symptom count whose report
    falls in the recent N-day window and whose postal code has ≥5 reporters.
    """
    max_date    = df["date_of_report"].max()
    win_start   = max_date - pd.Timedelta(days=N - 1)
    recent      = df[df["date_of_report"] >= win_start].copy()
    recent["n_symp"] = recent[SYMPTOM_COLS].sum(axis=1)
    candidates  = recent[recent["n_symp"] >= 1]

    # Prefer postal codes with richer community data
    pc_counts   = df.groupby("postal_code").size()
    good_pcs    = pc_counts[pc_counts >= 5].index
    rich        = candidates[candidates["postal_code"].isin(good_pcs)]
    pool        = rich if not rich.empty else candidates

    if pool.empty:
        return df[df["symptoms"] == 1].iloc[0]
    return pool.sort_values("n_symp", ascending=False).iloc[0]


# ════════════════════════════════════════════════════════════════════════════
# III.  COMMUNITY RATES — N-DAY WINDOW
# ════════════════════════════════════════════════════════════════════════════

def compute_community_rates(df: pd.DataFrame, postal_code: str, N: int) -> dict:
    """
    Compute symptom prevalence rates for a postal code over the past N days
    (anchored to the dataset max date).

    Returns
    -------
    dict with keys: rates (Series), n, window_start, window_end, community_df
    """
    max_date    = df["date_of_report"].max()
    win_start   = max_date - pd.Timedelta(days=N - 1)
    community   = df[
        (df["postal_code"] == postal_code) &
        (df["date_of_report"] >= win_start)
    ]

    rates = {}
    for col in SYMPTOM_COLS:
        valid = community[col].dropna()
        rates[col] = float(valid.sum() / len(valid)) if len(valid) else np.nan

    return {
        "rates"        : pd.Series(rates),
        "n"            : len(community),
        "window_start" : win_start,
        "window_end"   : max_date,
        "community_df" : community,
    }


# ════════════════════════════════════════════════════════════════════════════
# IV.  CONSISTENCY SCORE
# ════════════════════════════════════════════════════════════════════════════

def compute_consistency(participant: pd.Series, rates: pd.Series) -> dict:
    """
    For each symptom the participant reported (YES=1):
      - 'concordant'   : community rate > 0  (also circulating)
      - 'idiosyncratic': community rate == 0 (not seen in community)
      - 'novel'        : community rate is NaN (insufficient data)

    Returns a dict with symptom lists and an overall consistency score (0–100).
    """
    p_syms = [c for c in SYMPTOM_COLS if participant.get(c) == 1]
    if not p_syms:
        return {"score": 100, "concordant": [], "idiosyncratic": [], "novel": [],
                "participant_symptoms": []}

    concordant    = []
    idiosyncratic = []
    novel         = []
    for col in p_syms:
        r = rates.get(col, np.nan)
        if np.isnan(r):
            novel.append(col)
        elif r > 0:
            concordant.append(col)
        else:
            idiosyncratic.append(col)

    score = int(len(concordant) / len(p_syms) * 100) if p_syms else 100
    return {
        "score"                : score,
        "concordant"           : concordant,
        "idiosyncratic"        : idiosyncratic,
        "novel"                : novel,
        "participant_symptoms" : p_syms,
    }


# ════════════════════════════════════════════════════════════════════════════
# V.  LLM PROMPT TEMPLATE BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_llm_prompt(participant: pd.Series, rates: pd.Series,
                     community_info: dict, N: int) -> str:
    """
    Fill in User Story #2 LLM Prompt Template with real database values.

    Template:
    "The participant reports {participant_symptoms} and lives in postal area
     {postal_code}. In their community, {community_symptom_summary}
     (% of reporters with each symptom in the past 7 days). In 2–3 sentences,
     explain whether the participant's symptoms are consistent with community
     trends, and advise them on any precautions. Avoid stigmatizing language.
     Do not speculate about specific diseases."
    """
    # {participant_symptoms}
    p_syms = [SYMPTOM_LABELS[c] for c in SYMPTOM_COLS if participant.get(c) == 1]
    participant_symptoms_str = (
        ", ".join(p_syms) if p_syms else "no symptoms"
    )

    # {postal_code}
    postal_code = participant.get("postal_code", "unknown")

    # {community_symptom_summary}
    active = {col: rates[col] for col in SYMPTOM_COLS
              if not np.isnan(rates.get(col, np.nan)) and rates.get(col, 0) > 0}
    if active:
        parts = [
            f"{SYMPTOM_LABELS[col]}: {v*100:.1f}%"
            for col, v in sorted(active.items(), key=lambda x: -x[1])
        ]
        community_summary_str = "; ".join(parts)
    else:
        community_summary_str = "no symptoms are currently being reported above 0%"

    prompt = (
        "You are a public health data interpreter.\n\n"
        f"The participant reports {participant_symptoms_str} "
        f"and lives in postal area {postal_code}. "
        f"In their community, {community_summary_str} "
        f"(% of reporters with each symptom in the past {N} days). "
        "In 2–3 sentences, explain whether the participant's symptoms are "
        "consistent with community trends, and advise them on any precautions. "
        "Avoid stigmatizing language. "
        "Do not speculate about specific diseases.\n\n"
        f"Community data quality note: based on n={community_info['n']} "
        f"reporters in postal {postal_code} over the past {N} days "
        f"({community_info['window_start'].date()} – "
        f"{community_info['window_end'].date()}). "
        "Flag rates with n<10 as preliminary."
    )
    return prompt


# ════════════════════════════════════════════════════════════════════════════
# VI.  SIMULATED LLM NARRATIVE
# ════════════════════════════════════════════════════════════════════════════

def simulate_llm_narrative(participant: pd.Series, rates: pd.Series,
                            consistency: dict, community_info: dict,
                            N: int) -> str:
    """
    Rule-based narrative approximating a real LLM response.
    Production replacement:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
        msg = client.messages.create(
            model="claude-opus-4-6", max_tokens=200,
            messages=[{"role":"user","content": prompt}]
        )
        return msg.content[0].text
    """
    p_syms  = consistency["participant_symptoms"]
    conc    = consistency["concordant"]
    idio    = consistency["idiosyncratic"]
    score   = consistency["score"]
    n_comm  = community_info["n"]
    pc      = participant.get("postal_code", "your area")
    n_note  = f" (preliminary, n={n_comm})" if n_comm < 10 else ""

    # Sentence 1 — consistency assessment
    p_labels = [SYMPTOM_LABELS[c] for c in p_syms]
    sym_str  = (", ".join(p_labels[:-1]) + " and " + p_labels[-1]
                if len(p_labels) > 1 else p_labels[0]) if p_labels else "no symptoms"

    if not p_syms:
        s1 = (f"Your report of no active symptoms is reassuring and aligns well with "
              f"the low symptom burden currently observed in postal area {pc}{n_note}.")
    elif score >= 75:
        s1 = (f"Your reported symptoms — {sym_str} — are consistent with patterns "
              f"being seen across your community in postal area {pc}{n_note}, suggesting "
              f"these symptoms may be part of a broader circulating illness trend.")
    elif score >= 25:
        conc_labels = [SYMPTOM_LABELS[c] for c in conc]
        idio_labels = [SYMPTOM_LABELS[c] for c in idio]
        conc_str = ", ".join(conc_labels) if conc_labels else "none"
        idio_str = ", ".join(idio_labels) if idio_labels else "none"
        s1 = (f"Your symptoms show partial overlap with community trends in postal area "
              f"{pc}{n_note}: {conc_str} {'is' if len(conc_labels)==1 else 'are'} also "
              f"reported in the community, while {idio_str} "
              f"{'appears' if len(idio_labels)==1 else 'appear'} less common locally.")
    else:
        s1 = (f"Your reported symptoms — {sym_str} — are not widely reflected in the "
              f"current community surveillance data for postal area {pc}{n_note}, "
              f"which does not diminish their importance to your individual health.")

    # Sentence 2 — precautions (top 2 participant symptoms)
    prec_syms  = p_syms[:2] if p_syms else []
    prec_parts = [PRECAUTIONS.get(c, "monitor your symptoms and rest") for c in prec_syms]
    has_acute  = any(c in HIGH_ACUITY for c in p_syms)

    if has_acute:
        s2 = ("Given the nature of your reported symptoms, we recommend seeking "
              "clinical evaluation promptly rather than waiting to see if symptoms resolve.")
    elif prec_parts:
        s2 = ("As a precaution, we recommend " + " and ".join(prec_parts) + ".")
    else:
        s2 = ("Continue monitoring your health and maintain routine hygiene practices.")

    # Sentence 3 — encouragement / data quality note
    if n_comm < 10:
        s3 = (f"Please note that community rates are based on a small number of reporters "
              f"(n={n_comm}) and should be interpreted with caution — your participation "
              f"helps improve the accuracy of these estimates over time.")
    else:
        s3 = ("Your report strengthens community surveillance in your area, helping "
              "public health teams identify trends and allocate resources effectively.")

    return " ".join([s1, s2, s3])


# ════════════════════════════════════════════════════════════════════════════
# VII.  VISUALIZATION — 4-PANEL DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

def fig_dashboard(participant: pd.Series, rates: pd.Series,
                  consistency: dict, community_info: dict,
                  narrative: str, N: int, out_path: str) -> None:
    """
    4-panel dark-mode dashboard:
      Panel A — Community symptom rates bar chart (ranked), participant highlights
      Panel B — Radar chart: participant binary vs community prevalence
      Panel C — Consistency score gauge + concordance table
      Panel D — LLM narrative card
    """
    plt.rcParams.update({
        "figure.facecolor": BG,  "axes.facecolor": SURFACE,
        "axes.edgecolor":   "#2e3150", "text.color":  TEXT,
        "axes.labelcolor":  TEXT,  "xtick.color":   SUBTEXT,
        "ytick.color":      SUBTEXT, "grid.color":   "#252840",
        "font.family":      "DejaVu Sans", "font.size": 9,
    })

    fig = plt.figure(figsize=(18, 13), facecolor=BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                            left=0.06, right=0.97, top=0.91, bottom=0.05)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], polar=True)
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    uid  = participant.get("unique_id", "?")
    age  = int(participant.get("age", 0)) if pd.notna(participant.get("age")) else "?"
    sex  = participant.get("sex", "?")
    pc   = participant.get("postal_code", "?")
    occ  = participant.get("occupation", "?")

    fig.suptitle(
        f"EpiHack — Community Comparison Report (User Story #2)\n"
        f"Participant: ID {uid}  |  {sex}, Age {age}  |  Postal {pc}  |  {occ}  "
        f"|  Community window: past {N} days  |  n={community_info['n']} reporters",
        fontsize=13, fontweight="bold", color=TEXT, y=0.975
    )

    # ── Panel A: Community ranked bar chart ───────────────────────────────────
    p_syms = set(consistency["participant_symptoms"])
    active = [(col, rates[col]) for col in SYMPTOM_COLS
              if not np.isnan(rates.get(col, np.nan)) and rates.get(col, 0) > 0]
    active.sort(key=lambda x: x[1], reverse=True)

    if active:
        labels_a = [SYMPTOM_LABELS[c] for c, _ in active]
        vals_a   = [v * 100 for _, v in active]
        colors_a = []
        for col, _ in active:
            if col in p_syms:
                colors_a.append(ACCENT2 if col in consistency["concordant"] else YELLOW)
            else:
                colors_a.append(ACCENT + "80")  # dim community-only

        y_pos = np.arange(len(active))
        bars  = ax_a.barh(y_pos, vals_a, color=colors_a, height=0.6, zorder=2)
        ax_a.set_yticks(y_pos)
        ax_a.set_yticklabels(labels_a, fontsize=8.5)
        ax_a.set_xlabel("Community Prevalence (%)", fontsize=8)
        ax_a.set_title(
            f"Panel A — Community Symptom Rates\n"
            f"Postal {pc} · Past {N} days · n={community_info['n']}",
            color=ACCENT, fontsize=10, pad=8
        )
        ax_a.grid(axis="x", alpha=0.3, zorder=0)

        # Annotate bars
        for bar, val in zip(bars, vals_a):
            ax_a.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                      f"{val:.1f}%", va="center", fontsize=7.5, color=SUBTEXT)

        legend_els = [
            mpatches.Patch(color=ACCENT2, label="Participant & community (concordant)"),
            mpatches.Patch(color=YELLOW,  label="Participant only (idiosyncratic)"),
            mpatches.Patch(color=ACCENT + "80", label="Community only"),
        ]
        ax_a.legend(handles=legend_els, fontsize=7, facecolor=SURFACE,
                    edgecolor=ACCENT, labelcolor=TEXT)
    else:
        ax_a.text(0.5, 0.5, "No active community symptoms\nin this window",
                  ha="center", va="center", color=SUBTEXT, fontsize=11,
                  transform=ax_a.transAxes)
        ax_a.set_title(f"Panel A — Community Symptom Rates (Postal {pc})",
                       color=ACCENT, fontsize=10, pad=8)

    # ── Panel B: Radar chart ──────────────────────────────────────────────────
    # Include symptoms that either the participant reported or the community shows
    radar_cols = [col for col in SYMPTOM_COLS
                  if participant.get(col) == 1 or
                  (not np.isnan(rates.get(col, np.nan)) and rates.get(col, 0) > 0)]
    if not radar_cols:
        radar_cols = SYMPTOM_COLS[:6]   # fallback: show first 6

    N_ax  = len(radar_cols)
    theta = np.linspace(0, 2 * np.pi, N_ax, endpoint=False)
    theta = np.concatenate([theta, [theta[0]]])  # close polygon

    comm_vals = np.array([rates.get(c, 0) * 100 for c in radar_cols])
    pax_vals  = np.array([float(participant.get(c, 0) or 0) * 100 for c in radar_cols])
    comm_vals = np.concatenate([comm_vals, [comm_vals[0]]])
    pax_vals  = np.concatenate([pax_vals,  [pax_vals[0]]])
    radar_labels = [SYMPTOM_LABELS[c] for c in radar_cols]

    ax_b.plot(theta, comm_vals, color=ACCENT,  lw=2, label="Community %")
    ax_b.fill(theta, comm_vals, alpha=0.20,   color=ACCENT)
    ax_b.plot(theta, pax_vals,  color=ACCENT2, lw=2, label="Participant (×100)")
    ax_b.fill(theta, pax_vals,  alpha=0.18,   color=ACCENT2)
    ax_b.set_xticks(theta[:-1])
    ax_b.set_xticklabels(radar_labels, fontsize=7, color=TEXT)
    ax_b.tick_params(colors=SUBTEXT)
    ax_b.set_facecolor(SURFACE)
    ax_b.spines["polar"].set_color("#2e3150")
    ax_b.yaxis.set_tick_params(labelsize=6, labelcolor=SUBTEXT)
    ax_b.set_title("Panel B — Participant vs Community Radar\n(community %, participant binary×100)",
                   color=ACCENT, fontsize=10, pad=16)
    ax_b.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.28, 1.15),
                facecolor=SURFACE, edgecolor=ACCENT, labelcolor=TEXT)

    # ── Panel C: Consistency score + concordance table ────────────────────────
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    ax_c.axis("off")
    ax_c.set_facecolor(SURFACE)

    # Score gauge (simple arc)
    score = consistency["score"]
    gauge_bg = plt.matplotlib.patches.Wedge(
        (0.22, 0.72), 0.15, 180, 360,
        width=0.055, facecolor="#2a2a4a", edgecolor="none"
    )
    score_color = GREEN if score >= 75 else (YELLOW if score >= 25 else ACCENT2)
    fill_angle  = 180 + score * 1.8
    gauge_fill  = plt.matplotlib.patches.Wedge(
        (0.22, 0.72), 0.15, 180, fill_angle,
        width=0.055, facecolor=score_color, edgecolor="none"
    )
    ax_c.add_patch(gauge_bg)
    ax_c.add_patch(gauge_fill)
    ax_c.text(0.22, 0.67, f"{score}%", ha="center", va="center",
              fontsize=20, fontweight="bold", color=score_color,
              transform=ax_c.transAxes)
    ax_c.text(0.22, 0.60, "Consistency\nScore", ha="center", va="center",
              fontsize=9, color=SUBTEXT, transform=ax_c.transAxes)

    # Legend for score
    ax_c.text(0.55, 0.90, "Symptom Concordance Summary",
              ha="center", va="top", fontsize=10, fontweight="bold",
              color=ACCENT, transform=ax_c.transAxes)

    rows = [
        ("Concordant (also in community)",   consistency["concordant"],    GREEN),
        ("Idiosyncratic (not in community)", consistency["idiosyncratic"], ACCENT2),
        ("Insufficient data (community)",    consistency["novel"],         YELLOW),
    ]
    y_row = 0.80
    for label, syms, color in rows:
        ax_c.text(0.55, y_row, label + ":", ha="left", va="top",
                  fontsize=8.5, fontweight="bold", color=color,
                  transform=ax_c.transAxes)
        y_row -= 0.08
        sym_str = (", ".join(SYMPTOM_LABELS[s] for s in syms)
                   if syms else "none")
        for line in textwrap.wrap(sym_str, width=52):
            ax_c.text(0.55, y_row, line, ha="left", va="top",
                      fontsize=8, color=TEXT, transform=ax_c.transAxes)
            y_row -= 0.065

    # Data note
    ax_c.text(0.5, 0.03,
              f"Window: {community_info['window_start'].date()} – "
              f"{community_info['window_end'].date()} · "
              f"n={community_info['n']} reporters in postal {pc}",
              ha="center", va="bottom", fontsize=7, color=SUBTEXT,
              transform=ax_c.transAxes)

    ax_c.set_title("Panel C — Consistency Score & Concordance Analysis",
                   color=ACCENT, fontsize=10, pad=8)

    # ── Panel D: LLM narrative card ───────────────────────────────────────────
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.axis("off")

    rect = FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                          boxstyle="round,pad=0.02", linewidth=1.5,
                          edgecolor=ACCENT, facecolor="#12152a")
    ax_d.add_patch(rect)

    ax_d.text(0.5, 0.93, "Panel D — LLM-Generated Community Comparison Narrative",
              ha="center", va="top", fontsize=10, fontweight="bold",
              color=ACCENT, transform=ax_d.transAxes)
    ax_d.text(0.5, 0.87, "(Simulated — replace with live API call in production)",
              ha="center", va="top", fontsize=7, color=SUBTEXT, style="italic",
              transform=ax_d.transAxes)

    wrapped = textwrap.fill(narrative, width=80)
    ax_d.text(0.06, 0.79, wrapped,
              ha="left", va="top", fontsize=8.5, color=TEXT,
              transform=ax_d.transAxes, linespacing=1.65)

    # LLM prompt fields filled
    p_labels_str = ", ".join(SYMPTOM_LABELS[c] for c in consistency["participant_symptoms"]) or "None"
    ax_d.text(0.06, 0.20,
              f"Template fields resolved:\n"
              f"  {{participant_symptoms}} → {p_labels_str}\n"
              f"  {{postal_code}} → {pc}\n"
              f"  {{community_symptom_summary}} → [see Panel A]",
              ha="left", va="top", fontsize=7.5, color=SUBTEXT,
              transform=ax_d.transAxes, linespacing=1.5)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Dashboard saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# VIII.  CONSOLE REPORT
# ════════════════════════════════════════════════════════════════════════════

def print_report(participant: pd.Series, rates: pd.Series,
                 consistency: dict, community_info: dict,
                 prompt: str, narrative: str, N: int) -> None:
    sep = "═" * 72
    print(f"\n{sep}")
    print("  EpiHack — Community Comparison Report  (User Story #2)")
    print(sep)
    print(f"  Participant ID   : {participant['unique_id']}")
    print(f"  Age / Sex        : {participant.get('age','?')} / {participant.get('sex','?')}")
    print(f"  Occupation       : {participant.get('occupation','?')}")
    print(f"  Postal Code      : {participant.get('postal_code','?')}")
    print(f"  County           : {participant.get('county','?')}")
    print(f"  Report Date      : {participant.get('date_of_report','?')}")
    print(f"  Community Window : {N} days  "
          f"({community_info['window_start'].date()} – {community_info['window_end'].date()})")
    print(f"  Community n      : {community_info['n']} reporters")
    print()

    p_syms = [SYMPTOM_LABELS[c] for c in consistency["participant_symptoms"]]
    print(f"  Reported Symptoms    : {', '.join(p_syms) or 'None'}")
    print(f"  Consistency Score    : {consistency['score']}%")
    print()

    print(f"  {'SYMPTOM':<35} {'COMM%':>8}  STATUS")
    print(f"  {'-'*35} {'-'*8}  ------")
    for col in SYMPTOM_COLS:
        r   = rates.get(col, np.nan)
        pv  = participant.get(col, None)
        r_s = f"{r*100:.1f}%" if not np.isnan(r) else "—"
        if pv == 1:
            status = ("CONCORDANT" if col in consistency["concordant"]
                      else ("IDIOSYNCRATIC" if col in consistency["idiosyncratic"]
                            else "NO DATA"))
        else:
            status = "community only" if (not np.isnan(r) and r > 0) else ""
        marker = " ⚠" if col in HIGH_ACUITY and pv == 1 else ""
        print(f"  {SYMPTOM_LABELS[col]:<35} {r_s:>8}  {status}{marker}")
    print()

    print(f"  {'─'*68}")
    print("  LLM PROMPT (filled):")
    print(f"  {'─'*68}")
    for line in prompt.split("\n"):
        print(f"  {line}")
    print()
    print(f"  {'─'*68}")
    print("  SIMULATED LLM NARRATIVE (2–3 sentences):")
    print(f"  {'─'*68}")
    for line in textwrap.wrap(narrative, 68):
        print(f"  {line}")
    print(f"\n{sep}\n")


# ════════════════════════════════════════════════════════════════════════════
# IX.  CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EpiHack — Community Comparison Report (User Story #2)"
    )
    p.add_argument("--db",  default="epihack-mini.db",
                   help="Path to epihack SQLite database")
    p.add_argument("--uid", type=int, default=None,
                   help="Participant unique_id (default: auto-select)")
    p.add_argument("--n",   type=int, default=7,
                   help="Community window in days (default: 7)")
    p.add_argument("--out", default="figures/",
                   help="Output directory for PNG figures")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/6] Loading data from: {args.db}")
    df = load_data(args.db)
    print(f"      {len(df):,} records | "
          f"{df['date_of_report'].min().date()} – {df['date_of_report'].max().date()}")

    print(f"[2/6] Selecting participant (N={args.n}-day window)...")
    if args.uid is not None:
        participant = get_participant(df, args.uid)
        print(f"      uid={args.uid} found.")
    else:
        participant = default_participant(df, args.n)
        print(f"      Auto-selected uid={participant['unique_id']} "
              f"(postal {participant['postal_code']}).")

    print(f"[3/6] Computing community rates (postal {participant['postal_code']}, N={args.n})...")
    community_info = compute_community_rates(df, participant["postal_code"], args.n)
    print(f"      n={community_info['n']} reporters in window.")

    print("[4/6] Computing consistency score...")
    consistency = compute_consistency(participant, community_info["rates"])
    print(f"      Score={consistency['score']}%  "
          f"concordant={len(consistency['concordant'])}  "
          f"idiosyncratic={len(consistency['idiosyncratic'])}")

    print("[5/6] Filling prompt template and generating narrative...")
    prompt    = build_llm_prompt(participant, community_info["rates"],
                                  community_info, args.n)
    narrative = simulate_llm_narrative(participant, community_info["rates"],
                                       consistency, community_info, args.n)
    print_report(participant, community_info["rates"], consistency,
                 community_info, prompt, narrative, args.n)

    print("[6/6] Rendering 4-panel dashboard...")
    out_path = str(out_dir / "community_comparison_dashboard.png")
    fig_dashboard(participant, community_info["rates"], consistency,
                  community_info, narrative, args.n, out_path)

    print("\nDone. Outputs written to:", out_dir)


if __name__ == "__main__":
    main()
