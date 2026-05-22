"""
symptom_trajectory_report.py
=============================
EpiHack — Participant Symptom Trajectory Report (User Story #1)
----------------------------------------------------------------
Pipeline:
  1. Load epihack-mini.db (SQLite)
  2. Identify a participant by unique_id (default: sample with symptoms)
  3. Compute community-level symptom rates for the participant's postal code:
       • Recent window  : last 30 days relative to dataset max date
       • Baseline window: 30-day period immediately before the recent window
  4. Build the LLM prompt template with real values
  5. Simulate an LLM narrative (rule-based; swap for API call in production)
  6. Render a 4-panel matplotlib dashboard and save to PNG

Usage
-----
    python symptom_trajectory_report.py --db epihack-mini.db [--uid 384033] [--out figures/]

Requirements
------------
    pip install pandas matplotlib seaborn numpy
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

# ── Design tokens (dark-mode palette) ────────────────────────────────────────
BG      = "#0f1117"
SURFACE = "#1a1d27"
ACCENT  = "#6c63ff"
ACCENT2 = "#ff6584"
GREEN   = "#43e97b"
YELLOW  = "#f9ca24"
TEXT    = "#e8eaf6"
SUBTEXT = "#9e9eb0"


# ════════════════════════════════════════════════════════════════════════════
# I.  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_data(db_path: str) -> pd.DataFrame:
    """Load incidents table from SQLite into a DataFrame with typed columns."""
    con = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM incidents", con, parse_dates=["date_of_report", "date_of_illness"])
    con.close()

    # Map YES→1, NO→0, UNKNOWN / anything else → NaN  (Float64 keeps NaN in int-like col)
    for col in SYMPTOM_COLS + ["no_symptoms", "symptoms"]:
        df[col] = df[col].map({"YES": 1, "NO": 0}).astype("Float64")

    return df


# ════════════════════════════════════════════════════════════════════════════
# II.  PARTICIPANT SELECTION
# ════════════════════════════════════════════════════════════════════════════

def get_participant(df: pd.DataFrame, uid: int) -> pd.Series:
    """Return a single participant record by unique_id."""
    matches = df[df["unique_id"] == uid]
    if matches.empty:
        raise ValueError(f"unique_id {uid} not found in database.")
    return matches.iloc[0]


def default_participant(df: pd.DataFrame) -> pd.Series:
    """
    Choose a representative participant: someone who reported ≥2 symptoms,
    whose postal code has ≥100 community records, and whose report date
    falls within the recent window of the dataset.
    """
    max_date = df["date_of_report"].max()
    recent_start = max_date - pd.Timedelta(days=30)

    # Participants with ≥2 symptom flags in the recent window
    recent = df[df["date_of_report"] >= recent_start].copy()
    recent["n_symptoms"] = recent[SYMPTOM_COLS].sum(axis=1)
    candidates = recent[recent["n_symptoms"] >= 2].copy()

    if candidates.empty:
        # Fallback: any record with symptoms
        candidates = df[df["symptoms"] == 1].copy()
        candidates["n_symptoms"] = candidates[SYMPTOM_COLS].sum(axis=1)

    # Prefer candidates whose postal code has ≥100 records
    pc_counts = df.groupby("postal_code").size()
    good_pcs = pc_counts[pc_counts >= 100].index
    rich = candidates[candidates["postal_code"].isin(good_pcs)]
    pool = rich if not rich.empty else candidates

    # Sort by symptom count descending; take top
    return pool.sort_values("n_symptoms", ascending=False).iloc[0]


# ════════════════════════════════════════════════════════════════════════════
# III.  COMMUNITY WINDOW COMPUTATION
# ════════════════════════════════════════════════════════════════════════════

def compute_windows(df: pd.DataFrame, postal_code: str) -> dict:
    """
    For a given postal code, compute symptom prevalence rates in:
      • recent_window  : last 30 days relative to dataset max date
      • baseline_window: 30-day period immediately before the recent window

    Returns a dict with keys:
        max_date, recent_start, baseline_start, baseline_end,
        recent_df, baseline_df,
        recent_rates, baseline_rates,
        recent_n, baseline_n
    """
    community = df[df["postal_code"] == postal_code].copy()
    max_date = df["date_of_report"].max()

    recent_start   = max_date - pd.Timedelta(days=29)      # inclusive 30-day window
    baseline_end   = recent_start - pd.Timedelta(days=1)
    baseline_start = baseline_end - pd.Timedelta(days=29)  # prior 30-day window

    recent_df   = community[community["date_of_report"] >= recent_start]
    baseline_df = community[
        (community["date_of_report"] >= baseline_start) &
        (community["date_of_report"] <= baseline_end)
    ]

    def symptom_rates(subset: pd.DataFrame) -> pd.Series:
        """Prevalence: YES / (YES + NO), excludes NaN."""
        rates = {}
        for col in SYMPTOM_COLS:
            valid = subset[col].dropna()
            rates[col] = float(valid.sum() / len(valid)) if len(valid) else np.nan
        return pd.Series(rates)

    return {
        "max_date"       : max_date,
        "recent_start"   : recent_start,
        "baseline_start" : baseline_start,
        "baseline_end"   : baseline_end,
        "recent_df"      : recent_df,
        "baseline_df"    : baseline_df,
        "recent_rates"   : symptom_rates(recent_df),
        "baseline_rates" : symptom_rates(baseline_df),
        "recent_n"       : len(recent_df),
        "baseline_n"     : len(baseline_df),
        "community"      : community,
    }


def compute_daily_series(community: pd.DataFrame, recent_start: pd.Timestamp,
                         max_date: pd.Timestamp) -> pd.DataFrame:
    """
    Build a daily symptom burden time series (% of reporters with ≥1 symptom)
    for the full community spanning both windows.
    """
    start = recent_start - pd.Timedelta(days=30)  # include baseline in chart
    mask  = (community["date_of_report"] >= start) & (community["date_of_report"] <= max_date)
    sub   = community[mask].copy()

    sub["any_symptom"] = (sub[SYMPTOM_COLS].sum(axis=1) > 0).astype(float)
    daily = (
        sub.groupby("date_of_report")["any_symptom"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "symptomatic", "count": "total"})
        .reset_index()
    )
    daily["pct"] = daily["symptomatic"] / daily["total"] * 100
    return daily


# ════════════════════════════════════════════════════════════════════════════
# IV.  SIGNAL CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════

def classify_signal(rec_rate: float, bas_rate: float, threshold: float = 0.05) -> str:
    """
    Classify % change from baseline.
      'new'   : baseline=0, recent>0
      'gone'  : recent=0, baseline>0
      'up'    : pct_change > +threshold
      'dn'    : pct_change < -threshold
      'nc'    : within ±threshold
      'indef' : insufficient data
    """
    if np.isnan(rec_rate) or np.isnan(bas_rate):
        return "indef"
    if bas_rate == 0 and rec_rate > 0:
        return "new"
    if rec_rate == 0 and bas_rate > 0:
        return "gone"
    if bas_rate == 0:
        return "nc"
    pct = (rec_rate - bas_rate) / bas_rate
    if pct > threshold:
        return "up"
    if pct < -threshold:
        return "dn"
    return "nc"


def build_signal_table(recent_rates: pd.Series, baseline_rates: pd.Series) -> pd.DataFrame:
    rows = []
    for col in SYMPTOM_COLS:
        rec = recent_rates.get(col, np.nan)
        bas = baseline_rates.get(col, np.nan)
        sig = classify_signal(rec, bas)
        pct_chg = ((rec - bas) / bas * 100) if (
            not np.isnan(rec) and not np.isnan(bas) and bas != 0
        ) else (np.nan if np.isnan(rec) or bas == 0 else 0.0)
        rows.append({
            "symptom"      : col,
            "label"        : SYMPTOM_LABELS[col],
            "recent_rate"  : rec,
            "baseline_rate": bas,
            "pct_change"   : pct_chg,
            "signal"       : sig,
        })
    return pd.DataFrame(rows).sort_values("pct_change", ascending=False, key=lambda x: x.fillna(-9999))


# ════════════════════════════════════════════════════════════════════════════
# V.  LLM PROMPT TEMPLATE BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_llm_prompt(participant: pd.Series, windows: dict, signal_table: pd.DataFrame) -> str:
    """
    Fill in User Story #1 LLM Prompt Template with real values from the database.

    Template:
    "You are a public health data interpreter. The participant has reported the following
    symptoms over the past 30 days: {symptom_timeseries}. Their 30-day prior baseline was:
    {baseline_summary}. In 3–4 sentences, describe the participant's symptom trajectory in
    plain language, note any symptoms that have newly appeared or resolved, and recommend
    whether they should consult a healthcare provider. Do not diagnose. Use a supportive,
    non-alarming tone."
    """
    # Recent symptoms from participant record
    participant_syms = [
        SYMPTOM_LABELS[c] for c in SYMPTOM_COLS
        if participant.get(c) == 1
    ]
    if not participant_syms:
        participant_syms_str = "No symptoms reported in the current period"
    else:
        participant_syms_str = ", ".join(participant_syms)

    # Community symptom timeseries description (recent window)
    rec_active = signal_table[signal_table["recent_rate"] > 0].copy()
    if rec_active.empty:
        timeseries_str = "No community symptoms above 0% prevalence in the recent 30-day window."
    else:
        parts = [
            f"{r['label']} ({r['recent_rate']*100:.1f}% of reporters)"
            for _, r in rec_active.iterrows()
        ]
        timeseries_str = "; ".join(parts)
    timeseries_str = (
        f"Participant reported: {participant_syms_str}. "
        f"Community prevalence (postal {participant['postal_code']}, "
        f"n={windows['recent_n']} reporters, "
        f"{windows['recent_start'].date()} to {windows['max_date'].date()}): "
        f"{timeseries_str}."
    )

    # Baseline summary
    bas_active = signal_table[signal_table["baseline_rate"] > 0].copy()
    if bas_active.empty:
        baseline_str = "No symptoms reported in the 30-day baseline period."
    else:
        parts = [
            f"{r['label']} ({r['baseline_rate']*100:.1f}%)"
            for _, r in bas_active.iterrows()
        ]
        baseline_str = (
            f"Community baseline (n={windows['baseline_n']} reporters, "
            f"{windows['baseline_start'].date()} to {windows['baseline_end'].date()}): "
            + "; ".join(parts) + "."
        )

    prompt = (
        "You are a public health data interpreter.\n\n"
        f"The participant has reported the following symptoms over the past 30 days:\n"
        f"{timeseries_str}\n\n"
        f"Their 30-day prior baseline was:\n{baseline_str}\n\n"
        "In 3–4 sentences, describe the participant's symptom trajectory in plain language, "
        "note any symptoms that have newly appeared or resolved, and recommend whether they "
        "should consult a healthcare provider. Do not diagnose. "
        "Use a supportive, non-alarming tone."
    )
    return prompt


# ════════════════════════════════════════════════════════════════════════════
# VI.  SIMULATED LLM NARRATIVE (rule-based; replace with API call in production)
# ════════════════════════════════════════════════════════════════════════════

def simulate_llm_narrative(participant: pd.Series, signal_table: pd.DataFrame,
                            windows: dict) -> str:
    """
    Rule-based narrative generator that approximates what an LLM would produce.
    In production, replace with an API call:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
        message = client.messages.create(model="claude-opus-4-6", max_tokens=300,
                                         messages=[{"role":"user","content": prompt}])
        return message.content[0].text
    """
    participant_syms = [SYMPTOM_LABELS[c] for c in SYMPTOM_COLS if participant.get(c) == 1]
    new_syms  = signal_table[signal_table["signal"] == "new"]["label"].tolist()
    up_syms   = signal_table[signal_table["signal"] == "up"]["label"].tolist()
    gone_syms = signal_table[signal_table["signal"] == "gone"]["label"].tolist()
    high_acuity_present = any(c in HIGH_ACUITY for c in SYMPTOM_COLS if participant.get(c) == 1)

    name_str = f"postal area {participant['postal_code']}"
    age_str  = f"{int(participant['age'])}-year-old" if pd.notna(participant.get("age")) else ""
    sex_str  = participant.get("sex", "").lower()

    # Sentence 1: current symptom status
    if not participant_syms:
        s1 = (f"Your most recent report shows no active symptoms, which is encouraging news. "
              f"The community in {name_str} is also reporting relatively low symptom activity "
              f"(n={windows['recent_n']} reports over the past 30 days).")
    else:
        sym_list = ", ".join(participant_syms[:-1]) + (" and " + participant_syms[-1] if len(participant_syms) > 1 else participant_syms[0])
        s1 = (f"Your report indicates you are currently experiencing {sym_list}. "
              f"These symptoms are being tracked across your community ({name_str}, "
              f"n={windows['recent_n']} reporters in the past 30 days).")

    # Sentence 2: trajectory (new / resolved)
    trajectory_parts = []
    if new_syms:
        trajectory_parts.append(
            f"community data shows {', '.join(new_syms[:3])} emerging as newly reported symptoms "
            f"(not present in the prior 30-day baseline)"
        )
    if up_syms:
        trajectory_parts.append(
            f"{', '.join(up_syms[:3])} have increased above baseline levels in your area"
        )
    if gone_syms:
        trajectory_parts.append(
            f"{', '.join(gone_syms[:2])} that were present in the baseline have resolved in the community"
        )
    if trajectory_parts:
        s2 = "Comparing with the prior 30-day baseline period, " + "; ".join(trajectory_parts) + "."
    else:
        s2 = ("Symptom patterns in your community appear stable compared to the prior 30-day baseline, "
              "with no dramatic increases or newly emerging concerns.")

    # Sentence 3: recommendation
    if high_acuity_present:
        s3 = ("Given that your reported symptoms include signs that warrant clinical evaluation "
              "(such as difficulty breathing, bleeding, or jaundice), we strongly recommend "
              "contacting your healthcare provider promptly or calling the 24/7 nurse hotline.")
    elif participant_syms:
        s3 = ("If your symptoms persist beyond 3–5 days, worsen in severity, or are accompanied "
              "by high fever or difficulty breathing, please reach out to your healthcare provider "
              "for further evaluation — early assessment can provide peace of mind and appropriate care.")
    else:
        s3 = ("Continue monitoring your health and submitting weekly reports — your participation "
              "helps the community detect illness trends early, even when you are feeling well.")

    # Sentence 4: encouragement
    s4 = ("Thank you for participating in EpiHack; your report is a valuable contribution "
          "to Arizona's community health surveillance network.")

    return " ".join([s1, s2, s3, s4])


# ════════════════════════════════════════════════════════════════════════════
# VII.  VISUALIZATION — 4-PANEL DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

def fig_dashboard(participant: pd.Series, windows: dict,
                  signal_table: pd.DataFrame, narrative: str,
                  out_path: str) -> None:
    """
    4-panel dark-mode dashboard:
      Panel A — Participant symptom card (horizontal bar)
      Panel B — Community baseline vs recent comparison (grouped horizontal bar)
      Panel C — Daily community symptom burden time series
      Panel D — LLM narrative text card
    """
    plt.rcParams.update({
        "figure.facecolor"  : BG,
        "axes.facecolor"    : SURFACE,
        "axes.edgecolor"    : "#2e3150",
        "text.color"        : TEXT,
        "axes.labelcolor"   : TEXT,
        "xtick.color"       : SUBTEXT,
        "ytick.color"       : SUBTEXT,
        "grid.color"        : "#252840",
        "font.family"       : "DejaVu Sans",
        "font.size"         : 9,
    })

    fig = plt.figure(figsize=(18, 13), facecolor=BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                            left=0.06, right=0.97, top=0.91, bottom=0.05)

    ax_a = fig.add_subplot(gs[0, 0])   # participant symptoms
    ax_b = fig.add_subplot(gs[0, 1])   # baseline vs recent
    ax_c = fig.add_subplot(gs[1, 0])   # daily time series
    ax_d = fig.add_subplot(gs[1, 1])   # narrative

    # ── Title ────────────────────────────────────────────────────────────────
    sex  = participant.get("sex", "")
    age  = int(participant.get("age", 0)) if pd.notna(participant.get("age")) else "?"
    pc   = participant.get("postal_code", "")
    occ  = participant.get("occupation", "")
    uid  = participant.get("unique_id", "")
    fig.suptitle(
        f"EpiHack — Participant Symptom Trajectory Report\n"
        f"Participant: ID {uid}  |  {sex}, Age {age}  |  Postal Code {pc}  |  {occ}",
        fontsize=13, fontweight="bold", color=TEXT, y=0.975
    )

    # ── Panel A: Participant symptom bar chart ────────────────────────────────
    p_vals  = [float(participant.get(c, 0) or 0) for c in SYMPTOM_COLS]
    p_labels= [SYMPTOM_LABELS[c] for c in SYMPTOM_COLS]
    colors_a= [ACCENT2 if v == 1 else (GREEN if v == 0 else YELLOW) for v in p_vals]

    y_pos = np.arange(len(SYMPTOM_COLS))
    ax_a.barh(y_pos, p_vals, color=colors_a, height=0.6, zorder=2)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(p_labels, fontsize=8)
    ax_a.set_xlim(-0.05, 1.3)
    ax_a.set_xlabel("Reported (1 = YES)", fontsize=8)
    ax_a.set_title("Panel A — Participant Symptom Report", color=ACCENT, fontsize=10, pad=8)
    ax_a.axvline(0.5, color=SUBTEXT, lw=0.5, ls="--", zorder=1)
    ax_a.grid(axis="x", alpha=0.3, zorder=0)

    # Legend
    patches = [
        mpatches.Patch(color=ACCENT2, label="YES (reported)"),
        mpatches.Patch(color=GREEN,   label="NO (not reported)"),
        mpatches.Patch(color=YELLOW,  label="UNKNOWN"),
    ]
    ax_a.legend(handles=patches, fontsize=7, loc="lower right",
                facecolor=SURFACE, edgecolor=ACCENT, labelcolor=TEXT)

    # ── Panel B: Baseline vs Recent community comparison ──────────────────────
    st = signal_table.copy()
    # Filter to symptoms with any signal
    st_active = st[(st["recent_rate"] > 0) | (st["baseline_rate"] > 0)].copy()
    if st_active.empty:
        ax_b.text(0.5, 0.5, "No community symptoms\ndetected in either window",
                  ha="center", va="center", color=SUBTEXT, fontsize=10,
                  transform=ax_b.transAxes)
    else:
        st_active = st_active.sort_values("pct_change", ascending=True, key=lambda x: x.fillna(-9999))
        y2     = np.arange(len(st_active))
        bwidth = 0.35
        ax_b.barh(y2 - bwidth/2, st_active["baseline_rate"]*100, height=bwidth,
                  color=SUBTEXT, alpha=0.7, label="Baseline (prior 30d)", zorder=2)
        ax_b.barh(y2 + bwidth/2, st_active["recent_rate"]*100,   height=bwidth,
                  color=ACCENT,  alpha=0.9, label="Recent (last 30d)",    zorder=2)
        ax_b.set_yticks(y2)
        ax_b.set_yticklabels(st_active["label"].tolist(), fontsize=8)
        ax_b.set_xlabel("Community Prevalence (%)", fontsize=8)
        ax_b.set_title("Panel B — Community: Baseline vs Recent (Postal "+pc+")",
                       color=ACCENT, fontsize=10, pad=8)
        ax_b.grid(axis="x", alpha=0.3, zorder=0)
        ax_b.legend(fontsize=7, facecolor=SURFACE, edgecolor=ACCENT, labelcolor=TEXT)

        # Signal badges
        sig_colors = {"up": ACCENT2, "new": GREEN, "dn": YELLOW, "nc": SUBTEXT,
                      "gone": "#aaaaaa", "indef": "#555555"}
        for i, (_, row) in enumerate(st_active.iterrows()):
            sig = row["signal"]
            col = sig_colors.get(sig, SUBTEXT)
            ax_b.text(ax_b.get_xlim()[1]*0.98, i, sig.upper(),
                      ha="right", va="center", fontsize=6, color=col, fontweight="bold")

    # ── Panel C: Daily community burden time series ───────────────────────────
    daily = compute_daily_series(windows["community"], windows["recent_start"], windows["max_date"])
    daily = daily.sort_values("date_of_report")

    if len(daily) > 0:
        x_dates = daily["date_of_report"]
        y_pct   = daily["pct"]

        ax_c.fill_between(x_dates, y_pct, alpha=0.3, color=ACCENT, zorder=2)
        ax_c.plot(x_dates, y_pct, color=ACCENT, lw=2, zorder=3)

        # Shade windows
        ax_c.axvspan(windows["baseline_start"], windows["baseline_end"],
                     alpha=0.12, color=YELLOW, label="Baseline window")
        ax_c.axvspan(windows["recent_start"], windows["max_date"],
                     alpha=0.12, color=GREEN, label="Recent window")

        # Participant report date
        rdate = participant.get("date_of_report")
        if pd.notna(rdate):
            ax_c.axvline(rdate, color=ACCENT2, lw=1.5, ls="--", label="Participant report date", zorder=4)

        ax_c.set_xlabel("Date", fontsize=8)
        ax_c.set_ylabel("% Reporters with ≥1 Symptom", fontsize=8)
        ax_c.set_title(f"Panel C — Daily Community Symptom Burden (Postal {pc})",
                       color=ACCENT, fontsize=10, pad=8)
        ax_c.grid(alpha=0.3, zorder=0)
        ax_c.legend(fontsize=7, facecolor=SURFACE, edgecolor=ACCENT, labelcolor=TEXT)
        plt.setp(ax_c.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    else:
        ax_c.text(0.5, 0.5, "Insufficient daily data\nfor this postal code",
                  ha="center", va="center", color=SUBTEXT, transform=ax_c.transAxes)
        ax_c.set_title(f"Panel C — Daily Community Burden (Postal {pc})",
                       color=ACCENT, fontsize=10, pad=8)

    # ── Panel D: LLM Narrative card ───────────────────────────────────────────
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.axis("off")
    ax_d.set_facecolor(SURFACE)

    # Background box
    rect = FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                          boxstyle="round,pad=0.02", linewidth=1.5,
                          edgecolor=ACCENT, facecolor="#12152a")
    ax_d.add_patch(rect)

    ax_d.text(0.5, 0.93, "Panel D — LLM-Generated Health Narrative",
              ha="center", va="top", fontsize=10, fontweight="bold",
              color=ACCENT, transform=ax_d.transAxes)
    ax_d.text(0.5, 0.87, "(Simulated — replace with live API call in production)",
              ha="center", va="top", fontsize=7, color=SUBTEXT, style="italic",
              transform=ax_d.transAxes)

    wrapped = textwrap.fill(narrative, width=80)
    ax_d.text(0.06, 0.78, wrapped,
              ha="left", va="top", fontsize=8.5, color=TEXT,
              transform=ax_d.transAxes, wrap=True,
              linespacing=1.6)

    # Data quality note
    note = (f"Data: epihack-mini.db  |  "
            f"Recent window: {windows['recent_start'].date()} – {windows['max_date'].date()} "
            f"(n={windows['recent_n']})  |  "
            f"Baseline: {windows['baseline_start'].date()} – {windows['baseline_end'].date()} "
            f"(n={windows['baseline_n']})")
    ax_d.text(0.5, 0.04, note,
              ha="center", va="bottom", fontsize=6.5, color=SUBTEXT,
              transform=ax_d.transAxes)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Dashboard saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# VIII.  CONSOLE REPORT
# ════════════════════════════════════════════════════════════════════════════

def print_report(participant: pd.Series, windows: dict,
                 signal_table: pd.DataFrame, prompt: str, narrative: str) -> None:
    sep = "═" * 72
    print(f"\n{sep}")
    print("  EpiHack — Participant Symptom Trajectory Report")
    print(sep)
    print(f"  Participant ID : {participant['unique_id']}")
    print(f"  Age / Sex      : {participant.get('age', '?')} / {participant.get('sex', '?')}")
    print(f"  Occupation     : {participant.get('occupation', '?')}")
    print(f"  Postal Code    : {participant.get('postal_code', '?')}")
    print(f"  County         : {participant.get('county', '?')}")
    print(f"  Report Date    : {participant.get('date_of_report', '?')}")
    print()

    # Participant symptoms
    p_syms = [SYMPTOM_LABELS[c] for c in SYMPTOM_COLS if participant.get(c) == 1]
    print(f"  Reported Symptoms : {', '.join(p_syms) if p_syms else 'None'}")
    print()

    # Signal table
    print(f"  {'SYMPTOM':<35} {'BASELINE':>10} {'RECENT':>10} {'Δ%':>10}  SIG")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}  ---")
    for _, row in signal_table.iterrows():
        b = f"{row['baseline_rate']*100:.1f}%" if not np.isnan(row['baseline_rate']) else "—"
        r = f"{row['recent_rate']*100:.1f}%"   if not np.isnan(row['recent_rate'])   else "—"
        d = f"{row['pct_change']:+.1f}%"        if not np.isnan(row['pct_change'] or np.nan) else "—"
        print(f"  {row['label']:<35} {b:>10} {r:>10} {d:>10}  {row['signal'].upper()}")
    print()

    # LLM prompt
    print(f"  {'─'*68}")
    print("  LLM PROMPT TEMPLATE (filled):")
    print(f"  {'─'*68}")
    for line in prompt.split("\n"):
        print(f"  {line}")
    print()

    # Narrative
    print(f"  {'─'*68}")
    print("  SIMULATED LLM NARRATIVE:")
    print(f"  {'─'*68}")
    for line in textwrap.wrap(narrative, 68):
        print(f"  {line}")
    print(f"\n{sep}\n")


# ════════════════════════════════════════════════════════════════════════════
# IX.  CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EpiHack — Participant Symptom Trajectory Report (User Story #1)"
    )
    p.add_argument("--db",  default="epihack-mini.db",
                   help="Path to epihack SQLite database (default: epihack-mini.db)")
    p.add_argument("--uid", type=int, default=None,
                   help="Participant unique_id to analyze (default: auto-select)")
    p.add_argument("--out", default="figures/",
                   help="Output directory for PNG figures (default: figures/)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/6] Loading data from: {args.db}")
    df = load_data(args.db)
    print(f"      {len(df):,} records loaded | {df['date_of_report'].min().date()} – {df['date_of_report'].max().date()}")

    print(f"[2/6] Selecting participant...")
    if args.uid is not None:
        participant = get_participant(df, args.uid)
        print(f"      Participant uid={args.uid} found.")
    else:
        participant = default_participant(df)
        print(f"      Auto-selected uid={participant['unique_id']} "
              f"(postal {participant['postal_code']}, "
              f"report date {participant['date_of_report'].date()}).")

    print(f"[3/6] Computing community windows for postal {participant['postal_code']}...")
    windows = compute_windows(df, participant["postal_code"])
    print(f"      Recent   : {windows['recent_start'].date()} – {windows['max_date'].date()} "
          f"(n={windows['recent_n']})")
    print(f"      Baseline : {windows['baseline_start'].date()} – {windows['baseline_end'].date()} "
          f"(n={windows['baseline_n']})")

    print("[4/6] Building signal table...")
    signal_table = build_signal_table(windows["recent_rates"], windows["baseline_rates"])

    print("[5/6] Filling LLM prompt template and generating narrative...")
    prompt    = build_llm_prompt(participant, windows, signal_table)
    narrative = simulate_llm_narrative(participant, signal_table, windows)

    print_report(participant, windows, signal_table, prompt, narrative)

    print("[6/6] Rendering 4-panel dashboard...")
    out_path = str(out_dir / "symptom_trajectory_dashboard.png")
    fig_dashboard(participant, windows, signal_table, narrative, out_path)

    print("\nDone. Outputs written to:", out_dir)


if __name__ == "__main__":
    main()
