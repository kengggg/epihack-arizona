"""
emerging_signal_alert.py
=========================
EpiHack — Emerging Symptom Signal Alert (User Story #3)
---------------------------------------------------------
Pipeline:
  1. Load epihack-mini.db (SQLite)
  2. Compute per-postal-code, per-symptom signal classification:
       recent window  : last 7 days (anchored to dataset max date)
       baseline window: prior 30 days
       signal NEW     : baseline_rate == 0 AND recent_rate > 0
  3. Rank NEW signals by community size and emergence rate
  4. Identify a participant who reported the top NEW-signal symptom
  5. Fill the LLM Prompt Template:
       "The participant reported {new_symptom} today. Community surveillance
        data shows this symptom has shifted from 0% to {current_rate}%
        prevalence in postal area {postal_code} in the past 7 days
        (signal: NEW). Draft a 2–3 sentence alert message for the
        participant explaining the emerging pattern, recommending practical
        precautions (hygiene, ventilation, masking), and advising them to
        monitor symptoms. Do not alarm unnecessarily."
  6. Simulate LLM alert message (rule-based; swap for API call in production)
  7. Render 4-panel alert dashboard → PNG

Usage
-----
    python emerging_signal_alert.py \\
        --db epihack-mini.db [--postal 85301] [--symptom diarrhea] [--out figures/]

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
import matplotlib.dates as mdates

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
# Symptom-specific precautions
PRECAUTIONS = {
    "cough_congestion":              ("cover coughs with a tissue or elbow",
                                      "increase indoor ventilation",
                                      "consider wearing a mask in crowded settings"),
    "nauseas_vomiting":              ("wash hands thoroughly after any illness episode",
                                      "avoid sharing food or utensils",
                                      "stay hydrated with clear fluids"),
    "difficulty_breathing":          ("seek medical evaluation promptly",
                                      "avoid strenuous activity",
                                      "stay in well-ventilated spaces"),
    "sore_throat":                   ("practice hand hygiene",
                                      "avoid sharing drinks or utensils",
                                      "consider wearing a mask in group settings"),
    "rash":                          ("avoid touching or scratching affected skin",
                                      "wash hands before and after contact",
                                      "avoid sharing clothing or towels"),
    "fever":                         ("rest and stay hydrated",
                                      "monitor temperature and seek care if it exceeds 103 °F",
                                      "avoid close contact with others while febrile"),
    "chills":                        ("rest and keep warm",
                                      "stay hydrated",
                                      "monitor for fever development"),
    "diarrhea":                      ("wash hands thoroughly with soap after every bathroom visit",
                                      "avoid preparing food for others",
                                      "stay hydrated with oral rehydration fluids"),
    "bleeding_from_body_openings":   ("seek immediate clinical evaluation",
                                      "do not delay contacting a healthcare provider",
                                      "note any additional symptoms to report to your provider"),
    "red_eyes":                      ("avoid touching your eyes",
                                      "wash hands frequently",
                                      "avoid sharing towels, pillows, or eye drops"),
    "muscle_or_body_aches_and_pains":("rest adequately",
                                      "stay hydrated",
                                      "use appropriate OTC pain relief as directed"),
    "discolored_or_bloody_urine":    ("increase fluid intake",
                                      "contact your healthcare provider promptly",
                                      "note the color and duration for your provider"),
    "loss_of_smell_or_taste":        ("consider self-isolation until evaluated",
                                      "ventilate your home well",
                                      "inform household members and monitor for their symptoms"),
    "yellow_skin_yellow_eyes":       ("seek prompt medical evaluation",
                                      "avoid alcohol and unnecessary medications",
                                      "rest and stay hydrated"),
}
HIGH_ACUITY = {
    "difficulty_breathing", "bleeding_from_body_openings",
    "yellow_skin_yellow_eyes", "discolored_or_bloody_urine",
}

# ── Design tokens (alert-mode palette) ───────────────────────────────────────
BG      = "#0d0f18"
SURFACE = "#1a1d27"
SURFACE2= "#12152a"
ALERT   = "#ff4757"          # alert red
ORANGE  = "#ffa502"          # warning amber
ACCENT  = "#6c63ff"
GREEN   = "#43e97b"
YELLOW  = "#f9ca24"
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
# II.  SIGNAL SCANNER
# ════════════════════════════════════════════════════════════════════════════

def scan_new_signals(df: pd.DataFrame,
                     recent_days: int = 7,
                     baseline_days: int = 30) -> pd.DataFrame:
    """
    Scan every postal × symptom combination for NEW signals.

    A NEW signal satisfies:
      • baseline_rate == 0  (symptom absent in prior baseline_days)
      • recent_rate   >  0  (symptom newly present in last recent_days)
      • recent_n      >= 2  (minimum reporters for rate validity)
      • baseline_n    >= 5  (baseline must be large enough to be meaningful)

    Returns a DataFrame ranked by community size then recent_rate.
    """
    max_date    = df["date_of_report"].max()
    rec_start   = max_date - pd.Timedelta(days=recent_days - 1)
    bas_end     = rec_start - pd.Timedelta(days=1)
    bas_start   = bas_end   - pd.Timedelta(days=baseline_days - 1)

    recent_df   = df[df["date_of_report"] >= rec_start]
    baseline_df = df[(df["date_of_report"] >= bas_start) &
                     (df["date_of_report"] <= bas_end)]

    rows = []
    for pc in df["postal_code"].unique():
        rec_pc = recent_df[recent_df["postal_code"] == pc]
        bas_pc = baseline_df[baseline_df["postal_code"] == pc]
        if len(rec_pc) < 2 or len(bas_pc) < 5:
            continue
        for col in SYMPTOM_COLS:
            rec_valid = rec_pc[col].dropna()
            bas_valid = bas_pc[col].dropna()
            if len(rec_valid) == 0 or len(bas_valid) == 0:
                continue
            rec_rate = float(rec_valid.sum() / len(rec_valid))
            bas_rate = float(bas_valid.sum() / len(bas_valid))
            if bas_rate == 0 and rec_rate > 0:
                county = rec_pc["county"].iloc[0] if len(rec_pc) else ""
                rows.append({
                    "postal_code"  : pc,
                    "county"       : county,
                    "symptom"      : col,
                    "label"        : SYMPTOM_LABELS[col],
                    "recent_rate"  : rec_rate,
                    "recent_n"     : len(rec_pc),
                    "baseline_n"   : len(bas_pc),
                    "hi_acuity"    : col in HIGH_ACUITY,
                    "rec_start"    : rec_start,
                    "bas_start"    : bas_start,
                    "bas_end"      : bas_end,
                    "max_date"     : max_date,
                })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    # Rank: high-acuity first, then by community size, then by rate
    out["rank"] = (out["hi_acuity"].astype(int) * 10000 +
                   out["recent_n"] * 100 +
                   out["recent_rate"] * 100)
    return out.sort_values("rank", ascending=False).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# III.  PARTICIPANT LOOKUP
# ════════════════════════════════════════════════════════════════════════════

def find_participant(df: pd.DataFrame, postal: str,
                     symptom: str, rec_start: pd.Timestamp) -> pd.Series:
    """Return a participant in the postal area who reported the NEW symptom."""
    sub = df[(df["postal_code"] == postal) &
             (df["date_of_report"] >= rec_start) &
             (df[symptom] == 1)]
    if sub.empty:
        # Fallback: any recent reporter in that postal
        sub = df[(df["postal_code"] == postal) &
                 (df["date_of_report"] >= rec_start)]
    return sub.iloc[0] if not sub.empty else df.iloc[0]


# ════════════════════════════════════════════════════════════════════════════
# IV.  EMERGENCE TIMELINE
# ════════════════════════════════════════════════════════════════════════════

def build_timeline(df: pd.DataFrame, postal: str, symptom: str,
                   bas_start: pd.Timestamp, max_date: pd.Timestamp) -> pd.DataFrame:
    """
    Build a daily prevalence rate for the symptom in the postal code
    spanning the baseline + recent windows.
    """
    community = df[(df["postal_code"] == postal) &
                   (df["date_of_report"] >= bas_start) &
                   (df["date_of_report"] <= max_date)].copy()
    def daily_rate(group):
        valid = group[symptom].dropna()
        return float(valid.sum() / len(valid)) if len(valid) else np.nan

    timeline = (community.groupby("date_of_report")
                         .apply(daily_rate, include_groups=False)
                         .reset_index())
    timeline.columns = ["date", "rate"]
    return timeline.sort_values("date")


# ════════════════════════════════════════════════════════════════════════════
# V.  LLM PROMPT TEMPLATE BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_llm_prompt(signal: dict, participant: pd.Series) -> str:
    """
    Fill in User Story #3 LLM Prompt Template.

    Template:
    "The participant reported {new_symptom} today. Community surveillance data
     shows this symptom has shifted from 0% to {current_rate}% prevalence in
     postal area {postal_code} in the past 7 days (signal: NEW). Draft a 2–3
     sentence alert message for the participant explaining the emerging pattern,
     recommending practical precautions (hygiene, ventilation, masking), and
     advising them to monitor symptoms. Do not alarm unnecessarily."
    """
    new_symptom  = SYMPTOM_LABELS[signal["symptom"]]
    current_rate = f"{signal['recent_rate']*100:.1f}%"
    postal_code  = signal["postal_code"]

    prompt = (
        "You are a public health alert communications specialist.\n\n"
        f"The participant reported {new_symptom} today. "
        f"Community surveillance data shows this symptom has shifted from 0% "
        f"to {current_rate} prevalence in postal area {postal_code} "
        f"in the past 7 days (signal: NEW). "
        "Draft a 2–3 sentence alert message for the participant explaining the "
        "emerging pattern, recommending practical precautions (hygiene, "
        "ventilation, masking), and advising them to monitor symptoms. "
        "Do not alarm unnecessarily.\n\n"
        f"[Context: community n={signal['recent_n']} reporters in recent window; "
        f"baseline n={signal['baseline_n']} reporters over prior 30 days; "
        f"baseline rate was 0.0% throughout; "
        f"{'HIGH-ACUITY SYMPTOM — escalate to clinical review.' if signal['hi_acuity'] else 'standard symptom monitoring protocol applies.'}"
        f"]"
    )
    return prompt


# ════════════════════════════════════════════════════════════════════════════
# VI.  SIMULATED LLM ALERT MESSAGE
# ════════════════════════════════════════════════════════════════════════════

def simulate_alert_message(signal: dict, participant: pd.Series) -> str:
    """
    Rule-based alert message approximating a 2–3 sentence LLM response.

    Production replacement:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        msg = client.messages.create(
            model="claude-opus-4-6", max_tokens=150,
            system="You are a public health alert communications specialist. "
                   "Use clear, supportive language. Do not diagnose.",
            messages=[{"role":"user","content": prompt}]
        )
        return msg.content[0].text
    """
    sym_label    = SYMPTOM_LABELS[signal["symptom"]]
    rate_str     = f"{signal['recent_rate']*100:.1f}%"
    pc           = signal["postal_code"]
    n            = signal["recent_n"]
    is_hi_acuity = signal["hi_acuity"]
    prec         = PRECAUTIONS.get(signal["symptom"],
                                   ("practice hand hygiene",
                                    "increase ventilation",
                                    "consider wearing a mask"))

    # Sentence 1: emerging pattern (non-alarming framing)
    if is_hi_acuity:
        s1 = (f"Our surveillance system has detected {sym_label} as a newly "
              f"emerging symptom in postal area {pc}, with {rate_str} of recent "
              f"reporters in your community now reporting this symptom — "
              f"a change from zero reports over the prior 30 days — and we want "
              f"to make sure you have the information you need.")
    else:
        s1 = (f"Our community health monitoring system has picked up an early signal: "
              f"{sym_label} has newly appeared among {rate_str} of reporters "
              f"in your postal area ({pc}) this week, having been absent from "
              f"community reports over the prior 30 days.")

    # Sentence 2: precautions (draw from symptom-specific list)
    p1, p2, p3 = prec
    if is_hi_acuity:
        s2 = (f"Given the nature of this symptom, we recommend contacting your "
              f"healthcare provider or calling the 24/7 health line to discuss "
              f"your symptoms; in the meantime, {p1} and {p2}.")
    else:
        s2 = (f"As a precaution, we recommend {p1}, {p2}, and {p3} — "
              f"these simple steps can reduce transmission risk while the "
              f"community pattern is still emerging.")

    # Sentence 3: monitoring guidance + data note
    n_note = f" (note: community rates are based on n={n} reporters and are preliminary)" \
              if n < 10 else ""
    s3 = (f"Please monitor your {sym_label.lower()} symptoms over the next 3–5 days "
          f"and submit a follow-up EpiHack report — your continued reporting helps "
          f"us track whether this early signal grows into a broader community trend{n_note}.")

    return " ".join([s1, s2, s3])


# ════════════════════════════════════════════════════════════════════════════
# VII.  VISUALIZATION — 4-PANEL ALERT DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

def fig_dashboard(signal: dict, all_signals: pd.DataFrame, participant: pd.Series,
                  timeline: pd.DataFrame, alert_msg: str, prompt: str,
                  out_path: str) -> None:
    """
    4-panel alert dashboard:
      Panel A — NEW signal alert banner + symptom summary card
      Panel B — Symptom emergence timeline (daily rate, baseline vs recent)
      Panel C — All NEW signals across postal codes (heatmap-style table)
      Panel D — LLM alert message card with precautions
    """
    plt.rcParams.update({
        "figure.facecolor": BG,    "axes.facecolor": SURFACE,
        "axes.edgecolor":   "#2e3150", "text.color":  TEXT,
        "axes.labelcolor":  TEXT,  "xtick.color":   SUBTEXT,
        "ytick.color":      SUBTEXT, "grid.color":   "#252840",
        "font.family":      "DejaVu Sans", "font.size": 9,
    })

    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                            left=0.06, right=0.97, top=0.88, bottom=0.05)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    sym_label = SYMPTOM_LABELS[signal["symptom"]]
    rate_str  = f"{signal['recent_rate']*100:.1f}%"
    pc        = signal["postal_code"]
    alert_col = ALERT if signal["hi_acuity"] else ORANGE

    # ── Figure title (alert-mode) ─────────────────────────────────────────────
    fig.text(0.5, 0.965,
             f"[!]  EpiHack EMERGING SIGNAL ALERT  [!]",
             ha="center", fontsize=15, fontweight="bold",
             color=alert_col)
    fig.text(0.5, 0.945,
             f"NEW signal detected: {sym_label}  |  Postal {pc}  |  "
             f"0% → {rate_str} in 7 days  |  "
             f"Community n={signal['recent_n']} reporters  |  "
             f"{'[CRITICAL] HIGH-ACUITY' if signal['hi_acuity'] else 'Standard monitoring'}",
             ha="center", fontsize=11, color=TEXT)

    # ── Panel A: Signal summary + all NEW signals bar chart ───────────────────
    if all_signals.empty:
        ax_a.text(0.5, 0.5, "No NEW signals detected", ha="center",
                  va="center", color=SUBTEXT, transform=ax_a.transAxes)
    else:
        # Top 12 NEW signals for readability
        top12 = all_signals.head(12).copy()
        labels_a = [f"{SYMPTOM_LABELS[r['symptom']]}\n({r['postal_code']})"
                    for _, r in top12.iterrows()]
        vals_a   = top12["recent_rate"].values * 100
        colors_a = [ALERT if r["hi_acuity"] else (ORANGE if r["recent_rate"] >= 0.25
                    else ACCENT) for _, r in top12.iterrows()]

        y_pos = np.arange(len(top12))
        bars  = ax_a.barh(y_pos, vals_a, color=colors_a, height=0.65, zorder=2)
        ax_a.set_yticks(y_pos)
        ax_a.set_yticklabels(labels_a, fontsize=8)
        ax_a.set_xlabel("Emergence Rate — Recent 7-day Window (%)", fontsize=8)
        ax_a.set_title("Panel A — All NEW Signals Detected\n(baseline 0% → current rate)",
                       color=alert_col, fontsize=10, pad=8)
        ax_a.grid(axis="x", alpha=0.3, zorder=0)
        ax_a.axvline(0, color=SUBTEXT, lw=0.8)

        for bar, val in zip(bars, vals_a):
            ax_a.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                      f"{val:.1f}%", va="center", fontsize=7.5, color=SUBTEXT)

        # Highlight selected signal
        for i, (_, row) in enumerate(top12.iterrows()):
            if (row["postal_code"] == pc and row["symptom"] == signal["symptom"]):
                ax_a.get_yticklabels()[i].set_color(alert_col)
                ax_a.get_yticklabels()[i].set_fontweight("bold")

        legend_els = [
            mpatches.Patch(color=ALERT,  label="High-acuity symptom"),
            mpatches.Patch(color=ORANGE, label="Elevated rate (≥25%)"),
            mpatches.Patch(color=ACCENT, label="Standard rate"),
        ]
        ax_a.legend(handles=legend_els, fontsize=7, facecolor=SURFACE,
                    edgecolor=alert_col, labelcolor=TEXT)

    # ── Panel B: Emergence timeline ───────────────────────────────────────────
    if not timeline.empty:
        dates    = timeline["date"]
        rates    = timeline["rate"].values * 100
        rec_start = signal["rec_start"]

        # Split baseline vs recent
        mask_bas = dates < rec_start
        mask_rec = dates >= rec_start

        ax_b.fill_between(dates[mask_bas], rates[mask_bas],
                          alpha=0.25, color=ACCENT, zorder=2, label="Baseline window")
        ax_b.plot(dates[mask_bas], rates[mask_bas],
                  color=ACCENT, lw=1.5, alpha=0.7, zorder=3)
        ax_b.fill_between(dates[mask_rec], rates[mask_rec],
                          alpha=0.55, color=alert_col, zorder=4, label="Recent window")
        ax_b.plot(dates[mask_rec], rates[mask_rec],
                  color=alert_col, lw=2.5, zorder=5)

        # Emergence line
        ax_b.axvline(rec_start, color=YELLOW, lw=1.5, ls="--",
                     label="Emergence threshold", zorder=6)

        # Annotate peak
        peak_idx = np.nanargmax(rates)
        peak_val = rates[peak_idx]
        if peak_val > 0:
            peak_date = dates.iloc[peak_idx]
            ax_b.annotate(
                f"Peak: {peak_val:.1f}%",
                xy=(peak_date, peak_val),
                xytext=(peak_date, peak_val + max(rates) * 0.12),
                fontsize=8, color=alert_col, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=alert_col, lw=1.2)
            )

        # Baseline zero-line
        ax_b.axhline(0, color=GREEN, lw=0.8, ls=":", alpha=0.6, label="Zero baseline")

        ax_b.set_xlabel("Date", fontsize=8)
        ax_b.set_ylabel(f"{sym_label} Prevalence (%)", fontsize=8)
        ax_b.set_title(
            f"Panel B — Symptom Emergence Timeline\n"
            f"{sym_label} · Postal {pc} · "
            f"{signal['bas_start'].date()} – {signal['max_date'].date()}",
            color=alert_col, fontsize=10, pad=8
        )
        ax_b.set_ylim(bottom=-2)
        ax_b.grid(alpha=0.3, zorder=0)
        ax_b.legend(fontsize=7, facecolor=SURFACE, edgecolor=alert_col, labelcolor=TEXT)
        plt.setp(ax_b.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
        ax_b.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    else:
        ax_b.text(0.5, 0.5, "Insufficient timeline data",
                  ha="center", va="center", color=SUBTEXT, transform=ax_b.transAxes)
        ax_b.set_title("Panel B — Emergence Timeline", color=alert_col, fontsize=10)

    # ── Panel C: NEW signals matrix (postal × symptom) ────────────────────────
    ax_c.set_facecolor(SURFACE)
    ax_c.axis("off")

    if not all_signals.empty:
        matrix_data = all_signals.copy()
        postal_list = sorted(matrix_data["postal_code"].unique())
        symp_set    = matrix_data["symptom"].unique()
        symp_uniq   = [s for s in SYMPTOM_COLS if s in symp_set]
        symp_labels = [SYMPTOM_LABELS[s] for s in symp_uniq]

        # Build grid
        grid = np.full((len(postal_list), len(symp_uniq)), np.nan)
        for _, row in matrix_data.iterrows():
            pi = postal_list.index(row["postal_code"])
            si = symp_uniq.index(row["symptom"]) if row["symptom"] in symp_uniq else -1
            if si >= 0:
                grid[pi, si] = row["recent_rate"] * 100

        # Draw as colored table
        cell_h = 0.85 / max(len(postal_list), 1)
        cell_w = 0.90 / max(len(symp_uniq), 1)
        x0, y0 = 0.05, 0.88

        # Header row
        ax_c.text(0.025, y0 + 0.025, "Postal ↓  Symptom →",
                  va="center", fontsize=6.5, color=SUBTEXT, transform=ax_c.transAxes)
        for j, slbl in enumerate(symp_labels):
            ax_c.text(x0 + j * cell_w + cell_w / 2, y0 + 0.03,
                      slbl, ha="center", va="bottom", fontsize=5.5,
                      color=SUBTEXT, rotation=40, transform=ax_c.transAxes)

        for i, pc_row in enumerate(postal_list):
            yc = y0 - (i + 1) * cell_h
            highlight = (pc_row == pc)
            ax_c.text(0.025, yc + cell_h / 2, pc_row,
                      va="center", fontsize=6.5, fontweight="bold" if highlight else "normal",
                      color=alert_col if highlight else TEXT, transform=ax_c.transAxes)
            for j in range(len(symp_uniq)):
                val = grid[i, j]
                xc  = x0 + j * cell_w
                rect_color = (ALERT + "cc") if (not np.isnan(val) and val >= 40) else \
                             (ORANGE + "cc") if (not np.isnan(val) and val >= 20) else \
                             (ACCENT + "99") if not np.isnan(val) else SURFACE2
                rect = plt.Rectangle(
                    (xc, yc), cell_w * 0.9, cell_h * 0.85,
                    transform=ax_c.transAxes, clip_on=False,
                    facecolor=rect_color, edgecolor="#2e3150", linewidth=0.4
                )
                ax_c.add_patch(rect)
                if not np.isnan(val):
                    ax_c.text(xc + cell_w * 0.45, yc + cell_h * 0.42,
                               f"{val:.0f}%", ha="center", va="center",
                               fontsize=5.5, color=TEXT, fontweight="bold",
                               transform=ax_c.transAxes)

        ax_c.set_title("Panel C — NEW Signal Matrix (Postal × Symptom)",
                       color=alert_col, fontsize=10, pad=8)

        # Legend
        legend_items = [
            mpatches.Patch(color=ALERT   + "cc", label="≥40% rate"),
            mpatches.Patch(color=ORANGE  + "cc", label="20–39% rate"),
            mpatches.Patch(color=ACCENT  + "99", label="<20% rate"),
            mpatches.Patch(color=SURFACE2,        label="No NEW signal"),
        ]
        ax_c.legend(handles=legend_items, fontsize=6.5, loc="lower center",
                    facecolor=SURFACE, edgecolor=SUBTEXT, labelcolor=TEXT,
                    ncol=4, bbox_to_anchor=(0.5, -0.02))
    else:
        ax_c.text(0.5, 0.5, "No NEW signals found\nin current windows",
                  ha="center", va="center", color=SUBTEXT, fontsize=11,
                  transform=ax_c.transAxes)
        ax_c.set_title("Panel C — NEW Signal Matrix", color=alert_col, fontsize=10)

    # ── Panel D: LLM alert message card ──────────────────────────────────────
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.axis("off")

    # Alert-mode card with red border
    card = FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                          boxstyle="round,pad=0.02", linewidth=2.5,
                          edgecolor=alert_col, facecolor=SURFACE2)
    ax_d.add_patch(card)

    # Alert icon strip
    icon_strip = plt.Rectangle((0.01, 0.88), 0.98, 0.10,
                                transform=ax_d.transAxes, clip_on=False,
                                facecolor=alert_col + "33", edgecolor="none")
    ax_d.add_patch(icon_strip)
    ax_d.text(0.5, 0.935, f"[!]  EMERGING SIGNAL ALERT — {sym_label.upper()}  [!]",
              ha="center", va="center", fontsize=9.5, fontweight="bold",
              color=alert_col, transform=ax_d.transAxes)

    # Template fields
    ax_d.text(0.06, 0.84,
              f"{{new_symptom}} → {sym_label}   "
              f"{{current_rate}} → {rate_str}   "
              f"{{postal_code}} → {pc}",
              ha="left", va="top", fontsize=7.5, color=SUBTEXT,
              transform=ax_d.transAxes)

    # Alert message
    ax_d.text(0.06, 0.78,
              "Panel D — LLM Alert Message (2–3 sentences):",
              ha="left", va="top", fontsize=9, fontweight="bold",
              color=alert_col, transform=ax_d.transAxes)

    wrapped = textwrap.fill(alert_msg, width=78)
    ax_d.text(0.06, 0.71, wrapped,
              ha="left", va="top", fontsize=8, color=TEXT,
              transform=ax_d.transAxes, linespacing=1.6)

    # Precautions
    prec = PRECAUTIONS.get(signal["symptom"], ("practice hygiene", "ventilate spaces", "monitor symptoms"))
    ax_d.text(0.06, 0.28, "Recommended Precautions:",
              ha="left", va="top", fontsize=8.5, fontweight="bold",
              color=YELLOW, transform=ax_d.transAxes)
    for k, p in enumerate(prec):
        ax_d.text(0.06, 0.22 - k * 0.065, f"  {k+1}. {p}",
                  ha="left", va="top", fontsize=8, color=TEXT,
                  transform=ax_d.transAxes)

    # Data note
    ax_d.text(0.5, 0.03,
              f"Signal: NEW  |  Baseline: 0.0% (n={signal['baseline_n']}, 30d)  "
              f"→  Recent: {rate_str} (n={signal['recent_n']}, 7d)  |  "
              f"Priority: CRITICAL",
              ha="center", va="bottom", fontsize=6.5, color=SUBTEXT,
              transform=ax_d.transAxes)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Dashboard saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# VIII.  CONSOLE REPORT
# ════════════════════════════════════════════════════════════════════════════

def print_report(signal: dict, all_signals: pd.DataFrame, participant: pd.Series,
                 prompt: str, alert_msg: str) -> None:
    sep  = "═" * 72
    warn = "[!] " * 10
    print(f"\n{warn}")
    print(f"  EpiHack — EMERGING SIGNAL ALERT  (User Story #3 · CRITICAL)")
    print(f"{warn}")
    print(f"  Signal Type      : NEW (baseline 0% → {signal['recent_rate']*100:.1f}%)")
    print(f"  Symptom          : {SYMPTOM_LABELS[signal['symptom']]}")
    print(f"  Postal Code      : {signal['postal_code']} ({signal['county']} County)")
    print(f"  Recent Window    : {signal['rec_start'].date()} – {signal['max_date'].date()} (n={signal['recent_n']})")
    print(f"  Baseline Window  : {signal['bas_start'].date()} – {signal['bas_end'].date()} (n={signal['baseline_n']})")
    print(f"  High-Acuity      : {'YES — escalate to clinical review' if signal['hi_acuity'] else 'No'}")
    print()
    print(f"  Participant ID   : {participant['unique_id']}")
    print(f"  Age / Sex        : {participant.get('age','?')} / {participant.get('sex','?')}")
    print(f"  Occupation       : {participant.get('occupation','?')}")
    print(f"  Report Date      : {participant.get('date_of_report','?')}")
    print()
    print(f"  Total NEW signals found across all postal codes: {len(all_signals)}")
    print(f"  Postal codes with ≥1 NEW signal: {all_signals['postal_code'].nunique()}")
    print()
    print(f"  {'─'*68}")
    print("  LLM PROMPT TEMPLATE (filled):")
    print(f"  {'─'*68}")
    for line in prompt.split("\n"):
        print(f"  {line}")
    print()
    print(f"  {'─'*68}")
    print("  SIMULATED LLM ALERT MESSAGE:")
    print(f"  {'─'*68}")
    for line in textwrap.wrap(alert_msg, 68):
        print(f"  {line}")
    print(f"\n{sep}\n")


# ════════════════════════════════════════════════════════════════════════════
# IX.  CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EpiHack — Emerging Symptom Signal Alert (User Story #3)"
    )
    p.add_argument("--db",      default="epihack-mini.db",
                   help="Path to epihack SQLite database")
    p.add_argument("--postal",  default=None,
                   help="Force a specific postal code for analysis")
    p.add_argument("--symptom", default=None,
                   help="Force a specific symptom column name")
    p.add_argument("--out",     default="figures/",
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

    print("[2/6] Scanning for NEW signals (baseline 0% → recent >0%)...")
    all_signals = scan_new_signals(df, recent_days=7, baseline_days=30)
    if all_signals.empty:
        print("      No NEW signals found. Adjust window parameters.")
        return
    print(f"      Found {len(all_signals)} NEW signal(s) across "
          f"{all_signals['postal_code'].nunique()} postal code(s).")

    # Select target signal
    if args.postal and args.symptom:
        mask = ((all_signals["postal_code"] == args.postal) &
                (all_signals["symptom"] == args.symptom))
        if mask.any():
            signal = all_signals[mask].iloc[0].to_dict()
        else:
            print(f"      Specified postal/symptom not in NEW signals; using top signal.")
            signal = all_signals.iloc[0].to_dict()
    else:
        signal = all_signals.iloc[0].to_dict()

    print(f"      Selected: {SYMPTOM_LABELS[signal['symptom']]} "
          f"in postal {signal['postal_code']} "
          f"({signal['recent_rate']*100:.1f}%, n={signal['recent_n']})")

    print("[3/6] Finding participant who reported this symptom...")
    participant = find_participant(df, signal["postal_code"],
                                   signal["symptom"], signal["rec_start"])
    print(f"      uid={participant['unique_id']} "
          f"({participant.get('sex','?')}, age {participant.get('age','?')}, "
          f"postal {participant['postal_code']})")

    print("[4/6] Building emergence timeline...")
    timeline = build_timeline(df, signal["postal_code"], signal["symptom"],
                               signal["bas_start"], signal["max_date"])
    print(f"      Timeline points: {len(timeline)}")

    print("[5/6] Filling LLM prompt template and generating alert message...")
    prompt    = build_llm_prompt(signal, participant)
    alert_msg = simulate_alert_message(signal, participant)
    print_report(signal, all_signals, participant, prompt, alert_msg)

    print("[6/6] Rendering 4-panel alert dashboard...")
    out_path = str(out_dir / "emerging_signal_alert_dashboard.png")
    fig_dashboard(signal, all_signals, participant, timeline,
                  alert_msg, prompt, out_path)

    print(f"\nDone. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
