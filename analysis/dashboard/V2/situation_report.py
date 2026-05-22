"""
situation_report.py
====================
EpiHack — User Story #5: Epidemiological Intelligence Situation Report (SITREP)

LLM Prompt Template (filled at runtime):
    "You are an epidemiological intelligence analyst. Based on the following EpiHack
     signal table for the past {N} days: {signal_table} (columns: postal_code, symptom,
     recent_rate, baseline_rate, pct_change, signal_class). Produce a structured situation
     report with: (1) top 5 signals by magnitude (signal='up' or 'new'); (2) postal areas
     with 3+ elevated signals; (3) any newly emerging symptoms (signal='new'); (4) a
     recommended investigation priority list. Format as a bulleted executive brief.
     Flag small sample sizes (n < 10) explicitly."

Usage:
    python situation_report.py --db epihack-mini.db [--recent 7] [--baseline 30] [--out figures/]
"""

# ============================================================
# I. IMPORTS & CONFIGURATION
# ============================================================
import argparse
import sqlite3
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# Symptom column mapping: DB column -> display label
SYMPTOM_COLS = [
    ("cough_congestion",            "Cough / Congestion"),
    ("sore_throat",                 "Sore Throat"),
    ("difficulty_breathing",        "Difficulty Breathing"),
    ("nauseas_vomiting",            "Nausea / Vomiting"),
    ("diarrhea",                    "Diarrhea"),
    ("muscle_or_body_aches_and_pains", "Muscle / Body Aches"),
    ("fever",                       "Fever"),
    ("chills",                      "Chills"),
    ("red_eyes",                    "Red Eyes"),
    ("rash",                        "Rash"),
    ("loss_of_smell_or_taste",      "Loss of Smell/Taste"),
    ("bleeding_from_body_openings", "Bleeding (Body Openings)"),
    ("yellow_skin_yellow_eyes",     "Yellow Skin / Eyes"),
    ("discolored_or_bloody_urine",  "Discolored/Bloody Urine"),
]
COL_TO_LABEL = {c: l for c, l in SYMPTOM_COLS}

# High-acuity symptoms requiring priority escalation
HIGH_ACUITY = {
    "difficulty_breathing",
    "bleeding_from_body_openings",
    "yellow_skin_yellow_eyes",
    "discolored_or_bloody_urine",
}

# Design palette (dark-mode)
BG       = "#0f1117"
SURFACE  = "#1a1d27"
SURFACE2 = "#23263a"
ACCENT   = "#6c63ff"
GREEN    = "#43e97b"
YELLOW   = "#f9ca24"
ORANGE   = "#ffa502"
RED      = "#ff4757"
MUTED    = "#8892b0"
TEXT     = "#e8eaf6"

SIG_COLORS = {
    "new":  RED,
    "up":   ORANGE,
    "nc":   ACCENT,
    "dn":   GREEN,
    "gone": "#5a5d7a",
    "indef":"#3d3f55",
}


# ============================================================
# II. DATA LOADING
# ============================================================
def load_data(db_path: str) -> pd.DataFrame:
    """Load incidents table from SQLite; map YES/NO -> 1/0 (UNKNOWN -> NaN)."""
    con = sqlite3.connect(db_path)
    df  = pd.read_sql("SELECT * FROM incidents", con)
    con.close()
    df["date_of_report"] = pd.to_datetime(df["date_of_report"])
    for col, _ in SYMPTOM_COLS:
        if col in df.columns:
            df[col] = df[col].map({"YES": 1, "NO": 0}).astype("Float64")
    print(f"[1/7] Data loaded: {len(df):,} records | "
          f"{df['date_of_report'].min().date()} -- {df['date_of_report'].max().date()}")
    return df


# ============================================================
# III. SIGNAL TABLE COMPUTATION
# ============================================================
def classify_signal(rec_rate, bas_rate, rec_n, bas_n,
                    min_rec=2, min_bas=5, pct_thresh=5.0) -> str:
    """
    Classify a postal × symptom pair into one of six signal classes:
      new   : 0% -> >0%  (with sufficient n)
      up    : prevalence increase > pct_thresh percentage points
      dn    : prevalence decrease > pct_thresh percentage points
      nc    : no meaningful change (±pct_thresh)
      gone  : >0% -> 0%
      indef : insufficient data for inference
    """
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


def compute_signal_table(df: pd.DataFrame,
                         recent_days: int = 7,
                         baseline_days: int = 30) -> pd.DataFrame:
    """
    Compute a full signal table across all postal_code × symptom combinations.

    Temporal anchoring: windows are fixed relative to the dataset's max date
    (not system clock) to guarantee reproducibility.

    Returns a DataFrame with columns:
        postal_code, symptom, label,
        recent_n, baseline_n,
        recent_rate, baseline_rate, pct_change,
        signal_class, high_acuity, small_sample
    """
    max_date      = df["date_of_report"].max()
    rec_end       = max_date
    rec_start     = rec_end   - pd.Timedelta(days=recent_days - 1)
    bas_end       = rec_start - pd.Timedelta(days=1)
    bas_start     = bas_end   - pd.Timedelta(days=baseline_days - 1)

    df_rec = df[(df["date_of_report"] >= rec_start) & (df["date_of_report"] <= rec_end)]
    df_bas = df[(df["date_of_report"] >= bas_start) & (df["date_of_report"] <= bas_end)]

    print(f"[3/7] Windows: recent={rec_start.date()}..{rec_end.date()} "
          f"| baseline={bas_start.date()}..{bas_end.date()}")

    rows = []
    for postal, g_rec in df_rec.groupby("postal_code"):
        g_bas = df_bas[df_bas["postal_code"] == postal]
        for col, label in SYMPTOM_COLS:
            if col not in df.columns:
                continue
            v_rec = g_rec[col].dropna()
            v_bas = g_bas[col].dropna()
            rec_n = len(v_rec)
            bas_n = len(v_bas)
            rec_rate = float(v_rec.mean() * 100) if rec_n > 0 else 0.0
            bas_rate = float(v_bas.mean() * 100) if bas_n > 0 else 0.0
            pct_change = rec_rate - bas_rate

            sig = classify_signal(rec_rate, bas_rate, rec_n, bas_n)
            rows.append({
                "postal_code":  str(postal),
                "symptom":      col,
                "label":        label,
                "recent_n":     rec_n,
                "baseline_n":   bas_n,
                "recent_rate":  round(rec_rate, 2),
                "baseline_rate": round(bas_rate, 2),
                "pct_change":   round(pct_change, 2),
                "signal_class": sig,
                "high_acuity":  col in HIGH_ACUITY,
                "small_sample": rec_n < 10,
            })

    sig_df = pd.DataFrame(rows)
    print(f"      {len(sig_df):,} postal×symptom pairs computed "
          f"| {len(df_rec['postal_code'].unique())} active postal codes")
    return sig_df


# ============================================================
# IV. SITREP ANALYSIS
# ============================================================
def build_sitrep(sig_df: pd.DataFrame) -> dict:
    """
    Extract the four SITREP components:
      1. top_signals   — top 5 'up' or 'new' by pct_change magnitude
      2. hotspot_postals — postal codes with >= 3 elevated signals
      3. new_signals   — all 'new' emerging signals
      4. priority_list — ranked investigation recommendations
    """
    elevated = sig_df[sig_df["signal_class"].isin(["up", "new"])].copy()

    # 1. Top 5 signals by magnitude
    top5 = (
        elevated
        .sort_values(["high_acuity", "pct_change"], ascending=[False, False])
        .head(5)
    )

    # 2. Postal hotspots: >= 3 elevated signals
    postal_counts = (
        elevated
        .groupby("postal_code")
        .agg(
            n_elevated=("signal_class", "count"),
            n_new=("signal_class", lambda x: (x == "new").sum()),
            n_high_acuity=("high_acuity", "sum"),
        )
        .reset_index()
        .sort_values(["n_elevated", "n_high_acuity"], ascending=False)
    )
    hotspot_postals = postal_counts[postal_counts["n_elevated"] >= 3]

    # 3. New signals only
    new_signals = (
        sig_df[sig_df["signal_class"] == "new"]
        .sort_values(["high_acuity", "recent_rate"], ascending=[False, False])
    )

    # 4. Priority list (ranked by: high-acuity + hotspot overlap + magnitude)
    def priority_score(row):
        score  = row["pct_change"] * 10
        score += 500 if row["high_acuity"] else 0
        score += 200 if row["signal_class"] == "new" else 0
        score += 100 if row["postal_code"] in hotspot_postals["postal_code"].values else 0
        return score

    elevated["priority_score"] = elevated.apply(priority_score, axis=1)
    priority_list = (
        elevated
        .sort_values("priority_score", ascending=False)
        .drop_duplicates(subset=["postal_code", "symptom"])
        .head(10)
    )

    return {
        "top5":            top5,
        "hotspot_postals": hotspot_postals,
        "new_signals":     new_signals,
        "priority_list":   priority_list,
        "n_elevated":      len(elevated),
        "n_postals_active": sig_df["postal_code"].nunique(),
    }


# ============================================================
# V. LLM PROMPT TEMPLATE FILL
# ============================================================
def build_llm_prompt(sig_df: pd.DataFrame, recent_days: int) -> str:
    """
    Fill the User Story #5 LLM prompt template with the signal table summary.
    In production, replace simulate_llm_narrative() with:

        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        narrative = response.content[0].text
    """
    # Summarize signal_table for prompt (top 20 elevated, formatted)
    elevated = sig_df[sig_df["signal_class"].isin(["up", "new"])].copy()
    table_rows = []
    for _, r in elevated.sort_values("pct_change", ascending=False).head(20).iterrows():
        flag = " [SMALL SAMPLE]" if r["small_sample"] else ""
        table_rows.append(
            f"  {r['postal_code']} | {r['label']} | "
            f"rec={r['recent_rate']:.1f}% (n={r['recent_n']}) | "
            f"bas={r['baseline_rate']:.1f}% | "
            f"delta=+{r['pct_change']:.1f}pp | {r['signal_class'].upper()}{flag}"
        )
    signal_table_str = "\n".join(table_rows)

    prompt = textwrap.dedent(f"""
        You are an epidemiological intelligence analyst. Based on the following EpiHack
        signal table for the past {recent_days} days:

        {signal_table_str}

        (columns: postal_code, symptom, recent_rate, baseline_rate, pct_change, signal_class)

        Produce a structured situation report with:
        (1) top 5 signals by magnitude (signal='UP' or 'NEW');
        (2) postal areas with 3+ elevated signals;
        (3) any newly emerging symptoms (signal='NEW');
        (4) a recommended investigation priority list.
        Format as a bulleted executive brief. Flag small sample sizes (n < 10) explicitly.
    """).strip()
    return prompt


# ============================================================
# VI. SIMULATED LLM NARRATIVE (rule-based SITREP)
# ============================================================
def simulate_llm_narrative(sitrep: dict, recent_days: int) -> str:
    """
    Rule-based structured executive brief mimicking LLM output.

    Production replacement (inline comment):
        # response = client.messages.create(
        #     model="claude-opus-4-6", max_tokens=800,
        #     messages=[{"role":"user","content": prompt}]
        # )
        # return response.content[0].text
    """
    lines = []
    lines.append(f"EPIHACK SITUATION REPORT -- {recent_days}-Day Surveillance Window")
    lines.append("=" * 60)

    # --- Section 1: Top 5 signals ---
    lines.append("\n[1] TOP 5 ELEVATED SIGNALS (by magnitude)\n")
    for rank, (_, r) in enumerate(sitrep["top5"].iterrows(), 1):
        ss_flag = "  [!] Small sample (n<10)" if r["small_sample"] else ""
        ha_flag = "  [CRITICAL] High-acuity symptom" if r["high_acuity"] else ""
        lines.append(
            f"  {rank}. {r['label']} in postal {r['postal_code']} "
            f"-- {r['signal_class'].upper()} | recent={r['recent_rate']:.1f}% "
            f"(n={r['recent_n']}) | baseline={r['baseline_rate']:.1f}% "
            f"| delta=+{r['pct_change']:.1f}pp{ss_flag}{ha_flag}"
        )

    # --- Section 2: Postal hotspots ---
    lines.append("\n[2] POSTAL HOTSPOTS (>=3 elevated signals)\n")
    hp = sitrep["hotspot_postals"]
    if hp.empty:
        lines.append("  No postal code with 3+ simultaneously elevated signals detected.")
    else:
        for _, r in hp.iterrows():
            ha_note = f" | HIGH-ACUITY signals: {int(r['n_high_acuity'])}" if r["n_high_acuity"] > 0 else ""
            lines.append(
                f"  - Postal {r['postal_code']}: {int(r['n_elevated'])} elevated signals "
                f"({int(r['n_new'])} NEW){ha_note}"
            )

    # --- Section 3: Newly emerging symptoms ---
    lines.append("\n[3] NEWLY EMERGING SYMPTOMS (signal='NEW')\n")
    ns = sitrep["new_signals"]
    if ns.empty:
        lines.append("  No newly emerging symptoms detected in this window.")
    else:
        for _, r in ns.head(8).iterrows():
            ss_flag = "  [!] Small sample (n<10)" if r["small_sample"] else ""
            ha_flag = "  [CRITICAL]" if r["high_acuity"] else ""
            lines.append(
                f"  - {r['label']} | postal {r['postal_code']} | "
                f"recent={r['recent_rate']:.1f}% (n={r['recent_n']}) "
                f"[baseline was 0%]{ss_flag}{ha_flag}"
            )

    # --- Section 4: Investigation priority list ---
    lines.append("\n[4] RECOMMENDED INVESTIGATION PRIORITY LIST\n")
    for rank, (_, r) in enumerate(sitrep["priority_list"].head(8).iterrows(), 1):
        action = (
            "Immediate clinical referral + case investigation"
            if r["high_acuity"]
            else ("Field verification + household survey"
                  if r["signal_class"] == "new"
                  else "Enhanced community surveillance")
        )
        ss_note = " [preliminary -- verify n]" if r["small_sample"] else ""
        lines.append(
            f"  {rank}. [{r['signal_class'].upper()}] {r['label']} @ {r['postal_code']}"
            f" (delta=+{r['pct_change']:.1f}pp){ss_note}"
            f"\n     -> {action}"
        )

    lines.append(f"\n  Total elevated signals: {sitrep['n_elevated']} "
                 f"across {sitrep['n_postals_active']} active postal codes.")
    lines.append("\n[END SITREP]")
    return "\n".join(lines)


# ============================================================
# VII. 4-PANEL DASHBOARD (PNG)
# ============================================================
def fig_dashboard(sig_df: pd.DataFrame, sitrep: dict,
                  recent_days: int, out_dir: Path) -> None:
    """
    Render a 4-panel static dashboard:
      A. Top 15 elevated signals — horizontal bar chart (colored by signal class)
      B. Postal hotspot chart — bar chart of n_elevated per postal
      C. New-signal matrix — postal x symptom heatmap (only signal='new')
      D. Executive brief text panel
    """
    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    fig.suptitle(
        f"EpiHack SITREP — {recent_days}-Day Surveillance Window\n"
        f"Arizona Participatory Syndromic Surveillance 2023–2026",
        fontsize=14, fontweight="bold", color=TEXT, y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.06, right=0.97,
                           top=0.92, bottom=0.06,
                           hspace=0.38, wspace=0.32)

    # ---- Panel A: Top 15 elevated signals ----
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor(SURFACE)
    elevated = (
        sig_df[sig_df["signal_class"].isin(["up", "new"])]
        .sort_values(["high_acuity", "pct_change"], ascending=[False, False])
        .head(15)
        .iloc[::-1]  # reverse for horizontal bar (highest at top)
    )
    bar_labels = [f"{r['label']}\n{r['postal_code']}" for _, r in elevated.iterrows()]
    bar_colors = [SIG_COLORS.get(r["signal_class"], MUTED) for _, r in elevated.iterrows()]
    bars = ax_a.barh(range(len(elevated)), elevated["pct_change"].values,
                     color=bar_colors, edgecolor="none", height=0.72)

    # Annotate n and small-sample flag
    for i, (_, r) in enumerate(elevated.iterrows()):
        ss = " [!]" if r["small_sample"] else ""
        ax_a.text(r["pct_change"] + 0.2, i,
                  f"n={r['recent_n']}{ss}", va="center",
                  fontsize=7.5, color=TEXT, fontweight="bold" if r["small_sample"] else "normal")

    ax_a.set_yticks(range(len(elevated)))
    ax_a.set_yticklabels(bar_labels, fontsize=7.5, color=TEXT)
    ax_a.set_xlabel("Delta prevalence (percentage points)", fontsize=8, color=MUTED)
    ax_a.set_title("A — Top Elevated Signals", fontsize=9,
                   fontweight="bold", color=ORANGE, pad=8)
    ax_a.tick_params(colors=MUTED, labelsize=8)
    ax_a.spines[:].set_visible(False)
    ax_a.tick_params(axis="x", colors=MUTED)
    ax_a.set_facecolor(SURFACE)
    for spine in ax_a.spines.values():
        spine.set_edgecolor(SURFACE2)
    # legend
    patches = [mpatches.Patch(color=SIG_COLORS["new"], label="NEW"),
               mpatches.Patch(color=SIG_COLORS["up"],  label="UP")]
    ax_a.legend(handles=patches, fontsize=7.5, loc="lower right",
                facecolor=SURFACE2, edgecolor="none", labelcolor=TEXT)

    # ---- Panel B: Postal Hotspot Chart ----
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor(SURFACE)
    hp = sitrep["hotspot_postals"].head(15)
    if hp.empty:
        ax_b.text(0.5, 0.5, "No hotspot postal codes\ndetected (threshold: 3+ signals)",
                  ha="center", va="center", color=MUTED, fontsize=10,
                  transform=ax_b.transAxes)
    else:
        x = range(len(hp))
        ax_b.bar(x, hp["n_elevated"], color=ACCENT + "bb", label="Total elevated",
                 edgecolor="none", zorder=2)
        ax_b.bar(x, hp["n_new"], color=RED + "cc", label="NEW signals",
                 edgecolor="none", zorder=3)
        ax_b.bar(x, hp["n_high_acuity"], color=YELLOW + "dd", label="High-acuity",
                 edgecolor="none", zorder=4)
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(hp["postal_code"], rotation=45, ha="right",
                              fontsize=7.5, color=TEXT)
        ax_b.set_ylabel("Number of elevated signals", fontsize=8, color=MUTED)
        ax_b.axhline(3, color=ORANGE, linestyle="--", linewidth=1, alpha=0.7,
                     label="Hotspot threshold (3)")
        ax_b.legend(fontsize=7.5, facecolor=SURFACE2, edgecolor="none", labelcolor=TEXT)
    ax_b.set_title("B — Postal Hotspot Profile", fontsize=9,
                   fontweight="bold", color=ACCENT, pad=8)
    ax_b.tick_params(colors=MUTED, labelsize=8)
    ax_b.spines[:].set_visible(False)

    # ---- Panel C: New-Signal Matrix (postal x symptom) ----
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor(SURFACE)
    ns = sig_df[sig_df["signal_class"] == "new"]
    if ns.empty:
        ax_c.text(0.5, 0.5, "No NEW signals in this window.",
                  ha="center", va="center", color=MUTED, fontsize=10,
                  transform=ax_c.transAxes)
    else:
        # Pivot: postals (rows) x symptoms (cols), value = recent_rate
        pivot = ns.pivot_table(index="postal_code", columns="label",
                               values="recent_rate", aggfunc="max", fill_value=0)
        # Limit to top 20 postals by row sum
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).head(20).index]
        mat = pivot.values
        im = ax_c.imshow(mat, aspect="auto", cmap="OrRd",
                         vmin=0, vmax=max(mat.max(), 1))
        ax_c.set_xticks(range(len(pivot.columns)))
        ax_c.set_xticklabels(pivot.columns, rotation=55, ha="right", fontsize=6.5, color=TEXT)
        ax_c.set_yticks(range(len(pivot.index)))
        ax_c.set_yticklabels(pivot.index, fontsize=7, color=TEXT)
        plt.colorbar(im, ax=ax_c, fraction=0.035, pad=0.03,
                     label="Recent Prevalence %").ax.yaxis.label.set_color(MUTED)
    ax_c.set_title("C — NEW Signal Matrix (postal x symptom)", fontsize=9,
                   fontweight="bold", color=RED, pad=8)
    ax_c.tick_params(colors=MUTED, labelsize=7)
    ax_c.spines[:].set_visible(False)

    # ---- Panel D: Executive Brief Text ----
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor(SURFACE)
    ax_d.axis("off")
    ax_d.set_title("D — Executive Brief (Priority Signals)", fontsize=9,
                   fontweight="bold", color=GREEN, pad=8)

    brief_lines = []
    brief_lines.append(f"SURVEILLANCE PERIOD: {recent_days}-day window")
    brief_lines.append(f"Active postal codes: {sitrep['n_postals_active']}")
    brief_lines.append(f"Total elevated signals: {sitrep['n_elevated']}")
    brief_lines.append(f"Hotspot postal codes: {len(sitrep['hotspot_postals'])}")
    brief_lines.append(f"NEW signals: {len(sitrep['new_signals'])}")
    brief_lines.append("")
    brief_lines.append("TOP PRIORITY INVESTIGATIONS:")
    for rank, (_, r) in enumerate(sitrep["priority_list"].head(6).iterrows(), 1):
        ha = "[CRITICAL] " if r["high_acuity"] else ""
        ss = " [n<10]" if r["small_sample"] else ""
        line = (f"{rank}. {ha}{r['label']}\n"
                f"   @ {r['postal_code']} | {r['signal_class'].upper()}"
                f" | +{r['pct_change']:.1f}pp{ss}")
        brief_lines.append(line)

    brief_text = "\n".join(brief_lines)
    ax_d.text(0.04, 0.97, brief_text,
              transform=ax_d.transAxes,
              fontsize=7.8, color=TEXT, va="top", ha="left",
              fontfamily="monospace",
              bbox=dict(facecolor=SURFACE2, edgecolor=ACCENT + "55",
                        boxstyle="round,pad=0.5", alpha=0.85))

    out_path = out_dir / "situation_report_dashboard.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[7/7] Dashboard saved -> {out_path}")


# ============================================================
# VIII. MAIN
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="EpiHack SITREP — User Story #5")
    p.add_argument("--db",       required=True,  help="Path to epihack-mini.db")
    p.add_argument("--recent",   type=int, default=7,  help="Recent window (days)")
    p.add_argument("--baseline", type=int, default=30, help="Baseline window (days)")
    p.add_argument("--out",      default="figures/",   help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    df = load_data(args.db)

    # 2. Compute signal table
    print("[2/7] Computing full signal table (all postal x symptom pairs)...")
    sig_df = compute_signal_table(df, recent_days=args.recent, baseline_days=args.baseline)

    # 3. SITREP analysis
    print("[4/7] Building SITREP components...")
    sitrep = build_sitrep(sig_df)

    # 4. Fill LLM prompt template
    print("[5/7] Filling LLM prompt template...")
    prompt = build_llm_prompt(sig_df, recent_days=args.recent)

    # 5. Simulate narrative
    print("[6/7] Generating simulated SITREP narrative...")
    narrative = simulate_llm_narrative(sitrep, recent_days=args.recent)

    # 6. Print results
    sep = "=" * 72
    print(f"\n{sep}")
    print("  EpiHack -- Situation Report (User Story #5)")
    print(sep)
    print(f"\n  Signal table: {len(sig_df):,} postal x symptom pairs")
    top5 = sitrep["top5"]
    print(f"\n  TOP 5 SIGNALS:")
    for _, r in top5.iterrows():
        ha = " [HIGH-ACUITY]" if r["high_acuity"] else ""
        ss = " [SMALL SAMPLE]" if r["small_sample"] else ""
        print(f"    {r['signal_class'].upper():<4} | {r['label']:<30} "
              f"| postal {r['postal_code']} | +{r['pct_change']:.1f}pp "
              f"| n={r['recent_n']}{ha}{ss}")

    print(f"\n  HOTSPOT POSTAL CODES (>=3 elevated signals):")
    hp = sitrep["hotspot_postals"]
    if hp.empty:
        print("    None detected.")
    else:
        for _, r in hp.iterrows():
            print(f"    postal {r['postal_code']} — {int(r['n_elevated'])} elevated "
                  f"({int(r['n_new'])} NEW, {int(r['n_high_acuity'])} high-acuity)")

    print(f"\n  NEW SIGNALS ({len(sitrep['new_signals'])} total):")
    for _, r in sitrep["new_signals"].head(5).iterrows():
        ha = " [HIGH-ACUITY]" if r["high_acuity"] else ""
        print(f"    {r['label']:<30} | postal {r['postal_code']} "
              f"| {r['recent_rate']:.1f}% (n={r['recent_n']}){ha}")

    print(f"\n  {'─'*68}\n  FILLED LLM PROMPT (excerpt, first 600 chars):\n  {'─'*68}")
    print("  " + prompt[:600].replace("\n", "\n  "))

    print(f"\n  {'─'*68}\n  SIMULATED SITREP NARRATIVE:\n  {'─'*68}")
    print(narrative)
    print(f"\n{sep}\n")

    # 7. Render dashboard
    fig_dashboard(sig_df, sitrep, recent_days=args.recent, out_dir=out_dir)
    print(f"\nDone. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
