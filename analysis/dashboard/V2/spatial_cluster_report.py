"""
spatial_cluster_report.py
==========================
EpiHack — User Story #6: Spatial Cluster Analysis

LLM Prompt Template (filled at runtime):
    "You are a spatial epidemiology analyst. Given the following postal code x symptom
     signal matrix: {postal_symptom_matrix}. Identify postal codes where 3 or more
     symptoms show 'up' or 'new' signals in the past 7 days. For each identified cluster:
     (1) list the elevated symptoms and their % change from baseline; (2) calculate the
     composite burden score (sum of pct_changes); (3) rank clusters by composite burden.
     Output as a structured table. Note any postal codes with insufficient data
     (signal='indef')."

Composite Burden Score:
    CBS(postal) = sum(pct_change_i) for all symptoms_i where signal in {up, new}

Usage:
    python spatial_cluster_report.py --db epihack-mini.db [--recent 7] [--baseline 30]
                                      [--min-signals 3] [--out figures/]
"""

# ============================================================
# I. IMPORTS & CONFIGURATION
# ============================================================
import argparse
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

# Design palette
BG       = "#0f1117"
SURFACE  = "#1a1d27"
SURFACE2 = "#23263a"
ACCENT   = "#6c63ff"
RED      = "#ff4757"
ORANGE   = "#ffa502"
GREEN    = "#43e97b"
YELLOW   = "#f9ca24"
MUTED    = "#8892b0"
TEXT     = "#e8eaf6"

SIG_COLORS = {"new": RED, "up": ORANGE, "nc": ACCENT,
              "dn": GREEN, "gone": "#5a5d7a", "indef": "#2d2f45"}

# Symptom color palette for stacked bar decomposition
SYM_PALETTE = [
    "#6c63ff","#ff6584","#43e97b","#ffa502","#ff4757","#f9ca24",
    "#26de81","#fd9644","#45aaf2","#a55eea","#2bcbba","#fc5c65",
    "#4b7bec","#8854d0",
]


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
    print(f"[1/8] Data loaded: {len(df):,} records | "
          f"{df['date_of_report'].min().date()} -- {df['date_of_report'].max().date()}")
    return df


# ============================================================
# III. SIGNAL CLASSIFICATION
# ============================================================
def classify_signal(rec_rate: float, bas_rate: float,
                    rec_n: int, bas_n: int,
                    min_rec: int = 2, min_bas: int = 5,
                    pct_thresh: float = 5.0) -> str:
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


# ============================================================
# IV. POSTAL × SYMPTOM SIGNAL MATRIX
# ============================================================
def compute_postal_symptom_matrix(df: pd.DataFrame,
                                  recent_days: int = 7,
                                  baseline_days: int = 30) -> pd.DataFrame:
    """
    Compute the full postal_code × symptom signal matrix.

    Returns a long-format DataFrame with one row per (postal, symptom) pair,
    including signal classification and pct_change.
    Temporal anchoring is fixed to the dataset's max date for reproducibility.
    """
    max_date  = df["date_of_report"].max()
    rec_end   = max_date
    rec_start = rec_end   - pd.Timedelta(days=recent_days - 1)
    bas_end   = rec_start - pd.Timedelta(days=1)
    bas_start = bas_end   - pd.Timedelta(days=baseline_days - 1)

    df_rec = df[(df["date_of_report"] >= rec_start) & (df["date_of_report"] <= rec_end)]
    df_bas = df[(df["date_of_report"] >= bas_start) & (df["date_of_report"] <= bas_end)]

    print(f"[3/8] Windows: recent={rec_start.date()}..{rec_end.date()} "
          f"| baseline={bas_start.date()}..{bas_end.date()}")

    rows = []
    all_postals = df_rec["postal_code"].dropna().unique()

    for postal in all_postals:
        g_rec = df_rec[df_rec["postal_code"] == postal]
        g_bas = df_bas[df_bas["postal_code"] == postal]
        postal_total_rec = len(g_rec)

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
                "postal_code":       str(postal),
                "symptom":           col,
                "label":             label,
                "recent_n":          rec_n,
                "baseline_n":        bas_n,
                "postal_total_rec":  postal_total_rec,
                "recent_rate":       round(rec_rate, 2),
                "baseline_rate":     round(bas_rate, 2),
                "pct_change":        round(pct_change, 2),
                "signal_class":      sig,
                "high_acuity":       col in HIGH_ACUITY,
                "small_sample":      rec_n < 10,
            })

    mat = pd.DataFrame(rows)
    n_postals = mat["postal_code"].nunique()
    n_pairs   = len(mat)
    print(f"      Matrix: {n_pairs:,} pairs | {n_postals} postal codes")
    return mat


# ============================================================
# V. CLUSTER DETECTION
# ============================================================
def detect_clusters(mat: pd.DataFrame, min_signals: int = 3) -> pd.DataFrame:
    """
    Identify postal codes where >= min_signals symptoms show 'up' or 'new'.

    For each qualifying postal code (cluster), compute:
      - n_elevated       : count of symptoms with signal in {up, new}
      - n_new            : count of NEW signals
      - n_high_acuity    : count of high-acuity elevated symptoms
      - composite_burden : sum of pct_change values across all elevated symptoms
      - elevated_symptoms: list of (label, signal_class, pct_change, small_sample)
      - n_indef          : count of symptom pairs with insufficient data
      - rank             : rank by composite_burden (descending)
    """
    elevated_mask = mat["signal_class"].isin(["up", "new"])
    elevated      = mat[elevated_mask]

    postal_groups = elevated.groupby("postal_code")
    cluster_rows  = []

    for postal, grp in postal_groups:
        if len(grp) < min_signals:
            continue
        # Elevated symptom detail list
        syms = []
        for _, r in grp.sort_values("pct_change", ascending=False).iterrows():
            syms.append({
                "col":         r["symptom"],
                "label":       r["label"],
                "sig":         r["signal_class"],
                "rec_rate":    r["recent_rate"],
                "bas_rate":    r["baseline_rate"],
                "pct_change":  r["pct_change"],
                "rec_n":       r["recent_n"],
                "small_sample": r["small_sample"],
                "high_acuity": r["high_acuity"],
            })

        # Indef count for this postal
        postal_all  = mat[mat["postal_code"] == postal]
        n_indef     = int((postal_all["signal_class"] == "indef").sum())
        total_rec   = int(postal_all["postal_total_rec"].iloc[0])
        cbs         = round(float(grp["pct_change"].sum()), 2)

        cluster_rows.append({
            "postal_code":       postal,
            "n_elevated":        len(grp),
            "n_new":             int((grp["signal_class"] == "new").sum()),
            "n_high_acuity":     int(grp["high_acuity"].sum()),
            "composite_burden":  cbs,
            "elevated_symptoms": syms,
            "n_indef":           n_indef,
            "total_rec":         total_rec,
            "small_postal":      total_rec < 30,
        })

    clusters = pd.DataFrame(cluster_rows)
    if clusters.empty:
        print(f"      No clusters found with >= {min_signals} elevated signals.")
        return clusters

    clusters = clusters.sort_values("composite_burden", ascending=False).reset_index(drop=True)
    clusters["rank"] = clusters.index + 1
    print(f"[5/8] Clusters detected: {len(clusters)} postal codes with >= {min_signals} elevated signals")
    return clusters


def flag_indef_postals(mat: pd.DataFrame) -> pd.DataFrame:
    """
    Return postal codes where the majority of symptom pairs (> 50%) are 'indef'
    due to insufficient data, flagged as data-sparse.
    """
    total      = mat.groupby("postal_code").size().rename("total")
    indef_cnt  = mat[mat["signal_class"] == "indef"].groupby("postal_code").size().rename("n_indef")
    combined   = pd.concat([total, indef_cnt], axis=1).fillna(0)
    combined["indef_pct"] = combined["n_indef"] / combined["total"] * 100
    sparse = combined[combined["indef_pct"] > 50].sort_values("indef_pct", ascending=False)
    return sparse.reset_index()


# ============================================================
# VI. LLM PROMPT TEMPLATE FILL
# ============================================================
def build_llm_prompt(mat: pd.DataFrame, clusters: pd.DataFrame,
                     recent_days: int, min_signals: int) -> str:
    """
    Fill the User Story #6 LLM prompt template.

    Production replacement:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=900,
            messages=[{"role": "user", "content": prompt}]
        )
        narrative = response.content[0].text
    """
    # Build compact matrix string (elevated rows only)
    elevated = mat[mat["signal_class"].isin(["up", "new"])].sort_values(
        ["postal_code", "pct_change"], ascending=[True, False]
    )
    matrix_lines = []
    for postal, grp in elevated.groupby("postal_code"):
        for _, r in grp.iterrows():
            ss = " [n<10]" if r["small_sample"] else ""
            matrix_lines.append(
                f"  {postal} | {r['label']:<30} | {r['signal_class'].upper():<3} "
                f"| +{r['pct_change']:.1f}pp | rec={r['recent_rate']:.1f}% "
                f"(n={r['recent_n']}) | bas={r['baseline_rate']:.1f}%{ss}"
            )
    matrix_str = "\n".join(matrix_lines) if matrix_lines else "  [No elevated signals detected]"

    # Cluster table summary
    if not clusters.empty:
        cluster_lines = []
        for _, c in clusters.iterrows():
            ha = f" | {int(c['n_high_acuity'])} HIGH-ACUITY" if c["n_high_acuity"] > 0 else ""
            cluster_lines.append(
                f"  Rank {int(c['rank'])}: postal {c['postal_code']} — "
                f"CBS={c['composite_burden']:.1f}pp | {int(c['n_elevated'])} elevated "
                f"({int(c['n_new'])} NEW){ha}"
            )
        cluster_summary = "\n".join(cluster_lines)
    else:
        cluster_summary = "  No postal clusters detected at this threshold."

    prompt = textwrap.dedent(f"""
        You are a spatial epidemiology analyst. Given the following postal code x symptom
        signal matrix for the past {recent_days} days:

        {matrix_str}

        Identify postal codes where {min_signals} or more symptoms show 'UP' or 'NEW'
        signals. For each identified cluster:
        (1) list the elevated symptoms and their % change from baseline;
        (2) calculate the composite burden score (sum of pct_changes);
        (3) rank clusters by composite burden.

        Output as a structured table. Note any postal codes with insufficient data
        (signal='INDEF').

        [Pre-computed cluster summary for verification:
        {cluster_summary}]
    """).strip()
    return prompt


# ============================================================
# VII. SIMULATED LLM NARRATIVE
# ============================================================
def simulate_llm_narrative(clusters: pd.DataFrame,
                            indef_postals: pd.DataFrame,
                            recent_days: int,
                            min_signals: int) -> str:
    """
    Rule-based structured cluster report mimicking LLM output.

    Production replacement (inline comment):
        # response = client.messages.create(
        #     model="claude-opus-4-6", max_tokens=900,
        #     messages=[{"role":"user","content": prompt}]
        # )
        # return response.content[0].text
    """
    sep = "=" * 72
    lines = [
        f"SPATIAL CLUSTER REPORT — {recent_days}-Day Window | "
        f"Min signals threshold: {min_signals}",
        sep,
    ]

    if clusters.empty:
        lines.append(f"\nNo postal clusters with >= {min_signals} elevated signals detected.")
        lines.append("Consider widening the surveillance window or lowering the threshold.")
    else:
        # Ranked cluster table header
        lines.append(
            f"\n  {'RANK':<5} {'POSTAL':<8} {'N_ELEV':<8} "
            f"{'N_NEW':<7} {'N_HA':<6} {'CBS (pp)':<10} {'RISK FLAG'}"
        )
        lines.append("  " + "-" * 62)

        for _, c in clusters.iterrows():
            risk = ""
            if c["n_high_acuity"] > 0:
                risk = "[CRITICAL] High-acuity signal present"
            elif c["n_new"] >= 2:
                risk = "[ALERT] Multiple NEW signals"
            elif c["small_postal"]:
                risk = "[NOTE] Small reporter pool (<30)"
            lines.append(
                f"  {int(c['rank']):<5} {c['postal_code']:<8} {int(c['n_elevated']):<8} "
                f"{int(c['n_new']):<7} {int(c['n_high_acuity']):<6} "
                f"{c['composite_burden']:<10.1f} {risk}"
            )

        # Detail per cluster
        lines.append(f"\n{'─'*72}")
        lines.append("CLUSTER DETAIL (elevated symptoms per postal code):")
        lines.append(f"{'─'*72}\n")

        for _, c in clusters.iterrows():
            ha_note = f"  [!] {int(c['n_high_acuity'])} high-acuity symptom(s) present" if c["n_high_acuity"] > 0 else ""
            sp_note = f"  [NOTE] Small reporter pool (n_rec={c['total_rec']})" if c["small_postal"] else ""
            lines.append(
                f"  RANK {int(c['rank'])} | Postal {c['postal_code']} | "
                f"CBS = {c['composite_burden']:.1f} pp{ha_note}{sp_note}"
            )
            for s in c["elevated_symptoms"]:
                ss = "  [n<10]" if s["small_sample"] else ""
                ha = "  [HIGH-ACUITY]" if s["high_acuity"] else ""
                lines.append(
                    f"    • {s['label']:<30} "
                    f"| {s['sig'].upper():<3} | +{s['pct_change']:.1f}pp "
                    f"| rec={s['rec_rate']:.1f}% (n={s['rec_n']}) "
                    f"| bas={s['bas_rate']:.1f}%{ss}{ha}"
                )
            if c["n_indef"] > 0:
                lines.append(
                    f"    [NOTE] {int(c['n_indef'])} symptom pairs have insufficient "
                    f"data (INDEF) for this postal code."
                )
            lines.append("")

    # Indef postal summary
    lines.append(f"{'─'*72}")
    lines.append("POSTAL CODES WITH INSUFFICIENT DATA (>50% INDEF pairs):\n")
    if indef_postals.empty:
        lines.append("  None — data completeness is satisfactory across all postal codes.")
    else:
        for _, r in indef_postals.head(10).iterrows():
            lines.append(
                f"  postal {r['postal_code']}: {r['indef_pct']:.0f}% pairs INDEF "
                f"(n_indef={int(r['n_indef'])}/{int(r['total'])})"
            )
        if len(indef_postals) > 10:
            lines.append(f"  ... and {len(indef_postals)-10} additional data-sparse postal codes.")

    lines.append(f"\n{sep}")
    lines.append("[END SPATIAL CLUSTER REPORT]")
    return "\n".join(lines)


# ============================================================
# VIII. 4-PANEL DASHBOARD (PNG)
# ============================================================
def fig_dashboard(mat: pd.DataFrame, clusters: pd.DataFrame,
                  indef_postals: pd.DataFrame,
                  recent_days: int, min_signals: int,
                  out_dir: Path) -> None:
    """
    4-panel static dashboard:
      A. Ranked cluster bar chart (composite burden score)
      B. Symptom heatmap: postal × symptom (elevated signals only)
      C. Burden decomposition: stacked bars per postal (each symptom contribution)
      D. Indef postal summary + top cluster detail text
    """
    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    fig.suptitle(
        f"EpiHack Spatial Cluster Report — {recent_days}-Day Window | "
        f"Min {min_signals} Elevated Signals per Postal\n"
        f"Arizona Participatory Syndromic Surveillance 2023–2026",
        fontsize=13, fontweight="bold", color=TEXT, y=0.98
    )
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.06, right=0.97,
                           top=0.92, bottom=0.06,
                           hspace=0.38, wspace=0.32)

    # ---- Panel A: Ranked cluster composite burden ----
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor(SURFACE)

    if clusters.empty:
        ax_a.text(0.5, 0.5, f"No clusters detected\n(threshold: {min_signals}+ elevated signals)",
                  ha="center", va="center", color=MUTED, fontsize=11,
                  transform=ax_a.transAxes)
    else:
        clust_plot = clusters.sort_values("composite_burden").tail(20)
        bar_labels = [
            f"{r['postal_code']}\n(n={int(r['n_elevated'])})"
            for _, r in clust_plot.iterrows()
        ]
        bar_colors = [
            RED if r["n_high_acuity"] > 0 else
            ORANGE if r["n_new"] > 0 else ACCENT
            for _, r in clust_plot.iterrows()
        ]
        ax_a.barh(range(len(clust_plot)), clust_plot["composite_burden"],
                  color=bar_colors, edgecolor="none", height=0.7)

        for i, (_, r) in enumerate(clust_plot.iterrows()):
            sp = " [small pool]" if r["small_postal"] else ""
            ax_a.text(r["composite_burden"] + 0.5, i,
                      f"CBS={r['composite_burden']:.1f}{sp}",
                      va="center", fontsize=7.5, color=TEXT)

        ax_a.set_yticks(range(len(clust_plot)))
        ax_a.set_yticklabels(bar_labels, fontsize=7.5, color=TEXT)
        ax_a.set_xlabel("Composite Burden Score (sum of Δpp)", fontsize=8, color=MUTED)

        patches = [
            mpatches.Patch(color=RED,    label="High-acuity signal"),
            mpatches.Patch(color=ORANGE, label="NEW signal(s)"),
            mpatches.Patch(color=ACCENT, label="UP signal(s) only"),
        ]
        ax_a.legend(handles=patches, fontsize=7.5, loc="lower right",
                    facecolor=SURFACE2, edgecolor="none", labelcolor=TEXT)

    ax_a.set_title("A — Cluster Composite Burden Score (Ranked)",
                   fontsize=9, fontweight="bold", color=ORANGE, pad=8)
    ax_a.tick_params(colors=MUTED, labelsize=8)
    ax_a.spines[:].set_visible(False)

    # ---- Panel B: Symptom heatmap (postal × symptom, elevated only) ----
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor(SURFACE)

    elevated_mat = mat[mat["signal_class"].isin(["up", "new"])]
    if elevated_mat.empty or clusters.empty:
        ax_b.text(0.5, 0.5, "No elevated signals to display.",
                  ha="center", va="center", color=MUTED, fontsize=10,
                  transform=ax_b.transAxes)
    else:
        cluster_postals = clusters["postal_code"].tolist()
        em_cluster = elevated_mat[elevated_mat["postal_code"].isin(cluster_postals)]
        pivot = em_cluster.pivot_table(
            index="postal_code", columns="label",
            values="pct_change", aggfunc="max", fill_value=0
        )
        # Order postals by composite burden
        pivot = pivot.reindex(
            clusters.set_index("postal_code")
            .loc[clusters["postal_code"][clusters["postal_code"].isin(pivot.index)]]
            ["composite_burden"]
            .sort_values(ascending=False)
            .index
        ).dropna(how="all")

        mat_vals = pivot.values
        im = ax_b.imshow(mat_vals, aspect="auto", cmap="YlOrRd",
                         vmin=0, vmax=max(mat_vals.max(), 1))
        ax_b.set_xticks(range(len(pivot.columns)))
        ax_b.set_xticklabels(pivot.columns, rotation=50, ha="right",
                              fontsize=6.5, color=TEXT)
        ax_b.set_yticks(range(len(pivot.index)))
        ax_b.set_yticklabels(pivot.index, fontsize=7.5, color=TEXT)
        cb = plt.colorbar(im, ax=ax_b, fraction=0.035, pad=0.03)
        cb.set_label("Delta Prevalence (pp)", color=MUTED, fontsize=8)
        cb.ax.yaxis.set_tick_params(labelcolor=MUTED, labelsize=7)

    ax_b.set_title("B — Postal × Symptom Heatmap (Elevated Clusters)",
                   fontsize=9, fontweight="bold", color=ACCENT, pad=8)
    ax_b.tick_params(colors=MUTED, labelsize=7)
    ax_b.spines[:].set_visible(False)

    # ---- Panel C: Burden decomposition stacked bars ----
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor(SURFACE)

    if clusters.empty:
        ax_c.text(0.5, 0.5, "No clusters to display.", ha="center", va="center",
                  color=MUTED, fontsize=10, transform=ax_c.transAxes)
    else:
        # Gather all unique symptom labels across clusters
        all_sym_labels = []
        for _, c in clusters.iterrows():
            for s in c["elevated_symptoms"]:
                if s["label"] not in all_sym_labels:
                    all_sym_labels.append(s["label"])

        top_clusters = clusters.head(15)
        x_labels = top_clusters["postal_code"].tolist()
        x = np.arange(len(x_labels))
        bottom = np.zeros(len(x_labels))

        for si, sym_label in enumerate(all_sym_labels):
            vals = []
            for _, c in top_clusters.iterrows():
                sym_dict = {s["label"]: s["pct_change"]
                            for s in c["elevated_symptoms"]}
                vals.append(sym_dict.get(sym_label, 0))
            color = SYM_PALETTE[si % len(SYM_PALETTE)]
            ax_c.bar(x, vals, bottom=bottom, color=color,
                     label=sym_label, edgecolor="none", width=0.72)
            bottom += np.array(vals)

        ax_c.set_xticks(x)
        ax_c.set_xticklabels(x_labels, rotation=45, ha="right",
                              fontsize=7.5, color=TEXT)
        ax_c.set_ylabel("Composite Burden Score (Δpp)", fontsize=8, color=MUTED)

        # Legend outside if too many symptoms
        if len(all_sym_labels) <= 8:
            ax_c.legend(fontsize=6.5, facecolor=SURFACE2, edgecolor="none",
                        labelcolor=TEXT, ncol=2, loc="upper right")
        else:
            ax_c.legend(fontsize=6, facecolor=SURFACE2, edgecolor="none",
                        labelcolor=TEXT, ncol=3, loc="upper right",
                        bbox_to_anchor=(1.0, 1.0))

    ax_c.set_title("C — Burden Decomposition by Symptom (per Postal Cluster)",
                   fontsize=9, fontweight="bold", color=GREEN, pad=8)
    ax_c.tick_params(colors=MUTED, labelsize=8)
    ax_c.spines[:].set_visible(False)

    # ---- Panel D: Top cluster text + indef summary ----
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor(SURFACE)
    ax_d.axis("off")
    ax_d.set_title("D — Top Cluster Summary & Data Quality",
                   fontsize=9, fontweight="bold", color=YELLOW, pad=8)

    text_lines = ["CLUSTER RANKING SUMMARY\n"]
    if clusters.empty:
        text_lines.append("No clusters detected.")
    else:
        for _, c in clusters.head(8).iterrows():
            ha = " [HA]" if c["n_high_acuity"] > 0 else ""
            sp = " [small]" if c["small_postal"] else ""
            text_lines.append(
                f"#{int(c['rank'])} postal {c['postal_code']} — "
                f"CBS={c['composite_burden']:.1f} | "
                f"{int(c['n_elevated'])} elevated ({int(c['n_new'])} NEW){ha}{sp}"
            )
            # Show top 3 symptoms
            for s in c["elevated_symptoms"][:3]:
                text_lines.append(
                    f"   {s['sig'].upper()} {s['label']} +{s['pct_change']:.1f}pp"
                )
            if len(c["elevated_symptoms"]) > 3:
                text_lines.append(
                    f"   ... +{len(c['elevated_symptoms'])-3} more symptoms"
                )
            text_lines.append("")

    text_lines.append("DATA QUALITY — INDEF POSTALS")
    if indef_postals.empty:
        text_lines.append("All postal codes: data completeness OK")
    else:
        text_lines.append(
            f"{len(indef_postals)} postal codes >50% INDEF pairs"
        )
        for _, r in indef_postals.head(5).iterrows():
            text_lines.append(
                f"  postal {r['postal_code']}: {r['indef_pct']:.0f}% INDEF"
            )

    ax_d.text(0.04, 0.97, "\n".join(text_lines),
              transform=ax_d.transAxes, fontsize=7.5, color=TEXT,
              va="top", ha="left", fontfamily="monospace",
              bbox=dict(facecolor=SURFACE2, edgecolor=ACCENT + "55",
                        boxstyle="round,pad=0.5", alpha=0.85))

    out_path = out_dir / "spatial_cluster_dashboard.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[8/8] Dashboard saved -> {out_path}")


# ============================================================
# IX. MAIN
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="EpiHack Spatial Cluster Analysis — User Story #6")
    p.add_argument("--db",          required=True,  help="Path to epihack-mini.db")
    p.add_argument("--recent",      type=int, default=7,  help="Recent window (days)")
    p.add_argument("--baseline",    type=int, default=30, help="Baseline window (days)")
    p.add_argument("--min-signals", type=int, default=3,  help="Minimum elevated signals for cluster")
    p.add_argument("--out",         default="figures/",   help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = load_data(args.db)

    # 2-3. Compute matrix
    print("[2/8] Computing postal x symptom signal matrix...")
    mat = compute_postal_symptom_matrix(df,
                                        recent_days=args.recent,
                                        baseline_days=args.baseline)

    # 4. Detect clusters
    print("[4/8] Detecting spatial clusters...")
    clusters = detect_clusters(mat, min_signals=args.min_signals)

    # 5. Flag indef postals
    print("[5/8] Flagging data-sparse postal codes...")
    indef_postals = flag_indef_postals(mat)
    print(f"      {len(indef_postals)} postal codes with >50% INDEF pairs")

    # 6. Fill LLM prompt
    print("[6/8] Filling LLM prompt template...")
    prompt = build_llm_prompt(mat, clusters,
                               recent_days=args.recent,
                               min_signals=args.min_signals)

    # 7. Simulate narrative
    print("[7/8] Generating simulated cluster report...")
    narrative = simulate_llm_narrative(clusters, indef_postals,
                                        recent_days=args.recent,
                                        min_signals=args.min_signals)

    # Print outputs
    sep = "=" * 72
    print(f"\n{sep}")
    print("  EpiHack -- Spatial Cluster Analysis (User Story #6)")
    print(sep)

    if not clusters.empty:
        print(f"\n  {'RANK':<5} {'POSTAL':<8} {'ELEV':<6} {'NEW':<5} "
              f"{'HA':<4} {'CBS':>8}   FLAGS")
        print("  " + "-" * 55)
        for _, c in clusters.iterrows():
            flags = []
            if c["n_high_acuity"] > 0: flags.append("HIGH-ACUITY")
            if c["n_new"] >= 2:        flags.append("MULTI-NEW")
            if c["small_postal"]:      flags.append("small-pool")
            print(
                f"  {int(c['rank']):<5} {c['postal_code']:<8} {int(c['n_elevated']):<6} "
                f"{int(c['n_new']):<5} {int(c['n_high_acuity']):<4} "
                f"{c['composite_burden']:>8.1f}   {', '.join(flags)}"
            )

    print(f"\n  {'─'*68}\n  FILLED LLM PROMPT (excerpt):\n  {'─'*68}")
    print("  " + prompt[:700].replace("\n", "\n  "))
    print(f"\n  {'─'*68}\n  SIMULATED CLUSTER NARRATIVE:\n  {'─'*68}")
    print(narrative)
    print(f"\n{sep}\n")

    # 8. Render dashboard
    fig_dashboard(mat, clusters, indef_postals,
                  recent_days=args.recent,
                  min_signals=args.min_signals,
                  out_dir=out_dir)
    print(f"\nDone. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
