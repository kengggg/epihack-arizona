"""Story 1 — Temporal Risk.

Pipeline: incidents DB -> SQL trend query -> JSON payload -> LLM prompt -> message.
The LLM call is stubbed; swap `call_llm` for a real provider when ready.
"""

from __future__ import annotations

import json
import os
import sqlite3
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

STORY_NAME = "temporal_trend"

# Symptom columns that may be queried. Whitelisted because the symptom name is
# interpolated into the SQL as an identifier (sqlite cannot bind column names).
SYMPTOM_COLUMNS: frozenset[str] = frozenset({
    "no_symptoms",
    "symptoms",
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
})

# Below this many reports in a window, the ratio is flagged as not yet
# statistically meaningful (see STORY_1.md prompt rule 5).
REPORT_FLOOR = 30


@dataclass(frozen=True)
class TrendInputs:
    symptom: str
    area: str  # postal_code prefix, e.g. "850"
    anchor_date: date
    recent_days: int = 14
    baseline_days: int = 30


def _build_sql(symptom_col: str) -> str:
    # symptom_col is validated against SYMPTOM_COLUMNS before reaching here.
    #
    # All four window dates are bound as literal parameters (computed in
    # Python, see `compute_trend`). This is what lets the SQLite planner use
    # idx_incidents_date_postal: BETWEEN against bound literals becomes an
    # index range scan, whereas BETWEEN against a CTE-joined column stays a
    # full table scan even when an index exists.
    return f"""
WITH counts AS (
    SELECT
        SUM(CASE WHEN date_of_report BETWEEN :recent_start AND :recent_end
                  AND {symptom_col} = 'YES' THEN 1 ELSE 0 END) AS recent_yes,
        SUM(CASE WHEN date_of_report BETWEEN :recent_start AND :recent_end
                  AND {symptom_col} IS NOT NULL THEN 1 ELSE 0 END) AS recent_total,
        SUM(CASE WHEN date_of_report BETWEEN :baseline_start AND :baseline_end
                  AND {symptom_col} = 'YES' THEN 1 ELSE 0 END) AS baseline_yes,
        SUM(CASE WHEN date_of_report BETWEEN :baseline_start AND :baseline_end
                  AND {symptom_col} IS NOT NULL THEN 1 ELSE 0 END) AS baseline_total
    FROM incidents
    WHERE date_of_report BETWEEN :baseline_start AND :recent_end
      AND postal_code LIKE :area_prefix || '%'
)
SELECT
    recent_yes, recent_total, baseline_yes, baseline_total,
    ROUND(1.0 * recent_yes   / NULLIF(recent_total,   0), 4) AS recent_ratio,
    ROUND(1.0 * baseline_yes / NULLIF(baseline_total, 0), 4) AS baseline_ratio,
    ROUND(
        (1.0 * recent_yes   / NULLIF(recent_total,   0))
      / NULLIF(1.0 * baseline_yes / NULLIF(baseline_total, 0), 0),
        2
    ) AS fold_change
FROM counts;
"""


def compute_trend(db_path: str | Path, inputs: TrendInputs) -> dict[str, Any]:
    """Run the temporal-trend query and return the JSON payload from STORY_1.md."""
    if inputs.symptom not in SYMPTOM_COLUMNS:
        raise ValueError(
            f"unknown symptom {inputs.symptom!r}; "
            f"must be one of {sorted(SYMPTOM_COLUMNS)}"
        )

    # Compute window bounds in Python so they bind as literal parameters,
    # which lets SQLite use the (date_of_report, postal_code) index for a
    # range scan instead of falling back to a full table scan.
    anchor = inputs.anchor_date
    recent_start = anchor - timedelta(days=inputs.recent_days - 1)
    recent_end = anchor
    baseline_start = anchor - timedelta(days=inputs.recent_days + inputs.baseline_days - 1)
    baseline_end = anchor - timedelta(days=inputs.recent_days)

    sql = _build_sql(inputs.symptom)
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            sql,
            {
                "recent_start": recent_start.isoformat(),
                "recent_end": recent_end.isoformat(),
                "baseline_start": baseline_start.isoformat(),
                "baseline_end": baseline_end.isoformat(),
                "area_prefix": inputs.area,
            },
        ).fetchone()

    recent_total = row["recent_total"] or 0
    baseline_total = row["baseline_total"] or 0

    return {
        "story": STORY_NAME,
        "symptom": inputs.symptom,
        "area": inputs.area,
        "window": {
            "anchor_date": inputs.anchor_date.isoformat(),
            "recent_days": inputs.recent_days,
            "baseline_days": inputs.baseline_days,
        },
        "recent_yes": row["recent_yes"] or 0,
        "recent_total": recent_total,
        "baseline_yes": row["baseline_yes"] or 0,
        "baseline_total": baseline_total,
        "recent_ratio": row["recent_ratio"],
        "baseline_ratio": row["baseline_ratio"],
        "fold_change": row["fold_change"],
        "data_quality": {
            "recent_total_below_floor": recent_total < REPORT_FLOOR,
            "baseline_total_below_floor": baseline_total < REPORT_FLOOR,
            "floor": REPORT_FLOOR,
        },
    }


def trends_for_report(
    db_path: str | Path,
    user_report: dict[str, Any],
    area: str,
    anchor_date: date,
    recent_days: int = 14,
    baseline_days: int = 30,
) -> list[dict[str, Any]]:
    """Compute one trend payload per symptom the user reported as YES.

    `user_report` is the raw report dict (e.g. {"fever": "YES", "cough_congestion": "NO", ...}).
    Symptoms not in SYMPTOM_COLUMNS are ignored so unrelated profile keys
    (age, sex, email) pass through harmlessly.
    """
    reported = [
        sym for sym, val in user_report.items()
        if sym in SYMPTOM_COLUMNS and val == "YES"
    ]
    return [
        compute_trend(
            db_path,
            TrendInputs(
                symptom=sym,
                area=area,
                anchor_date=anchor_date,
                recent_days=recent_days,
                baseline_days=baseline_days,
            ),
        )
        for sym in reported
    ]


# Prompt copied from STORY_1.md. Kept in code so the LLM call is self-contained;
# update both if the spec changes.
PROMPT_TEMPLATE = """\
You generate a single short community advisory message for [APP NAME], a participatory
One Health surveillance app for Arizona residents. You will be given:
- a user profile (age, sex)
- a "story" name and a LIST of JSON payloads, one per symptom the user reported

Rules:
1. Surface all numerics as ratios or fold-changes. Never mention raw report counts.
2. Tone: factual, plain language, encouraging continued reporting, building
   health literacy and public-health awareness.
3. Never recommend the user seek medical care, see a doctor, or take any
   clinical action.
4. Keep to 2-3 short paragraphs, conversational. Cover every symptom in the
   payload list, but do not make the message longer than 3 paragraphs total -
   group symptoms together where the directions are similar.
5. If `data_quality.recent_total_below_floor` or `data_quality.baseline_total_below_floor`
   is true for a symptom, describe THAT symptom's trend as "early" or "not yet
   statistically meaningful" and lean harder on the value of continued reporting.
6. If `fold_change` is null for a symptom, the baseline had zero reports of
   that symptom - frame as a newly emerging signal worth watching, not as an alarm.
7. Address the user in second person. Refer to the community as "the [APP NAME]
   community" or "other reporters in Arizona".
8. End with one sentence reinforcing that the user's continued reporting matters.

For the "temporal_trend" story:
- For each symptom, describe how often it appeared in community reports in the
  recent window vs. the baseline window, using ratios.
- Characterise direction per symptom (rising / stable / falling) using fold_change:
  >=1.5 rising, <=0.67 falling, otherwise stable.
- Do not extrapolate causes.

USER PROFILE:
{user_profile_json}

STORY: {story}
PAYLOADS:
{payloads_json}
"""


def build_prompt(user_profile: dict[str, Any], payloads: list[dict[str, Any]]) -> str:
    if not payloads:
        raise ValueError("build_prompt requires at least one payload")
    return PROMPT_TEMPLATE.format(
        user_profile_json=json.dumps(user_profile, indent=2),
        story=payloads[0]["story"],
        payloads_json=json.dumps(payloads, indent=2),
    )


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "medgemma:latest")


def call_llm(prompt: str, *, timeout: float = 120.0) -> str:
    """Call a local Ollama server and return the generated advisory.

    Defaults to medgemma:latest at http://localhost:11434. Override with the
    OLLAMA_HOST / OLLAMA_MODEL env vars. Requires `ollama serve` to be running
    and the model pulled (`ollama pull medgemma:latest`).
    """
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST.rstrip('/')}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_HOST}. "
            "Is `ollama serve` running?"
        ) from exc
    return payload.get("response", "").strip()


def generate_message(
    db_path: str | Path,
    user_report: dict[str, Any],
    user_profile: dict[str, Any],
    area: str,
    anchor_date: date,
    recent_days: int = 14,
    baseline_days: int = 30,
) -> tuple[list[dict[str, Any]], str]:
    """End-to-end: user report -> per-symptom payloads -> prompt -> LLM message.

    Returns ([], "") when the user reported no recognised symptoms - caller
    decides whether to show anything in that case.
    """
    payloads = trends_for_report(
        db_path, user_report, area, anchor_date, recent_days, baseline_days
    )
    if not payloads:
        return [], ""
    prompt = build_prompt(user_profile, payloads)
    message = call_llm(prompt)
    return payloads, message
