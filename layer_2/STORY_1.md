# Story 1 - Temporal Risk

## Objectives
To give the user a temporal trends based on the reported symptom.

## Input
- `symptom`
- `area`

## Output

Reports of `symptom` in `area` are `+/- percentage` change the `30`-day baseline ratio in the past `N` days.

## SQL

The four window dates (`recent_start`, `recent_end`, `baseline_start`, `baseline_end`)
are computed in the caller and bound as literal parameters. This is what lets
SQLite use the `(date_of_report, postal_code)` index for a range scan instead
of falling back to a full table scan. See [schema.sql](schema.sql) for the
required index and [story_1.py](story_1.py) for the date arithmetic.

The `symptom` column name (e.g. `loss_of_smell_or_taste`) and the area prefix
are also bound parameters; `symptom` must be validated against an allowlist
before interpolation because SQLite cannot bind column identifiers.

**Denominator semantics:** `recent_total` / `baseline_total` count only rows
where the symptom column is `IS NOT NULL` — i.e. reports where the user
actually answered the question. Rows with the symptom column blank do not
inflate the denominator. This makes the ratio mean "of reporters who answered
this question, what fraction said YES" rather than "of all reports submitted",
and keeps the trend stable when the form mix (quick vs. full report) changes.

```sql
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
```

### Window date arithmetic (done in the caller)

Given `anchor_date`, `recent_days`, `baseline_days`:

- `recent_start  = anchor_date - (recent_days - 1) days`
- `recent_end    = anchor_date`
- `baseline_start = anchor_date - (recent_days + baseline_days - 1) days`
- `baseline_end   = anchor_date - recent_days days`

So with the defaults (`recent_days=14`, `baseline_days=30`) the recent window
covers the 14 days ending at the anchor, and the baseline covers the 30 days
immediately preceding the recent window (no overlap).

### Inputs

| Bind            | Meaning                                                          |
| --------------- | ---------------------------------------------------------------- |
| `:recent_start` | ISO date, start of recent window                                 |
| `:recent_end`   | ISO date, end of recent window (= anchor)                        |
| `:baseline_start` | ISO date, start of baseline window                             |
| `:baseline_end` | ISO date, end of baseline window                                 |
| `:area_prefix`  | Postal-code prefix (e.g. `"850"`); matched with `LIKE prefix‖'%'`|
| `{symptom_col}` | Column name interpolated after allowlist check                   |

## JSON Output

The JSON Output is a deterministic input for LLM to synthesise the final output message, to make sure it consistence across the stories.


```json
{
  "story": "temporal_trend",
  "symptom": "loss_of_smell_or_taste",
  "area": "850",
  "window": {
    "anchor_date": "2025-07-12",
    "recent_days": 14,
    "baseline_days": 30
  },
  "recent_yes": 2,
  "recent_total": 7,
  "baseline_yes": 3,
  "baseline_total": 15,
  "recent_ratio": 0.2857,
  "baseline_ratio": 0.2000,
  "fold_change": 1.43,
  "data_quality": {
    "recent_total_below_floor": true,
    "baseline_total_below_floor": true,
    "floor": 30
  }
}
```

### Result header

```json
{
  "story": "temporal_trend",
  "symptom": "loss_of_smell_or_taste",
  "window": {
    "anchor_date": "2025-07-12",
    "recent_days": 14,
    "baseline_days": 30
  },
```

### Result body
```json
  "recent_yes": 2,
  "recent_total": 7,
  "baseline_yes": 3,
  "baseline_total": 15,
  "recent_ratio": 0.2857,
  "baseline_ratio": 0.2000,
  "fold_change": 1.43,
```

### Deterministic quality data

```json
  "data_quality": {
    "recent_total_below_floor": true,
    "baseline_total_below_floor": true,
    "floor": 30
  }
```

The two `*_below_floor` booleans are sample-size confidence flags — they tell the LLM whether the underlying windows had enough reports for the ratio to be meaningful, so the LLM can babbling ("early signal", "not yet a confirmed trend") instead of stating a noisy ratio as fact.

```python
FLOOR = 30  # minimum total reports per window

data_quality = {
    "recent_total_below_floor":   row["recent_total"]   < FLOOR,  # 7  < 30  → True
    "baseline_total_below_floor": row["baseline_total"] < FLOOR,  # 15 < 30  → True
    "floor": FLOOR,
}
```

We could just as easily compute this in SQL as recent_total < `30` AS recent_total_below_floor.

The floor is a policy decision, not a data property
We may want different floors per story (a trend story can tolerate less data than a "your demographic" story where the cell is already sliced narrow). It keeps the SQL focused on aggregation only.


## LLM Prompt
```
You generate a single short community advisory message for [APP NAME], a participatory
One Health surveillance app for Arizona residents. You will be given:
- a user profile (age, sex)
- a "story" name and a JSON payload of computed community statistics

Rules:
1. Surface all numerics as ratios or fold-changes. Never mention raw report counts.
2. Tone: factual, plain language, encouraging continued reporting, building
   health literacy and public-health awareness.
3. Never recommend the user seek medical care, see a doctor, or take any
   clinical action.
4. Keep to 2–3 short paragraphs, conversational.
5. If `data_quality.recent_total_below_floor` or `data_quality.baseline_total_below_floor`
   is true, describe the trend as "early" or "not yet statistically meaningful"
   and lean harder on the value of continued reporting.
6. If `fold_change` is null, the baseline had zero reports of this symptom —
   frame as a newly emerging signal worth watching, not as an alarm.
7. Address the user in second person. Refer to the community as "the [APP NAME]
   community" or "other reporters in Arizona".
8. End with one sentence reinforcing that the user's continued reporting matters.

For the "temporal_trend" story:
- Describe how often this symptom appeared in community reports in the recent
  window vs. the baseline window, using ratios.
- Characterise direction (rising / stable / falling) using fold_change:
  >=1.5 rising, <=0.67 falling, otherwise stable.
- Do not extrapolate causes.
```
### Example of LLM Output

```
Thanks for reporting loss of smell or taste. In the [APP NAME] community, this symptom showed up in about 29% of reports over the past 14 days, compared with 20% in the 30 days before that — a modest increase, but the recent window is still small, so this isn't yet a confirmed trend.
What this means: more reporters in Arizona have flagged this symptom recently, but we'd need more reports to know whether it's a real shift or normal week-to-week variation. Watching this kind of pattern is exactly what community surveillance is designed for.
Your continued reporting is what makes this signal sharper — please keep submitting updates if your symptoms change.
```