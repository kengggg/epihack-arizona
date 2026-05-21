# Story 1 - Temporal Risk
## Objectives
To give the user a temporal trends based on the reported symptom.

## Input
- `symptom`
- `area`

## Output

Reports of `symptom` in `area` are `+/- percentage` change the `30`-day baseline ratio in the past `N` days.

## SQL

```
WITH params AS (
   SELECT
       DATE(:anchor_date)  AS anchor_date,
       :recent_days        AS recent_days,
       :baseline_days      AS baseline_days
),
windows AS (
   SELECT
       DATE(anchor_date, '-' || (recent_days - 1) || ' days')                AS recent_start,
       anchor_date                                                           AS recent_end,
       DATE(anchor_date, '-' || (recent_days + baseline_days - 1) || ' days')AS baseline_start,
       DATE(anchor_date, '-' || recent_days || ' days')                      AS baseline_end
   FROM params
),
counts AS (
   SELECT
       SUM(CASE WHEN date_of_report BETWEEN w.recent_start   AND w.recent_end
                 AND loss_of_smell_or_taste = 'YES' THEN 1 ELSE 0 END) AS recent_yes,
       SUM(CASE WHEN date_of_report BETWEEN w.recent_start   AND w.recent_end
                                             THEN 1 ELSE 0 END) AS recent_total,
       SUM(CASE WHEN date_of_report BETWEEN w.baseline_start AND w.baseline_end
                 AND loss_of_smell_or_taste = 'YES' THEN 1 ELSE 0 END) AS baseline_yes,
       SUM(CASE WHEN date_of_report BETWEEN w.baseline_start AND w.baseline_end
                                             THEN 1 ELSE 0 END) AS baseline_total
   FROM incidents, windows w
)
SELECT
   'loss_of_smell_or_taste' AS symptom,
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

## JSON Output
```
{
  "story": "temporal_trend",
  "symptom": "loss_of_smell_or_taste",
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
