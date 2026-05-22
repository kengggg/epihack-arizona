# Story 2 - Profile Prevalence

## Objectives
To give the user a prevalence analysis based on the user profile(s).

## Input
- `age-band`
- `gender`
- `area`

## Output

`X%` of reports from [`age band` + `gender` + `area`] in the past `N` days included `symptom`.

## SQL

```sql
WITH params AS (
    SELECT
        DATE('2026-05-21') AS anchor_date,
        90                 AS window_days,
        'Maricopa'         AS user_county,
        'FEMALE'           AS user_sex,
        '30-44'            AS user_age_band
),
windowed AS (
    SELECT i.*,
        CASE WHEN i.age < 30 THEN '18-29'
             WHEN i.age < 45 THEN '30-44'
             WHEN i.age < 65 THEN '45-64'
             ELSE '65+' END AS age_band
    FROM incidents i, params p
    WHERE i.date_of_report
          BETWEEN DATE(p.anchor_date, '-' || (p.window_days - 1) || ' days')
              AND p.anchor_date
)
SELECT 'L0_county_sex_age' AS level, 0 AS specificity,
       COUNT(*) AS denominator,
       SUM(CASE WHEN w.fever='YES' AND w.cough_congestion='YES' AND w.chills='YES'
                THEN 1 ELSE 0 END) AS numerator
FROM windowed w, params p
WHERE w.county   = p.user_county
  AND w.sex      = p.user_sex
  AND w.age_band = p.user_age_band
UNION ALL
SELECT 'L1_state_sex_age', 1,
       COUNT(*),
       SUM(CASE WHEN w.fever='YES' AND w.cough_congestion='YES' AND w.chills='YES'
                THEN 1 ELSE 0 END)
FROM windowed w, params p
WHERE w.sex      = p.user_sex
  AND w.age_band = p.user_age_band
UNION ALL
SELECT 'L2_state_age', 2,
       COUNT(*),
       SUM(CASE WHEN w.fever='YES' AND w.cough_congestion='YES' AND w.chills='YES'
                THEN 1 ELSE 0 END)
FROM windowed w, params p
WHERE w.age_band = p.user_age_band
UNION ALL
SELECT 'L3_state', 3,
       COUNT(*),
       SUM(CASE WHEN w.fever='YES' AND w.cough_congestion='YES' AND w.chills='YES'
                THEN 1 ELSE 0 END)
FROM windowed w
ORDER BY specificity;
```

## Raw Output

```
L0_county_sex_age  denom=742   num=3   ratio=0.0040
L1_state_sex_age   denom=1119  num=3   ratio=0.0027
L2_state_age       denom=1871  num=7   ratio=0.0037
L3_state           denom=5405  num=13  ratio=0.0024
```

## JSON Output

The JSON Output is a deterministic input for LLM to synthesise the final output message, to make sure it consistence across the stories.

When denominator >= floor, App layer should picks the most-specific level which is `L0_county_sex_age` which is combination of all 3 parameters:


```json
{
  "story": "profile_prevalence",
  "user_profile": {
    "age_band": "30-44",
    "sex": "FEMALE",
    "county": "Maricopa"
  },
  "user_symptoms": ["fever", "cough_congestion", "chills"],
  "window": {"anchor_date": "2026-05-21", "window_days": 90},
  "match_level": {
    "name": "L0_county_sex_age",
    "dimensions_used": ["county", "sex", "age_band"],
    "dimensions_dropped": [],
    "description": "women aged 30-44 in Maricopa County"
  },
  "ratio": 0.0040,
  "denominator_above_floor": true,
  "numerator_above_floor": false,
  "floor": {"denominator": 100, "numerator": 5}
}
```

In sparse area the same query returns:

```json
{
  "match_level": {
    "name": "L2_state_age",
    "dimensions_used": ["age_band"],
    "dimensions_dropped": ["county", "sex"],
    "description": "people aged 65+ in Arizona"
  },
  "ratio": 0.0000,
  "denominator_above_floor": true,
  "numerator_above_floor": false
}
```


### Result header

```json
{
  "story": "profile_prevalence",
  "user_profile": {
    "age_band": "30-44",
    "sex": "FEMALE",
    "county": "Maricopa"
  },
```

### Result body
```json
  },
  "user_symptoms": ["fever", "cough_congestion", "chills"],
  "window": {"anchor_date": "2026-05-21", "window_days": 90},
  "match_level": {
    "name": "L0_county_sex_age",
    "dimensions_used": ["county", "sex", "age_band"],
    "dimensions_dropped": [],
    "description": "women aged 30-44 in Maricopa County"
  },
```

### Deterministic quality data

```json
{
  "ratio": 0.0040,
  "denominator_above_floor": true,
  "numerator_above_floor": false,
  "floor": {"denominator": 100, "numerator": 5}
}
```

The floor is a policy decision, not a data property
We may want different floors per story (a trend story can tolerate less data than a "your demographic" story where the cell is already sliced narrow). It keeps the SQL focused on aggregation only.


## LLM Prompt
```
For the "profile_prevalence" story:

- Describe the proportion of reporters matching the user's profile who reported
  the SAME COMBINATION of symptoms as the user, over the time window.

- Use the pre-rendered `match_level.description` phrase verbatim when referring
  to the comparison group. Do not paraphrase ("women aged 30-44 in Maricopa
  County", not "middle-aged Maricopa women").

- If `match_level.dimensions_dropped` is non-empty, acknowledge the
  generalization gently — e.g., "we don't yet have enough reports specifically
  from [dropped dimension], so this looks across [used dimensions]".

- If `numerator_above_floor` is false:
    - If ratio is 0, frame as "this combination hasn't yet been reported by
      others in this group" — an interesting observation, not a concern.
    - Otherwise frame as "fewer than [ceiling]%" using the smallest sensible
      round number above the ratio.

- If `numerator_above_floor` is true, surface the ratio rounded to one
  significant figure (e.g., 4%, 0.4%, not 3.74%).

- Frame the ratio neutrally. Do not characterise it as "rare" / "common" /
  "unusual" unless the ratio is extreme (>20% or <0.1%) AND
  numerator_above_floor is true. This story is about relatability, not signal
  detection.

- Always list the user's symptom combination explicitly using human-readable
  names from the symptom mapping (e.g., "fever, cough/congestion, and chills").
```

### Example of LLM Output

For the dense area e.g. Maricopa, ratio 0.4%, numerator below floor:

```
Thanks for reporting fever, cough/congestion, and chills. Among women aged 30-44 in Maricopa County who reported to the [APP NAME] community over the past 90 days, fewer than 1% reported this same combination of symptoms together.

This combination is uncommon in your profile group — most reports involve fewer or different symptoms. That doesn't mean anything is wrong, just that what you're experiencing isn't a frequent pattern in the recent community data.

Your report adds an important data point. Please keep submitting updates if your symptoms change.
```

For the sparse area e.g. Greenlee, cascaded to L2, ratio 0%:

```
Thanks for reporting fever, cough/congestion, and chills. We don't yet have enough reports specifically from your county and gender, so we're looking across people aged 65+ in Arizona — and so far, this combination of symptoms hasn't been reported together by others in this group over the past 90 days.

Your report is the first signal of this pattern in your age group. That's exactly why community surveillance exists — patterns are visible only when people report them.

Please keep submitting updates if your symptoms change.
```
