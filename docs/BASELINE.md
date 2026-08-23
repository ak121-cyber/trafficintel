# Pre-fix baseline (captured 2026-08-24)

Recorded from the ten result JSONs produced while the web app was being built,
immediately before those files were deleted as build artifacts. Kept because an
event count means nothing without something to compare it against, and because
re-running these clips is the validation step for the accident and red-light
changes.

Every run below used the OLD logic: a 2-second cooldown with no incident identity,
and red-light enforcement that could never arm.

| run | dur (s) | vehicles | accidents | gaps ~2.0s | violations | enforcement armed | codec |
|---|---|---|---|---|---|---|---|
| `job_14ba7d35bb2a` | 41.17 | 31 | 10 | 3/9 | 0 | False | avc1 |
| `job_16547b220395` | 7.01 | 6 | 1 | 0/0 | 0 | False | avc1 |
| `job_433870d8f7ad` | 12.05 | 46 | 0 | 0/0 | 0 | False | avc1 |
| `job_4738d3d72eeb` | 30.15 | 6 | 2 | 0/1 | 0 | False | avc1 |
| `job_47ceb863e006` | 10.01 | 10 | 2 | 1/1 | 0 | False | avc1 |
| `job_729bdf075505` | 76.6 | 140 | 27 | 12/26 | 0 | False | avc1 |
| `job_ca618933a542` | 21.92 | 73 | 0 | 0/0 | 0 | False | avc1 |
| `job_faf3a8750bc1` | 10.01 | 10 | 1 | 0/0 | 0 | False | avc1 |
| `annotated` | 135.63 | 624 | 11 | 1/10 | 0 | False | avc1 |
| `cli_regression` | 76.6 | 140 | 27 | 12/26 | 0 | False | avc1 |

## What this baseline establishes

**Red-light enforcement never armed in a single run.** All ten report
`red_light_enforcement_active: false` and `red_light_violation_count: 0`. The zero
was not a finding about the footage; it was the feature being structurally inert,
because `RedLightMonitor.enabled` requires stop-line geometry, `default_stop_lines()`
returned `[]`, and the frontend never sent any. Any comparison of violation counts
against this baseline is therefore a comparison against *no measurement at all*.

**Accident events tracked the cooldown, not the footage.** 29 of 73
inter-event gaps across all runs fall in 1.9-2.3s, clustering on the 2.0s
`event_gap` constant. The worst case, `job_729bdf075505` (76.6s), produced 27
events with runs of gaps at exactly 2.0s - one ongoing situation re-reported every
time the cooldown expired while evidence stayed high.

**Two confidence values were fabricated.** `cli_regression` contains events with
confidence exactly `0.82` and `0.86`. These are not model outputs; they were
hardcoded for motion-derived events, which made invented numbers indistinguishable
from real detector scores. Motion-only events now report `confidence: null` with
`evidence: ["motion"]`.

## How to reproduce for comparison

The clips themselves are in `test_videos/` (untracked). Re-run the worst case:

```
python run_all.py --video test_videos/<clip>.mp4 --accident --traffic-light
```

Expected direction of change, to be confirmed on real footage:

- accident count falls sharply, and remaining inter-event gaps no longer cluster at 2.0s
- `accident_model_confirmed_count` reported alongside the total
- `stop_line_source` becomes `auto` when the stop line is visible, and
  `stop_line_calibration.reason` explains the refusal when it is not
- a violation count of 0 is now only meaningful when `red_light_enforcement_active` is true
