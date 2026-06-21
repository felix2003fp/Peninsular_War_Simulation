# Peninsular War — simulation vs reality

Key cities: MAD, LIS, BAR, VAL, MRC, OPO, SEV, CAD, MAL, GRA


## Tier 1

- Mean Brier (lower=better): **0.236**
- Mean weekly agreement (majority vote vs history): **0.750**
- Capture-order rank correlation (real vs sim median): **0.30**
- French win rate: **6.6%**  (real outcome: Allied victory)
- Mean game length: 312 weeks; ended by: {'turn_limit': np.int64(500)}
- Representative episodes (replay seed with simulate_history or watch_pygame `--base-seed <seed> --n-sims 1`):
    - closest_to_history: seed **10094** (distance 0.228)
    - median: seed **10059** (distance 0.269)
    - typical_french_win: seed **10328** (distance 0.239)

## Tier 2

- Mean Brier (lower=better): **0.247**
- Mean weekly agreement (majority vote vs history): **0.665**
- Capture-order rank correlation (real vs sim median): **0.58**
- French win rate: **49.8%**  (real outcome: Allied victory)
- Mean game length: 312 weeks; ended by: {'turn_limit': np.int64(493), 'allies_eliminated': np.int64(6), 'france_eliminated': np.int64(1)}
- Representative episodes (replay seed with simulate_history or watch_pygame `--base-seed <seed> --n-sims 1`):
    - closest_to_history: seed **10007** (distance 0.236)
    - median: seed **10395** (distance 0.387)
    - typical_french_win: seed **10360** (distance 0.266)

## Files

- `prob_curves.png` — per-city P(France) vs real step function
- `fidelity_over_time.png` — share of cities matching history each week
- `time_to_capture.png` — simulated first-capture spread vs real date (★)
- `outcomes.png` — winner / end-reason / game-length distributions
- `city_metrics.csv` — per-city Brier, agreement, capture weeks