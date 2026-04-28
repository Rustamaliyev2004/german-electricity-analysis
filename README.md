# German Electricity Market Analysis (2020–2024)

Analysis of 35,772 hourly observations from SMARD (German energy regulator).

## Scripts
- `crisis_analysis.py` — price crisis story, fossil vs renewable generation charts
- `regression_analysis.py` — OLS regression of day-ahead prices on generation mix and load

## Key Findings
- R² = 0.28: generation mix explains 28% of hourly price variation
- Load has the strongest positive effect on price (+0.80 EUR/MWh per MWh)
- Gas generation shows reverse causality typical in merit-order models

## Data Source
[SMARD](https://www.smard.de) — Bundesnetzagentur, hourly resolution, 2020–2024

## Tools
Python, pandas, statsmodels, matplotlib
