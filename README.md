# Walkability and housing prices

This project compares **EPA National Walkability Index** scores at the ZIP-code level with **typical home values** (Zillow research–style series, cleaned to a single `home_value` per ZIP in `data/zillow_clean.csv`).

## Research question

Do ZIP codes with higher walkability tend to have higher typical home prices—and how strong is that relationship at the ZIP level and when states are aggregated?

## What we did

1. **Data cleaning**: Standardized ZIP codes to five digits, merged walk scores with home values, dropped rows missing core fields, and trimmed extreme tails (1st–99th percentile) for both walk scores and prices so a handful of extreme markets does not dominate the visuals and regression.
2. **Exploratory analysis**: Distributions, scatter and density plots, state rankings by mean walk score, and state-level scatter of mean walk vs. median home value.
3. **Statistical analysis**: Pearson and Spearman correlations; simple ordinary least squares (OLS) of `log1p(home_value)` on walk score; residual plot to check the log-linear specification.

## Key findings

Across the cleaned merged sample, walkability and typical home value move together in the **positive** direction: higher walk scores are associated with higher prices on average. The relationship is clearer after a **log transform** of home values because sale and Zillow-style values are highly right-skewed. Correlation does **not** imply causation—walkable places often overlap with dense job centers, coastal markets, and amenity-rich neighborhoods that raise prices for many reasons beyond walking alone.

**Snapshot from the bundled data (re-run the script to refresh):** about **18.5k** ZIP codes after merge and trimming; Pearson *r* ≈ **0.39** between walk score and price; Spearman *ρ* ≈ **0.37**; simple OLS of log price on walk score yields *R*² ≈ **0.14** on the log scale.

## Repository / GitHub

After you create a remote repository on GitHub and push this project, add your link here:

**GitHub:** `https://github.com/<your-username>/<your-repo>`

```bash
git remote add origin https://github.com/<your-username>/<your-repo>.git
git add .
git commit -m "Walkability vs housing prices analysis and PDF report"
git push -u origin main
```

## PDF report (submission artifact)

Install dependencies and generate the PDF:

```bash
pip install -r requirements.txt
python scripts/generate_walkability_housing_report.py
```

Output: `reports/walkability_housing_report.pdf`

## Data sources (in `data/`)

| File | Description |
|------|-------------|
| `zip_walk_index.csv` | ZIP-level National Walkability Index (`NatWalkInd`) and weights |
| `zillow_clean.csv` | ZIP-level typical home value and geography (`State`, `City`, etc.) |

Raw Zillow time series (`zillow.csv`) are documented at [Zillow Research data](https://www.zillow.com/research/data/); this repo ignores the large raw file via `.gitignore` and ships the cleaned extract instead.

## Notebooks

`notebooks/cleaning zillow data.ipynb` — prior workflow for building `zillow_clean.csv` from the wide Zillow file.
