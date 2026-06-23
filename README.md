# Predicting Patterns in Sustainability and Development

Exploratory Data Analysis & Machine Learning project — Master's in Data Science, Faculty of Mathematics and Computer Science, University of Bucharest (Jan 2025). Team project (4 members).

Analyzes global energy, emissions, and economic indicators (2000–2020) to uncover patterns in renewable energy adoption, CO2 emissions, and economic development, and to build predictive models on top of them.

## Dataset

[Global Data on Sustainable Energy](https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy) (Kaggle) — 3,649 entries, 20 features, per country/year:
- Access to electricity / clean cooking fuels (%)
- Electricity generation by source: fossil fuels, nuclear, renewables (TWh)
- Low-carbon electricity share, renewable energy share
- CO2 emissions, energy intensity, primary energy consumption per capita
- GDP per capita, GDP growth
- Population density, land area, latitude/longitude

## Pipeline

1. **Continent mapping** — each country mapped to a continent via `pycountry_convert` (unmapped → "Unknown").
2. **Cleaning** — dropped French Guiana (single-year coverage) and 3 columns with >25% missing values (`Renewable-electricity-generating-capacity-per-capita`, `Financial flows to developing countries`, `Renewables (% equivalent primary energy)`).
3. **Imputation** — remaining missing values filled with the continent-wise median.
4. **Feature engineering** — added a `Population` column (`Density × Land Area`) for per-capita analysis.
5. **Scaling** — `StandardScaler` + L2 normalization (`sklearn.preprocessing.normalize`).
6. **EDA** — histograms, boxplots, and scatter plots per numerical feature; correlation matrix; animated Plotly choropleth maps (2000–2020 slider) for electricity sources, CO2 emissions, and GDP per capita.
7. **Split** — 80/20 train/test, both globally and per-continent (economic/energy trends differ significantly by region).

## Models & Results

**Classification** — predicting a country's continent from `Electricity from fossil fuels`, `Value_co2_emissions_kt_by_country`, `Electricity from renewables`, `Electricity from nuclear`, `gdp_per_capita` (all tuned via `GridSearchCV`):

| Model | Accuracy |
|---|---|
| Random Forest | **0.9027** |
| XGBoost | 0.8822 |
| KNN | 0.6425 |

**Regression** (Linear Regression / Random Forest / XGBoost / Gradient Boosting, compared per-continent and globally):
- GDP per capita — from energy consumption, clean fuel/electricity access, renewables generation
- CO2 emissions — from fossil fuel electricity, GDP, renewables electricity, population
- Electricity from fossil fuels — from CO2, renewables, nuclear, land area
- Electricity from nuclear / renewables — world + per-continent, with RFE feature selection

Tree-based models (Random Forest, XGBoost) consistently beat Linear Regression, with R² up to ~0.97–0.98 on several continents (e.g. Africa/Oceania fossil-fuel electricity, North America GDP). South America and Africa were the hardest regions to model accurately across most targets (R² as low as 0.4), suggesting the chosen features don't fully capture regional dynamics there.

**Clustering** — `DBSCAN` per year (2000–2020) on energy consumption per capita, GDP per capita, and continent, visualized as an animated 3D scatter plot. Found 7 stable clusters per year with silhouette scores between 0.67 and 0.80.

## Key findings

- Renewable energy adoption correlates strongly with national wealth; the poorest countries lag furthest behind in the transition away from fossil fuels.
- Random Forest and XGBoost outperform Linear Regression almost everywhere, confirming non-linear relationships dominate this dataset.
- Model performance is highly region-dependent — global models mask large per-continent variance, motivating the per-continent modeling approach used throughout.

## Stack

Python · pandas · NumPy · scikit-learn · XGBoost · SciPy · matplotlib · seaborn · Plotly · pycountry_convert

## Usage

```bash
jupyter notebook "EDA_complet (1).ipynb"
```

Notebook was originally run on Google Colab (`pd.read_csv("/content/...")`) — update the CSV path if running locally; the dataset file is included in this repo.

## Report

Full write-up with methodology, hyperparameter grids, and per-continent metrics: [`EDAProjectReport.pdf`](./EDAProjectReport.pdf).




# THIS IS A FINISHED PROJECT AND NO FURTHER COMMITS WILL BE MADE
