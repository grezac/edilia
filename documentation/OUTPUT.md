# JSON Output File Structure

## Introduction

The JSON output files documented here are produced by a statistical modelling pipeline designed to predict a target variable (e.g. daily minimum temperature) for a given weather station. The pipeline relies on a **genetic algorithm** that selects, from a large pool of georeferenced predictors stored in a SQLite database, the optimal subset of features to maximise prediction performance.

For each pipeline run, three regression models are trained and evaluated in parallel on the same selected predictors:

- a **linear** model (Ridge regression or similar),
- a **LightGBM** model,
- an **XGBoost** model.

Performance is measured on two distinct datasets: a **validation** set (used during optimisation) and a **test** set (unseen data, used for final evaluation).

Each file corresponds to a unique run, characterised by:

- a **target station** (e.g. `birmingham`),
- a **maximum number of predictors** allowed (e.g. `npreds90` for 90 predictors),
- a **random seed** fixing the initialisation of the genetic algorithm.

The file naming convention is: `{station}__{npreds}{N}.json`.

---

## General Structure

The JSON file is a single root object with six top-level keys:

```
{
  "total_elapsed_time": ...,
  "generations": ...,
  "linear_stats": { ... },
  "lightGBM_stats": { ... },
  "XGBoost_stats": { ... },
  "predictors": [ ... ]
}
```

---

## Key Descriptions

### `total_elapsed_time`

**Type:** `string`

Total wall-clock duration of the run, formatted as `HH h MM m SS s`.

**Example:** `"01 h 21 m 35 s"`

---

### `generations`

**Type:** `integer`

Number of generations completed by the genetic algorithm before convergence or stopping.

**Example:** `1455`

---

### `linear_stats` / `lightGBM_stats` / `XGBoost_stats`

These three keys share an identical structure. Each contains the performance metrics for the corresponding model (linear, LightGBM, and XGBoost respectively), evaluated on both the validation and test sets.

#### Common sub-keys

| Key | Type | Description |
|-----|------|-------------|
| `mae_val` | `float` | Mean Absolute Error on the **validation** set |
| `mae_test` | `float` | Mean Absolute Error on the **test** set |
| `mse_val` | `float` | Mean Squared Error on the **validation** set |
| `mse_test` | `float` | Mean Squared Error on the **test** set |
| `r2_val` | `float` | R² coefficient of determination on the **validation** set |
| `r2_test` | `float` | R² coefficient of determination on the **test** set |
| `bias` | `float` | Mean prediction bias (systematic error, in degrees) |
| `rate_1deg` | `float` | Proportion of predictions with an absolute error ≤ 1 °C |
| `rate_2deg` | `float` | Proportion of predictions with an absolute error ≤ 2 °C |
| `max_error` | `float` | Maximum absolute error observed on the test set |
| `extreme_threshold_5pct` | `float` | Threshold defining extreme events (bottom 5% of observed values) |

#### Sub-key `breakdown`

Breakdown of the MAE by temporal sub-period.

```json
"breakdown": {
  "annual": {
    "2023": ...,
    "2024": ...
  },
  "seasonal": {
    "winter": ...,
    "spring": ...,
    "summer": ...,
    "autumn": ...
  }
}
```

| Key | Type | Description |
|-----|------|-------------|
| `annual` | `object` | MAE per year (keys are year strings) |
| `seasonal` | `object` | MAE per meteorological season (`winter`, `spring`, `summer`, `autumn`) |

#### Sub-key `absolute_error_percentiles`

Percentiles of the absolute error distribution on the test set.

| Key | Type | Description |
|-----|------|-------------|
| `p50` | `float` | 50th percentile (median) of the absolute error |
| `p90` | `float` | 90th percentile of the absolute error |
| `p95` | `float` | 95th percentile of the absolute error |

#### Sub-key `extreme_events_5pct`

Detection scores for extreme events, defined as days where the observed value falls below the `extreme_threshold_5pct` threshold (bottom 5% of the series).

| Key | Type | Description |
|-----|------|-------------|
| `pod` | `float` | *Probability of Detection* — hit rate for extreme events |
| `far` | `float` | *False Alarm Rate* — proportion of false alarms among predicted extremes |
| `hss` | `float` | *Heidke Skill Score* — skill score relative to a random forecast |
| `pss` | `float` | *Peirce Skill Score* (also known as True Skill Statistic) |
| `n_events` | `integer` | Total number of extreme events observed in the test set |

---

### `predictors`

**Type:** `array` of objects

List of predictors (features) selected by the genetic algorithm. Each element represents a georeferenced predictor from the SQLite database.

#### Sub-keys of each predictor object

| Key | Type | Description |
|-----|------|-------------|
| `id` | `integer` | Unique identifier of the predictor in the SQLite database |
| `lat` | `float` | Latitude of the associated grid point (decimal degrees) |
| `lon` | `float` | Longitude of the associated grid point (decimal degrees) |
| `distance_km` | `float` | Distance in kilometres between the predictor grid point and the target station |
| `var_name` | `string` | Name of the meteorological or climatological variable (e.g. `2m_temperature_daily_minimum`, `total_precipitation_daily_sum`) |
| `linear_gain` | `float` | Predictor importance in the linear model (normalised coefficient) |
| `linear_split` | `float` | Same as `linear_gain` for the linear model (may differ for tree-based models) |
| `LGBM_gain` | `float` | Predictor importance in the LightGBM model, measured by total *gain* (cumulative loss reduction from splits using this feature) |
| `LGBM_split` | `integer` | Predictor importance in LightGBM measured by the number of times it is used in a split |
| `XGBoost_gain` | `float` | Predictor importance in the XGBoost model, measured by average *gain* |
| `XGBoost_split` | `integer` | Predictor importance in XGBoost measured by the total number of splits |

**Note:** the `gain` and `split` importance metrics are complementary. *Gain* reflects the effective contribution of a feature to error reduction, while *split* indicates how frequently it is used across the decision trees.
