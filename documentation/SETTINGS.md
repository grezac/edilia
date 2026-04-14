# Settings File Structure

## Introduction

The `settings.json` file defines the configuration used by the modelling pipeline based on a **genetic algorithm**. It specifies:

* the list of available **target sites** (weather stations),
* one or more **parameter configurations** controlling data extraction, model training, and evolutionary optimisation.

This configuration system is designed to be extensible and partially inherits from a broader parent project. As a result, some fields may appear redundant or over-specified in the present context but are retained for compatibility and reproducibility.

---

## General Structure

The JSON file is organised into two top-level keys:

```
{
  "SITES": { ... },
  "CONFIGS": { ... }
}
```

---

## `SITES`

### Description

The `SITES` section defines the list of available stations on which the modelling pipeline can operate. Each site is identified by a unique key (e.g. `BIRMINGHAM`, `PARIS`, etc.).

In practice, this list is expected to remain stable, although new stations may be added in the future if they are inserted into the database.

### Structure

Each site is described by the following fields:

| Key         | Type      | Description                                       |
| ----------- | --------- | ------------------------------------------------- |
| `id`        | `integer` | Identifier of the station in the database         |
| `name`      | `string`  | Human-readable name of the station                |
| `label`     | `string`  | Short label used in file names and visual outputs |
| `latitude`  | `float`   | Latitude in decimal degrees                       |
| `longitude` | `float`   | Longitude in decimal degrees                      |

### Example

```json
"BIRMINGHAM": {
  "id": 391229,
  "name": "Birmingham",
  "label": "birmingham",
  "latitude": 52.42,
  "longitude": -1.83
}
```

---

## `CONFIGS`

### Description

The `CONFIGS` section defines one or more configurations controlling the behaviour of the pipeline. Each configuration corresponds to a complete experimental setup.

The file currently contains a single configuration (`GENERIC_CONFIG`), which serves as a **template** for creating additional configurations.

For simplicity, all keys described below should be considered **mandatory**, even if some are not strictly required in all execution contexts.

---

## Configuration Keys

### `data_source`

**Type:** `string`
Relative path to the SQLite database containing all predictors and observations.

---

### `json_dir_name`

**Type:** `string`
Relative path to the directory where output JSON files will be written.

---

### `window_size`

**Type:** `integer`
Number of past time steps used as predictors.

* `1` → uses only the previous day
* `2` → uses the two previous days
* etc.

---

### `horizon`

**Type:** `integer`
Forecast horizon.

* `1` → prediction for the next day (J+1)
* `2` → prediction for J+2
* etc.

---

### `seed`

**Type:** `integer`
Random seed used to initialise all stochastic processes.
Ensures reproducibility of results.

---

### `time_unit`

**Type:** `string`
Time resolution of the data.

**Constraint:** must always be `"day"`.

---

### `test_period`

**Type:** `array[string, string]`
Defines the time range of the **test dataset**.

**Important:**
These values must remain unchanged to reproduce the results reported in the associated scientific article.

---

### `working_period`

**Type:** `array[string, string]`
Defines the time range of the **working dataset**:

```
working dataset = training set + validation set
```

**Important:**
These values must remain unchanged for reproducibility.

---

### `evolution_settings`

**Type:** `object`
Parameters controlling the genetic algorithm.

| Key                    | Type      | Description                          |
| ---------------------- | --------- | ------------------------------------ |
| `population_size`      | `integer` | Number of individuals per generation |
| `crossover_rate`       | `float`   | Probability of crossover             |
| `mutation_rate`        | `float`   | Probability of mutation              |
| `stop_criterion`       | `string`  | Stopping rule                        |
| `stop_criterion_value` | `integer` | Associated threshold                 |

**Constraints:**

* `stop_criterion` must always be `"patience"`.
* Other values can be modified freely, but must be preserved to reproduce published results.

---

### `excluded_vars`

**Type:** `array[string]`
List of variables present in the database but excluded from the predictor search space.

**Important:**
This list must remain unchanged to reproduce the results from the article.

---

### `train_split`

**Type:** `float`
Proportion of the working dataset used for training.

```
validation set = 1 - train_split
```

**Important:**
Must remain unchanged for reproducibility.

---

### `distance_metric`

**Type:** `string`
Distance computation method used for spatial queries.

**Constraint:**
Must remain unchanged (currently `"BallTree"`).

---

### `origin`

**Type:** `string`
Indicates the data source of predictors.

**Constraint:**
Must remain unchanged (currently `"ERA5"`).

---

### `rings`

**Type:** `array[object]`
Defines spatial constraints for predictor selection.

This structure is inherited from a broader project supporting multiple spatial rings. In the present project, only **one ring** is used.

#### Ring structure

| Key              | Type      | Description                                    |
| ---------------- | --------- | ---------------------------------------------- |
| `radius_min`     | `float`   | Minimum distance from the target station (km)  |
| `radius_max`     | `float`   | Maximum distance (km)                          |
| `max_predictors` | `integer` | Maximum number of predictors allowed           |
| `patience_stop`  | `integer` | Early stopping parameter specific to this ring |

**Notes:**

* The structure must be preserved as-is.
* Values can be adjusted for testing purposes (e.g. radii or predictor count).

---

## Adding a New Configuration

To create a new configuration:

1. Copy the existing configuration block.
2. Assign a new key (e.g. `"CONFIG2"`).
3. Modify the desired parameters.

### Example

```json
"CONFIG2": {
  "data_source": "/data/METEO_daily.db",
  "json_dir_name": "/data/outputs",
  "window_size": 2,
  "horizon": 2,
  "seed": 42,
  ...
}
```

---

## Validation and Runtime Adjustments

### Validation

The configuration is automatically:

* **validated**
* **completed (if needed)**

using the `get_settings.py` utility.

---

### Runtime Modifications

Some scripts may override specific parameters dynamically, including:

* number of predictors (`npreds.py`)
* temporal window (`windows.py`)

These modifications are intentional and part of the experimental workflow.

---

## Final Remarks

* The configuration system prioritises **reproducibility** and **flexibility**.
* Several fields are inherited from a larger framework and must be preserved even if they appear unused.
* For scientific publication purposes, specific parameters (periods, excluded variables, splits) must remain strictly unchanged.

---
