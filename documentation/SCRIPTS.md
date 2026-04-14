# Scripts

This document describes the scripts available in this repository (`scripts` folder) and how to use them.

> **Note:** To ensure test reproducibility, all scripts must be run from within the Docker container that replicates the test environment. Use the `run.py` launcher to start the container before executing any script.

---

## Table of contents

- [npreds.py](#npredspy) — Test the influence of the number of predictors
- [windows.py](#windowspy) — Test the influence of the time window size
- [lasso_test.py](#lasso_testpy) — LASSO test script with configurable feature count constraint

---

## `npreds.py`

Tests the effect of varying the number of predictors on model performance. Can also be used for more general experimentation purposes.

### Usage

```bash
python scripts/npreds.py --site <station_name> --seeds <int_list> --num_predictors <int_list>
```

All three parameters are required.

### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `--site` | string | ✅ | Name of the target station |
| `--seeds` | list of int | ✅ | Random seeds to use; the script runs once per seed |
| `--num_predictors` | list of int | ✅ | Numbers of predictors to test |

#### `--site`

Name of the target station. Case-insensitive.

```bash
--site BIRMINGHAM
--site paris
```

#### `--seeds`

A list of integers specifying the random seeds. The script is executed once for each seed, allowing reproducibility and variance estimation across runs.

```bash
--seeds [1,2,3,4,5]
--seeds [42]
```

#### `--num_predictors`

A list of integers specifying the numbers of predictors to test. The script runs once for each combination of seed and predictor count.

```bash
--num_predictors [50,70,90,110]
--num_predictors [90]
```

### Example

Run the script on the Birmingham station with 5 seeds and 4 predictor counts (20 runs total):

```bash
python scripts/npreds.py --site BIRMINGHAM --seeds [1,2,3,4,5] --num_predictors [50,70,90,110]
```

### Output

The script generates one file per run, created in:

```
data/outputs/results/<configuration_name>/seed_<seed_value>/
```

File names include the number of predictors:

```
<station_name>__npreds<num_predictors>.json
```

Examples:

```bash
data/outputs/results/GENERIC_CONFIG/seed_1/brest__npreds90.json
data/outputs/results/SPECIAL_CONFIG/seed_42/plymouth__npreds110.json
```

When all runs are complete, a summary file is written to:

```
data/outputs/results/<configuration_name>/
```

It contains the average results across all seeds, used as reference points.

Examples:

```bash
data/outputs/results/GENERIC_CONFIG/brest__npreds90_summary.json
data/outputs/results/SPECIAL_CONFIG/plymouth__npreds110_summary.json
```

> See `OUTPUT.md` for a detailed description of these files.

### Note

This script can be used to test settings other than the number of predictors. See the `npreds.py` source code and `SETTINGS.md` for more information.

---

## `windows.py`

Tests the effect of varying the time window size on model performance.

### Usage

```bash
python scripts/windows.py --site <station_name> --seeds <int_list> --windows <int_list>
```

All three parameters are required.

### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `--site` | string | ✅ | Name of the target station |
| `--seeds` | list of int | ✅ | Random seeds to use; the script runs once per seed |
| `--windows` | list of int | ✅ | Window sizes to test |

#### `--site`

Name of the target station. Case-insensitive.

```bash
--site BIRMINGHAM
--site paris
```

#### `--seeds`

A list of integers specifying the random seeds. The script is executed once for each seed, allowing reproducibility and variance estimation across runs.

```bash
--seeds [1,2,3,4,5]
--seeds [42]
```

#### `--windows`

A list of integers specifying the time window sizes to test. The script runs once for each combination of seed and window size.

```bash
--windows [1,2,3]
--windows [2]
```

### Example

Run the script on the Nice station with 3 seeds and 2 window sizes (6 runs total):

```bash
python scripts/windows.py --site nice --seeds [3,4,5] --windows [1,2]
```

### Output

The script generates one file per run, created in:

```
data/outputs/results/<configuration_name>/seed_<seed_value>/
```

File names include the window size:

```
<station_name>__w<window_size>.json
```

Examples:

```bash
data/outputs/results/GENERIC_CONFIG/seed_1/brest__w1.json
data/outputs/results/SPECIAL_CONFIG/seed_42/plymouth__w3.json
```

When all runs are complete, a summary file is written to:

```
data/outputs/results/<configuration_name>/
```

It contains the average results across all seeds, used as reference points.

Examples:

```bash
data/outputs/results/GENERIC_CONFIG/NICE_window1_summary.json
data/outputs/results/SPECIAL_CONFIG/PLYMOUTH_window3_summary.json
```

> See `OUTPUT.md` for a detailed description of these files.

### Note

By default, the number of predictors is set to 90. This can be changed in the script.

---

## `lasso_test.py`

LASSO test script with configurable feature count constraint.

### Usage

```bash
python scripts/lasso_test.py --site <station_name> [--num_features <int_list>]
```

Only `--site` is required.

### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `--site` | string | ✅ | Name of the target station |
| `--num_features` | list of int | ❌ | Maximum number of features to use; if omitted, LASSO determines the count automatically |

#### `--site`

Name of the target station. Case-insensitive.

```bash
--site BIRMINGHAM
--site paris
```

#### `--num_features`

```bash
--num_features 90
```

### Output

The script generates a single JSON file in:

```
data/outputs/results/
```

File names include the `num_features` value:

```
lasso_<station_name>_<num_features>features.json
```

If `--num_features` is not specified, the suffix is `_full`:

```
lasso_<station_name>_full.json
```

Examples:

```bash
data/outputs/results/lasso_BREST_90features.json
data/outputs/results/lasso_PARIS_full.json
```

### Notes

- If `--num_features` is not specified, LASSO selects the number of features it deems appropriate (potentially several hundreds).
- Running this script requires at least 16 GB of RAM.
- The LASSO algorithm is deterministic — no seeds are needed.
