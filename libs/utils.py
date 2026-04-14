"""
utils.py
---------------
Utility functions for time series forecasting, including data preparation, model training,
evaluation, and feature importance analysis for linear regression,XGBoost, and LightGBM models.
Also includes functions for calculating inclusive days between dates, formatting durations,
and computing Haversine distances.

Key functions:
- calculate_inclusive_days: computes total days between two dates, including boundaries.
- format_duration: converts seconds into a human-readable format (h/m/s).
- calculate_haversine_distance: calculates distance between two geographic points.
- prepare_data: prepares time series data for supervised learning with scaling and windowing.
- linear_regression: trains and evaluates a linear regression model with optional feature importance.
- xgboost_regression: trains and evaluates an XGBoost model with early stopping and feature importance.
- lgbm_regression: trains and evaluates a LightGBM model with early stopping and feature importance.
- mae_by_period_climate: computes MAE breakdowns by year and climatological season.

"""

from datetime import datetime
import warnings
import math
import random
import string
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import lightgbm as lgbm
import xgboost as xgb


def calculate_inclusive_days(start_date_str: str, end_date_str: str) -> int:
    """
    Calculates the total number of days between two dates, including both
    the start and end dates.

    Args:
        start_date_str (str): The start date string in "yyyy-mm-dd" format.
        end_date_str (str): The end date string in "yyyy-mm-dd" format.

    Returns:
        int: The total number of days (including the boundaries).

    Raises:
        ValueError: If the start date is later than the end date.
    """

    # 1. Convert date strings to date objects
    # The format "%Y-%m-%d" corresponds to "YYYY-MM-DD"
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    except ValueError as e:
        # Handle case where the date string format is incorrect
        raise ValueError(
            f"Date format error: Input must be 'yyyy-mm-dd'. Original error: {e}"
        ) from e

    # 2. Check for logical date order
    if start_date > end_date:
        raise ValueError(
            f"Invalid date range: Start date ({start_date_str}) cannot be later than end date ({end_date_str})."
        )

    # 3. Calculate the difference (timedelta)
    difference = end_date - start_date

    # 4. Extract the number of days from the difference
    # The difference is the number of days *between* the two dates.
    # To include both boundary dates, 1 day must be added.
    inclusive_day_count = difference.days + 1

    return inclusive_day_count


def format_duration(seconds):
    if seconds < 60:
        return f"{int(seconds):02d} s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d} m {secs:02d} s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d} h {minutes:02d} m {secs:02d} s"


def calculate_haversine_distance(latA, lonA, latB, lonB):
    """
    Calculates the distance in kilometers between two geographical points
    (A and B) using the Haversine formula.

    Args:
        latA (float): Latitude of point A (in degrees).
        lonA (float): Longitude of point A (in degrees).
        latB (float): Latitude of point B (in degrees).
        lonB (float): Longitude of point B (in degrees).

    Returns:
        float: The distance between the two points in kilometers.
    """
    # Average Earth radius in kilometers
    EARTH_RADIUS_KM = 6371.0

    # Convert degrees to radians
    latA_rad = math.radians(latA)
    lonA_rad = math.radians(lonA)
    latB_rad = math.radians(latB)
    lonB_rad = math.radians(lonB)

    # Coordinate differences
    dlon = lonB_rad - lonA_rad
    dlat = latB_rad - latA_rad

    # Haversine formula
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(latA_rad) * math.cos(latB_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance calculation
    distance_km = EARTH_RADIUS_KM * c

    return distance_km


def prepare_data(
    target_series,
    predictor_series,
    window_size,
    horizon,
    test_size,
    train_split,
):
    """
    Prepare time series data for supervised learning.

    This version fixes a critical bug:
    - y must ALWAYS come from the true target, not from column 0 of predictor_series
      unless the target is explicitly included as the first predictor.
    """

    # =========================
    # --- Align series length ---
    # =========================

    series_lengths = [len(s) for s in predictor_series]
    if len(set(series_lengths)) > 1:
        min_length = min(series_lengths)
        predictor_series = [s[:min_length] for s in predictor_series]
        target_series = target_series[:min_length]

    target_series = np.array(target_series)

    # =====================================================
    # --- Convert predictors into (n_samples, n_features) ---
    # =====================================================
    data = np.array(predictor_series).T

    # =================================================================
    # --- Split predictors into train/val/test with window adjustment ---
    # =================================================================
    extra = window_size + horizon - 1  # extra rows needed for window+horizon

    test_data = data[-(test_size + extra) :]
    train_val_data = data[:-test_size]

    # Train / validation split
    train_size = int(len(train_val_data) * train_split)
    train_data = train_val_data[:train_size]
    val_data = train_val_data[train_size:]

    # =======================
    # --- Scale predictors ---
    # =======================
    scalers = []
    train_scaled = np.zeros_like(train_data, dtype=np.float32)
    val_scaled = np.zeros_like(val_data, dtype=np.float32)
    test_scaled = np.zeros_like(test_data, dtype=np.float32)

    for i in range(data.shape[1]):
        scaler = StandardScaler()
        train_scaled[:, i] = scaler.fit_transform(
            train_data[:, i].reshape(-1, 1)
        ).flatten()
        val_scaled[:, i] = scaler.transform(val_data[:, i].reshape(-1, 1)).flatten()
        test_scaled[:, i] = scaler.transform(test_data[:, i].reshape(-1, 1)).flatten()
        scalers.append(scaler)

    # ===============================================================
    # --- Scale the TRUE target properly (critical bug fixed here) ---
    # ===============================================================
    # target needs same slicing as predictors
    target_extra = window_size + horizon - 1

    target_test = target_series[-(test_size + target_extra) :]
    target_train_val = target_series[:-test_size]

    train_size_target = int(len(target_train_val) * train_split)

    target_train = target_train_val[:train_size_target]
    target_val = target_train_val[train_size_target:]

    scaler_target = StandardScaler()

    scaler_target.fit(target_train.reshape(-1, 1))

    target_scaled_train = scaler_target.transform(target_train.reshape(-1, 1)).flatten()
    target_scaled_val = scaler_target.transform(target_val.reshape(-1, 1)).flatten()
    target_scaled_test = scaler_target.transform(target_test.reshape(-1, 1)).flatten()

    # ====================================================
    # --- Create sequences from predictors and true y ---
    # ====================================================
    def create_sequences(X, y, window_size, horizon):
        X_out, y_out = [], []
        for i in range(len(X) - window_size - horizon + 1):
            X_out.append(X[i : i + window_size].flatten())
            y_out.append(y[i + window_size + horizon - 1])
        return np.array(X_out, dtype=np.float32), np.array(y_out, dtype=np.float32)

    X_train, y_train = create_sequences(
        train_scaled, target_scaled_train, window_size, horizon
    )
    X_val, y_val = create_sequences(val_scaled, target_scaled_val, window_size, horizon)
    X_test, y_test = create_sequences(
        test_scaled, target_scaled_test, window_size, horizon
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_target


def __add_statistics__(results, y_true_test, y_pred_test):
    # Absolute error percentiles
    # The P90 tells: "90% of the forecasts have an error of less than X°C".
    errors = np.abs(y_true_test - y_pred_test)
    p50, p90, p95 = np.percentile(errors, [50, 90, 95])
    results["absolute_error_percentiles"] = {
        "p50": round(float(p50), 4),
        "p90": round(float(p90), 4),
        "p95": round(float(p95), 4),
    }

    # Bias (mean signed error)
    bias = np.mean(y_pred_test - y_true_test)
    results["bias"] = round(float(bias), 4)

    # Threshold success rate
    # "82% of forecasts within 1°C" says a lot.
    results["rate_1deg"] = round(
        float(np.mean(np.abs(y_true_test - y_pred_test) <= 1.0)), 4
    )
    results["rate_2deg"] = round(
        float(np.mean(np.abs(y_true_test - y_pred_test) <= 2.0)), 4
    )

    # Worst error
    results["max_error"] = round(float(np.max(np.abs(y_true_test - y_pred_test))), 4)

    # --- Extreme Events Statistics ---

    # 1. Threshold definition (θ): the coldest 5% of observed nights
    # y_true_test is used to define what is "truly" extreme for this specific station
    threshold_5pct = np.percentile(y_true_test, 5)
    results["extreme_threshold_5pct"] = round(float(threshold_5pct), 2)

    # 2. Binary vectors creation (Boolean)
    observed_extreme = y_true_test < threshold_5pct
    predicted_extreme = y_pred_test < threshold_5pct

    # 3. Confusion matrix calculation
    tp = np.sum(observed_extreme & predicted_extreme)  # True Positives
    fp = np.sum(~observed_extreme & predicted_extreme)  # False Positives
    fn = np.sum(observed_extreme & ~predicted_extreme)  # False Negatives
    tn = np.sum(~observed_extreme & ~predicted_extreme)  # True Negatives

    # 4. Recommended metrics calculation (with zero division protection)
    # POD: Probability of Detection (Hit Rate)
    # FAR: False Alarm Ratio
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0

    # Heidke Skill Score (HSS)
    expected_correct = ((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp)) / (
        tp + fp + tn + fn
    )
    hss = (
        (tp + tn - expected_correct) / (tp + fp + tn + fn - expected_correct)
        if (tp + fp + tn + fn - expected_correct) != 0
        else 0
    )

    # Peirce Skill Score (PSS)
    # POFD: Probability of False Detection (False Alarm Rate)
    pofd = fp / (fp + tn) if (fp + tn) > 0 else 0
    pss = pod - pofd

    # Adding to results dictionary
    results["extreme_events_5pct"] = {
        "pod": round(float(pod), 4),
        "far": round(float(far), 4),
        "hss": round(float(hss), 4),
        "pss": round(float(pss), 4),
        "n_events": int(tp + fn),  # Total number of actual extreme events
    }


def linear_regression(
    target_series,
    predictor_series,
    window_size,
    horizon,
    test_size,
    train_split,
    start_date,
    end_date,
    full_calculation=False,
):
    """
    Train and evaluate a simple linear regression model for time series forecasting.

    This function prepares the data using a rolling-window approach, trains a linear
    regression model, and computes performance metrics on validation and test sets,
    including seasonal breakdowns and optional feature importance analysis.

    Parameters
    ----------
    target_series : np.ndarray
        The target time series to be predicted. Typically represents the temperature
        or another meteorological variable at a single location.
    predictor_series : list of np.ndarray
        A list of time series used as predictors (features). Each array must have
        the same length as `target_series`. The target may or may not be included
        as the first predictor (index 0).
    window_size : int
        Number of past time steps (days, hours, etc.) used to build each input window.
    horizon : int
        Forecast horizon — how far ahead the model predicts (e.g., 1 for next day).
    test_size : int
        Number of time steps reserved for testing at the end of the series.
    train_split : float
        Fraction (0–1) of the remaining data to use for training; the rest is used
        for validation.
    start_date : str or datetime
        Start date for the date range used in seasonal breakdown.
    end_date : str or datetime
        End date for the date range used in seasonal breakdown.
    with_importance : bool, optional
        If True, compute and return feature importances (coefficient-based gain and
        split-like metrics). Defaults to False.

    Returns
    -------
    dict
        Dictionary containing:
        - mae_val, mse_val, r2_val: Validation metrics
        - mae_test, mse_test, r2_test: Test metrics
        - breakdown: Annual and seasonal MAE breakdown
        - importance_gain: Feature importances based on absolute coefficients (if with_importance=True)
        - importance_split: Feature importances based on standardized coefficients (if with_importance=True)

    Notes
    -----
    - The data preparation step is handled by `prepare_data()`, which normalizes
      all predictors and builds windowed input matrices.
    - The target variable is inverse-transformed after prediction using the
      corresponding scaler to report metrics in the original scale.
    """

    def r2_calculation(true_val, pred_val):
        """Compute the coefficient of determination (R²)."""
        ss_res = np.sum((true_val - pred_val) ** 2)
        ss_tot = np.sum((true_val - true_val.mean()) ** 2)
        return 1.0 if ss_tot == 0 else 1 - (ss_res / ss_tot)

    # --- Data preparation (scaling, windowing, and splitting) ---

    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_target = (
            prepare_data(
                target_series,
                predictor_series,
                window_size,
                horizon,
                test_size,
                train_split,
            )
        )

        if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
            print("WARNING: Empty datasets after preparation. Cannot train the model.")
            return {}

    except Exception as e:
        print(f"Error during data preparation for Linear Regression: {e}")
        return {}

    # --- Model training ---
    model = LinearRegression()
    model.fit(X_train, y_train)

    # --- Validation phase ---
    y_pred_val_norm = model.predict(X_val)
    y_true_val = scaler_target.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_pred_val = scaler_target.inverse_transform(
        y_pred_val_norm.reshape(-1, 1)
    ).flatten()

    # Compute validation metrics
    mae_val = mean_absolute_error(y_true_val, y_pred_val)
    mse_val = mean_squared_error(y_true_val, y_pred_val)
    r2_val = r2_calculation(y_true_val, y_pred_val)

    # --- Test phase ---
    y_pred_test_norm = model.predict(X_test)
    y_true_test = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_test = scaler_target.inverse_transform(
        y_pred_test_norm.reshape(-1, 1)
    ).flatten()

    # Compute test metrics
    mae_test = mean_absolute_error(y_true_test, y_pred_test)
    mse_test = mean_squared_error(y_true_test, y_pred_test)
    r2_test = r2_calculation(y_true_test, y_pred_test)

    # --- Results dictionary ---
    results = {
        "mae_val": round(mae_val, 4),
        "mae_test": round(mae_test, 4),
        "mse_val": round(mse_val, 4),
        "mse_test": round(mse_test, 4),
        "r2_val": round(float(r2_val), 4),
        "r2_test": round(float(r2_test), 4),
    }

    # --- Feature importance (if requested) ---
    if full_calculation:

        # --- Annual and seasonal breakdowns ---
        dates = pd.date_range(start_date, end_date, freq="D")
        breakdown = mae_by_period_climate(y_true_test, y_pred_test, dates)
        # For linear regression, we use the model coefficients
        coefficients = model.coef_

        # importance_gain: Absolute value of coefficients
        # Represents the direct impact of each feature on predictions
        importance_gain = np.abs(coefficients)

        # importance_split: Standardized importance based on feature variance
        # Accounts for the scale of features by considering their standard deviation
        feature_std = np.std(X_train, axis=0)
        importance_split = np.abs(coefficients * feature_std)

        results["breakdown"] = breakdown
        __add_statistics__(results, y_true_test, y_pred_test)
        results["importance_gain"] = importance_gain.tolist()
        results["importance_split"] = importance_split.tolist()

    return results


def mae_by_period_climate(y_true, y_pred, dates):

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    ).assign(
        year=lambda x: x["date"].dt.year,
        season=lambda x: x["date"].dt.month.map(
            lambda m: (
                "winter"
                if m in (12, 1, 2)
                else (
                    "spring"
                    if m in (3, 4, 5)
                    else "summer" if m in (6, 7, 8) else "autumn"
                )
            )
        ),
    )

    # === Annual MAE ===
    annual_mae = {
        str(year): round(mean_absolute_error(sub["y_true"], sub["y_pred"]), 3)
        for year, sub in df.groupby("year", sort=True)
    }

    # === Seasonal MAE ===
    season_order = ["winter", "spring", "summer", "autumn"]

    seasonal_mae = {
        season: round(mean_absolute_error(sub["y_true"], sub["y_pred"]), 3)
        for season, sub in sorted(
            df.groupby("season"),
            key=lambda kv: season_order.index(kv[0]),
        )
    }

    return {"annual": annual_mae, "seasonal": seasonal_mae}


# def mae_by_period_climate_autre2(y_true, y_pred, dates):
#     """
#     Calcule la MAE par année et par saison climatologique.

#     Paramètres
#     ----------
#     y_true : array-like
#         Valeurs réelles.
#     y_pred : array-like
#         Valeurs prédites.
#     dates : array-like (de type datetime)
#         Dates correspondant aux valeurs.

#     Retourne
#     --------
#     dict
#         {
#           "annual": {"2020": 1.23, "2021": 1.18, ...},
#           "seasonal": {"winter": 1.22, "spring": 1.08, "summer": 0.98, "autumn": 1.18}
#         }
#     """

#     df = pd.DataFrame(
#         {"date": pd.to_datetime(dates), "y_true": y_true, "y_pred": y_pred}
#     )

#     # === MAE annuelle ===
#     df["year"] = df["date"].dt.year
#     annual_mae = {
#         str(year): round(mean_absolute_error(sub.y_true, sub.y_pred), 3)
#         for year, sub in df.groupby("year")
#     }

#     # === MAE saisonnière (climatologique) ===
#     # saisons climatologiques : DJF, MAM, JJA, SON
#     def season_from_month(m):
#         if m in [12, 1, 2]:
#             return "winter"
#         elif m in [3, 4, 5]:
#             return "spring"
#         elif m in [6, 7, 8]:
#             return "summer"
#         else:
#             return "autumn"

#     df["season"] = df["date"].dt.month.map(season_from_month)
#     # Ordre conventionnel : hiver, printemps, été, automne
#     season_order = ["winter", "spring", "summer", "autumn"]

#     seasonal_mae = {
#         season: round(mean_absolute_error(sub.y_true, sub.y_pred), 3)
#         for season, sub in sorted(
#             df.groupby("season"), key=lambda kv: season_order.index(kv[0])
#         )
#     }

#     return {"annual": annual_mae, "seasonal": seasonal_mae}


# # --- Fin de votre fonction mae_by_period_climate ---


def xgboost_regression(
    target_series,
    predictor_series,
    window_size,
    horizon,
    test_size,
    train_split,
    start_date,
    end_date,
    the_seed,
    with_importance=False,
):
    """
    Trains an XGBoost model and returns MAE with seasonal breakdowns.
    Uses early stopping for optimal performance and speed.

    Args:
        target_series: The time series to predict.
        predictor_series: List of time series to use as features.
        window_size: Size of the time window.
        horizon: Prediction horizon.
        test_size: Proportion of the dataset to include in the test split.
        train_split: Proportion of the remaining data for training (after test_size).
        start_date: Start date for the full data range.
        end_date: End date for the full data range.
        the_seed: Random seed for reproducibility.
        with_importance: If True, include feature importances in the results.

    Returns:
        Dictionary of results containing performance metrics and optionally feature importances.
    """
    # 1. Data Preparation
    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_target = (
            prepare_data(
                target_series,
                predictor_series,
                window_size,
                horizon,
                test_size,
                train_split,
            )
        )

        if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
            print("WARNING: Empty datasets after preparation. Cannot train the model.")
            return {}  # Return an empty dictionary in case of error

    except Exception as e:
        print(f"Error during data preparation for XGBoost: {e}")
        return {}

    # 2. XGBoost Model Configuration (for xgb.train)
    # We define parameters slightly differently for the native API
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "eta": 0.02,  # learning_rate is called 'eta' in native API
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "lambda": 0.1,
        "alpha": 0.1,
        "max_depth": 6,
        "seed": the_seed,
        # "nthread": -1,  # n_jobs is called 'nthread' in native API
        "nthread": 1,  # For reproducibility across platforms
        "tree_method": "hist",
        # "silent": 1,  # Suppress verbose output // OBSOLETE parameter
    }

    # Create DMatrix objects for native XGBoost API
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)  # Dtest for prediction later

    evals = [(dtrain, "train"), (dval, "validation")]

    # 3. Training with Early Stopping using native API
    # The early_stopping_rounds argument is directly supported by xgb.train
    bst = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=2000,  # Max number of boosting rounds
        evals=evals,
        early_stopping_rounds=50,  # THIS WILL WORK with xgb.train
        verbose_eval=False,  # Suppress evaluation logs
    )

    # 4. Predictions
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Predictions using the booster. Use bst.best_iteration for early stopping
        y_pred_val_norm = bst.predict(dval, iteration_range=(0, bst.best_iteration + 1))
        y_pred_test_norm = bst.predict(
            dtest, iteration_range=(0, bst.best_iteration + 1)
        )

    # 5. Denormalization
    y_true_val = scaler_target.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_pred_val = scaler_target.inverse_transform(
        y_pred_val_norm.reshape(-1, 1)
    ).flatten()

    y_true_test = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_test = scaler_target.inverse_transform(
        y_pred_test_norm.reshape(-1, 1)
    ).flatten()

    # Calculate global MAEs
    mae_val = mean_absolute_error(y_true_val, y_pred_val)
    mse_val = mean_squared_error(y_true_val, y_pred_val)
    r2_val = r2_score(y_true_val, y_pred_val)

    mae_test = mean_absolute_error(y_true_test, y_pred_test)
    mse_test = mean_squared_error(y_true_test, y_pred_test)
    r2_test = r2_score(y_true_test, y_pred_test)

    # --- Correction pour les dates du jeu de test ---
    full_date_range = pd.date_range(start_date, end_date, freq="D")
    test_dates = full_date_range[-len(y_true_test) :]

    if len(test_dates) != len(y_true_test):
        print(
            f"ERREUR: La longueur de test_dates ({len(test_dates)}) ne correspond pas à la longueur de y_true_test ({len(y_true_test)})."
        )
        print(
            "Veuillez vérifier votre fonction `prepare_data` et la manière dont `start_date`/`end_date` sont utilisées."
        )
        return {}

    # Annual and seasonal breakdowns
    breakdown = mae_by_period_climate(y_true_test, y_pred_test, test_dates)

    results = {
        "mae_val": round(mae_val, 4),
        "mae_test": round(mae_test, 4),
        "mse_val": round(mse_val, 4),
        "mse_test": round(mse_test, 4),
        "r2_val": round(r2_val, 4),
        "r2_test": round(r2_test, 4),
        "breakdown": breakdown,
    }
    __add_statistics__(results, y_true_test, y_pred_test)
    if with_importance:
        # For native booster, feature importances are directly available
        feature_importances_gain = bst.get_score(importance_type="gain")
        feature_importances_split = bst.get_score(
            importance_type="weight"
        )  # 'weight' is for splits

        # If X_train was a pandas DataFrame, you could use its columns for feature names.
        # Otherwise, XGBoost assigns f0, f1, f2...
        if hasattr(X_train, "columns"):  # Check if X_train is a DataFrame
            feature_names = X_train.columns
        else:
            # Generate generic feature names f0, f1, f2...
            feature_names = [f"f{i}" for i in range(X_train.shape[1])]

        # Ensure consistent order for output
        # Create full dictionaries first, then extract sorted arrays
        importance_gain_dict = {
            name: feature_importances_gain.get(name, 0.0) for name in feature_names
        }
        importance_split_dict = {
            name: feature_importances_split.get(name, 0.0) for name in feature_names
        }

        # Sort based on default feature names (f0, f1, ...) or whatever order you prefer
        # Assuming numerical feature names if not explicitly set
        sorted_feature_keys = sorted(
            importance_gain_dict.keys(),
            key=lambda x: (
                int(x.replace("f", "")) if x.startswith("f") and x[1:].isdigit() else x
            ),
        )
        results["importance_gain"] = [
            importance_gain_dict[key] for key in sorted_feature_keys
        ]
        results["importance_split"] = [
            importance_split_dict[key] for key in sorted_feature_keys
        ]
    return results


##################################################################################################


def lgbm_regression(
    target_series,
    predictor_series,
    window_size,
    horizon,
    test_size,
    train_split,
    start_date,
    end_date,
    the_seed,
    with_importance=False,
):
    """
    Trains a LightGBM model and returns MAE with seasonal breakdowns.
    Uses early stopping for optimal performance and speed.

    Args:
        target_series: The time series to predict.
        predictor_series: List of time series to use as features.
        window_size: Size of the time window.
        horizon: Prediction horizon.
        test_size: Proportion of the dataset to include in the test split.
        train_split: Proportion of the remaining data for training (after test_size).
        start_date: Start date for the date range.
        end_date: End date for the date range.
        the_seed: Random seed for reproducibility.
        with_importance: If True, include feature importances in the results.

    Returns:
        Dictionary of results containing performance metrics and optionally feature importances.
    """
    # 1. Data Preparation
    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_target = (
            prepare_data(
                target_series,
                predictor_series,
                window_size,
                horizon,
                test_size,
                train_split,
            )
        )

        if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
            print("WARNING: Empty datasets after preparation. Cannot train the model.")
            return {}  # Return an empty dictionary in case of error

    except Exception as e:
        print(f"Error during data preparation for LightGBM: {e}")
        return {}

    # 2. LightGBM Model Configuration
    # "Fast version for GA (adjust as needed)
    lgbm_params = {
        "objective": "mae",
        "metric": "mae",
        "num_threads": 6,  # 8,  # si disponible
        "n_estimators": 1000,  # early stopping compensera
        "learning_rate": 0.1,  # 0.05,  # x2.5 vs 0.02
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "num_leaves": 15,  # 20,  # réduit
        "max_bin": 63,  # 127,  # réduit (défaut 255)
        "min_child_samples": 50,  # réduit les splits inutiles
        "verbose": -1,
        "seed": the_seed,
        "boosting_type": "gbdt",
    }
    model = lgbm.LGBMRegressor(**lgbm_params)

    # 3. Training with Early Stopping
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgbm.early_stopping(stopping_rounds=50, verbose=False)],
    )

    # 4. Predictions
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred_val_norm = model.predict(X_val, num_iteration=model.best_iteration_)
        y_pred_test_norm = model.predict(X_test, num_iteration=model.best_iteration_)

    # 5. Denormalization
    y_true_val = scaler_target.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_pred_val = scaler_target.inverse_transform(
        y_pred_val_norm.reshape(-1, 1)
    ).flatten()

    y_true_test = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_test = scaler_target.inverse_transform(
        y_pred_test_norm.reshape(-1, 1)
    ).flatten()

    # Calculate global MAEs

    mae_val = mean_absolute_error(y_true_val, y_pred_val)
    mse_val = mean_squared_error(y_true_val, y_pred_val)
    r2_val = r2_score(y_true_val, y_pred_val)

    mae_test = mean_absolute_error(y_true_test, y_pred_test)
    mse_test = mean_squared_error(y_true_test, y_pred_test)
    r2_test = r2_score(y_true_test, y_pred_test)

    # Annual and seasonal breakdowns
    dates = pd.date_range(start_date, end_date, freq="D")
    # dates = pd.date_range("2022-07-01", "2024-06-30", freq="D")
    breakdown = mae_by_period_climate(y_true_test, y_pred_test, dates)

    results = {
        "mae_val": round(mae_val, 4),
        "mae_test": round(mae_test, 4),
        "mse_val": round(mse_val, 4),
        "mse_test": round(mse_test, 4),
        "r2_val": round(r2_val, 4),
        "r2_test": round(r2_test, 4),
        "breakdown": breakdown,
    }
    __add_statistics__(results, y_true_test, y_pred_test)
    if with_importance:
        booster = model.booster_
        feature_names = booster.feature_name()

        # Retrieve the underlying Booster
        booster = model.booster_

        # Extract importances based on two criteria
        importance_gain = booster.feature_importance(importance_type="gain")
        importance_split = booster.feature_importance(importance_type="split")
        results["importance_gain"] = importance_gain.tolist()
        results["importance_split"] = importance_split.tolist()

    return results


def find_nearest_point(cursor, target_lat, target_lon, var_name=None):
    """
    Finds the point closest to the specified coordinates in the SQLite database.

    Parameters:
    -----------

    target_lat : float
        Latitude of the target point
    target_lon : float
        Longitude of the target point
    var_name : str, optional
        Name of the climate variable (None for all variables)

    Returns:
    --------
    dict : Information about the nearest point along with its time series
    """
    # Connection to database

    # Build the SQL query
    if var_name:
        query = """
        SELECT 
            fs.id, 
            fs.var_name, 
            s.latitude, 
            s.longitude, 
            fs.serie,
            (s.latitude - ?)*(s.latitude - ?) + (s.longitude - ?)*(s.longitude - ?) AS distance_squared
        FROM 
            feature_series AS fs
        JOIN 
            sites AS s ON fs.site_id = s.site_id
        WHERE 
            fs.var_name = ?
        ORDER BY 
            distance_squared ASC
        LIMIT 1;
        """
        params = (target_lat, target_lat, target_lon, target_lon, var_name)
    else:
        query = """
        SELECT id, var_name, latitude, longitude, serie,
               (latitude - ?)*(latitude - ?) + (longitude - ?)*(longitude - ?) AS distance_squared
        FROM climate_series
        ORDER BY distance_squared ASC
        LIMIT 1
        """
        params = (target_lat, target_lat, target_lon, target_lon)

    # Execute the query
    cursor.execute(query, params)
    result = cursor.fetchone()

    if result:
        id, var_name, latitude, longitude, serie, distance_squared = result

        # Calculate the approximate distance in kilometers (1 degree ≈ 111 km)
        distance_km = math.sqrt(distance_squared) * 111

        return {
            "id": id,
            "var_name": var_name,
            "latitude": latitude,
            "longitude": longitude,
            "distance_km": distance_km,
            "serie": serie,
        }
    else:
        return None


def generate_id(length: int = 6) -> str:
    """
    Generates a random alphanumeric identifier of the specified length.
    Usable characters: a-z, A-Z, 0-9.

    :param length: The desired length of the identifier (default is 6).
    :return: The randomly generated identifier (string).
    """
    # Define the set of usable characters
    # string.ascii_letters = a-z + A-Z
    # string.digits = 0-9
    characters = string.ascii_letters + string.digits

    # Randomly choose 'length' characters from the set, with replacement.
    # The result is a list of characters, which is then joined into a single string.
    identifier = "".join(random.choices(characters, k=length))

    return identifier
