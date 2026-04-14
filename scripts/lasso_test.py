"""
LASSO test script with configurable feature count constraint.

USAGE:
1. Run the script using command-line arguments:
   - For unrestricted LASSO:
     python lasso_test.py --site PARIS
   - To limit to a specific number of features (e.g., 90 to compare with GA):
     python lasso_test.py --site PARIS --num_features 90
   - To test other parsimony settings:
     python lasso_test.py --site PARIS --num_features 150

RESULTS:
- Displayed in the console
- Saved in /data/outputs/results/lasso_[SITE]_[N]features.json
"""

import argparse
import sqlite3
import pickle
import time
from datetime import datetime
import json
import os
import sys
from sklearn.linear_model import LassoCV
import numpy as np
from sklearn.neighbors import BallTree

# Go up one level to reach the root folder "edilia"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from libs.utils import prepare_data, calculate_inclusive_days
from xgboost import XGBRegressor


SITES = {
    "BIRMINGHAM": 391229,
    "BREST": 391227,
    "EDINBURGH": 391231,
    "LYON": 391225,
    "NICE": 391226,
    "PARIS": 391228,
    "PLYMOUTH": 391230,
    "STRASBOURG": 391224,
}


# Number of features to select after LASSO
# None = no constraint (LASSO selects freely)
# 90 = limit to 90 features (for comparison with GA)
# Other value = limit to this value
# N_FEATURES = None  # Adjust this value according to the desired test
# N_FEATURES = 90


def lasso_regression(
    target_series,
    predictor_series,  # here, all predictors
    window_size,
    horizon,
    test_size,
    train_split,
    n_features_to_select=None,  # None = no constraint, otherwise number of features to keep
):

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Preparing data...")
    start_time = time.time()
    # --- Data preparation ---
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_target = prepare_data(
        target_series,
        predictor_series,
        window_size,
        horizon,
        test_size,
        train_split,
    )

    prep_time = time.time() - start_time
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Data prepared in {prep_time/60:.1f} min"
    )
    print(f"Shape X_train: {X_train.shape}, y_train: {y_train.shape}")

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] Starting LASSO (this may take several tens of minutes)..."
    )
    fit_start = time.time()

    # --- LASSO model with internal CV ---

    model = LassoCV(
        cv=5,
        alphas=50,
        max_iter=50000,
        # n_jobs=-1,
        n_jobs=1,  # To avoid potential issues with parallelism in some environments (reproducibility, but slower...)
        random_state=42,
    )  # alphas rather than n_alphas

    model.fit(X_train, y_train)

    fit_time = time.time() - fit_start
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] LASSO terminé en {fit_time/60:.1f} min"
    )

    # --- Feature selection ---
    coef = model.coef_
    selected_idx = np.where(coef != 0)[0]

    print(f"LASSO selected {len(selected_idx)} predictors")

    # --- Limit on the number of features, if requested ---
    if n_features_to_select is not None and len(selected_idx) > n_features_to_select:
        print(
            f"Constraining to top {n_features_to_select} features by coefficient magnitude..."
        )
        # Keep the n_features_to_select coefficients with the highest values (in absolute terms)
        coef_abs = np.abs(coef[selected_idx])
        top_n_indices = np.argsort(coef_abs)[-n_features_to_select:]
        selected_idx = selected_idx[top_n_indices]
        print(f"Selected {len(selected_idx)} features after constraint")
    elif n_features_to_select is not None and len(selected_idx) < n_features_to_select:
        print(
            f"Warning: LASSO selected only {len(selected_idx)} features, less than requested {n_features_to_select}"
        )

    # --- Evaluation ---
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    mae_val = np.mean(np.abs(y_val - y_pred_val))
    mae_test = np.mean(np.abs(y_test - y_pred_test))

    # Denormalization to obtain the actual MAE
    y_pred_val_original = scaler_target.inverse_transform(
        y_pred_val.reshape(-1, 1)
    ).flatten()
    y_val_original = scaler_target.inverse_transform(y_val.reshape(-1, 1)).flatten()

    y_pred_test_original = scaler_target.inverse_transform(
        y_pred_test.reshape(-1, 1)
    ).flatten()
    y_test_original = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae_val_real = np.mean(np.abs(y_val_original - y_pred_val_original))
    mae_test_real = np.mean(np.abs(y_test_original - y_pred_test_original))

    print(f"Actual MAE (unscaled) - Val: {mae_val_real:.4f}, Test: {mae_test_real:.4f}")

    return {
        "mae_val": mae_val,
        "mae_test": mae_test,
        "mae_val_real": mae_val_real,
        "mae_test_real": mae_test_real,
        "selected_indices": selected_idx,
        "n_selected": len(selected_idx),
        "alpha": model.alpha_,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler_target": scaler_target,
    }


def main():
    parser = argparse.ArgumentParser(description="LASSO predictor search process")
    # Mandatory parameter
    parser.add_argument(
        "--site", help="Target site (e.g. PARIS, BIRMINGHAM,STRASBOURG)"
    )
    parser.add_argument(
        "--num_features",
        type=int,
        help="Number of features to select (default: None for no constraint)",
        default=None,
    )
    args = parser.parse_args()
    ref_site = args.site.upper()
    n_features_to_select = args.num_features

    if ref_site not in SITES:
        print(f"Error: site must be one of {list(SITES.keys())}")
        return
    SITE_ID = SITES[ref_site]
    conn = sqlite3.connect("/data/METEO_daily.db")
    conn.row_factory = sqlite3.Row  # To access fields by name
    cursor = conn.cursor()

    # Read the data from the reference site
    print("Selection of all features within a radius of 540 km...")
    query = f"SELECT fs.id, fs.series, s.latitude, s.longitude FROM feature_series fs JOIN sites s ON fs.site_id = s.site_id WHERE fs.id={SITE_ID}"
    cursor.execute(query)
    row = cursor.fetchone()
    series_blob = row[1]

    target_series = pickle.loads(
        series_blob
    )  # storing the series from the reference site
    target_coords_deg = np.array([[row[2], row[3]]])
    target_latitude = row[2]
    target_longitude = row[3]
    target_coords_rad = np.radians(target_coords_deg)

    excluded_vars = [
        "high_cloud_cover_daily_mean",
        "low_cloud_cover_daily_mean",
        "mean_sea_level_pressure_daily_mean",
        "medium_cloud_cover_daily_mean",
        "sea_surface_temperature_daily_mean",
    ]

    # Creating placeholders (e.g., ?, ?, ?)
    placeholders = ", ".join(["?"] * len(excluded_vars))

    query = f"""
            SELECT fs.id, s.latitude, s.longitude 
            FROM feature_series fs 
            JOIN sites s ON fs.site_id = s.site_id 
            WHERE fs.origin = ? 
            AND fs.var_name NOT IN ({placeholders})
        """

    # Execution with the parameters
    params = ["ERA5"] + excluded_vars
    cursor.execute(query, params)
    all_records = cursor.fetchall()

    # Data separation
    site_ids = [record[0] for record in all_records]  # IDs list

    all_sites_coords = [
        [record[1], record[2]] for record in all_records
    ]  # coordinates list
    # conversion from degrees to radians
    all_sites_coords_radians = np.radians(all_sites_coords)

    # Creating the index
    tree = BallTree(all_sites_coords_radians, metric="haversine")

    # Circle parameters
    R_EARTH_KM = 6371

    radius_min = 0
    radius_max = 540 / R_EARTH_KM
    inner_set_indexes = tree.query_radius(target_coords_rad, r=radius_min)[0]
    outer_set_indexes = tree.query_radius(target_coords_rad, r=radius_max)[0]

    # Conversion of indexes into sets of identifiers
    set_inner_c = {site_ids[i] for i in inner_set_indexes}
    set_outer_c = {site_ids[i] for i in outer_set_indexes}
    site_ids = list(set_outer_c - set_inner_c)

    site_ids = [x for x in site_ids]
    print(f"{len(site_ids)} features found")
    print("Reading time series...")

    # SQLite has a limit of 999 parameters, so you need to use chunking
    def chunked(iterable, size):
        for i in range(0, len(iterable), size):
            yield iterable[i : i + size]

    all_rows = []

    i = 1
    for chunk in chunked(site_ids, 900):
        i += 1
        placeholders = ",".join("?" for _ in chunk)
        query = f"""
        SELECT id, series
        FROM feature_series
        WHERE id IN ({placeholders})
        """
        cursor.execute(query, chunk)
        all_rows.extend(cursor.fetchall())

    predictor_series = []
    for id_, series in all_rows:
        predictor_series.append(pickle.loads(series))

    print("Calling lasso regression function...")
    test_size = calculate_inclusive_days("2023-01-01", "2024-12-31")

    if n_features_to_select is None:
        print("Running LASSO without feature constraint (free selection)...")
    else:
        print(f"Running LASSO with constraint to {n_features_to_select} features...")

    results = lasso_regression(
        target_series,
        predictor_series,  # this time = all predictors (typically ~430)
        window_size=1,
        horizon=1,
        test_size=test_size,
        train_split=0.85,
        n_features_to_select=n_features_to_select,
    )
    print(results)

    # XGBoost to get XGBoost_mae
    print(
        f"\n[{datetime.now().strftime('%H:%M:%S')}] Start training XGBoost on the features selected by LASSO..."
    )
    xgb_start = time.time()

    # Extraction of features selected by LASSO
    selected_idx = results["selected_indices"]
    X_train_selected = results["X_train"][:, selected_idx]
    X_test_selected = results["X_test"][:, selected_idx]
    y_train = results["y_train"]
    y_test = results["y_test"]

    # Using the XGBoost regression model included in utils.py

    scaler_target = results["scaler_target"]

    # Training XGBoost
    # xgb_model = XGBRegressor(
    #     n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1
    # )
    N_JOBS = 1  # To avoid potential issues with parallelism in some environments (reproducibility)
    xgb_model = XGBRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=N_JOBS
    )
    xgb_model.fit(X_train_selected, y_train)

    # Prediction with XGBoost
    y_pred_xgb = xgb_model.predict(X_test_selected)

    # Denormalization to get the actual MAE
    y_pred_xgb_original = scaler_target.inverse_transform(
        y_pred_xgb.reshape(-1, 1)
    ).flatten()
    y_test_original = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Computing MAE
    XGBoost_mae = np.mean(np.abs(y_test_original - y_pred_xgb_original))

    xgb_time = time.time() - xgb_start
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] XGBoost completed in {xgb_time/60:.1f} min"
    )
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS:")
    print(f"{'='*60}")
    print(f"LASSO alone      - MAE / test: {results['mae_test_real']:.4f}")
    print(f"LASSO + XGBoost - MAE / test: {XGBoost_mae:.4f}")
    print(
        f"Improvement: {((results['mae_test_real'] - XGBoost_mae) / results['mae_test_real'] * 100):.2f}%"
    )
    print(f"{'='*60}\n")

    # Adding the XGBoost MAE to the results
    results["mae_test_xgboost"] = float(XGBoost_mae)
    results["n_features_constraint"] = n_features_to_select

    # Cleaning up the results for the JSON backup
    results_to_save = {
        k: (
            v.tolist()
            if isinstance(v, np.ndarray)
            else float(v) if isinstance(v, (np.floating, np.integer)) else v
        )
        for k, v in results.items()
        if k not in ["X_train", "X_test", "y_train", "y_test", "scaler_target"]
    }

    print(XGBoost_mae)

    # File name based on the number of features
    if n_features_to_select is None:
        filename = f"lasso_{ref_site}_full.json"
    else:
        filename = f"lasso_{ref_site}_{n_features_to_select}features.json"

    file_name = os.path.join("/data/outputs/results", filename)

    with open(file_name, "w") as f:
        json.dump(results_to_save, f, indent=2)

    print(f"Final results saved in {filename}")

    conn.close()


if __name__ == "__main__":
    main()
