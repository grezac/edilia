"""
Model Evaluation Statistics Aggregator.

This module provides utilities to aggregate and summarize results from multiple
JSON experiment files. It is designed to process performance metrics for various
models (Linear, LightGBM, XGBoost) and calculate statistical averages and
standard deviations.

Key features:
- Recursive averaging of nested dictionaries/statistical blocks.
- Smart rounding based on metric types (MAE, MSE, R2, etc.).
- Time parsing and formatting (converting 'm s' strings to average durations).
- Automatic standard deviation calculation for key performance indicators.

Input: A list of JSON file paths and an output path.
Output: A single summary JSON file containing the aggregated statistics.
"""

import os
import json
import math


def stats_results(files, results_file_path):

    stats = {}
    num = 0
    data_list = []

    for file_path in files:
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            return None
        with open(file_path, "r") as file:
            data = json.load(file)
        data_list.append(data)
        num += 1

    if num == 0:
        return None

    # Keys for which we want a std deviation entry right after the mean
    STD_KEYS = {"mae_val", "mae_test", "mse_val", "mse_test", "r2_val", "r2_test"}

    # --- Helper: parse elapsed time string to total seconds ---
    def parse_time(s):
        parts = s.strip().split()
        minutes = 0
        seconds = 0
        for i, p in enumerate(parts):
            if p == "m":
                minutes = int(parts[i - 1])
            elif p == "s":
                seconds = int(parts[i - 1])
        return minutes * 60 + seconds

    def format_time(total_seconds):
        total_seconds = round(total_seconds)
        m = total_seconds // 60
        s = total_seconds % 60
        return f"{m:02d} m {s:02d} s"

    def get_decimals(key):
        key_lower = key.lower()
        if key_lower in ("n_events",):
            return 2
        if any(
            k in key_lower
            for k in ("mae", "mse", "bias", "max_error", "extreme_threshold")
        ):
            return 3
        if any(k in key_lower for k in ("pod", "far", "hss", "pss")):
            return 4
        if any(k in key_lower for k in ("r2",)):
            return 4
        if any(k in key_lower for k in ("p50", "p90", "p95")):
            return 4
        if any(k in key_lower for k in ("rate",)):
            return 4
        return 4

    def std_dev(values):
        n = len(values)
        if n < 2:
            return 0.0
        avg = sum(values) / n
        variance = sum((x - avg) ** 2 for x in values) / (n - 1)
        return math.sqrt(variance)

    def average_recursive(items, current_key=""):
        """items is a list of values (all same type) to average."""
        if isinstance(items[0], dict):
            result = {}
            for k in items[0]:
                sub_items = [item[k] for item in items]
                result[k] = average_recursive(sub_items, current_key=k)
                # Add std right after the mean for targeted keys
                if k in STD_KEYS:
                    decimals = get_decimals(k)
                    result[f"std_{k}"] = round(std_dev(sub_items), decimals)
            return result
        elif isinstance(items[0], (int, float)):
            avg = sum(items) / len(items)
            decimals = get_decimals(current_key)
            return round(avg, decimals)
        elif isinstance(items[0], str):
            return items
        else:
            return items

    # Build stats with same structure
    avg_seconds = sum(parse_time(d["total_elapsed_time"]) for d in data_list) / num
    stats["total_elapsed_time"] = format_time(avg_seconds)

    stats["generations"] = round(sum(d["generations"] for d in data_list) / num, 2)

    for key in ("linear_stats", "lightGBM_stats", "XGBoost_stats"):
        blocks = [d[key] for d in data_list]
        stats[key] = average_recursive(blocks, current_key=key)

    with open(results_file_path, "w") as f:
        json.dump(stats, f, indent=4)

    return stats
