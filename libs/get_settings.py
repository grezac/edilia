"""
check_settings.py
-----------------
Validation module for JSON configuration files used in the prediction project.

This module checks the structure and consistency of configuration parameters,
including time periods and geographic ring
configurations.

"""

import os
import sys
import json
import sqlite3
from datetime import datetime


# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================

# Required keys in any configuration
REQUIRED_KEYS = [
    "data_source",
    "window_size",
    "horizon",
    "seed",
    "test_period",
    "working_period",
    "train_split",
    "distance_metric",
    "origin",
    "rings",
]

# Required keys for a site
REQUIRED_SITE_KEYS = ["id", "name", "label", "latitude", "longitude"]

# Required keys for each ring
REQUIRED_SET_KEYS = ["radius_min", "radius_max", "max_predictors", "patience_stop"]

# Allowed values
VALID_DISTANCE_METRICS = ["BallTree"]


# =============================================================================
# UTILITY VALIDATION FUNCTIONS
# =============================================================================


def validate_int_range(value, min_val, max_val, name):
    """
    Validate that a value is an integer within a given range.

    Args:
        value: The value to validate
        min_val: Lower bound (inclusive)
        max_val: Upper bound (inclusive)
        name: Parameter name (for error message)

    Returns:
        str or None: Error message if invalid, None otherwise
    """
    if not isinstance(value, int) or value < min_val or value > max_val:
        return f"'{name}' must be an integer between {min_val} and {max_val}."
    return None


def validate_float_range(value, min_val, max_val, name):
    """
    Validate that a value is a float within a given range.

    Args:
        value: The value to validate
        min_val: Lower bound (inclusive)
        max_val: Upper bound (inclusive)
        name: Parameter name (for error message)

    Returns:
        str or None: Error message if invalid, None otherwise
    """
    if not isinstance(value, float) or value < min_val or value > max_val:
        return f"'{name}' must be a float between {min_val} and {max_val}."
    return None


def validate_non_empty_string(value, name):
    """
    Validate that a value is a non-empty string.

    Args:
        value: The value to validate
        name: Parameter name (for error message)

    Returns:
        str or None: Error message if invalid, None otherwise
    """
    if not isinstance(value, str) or not value:
        return f"'{name}' must be a non-empty string."
    return None


def validate_in_list(value, valid_values, name):
    """
    Validate that a value is among a list of allowed values.

    Args:
        value: The value to validate
        valid_values: List of allowed values
        name: Parameter name (for error message)

    Returns:
        str or None: Error message if invalid, None otherwise
    """
    if value not in valid_values:
        return f"'{name}' must be one of {valid_values}."
    return None


# =============================================================================
# SPECIALIZED VALIDATION FUNCTIONS
# =============================================================================


def validate_database(data_source):
    """
    Validate the existence and accessibility of a SQLite database.

    Args:
        data_source: Path to the database file

    Returns:
        list: List of errors encountered (empty if OK)
    """
    errors = []

    # Check path format
    if not isinstance(data_source, str) or not data_source.endswith(".db"):
        errors.append("'data_source' must be a path to a SQLite database file (.db).")
        return errors

    # Check file existence
    if not os.path.exists(data_source):
        errors.append(f"Database file '{data_source}' does not exist.")
        return errors

    # Test database connection
    try:
        conn = sqlite3.connect(data_source)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
        cursor.fetchone()
        conn.close()
    except sqlite3.Error as e:
        errors.append(f"Cannot connect to database '{data_source}': {str(e)}")

    return errors


def validate_site(site):
    """
    Validate a geographic site configuration.

    Args:
        site: Dictionary containing site information

    Returns:
        list: List of errors encountered (empty if OK)
    """
    errors = []

    # Check required keys
    for key in REQUIRED_SITE_KEYS:
        if key not in site:
            errors.append(f"SITE: Required parameter '{key}' is missing.")

    # If keys are missing, stop validation here
    if errors:
        return errors

    # Validate each field
    if not isinstance(site["id"], int):
        errors.append("SITE: 'id' must be an integer.")

    error = validate_non_empty_string(site["name"], "SITE: name")
    if error:
        errors.append(error)

    error = validate_non_empty_string(site["label"], "SITE: label")
    if error:
        errors.append(error)

    if not isinstance(site["latitude"], float) or not (
        -90.0 <= site["latitude"] <= 90.0
    ):
        errors.append("SITE: 'latitude' must be a float between -90 and 90.")

    if not isinstance(site["longitude"], float) or not (
        -180.0 <= site["longitude"] <= 180.0
    ):
        errors.append("SITE: 'longitude' must be a float between -180 and 180.")

    return errors


def validate_periods(settings):
    """
    Validate test and working periods, and their consistency with horizon.

    Periods must be lists of two dates in YYYY-MM-DD format.
    The gap between working_period end and test_period start must
    match exactly the 'horizon' value.

    Args:
        settings: Configuration dictionary containing test_period and working_period

    Returns:
        list: List of errors encountered (empty if OK)
    """
    errors = []
    test_period = settings["test_period"]
    working_period = settings["working_period"]

    # Check format (list of 2 elements)
    if not isinstance(test_period, list) or len(test_period) != 2:
        errors.append("'test_period' must be a list with two date strings.")
        return errors

    if not isinstance(working_period, list) or len(working_period) != 2:
        errors.append("'working_period' must be a list with two date strings.")
        return errors

    # Parse and validate dates
    try:
        test_start = datetime.strptime(test_period[0], "%Y-%m-%d")
        test_end = datetime.strptime(test_period[1], "%Y-%m-%d")
        working_start = datetime.strptime(working_period[0], "%Y-%m-%d")
        working_end = datetime.strptime(working_period[1], "%Y-%m-%d")

        # Check date consistency
        if test_end <= test_start:
            errors.append("'test_period': end date must be after start date.")

        if working_end <= working_start:
            errors.append("'working_period': end date must be after start date.")

        # Check gap between periods (must match horizon)
        expected_gap = settings.get("horizon", 1)
        actual_gap = (test_start - working_end).days

        if actual_gap != expected_gap:
            errors.append(
                f"Gap between 'working_period' end and 'test_period' start "
                f"must be {expected_gap} days (found {actual_gap} days)."
            )

    except ValueError as e:
        errors.append(f"Invalid date format in periods: {e}")

    return errors


def validate_ring(_ring, index, previous_radius_max):
    """
    Validate an individual geographic ring configuration.

    Args:
        _ring: Dictionary containing ring configuration
        index: ring index in the list (for error messages)
        previous_radius_max: Max radius of previous ring (to check continuity)

    Returns:
        list: List of errors encountered (empty if OK)
    """
    errors = []
    prefix = f"ring {index}"

    # Check required keys
    for key in REQUIRED_SET_KEYS:
        if key not in _ring:
            errors.append(f"{prefix}: Required parameter '{key}' is missing.")

    # If keys are missing, stop validation here
    if errors:
        return errors

    radius_min = _ring["radius_min"]
    radius_max = _ring["radius_max"]

    # Validate radii
    if not isinstance(radius_min, (int, float)) or not isinstance(
        radius_max, (int, float)
    ):
        errors.append(f"{prefix}: 'radius_min' and 'radius_max' must be numbers.")
    else:
        if radius_min < 0 or radius_max < 0:
            errors.append(
                f"{prefix}: 'radius_min' and 'radius_max' must be non-negative."
            )

        if radius_min >= radius_max:
            errors.append(f"{prefix}: 'radius_min' must be less than 'radius_max'.")

        # Check continuity with previous ring
        # Rem: This test should be eliminated! We could have two sets that differ not
        # in terms of geographic coverage, but, for example, in terms of the origin of the predictors.
        if index > 0 and radius_min < previous_radius_max:
            errors.append(
                f"{prefix}: 'radius_min' ({radius_min}) must be >= "
                f"previous ring's 'radius_max' ({previous_radius_max})."
            )

    # Validate max_predictors
    error = validate_int_range(
        _ring["max_predictors"], 1, 500, f"{prefix}: max_predictors"
    )
    if error:
        errors.append(error)

    # Validate patience_stop
    error = validate_int_range(
        _ring["patience_stop"], 1, 1000, f"{prefix}: patience_stop"
    )
    if error:
        errors.append(error)

    return errors


def validate_rings(rings):
    """
    Validate the complete list of geographic rings.

    Args:
        rings: List of ring configurations

    Returns:
        list: List of errors encountered (empty if OK)
    """
    errors = []

    # Check list format
    if not isinstance(rings, list) or len(rings) == 0 or len(rings) > 10:
        errors.append("'rings' must be a list containing between 1 and 10 elements.")
        return errors

    # Validate each ring
    previous_radius_max = 0
    for i, _ring in enumerate(rings):
        ring_errors = validate_ring(_ring, i, previous_radius_max)
        errors.extend(ring_errors)

        # Update max radius for next ring
        if "radius_max" in _ring and isinstance(_ring["radius_max"], (int, float)):
            previous_radius_max = _ring["radius_max"]

    return errors


def validate_anchor_vars(anchor_vars):
    """
    Validate the list of anchor variables.

    Args:
        anchor_vars: List of anchor variable names

    Returns:
        list: List of errors encountered (empty if OK)
    """
    errors = []

    if not isinstance(anchor_vars, list):
        errors.append("'anchor_vars' must be a list.")
        return errors

    # Check that all elements are strings
    if not all(isinstance(var, str) for var in anchor_vars):
        errors.append("'anchor_vars' must only contain strings.")

    return errors


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def read_settings(site_name, config_name):
    """
    Load and validate a JSON configuration file.

    This function checks the presence and validity of all required parameters,
    validates business constraints (periods, rings, etc.), and applies default
    values for optional parameters.

    Args:
        settings_file_name: Path to the JSON configuration file
        settings_section: Name of the configuration section to load (in CONFIGS)
        site_param: Name of the site to load (optional, in SITES)

    Returns:
        tuple: (True, settings) if configuration is valid

    Raises:
        ValueError: If configuration contains errors
        SystemExit: If JSON file is not found or invalid
    """
    errors = []

    # -------------------------------------------------------------------------
    # Load JSON file
    # -------------------------------------------------------------------------
    try:
        with open("./configs/settings.json", "r", encoding="utf-8") as json_file:
            json_content = json.load(json_file)
    except FileNotFoundError:
        print(f"Error: The file 'settings.json' cannot be found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: The file 'settings.json' is not a valid JSON.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Check basic structure
    # -------------------------------------------------------------------------
    if "CONFIGS" not in json_content:
        print("Required key 'CONFIGS' is missing in json file.")
        sys.exit(1)

    if config_name not in json_content["CONFIGS"]:
        print(f"The config section '{config_name}' is missing in json file.")
        sys.exit(1)

    config = json_content["CONFIGS"][config_name]

    # -------------------------------------------------------------------------
    # Validate site (if specified)
    # -------------------------------------------------------------------------

    if "SITES" not in json_content:
        print("Required key 'SITES' is missing in json file.")
        sys.exit(1)

    if site_name not in json_content["SITES"]:
        print(f"The site '{site_name}' is missing in the 'SITES' section.")
        sys.exit(1)

    site = json_content["SITES"][site_name]
    site_errors = validate_site(site)

    if site_errors:
        raise ValueError("Invalid site configuration:\n" + "\n".join(site_errors))

    # Add site to settings
    config["ref_site"] = dict(site)

    # -------------------------------------------------------------------------
    # Check required keys
    # -------------------------------------------------------------------------
    missing_keys = [key for key in REQUIRED_KEYS if key not in config]
    if missing_keys:
        for key in missing_keys:
            errors.append(f"Required parameter '{key}' is missing.")
        raise ValueError(
            "Invalid Configuration: Missing required parameters.\n" + "\n".join(errors)
        )

    # -------------------------------------------------------------------------
    # Validate data source
    # -------------------------------------------------------------------------
    errors.extend(validate_database(config["data_source"]))

    # -------------------------------------------------------------------------
    # Validate ref_site (if present but not via site_param)
    # -------------------------------------------------------------------------
    if "ref_site" in config:
        ref_site = config["ref_site"]
        if ref_site is None:
            errors.append("'ref_site' cannot be None.")
        elif not isinstance(ref_site, dict):
            errors.append("'ref_site' must be a dictionary.")

    # -------------------------------------------------------------------------
    # Validate simple numeric parameters
    # -------------------------------------------------------------------------

    # window_size: integer between 1 and 10
    error = validate_int_range(config["window_size"], 1, 10, "window_size")
    if error:
        errors.append(error)

    # horizon: integer between 1 and 10
    error = validate_int_range(config["horizon"], 1, 10, "horizon")
    if error:
        errors.append(error)

    # seed: integer >= 0
    if not isinstance(config["seed"], int) or config["seed"] < 0:
        errors.append("'seed' must be a non-negative integer.")

    # train_split: float between 0.6 and 0.95
    error = validate_float_range(config["train_split"], 0.6, 0.95, "train_split")
    if error:
        errors.append(error)

    # -------------------------------------------------------------------------
    # Validate constrained value parameters
    # -------------------------------------------------------------------------

    # distance_metric
    error = validate_in_list(
        config["distance_metric"], VALID_DISTANCE_METRICS, "distance_metric"
    )
    if error:
        errors.append(error)

    # origin
    error = validate_non_empty_string(config["origin"], "origin")
    if error:
        errors.append(error)

    # excluded_vars (optional)
    if "excluded_vars" in config:
        if not isinstance(config["excluded_vars"], list) or not all(
            isinstance(var, str) for var in config["excluded_vars"]
        ):
            errors.append("'excluded_vars' must be a list of strings.")
    else:
        config["excluded_vars"] = []

    # -------------------------------------------------------------------------
    # Validate time periods
    # -------------------------------------------------------------------------
    errors.extend(validate_periods(config))

    # -------------------------------------------------------------------------
    # Validate rings
    # -------------------------------------------------------------------------
    errors.extend(validate_rings(config["rings"]))

    # -------------------------------------------------------------------------
    # Validate and set defaults for optional parameters
    # -------------------------------------------------------------------------

    if "name_extension" in config:
        if not isinstance(config["name_extension"], str):
            errors.append("'name_extension' must be a string")
    else:
        config["name_extension"] = ""

    # ref_as_feature (default: True)
    if "ref_as_feature" in config:
        if not isinstance(config["ref_as_feature"], bool):
            errors.append("'ref_as_feature' must be a boolean (True or False).")
    else:
        config["ref_as_feature"] = True

    # dependent_rings (default: True)
    if "dependent_sets" in config:
        if not isinstance(config["dependent_rings"], bool):
            errors.append("'dependent_rings' must be a boolean.")
    else:
        config["dependent_rings"] = True

    # anchor_vars (optional)
    if "anchor_vars" in config:
        errors.extend(validate_anchor_vars(config["anchor_vars"]))

    # -------------------------------------------------------------------------
    # Final error report
    # -------------------------------------------------------------------------
    if errors:
        raise ValueError("Invalid Configuration:\n" + "\n".join(errors))

    settings = {"SITES": {site_name: site}, "CONFIGS": {config_name: config}}

    return settings
