#!/usr/bin/env python3
"""
NetCDF Consolidation and SQLite Import Script
Consolidates annual NetCDF files and imports data into SQLite database
"""

import os
import sys
import sqlite3
import pickle
import json
import time
from pathlib import Path
from datetime import datetime, date
import xarray as xr
import numpy as np
import csv

# =============================================================================
# CONFIGURATION
# =============================================================================


VARS_TO_IMPORT = [
    {
        "var_name": "2m_temperature_daily_minimum",
        "var_description": "minimum daily temperature at 2 meters",
        "unit": "K",
        "netcdf_var_name": "t2m",
    },
    {
        "var_name": "2m_temperature_daily_maximum",
        "var_description": "maximum daily temperature at 2 meters",
        "unit": "K",
        "netcdf_var_name": "t2m",
    },
    {
        "var_name": "10m_u_component_of_wind_daily_mean",
        "var_description": "daily mean zonal wind component at 10 meters",
        "unit": "m s**-1",
        "netcdf_var_name": "u10",
    },
    {
        "var_name": "10m_v_component_of_wind_daily_mean",
        "var_description": "daily mean meridional wind component at 10 meters",
        "unit": "m s**-1",
        "netcdf_var_name": "v10",
    },
    {
        "var_name": "10m_wind_gust_since_previous_post_processing_daily_maximum",
        "var_description": "daily maximum wind gust at 10 meters",
        "unit": "m s**-1",
        "netcdf_var_name": "fg10",
    },
    {
        "var_name": "2m_dewpoint_temperature_daily_minimum",
        "var_description": "minimum daily dewpoint temperature at 2 meters",
        "unit": "K",
        "netcdf_var_name": "d2m",
    },
    {
        "var_name": "boundary_layer_height_daily_maximum",
        "var_description": "daily maximum planetary boundary layer height",
        "unit": "m",
        "netcdf_var_name": "blh",
    },
    {
        "var_name": "evaporation_daily_sum",
        "var_description": "daily accumulated surface evaporation",
        "unit": "m of water equivalent",
        "netcdf_var_name": "e",
    },
    {
        "var_name": "low_cloud_cover_daily_mean",
        "var_description": "daily mean low cloud fraction",
        "unit": "(0 - 1)",
        "netcdf_var_name": "lcc",
    },
    {
        "var_name": "medium_cloud_cover_daily_mean",
        "var_description": "daily mean medium cloud fraction",
        "unit": "(0 - 1)",
        "netcdf_var_name": "mcc",
    },
    {
        "var_name": "sea_surface_temperature_daily_mean",
        "var_description": "daily mean sea surface temperature",
        "unit": "K",
        "netcdf_var_name": "sst",
    },
    {
        "var_name": "skin_temperature_daily_minimum",
        "var_description": "minimum daily land or sea skin temperature",
        "unit": "K",
        "netcdf_var_name": "skt",
    },
    {
        "var_name": "snow_depth_daily_mean",
        "var_description": "daily mean snow depth",
        "unit": "m of water equivalent",
        "netcdf_var_name": "sd",
    },
    {
        "var_name": "soil_temperature_level_1_daily_minimum",
        "var_description": "minimum daily soil temperature at level 1",
        "unit": "K",
        "netcdf_var_name": "stl1",
    },
    {
        "var_name": "surface_net_thermal_radiation_daily_sum",
        "var_description": "daily accumulated net thermal radiation at surface",
        "unit": "J m**-2",
        "netcdf_var_name": "str",
    },
    {
        "var_name": "surface_sensible_heat_flux_daily_sum",
        "var_description": "daily accumulated surface sensible heat flux",
        "unit": "J m**-2",
        "netcdf_var_name": "sshf",
    },
    {
        "var_name": "surface_solar_radiation_downwards_daily_sum",
        "var_description": "daily accumulated downward solar radiation at surface",
        "unit": "J m**-2",
        "netcdf_var_name": "ssrd",
    },
    {
        "var_name": "total_cloud_cover_daily_mean",
        "var_description": "daily mean total cloud cover",
        "unit": "(0 - 1)",
        "netcdf_var_name": "tcc",
    },
    {
        "var_name": "total_column_water_vapour_daily_mean",
        "var_description": "daily mean total column water vapour",
        "unit": "kg m**-2",
        "netcdf_var_name": "tcwv",
    },
    {
        "var_name": "volumetric_soil_water_layer_1_daily_mean",
        "var_description": "daily mean volumetric soil water content layer 1",
        "unit": "m**3 m**-3",
        "netcdf_var_name": "swvl1",
    },
    {
        "var_name": "boundary_layer_height_daily_minimum",
        "var_description": "daily minimum planetary boundary layer height",
        "unit": "m",
        "netcdf_var_name": "blh",
    },
    {
        "var_name": "high_cloud_cover_daily_mean",
        "var_description": "daily mean high cloud fraction",
        "unit": "(0 - 1)",
        "netcdf_var_name": "hcc",
    },
    {
        "var_name": "mean_sea_level_pressure_daily_mean",
        "var_description": "daily mean atmospheric pressure at mean sea level",
        "unit": "Pa",
        "netcdf_var_name": "msl",
    },
    {
        "var_name": "soil_temperature_level_2_daily_minimum",
        "var_description": "daily minimum soil temperature at level 2 (7-28 cm depth)",
        "unit": "K",
        "netcdf_var_name": "stl2",
    },
    {
        "var_name": "surface_latent_heat_flux_daily_sum",
        "var_description": "daily accumulated surface latent heat flux",
        "unit": "J m**-2",
        "netcdf_var_name": "slhf",
    },
    {
        "var_name": "surface_thermal_radiation_downwards_daily_mean",
        "var_description": "daily mean downward thermal radiation at surface",
        "unit": "J m**-2",
        "netcdf_var_name": "strd",
    },
    {
        "var_name": "temperature_850hPa_daily_mean",
        "var_description": "daily mean air temperature at 850 hPa pressure level",
        "unit": "K",
        "netcdf_var_name": "t",
    },
    {
        "var_name": "total_precipitation_daily_sum",
        "var_description": "daily accumulated total precipitation",
        "unit": "m",
        "netcdf_var_name": "tp",
    },
]

SITES = [
    {
        "name": "Strasbourg",
        "filename": "FR000007190_noNaN.csv",
        "interpolations": 0,
    },
    {
        "name": "Lyon",
        "filename": "FR069029001_noNaN.csv",
        "interpolations": 0,
    },
    {
        "name": "Nice",
        "filename": "FRE00104120_noNaN.csv",
        "interpolations": 0,
    },
    {
        "name": "Brest",
        "filename": "FRE00104484_noNaN.csv",
        "interpolations": 0,
    },
    {
        "name": "Paris",
        "filename": "FRM00007149_noNaN.csv",
        "interpolations": 0,
    },
    {
        "name": "Birmingham",
        "filename": "UK000000000_noNaN.csv",
        "interpolations": 1,  # 2013-06-30 (replaced -999 with 116)
    },
    {
        "name": "Plymouth",
        "filename": "UKE00105870_noNaN.csv",
        "interpolations": 8,
    },
    {
        "name": "Edinburgh",
        "filename": "UKE00105888_noNaN.csv",
        "interpolations": 10,
    },
]

FEATURES_DIR = "FEATURES"
REFERENCES_DIR = "REFERENCES"

# Years to process
YEARS = range(2004, 2025)  # 2004 to 2024 inclusive
NUM_FILES = 21  # to keep

# Database configuration
DB_NAME = "METEO_daily.db"
START_DATE = "2004-01-01"
END_DATE = "2024-12-31"
TOTAL_VALUES = 7671

# =============================================================================
# DATABASE DDL
# =============================================================================

DDL_SITES = """
CREATE TABLE IF NOT EXISTS sites (
    site_id INTEGER PRIMARY KEY AUTOINCREMENT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    UNIQUE(latitude, longitude)
);
"""

DDL_FEATURE_SERIES = """
CREATE TABLE IF NOT EXISTS feature_series (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    site_id INTEGER NOT NULL,
    var_name TEXT NOT NULL,
    var_description TEXT,
    unit TEXT,
    short_var_name TEXT,
    series BLOB NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    total_values INTEGER NOT NULL,
    original_nan_count INTEGER NOT NULL,
    nan_percentage REAL NOT NULL,
    max_consecutive_nans INTEGER NOT NULL,
    nans_interpolated INTEGER NOT NULL,
    interpolation_method TEXT,
    data_quality TEXT NOT NULL,
    quality_score REAL NOT NULL,
    import_timestamp TEXT NOT NULL,
    origin TEXT,
    FOREIGN KEY (site_id) REFERENCES sites (site_id),
    UNIQUE (site_id, var_name)
);
"""

DDL_IMPORT_METADATA = """
CREATE TABLE IF NOT EXISTS import_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    import_date TEXT NOT NULL,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    max_nan_threshold REAL NOT NULL,
    total_files_processed INTEGER NOT NULL,
    total_series_imported INTEGER NOT NULL,
    processing_time_seconds REAL NOT NULL
);
"""

DDL_INDEX_SITES = """
CREATE INDEX IF NOT EXISTS idx_sites_coords ON sites(latitude, longitude);
"""

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def print_progress(message, level="INFO"):
    """Print a formatted progress message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # symbols = {"INFO": "ℹ", "SUCCESS": "✓", "ERROR": "✗", "WARNING": "⚠"}
    # symbol = symbols.get(level, "•")
    # print(f"[{timestamp}] {symbol} {message}")
    print(f"[{timestamp}] {message}")


def validate_files_existence(var_name, base_path):
    """
    Verify the existence of all required annual files.

    Args:
        var_name: Variable name
        base_path: Folder path containing the files

    Returns:
        List of file paths sorted by year

    Raises:
        FileNotFoundError: If a file is missing
    """
    print_progress(f"Validating files for '{var_name}'...")

    file_paths = []
    missing_files = []

    for year in YEARS:
        filename = f"{var_name}_{year}.nc"
        filepath = base_path / filename

        if filepath.exists():
            file_paths.append(filepath)
            print_progress(f"  {filename} found", level="INFO")
        else:
            missing_files.append(filename)
            print_progress(f"  {filename} missing", level="ERROR")

    if missing_files:
        raise FileNotFoundError(
            f"Missing files for '{var_name}':\n"
            + "\n".join(f"  - {f}" for f in missing_files)
        )

    if len(file_paths) != NUM_FILES:
        raise ValueError(
            f"Incorrect number of files: {len(file_paths)} found, "
            f"{NUM_FILES} expected"
        )

    print_progress(f"All {NUM_FILES} files are present", level="SUCCESS")
    return file_paths


def validate_spatial_grid(datasets, var_name):
    """
    Verify spatial grid consistency across all files.

    Args:
        datasets: List of xarray datasets
        var_name: Variable name (for messages)

    Raises:
        ValueError: If grids are not consistent
    """
    print_progress("Validating spatial grid consistency...")

    ref_ds = datasets[0]
    ref_lat = ref_ds["latitude"].values
    ref_lon = ref_ds["longitude"].values

    for i, ds in enumerate(datasets[1:], start=1):
        lat = ds["latitude"].values
        lon = ds["longitude"].values

        if not np.allclose(lat, ref_lat, rtol=1e-9):
            raise ValueError(
                f"Inconsistent latitude grid in file {i+1}\n"
                f"Max difference: {np.max(np.abs(lat - ref_lat))}"
            )

        if not np.allclose(lon, ref_lon, rtol=1e-9):
            raise ValueError(
                f"Inconsistent longitude grid in file {i+1}\n"
                f"Max difference: {np.max(np.abs(lon - ref_lon))}"
            )

    print_progress(
        f"Consistent spatial grid: {len(ref_lat)} latitudes × "
        f"{len(ref_lon)} longitudes",
        level="SUCCESS",
    )


def load_datasets(file_paths, netcdf_var_name):
    """
    Load all NetCDF files.

    Args:
        file_paths: List of file paths
        netcdf_var_name: Variable name in NetCDF files

    Returns:
        List of xarray datasets
    """
    print_progress(f"Loading {len(file_paths)} NetCDF files...")

    datasets = []
    for i, filepath in enumerate(file_paths, start=1):
        print_progress(f"  Loading {i}/{len(file_paths)}: {filepath.name}")

        try:
            ds = xr.open_dataset(filepath)

            # Check that main variable exists
            if netcdf_var_name not in ds:
                raise ValueError(
                    f"Variable '{netcdf_var_name}' not found in {filepath.name}\n"
                    f"Available variables: {list(ds.data_vars)}"
                )

            datasets.append(ds)

        except Exception as e:
            raise RuntimeError(f"Error loading {filepath.name}: {str(e)}")

    print_progress(f"{len(datasets)} files loaded successfully", level="SUCCESS")
    return datasets


def consolidate_datasets(datasets, netcdf_var_name):
    """
    Consolidate datasets into a single dataset.

    Args:
        datasets: List of xarray datasets
        netcdf_var_name: Variable name in NetCDF files

    Returns:
        Consolidated dataset
    """

    print_progress("Consolidating temporal data...")

    # Concatenate along time dimension
    consolidated = xr.concat(datasets, dim="valid_time")

    # Recalculate valid_time for continuous sequence
    start_date = np.datetime64("2004-01-01")
    num_days = len(consolidated["valid_time"])

    new_valid_time = np.arange(num_days, dtype="int64")
    consolidated["valid_time"] = new_valid_time

    # Update valid_time attributes
    consolidated["valid_time"].attrs = {
        "units": "days since 2004-01-01 00:00:00",
        "calendar": "proleptic_gregorian",
        "long_name": "valid time",
        "standard_name": "time",
    }

    print_progress(
        f"Consolidation complete: {num_days} days " f"(2004-01-01 to 2024-12-31)",
        level="SUCCESS",
    )

    return consolidated


def save_consolidated_file(dataset, output_path, var_info):
    """
    Save consolidated dataset to NetCDF file.

    Args:
        dataset: Consolidated xarray dataset
        output_path: Output file path
        var_info: Dictionary containing variable info
    """
    print_progress(f"Saving consolidated file: {output_path.name}")

    # Add global metadata
    dataset.attrs.update(
        {
            "title": f"Consolidated {var_info['var_description']}",
            "description": f"{var_info['var_description']} - Consolidated data from 2004 to 2024",
            "temporal_coverage": "2004-01-01 to 2024-12-31",
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "created_by": "NetCDF Consolidation Script",
            "institution": "European Centre for Medium-Range Weather Forecasts",
            "Conventions": "CF-1.7",
        }
    )

    # Encoding for file size optimization
    encoding = {
        var_info["netcdf_var_name"]: {"zlib": True, "complevel": 4, "dtype": "float32"}
    }

    # Save
    dataset.to_netcdf(output_path, encoding=encoding, format="NETCDF4")

    # Display file statistics
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print_progress(f"File saved successfully ({file_size_mb:.2f} MB)", level="SUCCESS")


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================


def init_database(db_path):
    """
    Initialize SQLite database with required tables.

    Args:
        db_path: Path to database file

    Returns:
        Database connection
    """
    print_progress(f"Initializing database: {db_path.name}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute(DDL_SITES)
    cursor.execute(DDL_FEATURE_SERIES)
    cursor.execute(DDL_IMPORT_METADATA)
    cursor.execute(DDL_INDEX_SITES)

    conn.commit()

    print_progress("Database initialized successfully", level="SUCCESS")
    return conn


def get_or_create_site(cursor, latitude, longitude):
    """
    Get existing site_id or create new site.

    Args:
        cursor: Database cursor
        latitude: Site latitude (rounded to 4 decimals)
        longitude: Site longitude (rounded to 4 decimals)

    Returns:
        site_id
    """
    # Round to 4 decimals
    lat = round(latitude, 4)
    lon = round(longitude, 4)

    # Try to get existing site
    cursor.execute(
        "SELECT site_id FROM sites WHERE latitude = ? AND longitude = ?", (lat, lon)
    )
    result = cursor.fetchone()

    if result:
        return result[0]

    # Create new site
    cursor.execute("INSERT INTO sites (latitude, longitude) VALUES (?, ?)", (lat, lon))
    return cursor.lastrowid


def import_sites(conn, dataset):
    """
    Import all sites from dataset grid.

    Args:
        conn: Database connection
        dataset: xarray dataset with latitude/longitude

    Returns:
        Dictionary mapping (lat, lon) to site_id
    """
    print_progress("Importing sites into database...")

    cursor = conn.cursor()
    latitudes = dataset["latitude"].values
    longitudes = dataset["longitude"].values

    site_map = {}
    total_sites = len(latitudes) * len(longitudes)
    new_sites = 0

    cursor.execute("BEGIN TRANSACTION")

    for i, lat in enumerate(latitudes):
        for j, lon in enumerate(longitudes):
            lat_key = round(lat, 4)
            lon_key = round(lon, 4)

            # Check if site already exists
            cursor.execute(
                "SELECT site_id FROM sites WHERE latitude = ? AND longitude = ?",
                (lat_key, lon_key),
            )
            result = cursor.fetchone()

            if result:
                site_id = result[0]
            else:
                cursor.execute(
                    "INSERT INTO sites (latitude, longitude) VALUES (?, ?)",
                    (lat_key, lon_key),
                )
                site_id = cursor.lastrowid
                new_sites += 1

            site_map[(i, j)] = site_id

    conn.commit()

    print_progress(
        f"Sites processed: {total_sites} total, {new_sites} new", level="SUCCESS"
    )

    return site_map


def import_feature_series(conn, dataset, var_info, site_map):
    """
    Import feature series for all sites.
    """
    print_progress(f"Importing feature series for '{var_info['var_name']}'...")

    cursor = conn.cursor()
    netcdf_var_name = var_info["netcdf_var_name"]

    data_var = dataset[netcdf_var_name]

    # Gérer la dimension pressure_level si présente
    if "pressure_level" in data_var.dims:
        data_var = data_var.isel(pressure_level=0)
        print_progress(f"  Removed pressure_level dimension")

    # Identifier la dimension temporelle (time ou valid_time)
    time_dim = "time" if "time" in data_var.dims else "valid_time"

    # Transposer pour avoir (time, latitude, longitude)
    data_var = data_var.transpose(time_dim, "latitude", "longitude")

    print_progress(f"  Final shape: {data_var.shape}")

    var_data = data_var.values

    # Suite du code inchangée...
    import_timestamp = datetime.now().isoformat()
    num_imported = 0
    batch_size = 1000
    batch = []

    cursor.execute("BEGIN TRANSACTION")

    total_sites = len(site_map)

    for (lat_idx, lon_idx), site_id in site_map.items():
        series = var_data[:, lat_idx, lon_idx].astype(np.float32)

        nan_mask = np.isnan(series)
        contains_nan = nan_mask.any()
        if not np.all(series == series[0]) and not contains_nan:

            series_blob = pickle.dumps(series, protocol=pickle.HIGHEST_PROTOCOL)

            record = (
                site_id,
                var_info["var_name"],
                var_info["var_description"],
                var_info["unit"],
                var_info["netcdf_var_name"],
                series_blob,
                START_DATE,
                END_DATE,
                TOTAL_VALUES,
                0,
                0.0,
                0,
                0,
                "none",
                "perfect",
                100.0,
                import_timestamp,
                "ERA5",
            )

            batch.append(record)

            if len(batch) >= batch_size:
                cursor.executemany(
                    """
                    INSERT OR REPLACE INTO feature_series (
                        site_id, var_name, var_description, unit, short_var_name,
                        series, start_date, end_date, total_values,
                        original_nan_count, nan_percentage, max_consecutive_nans,
                        nans_interpolated, interpolation_method, data_quality,
                        quality_score, import_timestamp, origin
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                num_imported += len(batch)
                print_progress(
                    f"  Progress: {num_imported}/{total_sites} series imported"
                )
                batch = []

    if batch:
        cursor.executemany(
            """
            INSERT OR REPLACE INTO feature_series (
                site_id, var_name, var_description, unit, short_var_name,
                series, start_date, end_date, total_values,
                original_nan_count, nan_percentage, max_consecutive_nans,
                nans_interpolated, interpolation_method, data_quality,
                quality_score, import_timestamp, origin
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            batch,
        )
        num_imported += len(batch)

    conn.commit()

    print_progress(f"Import complete: {num_imported} series imported", level="SUCCESS")

    return num_imported


def import_feature_series_old(conn, dataset, var_info, site_map):
    """
    Import feature series for all sites.

    Args:
        conn: Database connection
        dataset: Consolidated xarray dataset
        var_info: Variable information dictionary
        site_map: Dictionary mapping (lat_idx, lon_idx) to site_id

    Returns:
        Number of series imported
    """
    print_progress(f"Importing feature series for '{var_info['var_name']}'...")

    cursor = conn.cursor()
    netcdf_var_name = var_info["netcdf_var_name"]
    var_data = dataset[netcdf_var_name].values

    import_timestamp = datetime.now().isoformat()
    num_imported = 0
    batch_size = 1000
    batch = []

    cursor.execute("BEGIN TRANSACTION")

    total_sites = len(site_map)

    for (lat_idx, lon_idx), site_id in site_map.items():
        # Extract time series for this site
        series = var_data[:, lat_idx, lon_idx].astype(np.float32)

        # On n'insère que les séries dont la variance n'est pas nulle (cas de snow_depth en mer)
        # et ne contenant pas de nan (cas de sea_surface_temperature)
        nan_mask = np.isnan(series)
        contains_nan = nan_mask.any()
        if not np.all(series == series[0]) and not contains_nan:

            # Serialize with pickle
            series_blob = pickle.dumps(series, protocol=pickle.HIGHEST_PROTOCOL)

            # Prepare record
            record = (
                site_id,
                var_info["var_name"],
                var_info["var_description"],
                var_info["unit"],
                var_info["netcdf_var_name"],
                series_blob,
                START_DATE,
                END_DATE,
                TOTAL_VALUES,
                0,  # original_nan_count
                0.0,  # nan_percentage
                0,  # max_consecutive_nans
                0,  # nans_interpolated
                "none",  # interpolation_method
                "perfect",  # data_quality
                100.0,  # quality_score
                import_timestamp,
                "ERA5",  # origin
            )

            batch.append(record)

            # Insert batch
            if len(batch) >= batch_size:
                cursor.executemany(
                    """
                    INSERT OR REPLACE INTO feature_series (
                        site_id, var_name, var_description, unit, short_var_name,
                        series, start_date, end_date, total_values,
                        original_nan_count, nan_percentage, max_consecutive_nans,
                        nans_interpolated, interpolation_method, data_quality,
                        quality_score, import_timestamp, origin
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                num_imported += len(batch)
                print_progress(
                    f"  Progress: {num_imported}/{total_sites} series imported"
                )
                batch = []

    # Insert remaining records
    if batch:
        cursor.executemany(
            """
            INSERT OR REPLACE INTO feature_series (
                site_id, var_name, var_description, unit, short_var_name,
                series, start_date, end_date, total_values,
                original_nan_count, nan_percentage, max_consecutive_nans,
                nans_interpolated, interpolation_method, data_quality,
                quality_score, import_timestamp, origin
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            batch,
        )
        num_imported += len(batch)

    conn.commit()

    print_progress(f"Import complete: {num_imported} series imported", level="SUCCESS")

    return num_imported


def insert_import_metadata(conn, var_info, num_series, processing_time):
    """
    Insert import metadata record.

    Args:
        conn: Database connection
        var_info: Variable information dictionary
        num_series: Number of series imported
        processing_time: Processing time in seconds
    """
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO import_metadata (
            import_date, period_start, period_end, max_nan_threshold,
            total_files_processed, total_series_imported, processing_time_seconds
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(),
            START_DATE,
            END_DATE,
            0.0,  # max_nan_threshold (perfect data)
            NUM_FILES,
            num_series,
            processing_time,
        ),
    )

    conn.commit()
    print_progress("Import metadata saved", level="SUCCESS")


def import_variable(var_info, script_dir, db_path):
    """
    Import consolidated NetCDF data into SQLite database.

    Args:
        var_info: Variable information dictionary
        script_dir: Script directory
        db_path: Database path

    Returns:
        True if successful, False otherwise
    """
    var_name = var_info["var_name"]

    print_progress("=" * 80)
    print_progress("." * 80)
    print_progress(f"DATABASE IMPORT: {var_name}")
    print_progress("=" * 80)
    start_time = time.time()

    try:
        # Path to consolidated file
        # var_folder = script_dir / var_name
        # var_folder = var_name
        # consolidated_file = var_folder / f"{var_name}_consolidated.nc"

        consolidated_file = script_dir / FEATURES_DIR / f"{var_name}_consolidated.nc"

        # print(f"consolidated_file: {consolidated_file}")

        if not consolidated_file.exists():
            raise FileNotFoundError(f"Consolidated file not found: {consolidated_file}")

        # Open database connection
        conn = init_database(db_path)

        # Load consolidated dataset
        print_progress(f"Loading consolidated file: {consolidated_file.name}")
        dataset = xr.open_dataset(consolidated_file)

        # Import sites
        site_map = import_sites(conn, dataset)

        # Import feature series
        num_imported = import_feature_series(conn, dataset, var_info, site_map)

        # Record metadata
        processing_time = time.time() - start_time
        insert_import_metadata(conn, var_info, num_imported, processing_time)

        # Close resources
        dataset.close()
        conn.close()

        print_progress(
            f"Database import completed in {processing_time:.2f} seconds",
            level="SUCCESS",
        )
        print_progress("")

        return True

    except Exception as e:
        print_progress(
            f"Error during database import for '{var_name}': {str(e)}", level="ERROR"
        )
        if "conn" in locals():
            conn.rollback()
            conn.close()
        return False


# =============================================================================
# CONSOLIDATION FUNCTIONS
# =============================================================================


def process_variable(var_info, script_dir):
    """
    Process a complete variable: validation, consolidation and saving.

    Args:
        var_info: Dictionary containing variable information
        script_dir: Script directory
    """
    var_name = var_info["var_name"]
    netcdf_var_name = var_info["netcdf_var_name"]

    print_progress("=" * 80)
    print_progress(f"PROCESSING VARIABLE: {var_name}")
    print_progress(f"Description: {var_info['var_description']}")
    print_progress(f"Unit: {var_info['unit']}")
    print_progress("=" * 80)

    try:
        # 1. Define paths
        # var_folder = script_dir / var_name
        var_folder = script_dir / FEATURES_DIR

        if not var_folder.exists():
            raise FileNotFoundError(
                f"Folder '{var_name}' does not exist in {script_dir}"
            )

        output_file = var_folder / f"{var_name}_consolidated.nc"

        # 2. Validate file existence
        file_paths = validate_files_existence(var_name, var_folder)

        # 3. Load datasets
        datasets = load_datasets(file_paths, netcdf_var_name)

        # 4. Validate spatial consistency
        validate_spatial_grid(datasets, var_name)

        # 5. Consolidate
        consolidated = consolidate_datasets(datasets, netcdf_var_name)

        # 6. Save
        save_consolidated_file(consolidated, output_file, var_info)

        # 7. Clean up memory
        for ds in datasets:
            ds.close()
        consolidated.close()

        print_progress(
            f"Processing of '{var_name}' completed successfully!", level="SUCCESS"
        )
        print_progress("")

        return True

    except Exception as e:
        print_progress(f"Error processing '{var_name}': {str(e)}", level="ERROR")
        return False


def import_reference_site(cursor, site):
    # Définition des dates limites
    date_debut = date(2004, 1, 1)
    date_fin = date(2024, 12, 31)

    # Listes pour stocker les données
    t_min = []
    latitude = 0.0
    longitude = 0.0
    var_description = ""

    print("Lecture du fichier CSV...")
    script_dir = Path(__file__).parent.resolve()
    csv_path = script_dir / REFERENCES_DIR / site["filename"]
    with open(
        # site["filename"],
        csv_path,
        mode="r",
        encoding="utf-8",
    ) as csv_file:

        lecteur = csv.DictReader(csv_file)

        for ligne in lecteur:
            try:
                date_ligne = datetime.strptime(ligne["DATE"], "%Y-%m-%d").date()
            except ValueError:
                continue

            if date_debut <= date_ligne <= date_fin:
                tmin_str = ligne["TMIN"].strip() if ligne["TMIN"] else ""
                latitude = float(ligne["LATITUDE"])
                longitude = float(ligne["LONGITUDE"])
                var_description = ligne["NAME"]
                t_min.append(round(int(tmin_str) / 10, 2))

        # 1. Insertion du site
        print("Insertion du site...")

        cursor.execute(
            "INSERT OR IGNORE INTO sites (latitude, longitude) VALUES (?, ?)",
            (latitude, longitude),
        )
        # cursor.execute(
        #     "INSERT INTO sites (latitude, longitude) VALUES (?, ?)",
        #     (latitude, longitude),
        # )
        site_id = cursor.lastrowid
        print(f"Site créé avec l'ID: {site_id}")

        # 2. Préparation des données communes
        unit = "°C"
        start_date = "2004-01-01"
        end_date = "2024-12-31"
        original_nan_count = site["interpolations"]
        nan_percentage = 0.0
        max_consecutive_nans = 0
        nans_interpolated = 0
        if original_nan_count == 0:
            interpolation_method = "NA"
            data_quality = "perfect"
            quality_score = 100.0
        else:
            interpolation_method = "linear"
            data_quality = "excellent"
            quality_score = 99.0
        origin = "NOAA"

        # 3. Insertion de la série climatique

        variables = [("t_min_noaa", t_min)]  # Que les tmin
        for var_name, data_list in variables:
            print(f"Insertion de {var_name} avec {len(data_list)} valeurs...")

            # Sérialisation des données
            data_array = np.array(data_list, dtype=np.float32)
            print(f"min: {data_array.min()} ; max: {data_array.max()}")
            serie_blob = pickle.dumps(data_array)
            data_array2 = pickle.loads(serie_blob)
            print(f"min: {data_array2.min()} ; max: {data_array2.max()}")

            import_timestamp = datetime.now().isoformat(timespec="microseconds")

            # Requête d'insertion
            query = """INSERT INTO feature_series (
                site_id, var_name, var_description, unit, series, 
                start_date, end_date, total_values, original_nan_count,
                nan_percentage, max_consecutive_nans, nans_interpolated, 
                interpolation_method, data_quality, quality_score, 
                import_timestamp, origin
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

            # Exécution de la requête
            cursor.execute(
                query,
                (
                    site_id,
                    var_name,
                    var_description,
                    unit,
                    serie_blob,  # ← Données correctement sérialisées
                    start_date,
                    end_date,
                    len(data_list),
                    original_nan_count,
                    nan_percentage,
                    max_consecutive_nans,
                    nans_interpolated,
                    interpolation_method,
                    data_quality,
                    quality_score,
                    import_timestamp,
                    origin,
                ),
            )
            last_id = cursor.lastrowid

            # print(
            #     f"{var_name} inséré avec succès pour le site {site["name"]} ; id de l'enregistrement créé : {last_id}"
            # )

    return last_id


def import_reference_sites():
    script_dir = Path(__file__).parent.resolve()
    db_path = script_dir / DB_NAME
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        site_ids = {}
        for site in SITES:
            site_id = import_reference_site(cursor, site)
            site_ids[site["name"]] = site_id
        conn.commit()

        # Save the site IDs
        with open("site_ids.json", "w", encoding="utf-8") as f:
            json.dump(site_ids, f, ensure_ascii=False, indent=4)

        print("Reference sites have been imported in database")
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        conn.rollback()
    except Exception as e:
        print(f"General error: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()
        print("Connection closed")


def create_var_index(db_path):
    print_progress("Creating variable index...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS var_name_idx ON feature_series (var_name)
        """
    )

    conn.commit()
    cursor.close()
    conn.close()

    print_progress("Variable index created successfully", level="SUCCESS")


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def main():
    # import_reference_sites()
    # sys.exit(0)
    """Main script function."""
    print_progress("=" * 80)
    print_progress("NETCDF CONSOLIDATION AND DATABASE IMPORT")
    print_progress("=" * 80)
    print_progress("")

    # Get script directory
    script_dir = Path(__file__).parent.resolve()
    db_path = script_dir / DB_NAME

    print_progress(f"Working directory: {script_dir}")
    print_progress(f"Database: {db_path}")
    print_progress(f"Variables to process: {len(VARS_TO_IMPORT)}")
    print_progress("")

    # Process each variable
    results = {}
    for i, var_info in enumerate(VARS_TO_IMPORT, start=1):
        print_progress(f"Variable {i}/{len(VARS_TO_IMPORT)}")

        # Step 1: Consolidation
        consolidation_success = process_variable(var_info, script_dir)

        # Step 2: Database import (if consolidation successful)
        if consolidation_success:
            import_success = import_variable(var_info, script_dir, db_path)
            results[var_info["var_name"]] = import_success
        else:
            results[var_info["var_name"]] = False

    create_var_index(db_path)

    # Final summary
    print_progress("=" * 80)
    print_progress("FINAL SUMMARY")
    print_progress("=" * 80)

    success_count = sum(results.values())
    total_count = len(results)

    for var_name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print_progress(f"{status}: {var_name}")

    print_progress("")
    print_progress(
        f"Processing completed: {success_count}/{total_count} variables processed",
        level="SUCCESS" if success_count == total_count else "WARNING",
    )

    print("*" * 80)
    print("Importing reference sites")

    import_reference_sites()

    return 0 if success_count == total_count else 1


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print_progress("\nInterrupted by user", level="WARNING")
        sys.exit(130)
    except Exception as e:
        print_progress(f"Fatal error: {str(e)}", level="ERROR")
        sys.exit(1)
