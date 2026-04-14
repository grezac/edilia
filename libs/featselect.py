"""
featselect.py
-----------------
Main module containing the Genetic Algorithm.
Three main classes are defined: Individual, PopulationPredictorSearch, and PredictorSearch.
- PredictorSearch is the main class that orchestrates the entire process, from configuration validation
to executing the evolution and summarizing results.
- PopulationPredictorSearch manages a population of individuals for a specific ring,
including their evaluation, selection, crossover, and mutation.
- Individual represents a single solution (a set of predictors) and contains methods to compute its error
and perform mutation.

JSON files are generated to store the results of the search, including the best predictors and their performance metrics.

"""

import sqlite3
import sys
import random
import os
import pickle
import json
import time
import numpy as np
from sklearn.neighbors import BallTree


from .utils import (
    find_nearest_point,
    calculate_haversine_distance,
    calculate_inclusive_days,
    format_duration,
    linear_regression,
    lgbm_regression,
    xgboost_regression,
)


class Individual:

    def __init__(self, cursor, config, ring_sites, ref_series):
        self.cursor = cursor
        self.config = config
        self.ref_series = ref_series
        self.active_ring = config["rings"][config["num_ring"]]  # for readibility
        self.ring_sites = ring_sites

        n_random = self.active_ring["max_predictors"] - len(self.active_ring["pool"])

        random_ids = random.sample(self.ring_sites, n_random)
        self.predictor_IDs = self.active_ring["pool"] + random_ids
        self.series = []
        self.var_names = []  # for debugging

        for predictor_ID in self.predictor_IDs:
            self.cursor.execute(
                """
                SELECT id, var_name, series
                FROM feature_series 
                WHERE id = ?
                """,
                (predictor_ID,),
            )
            row = self.cursor.fetchone()
            series_blob = row[2]
            array = pickle.loads(series_blob)
            self.series.append(array)
            self.var_names.append(row[1])  # pour debug !!!

    def compute_error(self, full_calculation=False):

        results = linear_regression(
            self.ref_series,
            self.series,
            self.config["window_size"],
            self.config["horizon"],
            self.config["test_size"],
            self.config["train_split"],
            self.config["test_period"][0],
            self.config["test_period"][1],
            full_calculation,
        )

        if full_calculation:
            self.mae_val = results["mae_val"]
            self.linear_stats = results
        else:
            self.mae_val = results["mae_val"]
            self.mae_test = results["mae_test"]

    def complete_stats(self):
        results = lgbm_regression(
            self.ref_series,
            self.series,
            self.config["window_size"],
            self.config["horizon"],
            self.config["test_size"],
            self.config["train_split"],
            self.config["test_period"][0],
            self.config["test_period"][1],
            self.config["seed"],
            True,
        )

        self.lightGBM = results
        results = xgboost_regression(
            self.ref_series,
            self.series,
            self.config["window_size"],
            self.config["horizon"],
            self.config["test_size"],
            self.config["train_split"],
            self.config["test_period"][0],
            self.config["test_period"][1],
            self.config["seed"],
            with_importance=True,
        )

        self.XGBoost = results

    def mutation(self):
        idx_to_mutate = np.random.randint(len(self.series))
        random_id = random.sample(self.ring_sites, 1)
        self.cursor.execute(
            """
                SELECT id, var_name, series
                FROM feature_series 
                WHERE id = ?
                """,
            (random_id[0],),  # Only pass the ID value, always as a one-element tuple
        )
        row = self.cursor.fetchone()
        if row:
            series_blob = row[2]
            array = pickle.loads(series_blob)
            self.series[idx_to_mutate] = array
            self.predictor_IDs[idx_to_mutate] = random_id[0]

    def copy_from(self, source_individual):
        self.mae_val = source_individual.mae_val
        self.mae_test = source_individual.mae_test
        self.series = source_individual.series.copy()
        self.predictor_IDs = source_individual.predictor_IDs.copy()


class PopulationPredictorSearch:

    def __init__(self, config, cursor):
        self.config = config
        self.cursor = cursor
        query = f"SELECT fs.id, fs.series, s.latitude, s.longitude FROM feature_series fs JOIN sites s ON fs.site_id = s.site_id WHERE fs.id={self.config['ref_site']['id']}"
        self.cursor.execute(query)
        row = self.cursor.fetchone()
        series_blob = row[1]
        self.ref_series = pickle.loads(
            series_blob
        )  # storing the series from the reference site

        target_coords_deg = np.array([[row[2], row[3]]])
        self.target_latitude = row[2]
        self.target_longitude = row[3]
        target_coords_rad = np.radians(target_coords_deg)
        excluded_vars = self.config["excluded_vars"]

        # creation of placeholders
        placeholders = ", ".join(["?"] * len(excluded_vars))

        query = f"""
            SELECT fs.id, s.latitude, s.longitude 
            FROM feature_series fs 
            JOIN sites s ON fs.site_id = s.site_id 
            WHERE fs.origin = ? 
            AND fs.var_name NOT IN ({placeholders})
        """

        # Execution with parameters
        params = [self.config["origin"]] + excluded_vars
        self.cursor.execute(query, params)

        all_records = self.cursor.fetchall()
        # Data separation
        site_ids = [record[0] for record in all_records]  # IDs list

        all_sites_coords = [
            [record[1], record[2]] for record in all_records
        ]  # coordinates list
        # conversion from degrees to radians
        all_sites_coords_radians = np.radians(all_sites_coords)

        # Index creation
        tree = BallTree(all_sites_coords_radians, metric="haversine")

        # Circle parameters
        R_EARTH_KM = 6371
        self.active_ring = config["rings"][config["num_ring"]]
        # The radius in kilometers must be expressed in radians (angle units)
        radius_min = self.active_ring["radius_min"] / R_EARTH_KM
        radius_max = self.active_ring["radius_max"] / R_EARTH_KM
        inner_ring_indexes = tree.query_radius(target_coords_rad, r=radius_min)[0]
        outer_ring_indexes = tree.query_radius(target_coords_rad, r=radius_max)[0]

        # Conversion of indexes into sets of identifiers
        ring_inner_c = {site_ids[i] for i in inner_ring_indexes}
        ring_outer_c = {site_ids[i] for i in outer_ring_indexes}
        ring_sites = list(ring_outer_c - ring_inner_c)

        # Calculate the ring IDs; remove the anchors to avoid duplicates
        self.ring_sites = [
            x for x in ring_sites if x not in set(self.active_ring["pool"])
        ]

        self.individuals = []
        for _ in range(self.config["evolution_settings"]["population_size"]):
            individual = Individual(
                self.cursor, self.config, self.ring_sites, self.ref_series
            )
            self.individuals.append(individual)

        if len(self.config["name_extension"]) > 0:
            name_extension = self.config["name_extension"]
        else:
            name_extension = ""

        self.json_file_name = os.path.join(
            self.config["json_dir_name"],
            f"{self.config['ref_site']['label']}_{self.config['num_ring']}{name_extension}.json",
        )
        self.snapshot = self.config
        self.snapshot["best_individuals"] = []

    def __write_json__(self, num_generation):
        best_mae_key = "linear_regression_on_val"
        best_individual_results = {
            "predictor_IDs": self.best_individual.predictor_IDs,
            "num_generation": num_generation,
            best_mae_key: round(self.best_individual.mae_val, 4),
        }
        self.snapshot["best_individuals"].append(best_individual_results)
        with open(self.json_file_name, "w", encoding="utf-8") as f:
            json.dump(self.snapshot, f, ensure_ascii=False, indent=4)

    def __finalize_json__(self, duration):
        self.best_individual.compute_error(full_calculation=True)
        self.best_individual.complete_stats()

        self.snapshot["final_stats"] = {
            "elapsed_time": duration,
            "num_potential_predictors": len(
                self.ring_sites
            ),  # In fact, related to the ring number !!!!!
            "linear_stats": self.best_individual.linear_stats,
            "lightGBM_stats": self.best_individual.lightGBM,
            "XGBoost_stats": self.best_individual.XGBoost,
        }
        with open(self.json_file_name, "w", encoding="utf-8") as f:
            json.dump(self.snapshot, f, ensure_ascii=False, indent=4)

    def __get_predictor_data__(self, predictor_ID):
        self.cursor.execute(
            """
                SELECT fs.id, fs.var_name, s.latitude, s.longitude 
                FROM feature_series fs JOIN sites s 
                ON fs.site_id = s.site_id WHERE fs.id=?
                """,
            (predictor_ID,),
        )
        row = self.cursor.fetchone()
        lat = row[2]
        lon = row[3]
        distance_km = calculate_haversine_distance(
            self.target_latitude, self.target_longitude, lat, lon
        )
        results = {
            "id": predictor_ID,
            "lat": round(lat, 2),
            "lon": round(lon, 2),
            "distance_km": round(distance_km, 2),
            "var_name": row[1],
        }
        return results

    def results_summary(self, duration):
        results = {}
        overview = self.snapshot.copy()
        del overview["best_individuals"]
        del overview["num_ring"]
        del overview["final_stats"]
        results["overview"] = overview

        self.best_individual.compute_error(full_calculation=True)
        self.best_individual.complete_stats()

        linear_stats = self.best_individual.linear_stats.copy()
        del linear_stats["importance_gain"]
        del linear_stats["importance_split"]

        lightGBM_stats = (
            self.best_individual.lightGBM.copy()
        )  # We need to make a copy of the dictionary!
        del lightGBM_stats["importance_gain"]
        del lightGBM_stats["importance_split"]
        XGBoost_stats = self.best_individual.XGBoost.copy()
        del XGBoost_stats["importance_gain"]
        del XGBoost_stats["importance_split"]
        results["total_elapsed_time"] = duration
        results["linear_stats"] = linear_stats
        results["lightGBM_stats"] = lightGBM_stats
        results["XGBoost_stats"] = XGBoost_stats
        predictors = []
        i = 0
        for pred in self.best_individual.predictor_IDs:
            predictor_data = self.__get_predictor_data__(pred)
            predictor_data["linear_gain"] = round(
                self.best_individual.linear_stats["importance_gain"][i], 4
            )
            predictor_data["linear_split"] = round(
                self.best_individual.linear_stats["importance_split"][i], 4
            )

            predictor_data["LGBM_gain"] = round(
                self.best_individual.lightGBM["importance_gain"][i], 4
            )
            predictor_data["LGBM_split"] = self.best_individual.lightGBM[
                "importance_split"
            ][i]
            predictor_data["XGBoost_gain"] = round(
                self.best_individual.XGBoost["importance_gain"][i], 4
            )
            predictor_data["XGBoost_split"] = int(
                round(self.best_individual.XGBoost["importance_split"][i])
            )
            predictors.append(predictor_data)
            i += 1
        results["predictors"] = predictors
        if self.config["name_extension"] == "":
            name_extension = ""
        else:
            name_extension = self.config["name_extension"]

        file_name = os.path.join(
            self.config["json_dir_name"],
            f"{self.config['ref_site']['label']}_summary{name_extension}.json",
        )

        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    def __evaluate_population__(self, stage_num=1):
        # We'll use something simple (no workers)
        record_broken = False
        for ind in self.individuals:
            ind.compute_error()
            if ind.mae_val < self.best_individual.mae_val:
                self.best_individual.copy_from(ind)
                self.best_individual.mae_val = round(self.best_individual.mae_val, 4)
                self.best_individual.mae_test = round(self.best_individual.mae_test, 4)
                record_broken = True
        return record_broken

    def __tournament__(self):
        # Random selection of two different individuals
        idx1, idx2 = np.random.choice(
            self.config["evolution_settings"]["population_size"], 2, replace=False
        )
        # Select the better of the two
        winner = (
            self.individuals[idx1]
            if (self.individuals[idx1].mae_val < self.individuals[idx2].mae_val)
            else self.individuals[idx2]
        )
        return winner

    def __crossover_uniform_column__(
        self, parent1: "Individual", parent2: "Individual", prob: float = 0.5
    ) -> "Individual":
        """
        Uniform crossover at the column level.

        Principle:
        - For each column position (except index 0), decide with a given probability
          whether to take the column from parent1 or parent2
        - Automatically avoids duplicates by building progressively
        """

        n_fixed_elements = self.active_ring["max_predictors"] - len(
            self.active_ring["pool"]
        )

        child = Individual(self.cursor, self.config, self.ring_sites, self.ref_series)

        child.copy_from(parent1)  # Initialize with parent1
        used_record_ids = set(parent1.predictor_IDs[:n_fixed_elements])

        for i in range(n_fixed_elements, len(child.series)):
            # Available candidates from both parents for this position
            candidates = []

            # Check parent1
            if parent1.predictor_IDs[i] not in used_record_ids:
                candidates.append(("parent1", i))

            # Check parent2 for this position and other available positions
            for j in range(n_fixed_elements, len(parent2.series)):
                if parent2.predictor_IDs[j] not in used_record_ids:
                    candidates.append(("parent2", j))
                    break  # Take the first available for this iteration

            if candidates:
                if len(candidates) == 1 or np.random.random() < prob:
                    chosen = candidates[0]
                else:
                    chosen = candidates[1]
                if chosen[0] == "parent2":
                    parent_source = parent2
                    source_idx = chosen[1]
                    child.series[i] = parent_source.series[source_idx].copy()
                    child.predictor_IDs[i] = parent_source.predictor_IDs[source_idx]
                used_record_ids.add(child.predictor_IDs[i])
        return child

    def static_evolution(self):
        start_time = time.time()

        self.best_individual = Individual(
            self.cursor, self.config, self.ring_sites, self.ref_series
        )
        self.best_individual.mae_val = float("inf")
        self.generation = 0
        patience_counter = 0
        self.__evaluate_population__(stage_num=1)
        active_ring = self.config["rings"][self.config["num_ring"]]
        record_broken = True
        record_broken_count = (
            0  # To ensure that evolution continues beyond the first record
        )
        stage = 1
        while (
            patience_counter < active_ring["patience_stop"] or record_broken_count < 2
        ):

            if record_broken:
                record_broken_count += 1
            print(
                f"Gen {self.generation} | best mae on validation set: {self.best_individual.mae_val}| pc: {patience_counter}"
            )

            new_individuals = []
            size = self.config["evolution_settings"][
                "population_size"
            ]  # No elitism => regenerate the entire population
            for _ in range(size):
                if (
                    np.random.rand()
                    < self.config["evolution_settings"]["crossover_rate"]
                ):
                    parent1 = self.__tournament__()
                    parent2 = self.__tournament__()
                    new_individual = self.__crossover_uniform_column__(parent1, parent2)
                else:
                    new_individual = (
                        self.__tournament__()
                    )  # No crossover, we select a winner
                # Apply mutation
                if (
                    np.random.rand()
                    < self.config["evolution_settings"]["mutation_rate"]
                ):
                    new_individual.mutation()

                new_individuals.append(new_individual)
            # Update population with new individuals

            for i in range(self.config["evolution_settings"]["population_size"]):
                self.individuals[i].copy_from(new_individuals[i])

            record_broken = self.__evaluate_population__(stage_num=stage)
            if record_broken:
                patience_counter = 0
            else:
                patience_counter += 1
            self.generation += 1

        end_time = time.time()


class PredictorSearch:

    def __init__(self, settings):

        start_time = time.time()
        configs = settings["CONFIGS"]

        first_key = next(iter(configs))
        config = configs[first_key]
        self.__complete_config__(config)
        print("Configuration successfully validated. Instance created.")
        self.config = config

        self.config_name = first_key

        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"])

    def __complete_config__(self, config):
        # calculation of test set size
        # test_size = calculate_inclusive_days("2023-01-01", "2024-12-31")

        test_size = calculate_inclusive_days(
            config["test_period"][0], config["test_period"][1]
        )
        config["test_size"] = test_size

        # creation of "pool" keys
        for c in config["rings"]:
            c["pool"] = []

    def __get_predictor_data__(self, predictor_ID):
        self.cursor.execute(
            """
                SELECT fs.id, fs.var_name, s.latitude, s.longitude 
                FROM feature_series fs JOIN sites s 
                ON fs.site_id = s.site_id WHERE fs.id=?
                """,
            (predictor_ID,),
        )
        row = self.cursor.fetchone()

        lat = row[2]
        lon = row[3]

        distance_km = calculate_haversine_distance(
            self.config["ref_site"]["latitude"],
            self.config["ref_site"]["longitude"],
            lat,
            lon,
        )
        results = {
            "id": predictor_ID,
            "lat": round(lat, 2),
            "lon": round(lon, 2),
            "distance_km": round(distance_km, 2),
            "var_name": row[1],
        }

        return results

    def __results_summary__(self, best_individuals, duration, generations):
        if self.config["dependent_rings"] == False:
            all_ids_flattened = [
                id for ind in best_individuals for id in ind.predictor_IDs
            ]
            best_individuals[-1].predictor_IDs = list(
                set(all_ids_flattened)
            )  # Avoid duplicates

        best_individual = best_individuals[-1]
        results = {}
        best_individual.compute_error(full_calculation=True)
        best_individual.complete_stats()

        linear_stats = best_individual.linear_stats.copy()
        del linear_stats["importance_gain"]
        del linear_stats["importance_split"]

        lightGBM_stats = (
            best_individual.lightGBM.copy()
        )  # We need to make a copy of the dictionary
        del lightGBM_stats["importance_gain"]
        del lightGBM_stats["importance_split"]
        XGBoost_stats = best_individual.XGBoost.copy()
        del XGBoost_stats["importance_gain"]
        del XGBoost_stats["importance_split"]
        results["total_elapsed_time"] = duration
        results["generations"] = generations
        results["linear_stats"] = linear_stats
        results["lightGBM_stats"] = lightGBM_stats
        results["XGBoost_stats"] = XGBoost_stats
        predictors = []
        i = 0
        for pred in best_individual.predictor_IDs:
            predictor_data = self.__get_predictor_data__(pred)

            predictor_data["linear_gain"] = round(
                best_individual.linear_stats["importance_gain"][i], 4
            )
            predictor_data["linear_split"] = round(
                best_individual.linear_stats["importance_split"][i], 4
            )

            predictor_data["LGBM_gain"] = round(
                best_individual.lightGBM["importance_gain"][i], 4
            )
            predictor_data["LGBM_split"] = best_individual.lightGBM["importance_split"][
                i
            ]
            predictor_data["XGBoost_gain"] = round(
                best_individual.XGBoost["importance_gain"][i], 4
            )
            predictor_data["XGBoost_split"] = int(
                round(best_individual.XGBoost["importance_split"][i])
            )
            predictors.append(predictor_data)
            i += 1
        results["predictors"] = predictors

        if self.config["name_extension"] == "":
            name_extension = ""
        else:
            name_extension = self.config["name_extension"]

        if self.config["name_extension"] == "":
            name_extension = ""
        else:
            name_extension = f'_{self.config["name_extension"]}'

        file_name = os.path.join(
            self.config["json_dir_name"],
            "results",
            self.config_name,
            f"seed_{str(self.config['seed'])}",
            f"{self.config['ref_site']['label']}_{name_extension}.json",
        )
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        return file_name

    def __evolution__(self):
        time_start = time.time()
        complete_folder_path = os.path.join(
            self.config["json_dir_name"],
            "results",
            self.config_name,
            f"seed_{str(self.config['seed'])}",
        )
        if not os.path.isdir(complete_folder_path):
            try:
                # Create the directory and subdirectories if necessary (makedirs)
                os.makedirs(complete_folder_path)
                print(f"Folder successfully created: {complete_folder_path}")
            except OSError as e:
                print(f"Error creating folder {complete_folder_path} : {e}")
                sys.exit()
        conn = sqlite3.connect(self.config["data_source"])
        conn.row_factory = sqlite3.Row  # To access fields by name
        self.cursor = conn.cursor()
        num_ring = 0

        # if ref_as_feature is True, the reference series is used as the first predictor.
        if self.config["ref_as_feature"]:
            for c in self.config["rings"]:
                c["pool"].append(self.config["ref_site"]["id"])
        if "anchor_vars" in self.config:
            ref_site = self.config["ref_site"]
            for v_name in self.config["anchor_vars"]:
                anchor = find_nearest_point(
                    self.cursor, ref_site["latitude"], ref_site["longitude"], v_name
                )
                for c in self.config["rings"]:
                    c["pool"].append(anchor["id"])

        best_individuals = []  # To store best individual from each set
        generations = 0
        for _ in self.config["rings"]:
            # print("*" * 50)
            # print(f"ring NUMBER {num_ring}")
            # print("*" * 50)
            self.config["num_ring"] = num_ring
            if num_ring > 0 and self.config["dependent_rings"]:
                self.config["rings"][num_ring][
                    "pool"
                ] = population.best_individual.predictor_IDs

            population = PopulationPredictorSearch(self.config, self.cursor)
            population.static_evolution()
            generations += population.generation
            best_individuals.append(population.best_individual)
            num_ring += 1

        time_end = time.time()

        self.__results_summary__(
            best_individuals, format_duration(time_end - time_start), generations
        )

        conn.close()

    def process(self):
        time_start = time.time()
        complete_folder_path = os.path.join(
            self.config["json_dir_name"],
            "results",
            self.config_name,
            f"seed_{str(self.config['seed'])}",
        )
        if not os.path.isdir(complete_folder_path):
            try:
                # Create the directory and subdirectories if necessary (makedirs)
                os.makedirs(complete_folder_path)
                print(f"Folder successfully created: {complete_folder_path}")
            except OSError as e:
                print(f"Error creating folder {complete_folder_path} : {e}")
                sys.exit()
        conn = sqlite3.connect(self.config["data_source"])
        conn.row_factory = sqlite3.Row  # To access fields by name
        self.cursor = conn.cursor()
        num_ring = 0
        # if ref_as_feature is True, the reference series is used as the first predictor.
        if self.config["ref_as_feature"]:
            for c in self.config["rings"]:
                c["pool"].append(self.config["ref_site"]["id"])
        if "anchor_vars" in self.config:
            ref_site = self.config["ref_site"]
            for v_name in self.config["anchor_vars"]:
                anchor = find_nearest_point(
                    self.cursor, ref_site["latitude"], ref_site["longitude"], v_name
                )

                for c in self.config["rings"]:
                    c["pool"].append(anchor["id"])

        best_individuals = []  # To store best individual from each set
        generations = 0
        for _ in self.config["rings"]:
            print("*" * 50)
            print(f"Ring NUMBER {num_ring}")
            print("*" * 50)
            self.config["num_ring"] = num_ring
            if num_ring > 0 and self.config["dependent_rings"]:
                self.config["rings"][num_ring][
                    "pool"
                ] = population.best_individual.predictor_IDs

            population = PopulationPredictorSearch(self.config, self.cursor)

            population.static_evolution()

            generations += population.generation
            best_individuals.append(population.best_individual)
            num_ring += 1

        time_end = time.time()
        file_name = self.__results_summary__(
            best_individuals, format_duration(time_end - time_start), generations
        )
        conn.close()
        return file_name
