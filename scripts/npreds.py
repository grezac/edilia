import os
import sys
import argparse

# Go up one level to reach the root folder "edilia"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from libs.featselect import PredictorSearch
from libs.get_settings import read_settings
from libs.stats_results import stats_results

CONFIG_NAME = "GENERIC_CONFIG"


def parse_int_list(string):
    # 1. Basic cleaning: remove brackets and spaces
    clean_string = string.strip("[]").replace(" ", "")

    # 2. Verification if the string is empty (case of --num_pred "[]" or "")
    if not clean_string:
        raise argparse.ArgumentTypeError("The list --num_pred cannot be empty.")

    # 3. split on commas and convert to integers
    parts = clean_string.split(",")
    int_list = []
    for p in parts:
        if p == "":
            continue

        try:
            int_list.append(int(p))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"The element '{p}' is not a valid integer in '{string}'."
            )

    # 4. Final verification in case we have only commas ",,,"
    if not int_list:
        raise argparse.ArgumentTypeError("No valid integers found in the list.")

    return int_list


def main():
    parser = argparse.ArgumentParser(description="Launch predictor search process")

    # Mandatory parameters
    parser.add_argument(
        "--site", help="Target site (e.g. PARIS, BIRMINGHAM,STRASBOURG)", required=True
    )
    # argument defining a list of integers, with flexible input format
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        help='List of integers, e.g., [1,2, 3] or "1, 2"',
        required=True,
    )

    parser.add_argument(
        "--num_predictors",
        type=parse_int_list,
        help='List of integers, e.g., [50,60, 70] or "50, 60"',
        required=True,
    )

    args = parser.parse_args()

    site = args.site.upper()
    seeds = args.seeds
    num_predictors = args.num_predictors

    for num in num_predictors:
        files = []
        for seed in seeds:
            settings = read_settings(site, CONFIG_NAME)
            settings["CONFIGS"][CONFIG_NAME]["seed"] = seed
            settings["CONFIGS"][CONFIG_NAME]["rings"][0]["max_predictors"] = num
            settings["CONFIGS"][CONFIG_NAME]["name_extension"] = f"npreds{num}"
            predictorSearch = PredictorSearch(settings)
            file_name = predictorSearch.process()
            files.append(file_name)
        stats_results(
            files,
            f"/data/outputs/results/{CONFIG_NAME}/{site}_npreds{num}_summary.json",
        )


if __name__ == "__main__":
    main()
