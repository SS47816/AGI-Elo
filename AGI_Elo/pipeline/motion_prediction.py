import argparse
import logging
from pathlib import Path

import pandas as pd

from AGI_Elo.utils.rating_system import (compute_ratings,
                                          convert_prediction_to_match_results)

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def compute_match_results(df: pd.DataFrame, model_name: str) -> list:
    """
    Compute match results from a DataFrame of predictions.

    Args:
        df (pd.DataFrame): A DataFrame with columns "Test Case", "True Label", and the model name.
        model_name (str): The model name to use for the match results.

    Returns:
        list: A list of match results, where each element is a list [true label, test case, model name, test case score, model score].
    """

    match_results = []
    for key, value in df.items():
        scenario_id = key
        mAP = value['overall_avg']['mean_average_precision_overall_avg']
        score = mAP
        match_results.append([0.0, scenario_id, model_name, 1 - score, score, mAP])
        # miss_rate = value['overall_avg']['miss_rate_overall_avg']
        # score = 1 - miss_rate
        # match_results.append([0.0, scenario_id, model_name, 1 - score, score, miss_rate])

    return match_results


def main():
    parser = argparse.ArgumentParser(description='Covert to match results & Compute ratings')
    parser.add_argument('--dataset_split', default="val", type=str, help='"train", "val"')
    parser.add_argument('--save_path', default="./data/motion_prediction/Waymo/",type=str, help='path to save processed data')
    args = parser.parse_args()

    # Input and Output Data folders
    output_folder = Path(args.save_path).joinpath(f"{args.dataset_split}/")
    output_folder.mkdir(parents=True, exist_ok=True)
    prediction_result_folder = output_folder.joinpath("predictions/")
    prediction_result_folder.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert prediction results to match results
    match_result_folder = output_folder.joinpath("matches/")
    match_result_folder.mkdir(parents=True, exist_ok=True)
    convert_prediction_to_match_results(prediction_result_folder, match_result_folder, compute_match_results)

    # Step 2 & 3: Update Ratings based on match results & Visualize results
    rating_result_folder = output_folder.joinpath("ratings/")
    rating_result_folder.mkdir(parents=True, exist_ok=True)
    compute_ratings(match_result_folder, rating_result_folder, dataset_name="Waymo", metric_name="mAP") # "Miss Rate", "mAP"

if __name__ == "__main__":
    main()
