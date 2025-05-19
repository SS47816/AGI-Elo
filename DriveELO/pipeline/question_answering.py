import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from AGI-Elo.utils.rating_system import (compute_ratings,
                                          convert_prediction_to_match_results)

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def extract_answer(response: str) -> str:
    # Remove <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    # Search for the first occurrence of A, B, C, or D as a standalone uppercase letter
    match = re.search(r"\b([ABCD])\b", cleaned)

    if match:
        return match.group(1)
    else:
        return None  # or raise an exception or return "Unknown"

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
    for _, row in df.iterrows():
        test_case_name: str = row["Test Case"]
        test_case_label: str = row["True Label"]
        accuracy: float = 0.0
        score: float = 0.0
        try:
            choice = extract_answer(row[model_name])
            accuracy = float(choice == test_case_label)
            score = accuracy
        except Exception as e:
            print(f"{str(e)} in answer: {row[model_name]}")

        match_results.append([test_case_label, test_case_name, model_name, 1 - score, score, accuracy])

    return match_results


def main():
    parser = argparse.ArgumentParser(description='Covert to match results & Compute ratings')
    parser.add_argument('--dataset_task', default="all", type=str, help=' "all", "high_school_physics", ... ')
    parser.add_argument('--dataset_split', default="test", type=str, help=' "test", "validation", "dev", "auxiliary_train" ')
    parser.add_argument('--save_path', default="./data/question_answering/MMLU/",type=str, help='path to save processed data')
    args = parser.parse_args()

    # Input and Output Data folders
    output_folder = Path(args.save_path).joinpath(f"{args.dataset_split}/")
    output_folder.mkdir(parents=True, exist_ok=True)

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
    compute_ratings(match_result_folder, rating_result_folder, dataset_name="MMLU", metric_name="Accuracy", left_margin=800)


if __name__ == "__main__":
    main()
