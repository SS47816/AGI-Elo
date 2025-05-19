import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             root_mean_squared_error)
from tqdm import tqdm

from AGI_Elo.utils.elo_system import update_elo
from AGI_Elo.utils.glicko_system import update_glicko
from AGI_Elo.utils.visualization import (plot_evaluation_trends,
                                          plot_model_rating_trends,
                                          plot_outcome_vs_rating, plot_ratings,
                                          plot_samples)

# Set up logging configuration
logger = logging.getLogger(__name__)

def convert_prediction_to_match_results(prediction_result_folder: Path, match_result_folder: Path, match_score_func: callable) -> None:
    # Loop through both csv and pkl files
    for file in prediction_result_folder.glob("*"):
        if file.suffix == ".csv":
            logger.info(f"Processing CSV: {file.name}")
            df = pd.read_csv(file)
        elif file.suffix == ".pkl":
            logger.info(f"Processing PKL: {file.name}")
            df = pd.read_pickle(file)
        else:
            continue

        model_name = file.stem

        try:
            match_results = match_score_func(df, model_name)
        except Exception as e:
            logger.error(f"Error when processing {file}: {str(e)}, skipped")

        # Save to new pkl file
        df_long = pd.DataFrame(match_results, columns=["True Label", "Test Case", "Model", "Test Case Score", "Model Score", "Model Performance"])
        match_result_path = match_result_folder.joinpath(f"match_{model_name}.pkl")
        df_long.to_pickle(match_result_path)
        logger.info(f"Long-format results saved to {match_result_path}")

    return

def compute_ratings(match_result_folder: Path, rating_result_folder: Path, dataset_name: str, metric_name: str, dataset_folder: Path=None,
                    num_rounds_list: list=[1], method: str='Glicko', init_mu: int=1500, init_rd: int=350,
                    bin_size: int=100, left_margin: int=400, min_rating_diff: int=-1250, max_rating_diff: int=1250, record: bool=True) -> None:
    # Load all .pkl files and concatenate all DataFrames
    pkl_files = sorted(match_result_folder.glob("*.pkl"))
    dfs = [pd.read_pickle(file) for file in pkl_files]
    df = pd.concat(dfs, ignore_index=True)
    num_mathces = len(df)

    # Initialize Ratings
    test_case_ratings = {}
    model_ratings = {}
    for _, row in df.iterrows():
        label = row["True Label"]
        test_case = row["Test Case"]
        model = row["Model"]
        if test_case not in test_case_ratings:
            test_case_ratings[test_case] = {'label': label, 'mu': init_mu, 'rd': init_rd, 'num_matches': 0, 'total_perf': 0.0, 'total_score': 0.0}
        if model not in model_ratings:
            model_ratings[model] = {'mu': init_mu, 'rd': init_rd, 'num_matches': 0, 'total_perf': 0.0, 'total_score': 0.0}

    # Process each match (test case vs model)
    total_num_rounds = max(num_rounds_list)
    for i in tqdm(range(total_num_rounds), desc=f"Running Match Rounds"):
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

        milestones = np.linspace(0, num_mathces-1, 101).astype(int)[1:]
        record_eval_df = pd.DataFrame(0.0, index=np.arange(1, 101, 1), columns=["MAE", "MSE", "RMSE", "Test Case rho", "Model rho"])
        record_model_df = pd.DataFrame(1500.0, index=np.arange(0, 101, 1), columns=list(model_ratings.keys()))

        # Start rated matches
        for j, (_, row) in enumerate(tqdm(df_shuffled.iterrows(), desc=f"Running Matches")):
            # Get match results
            test_case = row["Test Case"]
            model = row["Model"]
            perf_model = row["Model Performance"]
            score_test_case = row["Test Case Score"]
            score_model = row["Model Score"]

            # Extract current ratings
            rating_test_case: dict = test_case_ratings[test_case]
            rating_model: dict = model_ratings[model]

            # Update the total performance and score
            rating_test_case['total_perf'] += perf_model
            rating_model['total_perf'] += perf_model
            rating_test_case['total_score'] += score_test_case
            rating_model['total_score'] += score_model
            rating_test_case['num_matches'] += 1
            rating_model['num_matches'] += 1

            # Update Ratings
            if method == 'Elo':
                new_rating_test, new_rating_model = update_elo(rating_test_case, rating_model, score_test_case, score_model)
            elif method == 'Glicko':
                new_rating_test, new_rating_model = update_glicko(rating_test_case, rating_model, score_test_case, score_model)

            # Save updated ratings
            test_case_ratings[test_case] = new_rating_test
            model_ratings[model] = new_rating_model

            if record and j in milestones:
                progress_index = np.where(milestones == j)[0][0]
                (test_case_rho, model_rho), (test_case_df, model_df) = compute_rating_system_consistency(test_case_ratings, model_ratings)
                (mae, mse, rmse), _ = compute_rating_system_accuracy(df_shuffled[:j], test_case_df, model_df, min_rating_diff=min_rating_diff, max_rating_diff=max_rating_diff, bin_size=bin_size)
                record_eval_df.iloc[progress_index, :] = mae, mse, rmse, test_case_rho, model_rho
                record_model_df.iloc[progress_index+1, :] = [model_ratings[model]['mu'] for model in model_ratings.keys()]

        if i + 1 not in num_rounds_list:
            continue

        # Compute the final system consistency
        (test_case_rho, model_rho), (test_case_df, model_df) = compute_rating_system_consistency(test_case_ratings, model_ratings)
        logger.info(f"Test Cases Spearman correlation: {test_case_rho:.4f}")
        logger.info(f"  Models   Spearman correlation: {model_rho:.4f}")

        # Save ratings to pkl files
        test_case_ratings_path = rating_result_folder.joinpath(f"rating_{dataset_name}_{metric_name}_test_case_{method}_{i + 1}.pkl")
        model_ratings_path = rating_result_folder.joinpath(f"rating_{dataset_name}_{metric_name}_model_{method}_{i + 1}.pkl")
        test_case_df.to_pickle(test_case_ratings_path)
        model_df.to_pickle(model_ratings_path)
        # model_df.to_csv(model_ratings_path.with_name(model_ratings_path.stem + ".csv"))
        logger.info(f"Ratings saved to {test_case_ratings_path}, {model_ratings_path}")
        logger.info(f"# of test cases: {len(test_case_df)}, # of models: {len(model_df)}")

        # Visualize the Ratings
        rating_figure_path = rating_result_folder.joinpath(f"{dataset_name}_{metric_name}_dist_{method}_{i + 1}.pdf")
        plot_ratings(test_case_ratings_path, model_ratings_path, rating_figure_path, dataset_name, show_plot=False, bin_size=bin_size, left_margin=left_margin)
        if dataset_folder is not None:
            sample_figure_path = rating_result_folder.joinpath(f"{dataset_name}_{metric_name}_sample_{method}_{i + 1}.pdf")
            plot_samples(dataset_folder, test_case_ratings_path, sample_figure_path, num_samples_per_bin=3)

        # Compute & Show Evaluation Results
        results, assets = compute_rating_system_accuracy(df, test_case_df, model_df, min_rating_diff=min_rating_diff, max_rating_diff=max_rating_diff, bin_size=bin_size)
        logger.info(f"MAE  = {results[0]:.4f}")
        logger.info(f"MSE  = {results[1]:.4f}")
        logger.info(f"RMSE = {results[2]:.4f}")

        val_figure_path = rating_result_folder.joinpath(f"{dataset_name}_{metric_name}_val_{method}_{i + 1}.pdf")
        plot_outcome_vs_rating(*assets, val_figure_path, dataset_name, metric_name, min_rating_diff=min_rating_diff, max_rating_diff=max_rating_diff, show_plot=False)

        # Save Recorded Evaluation Figures across % of matches
        if record:
            eval_figure_path = rating_result_folder.joinpath(f"{dataset_name}_{metric_name}_eval_{method}_{i + 1}.pdf")
            plot_evaluation_trends(record_eval_df, eval_figure_path, dataset_name,show_plot=False)
            model_figure_path = rating_result_folder.joinpath(f"{dataset_name}_{metric_name}_model_{method}_{i + 1}.pdf")
            plot_model_rating_trends(record_model_df, model_figure_path, dataset_name, show_plot=False)

    return

def compute_rating_system_consistency(test_case_ratings: dict, model_ratings: dict) -> True:
    # Convert Ratings to a DataFrame
    test_case_data = [(values['label'], name, values['mu'], values['rd'], values['total_perf']/max(1, values['num_matches']), values['total_score']/max(1, values['num_matches'])) for name, values in test_case_ratings.items()]
    model_data = [(name, values['mu'], values['rd'], values['total_perf']/max(1, values['num_matches']), values['total_score']/max(1, values['num_matches'])) for name, values in model_ratings.items()]
    test_case_df = pd.DataFrame(test_case_data, columns=["Label", "Name", "Rating", "Deviation", "Average Performance", "Average Score"])
    model_df = pd.DataFrame(model_data, columns=["Name", "Rating", "Deviation", "Average Performance", "Average Score"])
    test_case_df = test_case_df.sort_values(by="Rating", ascending=False)
    model_df = model_df.sort_values(by="Rating", ascending=False)

    # Calculate Spearman correlation
    rho_test_case, p_value = spearmanr(test_case_df["Rating"], test_case_df["Average Performance"])
    # logger.info(f"Test Cases Spearman correlation: {rho_test_case:.4f}, P-value: {p_value}")
    rho_model, p_value = spearmanr(model_df["Rating"], model_df["Average Performance"])
    # logger.info(f"Models Spearman correlation: {rho_model:.4f}, P-value: {p_value}")

    return (rho_test_case, rho_model), (test_case_df, model_df)

def compute_rating_system_accuracy(matches_df: pd.DataFrame, test_case_df: pd.DataFrame, model_df: pd.DataFrame, min_rating_diff: int=-1250, max_rating_diff: int=1250, bin_size: int=100) -> tuple:
    # Build a pivot table for test case vs. model outcomes
    pivot_df_outcome = matches_df.pivot_table(index="Test Case", columns="Model", values="Model Score")
    pivot_df_outcome = pivot_df_outcome.sort_index().sort_index(axis=1)
    # print(pivot_df_outcome.head())

    # Build a pivot table for test case vs. model rating differences
    test_case_df = test_case_df.set_index("Name")
    model_df = model_df.set_index("Name")
    rating_diff = model_df["Rating"].values[None, :] - test_case_df["Rating"].values[:, None]
    pivot_df_rating_diff = pd.DataFrame(
        rating_diff,
        index=test_case_df.index,
        columns=model_df.index
    )
    pivot_df_rating_diff = pivot_df_rating_diff.sort_index().sort_index(axis=1)
    # print(pivot_df_rating_diff.head())

    # Define bins and labels from -800 to +800
    bin_edges = np.arange(min_rating_diff, max_rating_diff, bin_size)
    bin_labels = [f"[{bin_edges[i]},{bin_edges[i+1]})" for i in range(len(bin_edges)-1)]

    # Dict to collect average scores per bin per model
    binned_counts = {}
    binned_avg_scores = {}
    binned_avg_ratings = {}
    # Loop over each model
    for model in pivot_df_outcome.columns:
        rating_diffs = pivot_df_rating_diff[model]
        model_scores = pivot_df_outcome[model]

        # Bin the rating differences
        bins = pd.cut(rating_diffs, bins=bin_edges, labels=bin_labels, right=False)

        # Group scores by bin and calculate mean
        bin_count = model_scores.groupby(bins, observed=False).count()
        bin_avg_scores = model_scores.groupby(bins, observed=False).mean()
        bin_avg_ratings = rating_diffs.groupby(bins, observed=False).mean()

        # Ensure all bins are present (even if NaN)
        bin_count = bin_count.reindex(bin_labels)
        bin_avg_scores = bin_avg_scores.reindex(bin_labels)
        bin_avg_ratings = bin_avg_ratings.reindex(bin_labels)

        binned_counts[model] = bin_count
        binned_avg_scores[model] = bin_avg_scores
        binned_avg_ratings[model] = bin_avg_ratings

    # Combine into a DataFrame
    counts = pd.DataFrame(binned_counts).values.flatten()
    outcomes = pd.DataFrame(binned_avg_scores).values.flatten()
    rating_diffs = pd.DataFrame(binned_avg_ratings).values.flatten()

    # Theoretical Rating curve
    theoretical_rating_diffs = np.linspace(min_rating_diff, max_rating_diff, 1000)
    theoretical_outcomes = 1 / (1 + 10 ** (-theoretical_rating_diffs / 400))

    # Compute correlation R
    mask = ~np.isnan(rating_diffs) & ~np.isnan(outcomes)

    counts = counts[mask]
    rating_diffs = rating_diffs[mask]
    outcomes = outcomes[mask]
    expected_outcomes = 1 / (1 + 10 ** (-rating_diffs / 400))

    # r2 = r2_score(expected_outcomes, outcomes, sample_weight=counts)
    mae = mean_absolute_error(expected_outcomes, outcomes, sample_weight=counts)
    mse = mean_squared_error(expected_outcomes, outcomes, sample_weight=counts)
    rmse = root_mean_squared_error(expected_outcomes, outcomes, sample_weight=counts)

    return (mae, mse, rmse), (outcomes, rating_diffs, theoretical_outcomes, theoretical_rating_diffs, counts)
