from pathlib import Path

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm


def plot_ratings(test_case_ratings_path: Path, model_ratings_path: Path, figure_path: Path, dataset_name: str, show_plot: bool=False, bin_size: int=50, left_margin: int=200, right_margin: int=100) -> None:
    # Load Ratings
    test_cases_df = pd.read_pickle(test_case_ratings_path)
    models_df = pd.read_pickle(model_ratings_path)
    num_test_cases = test_cases_df.shape[0]
    num_models = models_df.shape[0]

    # Sort test case ratings for cumulative percentage calculation
    test_case_ratings = test_cases_df["Rating"].values
    sorted_ratings = np.sort(test_case_ratings)
    cumulative_percent = np.arange(1, len(sorted_ratings) + 1) / len(sorted_ratings) * 100

    # Histogram (Test Case Ratings)
    rating_start = (sorted_ratings.min() // 100) * 100
    rating_end = ((sorted_ratings.max() // 100) + 1) * 100 + 1
    rating_bins = np.arange(start=rating_start, stop=rating_end, step=bin_size)
    counts, bins = np.histogram(test_case_ratings, bins=rating_bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Pick colors
    fontsize = 14
    color_hist  = ( 74/255,  98/255, 138/255) # 'deepskyblue', 'aquamarine', 'mediumspringgreen', 'turquoise'
    color_curve = '#A5678E' # 'darkorange' # ( 72/255, 166/255, 167/255) # 'deepskyblue', 'red', 'crimson', 'gold'
    # ( 72/255, 166/255, 167/255) Teal https://colorhunt.co/palette/f2efe79acbd048a6a7006a71
    # (240/255, 160/255,  75/255) Orange https://colorhunt.co/palette/f2efe79acbd048a6a7006a71

    # Color for historgram
    cmap = cm.get_cmap("Blues")
    norm = Normalize(vmin=rating_start, vmax=rating_end)
    bin_colors = [cmap(norm(center)) for center in bin_centers]

    # https://colorhunt.co/palette/dff2ebb9e5e87ab2d34a628a
    # custom_colors = [
    #     (223/255, 242/255, 235/255),
    #     (185/255, 229/255, 232/255),
    #     (122/255, 178/255, 211/255),
    #     ( 74/255,  98/255, 138/255),
    # ]
    # # https://colorhunt.co/palette/00a9ff89cff3a0e9ffcdf5fd
    # custom_colors = [
    #     (205/255, 245/255, 253/255),
    #     (160/255, 233/255, 255/255),
    #     (137/255, 207/255, 243/255),
    #     (  0/255, 169/255, 255/255),
    # ]
    # # https://colorhunt.co/palette/82aae391d8e4bfeaf5eafdfc
    # custom_colors = [
    #     (234/255, 253/255, 252/255),
    #     (191/255, 234/255, 245/255),
    #     (145/255, 216/255, 228/255),
    #     (130/255, 170/255, 227/255),
    # ]
    # custom_cmap = LinearSegmentedColormap.from_list("custom_blue", custom_colors)
    # norm = TwoSlopeNorm(vmin=rating_start, vcenter=1500, vmax=rating_end)
    # bin_colors = [custom_cmap(norm(center)) for center in bin_centers]

    # Color for models
    model_colors = cm.get_cmap('tab20', num_models).colors

    # https://colorhunt.co/palette/ffedfaffb8e0ec7fa9be5985
    # custom_red_colors = [
    #     (190/255,  89/255, 133/255),
    #     (236/255, 127/255, 169/255),
    #     (255/255, 184/255, 224/255),
    #     # (255/255, 237/255, 250/255),
    # ]
    custom_red_colors = ['#DC3971', '#EC719F', '#F3B3CC', '#ABE5E8', '#34ADAE']
    custom_cmap = LinearSegmentedColormap.from_list("custom_red", custom_red_colors)
    model_colors = [custom_cmap(i) for i in np.linspace(0, 1, num_models)]

    # Create plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Draw each histogram bar manually
    ax1.hist(test_case_ratings, bins=rating_bins, color=bin_colors[len(bin_colors) // 2], alpha=0.8, edgecolor='black', label="Test Cases")
    for left, height, color in zip(bins[:-1], counts, bin_colors):
        ax1.bar(left, height, width=bin_size, color=color, edgecolor='black', align='edge')

    ax1.set_xlim(rating_start - left_margin, rating_end + right_margin)
    ax1.set_xlabel("Rating", fontsize=fontsize, fontweight='bold')
    ax1.set_ylabel("Number of Test Cases", color=color_hist, fontsize=fontsize, fontweight='bold')
    ax1.tick_params(axis='x', labelsize=fontsize)
    ax1.tick_params(axis='y', labelcolor=color_hist, labelsize=fontsize)

    # Secondary Y-axis for Cumulative Percentage
    ax2 = ax1.twinx()
    ax2.plot(sorted_ratings, cumulative_percent, color=color_curve, alpha=0.8, linestyle='-', linewidth=2, label="Cumulative %")
    ax2.set_ylabel("Cumulative Percentage", color=color_curve, fontsize=fontsize, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_curve, labelsize=fontsize)

    # Vertical lines for Model Ratings
    for name, rating, color in zip(models_df["Name"].values, models_df["Rating"].values, model_colors):
        ax1.axvline(rating, color=color, linestyle='dashed', linewidth=2, label=name)

    # Title & Legends
    # ax1.set_title(f"Model & Test Case Rating Distributions on {dataset_name}")
    ax1.legend(loc="upper left")
    ax2.legend(loc="center right", bbox_to_anchor=(1.0, 0.8))

    plt.savefig(figure_path, bbox_inches="tight")
    if show_plot:
        plt.show()

    # Creat a thumbnail version for graphical abstract
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_axis_off()
    fig.patch.set_facecolor('none')
    ax1.set_facecolor('none')

    # Histogram bar plot
    best_model_rating = models_df[models_df["Name"] != "Human"]["Rating"].max()
    for left, height, color in zip(bins[:-1], counts, bin_colors):
        ax1.bar(left, height, width=bin_size, color=color, edgecolor='black', align='edge')
        # if left >= best_model_rating:
        #     ax1.bar(left, height, width=bin_size, color="darkgray", edgecolor='black', align='edge', alpha=0.9)
        # else:
        #     ax1.bar(left, height, width=bin_size, color="#4E71FF", edgecolor='black', align='edge')

    # Save thumbnail with no background or axes
    thumbnail_path = figure_path.with_name(figure_path.stem + "_thumbnail.png")
    plt.savefig(thumbnail_path, bbox_inches="tight", dpi=300, transparent=True)

def plot_samples(dataset_path: Path, test_case_ratings_path: Path, figure_path: Path, num_samples_per_bin: int=3, show_plot: bool=False) -> None:
    df = pd.read_pickle(test_case_ratings_path)

    # Get test case ratings and define bins
    test_case_ratings = df["Rating"].values
    bin_width = 200
    rating_start = (test_case_ratings.min() // bin_width) * bin_width
    rating_end = ((test_case_ratings.max() // bin_width) + 1) * bin_width + 1
    rating_bins = np.arange(start=rating_start, stop=rating_end, step=bin_width)

    # Compute percentiles for each bin
    sorted_ratings = np.sort(test_case_ratings)
    # percentiles = [(np.sum(sorted_ratings <= bin_max) / len(sorted_ratings) * 100) for bin_max in rating_bins[1:]]
    percentiles = [(np.sum(sorted_ratings > bin_min) / len(sorted_ratings) * 100) for bin_min in rating_bins[:-1]]

    # Dictionary to store selected images
    selected_images = {}

    # Group images by rating bins and sample
    for i in range(len(rating_bins) - 1):
        bin_min, bin_max = rating_bins[i], rating_bins[i + 1]
        bin_df = df[(df["Rating"] >= bin_min) & (df["Rating"] < bin_max)]
        sampled = bin_df.sample(min(num_samples_per_bin, len(bin_df)), random_state=42)

        if not sampled.empty:
            selected_images[bin_min] = sampled[["Name", "Label"]].values.tolist()

    # Figure settings
    num_cols = len(selected_images)
    num_rows = max(len(images) for images in selected_images.values()) if selected_images else 1
    fig_width = max(7, num_cols * 1.5)
    fig_height = max(3, num_rows * 1.5)
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(fig_width, fig_height))

    # Ensure axes is always a 2D array (prevents indexing issues)
    axes = np.atleast_2d(axes)

    # Plot images
    for col, (bin_min, images) in enumerate(selected_images.items()):
        for row, (img_name, img_label) in enumerate(images):
            img_path = dataset_path.joinpath(img_label, img_name)

            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is None:  # Handle OpenCV read errors
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (64, 64))
                axes[row, col].imshow(img)
                axes[row, col].axis("off")

    # Titles for rating bins
    for col, (bin_min, percentile) in enumerate(zip(selected_images.keys(), percentiles)):
        axes[0, col].set_title(f"{int(bin_min)}-{int(bin_min + bin_width)}\nTop {percentile:.1f}% Hard", fontsize=10)

    plt.subplots_adjust(hspace=0.05, wspace=0.1)

    plt.savefig(figure_path)
    if show_plot:
        plt.show()


def plot_outcome_vs_rating(outcomes, rating_diffs, theoretical_outcomes, theoretical_rating_diffs, counts, figure_path: Path, dataset_name: str, metric_name: str, show_plot: bool=False, min_rating_diff: int=-850, max_rating_diff: int=850) -> None:
    curve_color = 'deepskyblue' # 'aquamarine'
    point_color = 'red'

    fontsize_l = 18
    fontsize_s = 14

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(theoretical_rating_diffs, theoretical_outcomes, label="Theoretical", color=curve_color, linewidth=4, zorder=1)
    plt.scatter(
        rating_diffs,
        outcomes,
        color=point_color,
        alpha=np.clip(0.9 * counts / counts.max() + 0.1, 0.1, 1),
        label="Empirical",
        zorder=2
    )

    plt.xlabel("Rating Difference (Model - Test Case)", fontsize=fontsize_l, fontweight="bold")
    plt.ylabel(f"Performance ({metric_name})", fontsize=fontsize_l, fontweight="bold")
    # plt.title(f"Observed vs. Theoretical Performance on {dataset_name}", fontsize=fontsize, fontweight="bold")

    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xticks(np.arange(min_rating_diff + 50, max_rating_diff - 49, 200), fontsize=fontsize_s)
    plt.yticks(np.linspace(0.0, 1.0, 11), fontsize=fontsize_s)

    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.axhline(0.5, color='gray', linestyle='--', linewidth=1)

    plt.legend(loc="upper left", fontsize=fontsize_l, frameon=True)
    plt.tight_layout()
    plt.savefig(figure_path, bbox_inches="tight")
    if show_plot:
        plt.show()

def plot_evaluation_trends(record_df: pd.DataFrame, save_path: str, dataset_name: str = None, show_plot: bool = True) -> None:
    x = record_df.index.values
    mae = record_df["MAE"].values
    mse = record_df["MSE"].values
    rho_test_case = record_df["Test Case rho"].values
    rho_model = record_df["Model rho"].values

    metrics = [("MAE", mae), ("MSE", mse), (r"$\rho_t$", rho_test_case), (r"$\rho_a$", rho_model), ]
    # Swap order of metrics
    # metric_keys = ["MAE", "MSE", "Test Case rho", "Model rho"]
    titles = ["MAE", "MSE", r"$\rho_t$", r"$\rho_a$"]
    # colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#ff7f0e"]
    colors = ['#F3CCDB', '#A8D1E1', '#62A9C8', '#147EBC'] # pink to blue '#E5E5F3',

    fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharex=True)
    for ax, (title, values), color in zip(axes, metrics, colors):
        ax.plot(x, values, marker="o", color=color, linewidth=2)
        # ax.set_title(title, fontsize=14)
        ax.set_xlabel("% of Matches", fontsize=20, fontweight="bold")
        ax.set_ylabel(title, fontsize=20, fontweight="bold")
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True, linestyle="--", alpha=0.6)

        # Legend in top right, slightly lower
        ax.legend(loc="upper right", fontsize=16)

        if title == titles[-1]:
            ax.set_ylim(0, 1)
            ax.legend(loc="lower right", fontsize=16)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show_plot:
        plt.show()
    plt.close()

def plot_evaluation_trends_summary(record_dfs: list[pd.DataFrame], labels: list[str], save_path: str, show_plot: bool = True) -> None:
    x = record_dfs[0].index.values  # Assume all DataFrames have same index

    # Swap order of metrics
    metric_keys = ["MAE", "MSE", "Test Case rho", "Model rho"]
    titles = ["MAE", "MSE", r"$\rho_t$", r"$\rho_a$"]

    # custom_colors = ['#33539E', '#7FACD6', '#BFB8DA', '#E8B7D4', '#A5678E'] # blue to red
    custom_colors = ['#F3CCDB', '#E5E5F3', '#A8D1E1', '#62A9C8', '#147EBC'] # pink to blue
    custom_cmap = LinearSegmentedColormap.from_list("custom_colors", custom_colors)
    colors = [custom_cmap(i) for i in np.linspace(0, 1, len(record_dfs))]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharex=True)

    for i, (ax, title) in enumerate(zip(axes, titles)):
        for j, (df, color) in enumerate(zip(record_dfs, colors)):
            values = df[metric_keys[i]].values
            label = labels[j]
            ax.plot(x, values, marker="o", color=color, linewidth=2, label=label)

        ax.set_xlabel("% of Matches", fontsize=20, fontweight="bold")
        ax.set_ylabel(title, fontsize=20, fontweight="bold")
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True, linestyle="--", alpha=0.6)

        # Legend in top right, slightly lower
        ax.legend(loc="upper right", fontsize=16)

        if title == titles[-1]:
            ax.set_ylim(0, 1)
            ax.legend(loc="lower right", fontsize=16)


    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show_plot:
        plt.show()
    plt.close()

def plot_model_rating_trends(rating_df: pd.DataFrame, save_path: str, dataset_name: str = None, show_plot: bool = True):
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(0, 101, 1)  # X-axis values from 0 to 100
    num_models = len(rating_df.columns)
    color_map = cm.get_cmap('tab20', num_models)

    for i, model_name in enumerate(rating_df.columns):
        ax.plot(x, rating_df[model_name], label=model_name,
                color=color_map(i), linewidth=2)

    # Axis labels
    ax.set_xlabel("% of Matches", fontsize=16, fontweight='bold')
    ax.set_ylabel("Rating", fontsize=16, fontweight='bold')

    # Center y-axis and grid around 1500
    min_rating = rating_df.min().min() - 400
    max_rating = rating_df.max().max()
    span = max(1500 - min_rating, max_rating - 1500)
    y_min = int((1500 - span) // 200) * 200
    y_max = int((1500 + span) // 200 + 1) * 200
    ax.set_ylim([y_min, y_max])
    y_ticks = np.arange(y_min, y_max + 1, 200)
    ax.set_yticks(y_ticks)

    # Ticks font size and bold
    ax.tick_params(axis='both', labelsize=14)
    # ax.set_xticklabels(ax.get_xticks(), fontweight='bold')
    # ax.set_yticklabels(ax.get_yticks(), fontweight='bold')

    ax.grid(True, which='major', axis='both', linestyle="--", alpha=0.5)
    ax.set_xlim([x.min(), x.max()])

    ax.legend(loc="lower center", ncol=3, frameon=False)

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show_plot:
        plt.show()
    plt.close()