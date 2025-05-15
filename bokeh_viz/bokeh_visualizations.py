#!/usr/bin/env python3
"""
Generate interactive Bokeh visualizations as standalone HTML files

This script creates two interactive visualizations:
1. Rating histogram with model ratings
2. Outcome vs rating difference plot

Both are saved as standalone HTML files.
"""

import argparse
from pathlib import Path
import matplotlib.cm as cm

import numpy as np
import pandas as pd
from bokeh.plotting import figure, save, output_file
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    Range1d,
    Toggle,
    BoxAnnotation,
    CustomJS,
    Div
)
from bokeh.layouts import column, row, Spacer
from bokeh.palettes import Category20
from bokeh.plotting import curdoc
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm


def create_ratings_html(test_case_ratings_path, model_ratings_path, output_html_path,
                       bin_size=100, left_margin=200, right_margin=100):
    """
    Create an interactive HTML visualization of test case and model ratings.
    Uses plot_ratings for data processing and adds Bokeh interactivity.

    Args:
        test_case_ratings_path: Path to the test case ratings pickle file
        model_ratings_path: Path to the model ratings pickle file
        output_html_path: Path to save the HTML output
        bin_size: Size of histogram bins
        left_margin: Left margin for x-axis
        right_margin: Right margin for x-axis
    """
    # Load data directly
    test_cases_df = pd.read_pickle(test_case_ratings_path)
    models_df = pd.read_pickle(model_ratings_path)
    num_models = models_df.shape[0]
    
    # Get test case ratings
    test_case_ratings = test_cases_df["Rating"].values
    sorted_ratings = np.sort(test_case_ratings)
    
    # Pick colors
    color_hist = '#EF7C00'  # NUS橙色
    color_curve = '#003D7C'  # NUS蓝色
    model_colors = cm.get_cmap('Paired', num_models).colors

    # Calculate histogram data first
    rating_start = (sorted_ratings.min() // 100) * 100
    rating_end = ((sorted_ratings.max() // 100) + 1) * 100 + 1
    rating_bins = np.arange(start=rating_start, stop=rating_end, step=bin_size)
    
    hist, edges = np.histogram(test_case_ratings, bins=rating_bins)
    
    # 使用更精确的累积百分比计算方法 - 与visualization.py一致
    cumulative_percent = np.arange(1, len(sorted_ratings) + 1) / len(sorted_ratings) * 100
    
    # Create Bokeh sources
    hist_source = ColumnDataSource(data=dict(
        top=hist,
        bottom=np.zeros(len(hist)),
        left=edges[:-1],
        right=edges[1:],
    ))
    
    # 使用排序后的原始数据点作为x轴，而非bin边界
    cumulative_source = ColumnDataSource(data=dict(
        x=sorted_ratings,
        y=cumulative_percent
    ))
    
    model_x = []
    model_y = []
    model_names = []
    model_ratings = []
    model_colors = []
    
    for i, (name, rating) in enumerate(zip(models_df["Name"].values, models_df["Rating"].values)):
        color = Category20[20][i % 20]
        model_x.append(rating)
        model_y.append(max(hist) / 2)
        model_names.append(name)
        model_ratings.append(rating)
        model_colors.append(color)
    
    # Create a single source for all models
    combined_model_source = ColumnDataSource(data=dict(
        x=model_x,
        y=model_y,
        name=model_names,
        rating=model_ratings,
        color=model_colors
    ))
    
    # Create Bokeh figure
    p = figure(
        x_axis_label="Rating",
        y_axis_label="Number of Test Cases",
        width=900,
        height=500,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        x_range=(rating_start - left_margin, rating_end + right_margin),
        y_range=(0, max(hist) * 1.1),  # Proper y-range for histogram
        output_backend="canvas",  # Change from webgl to canvas
        sizing_mode="stretch_both"  # Make the figure stretch to fill available space
    )
    
    # Add histogram
    hist_renderer = p.quad(
        top='top', bottom='bottom', left='left', right='right',
        source=hist_source,
        fill_color=color_hist, line_color="black", alpha=0.8,
        legend_label="Test Cases"
    )
    
    p.add_tools(HoverTool(
        renderers=[hist_renderer],
        tooltips=[
            ("Range", "@left - @right"),
            ("Count", "@top"),
        ]
    ))
    
    # Add cumulative line with proper y-range
    p.extra_y_ranges = {"cumulative": Range1d(start=-5, end=105)}
    
    cumulative_line = p.line(
        x='x', y='y',
        source=cumulative_source,
        y_range_name="cumulative",
        line_width=2,
        color=color_curve,
        legend_label="Cumulative %"
    )
    
    p.add_tools(HoverTool(
        renderers=[cumulative_line],
        tooltips=[
            ("Rating", "@x"),
            ("Cumulative %", "@y{0.0}%"),
        ]
    ))
    
    # Add axis for cumulative percentage - ensure proper alignment
    right_axis = p.yaxis[0].clone(y_range_name="cumulative")
    p.add_layout(right_axis, 'right')
    p.yaxis[0].axis_label_text_color = "black"  # 主y轴标签颜色
    p.yaxis[1].axis_label = "Cumulative Percentage"
    p.yaxis[1].axis_label_text_color = "black"  # 次y轴标签颜色
    p.yaxis[1].major_label_text_color = "black"  # 次y轴刻度标签颜色
    
    # Format tick marks to avoid scientific notation
    p.yaxis[0].formatter.use_scientific = False
    
    # Calculate the mapping factor between histogram y-scale and percentage y-scale
    # This ensures model lines extend to the full height (100%) on the right axis
    y_mapping_factor = 100 / (max(hist) * 1.1)
    
    # Create a function to interpolate cumulative percentage at any rating
    def get_cumulative_pct(rating):
        # Find indices where rating falls between two points
        if rating <= cumulative_source.data['x'][0]:
            return cumulative_source.data['y'][0]
        if rating >= cumulative_source.data['x'][-1]:
            return cumulative_source.data['y'][-1]
        
        # Find the two points to interpolate between
        idx = np.searchsorted(cumulative_source.data['x'], rating) - 1
        x1, y1 = cumulative_source.data['x'][idx], cumulative_source.data['y'][idx]
        x2, y2 = cumulative_source.data['x'][idx + 1], cumulative_source.data['y'][idx + 1]
        
        # Linear interpolation
        return y1 + (y2 - y1) * (rating - x1) / (x2 - x1)
    
    # Add vertical lines for all models at once with intersection points
    model_intersections = []
    for i, (rating, color, name) in enumerate(zip(model_x, model_colors, model_names)):
        # Calculate exact cumulative percentage at this rating
        cumulative_pct = get_cumulative_pct(rating)
        
        # Add vertical line
        p.line(
            x=[rating, rating],
            y=[0, 100 / y_mapping_factor],
            line_color=color,
            line_width=2,
            line_dash="dashed",
            legend_label=name
        )
        
        # Add intersection point with cumulative curve
        intersection = p.scatter(
            x=[rating], 
            y=[cumulative_pct], 
            size=8, 
            color=color,
            y_range_name="cumulative"
        )
        model_intersections.append(intersection)
        
        # Add hover tool for intersection point
        p.add_tools(HoverTool(
            renderers=[intersection],
            tooltips=[
                ("Model", name),
                ("Rating", f"{rating:.1f}"),
                ("Cumulative %", "@y{0.1f}%"),
            ],
            mode="mouse"
        ))
    
    # Remove the original model markers and hover tool
    # (Don't add the model_markers and its HoverTool from the original code) 
    
    # Legend settings
    p.legend.location = "top_left"
    p.legend.background_fill_alpha = 0.7
    p.legend.ncols = 2  # Use two columns for the legend
    p.legend.label_text_font_size = "8pt"  # Smaller font for legends
    p.legend.click_policy = "hide"  # Make legend interactive - can hide entries
    
    # Toggle按钮
    toggle = Toggle(label="Show/Hide Legend", button_type="primary", active=True, width=120, height=25)
    toggle.js_link('active', p.legend[0], 'visible')

    # 标题
    title_div = Div(text="<h2 style='margin: 0; text-align: center;'>Test Case & Model Rating Distributions</h3>",
                    width=400, height=30)

    # 左边按钮 + 中间 spacer + 中间标题 + 右边 spacer 实现居中
    header = row(
        toggle,
        Spacer(width=20),
        title_div,
        sizing_mode="stretch_width",
        height=35
    )

    layout = column(
        header,
        p,
        sizing_mode="stretch_both"
    )

    curdoc().add_root(layout)
    
    # Save the HTML file
    output_file(output_html_path, title="Rating Distributions")
    save(layout)


def create_outcome_vs_rating_html(outcomes, rating_diffs, counts, output_html_path,
                                 min_rating_diff=-850, max_rating_diff=850):
    """
    Create an interactive HTML visualization of outcomes vs rating differences.
    
    Args:
        outcomes: Array of outcome values
        rating_diffs: Array of rating differences
        counts: Array of counts for each data point
        output_html_path: Path to save the HTML output
        min_rating_diff: Minimum rating difference to display
        max_rating_diff: Maximum rating difference to display
    """

    # Theoretical Elo curve
    theoretical_rating_diffs = np.linspace(min_rating_diff, max_rating_diff, 1000)
    theoretical_outcomes = 1 / (1 + 10 ** (-theoretical_rating_diffs / 400))

    # Compute correlation R
    mask = ~np.isnan(rating_diffs) & ~np.isnan(outcomes)

    counts = counts[mask]
    rating_diffs = rating_diffs[mask]
    outcomes = outcomes[mask]
    expected_outcomes = 1 / (1 + 10 ** (-rating_diffs / 400))

    mse = mean_squared_error(expected_outcomes, outcomes)
    r2 = r2_score(expected_outcomes, outcomes)

    curve_color = '#003D7C'  # NUS蓝色
    point_color = '#EF7C00'  # NUS橙色
    label_color = '#000000'  # 黑色
    
    # Create Bokeh sources
    observed_source = ColumnDataSource(data=dict(
        x=rating_diffs,
        y=outcomes,
        expected=expected_outcomes,
        count=counts,
        alpha=0.9*counts/counts.max()+0.1,
    ))
    
    theoretical_source = ColumnDataSource(data=dict(
        x=theoretical_rating_diffs,
        y=theoretical_outcomes
    ))
    
    # Create figure
    p = figure(
        x_axis_label="Rating Difference (Model - Test Case)",
        y_axis_label="Model Score (Accuracy)",
        width=900,
        height=500,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        x_range=(min_rating_diff, max_rating_diff),
        y_range=(-0.05, 1.05),
        sizing_mode="stretch_both"  # Make the figure stretch to fill available space
    )
    
    # Add theoretical curve
    p.line(
        x='x', y='y',
        source=theoretical_source,
        line_width=2,
        color=curve_color,
        legend_label="Theoretical Curve"
    )
    
    # Add observed data points
    observed_points = p.scatter(
        x='x', y='y',
        source=observed_source,
        size=6,
        alpha="alpha",
        color=point_color,
        legend_label="Observed Data"
    )
    
    p.add_tools(HoverTool(
        renderers=[observed_points],
        tooltips=[
            ("Rating Diff", "@x{0.0}"),
            ("Observed Score", "@y{0.000}"),
            ("Expected Score", "@expected{0.000}"),
            ("Count", "@count"),
        ]
    ))
    
    # Calculate MAE and R²
    mae = mean_absolute_error(expected_outcomes, outcomes)
    r2 = r2_score(expected_outcomes, outcomes)
    
    # Add metrics text - moved to bottom right corner
    p.text(max_rating_diff - 350, 0.15, [f"MAE = {mae:.3f}"], text_font_size="12pt", text_color=label_color)
    p.text(max_rating_diff - 350, 0.10, [f"R² = {r2:.3f}"], text_font_size="12pt", text_color=label_color)
    
    # # Add reference lines
    # p.line(x=[0, 0], y=[0, 1], line_color="gray", line_dash="dashed", line_width=1)
    # p.line(x=[min_rating_diff, max_rating_diff], y=[0.5, 0.5], line_color="gray", line_dash="dashed", line_width=1)
    
    # Configure legend
    p.legend.location = "top_left"
    p.legend.background_fill_alpha = 0.8
    
    # # Add grid
    # p.grid.grid_line_alpha = 0.3
    
        # Toggle按钮
    toggle = Toggle(label="Show/Hide Legend", button_type="primary", active=True, width=120, height=25)
    toggle.js_link('active', p.legend[0], 'visible')

    # 标题
    title_div = Div(text="<h2 style='margin: 0; text-align: center;'>Test Case & Model Rating Distributions</h3>",
                    width=400, height=30)

    # 左边按钮 + 中间 spacer + 中间标题 + 右边 spacer 实现居中
    header = row(
        toggle,
        Spacer(width=20),
        title_div,
        sizing_mode="stretch_width",
        height=35
    )

    layout = column(
        header,
        p,
        sizing_mode="stretch_both"
    )

    curdoc().add_root(layout)
    
    # Save the HTML file
    output_file(output_html_path, title="Outcome vs Rating")
    save(layout)


def main():
    """Main function to generate the HTML visualizations."""
    parser = argparse.ArgumentParser(description='Generate interactive Bokeh visualizations')
    
    # Common arguments
    parser.add_argument('--output_dir', type=str, default='bokeh_html',
                        help='Directory to save HTML output')
    
    # Rating visualization arguments
    parser.add_argument('--test_case_ratings', type=str, required=True,
                        help='Path to test case ratings pickle file')
    parser.add_argument('--model_ratings', type=str, required=True,
                        help='Path to model ratings pickle file')
    parser.add_argument('--bin_size', type=int, default=100,
                        help='Size of histogram bins (default: 100)')
    parser.add_argument('--match_results', type=str, required=True,
                        help='Path to match results folder')
    parser.add_argument('--min_rating_diff', type=int, default=-850,
                        help='Minimum rating difference (default: -850)')
    parser.add_argument('--max_rating_diff', type=int, default=850,
                        help='Maximum rating difference (default: 850)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate ratings visualization
    ratings_html_path = output_dir / "ratings_visualization.html"
    print(f"Creating ratings visualization at {ratings_html_path}")
    create_ratings_html(
        args.test_case_ratings,
        args.model_ratings,
        ratings_html_path,
        bin_size=args.bin_size
    )
    
    # Generate outcome vs rating visualization
    outcome_html_path = output_dir / "outcome_vs_rating.html"
    print(f"Creating outcome vs rating visualization at {outcome_html_path}")
    
    # Load test case and model ratings
    test_case_df = pd.read_pickle(args.test_case_ratings)
    model_df = pd.read_pickle(args.model_ratings)
    
    # Load match results
    match_results_path = Path(args.match_results)
    pkl_files = sorted(match_results_path.glob("*.pkl"))
    
    if not pkl_files:
        print("No match result files found. Generating random data for demonstration.")
        # Generate sample data if match results are not provided
        rating_diffs = np.linspace(-800, 800, 50)
        rating_diffs = np.concatenate([rating_diffs + np.random.normal(0, 10, len(rating_diffs)) for _ in range(10)])
        theoretical_outcomes = 1 / (1 + 10 ** (-rating_diffs / 400))
        outcomes = theoretical_outcomes + np.random.normal(0, 0.1, len(rating_diffs))
        outcomes = np.clip(outcomes, 0, 1)
        counts = np.random.randint(1, 100, len(rating_diffs))
    else:
        print(f"Processing {len(pkl_files)} match result files...")
        # Load and process match results
        dfs = []
        for file in tqdm(pkl_files, desc="Loading pickle files"):
            dfs.append(pd.read_pickle(file))
        df = pd.concat(dfs, ignore_index=True)
        
        # Set up test case and model ratings for lookup
        test_case_df = test_case_df.set_index("Name")
        model_df = model_df.set_index("Name")
        
        # Build pivot tables for outcomes and rating differences
        print("Building pivot tables...")
        pivot_df_outcome = df.pivot_table(index="Test Case", columns="Model", values="Model Score")
        pivot_df_outcome = pivot_df_outcome.sort_index().sort_index(axis=1)
        
        # Calculate rating differences between models and test cases
        print("Calculating rating differences...")
        rating_diff = model_df["Rating"].values[None, :] - test_case_df["Rating"].values[:, None]
        pivot_df_rating_diff = pd.DataFrame(
            rating_diff,
            index=test_case_df.index,
            columns=model_df.index
        )
        pivot_df_rating_diff = pivot_df_rating_diff.sort_index().sort_index(axis=1)
        
        # Define bins for rating differences
        bin_edges = np.arange(args.min_rating_diff, args.max_rating_diff, args.bin_size)
        bin_labels = [f"[{bin_edges[i]},{bin_edges[i+1]})" for i in range(len(bin_edges)-1)]
        
        # Collect binned statistics
        binned_counts = {}
        binned_avg_scores = {}
        binned_avg_ratings = {}
        
        # Process each model
        for model in tqdm(pivot_df_outcome.columns, desc="Processing models"):
            rating_diffs = pivot_df_rating_diff[model]
            model_scores = pivot_df_outcome[model]
            
            # Bin the rating differences
            bins = pd.cut(rating_diffs, bins=bin_edges, labels=bin_labels, right=False)
            
            # Group scores by bin and calculate statistics
            bin_count = model_scores.groupby(bins, observed=False).count()
            bin_avg_scores = model_scores.groupby(bins, observed=False).mean()
            bin_avg_ratings = rating_diffs.groupby(bins, observed=False).mean()
            
            # Ensure all bins are present
            bin_count = bin_count.reindex(bin_labels)
            bin_avg_scores = bin_avg_scores.reindex(bin_labels)
            bin_avg_ratings = bin_avg_ratings.reindex(bin_labels)
            
            binned_counts[model] = bin_count
            binned_avg_scores[model] = bin_avg_scores
            binned_avg_ratings[model] = bin_avg_ratings
        
        # Combine into DataFrames
        binned_counts_df = pd.DataFrame(binned_counts)
        binned_avg_scores_df = pd.DataFrame(binned_avg_scores)
        binned_avg_ratings_df = pd.DataFrame(binned_avg_ratings)
        
        # Flatten the arrays for plotting
        outcomes = binned_avg_scores_df.values.flatten()
        rating_diffs = binned_avg_ratings_df.values.flatten()
        counts = binned_counts_df.values.flatten()
        
        # Remove NaN values
        mask = ~np.isnan(rating_diffs) & ~np.isnan(outcomes)
        rating_diffs = rating_diffs[mask]
        outcomes = outcomes[mask]
        counts = counts[mask]
    
    # Create the visualization
    create_outcome_vs_rating_html(
        outcomes,
        rating_diffs,
        counts,
        outcome_html_path,
        min_rating_diff=args.min_rating_diff, 
        max_rating_diff=args.max_rating_diff
    )
    
    print("\nVisualization files created successfully!")


if __name__ == "__main__":
    main()
