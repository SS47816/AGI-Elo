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
from matplotlib.colors import Normalize, LinearSegmentedColormap

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

# Global font size settings
fontsize_l = 12  # Large font
fontsize_m = 10  # Medium font
fontsize_s = 8   # Small font

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
    color_hist = "#{:02x}{:02x}{:02x}".format(74, 98, 138)  # Dark blue
    color_curve = '#A5678E'  # Purple
    
    # Create a custom color scheme for models
    custom_red_colors = ['#DC3971', '#EC719F', '#F3B3CC', '#ABE5E8', '#34ADAE']
    custom_cmap = LinearSegmentedColormap.from_list("custom_red", custom_red_colors)
    # Convert numpy color array to hex string
    model_colors = []
    for i in np.linspace(0, 1, num_models):
        rgba = custom_cmap(i)
        # Convert RGBA to hex color string
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgba[0] * 255), 
            int(rgba[1] * 255), 
            int(rgba[2] * 255)
        )
        model_colors.append(hex_color)
    
    # Create gradient colors for histogram
    # Note: Matplotlib 3.7+ has deprecated cm.get_cmap
    # Alternative:
    # import matplotlib.pyplot as plt
    # cmap = plt.get_cmap("Blues") or
    # import matplotlib as mpl
    # cmap = mpl.colormaps["Blues"]
    cmap = cm.get_cmap("Blues")
    
    rating_start = (sorted_ratings.min() // 100) * 100
    rating_end = ((sorted_ratings.max() // 100) + 1) * 100 + 1
    rating_bins = np.arange(start=rating_start, stop=rating_end, step=bin_size)
    
    hist, edges = np.histogram(test_case_ratings, bins=rating_bins)
    
    # Recalculate bin_colors using correct bin_centers
    norm = Normalize(vmin=rating_start, vmax=rating_end)
    bin_centers = [(edges[i] + edges[i+1])/2 for i in range(len(edges)-1)]
    # Convert numpy color array to hex string
    bin_colors = []
    for center in bin_centers:
        rgba = cmap(norm(center))
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgba[0] * 255), 
            int(rgba[1] * 255), 
            int(rgba[2] * 255)
        )
        bin_colors.append(hex_color)
    
    # Use a more accurate cumulative percentage calculation - consistent with visualization.py
    cumulative_percent = np.arange(1, len(sorted_ratings) + 1) / len(sorted_ratings) * 100
    
    # Create Bokeh sources
    hist_source = ColumnDataSource(data=dict(
        top=hist,
        bottom=np.zeros(len(hist)),
        left=edges[:-1],
        right=edges[1:],
        color=bin_colors,
    ))
    
    # Use sorted original data points as x-axis, not bin edges
    cumulative_source = ColumnDataSource(data=dict(
        x=sorted_ratings,
        y=cumulative_percent
    ))
    
    model_x = []
    model_y = []
    model_names = []
    model_ratings = []
    
    for i, (name, rating) in enumerate(zip(models_df["Name"].values, models_df["Rating"].values)):
        color = model_colors[i]
        model_x.append(rating)
        model_y.append(max(hist) / 2)
        model_names.append(name)
        model_ratings.append(rating)
    
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
    
    # Set font and axis label styles, consistent with visualization.py
    p.xaxis.axis_label = "Rating"
    p.yaxis.axis_label = "Number of Test Cases"
    p.xaxis.axis_label_text_font_size = f"{fontsize_l}pt"
    p.xaxis.axis_label_text_font_style = "bold"
    p.yaxis.axis_label_text_font_size = f"{fontsize_l}pt"
    p.yaxis.axis_label_text_font_style = "bold"
    p.yaxis.axis_label_text_color = color_hist
    
    # Set tick label styles
    p.xaxis.major_label_text_font_size = f"{fontsize_l}pt"
    p.yaxis.major_label_text_font_size = f"{fontsize_l}pt"
    p.yaxis.major_label_text_color = color_hist
    
    # Add histogram
    fixed_legend_color = "#4A628A"  # Dark blue for Test Cases legend fixed color
    hist_renderer = p.quad(
        top='top', bottom='bottom', left='left', right='right',
        source=hist_source,
        fill_color='color', line_color="black", alpha=0.8,
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
    
    # Set cumulative percentage axis style, consistent with visualization.py
    p.yaxis[1].axis_label = "Cumulative Percentage"
    p.yaxis[1].axis_label_text_font_size = f"{fontsize_l}pt"
    p.yaxis[1].axis_label_text_font_style = "bold"
    p.yaxis[1].axis_label_text_color = color_curve
    p.yaxis[1].major_label_text_font_size = f"{fontsize_l}pt"
    p.yaxis[1].major_label_text_color = color_curve
    
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
    p.legend.label_text_font_size = f"{fontsize_m}pt"  # Use medium font size
    p.legend.click_policy = "hide"  # Make legend interactive - can hide entries
    
    # Toggle button
    toggle = Toggle(label="Show/Hide Legend", button_type="primary", active=True, width=120, height=25)
    toggle.js_link('active', p.legend[0], 'visible')

    # Title
    title_div = Div(text="<h2 style='margin: 0; text-align: center;'>Test Case & Model Rating Distributions</h3>",
                    width=400, height=30)

    # Left button + middle spacer + center title + right spacer for centering
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


def create_outcome_vs_rating_html(outcomes, rating_diffs, theoretical_outcomes, theoretical_rating_diffs, counts, output_html_path,
                                 min_rating_diff=-1250, max_rating_diff=1250):
    """
    Create an interactive HTML visualization of outcomes vs rating differences.
    
    Args:
        outcomes: Array of outcome values
        rating_diffs: Array of rating differences
        theoretical_outcomes: Array of theoretical outcome values
        theoretical_rating_diffs: Array of theoretical rating differences
        counts: Array of counts for each data point
        output_html_path: Path to save the HTML output
        min_rating_diff: Minimum rating difference to display (default: -1250)
        max_rating_diff: Maximum rating difference to display (default: 1250)
    """    
    # Get the correct calculation method from rating_system.py
    mask = ~np.isnan(rating_diffs) & ~np.isnan(outcomes)
    masked_counts = counts[mask]
    masked_rating_diffs = rating_diffs[mask]
    masked_outcomes = outcomes[mask]
    expected_outcomes = 1 / (1 + 10 ** (-masked_rating_diffs / 400))
    
    # Use sklearn functions to calculate metrics with weights
    mae = mean_absolute_error(expected_outcomes, masked_outcomes, sample_weight=masked_counts)
    mse = mean_squared_error(expected_outcomes, masked_outcomes, sample_weight=masked_counts)

    # Get correct colors and font sizes from visualization.py
    curve_color = 'deepskyblue'  # Consistent with visualization.py
    point_color = 'red'  # Consistent with visualization.py 
    label_color = '#000000'  # Black text
    
    # Use global font size settings to avoid conflicts
    # fontsize_l and fontsize_s are defined at the top of the file
    
    # Calculate alpha values, using the same method as visualization.py
    alpha_values = np.clip(0.9 * counts / counts.max() + 0.1, 0.1, 1)
    
    # Create Bokeh sources
    observed_source = ColumnDataSource(data=dict(
        x=rating_diffs,
        y=outcomes,
        expected=expected_outcomes,
        count=counts,
        alpha=alpha_values,
    ))
    
    theoretical_source = ColumnDataSource(data=dict(
        x=theoretical_rating_diffs,
        y=theoretical_outcomes
    ))
    
    # Ensure the range of theoretical values is used
    x_range = (min_rating_diff, max_rating_diff)
    
    # Create figure
    p = figure(
        x_axis_label="Rating Difference (Model - Test Case)",
        y_axis_label="Performance (Accuracy)",
        width=900,
        height=500,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        x_range=x_range,
        y_range=(-0.05, 1.05),
        sizing_mode="stretch_both"  # Make the figure stretch to fill available space
    )
    
    # Set font and axis label styles
    p.xaxis.axis_label = "Rating Difference (Model - Test Case)"
    p.yaxis.axis_label = "Performance (Accuracy)"
    p.xaxis.axis_label_text_font_size = f"{fontsize_l}pt"
    p.xaxis.axis_label_text_font_style = "bold"
    p.yaxis.axis_label_text_font_size = f"{fontsize_l}pt"
    p.yaxis.axis_label_text_font_style = "bold"
    
    # Set tick label styles
    p.xaxis.major_label_text_font_size = f"{fontsize_s}pt"
    p.yaxis.major_label_text_font_size = f"{fontsize_s}pt"
    
    # Set X-axis ticks, consistent with visualization.py - fix tick calculation
    # Ensure correct tick generation even if parameters change
    step = 200
    x_ticks = np.arange(
        min_rating_diff + 50, 
        max_rating_diff - 49, 
        step
    )
    p.xaxis.ticker = x_ticks
    
    # Set Y-axis ticks, consistent with visualization.py
    p.yaxis.ticker = np.linspace(0.0, 1.0, 11)
    
    # Add grid lines, consistent with visualization.py
    p.grid.grid_line_alpha = 0.5
    p.grid.grid_line_dash = "dashed"
    
    # Add theoretical curve - increase linewidth to match matplotlib
    p.line(
        x='x', y='y',
        source=theoretical_source,
        line_width=4,  # Increase line width to match visualization.py
        color=curve_color,
        legend_label="Theoretical"
    )
    
    # Add observed data points
    observed_points = p.scatter(
        x='x', y='y',
        source=observed_source,
        size=8,  # Increase point size for visibility
        alpha="alpha",  # Use calculated alpha values
        color=point_color,
        legend_label="Empirical"
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
    
    # Add reference lines, consistent with visualization.py
    p.line(x=[0, 0], y=[0, 1], line_color="gray", line_dash="dashed", line_width=1)
    p.line(x=[min_rating_diff, max_rating_diff], y=[0.5, 0.5], line_color="gray", line_dash="dashed", line_width=1)
    
    # Configure legend, consistent with visualization.py
    p.legend.location = "top_left"  # Bokeh uses top_left instead of upper left
    p.legend.background_fill_alpha = 0.8
    p.legend.label_text_font_size = f"{fontsize_l}pt"
    p.legend.border_line_alpha = 0.5  # Slight border
    
    # Toggle button
    toggle = Toggle(label="Show/Hide Legend", button_type="primary", active=True, width=120, height=25)
    toggle.js_link('active', p.legend[0], 'visible')

    # Title includes accuracy metrics
    title_text = f"Performance Prediction VS. Reality: MAE={mae:.4f}, MSE={mse:.4f}"
    title_div = Div(text=f"<h2 style='margin: 0; text-align: center;'>{title_text}</h2>",
                    width=600, height=30)

    # Left button + middle spacer + center title + right spacer for centering
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
    parser.add_argument('--min_rating_diff', type=int, default=-1250,
                        help='Minimum rating difference (default: -1250)')
    parser.add_argument('--max_rating_diff', type=int, default=1250,
                        help='Maximum rating difference (default: 1250)')
    
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
        theoretical_rating_diffs = np.linspace(args.min_rating_diff, args.max_rating_diff, 1000)
        theoretical_outcomes = 1 / (1 + 10 ** (-theoretical_rating_diffs / 400))
        outcomes = 1 / (1 + 10 ** (-rating_diffs / 400)) + np.random.normal(0, 0.1, len(rating_diffs))
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
        
        # Calculate theoretical value curve, consistent with rating_system.py
        theoretical_rating_diffs = np.linspace(args.min_rating_diff, args.max_rating_diff, 1000)
        theoretical_outcomes = 1 / (1 + 10 ** (-theoretical_rating_diffs / 400))
        
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
        theoretical_outcomes,
        theoretical_rating_diffs,
        counts,
        outcome_html_path,
        min_rating_diff=args.min_rating_diff, 
        max_rating_diff=args.max_rating_diff
    )
    
    print("\nVisualization files created successfully!")


if __name__ == "__main__":
    main()
