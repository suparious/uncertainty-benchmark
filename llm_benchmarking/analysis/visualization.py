"""
Visualization module for LLM Uncertainty Benchmarking.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Union, Tuple
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

from ..utils.logging import get_logger
from ..utils.visualization import (
    setup_plot_style,
    create_figure,
    save_figure,
    prettify_model_name,
    create_colormap,
    add_labels_to_bars
)
from .reporting import generate_report

logger = get_logger(__name__)


def visualize_results(
    results: Dict[str, Dict[str, Any]],
    model_names: List[str] = None,
    output_file: Optional[str] = None,
    show: bool = True
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Visualize the benchmark results.
    
    Args:
        results: Dictionary with benchmark results
        model_names: List of model names to include. If None, include all.
        output_file: Path to save the visualization. If None, don't save.
        show: Whether to display the figures
        
    Returns:
        Tuple of (main_figure, task_figure)
    """
    # Set up plot style
    setup_plot_style()
    
    if model_names is None:
        model_names = list(results.keys())
    
    # Get report DataFrame
    report_df = generate_report(results, model_names)
    
    # Filter for average results
    avg_results = report_df[report_df['Task'] == 'Average'].copy()
    
    # Sort by accuracy for consistent coloring
    avg_results = avg_results.sort_values('Accuracy', ascending=False)
    
    # Create a custom colormap based on model count
    model_colors = create_colormap(len(avg_results), "viridis")
    color_dict = {model: color for model, color in zip(avg_results['Model'], model_colors)}
    
    # Create figure with 2 subplots
    main_fig, (ax1, ax2) = create_figure(1, 2, figsize=(18, 9))
    
    # Accuracy vs. Set Size plot
    for i, (_, row) in enumerate(avg_results.iterrows()):
        model = row['Model']
        ax1.scatter(
            row['Accuracy'], 
            row['Set Size'], 
            s=200, 
            color=color_dict[model],
            alpha=0.8,
            edgecolors='white',
            linewidth=1.5,
            label=prettify_model_name(model)
        )
    
    # Add annotations with improved positioning
    for i, row in avg_results.iterrows():
        offset_x = 0.5
        offset_y = 0.05 * (i % 3) + 0.05  # Stagger labels vertically
        ax1.annotate(
            prettify_model_name(row['Model']),
            (row['Accuracy'] + offset_x, row['Set Size'] + offset_y),
            fontsize=9,
            alpha=0.9,
            weight='bold'
        )
    
    ax1.set_title('Accuracy vs. Uncertainty (Set Size)', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Set Size (lower is better)', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_facecolor('#f8f9fa')
    
    # Coverage Rate vs. Set Size plot
    for i, (_, row) in enumerate(avg_results.iterrows()):
        model = row['Model']
        ax2.scatter(
            row['Coverage Rate'], 
            row['Set Size'], 
            s=200, 
            color=color_dict[model],
            alpha=0.8,
            edgecolors='white',
            linewidth=1.5,
            label=prettify_model_name(model)
        )
    
    # Add annotations with improved positioning
    for i, row in avg_results.iterrows():
        offset_x = 0.5
        offset_y = 0.05 * (i % 3) + 0.05  # Stagger labels vertically
        ax2.annotate(
            prettify_model_name(row['Model']),
            (row['Coverage Rate'] + offset_x, row['Set Size'] + offset_y),
            fontsize=9,
            alpha=0.9,
            weight='bold'
        )
    
    ax2.set_title('Coverage Rate vs. Uncertainty (Set Size)', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Coverage Rate (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Set Size (lower is better)', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_facecolor('#f8f9fa')
    
    # Add main title
    main_fig.suptitle('LLM Benchmark Results: Performance Overview', fontsize=20, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, wspace=0.2)
    
    # Save main figure if output_file is provided
    if output_file:
        save_figure(main_fig, output_file)
    
    # Show main figure if show is True
    if show:
        plt.show()
    else:
        plt.close(main_fig)
    
    # Create task-specific visualizations
    task_list = [task for task in report_df['Task'].unique() if task != 'Average']
    
    # Dynamic figure sizing based on number of tasks
    task_fig_height = max(6, 3 * len(task_list))
    task_fig, axes = create_figure(len(task_list), 1, figsize=(14, task_fig_height))
    
    if len(task_list) == 1:
        axes = [axes]  # Make sure axes is always a list
    
    # Create a consistent color map for models across all task plots
    all_models = report_df['Model'].unique()
    all_model_colors = create_colormap(len(all_models), "tab20")
    full_color_dict = {model: color for model, color in zip(all_models, all_model_colors)}
    
    for i, task in enumerate(task_list):
        task_df = report_df[report_df['Task'] == task].copy()
        
        ax = axes[i]
        
        # Sort by accuracy (descending)
        task_df = task_df.sort_values('Accuracy', ascending=False)
        
        # Set up x-axis
        x = np.arange(len(task_df))
        bar_width = 0.35
        
        # Get prettier model names for display
        model_names_display = [prettify_model_name(m) for m in task_df['Model']]
        
        # Plot accuracy bars
        accuracy_bars = ax.bar(
            x - bar_width/2, 
            task_df['Accuracy'], 
            bar_width, 
            label='Accuracy (%)',
            color='royalblue',
            alpha=0.8,
            edgecolor='white',
            linewidth=1
        )
        
        # Add accuracy value labels
        for j, bar in enumerate(accuracy_bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + 1.5,  # Offset above the bar
                f"{height:.1f}%",
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold',
                color='royalblue'
            )
        
        # Create twin axis for set size
        ax2 = ax.twinx()
        
        # Plot set size bars
        set_size_bars = ax2.bar(
            x + bar_width/2, 
            task_df['Set Size'], 
            bar_width, 
            label='Set Size',
            color='darkorange',
            alpha=0.8,
            edgecolor='white',
            linewidth=1
        )
        
        # Add set size value labels
        for j, bar in enumerate(set_size_bars):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.1,  # Offset above the bar
                f"{height:.1f}",
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold',
                color='darkorange'
            )
        
        # Set labels and title
        ax.set_title(f'Task: {task}', fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Models', fontsize=13, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold', color='royalblue')
        ax2.set_ylabel('Set Size', fontsize=13, fontweight='bold', color='darkorange')
        
        # Set colors for axis labels
        ax.tick_params(axis='y', colors='royalblue')
        ax2.tick_params(axis='y', colors='darkorange')
        
        # Set x-ticks and improve readability
        ax.set_xticks(x)
        ax.set_xticklabels(model_names_display, rotation=45, ha='right', fontsize=10)
        
        # Set y-axis limits with padding
        max_acc = max(task_df['Accuracy']) * 1.1
        max_ss = max(task_df['Set Size']) * 1.15
        ax.set_ylim(0, max_acc)
        ax2.set_ylim(0, max_ss)
        
        # Add grid for accuracy axis only
        ax.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.set_axisbelow(True)  # Place grid behind bars
        
        # Add subtle background
        ax.set_facecolor('#f8f9fa')
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend = ax.legend(
            lines1 + lines2, 
            labels1 + labels2, 
            loc='upper right', 
            frameon=True,
            framealpha=0.9,
            facecolor='white',
            edgecolor='lightgray'
        )
    
    # Add a main title
    task_fig.suptitle('LLM Benchmark Results: Task Breakdown', fontsize=20, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.4)
    
    # Save task figure if output_file is provided
    if output_file:
        # Create a properly named file for task visualizations
        task_output = output_file.replace('.', '_tasks.')
        save_figure(task_fig, task_output)
    
    # Show task figure if show is True
    if show:
        plt.show()
    else:
        plt.close(task_fig)
    
    return main_fig, task_fig


def visualize_task_comparisons(
    results: Dict[str, Dict[str, Any]],
    model_names: List[str] = None,
    output_file: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize model performance across different tasks.
    
    Args:
        results: Dictionary with benchmark results
        model_names: List of model names to include. If None, include all.
        output_file: Path to save the visualization. If None, don't save.
        show: Whether to display the figure
        
    Returns:
        Matplotlib figure
    """
    # Set up plot style
    setup_plot_style()
    
    if model_names is None:
        model_names = list(results.keys())
    
    # Get report DataFrame
    report_df = generate_report(results, model_names)
    
    # Filter out average results
    task_df = report_df[report_df['Task'] != 'Average'].copy()
    
    # Create prettier model names for display
    pretty_names = {model: prettify_model_name(model) for model in task_df['Model'].unique()}
    task_df['Display_Name'] = task_df['Model'].map(pretty_names)
    
    # Sort models by their average accuracy
    avg_acc = report_df[report_df['Task'] == 'Average'].set_index('Model')['Accuracy']
    models_sorted = avg_acc.sort_values(ascending=False).index.tolist()
    model_order = {model: i for i, model in enumerate(models_sorted)}
    task_df['Model_Order'] = task_df['Model'].map(model_order)
    
    # Sort the dataframe by model order
    task_df = task_df.sort_values('Model_Order')
    
    # Pivot the data for the heatmap
    pivot_df = task_df.pivot(index='Display_Name', columns='Task', values='Accuracy')
    
    # Create a figure with appropriate size based on model count
    fig_height = max(8, 0.5 * len(pivot_df))
    fig, ax = create_figure(figsize=(14, fig_height))
    
    # Create a visually appealing colormap
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    
    # Create the heatmap with improved formatting
    heatmap = sns.heatmap(
        pivot_df,
        annot=True,
        fmt='.1f',
        cmap=cmap,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Accuracy (%)', 'shrink': 0.8},
        ax=ax
    )
    
    # Improve annotations
    for text in heatmap.texts:
        if float(text.get_text()) > 90:
            text.set_weight('bold')
            text.set_color('white')
        else:
            text.set_color('black')
    
    # Set title and labels
    ax.set_title('Model Performance Across Tasks (Accuracy %)', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Models', fontsize=14, fontweight='bold')
    ax.set_xlabel('Tasks', fontsize=14, fontweight='bold')
    
    # Adjust y-axis tick parameters
    plt.yticks(rotation=0, fontsize=11)
    plt.xticks(rotation=30, ha='right', fontsize=11)
    
    # Add a grid
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output_file is provided
    if output_file:
        save_figure(fig, output_file)
    
    # Show figure if show is True
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def visualize_model_scaling(
    results: Dict[str, Dict[str, Any]],
    model_family: str,
    model_sizes: List[str],
    output_file: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize how model performance scales with size.
    
    Args:
        results: Dictionary with benchmark results
        model_family: Model family name (e.g., "llama")
        model_sizes: List of model sizes to include (e.g., ["7B", "13B", "70B"])
        output_file: Path to save the visualization. If None, don't save.
        show: Whether to display the figure
        
    Returns:
        Matplotlib figure
    """
    # Set up plot style
    setup_plot_style()
    
    # Construct model names
    model_names = [f"{model_family}-{size}" for size in model_sizes]
    
    # Check if we have results for all models
    available_models = [model for model in model_names if model in results]
    
    if not available_models:
        logger.warning(f"No results for any models in family {model_family}.")
        return None
    
    if len(available_models) < len(model_names):
        missing = set(model_names) - set(available_models)
        logger.warning(f"Missing results for models: {', '.join(missing)}")
    
    # Get report DataFrame for available models
    report_df = generate_report(results, available_models)
    
    # Filter for average results
    avg_results = report_df[report_df['Task'] == 'Average'].copy()
    
    # Sort by model size and create a proper categorical size column
    avg_results['Size'] = avg_results['Model'].apply(lambda x: x.split('-')[-1])
    avg_results['Size'] = pd.Categorical(avg_results['Size'], categories=model_sizes, ordered=True)
    avg_results = avg_results.sort_values('Size')
    
    # Create figure with 3 subplots
    fig, axes = create_figure(1, 3, figsize=(18, 6))
    
    metrics = ['Accuracy', 'Coverage Rate', 'Set Size']
    colors = ['#4285F4', '#34A853', '#FBBC05']
    
    for i, (metric, ax, color) in enumerate(zip(metrics, axes, colors)):
        # Create line plot with improved styling
        ax.plot(
            avg_results['Size'], 
            avg_results[metric], 
            marker='o', 
            markersize=10, 
            linewidth=2,
            color=color,
            markerfacecolor='white',
            markeredgewidth=2,
            markeredgecolor=color
        )
        
        # Add value labels
        for x, y in zip(avg_results['Size'], avg_results[metric]):
            ax.text(
                x, 
                y * 1.03, 
                f"{y:.1f}", 
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold',
                color=color
            )
        
        # Set labels and title
        ax.set_xlabel('Model Size', fontsize=13, fontweight='bold')
        y_label = metric
        if metric in ['Accuracy', 'Coverage Rate']:
            y_label += ' (%)'
        ax.set_ylabel(y_label, fontsize=13, fontweight='bold')
        ax.set_title(f'Effect of Model Scale on {metric}', fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set background color
        ax.set_facecolor('#f8f9fa')
    
    # Add global title
    plt.suptitle(
        f"Scaling Analysis for {model_family.upper()} Models", 
        fontsize=16, 
        fontweight='bold', 
        y=0.95
    )
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save figure if output_file is provided
    if output_file:
        save_figure(fig, output_file)
    
    # Show figure if show is True
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def visualize_prompt_strategies(
    results: Dict[str, Dict[str, Any]],
    model_name: str,
    output_file: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize the effectiveness of different prompt strategies for a model.
    
    Args:
        results: Dictionary with benchmark results
        model_name: Name of the model to analyze
        output_file: Path to save the visualization. If None, don't save.
        show: Whether to display the figure
        
    Returns:
        Matplotlib figure
    """
    # Set up plot style
    setup_plot_style()
    
    if model_name not in results:
        logger.warning(f"No results for model {model_name}. Cannot generate visualization.")
        return None
    
    model_results = results[model_name]
    
    # Collect data for visualization
    data = []
    
    for task, task_results in model_results.items():
        if task == 'overall':
            continue
        
        if 'prompt_strategies' not in task_results:
            logger.warning(f"No prompt strategy data for task {task}. Skipping.")
            continue
        
        for strategy, strategy_results in task_results['prompt_strategies'].items():
            data.append({
                'Task': task,
                'Prompt Strategy': strategy,
                'Accuracy': strategy_results['avg']['acc'] * 100,
                'Coverage Rate': strategy_results['avg']['cr'] * 100,
                'Set Size': strategy_results['avg']['ss']
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Check if we have data
    if len(df) == 0:
        logger.warning(f"No prompt strategy data available for model {model_name}.")
        return None
    
    # Get unique tasks and strategies
    tasks = df['Task'].unique()
    strategies = df['Prompt Strategy'].unique()
    
    # Create a figure with subplots for each metric
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
    
    metrics = ['Accuracy', 'Coverage Rate', 'Set Size']
    titles = [
        'Accuracy (%) by Task and Prompt Strategy',
        'Coverage Rate (%) by Task and Prompt Strategy',
        'Set Size by Task and Prompt Strategy'
    ]
    
    # Create a color palette for prompt strategies
    strategy_colors = create_colormap(len(strategies), "colorblind")
    color_dict = {strategy: color for strategy, color in zip(strategies, strategy_colors)}
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = fig.add_subplot(gs[i])
        
        # Create grouped bar chart
        pivot = df.pivot(index='Task', columns='Prompt Strategy', values=metric)
        
        # Set up bar positions
        x = np.arange(len(pivot.index))
        bar_width = 0.8 / len(pivot.columns)
        
        # Plot bars for each strategy
        for j, strategy in enumerate(pivot.columns):
            offset = (j - (len(pivot.columns) - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset, 
                pivot[strategy], 
                bar_width, 
                label=strategy,
                color=color_dict[strategy],
                alpha=0.8,
                edgecolor='white',
                linewidth=1
            )
            
            # Add value labels with improved formatting
            for k, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    height * 1.01,  # Place label slightly above bar
                    f"{height:.1f}",
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    color='dimgray'
                )
        
        # Set labels and title
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
        ax.set_xlabel('Task', fontsize=13, fontweight='bold')
        y_label = metric
        if metric in ['Accuracy', 'Coverage Rate']:
            y_label += ' (%)'
        ax.set_ylabel(y_label, fontsize=13, fontweight='bold')
        
        # Set x-ticks
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=0, fontsize=11)
        
        # Add grid
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        
        # Add background color
        ax.set_facecolor('#f8f9fa')
        
        # Add legend only to the first subplot
        if i == 0:
            ax.legend(
                title='Prompt Strategy',
                title_fontsize=12,
                frameon=True,
                facecolor='white',
                edgecolor='lightgray',
                loc='upper right'
            )
    
    # Add a main title with model name
    pretty_model = prettify_model_name(model_name)
    fig.suptitle(f'Prompt Strategy Evaluation for {pretty_model}', fontsize=18, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save figure if output_file is provided
    if output_file:
        save_figure(fig, output_file)
    
    # Show figure if show is True
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def visualize_correlation(
    results: Dict[str, Dict[str, Any]],
    model_names: List[str] = None,
    output_file: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize correlations between metrics across tasks and models.
    
    Args:
        results: Dictionary with benchmark results
        model_names: List of model names to include. If None, include all.
        output_file: Path to save the visualization. If None, don't save.
        show: Whether to display the figure
        
    Returns:
        Matplotlib figure
    """
    # Set up plot style
    setup_plot_style()
    
    if model_names is None:
        model_names = list(results.keys())
    
    # Get report DataFrame
    report_df = generate_report(results, model_names)
    
    # Filter out average results
    task_df = report_df[report_df['Task'] != 'Average'].copy()
    
    # Pivot the data to get metrics by model and task
    pivot_acc = task_df.pivot_table(index='Model', columns='Task', values='Accuracy')
    pivot_cr = task_df.pivot_table(index='Model', columns='Task', values='Coverage Rate')
    pivot_ss = task_df.pivot_table(index='Model', columns='Task', values='Set Size')
    
    # Rename columns to indicate metric
    pivot_acc.columns = [f"{col}_acc" for col in pivot_acc.columns]
    pivot_cr.columns = [f"{col}_cr" for col in pivot_cr.columns]
    pivot_ss.columns = [f"{col}_ss" for col in pivot_ss.columns]
    
    # Combine the dataframes
    combined = pd.concat([pivot_acc, pivot_cr, pivot_ss], axis=1)
    
    # Calculate correlation
    corr = combined.corr()
    
    # Create figure
    fig, ax = create_figure(figsize=(16, 14))
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Create a more visually pleasing diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Create correlation heatmap with improved styling
    sns.heatmap(
        corr, 
        mask=mask, 
        cmap=cmap, 
        vmax=1, 
        vmin=-1, 
        center=0,
        square=True, 
        linewidths=0.5, 
        cbar_kws={"shrink": .8, "label": "Correlation Coefficient"},
        annot=True,
        fmt=".2f",
        ax=ax
    )
    
    # Improve heatmap annotations
    for text in ax.texts:
        value = float(text.get_text())
        if abs(value) > 0.7:
            text.set_weight('bold')
        if value > 0:
            text.set_color('darkblue')
        elif value < 0:
            text.set_color('darkred')
    
    # Set title and adjust layout
    ax.set_title('Correlation between Metrics across Tasks', fontsize=18, fontweight='bold', pad=20)
    
    # Make the x and y labels more readable
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    
    # Save figure if output_file is provided
    if output_file:
        save_figure(fig, output_file)
    
    # Show figure if show is True
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig
