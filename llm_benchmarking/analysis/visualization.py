"""
Visualization module for LLM Uncertainty Benchmarking.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Union, Tuple

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
    
    # Create figure with 2 subplots
    main_fig, (ax1, ax2) = create_figure(1, 2, figsize=(15, 7))
    
    # Accuracy vs. Set Size plot
    sns.scatterplot(
        data=avg_results, 
        x='Accuracy', 
        y='Set Size', 
        s=100, 
        hue='Model', 
        ax=ax1
    )
    ax1.set_title('Accuracy vs. Uncertainty (Set Size)')
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_ylabel('Average Set Size (lower is better)')
    
    # Add text labels for each point
    for idx, row in avg_results.iterrows():
        model_name = prettify_model_name(row['Model'])
        ax1.text(row['Accuracy'] + 0.5, row['Set Size'] + 0.05, model_name, fontsize=8)
    
    # Coverage Rate vs. Set Size plot
    sns.scatterplot(
        data=avg_results, 
        x='Coverage Rate', 
        y='Set Size', 
        s=100, 
        hue='Model', 
        ax=ax2
    )
    ax2.set_title('Coverage Rate vs. Uncertainty (Set Size)')
    ax2.set_xlabel('Coverage Rate (%)')
    ax2.set_ylabel('Average Set Size (lower is better)')
    
    # Add text labels for each point
    for idx, row in avg_results.iterrows():
        model_name = prettify_model_name(row['Model'])
        ax2.text(row['Coverage Rate'] + 0.5, row['Set Size'] + 0.05, model_name, fontsize=8)
    
    plt.tight_layout()
    
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
    
    # Create figure with subplots (one for each task)
    task_fig, axes = create_figure(len(task_list), 1, figsize=(12, 5*len(task_list)))
    
    if len(task_list) == 1:
        axes = [axes]  # Make sure axes is always a list
    
    for i, task in enumerate(task_list):
        task_df = report_df[report_df['Task'] == task].copy()
        
        ax = axes[i]
        
        # Sort by accuracy
        task_df = task_df.sort_values('Accuracy', ascending=False)
        
        # Create bar plot with two metrics
        bar_width = 0.35
        x = np.arange(len(task_df))
        
        # Plot accuracy bars
        ax.bar(x - bar_width/2, task_df['Accuracy'], bar_width, label='Accuracy (%)')
        
        # Create twin axis for set size
        ax2 = ax.twinx()
        ax2.bar(x + bar_width/2, task_df['Set Size'], bar_width, color='orange', label='Set Size')
        
        # Set labels and title
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy (%)')
        ax2.set_ylabel('Set Size')
        ax.set_title(f'Task: {task}')
        
        # Set x-ticks
        ax.set_xticks(x)
        model_names = [prettify_model_name(m) for m in task_df['Model']]
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
    plt.tight_layout()
    
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
    
    # Pivot the data for easier plotting
    pivot_df = task_df.pivot(index='Model', columns='Task', values='Accuracy')
    
    # Create a heatmap
    fig, ax = create_figure(figsize=(12, len(model_names) * 0.6))
    
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        linewidths=.5,
        ax=ax
    )
    
    ax.set_title('Model Performance Across Tasks (Accuracy %)')
    ax.set_ylabel('Models')
    ax.set_xlabel('Tasks')
    
    # Prettify model names on y-axis
    y_labels = [prettify_model_name(model) for model in pivot_df.index]
    ax.set_yticklabels(y_labels, rotation=0)
    
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
    fig, ax = create_figure(figsize=(14, 12))
    
    # Create correlation heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr, 
        mask=mask, 
        cmap=cmap, 
        vmax=1, 
        vmin=-1, 
        center=0,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .5},
        annot=True,
        fmt=".2f",
        ax=ax
    )
    
    ax.set_title('Correlation between Metrics across Tasks')
    
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
    
    # Create figure with subplots for each metric
    fig, axes = create_figure(3, 1, figsize=(12, 15))
    
    metrics = ['Accuracy', 'Coverage Rate', 'Set Size']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Create grouped bar chart
        pivot = df.pivot(index='Task', columns='Prompt Strategy', values=metric)
        pivot.plot(kind='bar', ax=ax, rot=0)
        
        # Set labels and title
        ax.set_xlabel('Task')
        ax.set_ylabel(metric + (' (%)' if metric in ['Accuracy', 'Coverage Rate'] else ''))
        ax.set_title(f'{metric} by Task and Prompt Strategy')
        
        # Add legend
        ax.legend(title='Prompt Strategy')
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f')
    
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
    
    # Sort by model size
    avg_results['Size'] = avg_results['Model'].apply(lambda x: x.split('-')[-1])
    avg_results['Size'] = pd.Categorical(avg_results['Size'], categories=model_sizes, ordered=True)
    avg_results = avg_results.sort_values('Size')
    
    # Create figure with 3 subplots
    fig, axes = create_figure(1, 3, figsize=(18, 6))
    
    metrics = ['Accuracy', 'Coverage Rate', 'Set Size']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Create line plot
        ax.plot(avg_results['Size'], avg_results[metric], marker='o', markersize=10, linewidth=2)
        
        # Add value labels
        for x, y in zip(avg_results['Size'], avg_results[metric]):
            ax.text(x, y + (0.02 * max(avg_results[metric])), f"{y:.1f}", ha='center')
        
        # Set labels and title
        ax.set_xlabel('Model Size')
        ax.set_ylabel(metric + (' (%)' if metric in ['Accuracy', 'Coverage Rate'] else ''))
        ax.set_title(f'Effect of Model Scale on {metric}')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.suptitle(f"Scaling Analysis for {model_family.upper()} Models", fontsize=16)
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
