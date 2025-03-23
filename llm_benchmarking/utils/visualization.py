"""
Visualization utilities for LLM Uncertainty Benchmarking.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Union, Tuple

from .logging import get_logger

logger = get_logger(__name__)


def setup_plot_style():
    """
    Set up matplotlib style for consistent visualizations.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("talk")
    
    # Configure font sizes
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })


def create_figure(
    nrows: int = 1, 
    ncols: int = 1, 
    figsize: Tuple[int, int] = None,
    title: Optional[str] = None,
    tight_layout: bool = True
) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
    """
    Create a figure with the specified configuration.
    
    Args:
        nrows: Number of rows in the figure
        ncols: Number of columns in the figure
        figsize: Figure size in inches (width, height)
        title: Figure title
        tight_layout: Whether to apply tight layout
        
    Returns:
        Tuple of (figure, axes)
    """
    if figsize is None:
        figsize = (8 * ncols, 6 * nrows)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    if title:
        fig.suptitle(title)
    
    if tight_layout:
        plt.tight_layout()
        
        # Adjust spacing if we have a title
        if title:
            plt.subplots_adjust(top=0.9)
    
    return fig, axes


def save_figure(
    fig: plt.Figure,
    output_file: str,
    dpi: int = 300,
    bbox_inches: str = 'tight',
    pad_inches: float = 0.1
):
    """
    Save a figure to a file.
    
    Args:
        fig: Figure to save
        output_file: Output file path
        dpi: Resolution in dots per inch
        bbox_inches: Bounding box option
        pad_inches: Padding around the figure
    """
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save figure
    fig.savefig(
        output_file,
        dpi=dpi,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches
    )
    
    logger.info(f"Figure saved to {output_file}")


def prettify_model_name(model_name: str) -> str:
    """
    Make model name more readable for display in visualizations.
    
    Args:
        model_name: Original model name
        
    Returns:
        Prettified model name
    """
    # Remove organization prefix
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    
    # Replace underscores and hyphens with spaces
    model_name = model_name.replace("_", " ").replace("-", " ")
    
    # Capitalize key terms
    for term in ["gpt", "llama", "mistral", "falcon", "mpt", "phi"]:
        if term in model_name.lower():
            pattern = term
            replacement = term.upper()
            model_name = model_name.lower().replace(pattern, replacement)
    
    return model_name


def create_colormap(n_colors: int) -> List[str]:
    """
    Create a list of distinct colors for plotting.
    
    Args:
        n_colors: Number of colors needed
        
    Returns:
        List of color hex codes
    """
    # Use seaborn color palettes
    if n_colors <= 10:
        colors = sns.color_palette("tab10", n_colors)
    elif n_colors <= 20:
        colors = sns.color_palette("tab20", n_colors)
    else:
        # For more colors, create a custom palette
        colors = sns.color_palette("hsv", n_colors)
    
    # Convert to hex strings
    return [sns.utils.rgb2hex(c) for c in colors]


def add_labels_to_bars(
    ax: plt.Axes,
    labels: Optional[List[str]] = None,
    fontsize: int = 9,
    rotation: int = 0,
    ha: str = 'center',
    va: str = 'bottom',
    fmt: str = '.1f'
):
    """
    Add value labels to bars in a bar plot.
    
    Args:
        ax: Matplotlib axes
        labels: Custom labels to use (if None, use bar heights)
        fontsize: Font size for labels
        rotation: Rotation angle for labels
        ha: Horizontal alignment
        va: Vertical alignment
        fmt: Format string for numeric values
    """
    for i, patch in enumerate(ax.patches):
        # Get bar height
        height = patch.get_height()
        
        # Determine label
        if labels is None:
            label = f"{height:{fmt}}"
        else:
            label = labels[i]
        
        # Add text label
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + height * 0.01,  # Slight offset above bar
            label,
            ha=ha,
            va=va,
            fontsize=fontsize,
            rotation=rotation
        )


def create_benchmark_scatter_plot(
    data: Dict[str, Dict[str, float]],
    x_metric: str,
    y_metric: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_file: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a scatter plot comparing two metrics across models.
    
    Args:
        data: Dictionary of model name -> metrics dictionary
        x_metric: Metric to plot on x-axis
        y_metric: Metric to plot on y-axis
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        output_file: Optional file to save figure to
        show: Whether to display the figure
        
    Returns:
        Matplotlib figure
    """
    # Extract data
    models = list(data.keys())
    x_values = [data[model][x_metric] for model in models]
    y_values = [data[model][y_metric] for model in models]
    
    # Prettify model names
    display_names = [prettify_model_name(model) for model in models]
    
    # Create color map
    colors = create_colormap(len(models))
    
    # Create figure
    fig, ax = create_figure(figsize=(10, 8))
    
    # Create scatter plot
    scatter = ax.scatter(
        x_values,
        y_values,
        s=100,
        c=colors,
        alpha=0.8
    )
    
    # Add labels for each point
    for i, name in enumerate(display_names):
        ax.annotate(
            name,
            (x_values[i], y_values[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10
        )
    
    # Set labels and title
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output_file is provided
    if output_file:
        save_figure(fig, output_file)
    
    # Show figure if show is True
    if show:
        plt.show()
    
    return fig


def create_benchmark_bar_chart(
    data: Dict[str, Dict[str, float]],
    metric: str,
    title: str,
    ylabel: str,
    sort_by: Optional[str] = None,
    ascending: bool = False,
    output_file: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a bar chart of a metric across models.
    
    Args:
        data: Dictionary of model name -> metrics dictionary
        metric: Metric to plot
        title: Plot title
        ylabel: Y-axis label
        sort_by: Optional metric to sort by
        ascending: Whether to sort in ascending order
        output_file: Optional file to save figure to
        show: Whether to display the figure
        
    Returns:
        Matplotlib figure
    """
    import pandas as pd
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(
        {model: {metric: values[metric] for metric in values} for model, values in data.items()}
    ).T
    
    # Sort if requested
    if sort_by:
        df = df.sort_values(sort_by, ascending=ascending)
    
    # Extract the metric to plot
    values = df[metric].values
    
    # Prettify model names
    models = df.index.tolist()
    display_names = [prettify_model_name(model) for model in models]
    
    # Create color map
    colors = create_colormap(len(models))
    
    # Create figure
    fig, ax = create_figure(figsize=(12, 6))
    
    # Create bar chart
    bars = ax.bar(
        display_names,
        values,
        color=colors,
        alpha=0.8
    )
    
    # Add value labels
    add_labels_to_bars(ax)
    
    # Set labels and title
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Models")
    
    # Rotate x-axis labels if there are many models
    if len(models) > 5:
        plt.xticks(rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output_file is provided
    if output_file:
        save_figure(fig, output_file)
    
    # Show figure if show is True
    if show:
        plt.show()
    
    return fig


def create_multi_metric_comparison(
    data: Dict[str, Dict[str, Dict[str, float]]],
    metrics: List[str],
    task: str,
    title: str,
    output_file: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a multi-metric comparison chart for a specific task.
    
    Args:
        data: Dictionary of model -> task -> metrics
        metrics: List of metrics to include
        task: Task to visualize
        title: Plot title
        output_file: Optional file to save figure to
        show: Whether to display the figure
        
    Returns:
        Matplotlib figure
    """
    import pandas as pd
    
    # Extract task-specific data
    task_data = {}
    for model, model_data in data.items():
        if task in model_data:
            task_data[model] = model_data[task]
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'Model': prettify_model_name(model),
            **{metric: values.get(metric, 0) for metric in metrics}
        }
        for model, values in task_data.items()
    ])
    
    # Sort by the first metric
    df = df.sort_values(metrics[0], ascending=False)
    
    # Set up figure
    n_models = len(df)
    fig_width = max(10, n_models * 0.8)
    fig, ax = create_figure(figsize=(fig_width, 6))
    
    # Set up bar positions
    bar_width = 0.8 / len(metrics)
    positions = np.arange(len(df))
    
    # Create a bar for each metric
    for i, metric in enumerate(metrics):
        offset = (i - (len(metrics) - 1) / 2) * bar_width
        ax.bar(
            positions + offset,
            df[metric],
            width=bar_width,
            label=metric,
            alpha=0.7
        )
    
    # Set labels and title
    ax.set_title(title)
    ax.set_ylabel('Value')
    ax.set_xlabel('Models')
    
    # Set x-tick labels
    ax.set_xticks(positions)
    ax.set_xticklabels(df['Model'], rotation=45, ha='right')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output_file is provided
    if output_file:
        save_figure(fig, output_file)
    
    # Show figure if show is True
    if show:
        plt.show()
    
    return fig


def create_task_comparison_chart(
    data: Dict[str, Dict[str, Dict[str, float]]],
    model: str,
    metric: str,
    title: str,
    output_file: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a chart comparing a metric across different tasks for a specific model.
    
    Args:
        data: Dictionary of model -> task -> metrics
        model: Model to visualize
        metric: Metric to visualize
        title: Plot title
        output_file: Optional file to save figure to
        show: Whether to display the figure
        
    Returns:
        Matplotlib figure
    """
    # Extract model data
    model_data = data.get(model, {})
    
    # Sort tasks by metric value
    tasks = []
    values = []
    
    for task, metrics in model_data.items():
        if task != 'overall' and metric in metrics:
            tasks.append(task)
            values.append(metrics[metric])
    
    # Sort by value
    sorted_indices = np.argsort(values)[::-1]
    tasks = [tasks[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    
    # Create figure
    fig, ax = create_figure(figsize=(10, 6))
    
    # Create bar chart
    bars = ax.bar(
        tasks,
        values,
        alpha=0.8
    )
    
    # Add value labels
    add_labels_to_bars(ax)
    
    # Set labels and title
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xlabel('Tasks')
    
    # Add grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output_file is provided
    if output_file:
        save_figure(fig, output_file)
    
    # Show figure if show is True
    if show:
        plt.show()
    
    return fig
