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
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 18,
        'figure.figsize': (12, 8),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
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
        # Calculate reasonable figure size based on rows and columns
        width = max(10, 6 * ncols)
        height = max(8, 5 * nrows)
        figsize = (width, height)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    if title:
        fig.suptitle(title, fontweight='bold', fontsize=18)
    
    if tight_layout:
        # Add some extra space at the top for the title
        if title:
            plt.subplots_adjust(top=0.92)
    
    return fig, axes


def save_figure(
    fig: plt.Figure,
    output_file: str,
    dpi: int = 300,
    bbox_inches: str = 'tight',
    pad_inches: float = 0.2,
    transparent: bool = False
):
    """
    Save a figure to a file.
    
    Args:
        fig: Figure to save
        output_file: Output file path
        dpi: Resolution in dots per inch
        bbox_inches: Bounding box option
        pad_inches: Padding around the figure
        transparent: Whether to use transparent background
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
        pad_inches=pad_inches,
        transparent=transparent
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
    # Remove organization prefix if present
    if "/" in model_name:
        org, name = model_name.split("/", 1)
        # Keep a shortened organization name for clarity
        short_org = org.split("-")[0] if "-" in org else org
        model_name = f"{short_org}/{name}"
    
    # Replace underscores with spaces but keep hyphens for readability
    model_name = model_name.replace("_", " ")
    
    # Shorten common terms to save space
    replacements = {
        "Instruct": "Inst",
        "Instruction": "Inst",
        "International": "Int'l",
        "-AWQ": "",
        "-GPTQ": "",
        "-Quantized": "",
        "Language": "Lang",
        "Foundation": "Found",
    }
    
    for old, new in replacements.items():
        model_name = model_name.replace(old, new)
    
    # Capitalize model family names
    for family in ["gpt", "llama", "mistral", "falcon", "mpt", "phi", "qwen", "hermes"]:
        if family in model_name.lower():
            pattern = family
            replacement = family.upper()
            # Replace only the model family name, not if it's part of another word
            parts = []
            for part in model_name.split():
                if part.lower() == pattern:
                    parts.append(replacement)
                elif part.lower().startswith(pattern) and not part[len(pattern):].isalpha():
                    parts.append(replacement + part[len(pattern):])
                else:
                    parts.append(part)
            model_name = " ".join(parts)
    
    # Max length for chart readability
    if len(model_name) > 30:
        # Try to smartly truncate
        parts = model_name.split("/")
        if len(parts) > 1:
            # If there's an org/model split, keep both but truncate
            org, name = parts
            if len(org) > 10:
                org = org[:8] + ".."
            if len(name) > 18:
                name = name[:16] + ".."
            model_name = f"{org}/{name}"
        else:
            # Just truncate with ellipsis
            model_name = model_name[:28] + ".."
    
    return model_name


def rgb2hex(rgb):
    """Convert RGB tuple to hex color code.
    
    This is a local implementation to avoid dependency on specific seaborn versions.
    
    Args:
        rgb: Tuple of RGB values, each between 0 and 1
        
    Returns:
        Hex color code string
    """
    rgb = tuple(int(x * 255) for x in rgb)
    return "#{0:02x}{1:02x}{2:02x}".format(*rgb)


def create_colormap(n_colors: int, palette: str = "tab10") -> List[str]:
    """
    Create a list of distinct colors for plotting.
    
    Args:
        n_colors: Number of colors needed
        palette: Name of seaborn color palette to use
        
    Returns:
        List of color hex codes
    """
    # Use different palettes based on number of colors needed
    if n_colors <= 10:
        if palette == "tab10":
            colors = sns.color_palette("tab10", n_colors)
        else:
            colors = sns.color_palette(palette, n_colors)
    elif n_colors <= 20:
        if palette == "tab20":
            colors = sns.color_palette("tab20", n_colors)
        else:
            # Tab20 works better for large number of categories
            colors = sns.color_palette("tab20", n_colors)
    else:
        # For many colors, create a custom palette with good differentiation
        colors = sns.color_palette("hsv", n_colors)
    
    # Convert to hex strings using our local implementation
    return [rgb2hex(c) for c in colors]


def add_labels_to_bars(
    ax: plt.Axes,
    labels: Optional[List[str]] = None,
    fontsize: int = 9,
    rotation: int = 0,
    ha: str = 'center',
    va: str = 'bottom',
    fmt: str = '.1f',
    offset: float = 0.01
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
        offset: Y-offset as proportion of the data range
    """
    # Calculate y-offset based on data range
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    y_offset = y_range * offset

    for i, patch in enumerate(ax.patches):
        # Get bar height
        height = patch.get_height()
        
        # Skip labels for very small values (optional)
        if height < y_range * 0.02:  # Skip if less than 2% of range
            continue
        
        # Determine label
        if labels is None:
            label = f"{height:{fmt}}"
        else:
            if i < len(labels):
                label = labels[i]
            else:
                continue
        
        # Add text label
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + y_offset,  # Use calculated offset
            label,
            ha=ha,
            va=va,
            fontsize=fontsize,
            rotation=rotation,
            fontweight='bold',
            color='dimgrey'
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
    colors = create_colormap(len(models), palette="viridis")
    
    # Create figure
    fig, ax = create_figure(figsize=(12, 10))
    
    # Create scatter plot
    scatter = ax.scatter(
        x_values,
        y_values,
        s=200,  # Larger markers
        c=colors,
        alpha=0.8,
        edgecolors='white',
        linewidths=1.5
    )
    
    # Add labels for each point with improved positioning
    try:
        # Try to use adjustText for better label placement
        from adjustText import adjust_text
        
        texts = []
        for i, name in enumerate(display_names):
            texts.append(ax.text(
                x_values[i], 
                y_values[i], 
                name,
                fontsize=10,
                weight='bold'
            ))
        
        # Adjust text positions to avoid overlaps
        adjust_text(
            texts,
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
            expand_points=(1.5, 1.5),
            force_points=(0.1, 0.1)
        )
    except ImportError:
        # Fallback if adjustText is not available
        logger.debug("adjustText not available, using basic label placement")
        for i, name in enumerate(display_names):
            ax.annotate(
                name,
                (x_values[i] + 0.1, y_values[i] + 0.1),  # Add offset
                fontsize=9,
                fontweight='bold'
            )
    
    # Set labels and title
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Add a subtle background color to enhance readability
    ax.set_facecolor('#f8f9fa')
    
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
    
    # Create color map with a more pleasing palette
    colors = create_colormap(len(models), palette="muted")
    
    # Create figure with dynamic sizing based on number of models
    fig_width = max(12, len(models) * 0.8)
    fig, ax = create_figure(figsize=(fig_width, 8))
    
    # Create bar chart with improved styling
    bars = ax.bar(
        display_names,
        values,
        color=colors,
        alpha=0.8,
        width=0.7,
        edgecolor='white',
        linewidth=1.5
    )
    
    # Add value labels with improved formatting
    add_labels_to_bars(ax, fontsize=10, offset=0.02)
    
    # Set labels and title with improved styling
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_xlabel("Models", fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    if len(models) > 3:
        plt.xticks(rotation=30, ha='right')
    
    # Add grid for readability but only on the y-axis
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Set background color
    ax.set_facecolor('#f8f9fa')
    
    # Adjust layout for rotated labels
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
    
    # Set up figure with dynamic sizing
    n_models = len(df)
    fig_width = max(12, n_models * 1.2)
    fig, ax = create_figure(figsize=(fig_width, 8))
    
    # Get a color palette that differentiates metrics well
    colors = create_colormap(len(metrics), palette="colorblind")
    
    # Set up bar positions
    bar_width = 0.8 / len(metrics)
    positions = np.arange(len(df))
    
    # Create a bar for each metric with improved styling
    for i, metric in enumerate(metrics):
        offset = (i - (len(metrics) - 1) / 2) * bar_width
        ax.bar(
            positions + offset,
            df[metric],
            width=bar_width,
            label=metric,
            color=colors[i],
            alpha=0.8,
            edgecolor='white',
            linewidth=1
        )
    
    # Set labels and title with improved styling
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Value', fontsize=14, fontweight='bold')
    ax.set_xlabel('Models', fontsize=14, fontweight='bold')
    
    # Set x-tick labels with improved readability
    ax.set_xticks(positions)
    ax.set_xticklabels(df['Model'], rotation=30, ha='right')
    
    # Add legend with improved styling
    ax.legend(
        title='Metrics',
        title_fontsize=12,
        fontsize=10,
        frameon=True,
        facecolor='white',
        edgecolor='lightgray',
        loc='best'
    )
    
    # Add grid for readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Set background color
    ax.set_facecolor('#f8f9fa')
    
    # Adjust layout for rotated labels
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
    fig, ax = create_figure(figsize=(12, 8))
    
    # Choose a visually appealing color palette
    colors = create_colormap(len(tasks), palette="viridis")
    
    # Create bar chart with improved styling
    bars = ax.bar(
        tasks,
        values,
        color=colors,
        alpha=0.85,
        width=0.7,
        edgecolor='white',
        linewidth=1.5
    )
    
    # Add value labels with improved formatting
    add_labels_to_bars(ax, fontsize=11, offset=0.02)
    
    # Set labels and title with improved styling
    model_name = prettify_model_name(model)
    ax.set_title(f"{title}: {model_name}", fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel(metric, fontsize=14, fontweight='bold')
    ax.set_xlabel('Tasks', fontsize=14, fontweight='bold')
    
    # Add grid for readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Set background color
    ax.set_facecolor('#f8f9fa')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output_file is provided
    if output_file:
        save_figure(fig, output_file)
    
    # Show figure if show is True
    if show:
        plt.show()
    
    return fig
