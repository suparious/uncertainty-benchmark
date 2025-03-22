#!/usr/bin/env python
"""
Analyze and visualize existing benchmark results.

This script can be used to load and analyze previously saved benchmark results
without needing to re-run the benchmark.
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from main import LLMUncertaintyBenchmark

def load_and_visualize(input_dirs, output_dir, top_k=None):
    """
    Load results from multiple directories and create comparative visualizations.
    
    Args:
        input_dirs: List of directories containing benchmark results
        output_dir: Directory to save visualizations to
        top_k: If provided, only show the top K models by accuracy
    """
    # Create benchmark object (won't be used to run benchmarks)
    benchmark = LLMUncertaintyBenchmark(api_base_url="dummy")
    
    # Load results from each directory
    for input_dir in input_dirs:
        results_file = os.path.join(input_dir, "benchmark_results.json")
        
        if not os.path.exists(results_file):
            print(f"Warning: No results file found in {input_dir}. Skipping...")
            continue
        
        print(f"Loading results from {results_file}...")
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Add these results to the benchmark
        for model_name, model_results in results.items():
            benchmark.results[model_name] = model_results
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate report for all models
    report_df = benchmark.generate_report()
    
    # If top_k is provided, filter to only show top models
    if top_k is not None and top_k > 0:
        # Get top k models by average accuracy
        top_models = report_df[report_df['Task'] == 'Average'] \
            .sort_values('Accuracy', ascending=False) \
            .head(top_k)['Model'].tolist()
        
        # Filter report to only include these models
        report_df = report_df[report_df['Model'].isin(top_models)]
    
    # Save report
    report_file = os.path.join(output_dir, "model_comparison.csv")
    report_df.to_csv(report_file, index=False)
    print(f"Saved comparison report to {report_file}")
    
    # Create visualizations
    create_visualizations(report_df, output_dir)
    
    # Create task-specific comparisons
    create_task_comparisons(report_df, output_dir)
    
    # If we have many models, also create a correlation plot
    if len(report_df['Model'].unique()) >= 5:
        create_correlation_plot(report_df, output_dir)
    
    print(f"All visualizations saved to {output_dir}")

def create_visualizations(report_df, output_dir):
    """Create standard visualizations from the report data."""
    # Filter for average results
    avg_results = report_df[report_df['Task'] == 'Average'].copy()
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
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
        model_name = row['Model'].split('/')[-1]  # Get just the model name without org prefix
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
        model_name = row['Model'].split('/')[-1]  # Get just the model name without org prefix
        ax2.text(row['Coverage Rate'] + 0.5, row['Set Size'] + 0.05, model_name, fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    viz_file = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    plt.close()

def create_task_comparisons(report_df, output_dir):
    """Create task-specific comparison visualizations."""
    task_list = [task for task in report_df['Task'].unique() if task != 'Average']
    
    # Create figure with subplots for each task
    fig, axes = plt.subplots(len(task_list), 1, figsize=(12, 5*len(task_list)))
    
    if len(task_list) == 1:
        axes = [axes]  # Make sure axes is always a list
    
    # For each task, create a comparison plot
    for i, task in enumerate(task_list):
        task_df = report_df[report_df['Task'] == task].copy()
        
        # Sort by accuracy
        task_df = task_df.sort_values('Accuracy', ascending=False)
        
        ax = axes[i]
        
        # Create bar plot for accuracy
        x = np.arange(len(task_df))
        width = 0.35
        
        ax.bar(x - width/2, task_df['Accuracy'], width, label='Accuracy (%)')
        
        # Create twin axis for Set Size
        ax2 = ax.twinx()
        ax2.bar(x + width/2, task_df['Set Size'], width, color='orange', label='Set Size')
        
        # Set labels
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy (%)')
        ax2.set_ylabel('Set Size')
        ax.set_title(f'Task: {task}')
        
        # Set x-ticks with model names
        ax.set_xticks(x)
        model_names = [m.split('/')[-1] for m in task_df['Model']]  # Simplify names for display
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    
    # Save figure
    viz_file = os.path.join(output_dir, "task_comparisons.png")
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    plt.close()

def create_correlation_plot(report_df, output_dir):
    """Create a correlation plot to see relationships between metrics across models."""
    # Get all task-specific results (exclude Average)
    task_df = report_df[report_df['Task'] != 'Average'].copy()
    
    # Pivot the data to get metrics by model and task
    pivot_acc = task_df.pivot_table(index='Model', columns='Task', values='Accuracy')
    pivot_ss = task_df.pivot_table(index='Model', columns='Task', values='Set Size')
    
    # Rename columns to indicate metric
    pivot_acc.columns = [f"{col}_acc" for col in pivot_acc.columns]
    pivot_ss.columns = [f"{col}_ss" for col in pivot_ss.columns]
    
    # Combine the dataframes
    combined = pd.concat([pivot_acc, pivot_ss], axis=1)
    
    # Calculate correlation
    corr = combined.corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
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
        fmt=".2f"
    )
    
    plt.title('Correlation between Metrics across Tasks')
    plt.tight_layout()
    
    # Save figure
    viz_file = os.path.join(output_dir, "metric_correlations.png")
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and visualize benchmark results")
    parser.add_argument("--input-dirs", required=True, nargs="+", help="Directories containing benchmark results")
    parser.add_argument("--output-dir", default="./analysis_results", help="Directory to save visualizations to")
    parser.add_argument("--top-k", type=int, help="If provided, only show the top K models by accuracy")
    
    args = parser.parse_args()
    
    load_and_visualize(args.input_dirs, args.output_dir, args.top_k)