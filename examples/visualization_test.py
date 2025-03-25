#!/usr/bin/env python3
"""
Test script for visualizations in the LLM Benchmarking Suite.
This script tests all visualization modes using sample data.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_benchmarking.analysis.visualization import (
    visualize_results,
    visualize_task_comparisons,
    visualize_correlation,
    visualize_prompt_strategies,
    visualize_model_scaling
)
from llm_benchmarking.analysis.reporting import load_results, generate_report


def find_benchmark_results(benchmark_dir=None):
    """Find benchmark results in the given directory or default locations."""
    if benchmark_dir and os.path.isdir(benchmark_dir):
        search_dirs = [benchmark_dir]
    else:
        # Default locations to search for benchmark results
        search_dirs = [
            './benchmark_results',
            '../benchmark_results',
            './results',
            '../results',
        ]
    
    result_files = []
    
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
            
        # Walk through the directory and find benchmark_results.json files
        for root, _, files in os.walk(search_dir):
            for file in files:
                if file == "benchmark_results.json":
                    result_files.append(os.path.join(root, file))
    
    return result_files


def load_all_results(result_files):
    """Load results from all result files."""
    all_results = {}
    
    for result_file in result_files:
        print(f"Loading results from {result_file}")
        results = load_results(result_file)
        all_results.update(results)
    
    return all_results


def test_visualizations(results, output_dir=None, show=True):
    """Test all visualization functions and save results to the output directory."""
    if not output_dir:
        output_dir = './visualization_test'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model names
    model_names = list(results.keys())
    
    if not model_names:
        print("No models found in the results.")
        return
    
    print(f"Found {len(model_names)} models: {', '.join(model_names)}")
    
    # Test basic visualization
    print("\nTesting basic visualization...")
    main_fig, task_fig = visualize_results(
        results,
        output_file=os.path.join(output_dir, "test_basic.png"),
        show=show
    )
    
    # Test task comparison visualization
    print("\nTesting task comparison visualization...")
    comparison_fig = visualize_task_comparisons(
        results,
        output_file=os.path.join(output_dir, "test_comparison.png"),
        show=show
    )
    
    # Test correlation visualization
    print("\nTesting correlation visualization...")
    corr_fig = visualize_correlation(
        results,
        output_file=os.path.join(output_dir, "test_correlation.png"),
        show=show
    )
    
    # Test prompt strategy visualization for the first model
    print("\nTesting prompt strategy visualization...")
    model = model_names[0]
    prompt_fig = visualize_prompt_strategies(
        results,
        model_name=model,
        output_file=os.path.join(output_dir, f"test_prompt_{model}.png"),
        show=show
    )
    
    # Test model scaling visualization with a dummy model family
    # This is just a test, so we'll use the existing models as if they're from the same family
    print("\nTesting model scaling visualization...")
    # Create a dummy model family and sizes from the existing models
    model_family = "test"
    model_sizes = ["7B", "13B", "70B"]  # Example sizes
    
    # Rename the models temporarily for the test
    renamed_results = {}
    for i, model_name in enumerate(model_names[:3]):  # Use up to 3 models
        if i < len(model_sizes):
            renamed_results[f"{model_family}-{model_sizes[i]}"] = results[model_name]
    
    if len(renamed_results) > 1:
        scaling_fig = visualize_model_scaling(
            renamed_results,
            model_family=model_family,
            model_sizes=model_sizes[:len(renamed_results)],
            output_file=os.path.join(output_dir, "test_scaling.png"),
            show=show
        )
    
    print(f"\nAll tests completed. Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Test visualization functions for LLM Benchmarking")
    parser.add_argument("--benchmark-dir", help="Directory containing benchmark results")
    parser.add_argument("--output-dir", default="./visualization_test", help="Directory to save test visualizations")
    parser.add_argument("--no-show", action="store_true", help="Don't display the visualizations")
    
    args = parser.parse_args()
    
    # Find benchmark results
    result_files = find_benchmark_results(args.benchmark_dir)
    
    if not result_files:
        print("No benchmark results found. Please run benchmarks first or specify a directory with --benchmark-dir.")
        sys.exit(1)
    
    # Load all results
    results = load_all_results(result_files)
    
    # Test visualizations
    test_visualizations(results, args.output_dir, not args.no_show)


if __name__ == "__main__":
    main()
