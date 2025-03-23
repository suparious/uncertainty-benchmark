#!/usr/bin/env python
"""
Example of comparing multiple models using the LLM Uncertainty Benchmark.
"""

import os
import argparse
from llm_benchmarking.benchmark import LLMBenchmark, ParallelBenchmark
from llm_benchmarking.analysis import (
    generate_report,
    visualize_results,
    visualize_task_comparisons,
    visualize_correlation
)


def compare_models(
    api_base_url, 
    api_key, 
    model_names, 
    output_dir,
    use_parallel=False,
    sample_size=100
):
    """
    Compare multiple models on the benchmark.
    
    Args:
        api_base_url: Base URL for the API
        api_key: API key (if required)
        model_names: List of model names to compare
        output_dir: Directory to save results to
        use_parallel: Whether to use parallel processing
        sample_size: Number of samples per task
    """
    print(f"Comparing models: {', '.join(model_names)}")
    
    # Create benchmark instance
    if use_parallel:
        benchmark = ParallelBenchmark(
            api_base_url=api_base_url,
            api_key=api_key
        )
    else:
        benchmark = LLMBenchmark(
            api_base_url=api_base_url,
            api_key=api_key
        )
    
    # Prepare datasets
    benchmark.prepare_datasets(sample_size=sample_size)
    
    # Evaluate each model
    for model_name in model_names:
        print(f"Evaluating {model_name}...")
        
        # Determine if model uses chat template (simple heuristic)
        is_chat_model = any(keyword in model_name.lower() for keyword in ["chat", "instruct", "gpt"])
        
        try:
            benchmark.evaluate_model(
                model_name=model_name,
                use_chat_template=is_chat_model
            )
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
    
    # Save all results
    os.makedirs(output_dir, exist_ok=True)
    benchmark.save_results(output_dir)
    
    # Generate comprehensive model comparison
    create_comprehensive_comparison(benchmark.results, output_dir)


def create_comprehensive_comparison(results, output_dir):
    """
    Create a comprehensive model comparison.
    
    Args:
        results: Dictionary with benchmark results
        output_dir: Directory to save results to
    """
    # Generate comparison report
    report_file = os.path.join(output_dir, "model_comparison.csv")
    report_df = generate_report(results, output_file=report_file)
    print(f"Comparison report saved to {report_file}")
    print("\nModel Comparison Report:")
    print(report_df)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Main visualizations
    viz_file = os.path.join(output_dir, "model_comparison.png")
    visualize_results(results, output_file=viz_file)
    
    # Task comparison visualization
    compare_file = os.path.join(output_dir, "task_comparison.png")
    visualize_task_comparisons(results, output_file=compare_file)
    
    # Correlation visualization (if we have enough models)
    if len(results) >= 3:
        corr_file = os.path.join(output_dir, "metric_correlations.png")
        visualize_correlation(results, output_file=corr_file)
    
    print(f"All comparison results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Comparison Example")
    parser.add_argument("--api-base", required=True, help="Base URL for the OpenAI-compatible API")
    parser.add_argument("--api-key", help="API key (if required)")
    parser.add_argument("--models", required=True, nargs="+", help="Names of models to compare")
    parser.add_argument("--output-dir", default="./comparison_results", help="Output directory for results")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of samples per task")
    
    args = parser.parse_args()
    
    compare_models(
        api_base_url=args.api_base,
        api_key=args.api_key,
        model_names=args.models,
        output_dir=args.output_dir,
        use_parallel=args.parallel,
        sample_size=args.sample_size
    )
