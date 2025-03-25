"""
Analysis command module for LLM Uncertainty Benchmarking.
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any, Optional

from ..analysis.reporting import load_results, load_multiple_results, generate_report
from ..analysis.visualization import (
    visualize_results,
    visualize_task_comparisons, 
    visualize_correlation,
    visualize_prompt_strategies,
    visualize_model_scaling
)
from ..utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def analyze_command():
    """
    Main function for the analyze command.
    """
    parser = argparse.ArgumentParser(description="Analyze LLM Uncertainty Benchmark Results")
    
    # Input options
    parser.add_argument("--input-dirs", nargs="+", help="Directories containing benchmark results")
    parser.add_argument("--input-files", nargs="+", help="Specific result files to analyze")
    
    # Analysis options
    parser.add_argument("--models", nargs="+", help="Models to include in the analysis (if not specified, all models will be included)")
    parser.add_argument("--top-k", type=int, help="Only show the top K models by accuracy")
    
    # Visualization options
    parser.add_argument("--mode", choices=["basic", "comparative", "correlations", "prompt", "scaling"], default="basic", 
                        help="Analysis mode (basic: simple visualizations, comparative: task comparisons, correlations: metric correlations, prompt: prompt strategy analysis, scaling: model scaling analysis)")
    parser.add_argument("--model-family", help="Model family for scaling analysis (e.g., 'llama')")
    parser.add_argument("--model-sizes", nargs="+", help="Model sizes for scaling analysis (e.g., '7B' '13B' '70B')")
    parser.add_argument("--no-show", action="store_true", help="Don't display visualizations (only save to files)")
    
    # Output options
    parser.add_argument("--output-dir", default="./analysis_results", help="Output directory for analysis results")
    parser.add_argument("--output-prefix", default="model_comparison", help="Prefix for output files")
    parser.add_argument("--log-file", help="Path to save log output (if not specified, logs will only be written to console)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Check that at least one input option is provided
    if not args.input_dirs and not args.input_files:
        parser.error("At least one of --input-dirs or --input-files is required")
        
    # Ensure the proper dependencies are available
    try:
        import matplotlib
        import seaborn
    except ImportError:
        logger.error("Matplotlib and seaborn are required for analysis. Please install them with: pip install matplotlib seaborn")
        sys.exit(1)
    
    # Try to import adjustText which is used for better label placement
    try:
        from adjustText import adjust_text
        logger.info("Found 'adjustText' package for improved label placement.")
    except ImportError:
        logger.info("The 'adjustText' package is not installed. Labels might overlap in some visualizations.")
        logger.info("For improved label placement, install with: pip install adjustText")
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level, log_file=args.log_file)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = {}
    
    if args.input_dirs:
        for input_dir in args.input_dirs:
            logger.info(f"Looking for result files in {input_dir}")
            
            # Look for benchmark_results.json files
            result_files = []
            for root, _, files in os.walk(input_dir):
                for file in files:
                    if file == "benchmark_results.json":
                        result_files.append(os.path.join(root, file))
            
            if not result_files:
                logger.warning(f"No benchmark_results.json files found in {input_dir}")
                continue
            
            logger.info(f"Found {len(result_files)} result files in {input_dir}")
            
            # Load each file
            for result_file in result_files:
                file_results = load_results(result_file)
                results.update(file_results)
    
    if args.input_files:
        for input_file in args.input_files:
            if not os.path.exists(input_file):
                logger.warning(f"File not found: {input_file}")
                continue
            
            logger.info(f"Loading results from {input_file}")
            file_results = load_results(input_file)
            results.update(file_results)
    
    if not results:
        logger.error("No results loaded. Exiting.")
        sys.exit(1)
    
    logger.info(f"Loaded results for {len(results)} models: {', '.join(results.keys())}")
    
    # Filter models if specified
    if args.models:
        filtered_results = {model: results[model] for model in args.models if model in results}
        if not filtered_results:
            logger.error("No results found for specified models. Exiting.")
            sys.exit(1)
        results = filtered_results
        logger.info(f"Filtered to {len(results)} models: {', '.join(results.keys())}")
    
    # Filter to top-k models by accuracy if specified
    if args.top_k:
        # Generate report to get accuracies
        report_df = generate_report(results)
        
        # Get average accuracies
        avg_accuracies = report_df[report_df['Task'] == 'Average'].sort_values('Accuracy', ascending=False)
        
        # Get top-k models
        top_models = avg_accuracies.head(args.top_k)['Model'].tolist()
        
        # Filter results
        results = {model: results[model] for model in top_models}
        logger.info(f"Filtered to top {len(results)} models by accuracy: {', '.join(results.keys())}")
    
    # Generate output file paths
    report_file = os.path.join(args.output_dir, f"{args.output_prefix}.csv")
    viz_file = os.path.join(args.output_dir, f"{args.output_prefix}.png")
    
    # Generate report
    report_df = generate_report(results, output_file=report_file)
    logger.info(f"Report saved to {report_file}")
    
    # Run visualizations based on mode
    try:
        if args.mode == "basic":
            # Basic visualizations
            visualize_results(
                results,
                output_file=viz_file,
                show=not args.no_show
            )
            logger.info(f"Basic visualizations saved to:")
            logger.info(f"  - Overview: {viz_file}")
            logger.info(f"  - Task breakdown: {viz_file.replace('.png', '_tasks.png')}")
            logger.info(f"Use '--no-show' to disable automatic display of plots")
            
        elif args.mode == "comparative":
            # Task comparison visualization
            compare_file = os.path.join(args.output_dir, f"{args.output_prefix}_task_comparison.png")
            visualize_task_comparisons(
                results,
                output_file=compare_file,
                show=not args.no_show
            )
            logger.info(f"Task comparison visualization saved to {compare_file}")
            logger.info(f"Use '--no-show' to disable automatic display of plots")
            
        elif args.mode == "correlations":
            # Correlation visualization
            corr_file = os.path.join(args.output_dir, f"{args.output_prefix}_correlations.png")
            visualize_correlation(
                results,
                output_file=corr_file,
                show=not args.no_show
            )
            logger.info(f"Correlation visualization saved to {corr_file}")
            logger.info(f"Use '--no-show' to disable automatic display of plots")
            
        elif args.mode == "prompt":
            # Check that model is specified for prompt strategy analysis
            if not args.models or len(args.models) != 1:
                logger.error("Prompt strategy analysis requires exactly one model to be specified using --models.")
                sys.exit(1)
                
            model = args.models[0]
            
            # Prompt strategy visualization
            prompt_file = os.path.join(args.output_dir, f"{args.output_prefix}_prompt_strategies.png")
            visualize_prompt_strategies(
                results,
                model_name=model,
                output_file=prompt_file,
                show=not args.no_show
            )
            logger.info(f"Prompt strategy visualization saved to {prompt_file}")
            logger.info(f"Use '--no-show' to disable automatic display of plots")
            
        elif args.mode == "scaling":
            # Check that model family and sizes are specified for scaling analysis
            if not args.model_family:
                logger.error("Scaling analysis requires --model-family to be specified.")
                sys.exit(1)
                
            if not args.model_sizes:
                logger.error("Scaling analysis requires --model-sizes to be specified.")
                sys.exit(1)
                
            # Model scaling visualization
            scaling_file = os.path.join(args.output_dir, f"{args.output_prefix}_scaling.png")
            visualize_model_scaling(
                results,
                model_family=args.model_family,
                model_sizes=args.model_sizes,
                output_file=scaling_file,
                show=not args.no_show
            )
            logger.info(f"Model scaling visualization saved to {scaling_file}")
            logger.info(f"Use '--no-show' to disable automatic display of plots")
            
    except Exception as e:
        logger.error(f"Error during visualization: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Analysis completed successfully.")


def main():
    """
    Entry point for the analyze command.
    """
    import logging
    analyze_command()


if __name__ == "__main__":
    main()
