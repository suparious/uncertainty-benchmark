
import os
import argparse
from main import LLMUncertaintyBenchmark

def benchmark_single_model(api_base_url, api_key, model_name, output_dir, sample_size=100):
    """
    Benchmark a single model with a small sample size and save results.
    """
    print(f"Benchmarking {model_name} with reduced sample size of {sample_size} samples per task")
    
    benchmark = LLMUncertaintyBenchmark(
        api_base_url=api_base_url,
        api_key=api_key
    )
    
    # Prepare datasets for all tasks with smaller sample size
    benchmark.prepare_datasets(sample_size=sample_size)
    
    # Evaluate the model with just the base prompt strategy to save time
    is_chat_model = "chat" in model_name.lower() or "instruct" in model_name.lower()
    benchmark.evaluate_model(
        model_name=model_name,
        use_chat_template=is_chat_model,
        prompt_strategies=["base"]  # Only use base prompt to simplify testing
    )
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, model_name.replace("/", "_"))
    benchmark.save_results(model_dir)
    
    print(f"Benchmark results for {model_name} saved to {model_dir}")
    
    # Generate and display a simple report
    report = benchmark.generate_report()
    print("\nSummary Report:")
    print(report)

def compare_models(api_base_url, api_key, model_names, output_dir, sample_size=100):
    """
    Compare multiple models with a small sample size.
    """
    print(f"Comparing models: {', '.join(model_names)} with {sample_size} samples per task")
    
    benchmark = LLMUncertaintyBenchmark(
        api_base_url=api_base_url,
        api_key=api_key
    )
    
    # Prepare datasets for all tasks with smaller sample size
    benchmark.prepare_datasets(sample_size=sample_size)
    
    # Evaluate each model using just the base prompt strategy
    for model_name in model_names:
        print(f"\nEvaluating {model_name}...")
        is_chat_model = "chat" in model_name.lower() or "instruct" in model_name.lower()
        benchmark.evaluate_model(
            model_name=model_name,
            use_chat_template=is_chat_model,
            prompt_strategies=["base"]  # Only use base prompt to simplify testing
        )
    
    # Save all results
    os.makedirs(output_dir, exist_ok=True)
    benchmark.save_results(output_dir)
    
    # Generate comparison report and visualization
    comparison_file = os.path.join(output_dir, "model_comparison.csv")
    benchmark.generate_report(output_file=comparison_file)
    
    visualization_file = os.path.join(output_dir, "model_comparison.png")
    benchmark.visualize_results(output_file=visualization_file)
    
    # Display a simple report
    report = benchmark.generate_report()
    print("\nComparison Report:")
    print(report)
    
    print(f"\nComparison results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Uncertainty Benchmark Examples (Small Sample Size)")
    parser.add_argument("--api-base", required=True, help="Base URL for the OpenAI-compatible API")
    parser.add_argument("--api-key", help="API key (if required)")
    parser.add_argument("--output-dir", default="./benchmark_results", help="Output directory for results")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of samples per task")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Single model benchmark
    single_parser = subparsers.add_parser("single", help="Benchmark a single model")
    single_parser.add_argument("--model", required=True, help="Name of the model to benchmark")
    
    # Model comparison
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument("--models", required=True, nargs="+", help="Names of models to compare")
    
    args = parser.parse_args()
    
    if args.command == "single":
        benchmark_single_model(args.api_base, args.api_key, args.model, args.output_dir, args.sample_size)
    
    elif args.command == "compare":
        compare_models(args.api_base, args.api_key, args.models, args.output_dir, args.sample_size)
    
    else:
        parser.print_help()
