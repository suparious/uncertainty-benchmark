#!/usr/bin/env python
"""
Basic example of using the LLM Uncertainty Benchmark.
"""

import os
import argparse
from llm_benchmarking.benchmark import LLMBenchmark


def benchmark_model(api_base_url, api_key, model_name, use_chat_template, output_dir):
    """
    Run a benchmark on a single model.
    
    Args:
        api_base_url: Base URL for the API
        api_key: API key (if required)
        model_name: Name of the model to benchmark
        use_chat_template: Whether to use chat template
        output_dir: Directory to save results to
    """
    print(f"Benchmarking {model_name}...")
    
    # Create benchmark instance
    benchmark = LLMBenchmark(
        api_base_url=api_base_url,
        api_key=api_key
    )
    
    # Prepare datasets (use a smaller sample size for demonstration)
    benchmark.prepare_datasets(sample_size=100)
    
    # Evaluate the model
    benchmark.evaluate_model(
        model_name=model_name,
        use_chat_template=use_chat_template
    )
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, model_name.replace("/", "_"))
    benchmark.save_results(model_dir)
    
    print(f"Benchmark results for {model_name} saved to {model_dir}")
    
    # Generate and print report
    report = benchmark.generate_report()
    print("\nBenchmark Report:")
    print(report)
    
    # Visualize results
    benchmark.visualize_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic LLM Uncertainty Benchmark Example")
    parser.add_argument("--api-base", required=True, help="Base URL for the OpenAI-compatible API")
    parser.add_argument("--api-key", help="API key (if required)")
    parser.add_argument("--model", required=True, help="Name of the model to benchmark")
    parser.add_argument("--chat", action="store_true", help="Use chat template for instruction-tuned models")
    parser.add_argument("--output-dir", default="./benchmark_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    benchmark_model(
        api_base_url=args.api_base,
        api_key=args.api_key,
        model_name=args.model,
        use_chat_template=args.chat,
        output_dir=args.output_dir
    )
