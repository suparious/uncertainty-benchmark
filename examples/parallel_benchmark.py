#!/usr/bin/env python
"""
Example of using the Parallel LLM Uncertainty Benchmark.
"""

import os
import time
import argparse
from llm_benchmarking.benchmark import ParallelBenchmark


def benchmark_model_parallel(
    api_base_url, 
    api_key, 
    model_name, 
    use_chat_template, 
    output_dir,
    batch_size=4,
    max_workers=2,
    use_async=True,
    sample_size=100
):
    """
    Run a benchmark on a single model using parallel processing.
    
    Args:
        api_base_url: Base URL for the API
        api_key: API key (if required)
        model_name: Name of the model to benchmark
        use_chat_template: Whether to use chat template
        output_dir: Directory to save results to
        batch_size: Number of samples per batch
        max_workers: Maximum number of concurrent workers
        use_async: Whether to use async-based parallelization
        sample_size: Number of samples per task
    """
    start_time = time.time()
    
    print(f"Benchmarking {model_name} using {'async' if use_async else 'threaded'} parallel processing...")
    print(f"Parameters: batch_size={batch_size}, max_workers={max_workers}, sample_size={sample_size}")
    
    # Create benchmark instance
    benchmark = ParallelBenchmark(
        api_base_url=api_base_url,
        api_key=api_key,
        batch_size=batch_size,
        max_workers=max_workers,
        use_async=use_async
    )
    
    # Prepare datasets
    benchmark.prepare_datasets(sample_size=sample_size)
    
    # Evaluate the model
    benchmark.evaluate_model(
        model_name=model_name,
        use_chat_template=use_chat_template
    )
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, model_name.replace("/", "_"))
    benchmark.save_results(model_dir)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Benchmark results for {model_name} saved to {model_dir}")
    print(f"Total benchmark time: {elapsed_time:.2f} seconds")
    
    # Generate and print report
    report = benchmark.generate_report()
    print("\nBenchmark Report:")
    print(report)
    
    # Visualize results
    benchmark.visualize_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel LLM Uncertainty Benchmark Example")
    parser.add_argument("--api-base", required=True, help="Base URL for the OpenAI-compatible API")
    parser.add_argument("--api-key", help="API key (if required)")
    parser.add_argument("--model", required=True, help="Name of the model to benchmark")
    parser.add_argument("--chat", action="store_true", help="Use chat template for instruction-tuned models")
    parser.add_argument("--output-dir", default="./benchmark_results", help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of samples per batch")
    parser.add_argument("--max-workers", type=int, default=2, help="Maximum number of concurrent workers")
    parser.add_argument("--use-threads", action="store_true", help="Use thread-based parallelization instead of async")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of samples per task")
    
    args = parser.parse_args()
    
    benchmark_model_parallel(
        api_base_url=args.api_base,
        api_key=args.api_key,
        model_name=args.model,
        use_chat_template=args.chat,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        use_async=not args.use_threads,
        sample_size=args.sample_size
    )
