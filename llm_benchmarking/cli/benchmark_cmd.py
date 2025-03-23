"""
Benchmark command module for LLM Uncertainty Benchmarking.
"""

import os
import sys
import argparse
import time
import logging
from typing import List, Dict, Any, Optional

from ..benchmark import LLMBenchmark, ParallelBenchmark
from ..utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def benchmark_command():
    """
    Main function for the benchmark command.
    """
    parser = argparse.ArgumentParser(description="Run LLM Uncertainty Benchmark")
    
    # API connection options
    parser.add_argument("--api-base", required=True, help="Base URL for the OpenAI-compatible API")
    parser.add_argument("--api-key", help="API key (if required)")
    
    # Model options
    parser.add_argument("--model", required=True, help="Name of the model to benchmark")
    parser.add_argument("--chat", action="store_true", help="Use chat template for instruction-tuned models")
    
    # Task options
    task_choices = ["qa", "rc", "ci", "drs", "ds"]
    parser.add_argument("--tasks", nargs="+", choices=task_choices, help="Tasks to evaluate (if not specified, all tasks will be used)")
    
    # Prompt options
    prompt_choices = ["base", "shared_instruction", "task_specific"]
    parser.add_argument("--prompt-strategies", nargs="+", choices=prompt_choices, help="Prompt strategies to use (if not specified, all will be used)")
    
    # Sampling options
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of samples per task")
    parser.add_argument("--calibration-ratio", type=float, default=0.5, help="Ratio of data to use for calibration")
    parser.add_argument("--error-rate", type=float, default=0.1, help="Error rate alpha for conformal prediction")
    
    # Parallelization options
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of samples per batch for parallel processing")
    parser.add_argument("--max-workers", type=int, default=5, help="Maximum number of concurrent workers for parallel processing")
    parser.add_argument("--use-threads", action="store_true", help="Use thread-based parallelization instead of async")
    parser.add_argument("--timeout", type=float, default=60.0, help="Timeout for API requests in seconds")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for failed requests")
    parser.add_argument("--retry-delay", type=float, default=1.0, help="Initial delay between retries in seconds")
    
    # Output options
    parser.add_argument("--output-dir", default="./benchmark_results", help="Output directory for results")
    parser.add_argument("--log-file", help="Path to save log output (if not specified, logs will only be written to console)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level, log_file=args.log_file)
    
    # Print configuration
    logger.info("LLM Uncertainty Benchmark Configuration:")
    logger.info(f"  - API Base URL: {args.api_base}")
    logger.info(f"  - Model: {args.model}")
    logger.info(f"  - Chat template: {'Yes' if args.chat else 'No'}")
    logger.info(f"  - Tasks: {args.tasks if args.tasks else 'All'}")
    logger.info(f"  - Prompt strategies: {args.prompt_strategies if args.prompt_strategies else 'All'}")
    logger.info(f"  - Sample size: {args.sample_size}")
    logger.info(f"  - Calibration ratio: {args.calibration_ratio}")
    logger.info(f"  - Error rate: {args.error_rate}")
    logger.info(f"  - Parallel processing: {'Yes' if args.parallel else 'No'}")
    if args.parallel:
        logger.info(f"  - Batch size: {args.batch_size}")
        logger.info(f"  - Max workers: {args.max_workers}")
        logger.info(f"  - Parallelization: {'Thread-based' if args.use_threads else 'Async-based'}")
        logger.info(f"  - Timeout: {args.timeout}s")
        logger.info(f"  - Max retries: {args.max_retries}")
        logger.info(f"  - Retry delay: {args.retry_delay}s")
    logger.info(f"  - Output directory: {args.output_dir}")
    
    # Run benchmark
    start_time = time.time()
    
    try:
        # Create benchmark instance
        if args.parallel:
            benchmark = ParallelBenchmark(
                api_base_url=args.api_base,
                api_key=args.api_key,
                calibration_ratio=args.calibration_ratio,
                error_rate=args.error_rate,
                batch_size=args.batch_size,
                max_workers=args.max_workers,
                use_async=not args.use_threads,
                timeout=args.timeout,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay
            )
        else:
            benchmark = LLMBenchmark(
                api_base_url=args.api_base,
                api_key=args.api_key,
                calibration_ratio=args.calibration_ratio,
                error_rate=args.error_rate
            )
        
        # Prepare datasets
        benchmark.prepare_datasets(tasks=args.tasks, sample_size=args.sample_size)
        
        # Evaluate model
        benchmark.evaluate_model(
            model_name=args.model,
            tasks=args.tasks,
            prompt_strategies=args.prompt_strategies,
            use_chat_template=args.chat
        )
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        model_dir = os.path.join(args.output_dir, args.model.replace("/", "_"))
        benchmark.save_results(model_dir)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logger.info(f"Benchmark results for {args.model} saved to {model_dir}")
        logger.info(f"Total benchmark time: {elapsed_time:.2f} seconds")
        
        # Print summary report
        report = benchmark.generate_report()
        print("\nBenchmark Summary Report:")
        print(report)
        
    except Exception as e:
        logger.error(f"Error during benchmarking: {e}", exc_info=True)
        sys.exit(1)


def main():
    """
    Entry point for the benchmark command.
    """
    import logging
    benchmark_command()


if __name__ == "__main__":
    main()
