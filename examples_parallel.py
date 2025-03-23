import os
import argparse
import time
import logging
import sys
from typing import List, Dict, Any
from main import LLMUncertaintyBenchmark
from parallel_utils import ParallelProcessor, ThreadedProcessor, parallel_map

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ParallelBenchmark(LLMUncertaintyBenchmark):
    """
    Extended benchmark class that adds parallel processing capabilities.
    """
    
    def __init__(
        self, 
        api_base_url: str,
        api_key: str = None,
        calibration_ratio: float = 0.5,
        error_rate: float = 0.1,
        max_tokens: int = 2048,
        num_demonstrations: Dict[str, int] = None,
        batch_size: int = 10,
        max_workers: int = 5,
        use_async: bool = True,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the parallel benchmark.
        
        Args:
            api_base_url: Base URL for the OpenAI-compatible API
            api_key: API key for authentication (if required)
            calibration_ratio: Ratio of data to use for calibration
            error_rate: Error rate alpha for conformal prediction
            max_tokens: Maximum number of tokens for input context
            num_demonstrations: Number of demonstrations for each task
            batch_size: Number of samples to process in each batch
            max_workers: Maximum number of parallel workers
            use_async: Whether to use async-based or thread-based parallelization
            timeout: Timeout for API requests in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Initial delay between retries in seconds
        """
        super().__init__(
            api_base_url=api_base_url,
            api_key=api_key,
            calibration_ratio=calibration_ratio,
            error_rate=error_rate,
            max_tokens=max_tokens,
            num_demonstrations=num_demonstrations
        )
        
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.use_async = use_async
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        logger.info(f"Initializing {'Async' if use_async else 'Threaded'} parallel processor with:")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Max workers: {max_workers}")
        logger.info(f"  - Timeout: {timeout}s")
        logger.info(f"  - Max retries: {max_retries}")
        logger.info(f"  - Retry delay: {retry_delay}s")
        
        # Initialize parallel processor
        if use_async:
            self.processor = ParallelProcessor(
                api_base_url=api_base_url,
                api_key=api_key,
                batch_size=batch_size,
                max_workers=max_workers,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay
            )
        else:
            self.processor = ThreadedProcessor(
                api_base_url=api_base_url,
                api_key=api_key,
                batch_size=batch_size,
                max_workers=max_workers,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay
            )
    
    def evaluate_model(
        self, 
        model_name: str, 
        tasks: List[str] = None, 
        prompt_strategies: List[str] = None, 
        use_chat_template: bool = False,
        temperature: float = 0.0
    ) -> Dict:
        """
        Evaluate a model on the benchmark using parallel processing.
        
        Args:
            model_name: Name of the model to evaluate
            tasks: List of tasks to evaluate on. If None, use all.
            prompt_strategies: List of prompt strategies to use. If None, use all three.
            use_chat_template: Whether to use the chat template for instruction-tuned models
            temperature: Temperature for sampling (set to 0 for deterministic outputs)
            
        Returns:
            Dictionary with evaluation results
        """
        if tasks is None:
            tasks = list(self.tasks.keys())
        
        if prompt_strategies is None:
            prompt_strategies = ["base", "shared_instruction", "task_specific"]
        
        # Initialize results dictionary
        self.results[model_name] = {}
        
        # Process tasks sequentially (safer than parallel tasks)
        for task in tasks:
            logger.info(f"Evaluating {model_name} on {task} task...")
            self.results[model_name][task] = self._evaluate_task(
                model_name, task, prompt_strategies, use_chat_template, temperature
            )
        
        # Calculate overall average across tasks
        overall_avg = {
            'acc': sum(self.results[model_name][task]['avg']['acc'] for task in tasks) / len(tasks),
            'cr': sum(self.results[model_name][task]['avg']['cr'] for task in tasks) / len(tasks),
            'ss': sum(self.results[model_name][task]['avg']['ss'] for task in tasks) / len(tasks)
        }
        
        self.results[model_name]['overall'] = overall_avg
        
        logger.info(f"Overall results for {model_name}: Acc={overall_avg['acc']:.4f}, CR={overall_avg['cr']:.4f}, SS={overall_avg['ss']:.4f}")
        
        return self.results[model_name]
    
    def _evaluate_task(
        self,
        model_name: str,
        task: str,
        prompt_strategies: List[str],
        use_chat_template: bool,
        temperature: float
    ) -> Dict:
        """
        Evaluate a model on a specific task.
        
        Args:
            model_name: Name of the model to evaluate
            task: Task to evaluate on
            prompt_strategies: List of prompt strategies to use
            use_chat_template: Whether to use the chat template for instruction-tuned models
            temperature: Temperature for sampling
            
        Returns:
            Dictionary with task results
        """
        # Check if dataset is prepared
        if task not in self.datasets:
            logger.info(f"Dataset for {task} not prepared. Preparing now...")
            self.prepare_datasets([task])
        
        task_results = {}
        
        # Process prompt strategies sequentially for better reliability
        for prompt_strategy in prompt_strategies:
            logger.info(f"Using {prompt_strategy} prompt strategy...")
            task_results[prompt_strategy] = self._evaluate_with_prompt_strategy(
                model_name, task, prompt_strategy, use_chat_template, temperature
            )
        
        # Average results across prompt strategies
        avg_across_prompts = {
            'acc': sum(task_results[ps]['avg']['acc'] for ps in prompt_strategies) / len(prompt_strategies),
            'cr': sum(task_results[ps]['avg']['cr'] for ps in prompt_strategies) / len(prompt_strategies),
            'ss': sum(task_results[ps]['avg']['ss'] for ps in prompt_strategies) / len(prompt_strategies)
        }
        
        task_result = {
            'prompt_strategies': task_results,
            'avg': avg_across_prompts
        }
        
        logger.info(f"Results for {model_name} on {task} task: Acc={avg_across_prompts['acc']:.4f}, CR={avg_across_prompts['cr']:.4f}, SS={avg_across_prompts['ss']:.4f}")
        
        return task_result
    
    def _evaluate_with_prompt_strategy(
        self,
        model_name: str,
        task: str,
        prompt_strategy: str,
        use_chat_template: bool,
        temperature: float
    ) -> Dict:
        """
        Evaluate a model with a specific prompt strategy.
        
        Args:
            model_name: Name of the model to evaluate
            task: Task to evaluate on
            prompt_strategy: Prompt strategy to use
            use_chat_template: Whether to use the chat template for instruction-tuned models
            temperature: Temperature for sampling
            
        Returns:
            Dictionary with strategy results
        """
        # Get datasets
        calibration_data = self.datasets[task]['calibration']
        test_data = self.datasets[task]['test']
        
        # Create prompt formatter function
        def prompt_formatter(item):
            return self.format_prompt(task, item, prompt_strategy)
        
        # Process calibration set with proper error handling
        logger.info(f"Processing calibration set for {task} with {prompt_strategy} strategy...")
        try:
            calibration_results = self.processor.process_samples(
                calibration_data,
                model_name,
                prompt_formatter,
                use_chat_template,
                show_progress=True
            )
            
            if not calibration_results:
                logger.error(f"No valid results obtained from calibration set for {task} with {prompt_strategy} strategy. Skipping...")
                return {
                    'lac': {'acc': 0, 'cr': 0, 'ss': 0},
                    'aps': {'acc': 0, 'cr': 0, 'ss': 0},
                    'avg': {'acc': 0, 'cr': 0, 'ss': 0}
                }
                
            logger.info(f"Successfully processed {len(calibration_results)}/{len(calibration_data)} calibration samples")
        except Exception as e:
            logger.error(f"Error processing calibration set: {str(e)}")
            return {
                'lac': {'acc': 0, 'cr': 0, 'ss': 0},
                'aps': {'acc': 0, 'cr': 0, 'ss': 0},
                'avg': {'acc': 0, 'cr': 0, 'ss': 0}
            }
        
        # Process test set with proper error handling
        logger.info(f"Processing test set for {task} with {prompt_strategy} strategy...")
        try:
            test_results = self.processor.process_samples(
                test_data,
                model_name,
                prompt_formatter,
                use_chat_template,
                show_progress=True
            )
            
            if not test_results:
                logger.error(f"No valid results obtained from test set for {task} with {prompt_strategy} strategy. Skipping...")
                return {
                    'lac': {'acc': 0, 'cr': 0, 'ss': 0},
                    'aps': {'acc': 0, 'cr': 0, 'ss': 0},
                    'avg': {'acc': 0, 'cr': 0, 'ss': 0}
                }
                
            logger.info(f"Successfully processed {len(test_results)}/{len(test_data)} test samples")
        except Exception as e:
            logger.error(f"Error processing test set: {str(e)}")
            return {
                'lac': {'acc': 0, 'cr': 0, 'ss': 0},
                'aps': {'acc': 0, 'cr': 0, 'ss': 0},
                'avg': {'acc': 0, 'cr': 0, 'ss': 0}
            }
        
        # Calculate metrics using LAC conformal score function
        try:
            lac_results = self._calculate_metrics_with_conformal_prediction(
                calibration_results, test_results, score_function="lac"
            )
        except Exception as e:
            logger.error(f"Error calculating LAC metrics: {str(e)}")
            lac_results = {'acc': 0, 'cr': 0, 'ss': 0}
        
        # Calculate metrics using APS conformal score function
        try:
            aps_results = self._calculate_metrics_with_conformal_prediction(
                calibration_results, test_results, score_function="aps"
            )
        except Exception as e:
            logger.error(f"Error calculating APS metrics: {str(e)}")
            aps_results = {'acc': 0, 'cr': 0, 'ss': 0}
        
        # Average the results
        avg_results = {
            'acc': lac_results.get('acc', 0),  # Same for both
            'cr': (lac_results.get('cr', 0) + aps_results.get('cr', 0)) / 2,
            'ss': (lac_results.get('ss', 0) + aps_results.get('ss', 0)) / 2
        }
        
        return {
            'lac': lac_results,
            'aps': aps_results,
            'avg': avg_results
        }


def benchmark_single_model(
    api_base_url, 
    api_key, 
    model_name, 
    output_dir, 
    batch_size=10, 
    max_workers=5,
    use_async=True,
    sample_size=10000,
    timeout=60.0,
    max_retries=3,
    retry_delay=1.0,
    task=None
):
    """
    Benchmark a single model using parallel processing and save results.
    
    Args:
        api_base_url: Base URL for the API
        api_key: API key (if required)
        model_name: Name of the model to benchmark
        output_dir: Directory to save results to
        batch_size: Number of samples per batch
        max_workers: Maximum number of parallel workers
        use_async: Whether to use async-based or thread-based parallelization
        sample_size: Number of samples per task
        timeout: Timeout for API requests in seconds
        max_retries: Maximum number of retries for failed requests
        retry_delay: Initial delay between retries in seconds
        task: Optional specific task to evaluate (if None, evaluates all tasks)
    """
    start_time = time.time()
    
    logger.info(f"Benchmarking {model_name} using {'async' if use_async else 'threaded'} parallel processing")
    logger.info(f"Parameters: batch_size={batch_size}, max_workers={max_workers}, timeout={timeout}s, max_retries={max_retries}")
    
    benchmark = ParallelBenchmark(
        api_base_url=api_base_url,
        api_key=api_key,
        batch_size=batch_size,
        max_workers=max_workers,
        use_async=use_async,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay
    )
    
    # Prepare datasets for the specified task or all tasks
    tasks = [task] if task else None
    benchmark.prepare_datasets(tasks=tasks, sample_size=sample_size)
    
    # Evaluate the model
    is_chat_model = any(keyword in model_name.lower() for keyword in ["chat", "instruct", "conv"])
    benchmark.evaluate_model(
        model_name=model_name,
        tasks=tasks,
        use_chat_template=is_chat_model
    )
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, model_name.replace("/", "_"))
    benchmark.save_results(model_dir)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info(f"Benchmark results for {model_name} saved to {model_dir}")
    logger.info(f"Total benchmark time: {elapsed_time:.2f} seconds")
    
    # Generate and print summary report
    report = benchmark.generate_report()
    print("\nBenchmark Summary Report:")
    print(report)


def compare_models(
    api_base_url, 
    api_key, 
    model_names, 
    output_dir, 
    batch_size=10, 
    max_workers=5,
    use_async=True,
    sample_size=10000,
    timeout=60.0,
    max_retries=3,
    retry_delay=1.0,
    task=None
):
    """
    Compare multiple models using parallel processing and generate comparative reports.
    
    Args:
        api_base_url: Base URL for the API
        api_key: API key (if required)
        model_names: List of model names to compare
        output_dir: Directory to save results to
        batch_size: Number of samples per batch
        max_workers: Maximum number of parallel workers
        use_async: Whether to use async-based or thread-based parallelization
        sample_size: Number of samples per task
        timeout: Timeout for API requests in seconds
        max_retries: Maximum number of retries for failed requests
        retry_delay: Initial delay between retries in seconds
        task: Optional specific task to evaluate (if None, evaluates all tasks)
    """
    start_time = time.time()
    
    logger.info(f"Comparing models: {', '.join(model_names)} using {'async' if use_async else 'threaded'} parallel processing")
    logger.info(f"Parameters: batch_size={batch_size}, max_workers={max_workers}, timeout={timeout}s, max_retries={max_retries}")
    
    benchmark = ParallelBenchmark(
        api_base_url=api_base_url,
        api_key=api_key,
        batch_size=batch_size,
        max_workers=max_workers,
        use_async=use_async,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay
    )
    
    # Prepare datasets for the specified task or all tasks
    tasks = [task] if task else None
    benchmark.prepare_datasets(tasks=tasks, sample_size=sample_size)
    
    # Evaluate each model
    for model_name in model_names:
        logger.info(f"Evaluating {model_name}...")
        is_chat_model = any(keyword in model_name.lower() for keyword in ["chat", "instruct", "conv"])
        try:
            benchmark.evaluate_model(
                model_name=model_name,
                tasks=tasks,
                use_chat_template=is_chat_model
            )
        except Exception as e:
            logger.error(f"Failed to evaluate model {model_name}: {str(e)}")
    
    # Save all results
    os.makedirs(output_dir, exist_ok=True)
    benchmark.save_results(output_dir)
    
    # Generate comparison report and visualization
    comparison_file = os.path.join(output_dir, "model_comparison.csv")
    benchmark.generate_report(output_file=comparison_file)
    
    visualization_file = os.path.join(output_dir, "model_comparison.png")
    benchmark.visualize_results(output_file=visualization_file)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info(f"Comparison results saved to {output_dir}")
    logger.info(f"Total comparison time: {elapsed_time:.2f} seconds")
    
    # Generate and print summary report
    report = benchmark.generate_report()
    print("\nComparison Summary Report:")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel LLM Uncertainty Benchmark Examples")
    parser.add_argument("--api-base", required=True, help="Base URL for the OpenAI-compatible API")
    parser.add_argument("--api-key", help="API key (if required)")
    parser.add_argument("--output-dir", default="./benchmark_results", help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of samples per batch (default: 4)")
    parser.add_argument("--max-workers", type=int, default=2, help="Maximum number of parallel workers (default: 2)")
    parser.add_argument("--use-threads", action="store_true", help="Use thread-based parallelization instead of async")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of samples per task")
    parser.add_argument("--timeout", type=float, default=60.0, help="Timeout for API requests in seconds (default: 60)")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for failed requests (default: 3)")
    parser.add_argument("--retry-delay", type=float, default=1.0, help="Initial delay between retries in seconds (default: 1.0)")
    parser.add_argument("--task", help="Optional specific task to evaluate (qa, rc, ci, drs, ds)")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Single model benchmark
    single_parser = subparsers.add_parser("single", help="Benchmark a single model")
    single_parser.add_argument("--model", required=True, help="Name of the model to benchmark")
    
    # Model comparison
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument("--models", required=True, nargs="+", help="Names of models to compare")
    
    args = parser.parse_args()
    
    # Validate task if specified
    task = None
    if args.task:
        valid_tasks = ["qa", "rc", "ci", "drs", "ds"]
        if args.task not in valid_tasks:
            logger.error(f"Invalid task: {args.task}. Valid tasks are: {', '.join(valid_tasks)}")
            sys.exit(1)
        task = args.task
    
    if args.command == "single":
        benchmark_single_model(
            args.api_base, 
            args.api_key, 
            args.model, 
            args.output_dir,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            use_async=not args.use_threads,
            sample_size=args.sample_size,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            task=task
        )
    
    elif args.command == "compare":
        compare_models(
            args.api_base, 
            args.api_key, 
            args.models, 
            args.output_dir,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            use_async=not args.use_threads,
            sample_size=args.sample_size,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            task=task
        )
    
    else:
        parser.print_help()
