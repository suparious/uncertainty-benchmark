"""
Parallel benchmarking module for LLM Uncertainty Benchmarking.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Union

from .core import LLMBenchmark
from .metrics import calculate_metrics_with_conformal_prediction
from ..utils.logging import get_logger
from ..utils.parallel import ParallelProcessor, ThreadedProcessor

logger = get_logger(__name__)


class ParallelBenchmark(LLMBenchmark):
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
        
        # Check if we have valid calibration and test data
        if not calibration_results or not test_results:
            logger.warning(f"No valid calibration or test data.")
            # Return empty results
            lac_results = {'acc': 0, 'cr': 0, 'ss': 0}
            aps_results = {'acc': 0, 'cr': 0, 'ss': 0}
        else:
            # Calculate metrics using LAC conformal score function
            try:
                lac_results = calculate_metrics_with_conformal_prediction(
                    calibration_results, test_results, score_function="lac", error_rate=self.error_rate
                )
            except Exception as e:
                logger.error(f"Error calculating LAC metrics: {str(e)}")
                lac_results = {'acc': 0, 'cr': 0, 'ss': 0}
            
            # Calculate metrics using APS conformal score function
            try:
                aps_results = calculate_metrics_with_conformal_prediction(
                    calibration_results, test_results, score_function="aps", error_rate=self.error_rate
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
