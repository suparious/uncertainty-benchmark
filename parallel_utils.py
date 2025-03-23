"""
Utilities for parallel processing in the LLM Uncertainty Benchmark.
This module provides functionality for making concurrent API requests
and processing batches of samples in parallel.
"""

import asyncio
import aiohttp
import logging
import numpy as np
import time
from typing import List, Dict, Any, Callable, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.asyncio import tqdm as async_tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ParallelProcessor:
    """
    Class for processing samples in parallel using multiple concurrent API requests.
    """
    
    def __init__(
        self, 
        api_base_url: str, 
        api_key: Optional[str] = None,
        batch_size: int = 10,
        max_workers: int = 5,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 30.0
    ):
        """
        Initialize the parallel processor.
        
        Args:
            api_base_url: Base URL for the API
            api_key: API key for authentication
            batch_size: Number of samples to process in a batch
            max_workers: Maximum number of concurrent workers
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            timeout: Timeout for API requests in seconds
        """
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
    
    def process_samples(
        self, 
        samples: List[Dict], 
        model_name: str,
        prompt_formatter: Callable[[Dict], str],
        use_chat_template: bool = False,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Process a list of samples in parallel using batched API requests.
        
        Args:
            samples: List of samples to process
            model_name: Name of the model to use
            prompt_formatter: Function to format samples into prompts
            use_chat_template: Whether to use chat template
            show_progress: Whether to show progress bar
            
        Returns:
            List of processed samples with logits
        """
        # Split samples into batches
        batches = [samples[i:i+self.batch_size] for i in range(0, len(samples), self.batch_size)]
        
        # Process batches in parallel using asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                self._process_batches_async(
                    batches, 
                    model_name, 
                    prompt_formatter, 
                    use_chat_template, 
                    show_progress
                )
            )
        finally:
            loop.close()
        
        # Flatten results
        processed_samples = []
        for batch_results in results:
            processed_samples.extend(batch_results)
        
        return processed_samples
    
    async def _process_batches_async(
        self, 
        batches: List[List[Dict]], 
        model_name: str,
        prompt_formatter: Callable[[Dict], str],
        use_chat_template: bool,
        show_progress: bool
    ) -> List[List[Dict]]:
        """
        Process batches asynchronously.
        
        Args:
            batches: List of batches of samples
            model_name: Name of the model to use
            prompt_formatter: Function to format samples into prompts
            use_chat_template: Whether to use chat template
            show_progress: Whether to show progress bar
            
        Returns:
            List of lists of processed samples
        """
        async with aiohttp.ClientSession() as session:
            # Create tasks for each batch
            tasks = []
            for batch in batches:
                task = self._process_batch_async(
                    session, 
                    batch, 
                    model_name, 
                    prompt_formatter, 
                    use_chat_template
                )
                tasks.append(task)
            
            # Process tasks with progress bar if requested
            if show_progress:
                results = await async_tqdm.gather(
                    *tasks,
                    desc="Processing samples",
                    total=len(batches)
                )
            else:
                results = await asyncio.gather(*tasks)
            
            return results
    
    async def _process_batch_async(
        self, 
        session: aiohttp.ClientSession, 
        batch: List[Dict], 
        model_name: str,
        prompt_formatter: Callable[[Dict], str],
        use_chat_template: bool
    ) -> List[Dict]:
        """
        Process a batch of samples asynchronously.
        
        Args:
            session: aiohttp ClientSession
            batch: Batch of samples
            model_name: Name of the model to use
            prompt_formatter: Function to format samples into prompts
            use_chat_template: Whether to use chat template
            
        Returns:
            List of processed samples
        """
        # Format prompts for the batch
        formatted_prompts = [prompt_formatter(sample) for sample in batch]
        
        # Create a list of API request tasks
        tasks = []
        for i, prompt in enumerate(formatted_prompts):
            task = self._get_logits_async(
                session, 
                model_name, 
                prompt, 
                use_chat_template
            )
            tasks.append((i, task))
        
        # Process API requests concurrently
        processed_batch = [None] * len(batch)
        for i, task in tasks:
            try:
                logits = await task
                
                # Apply softmax to get probabilities
                softmax_probs = self._softmax(logits)
                
                # Store processed sample
                processed_batch[i] = {
                    'item': batch[i],
                    'logits': logits,
                    'softmax': softmax_probs
                }
            except Exception as e:
                logger.error(f"Error processing sample {batch[i].get('id', i)}: {e}")
        
        # Filter out failed samples
        processed_batch = [sample for sample in processed_batch if sample is not None]
        
        return processed_batch
    
    async def _get_logits_async(
        self, 
        session: aiohttp.ClientSession, 
        model_name: str, 
        prompt: str, 
        use_chat_template: bool
    ) -> List[float]:
        """
        Get logits from the API asynchronously.
        
        Args:
            session: aiohttp ClientSession
            model_name: Name of the model
            prompt: The formatted prompt
            use_chat_template: Whether to use chat template
            
        Returns:
            List of logits for the options A-F
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Prepare request data
        if use_chat_template:
            # Format as a chat message for instruction-tuned models
            data = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 10
            }
        else:
            # Use standard completion for base models
            data = {
                "model": model_name,
                "prompt": prompt,
                "temperature": 0.0,
                "max_tokens": 1,
                "logprobs": 10,
                "echo": False
            }
        
        # Send request with retries
        for attempt in range(self.max_retries):
            try:
                async with session.post(
                    f"{self.api_base_url}/completions",
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    # Extract logits/logprobs for options A-F
                    if not use_chat_template:
                        logprobs = result["choices"][0]["logprobs"]["top_logprobs"][0]
                        
                        # Extract logits for A, B, C, D, E, F
                        option_logits = []
                        for option in ["A", "B", "C", "D", "E", "F"]:
                            if option in logprobs:
                                option_logits.append(logprobs[option])
                            else:
                                # If option not in top logprobs, use a very low value
                                option_logits.append(-100.0)
                    else:
                        # Logic for chat models - adjust based on actual API response
                        logprobs = result["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                        
                        option_logits = []
                        for option in ["A", "B", "C", "D", "E", "F"]:
                            option_logit = next((lp["logprob"] for lp in logprobs if lp["token"] == option), -100.0)
                            option_logits.append(option_logit)
                    
                    return option_logits
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    # Wait before retrying
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    # Log and raise on final attempt
                    logger.error(f"Error getting logits from API after {self.max_retries} attempts: {e}")
                    raise
    
    def _softmax(self, logits: List[float]) -> List[float]:
        """Apply softmax to logits."""
        exp_logits = [np.exp(x) for x in logits]
        sum_exp_logits = sum(exp_logits)
        return [x / sum_exp_logits for x in exp_logits]


class ThreadedProcessor:
    """
    Class for processing samples in parallel using thread-based concurrency.
    This is an alternative to the async-based ParallelProcessor and may work
    better in some environments or for some APIs.
    """
    
    def __init__(
        self, 
        api_base_url: str, 
        api_key: Optional[str] = None,
        batch_size: int = 10,
        max_workers: int = 5,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 30.0
    ):
        """
        Initialize the threaded processor.
        
        Args:
            api_base_url: Base URL for the API
            api_key: API key for authentication
            batch_size: Number of samples to process in a batch
            max_workers: Maximum number of concurrent workers
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            timeout: Timeout for API requests in seconds
        """
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
    
    def process_samples(
        self, 
        samples: List[Dict], 
        model_name: str,
        prompt_formatter: Callable[[Dict], str],
        use_chat_template: bool = False,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Process a list of samples in parallel using thread-based concurrency.
        
        Args:
            samples: List of samples to process
            model_name: Name of the model to use
            prompt_formatter: Function to format samples into prompts
            use_chat_template: Whether to use chat template
            show_progress: Whether to show progress bar
            
        Returns:
            List of processed samples with logits
        """
        # Split samples into batches
        batches = [samples[i:i+self.batch_size] for i in range(0, len(samples), self.batch_size)]
        
        # Process batches using ThreadPoolExecutor
        processed_samples = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit batch processing tasks
            futures = []
            for batch in batches:
                future = executor.submit(
                    self._process_batch, 
                    batch, 
                    model_name, 
                    prompt_formatter, 
                    use_chat_template
                )
                futures.append(future)
            
            # Process completed futures with progress bar if requested
            if show_progress:
                from tqdm import tqdm
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                    batch_results = future.result()
                    processed_samples.extend(batch_results)
            else:
                for future in as_completed(futures):
                    batch_results = future.result()
                    processed_samples.extend(batch_results)
        
        return processed_samples
    
    def _process_batch(
        self, 
        batch: List[Dict], 
        model_name: str,
        prompt_formatter: Callable[[Dict], str],
        use_chat_template: bool
    ) -> List[Dict]:
        """
        Process a batch of samples.
        
        Args:
            batch: Batch of samples
            model_name: Name of the model to use
            prompt_formatter: Function to format samples into prompts
            use_chat_template: Whether to use chat template
            
        Returns:
            List of processed samples
        """
        processed_batch = []
        
        for sample in batch:
            try:
                # Format prompt
                prompt = prompt_formatter(sample)
                
                # Get logits
                logits = self._get_logits(model_name, prompt, use_chat_template)
                
                # Apply softmax to get probabilities
                softmax_probs = self._softmax(logits)
                
                # Store processed sample
                processed_batch.append({
                    'item': sample,
                    'logits': logits,
                    'softmax': softmax_probs
                })
            except Exception as e:
                logger.error(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
        
        return processed_batch
    
    def _get_logits(self, model_name: str, prompt: str, use_chat_template: bool) -> List[float]:
        """
        Get logits from the API with retries.
        
        Args:
            model_name: Name of the model
            prompt: The formatted prompt
            use_chat_template: Whether to use chat template
            
        Returns:
            List of logits for the options A-F
        """
        import requests
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Prepare request data
        if use_chat_template:
            # Format as a chat message for instruction-tuned models
            data = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 10
            }
        else:
            # Use standard completion for base models
            data = {
                "model": model_name,
                "prompt": prompt,
                "temperature": 0.0,
                "max_tokens": 1,
                "logprobs": 10,
                "echo": False
            }
        
        # Send request with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.api_base_url}/completions",
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
                
                # Extract logits/logprobs for options A-F
                if not use_chat_template:
                    logprobs = result["choices"][0]["logprobs"]["top_logprobs"][0]
                    
                    # Extract logits for A, B, C, D, E, F
                    option_logits = []
                    for option in ["A", "B", "C", "D", "E", "F"]:
                        if option in logprobs:
                            option_logits.append(logprobs[option])
                        else:
                            # If option not in top logprobs, use a very low value
                            option_logits.append(-100.0)
                else:
                    # Logic for chat models - adjust based on actual API response
                    logprobs = result["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                    
                    option_logits = []
                    for option in ["A", "B", "C", "D", "E", "F"]:
                        option_logit = next((lp["logprob"] for lp in logprobs if lp["token"] == option), -100.0)
                        option_logits.append(option_logit)
                
                return option_logits
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    # Wait before retrying
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    # Log and raise on final attempt
                    logger.error(f"Error getting logits from API after {self.max_retries} attempts: {e}")
                    raise
    
    def _softmax(self, logits: List[float]) -> List[float]:
        """Apply softmax to logits."""
        exp_logits = [np.exp(x) for x in logits]
        sum_exp_logits = sum(exp_logits)
        return [x / sum_exp_logits for x in exp_logits]


def parallel_map(
    func: Callable, 
    items: List[Any], 
    max_workers: int = None, 
    show_progress: bool = True,
    desc: str = "Processing"
) -> List[Any]:
    """
    Apply a function to each item in a list in parallel.
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        max_workers: Maximum number of workers (default: None, uses CPU count)
        show_progress: Whether to show a progress bar
        desc: Description for the progress bar
        
    Returns:
        List of results
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        futures = {executor.submit(func, item): i for i, item in enumerate(items)}
        
        # Process completed tasks
        if show_progress:
            from tqdm import tqdm
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                except Exception as e:
                    logger.error(f"Error processing item {idx}: {e}")
        else:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                except Exception as e:
                    logger.error(f"Error processing item {idx}: {e}")
    
    # Sort results by original index
    results.sort(key=lambda x: x[0])
    
    # Extract just the results
    return [r[1] for r in results]
