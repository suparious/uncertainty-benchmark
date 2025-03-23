"""
Parallel processing utilities for LLM Uncertainty Benchmarking.
"""

import asyncio
import aiohttp
import json
import time
import logging
import numpy as np
import requests
from typing import List, Dict, Any, Callable, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.asyncio import tqdm as async_tqdm

from .logging import get_logger
from .api import softmax

logger = get_logger(__name__)


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
        timeout: float = 60.0
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
        # Configure connection timeout options
        tcp_connector = aiohttp.TCPConnector(
            limit=self.max_workers,
            ssl=False,
            force_close=True,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(
            total=self.timeout,
            connect=self.timeout / 3,
            sock_connect=self.timeout / 3,
            sock_read=self.timeout
        )
        
        async with aiohttp.ClientSession(
            connector=tcp_connector,
            timeout=timeout
        ) as session:
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
        
        # Process each prompt individually
        # This is more reliable than gathering all tasks at once
        processed_batch = []
        
        for i, prompt in enumerate(formatted_prompts):
            try:
                sample = batch[i]
                logits = await self._get_logits_async(
                    session, 
                    model_name, 
                    prompt, 
                    use_chat_template,
                    sample.get('id', f"batch_{i}")
                )
                
                if logits:
                    # Apply softmax to get probabilities
                    softmax_probs = softmax(logits)
                    
                    # Store processed sample
                    processed_batch.append({
                        'item': sample,
                        'logits': logits,
                        'softmax': softmax_probs
                    })
                else:
                    # If logits is None, it means the request failed after all retries
                    # Log error and continue to the next sample
                    continue
            except Exception as e:
                logger.error(f"Exception in batch processing: {str(e)}")
        
        return processed_batch
    
    async def _get_logits_async(
        self, 
        session: aiohttp.ClientSession, 
        model_name: str, 
        prompt: str, 
        use_chat_template: bool,
        sample_id: str
    ) -> Optional[List[float]]:
        """
        Get logits from the API asynchronously with robust error handling.
        
        Args:
            session: aiohttp ClientSession
            model_name: Name of the model
            prompt: The formatted prompt
            use_chat_template: Whether to use chat template
            sample_id: Identifier for the sample (for logging)
            
        Returns:
            List of logits for the options A-F or None if all attempts failed
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
                # Progressive backoff delay
                if attempt > 0:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)
                
                # Handle API URLs that might or might not end with a slash
                base_url = self.api_base_url.rstrip('/')
                
                # First try with standard completions endpoint
                try:
                    async with session.post(
                        f"{base_url}/completions",
                        headers=headers,
                        json=data,
                        timeout=self.timeout
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.warning(f"HTTP Error {response.status} for sample {sample_id} (attempt {attempt+1}/{self.max_retries}): {error_text}")
                            continue
                        
                        result = await response.json()
                except aiohttp.ClientError as e:
                    # If first attempt fails, try with chat/completions endpoint
                    logger.warning(f"Error with standard completions endpoint: {e}, trying chat/completions endpoint")
                    try:
                        async with session.post(
                            f"{base_url}/chat/completions",
                            headers=headers,
                            json=data,
                            timeout=self.timeout
                        ) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                logger.warning(f"HTTP Error {response.status} for sample {sample_id} (attempt {attempt+1}/{self.max_retries}): {error_text}")
                                continue
                            
                            result = await response.json()
                    except aiohttp.ClientError as e:
                        logger.warning(f"Error with chat/completions endpoint: {e}")
                        continue
                
                # Check response structure
                if "choices" not in result or not result["choices"]:
                    logger.warning(f"Invalid response structure (no choices) for sample {sample_id} (attempt {attempt+1}/{self.max_retries})")
                    continue
                    
                # Extract logits/logprobs for options A-F
                option_logits = []
                try:
                    if not use_chat_template:
                        if "logprobs" not in result["choices"][0] or "top_logprobs" not in result["choices"][0]["logprobs"] or not result["choices"][0]["logprobs"]["top_logprobs"]:
                            logger.warning(f"Invalid logprobs structure for sample {sample_id} (attempt {attempt+1}/{self.max_retries})")
                            continue
                            
                        logprobs = result["choices"][0]["logprobs"]["top_logprobs"][0]
                        
                        # Extract logits for A, B, C, D, E, F
                        for option in ["A", "B", "C", "D", "E", "F"]:
                            if option in logprobs:
                                option_logits.append(logprobs[option])
                            else:
                                # If option not in top logprobs, use a very low value
                                option_logits.append(-100.0)
                    else:
                        # Logic for chat models
                        if "logprobs" not in result["choices"][0] or "content" not in result["choices"][0]["logprobs"] or not result["choices"][0]["logprobs"]["content"]:
                            logger.warning(f"Invalid chat logprobs structure for sample {sample_id} (attempt {attempt+1}/{self.max_retries})")
                            continue
                            
                        content_logprobs = result["choices"][0]["logprobs"]["content"]
                        if not content_logprobs or "top_logprobs" not in content_logprobs[0]:
                            logger.warning(f"Missing content logprobs for sample {sample_id} (attempt {attempt+1}/{self.max_retries})")
                            continue
                            
                        logprobs = content_logprobs[0]["top_logprobs"]
                        
                        for option in ["A", "B", "C", "D", "E", "F"]:
                            logprob_found = False
                            for lp in logprobs:
                                if "token" in lp and lp["token"] == option and "logprob" in lp:
                                    option_logits.append(lp["logprob"])
                                    logprob_found = True
                                    break
                            
                            if not logprob_found:
                                option_logits.append(-100.0)
                except Exception as e:
                    logger.warning(f"Error extracting logprobs for sample {sample_id} (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                    continue
                
                # Check if we have all 6 logits
                if len(option_logits) != 6:
                    logger.warning(f"Incomplete logits ({len(option_logits)}/6) for sample {sample_id} (attempt {attempt+1}/{self.max_retries})")
                    continue
                
                return option_logits
                
            except aiohttp.ClientError as e:
                logger.warning(f"Client error for sample {sample_id} (attempt {attempt+1}/{self.max_retries}): {str(e)}")
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout for sample {sample_id} (attempt {attempt+1}/{self.max_retries})")
            except json.JSONDecodeError:
                logger.warning(f"JSON decode error for sample {sample_id} (attempt {attempt+1}/{self.max_retries})")
            except Exception as e:
                logger.warning(f"Unexpected error for sample {sample_id} (attempt {attempt+1}/{self.max_retries}): {str(e)}")
        
        # All attempts failed
        logger.error(f"Error getting logits from API after {self.max_retries} attempts for sample {sample_id}")
        return None


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
        timeout: float = 60.0
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
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batches))) as executor:
            # Submit batch processing tasks
            futures = []
            for i, batch in enumerate(batches):
                future = executor.submit(
                    self._process_batch, 
                    batch, 
                    model_name, 
                    prompt_formatter, 
                    use_chat_template,
                    i  # batch index for tracking
                )
                futures.append(future)
            
            # Process completed futures with progress bar if requested
            if show_progress:
                from tqdm import tqdm
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                    try:
                        batch_results = future.result()
                        processed_samples.extend(batch_results)
                    except Exception as e:
                        logger.error(f"Error in batch processing: {str(e)}")
            else:
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        processed_samples.extend(batch_results)
                    except Exception as e:
                        logger.error(f"Error in batch processing: {str(e)}")
        
        return processed_samples
    
    def _process_batch(
        self, 
        batch: List[Dict], 
        model_name: str,
        prompt_formatter: Callable[[Dict], str],
        use_chat_template: bool,
        batch_idx: int
    ) -> List[Dict]:
        """
        Process a batch of samples.
        
        Args:
            batch: Batch of samples
            model_name: Name of the model to use
            prompt_formatter: Function to format samples into prompts
            use_chat_template: Whether to use chat template
            batch_idx: Index of the batch (for logging)
            
        Returns:
            List of processed samples
        """
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        processed_batch = []
        
        # Create a session with retry capabilities
        session = requests.Session()
        retries = Retry(
            total=0,  # We'll handle retries manually
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        for i, sample in enumerate(batch):
            try:
                # Format prompt
                prompt = prompt_formatter(sample)
                
                # Get logits
                sample_id = sample.get('id', f"batch_{batch_idx}_item_{i}")
                logits = self._get_logits(
                    session,
                    model_name, 
                    prompt, 
                    use_chat_template, 
                    sample_id
                )
                
                if logits:
                    # Apply softmax to get probabilities
                    softmax_probs = softmax(logits)
                    
                    # Store processed sample
                    processed_batch.append({
                        'item': sample,
                        'logits': logits,
                        'softmax': softmax_probs
                    })
            except Exception as e:
                logger.error(f"Error processing sample {sample.get('id', f'batch_{batch_idx}_item_{i}')}: {str(e)}")
        
        return processed_batch
    
    def _get_logits(
        self,
        session: requests.Session,
        model_name: str, 
        prompt: str, 
        use_chat_template: bool,
        sample_id: str
    ) -> Optional[List[float]]:
        """
        Get logits from the API with retries.
        
        Args:
            session: Requests session
            model_name: Name of the model
            prompt: The formatted prompt
            use_chat_template: Whether to use chat template
            sample_id: Identifier for the sample (for logging)
            
        Returns:
            List of logits for the options A-F or None if all attempts failed
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
                # Progressive backoff delay
                if attempt > 0:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    time.sleep(delay)
                
                # Handle API URLs that might or might not end with a slash
                base_url = self.api_base_url.rstrip('/')
                
                # Try first with completions endpoint
                try:
                    response = session.post(
                        f"{base_url}/completions",
                        headers=headers,
                        json=data,
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                except requests.RequestException as e:
                    # If first attempt fails, try with chat/completions endpoint
                    logger.warning(f"Error with standard completions endpoint: {e}, trying chat/completions endpoint")
                    try:
                        response = session.post(
                            f"{base_url}/chat/completions", 
                            headers=headers,
                            json=data,
                            timeout=self.timeout
                        )
                        response.raise_for_status()
                    except requests.RequestException as e:
                        logger.warning(f"Both completions endpoints failed: {e}")
                        continue
                
                if response.status_code != 200:
                    logger.warning(f"HTTP Error {response.status_code} for sample {sample_id} (attempt {attempt+1}/{self.max_retries}): {response.text}")
                    continue
                
                result = response.json()
                
                # Check response structure
                if "choices" not in result or not result["choices"]:
                    logger.warning(f"Invalid response structure (no choices) for sample {sample_id} (attempt {attempt+1}/{self.max_retries})")
                    continue
                
                # Extract logits/logprobs for options A-F
                option_logits = []
                try:
                    if not use_chat_template:
                        if "logprobs" not in result["choices"][0] or "top_logprobs" not in result["choices"][0]["logprobs"] or not result["choices"][0]["logprobs"]["top_logprobs"]:
                            logger.warning(f"Invalid logprobs structure for sample {sample_id} (attempt {attempt+1}/{self.max_retries})")
                            continue
                            
                        logprobs = result["choices"][0]["logprobs"]["top_logprobs"][0]
                        
                        # Extract logits for A, B, C, D, E, F
                        for option in ["A", "B", "C", "D", "E", "F"]:
                            if option in logprobs:
                                option_logits.append(logprobs[option])
                            else:
                                # If option not in top logprobs, use a very low value
                                option_logits.append(-100.0)
                    else:
                        # Logic for chat models
                        if "logprobs" not in result["choices"][0] or "content" not in result["choices"][0]["logprobs"] or not result["choices"][0]["logprobs"]["content"]:
                            logger.warning(f"Invalid chat logprobs structure for sample {sample_id} (attempt {attempt+1}/{self.max_retries})")
                            continue
                            
                        content_logprobs = result["choices"][0]["logprobs"]["content"]
                        if not content_logprobs or "top_logprobs" not in content_logprobs[0]:
                            logger.warning(f"Missing content logprobs for sample {sample_id} (attempt {attempt+1}/{self.max_retries})")
                            continue
                            
                        logprobs = content_logprobs[0]["top_logprobs"]
                        
                        for option in ["A", "B", "C", "D", "E", "F"]:
                            logprob_found = False
                            for lp in logprobs:
                                if "token" in lp and lp["token"] == option and "logprob" in lp:
                                    option_logits.append(lp["logprob"])
                                    logprob_found = True
                                    break
                            
                            if not logprob_found:
                                option_logits.append(-100.0)
                except Exception as e:
                    logger.warning(f"Error extracting logprobs for sample {sample_id} (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                    continue
                
                # Check if we have all 6 logits
                if len(option_logits) != 6:
                    logger.warning(f"Incomplete logits ({len(option_logits)}/6) for sample {sample_id} (attempt {attempt+1}/{self.max_retries})")
                    continue
                
                return option_logits
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error for sample {sample_id} (attempt {attempt+1}/{self.max_retries}): {str(e)}")
            except json.JSONDecodeError:
                logger.warning(f"JSON decode error for sample {sample_id} (attempt {attempt+1}/{self.max_retries})")
            except Exception as e:
                logger.warning(f"Unexpected error for sample {sample_id} (attempt {attempt+1}/{self.max_retries}): {str(e)}")
        
        # All attempts failed
        logger.error(f"Error getting logits from API after {self.max_retries} attempts for sample {sample_id}")
        return None


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
