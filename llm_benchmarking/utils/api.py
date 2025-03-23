"""
API utilities for LLM Uncertainty Benchmarking.
"""

import json
import logging
import requests
from typing import List, Dict, Optional, Any, Union

from .logging import get_logger

logger = get_logger(__name__)


def get_logits_from_api(
    api_base_url: str,
    model_name: str,
    prompt: str,
    use_chat_template: bool = False,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    timeout: float = 60.0,
) -> Optional[List[float]]:
    """
    Get logits from an OpenAI-compatible API.
    
    Args:
        api_base_url: Base URL for the API
        model_name: Name of the model
        prompt: The formatted prompt
        use_chat_template: Whether to use chat template
        api_key: Optional API key for authentication
        temperature: Temperature for sampling (default: 0.0)
        timeout: Timeout for API requests in seconds (default: 60.0)
        
    Returns:
        List of logits for the options A-F or None if the request failed
    """
    headers = {
        "Content-Type": "application/json"
    }
    
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    if use_chat_template:
        # Format as a chat message for instruction-tuned models
        data = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "logprobs": True,
            "top_logprobs": 10
        }
    else:
        # Use standard completion for base models
        data = {
            "model": model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": 1,
            "logprobs": 10,
            "echo": False
        }
    
    try:
        # Handle API URLs that might or might not end with a slash
        base_url = api_base_url.rstrip('/')
        
        # Try first with completions endpoint
        try:
            response = requests.post(
                f"{base_url}/completions", 
                headers=headers,
                json=data,
                timeout=timeout
            )
            response.raise_for_status()
        except requests.RequestException as e:
            # If first attempt fails, try with chat/completions endpoint
            logger.warning(f"Error with standard completions endpoint: {e}, trying chat/completions endpoint")
            try:
                response = requests.post(
                    f"{base_url}/chat/completions", 
                    headers=headers,
                    json=data,
                    timeout=timeout
                )
                response.raise_for_status()
            except requests.RequestException as e:
                logger.error(f"Both completions endpoints failed: {e}")
                return None
        
        result = response.json()
        
        # Extract logits/logprobs for options A-F
        option_logits = []
        
        # For standard completion API:
        if not use_chat_template:
            if "choices" not in result or not result["choices"]:
                logger.warning("Invalid response structure (no choices)")
                return None
                
            if "logprobs" not in result["choices"][0] or "top_logprobs" not in result["choices"][0]["logprobs"]:
                logger.warning("Invalid logprobs structure")
                return None
                
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
            if "choices" not in result or not result["choices"]:
                logger.warning("Invalid response structure (no choices)")
                return None
                
            if "logprobs" not in result["choices"][0] or "content" not in result["choices"][0]["logprobs"]:
                logger.warning("Invalid chat logprobs structure")
                return None
                
            content_logprobs = result["choices"][0]["logprobs"]["content"]
            if not content_logprobs or "top_logprobs" not in content_logprobs[0]:
                logger.warning("Missing content logprobs")
                return None
                
            logprobs = content_logprobs[0]["top_logprobs"]
            
            for option in ["A", "B", "C", "D", "E", "F"]:
                option_logit = -100.0  # Default value
                for lp in logprobs:
                    if "token" in lp and lp["token"] == option and "logprob" in lp:
                        option_logit = lp["logprob"]
                        break
                option_logits.append(option_logit)
        
        return option_logits
    
    except Exception as e:
        logger.error(f"Error getting logits from API: {e}")
        return None


def softmax(logits: List[float]) -> List[float]:
    """
    Apply softmax to logits.
    
    Args:
        logits: List of logit values
        
    Returns:
        List of probabilities
    """
    import numpy as np
    
    try:
        # Shift values for numerical stability
        shifted_logits = [x - max(logits) for x in logits]
        exp_logits = [np.exp(x) for x in shifted_logits]
        sum_exp_logits = sum(exp_logits)
        if sum_exp_logits == 0:
            # Handle degenerate case
            return [1.0/len(logits)] * len(logits)
        return [x / sum_exp_logits for x in exp_logits]
    except Exception as e:
        logger.error(f"Error in softmax calculation: {e}")
        # Return uniform distribution as fallback
        return [1.0/len(logits)] * len(logits)
