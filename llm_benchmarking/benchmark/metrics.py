"""
Metrics module for LLM Uncertainty Benchmarking.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

from ..utils.logging import get_logger

logger = get_logger(__name__)


def calculate_metrics_with_conformal_prediction(
    calibration_data: List[Dict],
    test_data: List[Dict],
    score_function: str = "lac",
    error_rate: float = 0.1
) -> Dict[str, Any]:
    """
    Calculate metrics using conformal prediction.
    
    Args:
        calibration_data: List of calibration items with logits
        test_data: List of test items with logits
        score_function: Conformal score function to use ("lac" or "aps")
        error_rate: Error rate alpha for conformal prediction
        
    Returns:
        Dictionary with metrics
    """
    # Calculate conformal scores for calibration data
    calibration_scores = []
    
    for item_data in calibration_data:
        item = item_data['item']
        softmax_probs = item_data['softmax']
        
        # Get the index of the correct answer
        correct_idx = _get_correct_answer_index(item)
        
        if correct_idx is None:
            logger.warning(f"Could not determine correct answer index for item {item.get('id', 'unknown')}")
            continue
        
        # Calculate conformal score
        score = _calculate_conformal_score(softmax_probs, correct_idx, score_function)
        calibration_scores.append(score)
    
    # Calculate threshold
    n = len(calibration_scores)
    
    # Check if we have any calibration scores
    if n == 0:
        logger.warning("No valid calibration scores. Using default threshold.")
        threshold = 0.5  # Default threshold
    else:
        quantile = int(np.ceil((n + 1) * (1 - error_rate))) / n
        threshold = np.quantile(calibration_scores, quantile)
    
    # Calculate metrics for test data
    correct_count = 0
    covered_count = 0
    prediction_set_sizes = []
    
    for item_data in test_data:
        item = item_data['item']
        softmax_probs = item_data['softmax']
        
        # Find the predicted label (highest probability)
        pred_idx = np.argmax(softmax_probs)
        pred_label = item['choice_labels'][pred_idx]
        
        # Find the correct label
        correct_idx = _get_correct_answer_index(item)
        
        if correct_idx is None:
            logger.warning(f"Could not determine correct answer index for item {item.get('id', 'unknown')}")
            continue
        
        correct_label = item['choice_labels'][correct_idx]
        
        # Check if prediction is correct
        if pred_label == correct_label:
            correct_count += 1
        
        # Construct prediction set
        prediction_set = _construct_prediction_set(
            softmax_probs,
            item['choice_labels'],
            threshold,
            score_function
        )
        
        # Ensure the prediction set is not empty
        if not prediction_set:
            prediction_set = [pred_label]
        
        # Check if the correct label is in the prediction set
        if correct_label in prediction_set:
            covered_count += 1
        
        # Record prediction set size
        prediction_set_sizes.append(len(prediction_set))
    
    # Calculate metrics
    n_test = len(test_data)
    
    acc = correct_count / n_test if n_test > 0 else 0
    cr = covered_count / n_test if n_test > 0 else 0
    ss = sum(prediction_set_sizes) / n_test if n_test > 0 else 0
    
    return {
        'acc': acc,
        'cr': cr,
        'ss': ss,
        'threshold': threshold,
        'prediction_set_sizes': prediction_set_sizes
    }


def _calculate_conformal_score(
    softmax_probs: List[float],
    correct_idx: int,
    score_function: str = "lac"
) -> float:
    """
    Calculate conformal score for a single item.
    
    Args:
        softmax_probs: Softmax probabilities
        correct_idx: Index of the correct answer
        score_function: Conformal score function to use ("lac" or "aps")
        
    Returns:
        Conformal score
    """
    if score_function == "lac":
        # LAC: 1 - probability of the true label
        score = 1 - softmax_probs[correct_idx]
    elif score_function == "aps":
        # APS: Sum of probabilities of labels with probability >= true label
        true_prob = softmax_probs[correct_idx]
        score = sum([p for p in softmax_probs if p >= true_prob])
    else:
        raise ValueError(f"Unknown score function: {score_function}")
    
    return score


def _construct_prediction_set(
    softmax_probs: List[float],
    choice_labels: List[str],
    threshold: float,
    score_function: str = "lac"
) -> List[str]:
    """
    Construct a prediction set based on conformal scores.
    
    Args:
        softmax_probs: Softmax probabilities
        choice_labels: List of choice labels
        threshold: Conformal threshold
        score_function: Conformal score function to use ("lac" or "aps")
        
    Returns:
        List of labels in the prediction set
    """
    prediction_set = []
    
    for i, (prob, label) in enumerate(zip(softmax_probs, choice_labels)):
        if score_function == "lac":
            # LAC: Add to set if 1 - probability <= threshold
            if 1 - prob <= threshold:
                prediction_set.append(label)
        elif score_function == "aps":
            # APS: Calculate sum for each label and add if <= threshold
            sum_prob = sum([p for p in softmax_probs if p >= prob])
            if sum_prob <= threshold:
                prediction_set.append(label)
    
    return prediction_set


def _get_correct_answer_index(item: Dict[str, Any]) -> Optional[int]:
    """
    Get the index of the correct answer from the item.
    
    Args:
        item: Dataset item
        
    Returns:
        Index of the correct answer or None if it can't be determined
    """
    if 'answer' not in item:
        return None
        
    try:
        if isinstance(item['answer'], int):
            # Direct integer index
            if 0 <= item['answer'] < len(item['choice_labels']):
                return item['answer']
        elif isinstance(item['answer'], str):
            if item['answer'] in item['choice_labels']:
                # If answer is already a label like 'A', 'B', etc.
                return item['choice_labels'].index(item['answer'])
            elif item['answer'].isdigit():
                # If answer is a digit string
                idx = int(item['answer'])
                if 0 <= idx < len(item['choice_labels']):
                    return idx
    except Exception as e:
        logger.warning(f"Error determining correct answer index: {e}")
    
    return None


def calculate_average_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate average metrics across multiple runs or configurations.
    
    Args:
        metrics_list: List of metrics dictionaries
        
    Returns:
        Dictionary with averaged metrics
    """
    # Keys to average
    keys = ["acc", "cr", "ss"]
    
    # Initialize result
    avg_metrics = {}
    
    # Calculate averages
    for key in keys:
        values = [m.get(key, 0) for m in metrics_list if key in m]
        avg_metrics[key] = sum(values) / len(values) if values else 0
    
    return avg_metrics
