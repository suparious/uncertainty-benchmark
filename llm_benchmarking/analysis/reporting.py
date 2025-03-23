"""
Reporting module for LLM Uncertainty Benchmarking.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Union

from ..utils.logging import get_logger

logger = get_logger(__name__)


def generate_report(
    results: Dict[str, Dict[str, Any]],
    model_names: List[str] = None,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate a report of the benchmark results.
    
    Args:
        results: Dictionary with benchmark results
        model_names: List of model names to include in the report. If None, include all.
        output_file: Path to save the report as CSV. If None, don't save.
        
    Returns:
        DataFrame with the report
    """
    if model_names is None:
        model_names = list(results.keys())
    
    # Initialize data for DataFrame
    report_data = []
    
    # Add rows for each model and task
    for model_name in model_names:
        if model_name not in results:
            logger.warning(f"No results for model {model_name}. Skipping...")
            continue
        
        model_results = results[model_name]
        
        # Add overall results
        report_data.append({
            'Model': model_name,
            'Task': 'Average',
            'Accuracy': model_results['overall']['acc'] * 100,
            'Coverage Rate': model_results['overall']['cr'] * 100,
            'Set Size': model_results['overall']['ss']
        })
        
        # Add task-specific results
        for task, task_results in model_results.items():
            if task == 'overall':
                continue
            
            report_data.append({
                'Model': model_name,
                'Task': task,
                'Accuracy': task_results['avg']['acc'] * 100,
                'Coverage Rate': task_results['avg']['cr'] * 100,
                'Set Size': task_results['avg']['ss']
            })
    
    # Create DataFrame
    report_df = pd.DataFrame(report_data)
    
    # Sort by model name and task
    report_df = report_df.sort_values(['Model', 'Task'])
    
    # Save to file if specified
    if output_file:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        report_df.to_csv(output_file, index=False)
        logger.info(f"Report saved to {output_file}")
    
    return report_df


def generate_prompt_strategy_report(
    results: Dict[str, Dict[str, Any]],
    model_name: str,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate a detailed report on prompt strategy effectiveness for a specific model.
    
    Args:
        results: Dictionary with benchmark results
        model_name: Name of the model to analyze
        output_file: Path to save the report as CSV. If None, don't save.
        
    Returns:
        DataFrame with the detailed prompt strategy report
    """
    if model_name not in results:
        logger.warning(f"No results for model {model_name}. Cannot generate report.")
        return pd.DataFrame()
    
    model_results = results[model_name]
    
    # Initialize data for DataFrame
    report_data = []
    
    # Add rows for each task and prompt strategy
    for task, task_results in model_results.items():
        if task == 'overall':
            continue
        
        if 'prompt_strategies' not in task_results:
            logger.warning(f"No prompt strategy data for task {task}. Skipping.")
            continue
        
        for strategy, strategy_results in task_results['prompt_strategies'].items():
            # Get results for each score function
            for score_func, metrics in strategy_results.items():
                if score_func == 'avg':
                    continue
                
                report_data.append({
                    'Task': task,
                    'Prompt Strategy': strategy,
                    'Score Function': score_func,
                    'Accuracy': metrics.get('acc', 0) * 100,
                    'Coverage Rate': metrics.get('cr', 0) * 100,
                    'Set Size': metrics.get('ss', 0),
                    'Threshold': metrics.get('threshold', 0)
                })
    
    # Create DataFrame
    report_df = pd.DataFrame(report_data)
    
    # Sort by task and prompt strategy
    report_df = report_df.sort_values(['Task', 'Prompt Strategy', 'Score Function'])
    
    # Save to file if specified
    if output_file:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        report_df.to_csv(output_file, index=False)
        logger.info(f"Prompt strategy report saved to {output_file}")
    
    return report_df


def load_results(input_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Load benchmark results from a file.
    
    Args:
        input_file: Path to the results file
        
    Returns:
        Dictionary with benchmark results
    """
    try:
        with open(input_file, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Loaded results from {input_file}")
        return results
    except Exception as e:
        logger.error(f"Error loading results from {input_file}: {e}")
        return {}


def load_multiple_results(input_files: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Load benchmark results from multiple files and merge them.
    
    Args:
        input_files: List of paths to results files
        
    Returns:
        Merged dictionary with benchmark results
    """
    merged_results = {}
    
    for input_file in input_files:
        results = load_results(input_file)
        merged_results.update(results)
    
    return merged_results
