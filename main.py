import os
import json
import numpy as np
import pandas as pd
import requests
import logging
from typing import List, Dict, Tuple, Any, Optional, Union
from tqdm import tqdm
import torch

# Import dataset utilities
from dataset_utils import (
    load_mmlu_dataset,
    load_cosmos_qa_dataset,
    load_hellaswag_dataset,
    load_halueval_dialogue_dataset,
    load_halueval_summarization_dataset
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMUncertaintyBenchmark:
    """
    A benchmarking framework for evaluating LLMs via uncertainty quantification
    based on the paper "Benchmarking LLMs via Uncertainty Quantification".
    """
    
    def __init__(
        self, 
        api_base_url: str,
        api_key: Optional[str] = None,
        calibration_ratio: float = 0.5,
        error_rate: float = 0.1,
        max_tokens: int = 2048,
        num_demonstrations: Dict[str, int] = None
    ):
        """
        Initialize the benchmark.

        Args:
            api_base_url: Base URL for the OpenAI-compatible API
            api_key: API key for authentication (if required)
            calibration_ratio: Ratio of data to use for calibration
            error_rate: Error rate alpha for conformal prediction
            max_tokens: Maximum number of tokens for input context
            num_demonstrations: Number of demonstrations for each task
        """
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.calibration_ratio = calibration_ratio
        self.error_rate = error_rate
        self.max_tokens = max_tokens
        
        if num_demonstrations is None:
            self.num_demonstrations = {
                "qa": 5,
                "rc": 5,
                "ci": 5,
                "drs": 3,
                "ds": 1
            }
        else:
            self.num_demonstrations = num_demonstrations
        
        # Define task datasets
        self.tasks = {
            "qa": "mmlu",
            "rc": "cosmos_qa",
            "ci": "hellaswag",
            "drs": "halueval_dialogue",
            "ds": "halueval_summarization"
        }
        
        # Initialize dataset caches
        self.datasets = {}
        self.demonstrations = {}
        self.results = {}
        
    def prepare_datasets(self, tasks: List[str] = None, sample_size: int = 10000):
        """
        Prepare datasets for benchmarking.
        
        Args:
            tasks: List of tasks to prepare datasets for. If None, prepare all.
            sample_size: Number of samples to use for each task
        """
        if tasks is None:
            tasks = list(self.tasks.keys())
        
        for task in tasks:
            logger.info(f"Preparing dataset for {task} task...")
            
            # Load the appropriate dataset based on the task
            if task == "qa":
                # Load MMLU dataset
                dataset = load_mmlu_dataset(sample_size)
                
            elif task == "rc":
                # Load CosmosQA dataset
                dataset = load_cosmos_qa_dataset(sample_size)
                
            elif task == "ci":
                # Load HellaSwag dataset
                dataset = load_hellaswag_dataset(sample_size)
                
            elif task == "drs":
                # Load HaluEval dialogue dataset
                dataset = load_halueval_dialogue_dataset(sample_size)
                
            elif task == "ds":
                # Load HaluEval summarization dataset
                dataset = load_halueval_summarization_dataset(sample_size)
            
            else:
                raise ValueError(f"Unknown task: {task}")
            
            # Make sure we have data
            if not dataset:
                logger.warning(f"No data loaded for task {task}. Skipping...")
                continue
                
            # Add options E and F to all questions
            for item in dataset:
                item['choices'].extend(['I don\'t know', 'None of the above'])
                item['choice_labels'].extend(['E', 'F'])
            
            # Split into calibration and test sets
            n_calibration = int(len(dataset) * self.calibration_ratio)
            calibration_set = dataset[:n_calibration]
            test_set = dataset[n_calibration:]
            
            # Store datasets
            self.datasets[task] = {
                'calibration': calibration_set,
                'test': test_set,
                'full': dataset
            }
            
            # Prepare demonstrations
            if self.num_demonstrations[task] > 0 and len(calibration_set) >= self.num_demonstrations[task]:
                self.demonstrations[task] = self._prepare_demonstrations(task)
            else:
                logger.warning(f"Not enough calibration samples for demonstrations in task {task}")
                self.demonstrations[task] = []
            
            logger.info(f"Prepared {len(dataset)} samples for {task} task "
                       f"({len(calibration_set)} for calibration, {len(test_set)} for testing)")
    
    # Dataset loading functions have been moved to dataset_utils.py
    
    def _prepare_demonstrations(self, task: str) -> List[Dict]:
        """Prepare demonstrations for a task from the calibration set."""
        calibration_set = self.datasets[task]['calibration']
        import random
        
        # Select random samples for demonstrations
        demo_samples = random.sample(calibration_set, self.num_demonstrations[task])
        
        return demo_samples
    
    def format_prompt(self, task: str, item: Dict, prompt_strategy: str) -> str:
        """
        Format prompt for a task based on the specified prompting strategy.
        
        Args:
            task: Task name
            item: Dataset item
            prompt_strategy: One of "base", "shared_instruction", or "task_specific"
            
        Returns:
            Formatted prompt
        """
        # Prepare demonstrations
        demos_text = ""
        if task in self.demonstrations and self.demonstrations[task]:
            for demo in self.demonstrations[task]:
                demo_prompt = self._format_single_prompt(task, demo, prompt_strategy, is_demo=True)
                demos_text += f"{demo_prompt}\n\n"
        
        # Format the current item
        item_prompt = self._format_single_prompt(task, item, prompt_strategy, is_demo=False)
        
        # Combine demonstrations and current item
        prompt = f"{demos_text}{item_prompt}"
        
        return prompt
    
    def _format_single_prompt(self, task: str, item: Dict, prompt_strategy: str, is_demo: bool) -> str:
        """Format a single prompt for a task item."""
        context_type = "Context" if task in ["rc", "ci"] else "Dialogue" if task == "drs" else "Document" if task == "ds" else None
        
        context_text = ""
        if context_type and item['context']:
            context_text = f"{context_type}: {item['context']}\n"
        
        choices_text = "Choices:\n"
        for label, choice in zip(item['choice_labels'], item['choices']):
            choices_text += f"{label}. {choice}\n"
        
        answer_text = ""
        if is_demo:
            # For demonstrations, include the answer
            answer_label = item['choice_labels'][int(item['answer'])] if isinstance(item['answer'], (int, str)) and item['answer'].isdigit() else item['answer']
            answer_text = f"Answer: {answer_label}"
        else:
            # For test items, just have the prompt "Answer:"
            answer_text = "Answer:"
        
        if prompt_strategy == "base":
            prompt = f"{context_text}Question: {item['question']}\n{choices_text}{answer_text}"
        
        elif prompt_strategy == "shared_instruction":
            if is_demo:
                instruction = ""  # No instruction for demos to save space
            else:
                instruction = "Below are some examples of multiple-choice questions with six potential answers. For each question, only one option is correct.\n\n"
                instruction += "Now make your best effort and select the correct answer for the following question. You only need to output the option.\n\n"
            
            prompt = f"{instruction}{context_text}Question: {item['question']}\n{choices_text}{answer_text}"
        
        elif prompt_strategy == "task_specific":
            if is_demo:
                instruction = ""  # No instruction for demos to save space
            else:
                if task == "qa":
                    instruction = "Below are some examples of multiple-choice questions about question answering. "
                    instruction += "Each question should be answered based on your world knowledge and problem solving ability.\n\n"
                elif task == "rc":
                    instruction = "Below are some examples of multiple-choice questions about reading comprehension. "
                    instruction += "Each question should be answered based on the given context and commonsense reasoning when necessary.\n\n"
                elif task == "ci":
                    instruction = "Below are some examples of multiple-choice questions about commonsense natural language inference. "
                    instruction += "For each question, there is a given context and the answer is the option that most likely follows the context.\n\n"
                elif task == "drs":
                    instruction = "Below are some examples of multiple-choice questions about dialogue response selection. "
                    instruction += "For each question, the answer is the option that represents the most suitable response for the given dialogue history, "
                    instruction += "without hallucination and non-factual information.\n\n"
                elif task == "ds":
                    instruction = "Below are some examples of multiple-choice questions about document summarization. "
                    instruction += "For each question, the answer is the option that accurately summarizes the given document without "
                    instruction += "hallucination and non-factual information.\n\n"
                
                instruction += "Now make your best effort and select the correct answer for the following question. You only need to output the option.\n\n"
            
            prompt = f"{instruction}{context_text}Question: {item['question']}\n{choices_text}{answer_text}"
        
        else:
            raise ValueError(f"Unknown prompt strategy: {prompt_strategy}")
        
        return prompt
    
    def evaluate_model(
        self, 
        model_name: str, 
        tasks: List[str] = None, 
        prompt_strategies: List[str] = None, 
        use_chat_template: bool = False,
        temperature: float = 0.0
    ) -> Dict:
        """
        Evaluate a model on the benchmark.
        
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
        
        for task in tasks:
            logger.info(f"Evaluating {model_name} on {task} task...")
            
            # Check if dataset is prepared
            if task not in self.datasets:
                logger.info(f"Dataset for {task} not prepared. Preparing now...")
                self.prepare_datasets([task])
            
            task_results = {}
            
            for prompt_strategy in prompt_strategies:
                logger.info(f"Using {prompt_strategy} prompt strategy...")
                
                # Get logits for calibration set
                calibration_logits = []
                calibration_data = self.datasets[task]['calibration']
                
                for item in tqdm(calibration_data, desc="Processing calibration set"):
                    prompt = self.format_prompt(task, item, prompt_strategy)
                    
                    try:
                        # Get logits from the model API
                        logits = self._get_logits_from_api(model_name, prompt, use_chat_template)
                        
                        # Store item with its logits
                        item_with_logits = {
                            'item': item,
                            'logits': logits,
                            'softmax': self._softmax(logits)
                        }
                        calibration_logits.append(item_with_logits)
                    except Exception as e:
                        logger.error(f"Error getting logits for calibration item {item['id']}: {e}")
                
                # Get logits for test set
                test_logits = []
                test_data = self.datasets[task]['test']
                
                for item in tqdm(test_data, desc="Processing test set"):
                    prompt = self.format_prompt(task, item, prompt_strategy)
                    
                    try:
                        # Get logits from the model API
                        logits = self._get_logits_from_api(model_name, prompt, use_chat_template)
                        
                        # Store item with its logits
                        item_with_logits = {
                            'item': item,
                            'logits': logits,
                            'softmax': self._softmax(logits)
                        }
                        test_logits.append(item_with_logits)
                    except Exception as e:
                        logger.error(f"Error getting logits for test item {item['id']}: {e}")
                
                # Calculate metrics using LAC conformal score function
                lac_results = self._calculate_metrics_with_conformal_prediction(
                    calibration_logits, test_logits, score_function="lac"
                )
                
                # Calculate metrics using APS conformal score function
                aps_results = self._calculate_metrics_with_conformal_prediction(
                    calibration_logits, test_logits, score_function="aps"
                )
                
                # Average the results
                avg_results = {
                    'acc': lac_results['acc'],  # Same for both
                    'cr': (lac_results['cr'] + aps_results['cr']) / 2,
                    'ss': (lac_results['ss'] + aps_results['ss']) / 2
                }
                
                # Store results for this prompt strategy
                task_results[prompt_strategy] = {
                    'lac': lac_results,
                    'aps': aps_results,
                    'avg': avg_results
                }
            
            # Average results across prompt strategies
            avg_across_prompts = {
                'acc': np.mean([task_results[ps]['avg']['acc'] for ps in prompt_strategies]),
                'cr': np.mean([task_results[ps]['avg']['cr'] for ps in prompt_strategies]),
                'ss': np.mean([task_results[ps]['avg']['ss'] for ps in prompt_strategies])
            }
            
            # Store results for this task
            self.results[model_name][task] = {
                'prompt_strategies': task_results,
                'avg': avg_across_prompts
            }
            
            logger.info(f"Results for {model_name} on {task} task: Acc={avg_across_prompts['acc']:.4f}, CR={avg_across_prompts['cr']:.4f}, SS={avg_across_prompts['ss']:.4f}")
        
        # Calculate overall average across tasks
        overall_avg = {
            'acc': np.mean([self.results[model_name][task]['avg']['acc'] for task in tasks]),
            'cr': np.mean([self.results[model_name][task]['avg']['cr'] for task in tasks]),
            'ss': np.mean([self.results[model_name][task]['avg']['ss'] for task in tasks])
        }
        
        self.results[model_name]['overall'] = overall_avg
        
        logger.info(f"Overall results for {model_name}: Acc={overall_avg['acc']:.4f}, CR={overall_avg['cr']:.4f}, SS={overall_avg['ss']:.4f}")
        
        return self.results[model_name]
    
    def _get_logits_from_api(self, model_name: str, prompt: str, use_chat_template: bool) -> List[float]:
        """
        Get logits from the OpenAI-compatible API.
        
        Args:
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
        
        try:
            response = requests.post(
                f"{self.api_base_url}/completions", 
                headers=headers,
                json=data
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract logits/logprobs for options A-F
            # The exact logic will depend on the API response structure
            
            # For standard completion API:
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
                # This is a placeholder and would need to be adapted
                logprobs = result["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                
                option_logits = []
                for option in ["A", "B", "C", "D", "E", "F"]:
                    option_logit = next((lp["logprob"] for lp in logprobs if lp["token"] == option), -100.0)
                    option_logits.append(option_logit)
            
            return option_logits
        
        except Exception as e:
            logger.error(f"Error getting logits from API: {e}")
            raise
    
    def _softmax(self, logits: List[float]) -> List[float]:
        """Apply softmax to logits."""
        exp_logits = [np.exp(x) for x in logits]
        sum_exp_logits = sum(exp_logits)
        return [x / sum_exp_logits for x in exp_logits]
    
    def _calculate_metrics_with_conformal_prediction(
        self, 
        calibration_data: List[Dict], 
        test_data: List[Dict],
        score_function: str = "lac"
    ) -> Dict:
        """
        Calculate metrics using conformal prediction.
        
        Args:
            calibration_data: List of calibration items with logits
            test_data: List of test items with logits
            score_function: Conformal score function to use ("lac" or "aps")
            
        Returns:
            Dictionary with metrics
        """
        # Calculate conformal scores for calibration data
        calibration_scores = []
        
        for item_data in calibration_data:
            item = item_data['item']
            softmax_probs = item_data['softmax']
            
            # Get the index of the correct answer
            correct_idx = None
            if isinstance(item['answer'], (int, str)) and item['answer'].isdigit():
                correct_idx = int(item['answer'])
            else:
                for i, label in enumerate(item['choice_labels']):
                    if label == item['answer']:
                        correct_idx = i
                        break
            
            if correct_idx is None:
                logger.warning(f"Could not determine correct answer index for item {item['id']}")
                continue
            
            # Calculate conformal score
            if score_function == "lac":
                # LAC: 1 - probability of the true label
                score = 1 - softmax_probs[correct_idx]
            elif score_function == "aps":
                # APS: Sum of probabilities of labels with probability >= true label
                true_prob = softmax_probs[correct_idx]
                score = sum([p for p in softmax_probs if p >= true_prob])
            else:
                raise ValueError(f"Unknown score function: {score_function}")
            
            calibration_scores.append(score)
        
        # Calculate threshold
        n = len(calibration_scores)
        quantile = int(np.ceil((n + 1) * (1 - self.error_rate))) / n
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
            correct_idx = None
            if isinstance(item['answer'], (int, str)) and item['answer'].isdigit():
                correct_idx = int(item['answer'])
            else:
                for i, label in enumerate(item['choice_labels']):
                    if label == item['answer']:
                        correct_idx = i
                        break
            
            if correct_idx is None:
                logger.warning(f"Could not determine correct answer index for item {item['id']}")
                continue
            
            correct_label = item['choice_labels'][correct_idx]
            
            # Check if prediction is correct
            if pred_label == correct_label:
                correct_count += 1
            
            # Construct prediction set
            prediction_set = []
            for i, (prob, label) in enumerate(zip(softmax_probs, item['choice_labels'])):
                if score_function == "lac":
                    # LAC: Add to set if 1 - probability <= threshold
                    if 1 - prob <= threshold:
                        prediction_set.append(label)
                elif score_function == "aps":
                    # APS: Calculate sum for each label and add if <= threshold
                    sum_prob = sum([p for p in softmax_probs if p >= prob])
                    if sum_prob <= threshold:
                        prediction_set.append(label)
            
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
    
    def generate_report(self, model_names: List[str] = None, output_file: str = None) -> pd.DataFrame:
        """
        Generate a report of the benchmark results.
        
        Args:
            model_names: List of model names to include in the report. If None, include all.
            output_file: Path to save the report as CSV. If None, don't save.
            
        Returns:
            DataFrame with the report
        """
        if model_names is None:
            model_names = list(self.results.keys())
        
        # Initialize data for DataFrame
        report_data = []
        
        # Add rows for each model and task
        for model_name in model_names:
            if model_name not in self.results:
                logger.warning(f"No results for model {model_name}. Skipping...")
                continue
            
            model_results = self.results[model_name]
            
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
            report_df.to_csv(output_file, index=False)
            logger.info(f"Report saved to {output_file}")
        
        return report_df
    
    def visualize_results(self, model_names: List[str] = None, output_file: str = None):
        """
        Visualize the benchmark results.
        
        Args:
            model_names: List of model names to include. If None, include all.
            output_file: Path to save the visualization. If None, display it.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if model_names is None:
            model_names = list(self.results.keys())
        
        # Get report DataFrame
        report_df = self.generate_report(model_names)
        
        # Filter for average results
        avg_results = report_df[report_df['Task'] == 'Average'].copy()
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Accuracy vs. Set Size plot
        sns.scatterplot(
            data=avg_results, 
            x='Accuracy', 
            y='Set Size', 
            s=100, 
            hue='Model', 
            ax=ax1
        )
        ax1.set_title('Accuracy vs. Uncertainty (Set Size)')
        ax1.set_xlabel('Accuracy (%)')
        ax1.set_ylabel('Average Set Size (lower is better)')
        
        # Add text labels for each point
        for idx, row in avg_results.iterrows():
            ax1.text(row['Accuracy'] + 0.5, row['Set Size'] + 0.05, row['Model'], fontsize=8)
        
        # Coverage Rate vs. Set Size plot
        sns.scatterplot(
            data=avg_results, 
            x='Coverage Rate', 
            y='Set Size', 
            s=100, 
            hue='Model', 
            ax=ax2
        )
        ax2.set_title('Coverage Rate vs. Uncertainty (Set Size)')
        ax2.set_xlabel('Coverage Rate (%)')
        ax2.set_ylabel('Average Set Size (lower is better)')
        
        # Add text labels for each point
        for idx, row in avg_results.iterrows():
            ax2.text(row['Coverage Rate'] + 0.5, row['Set Size'] + 0.05, row['Model'], fontsize=8)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {output_file}")
        else:
            plt.show()
        
        # Create task-specific visualizations
        task_list = [task for task in report_df['Task'].unique() if task != 'Average']
        
        # Create figure with 5 subplots (one for each task)
        fig, axes = plt.subplots(len(task_list), 1, figsize=(10, 4*len(task_list)))
        
        for i, task in enumerate(task_list):
            task_df = report_df[report_df['Task'] == task].copy()
            
            ax = axes[i] if len(task_list) > 1 else axes
            
            # Sort by accuracy
            task_df = task_df.sort_values('Accuracy', ascending=False)
            
            # Create bar plot with two metrics
            bar_width = 0.35
            x = np.arange(len(task_df))
            
            # Plot accuracy bars
            ax.bar(x - bar_width/2, task_df['Accuracy'], bar_width, label='Accuracy (%)')
            
            # Create twin axis for set size
            ax2 = ax.twinx()
            ax2.bar(x + bar_width/2, task_df['Set Size'], bar_width, color='orange', label='Set Size')
            
            # Set labels and title
            ax.set_xlabel('Models')
            ax.set_ylabel('Accuracy (%)')
            ax2.set_ylabel('Set Size')
            ax.set_title(f'Task: {task}')
            
            # Set x-ticks
            ax.set_xticks(x)
            ax.set_xticklabels(task_df['Model'], rotation=45, ha='right')
            
            # Add legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
        plt.tight_layout()
        
        if output_file:
            task_output = output_file.replace('.', '_tasks.')
            plt.savefig(task_output, dpi=300, bbox_inches='tight')
            logger.info(f"Task-specific visualization saved to {task_output}")
        else:
            plt.show()
    
    def save_results(self, output_dir: str):
        """
        Save benchmark results to files.
        
        Args:
            output_dir: Directory to save files to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results as JSON
        results_file = os.path.join(output_dir, "benchmark_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate and save report
        report_file = os.path.join(output_dir, "benchmark_report.csv")
        self.generate_report(output_file=report_file)
        
        # Generate and save visualizations
        vis_file = os.path.join(output_dir, "benchmark_visualization.png")
        self.visualize_results(output_file=vis_file)
        
        logger.info(f"All results saved to {output_dir}")
    
    def load_results(self, input_file: str):
        """
        Load benchmark results from a file.
        
        Args:
            input_file: Path to the results file
        """
        with open(input_file, 'r') as f:
            self.results = json.load(f)
        
        logger.info(f"Loaded results from {input_file}")