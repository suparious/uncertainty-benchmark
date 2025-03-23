"""
Task-specific prompt strategy for LLM Uncertainty Benchmarking.
"""

from typing import Dict, List, Optional, Any
from .base import BasePromptStrategy


class TaskSpecificPromptStrategy(BasePromptStrategy):
    """
    Prompt strategy with task-specific instructions.
    """
    
    def __init__(self):
        """
        Initialize the task-specific prompt strategy.
        """
        super().__init__()
        self.name = "task_specific"
        
        # Task-specific instructions
        self.task_instructions = {
            "qa": "Below are some examples of multiple-choice questions about question answering. "
                 "Each question should be answered based on your world knowledge and problem solving ability.",
            
            "rc": "Below are some examples of multiple-choice questions about reading comprehension. "
                 "Each question should be answered based on the given context and commonsense reasoning when necessary.",
            
            "ci": "Below are some examples of multiple-choice questions about commonsense natural language inference. "
                 "For each question, there is a given context and the answer is the option that most likely follows the context.",
            
            "drs": "Below are some examples of multiple-choice questions about dialogue response selection. "
                  "For each question, the answer is the option that represents the most suitable response for the given dialogue history, "
                  "without hallucination and non-factual information.",
            
            "ds": "Below are some examples of multiple-choice questions about document summarization. "
                 "For each question, the answer is the option that accurately summarizes the given document without "
                 "hallucination and non-factual information."
        }
    
    def format_prompt(
        self,
        item: Dict[str, Any],
        is_demo: bool = False,
        task_type: Optional[str] = None
    ) -> str:
        """
        Format a single prompt with task-specific instructions.
        
        Args:
            item: Dataset item
            is_demo: Whether this is a demonstration example
            task_type: Optional task type for context
            
        Returns:
            Formatted prompt
        """
        # Get base format without instructions
        base_prompt = super().format_prompt(item, is_demo, task_type)
        
        # For demonstrations, we don't add instructions to save space
        if is_demo:
            return base_prompt
        
        # Get task-specific instruction if available
        instruction = ""
        if task_type in self.task_instructions:
            instruction = self.task_instructions[task_type] + "\n\n"
            
        # Add common instruction
        instruction += "Now make your best effort and select the correct answer for the following question. You only need to output the option.\n\n"
        
        return f"{instruction}{base_prompt}"
