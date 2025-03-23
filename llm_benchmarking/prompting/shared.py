"""
Shared instruction prompt strategy for LLM Uncertainty Benchmarking.
"""

from typing import Dict, List, Optional, Any
from .base import BasePromptStrategy


class SharedInstructionPromptStrategy(BasePromptStrategy):
    """
    Prompt strategy with shared instructions for all tasks.
    """
    
    def __init__(self):
        """
        Initialize the shared instruction prompt strategy.
        """
        super().__init__()
        self.name = "shared_instruction"
    
    def format_prompt(
        self,
        item: Dict[str, Any],
        is_demo: bool = False,
        task_type: Optional[str] = None
    ) -> str:
        """
        Format a single prompt with shared instructions.
        
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
        
        # Add shared instructions for all tasks
        instruction = "Below are some examples of multiple-choice questions with six potential answers. For each question, only one option is correct.\n\n"
        instruction += "Now make your best effort and select the correct answer for the following question. You only need to output the option.\n\n"
        
        return f"{instruction}{base_prompt}"
