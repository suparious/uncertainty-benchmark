"""
Prompt formatter for LLM Uncertainty Benchmarking.
"""

from typing import Dict, List, Optional, Any, Union
from .base import BasePromptStrategy
from .shared import SharedInstructionPromptStrategy
from .task_specific import TaskSpecificPromptStrategy


class PromptFormatter:
    """
    Class for formatting prompts using different strategies.
    """
    
    def __init__(self):
        """
        Initialize the prompt formatter with all available strategies.
        """
        self.strategies = {
            "base": BasePromptStrategy(),
            "shared_instruction": SharedInstructionPromptStrategy(),
            "task_specific": TaskSpecificPromptStrategy()
        }
    
    def format_prompt(
        self,
        item: Dict[str, Any],
        strategy: str = "base",
        is_demo: bool = False,
        task_type: Optional[str] = None
    ) -> str:
        """
        Format a prompt using the specified strategy.
        
        Args:
            item: Dataset item
            strategy: Name of the prompt strategy to use
            is_demo: Whether this is a demonstration example
            task_type: Optional task type for context
            
        Returns:
            Formatted prompt
        """
        if strategy not in self.strategies:
            raise ValueError(f"Unknown prompt strategy: {strategy}")
        
        return self.strategies[strategy].format_prompt(
            item=item,
            is_demo=is_demo,
            task_type=task_type
        )
    
    def format_with_demonstrations(
        self,
        item: Dict[str, Any],
        demonstrations: List[Dict[str, Any]],
        strategy: str = "base",
        task_type: Optional[str] = None
    ) -> str:
        """
        Format a prompt with demonstrations using the specified strategy.
        
        Args:
            item: Dataset item to create prompt for
            demonstrations: List of demonstration examples
            strategy: Name of the prompt strategy to use
            task_type: Optional task type for context
            
        Returns:
            Formatted prompt with demonstrations
        """
        if strategy not in self.strategies:
            raise ValueError(f"Unknown prompt strategy: {strategy}")
        
        # Format each demonstration
        demos_text = ""
        for demo in demonstrations:
            demo_prompt = self.strategies[strategy].format_prompt(
                item=demo,
                is_demo=True,
                task_type=task_type
            )
            demos_text += f"{demo_prompt}\n\n"
        
        # Format the current item
        item_prompt = self.strategies[strategy].format_prompt(
            item=item,
            is_demo=False,
            task_type=task_type
        )
        
        # Combine demonstrations and current item
        prompt = f"{demos_text}{item_prompt}"
        
        return prompt
