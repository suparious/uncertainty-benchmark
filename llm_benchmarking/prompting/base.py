"""
Base prompt strategy for LLM Uncertainty Benchmarking.
"""

from typing import Dict, List, Optional, Any


class BasePromptStrategy:
    """
    Base prompt strategy with minimal instructions.
    """
    
    def __init__(self):
        """
        Initialize the base prompt strategy.
        """
        self.name = "base"
    
    def format_prompt(
        self,
        item: Dict[str, Any],
        is_demo: bool = False,
        task_type: Optional[str] = None
    ) -> str:
        """
        Format a single prompt for a task item.
        
        Args:
            item: Dataset item
            is_demo: Whether this is a demonstration example
            task_type: Optional task type for context
            
        Returns:
            Formatted prompt
        """
        context_type = self._get_context_type(task_type)
        
        context_text = ""
        if context_type and item.get('context'):
            context_text = f"{context_type}: {item['context']}\n"
        
        choices_text = "Choices:\n"
        for label, choice in zip(item['choice_labels'], item['choices']):
            choices_text += f"{label}. {choice}\n"
        
        answer_text = ""
        if is_demo:
            # For demonstrations, include the answer
            answer_label = self._get_answer_label(item)
            answer_text = f"Answer: {answer_label}"
        else:
            # For test items, just have the prompt "Answer:"
            answer_text = "Answer:"
        
        # Assemble the prompt
        prompt = f"{context_text}Question: {item['question']}\n{choices_text}{answer_text}"
        
        return prompt
    
    def _get_context_type(self, task_type: Optional[str]) -> Optional[str]:
        """
        Get the appropriate context type label based on the task type.
        
        Args:
            task_type: Task type code
            
        Returns:
            Context type label or None if not applicable
        """
        if task_type in ["rc", "ci"]:
            return "Context"
        elif task_type == "drs":
            return "Dialogue"
        elif task_type == "ds":
            return "Document"
        else:
            return None
    
    def _get_answer_label(self, item: Dict[str, Any]) -> str:
        """
        Get the answer label from the item, handling different answer formats.
        
        Args:
            item: Dataset item
            
        Returns:
            Answer label (e.g., 'A', 'B', etc.)
        """
        if 'answer' not in item:
            return "Unknown"
            
        # Handle different answer formats (int, str, etc.)
        answer = item['answer']
        
        if isinstance(answer, int):
            # If it's already an integer index
            if 0 <= answer < len(item['choice_labels']):
                return item['choice_labels'][answer]
        elif isinstance(answer, str):
            if answer in item['choice_labels']:
                # If it's already a label like 'A', 'B', etc.
                return answer
            elif answer.isdigit():
                # If it's a string representing a digit
                idx = int(answer)
                if 0 <= idx < len(item['choice_labels']):
                    return item['choice_labels'][idx]
        
        # If we can't determine the answer format, return as is
        return str(answer)
