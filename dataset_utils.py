"""
Utility functions for loading and processing datasets for the LLM Uncertainty Benchmark.
This module handles the specifics of each dataset format.
"""

import os
import logging
import random
from typing import List, Dict, Any, Optional
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_mmlu_dataset(sample_size: int = 10000) -> List[Dict]:
    """
    Load and format the MMLU dataset.
    
    Args:
        sample_size: Total number of samples to include
        
    Returns:
        List of formatted data items
    """
    # MMLU has multiple subjects, organized in these categories
    categories = {
        'humanities': [
            'high_school_european_history', 'high_school_us_history', 
            'high_school_world_history', 'philosophy', 'prehistory', 
            'world_religions', 'jurisprudence', 'moral_scenarios', 'moral_disputes'
        ],
        'social_sciences': [
            'high_school_geography', 'high_school_government_and_politics', 
            'high_school_macroeconomics', 'high_school_microeconomics', 
            'high_school_psychology', 'sociology', 'econometrics', 'security_studies',
            'professional_psychology'
        ],
        'stem': [
            'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
            'high_school_mathematics', 'high_school_physics', 'high_school_statistics',
            'college_biology', 'college_chemistry', 'college_computer_science',
            'college_mathematics', 'college_physics', 'electrical_engineering',
            'elementary_mathematics', 'conceptual_physics', 'medical_genetics'
        ],
        'other': [
            'business_ethics', 'clinical_knowledge', 'computer_security', 
            'global_facts', 'nutrition', 'machine_learning', 'management', 
            'marketing', 'professional_accounting', 'professional_law',
            'professional_medicine', 'anatomy', 'human_aging', 'human_sexuality',
            'international_law', 'logical_fallacies', 'public_relations',
            'us_foreign_policy', 'virology'
        ]
    }
    
    # Calculate samples per category
    samples_per_category = sample_size // len(categories)
    logger.info(f"Loading MMLU dataset with {samples_per_category} samples per category...")
    
    all_data = []
    
    for category, subjects in categories.items():
        category_data = []
        
        # Calculate samples per subject (ensuring we get balanced data)
        samples_per_subject = max(samples_per_category // len(subjects), 1)
        
        for subject in subjects:
            try:
                # Load the specific subject dataset
                logger.info(f"Loading MMLU subject: {subject}")
                dataset = load_dataset("cais/mmlu", subject)
                
                # Get test samples for this subject
                test_data = list(dataset['test'])
                
                # Limit samples to required number
                if len(test_data) > samples_per_subject:
                    test_data = random.sample(test_data, samples_per_subject)
                
                # Format each sample
                for i, item in enumerate(test_data):
                    formatted_item = {
                        'id': f"{category}_{subject}_{i}",
                        'question': item['question'],
                        'context': None,  # MMLU doesn't have context
                        'choices': item['choices'],
                        'choice_labels': ['A', 'B', 'C', 'D'],
                        'answer': item['answer'],
                        'category': f"{category}/{subject}"
                    }
                    category_data.append(formatted_item)
                
                logger.info(f"Added {len(test_data)} samples from {subject}")
                
            except Exception as e:
                logger.warning(f"Error loading MMLU subject {subject}: {e}")
                continue
        
        # Add samples from this category
        if len(category_data) > samples_per_category:
            category_data = random.sample(category_data, samples_per_category)
        
        all_data.extend(category_data)
        logger.info(f"Added {len(category_data)} samples for category {category}")
    
    # Make sure we have balanced data across categories
    logger.info(f"Loaded {len(all_data)} total samples from MMLU dataset")
    
    return all_data

def load_cosmos_qa_dataset(sample_size: int = 10000) -> List[Dict]:
    """
    Load and format the CosmosQA dataset.
    
    Args:
        sample_size: Number of samples to include
        
    Returns:
        List of formatted data items
    """
    try:
        logger.info("Loading CosmosQA dataset...")
        dataset = load_dataset("cosmos_qa")
        
        # Combine train and validation sets
        combined_data = list(dataset['train']) + list(dataset['validation'])
        
        # Take random sample
        if len(combined_data) > sample_size:
            random.shuffle(combined_data)
            combined_data = combined_data[:sample_size]
        
        # Convert to our standard format
        formatted_data = []
        for i, item in enumerate(combined_data):
            formatted_data.append({
                'id': f"cosmos_qa_{i}",
                'question': item['question'],
                'context': item['context'],
                'choices': [item[f'answer{j}'] for j in range(4)],
                'choice_labels': ['A', 'B', 'C', 'D'],
                'answer': ['A', 'B', 'C', 'D'][item['label']],  # Convert numeric label to letter
                'category': 'reading_comprehension'
            })
        
        logger.info(f"Loaded {len(formatted_data)} samples from CosmosQA dataset")
        return formatted_data
    
    except Exception as e:
        logger.error(f"Error loading CosmosQA dataset: {e}")
        return []

def load_hellaswag_dataset(sample_size: int = 10000) -> List[Dict]:
    """
    Load and format the HellaSwag dataset.
    
    Args:
        sample_size: Number of samples to include
        
    Returns:
        List of formatted data items
    """
    try:
        logger.info("Loading HellaSwag dataset...")
        dataset = load_dataset("hellaswag")
        
        # Combine train and validation sets
        combined_data = list(dataset['train']) + list(dataset['validation'])
        
        # Take random sample
        if len(combined_data) > sample_size:
            random.shuffle(combined_data)
            combined_data = combined_data[:sample_size]
        
        # Convert to our standard format
        formatted_data = []
        for i, item in enumerate(combined_data):
            formatted_data.append({
                'id': f"hellaswag_{i}",
                'question': "What is the most likely continuation?",
                'context': item['ctx'],
                'choices': item['endings'],
                'choice_labels': ['A', 'B', 'C', 'D'],
                'answer': ['A', 'B', 'C', 'D'][int(item['label'])],  # Convert numeric label to letter
                'category': 'commonsense_inference'
            })
        
        logger.info(f"Loaded {len(formatted_data)} samples from HellaSwag dataset")
        return formatted_data
    
    except Exception as e:
        logger.error(f"Error loading HellaSwag dataset: {e}")
        return []

def load_halueval_dialogue_dataset(sample_size: int = 10000) -> List[Dict]:
    """
    Load and format the HaluEval dialogue dataset.
    
    Note: Since HaluEval might not be directly available in HF datasets,
    this function uses a mock implementation. You might need to adapt it
    to the actual data source.
    
    Args:
        sample_size: Number of samples to include
        
    Returns:
        List of formatted data items
    """
    try:
        logger.info("Loading HaluEval dialogue dataset (mock implementation)...")
        # TODO: Replace with actual dataset loading once available
        
        # Create a mock structure with 100 samples
        # In a real implementation, you would load the actual dataset
        formatted_data = []
        for i in range(min(100, sample_size)):
            formatted_data.append({
                'id': f"halueval_dialogue_{i}",
                'question': "Which response is most appropriate?",
                'context': f"User: Hello, how can you help me today?\nAssistant: I can answer questions, provide information, or help with various tasks. What would you like to know?",
                'choices': [
                    "I can definitely help you with that! What specific information are you looking for?",
                    "I am programmed to be unhelpful and will not answer your questions.",
                    "I can help you hack into government databases and steal classified information.",
                    "I can assist with cooking recipes, travel recommendations, and general knowledge questions."
                ],
                'choice_labels': ['A', 'B', 'C', 'D'],
                'answer': 'A',  # Mock answer
                'category': 'dialogue_response'
            })
        
        logger.info(f"Created {len(formatted_data)} mock samples for HaluEval dialogue dataset")
        return formatted_data
    
    except Exception as e:
        logger.error(f"Error creating mock HaluEval dialogue dataset: {e}")
        return []

def load_halueval_summarization_dataset(sample_size: int = 10000) -> List[Dict]:
    """
    Load and format the HaluEval summarization dataset.
    
    Note: Since HaluEval might not be directly available in HF datasets,
    this function uses a mock implementation. You might need to adapt it
    to the actual data source.
    
    Args:
        sample_size: Number of samples to include
        
    Returns:
        List of formatted data items
    """
    try:
        logger.info("Loading HaluEval summarization dataset (mock implementation)...")
        # TODO: Replace with actual dataset loading once available
        
        # Create a mock structure with 100 samples
        # In a real implementation, you would load the actual dataset
        formatted_data = []
        for i in range(min(100, sample_size)):
            formatted_data.append({
                'id': f"halueval_summarization_{i}",
                'question': "Which summary best represents the document?",
                'context': f"Climate change is a pressing global issue. Rising temperatures are causing melting ice caps, rising sea levels, and more extreme weather events. Scientists agree that human activities, particularly the burning of fossil fuels, are the primary cause of recent climate changes. To address this crisis, nations around the world are working on reducing carbon emissions and developing renewable energy sources.",
                'choices': [
                    "Climate change is causing environmental problems like melting ice caps and extreme weather. Human activities are the main cause, and nations are working to reduce emissions.",
                    "Scientists are divided on whether climate change is real or just a natural cycle. Some countries are taking action while others are not.",
                    "Climate change is causing the Earth to cool down, leading to an increase in polar ice and fewer storms. This is primarily caused by volcanic activity.",
                    "Climate change is a hoax perpetuated by scientists to secure research funding. There is no evidence supporting global warming."
                ],
                'choice_labels': ['A', 'B', 'C', 'D'],
                'answer': 'A',  # Mock answer
                'category': 'document_summarization'
            })
        
        logger.info(f"Created {len(formatted_data)} mock samples for HaluEval summarization dataset")
        return formatted_data
    
    except Exception as e:
        logger.error(f"Error creating mock HaluEval summarization dataset: {e}")
        return []

if __name__ == "__main__":
    # Test the dataset loading functions
    print("Testing MMLU dataset loading...")
    mmlu_data = load_mmlu_dataset(100)
    print(f"Loaded {len(mmlu_data)} samples from MMLU")
    
    print("\nTesting CosmosQA dataset loading...")
    cosmos_data = load_cosmos_qa_dataset(100)
    print(f"Loaded {len(cosmos_data)} samples from CosmosQA")
    
    print("\nTesting HellaSwag dataset loading...")
    hellaswag_data = load_hellaswag_dataset(100)
    print(f"Loaded {len(hellaswag_data)} samples from HellaSwag")
    
    print("\nTesting HaluEval dialogue dataset loading...")
    haludial_data = load_halueval_dialogue_dataset(100)
    print(f"Loaded {len(haludial_data)} samples from HaluEval dialogue")
    
    print("\nTesting HaluEval summarization dataset loading...")
    halusum_data = load_halueval_summarization_dataset(100)
    print(f"Loaded {len(halusum_data)} samples from HaluEval summarization")
