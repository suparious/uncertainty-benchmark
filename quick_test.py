#!/usr/bin/env python
"""
Quick test script for the LLM Uncertainty Benchmark.
This script tests the benchmark with a small sample size on one task.
It uses the improved dataset loading from dataset_utils.py.
"""

import os
import argparse
import logging
import random
from typing import List, Dict, Any
import numpy as np
import requests
import json
from tqdm import tqdm

# Import dataset utilities directly
from dataset_utils import load_hellaswag_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def quick_test(api_base_url, api_key, model_name, output_dir):
    """
    Run a quick test of the benchmark with one model on a few samples.
    """
    print(f"Running quick test with model: {model_name}")
    
    # Load a small set of samples (just 5 for quick testing)
    sample_size = 5
    print(f"Loading {sample_size} samples from HellaSwag dataset for quick testing...")
    
    try:
        dataset = load_hellaswag_dataset(sample_size)
        
        if not dataset:
            print("Failed to load dataset. Exiting test.")
            return
            
        print(f"Successfully loaded {len(dataset)} samples.")
        
        # Add options E and F to all questions
        for item in dataset:
            item['choices'].extend(['I don\'t know', 'None of the above'])
            item['choice_labels'].extend(['E', 'F'])
        
        # Process each sample
        correct_count = 0
        
        for i, item in enumerate(dataset):
            print(f"\nTesting sample {i+1}/{len(dataset)}")
            
            # Format the prompt
            prompt = format_prompt(item)
            
            # Get prediction
            try:
                prediction = get_prediction(api_base_url, api_key, model_name, prompt)
                print(f"Model prediction: {prediction}")
                print(f"Correct answer: {item['answer']}")
                
                if prediction == item['answer']:
                    correct_count += 1
                    print("✓ Correct!")
                else:
                    print("✗ Incorrect")
            except Exception as e:
                print(f"Error getting prediction: {e}")
                continue
        
        # Report results
        if len(dataset) > 0:
            accuracy = correct_count / len(dataset) * 100
            print(f"\nQuick test results: {correct_count}/{len(dataset)} correct ({accuracy:.1f}%)")
        
        print("\nQuick test completed successfully!")
        
    except Exception as e:
        print(f"Error during quick test: {e}")

def format_prompt(item):
    """Format a simple prompt for the item."""
    context_text = ""
    if item['context']:
        context_text = f"Context: {item['context']}\n\n"
    
    choices_text = "Choices:\n"
    for label, choice in zip(item['choice_labels'], item['choices']):
        choices_text += f"{label}. {choice}\n"
    
    prompt = f"{context_text}Question: {item['question']}\n\n{choices_text}\nAnswer:"
    
    return prompt

def get_prediction(api_base_url, api_key, model_name, prompt):
    """Get prediction from the model API."""
    headers = {
        "Content-Type": "application/json"
    }
    
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Use standard completion API
    data = {
        "model": model_name,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": 1,
        "stop": ["\n"]
    }
    
    response = requests.post(
        f"{api_base_url}/completions", 
        headers=headers,
        json=data
    )
    
    response.raise_for_status()
    result = response.json()
    
    # Extract the prediction (should be a single letter A-F)
    prediction = result["choices"][0]["text"].strip()
    
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick test for LLM Uncertainty Benchmark")
    parser.add_argument("--api-base", required=True, help="Base URL for the OpenAI-compatible API")
    parser.add_argument("--api-key", help="API key (if required)")
    parser.add_argument("--model", required=True, help="Name of the model to test")
    parser.add_argument("--output-dir", default="./quicktest_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    quick_test(args.api_base, args.api_key, args.model, args.output_dir)
