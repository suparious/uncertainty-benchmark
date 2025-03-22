#!/usr/bin/env python
"""
Quick test script for the LLM Uncertainty Benchmark.
This script tests the benchmark with a small sample size on one task.
"""

import os
import argparse
from main import LLMUncertaintyBenchmark

def quick_test(api_base_url, api_key, model_name, output_dir):
    """
    Run a quick test of the benchmark with one model on one task.
    """
    print(f"Running quick test with model: {model_name}")
    
    # Initialize with small sample size to test quickly
    benchmark = LLMUncertaintyBenchmark(
        api_base_url=api_base_url,
        api_key=api_key,
        calibration_ratio=0.5,
        error_rate=0.1,
        num_demonstrations={"qa": 2}  # Only use 2 demos for faster testing
    )
    
    # Only prepare the QA dataset with a small sample
    class QuickTestBenchmark(LLMUncertaintyBenchmark):
        def _load_mmlu_dataset(self, category, samples_per_category):
            """Override to use a much smaller dataset for testing."""
            return super()._load_mmlu_dataset(category, min(samples_per_category, 5))
    
    benchmark = QuickTestBenchmark(
        api_base_url=api_base_url,
        api_key=api_key,
        calibration_ratio=0.5,
        error_rate=0.1,
        num_demonstrations={"qa": 2}
    )
    
    # Prepare only QA dataset
    print("Preparing QA dataset (small sample)...")
    benchmark.prepare_datasets(tasks=["qa"], sample_size=20)
    
    # Test with just one prompting strategy
    print("Evaluating model...")
    is_chat_model = "chat" in model_name.lower() or "instruct" in model_name.lower()
    benchmark.evaluate_model(
        model_name=model_name,
        tasks=["qa"],
        prompt_strategies=["base"],
        use_chat_template=is_chat_model
    )
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_dir}...")
    benchmark.save_results(output_dir)
    
    # Print results
    report = benchmark.generate_report()
    print("\nQUICK TEST RESULTS:")
    print(report)
    
    print("\nTest completed successfully! If everything worked, you can now run the full benchmark.")
    print("Note: The results from this quick test are based on a very small sample and shouldn't be used for any real evaluation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick test for LLM Uncertainty Benchmark")
    parser.add_argument("--api-base", required=True, help="Base URL for the OpenAI-compatible API")
    parser.add_argument("--api-key", help="API key (if required)")
    parser.add_argument("--model", required=True, help="Name of the model to test")
    parser.add_argument("--output-dir", default="./quicktest_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    quick_test(args.api_base, args.api_key, args.model, args.output_dir)