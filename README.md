# LLM Uncertainty Benchmarking

A framework for evaluating Language Models via uncertainty quantification based on conformal prediction.

## Overview

This package implements a benchmarking framework for LLMs that evaluates both accuracy and uncertainty quantification, based on the paper "Benchmarking LLMs via Uncertainty Quantification". It offers:

- Multiple benchmark tasks (QA, reading comprehension, commonsense inference, etc.)
- Different prompt strategies to test model robustness
- Conformal prediction for uncertainty quantification
- Parallel processing capabilities for faster evaluation
- Comprehensive analysis and visualization tools

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-benchmarking.git
cd llm-benchmarking

# Install the package
pip install -e .
```

## Quick Start

```python
from llm_benchmarking.benchmark import LLMBenchmark

# Create a benchmark instance
benchmark = LLMBenchmark(
    api_base_url="http://localhost:8000/v1",
    api_key="your-api-key"  # Optional, if needed
)

# Prepare datasets
benchmark.prepare_datasets(sample_size=100)  # Use a small sample size for quick testing

# Evaluate a model
benchmark.evaluate_model(
    model_name="mistral-7b",
    use_chat_template=False  # Set to True for instruction-tuned models
)

# Save results
benchmark.save_results("./results/mistral-7b")

# Generate a report
report = benchmark.generate_report()
print(report)

# Visualize results
benchmark.visualize_results()
```

## Command-Line Interface

The package provides command-line tools for benchmarking and analysis:

```bash
# Run a benchmark
llm-benchmark --api-base http://localhost:8000/v1 --model mistral-7b --sample-size 100

# Run benchmark with parallel processing
llm-benchmark --api-base http://localhost:8000/v1 --model mistral-7b --parallel --batch-size 10 --max-workers 4

# Analyze results
llm-analyze --input-dirs ./benchmark_results --mode basic

# Compare model scaling (e.g., for different sizes of the same model family)
llm-analyze --input-dirs ./benchmark_results --mode scaling --model-family llama --model-sizes 7B 13B 70B
```

## Features

### Tasks

The benchmark includes the following tasks:

- **qa**: Question Answering (MMLU dataset)
- **rc**: Reading Comprehension (CosmosQA dataset)
- **ci**: Commonsense Inference (HellaSwag dataset)
- **drs**: Dialogue Response Selection (HaluEval dialogue dataset)
- **ds**: Document Summarization (HaluEval summarization dataset)

### Prompt Strategies

Three prompt strategies are available:

- **base**: Minimal instructions
- **shared_instruction**: Shared instructions for all tasks
- **task_specific**: Task-specific instructions

### Metrics

The benchmark evaluates models using three key metrics:

- **Accuracy**: Percentage of correct predictions
- **Coverage Rate**: Percentage of tests where the correct answer is in the prediction set
- **Set Size**: Average size of the prediction set (smaller is better)

### Parallel Processing

For faster evaluation, the benchmark offers two parallelization approaches:

- **Async-based**: Using asyncio and aiohttp
- **Thread-based**: Using ThreadPoolExecutor

## Examples

See the `examples` directory for sample scripts:

- `basic_benchmark.py`: Basic benchmarking
- `parallel_benchmark.py`: Benchmarking with parallel processing
- `model_comparison.py`: Comparing multiple models

## Documentation

For more detailed documentation, please refer to the docstrings in the code and the examples.

## Citation

If you use this framework in your research, please cite the original paper:

```bibtex
@article{ye2024llm_uq,
  title={Benchmarking LLMs via Uncertainty Quantification},
  author={Ye, Fanghua and Yang MingMing and Pang, Jianhui and Wang, Longyue and Wong, Derek F and Yilmaz Emine and Shi, Shuming and Tu, Zhaopeng},
  journal={arXiv preprint arXiv:2401.12794},
  year={2024}
  }
```

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
