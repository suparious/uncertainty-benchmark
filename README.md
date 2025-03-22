# LLM Uncertainty Benchmark

A comprehensive benchmarking framework for evaluating Large Language Models (LLMs) using uncertainty quantification, based on the paper ["Benchmarking LLMs via Uncertainty Quantification"](https://arxiv.org/abs/2401.12794).

## Overview

Traditional LLM benchmarks typically focus solely on accuracy metrics. This framework extends LLM evaluation by incorporating uncertainty quantification using conformal prediction, providing a more comprehensive assessment of model performance.

The benchmark evaluates LLMs on 5 core NLP tasks:
1. **Question Answering (QA)** - MMLU dataset
2. **Reading Comprehension (RC)** - CosmosQA dataset
3. **Commonsense Inference (CI)** - HellaSwag dataset
4. **Dialogue Response Selection (DRS)** - HaluEval dataset
5. **Document Summarization (DS)** - HaluEval dataset

Each task is formulated as a multiple-choice question with 6 options (A-F), with the last two options being "I don't know" and "None of the above".

## Key Metrics

The benchmark provides three key metrics for each model:

- **Accuracy (Acc)**: Traditional accuracy measure - percentage of questions answered correctly
- **Coverage Rate (CR)**: Percentage of test instances where the true label is in the prediction set
- **Set Size (SS)**: Average size of the prediction sets - a measure of model uncertainty (smaller is better)

## Features

- Evaluate LLMs using conformal prediction for uncertainty quantification
- Support for both base and instruction-tuned models
- Multiple prompting strategies for robust evaluation
- Comprehensive reporting and visualization tools
- Analysis capabilities for model scaling and instruction tuning effects

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-uncertainty-benchmark.git
cd llm-uncertainty-benchmark

# Install required packages
pip install -r requirements.txt
```

## Usage

### Basic Model Evaluation

```python
from main import LLMUncertaintyBenchmark

# Initialize the benchmark
benchmark = LLMUncertaintyBenchmark(
    api_base_url="http://localhost:8000/v1",  # Your vLLM-OpenAI server endpoint
    api_key=None,  # API key if required
    calibration_ratio=0.5,  # Ratio of data to use for calibration
    error_rate=0.1  # Error rate alpha for conformal prediction
)

# Prepare datasets for all tasks
benchmark.prepare_datasets()

# Evaluate a model
benchmark.evaluate_model(
    model_name="meta-llama/Llama-2-13b-hf",
    use_chat_template=False,  # Set to True for instruction-tuned models
    prompt_strategies=["base", "shared_instruction", "task_specific"]
)

# Generate report
report = benchmark.generate_report()
print(report)

# Visualize results
benchmark.visualize_results()

# Save results
benchmark.save_results("./results/llama-2-13b")
```

### Command-line Interface

The package also provides a command-line interface for common benchmarking scenarios:

```bash
# Benchmark a single model
python examples.py --api-base http://localhost:8000/v1 single --model meta-llama/Llama-2-13b-hf

# Compare multiple models
python examples.py --api-base http://localhost:8000/v1 compare --models meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf

# Analyze effect of model scale
python examples.py --api-base http://localhost:8000/v1 scale --family meta-llama/Llama-2 --sizes 7b 13b 70b

# Analyze effect of instruction tuning
python examples.py --api-base http://localhost:8000/v1 instruct --family meta-llama/Llama-2 --sizes 7b 13b
```

## Understanding Results

When analyzing the results, keep in mind:

1. **Accuracy vs. Uncertainty**: Models with higher accuracy don't necessarily have lower uncertainty.
2. **Model Scale Effects**: Larger models may demonstrate different uncertainty patterns compared to smaller ones.
3. **Instruction Tuning Effects**: Instruction-tuned models often show different uncertainty characteristics than their base counterparts.

The key insight from the paper is that both accuracy and uncertainty are important metrics, and they don't always correlate. This provides a more nuanced understanding of model performance.

## Example Results Visualization

![Example Benchmark Visualization](./docs/example_visualization.png)

## Requirements

The framework requires:

1. Access to LLMs via an OpenAI-compatible API (e.g., vLLM server)
2. Python 3.8+ with the following packages:
   - numpy
   - pandas
   - matplotlib
   - seaborn
   - requests
   - tqdm
   - datasets
   - torch

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{ye2024benchmarking,
  title={Benchmarking LLMs via Uncertainty Quantification},
  author={Ye, Fanghua and Yang, Mingming and Pang, Jianhui and Wang, Longyue and Wong, Derek F. and Yilmaz, Emine and Shi, Shuming and Tu, Zhaopeng},
  journal={arXiv preprint arXiv:2401.12794},
  year={2024}
}
```