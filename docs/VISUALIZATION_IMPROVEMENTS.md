# LLM Benchmarking Visualization Improvements

This document outlines the visualization improvements made to the LLM Benchmarking Suite.

## Major Enhancements

1. **Overall Visualization Improvements**
   - Better color schemes for all charts
   - Improved readability with background colors and grid lines
   - Enhanced label placement to avoid overlapping
   - Optimized figure sizes based on content
   - Added clear, descriptive titles and consistent styling

2. **Model Comparison Charts**
   - Enhanced scatter plots with better model name displays
   - Improved bar charts with clear value labels
   - Better heatmap visualization for task comparisons

3. **Task-specific Visualizations**
   - Dual-axis charts for comparing accuracy and set size
   - Consistent color schemes across all task charts
   - Clear separation between tasks for easier comparison

4. **Advanced Visualization Types**
   - Added correlation heatmaps for metric analysis
   - Improved prompt strategy comparison charts
   - Enhanced model scaling visualizations

5. **Code Quality Improvements**
   - Better dependency management
   - Improved error handling for visualization dependencies
   - Enhanced logging with more detailed information
   - Support for saving figures with consistent naming

## Installation

To use the enhanced visualization capabilities, install the additional dependencies:

```bash
# Option 1: Using pip with requirements file
pip install -r requirements-viz.txt

# Option 2: Using setup.py extras
pip install -e .[viz]
```

## Usage Examples

```bash
# Basic visualization with overview and task breakdown
llm-analyze --input-dirs ./benchmark_results --mode basic

# Task comparison visualization
llm-analyze --input-dirs ./benchmark_results --mode comparative

# Metric correlation analysis
llm-analyze --input-dirs ./benchmark_results --mode correlations

# Prompt strategy comparison for a specific model
llm-analyze --input-dirs ./benchmark_results --mode prompt --models "mistral-7b"

# Model scaling analysis
llm-analyze --input-dirs ./benchmark_results --mode scaling --model-family llama --model-sizes 7B 13B 70B
```

## Output Files

The visualizations are saved with the following naming convention:

- Basic mode: `model_comparison.png` and `model_comparison_tasks.png`
- Comparative mode: `model_comparison_task_comparison.png`
- Correlations mode: `model_comparison_correlations.png`
- Prompt mode: `model_comparison_prompt_strategies.png`
- Scaling mode: `model_comparison_scaling.png`

You can specify a custom output directory with `--output-dir` and a custom prefix with `--output-prefix`.
