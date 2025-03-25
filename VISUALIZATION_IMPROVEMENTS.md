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

5. **Robustness Improvements**
   - Enhanced error handling for all visualization functions
   - Added automatic detection and handling of duplicate model names
   - Improved model name formatting to better distinguish similar models
   - Added detailed error messages with helpful suggestions for common issues
   - Better dependency management with optional visualization enhancements
   - Improved logging with more detailed information

## Installation

The visualization functionality works with the core dependencies, but you can enhance it with optional packages:

```bash
# Basic installation - all core features will work
pip install -e .

# Enhanced visualization with better label placement (recommended)
pip install -e ".[viz]"   # Regular shell
pip install -e './[viz]'  # For zsh and other shells that interpret square brackets
```

## Compatibility Notes

- The code now includes a fallback mechanism for when the `adjustText` package is not available
- A custom `rgb2hex` function is implemented to ensure compatibility with all versions of seaborn
- All visualization functions now handle missing dependencies gracefully
- Added support for handling duplicate model names in comparative visualizations
- Improved error messages that provide specific guidance when problems occur

## Handling Many Similar Models

When analyzing many similar models (e.g., multiple variants of LLAMA or Mistral), you may encounter challenges with distinguishing between them in visualizations. Here are some recommended approaches:

1. **Use model filtering**:
   ```bash
   # Analyze only specific models of interest
   llm-analyze --input-dirs ./benchmark_results --models "modelA" "modelB" "modelC"
   ```

2. **Focus on top performers**:
   ```bash
   # Show only the top 5 models by accuracy
   llm-analyze --input-dirs ./benchmark_results --top-k 5
   ```

3. **Use basic mode** if comparative visualization fails due to duplicate model names:
   ```bash
   llm-analyze --input-dirs ./benchmark_results --mode basic
   ```

4. **Use scaling analysis** for comparing models within a family:
   ```bash
   llm-analyze --input-dirs ./benchmark_results --mode scaling --model-family mistral --model-sizes 7B 8x7B 8x22B
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
