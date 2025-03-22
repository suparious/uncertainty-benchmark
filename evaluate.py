import os
import argparse
from main import LLMUncertaintyBenchmark

def benchmark_single_model(api_base_url, api_key, model_name, output_dir):
    """
    Benchmark a single model and save results.
    """
    benchmark = LLMUncertaintyBenchmark(
        api_base_url=api_base_url,
        api_key=api_key
    )
    
    # Prepare datasets for all tasks
    benchmark.prepare_datasets()
    
    # Evaluate the model
    is_chat_model = "chat" in model_name.lower() or "instruct" in model_name.lower()
    benchmark.evaluate_model(
        model_name=model_name,
        use_chat_template=is_chat_model
    )
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, model_name.replace("/", "_"))
    benchmark.save_results(model_dir)
    
    print(f"Benchmark results for {model_name} saved to {model_dir}")

def compare_models(api_base_url, api_key, model_names, output_dir):
    """
    Compare multiple models and generate comparative reports.
    """
    benchmark = LLMUncertaintyBenchmark(
        api_base_url=api_base_url,
        api_key=api_key
    )
    
    # Prepare datasets for all tasks
    benchmark.prepare_datasets()
    
    # Evaluate each model
    for model_name in model_names:
        print(f"Evaluating {model_name}...")
        is_chat_model = "chat" in model_name.lower() or "instruct" in model_name.lower()
        benchmark.evaluate_model(
            model_name=model_name,
            use_chat_template=is_chat_model
        )
    
    # Save all results
    os.makedirs(output_dir, exist_ok=True)
    benchmark.save_results(output_dir)
    
    # Generate comparison report and visualization
    comparison_file = os.path.join(output_dir, "model_comparison.csv")
    benchmark.generate_report(output_file=comparison_file)
    
    visualization_file = os.path.join(output_dir, "model_comparison.png")
    benchmark.visualize_results(output_file=visualization_file)
    
    print(f"Comparison results saved to {output_dir}")

def scale_analysis(api_base_url, api_key, model_family, model_sizes, output_dir):
    """
    Analyze how model scale affects uncertainty and accuracy.
    """
    benchmark = LLMUncertaintyBenchmark(
        api_base_url=api_base_url,
        api_key=api_key
    )
    
    # Prepare datasets for all tasks
    benchmark.prepare_datasets()
    
    # Evaluate each model size
    model_names = []
    for size in model_sizes:
        model_name = f"{model_family}-{size}"
        model_names.append(model_name)
        
        print(f"Evaluating {model_name}...")
        is_chat_model = "chat" in model_name.lower() or "instruct" in model_name.lower()
        benchmark.evaluate_model(
            model_name=model_name,
            use_chat_template=is_chat_model
        )
    
    # Save all results
    os.makedirs(output_dir, exist_ok=True)
    benchmark.save_results(output_dir)
    
    # Generate scale analysis visualization
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    # Get report DataFrame
    report_df = benchmark.generate_report(model_names)
    
    # Filter for average results
    avg_results = report_df[report_df['Task'] == 'Average'].copy()
    
    # Sort by model size
    avg_results['Size'] = avg_results['Model'].apply(lambda x: x.split('-')[-1])
    avg_results['Size'] = pd.Categorical(avg_results['Size'], categories=model_sizes, ordered=True)
    avg_results = avg_results.sort_values('Size')
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Accuracy by size
    sns.lineplot(
        data=avg_results,
        x='Size',
        y='Accuracy',
        marker='o',
        markersize=10,
        ax=ax1
    )
    ax1.set_title('Effect of Model Scale on Accuracy')
    ax1.set_xlabel('Model Size')
    ax1.set_ylabel('Accuracy (%)')
    
    # Set Size by model size
    sns.lineplot(
        data=avg_results,
        x='Size',
        y='Set Size',
        marker='o',
        markersize=10,
        ax=ax2
    )
    ax2.set_title('Effect of Model Scale on Uncertainty (Set Size)')
    ax2.set_xlabel('Model Size')
    ax2.set_ylabel('Average Set Size (lower is better)')
    
    plt.tight_layout()
    
    # Save visualization
    viz_file = os.path.join(output_dir, f"{model_family}_scale_analysis.png")
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    
    print(f"Scale analysis for {model_family} saved to {output_dir}")

def instruction_analysis(api_base_url, api_key, model_family, model_sizes, output_dir):
    """
    Analyze how instruction tuning affects uncertainty and accuracy.
    """
    benchmark = LLMUncertaintyBenchmark(
        api_base_url=api_base_url,
        api_key=api_key
    )
    
    # Prepare datasets for all tasks
    benchmark.prepare_datasets()
    
    # Evaluate each model variant
    model_names = []
    for size in model_sizes:
        # Base model
        base_model = f"{model_family}-{size}"
        model_names.append(base_model)
        
        print(f"Evaluating {base_model}...")
        benchmark.evaluate_model(
            model_name=base_model,
            use_chat_template=False
        )
        
        # Chat/instruct model
        chat_model = f"{model_family}-{size}-chat"
        model_names.append(chat_model)
        
        print(f"Evaluating {chat_model}...")
        benchmark.evaluate_model(
            model_name=chat_model,
            use_chat_template=True
        )
    
    # Save all results
    os.makedirs(output_dir, exist_ok=True)
    benchmark.save_results(output_dir)
    
    # Generate instruction analysis visualization
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    # Get report DataFrame
    report_df = benchmark.generate_report(model_names)
    
    # Filter for average results
    avg_results = report_df[report_df['Task'] == 'Average'].copy()
    
    # Extract model info
    avg_results['Size'] = avg_results['Model'].apply(lambda x: x.split('-')[-2] if 'chat' in x else x.split('-')[-1])
    avg_results['Type'] = avg_results['Model'].apply(lambda x: 'Chat' if 'chat' in x else 'Base')
    
    # Sort by size
    avg_results['Size'] = pd.Categorical(avg_results['Size'], categories=model_sizes, ordered=True)
    avg_results = avg_results.sort_values(['Size', 'Type'])
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Accuracy comparison
    sizes = sorted(avg_results['Size'].unique())
    width = 0.35
    x = np.arange(len(sizes))
    
    base_acc = avg_results[avg_results['Type'] == 'Base']['Accuracy'].values
    chat_acc = avg_results[avg_results['Type'] == 'Chat']['Accuracy'].values
    
    ax1.bar(x - width/2, base_acc, width, label='Base')
    ax1.bar(x + width/2, chat_acc, width, label='Chat/Instruct')
    
    ax1.set_title('Effect of Instruction Tuning on Accuracy')
    ax1.set_xlabel('Model Size')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes)
    ax1.legend()
    
    # Set Size comparison
    base_ss = avg_results[avg_results['Type'] == 'Base']['Set Size'].values
    chat_ss = avg_results[avg_results['Type'] == 'Chat']['Set Size'].values
    
    ax2.bar(x - width/2, base_ss, width, label='Base')
    ax2.bar(x + width/2, chat_ss, width, label='Chat/Instruct')
    
    ax2.set_title('Effect of Instruction Tuning on Uncertainty (Set Size)')
    ax2.set_xlabel('Model Size')
    ax2.set_ylabel('Average Set Size (lower is better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sizes)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save visualization
    viz_file = os.path.join(output_dir, f"{model_family}_instruction_analysis.png")
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    
    print(f"Instruction tuning analysis for {model_family} saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Uncertainty Benchmark Examples")
    parser.add_argument("--api-base", required=True, help="Base URL for the OpenAI-compatible API")
    parser.add_argument("--api-key", help="API key (if required)")
    parser.add_argument("--output-dir", default="./benchmark_results", help="Output directory for results")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Single model benchmark
    single_parser = subparsers.add_parser("single", help="Benchmark a single model")
    single_parser.add_argument("--model", required=True, help="Name of the model to benchmark")
    
    # Model comparison
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument("--models", required=True, nargs="+", help="Names of models to compare")
    
    # Scale analysis
    scale_parser = subparsers.add_parser("scale", help="Analyze effect of model scale")
    scale_parser.add_argument("--family", required=True, help="Model family (e.g., llama, mistral)")
    scale_parser.add_argument("--sizes", required=True, nargs="+", help="Model sizes to analyze (e.g., 7B 13B 70B)")
    
    # Instruction tuning analysis
    instruct_parser = subparsers.add_parser("instruct", help="Analyze effect of instruction tuning")
    instruct_parser.add_argument("--family", required=True, help="Model family (e.g., llama, mistral)")
    instruct_parser.add_argument("--sizes", required=True, nargs="+", help="Model sizes to analyze (e.g., 7B 13B)")
    
    args = parser.parse_args()
    
    if args.command == "single":
        benchmark_single_model(args.api_base, args.api_key, args.model, args.output_dir)
    
    elif args.command == "compare":
        compare_models(args.api_base, args.api_key, args.models, args.output_dir)
    
    elif args.command == "scale":
        scale_analysis(args.api_base, args.api_key, args.family, args.sizes, args.output_dir)
    
    elif args.command == "instruct":
        instruction_analysis(args.api_base, args.api_key, args.family, args.sizes, args.output_dir)
    
    else:
        parser.print_help()