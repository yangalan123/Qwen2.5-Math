#!/usr/bin/env python3
"""
Standalone visualization script for drawing best@K curves comparing normal and annealed sampling.
This script imports necessary components from collect_majority_results.py and creates
visualizations for data from ${OUTPUT_DIR}/annealed and ${OUTPUT_DIR}/normal directories.
"""

import argparse
import glob
import json
import os
import random
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import pandas as pd

# Import necessary functions from collect_majority_results.py
from collect_majority_results import (
    process_results, 
    calculate_model_performance,
    DEFAULT_VISUALIZATION_DIR, 
    DEFAULT_FIG_SIZE, 
    DEFAULT_FONT_SIZE, 
    DEFAULT_LINE_WIDTH
)


def parse_arguments():
    """Parse command line arguments for the visualization script."""
    parser = argparse.ArgumentParser(description='Visualize best@K curves for normal vs annealed sampling')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Base output directory containing annealed/ and normal/ subdirectories')
    parser.add_argument('--n_values', type=int, nargs='+', default=[1, 3, 8, 16, 24, 32],
                        help='N values for majority voting (default: [1, 3, 8, 16, 24, 32])')
    parser.add_argument('--bootstrap_num', type=int, default=100,
                        help='Number of bootstrap iterations (default: 100)')
    parser.add_argument('--visualization_dir', type=str, default='./best_at_k_visualizations',
                        help='Directory to save visualization plots (default: ./best_at_k_visualizations)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Specific model name to visualize (if not specified, will auto-detect)')
    parser.add_argument('--data_name', type=str, default='mmlu_stem',
                        help='Data name to look for in results (default: mmlu_stem)')
    parser.add_argument('--prompt_type', type=str, default='cot',
                        help='Prompt type used in evaluation (default: cot)')
    parser.add_argument('--sample_nums', type=int, default=200,
                        help='Number of samples used in evaluation (default: 200)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed used in evaluation (default: 0)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature used in evaluation (default: 1.0)')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p used in evaluation (default: 0.9)')
    parser.add_argument('--n_sampling', type=int, default=64,
                        help='Number of samplings used in evaluation (default: 64)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[10, 6],
                        help='Figure size as width height (default: [10, 6])')
    parser.add_argument('--fontsize', type=int, default=14,
                        help='Font size for plots (default: 14)')
    parser.add_argument('--linewidth', type=int, default=2,
                        help='Line width for plots (default: 2)')
    return parser.parse_args()


def find_result_files(output_dir, data_name, prompt_type, sample_nums, seed, temperature, top_p, n_sampling):
    """
    Find result files in the specified output directory structure.
    
    Args:
        output_dir: Base output directory
        data_name: Name of the dataset
        prompt_type: Type of prompt used
        sample_nums: Number of samples
        seed: Random seed
        temperature: Temperature parameter
        top_p: Top-p parameter
        n_sampling: Number of samplings
        
    Returns:
        tuple: (annealed_file, normal_file) paths, or (None, None) if not found
    """
    # Construct the expected filename pattern
    filename_pattern = f"test_{prompt_type}_{sample_nums}_seed{seed}_t{temperature}_5shot_s0_e-1.jsonl"
    
    # Look for annealed results
    annealed_dir = os.path.join(output_dir, "annealed", data_name)
    annealed_file = os.path.join(annealed_dir, filename_pattern)
    
    # Look for normal results
    normal_dir = os.path.join(output_dir, "normal", data_name)
    normal_file = os.path.join(normal_dir, filename_pattern)
    
    # Check if files exist
    annealed_exists = os.path.exists(annealed_file)
    normal_exists = os.path.exists(normal_file)
    
    if not annealed_exists:
        print(f"Warning: Annealed results file not found: {annealed_file}")
        annealed_file = None
    
    if not normal_exists:
        print(f"Warning: Normal results file not found: {normal_file}")
        normal_file = None
    
    return annealed_file, normal_file


def process_and_analyze_data(result_file, n_values, bootstrap_num):
    """
    Process result file and calculate performance metrics.
    
    Args:
        result_file: Path to the result file
        n_values: List of N values for majority voting
        bootstrap_num: Number of bootstrap iterations
        
    Returns:
        dict: Performance data dictionary
    """
    if result_file is None:
        return None
    
    print(f"Processing file: {result_file}")
    
    # Process results using imported function
    results_by_n, prob_weighted_results_by_n, individual_results = process_results(
        result_file, n_values, bootstrap_num
    )
    
    # Calculate performance metrics
    performance_data = calculate_model_performance(results_by_n)
    
    return performance_data


def plot_best_at_n_scores(annealed_perf, normal_perf, n_values, output_dir, 
                          model_name, temperature, top_p, figsize, fontsize, linewidth):
    """
    Create a plot of best@N scores for annealed vs normal sampling.
    
    Args:
        annealed_perf: Performance data for annealed sampling
        normal_perf: Performance data for normal sampling
        n_values: List of N values
        output_dir: Directory to save the plot
        model_name: Name of the model
        temperature: Temperature parameter
        top_p: Top-p parameter
        figsize: Figure size tuple
        fontsize: Font size
        linewidth: Line width
    """
    plt.rc('font', size=fontsize)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Macaroon color palette
    annealed_color = '#E6B17A'  # Warm beige
    normal_color = '#8B4513'    # Saddle brown
    
    # Plot annealed sampling
    if annealed_perf is not None:
        ax.errorbar(
            annealed_perf['n_values'],
            annealed_perf['mean_scores'],
            yerr=annealed_perf['std_errors'],
            fmt='o-',
            linewidth=linewidth,
            capsize=5,
            label='Annealed Sampling',
            color=annealed_color,
            markersize=8,
            markerfacecolor=annealed_color,
            markeredgecolor='white',
            markeredgewidth=1
        )
    
    # Plot normal sampling
    if normal_perf is not None:
        ax.errorbar(
            normal_perf['n_values'],
            normal_perf['mean_scores'],
            yerr=normal_perf['std_errors'],
            fmt='s-',
            linewidth=linewidth,
            capsize=5,
            label='Normal Sampling',
            color=normal_color,
            markersize=8,
            markerfacecolor=normal_color,
            markeredgecolor='white',
            markeredgewidth=1
        )
    
    ax.set_xscale('log')
    ax.set_xlabel('N (majority voting size)', fontsize=fontsize)
    ax.set_ylabel('Mean Score', fontsize=fontsize)
    ax.set_title(f'Best@N Score Comparison - {model_name}', fontsize=fontsize+2, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=fontsize)
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(n) for n in n_values], fontsize=fontsize-1)
    ax.tick_params(axis='y', labelsize=fontsize-1)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"{model_name}_best_at_n_scores_T{temperature}_p{top_p}.pdf"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved Best@N scores plot to: {plot_path}")
    
    plt.close()


def plot_standard_errors(annealed_perf, normal_perf, n_values, output_dir, 
                         model_name, temperature, top_p, figsize, fontsize, linewidth):
    """
    Create a plot of standard errors for annealed vs normal sampling.
    
    Args:
        annealed_perf: Performance data for annealed sampling
        normal_perf: Performance data for normal sampling
        n_values: List of N values
        output_dir: Directory to save the plot
        model_name: Name of the model
        temperature: Temperature parameter
        top_p: Top-p parameter
        figsize: Figure size tuple
        fontsize: Font size
        linewidth: Line Width
    """
    plt.rc('font', size=fontsize)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Macaroon color palette
    annealed_color = '#E6B17A'  # Warm beige
    normal_color = '#8B4513'    # Saddle brown
    
    # Plot annealed sampling
    if annealed_perf is not None:
        ax.plot(
            annealed_perf['n_values'],
            annealed_perf['std_errors'],
            'o-',
            linewidth=linewidth,
            label='Annealed Sampling',
            color=annealed_color,
            markersize=8,
            markerfacecolor=annealed_color,
            markeredgecolor='white',
            markeredgewidth=1
        )
    
    # Plot normal sampling
    if normal_perf is not None:
        ax.plot(
            normal_perf['n_values'],
            normal_perf['std_errors'],
            's-',
            linewidth=linewidth,
            label='Normal Sampling',
            color=normal_color,
            markersize=8,
            markerfacecolor=normal_color,
            markeredgecolor='white',
            markeredgewidth=1
        )
    
    ax.set_xscale('log')
    ax.set_xlabel('N (majority voting size)', fontsize=fontsize)
    ax.set_ylabel('Standard Error', fontsize=fontsize)
    ax.set_title(f'Standard Error Comparison - {model_name}', fontsize=fontsize+2, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=fontsize)
    ax.set_xticks(n_values)
    ax.set_xticklabels([str(n) for n in n_values], fontsize=fontsize-1)
    ax.tick_params(axis='y', labelsize=fontsize-1)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"{model_name}_standard_errors_T{temperature}_p{top_p}.pdf"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved Standard Error plot to: {plot_path}")
    
    plt.close()


def export_data_to_csv(annealed_perf, normal_perf, output_dir, model_name, temperature, top_p):
    """
    Export performance data to CSV files for later figure merging.
    
    Args:
        annealed_perf: Performance data for annealed sampling
        normal_perf: Performance data for normal sampling
        output_dir: Directory to save CSV files
        model_name: Name of the model
        temperature: Temperature parameter
        top_p: Top-p parameter
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data for Best@N scores
    best_at_n_data = []
    if annealed_perf is not None:
        for i, n in enumerate(annealed_perf['n_values']):
            best_at_n_data.append({
                'N': n,
                'Method': 'Annealed',
                'Mean_Score': annealed_perf['mean_scores'][i],
                'Std_Error': annealed_perf['std_errors'][i],
                'Model': model_name,
                'Temperature': temperature,
                'Top_P': top_p
            })
    
    if normal_perf is not None:
        for i, n in enumerate(normal_perf['n_values']):
            best_at_n_data.append({
                'N': n,
                'Method': 'Normal',
                'Mean_Score': normal_perf['mean_scores'][i],
                'Std_Error': normal_perf['std_errors'][i],
                'Model': model_name,
                'Temperature': temperature,
                'Top_P': top_p
            })
    
    # Save Best@N scores data
    if best_at_n_data:
        best_at_n_df = pd.DataFrame(best_at_n_data)
        csv_filename = f"{model_name}_best_at_n_scores_T{temperature}_p{top_p}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        best_at_n_df.to_csv(csv_path, index=False)
        print(f"Saved Best@N scores data to: {csv_path}")
    
    # Create data for Standard Errors
    std_error_data = []
    if annealed_perf is not None:
        for i, n in enumerate(annealed_perf['n_values']):
            std_error_data.append({
                'N': n,
                'Method': 'Annealed',
                'Std_Error': annealed_perf['std_errors'][i],
                'Model': model_name,
                'Temperature': temperature,
                'Top_P': top_p
            })
    
    if normal_perf is not None:
        for i, n in enumerate(normal_perf['n_values']):
            std_error_data.append({
                'N': n,
                'Method': 'Normal',
                'Std_Error': normal_perf['std_errors'][i],
                'Model': model_name,
                'Temperature': temperature,
                'Top_P': top_p
            })
    
    # Save Standard Error data
    if std_error_data:
        std_error_df = pd.DataFrame(std_error_data)
        csv_filename = f"{model_name}_standard_errors_T{temperature}_p{top_p}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        std_error_df.to_csv(csv_path, index=False)
        print(f"Saved Standard Error data to: {csv_path}")


def print_performance_summary(annealed_perf, normal_perf, model_name):
    """
    Print a summary of performance metrics.
    
    Args:
        annealed_perf: Performance data for annealed sampling
        normal_perf: Performance data for normal sampling
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    print(f"Performance Summary for {model_name}")
    print(f"{'='*60}")
    
    if annealed_perf is not None:
        print(f"\nAnnealed Sampling Performance:")
        for i, n in enumerate(annealed_perf['n_values']):
            print(f"  N={n}: Score: {annealed_perf['mean_scores'][i]:.4f} ± {annealed_perf['std_errors'][i]:.4f}")
    
    if normal_perf is not None:
        print(f"\nNormal Sampling Performance:")
        for i, n in enumerate(normal_perf['n_values']):
            print(f"  N={n}: Score: {normal_perf['mean_scores'][i]:.4f} ± {normal_perf['std_errors'][i]:.4f}")
    
    # Calculate improvement if both are available
    if annealed_perf is not None and normal_perf is not None:
        print(f"\nImprovement Analysis:")
        for i, n in enumerate(annealed_perf['n_values']):
            if n in normal_perf['n_values']:
                normal_idx = normal_perf['n_values'].index(n)
                annealed_score = annealed_perf['mean_scores'][i]
                normal_score = normal_perf['mean_scores'][normal_idx]
                improvement = annealed_score - normal_score
                improvement_pct = (improvement / normal_score) * 100 if normal_score > 0 else 0
                print(f"  N={n}: Annealed: {annealed_score:.4f}, Normal: {normal_score:.4f}, "
                      f"Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")


def main():
    """Main function to run the visualization script."""
    args = parse_arguments()
    
    print(f"Starting visualization for output directory: {args.output_dir}")
    print(f"Looking for data: {args.data_name}")
    print(f"Prompt type: {args.prompt_type}")
    print(f"N values: {args.n_values}")
    print(f"Bootstrap iterations: {args.bootstrap_num}")
    
    # Auto-detect model name if not specified
    if args.model_name is None:
        # Try to extract model name from the output directory path
        basename = os.path.basename(args.output_dir.rstrip('/'))
        if basename:
            args.model_name = basename
        else:
            args.model_name = "Unknown_Model"
    
    print(f"Model name: {args.model_name}")
    
    # Find result files
    annealed_file, normal_file = find_result_files(
        args.output_dir, args.data_name, args.prompt_type, 
        args.sample_nums, args.seed, args.temperature, 
        args.top_p, args.n_sampling
    )
    
    if annealed_file is None and normal_file is None:
        print("Error: No result files found. Please check the output directory and parameters.")
        return
    
    # Process data
    print("\nProcessing annealed sampling results...")
    annealed_perf = process_and_analyze_data(annealed_file, args.n_values, args.bootstrap_num)
    
    print("\nProcessing normal sampling results...")
    normal_perf = process_and_analyze_data(normal_file, args.n_values, args.bootstrap_num)
    
    if annealed_perf is None and normal_perf is None:
        print("Error: No valid performance data could be extracted.")
        return
    
    # Create visualizations
    print("\nCreating Best@N scores plot...")
    plot_best_at_n_scores(
        annealed_perf, normal_perf, args.n_values, args.visualization_dir,
        args.model_name, args.temperature, args.top_p,
        args.figsize, args.fontsize, args.linewidth
    )
    
    print("\nCreating Standard Error plot...")
    plot_standard_errors(
        annealed_perf, normal_perf, args.n_values, args.visualization_dir,
        args.model_name, args.temperature, args.top_p,
        args.figsize, args.fontsize, args.linewidth
    )
    
    # Export data to CSV
    print("\nExporting data to CSV files...")
    export_data_to_csv(
        annealed_perf, normal_perf, args.visualization_dir,
        args.model_name, args.temperature, args.top_p
    )
    
    # Print performance summary
    print_performance_summary(annealed_perf, normal_perf, args.model_name)
    
    print(f"\nVisualization complete! Check {args.visualization_dir} for output files.")
    print("Generated files:")
    print(f"  - {args.model_name}_best_at_n_scores_T{args.temperature}_p{args.top_p}.pdf")
    print(f"  - {args.model_name}_standard_errors_T{args.temperature}_p{args.top_p}.pdf")
    print(f"  - {args.model_name}_best_at_n_scores_T{args.temperature}_p{args.top_p}.csv")
    print(f"  - {args.model_name}_standard_errors_T{args.temperature}_p{args.top_p}.csv")


if __name__ == "__main__":
    main()
