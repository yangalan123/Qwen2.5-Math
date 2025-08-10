import argparse
import glob
import json
import os.path
import random
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import pandas as pd

from rm_maj_eval import eval_maj_k_metrics
from uncertainty_quantification.visualization_utils import (
    DEFAULT_VISUALIZATION_DIR, DEFAULT_FIG_SIZE, DEFAULT_FONT_SIZE, DEFAULT_LINE_WIDTH)

from uncertainty_quantification.consts import root_path
from uncertainty_quantification.uncertainty_computation import DEFAULT_ASYMPTOTIC_LIMIT


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate model performance with majority voting')
    parser.add_argument('--top_p_values', type=float, nargs='+', default=[0.9, 1.0], help='Top-p sampling parameters')
    parser.add_argument('--temperature_values', type=float, nargs='+', default=[0.6, 1.0],
                        help='Temperature values for sampling')
    parser.add_argument('--sample_nums', type=int, default=200, help='Number of samples')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--n_sampling', type=int, default=64, help='Number of samplings')
    parser.add_argument('--prompt_template', type=str, default='cot', help='Prompt template type')
    parser.add_argument('--n_values', type=int, nargs='+', default=[1, 3, 8, 16, 24, 32],
                        help='N values for majority voting')
    parser.add_argument('--bootstrap_num', type=int, default=100, help='Number of bootstrap iterations')
    return parser.parse_args()

def process_results(result_file, n_values=[1, 4, 8, 16, 32], bootstrap_num=5):
    """Process results file and calculate accuracy for different N values"""
    # Initialize results dictionary
    results_by_n = {n: [] for n in n_values}
    prob_weighted_results_by_n = {n: [] for n in n_values}

    # For tracking shortest and longest outputs
    all_individual_results = []

    with open(result_file, "r", encoding='utf-8') as f:
        for line in f:
            all_sample_results = []
            data = json.loads(line)
            preds = data["pred"]
            scores = data['score']
            token_num_ids = data['token_ids_num']
            prompt_token_nums = [x[0] for x in token_num_ids]
            output_token_nums = [x[1] for x in token_num_ids]
            logliks = data['logliks'] if "logliks" in data else []
            entropies = data['entropies'] if "entropies" in data else []
            data_idx = data['idx']
            gt = data['gt']

            counter = 0
            for pred, score, output_token_num in zip(preds, scores, output_token_nums):
                result = {
                    "pred": pred,
                    "score": score,
                    "data_idx": data_idx,
                    "output_length": output_token_num,
                    "gt": gt
                }
                if len(logliks) > counter:
                    # cumulative logprob
                    result['loglik'] = logliks[counter] / output_token_num
                    assert len(entropies[counter]) == output_token_num
                    result['entropy'] = sum(entropies[counter]) / output_token_num
                    # begin_entropy = entropies[counter][:5]
                    for prefix_length in [1, 5, 25, 200]:
                        begin_entropy = entropies[counter][:prefix_length]
                        result[f'entropy@{prefix_length}'] = np.mean(begin_entropy)
                    if output_token_num >= DEFAULT_ASYMPTOTIC_LIMIT:
                        result['BF'] = -result['loglik']
                    else:
                        result['BF'] = result['entropy']

                all_sample_results.append(result)
                all_individual_results.append(result)
                counter += 1

            # Process for each N value
            for n in n_values:
                majority_scores = []
                prob_majority_scores = []
                for j in range(bootstrap_num):
                    segment_result = random.sample(all_sample_results, n)
                    # Get majority prediction
                    majority_pick = Counter([x['pred'] for x in segment_result]).most_common(1)[0][0]

                    # Find the score for the majority prediction
                    majority_score = None
                    for result in segment_result:
                        if result['pred'] == majority_pick:
                            majority_score = result['score']
                            break

                    # Store result
                    if majority_score is not None:
                        # results_by_n[n].append(majority_score)
                        majority_scores.append(majority_score)

                    pred_probs = {}
                    for result in segment_result:
                        pred = result['pred']
                        # Convert log-likelihood to probability
                        prob = np.exp(result['loglik'])

                        if pred not in pred_probs:
                            pred_probs[pred] = 0
                        pred_probs[pred] += prob

                    # Find prediction with highest probability sum
                    prob_weighted_pick = max(pred_probs.items(), key=lambda x: x[1])[0]

                    # Find the score for the probability-weighted prediction
                    prob_weighted_score = None
                    for result in segment_result:
                        if result['pred'] == prob_weighted_pick:
                            prob_weighted_score = result['score']
                            break

                    # Store probability-weighted voting result
                    if prob_weighted_score is not None:
                        # prob_weighted_results_by_n[n].append(prob_weighted_score)
                        prob_majority_scores.append(prob_weighted_score)
                results_by_n[n].append(majority_scores)
                prob_weighted_results_by_n[n].append(prob_majority_scores)

    # Return both the results_by_n and all individual results for length-based analysis
    return results_by_n, prob_weighted_results_by_n, all_individual_results


def calculate_model_performance(results_by_n):
    """Calculate mean score and standard error for each N value"""
    performance_data = {
        'n_values': [],
        'mean_scores': [],
        'std_errors': [],
        'sample_counts': []
    }

    for n, scores in sorted(results_by_n.items()):
        if scores:
            # percentage_scores = 100 * scores
            performance_data['n_values'].append(n)
            performance_data['mean_scores'].append(np.mean([x for xs in scores for x in xs]))
            # performance_data['std_errors'].append(sem(percentage_scores) if len(scores) > 1 else 0)
            performance_data['std_errors'].append(np.mean([np.std(x) for x in scores]))
            performance_data['sample_counts'].append(len(scores))

    return performance_data


def plot_score_vs_n(performance_data, model_name, visualization_dir="majority_results"):
    """Plot score vs N for the model"""
    plt.figure(figsize=DEFAULT_FIG_SIZE)
    plt.rc('font', size=DEFAULT_FONT_SIZE)

    plt.errorbar(
        performance_data['n_values'],
        performance_data['mean_scores'],
        yerr=performance_data['std_errors'],
        fmt='o-',
        linewidth=DEFAULT_LINE_WIDTH,
        capsize=5,
        label=f"Model: {model_name} (n={sum(performance_data['sample_counts'])})"
    )

    plt.xscale('log')
    plt.xlabel('N (majority voting size)')
    plt.ylabel('Score')
    plt.title(f'Best@N Score - {model_name}')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')

    # Set x-ticks to the specific N values
    plt.xticks(performance_data['n_values'], [str(n) for n in performance_data['n_values']])

    # Print statistics
    print(f"Model: {model_name} performance:")
    for i, n in enumerate(performance_data['n_values']):
        print(f"  N={n}: {performance_data['sample_counts'][i]} samples, "
              f"Score: {performance_data['mean_scores'][i]:.4f} ± {performance_data['std_errors'][i]:.4f}")

    plt.tight_layout()
    os.makedirs(visualization_dir, exist_ok=True)
    plt.savefig(f"{visualization_dir}/{model_name}_score_vs_n.pdf")

    # Also create a version with a complete y-axis (0 to 1)
    plt.ylim(0, 1)
    plt.savefig(f"{visualization_dir}/{model_name}_score_vs_n_full_scale.pdf")
    plt.close()


def plot_multi_model_comparison(args, all_models_data, temp, top_p, visualization_dir="majority_results"):
    """Create a comparison plot of multiple models"""
    plt.figure(figsize=DEFAULT_FIG_SIZE)
    plt.rc('font', size=DEFAULT_FONT_SIZE)

    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|']
    colors = plt.cm.tab10.colors

    for i, (model_name, performance_data) in enumerate(all_models_data.items()):
        if "70" in model_name:
            continue
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]

        plt.errorbar(
            performance_data['n_values'][2:],
            performance_data['mean_scores'][2:],
            yerr=performance_data['std_errors'][2:],
            fmt=f'{marker}-',
            linewidth=DEFAULT_LINE_WIDTH,
            color=color,
            capsize=5,
            label=f"{model_name}"
        )

    plt.xscale('log')
    plt.xlabel('N (majority voting size)')
    plt.ylabel('Score')
    plt.title(f'Maj@N Score Comparison (temp={temp}, top_p={top_p})')
    # plt.grid(True, alpha=0.3)
    plt.legend(loc='best')

    # Set x-ticks to the specific N values
    # all_n_values = sorted(list(set().union(*[data['n_values'] for data in all_models_data.values()])))
    all_n_values = [1, 3, 8, 16][1:]
    plt.xticks(all_n_values, [str(n) for n in all_n_values])

    plt.tight_layout()
    os.makedirs(visualization_dir, exist_ok=True)
    plt.savefig(f"{visualization_dir}/all_models_comparison_score_vs_n_t{temp}_p{top_p}.pdf")

    # Also create a version with a complete y-axis (0 to 1)
    plt.ylim(0, 1)
    plt.savefig(f"{visualization_dir}/all_models_comparison_score_vs_n_full_scale_t{temp}_p{top_p}.pdf")
    plt.close()


def plot_comparison_voting_methods(performance_data_maj, performance_data_prob, model_name,
                                   visualization_dir="majority_results"):
    """Plot comparison of majority voting vs probability-weighted voting for the model"""
    plt.figure(figsize=DEFAULT_FIG_SIZE)
    plt.rc('font', size=DEFAULT_FONT_SIZE)

    # Plot majority voting
    plt.errorbar(
        performance_data_maj['n_values'],
        performance_data_maj['mean_scores'],
        yerr=performance_data_maj['std_errors'],
        fmt='o-',
        linewidth=DEFAULT_LINE_WIDTH,
        capsize=5,
        label=f"Majority Voting"
    )

    # Plot probability-weighted voting
    plt.errorbar(
        performance_data_prob['n_values'],
        performance_data_prob['mean_scores'],
        yerr=performance_data_prob['std_errors'],
        fmt='s--',
        linewidth=DEFAULT_LINE_WIDTH,
        capsize=5,
        label=f"Probability-Weighted Voting"
    )

    plt.xscale('log')
    plt.xlabel('N (voting size)')
    plt.ylabel('Score')
    plt.title(f'Voting Methods Comparison - {model_name}')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')

    # Set x-ticks to the specific N values
    plt.xticks(performance_data_maj['n_values'], [str(n) for n in performance_data_maj['n_values']])

    # Print statistics
    # print(f"Model: {model_name} performance comparison:")
    # for i, n in enumerate(performance_data_maj['n_values']):
    #     maj_score = performance_data_maj['mean_scores'][i]
    #     maj_error = performance_data_maj['std_errors'][i]
    #     prob_score = performance_data_prob['mean_scores'][i]
    #     prob_error = performance_data_prob['std_errors'][i]
    #
    #     print(f"  N={n}: Majority: {maj_score:.4f} ± {maj_error:.4f}, "
    #           f"Prob-weighted: {prob_score:.4f} ± {prob_error:.4f}, "
    #           f"Difference: {prob_score - maj_score:.4f}")

    plt.tight_layout()
    os.makedirs(visualization_dir, exist_ok=True)
    plt.savefig(f"{visualization_dir}/{model_name}_voting_methods_comparison.pdf")

    # Also create a version with a complete y-axis (0 to 1)
    plt.ylim(0, 1)
    plt.savefig(f"{visualization_dir}/{model_name}_voting_methods_comparison_full_scale.pdf")
    plt.close()


def find_burning_period(performance_data, difference_ratio=0.90, std_errors=None):
    """Find the earliest n that unlocks 90% of maximum performance gain"""
    n_values = performance_data['n_values']
    scores = performance_data['mean_scores']
    # std_errors = performance_data['std_errors']

    # Calculate max possible gain
    min_score = scores[0]  # Assuming scores are ordered by n_values
    max_score = max(scores)
    max_gain = max_score - min_score

    # If there's no gain, return the smallest n
    if max_gain <= 0:
        return n_values[0]

    # Find the earliest n that achieves 90% of max gain
    # target_score = min_score + difference_ratio * max_gain
    # print(min_score, std_errors[0])
    # exit()
    target_score = min_score + 0.25 * std_errors[1]

    for i, score in enumerate(scores):
        if score >= target_score:
            return n_values[i]

    # If no n satisfies, return the largest n
    return n_values[-1]


def calculate_length_based_std(individual_results, n_value=1, bootstrap_num=5, top_k=-1, shortest=False):
    """
    Calculate standard deviation for acc@n

    Args:
        individual_results: List of individual prediction results
        n_value: The N value for majority voting (1, 3, 8, 16)
        bootstrap_num: Number of bootstrap iterations
        top_k: Number of top-k results to consider (-1 for all)
        shortest: Whether to sort by shortest (True) or longest (False)

    Returns:
        Average standard deviation across all data points
    """
    # Group by data_idx
    grouped_results = {}
    for result in individual_results:
        idx = result['data_idx']
        if idx not in grouped_results:
            grouped_results[idx] = []
        grouped_results[idx].append(result)

    # For each data point, calculate std of scores
    std_values = []

    for idx, results in grouped_results.items():
        # Sort by length if specified
        if top_k > 0:
            sorted_results = sorted(results, key=lambda x: x['output_length'],
                                    reverse=not shortest)
            results_to_use = sorted_results[:top_k]
        else:
            results_to_use = results

        # Skip if we don't have enough samples
        if len(results_to_use) < n_value:
            continue

        # Calculate acc@n with bootstrapping
        bootstrap_scores = []
        for _ in range(bootstrap_num):
            # Randomly sample n_value results
            if len(results_to_use) == n_value:
                # If we have exactly n_value results, use all of them
                segment_result = results_to_use
            else:
                # Otherwise, randomly sample n_value results
                segment_result = random.sample(results_to_use, n_value)

            # Get majority prediction
            majority_pick = Counter([x['pred'] for x in segment_result]).most_common(1)[0][0]

            # Find the score for the majority prediction
            majority_score = None
            for result in segment_result:
                if result['pred'] == majority_pick:
                    majority_score = result['score']
                    break

            # Store result
            if majority_score is not None:
                bootstrap_scores.append(majority_score)

        # Calculate std for this data point if we have enough bootstrap samples
        if len(bootstrap_scores) >= 2:
            std_values.append(np.std(bootstrap_scores))

    # Return average std across all data points
    return np.mean(std_values) if std_values else 0


def generate_latex_table(model_data, temp, top_p, output_dir, bootstrap_num=5):
    """Generate LaTeX table with the requested metrics for acc@1, acc@3, acc@8, acc@16"""
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for table
    table_data = []

    # N values to compute std for
    n_values_for_std = [1, 3, 8, 16]

    for model_name, data in model_data.items():
        performance_data = data['performance']
        individual_results = data['individual_results']

        # Find burning period

        # Calculate std for different N values
        std_values = {}
        for n in n_values_for_std:
            std_values[n] = calculate_length_based_std(
                individual_results,
                n_value=n,
                bootstrap_num=bootstrap_num,
                top_k=-1,
                shortest=False
            )
        burning_period = find_burning_period(performance_data, std_errors=std_values)

        # Add to table data
        model_row = {
            'Model': model_name,
            # 'Compute-Optimal (n)': burning_period
        }

        # Add std values for different N
        for n in n_values_for_std:
            model_row[f'Acc@{n} Std'] = std_values[n]
        if "loglik" in individual_results[0]:
            model_row["loglik"] = np.exp(-np.mean([x['loglik'] for x in individual_results if x is not None]))
        if "entropy" in individual_results[0]:
            model_row["ppl"] = np.exp(np.mean([x['entropy'] for x in individual_results if x is not None]))
        # if "entropy_begin" in individual_results[0]:
        #     model_row["ppl_begin"] = np.exp(np.mean([x['entropy_begin'] for x in individual_results if x is not None]))
        all_keys = list(individual_results[0].keys())
        for _key in all_keys:
            if "entropy@" in _key or "BF" in _key:
                model_row[_key] = np.exp(np.mean([x[_key] for x in individual_results if x is not None]))
        if "output_length" in individual_results[0]:
            model_row['Output Length (mean)'] = np.mean([x['output_length'] for x in individual_results if x is not None])
            model_row['Output Length (std)'] = np.std([x['output_length'] for x in individual_results if x is not None])

        table_data.append(model_row)

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Generate LaTeX table
    latex_table = df.to_latex(index=False, float_format="%.4f", escape=False)

    # Add table environment, caption and label
    latex_table = f"\\begin{{table}}\n\\caption{{Model performance metrics (temp={temp}, top\\_p={top_p}, bootstrap={bootstrap_num})}}\n" + latex_table
    latex_table = latex_table.replace('\\end{tabular}',
                                      f'\\end{{tabular}}\n\\label{{tab:model_metrics_t{temp}_p{top_p}}}\n\\end{{table}}')

    # Write to file
    with open(f"{output_dir}/model_metrics_t{temp}_p{top_p}.tex", 'w') as f:
        f.write(latex_table)

    return latex_table


def plot_comparison_lines(perf_standard, perf_annealed, n_values, model_name, temp, top_p, output_dir, perf_standard_t06=None, nvals_t06=None):
    import os
    # Mean scores
    plt.figure()
    plt.plot(n_values, perf_standard['mean_scores'], marker='o', label='Standard', color='blue')
    plt.plot(n_values, perf_annealed['mean_scores'], marker='o', label='Annealed', color='orange')
    if perf_standard_t06 is not None and nvals_t06 is not None:
        plt.plot(nvals_t06, perf_standard_t06['mean_scores'], marker='o', label='Standard (T=0.6)', color='green')
    plt.xlabel('N (majority voting size)')
    plt.ylabel('Mean Score')
    plt.title(f'{model_name} Mean Score vs N (T={temp}, p={top_p})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{model_name}_mean_scores_T{temp}_p{top_p}.pdf'))
    plt.close()
    # Std errors
    plt.figure()
    plt.plot(n_values, perf_standard['std_errors'], marker='o', label='Standard', color='blue')
    plt.plot(n_values, perf_annealed['std_errors'], marker='o', label='Annealed', color='orange')
    if perf_standard_t06 is not None and nvals_t06 is not None:
        plt.plot(nvals_t06, perf_standard_t06['std_errors'], marker='o', label='Standard (T=0.6)', color='green')
    plt.xlabel('N (majority voting size)')
    plt.ylabel('Std Error')
    plt.title(f'{model_name} Std Error vs N (T={temp}, p={top_p})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_std_errors_T{temp}_p{top_p}.pdf'))
    plt.close()


def plot_comparison_lines_multi_annealed(perf_standard, perf_annealed_dict, n_values, model_name, temp, top_p, output_dir, perf_standard_t06=None, nvals_t06=None):
    import os
    # Use a large color palette for annealed curves
    palette = plt.cm.get_cmap('tab20', max(3, len(perf_annealed_dict)))
    # Mean scores
    plt.figure()
    plt.plot(n_values, perf_standard['mean_scores'], marker='o', label='Standard', color='blue')
    for idx, (annealed_label, perf_annealed) in enumerate(perf_annealed_dict.items()):
        plt.plot(n_values, perf_annealed['mean_scores'], marker='o', label=annealed_label, color=palette(idx))
    if perf_standard_t06 is not None and nvals_t06 is not None:
        plt.plot(nvals_t06, perf_standard_t06['mean_scores'], marker='o', label='Standard (T=0.6)', color='green')
    plt.xlabel('N (majority voting size)')
    plt.ylabel('Mean Score')
    plt.title(f'{model_name} Mean Score vs N (T={temp}, p={top_p})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{model_name}_mean_scores_T{temp}_p{top_p}_multi_annealed.pdf'))
    plt.close()
    # Std errors
    plt.figure()
    plt.plot(n_values, perf_standard['std_errors'], marker='o', label='Standard', color='blue')
    for idx, (annealed_label, perf_annealed) in enumerate(perf_annealed_dict.items()):
        plt.plot(n_values, perf_annealed['std_errors'], marker='o', label=annealed_label, color=palette(idx))
    if perf_standard_t06 is not None and nvals_t06 is not None:
        plt.plot(nvals_t06, perf_standard_t06['std_errors'], marker='o', label='Standard (T=0.6)', color='green')
    plt.xlabel('N (majority voting size)')
    plt.ylabel('Std Error')
    plt.title(f'{model_name} Std Error vs N (T={temp}, p={top_p})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_std_errors_T{temp}_p{top_p}_multi_annealed.pdf'))
    plt.close()


def main():
    args = parse_arguments()
    temperature_values = args.temperature_values
    top_p_values = args.top_p_values
    n_values = range(1, 33, 2)
    bootstrap_num = args.bootstrap_num
    config_pairs = [
        {
            'config_name': 'standard',
            'latex_tables_dir': './latex_tables_majority',
            'results_root_dir_template': f"{root_path}/sampling_effects/Qwen2.5-Math/evaluation/outputs/math_eval_Seed{{seed}}_N{{n_sampling}}_MAXTOKENS{{max_tokens}}_TOPP{{top_p}}_logits"
        },
    ]
    # Collect results for standard config
    results_by_key = {}  # key: (model_name, temp, top_p), value: { 'standard': perf }
    for config in config_pairs:
        config_name = config['config_name']
        latex_tables_dir = config['latex_tables_dir']
        results_root_dir_template = config['results_root_dir_template']
        os.makedirs(latex_tables_dir, exist_ok=True)
        for temp in temperature_values:
            for top_p in top_p_values:
                max_tokens_values = [2048, 32768]
                for max_tokens in max_tokens_values:
                    results_root_dir = results_root_dir_template.format(
                        seed=args.seed,
                        n_sampling=args.n_sampling,
                        max_tokens=max_tokens,
                        top_p=top_p
                    )
                    model_dirs = glob.glob(f"{results_root_dir}/*")
                    for model_dir in model_dirs:
                        model_name = os.path.basename(model_dir)
                        if "llama" not in model_name.lower():
                            continue
                        results_file_pattern = f"test_{args.prompt_template}_{args.sample_nums}_seed{args.seed}_t{temp}_5shot_s0_e-1.jsonl"
                        result_files = glob.glob(f"{model_dir}/mmlu_stem/{results_file_pattern}")
                        if not result_files:
                            continue
                        result_file = result_files[0]
                        results_by_n, prob_weighted_results_by_n, individual_results = process_results(result_file, n_values, bootstrap_num)
                        performance_data = calculate_model_performance(results_by_n)
                        key = (model_name, temp, top_p)
                        if key not in results_by_key:
                            results_by_key[key] = {}
                        results_by_key[key][config_name] = performance_data
    # Now collect all annealed configs
    # Find all annealed result dirs matching the pattern
    import re
    annealed_root = f"{root_path}/sampling_effects/Qwen2.5-Math/evaluation/outputs/"
    annealed_dirs = glob.glob(os.path.join(annealed_root, "math_eval_Seed*_logits_annealed*"))
    # For each annealed dir, extract params for labeling
    annealed_info = []  # List of (dir, label, params_dict)
    annealed_pattern = re.compile(r"math_eval_Seed(?P<seed>\d+)_N(?P<n_sampling>\d+)_MAXTOKENS(?P<max_tokens>\d+)_TOPP(?P<top_p>[\d\.]+)_logits_annealed(_s(?P<start_temp>[\d\.]+)_t(?P<end_temp>[\d\.]+)_d(?P<decay_rate>[\d\.]+))?")
    for d in annealed_dirs:
        m = annealed_pattern.search(os.path.basename(d))
        if m:
            params = m.groupdict()
            # Build a label string from params
            label = f"Annealed"
            if params.get("start_temp"):
                label += f" s={params['start_temp']} t={params['end_temp']} d={params['decay_rate']}"
            else:
                label += " (default)"
            label += f" (Seed={params['seed']} N={params['n_sampling']} MaxTok={params['max_tokens']} TopP={params['top_p']})"
            annealed_info.append((d, label, params))
    # Now plot comparisons for each (model, temp, top_p) that has standard config
    output_dir = './majority_results_linecharts'
    os.makedirs(output_dir, exist_ok=True)
    for key, perf_dict in results_by_key.items():
        model_name, temp, top_p = key
        perf_standard = perf_dict['standard']
        # For this model/temp/top_p, find all matching annealed dirs
        perf_annealed_dict = {}
        nvals = perf_standard['n_values']
        for d, label, params in annealed_info:
            # Only match if temp/top_p match
            if float(params['top_p']) != float(top_p):
                continue
            # Find model dir inside this annealed dir
            model_dir = os.path.join(d, model_name)
            if not os.path.isdir(model_dir):
                continue
            results_file_pattern = f"test_{args.prompt_template}_{args.sample_nums}_seed{params['seed']}_t{temp}_5shot_s0_e-1.jsonl"
            result_files = glob.glob(f"{model_dir}/mmlu_stem/{results_file_pattern}")
            if not result_files:
                continue
            result_file = result_files[0]
            results_by_n, prob_weighted_results_by_n, individual_results = process_results(result_file, nvals, bootstrap_num)
            performance_data = calculate_model_performance(results_by_n)
            perf_annealed_dict[label] = performance_data
        # Try to get standard (T=0.6) for same model_name and top_p
        key_t06 = (model_name, 0.6, top_p)
        perf_standard_t06 = None
        nvals_t06 = None
        if key_t06 in results_by_key and 'standard' in results_by_key[key_t06]:
            perf_t06 = results_by_key[key_t06]['standard']
            nvals_t06 = sorted(list(set(perf_t06['n_values']) & set(nvals)))
            def align(perf, nvals):
                idxs = [perf['n_values'].index(n) for n in nvals]
                return {
                    'mean_scores': [perf['mean_scores'][i] for i in idxs],
                    'std_errors': [perf['std_errors'][i] for i in idxs]
                }
            perf_standard_t06 = align(perf_t06, nvals_t06)
        if perf_annealed_dict:
            plot_comparison_lines_multi_annealed(perf_standard, perf_annealed_dict, nvals, model_name, temp, top_p, output_dir, perf_standard_t06, nvals_t06)
        # else: fallback to old plot if only one annealed config (optional)


if __name__ == '__main__':
    main()
