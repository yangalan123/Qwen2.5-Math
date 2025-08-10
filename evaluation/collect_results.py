import os
import json
import glob
import numpy as np
import re
from collections import defaultdict
import torch
import scipy.stats as st


def collect_data(base_dir='outputs'):
    """
    Collect accuracy data from JSON files matching the pattern:
    outputs/math_eval_Seed{seed}_N1_MAXTOKENS{max_tokens}_TOPP{top_p}/{model_name}/{dataset_name}/test_{prompt_type}_seed{seed}_t{temperature}_{shot_num}shot_s0_e-1_{prompt_type}_metrics.json
    """
    # Create a data structure to store results
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    # Regular expression to extract information from file paths
    pattern = re.compile(
        r'outputs/math_eval_Seed(\d+)_N64_MAXTOKENS(\d+)_TOPP([\d\.]+)[^/]*/([^/]+)/([^/]+)/test_([^_]+)_200_seed\d+_t([\d\.]+)_(\d+)shot_s0_e-1_[^_]+_metrics\.json'
    )
    # pattern = re.compile(
    #     r'outputs/math_eval_Seed(\d+)_N5_MAXTOKENS(\d+)_TOPP([\d\.]+)/([^/]+)/([^/]+)/test_([^_]+)_-1_seed\d+_t([\d\.]+)_(\d+)shot_s0_e-1_[^_]+_metrics\.json'
    # )
    #pattern = re.compile(
        #r'outputs/math_eval_Seed(\d+)_N1_MAXTOKENS(\d+)_TOPP([\d\.]+)/([^/]+)/([^/]+)/test_([^_]+)_-1_seed\d+_t([\d\.]+)_(\d+)shot_s0_e-1_[^_]+_metrics\.json'
    #)
    ## neurips last minute
    pattern = re.compile(
        r'outputs/math_eval_Seed(\d+)_N64_MAXTOKENS(\d+)_TOPP([\d\.]+)[^/]*/([^/]+)/([^/]+)/test_([^_]+)_200_seed\d+_t([\d\.]+)_(\d+)shot_s0_e-1\.jsonl'
    )

    # Find all JSON files that match the pattern
    json_files = glob.glob(os.path.join(base_dir, "**/*.jsonl"), recursive=True)

    # Track max_tokens per model
    max_tokens_by_model = {}
    # Track shot_num per (model, dataset, prompt_type)
    shot_num_by_key = {}

    for file_path in json_files:
        match = pattern.match(file_path)
        if not match:
            if "deprecated" not in file_path and "N64" in file_path:
                print(f"{file_path} not matched")
            continue

        seed, max_tokens, top_p, model_name, dataset_name, prompt_type, temperature, shot_num = match.groups()
        # if "deepseek" in model_name.lower() and int(shot_num) == 5:
        #     continue

        # Convert numeric values
        seed = int(seed)
        max_tokens = int(max_tokens)
        top_p = float(top_p)
        temperature = float(temperature)
        shot_num = int(shot_num)

        # Store max_tokens by model
        if model_name not in max_tokens_by_model:
            max_tokens_by_model[model_name] = max_tokens

        # Store shot_num by (model, dataset, prompt_type)
        key = (model_name, dataset_name, prompt_type)
        if key not in shot_num_by_key:
            shot_num_by_key[key] = shot_num
        else:
            if shot_num != shot_num_by_key[key]:
                print(f" key for shot_num! {shot_num_by_key[key]} ({key} {shot_num})")

        # Try to read the JSON file
        try:
            accs = []
            stds = []
            with open(file_path, "r", encoding='utf-8') as f:
                for line in f:
                    all_sample_results = []
                    data = json.loads(line)
                    preds = data["pred"]
                    scores = data['score']
                    # token_num_ids = data['token_ids_num']
                    # prompt_token_nums = [x[0] for x in token_num_ids]
                    # output_token_nums = [x[1] for x in token_num_ids]
                    # logliks = data['logliks'] if "logliks" in data else []
                    # entropies = data['entropies'] if "entropies" in data else []
                    data_idx = data['idx']
                    gt = data['gt']
                    _acc = np.mean(scores)
                    _std = st.sem(scores)
                    accs.append(_acc)
                    stds.append(_std)
                results[prompt_type][dataset_name][model_name][(temperature, top_p)].append({
                    "acc": np.mean(accs),
                    "std": np.mean(stds),
                })
            # with open(file_path, 'r') as f:
            #     try:
            #         data = json.load(f)
            #     except:
            #         print(f"{file_path} perhaps not saved as a real json, try torch load")
            #         data = torch.load(f)
            #
            # # Get the accuracy value
            # if 'acc' in data:
            #     acc = data['acc']
            #     # Store the result
            #     results[prompt_type][dataset_name][model_name][(temperature, top_p)].append(acc)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return results, max_tokens_by_model, shot_num_by_key


def generate_latex_tables(results, max_tokens_by_model, shot_num_by_key, output_dir='latex_tables_N64'):
    """
    Generate LaTeX tables from the collected data.
    """
    os.makedirs(output_dir, exist_ok=True)

    for prompt_type in results:
        for dataset_name in results[prompt_type]:
            # Get models and unique (temperature, top_p) settings for this dataset
            models = sorted(list(set(model for model in results[prompt_type][dataset_name].keys())))
            temp_top_p_values = sorted(list(set(
                tp for model in results[prompt_type][dataset_name]
                for tp in results[prompt_type][dataset_name][model].keys()
            )))

            # If no data, skip
            if not models or not temp_top_p_values:
                continue

            # Create the LaTeX table
            output_file = os.path.join(output_dir, f"{prompt_type}_{dataset_name}.tex")

            with open(output_file, 'w') as f:
                # Begin table
                f.write("\\begin{table}[htbp]\n")
                f.write("\\centering\n")
                f.write("\\resizebox{\\textwidth}{!}{\n")

                # Table header
                f.write("\\begin{tabular}{l" + "c" * len(temp_top_p_values) + "}\n")
                f.write("\\toprule\n")

                # Header row
                header = ["Models"]
                for t, p in temp_top_p_values:
                    header.append(f"T={t}, P={p}")

                f.write(" & ".join(header) + " \\\\\n")
                f.write("\\midrule\n")

                # Data rows
                for model in models:
                    row = [model]
                    for t, p in temp_top_p_values:
                        if (t, p) in results[prompt_type][dataset_name][model]:
                            accs = results[prompt_type][dataset_name][model][(t, p)]
                            if accs:
                                # Calculate mean and std if multiple seeds are available
                                # if len(accs) > 1:
                                #     mean = np.mean(accs)
                                #     std = np.std(accs)
                                #     row.append(f"{mean:.2f} ({std:.2f})")
                                # else:
                                #     # Just show the value if only one seed is available
                                #     row.append(f"{accs[0]:.2f}")
                                # row.append(f"{max(accs):.2f}")
                                row.append(f"{100*accs[0]['acc']:.2f} ($\pm$ {100*accs[0]['std']:.2f})")
                            else:
                                row.append("\\text{Incomplete}")
                        else:
                            row.append("\\text{Incomplete}")

                    f.write(" & ".join(row) + " \\\\\n")

                # End table
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("}\n")

                # Caption with max_tokens and shot_num information
                f.write("\\caption{")
                f.write(f"Results for dataset: {dataset_name}, prompt type: {prompt_type}. ")

                # Add shot_num information
                f.write("Shot numbers used: ")
                model_shot_info = []
                for model in models:
                    key = (model, dataset_name, prompt_type)
                    if key in shot_num_by_key:
                        model_shot_info.append(f"{model}: {shot_num_by_key[key]}")
                f.write(", ".join(model_shot_info))

                # Add max_tokens information
                f.write(". Max tokens: ")
                model_token_info = []
                for model in models:
                    if model in max_tokens_by_model:
                        model_token_info.append(f"{model}: {max_tokens_by_model[model]}")
                f.write(", ".join(model_token_info))

                f.write("}\n")
                f.write("\\label{tab:" + f"{prompt_type}_{dataset_name}" + "}\n")
                f.write("\\end{table}\n")

            print(f"Generated table: {output_file}")


def main():
    # Collect data
    print("Collecting data...")
    results, max_tokens_by_model, shot_num_by_key = collect_data()

    # Generate LaTeX tables
    print("Generating LaTeX tables...")
    generate_latex_tables(results, max_tokens_by_model, shot_num_by_key)

    print("Done!")


if __name__ == "__main__":
    main()
