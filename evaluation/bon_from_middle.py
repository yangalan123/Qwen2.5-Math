from uncertainty_quantification.consts import ALL_MODELS, root_path
import glob
import os
import argparse
import json
from math_eval import setup_llm, create_stop_words, run_execute, is_multi_choice, run_llm_via_vllm
from transformers import AutoTokenizer
import random
from python_executor import PythonExecutor
from vllm import SamplingParams
import torch
from uncertainty_quantification.loglik_computation import get_tokenwise_entropy_from_vllm_outputs
from parser import choice_answer_clean
from evaluate import evaluate

def process_results(result_file):
    """Process results file"""

    # For tracking shortest and longest outputs
    all_individual_results = []
    idx2prompts = dict()

    with open(result_file, "r", encoding='utf-8') as f:
        for line in f:
            # all_sample_results = []
            data = json.loads(line)
            preds = data["pred"]
            scores = data['score']
            codes = data['code']
            token_num_ids = data['token_ids_num']
            prompt_token_nums = [x[0] for x in token_num_ids]
            output_token_nums = [x[1] for x in token_num_ids]
            logliks = data['logliks']
            data_idx = data['idx']
            gt = data['gt']
            assert data_idx not in idx2prompts, f"IDX-{data_idx} already exists!"
            idx2prompts[data_idx] = data['prompt']

            for code, pred, score, output_token_num in zip(codes, preds, scores, output_token_nums):
                result = {
                    "code": code,
                    "pred": pred,
                    "score": score,
                    "data_idx": data_idx,
                    "output_length": output_token_num,
                    "prompt_token_nums": prompt_token_nums,
                    "logliks": logliks,
                    "gt": gt
                }
                # all_sample_results.append(result)
                all_individual_results.append(result)

    return all_individual_results, idx2prompts

# def run_llm_via_vllm(args, llm, prompts, stop_words=None, checkpoint_filename=None):
#     if stop_words is None:
#         stop_words = create_stop_words(args)
#     logliks = []
#     token_ids_num = []
#     output_texts = []
#     entropies = []
#     start = 0
#     sampling_param = SamplingParams(
#         temperature=args.temperature,
#         top_p=args.top_p,
#         max_tokens=args.max_tokens_per_call,
#         logprobs=50,
#         n=args.n_sampling,
#         stop=stop_words,
#         stop_token_ids=(
#             [151645, 151643]
#             if "qwen2" in args.model_name_or_path.lower()
#             else None
#         ),
#     )
#     if os.path.exists(checkpoint_filename):
#         logliks, token_ids_num, output_texts, entropies = torch.load(checkpoint_filename)
#         start = len(output_texts)
#     if args.ckpt_freq > 0:
#         for j in range(start, len(prompts), args.ckpt_freq):
#             outputs = llm.generate(prompts[j: j + args.ckpt_freq], sampling_param)
#             inc_logliks, inc_token_ids_num, inc_output_texts, inc_entropies = postprocessing_vllm_outputs(outputs)
#             logliks.extend(inc_logliks)
#             token_ids_num.extend(inc_token_ids_num)
#             output_texts.extend(inc_output_texts)
#             entropies.extend(inc_entropies)
#             torch.save([logliks, token_ids_num, output_texts, entropies], checkpoint_filename)
#             print(f"Saved Checkpoints at {j} / {len(prompts)} to {checkpoint_filename}")
#     else:
#         outputs = llm.generate(
#             prompts[start:], sampling_param
#         )
#         inc_logliks, inc_token_ids_num, inc_output_texts, inc_entropies = postprocessing_vllm_outputs(outputs)
#         logliks.extend(inc_logliks)
#         token_ids_num.extend(inc_token_ids_num)
#         output_texts.extend(inc_output_texts)
#         entropies.extend(inc_entropies)
#     torch.save([logliks, token_ids_num, output_texts, entropies], checkpoint_filename)
#     print(f"LLM Generation Complete, Save Intermediate Results at {checkpoint_filename}")
#     outputs = output_texts
#     return logliks, token_ids_num, output_texts, entropies, outputs
#
# def postprocessing_vllm_outputs(vllm_outputs):
#     outputs = sorted(
#         vllm_outputs, key=lambda x: int(x.request_id)
#     )  # sort outputs by request_id
#     # all_vllm_outputs = [output.outputs[0] for output in outputs]
#     all_vllm_outputs = []
#     token_ids_num = []
#     logliks = []
#     output_texts = []
#     for output_per_instance in outputs:
#         prompt_token_nums = len(output_per_instance.prompt_token_ids)
#         for _output in output_per_instance.outputs:
#             all_vllm_outputs.append(_output)
#             token_ids_num.append((prompt_token_nums, len(_output.token_ids)))
#             logliks.append(_output.cumulative_logprob)
#             output_texts.append(_output.text)
#
#     # logliks = [output.outputs[0].cumulative_logprob for output in outputs]
#     # token_ids_num = [(len(output.prompt_token_ids), len(output.outputs[0].token_ids)) for output in outputs]
#     # output_texts = [output.outputs[0].text for output in outputs]
#     entropies_w_token_ids = get_tokenwise_entropy_from_vllm_outputs(all_vllm_outputs, p=args.top_p, top_p_mode=True)
#     entropies = [x[0] for x in entropies_w_token_ids]
#     return logliks, token_ids_num, output_texts, entropies

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate model performance with majority voting')
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling parameters')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Temperature values for sampling')
    # parser.add_argument('--top_p_values', type=float, nargs='+', default=[0.9, 1.0], help='Top-p sampling parameters')
    # parser.add_argument('--temperature_values', type=float, nargs='+', default=[0.6, 1.0],
    #                     help='Temperature values for sampling')
    parser.add_argument('--sample_nums', type=int, default=200, help='Number of samples')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Number of samples')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--n_sampling', type=int, default=64, help='Number of samplings')
    parser.add_argument('--prompt_type', type=str, default='cot', help='Prompt template type')
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    # parser.add_argument('--n_values', type=int, nargs='+', default=[1, 3, 8, 16, 24, 32],
    #                     help='N values for majority voting')
    # parser.add_argument('--bootstrap_num', type=int, default=5, help='Number of bootstrap iterations')
    parser.add_argument('--prefix_token_num', type=int, default=20, help='Number of prefix tokens generated')
    parser.add_argument('--prefix_num_per_task', type=int, default=3, help='Number of prefix tokens generated')
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--ckpt_freq", type=int, default=-1, help="Checkpoint frequency for vllm outputs.")
    parser.add_argument("--data_name", default="mmlu_stem", type=str)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.5, help='(for vllm) gpu memory utilization.')
    args = parser.parse_args()
    args.max_tokens_per_call = args.max_tokens
    args.use_vllm = True
    args.apply_chat_template = False
    return args

if __name__ == '__main__':
    args = parse_arguments()
    random.seed(args.seed)
    assert len(set([os.path.basename(x) for x in ALL_MODELS])) == len(ALL_MODELS), "duplicated model basename found!"
    # base_model_name_mapping = {}
    # for model_full_name in ALL_MODELS:
    #     base_model_name_mapping[os.path.basename(model_full_name)] = model_full_name
    results_root_dir = f"{root_path}/sampling_effects/Qwen2.5-Math/evaluation/outputs/math_eval_Seed{args.seed}_N{args.n_sampling}_MAXTOKENS{args.max_tokens}_TOPP{args.top_p}"
    print(f"Processing results from: {results_root_dir}")
    model_dirs = glob.glob(f"{results_root_dir}/*")

    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)
    print("Created executor")

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        if os.path.basename(args.model_name_or_path).lower() != model_name.lower():
            print(f"Ignore {model_name}")
            continue
        # if "llama" not in model_name.lower():
        #     continue
        print(f"Processing model: {model_name}")

        # Find results file
        results_file_pattern = f"test_{args.prompt_type}_{args.sample_nums}_seed{args.seed}_t{args.temperature}_5shot_s0_e-1.jsonl"
        result_files = glob.glob(f"{model_dir}/{args.data_name}/{results_file_pattern}")

        if not result_files:
            print(f"No matching results file found for {model_name}")
            continue

        if len(result_files) > 1:
            print(f"Warning: Multiple result files found for {model_name}, using the first one")

        result_file = result_files[0]
        output_filename = result_file.replace("jsonl", f"bon_from_middle_{args.prefix_token_num}_pnumt{args.prefix_num_per_task}.out")
        output_metric_filename = result_file.replace("jsonl", f"bon_from_middle_{args.prefix_token_num}_pnumt{args.prefix_num_per_task}.out.metric")
        checkpoint_filename = result_file.replace("jsonl", f"bon_from_middle_{args.prefix_token_num}_pnumt{args.prefix_num_per_task}.ckpt")
        if os.path.exists(output_filename):
            print(f"{output_filename} exists, skipping...")
            continue
        print(f"Using results file: {result_file}")
        all_individual_results, idx2prompt = process_results(result_file)
        llm, _ = setup_llm(args)
        # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer = llm.get_tokenizer()
        all_tasks = dict()
        for result in all_individual_results:
            pred = result["pred"]
            score = result['score']
            data_idx = result['data_idx']
            code = result['code']
            code_token_ids = tokenizer(code)['input_ids']
            decoded_prefix = tokenizer.decode(code_token_ids[:args.prefix_token_num])
            key = (data_idx, pred)
            if key not in all_tasks:
                all_tasks[key] = []
            result['recombined_prompt'] = idx2prompt[data_idx] + decoded_prefix
            all_tasks[key].append(result)
        prompts = []
        linearized_all_tasks= []
        for key in all_tasks:
            if len(all_tasks[key]) >= args.prefix_num_per_task:
                all_tasks[key] = random.sample(all_tasks[key], args.prefix_num_per_task)
            for item in all_tasks[key]:
                prompts.append(item['recombined_prompt'])
                linearized_all_tasks.append(item)
        stop_words = create_stop_words(args)
        prompts = [x for x in prompts for _ in range(args.n_sampling)]
        logliks, token_ids_num, _, entropies, outputs = run_llm_via_vllm(args, llm, prompts, stop_words,
                                                                                    checkpoint_filename)
        codes = []
        for code_i in range(len(outputs)):
            code = outputs[code_i].strip()
            for stop_word in stop_words:
                if stop_word in code:
                    code = code.split(stop_word)[0].strip()
            codes.append(code)

        results = [
            run_execute(executor, code, args.prompt_type, args.data_name) for code in codes
        ]
        preds = [item[0] for item in results]
        reports = [item[1] for item in results]
        all_samples = []
        for i, sample in enumerate(linearized_all_tasks):
            code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
            result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
            preds = [item[0] for item in result]
            reports = [item[1] for item in result]
            loglik_piece = logliks[i * args.n_sampling : (i + 1) * args.n_sampling]
            token_ids_num_piece = token_ids_num[i * args.n_sampling : (i + 1) * args.n_sampling]
            entropies_piece = entropies[i * args.n_sampling: (i + 1) * args.n_sampling]
            for j in range(len(preds)):
                if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                    "A",
                    "B",
                    "C",
                    "D",
                    "E",
                ]:
                    preds[j] = choice_answer_clean(code[j])
                elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                    # remove any non-choice char
                    preds[j] = "".join(
                        [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                    )

            # sample.pop("prompt")
            sample.update({"bon_middle_code": code, "bon_middle_pred": preds, "bon_middle_report": reports, "bon_middle_logliks": loglik_piece, "bon_middle_token_ids_num": token_ids_num_piece, "bon_middle_entropies": entropies_piece})
            all_samples.append(sample)

        # add processed samples
        # all_samples.extend(processed_samples)
        all_samples, result_json = evaluate(
            samples=all_samples,
            data_name=args.data_name,
            prompt_type=args.prompt_type,
            execute=True,
        )

        # save outputs
        # save_jsonl(all_samples, output_filename)
        torch.save(all_samples, output_filename)

        # result_json["time_use_in_second"] = time_use
        # result_json["time_use_in_minute"] = (
        #     f"{int(time_use // 60)}:{int(time_use % 60):02d}"
        # )

        with open(
                output_metric_filename, "w"
        ) as f:
            json.dump(result_json, f, indent=4)
