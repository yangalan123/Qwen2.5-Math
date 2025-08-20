import random
import pickle
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions
import numpy as np
# from uncertainty_quantification.loglik_computation import get_tokenwise_entropy_from_vllm_outputs
tolerance_inf = 1e-12
def get_logprob_per_token_from_vllm_outputs(vllm_output_token):
    # vllm updated implementation, for backward compatibility
    if isinstance(vllm_output_token, float):
        return vllm_output_token
    else:
        return vllm_output_token.logprob

def get_truncated_dist_top_p(sorted_dist, top_p, tolerance_inf=tolerance_inf):
    # 2025-01-10: this is a new implementation take numerical instability into consideration
    # esp. for entropy contribution, when a logprob is sufficiently small, we can ignore it
    _cumulative_probs = 0.0
    truncated_dist = dict()
    for key, value in sorted_dist:
        _exp_logprob = np.exp(value)
        if np.isinf(value) or _exp_logprob < tolerance_inf:
            # too small to contribute meaningfully, so we can save some consideration for numerical stability
            continue
        _cumulative_probs += _exp_logprob
        truncated_dist[key] = value
        if _cumulative_probs >= top_p:
            break
    return truncated_dist
def get_token_truncated_dist_from_vllm_outputs(unnormalized_dist, token_ids, length_i, p, top_p_mode=False):
    if not top_p_mode:
        # min_p mode
        truncated_dist = dict()
        for key in unnormalized_dist:
            # if isinstance(unnormalized_dist[key], float):
            #     _value = unnormalized_dist[key]
            # else:
            #     _value = unnormalized_dist[key].logprob
            _value = get_logprob_per_token_from_vllm_outputs(unnormalized_dist[key])
            if np.exp(_value) >= p or key == token_ids[length_i]:
                # "or" part: allow minor error
                # delay all normalization in the end to avoid numerical instability
                # normalized_dist[key] = unnormalized_dist[key]
                truncated_dist[key] = _value
    else:
        _unnormalized_dist = {key: get_logprob_per_token_from_vllm_outputs(unnormalized_dist[key]) for key in
                              unnormalized_dist}
        _sorted_dist = sorted(_unnormalized_dist.items(), key=lambda x: x[1], reverse=True)
        # for key, value in _sorted_dist:
        #     _cumulative_probs += np.exp(value)
        #     normalized_dist[key] = value
        #     if _cumulative_probs >= p:
        #         break
        truncated_dist = get_truncated_dist_top_p(_sorted_dist, p)
        # "or" part: allow minor error
        # delay all normalization in the end to avoid numerical instability
        if token_ids[length_i] not in truncated_dist:
            truncated_dist[token_ids[length_i]] = get_logprob_per_token_from_vllm_outputs(
                unnormalized_dist[token_ids[length_i]])
    keys = list(truncated_dist.keys())
    assert token_ids[length_i] in keys, "Token not in the distribution: length-i:{}, token: {}, prob: {}".format(
        length_i, token_ids[length_i], np.exp(unnormalized_dist[token_ids[length_i]].logprob))
    values = [truncated_dist[key] for key in keys]
    normalized_dist_values = torch.softmax(torch.tensor(values), dim=0)
    return keys, normalized_dist_values

def get_tokenwise_entropy_from_vllm_outputs(outputs, p, max_length=None, top_p_mode=False):
    buf = []
    for output in outputs:
        _per_output_buf = []
        gen_seq_len = len(output.logprobs)
        token_ids = output.token_ids
        _max_length = min(gen_seq_len, max_length) if max_length is not None else gen_seq_len
        for length_i in range(_max_length):
            unnormalized_dist = output.logprobs[length_i]
            if unnormalized_dist is None:
                break
            keys, normalized_dist_values = get_token_truncated_dist_from_vllm_outputs(unnormalized_dist, token_ids,
                                                                                      length_i, p, top_p_mode)
            entropy = -torch.sum(normalized_dist_values * torch.log(normalized_dist_values))
            _per_output_buf.append(entropy.item())
        buf.append([_per_output_buf, token_ids])
    return buf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--use_annealed_sampling", action="store_true",)
    parser.add_argument("--annealed_sampling_decay_mode", type=str, default="negexp", choices=["global_step", "token_length", "both", "both_v_1_5", "negexp", "steps_variant", "steps_variant_rev", "none"])
    parser.add_argument("--annealed_sampling_exploration_temp", type=float, default=1.2)
    parser.add_argument("--annealed_sampling_stability_temp", type=float, default=0.1)
    parser.add_argument("--annealed_sampling_decay_freq", type=int, default=25)
    parser.add_argument("--annealed_sampling_global_step", type=int, default=200)
    parser.add_argument("--annealed_sampling_warmup_period", type=int, default=10)
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    parser.add_argument("--ckpt_freq", type=int, default=-1, help="Checkpoint frequency for vllm outputs.")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.7, help='(for vllm) gpu memory utilization.')
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    if "deepseek" in args.model_name_or_path.lower() and args.n_sampling > 1:
        if args.ckpt_freq == -1:
            args.ckpt_freq = 64
    if "global_step_" in args.model_name_or_path:
        # global_step_x is part of the path, use regex to get the global_step
        try:
            args.annealed_sampling_global_step = int(re.search(r"global_step_(\d+)", args.model_name_or_path).group(1))
        except:
            print(f"Warning: global_step_x is part of the path, but cannot find the global_step, use default value {args.annealed_sampling_global_step}")
    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        random.seed(args.seed)
        examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        # examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}_{args.num_shots}shot"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file

def setup_llm(args):
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if args.use_vllm:
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_logprobs=100
        )
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    else:
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
        )
    return llm, tokenizer

def create_stop_words(args):
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")
    return stop_words

def annealed_sampling_processor(token_ids: Union[list[int], tuple[int]], logits: torch.Tensor, 
                               exploration_temp: float = 1.0, stability_temp: float = 0.1, 
                               decay_freq: int = 50, global_step: int = 0,
                               decay_mode: str = 'both', warmup_period: int = 10) -> torch.Tensor:
    """
    Annealed sampling logits processor for vLLM.
    
    Args:
        token_ids: List of token IDs generated so far
        logits: Logits tensor from the model
        exploration_temp: Exploration temperature (higher = more exploration)
        stability_temp: Stability temperature (lower = more focused)
        decay_freq: Decay frequency for temperature annealing
        global_step: Current global optimization step
        decay_mode: Which annealing mode to use. Options: 'global_step', 'token_length', 'both', 'none', 'adaptive'.
        warmup_period: If len(token_ids) < warmup_period, do not apply temperature scaling (default: 10)
        adaptive_decay: Whether to use adaptive decay based on historical performance
        uid: Unique identifier for the current trajectory (required for adaptive decay)
        historical_manager: Historical data manager instance (optional, will use global if None)
    Returns:
        Modified logits tensor
    """
    # If in warmup period, do not apply temperature scaling
    if len(token_ids) < warmup_period:
        return logits
    
    # Calculate the current temperature based on the selected decay mode
    if decay_mode == 'global_step':
        current_temp = stability_temp + (exploration_temp - stability_temp) * np.exp(-global_step / decay_freq)
    elif decay_mode == 'token_length':
        current_temp = stability_temp + (exploration_temp - stability_temp) * np.exp(-len(token_ids) / (20 * decay_freq))
    elif decay_mode == 'both':
        _exploration_temp = exploration_temp * np.exp(-global_step / decay_freq)
        current_temp = stability_temp + (_exploration_temp - stability_temp) * np.exp(-len(token_ids) / (20 * decay_freq))
    elif decay_mode == "both_v_1_5":
        _decay_freq = min(decay_freq + 5 * global_step, 2000)
        current_temp = stability_temp + (exploration_temp - stability_temp) * np.exp(-len(token_ids) / (20 * _decay_freq))
    elif decay_mode == "negexp":
        # as we use -exp(x/d), we need to use a larger decay_freq to get a smaller temperature and to keep the temperature >= 0
        _decay_freq = min(decay_freq + 5 * global_step, 40000)
        current_temp = 1 + exploration_temp - np.exp(len(token_ids) / (20 * _decay_freq))
        # avoid temperature < stability_temp
        current_temp = max(current_temp, stability_temp)
    elif decay_mode == "steps_variant":
        # use the same temperature at all positions, no matter how long token_ids is
        # the temperature gradually increases from stability_temp to exploration_temp
        current_temp = exploration_temp + (stability_temp - exploration_temp) * np.exp(-global_step / decay_freq)
        current_temp = min(current_temp, 1.0)
    elif decay_mode == "steps_variant_rev":
        # use the same temperature at all positions, no matter how long token_ids is
        # the temperature gradually increases from stability_temp to exploration_temp
        current_temp = stability_temp + (exploration_temp - stability_temp) * np.exp(-global_step / decay_freq)
        current_temp = max(current_temp, 0.1)
    elif decay_mode == 'none':
        current_temp = exploration_temp
    else:
        raise ValueError(f"Unknown decay_mode: {decay_mode}")

    # Apply temperature scaling to logits
    logits = logits / current_temp
    
    return logits

def run_llm_via_vllm(args, llm, prompts, stop_words=None, checkpoint_filename=None):
    if stop_words is None:
        stop_words = create_stop_words(args)
    logliks = []
    token_ids_num = []
    output_texts = []
    entropies = []
    start = 0
    _annealed_sampling_processor = lambda token_ids, logits: annealed_sampling_processor(
        token_ids, 
        logits, 
        exploration_temp=args.annealed_sampling_exploration_temp, 
        stability_temp=args.annealed_sampling_stability_temp, 
        decay_freq=args.annealed_sampling_decay_freq, 
        global_step=args.annealed_sampling_global_step,
        decay_mode=args.annealed_sampling_decay_mode, 
        warmup_period=args.annealed_sampling_warmup_period
    )
    sampling_param = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens_per_call,
        logprobs=50,
        n=1,
        stop=stop_words,
        stop_token_ids=(
            [151645, 151643]
            if "qwen2" in args.model_name_or_path.lower()
            else None
        ),
        logits_processors=[_annealed_sampling_processor] if args.use_annealed_sampling else None,
    )
    if os.path.exists(checkpoint_filename):
        logliks, token_ids_num, output_texts, entropies = torch.load(checkpoint_filename)
        start = len(output_texts)
    if args.ckpt_freq > 0:
        for j in range(start, len(prompts), args.ckpt_freq):
            outputs = llm.generate(prompts[j: j + args.ckpt_freq], sampling_param)
            inc_logliks, inc_token_ids_num, inc_output_texts, inc_entropies = postprocessing_vllm_outputs(outputs, args)
            logliks.extend(inc_logliks)
            token_ids_num.extend(inc_token_ids_num)
            output_texts.extend(inc_output_texts)
            entropies.extend(inc_entropies)
            torch.save([logliks, token_ids_num, output_texts, entropies], checkpoint_filename)
            print(f"Saved Checkpoints at {j} / {len(prompts)} to {checkpoint_filename}")
    else:
        outputs = llm.generate(
            prompts[start:], sampling_param
        )
        inc_logliks, inc_token_ids_num, inc_output_texts, inc_entropies = postprocessing_vllm_outputs(outputs, args)
        logliks.extend(inc_logliks)
        token_ids_num.extend(inc_token_ids_num)
        output_texts.extend(inc_output_texts)
        entropies.extend(inc_entropies)
    torch.save([logliks, token_ids_num, output_texts, entropies], checkpoint_filename)
    print(f"LLM Generation Complete, Save Intermediate Results at {checkpoint_filename}")
    outputs = output_texts
    return logliks, token_ids_num, output_texts, entropies, outputs


def setup(args):
    # load model

    # infer & eval
    data_list = args.data_names.split(",")
    # delay the loading of LLM until necessary
    global llm
    global tokenizer
    llm, tokenizer = None, None
    results = []
    for data_name in data_list:
        # results.append(main(llm, tokenizer, data_name, args))
        results.append(main(data_name, args))

    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )

    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True

def postprocessing_vllm_outputs(vllm_outputs, args):
    outputs = sorted(
        vllm_outputs, key=lambda x: int(x.request_id)
    )  # sort outputs by request_id
    all_vllm_outputs = [output.outputs[0] for output in outputs]
    logliks = [output.outputs[0].cumulative_logprob for output in outputs]
    token_ids_num = [(len(output.prompt_token_ids), len(output.outputs[0].token_ids)) for output in outputs]
    output_texts = [output.outputs[0].text for output in outputs]
    entropies_w_token_ids = get_tokenwise_entropy_from_vllm_outputs(all_vllm_outputs, p=args.top_p, top_p_mode=True)
    entropies = [x[0] for x in entropies_w_token_ids]
    return logliks, token_ids_num, output_texts, entropies


def main(data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    output_metric_filename = out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json")
    checkpoint_filename = out_file.replace(".jsonl", ".ckpt")
    if not args.overwrite and len(examples) == 0 and os.path.exists(output_metric_filename):
        # no remained samples, just ends
        print(f"No remained examples and the user do not want to overwrite, report last time results for {data_name} wo llm initialization to save time")
        results_json = json.load(open(output_metric_filename))
        return results_json
    else:
        global llm, tokenizer
        if llm is None:
            llm, tokenizer = setup_llm(args)

    print("=" * 50)
    print("data:", data_name, " , remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    stop_words = create_stop_words(args)

    # start inference
    # measure time use
    start_time = time.time()
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
        if args.use_vllm:
            logliks, token_ids_num, output_texts, entropies, outputs = run_llm_via_vllm(args, llm, prompts, stop_words, checkpoint_filename)
        else:
            outputs = generate_completions(
                model=llm,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_tokens_per_call,
                batch_size=16,
                stop_id_sequences=stop_words,
            )

        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif "boxed" not in output and output.endswith("```"):
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute the remain prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
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
        sample.update({"code": code, "pred": preds, "report": reports, "logliks": loglik_piece, "token_ids_num": token_ids_num_piece, "entropies": entropies_piece})
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)

    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )

    with open(
        output_metric_filename, "w"
    ) as f:
        json.dump(result_json, f, indent=4)
        # pickle.dump(result_json, f)
    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
