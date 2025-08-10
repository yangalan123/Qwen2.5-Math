#!/bin/bash
#SBATCH --mail-user=chenghao@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/net/scratch2/chenghao/persona_following/slurm_output/%A_%a.%N.stdout
#SBATCH --error=/net/scratch2/chenghao/persona_following/slurm_output/%A_%a.%N.stderr
#SBATCH --chdir=/net/scratch2/chenghao/persona_following/slurm_output
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=400gb
#SBATCH --job-name=small_math_eval_sampling_effects
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=11:59:00
#SBATCH --signal=SIGUSR1@120

echo $PATH
#export OUTLINES_CACHE_DIR=/net/scratch/chenghao/tmp/.outlines
#export VLLM_LOGGING_LEVEL=DEBUG
#export CUDA_LAUNCH_BLOCKING=1
#export NCCL_DEBUG=TRACE
#export VLLM_TRACE_FUNCTION=1
# please uncomment -- just comment for debugging at 10/1/2023
cd /net/scratch2/chenghao/persona_following
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /net/scratch2/chenghao/persona_following/env
cd sampling_effects/Qwen2.5-Math/evaluation

models=("deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        "meta-llama/Llama-3.1-8B"
        "meta-llama/Llama-3.1-8B-Instruct"
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        "Qwen/Qwen2.5-Math-7B-Instruct"
        "Qwen/Qwen2.5-Math-7B"
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        "meta-llama/Meta-Llama-3-70B"
        "meta-llama/Meta-Llama-3-70B-Instruct"
        "meta-llama/Llama-3.3-70B-Instruct")

#prompt_types=("cot" "direct" "qwen25-math-cot")
#prompt_types=("qwen25-math-cot")
prompt_types=("cot")

total_tasks=${#models[@]}*${#prompt_types[@]}
if ((SLURM_ARRAY_TASK_ID >= total_tasks)); then
  echo "Error: Invalid task ID $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

#model_idx=$((SLURM_ARRAY_TASK_ID / (${#constraints[@]} * ${#top_ps[@]})))
model_idx=$((SLURM_ARRAY_TASK_ID / ${#prompt_types[@]}))
#constraint_idx=$((SLURM_ARRAY_TASK_ID / ${#top_ps[@]} % ${#constraints[@]}))
prompt_type_idx=$((SLURM_ARRAY_TASK_ID % ${#prompt_types[@]}))
#top_p_idx=$((SLURM_ARRAY_TASK_ID % ${#top_ps[@]}))
#temperature=0.5
#temperature=0.6
#temperature=0
n_sampling=64
#top_p=1.0
#top_p=0.9
NUM_TEST_SAMPLE=200

model=${models[$model_idx]}
prompt_type=${prompt_types[$prompt_type_idx]}

for temperature in 0.6 1.0
do
  for top_p in 0.9 1.0
  do
    bash sh/eval.sh ${prompt_type} ${model} ${temperature} ${n_sampling} ${top_p} ${NUM_TEST_SAMPLE}
  done
done
