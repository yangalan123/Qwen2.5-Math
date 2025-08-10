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
model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
prompt_type="cot"
n_sampling=64
NUM_TEST_SAMPLE=200
if [[ "${model,,}" =~ deepseek ]]; then
  #    MAX_TOKENS=16384
  MAX_TOKENS=32768
else
  MAX_TOKENS=2048
fi
temperature=0.6
top_p=0.9
  #--prefix_token_num 25 \
  #--prefix_num_per_task 5 \
  #--prefix_token_num 200 \
  #--prefix_num_per_task 1 \
python bon_from_middle_load_ckpt_results.py \
  --model_name_or_path ${model} \
  --top_p ${top_p} \
  --prefix_token_num 25 \
  --prefix_num_per_task 5 \
  --temperature ${temperature} \
  --data_name "mmlu_stem" \
  --gpu_memory_utilization 0.9 \
  --max_tokens ${MAX_TOKENS} \
  --n_sampling ${n_sampling} \
  --ckpt_freq 500
