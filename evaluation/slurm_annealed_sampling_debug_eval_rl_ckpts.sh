#!/bin/bash
#SBATCH --mail-user=chenghao@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/net/scratch/chenghao/persona_following/slurm_output/%A_%a.%N.stdout
#SBATCH --error=/net/scratch/chenghao/persona_following/slurm_output/%A_%a.%N.stderr
#SBATCH --chdir=/net/scratch/chenghao/persona_following/slurm_output
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=400gb
#SBATCH --job-name=run_math_eval_sampling_effects
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
cd /net/scratch/chenghao/persona_following
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /net/scratch/chenghao/persona_following/env
cd sampling_effects/Qwen2.5-Math/evaluation
root_dir=/net/scratch2/chenghao/annealing_sampling/checkpoints/minimal_rl_numina_math/annealed_sampling_grpo_explore_1.2_stable_0.1_decay_freq_100
# prompt_type=cot
prompt_type="qwen25-math-cot"
temperature=1.0
n_sampling=64
top_p=0.9
NUM_TEST_SAMPLE=200
for model in ${root_dir}/global_step_*/huggingface; do
    echo "Processing ${model}"
    CUDA_VISIBLE_DEVICES=3 bash sh/eval_annealed_rl.sh ${prompt_type} ${model} ${temperature} ${n_sampling} ${top_p} ${NUM_TEST_SAMPLE}
done