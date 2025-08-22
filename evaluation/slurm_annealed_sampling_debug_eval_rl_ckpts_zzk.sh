#!/bin/bash

#SBATCH --chdir=/fsx/zhuokai/Qwen2.5-Math/evaluation
#SBATCH --gres=gpu:1
#SBATCH --mem 128G
#SBATCH -c 64
#SBATCH --job-name=eval_annealed_sampling_llama_3.2_1b
#SBATCH --output=/fsx/zhuokai/Qwen2.5-Math/evaluation/slurm/eval_annealed_sampling_llama_3.2_1b.stdout
#SBATCH --error=/fsx/zhuokai/Qwen2.5-Math/evaluation/slurm/eval_annealed_sampling_llama_3.2_1b.stderr

ROOT_DIR=/fsx/zhuokai/verl/

project_name="minimal_rl_numina_math"
#[TODO]: fill in the experiment name
experiment_name="annealed_sampling_minimal_rl_negexp_explore_1.2_stable_0.1_decay_freq_25_warmup_period_10_negexp_Llama-3.2-1B-Instruct_zzk"
ckpt_dir=/fsx/zhuokai/verl/checkpoints/${project_name}/${experiment_name}
# prompt_type=cot
prompt_type="qwen25-math-cot"
# dummy setup for temperature and top_p -- if you want to change it, you can do it in the sh/eval_annealed_rl.sh script
temperature=1.0
n_sampling=64
top_p=0.9
NUM_TEST_SAMPLE=200
pwd=$(pwd)
model_path=Qwen/Qwen2.5-Math-1.5B
for global_step_dir in ${ckpt_dir}/global_step_*; do
    echo "Processing ${global_step_dir}"
    cd ${ROOT_DIR}/scripts
    python legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir ${global_step_dir}/actor \
    --hf_model_path ${model_path} \
    --target_dir ${global_step_dir}/huggingface
    cd ${pwd}
    bash sh/eval_annealed_rl.sh ${prompt_type} ${global_step_dir}/huggingface ${temperature} ${n_sampling} ${top_p} ${NUM_TEST_SAMPLE}
done