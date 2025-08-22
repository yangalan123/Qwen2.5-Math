#!/bin/bash

#SBATCH --chdir=/fsx/zhuokai/verl/
#SBATCH --mem 128G
#SBATCH -c 64
#SBATCH --job-name=draw_best_k_visualization_zzk
#SBATCH --output=/fsx/zhuokai/verl/slurm/draw_best_k_visualization_zzk.stdout
#SBATCH --error=/fsx/zhuokai/verl/slurm/draw_best_k_visualization_zzk.stderr

ROOT_DIR=/fsx/zhuokai/verl/
ROOT_DIR=/fsx/zhuokai/verl/

project_name="minimal_rl_numina_math"
#[TODO]: fill in the experiment name
experiment_name=""
#[TODO]: fill in the base model name, use llama-3.2-1b-instruct for now
# This base_model_name is used for filename and plot title
base_model_name=Llama-3.2-1B-Instruct
ckpt_dir=checkpoints/${project_name}/${experiment_name}

for global_step_dir in ${ckpt_dir}/global_step_*; do
    basename=$(basename ${global_step_dir})
    python visualize_best_at_k.py --output_dir outputs/${experiment_name}/${basename} --visualization_dir ./rl_ckpts_evaluation/debug_visualization_two_curves --model_name $base_model_name
done