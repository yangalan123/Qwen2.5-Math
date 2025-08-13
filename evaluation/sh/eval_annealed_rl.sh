set -ex

PROMPT_TYPE=${1:-"cot"}
MODEL_NAME_OR_PATH=${2:-"Qwen/Qwen2.5-Math-7B"}
TEMPERATURE=${3:-1.0}
N_SAMPLING=${4:-1}
TOP_P=${5:-1.0}
NUM_TEST_SAMPLE=${6:-"-1"}

if [[ "${MODEL_NAME_OR_PATH,,}" =~ deepseek ]]; then
  #    MAX_TOKENS=16384
  MAX_TOKENS=32768
else
  MAX_TOKENS=3072
fi

if [[ "$TEMPERATURE" == "0" ]]; then
  seeds=(1)
else
#  seeds=(0 1 2)
  seeds=(0)
fi

SPLIT="test"
#NUM_TEST_SAMPLE=-1
for seed in "${seeds[@]}"; do
  # basename=${MODEL_NAME_OR_PATH##*/}
  # get the the last directory
  basename=$(basename $(dirname ${MODEL_NAME_OR_PATH}))
  # get the second to last directory
  exp_name=$(basename $(dirname $(dirname ${MODEL_NAME_OR_PATH})))
  OUTPUT_DIR=${exp_name}/${basename}
  echo "OUTPUT_DIR: ${OUTPUT_DIR}"
  echo "MODEL_NAME_OR_PATH: ${MODEL_NAME_OR_PATH}"
  echo "SEED: ${seed}, TEMPERATURE: ${TEMPERATURE}, N_SAMPLING: ${N_SAMPLING}, MAX_TOKENS: ${MAX_TOKENS}"
#  echo "Evaluating MATH-500 Datasets"
#  #  DATA_NAME="gsm8k,math,svamp,asdiv,mawps,carp_en,tabmwp,minerva_math,gaokao2023en,olympiadbench,college_math"
#  DATA_NAME="math-500"
#
#  if [[ "${MODEL_NAME_OR_PATH,,}" =~ deepseek ]]; then
#    MATH_NUM_SHOTS=0
#  else
#    if [[ "${PROMPT_TYPE,,}" =~ cot ]]; then
#      MATH_NUM_SHOTS=0
#    else
#      MATH_NUM_SHOTS=5
#    fi
#  fi
#  TOKENIZERS_PARALLELISM=false \
#    python3 -u math_eval.py \
#    --model_name_or_path ${MODEL_NAME_OR_PATH} \
#    --data_name ${DATA_NAME} \
#    --output_dir ${OUTPUT_DIR} \
#    --split ${SPLIT} \
#    --prompt_type ${PROMPT_TYPE} \
#    --num_test_sample ${NUM_TEST_SAMPLE} \
#    --max_tokens_per_call ${MAX_TOKENS} \
#    --seed ${seed} \
#    --temperature ${TEMPERATURE} \
#    --n_sampling ${N_SAMPLING} \
#    --top_p ${TOP_P} \
#    --start 0 \
#    --end -1 \
#    --num_shots ${MATH_NUM_SHOTS} \
#    --use_vllm \
#    --save_outputs

#  # English competition datasets
#  if [[ "${MODEL_NAME_OR_PATH,,}" =~ deepseek ]]; then
#    echo "Evaluating English Competition Datasets"
#    #  DATA_NAME="aime24,amc23"
#    DATA_NAME="aime24"
#    TOKENIZERS_PARALLELISM=false \
#      python3 -u math_eval.py \
#      --model_name_or_path ${MODEL_NAME_OR_PATH} \
#      --data_name ${DATA_NAME} \
#      --output_dir ${OUTPUT_DIR} \
#      --split ${SPLIT} \
#      --prompt_type ${PROMPT_TYPE} \
#      --num_test_sample ${NUM_TEST_SAMPLE} \
#      --max_tokens_per_call ${MAX_TOKENS} \
#      --seed ${seed} \
#      --temperature ${TEMPERATURE} \
#      --n_sampling ${N_SAMPLING} \
#      --top_p ${TOP_P} \
#      --start 0 \
#      --end -1 \
#      --use_vllm \
#      --save_outputs
#  fi

  # English multiple-choice datasets
  echo "Evaluating English Multiple-Choice Datasets"
  #  DATA_NAME="aqua,sat_math,mmlu_stem"
  DATA_NAME="mmlu_stem,aime24,math-500"
#  if [[ "${MODEL_NAME_OR_PATH,,}" =~ deepseek ]]; then
#    DATA_NAME="mmlu_stem"
#  else
#    DATA_NAME="mmlu_stem,mmlu"
#  fi
  TOKENIZERS_PARALLELISM=false VLLM_USE_V1=0 \
    python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR}/annealed \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --max_tokens_per_call ${MAX_TOKENS} \
    --seed ${seed} \
    --temperature ${TEMPERATURE} \
    --n_sampling ${N_SAMPLING} \
    --top_p ${TOP_P} \
    --start 0 \
    --end -1 \
    --use_vllm \
    --use_annealed_sampling \
    --save_outputs \
    --num_shots 5

  TOKENIZERS_PARALLELISM=false VLLM_USE_V1=0 \
    python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR}/normal \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --max_tokens_per_call ${MAX_TOKENS} \
    --seed ${seed} \
    --temperature 0.6 \
    --n_sampling ${N_SAMPLING} \
    --top_p 0.9 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --num_shots 5


  #  echo "Evaluating English Open Datasets"
  #  DATA_NAME="olympiadbench,math"
  #  TOKENIZERS_PARALLELISM=false \
  #  python3 -u math_eval.py \
  #      --model_name_or_path ${MODEL_NAME_OR_PATH} \
  #      --data_name ${DATA_NAME} \
  #      --output_dir ${OUTPUT_DIR} \
  #      --split ${SPLIT} \
  #      --prompt_type ${PROMPT_TYPE} \
  #      --num_test_sample ${NUM_TEST_SAMPLE} \
  #      --max_tokens_per_call ${MAX_TOKENS} \
  #      --seed ${seed} \
  #      --temperature ${TEMPERATURE} \
  #      --n_sampling ${N_SAMPLING} \
  #      --top_p ${TOP_P} \
  #      --start 0 \
  #      --end -1 \
  #      --use_vllm \
  #      --save_outputs \
  #      --overwrite
done
# English open datasets

# Chinese gaokao collections
#DATA_NAME="gaokao2024_I,gaokao2024_II,gaokao2024_mix,gaokao_math_cloze,gaokao_math_qa"
#TOKENIZERS_PARALLELISM=false \
#python3 -u math_eval.py \
#    --model_name_or_path ${MODEL_NAME_OR_PATH} \
#    --data_name ${DATA_NAME} \
#    --output_dir ${OUTPUT_DIR} \
#    --split ${SPLIT} \
#    --prompt_type ${PROMPT_TYPE} \
#    --num_test_sample ${NUM_TEST_SAMPLE} \
#    --seed 0 \
#    --temperature 0 \
#    --n_sampling 1 \
#    --top_p 1 \
#    --start 0 \
#    --end -1 \
#    --use_vllm \
#    --save_outputs \
#    --overwrite \
#    --adapt_few_shot

# Chinese other datasets
#DATA_NAME="cmath,cn_middle_school"
#TOKENIZERS_PARALLELISM=false \
#python3 -u math_eval.py \
#    --model_name_or_path ${MODEL_NAME_OR_PATH} \
#    --data_name ${DATA_NAME} \
#    --output_dir ${OUTPUT_DIR} \
#    --split ${SPLIT} \
#    --prompt_type ${PROMPT_TYPE} \
#    --num_test_sample ${NUM_TEST_SAMPLE} \
#    --seed 0 \
#    --temperature 0 \
#    --n_sampling 1 \
#    --top_p 1 \
#    --start 0 \
#    --end -1 \
#    --use_vllm \
#    --save_outputs \
#    --overwrite \
#    --adapt_few_shot
