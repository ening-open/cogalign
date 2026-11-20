#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: bash %s MODEL_PATH IMAGE_DIR TRAIN_JSON EVAL_JSON OUTPUT_DIR\n' "$0"
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi

if [[ $# -ne 5 ]]; then
    usage >&2
    exit 2
fi

model_path=$1
image_dir=$2
train_data=$3
eval_data=$4
output_dir=$5
if [[ ! -d $image_dir || ! -f $train_data || ! -f $eval_data ]]; then
    printf 'Expected an image directory and existing training and evaluation JSON files.\n' >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

exec bash "$script_dir/sft_qwenvl.sh" "${NUM_PROCESSES:-6}" \
    --seed 42 \
    --output_dir "$output_dir" \
    --dataset base \
    --image_dir "$image_dir" \
    --train_data "$train_data" \
    --eval_data "$eval_data" \
    --model_type qwen3vl_4b_it_align_ds_rationale_steer_align \
    --model_name "$model_path" \
    --attn_impl flash_attention_2 \
    --tune_vision false \
    --tune_merger true \
    --tune_llm true \
    --deepspeed 2 \
    --per_device_train_batch_size "${TRAIN_BATCH_SIZE:-4}" \
    --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-4}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-6}" \
    --offload_optimizer false \
    --offload_param false \
    --lr 1e-5 \
    --epochs 1 \
    --skip_eval false \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --max_pixels 451584 \
    --min_pixels 12544 \
    --use_4bit_quantization false \
    --gradient_checkpointing true \
    --do_eval true \
    --swanlab_enable "${SWANLAB_ENABLE:-false}"
