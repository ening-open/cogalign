#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: bash %s SPLITS DATA_DIR IMAGE_DIR MODEL_PATH OUTPUT_DIR\n' "$0"
    printf 'Use comma-separated split IDs, for example 01,02.\n'
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi

if [[ $# -ne 5 || ! $1 =~ ^[0-9]{1,2}(,[0-9]{1,2})*$ ]]; then
    usage >&2
    exit 2
fi

IFS=, read -r -a requested_splits <<< "$1"
data_dir=$2
image_dir=$3
model_path=$4
output_dir=$5
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

splits=()
for split in "${requested_splits[@]}"; do
    if [[ ${#split} -eq 1 ]]; then
        split="0$split"
    fi
    split_data="$data_dir/o1o_48object_base_split_${split}.json"
    if [[ ! -f $split_data ]]; then
        printf 'Split file not found: %s\n' "$split_data" >&2
        exit 2
    fi
    splits+=("$split")
done

for split in "${splits[@]}"; do
    bash "$script_dir/infer_qwenvl.sh" infer \
        "$output_dir/$split/chunks" \
        "$data_dir/o1o_48object_base_split_${split}.json" \
        "$image_dir" "$model_path" qwen3vl_4b_it_o1o48obj_align \
        --image-cache 1600 \
        --max-new-tokens 2048 \
        --temperature 0.0 \
        --top-p 0.001 \
        --top-k 1 \
        --repetition-penalty 1.0 \
        --presence-penalty 0.0 \
        --min-pixels 12544 \
        --max-pixels 451584 \
        --save-interval 256 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.92 \
        --max-model-len 8192
done
