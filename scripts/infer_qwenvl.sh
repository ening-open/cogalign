#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: bash %s {infer|hf_infer} OUTPUT_DIR DATA_JSON IMAGE_DIR MODEL_PATH MODEL_TAG [inference options ...]\n' "$0"
    printf 'For backend options: python -m cogalign.inference.qwenvl {infer|hf_infer} --help\n'
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi

if [[ $# -lt 6 || ( $1 != infer && $1 != hf_infer ) ]]; then
    usage >&2
    exit 2
fi

backend=$1
output_dir=$2
data_path=$3
image_dir=$4
model_path=$5
model_tag=$6
shift 6

if [[ ! -f $data_path || ! -d $image_dir ]]; then
    printf 'Expected an existing prediction input JSON file and image directory.\n' >&2
    exit 2
fi

mkdir -p -- "$output_dir"
shard_prefix="${model_tag}_response_"
if [[ $backend == hf_infer ]]; then
    shard_prefix="${model_tag}_hf_response_"
fi
shopt -s nullglob
existing_shards=("$output_dir/$shard_prefix"*.json)
if ((${#existing_shards[@]} > 0)); then
    printf 'Prediction shards already exist for %s in %s. Use a fresh output directory or model tag.\n' "$model_tag" "$output_dir" >&2
    exit 2
fi

exec python -m cogalign.inference.qwenvl "$backend" \
    "$@" \
    --output-dir "$output_dir" \
    --data "$data_path" \
    --image-dir "$image_dir" \
    --model-path "$model_path" \
    --model-tag "$model_tag"
