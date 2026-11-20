#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: bash %s NUM_PROCESSES [SFT arguments ...]\n' "$0"
    printf 'For SFT options: python -m cogalign.training.sft_qwenvl --help\n'
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi

if [[ ! ${1:-} =~ ^[1-9][0-9]*$ ]]; then
    usage >&2
    exit 2
fi
num_processes=$1
shift

launch_args=(--num_processes "$num_processes")
if [[ -n ${ACCELERATE_CONFIG_FILE:-} ]]; then
    launch_args+=(--config_file "$ACCELERATE_CONFIG_FILE")
elif [[ $num_processes != 1 ]]; then
    launch_args+=(--multi_gpu)
fi

exec python -m accelerate.commands.launch "${launch_args[@]}" \
    --module cogalign.training.sft_qwenvl "$@"
