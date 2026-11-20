#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s RSM_DIR RDM_OUTPUT_DIR HUMAN_RDM_DIR TAG [TAG ...]\n' "$0"
}

if [[ $# -eq 1 && $1 == --help ]]; then
    usage
    exit 0
fi

if [[ $# -lt 4 ]]; then
    usage >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export COGALIGN_RSA_RSM_DIR="$1"
export COGALIGN_RSA_RDM_DIR="$2"
export COGALIGN_RSA_HUMAN_RDM_DIR="$3"
shift 3

# Keep each model tag separate while passing it through the environment.
for tag in "$@"; do
    if [[ -z $tag || $tag == *$'\n'* ]]; then
        printf 'Model tags must be non-empty and contain no newlines.\n' >&2
        exit 2
    fi
done
COGALIGN_RSA_MODEL_TAGS=$(printf '%s\n' "$@")
export COGALIGN_RSA_MODEL_TAGS

exec octave --no-gui --quiet --norc --path "$script_dir" --eval '
    rsm_dir = getenv("COGALIGN_RSA_RSM_DIR");
    rdm_dir = getenv("COGALIGN_RSA_RDM_DIR");
    human_rdm_dir = getenv("COGALIGN_RSA_HUMAN_RDM_DIR");
    model_tags = strsplit(getenv("COGALIGN_RSA_MODEL_TAGS"), char(10));
    get_rdm_48object(rsm_dir, rdm_dir, model_tags);
    get_human_behavior_consistency(rdm_dir, human_rdm_dir, model_tags);
'
