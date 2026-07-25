#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
    echo "usage: $0 [output-directory]" >&2
    exit 2
fi

collection_id="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="${1:-paper/data/raw/${collection_id}}"
repetitions="${BLADE_SYNC_REPETITIONS:-3}"
allow_dirty="${BLADE_SYNC_ALLOW_DIRTY:-0}"
device_id="${BLADE_SYNC_DEVICE_ID:-}"

if [[ ! "${repetitions}" =~ ^[1-9][0-9]*$ ]]; then
    echo "BLADE_SYNC_REPETITIONS must be a positive integer" >&2
    exit 2
fi
if [[ "${allow_dirty}" != "0" && "${allow_dirty}" != "1" ]]; then
    echo "BLADE_SYNC_ALLOW_DIRTY must be 0 or 1" >&2
    exit 2
fi
if [[ -n "${device_id}" && ! "${device_id}" =~ ^[0-9]+$ ]]; then
    echo "BLADE_SYNC_DEVICE_ID must be an unsigned integer" >&2
    exit 2
fi
if [[ -e "${output_dir}" ]]; then
    echo "refusing to reuse existing output directory: ${output_dir}" >&2
    exit 2
fi
if [[ "${allow_dirty}" == "0" && -n "$(git status --porcelain)" ]]; then
    echo "refusing to collect final data from a dirty worktree" >&2
    echo "commit the artifact or set BLADE_SYNC_ALLOW_DIRTY=1 for provisional runs" >&2
    exit 2
fi
if ! command -v shuf >/dev/null; then
    echo "the matrix runner requires shuf to randomize configuration order" >&2
    exit 2
fi

mkdir -p "${output_dir}"

git status --short --branch >"${output_dir}/git-status.txt"
git rev-parse HEAD >"${output_dir}/git-revision.txt"
{
    date -u --iso-8601=seconds
    uname -a
    rustc --version
    cargo --version
    sha256sum Cargo.lock
    echo "BLADE_SYNC_REPETITIONS=${repetitions}"
    echo "BLADE_SYNC_DEVICE_ID=${device_id}"
} >"${output_dir}/system.txt"
vulkaninfo --summary >"${output_dir}/vulkaninfo.txt" 2>&1
vulkaninfo >"${output_dir}/vulkaninfo-full.txt" 2>&1

if command -v nvidia-smi >/dev/null; then
    nvidia-smi -q >"${output_dir}/nvidia-smi.txt" 2>&1 || true
fi
if command -v rocm-smi >/dev/null; then
    rocm-smi --showallinfo >"${output_dir}/rocm-smi.txt" 2>&1 || true
fi

workloads=(
    compute-independent
    compute-chain
    graphics-independent
    graphics-chain
)
policies=(
    automatic
    hazard-only
    explicit-all
)

configurations=()
for workload in "${workloads[@]}"; do
    for policy in "${policies[@]}"; do
        configurations+=("${workload}__${policy}")
    done
done

cargo build --quiet --release --example sync-bench

device_arguments=()
if [[ -n "${device_id}" ]]; then
    device_arguments=(--device-id "${device_id}")
fi

echo "repetition,index,configuration" >"${output_dir}/order.txt"
for ((repetition = 1; repetition <= repetitions; repetition++)); do
    mapfile -t randomized < <(shuf -e -- "${configurations[@]}")
    for index in "${!randomized[@]}"; do
        configuration="${randomized[index]}"
        workload="${configuration%%__*}"
        policy="${configuration##*__}"
        printf -v run_id "r%02d__%s" "${repetition}" "${configuration}"
        echo "${repetition},${index},${configuration}" >>"${output_dir}/order.txt"
        echo "${run_id}" >&2
        cargo run --quiet --release --example sync-bench -- \
            --workload "${workload}" \
            --policy "${policy}" \
            "${device_arguments[@]}" \
            >"${output_dir}/${run_id}.csv"
    done
done

echo "Raw results: ${output_dir}" >&2
