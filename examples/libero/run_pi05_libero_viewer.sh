#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_ROOT="${LEROBOT_RUNTIME_ROOT:-/extdata/hdd2/lerobot-smolvla}"
HF_CACHE="${HF_HOME:-${RUNTIME_ROOT}/hf-cache}"
PYTHON="${RUNTIME_ROOT}/.venv/bin/python"
PI05_CHECKPOINT="${PI05_LIBERO_CHECKPOINT:-${HF_CACHE}/pi05_libero_finetuned}"
LIBERO_SUITE="${LIBERO_SUITE:-libero_object}"
LIBERO_TASK_ID="${LIBERO_TASK_ID:-0}"

for ((i = 1; i <= $#; i++)); do
    argument="${!i}"
    if [[ "${argument}" == "--task" && $i -lt $# ]]; then
        next_index=$((i + 1))
        LIBERO_SUITE="${!next_index}"
    elif [[ "${argument}" == --task=* ]]; then
        LIBERO_SUITE="${argument#*=}"
    elif [[ "${argument}" == "--task-id" && $i -lt $# ]]; then
        next_index=$((i + 1))
        LIBERO_TASK_ID="${!next_index}"
    elif [[ "${argument}" == --task-id=* ]]; then
        LIBERO_TASK_ID="${argument#*=}"
    fi
done

case "${LIBERO_SUITE}" in
    libero_spatial) DEFAULT_MAX_STEPS=280 ;;
    libero_object) DEFAULT_MAX_STEPS=280 ;;
    libero_goal) DEFAULT_MAX_STEPS=300 ;;
    libero_10) DEFAULT_MAX_STEPS=520 ;;
    *)
        echo "Unsupported LIBERO suite: ${LIBERO_SUITE}" >&2
        echo "Expected one of: libero_spatial, libero_object, libero_goal, libero_10" >&2
        exit 1
        ;;
esac
MAX_STEPS="${LIBERO_MAX_STEPS:-${DEFAULT_MAX_STEPS}}"
if [[ ! "${LIBERO_TASK_ID}" =~ ^[0-9]+$ ]] || ((LIBERO_TASK_ID < 0 || LIBERO_TASK_ID > 9)); then
    echo "Invalid LIBERO task id: ${LIBERO_TASK_ID}; expected 0..9" >&2
    exit 1
fi

if [[ ! -f "${PI05_CHECKPOINT}/model.safetensors" ]]; then
    echo "Pi0.5-LIBERO checkpoint not found: ${PI05_CHECKPOINT}/model.safetensors" >&2
    echo "Download lerobot/pi05_libero_finetuned before running this script." >&2
    exit 1
fi
PI_TOKENIZER="${PI_TOKENIZER_PATH:-${HF_CACHE}/paligemma-tokenizer}"
if [[ ! -f "${PI_TOKENIZER}/tokenizer.json" ]]; then
    echo "Local PaliGemma tokenizer not found: ${PI_TOKENIZER}" >&2
    exit 1
fi

cd "${REPO_ROOT}"
echo "Policy: ${PI05_CHECKPOINT}"
echo "LIBERO suite: ${LIBERO_SUITE} | task id: ${LIBERO_TASK_ID} | max steps: ${MAX_STEPS}"
echo "Pi0.5 uses one LIBERO-finetuned checkpoint for all four suites."

exec env \
    HF_HOME="${HF_CACHE}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    LIBERO_CONFIG_PATH="${RUNTIME_ROOT}/libero-config" \
    MUJOCO_GL=glfw \
    DISPLAY="${DISPLAY:-:0}" \
    XAUTHORITY="${XAUTHORITY:-/home/aitech/.Xauthority}" \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    LD_LIBRARY_PATH="${RUNTIME_ROOT}/.venv/lib/python3.12/site-packages/nvidia/npp/lib:${LD_LIBRARY_PATH:-}" \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PYTHON}" examples/libero/run_pi0_libero_viewer.py \
        --policy-path "${PI05_CHECKPOINT}" \
        --task "${LIBERO_SUITE}" \
        --task-id "${LIBERO_TASK_ID}" \
        --device cuda \
        --dtype float32 \
        --n-action-steps 10 \
        --tokenizer-path "${PI_TOKENIZER}" \
        --render-resolution 360 \
        --window-width 960 \
        --window-height 720 \
        --max-steps "${MAX_STEPS}" \
        --start-free-camera \
        "$@"
