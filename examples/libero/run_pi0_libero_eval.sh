#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_ROOT="${LEROBOT_RUNTIME_ROOT:-/extdata/hdd2/lerobot-smolvla}"
HF_CACHE="${HF_HOME:-${RUNTIME_ROOT}/hf-cache}"
EVAL_BIN="${RUNTIME_ROOT}/.venv/bin/lerobot-eval"
PI0_CHECKPOINT="${PI0_LIBERO_CHECKPOINT:-${HF_CACHE}/pi0_libero_finetuned}"

LIBERO_SUITE="${LIBERO_SUITE:-libero_object}"
LIBERO_TASK_IDS="${LIBERO_TASK_IDS:-[0]}"
N_EPISODES="${N_EPISODES:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/pi0_libero_${LIBERO_SUITE}}"

if [[ ! -x "${EVAL_BIN}" ]]; then
    echo "lerobot-eval not found: ${EVAL_BIN}" >&2
    exit 1
fi
if [[ ! -f "${PI0_CHECKPOINT}/model.safetensors" ]]; then
    echo "Pi0-LIBERO checkpoint not found: ${PI0_CHECKPOINT}/model.safetensors" >&2
    echo "Download lerobot/pi0_libero_finetuned before running this script." >&2
    exit 1
fi

cd "${REPO_ROOT}"
echo "Policy: ${PI0_CHECKPOINT}"
echo "LIBERO suite: ${LIBERO_SUITE} | task ids: ${LIBERO_TASK_IDS} | episodes: ${N_EPISODES}"
echo "Results: ${OUTPUT_DIR}"

# The repository's standard evaluation path is used here. This machine-specific
# launcher selects GLFW because EGL is unavailable on the Tesla M40 host, and
# converts the BF16 checkpoint to FP32 while streaming because compute capability
# 5.2 does not support BF16 execution.
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
    "${EVAL_BIN}" \
        --output_dir="${OUTPUT_DIR}" \
        --policy.path="${PI0_CHECKPOINT}" \
        --policy.device=cuda \
        --policy.dtype=float32 \
        --policy.compile_model=false \
        --env.type=libero \
        --env.task="${LIBERO_SUITE}" \
        --env.task_ids="${LIBERO_TASK_IDS}" \
        --env.control_mode=relative \
        --eval.batch_size=1 \
        --eval.n_episodes="${N_EPISODES}" \
        --eval.use_async_envs=false \
        --env.max_parallel_tasks=1 \
        "$@"
