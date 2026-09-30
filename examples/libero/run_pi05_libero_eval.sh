#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_ROOT="${LEROBOT_RUNTIME_ROOT:-/extdata/hdd2/lerobot-smolvla}"
HF_CACHE="${HF_HOME:-${RUNTIME_ROOT}/hf-cache}"
EVAL_BIN="${RUNTIME_ROOT}/.venv/bin/lerobot-eval"
PI05_CHECKPOINT="${PI05_LIBERO_CHECKPOINT:-${HF_CACHE}/pi05_libero_finetuned}"

LIBERO_SUITE="${LIBERO_SUITE:-libero_object}"
LIBERO_TASK_IDS="${LIBERO_TASK_IDS:-[0]}"
N_EPISODES="${N_EPISODES:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/pi05_libero_${LIBERO_SUITE}}"

case "${LIBERO_SUITE}" in
    libero_spatial|libero_object|libero_goal|libero_10) ;;
    *)
        echo "Unsupported LIBERO suite: ${LIBERO_SUITE}" >&2
        echo "Expected one of: libero_spatial, libero_object, libero_goal, libero_10" >&2
        exit 1
        ;;
esac

if [[ ! -x "${EVAL_BIN}" ]]; then
    echo "lerobot-eval not found: ${EVAL_BIN}" >&2
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
echo "LIBERO suite: ${LIBERO_SUITE} | task ids: ${LIBERO_TASK_IDS} | episodes: ${N_EPISODES}"
echo "Action steps: 10 (official OpenPI Pi0.5 LIBERO setting)"
echo "Results: ${OUTPUT_DIR}"

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
        --policy.path="${PI05_CHECKPOINT}" \
        --policy.device=cuda \
        --policy.dtype=float32 \
        --policy.compile_model=false \
        --policy.n_action_steps=10 \
        --tokenizer_path="${PI_TOKENIZER}" \
        --env.type=libero \
        --env.task="${LIBERO_SUITE}" \
        --env.task_ids="${LIBERO_TASK_IDS}" \
        --env.control_mode=relative \
        --eval.batch_size=1 \
        --eval.n_episodes="${N_EPISODES}" \
        --eval.use_async_envs=false \
        --env.max_parallel_tasks=1 \
        "$@"
