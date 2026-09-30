#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_ROOT="${LEROBOT_RUNTIME_ROOT:-/extdata/hdd2/lerobot-smolvla}"
HF_CACHE="${GROOT_HF_HOME:-${RUNTIME_ROOT}/hf-cache-groot}"
EVAL_BIN="${RUNTIME_ROOT}/.venv-groot/bin/lerobot-eval"
PROCESSOR_ASSETS="${GROOT_PROCESSOR_ASSETS:-${HF_CACHE}/qwen3-vl-2b-processor-assets}"

LIBERO_SUITE="${LIBERO_SUITE:-libero_object}"
LIBERO_TASK_IDS="${LIBERO_TASK_IDS:-[0]}"
N_EPISODES="${N_EPISODES:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/groot_libero_${LIBERO_SUITE}}"

case "${LIBERO_SUITE}" in
    libero_spatial|libero_object|libero_goal|libero_10) ;;
    *)
        echo "Unsupported LIBERO suite: ${LIBERO_SUITE}" >&2
        echo "Expected one of: libero_spatial, libero_object, libero_goal, libero_10" >&2
        exit 1
        ;;
esac
GROOT_CHECKPOINT="${GROOT_LIBERO_CHECKPOINT:-${HF_CACHE}/gr00t17-lerobot-${LIBERO_SUITE}-640}"

if [[ ! -x "${EVAL_BIN}" ]]; then
    echo "GR00T lerobot-eval not found: ${EVAL_BIN}" >&2
    exit 1
fi
if [[ ! -f "${GROOT_CHECKPOINT}/model.safetensors" ]]; then
    echo "GR00T-LIBERO checkpoint not found: ${GROOT_CHECKPOINT}/model.safetensors" >&2
    echo "Download: nvidia/gr00t17-lerobot-${LIBERO_SUITE}-640" >&2
    exit 1
fi
if [[ ! -f "${PROCESSOR_ASSETS}/tokenizer.json" ]]; then
    echo "GR00T processor assets not found: ${PROCESSOR_ASSETS}/tokenizer.json" >&2
    exit 1
fi

cd "${REPO_ROOT}"
echo "Policy: ${GROOT_CHECKPOINT}"
echo "LIBERO suite: ${LIBERO_SUITE} | task ids: ${LIBERO_TASK_IDS} | episodes: ${N_EPISODES}"
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
    LD_LIBRARY_PATH="${RUNTIME_ROOT}/.venv-groot/lib/python3.12/site-packages/nvidia/npp/lib:${LD_LIBRARY_PATH:-}" \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${EVAL_BIN}" \
        --output_dir="${OUTPUT_DIR}" \
        --policy.path="${GROOT_CHECKPOINT}" \
        --policy.device=cuda \
        --policy.n_action_steps=8 \
        --policy.use_bf16=false \
        --policy.model_params_fp32=true \
        --policy.use_flash_attention=false \
        --tokenizer_path="${PROCESSOR_ASSETS}" \
        --rename_map='{"observation.images.image2":"observation.images.wrist_image"}' \
        --env.type=libero \
        --env.task="${LIBERO_SUITE}" \
        --env.task_ids="${LIBERO_TASK_IDS}" \
        --env.control_mode=relative \
        --eval.batch_size=1 \
        --eval.n_episodes="${N_EPISODES}" \
        --eval.use_async_envs=false \
        --env.max_parallel_tasks=1 \
        "$@"
