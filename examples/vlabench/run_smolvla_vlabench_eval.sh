#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_ROOT="${LEROBOT_RUNTIME_ROOT:-/extdata/hdd2/lerobot-smolvla}"
HF_CACHE="${HF_HOME:-${RUNTIME_ROOT}/hf-cache}"
EVAL_BIN="${RUNTIME_ROOT}/.venv/bin/lerobot-eval"
SMOLVLA_CHECKPOINT="${SMOLVLA_VLABENCH_CHECKPOINT:-${HF_CACHE}/hub/models--lerobot--smolvla_vlabench/snapshots/4fd586e12dc14b04d9d606ddbb77448df4f0ff29}"

VLABENCH_TASK="${VLABENCH_TASK:-select_toy}"
N_EPISODES="${N_EPISODES:-5}"
N_ACTION_STEPS="${N_ACTION_STEPS:-50}"
EPISODE_LENGTH="${EPISODE_LENGTH:-500}"
RENDER_RESOLUTION="${RENDER_RESOLUTION:-[256,256]}"
SEED="${SEED:-1000}"
DETERMINISTIC_TRACK="${VLABENCH_DETERMINISTIC_TRACK:-}"
TRACK_SEED_OFFSET="${VLABENCH_TRACK_SEED_OFFSET:-${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/smolvla_vlabench_${VLABENCH_TASK}_n${N_EPISODES}}"

TRACK_ARGS=()
if [[ -n "${DETERMINISTIC_TRACK}" ]]; then
    TRACK_ARGS+=("--env.deterministic_track=${DETERMINISTIC_TRACK}")
    TRACK_ARGS+=("--env.track_seed_offset=${TRACK_SEED_OFFSET}")
fi

if [[ ! -x "${EVAL_BIN}" ]]; then
    echo "lerobot-eval not found: ${EVAL_BIN}" >&2
    exit 1
fi
if [[ ! -f "${SMOLVLA_CHECKPOINT}/model.safetensors" ]]; then
    echo "SmolVLA VLABench checkpoint not found: ${SMOLVLA_CHECKPOINT}/model.safetensors" >&2
    exit 1
fi

cd "${REPO_ROOT}"
echo "Policy: ${SMOLVLA_CHECKPOINT}"
echo "VLABench task: ${VLABENCH_TASK} | episodes: ${N_EPISODES} | seed: ${SEED}"
echo "Action steps: ${N_ACTION_STEPS} | episode length: ${EPISODE_LENGTH}"
if [[ -n "${DETERMINISTIC_TRACK}" ]]; then
    echo "Deterministic track: ${DETERMINISTIC_TRACK} | seed offset: ${TRACK_SEED_OFFSET}"
fi
echo "Results: ${OUTPUT_DIR}"

exec env \
    VLABENCH_ROOT="${RUNTIME_ROOT}/sim/VLABench/VLABench" \
    HF_HOME="${HF_CACHE}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
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
        --policy.path="${SMOLVLA_CHECKPOINT}" \
        --policy.device=cuda \
        --policy.n_action_steps="${N_ACTION_STEPS}" \
        --env.type=vlabench \
        --env.task="${VLABENCH_TASK}" \
        --env.episode_length="${EPISODE_LENGTH}" \
        --env.render_resolution="${RENDER_RESOLUTION}" \
        --env.max_parallel_tasks=1 \
        --eval.batch_size=1 \
        --eval.n_episodes="${N_EPISODES}" \
        --eval.use_async_envs=false \
        --seed="${SEED}" \
        '--rename_map={"observation.images.image":"observation.images.camera1","observation.images.second_image":"observation.images.camera2","observation.images.wrist_image":"observation.images.camera3"}' \
        "${TRACK_ARGS[@]}" \
        "$@"
