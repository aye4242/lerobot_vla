#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_ROOT="${LEROBOT_RUNTIME_ROOT:-/extdata/hdd2/lerobot-smolvla}"
HF_CACHE="${HF_HOME:-${RUNTIME_ROOT}/hf-cache}"
PYTHON="${RUNTIME_ROOT}/.venv/bin/python"
PI0_CHECKPOINT="${PI0_CHECKPOINT:-${HF_CACHE}/pi0_base}"
PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-${HF_CACHE}/paligemma-tokenizer}"

if [[ ! -f "${PI0_CHECKPOINT}/model.safetensors" ]]; then
    echo "Pi0 checkpoint not found: ${PI0_CHECKPOINT}/model.safetensors" >&2
    exit 1
fi
if [[ ! -f "${PALIGEMMA_TOKENIZER}/tokenizer.json" ]]; then
    echo "PaliGemma tokenizer not found: ${PALIGEMMA_TOKENIZER}/tokenizer.json" >&2
    exit 1
fi

cd "${REPO_ROOT}"
echo "Launching policy=pi0 checkpoint=${PI0_CHECKPOINT}"

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
    "${PYTHON}" examples/vlabench/run_vlabench_viewer.py \
        --viewer dm-control \
        --task select_toy \
        --policy-path "${PI0_CHECKPOINT}" \
        --tokenizer-path "${PALIGEMMA_TOKENIZER}" \
        --device cuda \
        --n-action-steps 50 \
        --render-resolution 256 \
        --window-width 960 \
        --window-height 720 \
        --timestep 0.001 \
        --solver newton \
        --integrator implicitfast \
        --iterations 100 \
        --tolerance 1e-8 \
        --gravity-z -9.81 \
        --log-actions-every 1 \
        "$@"
