#!/usr/bin/env bash

set -euo pipefail

# Train one controlled PI05 memory variant on the local 10 Hz replay dataset.
# The three variants are selected with MEM_MODE=base|visual|full.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_ROOT="${PI05_MEM_DATASET_ROOT:-${REPO_ROOT}/data/datasets/pi05_three_objects_replay_10hz_cream_first}"
DATASET_REPO_ID="${PI05_MEM_DATASET_REPO_ID:-local/pi05_three_objects_replay_10hz_cream_first}"
CHECKPOINT="${PI05_MEM_CHECKPOINT:-${REPO_ROOT}/data/models/pi05_libero_finetuned}"
TOKENIZER_PATH="${PI05_MEM_TOKENIZER_PATH:-${REPO_ROOT}/data/models/paligemma-tokenizer}"
OUTPUT_ROOT="${PI05_MEM_OUTPUT_ROOT:-${REPO_ROOT}/outputs/train}"
MEM_MODE="${MEM_MODE:-base}"
STEPS="${PI05_MEM_STEPS:-1000}"
BATCH_SIZE="${PI05_MEM_BATCH_SIZE:-1}"
NUM_WORKERS="${PI05_MEM_NUM_WORKERS:-4}"
MEMORY_STRIDE="${PI05_MEM_STRIDE:-10}"

case "${MEM_MODE}" in
    base)
        MEMORY_FLAGS=(
            --policy.use_visual_memory=false
            --policy.use_proprioceptive_memory=false
        )
        ;;
    visual)
        MEMORY_FLAGS=(
            --policy.use_visual_memory=true
            --policy.use_proprioceptive_memory=false
        )
        ;;
    full)
        MEMORY_FLAGS=(
            --policy.use_visual_memory=true
            --policy.use_proprioceptive_memory=true
        )
        ;;
    *)
        echo "MEM_MODE must be base, visual, or full; got: ${MEM_MODE}" >&2
        exit 2
        ;;
esac

if [[ ! -f "${CHECKPOINT}/model.safetensors" ]]; then
    echo "PI05 checkpoint not found: ${CHECKPOINT}/model.safetensors" >&2
    exit 1
fi
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
    echo "Dataset metadata not found: ${DATASET_ROOT}/meta/info.json" >&2
    exit 1
if [[ ! -f "${TOKENIZER_PATH}/tokenizer.json" ]]; then
    echo "PaliGemma tokenizer not found: ${TOKENIZER_PATH}/tokenizer.json" >&2
    exit 1
fi
fi

cd "${REPO_ROOT}"
echo "MEM_MODE=${MEM_MODE}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "DATASET_REPO_ID=${DATASET_REPO_ID}"
echo "STEPS=${STEPS} BATCH_SIZE=${BATCH_SIZE}"

echo "MEMORY_STRIDE=${MEMORY_STRIDE}"
exec .venv/bin/lerobot-train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --policy.type=pi05 \
    --policy.pretrained_path="${CHECKPOINT}" \
    --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}' \
    --policy.n_action_steps=10 \
    --policy.empty_cameras=1 \
    --policy.gradient_checkpointing=true \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --tokenizer_path="${TOKENIZER_PATH}" \
    --policy.push_to_hub=false \
    --policy.memory_frames=6 \
    --policy.memory_stride="${MEMORY_STRIDE}" \
    "${MEMORY_FLAGS[@]}" \
    --output_dir="${OUTPUT_ROOT}/pi05_mem_${MEM_MODE}" \
    --job_name="pi05_mem_${MEM_MODE}" \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --steps="${STEPS}" \
    --save_freq="${STEPS}" \
    --seed=1000
