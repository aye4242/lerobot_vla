#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

LIBERO_SUITE="${LIBERO_SUITE:-libero_object}"
LIBERO_TASK_IDS="${LIBERO_TASK_IDS:-[0]}"
N_EPISODES="${N_EPISODES:-5}"
SEED="${SEED:-1000}"
PI0_ACTION_STEPS="${PI0_ACTION_STEPS:-50}"
PI05_ACTION_STEPS="${PI05_ACTION_STEPS:-10}"
PI0_OUTPUT_DIR="${PI0_OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/pi0_${LIBERO_SUITE}_compare_n${N_EPISODES}}"
PI05_OUTPUT_DIR="${PI05_OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/pi05_${LIBERO_SUITE}_compare_n${N_EPISODES}}"

cd "${REPO_ROOT}"

echo "Pi0 vs Pi0.5 LIBERO comparison"
echo "Suite: ${LIBERO_SUITE} | task ids: ${LIBERO_TASK_IDS} | episodes: ${N_EPISODES} | seed: ${SEED}"
echo "Pi0 action steps: ${PI0_ACTION_STEPS} | Pi0.5 action steps: ${PI05_ACTION_STEPS}"
echo "Pi0 output: ${PI0_OUTPUT_DIR}"
echo "Pi0.5 output: ${PI05_OUTPUT_DIR}"

LIBERO_SUITE="${LIBERO_SUITE}" \
LIBERO_TASK_IDS="${LIBERO_TASK_IDS}" \
N_EPISODES="${N_EPISODES}" \
OUTPUT_DIR="${PI0_OUTPUT_DIR}" \
bash ./examples/libero/run_pi0_libero_eval.sh \
    --seed="${SEED}" \
    --policy.n_action_steps="${PI0_ACTION_STEPS}"

LIBERO_SUITE="${LIBERO_SUITE}" \
LIBERO_TASK_IDS="${LIBERO_TASK_IDS}" \
N_EPISODES="${N_EPISODES}" \
OUTPUT_DIR="${PI05_OUTPUT_DIR}" \
bash ./examples/libero/run_pi05_libero_eval.sh \
    --seed="${SEED}" \
    --policy.n_action_steps="${PI05_ACTION_STEPS}"

echo "Comparison complete. Inspect both eval_info.json files under the output directories above."
