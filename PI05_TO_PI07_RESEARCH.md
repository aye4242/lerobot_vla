# PI0.5 to PI0.7 Incremental Research Log

This document records the first incremental experiments based on the local
`pi05.pdf` and `pi0.7.pdf` papers. The goal is to move toward the pi0.7
capabilities one controlled change at a time, not to claim a full pi0.7
implementation.

## Paper Findings

The important pi0.7 changes relative to pi0.5 are:

1. Richer context: task, subtask, subgoal images, episode speed, episode
   quality, mistake labels, and control mode.
2. History-aware observations: MEM-style temporal and spatial compression,
   with up to six history frames and continuous history state embeddings.
3. Broader data: high-quality and low-quality demonstrations, failures,
   autonomous policy rollouts, RL data, human videos, and web data.
4. Larger model: Gemma3 4B VLM and an 860M action expert, approximately 5B
   parameters in total.
5. Runtime additions: generated subgoal images, asynchronous world-model
   updates, training-time RTC, and optional metadata classifier-free guidance.

The pi0.7 paper states that richer context is what makes mixed-quality data
usable. It does not present the change as a simple backbone-size upgrade.

## Implemented Increment

The repository now has an opt-in pi0.7-style episode metadata path:

- `src/lerobot/datasets/pi05_metadata.py` loads and validates a JSON sidecar
  keyed by episode index.
- `src/lerobot/datasets/factory.py` wraps map-style train/eval datasets when
  metadata is enabled.
- `src/lerobot/processor/converters.py` preserves metadata through
  `batch_to_transition`.
- `src/lerobot/policies/pi05/processor_pi05.py` renders `Speed`, `Quality`,
  `Mistake`, and `Control Mode` into the prompt.
- `src/lerobot/policies/pi05/modeling_pi05.py` and the policy factory preserve
  MEM projection and processor settings when loading old pi0.5 checkpoints.

The metadata path is disabled by default, so existing pi0.5 prompt behavior
and checkpoint loading remain the default.

## MEM Ablation

The reproducible training entry point is:

`examples/libero/run_pi05_mem_ablation_train.sh`

The short runs used the readable local 20 Hz dataset
`data/datasets/pi05_three_objects_basket`, so the history stride was 20 frames
per second of wall-clock history. All variants used:

- the same pi0.5 checkpoint: `data/models/pi05_libero_finetuned`;
- the same dataset, seed, optimizer, batch size, and 1000 training steps;
- `train_expert_only=true`, `freeze_vision_encoder=true`, and batch size 1;
- `memory_frames=6` and `memory_stride=20`;
- the same 300M action expert and 2B PaliGemma backbone.

| Variant | Visual memory | Proprioceptive memory | Final train loss | 16-sample offline loss | Mean forward time |
|---|---:|---:|---:|---:|---:|
| Base | no | no | 0.318 | 0.4152 | 0.170 s |
| Visual-MEM | yes | no | 0.320 | 0.4219 | 0.328 s |
| Full-MEM | yes | yes | 0.377 | 0.4430 | 0.340 s |

The offline evaluation used the same 16 evenly spaced dataset indices and the
same per-sample random seeds. It is a flow-matching loss check, not a robot
success-rate evaluation.

## Current Decision

The result does not justify enabling MEM by default:

- all three variants train and save valid checkpoints;
- Visual-MEM adds roughly 1.9x forward cost in this measurement without a
  lower offline loss;
- Full-MEM adds a new randomly initialized state projection when starting from
  the old checkpoint and is therefore not fairly judged by a 1000-step run;
- no LIBERO success-rate comparison was run in this workspace because the
  local environment lacks `libero`, `robosuite`, and `mujoco`.

The next research step should therefore be a longer, properly controlled
experiment rather than a claim that MEM improved the policy.

## Next Experiments

1. Run the same three variants for a longer budget, preferably 3k-6k steps,
   with at least three random seeds if GPU time permits.
2. Evaluate each checkpoint in the same LIBERO runtime and report success,
   progress, completion time, and inference latency.
3. Build a real mixed-quality LeRobot dataset from successful, failed, and
   old-policy rollouts. Do not infer episode labels from unrelated debug video
   indices.
4. Enable the metadata prompt on that dataset and compare mixed-quality data
   with and without metadata.
5. Test real future frames as subgoal images before introducing a 14B BAGEL
   world model.

The migration order remains:

`pi05 base -> MEM validation -> mixed-quality data + metadata -> real subgoal
images -> generated subgoal images -> larger action expert -> Gemma3 4B`.
