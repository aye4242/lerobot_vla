#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run a pretrained LeRobot policy in VLABench's interactive MuJoCo viewer.

The regular ``lerobot-eval`` command renders offscreen and writes an MP4. This
script keeps the same observation, policy processor, and EEF-to-joint control
path while using either the native MuJoCo Viewer or ``dm_control.viewer``.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.envs.configs import VLABenchEnv as VLABenchEnvConfig
from lerobot.envs.utils import preprocess_observation
from lerobot.envs.vlabench import VLABenchEnv, _extract_task_description
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.random_utils import set_seed

SMOLVLA_CAMERA_RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.second_image": "observation.images.camera2",
    "observation.images.wrist_image": "observation.images.camera3",
}

# ``lerobot/pi0_base`` was trained with the multi-embodiment camera names below.
# This is an interface smoke-test mapping for VLABench, not a claim that the
# base checkpoint has VLABench-calibrated camera semantics. A Pi0 checkpoint
# fine-tuned on VLABench should normally use identity or an explicit --camera-map.
PI0_BASE_CAMERA_RENAME_MAP = {
    "observation.images.image": "observation.images.base_0_rgb",
    "observation.images.second_image": "observation.images.right_wrist_0_rgb",
    "observation.images.wrist_image": "observation.images.left_wrist_0_rgb",
}

# Kept as a compatibility alias for existing scripts importing this constant.
CAMERA_RENAME_MAP = SMOLVLA_CAMERA_RENAME_MAP


def _parse_camera_map(value: str | None) -> dict[str, str] | None:
    """Parse an optional JSON source-to-policy camera feature mapping."""
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("--camera-map must be a JSON object") from exc
    if not isinstance(parsed, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()):
        raise ValueError("--camera-map must be a JSON object with string keys and values")
    return parsed


def _resolve_camera_rename_map(policy_cfg: PreTrainedConfig, explicit_map: str | None) -> dict[str, str]:
    """Choose a camera adapter without changing the model-independent environment."""
    parsed_map = _parse_camera_map(explicit_map)
    if parsed_map is not None:
        return parsed_map

    policy_type = getattr(policy_cfg, "type", "").lower()
    if policy_type == "smolvla":
        return SMOLVLA_CAMERA_RENAME_MAP

    visual_features = {
        key.lower()
        for key, feature in getattr(policy_cfg, "input_features", {}).items()
        if getattr(feature, "type", None) is not None and str(feature.type).lower().endswith("visual")
    }
    canonical = {
        "observation.images.image",
        "observation.images.second_image",
        "observation.images.wrist_image",
    }
    if canonical.issubset(visual_features):
        return {}

    if policy_type in {"pi0", "pi05"}:
        targets = {key.lower(): key for key in getattr(policy_cfg, "input_features", {})}
        base = next((key for key in targets if "base_0_rgb" in key or "front" in key), None)
        right = next((key for key in targets if "right_wrist" in key), None)
        left = next((key for key in targets if "left_wrist" in key), None)
        if base and right and left:
            return {
                "observation.images.image": targets[base],
                "observation.images.second_image": targets[right],
                "observation.images.wrist_image": targets[left],
            }

    raise ValueError(
        f"Policy '{policy_type}' expects visual features {sorted(visual_features)}, but VLABench exposes "
        "image/second_image/wrist_image. Pass --camera-map with a JSON source-to-policy mapping."
    )


@dataclass(frozen=True)
class PhysicsOverrides:
    timestep: float | None = None
    solver: str | None = None
    integrator: str | None = None
    iterations: int | None = None
    tolerance: float | None = None
    gravity_z: float | None = None

    def apply(self, physics: Any) -> None:
        import mujoco

        opt = physics.model.opt
        if self.timestep is not None:
            opt.timestep = self.timestep
        if self.solver is not None:
            opt.solver = {
                "pgs": mujoco.mjtSolver.mjSOL_PGS,
                "cg": mujoco.mjtSolver.mjSOL_CG,
                "newton": mujoco.mjtSolver.mjSOL_NEWTON,
            }[self.solver]
        if self.integrator is not None:
            opt.integrator = {
                "euler": mujoco.mjtIntegrator.mjINT_EULER,
                "rk4": mujoco.mjtIntegrator.mjINT_RK4,
                "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
                "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
            }[self.integrator]
        if self.iterations is not None:
            opt.iterations = self.iterations
        if self.tolerance is not None:
            opt.tolerance = self.tolerance
        if self.gravity_z is not None:
            opt.gravity[2] = self.gravity_z

        print(
            "MuJoCo options: "
            f"timestep={opt.timestep:g}, solver={int(opt.solver)}, integrator={int(opt.integrator)}, "
            f"iterations={opt.iterations}, tolerance={opt.tolerance:g}, gravity={list(opt.gravity)}",
            flush=True,
        )


class NativeViewerControls:
    """Keyboard state shared with MuJoCo's passive Viewer."""

    # GLFW key codes. Keeping the values local avoids requiring the optional
    # Python glfw package solely for three constants.
    KEY_SPACE = 32
    KEY_BACKSPACE = 259
    KEY_RIGHT = 262

    def __init__(self) -> None:
        self.paused = True
        self.reset_requested = False
        self.single_step_requested = False

    def key_callback(self, keycode: int) -> None:
        if keycode == self.KEY_SPACE:
            self.paused = not self.paused
        elif keycode == self.KEY_BACKSPACE:
            self.reset_requested = True
        elif keycode == self.KEY_RIGHT and self.paused:
            self.single_step_requested = True


class VLABenchViewerPolicy:
    """Adapt LeRobot policy inference to a dm_control Viewer callback."""

    def __init__(
        self,
        env: VLABenchEnv,
        policy: Any,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]],
        postprocessor: Callable[[Any], Any],
        physics_overrides: PhysicsOverrides | None = None,
        log_actions_every: int = 0,
        policy_name: str = "policy",
        observation_preprocessor: Callable[[dict[str, Any]], dict[str, Any]] = preprocess_observation,
    ) -> None:
        self.env = env
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.physics_overrides = physics_overrides or PhysicsOverrides()
        self.log_actions_every = log_actions_every
        self.policy_name = policy_name
        self.observation_preprocessor = observation_preprocessor
        self._episode_started = False
        self._action_count = 0

    def reset(self) -> None:
        """Clear queued actions when the Viewer resets the environment."""
        self.policy.reset()
        self._episode_started = False
        self._action_count = 0

    def _refresh_episode(self) -> None:
        assert self.env._env is not None
        self.env.task_description = _extract_task_description(self.env._env.task, self.env.task)
        self.physics_overrides.apply(self.env._env.physics)
        print(f"VLABench instruction: {self.env.task_description}", flush=True)
        self._episode_started = True

    def __call__(self, timestep: Any) -> np.ndarray:
        # A dm_control environment can reset itself at episode boundaries, without
        # going through the Gym wrapper. Clear the policy's action queue in that case.
        if timestep.first():
            if self._episode_started:
                self.policy.reset()
            self._refresh_episode()
        elif not self._episode_started:
            self._refresh_episode()

        raw_observation = self.env._get_obs()
        current_eef = np.asarray(raw_observation.get("agent_pos", []), dtype=np.float64).ravel()
        observation = self.observation_preprocessor(raw_observation)
        observation["task"] = [self.env.task_description]
        observation = self.preprocessor(observation)

        device_type = torch.device(self.policy.config.device).type
        amp_context = torch.autocast(device_type=device_type) if self.policy.config.use_amp else nullcontext()
        inference_started = time.perf_counter()
        with torch.inference_mode(), amp_context:
            action = self.policy.select_action(observation)
        inference_s = time.perf_counter() - inference_started
        action = self.postprocessor(action)
        action_numpy = np.asarray(action.detach().to("cpu").numpy()).squeeze(0)
        if action_numpy.ndim != 1:
            raise ValueError(f"Expected one policy action, got shape {action_numpy.shape}")

        assert self.env._env is not None
        ctrl_dim = int(self.env._env.physics.data.ctrl.shape[0])
        ctrl = self.env._build_ctrl_from_action(action_numpy, ctrl_dim)
        self._action_count += 1
        if self.log_actions_every and self._action_count % self.log_actions_every == 0:
            position_delta = (
                np.round(action_numpy[:3] - current_eef[:3], 4).tolist()
                if current_eef.size >= 3
                else None
            )
            print(
                f"{self.policy_name} action #{self._action_count} | inference={inference_s:.3f}s | "
                f"current_eef={np.round(current_eef, 4).tolist()} | "
                f"target_eef={np.round(action_numpy, 4).tolist()} | delta_pos={position_delta} | "
                f"ctrl={np.round(ctrl, 4).tolist()}",
                flush=True,
            )
        return ctrl


# Backward-compatible name used by the original SmolVLA tests and scripts.
SmolVLAViewerPolicy = VLABenchViewerPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="select_toy", help="VLABench task name")
    parser.add_argument(
        "--policy-path",
        default="lerobot/smolvla_vlabench",
        help="Local path or Hub id of the pretrained policy (defaults to the SmolVLA VLABench checkpoint)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--camera-map",
        default=None,
        help=(
            "Optional JSON mapping from VLABench image keys to policy image keys, "
            "for example '{\"observation.images.image\": \"observation.images.base_0_rgb\"}'"
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Optional local tokenizer path override (useful for gated PaliGemma tokenizer files)",
    )
    parser.add_argument(
        "--viewer",
        choices=("native", "dm-control"),
        default="dm-control",
        help="dm-control is the stable VLABench Viewer; native is experimental on MuJoCo 3.2.2",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Actions executed before replanning (default: preserve the checkpoint value)",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--render-resolution", type=int, default=256)
    parser.add_argument("--window-width", type=int, default=1024)
    parser.add_argument("--window-height", type=int, default=768)
    parser.add_argument(
        "--headless-steps",
        type=int,
        default=0,
        help="Run this many policy/environment steps without opening a Viewer, then exit",
    )
    parser.add_argument("--timestep", type=float, help="Override MuJoCo physics timestep")
    parser.add_argument("--solver", choices=("pgs", "cg", "newton"))
    parser.add_argument("--integrator", choices=("euler", "rk4", "implicit", "implicitfast"))
    parser.add_argument("--iterations", type=int, help="Override solver iterations")
    parser.add_argument("--tolerance", type=float, help="Override solver tolerance")
    parser.add_argument("--gravity-z", type=float, help="Override vertical gravity (default is usually -9.81)")
    parser.add_argument(
        "--log-actions-every",
        type=int,
        default=0,
        help="Print predicted EEF actions, joint controls, and inference time every N actions (0 disables)",
    )
    return parser.parse_args()


def launch_dm_control_viewer(
    env: VLABenchEnv,
    viewer_policy: VLABenchViewerPolicy,
    task: str,
    width: int,
    height: int,
) -> None:
    from dm_control import viewer

    assert env._env is not None
    print(
        "dm_control Viewer starts paused: Space=run/pause, Backspace=reset, "
        "F1=help, [ or ]=camera, mouse=orbit/pan/zoom",
        flush=True,
    )
    viewer.launch(
        env._env,
        policy=viewer_policy,
        title=f"{viewer_policy.policy_name} - {task}",
        width=width,
        height=height,
    )


def launch_native_viewer(env: VLABenchEnv, viewer_policy: VLABenchViewerPolicy) -> None:
    """Run the dm_control environment while displaying its live native MuJoCo data."""
    import mujoco.viewer

    assert env._env is not None
    controls = NativeViewerControls()

    while True:
        timestep = env._env.reset()
        viewer_policy.reset()
        controls.paused = True
        controls.reset_requested = False
        controls.single_step_requested = False

        # VLABench may rebuild Physics during reset. Launch after every reset so
        # the native Viewer always owns the current model and data pointers.
        physics = env._env.physics
        with mujoco.viewer.launch_passive(
            physics.model.ptr,
            physics.data.ptr,
            key_callback=controls.key_callback,
            show_left_ui=True,
            show_right_ui=True,
        ) as handle:
            print(
                "Native MuJoCo Viewer starts paused: Space=run/pause, Right=one policy step, "
                "Backspace=reset/reopen, use the side panels to inspect or edit simulation parameters",
                flush=True,
            )
            while handle.is_running() and not controls.reset_requested:
                should_step = not controls.paused or controls.single_step_requested
                if should_step:
                    controls.single_step_requested = False
                    ctrl = viewer_policy(timestep)
                    timestep = env._env.step(ctrl)
                    if timestep.last():
                        controls.paused = True
                        print("Episode terminated. Press Backspace to reset.", flush=True)
                handle.sync()
                if not should_step:
                    time.sleep(0.01)

            window_closed = not handle.is_running()

        if window_closed:
            return
        print(
            "Resetting VLABench; the native Viewer window will reopen with the new Physics state.",
            flush=True,
        )


def run_headless(env: VLABenchEnv, viewer_policy: VLABenchViewerPolicy, steps: int) -> None:
    """Execute a bounded policy smoke test without requiring a desktop Viewer."""

    assert env._env is not None
    timestep = env._env.reset()
    viewer_policy.reset()
    for step in range(1, steps + 1):
        ctrl = viewer_policy(timestep)
        timestep = env._env.step(ctrl)
        print(f"Headless VLABench step {step}/{steps} completed", flush=True)
        if timestep.last():
            print(f"Episode terminated after {step} headless steps", flush=True)
            break


def main() -> None:
    args = parse_args()
    if args.viewer == "native":
        raise SystemExit(
            "The native MuJoCo Viewer is incompatible with the current VLABench model on MuJoCo 3.2.2 "
            "(mjv_makeSceneState failure). Use '--viewer dm-control'."
        )
    if args.n_action_steps is not None and args.n_action_steps <= 0:
        raise ValueError("--n-action-steps must be positive")
    if args.timestep is not None and args.timestep <= 0:
        raise ValueError("--timestep must be positive")
    if args.iterations is not None and args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.tolerance is not None and args.tolerance < 0:
        raise ValueError("--tolerance must be non-negative")
    if args.log_actions_every < 0:
        raise ValueError("--log-actions-every must be non-negative")
    if args.headless_steps < 0:
        raise ValueError("--headless-steps must be non-negative")

    # VLABench currently has one malformed logging call. Suppress logging's
    # internal formatting traceback while retaining normal project logs.
    logging.raiseExceptions = False
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    set_seed(args.seed)

    env_cfg = VLABenchEnvConfig(
        task=args.task,
        render_resolution=(args.render_resolution, args.render_resolution),
    )
    env = VLABenchEnv(
        task=args.task,
        render_resolution=(args.render_resolution, args.render_resolution),
    )

    policy_overrides = [f"--device={args.device}"]
    if args.n_action_steps is not None:
        policy_overrides.append(f"--n_action_steps={args.n_action_steps}")
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path, cli_overrides=policy_overrides)
    policy_cfg.pretrained_path = Path(args.policy_path)
    camera_rename_map = _resolve_camera_rename_map(policy_cfg, args.camera_map)
    print(
        f"Policy type: {policy_cfg.type} | camera rename map: {camera_rename_map or 'identity'}",
        flush=True,
    )

    print(
        f"Loading policy: {args.policy_path} | n_action_steps={policy_cfg.n_action_steps}",
        flush=True,
    )
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=camera_rename_map)
    policy.eval()
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": camera_rename_map},
    }
    if args.tokenizer_path is not None:
        preprocessor_overrides["tokenizer_processor"] = {"tokenizer_name": args.tokenizer_path}

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
        pretrained_revision=policy_cfg.pretrained_revision,
        preprocessor_overrides=preprocessor_overrides,
    )

    print(f"Loading VLABench task: {args.task}", flush=True)
    env._ensure_env()
    assert env._env is not None
    env._seed_inner_env(args.seed)
    viewer_policy = VLABenchViewerPolicy(
        env,
        policy,
        preprocessor,
        postprocessor,
        physics_overrides=PhysicsOverrides(
            timestep=args.timestep,
            solver=args.solver,
            integrator=args.integrator,
            iterations=args.iterations,
            tolerance=args.tolerance,
            gravity_z=args.gravity_z,
        ),
        log_actions_every=args.log_actions_every,
        policy_name=policy_cfg.type,
    )
    viewer_policy.physics_overrides.apply(env._env.physics)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        if args.headless_steps:
            run_headless(env, viewer_policy, args.headless_steps)
        elif args.viewer == "native":
            launch_native_viewer(env, viewer_policy)
        else:
            launch_dm_control_viewer(
                env,
                viewer_policy,
                task=args.task,
                width=args.window_width,
                height=args.window_height,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
