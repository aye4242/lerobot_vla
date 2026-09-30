#!/usr/bin/env python

"""Record clean scripted demonstrations for the custom three-object LIBERO task."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from libero.libero import benchmark
from robosuite.utils.transform_utils import quat2axisangle

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.libero import LiberoEnv
from lerobot.utils.io_utils import write_video


TASK = "put the cream cheese box, butter, and alphabet soup in the basket, one at a time"
OBJECTS = ("alphabet_soup_1", "butter_1", "cream_cheese_1")
# Keep the language and all expert action trajectories in the same order.  The
# alphabetical order above is retained for stable reporting of per-object
# metrics, while this order is used only to generate demonstrations.
DEMONSTRATION_ORDER = ("cream_cheese_1", "butter_1", "alphabet_soup_1")
SUITE = "libero_10"
TASK_ID = 0

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OracleConfig:
    position_scale: float = 0.04
    max_translation: float = 0.85
    position_tolerance: float = 0.012
    stable_steps: int = 4
    transit_z: float = 0.68
    grasp_z_offset: float = 0.005
    cream_grasp_z: float = 0.44
    lift_z: float = 0.64
    # The Panda grip site cannot descend below roughly z=0.54 over the basket
    # without colliding with its rim. Release just above that limit and let the
    # object settle into the contain region.
    release_z_offset: float = 0.18
    phase_timeout: int = 100
    close_steps: int = 18
    open_steps: int = 16
    settle_steps: int = 30


class OracleFailure(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/datasets/pi05_three_objects_basket"))
    parser.add_argument("--repo-id", default="local/pi05_three_objects_basket")
    parser.add_argument("--num-successful", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=60)
    parser.add_argument("--start-init-state", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--controller", choices=("hybrid", "policy", "scripted"), default="hybrid")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/models/pi05_libero_finetuned"),
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path("data/models/paligemma-tokenizer"),
    )
    parser.add_argument("--subtask-max-steps", type=int, default=240)
    parser.add_argument(
        "--use-pruned-init-states",
        action="store_true",
        help=(
            "Use the installed fixed states. They are unsuitable if copied from the original "
            "two-object task."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Run the oracle without creating a dataset.")
    parser.add_argument("--save-debug-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Append until --num-successful total episodes.")
    return parser.parse_args()


def dataset_features() -> dict:
    return {
        "observation.images.image": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.image2": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.success": {"dtype": "bool", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }


def make_env(init_state: int, fps: int, max_steps: int, use_pruned_init_states: bool = False) -> LiberoEnv:
    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(TASK_ID)
    if task.language != TASK:
        raise RuntimeError(
            f"{SUITE} task {TASK_ID} is not the installed custom task. Got: {task.language!r}. "
            "Install the custom BDDL, init-state file, and benchmark entry in the container first."
        )
    return LiberoEnv(
        task_suite=suite,
        task_id=TASK_ID,
        task_suite_name=SUITE,
        episode_length=max_steps,
        camera_name="agentview_image,robot0_eye_in_hand_image",
        obs_type="pixels_agent_pos",
        observation_width=256,
        observation_height=256,
        init_states=use_pruned_init_states,
        episode_index=init_state,
        n_envs=1,
        num_steps_wait=10,
        control_freq=fps,
        control_mode="relative",
        hard_reset=True,
        auto_reset_on_termination=False,
        terminate_on_success=False,
    )


def processed_frame(observation: dict, action: np.ndarray, reward: float, success: bool, done: bool) -> dict:
    state = observation["robot_state"]
    axis_angle = quat2axisangle(np.asarray(state["eef"]["quat"], dtype=np.float64))
    robot_state = np.concatenate(
        (
            np.asarray(state["eef"]["pos"], dtype=np.float32),
            np.asarray(axis_angle, dtype=np.float32),
            np.asarray(state["gripper"]["qpos"], dtype=np.float32),
        )
    )
    return {
        # LiberoProcessorStep applies the same 180-degree correction during evaluation.
        "observation.images.image": np.ascontiguousarray(observation["pixels"]["image"][::-1, ::-1]),
        "observation.images.image2": np.ascontiguousarray(observation["pixels"]["image2"][::-1, ::-1]),
        "observation.state": robot_state,
        "action": np.asarray(action, dtype=np.float32),
        "next.reward": np.asarray([reward], dtype=np.float32),
        "next.success": np.asarray([success], dtype=bool),
        "next.done": np.asarray([done], dtype=bool),
        "task": TASK,
    }


class ScriptedRecorder:
    def __init__(self, env: LiberoEnv, config: OracleConfig, dataset: LeRobotDataset | None):
        self.env = env
        self.config = config
        self.dataset = dataset
        self.observation: dict = {}
        self.frames: list[np.ndarray] = []
        self.steps = 0
        self.last_info: dict = {}

    @property
    def inner_env(self):
        assert self.env._env is not None
        return self.env._env.env

    def object_position(self, name: str) -> np.ndarray:
        raw = self.inner_env._get_observations()
        return np.asarray(raw[f"{name}_pos"], dtype=np.float64)

    def eef_position(self) -> np.ndarray:
        return np.asarray(self.observation["robot_state"]["eef"]["pos"], dtype=np.float64)

    def eef_axis_angle(self) -> np.ndarray:
        return quat2axisangle(
            np.asarray(self.observation["robot_state"]["eef"]["quat"], dtype=np.float64)
        )

    def step(self, action: np.ndarray) -> None:
        if self.steps >= self.env._max_episode_steps:
            raise OracleFailure(f"episode exceeded {self.env._max_episode_steps} steps")
        before = self.observation
        next_observation, reward, terminated, truncated, info = self.env.step(action)
        success = bool(info["is_success"])
        done = bool(terminated or truncated)
        if self.dataset is not None:
            self.dataset.add_frame(processed_frame(before, action, reward, success, done))
        self.frames.append(np.ascontiguousarray(before["pixels"]["image"][::-1, ::-1]))
        self.observation = next_observation
        self.last_info = info
        self.steps += 1

    def hold(self, count: int, gripper: float) -> None:
        action = np.zeros(7, dtype=np.float32)
        action[6] = gripper
        for _ in range(count):
            self.step(action)

    def rotate_gripper_z(self, increments: int, gripper: float) -> None:
        """Rotate around the tool z axis in converged 0.5 rad increments."""
        for _ in range(abs(increments)):
            action = np.zeros(7, dtype=np.float32)
            action[5] = float(np.sign(increments))
            action[6] = gripper
            self.step(action)
            # A zero rotational command preserves the new target orientation
            # while allowing the OSC controller to converge to it.
            self.hold(18, gripper)

    def move_to(self, target: np.ndarray, gripper: float, label: str) -> None:
        stable = 0
        for _ in range(self.config.phase_timeout):
            error = np.asarray(target, dtype=np.float64) - self.eef_position()
            if np.linalg.norm(error) <= self.config.position_tolerance:
                stable += 1
                if stable >= self.config.stable_steps:
                    return
            else:
                stable = 0
            action = np.zeros(7, dtype=np.float32)
            action[:3] = np.clip(
                error / self.config.position_scale,
                -self.config.max_translation,
                self.config.max_translation,
            )
            action[6] = gripper
            self.step(action)
        raise OracleFailure(
            f"{label} timed out: target={target.round(4)}, eef={self.eef_position().round(4)}"
        )

    def insert_final_object(self, object_name: str, basket: np.ndarray) -> None:
        """Contact-search free basket space and stop at the first native In success."""
        search_offsets = (
            np.array([0.0, -0.04]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.025]),
            np.array([-0.055, 0.01]),
            np.array([0.055, 0.01]),
        )
        for offset in search_offsets:
            target = np.array([basket[0] + offset[0], basket[1] + offset[1], basket[2] + 0.10])
            for _ in range(45):
                error = target - self.eef_position()
                action = np.zeros(7, dtype=np.float32)
                action[:3] = np.clip(
                    error / self.config.position_scale,
                    -self.config.max_translation,
                    self.config.max_translation,
                )
                action[6] = 1.0
                self.step(action)
                if self.env.is_object_in_basket(object_name):
                    self.hold(self.config.open_steps + self.config.settle_steps, -1.0)
                    if self.env.is_object_in_basket(object_name):
                        logger.info("Placed %s in basket at step %d", object_name, self.steps)
                        return
        raise OracleFailure(
            f"contact search could not put {object_name} in basket; "
            f"eef={self.eef_position().round(4)}, object={self.object_position(object_name).round(4)}"
        )

    def pick_and_place(self, object_name: str, basket_offset: np.ndarray) -> None:
        cfg = self.config
        object_pos = self.object_position(object_name)
        logger.info("Picking %s at %s", object_name, object_pos.round(4))

        above_object = np.array([object_pos[0], object_pos[1], cfg.transit_z])
        self.move_to(above_object, -1.0, f"move above {object_name}")
        if object_name == "cream_cheese_1":
            # Its narrow upright profile is not graspable with the initial
            # finger axis. A ~90-degree yaw also lets the fingertips descend
            # beside the box instead of colliding with its top.
            self.rotate_gripper_z(3, -1.0)
            # Its narrow upright profile needs a lower side grasp after the
            # yaw rotation; successful trajectories are retained only after
            # the subsequent lifted-object verification.
            grasp_z = cfg.cream_grasp_z
        else:
            grasp_z = max(object_pos[2] + cfg.grasp_z_offset, 0.515)
        grasp = np.array([object_pos[0], object_pos[1], grasp_z])
        self.move_to(grasp, -1.0, f"descend to {object_name}")
        self.hold(cfg.close_steps, 1.0)

        lift = np.array([grasp[0], grasp[1], cfg.lift_z])
        self.move_to(lift, 1.0, f"lift {object_name}")
        lifted_pos = self.object_position(object_name)
        if (
            lifted_pos[2] < object_pos[2] + 0.06
            or np.linalg.norm(lifted_pos[:2] - self.eef_position()[:2]) > 0.08
        ):
            raise OracleFailure(
                f"failed to grasp {object_name}: before={object_pos.round(4)}, after={lifted_pos.round(4)}"
            )

        basket = self.object_position("basket_1")
        above_basket = np.array([basket[0] + basket_offset[0], basket[1] + basket_offset[1], cfg.transit_z])
        release_z_offset = 0.11 if object_name == "cream_cheese_1" else cfg.release_z_offset
        release = np.array([above_basket[0], above_basket[1], basket[2] + release_z_offset])
        self.move_to(above_basket, 1.0, f"move {object_name} above basket")
        if object_name == "cream_cheese_1":
            self.insert_final_object(object_name, basket)
            return
        try:
            self.move_to(release, 1.0, f"lower {object_name} into basket")
        except OracleFailure:
            # An already occupied basket can stop the held object a few
            # centimeters above the nominal release pose. If it is still well
            # inside the basket footprint and below the safe drop height,
            # release and let the native In predicate decide validity.
            error = release - self.eef_position()
            if np.linalg.norm(error[:2]) >= 0.04 or self.eef_position()[2] >= 0.66:
                raise
        self.hold(cfg.open_steps, -1.0)
        self.move_to(above_basket, -1.0, f"retreat after {object_name}")
        self.hold(cfg.settle_steps, -1.0)
        if not self.env.is_object_in_basket(object_name):
            raise OracleFailure(f"{object_name} was released but LIBERO's In predicate is false")
        logger.info("Placed %s in basket at step %d", object_name, self.steps)

    def run(self, seed: int) -> dict:
        self.observation, _ = self.env.reset(seed=seed)
        self.frames = []
        self.steps = 0
        self.last_info = {}
        offsets = (
            np.array([-0.035, -0.015]),
            np.array([0.035, -0.015]),
            np.array([0.0, -0.04]),
        )
        home_position = self.eef_position().copy()
        home_axis_angle = self.eef_axis_angle().copy()
        logger.info("Oracle reset pose: position=%s axis_angle=%s", home_position.round(4), home_axis_angle.round(4))
        initial_positions = {name: self.object_position(name).tolist() for name in (*OBJECTS, "basket_1")}
        for object_name, offset in zip(DEMONSTRATION_ORDER, offsets, strict=True):
            self.pick_and_place(object_name, offset)
            if object_name == "cream_cheese_1":
                # Cream cheese needs a 90-degree yaw for grasping. Restore the
                # nominal finger orientation at a collision-free transit height
                # before attempting the cylindrical objects.
                eef = self.eef_position()
                self.move_to(
                    np.array([eef[0], eef[1], self.config.transit_z]),
                    -1.0,
                    "lift after cream_cheese_1",
                )
                self.rotate_gripper_z(-3, -1.0)
                self.move_to(home_position, -1.0, "return home after cream_cheese_1")
                logger.info(
                    "Oracle pose after cream recovery: position=%s axis_angle=%s",
                    self.eef_position().round(4),
                    self.eef_axis_angle().round(4),
                )
        self.hold(self.config.settle_steps, -1.0)
        in_basket = {name: self.env.is_object_in_basket(name) for name in OBJECTS}
        success = bool(self.env._env.check_success()) if self.env._env is not None else False
        return {
            "success": success and all(in_basket.values()),
            "steps": self.steps,
            "in_basket": in_basket,
            "initial_positions": initial_positions,
            "final_positions": {name: self.object_position(name).tolist() for name in (*OBJECTS, "basket_1")},
        }


def _add_batch_dimension(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _add_batch_dimension(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return np.expand_dims(value, axis=0)
    return value


class Pi05SubtaskController:
    """Use the existing LIBERO policy as a clean atomic-skill demonstrator."""

    def __init__(self, checkpoint: Path, tokenizer_path: Path):
        import torch

        from lerobot.envs.utils import preprocess_observation
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.processor import LiberoProcessorStep

        checkpoint = checkpoint.expanduser().resolve()
        tokenizer_path = tokenizer_path.expanduser().resolve()
        if not (checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(f"Pi0.5 checkpoint not found: {checkpoint}")
        if not (tokenizer_path / "tokenizer.json").is_file():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

        config = PI05Config.from_pretrained(checkpoint)
        config.device = "cuda"
        config.dtype = "bfloat16"
        config.n_action_steps = 10
        config.compile_model = False
        config.pretrained_path = checkpoint
        self.policy = PI05Policy.from_pretrained(pretrained_name_or_path=checkpoint, config=config)
        self.policy.to("cuda")
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=checkpoint,
            preprocessor_overrides={
                "device_processor": {"device": "cuda"},
                "tokenizer_processor": {"tokenizer_name": str(tokenizer_path)},
            },
        )
        self.env_processor = LiberoProcessorStep()
        self.preprocess_observation = preprocess_observation
        self.torch = torch

    def action(self, observation: dict, prompt: str) -> np.ndarray:
        batch = self.preprocess_observation(_add_batch_dimension(observation))
        batch["task"] = [prompt]
        batch = self.env_processor.observation(batch)
        batch = self.preprocessor(batch)
        with self.torch.inference_mode():
            action = self.postprocessor(self.policy.select_action(batch))
        return action[0].detach().float().cpu().numpy()

    def run(self, recorder: ScriptedRecorder, seed: int, subtask_max_steps: int) -> dict:
        recorder.observation, _ = recorder.env.reset(seed=seed)
        recorder.frames = []
        recorder.steps = 0
        recorder.last_info = {}
        initial_positions = {
            name: recorder.object_position(name).tolist() for name in (*OBJECTS, "basket_1")
        }
        home_position = recorder.eef_position().copy()
        prompts = (
            # The flat box is most reliable while the basket is empty. Returning
            # home afterward keeps the remaining subtasks in distribution.
            ("cream_cheese_1", "put the cream cheese box in the basket"),
            ("butter_1", "put the butter in the basket"),
            ("alphabet_soup_1", "put the alphabet soup in the basket"),
        )
        subtask_steps: dict[str, int] = {}
        for subtask_index, (object_name, prompt) in enumerate(prompts):
            self.policy.reset()
            stable = 0
            start_step = recorder.steps
            for _ in range(subtask_max_steps):
                recorder.step(self.action(recorder.observation, prompt))
                qpos = np.asarray(recorder.observation["robot_state"]["gripper"]["qpos"])
                released = float(np.max(np.abs(qpos))) >= 0.03
                if recorder.env.is_object_in_basket(object_name) and released:
                    stable += 1
                    if stable >= 5:
                        break
                else:
                    stable = 0
            else:
                raise OracleFailure(
                    f"Pi0.5 subtask failed for {object_name} after {subtask_max_steps} steps; "
                    f"in_basket={recorder.env.is_object_in_basket(object_name)}"
                )
            subtask_steps[object_name] = recorder.steps - start_step
            if subtask_index < len(prompts) - 1:
                recorder.move_to(home_position, -1.0, f"return home after {object_name}")

        in_basket = {name: recorder.env.is_object_in_basket(name) for name in OBJECTS}
        success = bool(recorder.env._env.check_success()) if recorder.env._env is not None else False
        return {
            "success": success and all(in_basket.values()),
            "steps": recorder.steps,
            "subtask_steps": subtask_steps,
            "in_basket": in_basket,
            "initial_positions": initial_positions,
            "final_positions": {
                name: recorder.object_position(name).tolist() for name in (*OBJECTS, "basket_1")
            },
        }

    def run_hybrid(self, recorder: ScriptedRecorder, seed: int, subtask_max_steps: int) -> dict:
        """Let Pi0.5 handle cream cheese, then place butter and alphabet soup."""
        recorder.observation, _ = recorder.env.reset(seed=seed)
        recorder.frames = []
        recorder.steps = 0
        recorder.last_info = {}
        initial_positions = {
            name: recorder.object_position(name).tolist() for name in (*OBJECTS, "basket_1")
        }
        home_position = recorder.eef_position().copy()

        object_name = "cream_cheese_1"
        prompt = "put the cream cheese box in the basket"
        self.policy.reset()
        stable = 0
        for _ in range(subtask_max_steps):
            recorder.step(self.action(recorder.observation, prompt))
            qpos = np.asarray(recorder.observation["robot_state"]["gripper"]["qpos"])
            released = float(np.max(np.abs(qpos))) >= 0.03
            if recorder.env.is_object_in_basket(object_name) and released:
                stable += 1
                if stable >= 5:
                    break
            else:
                stable = 0
        else:
            raise OracleFailure(
                f"Pi0.5 subtask failed for {object_name} after {subtask_max_steps} steps; "
                f"in_basket={recorder.env.is_object_in_basket(object_name)}"
            )
        cream_steps = recorder.steps
        recorder.move_to(home_position, -1.0, "return home after cream_cheese_1")

        recorder.pick_and_place("butter_1", np.array([-0.045, -0.025]))
        recorder.pick_and_place("alphabet_soup_1", np.array([0.045, -0.025]))
        in_basket = {name: recorder.env.is_object_in_basket(name) for name in OBJECTS}
        success = bool(recorder.env._env.check_success()) if recorder.env._env is not None else False
        return {
            "success": success and all(in_basket.values()),
            "steps": recorder.steps,
            "subtask_steps": {"cream_cheese_1": cream_steps},
            "in_basket": in_basket,
            "initial_positions": initial_positions,
            "final_positions": {
                name: recorder.object_position(name).tolist() for name in (*OBJECTS, "basket_1")
            },
        }


def prepare_output(args: argparse.Namespace) -> None:
    if args.resume:
        if args.overwrite:
            raise ValueError("--resume and --overwrite are mutually exclusive")
        if not args.output_root.exists():
            raise FileNotFoundError(f"Cannot resume missing dataset: {args.output_root}")
        return
    if not args.output_root.exists():
        return
    if args.dry_run:
        return
    if not args.overwrite:
        raise FileExistsError(
            f"Dataset output already exists: {args.output_root}. Use --overwrite to replace it."
        )
    import shutil

    shutil.rmtree(args.output_root)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    if args.num_successful <= 0:
        raise ValueError("--num-successful must be positive")
    prepare_output(args)
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    debug_dir = args.output_root.parent / f"{args.output_root.name}_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    dataset = None
    if not args.dry_run:
        if args.resume:
            dataset = LeRobotDataset.resume(
                repo_id=args.repo_id,
                root=args.output_root,
                image_writer_threads=4,
            )
        else:
            dataset = LeRobotDataset.create(
                repo_id=args.repo_id,
                fps=args.fps,
                features=dataset_features(),
                root=args.output_root,
                robot_type="panda_libero",
                use_videos=True,
                image_writer_threads=4,
            )

    policy_controller = None
    if args.controller in {"policy", "hybrid"}:
        policy_controller = Pi05SubtaskController(args.checkpoint, args.tokenizer_path)

    successes = dataset.num_episodes if dataset is not None and args.resume else 0
    attempts = 0
    attempts_path = debug_dir / "attempts.json"
    summaries: list[dict] = []
    if args.resume and attempts_path.is_file():
        summaries = json.loads(attempts_path.read_text(encoding="utf-8"))
    attempt_offset = len(summaries)
    try:
        while successes < args.num_successful and attempts < args.max_attempts:
            init_state = args.start_init_state + attempts
            seed = args.seed + attempts
            env = make_env(init_state, args.fps, args.max_steps, args.use_pruned_init_states)
            recorder = ScriptedRecorder(env, OracleConfig(), dataset)
            summary = {"attempt": attempt_offset + attempts, "init_state": init_state, "seed": seed}
            try:
                if policy_controller is None:
                    result = recorder.run(seed)
                elif args.controller == "hybrid":
                    result = policy_controller.run_hybrid(recorder, seed, args.subtask_max_steps)
                else:
                    result = policy_controller.run(recorder, seed, args.subtask_max_steps)
                summary.update(result)
                if not result["success"]:
                    raise OracleFailure(f"final BDDL goal failed: {result['in_basket']}")
                if dataset is not None:
                    dataset.save_episode()
                summary["episode_index"] = successes
                successes += 1
                logger.info(
                    "Saved successful episode %d/%d (%d steps)",
                    successes,
                    args.num_successful,
                    result["steps"],
                )
            except OracleFailure as exc:
                summary.update({"success": False, "error": str(exc), "steps": recorder.steps})
                if dataset is not None:
                    dataset.clear_episode_buffer()
                logger.warning("Attempt %d failed: %s", attempt_offset + attempts, exc)
            finally:
                if args.save_debug_videos or not summary.get("success", False):
                    if recorder.frames:
                        attempt_index = attempt_offset + attempts
                        outcome = "success" if summary.get("success") else "failed"
                        video_path = debug_dir / f"attempt_{attempt_index:03d}_{outcome}.mp4"
                        write_video(video_path, recorder.frames, args.fps)
                        summary["debug_video"] = str(video_path)
                env.close()
            summaries.append(summary)
            attempts_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
            attempts += 1
    finally:
        if dataset is not None:
            if dataset.has_pending_frames():
                dataset.clear_episode_buffer()
            dataset.finalize()

    logger.info("Collection complete: %d successes in %d attempts", successes, attempts)
    if successes < args.num_successful:
        raise RuntimeError(f"Only collected {successes}/{args.num_successful} successful episodes")


if __name__ == "__main__":
    main()
