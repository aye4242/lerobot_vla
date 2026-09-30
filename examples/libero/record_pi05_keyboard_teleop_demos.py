#!/usr/bin/env python

"""Record strict, human-operated demonstrations for the custom three-object task.

Run this in the dedicated TurboVNC-enabled container.  Each saved episode is
performed entirely by the human operator: the recorder accepts it only after
LIBERO's goal is true for a short settling period and every target has been
released from the gripper.  Press ``q`` in the MuJoCo window to discard the
current attempt and start a fresh randomized scene; use Ctrl-C in the terminal
to finish recording.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from pathlib import Path

import glfw
import mujoco
import numpy as np
from libero.libero.envs.env_wrapper import ControlEnv
from robosuite.devices import Keyboard
from robosuite.utils.input_utils import input2action
from robosuite.utils.transform_utils import quat2axisangle

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.libero import LiberoEnv

from record_pi05_composition_demos import OBJECTS, TASK, dataset_features, make_env


logger = logging.getLogger(__name__)
SAFE_EEF_SEPARATION_M = 0.08
CAMERA_STATE_PATH = Path("data/config/pi05_teleop_camera.json")
DISTRACTOR_OBJECTS = ("ketchup_1", "orange_juice_1", "milk_1", "tomato_sauce_1")
PERMUTABLE_OBJECTS = (*OBJECTS, *DISTRACTOR_OBJECTS)
SCENE_OBJECTS = (*OBJECTS, *DISTRACTOR_OBJECTS, "basket_1")
CAMERA_VIEWS = {
    "1": "agentview",  # broad scene view, best for teleoperation
    "2": "frontview",
    "3": "birdview",
    "4": "sideview",
    "5": "robot0_eye_in_hand",
}


class TeleopKeyboard(Keyboard):
    """Keyboard control plus optional fixed-camera requests.

    The robosuite defaults are reversed relative to this scene's preferred
    operating view, so horizontal translation is defined explicitly here:
    W/S move along +X/-X and A/D move along +Y/-Y.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.view_request: str | None = None
        self.zoom_delta = 0
        self.save_camera_request = False
        super().__init__(*args, **kwargs)

    @staticmethod
    def _display_controls() -> None:
        print(
            """
Robot controls (fixed world axes; camera movement does not change them)
  W / S       +X / -X horizontal translation
  A / D       +Y / -Y horizontal translation
  R / F       +Z / -Z, raise / lower the gripper
  Z / X       negative / positive rotation about X
  T / G       positive / negative rotation about Y
  C / V       negative / positive rotation about Z
  Space       toggle gripper closed / open

Display controls
  Mouse left-drag   rotate free camera
  Mouse right-drag  pan free camera
  Mouse wheel       zoom free camera
                    (mouse changes are saved automatically)
  0                 reset tabletop free camera
  1 / 2 / 3 / 4 / 5 agent / front / top / side / wrist camera
  + / -             zoom in / out
  P                 save the current free camera as the startup/reset view

Recording controls
  Q           discard the current attempt and load a new randomized scene
  Ctrl-C      stop recording; completed episodes remain saved
"""
        )

    def on_press(self, key) -> None:
        try:
            char = key.char.lower()
            if char == "w":
                self.pos[0] += self._pos_step * self.pos_sensitivity
            elif char == "s":
                self.pos[0] -= self._pos_step * self.pos_sensitivity
            elif char == "a":
                self.pos[1] += self._pos_step * self.pos_sensitivity
            elif char == "d":
                self.pos[1] -= self._pos_step * self.pos_sensitivity
            else:
                super().on_press(key)

            if char in CAMERA_VIEWS:
                self.view_request = CAMERA_VIEWS[char]
            elif char == "0":
                self.view_request = "__free__"
            elif char in {"+", "="}:
                self.zoom_delta += 1
            elif char == "-":
                self.zoom_delta -= 1
            elif char == "p":
                self.save_camera_request = True
        except AttributeError:
            super().on_press(key)


class NativeFreeCameraViewer:
    """Minimal GLFW renderer with mouse camera controls and no keyboard shortcuts."""

    def __init__(self, sim, camera_state_path: Path = CAMERA_STATE_PATH) -> None:
        self._sim = sim
        self._model = sim.model._model
        self._data = sim.data._data
        self._fixed_camera_name: str | None = None
        self._camera_state_path = camera_state_path
        self._last_cursor = (0.0, 0.0)
        self._camera_lock = threading.RLock()
        self._ready = threading.Event()
        self._render_request = threading.Event()
        self._render_done = threading.Event()
        self._close_request = threading.Event()
        self._render_error: BaseException | None = None
        self._window = None
        self._camera = mujoco.MjvCamera()
        self._option = mujoco.MjvOption()
        self._perturb = mujoco.MjvPerturb()
        self._scene = None
        self._render_context = None
        # Draw textured visual geometry only. The default collision geometry
        # produces the green/yellow overlays seen in earlier VNC screenshots.
        self._option.geomgroup[0] = 0
        self._option.geomgroup[1] = 1
        self._reset_free_camera()
        self._render_thread = threading.Thread(
            target=self._render_loop, name="pi05-teleop-render", daemon=True
        )
        self._render_thread.start()
        if not self._ready.wait(timeout=15.0):
            raise RuntimeError("Timed out while creating the MuJoCo render window.")
        if self._render_error is not None:
            raise RuntimeError("Unable to create the MuJoCo render window.") from self._render_error

    def _render_loop(self) -> None:
        try:
            if not glfw.init():
                raise RuntimeError("GLFW initialization failed; check DISPLAY and TurboVNC.")
            glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
            self._window = glfw.create_window(1240, 760, "MuJoCo : base", None, None)
            if self._window is None:
                raise RuntimeError("Unable to create the MuJoCo GLFW window.")
            glfw.make_context_current(self._window)
            glfw.swap_interval(1)
            self._scene = mujoco.MjvScene(self._model, maxgeom=10000)
            self._render_context = mujoco.MjrContext(
                self._model, mujoco.mjtFontScale.mjFONTSCALE_150
            )
            # glfw 2.10.2's public callback wrappers call ctypes.pointer() on
            # an already-created pointer, which fails on Python 3.12. Register
            # the C callbacks directly and retain them for the window lifetime.
            self._mouse_button_callback = glfw._GLFWmousebuttonfun(self._on_mouse_button)
            self._cursor_pos_callback = glfw._GLFWcursorposfun(self._on_cursor_move)
            self._scroll_callback = glfw._GLFWscrollfun(self._on_scroll)
            glfw._glfw.glfwSetMouseButtonCallback(self._window, self._mouse_button_callback)
            glfw._glfw.glfwSetCursorPosCallback(self._window, self._cursor_pos_callback)
            glfw._glfw.glfwSetScrollCallback(self._window, self._scroll_callback)
            self._ready.set()
            while not self._close_request.is_set() and not glfw.window_should_close(self._window):
                glfw.poll_events()
                if not self._render_request.wait(timeout=0.02):
                    continue
                self._render_request.clear()
                with self._camera_lock:
                    width, height = glfw.get_framebuffer_size(self._window)
                    viewport = mujoco.MjrRect(0, 0, width, height)
                    mujoco.mjv_updateScene(
                        self._model,
                        self._data,
                        self._option,
                        self._perturb,
                        self._camera,
                        mujoco.mjtCatBit.mjCAT_ALL,
                        self._scene,
                    )
                    mujoco.mjr_render(viewport, self._scene, self._render_context)
                    glfw.swap_buffers(self._window)
                self._render_done.set()
        except BaseException as error:
            self._render_error = error
            self._ready.set()
            self._render_done.set()
        finally:
            if self._render_context is not None:
                self._render_context.free()
            if self._window is not None:
                glfw.destroy_window(self._window)
                self._window = None
    def _on_mouse_button(self, _window, _button, action, _mods) -> None:
        self._last_cursor = glfw.get_cursor_pos(self._window)
        if action == glfw.RELEASE and self._camera.type == mujoco.mjtCamera.mjCAMERA_FREE:
            state = self.save_free_camera()
            logger.info("Auto-saved free camera after mouse drag: %s", state)

    def _on_cursor_move(self, _window, xpos: float, ypos: float) -> None:
        left = glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        right = glfw.get_mouse_button(self._window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        if not left and not right:
            self._last_cursor = (xpos, ypos)
            return
        previous_x, previous_y = self._last_cursor
        self._last_cursor = (xpos, ypos)
        _, height = glfw.get_window_size(self._window)
        scale = max(float(height), 1.0)
        dx = (xpos - previous_x) / scale
        dy = (ypos - previous_y) / scale
        with self._camera_lock:
            if left:
                mujoco.mjv_moveCamera(
                    self._model, mujoco.mjtMouse.mjMOUSE_ROTATE_H, dx, 0.0, self._scene, self._camera
                )
                mujoco.mjv_moveCamera(
                    self._model, mujoco.mjtMouse.mjMOUSE_ROTATE_V, 0.0, dy, self._scene, self._camera
                )
            else:
                mujoco.mjv_moveCamera(
                    self._model, mujoco.mjtMouse.mjMOUSE_MOVE_H, dx, 0.0, self._scene, self._camera
                )
                mujoco.mjv_moveCamera(
                    self._model, mujoco.mjtMouse.mjMOUSE_MOVE_V, 0.0, dy, self._scene, self._camera
                )

    def _on_scroll(self, _window, _xoffset: float, yoffset: float) -> None:
        with self._camera_lock:
            mujoco.mjv_moveCamera(
                self._model,
                mujoco.mjtMouse.mjMOUSE_ZOOM,
                0.0,
                -0.05 * yoffset,
                self._scene,
                self._camera,
            )
        state = self.save_free_camera()
        logger.info("Auto-saved free camera after zoom: %s", state)

    def _reset_free_camera(self) -> None:
        """Restore the saved tabletop camera, or the screenshot-derived default."""
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera_state = {
            "lookat": [-0.22, 0.12, 0.73],
            "distance": 1.65,
            "azimuth": 135.0,
            "elevation": -25.0,
        }
        if self._camera_state_path.exists():
            camera_state.update(json.loads(self._camera_state_path.read_text()))
        self._camera.lookat[:] = camera_state["lookat"]
        self._camera.distance = float(camera_state["distance"])
        self._camera.azimuth = float(camera_state["azimuth"])
        self._camera.elevation = float(camera_state["elevation"])
        self._fixed_camera_name = None

    def save_free_camera(self) -> dict[str, object]:
        """Persist the current free camera for future starts and resets."""
        with self._camera_lock:
            if self._camera.type != mujoco.mjtCamera.mjCAMERA_FREE:
                raise RuntimeError("Press 0 before P; only a free camera pose can be saved.")
            state = {
                "lookat": [float(value) for value in self._camera.lookat],
                "distance": float(self._camera.distance),
                "azimuth": float(self._camera.azimuth),
                "elevation": float(self._camera.elevation),
            }
        self._camera_state_path.parent.mkdir(parents=True, exist_ok=True)
        self._camera_state_path.write_text(json.dumps(state, indent=2) + "\n")
        return state

    def set_free_camera(self) -> None:
        with self._camera_lock:
            self._reset_free_camera()

    def set_fixed_camera(self, camera_name: str) -> None:
        with self._camera_lock:
            self._camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self._camera.fixedcamid = self._sim.model.camera_name2id(camera_name)
            self._fixed_camera_name = camera_name

    def zoom_by(self, delta: int) -> None:
        with self._camera_lock:
            self._camera.distance = float(
                np.clip(self._camera.distance / (1.25**delta), 0.05, 10.0)
            )

    def sync(self) -> None:
        if self._render_error is not None:
            raise RuntimeError("MuJoCo render thread failed.") from self._render_error
        with self._camera_lock:
            if self._camera.type == mujoco.mjtCamera.mjCAMERA_FREE:
                # TurboVNC can occasionally deliver a large drag delta. Keep
                # the interactive camera inside the tabletop work volume.
                self._camera.lookat[0] = float(np.clip(self._camera.lookat[0], -0.85, 0.35))
                self._camera.lookat[1] = float(np.clip(self._camera.lookat[1], -0.50, 0.75))
                self._camera.lookat[2] = float(np.clip(self._camera.lookat[2], 0.25, 1.20))
                self._camera.distance = float(np.clip(self._camera.distance, 1.20, 4.20))
                self._camera.elevation = float(np.clip(self._camera.elevation, -85.0, -8.0))
        self._render_done.clear()
        self._render_request.set()
        if not self._render_done.wait(timeout=2.0):
            raise RuntimeError("Timed out while rendering the MuJoCo window.")

    def close(self) -> None:
        self._close_request.set()
        self._render_request.set()
        self._render_thread.join(timeout=5.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/datasets/pi05_cream_butter_alphabet_basket_teleop_clean_20hz"),
    )
    parser.add_argument(
        "--repo-id", default="local/pi05_cream_butter_alphabet_basket_teleop_clean_20hz"
    )
    parser.add_argument("--num-successful", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    # Human keyboard operation needs substantially more wall time than the
    # scripted controller: 3 minutes at 20 Hz is a safe upper bound for a
    # careful three-pick demonstration.
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=4000)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument(
        "--min-basket-clearance",
        type=float,
        default=0.18,
        help="Minimum initial planar centre distance (m) from every target to the basket.",
    )
    parser.add_argument(
        "--min-object-clearance",
        type=float,
        default=0.11,
        help="Minimum initial planar centre distance (m) between a target and every other scene object.",
    )
    parser.add_argument(
        "--max-layout-attempts",
        type=int,
        default=48,
        help="Number of deterministic layout seeds to try before failing a reset.",
    )
    parser.add_argument(
        "--layout-mode",
        choices=("permuted", "bddl"),
        default="permuted",
        help=(
            "`permuted` rearranges all target and distractor object slots for visibly distinct scenes; "
            "`bddl` uses only the BDDL sampler."
        ),
    )
    parser.add_argument(
        "--permutation-attempts",
        type=int,
        default=512,
        help="Valid slot permutations sampled within each BDDL reset when --layout-mode=permuted.",
    )
    parser.add_argument("--pos-sensitivity", type=float, default=0.35)
    parser.add_argument("--rot-sensitivity", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true", help="Append to an existing recording dataset.")
    return parser.parse_args()


def raw_to_frame(raw_obs: dict, action: np.ndarray, reward: float, success: bool, done: bool) -> dict:
    """Map ControlEnv's raw robosuite observation to the Pi0.5 LIBERO schema."""
    axis_angle = quat2axisangle(np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float64))
    state = np.concatenate(
        (
            np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32),
            np.asarray(axis_angle, dtype=np.float32),
            np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    )
    return {
        # This is the same camera correction used by the existing custom
        # recordings and by LiberoProcessorStep at evaluation time.
        "observation.images.image": np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1]),
        "observation.images.image2": np.ascontiguousarray(raw_obs["robot0_eye_in_hand_image"][::-1, ::-1]),
        "observation.state": state,
        "action": np.asarray(action, dtype=np.float32),
        "next.reward": np.asarray([reward], dtype=np.float32),
        "next.success": np.asarray([success], dtype=bool),
        "next.done": np.asarray([done], dtype=bool),
        "task": TASK,
    }


def task_bddl_path() -> str:
    """Obtain the installed custom task path through the project task registry."""
    probe: LiberoEnv = make_env(
        init_state=0,
        fps=20,
        max_steps=1,
        use_pruned_init_states=False,
    )
    return probe._task_bddl_file


def create_env(args: argparse.Namespace) -> ControlEnv:
    return ControlEnv(
        bddl_file_name=task_bddl_path(),
        control_freq=args.fps,
        horizon=args.max_steps,
        ignore_done=True,
        hard_reset=True,
        # Creating robosuite's OpenCV viewer during every hard reset can block
        # under TurboVNC. Create it explicitly only after the first reset.
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
    )


def target_status(control_env: ControlEnv, raw_obs: dict) -> tuple[bool, dict[str, float]]:
    """Require real release, avoiding LIBERO's transient `In` false positives."""
    inner = control_env.env
    regions = [
        name
        for name in inner.object_states_dict
        if name.startswith("basket_") and name.endswith("_contain_region")
    ]
    if len(regions) != 1:
        raise RuntimeError(f"Expected exactly one basket contain region, got {regions}")
    eef = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float64)
    distances = {
        name: float(np.linalg.norm(np.asarray(raw_obs[f"{name}_pos"], dtype=np.float64) - eef))
        for name in OBJECTS
    }
    in_basket = all(bool(inner._eval_predicate(["in", name, regions[0]])) for name in OBJECTS)
    released = all(distance > SAFE_EEF_SEPARATION_M for distance in distances.values())
    return bool(control_env.check_success() and in_basket and released), distances


def reset_with_clearance(
    control_env: ControlEnv,
    start_seed: int,
    min_basket_clearance: float,
    min_object_clearance: float,
    max_layout_attempts: int,
    layout_mode: str,
    permutation_attempts: int,
) -> tuple[dict, int, dict[str, dict[str, float]]]:
    """Reset to a randomized layout with enough room to grasp each target.

    LIBERO's placement sampler prevents physical overlap, but its valid states
    can still put a target against the basket rim or a distractor. Such scenes
    encourage the operator to push another object before grasping and are
    inappropriate for clean-action demonstrations.
    """
    for offset in range(max_layout_attempts):
        layout_seed = start_seed + offset
        control_env.seed(layout_seed)
        raw_obs = control_env.reset()
        if layout_mode == "permuted":
            raw_obs = apply_permuted_layout(
                control_env,
                raw_obs,
                layout_seed,
                min_basket_clearance,
                min_object_clearance,
                permutation_attempts,
            )
            if raw_obs is None:
                continue
        clearances = layout_clearances(raw_obs)
        if layout_is_clear(clearances, min_basket_clearance, min_object_clearance):
            return raw_obs, layout_seed, clearances
    raise RuntimeError(
        "Could not sample a layout with sufficient target-to-basket clearance after "
        f"{max_layout_attempts} seeds starting at {start_seed}."
    )


def layout_clearances(raw_obs: dict) -> dict[str, dict[str, float]]:
    object_names = tuple(name for name in SCENE_OBJECTS if f"{name}_pos" in raw_obs)
    positions = {name: np.asarray(raw_obs[f"{name}_pos"], dtype=np.float64)[:2] for name in object_names}
    if "basket_1" not in positions:
        raise RuntimeError("The scene has no basket_1 position observation")
    basket_clearance = {
        name: float(np.linalg.norm(positions[name] - positions["basket_1"]))
        for name in OBJECTS
        if name in positions
    }
    nearest_object_clearance = {
        name: min(
            float(np.linalg.norm(positions[name] - positions[other]))
            for other in object_names
            if other != name
        )
        for name in OBJECTS
        if name in positions
    }
    return {"basket": basket_clearance, "nearest_object": nearest_object_clearance}


def layout_is_clear(
    clearances: dict[str, dict[str, float]], min_basket_clearance: float, min_object_clearance: float
) -> bool:
    return all(distance >= min_basket_clearance for distance in clearances["basket"].values()) and all(
        distance >= min_object_clearance for distance in clearances["nearest_object"].values()
    )


def apply_permuted_layout(
    control_env: ControlEnv,
    raw_obs: dict,
    layout_seed: int,
    min_basket_clearance: float,
    min_object_clearance: float,
    permutation_attempts: int,
) -> dict | None:
    """Reassign the sampled tabletop slots across targets and distractors.

    Each source slot was produced by LIBERO's collision-aware sampler.  By
    permuting the seven movable non-basket items after reset, every episode
    changes the full scene rather than making an imperceptibly small local
    perturbation around the same named-object positions.  Target clearance is
    still enforced by :func:`reset_with_clearance` before the episode is shown.
    """
    object_names = tuple(name for name in PERMUTABLE_OBJECTS if f"{name}_pos" in raw_obs)
    if len(object_names) < 2:
        return raw_obs
    slot_positions = [np.asarray(raw_obs[f"{name}_pos"], dtype=np.float64)[:2] for name in object_names]
    rng = np.random.default_rng(layout_seed)
    selected_order: np.ndarray | None = None
    basket_position = np.asarray(raw_obs["basket_1_pos"], dtype=np.float64)[:2]
    for _ in range(permutation_attempts):
        slot_order = rng.permutation(len(object_names))
        candidate_positions = {
            object_name: slot_positions[slot_index]
            for object_name, slot_index in zip(object_names, slot_order, strict=True)
        }
        candidate_positions["basket_1"] = basket_position
        candidate_raw_obs = {
            f"{name}_pos": position for name, position in candidate_positions.items()
        }
        clearances = layout_clearances(candidate_raw_obs)
        if layout_is_clear(clearances, min_basket_clearance, min_object_clearance):
            selected_order = slot_order
            break
    if selected_order is None:
        return None
    inner = control_env.env
    for object_name, slot_index in zip(object_names, selected_order, strict=True):
        joint_name = inner.objects_dict[object_name].joints[-1]
        qpos = np.asarray(inner.sim.data.get_joint_qpos(joint_name), dtype=np.float64)
        qpos[:2] = slot_positions[slot_index]
        inner.sim.data.set_joint_qpos(joint_name, qpos)
    inner.sim.forward()
    inner._update_observables(force=True)
    return inner._get_observations()


def open_dataset(args: argparse.Namespace) -> LeRobotDataset:
    if args.num_successful <= 0:
        raise ValueError("--num-successful must be positive")
    if args.resume:
        if not args.output_root.exists():
            raise FileNotFoundError(f"Cannot resume missing dataset: {args.output_root}")
        return LeRobotDataset.resume(
            repo_id=args.repo_id,
            root=args.output_root,
            image_writer_threads=4,
        )
    if args.output_root.exists():
        raise FileExistsError(
            f"Dataset output already exists: {args.output_root}. Use --resume to append rather than overwrite."
        )
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    return LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=dataset_features(),
        root=args.output_root,
        robot_type="panda_libero",
        use_videos=True,
        image_writer_threads=4,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    if args.fps <= 0 or args.max_steps <= 0 or args.settle_steps <= 0:
        raise ValueError("--fps, --max-steps, and --settle-steps must be positive")
    if (
        args.min_basket_clearance <= 0
        or args.min_object_clearance <= 0
        or args.max_layout_attempts <= 0
        or args.permutation_attempts <= 0
    ):
        raise ValueError(
            "--min-basket-clearance, --min-object-clearance, --max-layout-attempts, and "
            "--permutation-attempts must be positive"
        )

    dataset = open_dataset(args)
    env = create_env(args)
    viewer: NativeFreeCameraViewer | None = None
    keyboard = TeleopKeyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    saved = dataset.num_episodes
    next_layout_seed = args.seed
    deadline = 1.0 / args.fps
    logger.info("Task: %s", TASK)
    logger.info(
        "Use the MuJoCo window's native mouse camera: left-drag rotates, right-drag pans, and the wheel zooms. "
        "q discards/restarts; space toggles the gripper; 0 resets the free camera; "
        "1/2/3/4/5 select full/front/top/side/wrist fixed views; P saves the free camera; "
        "+/- zoom; Ctrl-C ends. "
        "Saved episodes need %d stable released-success steps.",
        args.settle_steps,
    )

    try:
        while saved < args.num_successful:
            raw_obs, layout_seed, clearances = reset_with_clearance(
                env,
                start_seed=next_layout_seed,
                min_basket_clearance=args.min_basket_clearance,
                min_object_clearance=args.min_object_clearance,
                max_layout_attempts=args.max_layout_attempts,
                layout_mode=args.layout_mode,
                permutation_attempts=args.permutation_attempts,
            )
            next_layout_seed = layout_seed + 1
            # ``hard_reset=True`` reconstructs MuJoCo's MjSim. Its previous
            # native viewer therefore points at a destroyed simulation after
            # `q`; re-create it for every attempt.
            if viewer is not None:
                viewer.close()
            viewer = NativeFreeCameraViewer(env.sim)
            keyboard.start_control()
            stable = 0
            steps = 0
            logger.info(
                "Attempt for episode %d is ready (%s layout seed %d; target-to-basket=%s; target-to-nearest=%s).",
                saved,
                args.layout_mode,
                layout_seed,
                {name: round(distance, 3) for name, distance in clearances["basket"].items()},
                {
                    name: round(distance, 3)
                    for name, distance in clearances["nearest_object"].items()
                },
            )
            while steps < args.max_steps:
                tick_start = time.monotonic()
                action, _ = input2action(keyboard, env.robots[0])
                if action is None:
                    dataset.clear_episode_buffer()
                    logger.info("Attempt discarded. Starting a fresh randomized scene.")
                    break
                view = keyboard.view_request
                if view is not None:
                    if view == "__free__":
                        viewer.set_free_camera()
                    else:
                        viewer.set_fixed_camera(view)
                    keyboard.view_request = None
                    logger.info("Switched display camera to %s", view)
                zoom_delta = keyboard.zoom_delta
                if zoom_delta:
                    viewer.zoom_by(zoom_delta)
                    keyboard.zoom_delta = 0
                if keyboard.save_camera_request:
                    camera_state = viewer.save_free_camera()
                    keyboard.save_camera_request = False
                    logger.info("Saved startup camera to %s: %s", CAMERA_STATE_PATH, camera_state)
                raw_next, reward, _done, _info = env.step(action)
                success, distances = target_status(env, raw_next)
                stable = stable + 1 if success else 0
                done = stable >= args.settle_steps
                dataset.add_frame(raw_to_frame(raw_obs, action, reward, success, done))
                raw_obs = raw_next
                steps += 1
                viewer.sync()
                if done:
                    dataset.save_episode()
                    saved += 1
                    logger.info(
                        "Saved episode %d with %d frames; final EEF distances: %s",
                        saved,
                        steps,
                        {name: round(value, 3) for name, value in distances.items()},
                    )
                    break
                remaining = deadline - (time.monotonic() - tick_start)
                if remaining > 0:
                    time.sleep(remaining)
            else:
                dataset.clear_episode_buffer()
                logger.warning("Attempt exceeded %d steps and was discarded.", args.max_steps)
    except KeyboardInterrupt:
        if dataset.writer is not None:
            dataset.clear_episode_buffer()
        logger.info("Stopped. Saved %d episode(s) at %s", saved, args.output_root)
    finally:
        keyboard.listener.stop()
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
