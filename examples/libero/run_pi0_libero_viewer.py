#!/usr/bin/env python

"""Run a Pi-family policy in LIBERO with a live interactive viewer."""

from __future__ import annotations

import argparse
import textwrap
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import simpledialog

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageTk

from lerobot.configs import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--task", default="libero_object", help="LIBERO suite name")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--n-action-steps", type=int, default=50)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Optional local tokenizer/vision processor directory overriding the checkpoint config.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Interactive episode step limit; use a larger value for multi-object runs.",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="Override the LIBERO language instruction passed to the policy.",
    )
    parser.add_argument(
        "--post-success-steps",
        type=int,
        default=50,
        help="Keep executing this many steps after LIBERO first reports success.",
    )
    parser.add_argument(
        "--auto-home-after-success",
        action="store_true",
        help="Return the robot to its initial pose after each successful object task.",
    )
    parser.add_argument("--render-resolution", type=int, default=360)
    parser.add_argument("--window-width", type=int, default=960)
    parser.add_argument("--window-height", type=int, default=720)
    parser.add_argument("--start-running", action="store_true")
    parser.add_argument(
        "--start-free-camera",
        action="store_true",
        dest="start_free_camera",
        help="Start with the mouse-controlled observer view (default).",
    )
    parser.add_argument(
        "--start-fixed-camera",
        action="store_false",
        dest="start_free_camera",
        help="Start with the policy's fixed camera instead of the free observer view.",
    )
    parser.set_defaults(start_free_camera=True)
    parser.add_argument(
        "--max-viewer-steps",
        type=int,
        default=None,
        help="Exit after this many total environment steps (useful for smoke tests).",
    )
    parser.add_argument("--no-realtime", action="store_true")
    return parser.parse_args()


def _match_instruction_target(
    instruction: str, scene_objects: list[dict[str, str]]
) -> dict[str, str] | None:
    normalized_instruction = instruction.lower().replace("_", " ")
    candidates = sorted(scene_objects, key=lambda item: len(item["name"]), reverse=True)
    return next(
        (item for item in candidates if item["name"] in normalized_instruction),
        None,
    )


def _libero_success(info: dict[str, object]) -> bool:
    """Read the vectorized LIBERO BDDL success predicate for the active task."""
    value = info.get("is_success", False)
    values = np.asarray(value)
    return bool(values.reshape(-1)[0]) if values.size else False


def _draw_overlay(
    rgb_frame: np.ndarray,
    *,
    task_description: str,
    step: int,
    max_steps: int,
    inference_s: float,
    paused: bool,
    message: str,
    free_camera: bool,
) -> Image.Image:
    frame = Image.fromarray(rgb_frame.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(frame)
    instruction_lines = textwrap.wrap(task_description, width=76)[:2] or [""]
    message_y = 31 + 18 * len(instruction_lines) + 5
    camera_y = message_y + 23
    overlay_height = camera_y + 22
    draw.rectangle((0, 0, frame.width, overlay_height), fill=(0, 0, 0))
    status = "PAUSED" if paused else "RUNNING"
    draw.text(
        (8, 8),
        f"{status}  step={step}/{max_steps}  inference={inference_s:.3f}s",
        fill=(80, 255, 80),
    )
    for line_index, instruction_line in enumerate(instruction_lines):
        draw.text((8, 31 + 18 * line_index), instruction_line, fill=(255, 255, 255))
    draw.text(
        (8, message_y),
        message or "Space=run/pause  I=next instruction",
        fill=(100, 220, 255),
    )
    camera_text = (
        "FREE VIEW: left-drag=orbit  right-drag=pan  wheel=zoom  C=fixed view"
        if free_camera
        else "POLICY VIEW (fixed): C=free mouse view"
    )
    draw.text((8, camera_y), camera_text, fill=(255, 210, 100))
    return frame


def _fit_to_window(image: Image.Image, width: int, height: int) -> Image.Image:
    width = max(width, 1)
    height = max(height, 1)
    fitted = image.copy()
    fitted.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    canvas.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
    return canvas


def main() -> None:
    args = parse_args()
    if args.post_success_steps < 0:
        raise ValueError("--post-success-steps must be non-negative")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    checkpoint = args.policy_path.expanduser().resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {checkpoint / 'model.safetensors'}")

    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy_type = str(policy_cfg.type)
    policy_label = {"pi0": "Pi0", "pi05": "Pi0.5", "groot": "GR00T N1.7"}.get(
        policy_type, policy_type
    )
    policy_cfg.pretrained_path = checkpoint
    policy_cfg.device = args.device
    policy_cfg.dtype = args.dtype
    policy_cfg.compile_model = False
    policy_cfg.n_action_steps = args.n_action_steps
    if policy_type == "groot":
        policy_cfg.base_model_path = str(checkpoint)
        policy_cfg.use_bf16 = False
        policy_cfg.model_params_fp32 = True
        policy_cfg.use_flash_attention = False

    env_cfg = LiberoEnvConfig(
        task=args.task,
        task_ids=[args.task_id],
        control_mode="relative",
        observation_height=args.render_resolution,
        observation_width=args.render_resolution,
        max_parallel_tasks=1,
        auto_reset_on_termination=False,
        terminate_on_success=False,
        episode_length=args.max_steps,
    )
    env_groups = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = env_groups[args.task][args.task_id]

    print(f"Loading {policy_label} policy: {checkpoint}", flush=True)
    camera_rename_map = (
        {"observation.images.image2": "observation.images.wrist_image"}
        if policy_type == "groot"
        else {}
    )
    if camera_rename_map:
        print(f"Camera rename map: {camera_rename_map}", flush=True)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=camera_rename_map)
    policy.eval()

    preprocessor_overrides: dict[str, dict[str, str]] = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": camera_rename_map},
    }
    if args.tokenizer_path is not None:
        tokenizer_path = args.tokenizer_path.expanduser().resolve()
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_path}")
        if policy_type == "groot":
            preprocessor_overrides["groot_n1_7_vlm_encode_v1"] = {
                "model_name": str(tokenizer_path)
            }
            print(f"GR00T VLM processor assets: {tokenizer_path}", flush=True)
        else:
            preprocessor_overrides["tokenizer_processor"] = {
                "tokenizer_name": str(tokenizer_path)
            }
            print(f"Tokenizer override: {tokenizer_path}", flush=True)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    libero_instruction = str(env.call("task_description")[0])
    task_description = args.instruction or libero_instruction
    scene_objects = list(env.call("get_scene_objects")[0])
    active_target = _match_instruction_target(task_description, scene_objects)
    max_steps = int(env.call("_max_episode_steps")[0])
    fps = int(env.unwrapped.metadata.get("render_fps", env_cfg.fps))
    window_title = f"{policy_label} LIBERO | {args.task} task {args.task_id}"

    root = tk.Tk()
    root.title(window_title)
    root.geometry(f"{args.window_width}x{args.window_height}")
    image_label = tk.Label(root, background="black")
    image_label.pack(fill=tk.BOTH, expand=True)
    pressed_keys: list[str] = []
    camera_events: list[tuple[str, float, float]] = []
    drag_state: dict[str, int | None] = {"button": None, "x": None, "y": None}

    def record_key(event: tk.Event) -> None:
        if isinstance(event.widget, (tk.Entry, tk.Text)):
            return
        key = event.keysym.lower()
        if key not in {"space", "i", "h", "c", "r", "q", "escape"}:
            return
        pressed_keys.append(key)
        display_key = "SPACE" if key == "space" else key.upper()
        print(f"Viewer key detected: {display_key} (queued)", flush=True)

    root.bind_all("<Key>", record_key)

    def start_drag(event: tk.Event, button: int) -> None:
        drag_state.update(button=button, x=event.x, y=event.y)

    def drag(event: tk.Event) -> None:
        old_x = drag_state["x"]
        old_y = drag_state["y"]
        button = drag_state["button"]
        if old_x is None or old_y is None or button is None:
            return
        width = max(image_label.winfo_width(), 1)
        height = max(image_label.winfo_height(), 1)
        dx = (event.x - old_x) / width
        dy = (event.y - old_y) / height
        if button == 1:
            camera_events.extend((("rotate_h", dx, 0.0), ("rotate_v", 0.0, dy)))
        else:
            camera_events.extend((("move_h", dx, 0.0), ("move_v", 0.0, dy)))
        drag_state.update(x=event.x, y=event.y)

    def stop_drag(_event: tk.Event) -> None:
        drag_state.update(button=None, x=None, y=None)

    def wheel(event: tk.Event) -> None:
        direction = -1.0 if event.delta > 0 else 1.0
        camera_events.append(("zoom", 0.0, direction * 0.08))

    image_label.bind("<ButtonPress-1>", lambda event: start_drag(event, 1))
    image_label.bind("<ButtonPress-3>", lambda event: start_drag(event, 3))
    image_label.bind("<B1-Motion>", drag)
    image_label.bind("<B3-Motion>", drag)
    image_label.bind("<ButtonRelease-1>", stop_drag)
    image_label.bind("<ButtonRelease-3>", stop_drag)
    image_label.bind("<MouseWheel>", wheel)
    image_label.bind("<Button-4>", lambda _event: camera_events.append(("zoom", 0.0, -0.08)))
    image_label.bind("<Button-5>", lambda _event: camera_events.append(("zoom", 0.0, 0.08)))
    root.update_idletasks()
    root.update()
    root.focus_force()
    print(f"LIBERO instruction: {libero_instruction}", flush=True)
    print(f"{policy_label} instruction: {task_description}", flush=True)
    print(
        "Scene objects: " + ", ".join(item["name"] for item in scene_objects),
        flush=True,
    )
    print(
        f"Active target: {active_target['name'] if active_target else 'manual / not detected'}",
        flush=True,
    )
    print(
        f"Success handling: continue {args.post_success_steps} steps after first detection",
        flush=True,
    )
    print(
        "Primary controls: Space=run/pause, I=next instruction",
        flush=True,
    )
    print(
        "Optional controls: H=robot home, C=fixed/free camera, R=reset, "
        "Q/Esc=quit; mouse drag/wheel controls the free camera",
        flush=True,
    )

    episode_index = 0
    total_steps = 0
    step = 0
    inference_s = 0.0
    first_success_step: int | None = None
    paused = not args.start_running
    free_camera = args.start_free_camera
    message = "Press Space to start"
    observation, _ = env.reset(seed=[args.seed])
    policy.reset()
    if free_camera:
        env.call("reset_free_camera")
    rgb_frame = np.asarray(env.call("render_free_camera" if free_camera else "render")[0])
    home_hold_action = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]], dtype=np.float32)

    def return_robot_home() -> None:
        nonlocal observation, step, total_steps, inference_s, first_success_step, rgb_frame
        print("Robot cleanup START: release object and return arm to Home", flush=True)
        env.call("return_robot_home")
        observation, _, _, _, _ = env.step(home_hold_action)
        step += 1
        total_steps += 1
        policy.reset()
        inference_s = 0.0
        first_success_step = None
        rgb_frame = np.asarray(env.call("render_free_camera" if free_camera else "render")[0])
        print("Robot cleanup DONE: Home is ready for the next instruction", flush=True)

    inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi-inference")
    try:
        while True:
            display_image = _draw_overlay(
                rgb_frame,
                task_description=task_description,
                step=step,
                max_steps=max_steps,
                inference_s=inference_s,
                paused=paused,
                message=message,
                free_camera=free_camera,
            )
            display_image = _fit_to_window(
                display_image, image_label.winfo_width(), image_label.winfo_height()
            )
            photo = ImageTk.PhotoImage(display_image)
            image_label.configure(image=photo)
            image_label.image = photo
            try:
                root.update_idletasks()
                root.update()
            except tk.TclError:
                break
            key = pressed_keys.pop(0) if pressed_keys else ""

            if key in ("q", "escape"):
                print("Viewer command: QUIT", flush=True)
                break
            if key == "space":
                paused = not paused
                message = ""
                print(f"Viewer state: {'PAUSED' if paused else 'RUNNING'}", flush=True)
            if key == "i":
                paused = True
                new_instruction = simpledialog.askstring(
                    f"{policy_label} instruction",
                    "Enter the next instruction:",
                    initialvalue=task_description,
                    parent=root,
                )
                if new_instruction and new_instruction.strip():
                    task_description = new_instruction.strip()
                    active_target = _match_instruction_target(task_description, scene_objects)
                    policy.reset()
                    first_success_step = None
                    inference_s = 0.0
                    target_text = active_target["name"] if active_target else "manual"
                    message = f"New instruction ready ({target_text}); press Space"
                    print(f"{policy_label} instruction changed: {task_description}", flush=True)
                    print(f"Active target: {target_text}", flush=True)
            if key == "h":
                paused = True
                return_robot_home()
                message = "Robot returned Home; press I for a task or Space to continue"
            if key == "c":
                free_camera = not free_camera
                if free_camera:
                    env.call("reset_free_camera")
                    rgb_frame = np.asarray(env.call("render_free_camera")[0])
                    message = f"Free observer camera; {policy_label} inputs remain fixed"
                else:
                    rgb_frame = np.asarray(env.call("render")[0])
                    message = "Fixed policy camera"
                print(
                    f"Viewer camera: {'FREE OBSERVER' if free_camera else 'FIXED POLICY VIEW'}",
                    flush=True,
                )
            camera_changed = False
            while camera_events:
                camera_action, rel_x, rel_y = camera_events.pop(0)
                if free_camera:
                    env.call("move_free_camera", camera_action, rel_x, rel_y)
                    camera_changed = True
            if free_camera and (key == "c" or camera_changed):
                rgb_frame = np.asarray(env.call("render_free_camera")[0])
            if key == "r":
                print("Viewer command: RESET", flush=True)
                episode_index += 1
                observation, _ = env.reset(seed=[args.seed + episode_index])
                policy.reset()
                step = 0
                inference_s = 0.0
                first_success_step = None
                paused = True
                message = f"Reset complete (episode {episode_index}); press Space"
                rgb_frame = np.asarray(
                    env.call("render_free_camera" if free_camera else "render")[0]
                )
                continue
            if paused:
                time.sleep(0.03)
                continue

            tick = time.perf_counter()
            policy_input = preprocess_observation(observation)
            policy_input["task"] = [task_description]
            policy_input = env_preprocessor(policy_input)
            policy_input = preprocessor(policy_input)

            inference_start = time.perf_counter()
            is_new_chunk = len(policy._action_queue) == 0
            if is_new_chunk:
                message = f"INFERENCE: computing action chunk at step {step + 1}"
                print(
                    f"{policy_label} chunk inference START | step={step + 1} | "
                    f"n_action_steps={args.n_action_steps} | instruction={task_description!r}",
                    flush=True,
                )
                inference_image = _draw_overlay(
                    rgb_frame,
                    task_description=task_description,
                    step=step,
                    max_steps=max_steps,
                    inference_s=inference_s,
                    paused=paused,
                    message=message,
                    free_camera=free_camera,
                )
                inference_image = _fit_to_window(
                    inference_image, image_label.winfo_width(), image_label.winfo_height()
                )
                inference_photo = ImageTk.PhotoImage(inference_image)
                image_label.configure(image=inference_photo)
                image_label.image = inference_photo
                root.update_idletasks()
                root.update()

                def compute_action() -> torch.Tensor:
                    with torch.inference_mode():
                        return policy.select_action(policy_input)

                inference_future = inference_executor.submit(compute_action)
                next_progress_log = 2.0
                while not inference_future.done():
                    try:
                        root.update_idletasks()
                        root.update()
                    except tk.TclError:
                        pass
                    elapsed_inference = time.perf_counter() - inference_start
                    if elapsed_inference >= next_progress_log:
                        print(
                            f"{policy_label} chunk inference still running | step={step + 1} | "
                            f"elapsed={elapsed_inference:.1f}s",
                            flush=True,
                        )
                        next_progress_log += 2.0
                    time.sleep(0.03)
                action = inference_future.result()
            else:
                with torch.inference_mode():
                    action = policy.select_action(policy_input)
            inference_s = time.perf_counter() - inference_start
            action = postprocessor(action)
            action_transition = env_postprocessor({ACTION: action})
            action_numpy = action_transition[ACTION].detach().cpu().numpy()

            observation, _, terminated, truncated, info = env.step(action_numpy)
            step += 1
            total_steps += 1
            rgb_frame = np.asarray(env.call("render_free_camera" if free_camera else "render")[0])
            # This is LIBERO's generic BDDL predicate. It covers all four suites,
            # including drawers, spatial relations, and multi-stage LIBERO-10 tasks.
            success = _libero_success(info)

            if success and first_success_step is None:
                first_success_step = step
                message = (
                    f"SUCCESS detected; completing {args.post_success_steps} more steps"
                )
                print(
                    f"LIBERO success detected | step={step} | continuing for "
                    f"{args.post_success_steps} post-success steps",
                    flush=True,
                )

            if is_new_chunk:
                print(
                    f"{policy_label} chunk inference DONE | step={step} | {inference_s:.3f}s | "
                    f"action={np.round(action_numpy[0], 4).tolist()}",
                    flush=True,
                )
                message = f"Action chunk ready in {inference_s:.2f}s"
            elif step % 10 == 0:
                print(
                    f"{policy_label} rollout active | step={step} | "
                    f"queued_actions={len(policy._action_queue)}",
                    flush=True,
                )

            if first_success_step is not None:
                remaining_post_success_steps = max(
                    args.post_success_steps - (step - first_success_step), 0
                )
                message = (
                    f"SUCCESS detected; {remaining_post_success_steps} continuation steps remaining"
                )

            post_success_complete = (
                first_success_step is not None
                and step - first_success_step >= args.post_success_steps
            )
            if post_success_complete and success:
                paused = True
                if args.auto_home_after_success:
                    return_robot_home()
                    message = "SUCCESS + HOME complete - press I for the next object"
                else:
                    policy.reset()
                    first_success_step = None
                    message = "SUCCESS complete - press I for the next object (H=optional Home)"
                print(f"{policy_label} post-success sequence complete | step={step}", flush=True)
            elif post_success_complete:
                print(
                    f"LIBERO success was lost after continuation | step={step}; continuing rollout",
                    flush=True,
                )
                first_success_step = None

            episode_finished = bool(terminated[0] or truncated[0]) or step >= max_steps
            if episode_finished and not paused:
                paused = True
                message = "SUCCESS - press R" if success else "TIMEOUT - press R"

            if args.max_viewer_steps is not None and total_steps >= args.max_viewer_steps:
                print(f"Reached --max-viewer-steps={args.max_viewer_steps}", flush=True)
                break

            if not args.no_realtime:
                elapsed = time.perf_counter() - tick
                time.sleep(max(0.0, 1.0 / fps - elapsed))
    finally:
        inference_executor.shutdown(wait=True, cancel_futures=True)
        env.close()
        try:
            root.destroy()
        except tk.TclError:
            pass


if __name__ == "__main__":
    main()
