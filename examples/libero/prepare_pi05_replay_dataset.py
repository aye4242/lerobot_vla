#!/usr/bin/env python

"""Build a targeted 10 Hz replay dataset for the three-object Pi0.5 task.

Official LIBERO demonstrations are already recorded at 10 Hz. Custom sources
may be recorded at 10 Hz or a higher integer multiple; each is normalized to
10 Hz before the sources are combined. This makes it possible to mix the
original 20 Hz demonstrations with newly recorded 10 Hz demonstrations.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


REPLAY_TASK_INDICES = (4, 5, 7)
CUSTOM_TASK = "put the cream cheese box, alphabet soup, and tomato sauce in the basket, one at a time"
COMMON_FEATURES = {
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
    "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument(
        "--custom-roots",
        type=Path,
        nargs="+",
        required=True,
        help="One or more custom successful-demonstration dataset roots.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/pi05_three_objects_replay_10hz")
    parser.add_argument("--old-episodes-per-task", type=int, default=15)
    parser.add_argument("--custom-repeats", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_official_episodes(root: Path, episodes_per_task: int) -> list[int]:
    data_files = sorted((root / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No official parquet files found under {root / 'data'}")
    episode_tasks = pd.concat(
        [pd.read_parquet(path, columns=["episode_index", "task_index"]) for path in data_files],
        ignore_index=True,
    ).drop_duplicates()

    selected: list[int] = []
    for task_index in REPLAY_TASK_INDICES:
        candidates = sorted(
            episode_tasks.loc[episode_tasks["task_index"] == task_index, "episode_index"]
            .astype(int)
            .tolist()
        )
        if len(candidates) < episodes_per_task:
            raise ValueError(
                f"Task {task_index} only has {len(candidates)} episodes; requested {episodes_per_task}"
            )
        selected.extend(candidates[:episodes_per_task])
    return sorted(selected)


def frame_for_writer(item: dict, *, task_override: str | None = None) -> dict:
    frame = {"task": task_override or item["task"]}
    for key in ("observation.images.image", "observation.images.image2"):
        image = item[key]
        if not isinstance(image, torch.Tensor) or image.ndim != 3:
            raise TypeError(f"Expected CHW tensor for {key}, got {type(image)} {getattr(image, 'shape', None)}")
        frame[key] = image.permute(1, 2, 0).contiguous().cpu().numpy()
    frame["observation.state"] = item["observation.state"].cpu().numpy()
    frame["action"] = item["action"].cpu().numpy()
    return frame


def append_source(
    output: LeRobotDataset,
    source: LeRobotDataset,
    *,
    stride: int,
    description: str,
    task_override: str | None = None,
) -> tuple[int, int]:
    source_episode: int | None = None
    kept_frames = 0
    saved_episodes = 0
    for item in tqdm(source, desc=description):
        episode_index = int(item["episode_index"])
        if source_episode is not None and episode_index != source_episode:
            output.save_episode()
            saved_episodes += 1
        source_episode = episode_index
        if int(item["frame_index"]) % stride != 0:
            continue
        output.add_frame(frame_for_writer(item, task_override=task_override))
        kept_frames += 1
    if source_episode is not None:
        output.save_episode()
        saved_episodes += 1
    return saved_episodes, kept_frames


def main() -> None:
    args = parse_args()
    if args.old_episodes_per_task <= 0 or args.custom_repeats <= 0:
        raise ValueError("Episode count and custom repeats must be positive")
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {args.output_root}")
        shutil.rmtree(args.output_root)

    official_episodes = select_official_episodes(args.official_root, args.old_episodes_per_task)
    official = LeRobotDataset(
        "local/libero_official_replay",
        root=args.official_root,
        # Local-only datasets do not have Hub version refs. "main" skips the
        # compatibility lookup while preserving the metadata under root.
        revision="main",
        episodes=official_episodes,
        download_videos=False,
        return_uint8=True,
    )
    customs = [
        LeRobotDataset(
            f"local/pi05_three_objects_basket_{index}",
            root=root,
            revision="main",
            download_videos=False,
            return_uint8=True,
        )
        for index, root in enumerate(args.custom_roots)
    ]
    if official.fps != 10:
        raise ValueError(f"Expected official FPS 10, got {official.fps}")
    for root, custom in zip(args.custom_roots, customs, strict=True):
        if custom.fps < 10 or custom.fps % 10 != 0:
            raise ValueError(f"Custom FPS must be an integer multiple of 10: {root} has {custom.fps}")

    output = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=10,
        features=COMMON_FEATURES,
        root=args.output_root,
        robot_type="panda",
        use_videos=True,
        image_writer_threads=8,
    )
    try:
        old_episodes, old_frames = append_source(
            output, official, stride=1, description="Copying official replay"
        )
        new_episodes = 0
        new_frames = 0
        for repeat in range(args.custom_repeats):
            for root, custom in zip(args.custom_roots, customs, strict=True):
                stride = custom.fps // 10
                episodes, frames = append_source(
                    output,
                    custom,
                    stride=stride,
                    description=(
                        f"Copying custom demonstrations from {root.name} "
                        f"({repeat + 1}/{args.custom_repeats}, stride {stride})"
                    ),
                    task_override=CUSTOM_TASK,
                )
                new_episodes += episodes
                new_frames += frames
    finally:
        if output.has_pending_frames():
            output.clear_episode_buffer()
        output.finalize()

    print(f"official episodes/frames: {old_episodes}/{old_frames}")
    print(f"custom episodes/frames:   {new_episodes}/{new_frames}")
    print(f"total episodes/frames:    {output.meta.total_episodes}/{output.meta.total_frames}")
    print(f"output: {args.output_root}")


if __name__ == "__main__":
    main()
