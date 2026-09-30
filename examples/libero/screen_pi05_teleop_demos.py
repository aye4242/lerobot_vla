#!/usr/bin/env python

"""Rank teleoperated Pi0.5 demonstrations for manual clean-action review.

This script is deliberately non-destructive. It cannot prove a collision from
recorded actions alone, so it identifies episodes that are most likely to
contain operator corrections (for example, moving in one direction and quickly
undoing it). Review the resulting candidates in the saved videos before
excluding or replacing any episode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/datasets/pi05_three_objects_basket_teleop_20hz"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Defaults to <dataset-root>/reports/action_screening.json.",
    )
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=0.05,
        help="Per-axis action magnitude that counts as an intentional movement command.",
    )
    parser.add_argument(
        "--correction-window-frames",
        type=int,
        default=40,
        help="Opposite-direction commands within this window count as a correction.",
    )
    return parser.parse_args()


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Return deterministic [0, 1] ranks, robust when values repeat."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(values, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(values), endpoint=True)
    return ranks


def rapid_reversals(actions: np.ndarray, threshold: float, window: int) -> int:
    """Count likely direction-correction pairs in the translational actions."""
    count = 0
    for axis in range(3):
        command_indices = np.flatnonzero(np.abs(actions[:, axis]) >= threshold)
        if len(command_indices) < 2:
            continue
        signs = np.sign(actions[command_indices, axis])
        for previous_index, current_index, previous_sign, current_sign in zip(
            command_indices[:-1], command_indices[1:], signs[:-1], signs[1:], strict=True
        ):
            if current_index - previous_index <= window and previous_sign != current_sign:
                count += 1
    return count


def main() -> None:
    args = parse_args()
    if args.active_threshold <= 0 or args.correction_window_frames <= 0:
        raise ValueError("--active-threshold and --correction-window-frames must be positive")
    root = args.dataset_root.resolve()
    report_path = args.report or root / "reports" / "action_screening.json"
    table = ds.dataset(root / "data", format="parquet").to_table(
        columns=["episode_index", "action", "next.success"]
    )
    episode_indices = np.asarray(table.column("episode_index").to_numpy(), dtype=np.int64)
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    successes = np.asarray(table.column("next.success").to_numpy(), dtype=bool)

    records: list[dict[str, int | float | bool]] = []
    for episode_index in sorted(np.unique(episode_indices)):
        episode_actions = actions[episode_indices == episode_index]
        translation = episode_actions[:, :3]
        rotation = episode_actions[:, 3:6]
        active = np.linalg.norm(translation, axis=1) >= args.active_threshold
        records.append(
            {
                "episode_index": int(episode_index),
                "frames": int(len(episode_actions)),
                "duration_seconds": round(float(len(episode_actions) / 20.0), 2),
                "translation_command_frames": int(active.sum()),
                "translation_command_distance": round(float(np.linalg.norm(translation, axis=1).sum()), 3),
                "rotation_command_magnitude": round(float(np.linalg.norm(rotation, axis=1).sum()), 3),
                "rapid_direction_corrections": rapid_reversals(
                    episode_actions, args.active_threshold, args.correction_window_frames
                ),
                "terminal_success": bool(successes[episode_indices == episode_index].any()),
            }
        )

    lengths = np.asarray([record["frames"] for record in records], dtype=np.float64)
    corrections = np.asarray(
        [record["rapid_direction_corrections"] for record in records], dtype=np.float64
    )
    distance = np.asarray(
        [record["translation_command_distance"] for record in records], dtype=np.float64
    )
    score = (
        0.45 * percentile_rank(lengths)
        + 0.40 * percentile_rank(corrections)
        + 0.15 * percentile_rank(distance)
    )
    review_cutoff = float(np.quantile(score, 0.75))
    for record, quality_score in zip(records, score, strict=True):
        record["review_score"] = round(float(quality_score), 3)
        record["manual_review"] = bool(quality_score >= review_cutoff)
        record["review_reason"] = (
            "long trajectory / frequent rapid directional corrections; inspect video for collision or recovery"
            if record["manual_review"]
            else "no automatic concern; still retain only after any known collision is reviewed"
        )

    records.sort(key=lambda record: float(record["review_score"]), reverse=True)
    report = {
        "purpose": "Non-destructive ranking for human clean-action review; not an automatic collision label.",
        "dataset_root": str(root),
        "total_episodes": len(records),
        "manual_review_candidates": [
            int(record["episode_index"]) for record in records if bool(record["manual_review"])
        ],
        "settings": {
            "active_threshold": args.active_threshold,
            "correction_window_frames": args.correction_window_frames,
            "review_score_75th_percentile": round(review_cutoff, 3),
        },
        "episodes_ranked": records,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"Manual-review candidates: {report['manual_review_candidates']}")
    for record in records[:10]:
        print(
            "episode={episode_index:02d} score={review_score:.3f} frames={frames} "
            "corrections={rapid_direction_corrections} distance={translation_command_distance}".format(**record)
        )


if __name__ == "__main__":
    main()
