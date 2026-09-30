#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from types import SimpleNamespace

import pytest

from lerobot.envs.vlabench import (
    VLABenchEnv,
    _episode_index_from_seed,
    _extract_task_description,
    _load_deterministic_episode_configs,
    _resolve_camera_frame_indices,
)


def test_resolve_camera_frame_indices_matches_training_dataset():
    camera_names = ["right", "left", "forward", "franka/Franka_wrist_cam"]

    assert _resolve_camera_frame_indices(camera_names, frame_count=4) == {
        "image": 2,
        "second_image": 0,
        "wrist_image": 3,
    }


def test_resolve_camera_frame_indices_uses_standard_order_without_names():
    assert _resolve_camera_frame_indices([], frame_count=4) == {
        "image": 2,
        "second_image": 0,
        "wrist_image": 3,
    }


def test_extract_task_description_prefers_get_instruction():
    task = SimpleNamespace(
        get_instruction=lambda: "  Please take the red toy.  ",
        task_description="stale description",
    )

    assert _extract_task_description(task, "select_toy") == "Please take the red toy."


def test_extract_task_description_supports_instruction_list():
    task = SimpleNamespace(instructions=["", "Please take the book."])

    assert _extract_task_description(task, "select_book") == "Please take the book."


def test_extract_task_description_supports_legacy_attribute():
    task = SimpleNamespace(language_instruction="Insert the flower.")

    assert _extract_task_description(task, "insert_flower") == "Insert the flower."


def test_extract_task_description_falls_back_to_task_name():
    task = SimpleNamespace(get_instruction=lambda: None, instructions=[""])

    assert _extract_task_description(task, "select_drink") == "select_drink"


def test_load_deterministic_episode_configs_selects_task(tmp_path):
    track = tmp_path / "track.json"
    track.write_text(
        '{"select_toy": [{"task": {"instructions": ["Put the toy in the box"]}}]}', encoding="utf-8"
    )

    assert _load_deterministic_episode_configs(str(track), "select_toy") == [
        {"task": {"instructions": ["Put the toy in the box"]}}
    ]


def test_load_deterministic_episode_configs_rejects_missing_task(tmp_path):
    track = tmp_path / "track.json"
    track.write_text('{"select_fruit": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="select_toy"):
        _load_deterministic_episode_configs(str(track), "select_toy")


def test_episode_index_from_seed_uses_offset():
    assert _episode_index_from_seed(1003, 1000, 50, 0) == 3


def test_episode_index_from_seed_wraps_seedless_autoreset():
    assert _episode_index_from_seed(None, 1000, 50, 50) == 0


def test_episode_index_from_seed_rejects_out_of_range():
    with pytest.raises(ValueError, match="outside"):
        _episode_index_from_seed(1050, 1000, 50, 0)


def test_episode_info_contains_stage_diagnostics():
    env = object.__new__(VLABenchEnv)
    env.task = "select_toy"
    env.task_description = "Put the mickey into the giftbox_seen"
    env.target_entity = "mickey"
    env.target_container = "giftbox_seen"
    env.episode_config_index = 0
    env._stage_metrics = {
        "stage_target_reached": True,
        "stage_target_grasped": True,
        "stage_target_lifted": False,
        "stage_wrong_object_grasped": False,
        "min_eef_target_distance_m": 0.02,
        "max_target_lift_m": 0.03,
    }

    info = env._episode_info(is_success=False)

    assert info["instruction"] == "Put the mickey into the giftbox_seen"
    assert info["target_entity"] == "mickey"
    assert info["stage_target_grasped"] is True
    assert info["stage_placed"] is False
