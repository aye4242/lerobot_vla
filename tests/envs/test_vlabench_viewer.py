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

import sys
from types import SimpleNamespace

import numpy as np
import torch

from examples.vlabench.run_smolvla_viewer import (
    NativeViewerControls,
    PI0_BASE_CAMERA_RENAME_MAP,
    SMOLVLA_CAMERA_RENAME_MAP,
    SmolVLAViewerPolicy,
    _parse_camera_map,
    _resolve_camera_rename_map,
    parse_args,
)
from lerobot.configs import FeatureType, PolicyFeature


class _FakePolicy:
    config = SimpleNamespace(device="cpu", use_amp=False)

    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def select_action(self, observation):
        assert observation["task"] == ["Move the toy."]
        return torch.arange(7, dtype=torch.float32).unsqueeze(0)


class _FakeEnv:
    task = "select_toy"
    task_description = ""

    def __init__(self):
        task = SimpleNamespace(get_instruction=lambda: "Move the toy.")
        data = SimpleNamespace(ctrl=np.zeros(9, dtype=np.float64))
        opt = SimpleNamespace(
            timestep=0.002,
            solver=2,
            integrator=0,
            iterations=100,
            tolerance=1e-8,
            gravity=np.array([0.0, 0.0, -9.81]),
        )
        model = SimpleNamespace(opt=opt)
        self._env = SimpleNamespace(task=task, physics=SimpleNamespace(data=data, model=model))

    def _get_obs(self):
        return {"agent_pos": np.zeros(7, dtype=np.float32)}

    def _build_ctrl_from_action(self, action, ctrl_dim):
        assert action.shape == (7,)
        assert ctrl_dim == 9
        return np.pad(action, (0, 2))


def test_viewer_policy_resets_queue_and_returns_joint_ctrl():
    env = _FakeEnv()
    policy = _FakePolicy()
    callback = SmolVLAViewerPolicy(
        env,
        policy,
        preprocessor=lambda observation: observation,
        postprocessor=lambda action: action,
        observation_preprocessor=lambda observation: observation,
    )

    callback.reset()
    ctrl = callback(SimpleNamespace(first=lambda: True))

    assert policy.reset_count == 1
    assert ctrl.shape == (9,)
    np.testing.assert_array_equal(ctrl[:7], np.arange(7))


def test_native_viewer_controls():
    controls = NativeViewerControls()

    controls.key_callback(controls.KEY_SPACE)
    assert controls.paused is False
    controls.key_callback(controls.KEY_SPACE)
    controls.key_callback(controls.KEY_RIGHT)
    controls.key_callback(controls.KEY_BACKSPACE)

    assert controls.paused is True
    assert controls.single_step_requested is True
    assert controls.reset_requested is True


def test_parse_args_preserves_checkpoint_action_steps_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_smolvla_viewer.py"])

    args = parse_args()

    assert args.n_action_steps is None


def test_parse_args_accepts_action_steps_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_smolvla_viewer.py", "--n-action-steps", "10"])

    args = parse_args()

    assert args.n_action_steps == 10


def test_camera_map_defaults_follow_policy_checkpoint_features():
    smolvla_cfg = SimpleNamespace(type="smolvla", input_features={})
    pi0_cfg = SimpleNamespace(
        type="pi0",
        input_features={
            "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.images.left_wrist_0_rgb": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            "observation.images.right_wrist_0_rgb": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
        },
    )

    assert _resolve_camera_rename_map(smolvla_cfg, None) == SMOLVLA_CAMERA_RENAME_MAP
    assert _resolve_camera_rename_map(pi0_cfg, None) == PI0_BASE_CAMERA_RENAME_MAP


def test_explicit_camera_map_overrides_policy_default():
    value = '{"observation.images.image": "observation.images.custom"}'

    assert _parse_camera_map(value) == {
        "observation.images.image": "observation.images.custom",
    }
    assert _resolve_camera_rename_map(SimpleNamespace(type="pi0", input_features={}), value) == {
        "observation.images.image": "observation.images.custom",
    }
