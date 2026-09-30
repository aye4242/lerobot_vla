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

from dataclasses import dataclass

import torch

from lerobot.policies.smolvla import smolvlm_with_expert


@dataclass
class _TensorMetadata:
    dtype: torch.dtype
    device: torch.device


def test_pre_ampere_bf16_attention_requires_fp32_value_matmul(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (5, 2))
    value_states = _TensorMetadata(dtype=torch.bfloat16, device=torch.device("cuda"))

    assert smolvlm_with_expert._requires_fp32_attention_value_matmul(value_states)


def test_ampere_bf16_attention_keeps_native_value_matmul(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 0))
    value_states = _TensorMetadata(dtype=torch.bfloat16, device=torch.device("cuda"))

    assert not smolvlm_with_expert._requires_fp32_attention_value_matmul(value_states)


def test_fp32_attention_does_not_require_compatibility_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (5, 2))
    value_states = _TensorMetadata(dtype=torch.float32, device=torch.device("cuda"))

    assert not smolvlm_with_expert._requires_fp32_attention_value_matmul(value_states)
