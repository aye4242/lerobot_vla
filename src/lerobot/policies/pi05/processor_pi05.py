#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    TokenizerProcessorStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import (
    OBS_STATE,
    PI05_METADATA_CONTROL_MODE,
    PI05_METADATA_MISTAKE,
    PI05_METADATA_QUALITY,
    PI05_METADATA_SPEED_STEPS,
)

from .configuration_pi05 import PI05Config


@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"
    # MEM section III-D represents proprioception with a linear projection into the
    # backbone instead of discretized prompt tokens, so the state is carried once.
    # Set from `PI05Config.use_proprioceptive_memory`; stock PI0.5 keeps it in the prompt.
    include_state_in_prompt: bool = True
    use_episode_metadata: bool = False
    metadata_missing_policy: str = "error"
    metadata_default_speed_steps: int | None = None
    metadata_default_quality: int | None = None
    metadata_default_mistake: bool | None = None
    metadata_default_control_mode: str | None = None

    def __post_init__(self) -> None:
        if self.metadata_missing_policy not in {"error", "omit"}:
            raise ValueError("metadata_missing_policy must be either 'error' or 'omit'")

    def get_config(self) -> dict[str, Any]:
        return {
            "max_state_dim": self.max_state_dim,
            "task_key": self.task_key,
            "include_state_in_prompt": self.include_state_in_prompt,
            "use_episode_metadata": self.use_episode_metadata,
            "metadata_missing_policy": self.metadata_missing_policy,
            "metadata_default_speed_steps": self.metadata_default_speed_steps,
            "metadata_default_quality": self.metadata_default_quality,
            "metadata_default_mistake": self.metadata_default_mistake,
            "metadata_default_control_mode": self.metadata_default_control_mode,
        }

    @staticmethod
    def _batch_value(value: Any, index: int) -> Any:
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.item()
            item = value[index]
            return item.item() if item.ndim == 0 else item.tolist()
        if isinstance(value, (list, tuple)):
            return value[index]
        return value

    def _render_metadata(self, complementary_data: dict[str, Any], index: int) -> str | None:
        fields = {
            "speed_steps": complementary_data.get(PI05_METADATA_SPEED_STEPS),
            "quality": complementary_data.get(PI05_METADATA_QUALITY),
            "mistake": complementary_data.get(PI05_METADATA_MISTAKE),
            "control_mode": complementary_data.get(PI05_METADATA_CONTROL_MODE),
        }
        defaults = {
            "speed_steps": self.metadata_default_speed_steps,
            "quality": self.metadata_default_quality,
            "mistake": self.metadata_default_mistake,
            "control_mode": self.metadata_default_control_mode,
        }
        values = {}
        for name, value in fields.items():
            value = self._batch_value(value, index) if value is not None else None
            values[name] = defaults[name] if value is None else value

        if all(value is None for value in values.values()):
            if self.metadata_missing_policy == "omit":
                return None
            raise ValueError("PI05 episode metadata is enabled but no metadata was provided")
        missing = [name for name, value in values.items() if value is None]
        if missing:
            if self.metadata_missing_policy == "omit":
                return None
            raise ValueError(f"Missing PI05 episode metadata fields: {missing}")

        speed_steps = values["speed_steps"]
        quality = values["quality"]
        mistake = values["mistake"]
        control_mode = values["control_mode"]
        if isinstance(speed_steps, bool) or not isinstance(speed_steps, int):
            raise ValueError("PI05 metadata speed_steps must be a positive integer")
        if isinstance(quality, bool) or not isinstance(quality, int):
            raise ValueError("PI05 metadata quality must be an integer from 1 to 5")
        if not isinstance(mistake, bool):
            raise ValueError("PI05 metadata mistake must be a boolean")
        if not isinstance(control_mode, str):
            raise ValueError("PI05 metadata control_mode must be a string")
        if speed_steps <= 0:
            raise ValueError("PI05 metadata speed_steps must be positive")
        if not 1 <= quality <= 5:
            raise ValueError("PI05 metadata quality must be between 1 and 5")
        if control_mode not in {"joint", "ee", "end_effector"}:
            raise ValueError("PI05 metadata control_mode must be 'joint', 'ee', or 'end_effector'")
        if control_mode == "end_effector":
            control_mode = "ee"
        return (
            f"Speed: {speed_steps} steps.\n"
            f"Quality: {quality}.\n"
            f"Mistake: {str(mistake).lower()}.\n"
            f"Control Mode: {control_mode}."
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        observation = transition.get(TransitionKey.OBSERVATION)
        state = observation.get(OBS_STATE) if observation is not None else None
        if state is None:
            raise ValueError("State is required for PI05")
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None:
            raise ValueError("Complementary data is required for PI05")
        tasks = complementary_data.get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # TODO: check if this necessary
        state = deepcopy(state)

        discretized_states = None
        if self.include_state_in_prompt:
            # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
            # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
            prompt_state = state[:, -1] if state.ndim == 3 else state
            state_np = prompt_state.cpu().numpy()
            discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            if discretized_states is None:
                full_prompt = f"Task: {cleaned_text};"
            else:
                state_str = " ".join(map(str, discretized_states[i]))
                full_prompt = f"Task: {cleaned_text}, State: {state_str};"
            if self.use_episode_metadata:
                metadata_prompt = self._render_metadata(transition[TransitionKey.COMPLEMENTARY_DATA], i)
                if metadata_prompt is not None:
                    full_prompt = f"{full_prompt}\n{metadata_prompt}"
            full_prompt = f"{full_prompt}\nAction: "
            full_prompts.append(full_prompt)

        complementary_data[self.task_key] = full_prompts
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    steps = make_default_policy_processor_steps(config, dataset_stats)

    # OpenPI order: raw → relative → normalize → model → unnormalize → absolute
    input_steps: list[ProcessorStep] = [
        steps.rename_observations,  # To mimic the same processor as pretrained one
        steps.add_batch_dim,
        relative_step,
        # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        steps.normalize,
        Pi05PrepareStateTokenizerProcessorStep(
            max_state_dim=config.max_state_dim,
            include_state_in_prompt=not config.use_proprioceptive_memory,
            use_episode_metadata=getattr(config, "use_episode_metadata", False),
            metadata_missing_policy=getattr(config, "metadata_missing_policy", "error"),
            metadata_default_speed_steps=getattr(config, "metadata_default_speed_steps", None),
            metadata_default_quality=getattr(config, "metadata_default_quality", None),
            metadata_default_mistake=getattr(config, "metadata_default_mistake", None),
            metadata_default_control_mode=getattr(config, "metadata_default_control_mode", None),
        ),
        TokenizerProcessorStep(
            tokenizer_name=config.text_tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        steps.to_device,
    ]

    output_steps: list[ProcessorStep] = [
        steps.unnormalize,
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        steps.to_cpu,
    ]

    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
