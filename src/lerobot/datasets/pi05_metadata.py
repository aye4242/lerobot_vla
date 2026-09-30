#!/usr/bin/env python

"""Episode-level metadata support for the incremental PI05 experiment."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from lerobot.utils.constants import (
    PI05_METADATA_CONTROL_MODE,
    PI05_METADATA_MISTAKE,
    PI05_METADATA_QUALITY,
    PI05_METADATA_SPEED_STEPS,
)

_FIELD_TO_BATCH_KEY = {
    "speed_steps": PI05_METADATA_SPEED_STEPS,
    "quality": PI05_METADATA_QUALITY,
    "mistake": PI05_METADATA_MISTAKE,
    "control_mode": PI05_METADATA_CONTROL_MODE,
}


def load_pi05_episode_metadata(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a JSON mapping keyed by episode index.

    A multi-dataset mapping may use ``"dataset_index:episode_index"`` keys;
    single-dataset mappings normally use just ``"episode_index"``.
    """
    metadata_path = Path(path).expanduser()
    with metadata_path.open(encoding="utf-8") as metadata_file:
        payload = json.load(metadata_file)

    if isinstance(payload, dict) and "episodes" in payload:
        payload = payload["episodes"]
    if not isinstance(payload, dict):
        raise ValueError(f"PI05 metadata must be a JSON object, got {type(payload).__name__}")

    result: dict[str, dict[str, Any]] = {}
    for episode_key, value in payload.items():
        if not isinstance(episode_key, str) or not episode_key:
            raise ValueError("PI05 metadata episode keys must be non-empty strings")
        if not isinstance(value, dict):
            raise ValueError(f"Metadata for episode {episode_key!r} must be an object")
        unknown = set(value) - set(_FIELD_TO_BATCH_KEY)
        if unknown:
            raise ValueError(f"Unknown PI05 metadata fields for episode {episode_key!r}: {sorted(unknown)}")
        speed_steps = value.get("speed_steps")
        if speed_steps is not None and (
            isinstance(speed_steps, bool) or not isinstance(speed_steps, int) or speed_steps <= 0
        ):
            raise ValueError(f"speed_steps for episode {episode_key!r} must be a positive integer")
        quality = value.get("quality")
        if quality is not None and (
            isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 5
        ):
            raise ValueError(f"quality for episode {episode_key!r} must be an integer from 1 to 5")
        if "mistake" in value and not isinstance(value["mistake"], bool):
            raise ValueError(f"mistake for episode {episode_key!r} must be a boolean")
        control_mode = value.get("control_mode")
        if control_mode is not None and control_mode not in {"joint", "ee", "end_effector"}:
            raise ValueError(
                f"control_mode for episode {episode_key!r} must be 'joint', 'ee', or 'end_effector'"
            )
        result[episode_key] = dict(value)
    return result


def _scalar(value: Any) -> int | str:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected a scalar index, got tensor with shape {tuple(value.shape)}")
        value = value.detach().cpu().item()
    if isinstance(value, bool):
        raise ValueError("Episode indices cannot be boolean")
    if isinstance(value, (int, str)):
        return value
    raise TypeError(f"Expected an integer or string index, got {type(value).__name__}")


class PI05EpisodeMetadataDataset(Dataset):
    """Add validated PI05 episode metadata to samples from a map-style dataset."""

    def __init__(
        self,
        dataset: Dataset,
        metadata_path: str | Path,
        *,
        missing_policy: str = "error",
    ) -> None:
        if missing_policy not in {"error", "omit"}:
            raise ValueError(f"Unsupported PI05 metadata missing policy: {missing_policy!r}")
        self.dataset = dataset
        self.metadata_path = str(Path(metadata_path).expanduser())
        self.missing_policy = missing_policy
        self.metadata = load_pi05_episode_metadata(self.metadata_path)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str) -> Any:
        # Preserve the dataset facade expected by samplers and training helpers.
        return getattr(self.dataset, name)

    def _metadata_for(self, item: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if "episode_index" not in item:
            raise KeyError("PI05 metadata requires every dataset sample to expose episode_index")
        episode_index = _scalar(item["episode_index"])
        dataset_index = _scalar(item["dataset_index"]) if "dataset_index" in item else None
        candidates = []
        if dataset_index is not None:
            candidates.append(f"{dataset_index}:{episode_index}")
        candidates.append(str(episode_index))
        for key in candidates:
            if key in self.metadata:
                return self.metadata[key]
        if self.missing_policy == "omit":
            return None
        raise KeyError(
            f"No PI05 metadata found for sample episode {episode_index!r}; "
            f"looked up keys {candidates} in {self.metadata_path}"
        )

    def _enrich(self, item: dict[str, Any]) -> dict[str, Any]:
        metadata = self._metadata_for(item)
        if metadata is None:
            return item
        for field, batch_key in _FIELD_TO_BATCH_KEY.items():
            if field in metadata:
                item[batch_key] = metadata[field]
        return item

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._enrich(dict(self.dataset[index]))

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        if hasattr(self.dataset, "__getitems__"):
            items = self.dataset.__getitems__(indices)
        else:
            items = [self.dataset[index] for index in indices]
        return [self._enrich(dict(item)) for item in items]
