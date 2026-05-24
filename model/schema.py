from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass(frozen=True)
class CameraSpec:
    role: str
    frame: str
    intrinsics: Optional[torch.Tensor] = None
    extrinsics: Optional[torch.Tensor] = None
    modality: str = "rgb"
    temporal_rate_hz: Optional[float] = None


@dataclass(frozen=True)
class ObservationSpec:
    cameras: list[CameraSpec] = field(default_factory=list)
    base_state_dim: int = 0
    joint_state_dim: int = 0
    ee_state_dim: int = 0
    task_state_dim: int = 0


@dataclass(frozen=True)
class ActionSpec:
    base_dim: int = 3
    joint_dim: int = 0
    gripper_dim: int = 0
    chunk_size: int = 1

    @property
    def total_dim(self) -> int:
        return self.base_dim + self.joint_dim + self.gripper_dim


@dataclass(frozen=True)
class StateBatch:
    base_state_abs: torch.Tensor
    joint_state_abs: torch.Tensor
    ee_state_abs: Optional[torch.Tensor] = None
    task_state: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class ActionBatch:
    base_action_rel: torch.Tensor
    joint_action_rel: torch.Tensor
    gripper_action_rel: Optional[torch.Tensor] = None

    def as_tensor(self) -> torch.Tensor:
        parts = [self.base_action_rel, self.joint_action_rel]
        if self.gripper_action_rel is not None:
            parts.append(self.gripper_action_rel)
        return torch.cat(parts, dim=-1)


@dataclass(frozen=True)
class ObservationBatch:
    images: torch.Tensor
    camera_types: Optional[torch.Tensor] = None
    view_roles: Optional[torch.Tensor] = None
    base_state_abs: Optional[torch.Tensor] = None
    joint_state_abs: Optional[torch.Tensor] = None
    ee_state_abs: Optional[torch.Tensor] = None
    task_state: Optional[torch.Tensor] = None
    state_history: Optional[torch.Tensor] = None
    view_mask: Optional[torch.Tensor] = None

    def to(self, device: torch.device | str) -> "ObservationBatch":
        return ObservationBatch(
            images=self.images.to(device),
            camera_types=None if self.camera_types is None else self.camera_types.to(device),
            view_roles=None if self.view_roles is None else self.view_roles.to(device),
            base_state_abs=None if self.base_state_abs is None else self.base_state_abs.to(device),
            joint_state_abs=None if self.joint_state_abs is None else self.joint_state_abs.to(device),
            ee_state_abs=None if self.ee_state_abs is None else self.ee_state_abs.to(device),
            task_state=None if self.task_state is None else self.task_state.to(device),
            state_history=None if self.state_history is None else self.state_history.to(device),
            view_mask=None if self.view_mask is None else self.view_mask.to(device),
        )


@dataclass(frozen=True)
class TrajectorySample:
    observation: ObservationBatch
    state_abs: StateBatch
    action_rel: ActionBatch
    next_state_abs: Optional[StateBatch] = None
    episode_id: Optional[str] = None
    step_id: Optional[int] = None
    timestamp: Optional[float] = None
