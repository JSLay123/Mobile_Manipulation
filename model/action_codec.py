from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from model.schema import ActionSpec


@dataclass(frozen=True)
class RelativeActionBatch:
    base_action_rel: torch.Tensor
    joint_action_rel: torch.Tensor
    gripper_action_rel: Optional[torch.Tensor] = None

    def as_tensor(self) -> torch.Tensor:
        parts = [self.base_action_rel, self.joint_action_rel]
        if self.gripper_action_rel is not None:
            parts.append(self.gripper_action_rel)
        return torch.cat(parts, dim=-1)


class ActionCodec:
    def __init__(self, action_spec: ActionSpec) -> None:
        self.action_spec = action_spec

    def split(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        base_end = self.action_spec.base_dim
        joint_end = base_end + self.action_spec.joint_dim
        base_action = action[..., :base_end]
        joint_action = action[..., base_end:joint_end]
        gripper_action = None
        if self.action_spec.gripper_dim > 0:
            gripper_action = action[..., joint_end:joint_end + self.action_spec.gripper_dim]
        return base_action, joint_action, gripper_action

    def pack(self, base_action_rel: torch.Tensor, joint_action_rel: torch.Tensor, gripper_action_rel: Optional[torch.Tensor] = None) -> torch.Tensor:
        parts = [base_action_rel, joint_action_rel]
        if gripper_action_rel is not None:
            parts.append(gripper_action_rel)
        return torch.cat(parts, dim=-1)

    def absolute_to_relative(
        self,
        base_control_state_abs: torch.Tensor,
        joint_control_state_abs: torch.Tensor,
        base_action_abs: torch.Tensor,
        joint_action_abs: torch.Tensor,
        gripper_action_abs: Optional[torch.Tensor] = None,
    ) -> RelativeActionBatch:
        base_action_rel = base_action_abs - base_control_state_abs[..., : self.action_spec.base_dim]
        joint_action_rel = joint_action_abs - joint_control_state_abs[..., : self.action_spec.joint_dim]
        gripper_action_rel = None
        if gripper_action_abs is not None:
            gripper_start = self.action_spec.joint_dim
            gripper_end = gripper_start + self.action_spec.gripper_dim
            gripper_action_rel = gripper_action_abs - joint_control_state_abs[..., gripper_start:gripper_end]
        return RelativeActionBatch(
            base_action_rel=base_action_rel,
            joint_action_rel=joint_action_rel,
            gripper_action_rel=gripper_action_rel,
        )

    def relative_to_absolute(
        self,
        base_control_state_abs: torch.Tensor,
        joint_control_state_abs: torch.Tensor,
        action_rel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        base_rel, joint_rel, gripper_rel = self.split(action_rel)
        base_action_abs = base_control_state_abs[..., : self.action_spec.base_dim] + base_rel
        joint_action_abs = joint_control_state_abs[..., : self.action_spec.joint_dim] + joint_rel
        gripper_action_abs = None
        if gripper_rel is not None:
            gripper_start = self.action_spec.joint_dim
            gripper_end = gripper_start + self.action_spec.gripper_dim
            gripper_action_abs = joint_control_state_abs[..., gripper_start:gripper_end] + gripper_rel
        return base_action_abs, joint_action_abs, gripper_action_abs
