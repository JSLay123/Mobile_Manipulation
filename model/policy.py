from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from model.mm_blocks import CrossAttention, DiT_Block, FinalLayer, SharedDiTBlock, TimeEmbedding
from model.position_embedding import PositionalEmbedding
from model.schema import ActionSpec, ObservationBatch
from model.vision_head import MultiViewVisionEncoder
from model.gt_module import GatingCoefficient


class StateEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedSplitFlowPolicy(nn.Module):
    """Shared-Split DiT policy.

    Flow:
        1. Encode vision: each camera → one CLS token [B, V, D]
        2. Compute g_t from state history [B, K, 15] → [B, 1]
        3. Gating: split vision into fixed/dynamic, blend with g_t
        4. Encode state + action → hidden_size
        5. Shared DiT blocks: self-attn(x) → cross-attn(all vision)
        6. Split: take action portion [B, T, D]
        7. Base branch: self-attn → cross-attn(joint_state) → self-attn → cross-attn(V_base)
        8. Joint branch: self-attn → cross-attn(base_state) → self-attn → cross-attn(V_joint)
        9. Head: base_vel [B,T,3], joint_vel [B,T,N_joints+gripper]
    """

    def __init__(
        self,
        vision_backbone: nn.Module,
        action_spec: ActionSpec,
        base_state_dim: int,
        joint_state_dim: int,
        num_fixed_cameras: int = 4,
        hidden_size: int = 512,
        num_heads: int = 8,
        num_shared_layers: int = 2,
        num_branch_layers: int = 4,
        max_action_chunk: int = 32,
        history_window: int = 10,
        gt_hidden_size: int = 256,
    ) -> None:
        super().__init__()

        self.action_spec = action_spec
        self.hidden_size = hidden_size
        self.max_action_chunk = max_action_chunk
        self.num_fixed_cameras = num_fixed_cameras

        # --- Time embedding ---
        self.time_embed = TimeEmbedding(hidden_size=hidden_size)

        # --- Vision encoder ---
        self.vision_encoder = MultiViewVisionEncoder(
            backbone=vision_backbone,
            dino_type='base',
            proj_dim=hidden_size,
        )

        # --- Gating coefficient ---
        self.gt_module = GatingCoefficient(
            state_dim=base_state_dim + joint_state_dim,
            hidden_size=gt_hidden_size,
            num_heads=4,
            num_layers=2,
            history_window=history_window,
        )

        # --- State encoders ---
        self.base_state_encoder = StateEncoder(base_state_dim, hidden_size)
        self.joint_state_encoder = StateEncoder(joint_state_dim, hidden_size)

        # --- Action encoders ---
        self.base_action_encoder = ActionEncoder(action_spec.base_dim, hidden_size)
        self.joint_action_encoder = ActionEncoder(
            action_spec.joint_dim + action_spec.gripper_dim, hidden_size,
        )

        # --- Positional embedding for action tokens ---
        self.action_pos_embed = PositionalEmbedding(hidden_size, max_seq_len=max_action_chunk)

        # --- Shared DiT blocks ---
        self.shared_blocks = nn.ModuleList([
            SharedDiTBlock(hidden_size=hidden_size, num_heads=num_heads)
            for _ in range(num_shared_layers)
        ])

        # --- Branch DiT blocks ---
        self.base_blocks = nn.ModuleList([
            DiT_Block(hidden_size=hidden_size, num_heads=num_heads)
            for _ in range(num_branch_layers)
        ])
        self.joint_blocks = nn.ModuleList([
            DiT_Block(hidden_size=hidden_size, num_heads=num_heads)
            for _ in range(num_branch_layers)
        ])

        # --- Output heads ---
        self.base_head = FinalLayer(hidden_size=hidden_size, out_dim=action_spec.base_dim)
        self.joint_head = FinalLayer(
            hidden_size=hidden_size,
            out_dim=action_spec.joint_dim + action_spec.gripper_dim,
        )

    def encode_observation(
        self,
        images: torch.Tensor,
        camera_types: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode multi-view images into vision tokens.

        Args:
            images: [B, V, 3, H, W]
            camera_types: [V] or [B, V], 0=fixed, 1=dynamic
        Returns:
            all_vision: [B, V, D]
            fixed_vision: [B, N_fixed, D]
            dynamic_vision: [B, N_dynamic, D]
        """
        all_vision = self.vision_encoder(images, camera_types=camera_types)  # [B, V, D]

        if camera_types is not None:
            # camera_types: [V] — use first batch for indexing
            if camera_types.dim() == 2:
                ct = camera_types[0]
            else:
                ct = camera_types
            fixed_mask = ct == 0
            dynamic_mask = ct == 1
            fixed_vision = all_vision[:, fixed_mask, :]
            dynamic_vision = all_vision[:, dynamic_mask, :]
        else:
            # default: first num_fixed_cameras are fixed
            fixed_vision = all_vision[:, :self.num_fixed_cameras, :]
            dynamic_vision = all_vision[:, self.num_fixed_cameras:, :]

        return all_vision, fixed_vision, dynamic_vision

    def apply_gt_gating(
        self,
        fixed_vision: torch.Tensor,
        dynamic_vision: torch.Tensor,
        g_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend fixed/dynamic vision tokens using g_t.

        V_base  = (1-g_t) * V_fixed  + g_t * mean(V_dynamic)  → attend mostly to fixed
        V_joint = g_t * V_dynamic    + (1-g_t) * mean(V_fixed) → attend mostly to dynamic

        Args:
            fixed_vision: [B, N_fixed, D]
            dynamic_vision: [B, N_dynamic, D]
            g_t: [B, 1]
        Returns:
            V_base: [B, N_fixed, D]
            V_joint: [B, N_dynamic, D]
        """
        # g_t broadcast: [B,1] → [B,1,1]
        gt = g_t.unsqueeze(-1)

        fixed_mean = fixed_vision.mean(dim=1, keepdim=True)    # [B, 1, D]
        dynamic_mean = dynamic_vision.mean(dim=1, keepdim=True)  # [B, 1, D]

        V_base = (1 - gt) * fixed_vision + gt * dynamic_mean
        V_joint = gt * dynamic_vision + (1 - gt) * fixed_mean

        return V_base, V_joint

    def forward_branch(
        self,
        h: torch.Tensor,
        state_cond: torch.Tensor,
        vision_cond: torch.Tensor,
        blocks: nn.ModuleList,
    ) -> torch.Tensor:
        """Run branch DiT blocks: self-attn → cross-attn(state) → self-attn → cross-attn(vision).

        Args:
            h: [B, T, D]
            state_cond: [B, 1, D] — other branch's state token
            vision_cond: [B, N_v, D] — gated vision tokens
            blocks: list of DiT_Block (alternating self-attn+cross-attn)
        Returns:
            [B, T, D]
        """
        for i, block in enumerate(blocks):
            if i % 2 == 0:
                h = block(h, state_cond)
            else:
                h = block(h, vision_cond)
        return h

    def forward(
        self,
        images: torch.Tensor,
        state_base: torch.Tensor,
        state_joint: torch.Tensor,
        action_base: torch.Tensor,
        action_joint: torch.Tensor,
        t: torch.Tensor,
        state_history: Optional[torch.Tensor] = None,
        camera_types: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: [B, V, 3, H, W]
            state_base: [B, base_dim] — current base state
            state_joint: [B, joint_dim] — current joint+gripper state
            action_base: [B, T, base_dim] — base action sequence
            action_joint: [B, T, joint_dim+gripper_dim] — joint action sequence
            t: [B] — diffusion timestep
            state_history: [B, K, state_dim] — for g_t
            camera_types: [V] or [B, V]
        Returns:
            dict with base_vel, joint_vel, g_t, (optionally h_base, h_joint)
        """
        B = images.size(0)
        T = action_base.size(1)

        # 1. Vision encoding
        all_vision, fixed_vision, dynamic_vision = self.encode_observation(images, camera_types)

        # 2. Gating coefficient
        if state_history is not None:
            g_t = self.gt_module(state_history)  # [B, 1]
        else:
            g_t = torch.zeros(B, 1, device=images.device)

        # 3. Vision gating
        V_base, V_joint = self.apply_gt_gating(fixed_vision, dynamic_vision, g_t)

        # 4. Encode state and action
        t_emb = self.time_embed(t)  # [B, D]

        state_base_token = self.base_state_encoder(state_base)   # [B, D]
        state_joint_token = self.joint_state_encoder(state_joint)  # [B, D]

        base_action_tokens = self.base_action_encoder(action_base)   # [B, T, D]
        joint_action_tokens = self.joint_action_encoder(action_joint)  # [B, T, D]

        # 5. Build shared sequence: [state_base, state_joint, action_base+joint]
        state_base_token = state_base_token.unsqueeze(1)   # [B, 1, D]
        state_joint_token = state_joint_token.unsqueeze(1)  # [B, 1, D]
        action_tokens = base_action_tokens + joint_action_tokens  # [B, T, D]
        action_tokens = self.action_pos_embed(action_tokens)

        # add time embedding to state tokens
        t_emb_seq = t_emb.unsqueeze(1)  # [B, 1, D]
        state_base_token = state_base_token + t_emb_seq
        state_joint_token = state_joint_token + t_emb_seq

        x = torch.cat([state_base_token, state_joint_token, action_tokens], dim=1)  # [B, T+2, D]

        # 6. Shared DiT blocks
        for block in self.shared_blocks:
            x = block(x, all_vision)

        # 7. Split: take action portion [B, T, D]
        h = x[:, 2:, :]  # [B, T, D]

        # 8. Branch processing
        # state_cond for base = joint_state_token, for joint = base_state_token
        state_joint_cond = state_joint_token  # [B, 1, D]
        state_base_cond = state_base_token    # [B, 1, D]

        h_base = self.forward_branch(h, state_joint_cond, V_base, self.base_blocks)
        h_joint = self.forward_branch(h, state_base_cond, V_joint, self.joint_blocks)

        # 9. Output heads
        base_vel = self.base_head(h_base)     # [B, T, base_dim]
        joint_vel = self.joint_head(h_joint)   # [B, T, joint_dim+gripper_dim]

        return {
            'base_vel': base_vel,
            'joint_vel': joint_vel,
            'g_t': g_t,
            'h_base': h_base,
            'h_joint': h_joint,
        }
