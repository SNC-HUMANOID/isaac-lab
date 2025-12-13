"""Custom action term that inverts left hip pitch to fix axis issue."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class InvertedJointPositionActionCfg(ActionTermCfg):
    """Configuration for inverted joint position action."""

    class_type: type = None  # Will be set below

    scale: float = 1.0
    """Scale for the action."""

    offset: float = 0.0
    """Offset for the action."""

    use_default_offset: bool = True
    """Whether to use default joint positions as offset."""

    invert_joints: list[str] = []
    """List of joint name patterns to invert (flip sign)."""


class InvertedJointPositionAction(ActionTerm):
    """Joint position action with ability to invert specific joints.

    This fixes the hip axis issue by inverting left hip pitch actions.
    """

    cfg: InvertedJointPositionActionCfg

    def __init__(self, cfg: InvertedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Get asset
        self._asset: Articulation = env.scene[cfg.asset_name]

        # Create buffers
        self._raw_actions = torch.zeros(env.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # Default positions
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, cfg.joint_ids].clone()
        else:
            self._offset = torch.zeros_like(self._raw_actions)
            if cfg.offset is not None:
                self._offset += cfg.offset

        # Find joints to invert
        self._invert_mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self.device)
        joint_names = self._asset.data.joint_names

        for pattern in cfg.invert_joints:
            import re
            for i, idx in enumerate(cfg.joint_ids):
                if idx < len(joint_names):
                    if re.match(pattern, joint_names[idx]):
                        self._invert_mask[i] = True
                        print(f"[InvertedAction] Will invert: {joint_names[idx]}")

    @property
    def action_dim(self) -> int:
        return len(self.cfg.joint_ids)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        # Store raw
        self._raw_actions[:] = actions

        # Scale
        self._processed_actions = self.cfg.scale * actions

        # Invert specified joints
        self._processed_actions[:, self._invert_mask] *= -1.0

        # Add offset
        self._processed_actions += self._offset

    def apply_actions(self):
        self._asset.set_joint_position_target(
            self._processed_actions,
            joint_ids=self.cfg.joint_ids
        )


# Register the class
InvertedJointPositionActionCfg.class_type = InvertedJointPositionAction
