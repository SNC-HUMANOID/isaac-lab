"""Custom action term that inverts left hip pitch joint."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class InvertedLeftHipJointAction(JointPositionAction):
    """Joint position action that inverts the left hip pitch joint.

    This is a workaround for URDF axis inversion bug where left_hip_pitch_joint
    has opposite axis direction compared to right_hip_pitch_joint.
    """

    def __init__(self, cfg: JointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Find left hip pitch index
        self._left_hip_idx = None
        for i, name in enumerate(self._joint_names):
            if name == "left_hip_pitch_joint":
                self._left_hip_idx = i
                print(f"[InvertedLeftHipAction] Found left_hip_pitch_joint at index {i}")
                print(f"[InvertedLeftHipAction] Will INVERT action for this joint to fix axis bug")
                break

        if self._left_hip_idx is None:
            print(f"[InvertedLeftHipAction] WARNING: left_hip_pitch_joint not found in {self._joint_names}")

    def process_actions(self, actions: torch.Tensor):
        """Process actions - invert left hip pitch action AFTER scaling."""
        # First apply parent's scale and offset (stores in self._processed_actions)
        super().process_actions(actions)

        # Then invert left hip pitch in the processed actions
        if self._left_hip_idx is not None:
            self._processed_actions[:, self._left_hip_idx] *= -1.0
