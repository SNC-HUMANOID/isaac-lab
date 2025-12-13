"""Gym wrapper to invert left hip pitch action."""

import torch
import gymnasium as gym


class LeftHipInverterWrapper(gym.Wrapper):
    """Inverts left_hip_pitch_joint action to fix URDF axis bug.

    This wrapper intercepts actions BEFORE they reach the environment
    and inverts the left hip pitch action to compensate for the inverted
    axis in the URDF file.
    """

    def __init__(self, env):
        super().__init__(env)
        self.left_hip_idx = None

        # Find left hip pitch index in the unwrapped environment
        unwrapped = env.unwrapped
        if hasattr(unwrapped, 'action_manager'):
            joint_action_term = unwrapped.action_manager._terms.get('joint_pos')
            if joint_action_term and hasattr(joint_action_term, '_joint_names'):
                for i, name in enumerate(joint_action_term._joint_names):
                    if name == "left_hip_pitch_joint":
                        self.left_hip_idx = i
                        print(f"\n{'='*60}")
                        print(f"[LeftHipInverterWrapper] Found left_hip_pitch_joint at index {i}")
                        print(f"[LeftHipInverterWrapper] Will INVERT action to fix URDF axis bug")
                        print(f"{'='*60}\n")
                        break

        if self.left_hip_idx is None:
            print(f"\n[LeftHipInverterWrapper] WARNING: Could not find left_hip_pitch_joint!")
            print(f"[LeftHipInverterWrapper] Wrapper will NOT invert any actions\n")

    def step(self, action):
        """Invert left hip pitch action before sending to environment."""
        if self.left_hip_idx is not None:
            # Clone to avoid modifying original
            action = action.clone() if isinstance(action, torch.Tensor) else action.copy()
            # Invert left hip pitch
            action[:, self.left_hip_idx] *= -1.0

        return self.env.step(action)
