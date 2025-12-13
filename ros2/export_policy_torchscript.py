#!/usr/bin/env python3
"""
Export RSL-RL policy checkpoint to TorchScript for deployment on real robot.

Usage:
    python export_policy_torchscript.py \
        --checkpoint logs/rsl_rl/humanoid_10102025/2025-11-13_08-57-58/model_9050.pt \
        --output ros2/humanoid_policy.pt
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch
import yaml


class PolicyForInference(torch.nn.Module):
    """Lightweight policy wrapper for inference - includes normalization."""
    
    def __init__(self, actor_state_dict, obs_mean=None, obs_std=None):
        super().__init__()
        
        # Rebuild actor network from state dict
        self.actor = self._rebuild_actor(actor_state_dict)
        
        # Register normalization buffers
        if obs_mean is not None and obs_std is not None:
            self.register_buffer('obs_mean', obs_mean)
            self.register_buffer('obs_std', obs_std)
            self.normalize = True
        else:
            self.normalize = False
            print("WARNING: No observation normalization found in checkpoint!")
    
    def _rebuild_actor(self, state_dict):
        """Rebuild actor network from state dict."""
        layers = []
        
        # Get all layer indices (e.g., 0, 2, 4, 6 for network with activations)
        layer_indices = sorted(set(int(k.split('.')[0]) for k in state_dict.keys() if 'weight' in k))
        
        for i, layer_idx in enumerate(layer_indices):
            weight_key = f'{layer_idx}.weight'
            bias_key = f'{layer_idx}.bias'
            
            # Get layer dimensions
            out_features, in_features = state_dict[weight_key].shape
            
            # Create linear layer
            linear = torch.nn.Linear(in_features, out_features)
            linear.weight.data = state_dict[weight_key]
            linear.bias.data = state_dict[bias_key]
            layers.append(linear)
            
            # Add activation (ELU) except for last layer
            if i < len(layer_indices) - 1:
                layers.append(torch.nn.ELU())
        
        return torch.nn.Sequential(*layers)
    
    def forward(self, obs):
        """
        Forward pass with normalization.
        
        Args:
            obs: [batch, obs_dim] or [obs_dim] observation tensor
        
        Returns:
            action: [batch, action_dim] or [action_dim] action tensor
        """
        # Handle single observation (no batch dim)
        squeeze_output = False
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze_output = True
        
        # Normalize observation
        if self.normalize:
            obs = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        
        # Forward through actor
        action = self.actor(obs)
        
        # Remove batch dim if input was single observation
        if squeeze_output:
            action = action.squeeze(0)
        
        return action


def _load_yaml(path: Path):
    """Load YAML file, handling python-specific tags if present."""
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as fp:
        contents = fp.read()

    try:
        return yaml.safe_load(contents)
    except yaml.constructor.ConstructorError:
        try:
            # Fallback to the full loader for python-specific tags (e.g. tuples).
            return yaml.load(contents, Loader=yaml.FullLoader)
        except yaml.constructor.ConstructorError as exc:
            warnings.warn(
                f"Failed to parse YAML file '{path}': {exc}. Continuing without this file.",
                RuntimeWarning,
            )
            return None


def load_params(params_dir):
    """Load environment and agent parameters."""
    params = {}

    env_cfg = _load_yaml(params_dir / "env.yaml")
    if env_cfg is not None:
        params["env"] = env_cfg

    agent_cfg = _load_yaml(params_dir / "agent.yaml")
    if agent_cfg is not None:
        params["agent"] = agent_cfg

    return params


def export_policy(checkpoint_path, output_path):
    """
    Export policy from checkpoint to TorchScript.
    
    Args:
        checkpoint_path: Path to model_XXXX.pt checkpoint
        output_path: Path to save TorchScript policy
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract components
    if 'model_state_dict' in ckpt:
        model_state = ckpt['model_state_dict']
    else:
        raise ValueError(f"Cannot find 'model_state_dict' in checkpoint. Available keys: {ckpt.keys()}")
    
    # Extract actor weights
    actor_state = {}
    for key, value in model_state.items():
        if key.startswith('actor.'):
            # Remove 'actor.' prefix
            new_key = key.replace('actor.', '')
            actor_state[new_key] = value
    
    if not actor_state:
        raise ValueError("No actor weights found in model_state_dict!")
    
    # Get normalization stats
    obs_mean = ckpt.get('obs_mean', None)
    obs_std = ckpt.get('obs_std', None)
    
    # Load params
    params_dir = checkpoint_path.parent / "params"
    params = load_params(params_dir)
    
    # Print info
    print(f"\n{'='*60}")
    print(f"Checkpoint Information:")
    print(f"{'='*60}")
    print(f"  Iteration: {ckpt.get('iter', 'unknown')}")
    print(f"  Actor layers: {len([k for k in actor_state.keys() if 'weight' in k])}")
    
    # Infer dimensions from actor network
    first_weight = [v for k, v in actor_state.items() if 'weight' in k and '0.weight' in k][0]
    last_weight = [v for k, v in actor_state.items() if 'weight' in k][-1]
    obs_dim = first_weight.shape[1]
    action_dim = last_weight.shape[0]
    
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Normalization: {'Yes' if obs_mean is not None else 'No'}")
    
    if params.get('env'):
        print(f"\nEnvironment Config:")
        env_cfg = params['env']
        if 'scene' in env_cfg:
            print(f"  Num envs: {env_cfg['scene'].get('num_envs', 'N/A')}")
        if 'sim' in env_cfg:
            print(f"  Sim dt: {env_cfg['sim'].get('dt', 'N/A')}")
    
    if params.get('agent'):
        print(f"\nAgent Config:")
        agent_cfg = params['agent']
        if 'policy' in agent_cfg:
            print(f"  Hidden dims: {agent_cfg['policy'].get('actor_hidden_dims', 'N/A')}")
            print(f"  Activation: {agent_cfg['policy'].get('activation', 'N/A')}")
    
    print(f"{'='*60}\n")
    
    # Create policy module
    print("Building policy module...")
    policy = PolicyForInference(actor_state, obs_mean, obs_std)
    policy.eval()
    
    # Test with dummy input
    print("Testing policy with dummy observation...")
    dummy_obs = torch.randn(1, obs_dim)
    with torch.no_grad():
        dummy_action = policy(dummy_obs)
    print(f"  ✓ Output shape: {dummy_action.shape}")
    print(f"  ✓ Action range: [{dummy_action.min():.3f}, {dummy_action.max():.3f}]")
    
    # Trace to TorchScript
    print("\nTracing to TorchScript...")
    traced_policy = torch.jit.trace(policy, dummy_obs)

    # Prepare metadata for downstream tooling
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "iteration": ckpt.get("iter", None),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "has_normalization": bool(obs_mean is not None and obs_std is not None),
    }

    # Save with metadata using TorchScript extra files
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced_policy.save(
        str(output_path),
        _extra_files={"metadata.json": json.dumps(metadata)},
    )
    print(f"✓ Saved TorchScript policy to: {output_path}")
    
    # Verify by loading
    print("\nVerifying saved policy...")
    extra_files = {"metadata.json": ""}
    loaded_policy = torch.jit.load(str(output_path), _extra_files=extra_files)
    with torch.no_grad():
        verify_action = loaded_policy(dummy_obs)
    
    if torch.allclose(dummy_action, verify_action, atol=1e-6):
        print("✓ Verification passed! Policy loaded correctly.")
    else:
        print("WARNING: Verification failed! Actions don't match.")

    if extra_files["metadata.json"]:
        print("✓ Metadata embedded in TorchScript file.")
    else:
        print("WARNING: Metadata missing in TorchScript file.")
    
    print(f"\n{'='*60}")
    print("Export complete!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. Copy {output_path.name} to your robot")
    print(f"2. Use with ROS 2 node: humanoid_policy_node.py")
    print(f"3. Make sure to use the same observation structure as training")
    print(f"4. Apply action_scale and default_joint_pos from your robot config")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export RSL-RL policy checkpoint to TorchScript for deployment"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (model_XXXX.pt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for TorchScript policy (default: same dir as checkpoint)"
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        checkpoint_path = Path(args.checkpoint)
        output_path = checkpoint_path.parent / "policy_torchscript.pt"
    else:
        output_path = Path(args.output)
    
    try:
        export_policy(args.checkpoint, output_path)
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
