#!/usr/bin/env python3
"""Script to inspect checkpoint parameters."""

import torch
import sys


def print_dict(d, indent=0):
    """Recursively print dictionary."""
    if not isinstance(d, dict):
        print("  " * indent + str(d))
        return
    
    for k, v in d.items():
        if isinstance(v, dict):
            print("  " * indent + f"{k}:")
            print_dict(v, indent + 1)
        elif isinstance(v, (list, tuple)) and len(v) > 5:
            print("  " * indent + f"{k}: [{type(v).__name__} len={len(v)}]")
        else:
            print("  " * indent + f"{k}: {v}")


if len(sys.argv) < 2:
    print("Usage: python inspect_checkpoint.py <checkpoint_path>")
    sys.exit(1)

ckpt_path = sys.argv[1]
print(f"Loading checkpoint: {ckpt_path}")

ckpt = torch.load(ckpt_path, map_location="cpu")

print("\n" + "="*80)
print("CHECKPOINT CONTENTS")
print("="*80)

if isinstance(ckpt, dict):
    print(f"\nTop-level keys: {list(ckpt.keys())}")
    
    # Check iteration
    for k in ["iter", "it", "iteration", "iter_num", "step"]:
        if k in ckpt:
            print(f"\nIteration: {ckpt[k]}")
            break
    
    # Check for config/params
    if "train_cfg" in ckpt:
        print("\n" + "-"*80)
        print("TRAINING CONFIG:")
        print("-"*80)
        cfg = ckpt["train_cfg"]
        print_dict(cfg, indent=0)
    
    if "env_cfg" in ckpt:
        print("\n" + "-"*80)
        print("ENVIRONMENT CONFIG:")
        print("-"*80)
        cfg = ckpt["env_cfg"]
        print_dict(cfg, indent=0)
        
    # Look for reward weights in config
    if "env_cfg" in ckpt and isinstance(ckpt["env_cfg"], dict):
        env = ckpt["env_cfg"]
        if "rewards" in env:
            print("\n" + "-"*80)
            print("REWARD WEIGHTS:")
            print("-"*80)
            rewards = env["rewards"]
            if isinstance(rewards, dict):
                for name, cfg in rewards.items():
                    if isinstance(cfg, dict) and "weight" in cfg:
                        print(f"  {name}: {cfg['weight']}")
    
    # State dicts
    for key in ["model_state_dict", "actor_critic_state_dict", "policy_state_dict"]:
        if key in ckpt:
            sd = ckpt[key]
            if isinstance(sd, dict):
                total_params = sum(t.numel() for t in sd.values() if hasattr(t, "numel"))
                print(f"\n{key}: {len(sd)} tensors, {total_params:,} parameters")
                break

else:
    print(f"Checkpoint type: {type(ckpt)}")
    print("Not a standard dict checkpoint")
