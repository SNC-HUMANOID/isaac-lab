#!/usr/bin/env python3
"""Simple script to analyze SNC R5 training results."""

import os
import torch
import numpy as np
from pathlib import Path

def analyze_checkpoint(checkpoint_path):
    """Analyze a single checkpoint file."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"\n=== Checkpoint Analysis: {checkpoint_path.name} ===")
        
        # Print keys available in checkpoint
        print("Available keys:", list(checkpoint.keys()))
        
        if 'model_state_dict' in checkpoint:
            model_dict = checkpoint['model_state_dict']
            print(f"Model parameters count: {len(model_dict)}")
            
            # Count total parameters
            total_params = 0
            for key, tensor in model_dict.items():
                params = tensor.numel() if hasattr(tensor, 'numel') else 0
                total_params += params
                if 'actor' in key and len(key.split('.')) <= 3:  # Show main actor layers
                    print(f"  {key}: {tensor.shape}")
                elif 'critic' in key and len(key.split('.')) <= 3:  # Show main critic layers
                    print(f"  {key}: {tensor.shape}")
            
            print(f"Total parameters: {total_params:,}")
        
        if 'optimizer_state_dict' in checkpoint:
            print("Optimizer state included: Yes")
        
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
        
        if 'total_steps' in checkpoint:
            print(f"Total steps: {checkpoint['total_steps']}")
            
        return True
        
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return False

def main():
    """Main analysis function."""
    # Find the training results directory
    log_dir = Path("/home/sncbot/IsaacLab/logs/rsl_rl/humanoid_snc_r5_standing/2025-09-22_13-18-43")
    
    if not log_dir.exists():
        print(f"Training directory not found: {log_dir}")
        return
    
    print("🚀 SNC R5 Humanoid Training Results Analysis")
    print("=" * 50)
    
    # List all model files
    model_files = sorted(log_dir.glob("model_*.pt"))
    print(f"\nFound {len(model_files)} model checkpoints")
    
    # Analyze first, middle, and last checkpoints
    checkpoints_to_analyze = []
    if model_files:
        checkpoints_to_analyze.append(model_files[0])  # First
        if len(model_files) > 2:
            checkpoints_to_analyze.append(model_files[len(model_files)//2])  # Middle
        if len(model_files) > 1:
            checkpoints_to_analyze.append(model_files[-1])  # Last
    
    for checkpoint_path in checkpoints_to_analyze:
        analyze_checkpoint(checkpoint_path)
    
    # Check for tensorboard logs
    tb_files = list(log_dir.glob("events.out.tfevents.*"))
    if tb_files:
        print(f"\n📊 Tensorboard logs found: {len(tb_files)} files")
        print("To view training curves, run:")
        print(f"tensorboard --logdir {log_dir}")
    
    # Check configuration files
    params_dir = log_dir / "params"
    if params_dir.exists():
        config_files = list(params_dir.glob("*.yaml"))
        print(f"\n⚙️  Configuration files: {len(config_files)}")
        for config_file in config_files:
            print(f"  - {config_file.name}")
    
    print("\n✅ Analysis complete!")
    print("\nNext steps:")
    print("1. Run tensorboard to view training curves")
    print("2. Test the trained policy with play.py script")
    print("3. Evaluate performance on different tasks")

if __name__ == "__main__":
    main()