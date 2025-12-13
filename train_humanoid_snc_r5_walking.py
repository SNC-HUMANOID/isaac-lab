#!/usr/bin/env python3

"""Train SNC R5 humanoid for natural walking with improved arm movement."""

import argparse

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train SNC R5 humanoid for natural walking")
    parser.add_argument("--headless", action="store_true", default=False,
                        help="Force display off at all times")
    parser.add_argument("--num_envs", type=int, default=2048,
                        help="Number of environments to simulate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed used for the environment")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume training from checkpoint")
    parser.add_argument("--load_run", type=str, default="",
                        help="Load run directory for resuming")
    
    args = parser.parse_args()
    
    print("🚶 Training SNC R5 for Natural Walking")
    print("=" * 50)
    print(f"Environment: Isaac-Humanoid-SNC-R5-Walking-Direct-v0")
    print(f"Number of environments: {args.num_envs}")
    print(f"Seed: {args.seed}")
    print(f"Headless: {args.headless}")
    
    if args.resume:
        print(f"Resuming from: {args.load_run}")
    
    print("\nKey improvements in this environment:")
    print("✅ Natural gait cycle rewards")
    print("✅ Coordinated arm swing (opposite to legs)")  
    print("✅ Forward velocity targets")
    print("✅ Torso stability optimization")
    print("✅ Energy-efficient movement")
    print("✅ Improved termination conditions")
    print("\nStarting training...")

if __name__ == "__main__":
    main()