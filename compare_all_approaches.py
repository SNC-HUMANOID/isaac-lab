#!/usr/bin/env python3
"""
📊 Compare All Training Approaches
Run different configs and see which works best
"""

import subprocess
import time
import os

def run_training(script_name, description, args="", max_time=300):
    """Run a training script for limited time and capture results"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {description}")
    print(f"Script: {script_name}")
    print(f"Args: {args}")
    print(f"Max time: {max_time} seconds")
    print(f"{'='*60}")

    cmd = f"./isaaclab.sh -p {script_name} {args} --headless --num_envs 128"

    try:
        # Run for limited time
        process = subprocess.Popen(
            cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/home/sncbot/IsaacLab"
        )

        start_time = time.time()
        output_lines = []

        # Monitor output for max_time seconds
        while time.time() - start_time < max_time:
            output = process.stdout.readline()
            if output:
                output_lines.append(output.strip())
                print(output.strip())

                # Check for success indicators
                if "Mean reward:" in output:
                    # Extract reward
                    try:
                        reward_part = output.split("Mean reward:")[1].strip()
                        reward = float(reward_part.split()[0])
                        if reward > -100 and reward < 100:  # Reasonable reward
                            print(f"✅ Good reward found: {reward}")
                    except:
                        pass

                # Check for failure indicators
                if "nan" in output.lower() or "inf" in output.lower():
                    print(f"❌ NaN/inf detected!")
                    break

            elif process.poll() is not None:
                break

        # Kill process
        process.terminate()
        process.wait(timeout=5)

        return output_lines

    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return []

def main():
    print("📊 Comparing All Humanoid Training Approaches")
    print("Will test each approach for 5 minutes...")

    approaches = [
        ("train_humanoid_super_safe.py", "🛡️ Super Safe (very conservative)", ""),
        ("train_humanoid_fast_safe.py", "⚡ Fast & Safe (balanced)", ""),
        ("train_humanoid_simple_movement.py", "🚶 Simple Movement (movement focus)", "--speed 0.5"),
        ("scripts/reinforcement_learning/rsl_rl/train.py", "🏃 Official G1 (for comparison)", "--task Isaac-Velocity-Flat-G1-v0"),
    ]

    results = {}

    for script, description, args in approaches:
        print(f"\n🔄 Starting test: {description}")
        output = run_training(script, description, args, max_time=300)  # 5 minutes each
        results[description] = output

        # Brief pause between tests
        print(f"\n⏸️  Pausing 10 seconds before next test...")
        time.sleep(10)

    # Summary
    print(f"\n{'='*80}")
    print("📊 FINAL COMPARISON SUMMARY")
    print(f"{'='*80}")

    for description, output in results.items():
        print(f"\n{description}:")

        # Look for last reward line
        last_reward = None
        has_nan = False

        for line in output:
            if "Mean reward:" in line:
                try:
                    reward_part = line.split("Mean reward:")[1].strip()
                    last_reward = float(reward_part.split()[0])
                except:
                    pass
            if "nan" in line.lower() or "inf" in line.lower():
                has_nan = True

        if has_nan:
            print("   ❌ Status: FAILED (NaN/inf detected)")
        elif last_reward is not None:
            if -10 <= last_reward <= 20:
                print(f"   ✅ Status: GOOD (reward: {last_reward:.2f})")
            elif last_reward > 20:
                print(f"   ⚠️  Status: HIGH REWARD (reward: {last_reward:.2f})")
            else:
                print(f"   ⚠️  Status: LOW REWARD (reward: {last_reward:.2f})")
        else:
            print("   ❓ Status: UNKNOWN (no reward data)")

    print(f"\n🎯 RECOMMENDATION:")
    print("Use the approach marked as ✅ GOOD for your actual training!")

if __name__ == "__main__":
    main()