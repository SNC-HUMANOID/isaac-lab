#!/usr/bin/env python3
"""
Test script to verify exported TorchScript policy.

Usage:
    python ros2/test_policy_export.py --policy ros2/humanoid_policy.pt
"""

import argparse
import json
import warnings
from typing import Optional

import numpy as np
import torch


def test_policy(policy_path):
    """Test that exported policy loads and runs correctly."""
    
    print(f"\n{'='*60}")
    print(f"Testing TorchScript Policy")
    print(f"{'='*60}\n")
    
    print(f"Loading policy from: {policy_path}")
    
    extra_files = {"metadata.json": ""}
    try:
        policy = torch.jit.load(policy_path, map_location='cpu', _extra_files=extra_files)
        policy.eval()
        print("✓ Policy loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load policy: {e}")
        return False

    metadata = {}
    if extra_files.get("metadata.json"):
        try:
            metadata = json.loads(extra_files["metadata.json"])
            print(f"✓ Loaded metadata: obs_dim={metadata.get('obs_dim')}, action_dim={metadata.get('action_dim')}")
        except json.JSONDecodeError as exc:
            warnings.warn(f"Failed to parse metadata.json: {exc}")
    
    # Check if policy has normalization buffers
    has_normalization = False
    for name, buffer in policy.named_buffers():
        if 'obs_mean' in name or 'obs_std' in name:
            has_normalization = True
            print(f"  Found buffer: {name} with shape {buffer.shape}")
    
    if has_normalization:
        print("✓ Policy includes observation normalization\n")
    else:
        if metadata.get("has_normalization"):
            print("⚠ WARNING: Normalization expected (per metadata) but buffers missing!\n")
        else:
            print("⚠ WARNING: No observation normalization found!\n")
    
    # Infer observation dimension
    inferred_obs_dim: Optional[int] = metadata.get("obs_dim") if metadata else None
    inferred_action_dim: Optional[int] = metadata.get("action_dim") if metadata else None

    # Try to inspect the scripted graph for the input tensor shape (PyTorch 2.x)
    try:
        graph = policy.inlined_graph if hasattr(policy, "inlined_graph") else policy.graph
        for node in graph.inputs():
            node_type = node.type()
            if hasattr(node_type, "sizes") and node_type.sizes():
                sizes = list(node_type.sizes())
                if len(sizes) == 2 and sizes[0] is None and isinstance(sizes[1], int):
                    inferred_obs_dim = sizes[1]
                    break
    except Exception as exc:
        warnings.warn(f"Unable to infer observation dimension from TorchScript graph: {exc}")

    # Fallback brute-force probing if graph inspection failed
    probe_dims = [48, 51, 54, 57, 60, 63, 66]
    if inferred_obs_dim is None:
        for candidate in probe_dims:
            try:
                dummy_obs = torch.randn(1, candidate)
                with torch.no_grad():
                    action = policy(dummy_obs)
                inferred_obs_dim = candidate
                inferred_action_dim = action.shape[1]
                break
            except RuntimeError:
                continue

    if inferred_obs_dim is None:
        print("✗ Could not infer observation dimension")
        return False

    # Ensure we have the action dimension
    if inferred_action_dim is None:
        dummy_obs = torch.randn(1, inferred_obs_dim)
        with torch.no_grad():
            action = policy(dummy_obs)
        inferred_action_dim = action.shape[1]

    print(f"✓ Observation dimension: {inferred_obs_dim}")
    print(f"✓ Action dimension: {inferred_action_dim}\n")
    
    # Test multiple random observations
    print("Running inference tests with random observations...")
    test_results = []
    
    for _ in range(10):
        obs = torch.randn(1, inferred_obs_dim)
        with torch.no_grad():
            action = policy(obs)
        test_results.append(action.numpy())
    
    test_results = np.array(test_results).squeeze()
    
    print(f"  ✓ Completed {len(test_results)} inference calls")
    print(f"  Action statistics:")
    print(f"    Mean: {test_results.mean(axis=0)[:5]} ... (first 5)")
    print(f"    Std:  {test_results.std(axis=0)[:5]} ... (first 5)")
    print(f"    Min:  {test_results.min():.3f}")
    print(f"    Max:  {test_results.max():.3f}\n")
    
    # Test single observation (no batch)
    print("Testing single observation input (no batch dimension)...")
    try:
        obs_single = torch.randn(inferred_obs_dim)
        with torch.no_grad():
            action_single = policy(obs_single)
        print(f"  ✓ Single obs input: {obs_single.shape} → {action_single.shape}\n")
    except Exception as e:
        print(f"  ⚠ Single obs input failed: {e}\n")
    
    # Performance test
    print("Running performance test...")
    import time
    
    obs_batch = torch.randn(1, inferred_obs_dim)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = policy(obs_batch)
    
    # Benchmark
    n_iters = 1000
    start_time = time.time()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = policy(obs_batch)
    elapsed = time.time() - start_time
    
    freq = n_iters / elapsed
    latency = elapsed / n_iters * 1000  # ms
    
    print(f"  ✓ Inference frequency: {freq:.1f} Hz")
    print(f"  ✓ Average latency: {latency:.2f} ms\n")
    
    if freq < 50:
        print(f"  ⚠ WARNING: Inference too slow for real-time control (< 50 Hz)")
    elif freq < 200:
        print(f"  ⚠ WARNING: May not achieve 200 Hz control rate")
    else:
        print(f"  ✓ Performance sufficient for real-time control")
    
    print(f"\n{'='*60}")
    print("All tests passed! ✓")
    print(f"{'='*60}\n")
    
    print("Next steps:")
    print("1. Update joint_names in humanoid_policy_node.py to match your robot")
    print("2. Update default_joint_pos and joint_limits")
    print("3. Verify observation structure matches training")
    print("4. Test with reduced action_scale first (0.1)")
    print()
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test exported TorchScript policy")
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Path to TorchScript policy (.pt file)"
    )
    
    args = parser.parse_args()
    
    success = test_policy(args.policy)
    
    if not success:
        print("\n✗ Tests failed!")
        exit(1)
    else:
        print("✓ All tests passed!")
        exit(0)


if __name__ == "__main__":
    main()
