#!/bin/bash

# Quick test script for Humanoid 10102025 locomotion environment
# This will run a quick training test with 4 environments for 10 iterations

echo "========================================="
echo "Testing Humanoid 10102025 Locomotion"
echo "========================================="
echo ""
echo "This is a quick test to verify the setup works."
echo "For full training, use: train_humanoid_10102025_locomotion.py"
echo ""

./isaaclab.sh -p train_humanoid_10102025_locomotion.py \
    --num_envs 4 \
    --seed 42

echo ""
echo "========================================="
echo "Test completed!"
echo "========================================="
