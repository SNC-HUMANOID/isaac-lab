#!/bin/bash

# Training script for Humanoid 10102025

echo "========================================"
echo "Training Humanoid 10102025 with RSL-RL"
echo "========================================"

# Default parameters
NUM_ENVS=4096
MAX_ITERATIONS=5000
HEADLESS="--headless"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gui)
            HEADLESS=""
            shift
            ;;
        --num_envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --max_iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run training
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Humanoid-10102025-Direct-v0 \
    --num_envs ${NUM_ENVS} \
    --max_iterations ${MAX_ITERATIONS} \
    ${HEADLESS}

echo ""
echo "Training complete!"
echo "Logs saved to: logs/rsl_rl/isaac_humanoid_10102025_direct/"

