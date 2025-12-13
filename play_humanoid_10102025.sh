#!/bin/bash

# Play trained Humanoid 10102025 policy
# Usage: ./play_humanoid_10102025.sh [checkpoint_path]

if [ -z "$1" ]; then
    echo "Usage: ./play_humanoid_10102025.sh <checkpoint_path>"
    echo ""
    echo "Example:"
    echo "  ./play_humanoid_10102025.sh logs/rsl_rl/humanoid_10102025/2025-11-03_13-00-00/model_1000.pt"
    echo ""
    echo "Or to use the latest checkpoint:"
    echo "  ./play_humanoid_10102025.sh logs/rsl_rl/humanoid_10102025/2025-11-03_13-00-00/"
    exit 1
fi

CHECKPOINT_PATH=$1

echo "========================================="
echo "Playing Humanoid 10102025 Policy"
echo "========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo ""

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Humanoid-10102025-Direct-v0 \
    --num_envs 16 \
    --load_run "$CHECKPOINT_PATH"
