#!/bin/bash

echo "Testing Humanoid SNC R5 Model"
echo "=============================="
echo ""
echo "Model: logs/rsl_rl/humanoid_snc_r5_velocity/2025-09-29_16-13-37/model_1499.pt"
echo "Task: Isaac-Velocity-Flat-Humanoid-SNC-R5-v0"
echo ""

# Run the test
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Humanoid-SNC-R5-v0 \
  --checkpoint $(pwd)/logs/rsl_rl/humanoid_snc_r5_velocity/2025-09-29_16-13-37/model_1499.pt \
  --num_envs 1 \
  --headless \
  2>&1 | grep -E "(Step|reward|Episode|Walking|Falling|Test)"