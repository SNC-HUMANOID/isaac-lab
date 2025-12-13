#!/bin/bash

# Retrain Humanoid SNC R5 with fixed URDF and improved reward configuration
cd /home/sncbot/IsaacLab
source /home/sncbot/isaac-sim/setup_conda_env.sh

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Humanoid-SNC-R5-v0 \
    --num_envs 1024 \
    --headless
