#!/bin/bash

echo "========================================="
echo "Testing Humanoid 10102025 Environment"
echo "========================================="

# Test with minimal number of environments
./isaaclab.sh -p scripts/environments/random_agent.py \
    --task Isaac-Humanoid-10102025-Direct-v0 \
    --num_envs 4 \
    --headless

