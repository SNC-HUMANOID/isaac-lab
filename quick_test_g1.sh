#!/bin/bash

echo "============================================"
echo "Quick Test: Humanoid 10102025 with G1 Limits"
echo "============================================"
echo ""

# Test environment with minimal setup
./isaaclab.sh -p scripts/environments/random_agent.py \
    --task Isaac-Humanoid-10102025-Direct-v0 \
    --num_envs 2 \
    --headless

