#!/bin/bash

# Simple G1 Model Test using existing play script
echo "🤖 Testing G1 Model with Isaac Lab Play Script"
echo "=============================================="

# Check if model exists
MODEL_PATH="logs/rsl_rl/g1_flat/2025-07-29_15-27-26/model_300.pt"
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    exit 1
fi

echo "📁 Model: $MODEL_PATH"
echo "🎯 Task: Isaac-Velocity-Flat-G1-v0"
echo ""

# Check available tasks
echo "🔍 Available G1 tasks:"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --help | grep -i g1 || echo "No G1 tasks found in help"

echo ""
echo "🚀 Attempting to run with Isaac Lab play script..."

# Try to find the right task name
if ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-G1-v0 --load_run logs/rsl_rl/g1_flat/2025-07-29_15-27-26/ --num_envs 1 --headless; then
    echo "✅ Success with Isaac-Velocity-Flat-G1-v0"
elif ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-G1-Flat-v0 --load_run logs/rsl_rl/g1_flat/2025-07-29_15-27-26/ --num_envs 1 --headless; then
    echo "✅ Success with Isaac-G1-Flat-v0"
elif ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Humanoid-v0 --load_run logs/rsl_rl/g1_flat/2025-07-29_15-27-26/ --num_envs 1 --headless; then
    echo "✅ Success with Isaac-Humanoid-v0"
else
    echo "❌ Failed with standard task names"
    echo ""
    echo "💡 Let's check what tasks are actually available:"
    find /home/sncbot/IsaacLab/source/isaaclab_tasks -name "*g1*" -type f | head -5
    echo ""
    echo "🔍 Searching for environment configs:"
    find /home/sncbot/IsaacLab/source/isaaclab_tasks -name "*_env_cfg.py" | grep -i g1 | head -3
fi