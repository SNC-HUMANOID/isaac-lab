#!/bin/bash

# Play trained Humanoid SNC R5 model
# Usage: ./play_snc_r5.sh

# ใช้ run name เท่านั้น (ไม่ใช่ full path)
RUN_NAME="2025-10-03_08-30-38"
FULL_PATH="logs/rsl_rl/humanoid_snc_r5_velocity/$RUN_NAME"

echo "🤖 Playing Humanoid SNC R5 Trained Model"
echo "=========================================="
echo "Model: $FULL_PATH"
echo ""

# Check if model exists
if [ ! -d "$FULL_PATH" ]; then
    echo "❌ Error: Model directory not found: $FULL_PATH"
    exit 1
fi

# Find latest checkpoint
LATEST_MODEL=$(ls -t "$FULL_PATH"/model_*.pt 2>/dev/null | head -1)
if [ -z "$LATEST_MODEL" ]; then
    echo "❌ Error: No model checkpoints found in $FULL_PATH"
    exit 1
fi

echo "📦 Latest checkpoint: $(basename $LATEST_MODEL)"
echo ""

# Run with GUI (not headless)
# ⚠️ --load_run ต้องเป็น run name เท่านั้น (ไม่ใช่ full path)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-Humanoid-SNC-R5-v0 --num_envs 16 --load_run "$RUN_NAME"

# Alternative: Run headless for testing
# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-Humanoid-SNC-R5-v0 --num_envs 16 --load_run "$RUN_NAME" --headless
