#!/bin/bash

# Test SNC R5 Trained Model
echo "🤖 Testing Humanoid SNC R5 Trained Model"
echo "=============================================="

# Model path
MODEL_DIR="/home/sncbot/IsaacLab/logs/rsl_rl/humanoid_snc_r5_velocity/2025-10-03_08-30-38"
CHECKPOINT="model_1499.pt"

if [ ! -f "$MODEL_DIR/$CHECKPOINT" ]; then
    echo "❌ Model not found: $MODEL_DIR/$CHECKPOINT"
    exit 1
fi

echo "📁 Model: $MODEL_DIR/$CHECKPOINT"
echo "🎯 Task: Isaac-Velocity-Flat-Humanoid-SNC-R5-v0"
echo "📊 Iteration: 1499 (final checkpoint)"
echo ""

# Try with headless mode first (faster)
echo "🚀 Running in headless mode (no GUI)..."
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Humanoid-SNC-R5-v0 \
    --load_run "$MODEL_DIR" \
    --checkpoint "$CHECKPOINT" \
    --num_envs 16 \
    --headless

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Test completed successfully!"
    echo ""
    echo "To run with GUI visualization:"
    echo "./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\"
    echo "    --task Isaac-Velocity-Flat-Humanoid-SNC-R5-v0 \\"
    echo "    --load_run $MODEL_DIR \\"
    echo "    --num_envs 4"
else
    echo ""
    echo "❌ Test failed"
    echo "Trying alternative task names..."

    # Try rough terrain version
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Velocity-Rough-Humanoid-SNC-R5-v0 \
        --load_run "$MODEL_DIR" \
        --checkpoint "$CHECKPOINT" \
        --num_envs 16 \
        --headless
fi
