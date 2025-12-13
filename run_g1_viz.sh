#!/bin/bash

# G1 Model Visualization Runner
# ใช้สำหรับดูการทำงานของโมเดล AI ใน Isaac Sim

echo "🤖 Starting G1 Model Visualization..."
echo "================================================"

# Default parameters
MODEL_PATH="logs/rsl_rl/g1_flat/2025-07-29_15-27-26/model_300.pt"
NUM_ENVS=1
HEADLESS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --num_envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --headless)
            HEADLESS=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model PATH      Path to trained model (default: latest G1 model)"
            echo "  --num_envs N      Number of environments (default: 1)"
            echo "  --headless        Run without GUI"
            echo "  --help            Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run with default settings"
            echo "  $0 --num_envs 4                      # Run with 4 robots"
            echo "  $0 --model logs/.../model_200.pt     # Use specific model"
            echo "  $0 --headless                        # Run without GUI"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model file not found: $MODEL_PATH"
    echo "Available models:"
    find logs/rsl_rl/g1_flat/ -name "*.pt" 2>/dev/null | head -5
    exit 1
fi

echo "📁 Model: $MODEL_PATH"
echo "🔢 Environments: $NUM_ENVS"
echo "👁️  Headless: $HEADLESS"
echo ""

# Build command
CMD="./isaaclab.sh -p visualize_g1_model.py"
CMD="$CMD --model_path $MODEL_PATH"
CMD="$CMD --num_envs $NUM_ENVS"

if [ "$HEADLESS" = true ]; then
    CMD="$CMD --headless"
fi

echo "🚀 Running command:"
echo "$CMD"
echo ""

# Run the visualization
exec $CMD