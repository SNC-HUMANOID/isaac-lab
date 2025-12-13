#!/bin/bash

echo "🤖 Opening SNC R5 Robot in Isaac Sim..."
echo "=========================================="

# Path to URDF
URDF_PATH="/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_fixed_knees.urdf"

if [ ! -f "$URDF_PATH" ]; then
    echo "❌ URDF not found: $URDF_PATH"
    exit 1
fi

echo "📁 URDF: $URDF_PATH"
echo ""
echo "Opening Isaac Sim..."
echo "You can manually import the URDF:"
echo "  1. File → Import → URDF"
echo "  2. Select: $URDF_PATH"
echo ""

# Launch Isaac Sim GUI
/home/sncbot/isaac-sim/isaac-sim.sh
