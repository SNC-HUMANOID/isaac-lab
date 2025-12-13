#!/bin/bash

echo "🤖 Opening SNC R5 UPDATE Model in Isaac Sim..."
echo "================================================"

URDF_PATH="/home/sncbot/Downloads/SNC_R5_UPDATE/Humanoid_SNC.urdf"

if [ ! -f "$URDF_PATH" ]; then
    echo "❌ URDF not found: $URDF_PATH"
    exit 1
fi

echo "📁 URDF: $URDF_PATH"
echo "📁 STL Directory: /home/sncbot/Downloads/SNC_R5_UPDATE/"
echo ""
echo "🚀 Launching Isaac Sim..."
echo ""
echo "📋 To import the robot:"
echo "  1. File → Import → URDF"
echo "  2. Browse to: $URDF_PATH"
echo "  3. Click 'Import'"
echo ""

# Launch Isaac Sim
cd /home/sncbot/Downloads/SNC_R5_UPDATE
/home/sncbot/isaac-sim/isaac-sim.sh
