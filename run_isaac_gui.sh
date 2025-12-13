#!/bin/bash

echo "🖥️ Isaac Sim GUI Launcher - FIXED VERSION"
echo "========================================="

# Fix TLS memory allocation issue
export LD_PRELOAD=""
unset LD_PRELOAD

echo "✅ Fixed TLS memory allocation issue"
echo "🚀 Opening Isaac Sim GUI..."
echo ""
echo "Manual Loading Instructions:"
echo "1. Wait for Isaac Sim to fully load"
echo "2. Go to: File → Open"
echo "3. Navigate to:"
echo "   /home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/usd/Humanoid_SNC.usd"
echo "4. Click Open"
echo "5. Press F to focus on robot"
echo ""
echo "Loading Isaac Sim..."

# Run Isaac Sim GUI with fix
LD_PRELOAD="" ./isaaclab.sh -p open_isaac_sim.py

echo ""
echo "👋 Isaac Sim closed."