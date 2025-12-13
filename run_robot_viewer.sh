#!/bin/bash

echo "🤖 Humanoid_SNC Robot Viewer - FIXED VERSION"
echo "============================================="

# Fix TLS memory allocation issue
export LD_PRELOAD=""
unset LD_PRELOAD

echo "✅ Fixed TLS memory allocation issue"
echo "🚀 Starting Isaac Sim with robot viewer..."
echo ""
echo "This will:"
echo "1. Open Isaac Sim GUI (wait 30-60 seconds)"
echo "2. Load the Humanoid_SNC robot automatically"
echo "3. Show robot in 3D viewport"
echo ""
echo "Controls:"
echo "- Mouse drag: Rotate view"
echo "- Mouse wheel: Zoom"
echo "- F key: Focus on robot"
echo "- Close window to exit"
echo ""
echo "Loading..."

# Run the fixed robot viewer
LD_PRELOAD="" ./isaaclab.sh -p fixed_robot_viewer.py

echo ""
echo "👋 Robot viewer closed."