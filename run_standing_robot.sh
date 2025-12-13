#!/bin/bash

echo "🤖 Humanoid_SNC Robot - STANDING UPRIGHT FIX"
echo "============================================="

# Fix TLS memory allocation issue
export LD_PRELOAD=""
unset LD_PRELOAD

echo "✅ Fixed TLS memory allocation issue"
echo "🔄 Applying orientation fix for standing robot..."
echo ""
echo "Problem: Robot was lying down along X-axis"
echo "Solution: Rotate 90° around Y-axis to make it stand upright"
echo ""
echo "This will:"
echo "1. Open Isaac Sim GUI (wait 30-60 seconds)"
echo "2. Load robot with correct upright orientation"
echo "3. Position robot standing on ground plane"
echo ""
echo "Expected result:"
echo "✅ Robot standing upright (head up, feet down)"
echo "✅ Proper humanoid pose"
echo "✅ Not lying on its side"
echo ""
echo "Loading..."

# Run the fixed standing robot viewer
LD_PRELOAD="" ./isaaclab.sh -p fixed_standing_robot.py

echo ""
echo "👋 Standing robot viewer closed."