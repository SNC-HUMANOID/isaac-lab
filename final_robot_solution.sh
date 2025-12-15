#!/bin/bash

echo "🤖 FINAL HUMANOID_SNC ROBOT SOLUTION"
echo "===================================="
echo ""
echo "PROBLEM: Isaac Sim GUI hangs/freezes"
echo "SOLUTION: Generate scene file for manual opening"
echo ""

# Fix environment
export LD_PRELOAD=""
unset LD_PRELOAD
export DISPLAY=:0

echo "✅ Environment configured"
echo "🔄 Generating robot scene with upright orientation..."
echo ""

# Generate the scene file
./isaaclab.sh -p simple_robot_image.py

echo ""
echo "="*50
echo "🎉 SOLUTION COMPLETE!"
echo "="*50
echo ""
echo "📁 FILES CREATED:"
echo "   • Robot scene: robot_scene_upright.usd"
echo "   • Original USD: Humaniod_SNC_ROS_URDF/Humanoid_SNC/usd/Humanoid_SNC.usd"
echo ""
echo "🔧 TO VIEW THE ROBOT:"
echo "1. Open Isaac Sim manually:"
echo "   cd /home/sncbot/isaac-sim && ./isaac-sim.sh"
echo ""
echo "2. In Isaac Sim GUI:"
echo "   File → Open → Select robot_scene_upright.usd"
echo ""
echo "3. Press F to focus on robot"
echo ""
echo "🎯 ROBOT SHOULD APPEAR:"
echo "   ✅ Standing upright (not lying down)"
echo "   ✅ Head pointing up"
echo "   ✅ Feet on ground"
echo "   ✅ Proper humanoid pose"
echo ""
echo "📖 For more solutions see: GUI_HANG_SOLUTIONS.md"
echo "="*50