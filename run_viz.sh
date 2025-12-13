#!/bin/bash

echo "🤖 Starting Humanoid_SNC Robot Visualization..."
echo "📱 This will open Isaac Sim GUI to show your robot"
echo "⏳ Please wait for Isaac Sim to load (may take 30-60 seconds)"
echo "❌ Press Ctrl+C to cancel if needed"
echo ""

# Run the FIXED visualization script
./isaaclab.sh -p fixed_robot_viewer.py

echo "👋 Visualization closed."