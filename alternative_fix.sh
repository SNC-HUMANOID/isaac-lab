#!/bin/bash

echo "🔧 Alternative Isaac Sim TLS Fix..."

# Alternative method - disable problematic libraries loading
export LD_PRELOAD=""
unset LD_PRELOAD

# Set specific environment variables for Isaac Sim
export OMNI_KIT_ACCEPT_EULA=yes
export ISAACSIM_PATH=/home/sncbot/isaac-sim

# Try different memory management
ulimit -v unlimited 2>/dev/null || echo "Could not set unlimited virtual memory"

echo "Environment variables set:"
echo "- LD_PRELOAD cleared"  
echo "- EULA accepted"
echo "- Isaac Sim path set"

echo ""
echo "🚀 Trying to start Isaac Sim directly..."

# Try direct isaac-sim command with environment fixes
cd /home/sncbot/isaac-sim
LD_PRELOAD="" ./isaac-sim.sh --no-window --headless &
ISAAC_PID=$!

sleep 5

if kill -0 $ISAAC_PID 2>/dev/null; then
    echo "✅ Isaac Sim is running in background (PID: $ISAAC_PID)"
    echo "Now trying Isaac Lab..."
    kill $ISAAC_PID
    cd /home/sncbot/IsaacLab
    LD_PRELOAD="" ./isaaclab.sh -p open_isaac_sim.py
else
    echo "❌ Isaac Sim failed to start"
    echo "Let's try the Isaac Lab approach with environment fixes..."
    cd /home/sncbot/IsaacLab
    LD_PRELOAD="" MALLOC_MMAP_THRESHOLD_=131072 ./isaaclab.sh -s
fi