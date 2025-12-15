#!/bin/bash

echo "🔧 Fixing Isaac Sim TLS memory allocation error..."

# Method 1: Set LD_PRELOAD to fix TLS issue
export LD_PRELOAD="/lib/x86_64-linux-gnu/libdl.so.2"

echo "✅ Set LD_PRELOAD environment variable"

# Method 2: Alternative - set RTLD_DEEPBIND
export PYTHONPATH="/home/sncbot/isaac-sim/kit/python/lib/python3.10/site-packages:$PYTHONPATH"

echo "✅ Updated PYTHONPATH"

# Method 3: Try with different memory allocation
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_TOP_PAD_=131072

echo "✅ Set memory allocation parameters"

# Now try to run Isaac Sim
echo ""
echo "🚀 Attempting to start Isaac Sim with fixes..."
echo "If this works, you should see Isaac Sim GUI opening"

cd /home/sncbot/IsaacLab
./isaaclab.sh -s