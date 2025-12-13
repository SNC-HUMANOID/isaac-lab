#!/bin/bash

# RViz2 wrapper script to avoid snap library conflicts
unset SNAP
unset SNAP_DATA
unset SNAP_COMMON

# Remove all snap paths from LD_LIBRARY_PATH
if [ -n "$LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed 's|/snap[^:]*:*||g' | sed 's/:*$//' | sed 's/^:*//')
fi

# Preload system pthread library to avoid snap conflicts
export LD_PRELOAD="/lib/x86_64-linux-gnu/libpthread.so.0"

# Source ROS2 environment
source /home/code/ros2_humble/install/setup.bash

# Add workspace to AMENT_PREFIX_PATH
export AMENT_PREFIX_PATH="/home/code/Humanoid/install/humanoid_snc:$AMENT_PREFIX_PATH"

# Run RViz2
exec /home/code/ros2_humble/install/rviz2/lib/rviz2/rviz2 "$@"