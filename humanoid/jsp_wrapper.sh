#!/bin/bash
unset GTK_PATH
unset LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
export LD_PRELOAD=/lib/x86_64-linux-gnu/libpthread.so.0
exec /opt/ros/humble/lib/joint_state_publisher_gui/joint_state_publisher_gui "$@"
