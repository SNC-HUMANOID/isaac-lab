#!/bin/bash

# Unitree G1 Training Examples with Monitoring
# Usage examples for training G1 robot in Isaac Lab

echo "==============================================="
echo "UNITREE G1 TRAINING EXAMPLES"
echo "==============================================="

# Basic G1 training commands
echo "Available G1 training tasks:"
echo "- Isaac-Velocity-Flat-G1-v0 (flat terrain)"
echo "- Isaac-Velocity-Rough-G1-v0 (rough terrain)"
echo ""

# Example 1: Basic G1 training with monitoring
echo "Example 1: Basic G1 flat terrain training"
echo "Command: python3 training_monitor_wrapper.py --task Isaac-Velocity-Flat-G1-v0 --framework rsl_rl"
echo ""

# Example 2: G1 rough terrain training
echo "Example 2: G1 rough terrain training"  
echo "Command: python3 training_monitor_wrapper.py --task Isaac-Velocity-Rough-G1-v0 --framework rsl_rl --headless"
echo ""

# Example 3: G1 training with dashboard monitoring
echo "Example 3: G1 training with GUI dashboard"
echo "Command: python3 training_monitor_wrapper.py --task Isaac-Velocity-Flat-G1-v0 --framework rsl_rl --monitor-mode dashboard"
echo ""

# Example 4: Play trained G1 model
echo "Example 4: Play trained G1 model"
echo "Command: ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-G1-v0 --load_run logs/rsl_rl/[run_folder]/"
echo ""

# Quick commands section
echo "==============================================="
echo "QUICK COMMANDS"
echo "==============================================="

# Function to run G1 training
run_g1_flat() {
    echo "Starting G1 flat terrain training with monitoring..."
    python3 training_monitor_wrapper.py --task Isaac-Velocity-Flat-G1-v0 --framework rsl_rl --save-stats
}

run_g1_rough() {
    echo "Starting G1 rough terrain training with monitoring..."
    python3 training_monitor_wrapper.py --task Isaac-Velocity-Rough-G1-v0 --framework rsl_rl --save-stats --headless
}

run_g1_dashboard() {
    echo "Starting G1 training with dashboard monitoring..."
    python3 training_monitor_wrapper.py --task Isaac-Velocity-Flat-G1-v0 --framework rsl_rl --monitor-mode dashboard
}

# Check arguments
case "$1" in
    "flat")
        run_g1_flat
        ;;
    "rough") 
        run_g1_rough
        ;;
    "dashboard")
        run_g1_dashboard
        ;;
    "check")
        echo "Checking system status..."
        python3 quick_gpu_check.py
        ;;
    "monitor")
        echo "Starting system monitoring..."
        python3 training_analysis.py --mode monitor
        ;;
    *)
        echo "Usage: $0 {flat|rough|dashboard|check|monitor}"
        echo ""
        echo "Commands:"
        echo "  flat      - Train G1 on flat terrain"
        echo "  rough     - Train G1 on rough terrain"  
        echo "  dashboard - Train G1 with GUI monitoring"
        echo "  check     - Quick system check"
        echo "  monitor   - Start system monitoring"
        ;;
esac