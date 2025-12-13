#!/bin/bash

# Simple test script to verify Humanoid 10102025 setup
# This script tests the robot model loading and basic visualization

echo "========================================="
echo "Testing Humanoid 10102025 Setup"
echo "========================================="

# Test 1: Check if USD file exists
echo ""
echo "Test 1: Checking USD file..."
if [ -f "humanoid_10102025_assets/Humanoid_10102025.usd" ]; then
    echo "✓ USD file found"
    ls -lh humanoid_10102025_assets/Humanoid_10102025.usd
else
    echo "✗ USD file not found!"
    exit 1
fi

# Test 2: Check robot configuration
echo ""
echo "Test 2: Checking robot configuration..."
if [ -f "source/isaaclab_assets/isaaclab_assets/robots/humanoid_10102025.py" ]; then
    echo "✓ Robot configuration found"
else
    echo "✗ Robot configuration not found!"
    exit 1
fi

# Test 3: Print robot info
echo ""
echo "Test 3: Robot Information"
echo "-------------------------"
echo "Robot name: Humanoid 10102025"
echo "Joints: 20 actuated joints"
echo "  - Legs: 12 joints (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll) x2"
echo "  - Arms: 8 joints (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow) x2"
echo ""
echo "USD Path: $(pwd)/humanoid_10102025_assets/Humanoid_10102025.usd"
echo ""

# Test 4: Test Python import
echo "Test 4: Testing Python imports..."
./isaaclab.sh -p -c "from isaaclab_assets.robots.humanoid_10102025 import HUMANOID_10102025_CFG; print('✓ Robot config imported successfully')" 2>&1 | grep "✓"

if [ $? -eq 0 ]; then
    echo "✓ Python imports working"
else
    echo "✗ Python imports failed"
    exit 1
fi

echo ""
echo "========================================="
echo "All tests passed! ✓"
echo "========================================="
echo ""
echo "Ready to train! Use one of these commands:"
echo ""
echo "  # Train with existing humanoid environment (quick test):"
echo "  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Humanoid-Direct-v0 --num_envs 64 --headless"
echo ""
echo "  # Create custom environment (advanced):"
echo "  # Edit the provided training scripts in the repository root"
echo ""
