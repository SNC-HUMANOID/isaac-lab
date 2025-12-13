#!/bin/bash

echo "======================================"
echo "Testing Humanoid_10102025 USD Model"
echo "======================================"
echo ""

# Check if USD files exist
echo "1. Checking USD files..."
if [ -f "/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/Humanoid_10102025.usd" ]; then
    echo "   ✓ Main USD file found"
else
    echo "   ✗ Main USD file NOT found"
    exit 1
fi

if [ -f "/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/configuration/Humanoid_10102025_base.usd" ]; then
    echo "   ✓ Base USD file found"
else
    echo "   ✗ Base USD file NOT found"
    exit 1
fi

echo ""
echo "2. File sizes:"
ls -lh /home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/ | grep -E "\.usd$|\.yaml$"
echo ""

echo "3. Configuration files:"
ls -lh /home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/configuration/
echo ""

echo "======================================"
echo "✓ All checks passed!"
echo "======================================"
echo ""
echo "To view the robot model, run:"
echo "  ./isaaclab.sh -p view_humanoid_10102025.py"
echo ""
echo "Or to view with GUI headless mode off:"
echo "  ./isaaclab.sh -p view_humanoid_10102025.py"
echo ""

