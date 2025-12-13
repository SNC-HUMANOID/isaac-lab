#!/bin/bash

echo "================================================"
echo "  Verifying Humanoid 10102025 G1 Setup"
echo "================================================"
echo ""

# Check URDF
echo "✓ Checking URDF files..."
if [ -f "/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/urdf/Humanoid_10102025_g1_limits.urdf" ]; then
    echo "  ✅ G1 URDF exists"
else
    echo "  ❌ G1 URDF NOT found"
fi

# Check USD
echo ""
echo "✓ Checking USD files..."
if [ -f "/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/Humanoid_10102025_g1.usd" ]; then
    echo "  ✅ Main G1 USD exists"
else
    echo "  ❌ Main G1 USD NOT found"
fi

if [ -f "/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/configuration/Humanoid_10102025_g1_base.usd" ]; then
    echo "  ✅ Base G1 USD exists ($(du -h /home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/configuration/Humanoid_10102025_g1_base.usd | cut -f1))"
else
    echo "  ❌ Base G1 USD NOT found"
fi

# Check Robot Config
echo ""
echo "✓ Checking robot configuration..."
if grep -q "Humanoid_10102025_g1.usd" /home/sncbot/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/humanoid_10102025.py; then
    echo "  ✅ Robot config using G1 USD"
else
    echo "  ⚠️  Robot config NOT using G1 USD"
fi

# Check Environment
echo ""
echo "✓ Checking environment registration..."
if grep -q "Isaac-Humanoid-10102025-Direct-v0" /home/sncbot/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_10102025/__init__.py; then
    echo "  ✅ Environment registered"
else
    echo "  ❌ Environment NOT registered"
fi

echo ""
echo "================================================"
echo "  Summary"
echo "================================================"
echo ""
echo "Files ready:"
echo "  📁 URDF: Humanoid_10102025_g1_limits.urdf"
echo "  📁 USD:  Humanoid_10102025_g1.usd"
echo "  📁 Config: humanoid_10102025.py (updated)"
echo ""
echo "Ready to train with G1 joint limits!"
echo ""
echo "Run: ./train_humanoid_10102025.sh"
echo ""

