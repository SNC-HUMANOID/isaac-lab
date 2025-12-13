# 🤖 Robot Orientation Fix - Standing Upright

## ❌ Problem Identified
Your Humanoid_SNC robot is **lying down along the X-axis** instead of standing upright. This is a common issue with URDF files exported from SolidWorks.

## ✅ Solutions Available

### Method 1: Quick Fix with Rotation (Recommended)
```bash
./run_standing_robot.sh
```
- **What it does**: Rotates robot 90° around Y-axis in Isaac Sim
- **Result**: Robot stands upright immediately
- **Pros**: Fast, no file modification needed
- **Cons**: Temporary fix (need to apply each time)

### Method 2: Manual Loading with Rotation
1. Open Isaac Sim: `./run_isaac_gui.sh`
2. Load USD file manually
3. In Isaac Sim, select robot
4. In Properties panel → Transform → Rotation
5. Set Y rotation to **90 degrees**

### Method 3: Test Multiple Orientations
```bash
export LD_PRELOAD="" && ./isaaclab.sh -p test_robot_orientations.py
```
- Shows 6 different robot orientations
- Find which one looks best standing upright
- Usually **Rotate_Y_90** is correct

## 🎯 Expected Standing Pose

When fixed correctly, you should see:
- ✅ **Head pointing UP** (toward +Z axis)
- ✅ **Feet pointing DOWN** (toward ground)
- ✅ **Arms at sides** (extending in ±Y direction)  
- ✅ **Robot facing forward** (along X or Y axis)
- ✅ **Upright humanoid posture**

## 🔧 Technical Details

**Original Problem:**
- Robot base_link oriented with Z-axis along robot's "front"
- SolidWorks export typically has different coordinate system
- Isaac Sim expects Z-up coordinate system

**Fix Applied:**
- Rotation: 90° around Y-axis
- Formula: `rpy="0 1.5708 0"` (in radians)
- Coordinate transform: X→Z, Y→Y, Z→-X

## 🚀 Quick Start

**Just want to see standing robot?**
```bash
./run_standing_robot.sh
```

**Want to understand the problem?**
- Original: Robot lying on side
- Fixed: Robot standing up like a person
- Key: Y-axis rotation to correct orientation

## 📋 Files Created

- `fixed_standing_robot.py` - Main fix script
- `run_standing_robot.sh` - Easy launcher
- `test_robot_orientations.py` - Test multiple poses
- This guide - `ROBOT_ORIENTATION_FIX.md`

Your robot will now stand proudly like a proper humanoid! 🤖✨