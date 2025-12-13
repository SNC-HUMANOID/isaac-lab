# Humanoid_SNC Robot Viewing Guide

## Problem: Robot scripts hang or don't show anything

The issue is likely that Isaac Sim is having trouble with automated robot loading. Here's the **MANUAL** approach that will definitely work:

## ✅ WORKING SOLUTION - Manual Loading

### Step 1: Open Isaac Sim GUI
```bash
./isaaclab.sh -p open_isaac_sim.py
```

### Step 2: Manual File Loading
1. Wait for Isaac Sim GUI to fully load (30-60 seconds)
2. In Isaac Sim menu: **File → Open**
3. Navigate to: `/home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/usd/Humanoid_SNC.usd`
4. Click **Open**
5. Press **F** key to focus on robot
6. Use mouse to explore the robot

## 🎮 Navigation Controls
- **Mouse drag**: Rotate camera
- **Mouse wheel**: Zoom in/out  
- **Middle click + drag**: Pan view
- **F key**: Focus on selected object
- **Alt + Mouse drag**: Orbit around selection

## 📁 Robot Files Location
- **USD file**: `/home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/usd/Humanoid_SNC.usd`
- **Original URDF**: `/home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/urdf/Humanoid_SNC.urdf`
- **Meshes**: `/home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/meshes/`

## 🔧 Alternative Methods (if automated scripts work)

### Method 1: Fixed Viewer
```bash
./isaaclab.sh -p fixed_robot_viewer.py
```

### Method 2: Simple Scripts
```bash
./run_viz.sh
```

## 🐛 Troubleshooting
- **Program hangs**: Use the manual loading method above
- **Can't see robot**: Press F to focus, then zoom out
- **Isaac Sim won't open**: Check that you have GUI/X11 forwarding enabled
- **Robot looks wrong**: The robot should appear as grey/silver colored

## 🎉 Expected Result
You should see the **Humanoid_SNC robot** - a humanoid robot with:
- Head/torso (base_link)
- Two arms with shoulder, elbow joints
- Two legs with hip, knee, ankle joints
- Grey/metallic appearance

The robot will be displayed in Isaac Sim's 3D viewport where you can explore it from all angles!