# 🔧 Isaac Sim GUI Hang Solutions

## ❌ Problem
Isaac Sim GUI **hangs/freezes** and doesn't display properly. The application starts but the window either:
- Doesn't appear
- Appears but is frozen/unresponsive
- Shows black screen
- Hangs with loading message

## ✅ Solutions (Try in Order)

### Solution 1: Force GUI with Display Settings
```bash
export DISPLAY=:0
export QT_X11_NO_MITSHM=1
export LD_PRELOAD=""
./isaaclab.sh -p fixed_standing_robot.py
```

### Solution 2: Manual USD File Opening
```bash
# 1. Generate scene file
export LD_PRELOAD="" && ./isaaclab.sh -p simple_robot_image.py

# 2. Open Isaac Sim GUI manually
./isaaclab.sh -s

# 3. In Isaac Sim: File → Open → Select generated USD file
```

### Solution 3: Use Isaac Sim Directly
```bash
# Open Isaac Sim without Isaac Lab
cd /home/sncbot/isaac-sim
export LD_PRELOAD=""
./isaac-sim.sh

# Then File → Open → /home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/usd/Humanoid_SNC.usd
```

### Solution 4: Check System Resources
```bash
# Check memory and processes
free -h
ps aux | grep isaac

# Kill hanging processes
pkill -f isaac
```

### Solution 5: Alternative Viewers
```bash
# Use USD viewer tools (if available)
usdview /home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/usd/Humanoid_SNC.usd

# Or use Blender with USD plugin
# Or use other 3D viewers that support USD format
```

## 🔍 Debug Information

**Working Parts:**
- ✅ Isaac Sim starts in headless mode
- ✅ Robot USD files load correctly  
- ✅ Scene creation works
- ✅ Orientation fix applies correctly

**Problem Area:**
- ❌ GUI rendering/display
- ❌ Window management
- ❌ Graphics pipeline freeze

## 🎯 Root Causes (Most Common)

1. **X11/Display Issues**: Remote desktop or SSH without proper display forwarding
2. **GPU/Graphics**: Vulkan/OpenGL conflicts with remote systems
3. **Memory**: Insufficient GPU memory or system RAM
4. **Process Conflicts**: Previous Isaac Sim instances still running
5. **Environment**: Missing GUI libraries or display server issues

## 📋 Quick Checklist

- [ ] Kill all existing Isaac processes: `pkill -f isaac`
- [ ] Check display: `echo $DISPLAY` should show `:0` or similar
- [ ] Test X11: `xeyes` or `xclock` should work
- [ ] Check memory: `free -h` should show sufficient RAM
- [ ] Try headless first: Add `--headless` flag
- [ ] Use alternative viewer if available

## 🎉 Workaround Summary

**If GUI won't work:**
1. Generate scene with `simple_robot_image.py`
2. Open generated USD file manually
3. Robot will be positioned upright and ready to view
4. All fixes (orientation, position) are preserved in the USD file

The robot **IS** working correctly - it's just a GUI display issue! 🤖✨