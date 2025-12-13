# ROS 2 Integration for IsaacLab Humanoid Policy

This directory contains tools for deploying IsaacLab-trained policies on real robots using ROS 2.

## Files

1. **`export_policy_torchscript.py`** - Export RSL-RL checkpoint to TorchScript
2. **`humanoid_policy_node.py`** - ROS 2 node for real-time policy inference
3. **`test_policy_export.py`** - Test script to verify exported policy
4. **`robot_config.yaml`** - Robot-specific configuration

## Quick Start

### 1. Export Your Trained Policy

```bash
cd /home/code/SNC Lab/isaac-lab/IsaacLab

# Export checkpoint to TorchScript
python ros2/export_policy_torchscript.py \
    --checkpoint logs/rsl_rl/humanoid_10102025/2025-11-13_08-57-58/model_9050.pt \
    --output ros2/humanoid_policy.pt
```

This will create `humanoid_policy.pt` which includes:
- Policy network weights
- Observation normalization (mean/std)
- Optimized for inference (no training components)

### 2. Test the Exported Policy (Optional)

```bash
# Test that policy loads and runs correctly
python ros2/test_policy_export.py --policy ros2/humanoid_policy.pt
```

### 3. Configure for Your Robot

Edit `robot_config.yaml` to match your robot:
- Joint names and order
- Default positions (standing pose)
- Joint limits
- Action scale
- Control rate

### 4. Run ROS 2 Node

**Method A: Direct Python execution (for testing)**
```bash
python ros2/humanoid_policy_node.py \
    --ros-args \
    -p policy_path:=ros2/humanoid_policy.pt \
    -p control_rate:=200.0 \
    -p enable_safety:=true
```

**Method B: Create ROS 2 package (for deployment)**
```bash
# Create package
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python humanoid_control

# Copy files
cp /home/code/SNC Lab/isaac-lab/IsaacLab/ros2/humanoid_policy_node.py \
   ~/ros2_ws/src/humanoid_control/humanoid_control/
cp /home/code/SNC Lab/isaac-lab/IsaacLab/ros2/humanoid_policy.pt \
   ~/ros2_ws/src/humanoid_control/config/

# Build and run
cd ~/ros2_ws
colcon build --packages-select humanoid_control
source install/setup.bash
ros2 run humanoid_control humanoid_policy_node
```

## Architecture

### Observation Flow
```
Robot Sensors → ROS Topics → build_observation() → Normalize → Policy → Actions → Joint Commands
```

### Expected ROS 2 Topics

**Subscriptions:**
- `/joint_states` (sensor_msgs/JointState) - Current joint positions and velocities
- `/imu/data` (sensor_msgs/Imu) - IMU measurements (ang_vel, lin_acc)
- `/cmd_vel` (geometry_msgs/Twist) - Velocity commands from teleop/planner

**Publications:**
- `/joint_position_command` (std_msgs/Float32MultiArray) - Target joint positions

### Observation Structure

The observation **must match exactly** what was used during training:

```python
obs = [
    joint_pos_normalized,  # 21 elements (joint_pos - default_pos)
    joint_vel,             # 21 elements
    base_ang_vel,          # 3 elements (from IMU)
    gravity_vec,           # 3 elements (normalized lin_acc from IMU)
    command,               # 3 elements [vx, vy, yaw_rate]
]
# Total: 51 elements
```

⚠️ **CRITICAL**: Verify your training observation size matches!

Check in: `logs/rsl_rl/humanoid_10102025/2025-11-13_08-57-58/params/env.yaml`

## Configuration Parameters

Edit in `humanoid_policy_node.py` or pass as ROS parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `policy_path` | `ros2/humanoid_policy.pt` | Path to TorchScript policy |
| `control_rate` | `200.0` | Control loop frequency (Hz) |
| `enable_safety` | `true` | Enable safety checks and filters |
| `max_joint_vel` | `10.0` | Max joint velocity (rad/s) |
| `action_scale` | `0.25` | Action scaling factor |

## Safety Features

The node includes several safety mechanisms:

1. **Joint Limit Clipping** - Prevents exceeding hardware limits
2. **Velocity Limiting** - Limits max joint velocity change per timestep
3. **Action Filtering** - Moving average filter to smooth actions
4. **Sensor Timeout** - Stops if no sensor data received
5. **Emergency Stop** - Graceful shutdown on Ctrl+C

### Testing Safely

**Start with reduced action scale:**
```bash
ros2 run humanoid_control humanoid_policy_node \
    --ros-args -p action_scale:=0.1  # 40% of training scale
```

**Monitor in real-time:**
```bash
# Watch joint commands
ros2 topic echo /joint_position_command

# Watch robot state
ros2 topic echo /joint_states
```

## Troubleshooting

### Policy loads but robot doesn't move well

**Check observation structure:**
```python
# Add debug output in humanoid_policy_node.py control_loop():
self.get_logger().info(f"Obs shape: {obs.shape}, Action: {action}")
```

**Verify normalization:**
- Check that IMU frame matches training (world frame vs body frame)
- Verify gravity vector points down (~[0, 0, -9.81])
- Check command scaling matches training

**Adjust action scale:**
- Start low (0.1) and gradually increase
- Training used 0.25, real robot may need 0.15-0.2

### Robot is unstable

**Check control rate:**
- Must match training rate (typically 50-200 Hz)
- Verify ROS timer is actually achieving target rate

**Increase safety margins:**
```python
self.max_joint_vel = 5.0  # Reduce from 10.0
self.action_history = deque(maxlen=5)  # Increase filtering
```

**Check joint mapping:**
- Verify joint_names order matches training
- Test individual joints in isolation

### Observation size mismatch

```
Error: Expected obs size 51, got 48
```

**Solution:**
1. Check `params/env.yaml` in your training logs
2. Count observation components in your training config
3. Adjust `build_observation()` to match exactly

## Advanced: Custom Observation Structure

If your training used different observations, modify `build_observation()`:

```python
def build_observation(self):
    # Example: Adding contact sensors
    obs = np.concatenate([
        joint_pos_normalized,
        joint_vel,
        base_ang_vel,
        gravity_vec,
        command,
        foot_contacts,  # Add this if used in training
    ])
    return obs
```

## Deployment Checklist

- [ ] Export policy with correct checkpoint
- [ ] Test exported policy loads without errors
- [ ] Verify observation size matches training
- [ ] Configure joint names and limits for your robot
- [ ] Test with reduced action_scale first (0.1)
- [ ] Check IMU orientation and gravity vector
- [ ] Verify control loop runs at target frequency
- [ ] Test emergency stop (Ctrl+C)
- [ ] Gradually increase action_scale
- [ ] Test on different terrains/commands

## References

- IsaacLab Documentation: https://isaac-sim.github.io/IsaacLab/
- RSL-RL: https://github.com/leggedrobotics/rsl_rl
- ROS 2 Humble: https://docs.ros.org/en/humble/

## Support

For issues with:
- **Policy export**: Check IsaacLab logs and checkpoint structure
- **ROS 2 integration**: Verify topic names and message types
- **Robot behavior**: Review observation normalization and action scaling

Good luck with your deployment! 🤖🚀
