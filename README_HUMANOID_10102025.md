# Humanoid 10102025 Training Setup

## Overview
This document describes the setup for training the Humanoid 10102025 robot in Isaac Lab.

## Robot Specifications
- **Name**: Humanoid 10102025
- **Total Joints**: 20 actuated joints
- **Legs**: 12 joints (6 per leg)
  - hip_pitch, hip_roll, hip_yaw
  - knee
  - ankle_pitch, ankle_roll
- **Arms**: 8 joints (4 per arm)
  - shoulder_pitch, shoulder_roll, shoulder_yaw
  - elbow
- **Sensors**: IMU (imu_joint) connected to pelvis

## Files Created

### 1. Robot Model Files
- **URDF**: `Humanoid_10102025/urdf/Humanoid_10102025.urdf` (with IMU sensor)
- **USD (with IMU)**: `humanoid_10102025_assets/Humanoid_10102025_with_imu.usd`
- **USD (original)**: `humanoid_10102025_assets/Humanoid_10102025.usd`

### 2. Robot Configuration
- **Config**: `source/isaaclab_assets/isaaclab_assets/robots/humanoid_10102025.py`
- **Registration**: Updated `source/isaaclab_assets/isaaclab_assets/robots/__init__.py`

### 3. Training Scripts
- **Simple Test**: `train_humanoid_10102025_simple.py` - Basic environment with random actions
- **Full Training**: `train_humanoid_10102025.py` - RSL-RL PPO training
- **Test Script**: `simple_test_humanoid_10102025.sh` - Verification script

## Usage

### Quick Test
Run the verification script to check if everything is set up correctly:
```bash
./simple_test_humanoid_10102025.sh
```

### Simple Training Test (Random Actions)
Test the environment with random actions:
```bash
# Headless mode
./isaaclab.sh -p train_humanoid_10102025_simple.py --headless --num_envs 64 --max_iterations 100

# With visualization
./isaaclab.sh -p train_humanoid_10102025_simple.py --num_envs 4 --max_iterations 50
```

### Full RL Training
Train with RSL-RL PPO algorithm:
```bash
# Headless training with 4096 environments
./isaaclab.sh -p train_humanoid_10102025.py --num_envs 4096 --max_iterations 2000 --headless

# Lower environment count for testing
./isaaclab.sh -p train_humanoid_10102025.py --num_envs 512 --max_iterations 500 --headless
```

### Play Trained Policy
After training, play back the learned policy:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Humanoid-10102025-Direct-v0 \
  --load_run logs/humanoid_10102025/rsl_rl/YYYY-MM-DD_HH-MM-SS/
```

## Robot Configuration Details

### Actuator Parameters

**Legs** (hip, knee, ankle joints):
- Hip joints: Stiffness 200.0, Damping 10.0
- Knee joints: Stiffness 200.0, Damping 10.0
- Ankle joints: Stiffness 40.0, Damping 5.0

**Arms** (shoulder, elbow joints):
- All arm joints: Stiffness 40.0, Damping 5.0

### Initial Pose
- **Height**: 1.0 meters above ground
- **Leg Configuration**: Slightly bent stance
  - hip_pitch: -0.1 rad
  - knee: 0.2 rad
  - ankle_pitch: -0.1 rad
- **Arm Configuration**: Neutral position (all joints at 0.0)

### Simulation Settings
- **Physics dt**: 0.005s (5ms)
- **Decimation**: 4 (control at 20ms = 50Hz)
- **Friction**: 1.0 (static and dynamic)
- **Restitution**: 0.0 (no bouncing)

## Environment Details

### Observation Space (74 dimensions)
- Root position (3)
- Root orientation quaternion (4)
- Root linear velocity (3)
- Root angular velocity (3)
- Joint positions (20)
- Joint velocities (20)
- Velocity commands (3)
- Previous actions (20)

### Action Space
- 20 continuous actions (one per joint)
- Action scale: 1.0

### Rewards
- **Linear velocity tracking**: 1.0
- **Angular velocity tracking**: 0.5
- **Joint acceleration penalty**: -2.5e-7
- **Action rate penalty**: -0.01
- **Alive reward**: 2.0
- **DOF at limit penalty**: -1.0
- **Feet air time reward**: 0.5

### Termination Conditions
- Robot falls (pelvis height < 0.5m or > 2.0m)
- Episode timeout (20 seconds)

## Training Output
Training logs are saved to:
```
logs/humanoid_10102025/rsl_rl/YYYY-MM-DD_HH-MM-SS/
```

This directory contains:
- Model checkpoints
- Training statistics (rewards, episode lengths)
- TensorBoard logs

## IMU Sensor
The IMU sensor is attached to the pelvis via a fixed joint named `imu_joint`.
- **Link name**: `imu_link`
- **Joint type**: Fixed
- **Parent**: `pelvis`
- **Mass**: 0.001 kg
- **Dimensions**: 1cm × 1cm × 0.5cm (visualized as red box)

## Troubleshooting

### Issue: USD file not found
**Solution**: Ensure the USD file exists at the path specified in the robot configuration:
```bash
ls -lh humanoid_10102025_assets/Humanoid_10102025.usd
```

### Issue: Training crashes or robot falls immediately
**Solution**: Try adjusting actuator parameters (stiffness/damping) or initial joint positions in the robot configuration file.

### Issue: Import errors
**Solution**: Ensure Isaac Lab extensions are installed:
```bash
./isaaclab.sh -i
```

## Next Steps

1. **Tune Rewards**: Adjust reward scales in the environment configuration
2. **Adjust Physics**: Modify actuator stiffness/damping for better stability
3. **Add Observations**: Include IMU data, contact forces, or terrain information
4. **Curriculum Learning**: Start with simple tasks and gradually increase difficulty
5. **Domain Randomization**: Add noise to physics parameters for robust policies

## References

- Isaac Lab Documentation: https://docs.omniverse.nvidia.com/isaacsim/latest/index.html
- RSL-RL: https://github.com/leggedrobotics/rsl_rl
- Original URDF: `Humanoid_10102025/urdf/Humanoid_10102025.urdf`
