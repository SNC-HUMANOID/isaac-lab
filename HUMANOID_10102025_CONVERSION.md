# Humanoid_10102025 URDF to USD Conversion

## Summary

Successfully converted the Humanoid_10102025 URDF model to USD format for use in Isaac Lab.

## Files Created

### 1. USD Files
- **Main USD**: `/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/Humanoid_10102025.usd`
- **Base USD**: `/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/configuration/Humanoid_10102025_base.usd` (95 MB)
- **Physics USD**: `/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/configuration/Humanoid_10102025_physics.usd` (8.0 KB)
- **Sensor USD**: `/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/configuration/Humanoid_10102025_sensor.usd` (657 B)
- **Config YAML**: `/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/config.yaml`

### 2. Viewer Script
- **Script**: `/home/sncbot/IsaacLab/view_humanoid_10102025.py`

## Robot Structure

### Joints (20 DOF)

**Left Leg (6 DOF):**
- left_hip_pitch_joint
- left_hip_roll_joint
- left_hip_yaw_joint
- left_knee_joint
- left_ankle_pitch_joint
- left_ankle_roll_joint

**Right Leg (6 DOF):**
- right_hip_pitch_joint
- right_hip_roll_joint
- right_hip_yaw_joint
- right_knee_joint
- right_ankle_pitch_joint
- right_ankle_roll_joint

**Left Arm (4 DOF):**
- left_shoulder_pitch_joint
- left_shoulder_roll_joint
- left_shoulder_yaw_joint
- left_elbow_joint

**Right Arm (4 DOF):**
- right_shoulder_pitch_joint
- right_shoulder_roll_joint
- right_shoulder_yaw_joint
- right_elbow_joint

### Sensors
- **IMU**: imu_link (attached to pelvis_link)

## Usage

### 1. View the Robot Model

```bash
./isaaclab.sh -p view_humanoid_10102025.py
```

### 2. Use in Isaac Lab Configuration

```python
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

robot_cfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/Humanoid_10102025.usd",
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "left_knee_joint": 0.2,
            "left_ankle_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            "right_knee_joint": 0.2,
            "right_ankle_pitch_joint": -0.1,
        },
    ),
    actuators={
        "legs": sim_utils.ImplicitActuatorCfg(
            joint_names_expr=[
                ".*hip.*",
                ".*knee.*",
                ".*ankle.*",
            ],
            stiffness=25.0,
            damping=5.0,
        ),
    },
)
```

## Conversion Command

The conversion was performed using:

```bash
./isaaclab.sh -p scripts/tools/convert_urdf.py \
  /home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/urdf/Humanoid_10102025.urdf \
  /home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/Humanoid_10102025.usd \
  --merge-joints
```

## Next Steps

1. **Visualize**: Run the viewer script to inspect the model
2. **Create Asset Config**: Create a robot asset configuration in `isaaclab_assets`
3. **Create Environment**: Set up an RL environment for training
4. **Train Policy**: Train a locomotion policy using RSL-RL or other frameworks

## Joint Limits and Efforts

All joints have:
- **Lower limit**: -3.142 rad (-180°)
- **Upper limit**: 3.142 rad (180°)
- **Velocity**: 20 rad/s

**Effort limits:**
- Hip/Shoulder joints: 25-27 Nm
- Knee/Elbow joints: 12-25 Nm
- Ankle joints: 12 Nm

## Mass Properties

- **Pelvis**: 3.40 kg (main body)
- **Total estimated**: ~4.5 kg (lightweight humanoid)

