# Humanoid_abs Training Setup

## Overview
This setup enables training the Humanoid_abs robot (from `/home/code/IsaacLab/source/isaaclab_assets/isaaclab_assets/humanoid_snc/urdf/Humanoid_abs.urdf`) to walk forward using reinforcement learning.

## Files Created

### 1. Robot Configuration
**File**: `/home/code/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/humanoid_abs.py`
- Defines the `HUMANOID_ABS_CFG` configuration
- Loads the URDF file
- Configures actuators for legs, feet, torso, and arms
- Sets initial joint positions for stable standing

### 2. Environment Configuration
**File**: `/home/code/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/humanoid_abs/flat_env_cfg.py`
- Defines `HumanoidAbsFlatEnvCfg` for training
- Defines `HumanoidAbsFlatEnvCfg_PLAY` for evaluation
- Custom reward function focused on forward walking
- Flat terrain (no obstacles)

### 3. Training Configuration
**File**: `/home/code/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/humanoid_abs/agents/rsl_rl_ppo_cfg.py`
- PPO hyperparameters optimized for bipedal locomotion
- 2000 max iterations
- Neural network: [256, 256, 128] hidden layers

### 4. Environment Registration
**File**: `/home/code/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/humanoid_abs/__init__.py`
- Registers two environments:
  - `Isaac-Velocity-Flat-Humanoid-Abs-v0` (training)
  - `Isaac-Velocity-Flat-Humanoid-Abs-Play-v0` (evaluation)

## Robot Specifications

### Joint Structure
The Humanoid_abs robot has the following joints:
- **Legs (per side)**: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
- **Torso**: waist_joint
- **Arms (per side)**: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow

### Key Parameters
- **Initial height**: 0.65m
- **Target height**: 0.65m
- **Forward velocity range**: 0.5 - 1.5 m/s
- **Contact bodies**: `.*_ankle_roll_link` (feet)

## Training Commands

### Start Training
```bash
cd /home/code/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Flat-Humanoid-Abs-v0
```

### Training with Custom Parameters
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Velocity-Flat-Humanoid-Abs-v0 \
    --num_envs=4096 \
    --headless
```

### Resume Training from Checkpoint
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Isaac-Velocity-Flat-Humanoid-Abs-v0 \
    --resume
```

### Play/Evaluate Trained Policy
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task=Isaac-Velocity-Flat-Humanoid-Abs-Play-v0 \
    --num_envs=50 \
    --load_run=<run_folder>
```

## Reward Structure

The reward function is designed to encourage forward walking:

### Primary Rewards
- **track_lin_vel_xy_exp** (weight: 1.5): Rewards tracking commanded forward velocity
- **track_ang_vel_z_exp** (weight: 1.0): Rewards tracking angular velocity commands
- **feet_air_time** (weight: 0.5): Encourages proper gait with swing phase
- **termination_penalty** (weight: -200): Large penalty for falling

### Stability Penalties
- **feet_slide** (weight: -0.15): Penalizes foot slipping
- **flat_orientation_l2** (weight: -1.5): Keeps torso upright
- **base_height_l2** (weight: -0.5): Maintains target height

### Efficiency Penalties
- **action_rate_l2** (weight: -0.01): Smooth actions
- **dof_acc_l2** (weight: -2.5e-7): Smooth joint accelerations
- **dof_torques_l2** (weight: -2.0e-7): Energy efficiency

### Joint Deviation Penalties
- Penalizes deviations from default positions for non-critical joints (arms, hip yaw/roll, waist)

## Expected Results

### Training Progress
- **Iterations**: ~1500-2000 for basic walking
- **Time**: Depends on GPU and number of environments
- **Checkpoints**: Saved every 50 iterations in `logs/rsl_rl/humanoid_abs_flat/`

### Performance Metrics
- **Forward velocity**: Should reach 0.8-1.2 m/s
- **Episode length**: Should increase as robot learns to walk
- **Reward**: Should steadily increase and plateau

## Troubleshooting

### Robot Falls Immediately
- Check initial joint positions in `humanoid_abs.py`
- Verify actuator gains (stiffness/damping)
- Reduce initial velocity commands

### Training is Unstable
- Reduce learning rate in `rsl_rl_ppo_cfg.py`
- Increase number of environments for better exploration
- Adjust reward weights

### Slow Training
- Increase `--num_envs` (requires more GPU memory)
- Use `--headless` mode to disable rendering
- Check GPU utilization

## Next Steps

### For Forward Walking Only
The current configuration is optimized for forward walking. To maintain this:
- Keep `lin_vel_y` at (0.0, 0.0)
- Keep `ang_vel_z` small or at (0.0, 0.0)

### For More Complex Locomotion
To enable turning and lateral movement:
- Increase `ang_vel_z` range in the environment config
- Increase `lin_vel_y` range for sideways walking
- Adjust reward weights accordingly

### Transfer to Real Robot
1. Save the trained policy
2. Export to ONNX format if needed
3. Deploy with appropriate control frequency
4. May need sim-to-real adaptation techniques

## File Structure Summary
```
IsaacLab/
├── source/
│   ├── isaaclab_assets/
│   │   └── isaaclab_assets/
│   │       ├── robots/
│   │       │   ├── __init__.py (updated)
│   │       │   └── humanoid_abs.py (new)
│   │       └── humanoid_snc/
│   │           └── urdf/
│   │               └── Humanoid_abs.urdf (existing)
│   └── isaaclab_tasks/
│       └── isaaclab_tasks/
│           └── manager_based/
│               └── locomotion/
│                   └── velocity/
│                       └── config/
│                           └── humanoid_abs/ (new)
│                               ├── __init__.py
│                               ├── flat_env_cfg.py
│                               └── agents/
│                                   ├── __init__.py
│                                   └── rsl_rl_ppo_cfg.py
└── logs/ (created during training)
    └── rsl_rl/
        └── humanoid_abs_flat/
            └── <timestamp>/
                ├── model_*.pt (checkpoints)
                └── events.tfevents.* (tensorboard logs)
```

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir logs/rsl_rl/humanoid_abs_flat
```

### Key Metrics to Watch
- `Episode/cumulative_reward`: Should increase
- `Episode/episode_length`: Should increase (less falling)
- `Loss/value_function`: Should decrease and stabilize
- `Policy/mean_forward_velocity`: Should approach target range

---
**Created**: November 10, 2025
**Robot**: Humanoid_abs (Custom SNC Humanoid)
**Task**: Forward Walking on Flat Terrain
