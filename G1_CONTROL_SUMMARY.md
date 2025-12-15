# 🤖 G1 Robot Control Summary

## 📄 ไฟล์ Output ที่ได้จากการเทรน

### **1. Model Files (Neural Network)**
```
logs/rsl_rl/g1_flat/2025-07-29_15-27-26/
├── model_300.pt          # โมเดลสุดท้าย (ใช้ไฟล์นี้)
├── model_250.pt          # Checkpoint ก่อนหน้า
├── model_200.pt
├── model_150.pt
├── model_100.pt
├── model_50.pt
└── model_0.pt
```

**การใช้งาน:**
```python
import torch
checkpoint = torch.load("logs/rsl_rl/g1_flat/2025-07-29_15-27-26/model_300.pt")
model_weights = checkpoint['model_state_dict']  # Neural network weights
```

### **2. Configuration Files**
```
logs/rsl_rl/g1_flat/2025-07-29_15-27-26/params/
├── agent.yaml       # PPO algorithm settings
├── env.yaml         # Environment & robot settings
├── agent.pkl        # Binary config
└── env.pkl          # Binary config
```

**ข้อมูลสำคัญใน agent.yaml:**
- Learning rate: 0.001
- Network: [256, 128, 128]
- PPO parameters

**ข้อมูลสำคัญใน env.yaml:**
- Action scale: 0.5
- 37 joints controlled
- Joint stiffness/damping values

### **3. Training Logs**
```
logs/rsl_rl/g1_flat/2025-07-29_15-27-26/
└── events.out.tfevents.*  # TensorBoard logs (729KB)
```

**การดู logs:**
```bash
tensorboard --logdir logs/rsl_rl/g1_flat/2025-07-29_15-27-26/
```

---

## 🎮 การสั่งแต่ละ Joint

### **Input ที่โมเดลต้องการ (123 values):**

```python
observation_vector = [
    # Base state (12 values)
    base_linear_velocity_x,     # m/s
    base_linear_velocity_y,     # m/s  
    base_linear_velocity_z,     # m/s
    base_angular_velocity_x,    # rad/s
    base_angular_velocity_y,    # rad/s
    base_angular_velocity_z,    # rad/s
    projected_gravity_x,        # normalized
    projected_gravity_y,        # normalized
    projected_gravity_z,        # normalized
    velocity_command_x,         # desired vx
    velocity_command_y,         # desired vy
    velocity_command_z,         # desired wz
    
    # Joint positions (37 values) - current positions relative to default
    joint_pos_0, joint_pos_1, ..., joint_pos_36,
    
    # Joint velocities (37 values) - current velocities
    joint_vel_0, joint_vel_1, ..., joint_vel_36,
    
    # Previous actions (37 values) - last sent commands
    prev_action_0, prev_action_1, ..., prev_action_36
]
# Total: 12 + 37 + 37 + 37 = 123 values
```

### **Output ที่โมเดลให้ (37 values):**

```python
joint_targets = [
    # Legs (9 joints)
    left_hip_yaw_target,        # radians
    left_hip_roll_target,       # radians
    left_hip_pitch_target,      # radians
    left_knee_target,           # radians
    right_hip_yaw_target,       # radians
    right_hip_roll_target,      # radians
    right_hip_pitch_target,     # radians
    right_knee_target,          # radians
    torso_target,               # radians
    
    # Feet (4 joints)
    left_ankle_pitch_target,    # radians
    left_ankle_roll_target,     # radians
    right_ankle_pitch_target,   # radians
    right_ankle_roll_target,    # radians
    
    # Arms & Hands (24 joints)
    left_shoulder_pitch_target,
    left_shoulder_roll_target,
    left_shoulder_yaw_target,
    left_elbow_pitch_target,
    left_elbow_roll_target,
    # ... และ finger joints
    # ... รวม 37 joints
]
```

### **การแปลงและส่งไปยัง Motors:**

```python
# 1. Raw output จาก neural network (-1 to +1)
raw_actions = model(observations)  # 37 values

# 2. Scale และ offset
scale = 0.5  # จาก config
default_positions = [0.0, 0.0, -0.2, 0.42, ...]  # 37 values

final_joint_positions = default_positions + (raw_actions * scale)

# 3. ส่งไปยัง PD controllers
for i, joint_name in enumerate(joint_names):
    target_position = final_joint_positions[i]  # radians
    robot.set_joint_position_target(joint_name, target_position)
```

---

## ⚙️ การเซ็ต Environment

### **1. Environment Configuration**

```python
# Environment settings
num_envs = 4096              # Parallel environments ขณะเทรน
env_spacing = 2.5           # Distance between robots (meters)
episode_length = 20.0       # Episode duration (seconds)
control_frequency = 50      # Hz (every 0.02 seconds)
simulation_dt = 0.005       # Physics timestep
decimation = 4              # Control decimation (0.005 * 4 = 0.02s)
```

### **2. Robot Initial State**

```python
initial_state = {
    'position': [0.0, 0.0, 0.74],      # [x, y, z] meters
    'orientation': [1.0, 0.0, 0.0, 0.0], # quaternion [w, x, y, z]
    'joint_positions': {
        # Legs - standing pose
        '*_hip_pitch_joint': -0.2,      # slightly backward
        '*_knee_joint': 0.42,           # bent knees
        '*_ankle_pitch_joint': -0.23,   # level feet
        
        # Arms - natural pose  
        '*_shoulder_pitch_joint': 0.35,  # arms down
        'left_shoulder_roll_joint': 0.16,   # away from body
        'right_shoulder_roll_joint': -0.16, # away from body
        '*_elbow_pitch_joint': 0.87,     # slightly bent
        
        # Hands
        'left_one_joint': 1.0,           # finger position
        'right_one_joint': -1.0,        # finger position
        'left_two_joint': 0.52,
        'right_two_joint': -0.52,
    }
}
```

### **3. Actuator Settings**

```python
actuator_config = {
    'legs': {
        'joints': ['*_hip_*_joint', '*_knee_joint', 'torso_joint'],
        'effort_limit': 300,        # Nm
        'velocity_limit': 100.0,    # rad/s
        'stiffness': {
            '*_hip_yaw_joint': 150.0,
            '*_hip_roll_joint': 150.0, 
            '*_hip_pitch_joint': 200.0,
            '*_knee_joint': 200.0,
            'torso_joint': 200.0
        },
        'damping': {
            '*_hip_yaw_joint': 5.0,
            '*_hip_roll_joint': 5.0,
            '*_hip_pitch_joint': 5.0, 
            '*_knee_joint': 5.0,
            'torso_joint': 5.0
        }
    },
    'feet': {
        'joints': ['*_ankle_*_joint'],
        'effort_limit': 20,         # Nm (lower for fine control)
        'stiffness': 20.0,
        'damping': 2.0
    },
    'arms': {
        'joints': ['*_shoulder_*_joint', '*_elbow_*_joint', '*_*_joint'],
        'effort_limit': 300,        # Nm
        'velocity_limit': 100.0,    # rad/s
        'stiffness': 40.0,          # Lower than legs
        'damping': 10.0
    }
}
```

### **4. Observation Configuration**

```python
observations = {
    'base_lin_vel': {
        'noise': 'uniform(-0.1, 0.1)'    # Add noise for robustness
    },
    'base_ang_vel': {
        'noise': 'uniform(-0.2, 0.2)'
    },
    'projected_gravity': {
        'noise': 'uniform(-0.05, 0.05)'
    },
    'joint_pos': {
        'noise': 'uniform(-0.01, 0.01)'  # Small noise on joint positions
    },
    'joint_vel': {
        'noise': 'uniform(-1.5, 1.5)'    # Larger noise on velocities
    }
}
```

### **5. Reward Configuration**

```python
rewards = {
    # Main objectives
    'track_lin_vel_xy_exp': 1.0,        # Follow linear velocity commands
    'track_ang_vel_z_exp': 1.0,         # Follow angular velocity commands
    'feet_air_time': 0.75,              # Encourage proper walking gait
    
    # Penalties
    'termination_penalty': -200.0,      # Don't fall down
    'flat_orientation_l2': -1.0,        # Stay upright
    'lin_vel_z_l2': -0.2,              # Don't jump
    'feet_slide': -0.1,                 # Don't slip
    'dof_torques_l2': -2e-6,           # Use less torque
    'action_rate_l2': -0.005,          # Smooth movements
    
    # Joint deviations (stay near default poses)
    'joint_deviation_hip': -0.1,
    'joint_deviation_arms': -0.1,
    'joint_deviation_fingers': -0.05,
    'joint_deviation_torso': -0.1
}
```

---

## 🚀 การใช้งานจริง

### **Control Loop ที่ 50 Hz:**

```python
import torch
import numpy as np

# Load model
checkpoint = torch.load("logs/rsl_rl/g1_flat/2025-07-29_15-27-26/model_300.pt")
model = create_policy_network(checkpoint['model_state_dict'])

# Control parameters
CONTROL_FREQ = 50  # Hz
dt = 1.0 / CONTROL_FREQ
scale = 0.5

# Default joint positions (37 values)
default_positions = np.array([...])  # From training config

# Previous action memory
previous_actions = np.zeros(37)

while True:
    # 1. Get sensor data
    base_vel = robot.get_base_velocity()              # 6 values
    gravity = robot.get_projected_gravity()           # 3 values
    velocity_command = get_user_command()             # 3 values [vx, vy, wz]
    joint_pos = robot.get_joint_positions() - default_positions  # 37 values (relative)
    joint_vel = robot.get_joint_velocities()         # 37 values
    
    # 2. Combine observations
    obs = np.concatenate([
        base_vel, gravity, velocity_command,
        joint_pos, joint_vel, previous_actions
    ])  # 123 values total
    
    # 3. AI inference
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    with torch.no_grad():
        raw_actions = model(obs_tensor).squeeze(0).numpy()  # 37 values
    
    # 4. Scale actions
    final_positions = default_positions + (raw_actions * scale)
    
    # 5. Send to robot
    robot.set_joint_position_targets(final_positions)
    
    # 6. Update memory
    previous_actions = raw_actions
    
    # 7. Wait for next control cycle
    time.sleep(dt)
```

**นี่คือทุกอย่างที่คุณต้องรู้เพื่อนำโมเดลไปใช้กับหุ่นยนต์จริง!** 🤖✨