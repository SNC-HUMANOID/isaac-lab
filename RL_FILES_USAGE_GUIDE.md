# 🗂️ ไฟล์จากการเทรน RL และการใช้งาน

## 📁 ไฟล์ที่ได้จากการเทรน RL

```
logs/rsl_rl/g1_flat/2025-07-29_15-27-26/
├── 🧠 MODEL FILES (สำคัญที่สุด)
│   ├── model_300.pt        ← ใช้ไฟล์นี้! (โมเดลสุดท้าย)
│   ├── model_250.pt        ← Backup models
│   ├── model_200.pt
│   ├── model_150.pt
│   └── ...
│
├── ⚙️ CONFIG FILES (การตั้งค่า)
│   └── params/
│       ├── agent.yaml     ← การตั้ค่า AI algorithm
│       ├── env.yaml       ← การตั้งค่า robot & environment
│       ├── agent.pkl      ← Binary config
│       └── env.pkl        ← Binary config
│
├── 📊 TRAINING LOGS (ดูผลการเทรน)
│   └── events.out.tfevents.* ← TensorBoard logs
│
└── 📝 GIT INFO (เวอร์ชันโค้ด)
    └── git/IsaacLab.diff
```

---

## 🧠 ไฟล์ Model (model_300.pt) - ใช้ทำอะไร?'vel'

### **1. โครงสร้างภายในไฟล์:**
```python
import torch
checkpoint = torch.load("model_300.pt")

checkpoint = {
    'model_state_dict': {        # 🔥 ส่วนสำคัญที่สุด!
        'actor.0.weight': tensor,     # Neural network weights
        'actor.0.bias': tensor,       # Neural network biases  
        'actor.2.weight': tensor,
        'actor.2.bias': tensor,
        # ... รวม 167,243 parameters
        
        'critic.0.weight': tensor,    # สำหรับ training เท่านั้น
        'critic.0.bias': tensor,      # ไม่ใช้ตอน deployment
        # ...
    },
    'iter': 300,                 # Training iteration
    'infos': {...}              # Training statistics
}
```

### **2. การใช้งาน Neural Network:**

#### **🔥 สร้าง Policy Network (Actor):**
```python
import torch
import torch.nn as nn

# 1. โหลดโมเดล
checkpoint = torch.load("model_300.pt", map_location='cpu')
state_dict = checkpoint['model_state_dict']

# 2. แยกเอาส่วน Actor (Policy Network)
actor_weights = {k.replace('actor.', ''): v 
                for k, v in state_dict.items() 
                if 'actor' in k}

# 3. สร้าง Neural Network
policy_net = nn.Sequential(
    nn.Linear(123, 256), nn.ELU(),    # Input layer
    nn.Linear(256, 128), nn.ELU(),    # Hidden layer 1  
    nn.Linear(128, 128), nn.ELU(),    # Hidden layer 2
    nn.Linear(128, 37)                # Output layer (37 joints)
)

# 4. โหลด weights
policy_net.load_state_dict(actor_weights)
policy_net.eval()  # เปลี่ยนเป็น inference mode
```

---

## 🎮 Input/Output ของ Neural Network

### **📥 INPUT (123 ค่า) - ข้อมูลที่ใส่เข้าไป:**

```python
def get_robot_observations():
    """รวบรวมข้อมูลจากเซนเซอร์ robot (123 ค่า)"""
    
    # 1. Base velocity (6 ค่า) - ความเร็วของลำตัว
    base_lin_vel = robot.get_linear_velocity()    # [vx, vy, vz] m/s
    base_ang_vel = robot.get_angular_velocity()   # [wx, wy, wz] rad/s
    
    # 2. Gravity vector (3 ค่า) - ทิศทางแรงโน้มถ่วง  
    gravity = robot.get_projected_gravity()       # [gx, gy, gz] normalized
    
    # 3. Commands (3 ค่า) - คำสั่งที่ต้องการ
    commands = get_user_commands()                # [desired_vx, desired_vy, desired_wz]
    
    # 4. Joint positions (37 ค่า) - ตำแหน่งปัจจุบัน.yaml** | 🏗️ Network architecture | ⭐⭐⭐ ต้องมี! |
| **events.out.tfevents** | 📊 ดูกราฟการเทรน | ⭐⭐ ไม่จำเป็น |
| **model_250.pt, etc** | 💾 Backup models | ⭐ ไม่จำเป็น |

### **🔥 สิ่งสำคัญที่ต้องจำ:**
1. **AI ส่งออกตำแหน่ง joint** (37 ค่า) ไม่ใช่ torque
2. **Scale ด้วย 0.5** แล้วบวกกับ default positions  
3. **ใช้ PD Controller** แปลงตำแหน่งเป็น torque
4. **รันที่ 50 Hz** (ทุก 0.02 วินาที)
5. **Input ต้องครบ 123 ค่า** จากเซนเซอร์ต่างๆ

**ตอนนี้คุณรู้แล้วว่าทุกไฟล์ใช้ทำอะไร และค่าไหนไปสั่ง robot!** 🤖✨
    joint_pos = robot.get_joint_positions()      # [joint_0_pos, joint_1_pos, ...] radians
    joint_pos_relative = joint_pos - default_joint_positions  # เทียบกับค่าเริ่มต้น
    
    # 5. Joint velocities (37 ค่า) - ความเร็วปัจจุบัน  
    joint_vel = robot.get_joint_velocities()     # [joint_0_vel, joint_1_vel, ...] rad/s
    
    # 6. Previous actions (37 ค่า) - คำสั่งก่อนหน้า
    prev_actions = last_sent_actions              # [prev_joint_0, prev_joint_1, ...]
    
    # รวมทั้งหมด
    observations = np.concatenate([
        base_lin_vel,      # 3 ค่า
        base_ang_vel,      # 3 ค่า  
        gravity,           # 3 ค่า
        commands,          # 3 ค่า
        joint_pos_relative,# 37 ค่า
        joint_vel,         # 37 ค่า
        prev_actions       # 37 ค่า
    ])  # รวม = 123 ค่า
    
    return observations
```

### **📤 OUTPUT (37 ค่า) - ค่าที่ใช้สั่ง Robot:**

```python
def run_ai_control():
    """ใช้ AI ควบคุมหุ่นยนต์"""
    
    # 1. รับข้อมูลจากเซนเซอร์
    observations = get_robot_observations()  # 123 ค่า
    
    # 2. ประมวลผลด้วย AI
    obs_tensor = torch.FloatTensor(observations).unsqueeze(0)
    with torch.no_grad():
        raw_actions = policy_net(obs_tensor).squeeze().numpy()  # 37 ค่า
    
    print("🤖 AI Raw Output:", raw_actions[:5])  # แสดง 5 ค่าแรก
    # ตัวอย่าง: [-0.234, 0.156, -0.089, 0.345, 0.012]
    
    # 3. แปลงเป็นตำแหน่ง joint จริง
    action_scale = 0.5  # จาก config file
    default_positions = np.array([...])  # 37 ค่า จาก config
    
    final_joint_targets = default_positions + (raw_actions * action_scale)
    
    print("🎯 Final Joint Targets:", final_joint_targets[:5])
    # ตัวอย่าง: [0.0, -0.122, -0.244, 0.592, -0.224] radians
    
    # 4. ส่งไปยัง robot motors
    robot.set_joint_position_targets(final_joint_targets)
    
    return raw_actions  # เก็บไว้สำหรับรอบถัดไป
```

---

## 🎯 ค่าที่ไปสั่ง Robot จริงๆ

### **🔥 สิ่งสำคัญ: AI ส่งออก "ตำแหน่งเป้าหมาย" ไม่ใช่ Torque!**

```python
# AI Output -> Joint Position Targets (37 ค่า)
joint_targets = [
    0.15,    # left_hip_yaw_joint target (radians)
    -0.05,   # left_hip_roll_joint target
    -0.25,   # left_hip_pitch_joint target  
    0.45,    # left_knee_joint target
    # ... รวม 37 joints
]

# Robot ใช้ PD Controller แปลงเป็น Torque
for i, joint_name in enumerate(joint_names):
    target_pos = joint_targets[i]         # เป้าหมาย (radians)
    current_pos = robot.get_joint_pos(i)  # ปัจจุบัน (radians)
    current_vel = robot.get_joint_vel(i)  # ความเร็ว (rad/s)
    
    # PD Control Formula
    error = target_pos - current_pos      # ผิดพลาดตำแหน่ง
    torque = (Kp * error) - (Kd * current_vel)
    
    robot.set_joint_torque(joint_name, torque)  # ส่ง torque ไปมอเตอร์
```

### **⚙️ PD Controller Parameters (จาก config):**
```python
pd_gains = {
    'legs': {'Kp': 150-200, 'Kd': 5.0},    # ขา - แรงสูง  
    'feet': {'Kp': 20, 'Kd': 2.0},         # เท้า - แรงต่ำ
    'arms': {'Kp': 40, 'Kd': 10.0}         # แขน - แรงกลาง
}
```

---

## 📊 Config Files - ใช้ทำอะไร?

### **1. agent.yaml - การตั้งค่า AI:**
```yaml
# ข้อมูลสำคัญที่ใช้
policy:
  actor_hidden_dims: [256, 128, 128]  # โครงสร้าง Neural Network
  activation: elu                     # Activation function

algorithm:
  learning_rate: 0.001               # ไม่ใช้ตอน deployment
  
num_steps_per_env: 24               # ไม่ใช้ตอน deployment
max_iterations: 500                 # ไม่ใช้ตอน deployment
```

### **2. env.yaml - การตั้งค่า Robot:**
```yaml
# ข้อมูลสำคัญที่ใช้
actions:
  joint_pos:
    scale: 0.5                      # 🔥 สำคัญ! ใช้ scale AI output
    use_default_offset: true        # ใช้ default positions

# Default joint positions (37 ค่า)
init_state:
  joint_pos:
    "*_hip_pitch_joint": -0.2       # 🔥 สำคัญ! Default positions
    "*_knee_joint": 0.42
    "*_ankle_pitch_joint": -0.23
    # ...

# PD Controller gains  
actuators:
  legs:
    stiffness: 150.0-200.0          # 🔥 สำคัญ! Kp values
    damping: 5.0                    # 🔥 สำคัญ! Kd values
```

---

## 🚀 การนำไปใช้จริง - ทีละขั้น

### **Step 1: โหลดโมเดลและ config**
```python
# โหลดโมเดล
model = torch.load("model_300.pt")
policy_net = create_policy_network(model)

# โหลด config
with open("params/env.yaml") as f:
    config = yaml.safe_load(f)
    
action_scale = config['actions']['joint_pos']['scale']  # 0.5
default_positions = extract_default_positions(config)   # 37 ค่า
```

### **Step 2: Control Loop ที่ 50 Hz**
```python
import time

dt = 0.02  # 50 Hz
previous_actions = np.zeros(37)

while True:
    start_time = time.time()
    
    # 1. อ่านเซนเซอร์
    observations = get_robot_observations()  # 123 ค่า
    
    # 2. ประมวลผล AI  
    actions = policy_net(observations)       # 37 ค่า (-1 to +1)
    
    # 3. แปลงเป็นตำแหน่งจริง
    targets = default_positions + (actions * action_scale)
    
    # 4. ส่งไปยัง robot
    robot.set_joint_targets(targets)         # 37 ค่า (radians)
    
    # 5. เก็บสำหรับรอบถัดไป
    previous_actions = actions
    
    # 6. รอให้ครบ 50 Hz
    elapsed = time.time() - start_time
    time.sleep(max(0, dt - elapsed))
```

---

## 🎯 สรุป: ไฟล์ไหนทำอะไร

| ไฟล์ | ใช้ทำอะไร | สำคัญเพียงใด |
|------|-----------|---------------|
| **model_300.pt** | 🧠 Neural Network weights | ⭐⭐⭐⭐⭐ ต้องมี! |
| **env.yaml** | ⚙️ Robot config, default positions, PD gains | ⭐⭐⭐⭐ ต้องมี! |
| **agent.yaml** | 🏗️ Network architecture | ⭐⭐⭐ ต้องมี! |
| **events.out.tfevents** | 📊 ดูกราฟการเทรน | ⭐⭐ ไม่จำเป็น |
| **model_250.pt, etc** | 💾 Backup models | ⭐ ไม่จำเป็น |

### **🔥 สิ่งสำคัญที่ต้องจำ:**
1. **AI ส่งออกตำแหน่ง joint** (37 ค่า) ไม่ใช่ torque
2. **Scale ด้วย 0.5** แล้วบวกกับ default positions  
3. **ใช้ PD Controller** แปลงตำแหน่งเป็น torque
4. **รันที่ 50 Hz** (ทุก 0.02 วินาที)
5. **Input ต้องครบ 123 ค่า** จากเซนเซอร์ต่างๆ

**ตอนนี้คุณรู้แล้วว่าทุกไฟล์ใช้ทำอะไร และค่าไหนไปสั่ง robot!** 🤖✨