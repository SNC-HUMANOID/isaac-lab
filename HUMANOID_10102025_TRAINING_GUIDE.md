# 🤖 Humanoid 10102025 - Training Guide

## 🎯 การเทรนแบบที่แนะนำ (แบบเดียวกับ SNC R5)

ตอนนี้ Humanoid 10102025 ใช้ **LocomotionEnv** ซึ่งเป็น environment ที่ออกแบบมาเฉพาะสำหรับการเดิน เหมือนกับที่ SNC R5 ใช้!

### ✨ ข้อดีของ LocomotionEnv

1. **Reward Structure ที่ดีกว่า**: ออกแบบมาเพื่อการเดินโดยเฉพาะ
2. **Joint Gears**: มี torque scaling ที่เหมาะสมกับแต่ละข้อต่อ
3. **Heading & Up Rewards**: รางวัลสำหรับการเดินไปข้างหน้าและการยืนตรง
4. **Energy Cost**: ลงโทษการใช้พลังงานมากเกินไป
5. **Termination ที่ชัดเจน**: หยุดทันทีเมื่อหุ่นยนต์ล้ม

---

## 🚀 วิธีการเทรน

### 1. เทรนแบบเต็มรูปแบบ (แนะนำ!)

**เทรนด้วย 4096 environments (แรงมาก, ใช้เวลานาน):**
```bash
./isaaclab.sh -p train_humanoid_10102025_locomotion.py \
    --num_envs 4096 \
    --seed 42 \
    --headless
```

**เทรนด้วย 1024 environments (กลางๆ):**
```bash
./isaaclab.sh -p train_humanoid_10102025_locomotion.py \
    --num_envs 1024 \
    --seed 42 \
    --headless
```

**เทรนด้วย 512 environments (เร็ว, ทดสอบ):**
```bash
./isaaclab.sh -p train_humanoid_10102025_locomotion.py \
    --num_envs 512 \
    --seed 42 \
    --headless
```

### 2. เทรนแบบมีหน้าจอ (เพื่อดูการเทรน)

```bash
./isaaclab.sh -p train_humanoid_10102025_locomotion.py \
    --num_envs 64 \
    --seed 42
```

### 3. ทดสอบ Environment

```bash
./test_humanoid_10102025_locomotion.sh
```

---

## 📊 Configuration Details

### Algorithm (RSL-RL PPO) - เหมือน SNC R5

```python
Learning Rate: 3.0e-4
Schedule: adaptive
Gamma: 0.99
Lambda (GAE): 0.95
Clip Param: 0.2
Entropy Coefficient: 0.01
Value Loss Coefficient: 1.0
Max Gradient Norm: 1.0

Learning Epochs: 5
Mini Batches: 4
Desired KL: 0.01
```

### Policy Network - เหมือน SNC R5

```python
Actor Hidden Layers: [512, 256, 128]
Critic Hidden Layers: [512, 256, 128]
Activation: ELU
Init Noise Std: 1.0
```

### Training Settings

```python
Max Iterations: 1500
Save Interval: 50 iterations
Episode Length: 15 seconds
Simulation dt: 1/120 (120 Hz)
Control Decimation: 2 (60 Hz control)
```

### Joint Gears (Torque Scaling)

```python
# Left leg
hip_pitch, hip_roll, hip_yaw: 45.0
knee: 90.0
ankle_pitch, ankle_roll: 22.5

# Right leg
hip_pitch, hip_roll, hip_yaw: 45.0
knee: 90.0
ankle_pitch, ankle_roll: 22.5

# Left arm
shoulder_pitch, shoulder_roll, shoulder_yaw: 45.0
elbow: 22.5

# Right arm
shoulder_pitch, shoulder_roll, shoulder_yaw: 45.0
elbow: 22.5
```

### Reward Weights

```python
heading_weight: 0.5          # รางวัลการเดินไปข้างหน้า
up_weight: 0.1               # รางวัลการยืนตรง
alive_reward_scale: 2.0      # รางวัลการมีชีวิต
energy_cost_scale: 0.05      # ลงโทษการใช้พลังงาน
actions_cost_scale: 0.01     # ลงโทษการเคลื่อนไหวรุนแรง
dof_vel_scale: 0.1           # ลงโทษความเร็วข้อต่อ
death_cost: -10.0            # ลงโทษการล้ม
termination_height: 0.8m     # ความสูงที่ถือว่าล้ม
```

---

## 📁 ผลลัพธ์การเทรน

### ตำแหน่งไฟล์

```
logs/rsl_rl/humanoid_10102025/
└── 2025-11-03_13-00-00_42/          # วันที่-เวลา_seed
    ├── model_50.pt                   # Checkpoint ทุกๆ 50 iterations
    ├── model_100.pt
    ├── model_150.pt
    ├── ...
    ├── model_1500.pt
    ├── policy.pt                     # Policy สุดท้าย (JIT format)
    ├── policy.onnx                   # Policy สุดท้าย (ONNX format)
    ├── params/
    │   ├── env.yaml                  # Environment configuration
    │   ├── env.pkl
    │   ├── agent.yaml                # Agent configuration
    │   └── agent.pkl
    └── summaries/                    # TensorBoard logs
        └── ...
```

### ดู TensorBoard

```bash
tensorboard --logdir logs/rsl_rl/humanoid_10102025/
```

เปิดเบราว์เซอร์ที่: http://localhost:6006

---

## 🎮 เล่นด้วย Policy ที่เทรนแล้ว

### วิธีที่ 1: ใช้ Script

```bash
./play_humanoid_10102025.sh logs/rsl_rl/humanoid_10102025/2025-11-03_13-00-00_42/
```

### วิธีที่ 2: คำสั่งเต็ม

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Humanoid-10102025-Direct-v0 \
    --num_envs 16 \
    --load_run logs/rsl_rl/humanoid_10102025/2025-11-03_13-00-00_42/
```

### เล่นด้วย Checkpoint เฉพาะ

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Humanoid-10102025-Direct-v0 \
    --num_envs 16 \
    --checkpoint logs/rsl_rl/humanoid_10102025/2025-11-03_13-00-00_42/model_1000.pt
```

---

## 📈 การตรวจสอบความคืบหน้า

### Metrics ที่ควรดู

1. **Reward Mean**: ควรเพิ่มขึ้นเรื่อยๆ
2. **Episode Length Mean**: ควรเพิ่มขึ้น (หุ่นยนต์อยู่ได้นานขึ้น)
3. **Learning Rate**: จะปรับตัวอัตโนมัติด้วย adaptive schedule
4. **Policy Loss**: ควรลดลง
5. **Value Loss**: ควรลดลง

### Reward ที่ดี

หลังจากเทรน 1500 iterations:
- **Total Reward**: > 50 (ยิ่งมากยิ่งดี)
- **Episode Length**: ใกล้ 15 วินาที (max episode length)

---

## 🔧 การปรับแต่ง

### เพิ่มความเร็วการเทรน

```bash
# เพิ่มจำนวน environments
./isaaclab.sh -p train_humanoid_10102025_locomotion.py \
    --num_envs 8192 \
    --headless
```

### ปรับ Reward Weights

แก้ไขไฟล์:
```
source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_10102025/humanoid_10102025_env.py
```

เปลี่ยนค่า:
```python
heading_weight: float = 0.5      # เพิ่มเป็น 1.0 ถ้าอยากให้เดินไปข้างหน้าเร็วขึ้น
up_weight: float = 0.1           # เพิ่มเป็น 0.2 ถ้าอยากให้ยืนตรงมากขึ้น
alive_reward_scale: float = 2.0  # เพิ่มถ้าอยากให้หุ่นอยู่นานขึ้น
energy_cost_scale: float = 0.05  # ลดถ้าอยากให้เคลื่อนไหวเร็วขึ้น
```

### ปรับ Joint Gears

แก้ไขค่า `joint_gears` ในไฟล์เดียวกัน:
```python
joint_gears: list[float] = [
    # เพิ่มค่าถ้าอยากได้ torque มากขึ้น
    # ลดค่าถ้าอยากได้การเคลื่อนไหวที่ละเอียดขึ้น
]
```

---

## 🆚 เปรียบเทียบกับวิธีเดิม

### ❌ วิธีเดิม (train_humanoid_10102025.py)

- ใช้ HumanoidEnv ทั่วไป
- Reward structure ซับซ้อน
- ต้องปรับแต่งเยอะ
- เหมาะสำหรับ custom tasks

### ✅ วิธีใหม่ (train_humanoid_10102025_locomotion.py) **แนะนำ!**

- ใช้ LocomotionEnv เหมือน SNC R5
- Reward structure เหมาะสำหรับการเดิน
- มี joint gears ที่ปรับแล้ว
- ใช้งานได้ทันที!

---

## 📚 เอกสารเพิ่มเติม

- **Quick Start**: `HUMANOID_10102025_QUICKSTART.md`
- **Full Documentation**: `README_HUMANOID_10102025.md`
- **SNC R5 Comparison**: ดูไฟล์ `train_humanoid_snc_r5.py` เพื่อเปรียบเทียบ

---

## 🎉 สรุป

### คำสั่งที่ใช้บ่อย

```bash
# 1. เทรน (แนะนำ)
./isaaclab.sh -p train_humanoid_10102025_locomotion.py --num_envs 1024 --seed 42 --headless

# 2. ดู TensorBoard
tensorboard --logdir logs/rsl_rl/humanoid_10102025/

# 3. เล่น Policy
./play_humanoid_10102025.sh logs/rsl_rl/humanoid_10102025/2025-11-03_13-00-00_42/

# 4. ทดสอบ Environment
./test_humanoid_10102025_locomotion.sh
```

### Timeline การเทรน (โดยประมาณ)

- **512 envs**: ~2-3 ชั่วโมง สำหรับ 1500 iterations
- **1024 envs**: ~1.5-2 ชั่วโมง
- **4096 envs**: ~30-45 นาที (ต้องการ GPU แรง)

---

**Happy Training! 🤖✨**
