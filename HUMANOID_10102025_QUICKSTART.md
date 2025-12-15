# 🤖 Humanoid 10102025 - Quick Start Guide

## ✅ การติดตั้งเสร็จสมบูรณ์แล้ว!

### 📁 ไฟล์ที่สร้างขึ้น

#### 1. โมเดลหุ่นยนต์
- ✅ **URDF (พร้อม IMU)**: `Humanoid_10102025/urdf/Humanoid_10102025.urdf`
- ✅ **USD (ต้นฉบับ)**: `humanoid_10102025_assets/Humanoid_10102025.usd`
- ✅ **USD (พร้อม IMU)**: `humanoid_10102025_assets/Humanoid_10102025_with_imu.usd`

#### 2. ไฟล์คอนฟิก
- ✅ **Robot Config**: `source/isaaclab_assets/isaaclab_assets/robots/humanoid_10102025.py`
- ✅ **Package Init**: `source/isaaclab_assets/isaaclab_assets/robots/__init__.py` (อัพเดทแล้ว)

#### 3. สคริปต์เทรนนิ่ง
- ✅ **Simple Test**: `train_humanoid_10102025_simple.py` (ทดสอบพื้นฐาน)
- ✅ **Full Training**: `train_humanoid_10102025.py` (เทรนด้วย RSL-RL)
- ✅ **Viewer**: `view_humanoid_10102025.py` (ดูโมเดล)
- ✅ **Test Script**: `simple_test_humanoid_10102025.sh` (ตรวจสอบระบบ)

---

## 🚀 วิธีการใช้งาน

### 1️⃣ ตรวจสอบว่าติดตั้งถูกต้อง
```bash
./simple_test_humanoid_10102025.sh
```

### 2️⃣ ดูโมเดลหุ่นยนต์

**ดูโมเดลปกติ:**
```bash
./isaaclab.sh -p view_humanoid_10102025.py
```

**ดูโมเดลพร้อม IMU:**
```bash
./isaaclab.sh -p view_humanoid_10102025.py --with_imu
```

### 3️⃣ ทดสอบ Environment ด้วย Random Actions

**แบบมีหน้าจอ (4 environments):**
```bash
./isaaclab.sh -p train_humanoid_10102025_simple.py --num_envs 4 --max_iterations 50
```

**แบบไม่มีหน้าจอ (Headless - 64 environments):**
```bash
./isaaclab.sh -p train_humanoid_10102025_simple.py --headless --num_envs 64 --max_iterations 100
```

### 4️⃣ เทรนแบบเต็มรูปแบบด้วย RSL-RL PPO

**เทรนแบบปกติ (4096 environments):**
```bash
./isaaclab.sh -p train_humanoid_10102025.py --num_envs 4096 --max_iterations 2000 --headless
```

**เทรนแบบทดสอบ (512 environments, น้อยกว่าปกติ):**
```bash
./isaaclab.sh -p train_humanoid_10102025.py --num_envs 512 --max_iterations 500 --headless
```

**เทรนแบบมีหน้าจอเพื่อดูการเทรน (ช้า):**
```bash
./isaaclab.sh -p train_humanoid_10102025.py --num_envs 64 --max_iterations 200
```

---

## ⚠️ สำคัญ! อย่าใช้คำสั่งนี้

**❌ อย่าใช้:**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Humanoid
```
**เหตุผล:** Environment นี้ยังไม่ได้ลงทะเบียนในระบบ และต้องการ configuration ที่ซับซ้อนกว่า

**✅ ใช้แทน:**
```bash
./isaaclab.sh -p train_humanoid_10102025.py --num_envs 512 --max_iterations 500 --headless
```

---

## 📊 ผลลัพธ์การเทรน

ไฟล์ผลลัพธ์จะถูกบันทึกที่:
```
logs/humanoid_10102025/rsl_rl/YYYY-MM-DD_HH-MM-SS/
```

ภายในโฟลเดอร์จะมี:
- 📄 Model checkpoints (`.pt` files)
- 📈 Training statistics (rewards, episode lengths)
- 📊 TensorBoard logs

### ดู TensorBoard
```bash
tensorboard --logdir logs/humanoid_10102025/rsl_rl/
```

---

## 🎮 เล่นด้วย Policy ที่เทรนแล้ว

**หลังจากเทรนเสร็จ** ใช้คำสั่งนี้เพื่อดูผลลัพธ์:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Humanoid-10102025-Direct-v0 \
  --load_run logs/humanoid_10102025/rsl_rl/YYYY-MM-DD_HH-MM-SS/
```

**หมายเหตุ:** เปลี่ยน `YYYY-MM-DD_HH-MM-SS` เป็นชื่อโฟลเดอร์ที่ระบบสร้างขึ้นหลังจากเทรน

---

## 🤖 ข้อมูลหุ่นยนต์

### Joints (20 ข้อต่อ)

**ขา (12 joints):**
- Left leg: `left_hip_pitch`, `left_hip_roll`, `left_hip_yaw`, `left_knee`, `left_ankle_pitch`, `left_ankle_roll`
- Right leg: `right_hip_pitch`, `right_hip_roll`, `right_hip_yaw`, `right_knee`, `right_ankle_pitch`, `right_ankle_roll`

**แขน (8 joints):**
- Left arm: `left_shoulder_pitch`, `left_shoulder_roll`, `left_shoulder_yaw`, `left_elbow`
- Right arm: `right_shoulder_pitch`, `right_shoulder_roll`, `right_shoulder_yaw`, `right_elbow`

### Actuator Parameters

**ขา:**
- Hip joints: Stiffness=200.0, Damping=10.0
- Knee joints: Stiffness=200.0, Damping=10.0
- Ankle joints: Stiffness=40.0, Damping=5.0

**แขน:**
- All joints: Stiffness=40.0, Damping=5.0

### IMU Sensor
- **ชื่อ Link**: `imu_link`
- **ชื่อ Joint**: `imu_joint`
- **เชื่อมต่อกับ**: `pelvis`
- **ประเภท**: Fixed joint
- **สี**: แดง (Red box visualization)

---

## 🔧 การแก้ปัญหา

### ปัญหา 1: ไฟล์ USD ไม่พบ
```bash
# ตรวจสอบว่าไฟล์มีอยู่
ls -lh humanoid_10102025_assets/*.usd
```

### ปัญหา 2: หุ่นยนต์ล้มทันที
**แก้ไข:** ปรับค่า actuator stiffness/damping ในไฟล์:
```
source/isaaclab_assets/isaaclab_assets/robots/humanoid_10102025.py
```

### ปัญหา 3: Import Error
```bash
# ติดตั้ง Isaac Lab extensions ใหม่
./isaaclab.sh -i
```

### ปัญหา 4: Environment not found
**อาการ:** `gymnasium.error.NameNotFound: Environment 'Isaac-...' doesn't exist`

**สาเหตุ:** Environment ยังไม่ได้ลงทะเบียน หรือใช้ชื่อ environment ผิด

**แก้ไข:** ใช้สคริปต์ที่สร้างไว้แล้วแทน:
```bash
./isaaclab.sh -p train_humanoid_10102025.py --num_envs 512 --max_iterations 500 --headless
```

---

## 📚 ข้อมูลเพิ่มเติม

### Observation Space (74 มิติ)
- Root position (3)
- Root orientation (4) - quaternion
- Root linear velocity (3)
- Root angular velocity (3)
- Joint positions (20)
- Joint velocities (20)
- Velocity commands (3)
- Previous actions (20)

### Action Space
- 20 continuous actions (หนึ่ง action ต่อหนึ่ง joint)
- Action scale: 1.0

### Rewards
- ✅ Linear velocity tracking: 1.0
- ✅ Angular velocity tracking: 0.5
- ❌ Joint acceleration penalty: -2.5e-7
- ❌ Action rate penalty: -0.01
- ✅ Alive reward: 2.0
- ❌ DOF at limit penalty: -1.0
- ✅ Feet air time reward: 0.5

### Termination
- หุ่นยนต์ล้ม (ความสูง pelvis < 0.5m หรือ > 2.0m)
- หมดเวลา episode (20 วินาที)

---

## 🎯 ขั้นตอนต่อไป

1. **ปรับแต่ง Rewards** - แก้ไขค่า reward scales เพื่อให้หุ่นยนต์เดินได้ดีขึ้น
2. **ปรับแต่ง Physics** - เปลี่ยนค่า stiffness/damping ของ actuators
3. **เพิ่ม Observations** - เพิ่มข้อมูล IMU, contact forces, terrain information
4. **Curriculum Learning** - เริ่มจากงานง่ายแล้วค่อยๆ เพิ่มความยาก
5. **Domain Randomization** - เพิ่ม noise ให้กับ physics parameters

---

## 📖 เอกสารอ้างอิง

- [Isaac Lab Documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html)
- [RSL-RL GitHub](https://github.com/leggedrobotics/rsl_rl)
- **README ฉบับเต็ม**: `README_HUMANOID_10102025.md`

---

## ✨ สรุป

### สิ่งที่ทำเสร็จแล้ว ✅
1. ✅ แปลง URDF เป็น USD (2 versions: with/without IMU)
2. ✅ สร้าง robot configuration สำหรับ Isaac Lab
3. ✅ ลงทะเบียน robot ในระบบ
4. ✅ สร้างสคริปต์เทรนแบบทดสอบ (random actions)
5. ✅ สร้างสคริปต์เทรนแบบเต็มรูปแบบ (RSL-RL PPO)
6. ✅ สร้างสคริปต์สำหรับดูโมเดล
7. ✅ เพิ่ม IMU sensor เข้าไปใน URDF

### พร้อมใช้งาน! 🚀

เริ่มได้เลยด้วยคำสั่ง:
```bash
# ขั้นตอนที่ 1: ตรวจสอบระบบ
./simple_test_humanoid_10102025.sh

# ขั้นตอนที่ 2: ดูโมเดล
./isaaclab.sh -p view_humanoid_10102025.py --with_imu

# ขั้นตอนที่ 3: ทดสอบ environment
./isaaclab.sh -p train_humanoid_10102025_simple.py --num_envs 4 --max_iterations 50

# ขั้นตอนที่ 4: เทรนจริง!
./isaaclab.sh -p train_humanoid_10102025.py --num_envs 512 --max_iterations 500 --headless
```

---

**หมายเหตุ:** หากมีคำถามหรือปัญหา ดูเพิ่มเติมได้ที่ `README_HUMANOID_10102025.md`
