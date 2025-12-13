# 🤖 Humanoid SNC Robot for Isaac Lab

## 📋 สรุป
คุณมีหุ่นยนต์ Humanoid SNC พร้อมใช้งานใน Isaac Lab แล้ว!

## 🎯 การใช้งาน

### 1️⃣ แสดง Model (ง่ายที่สุด)
```bash
./isaaclab.sh -p 01_show_model.py
```
- เห็นรูปร่างหุ่นยนต์ 3D
- ดูจำนวน joints และชื่อ
- หมุนดูได้รอบๆ

### 2️⃣ ทดสอบ Physics
```bash
./isaaclab.sh -p 02_test_physics.py --num_envs 4
```
- เห็นหุ่นยนต์เคลื่อนไหวด้วย random actions
- ดู reward และ physics simulation
- ทดสอบว่า environment ทำงานถูกต้อง

### 3️⃣ เทรน RL 
```bash
./isaaclab.sh -p 03_train_rsl_rl.py --num_envs 512 --max_iterations 1000 --headless
```
- เทรน reinforcement learning
- บันทึก logs ใน `logs/rsl_rl/humanoid_snc/`
- ดู progress ด้วย TensorBoard

## 📊 ข้อมูลหุ่นยนต์
- **จำนวน joints**: 21 ข้อต่อ
- **Action space**: (batch_size, 21) สำหรับควบคุม joints
- **Observation space**: (batch_size, 75) ข้อมูล sensors
- **Robot height**: ประมาณ 1.0-1.2 เมตร

## 📂 ไฟล์ที่สำคัญ
- `humanoid_snc_assets/humanoid_snc.usd` - Robot model (USD format)
- `source/isaaclab_assets/isaaclab_assets/robots/humanoid_snc.py` - Robot config
- `01_show_model.py` - แสดง model
- `02_test_physics.py` - ทดสอบ physics
- `03_train_rsl_rl.py` - เทรน RL

## 🎮 Joint Names
```
left_shoulder_pitch_joint, left_shoulder_roll_joint, left_shoulder_yaw_joint,
left_elbow_joint, right_shoulder_pitch_joint, right_shoulder_roll_joint,
right_shoulder_yaw_joint, right_elbow_joint, Pelvis_joint,
left_hip_pitch_joint, left_hip_roll_joint, left_hip_yaw_joint,
left_ankle_pitch_joint, left_ankle_roll_joint, left_ankle_yaw_joint,
right_hip_pitch_joint, right_hip_roll_joint, right_hip_yaw_joint,
right_ankle_pitch_joint, right_ankle_roll_joint, right_ankle_yaw_joint
```

## 🔧 Tips
- เริ่มด้วย `01_show_model.py` เพื่อดู model
- ใช้ `--headless` สำหรับ training
- ใช้ `--num_envs` เพื่อปรับจำนวน environments
- Logs จะถูกบันทึกใน `logs/rsl_rl/humanoid_snc/`

## 🎯 เป้าหมายการเทรน
หุ่นยนต์จะเรียนรู้:
- การเดินตรง
- การรักษาความสมดุล
- การหลีกเลี่ยงการล้ม
- การเคลื่อนไหวอย่างมีประสิทธิภาพ

**ขั้นตอนแนะนำ:**
1. รัน `01_show_model.py` ดู model
2. รัน `02_test_physics.py` ทดสอบ
3. รัน `03_train_rsl_rl.py` เทรน RL

🚀 **พร้อมใช้งานแล้ว!**