# ✅ Humanoid 10102025 - คัดลอกการตั้งค่า G1 เรียบร้อย!

## 🎉 การตั้งค่าเสร็จสมบูรณ์

ตอนนี้ Humanoid 10102025 ใช้การตั้งค่าเหมือน G1 เป๊ะๆ แล้ว!

### การเปลี่ยนแปลงที่สำคัญ:

1. **Reward Structure** (คัดลอกจาก G1):
   - `feet_air_time_positive_biped` - รางวัลสำหรับเวลาที่เท้าลอยในอากาศ
   - `track_ang_vel_z_exp` weight = 2.0 (rough) / 1.0 (flat)
   - `feet_slide` penalty = -0.1
   - `joint_deviation` penalties for hip/arms

2. **Physics Parameters** (คัดลอกจาก G1):
   - `flat_orientation_l2` weight = -1.0
   - `action_rate_l2` weight = -0.005
   - `dof_acc_l2` weight = -1.25e-7 (rough) / -1.0e-7 (flat)
   - `dof_torques_l2` weight = -1.5e-7 (rough) / -2.0e-6 (flat)

3. **Command Ranges** (คัดลอกจาก G1):
   - Flat: lin_vel_y = (-0.5, 0.5) - เดินข้างได้
   - Rough: lin_vel_y = (0.0, 0.0) - เดินหน้าอย่างเดียว

---

## 🚀 คำสั่งเทรน (เหมือนเดิม)

### 1. Flat Terrain (แนะนำเริ่มที่นี่)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Humanoid-10102025-v0 \
    --num_envs 1024 \
    --headless
```

### 2. Rough Terrain (พื้นขรุขระ)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Humanoid-10102025-v0 \
    --num_envs 1024 \
    --headless
```

---

## 🎮 เล่น Policy

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Humanoid-10102025-Play-v0 \
    --num_envs 16 \
    --load_run logs/rsl_rl/humanoid_10102025/YYYY-MM-DD_HH-MM-SS
```

---

## 📊 ดู TensorBoard

```bash
tensorboard --logdir logs/rsl_rl/humanoid_10102025/
```

---

## 🔧 ไฟล์ที่อัพเดท

1. [rough_env_cfg.py](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/humanoid_10102025/rough_env_cfg.py) - Rough terrain config
2. [flat_env_cfg.py](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/humanoid_10102025/flat_env_cfg.py) - Flat terrain config

---

## 🎯 ความแตกต่างจาก G1

**Body Names ที่ปรับเปลี่ยน:**
- G1: `torso_link` → Humanoid 10102025: `pelvis`
- G1: `.*_ankle_roll_link` → Humanoid 10102025: `.*ankle.*`

**Joint Names ที่เอาออก:**
- เอา finger joints ออก (G1 มีนิ้ว, Humanoid 10102025 ไม่มี)
- เอา `torso_joint` deviation ออก (Humanoid 10102025 ไม่มี torso joint)

---

## ✨ คาดหวังผลลัพธ์

ด้วยการตั้งค่าเหมือน G1 หุ่นยนต์ควรจะ:
- เดินได้อย่างธรรมชาติมากขึ้น
- มีการเคลื่อนไหวที่ลื่นไหล
- สมดุลได้ดีขึ้น
- ไม่แข็งเหมือนเดิม

ลองเทรนดูครับ! 🚀
