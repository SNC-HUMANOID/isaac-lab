# Humanoid 10102025 Training Setup - พร้อมเทรน!

## ✅ สรุปการตั้งค่า

### 1. โมเดลโรบอท
- **USD File**: `/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/Humanoid_10102025.usd`
- **จำนวน Joints**: 21 DOF
  - Left leg: 6 joints
  - Right leg: 6 joints  
  - Pelvis: 1 joint
  - Left arm: 4 joints
  - Right arm: 4 joints

### 2. ไฟล์ที่สร้าง

**Robot Asset:**
- `/home/sncbot/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/humanoid_10102025.py`

**Environment:**
- `/home/sncbot/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_10102025/`
  - `__init__.py` - ลงทะเบียน environment
  - `humanoid_env.py` - environment configuration
  - `agents/` - agent configurations

**Scripts:**
- `/home/sncbot/IsaacLab/train_humanoid_10102025.sh` - training script
- `/home/sncbot/IsaacLab/test_humanoid_10102025_env.sh` - test script

## 🚀 วิธีเทรน

### เทรนแบบ Headless (แนะนำ)
```bash
./train_humanoid_10102025.sh
```

### เทรนแบบมี GUI
```bash
./train_humanoid_10102025.sh --gui
```

### ปรับแต่งพารามิเตอร์
```bash
./train_humanoid_10102025.sh --num_envs 2048 --max_iterations 10000
```

### เทรนด้วยคำสั่งโดยตรง
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Humanoid-10102025-Direct-v0 \
    --num_envs 4096 \
    --headless
```

## 📊 การตั้งค่า Environment

- **Environment ID**: `Isaac-Humanoid-10102025-Direct-v0`
- **Observation Space**: 75 dimensions
- **Action Space**: 21 dimensions (1 per joint)
- **Episode Length**: 15 seconds
- **Physics dt**: 1/120 seconds
- **Decimation**: 2
- **Default num_envs**: 4096

## 🎯 Reward Configuration

- **Heading weight**: 0.5 - รางวัลสำหรับการเดินไปข้างหน้า
- **Up weight**: 0.1 - รางวัลสำหรับการยืนตรง
- **Energy cost scale**: 0.05 - ลงโทษการใช้พลังงาน
- **Actions cost scale**: 0.01 - ลงโทษการเคลื่อนไหวมากเกินไป
- **Alive reward**: 2.0 - รางวัลสำหรับการอยู่รอด
- **Death cost**: -10.0 - ลงโทษเมื่อหกล้ม
- **Termination height**: 0.8m - ความสูงที่ถือว่าหกล้ม

## 📁 ผลการเทรน

ผลการเทรนจะถูกบันทึกที่:
```
logs/rsl_rl/isaac_humanoid_10102025_direct/YYYY-MM-DD_HH-MM-SS/
```

## 🎮 เล่นโมเดลที่เทรนแล้ว

หลังจากเทรนเสร็จ ให้เล่นโมเดลด้วย:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Humanoid-10102025-Direct-v0 \
    --num_envs 16 \
    --load_run logs/rsl_rl/isaac_humanoid_10102025_direct/YYYY-MM-DD_HH-MM-SS
```

## ⚙️ การปรับแต่งเพิ่มเติม

### เปลี่ยน Learning Rate
แก้ไขไฟล์: `source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_10102025/agents/rsl_rl_ppo_cfg.py`

### เปลี่ยน Reward Weights
แก้ไขไฟล์: `source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_10102025/humanoid_env.py`

### เปลี่ยน Actuator Parameters
แก้ไขไฟล์: `source/isaaclab_assets/isaaclab_assets/robots/humanoid_10102025.py`

## 🐛 Troubleshooting

### Out of Memory
ลด num_envs:
```bash
./train_humanoid_10102025.sh --num_envs 2048
```

### Robot Falls Immediately
ตรวจสอบ initial joint positions ใน robot config

### Slow Training
ใช้ headless mode และลด decimation

## 📝 Notes

- GPU แนะนำ: RTX 3090 ขึ้นไป หรือ A2000 12GB (ที่ใช้อยู่)
- RAM แนะนำ: 32GB ขึ้นไป
- Training time: ประมาณ 2-4 ชั่วโมงสำหรับ 5000 iterations

## 🎉 พร้อมเทรนแล้ว!

รันคำสั่งนี้เพื่อเริ่มเทรน:
```bash
./train_humanoid_10102025.sh
```

Good luck! 🤖
