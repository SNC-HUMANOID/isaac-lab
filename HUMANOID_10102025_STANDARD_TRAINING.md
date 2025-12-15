# 🎯 Humanoid 10102025 - การเทรนแบบมาตรฐาน Isaac Lab

## ✅ ตอนนี้ใช้คำสั่งมาตรฐานได้แล้ว!

คุณสามารถใช้คำสั่งมาตรฐานของ Isaac Lab เหมือนกับหุ่นยนต์ตัวอื่นๆ เช่น H1, G1, หรือ SNC R5!

---

## 🚀 การเทรน

### 1. Flat Terrain (แนะนำสำหรับเริ่มต้น)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Humanoid-10102025-v0 \
    --num_envs 1024 \
    --headless
```

### 2. Rough Terrain (ยากขึ้น - พื้นขรุขระ)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Humanoid-10102025-v0 \
    --num_envs 1024 \
    --headless
```

---

## 🎮 การเล่น Policy

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Humanoid-10102025-Play-v0 \
    --num_envs 16 \
    --load_run logs/rsl_rl/velocity_flat_humanoid_10102025/YYYY-MM-DD_HH-MM-SS
```

---

## 📊 ผลลัพธ์

ไฟล์จะอยู่ที่:
```
logs/rsl_rl/velocity_flat_humanoid_10102025/YYYY-MM-DD_HH-MM-SS/
```

ดู TensorBoard:
```bash
tensorboard --logdir logs/rsl_rl/velocity_flat_humanoid_10102025/
```

---

## 🎯 Environments

- `Isaac-Velocity-Flat-Humanoid-10102025-v0` - พื้นเรียบ (เทรน)
- `Isaac-Velocity-Flat-Humanoid-10102025-Play-v0` - พื้นเรียบ (เล่น)
- `Isaac-Velocity-Rough-Humanoid-10102025-v0` - พื้นขรุขระ (เทรน)
- `Isaac-Velocity-Rough-Humanoid-10102025-Play-v0` - พื้นขรุขระ (เล่น)
