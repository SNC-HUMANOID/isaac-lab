# 03_training - การเทรน RL

## ⚠️ สถานะ: กำลังแก้ไข
ไฟล์ในโฟลเดอร์นี้ยังมีปัญหาเรื่อง RSL-RL configuration

## 📋 ไฟล์ในโฟลเดอร์นี้

### 🔧 03_train_rsl_rl.py (ล่าสุด)
**สถานะ**: มีปัญหา std >= 0.0 error
**ใช้งาน**: `./isaaclab.sh -p humanoid_snc_scripts/03_training/03_train_rsl_rl.py --num_envs 32 --headless`
- พยายามใช้ RSL-RL wrapper
- มี environment configuration ครบ
- แต่ยัง error ตอน policy initialization

### ❌ train_humanoid_snc.py (เก่า)
**สถานะ**: ไม่ทำงาน
- มีปัญหา environment wrapper
- ขาด get_observations method

### 🔄 train_humanoid_snc_correct.py
**สถานะ**: ไม่ครบ
- พยายามใช้ RslRlVecEnvWrapper
- ไม่มี environment definition

## 🐛 ปัญหาที่พบ
1. **RSL-RL Configuration**: format ยุ่งยาก
2. **Policy Initialization**: std deviation ติดลบ
3. **Environment Wrapper**: ไม่ match กับ RSL-RL interface

## 💡 ทางออก
ใช้ไฟล์ใน `02_physics_tests/` แทน:
- `04_simple_movement.py` - เห็นการเดิน
- `02_test_physics.py` - ทดสอบพื้นฐาน

## 🎯 สำหรับอนาคต
- ศึกษา existing Isaac Lab RL environments
- ใช้ config ที่มีอยู่แล้วเป็นแม่แบบ
- หรือใช้ direct gymnasium interface