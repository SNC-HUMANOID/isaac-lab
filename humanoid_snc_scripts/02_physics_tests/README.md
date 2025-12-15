# 02_physics_tests - ทดสอบ Physics

## 📋 ไฟล์ในโฟลเดอร์นี้

### ✅ 02_test_physics.py (แนะนำ)
**ใช้งาน**: `./isaaclab.sh -p humanoid_snc_scripts/02_physics_tests/02_test_physics.py --num_envs 4`
- ทดสอบ physics simulation
- ใช้ random actions
- ดู reward และ observation
- **เหมาะสำหรับ**: ทดสอบว่า environment ทำงาน

### 🚶‍♂️ 04_simple_movement.py (น่าสนใจ)
**ใช้งาน**: `./isaaclab.sh -p humanoid_snc_scripts/02_physics_tests/04_simple_movement.py --num_envs 4`
- การเคลื่อนไหวด้วย walking pattern
- ใช้ sine wave สร้าง gait
- ดูการเดินแบบมีจุดประสงค์
- **เหมาะสำหรับ**: เห็นหุ่นยนต์พยายามเดิน

### 🔬 simple_train_humanoid_snc.py
**ใช้งาน**: `./isaaclab.sh -p humanoid_snc_scripts/02_physics_tests/simple_train_humanoid_snc.py --num_envs 4`
- ทดสอบ environment สำหรับ training
- ไม่ใช่การเทรนจริง
- แค่ทดสอบว่าระบบพร้อม

## 🎯 ลำดับการทดสอบ
1. **Physics**: `02_test_physics.py` - ดูว่าหุ่นยนต์มีชีวิต
2. **Movement**: `04_simple_movement.py` - ดูการเคลื่อนไหว
3. **Environment**: `simple_train_humanoid_snc.py` - ทดสอบ RL env