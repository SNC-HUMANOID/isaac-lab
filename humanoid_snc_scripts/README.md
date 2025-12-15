# 🤖 Humanoid SNC Scripts Collection

## 📁 โครงสร้างโฟลเดอร์

### 01_basic/ - สคริปต์พื้นฐาน
```
01_basic/
├── 01_show_model.py          # แสดง model 3D (เริ่มต้น)
├── show_humanoid_model.py    # แสดง model แบบเก่า
├── test_humanoid_snc.py      # ทดสอบการโหลด robot
└── visualize_humanoid_snc.py # แสดง robot พร้อม control
```

### 02_physics_tests/ - ทดสอบ Physics
```
02_physics_tests/
├── 02_test_physics.py        # ทดสอบ physics พื้นฐาน
├── 04_simple_movement.py     # การเคลื่อนไหวแบบ pattern
└── simple_train_humanoid_snc.py # ทดสอบ environment
```

### 03_training/ - การเทรน RL
```
03_training/
├── 03_train_rsl_rl.py        # เทรน RSL-RL (ใหม่ล่าสุด)
├── train_humanoid_snc.py     # เทรนแบบเก่า (มีปัญหา)
└── train_humanoid_snc_correct.py # เทรนแบบแก้ไข
```

### 04_configs/ - Configuration Files
```
04_configs/
└── humanoid_snc_locomotion.py # Environment configuration
```

## 🚀 วิธีใช้งาน

### 1️⃣ เริ่มต้น - ดู Model
```bash
./isaaclab.sh -p humanoid_snc_scripts/01_basic/01_show_model.py
```

### 2️⃣ ทดสอบ Physics
```bash
./isaaclab.sh -p humanoid_snc_scripts/02_physics_tests/02_test_physics.py --num_envs 4
```

### 3️⃣ ทดสอบการเคลื่อนไหว
```bash
./isaaclab.sh -p humanoid_snc_scripts/02_physics_tests/04_simple_movement.py --num_envs 4
```

### 4️⃣ เทรน RL (ยังมีปัญหา)
```bash
./isaaclab.sh -p humanoid_snc_scripts/03_training/03_train_rsl_rl.py --num_envs 32 --max_iterations 100 --headless
```

## 📊 ข้อมูลหุ่นยนต์
- **จำนวน joints**: 21 ข้อต่อ
- **Action space**: (batch_size, 21)
- **Observation space**: (batch_size, 75)
- **Robot height**: ~1.0-1.2 เมตร

## 🎯 แนะนำการใช้งาน
1. **เริ่มต้น**: `01_basic/01_show_model.py` - ดู model
2. **ทดสอบ**: `02_physics_tests/02_test_physics.py` - ทดสอบ physics
3. **เคลื่อนไหว**: `02_physics_tests/04_simple_movement.py` - เห็นการเดิน
4. **เทรน**: `03_training/03_train_rsl_rl.py` - เทรน RL (กำลังแก้ไข)

## ⚠️ หมายเหตุ
- ไฟล์ใน `03_training/` ยังมีปัญหาเรื่อง RSL-RL configuration
- ใช้ไฟล์ใน `02_physics_tests/` สำหรับดูการทำงาน
- ไฟล์ใน `01_basic/` เหมาะสำหรับเริ่มต้น