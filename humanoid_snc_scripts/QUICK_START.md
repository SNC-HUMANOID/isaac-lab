# 🚀 Quick Start Guide

## 🎯 เริ่มต้นใช้งาน Humanoid SNC

### 1️⃣ ดู Model (เริ่มต้นที่นี่)
```bash
./isaaclab.sh -p humanoid_snc_scripts/01_basic/01_show_model.py
```
**ผลลัพธ์**: เห็นหุ่นยนต์ 3D, จำนวน joints, สามารถหมุนดูได้

### 2️⃣ ทดสอบ Physics
```bash
./isaaclab.sh -p humanoid_snc_scripts/02_physics_tests/02_test_physics.py --num_envs 4
```
**ผลลัพธ์**: หุ่นยนต์เคลื่อนไหวแบบสุ่ม, ดู reward scores

### 3️⃣ ดูการเดิน (แนะนำ!)
```bash
./isaaclab.sh -p humanoid_snc_scripts/02_physics_tests/04_simple_movement.py --num_envs 4
```
**ผลลัพธ์**: หุ่นยนต์พยายามเดินด้วย walking pattern

## 🔥 Commands ที่ใช้บ่อย

### แสดง Model
```bash
./isaaclab.sh -p humanoid_snc_scripts/01_basic/01_show_model.py
```

### ทดสอบ Physics (หลายหุ่นยนต์)
```bash
./isaaclab.sh -p humanoid_snc_scripts/02_physics_tests/02_test_physics.py --num_envs 8
```

### การเดิน (แบบ headless)
```bash
./isaaclab.sh -p humanoid_snc_scripts/02_physics_tests/04_simple_movement.py --num_envs 4 --headless
```

## 📁 โครงสร้างโฟลเดอร์
```
humanoid_snc_scripts/
├── 01_basic/          # เริ่มต้นที่นี่
├── 02_physics_tests/  # ทดสอบการทำงาน  
├── 03_training/       # เทรน RL (ยังมีปัญหา)
└── 04_configs/        # Configuration files
```

## 💡 Tips
- เริ่มด้วย `01_basic/01_show_model.py` เสมอ
- ใช้ `--headless` เพื่อประหยัด GPU
- ใช้ `--num_envs` เพื่อปรับจำนวนหุ่นยนต์
- กด `Ctrl+C` เพื่อหยุด
- ดู README ในแต่ละโฟลเดอร์สำหรับรายละเอียด

## ⚠️ หมายเหตุ
- ไฟล์ training ใน `03_training/` ยังมีปัญหา
- ใช้ไฟล์ใน `02_physics_tests/` สำหรับดูการทำงาน
- หุ่นยนต์มี 21 joints สามารถควบคุมได้