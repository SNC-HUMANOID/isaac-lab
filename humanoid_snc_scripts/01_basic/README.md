# 01_basic - สคริปต์พื้นฐาน

## 📋 ไฟล์ในโฟลเดอร์นี้

### ✅ 01_show_model.py (แนะนำ)
**ใช้งาน**: `./isaaclab.sh -p humanoid_snc_scripts/01_basic/01_show_model.py`
- แสดง robot model 3D
- ดูจำนวน joints และชื่อ
- หมุนดูได้รอบๆ
- **เหมาะสำหรับ**: เริ่มต้นครั้งแรก

### 📄 show_humanoid_model.py
**ใช้งาน**: `./isaaclab.sh -p humanoid_snc_scripts/01_basic/show_humanoid_model.py`
- แสดง model แบบเก่า
- มี logging ละเอียดกว่า

### 🔧 test_humanoid_snc.py
**ใช้งาน**: `./isaaclab.sh -p humanoid_snc_scripts/01_basic/test_humanoid_snc.py --headless`
- ทดสอบการโหลด robot
- ใช้ low-level API

### 🎮 visualize_humanoid_snc.py
**ใช้งาน**: `./isaaclab.sh -p humanoid_snc_scripts/01_basic/visualize_humanoid_snc.py --num_envs 1`
- แสดง robot พร้อม position control
- พยายามให้หุ่นยนต์ยืน

## 🎯 เริ่มต้นที่นี่
```bash
./isaaclab.sh -p humanoid_snc_scripts/01_basic/01_show_model.py
```