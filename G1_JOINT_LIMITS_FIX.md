# ✅ แก้ไข Initial Joint Positions สำหรับ G1 Limits

## ปัญหาที่พบ

```
ValueError: The following joints have default positions out of the limits: 
	- 'left_elbow_joint': 0.870 not in [-2.230, 0.090]
	- 'right_elbow_joint': 0.870 not in [-2.230, 0.090]
```

## สาเหตุ

Initial joint positions ที่ตั้งไว้เดิมไม่เข้ากับ G1 joint limits:
- Elbow เดิม: 0.87 rad 
- G1 elbow range: [-2.23, 0.09] rad
- **0.87 อยู่นอก range!**

## การแก้ไข

อัพเดทไฟล์: `source/isaaclab_assets/isaaclab_assets/robots/humanoid_10102025.py`

### ก่อนแก้ไข
```python
# Arms
".*_shoulder_pitch_joint": 0.35,
"left_shoulder_roll_joint": 0.16,
"right_shoulder_roll_joint": -0.16,
".*_elbow.*": 0.87,  # ❌ นอก range!
```

### หลังแก้ไข
```python
# Arms - updated for G1 joint limits
".*_shoulder_pitch_joint": 0.0,
"left_shoulder_roll_joint": 0.5,
"right_shoulder_roll_joint": -0.5,
".*_elbow.*": -0.5,  # ✅ ใน range [-2.23, 0.09]
```

## ตรวจสอบ Initial Positions ทั้งหมด

| Joint | Initial Pos | G1 Limit | ✓/✗ |
|-------|-------------|----------|-----|
| **Legs** |
| hip_pitch | -0.25 | [-0.43, 2.79] | ✅ |
| hip_roll | 0.0 | [-0.43, 0.43] | ✅ |
| hip_yaw | 0.0 | [-0.43, 0.43] | ✅ |
| knee | 0.5 | [-0.26, 2.05] | ✅ |
| ankle_pitch | -0.25 | [-0.87, 0.52] | ✅ |
| ankle_roll | 0.0 | [-0.26, 0.26] | ✅ |
| **Torso** |
| pelvis | 0.0 | [-2.79, 2.79] | ✅ |
| **Arms** |
| shoulder_pitch | 0.0 | [-2.87, 2.87] | ✅ |
| shoulder_roll (L) | 0.5 | [-1.57, 3.11] | ✅ |
| shoulder_roll (R) | -0.5 | [-3.11, 1.57] | ✅ |
| shoulder_yaw (L) | 0.0 | [-3.01, 1.32] | ✅ |
| shoulder_yaw (R) | 0.0 | [-1.32, 3.01] | ✅ |
| elbow | -0.5 | [-2.23, 0.09] | ✅ |

## ท่าทางเริ่มต้นของโรบอท

```
        Arms bent at elbows
            \  |  /
             \ | /
    Left      \|/      Right
    shoulder   O   shoulder
               |
             Torso
               |
         Bent knees
        /           \
       /             \
   Left leg      Right leg
```

### ลักษณะท่าทาง:
- **Legs**: เข่างอเล็กน้อย พร้อมยืน
- **Arms**: แขนอยู่ข้างตัว ข้อศอกงอ
- **Torso**: ตั้งตรง

## ✅ สรุป

การแก้ไข:
1. เปลี่ยน elbow จาก 0.87 → -0.5 rad
2. ปรับ shoulder positions ให้เหมาะสม
3. ตรวจสอบว่าทุก joint อยู่ใน G1 limits

## 🚀 พร้อมเทรนแล้ว!

```bash
./train_humanoid_10102025.sh
```

