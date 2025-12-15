# ✅ อัพเดท Joint Limits เป็น G1 Spec สำเร็จ!

## สรุปการอัพเดท

### 1. ไฟล์ที่สร้าง

**URDF ใหม่ (พร้อม G1 limits):**
- `/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/urdf/Humanoid_10102025_g1_limits.urdf`

**USD ใหม่ (converted จาก URDF ด้านบน):**
- Main: `/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/Humanoid_10102025_g1.usd`
- Base: `/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/configuration/Humanoid_10102025_g1_base.usd` (95 MB)
- Physics: `/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/configuration/Humanoid_10102025_g1_physics.usd`
- Sensor: `/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/configuration/Humanoid_10102025_g1_sensor.usd`

### 2. Joint Limits ที่อัพเดท (21 joints)

#### Leg Joints

**Hip Pitch** (ซ้าย-ขวา)
- Range: [-0.43, 2.79] rad (-24.6° to 160°)
- Effort: 200 Nm
- Velocity: 30 rad/s

**Hip Roll** (ซ้าย-ขวา)
- Range: [-0.43, 0.43] rad (-24.6° to 24.6°)
- Effort: 200 Nm
- Velocity: 30 rad/s

**Hip Yaw** (ซ้าย-ขวา)
- Range: [-0.43, 0.43] rad (-24.6° to 24.6°)
- Effort: 200 Nm
- Velocity: 30 rad/s

**Knee** (ซ้าย-ขวา)
- Range: [-0.26, 2.05] rad (-14.9° to 117.5°)
- Effort: 300 Nm
- Velocity: 30 rad/s

**Ankle Pitch** (ซ้าย-ขวา)
- Range: [-0.87, 0.52] rad (-49.8° to 29.8°)
- Effort: 45 Nm
- Velocity: 30 rad/s

**Ankle Roll** (ซ้าย-ขวา)
- Range: [-0.26, 0.26] rad (-14.9° to 14.9°)
- Effort: 45 Nm
- Velocity: 30 rad/s

#### Torso Joint

**Pelvis (Waist)**
- Range: [-2.79, 2.79] rad (-160° to 160°)
- Effort: 200 Nm
- Velocity: 30 rad/s

#### Arm Joints

**Shoulder Pitch** (ซ้าย-ขวา)
- Range: [-2.87, 2.87] rad (-164.5° to 164.5°)
- Effort: 45 Nm
- Velocity: 30 rad/s

**Shoulder Roll**
- Left: [-1.57, 3.11] rad (-90° to 178°)
- Right: [-3.11, 1.57] rad (-178° to 90°)
- Effort: 45 Nm
- Velocity: 30 rad/s

**Shoulder Yaw**
- Left: [-3.01, 1.32] rad (-172.5° to 75.6°)
- Right: [-1.32, 3.01] rad (-75.6° to 172.5°)
- Effort: 45 Nm
- Velocity: 30 rad/s

**Elbow** (ซ้าย-ขวา)
- Range: [-2.23, 0.09] rad (-127.7° to 5.2°)
- Effort: 45 Nm
- Velocity: 30 rad/s

### 3. การเปลี่ยนแปลงจากเดิม

| Joint Type | เดิม | G1 Spec |
|-----------|------|---------|
| **Leg Joints** |
| Hip effort | 25 Nm | 200 Nm |
| Knee effort | 25 Nm | 300 Nm |
| Ankle effort | 12 Nm | 45 Nm |
| **Arm Joints** |
| Shoulder effort | 27 Nm | 45 Nm |
| Elbow effort | 12 Nm | 45 Nm |
| **Velocity** |
| All joints | 20 rad/s | 30 rad/s |

### 4. ข้อดีของ G1 Limits

✅ **Effort สูงขึ้น** - แรงบิดมากขึ้น ทำให้ควบคุมได้ดีกว่า
✅ **Velocity เร็วขึ้น** - เคลื่อนไหวได้รวดเร็วกว่า
✅ **Range เหมาะสม** - จำกัดขอบเขตการเคลื่อนไหวที่เป็นธรรมชาติ
✅ **Realistic** - ใกล้เคียงกับโรบอทจริงมากขึ้น

## 🚀 วิธีใช้งาน

### ตัวเลือก 1: ใช้ตามที่ตั้งค่าไว้ (อัพเดทอัตโนมัติแล้ว)

Robot config ได้อัพเดทให้ใช้ USD ใหม่แล้ว:
```bash
./train_humanoid_10102025.sh
```

### ตัวเลือก 2: เทียบเคียง USD เก่ากับใหม่

**USD เก่า (original limits):**
```python
usd_path="/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/Humanoid_10102025.usd"
```

**USD ใหม่ (G1 limits):**
```python
usd_path="/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/Humanoid_10102025_g1.usd"
```

## 📊 ตาราง Joint Limits แบบละเอียด

```
Joint Name                  | Lower   | Upper  | Range   | Effort | Velocity
---------------------------|---------|--------|---------|--------|----------
left_hip_pitch_joint       | -0.43   | 2.79   | 3.22    | 200    | 30
left_hip_roll_joint        | -0.43   | 0.43   | 0.86    | 200    | 30
left_hip_yaw_joint         | -0.43   | 0.43   | 0.86    | 200    | 30
left_knee_joint            | -0.26   | 2.05   | 2.31    | 300    | 30
left_ankle_pitch_joint     | -0.87   | 0.52   | 1.39    | 45     | 30
left_ankle_roll_joint      | -0.26   | 0.26   | 0.52    | 45     | 30
right_hip_pitch_joint      | -0.43   | 2.79   | 3.22    | 200    | 30
right_hip_roll_joint       | -0.43   | 0.43   | 0.86    | 200    | 30
right_hip_yaw_joint        | -0.43   | 0.43   | 0.86    | 200    | 30
right_knee_joint           | -0.26   | 2.05   | 2.31    | 300    | 30
right_ankle_pitch_joint    | -0.87   | 0.52   | 1.39    | 45     | 30
right_ankle_roll_joint     | -0.26   | 0.26   | 0.52    | 45     | 30
pelvis_joint               | -2.79   | 2.79   | 5.58    | 200    | 30
left_shoulder_pitch_joint  | -2.87   | 2.87   | 5.74    | 45     | 30
left_shoulder_roll_joint   | -1.57   | 3.11   | 4.68    | 45     | 30
left_shoulder_yaw_joint    | -3.01   | 1.32   | 4.33    | 45     | 30
left_elbow_joint           | -2.23   | 0.09   | 2.32    | 45     | 30
right_shoulder_pitch_joint | -2.87   | 2.87   | 5.74    | 45     | 30
right_shoulder_roll_joint  | -3.11   | 1.57   | 4.68    | 45     | 30
right_shoulder_yaw_joint   | -1.32   | 3.01   | 4.33    | 45     | 30
right_elbow_joint          | -2.23   | 0.09   | 2.32    | 45     | 30
```

## 🎯 คำสั่งที่ใช้

### 1. อัพเดท URDF
```bash
python3 update_joint_limits_g1.py
```

### 2. Convert เป็น USD
```bash
./isaaclab.sh -p scripts/tools/convert_urdf.py \
  /home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/urdf/Humanoid_10102025_g1_limits.urdf \
  /home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/Humanoid_10102025_g1.usd \
  --merge-joints
```

### 3. เทรนด้วย USD ใหม่
```bash
./train_humanoid_10102025.sh
```

## ✅ เสร็จสมบูรณ์!

โมเดล Humanoid_10102025 ได้รับการอัพเดท joint limits ให้เป็นมาตรฐาน G1 humanoid แล้ว พร้อมเทรน! 🎉

