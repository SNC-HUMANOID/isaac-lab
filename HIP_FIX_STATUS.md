# Hip Axis Fix - Current Status

## ปัญหา
ขาซ้ายไม่ก้าวไปข้างหน้า - อาจก้าวถอยหลังเพราะแกนหมุนกลับกัน

## สาเหตุที่พบ
URDF มีแกนหมุน hip pitch กลับกัน:
- Left hip: `axis xyz="0.006, -0.9999, 0"` (Y เป็นลบ)
- Right hip: `axis xyz="-0.006, +0.9999, 0"` (Y เป็นบวก)

## การแก้ไขที่ทำไปแล้ว

### ✓ 1. Action Scale Dictionary
**ไฟล์:** `rough_env_cfg.py:108-135`

```python
self.actions.joint_pos.scale = {
    "left_hip_pitch_joint": -0.5,  # ← กลับเครื่องหมาย
    # ... ข้อต่ออื่นๆ ใช้ +0.5
}
```

**ทฤษฎี:** Action scale ควรชดเชยแกนที่กลับกัน
- Policy ส่ง +0.5 → Left hip ได้ -0.25 → URDF axis กลับอีกที → สุทธิเป็น +0.25 (เดินหน้า)

### ✓ 2. สร้าง URDF ที่แก้แกนหมุนแล้ว
**ไฟล์:** `Humanoid_SNC_R5_fixed_hip_axis.urdf`
- แก้ left hip axis เป็น: `-0.006, +0.9999, 0` (เหมือน right hip)
- **แต่ยังไม่ได้แปลงเป็น USD** เพราะ Isaac Sim environment ไม่พร้อม

## ปัญหาที่ยังเหลือ

### 1. USD Conversion ไม่สำเร็จ
```
[ERROR] Unable to find any Python executable at path: '_isaac_sim/python.sh'
```

**สาเหตุ:** Isaac Sim ไม่ได้ติดตั้งหรือ environment ไม่ถูกต้อง

### 2. Action Scale อาจไม่ทำงาน
แม้ตั้งค่าแล้ว แต่:
- ไม่สามารถทดสอบว่าโหลดจริงหรือไม่ (ไม่มี Isaac Sim)
- อาจมีปัญหาอื่นที่ทำให้เดินไม่ได้

## แนวทางแก้ไข

### ทางเลือกที่ 1: แก้ USD ด้วยมือ (แนะนำ)
เนื่องจากไม่สามารถรัน Isaac Sim ได้ ให้:

1. **คัดลอก USD เดิม**
   ```bash
   cp mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_fixed_axis.usd \
      mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_hip_fixed.usd
   ```

2. **แก้ไขใน USD Editor (ถ้ามี)** หรือ
3. **ใช้ action scale dictionary ต่อไป** แต่ต้องแน่ใจว่า:
   - Configuration ถูกโหลด
   - ไม่มีปัญหาอื่น (reward, stiffness, etc.)

### ทางเลือกที่ 2: ตรวจสอบปัญหาอื่น

อาจมีปัญหาอื่นที่ทำให้เดินไม่ได้:

#### A. Initial Pose ไม่เหมาะสม
ตรวจสอบใน `humanoid_snc_r5.py`:
```python
"left_hip_pitch_joint": -0.20,   # ถูกต้องหรือไม่?
"right_hip_pitch_joint": -0.20,  # เหมือนกันหรือไม่?
```

#### B. Stiffness/Damping ไม่เหมาะ
```python
# Hip
stiffness: 150-200
damping: 5.0

# Knee
stiffness: 200
damping: 5.0
```

#### C. Reward Function
- `feet_air_time`: weight=2.0 (สูงเกินไป?)
- `flat_orientation_l2`: weight=-0.2 (ต่ำเกินไป?)

### ทางเลือกที่ 3: ลองเทรน G1 เปรียบเทียบ

รัน G1 environment เพื่อดูว่าเดินได้หรือไม่:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-G1-v0 \
    --num_envs 1024 \
    --headless
```

ถ้า G1 เดินได้ → ปัญหาอยู่ที่ SNC R5 config
ถ้า G1 ก็เดินไม่ได้ → ปัญหาอยู่ที่ training setup

## คำถามสำหรับคุณ

1. **G1 เทรนได้หรือไม่?** ลองเทรนดูหรือยัง?
2. **SNC R5 ล้มเร็วมากหรือไม่?** Episode length เท่าไร?
3. **Reward เป็นลบมากหรือไม่?** Episode reward เท่าไร?
4. **เห็นอะไรเวลาเทรน?** ขาซ้ายไม่ขยับเลย หรือ ขยับแต่ทิศทางผิด?

## ขั้นตอนถัดไป (ตามลำดับความสำคัญ)

1. ✓ **Action scale dictionary พร้อมใช้แล้ว** - ลองเทรนดู
2. ⏸ **USD conversion รออุปกรณ์พร้อม** - ต้องการ Isaac Sim
3. ⏸ **ตรวจสอบ config อื่นๆ** - ถ้ายังเดินไม่ได้

---

**สถานะปัจจุบัน:** Action scale fix พร้อมแล้ว แต่ต้อง**ทดสอบจริง**เพื่อดูผล
