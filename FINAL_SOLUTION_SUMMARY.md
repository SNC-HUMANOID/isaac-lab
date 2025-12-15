# Final Solution Summary - Left Leg Issue

## สถานะปัจจุบัน

### ✅ การแก้ไขที่ทำแล้ว:

1. **Action Scale Dictionary** - กลับเครื่องหมาย left hip
   ```python
   # rough_env_cfg.py:115
   "left_hip_pitch_joint": -0.5,   # NEGATIVE
   "right_hip_pitch_joint": 0.5,   # POSITIVE
   ```

2. **Initial Pose** - ใช้ค่าเดียวกัน
   ```python
   # humanoid_snc_r5.py:42
   ".*_hip_pitch_joint": -0.20,  # G1 EXACT
   ```

### ❌ ปัญหาที่เหลือ:

**Initial pose ใช้ค่าเดียวกัน แต่แกนกลับกัน → ขาซ้ายชี้หลัง**

```
Left hip:  value=-0.20 × axis(Y=-0.9999) = +0.20 (ชี้หลัง!)
Right hip: value=-0.20 × axis(Y=+0.9999) = -0.20 (ชี้หน้า)
```

## วิธีแก้ 3 ทาง

### ทางที่ 1: แก้ Initial Pose (ง่ายที่สุด) ⭐

```python
# humanoid_snc_r5.py
joint_pos={
    "left_hip_pitch_joint": +0.20,   # กลับเครื่องหมาย
    "right_hip_pitch_joint": -0.20,  # ปกติ
    # ...
}
```

**ผลลัพธ์:**
- ทั้งสองขาชี้หน้าเมื่อเริ่ม ✓
- Action scale จัดการเวลาเดิน ✓

### ทางที่ 2: ใช้ Initial State Event (ซับซ้อน)

สร้าง event ที่กลับเครื่องหมาย left hip หลัง reset:

```python
# rough_env_cfg.py
self.events.reset_left_hip = EventTerm(
    func=custom_reset_left_hip,
    mode="reset",
)
```

### ทางที่ 3: แก้ URDF และสร้าง USD ใหม่ (ถาวร)

```bash
# แก้ URDF ให้แกนเหมือนกัน
# แล้วแปลงเป็น USD ใหม่
./isaaclab.sh -p scripts/tools/convert_urdf.py \
    mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_fixed_hip_axis.urdf \
    mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_fixed_hip_axis.usd
```

จากนั้นลบ action scale dictionary ออกทั้งหมด

## แนะนำ: ทางที่ 1 ⭐

แก้ initial pose ให้ใช้ค่าตรงกันข้าม:

```python
# source/isaaclab_assets/isaaclab_assets/robots/humanoid_snc_r5.py line 40-44
joint_pos={
    "left_hip_pitch_joint": +0.20,   # ← CHANGE TO POSITIVE
    "right_hip_pitch_joint": -0.20,  # ← KEEP NEGATIVE
    ".*_knee_joint": 0.42,
    ".*_ankle_pitch_joint": -0.23,
}
```

**เหตุผล:**
- ง่าย ใช้เวลา 1 นาที
- ไม่ต้องรอ USD conversion
- ทำงานร่วมกับ action scale dictionary ที่มีอยู่

## คำสั่งแก้ไข

```bash
# 1. แก้ initial pose
nano source/isaaclab_assets/isaaclab_assets/robots/humanoid_snc_r5.py

# เปลี่ยนบรรทัด 42 จาก:
".*_hip_pitch_joint": -0.20,

# เป็น:
"left_hip_pitch_joint": +0.20,
"right_hip_pitch_joint": -0.20,

# 2. เทรนใหม่
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Humanoid-SNC-R5-v0 \
    --num_envs 4096 \
    --headless
```

## ผลที่คาดหวัง

✅ ขาซ้ายชี้หน้าตั้งแต่เริ่มต้น (ไม่ชี้หลัง)
✅ เวลาเดิน ทั้งสองขาก้าวหน้าพร้อมกัน
✅ Episode length เพิ่มขึ้น (50+)
✅ Reward เพิ่มขึ้น (0 → positive)
✅ Robot เรียนรู้เดินได้

## หาก USD Conversion สำเร็จ

ถ้า background process convert URDF เสร็จ:

```bash
# 1. ใช้ USD ใหม่
# แก้ humanoid_snc_r5.py line ~15:
usd_path="...Humanoid_SNC_R5_fixed_hip_axis.usd"

# 2. ลบ action scale dictionary
# แก้ rough_env_cfg.py line 112-136 เป็น:
self.actions.joint_pos.scale = 0.5

# 3. ใช้ initial pose ปกติ
".*_hip_pitch_joint": -0.20,
```

แต่ตอนนี้แนะนำ **ทางที่ 1** ก่อน - แก้ initial pose!
