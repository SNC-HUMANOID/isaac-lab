# Troubleshooting: Left Leg Walking Backward

## ปัญหา
**ขาซ้ายยังคงก้าวถอยหลัง** แม้จะแก้ action scale dictionary แล้ว

## การแก้ไขที่ทำไปแล้ว ✅

### 1. Action Scale Dictionary
**ไฟล์:** `rough_env_cfg.py:112-136`
```python
"left_hip_pitch_joint": -0.5,   # กลับเครื่องหมาย
"right_hip_pitch_joint": 0.5,   # ปกติ
```
✅ **ยืนยัน:** อยู่ในโค้ดแล้ว

### 2. Initial Pose
**ไฟล์:** `humanoid_snc_r5.py:42`
```python
".*_hip_pitch_joint": -0.20,  # ทั้งสองขาเหมือนกัน
```
✅ **ยืนยัน:** ใช้ค่าเดียวกัน

## เหตุผลที่อาจยังไม่ได้

### สาเหตุที่เป็นไปได้:

#### 1. **Flat terrain vs Rough terrain config**
- คุณใช้ task ไหน? `Flat` หรือ `Rough`?
- Flat config อาจ inherit แต่ไม่ override scale

**ตรวจสอบ:**
```bash
grep -r "actions.joint_pos" source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/humanoid_snc_r5/flat_env_cfg.py
```

#### 2. **URDF axis ยังไม่ได้แก้**
- Scale dictionary ช่วย compensate แต่ initial pose อาจยังผิด
- เพราะ initial pose ใช้ค่าเดียวกัน แต่แกนกลับ → ขาซ้ายชี้หลังตั้งแต่เริ่ม

**แก้ไข:**
```python
# humanoid_snc_r5.py
"left_hip_pitch_joint": +0.20,   # บวก (กลับเครื่องหมาย)
"right_hip_pitch_joint": -0.20,  # ลบ (ปกติ)
```

#### 3. **Observation มีปัญหา**
- Policy อาจได้ observation ที่ผิด (joint pos ที่กลับกัน)
- ทำให้เรียนรู้ผิด

#### 4. **Reward function ผิด**
- Reward อาจให้คะแนนพฤติกรรมที่ผิด
- ทำให้ policy เรียนรู้ให้ขาซ้ายก้าวถอยหลัง

## แนวทางแก้ไข

### วิธีที่ 1: แก้ Initial Pose (แนะนำ)

เพราะ URDF มีแกนกลับ initial pose ต้องกลับเครื่องหมายด้วย:

```python
# source/isaaclab_assets/isaaclab_assets/robots/humanoid_snc_r5.py
joint_pos={
    "left_hip_pitch_joint": +0.20,   # ← บวก (compensate for inverted axis)
    "right_hip_pitch_joint": -0.20,  # ← ลบ (normal)
    # ... rest same
}
```

### วิธีที่ 2: ตรวจสอบ Flat config

```python
# flat_env_cfg.py
def __post_init__(self):
    super().__post_init__()
    # ต้องแน่ใจว่าไม่ได้ override actions.joint_pos.scale
```

### วิธีที่ 3: แก้ Observation

อาจต้องกลับเครื่องหมาย joint_pos observation ของ left hip ด้วย

### วิธีที่ 4: Debug Training

เพิ่ม log เพื่อดูว่า action ถูกส่งอย่างไร:

```python
# ใน rough_env_cfg.py หลัง scale
print(f"Actions after scale: left_hip={actions[:, left_hip_idx]}, right_hip={actions[:, right_hip_idx]}")
```

## ขั้นตอนถัดไป (เรียงตามลำดับ)

### 1. ✅ ลองแก้ initial pose
```python
# humanoid_snc_r5.py - line 42
"left_hip_pitch_joint": +0.20,
"right_hip_pitch_joint": -0.20,
```

### 2. ✅ เทรนใหม่ทันที
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Humanoid-SNC-R5-v0 \
    --num_envs 1024 \
    --headless
```

### 3. ✅ สังเกตพฤติกรรมตอนเริ่มต้น
- Reset แล้วขาซ้ายชี้หลังหรือไม่?
- ถ้าชี้หลัง → initial pose ผิด
- ถ้าชี้หน้า → แต่เดินถอยหลัง → action/reward ผิด

### 4. ❓ถ้ายังไม่ได้
บอกผมว่าเห็นอะไร:
- Episode reward เป็นเท่าไร?
- Episode length เป็นเท่าไร?
- ขาซ้ายตอนเริ่มต้นชี้ทิศไหน?
- เวลาเดินขาซ้ายทำอะไร?

## สรุป

**ทฤษฎี:** Scale dictionary ควรทำงาน ✓
**ปัญหา:** Initial pose อาจยังผิด (ใช้ค่าเดียวกันทั้งที่แกนกลับ)

**แก้ไข:**
1. Initial pose: `left=+0.20, right=-0.20`
2. Action scale: `left=-0.5, right=+0.5` (มีอยู่แล้ว)

→ **ทั้งสองจุดต้องกลับเครื่องหมาย** เพื่อ compensate แกนที่กลับกัน

---

ลองแก้ initial pose แล้วเทรนใหม่ดูครับ! 🚀
