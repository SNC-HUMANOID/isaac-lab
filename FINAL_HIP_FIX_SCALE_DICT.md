# Hip Axis Fix - Scale Dictionary Solution

## ปัญหา
**ขาซ้ายก้าวถอยหลัง แต่ขาขวาก้าวไปข้างหน้า**

## สาเหตุ
URDF มีแกนหมุน hip pitch กลับกัน:
- Left hip axis: Y = **-0.9999** (ลบ)
- Right hip axis: Y = **+0.9999** (บวก)

→ คำสั่งเดียวกัน → ทิศทางตรงกันข้าม

## การแก้ไข FINAL

### 1. Initial Pose: ใช้ค่าเดียวกัน
```python
# humanoid_snc_r5.py
".*_hip_pitch_joint": -0.20,  # ทั้งสองขาใช้ค่าเดียวกัน
```

### 2. Action Scale: กลับเครื่องหมายขาซ้าย
```python
# rough_env_cfg.py
self.actions.joint_pos.scale = {
    "left_hip_pitch_joint": -0.5,   # ← ลบ (กลับเครื่องหมาย)
    "right_hip_pitch_joint": 0.5,   # ← บวก (ปกติ)
    # ... ข้อต่ออื่นๆ ใช้ 0.5
}
```

## วิธีการทำงาน

```
Policy ส่ง: [+1.0, +1.0] (left, right - เหมือนกัน)
       ↓
Scale apply:
  Left:  +1.0 × (-0.5) = -0.5
  Right: +1.0 × (+0.5) = +0.5
       ↓
URDF axis apply:
  Left:  -0.5 × (Y=-0.9999) = +0.5 (forward!)
  Right: +0.5 × (Y=+0.9999) = +0.5 (forward!)
       ↓
ผลลัพธ์: ✅ ทั้งสองขาก้าวหน้าเหมือนกัน!
```

## ไฟล์ที่แก้

1. **[humanoid_snc_r5.py:42](source/isaaclab_assets/isaaclab_assets/robots/humanoid_snc_r5.py#L42)**
   - Initial pose ใช้ค่าเดียวกันทั้งสองขา

2. **[rough_env_cfg.py:112-136](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/humanoid_snc_r5/rough_env_cfg.py#L112-L136)**
   - Scale dictionary ที่ระบุทุกข้อต่อแยกกัน
   - `left_hip_pitch_joint: -0.5` (กลับเครื่องหมาย)

## คำสั่งเทรน

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Humanoid-SNC-R5-v0 \
    --num_envs 4096 \
    --headless
```

## ผลที่คาดหวัง

✅ ขาซ้ายและขาขวาก้าวไปข้างหน้าพร้อมกัน
✅ ตำแหน่งขาสมมาตร
✅ Episode length เพิ่มขึ้น
✅ Reward เพิ่มขึ้น
✅ Robot เรียนรู้เดินได้

---

**สถานะ:** ✅ แก้ไขเสร็จสมบูรณ์ - ใช้ scale dictionary (ไม่ใช่ custom action class)
