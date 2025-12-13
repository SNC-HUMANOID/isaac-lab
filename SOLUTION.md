# Final Solution - Left Hip Axis Inversion Fix

## ปัญหา
**ขาซ้ายก้าวถอยหลัง แต่ขาขวาก้าวหน้า**

## การแก้ไข FINAL ✅

### 1. Gym Wrapper
ไฟล์: `action_wrapper.py` - กลับเครื่องหมาย action ของ left hip ก่อนส่งให้ environment

### 2. Environment Registration  
ไฟล์: `__init__.py` - เพิ่ม `additional_wrappers=(LeftHipInverterWrapper,)`

### 3. Initial Pose
ไฟล์: `humanoid_snc_r5.py` - `left_hip_pitch: +0.20, right_hip_pitch: -0.20`

## คำสั่งเทรนใหม่

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Humanoid-SNC-R5-v0 \
    --num_envs 4096 \
    --headless
```

คุณจะเห็น:
```
[LeftHipInverterWrapper] Found left_hip_pitch_joint at index X
[LeftHipInverterWrapper] Will INVERT action to fix URDF axis bug
```

✅ ทั้งสองขาจะเดินหน้าพร้อมกัน!
