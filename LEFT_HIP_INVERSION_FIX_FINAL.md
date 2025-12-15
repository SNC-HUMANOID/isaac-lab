# Left Hip Inversion Fix - FINAL SOLUTION

## ปัญหา
**ขาขวาเดินหน้า แต่ขาซ้ายเดินถอยหลัง**

## สาเหตุ
URDF มีแกนหมุน hip pitch กลับกัน:
```xml
<joint name="left_hip_pitch_joint">
    <axis xyz="0.006, -0.9999, 0" />  ← Y เป็นลบ (ผิด)
</joint>

<joint name="right_hip_pitch_joint">
    <axis xyz="-0.006, +0.9999, 0" />  ← Y เป็นบวก (ถูก)
</joint>
```

เมื่อ policy ส่งคำสั่ง +0.5 ให้ทั้งสองข้อ:
- Right hip: หมุนไปข้างหน้า (ขาก้าวหน้า) ✓
- Left hip: หมุนไปข้างหลัง (ขาก้าวหลัง) ✗

## การแก้ไข

### 1. สร้าง Custom Action Class
**ไฟล์:** [inverted_hip_action.py](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/humanoid_snc_r5/inverted_hip_action.py)

```python
class InvertedLeftHipJointAction(JointPositionAction):
    """กลับเครื่องหมาย action ของ left_hip_pitch_joint"""

    def process_actions(self, actions: torch.Tensor):
        # Invert left hip pitch action
        if self._left_hip_idx is not None:
            actions[:, self._left_hip_idx] *= -1.0

        return super().process_actions(actions)
```

### 2. ใช้ Custom Class ในการตั้งค่า
**ไฟล์:** [rough_env_cfg.py:110-118](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/humanoid_snc_r5/rough_env_cfg.py#L110-L118)

```python
self.actions.joint_pos = JointPositionActionCfg(
    asset_name="robot",
    joint_names=[".*"],
    scale=0.5,
    use_default_offset=True,
    class_type=InvertedLeftHipJointAction,  # ← ใช้ custom class
)
```

## ผลลัพธ์

### ก่อนแก้ไข:
- Policy ส่ง: [+0.5, +0.5] สำหรับ [left, right]
- URDF ได้: [+0.5, +0.5]
- แกนทำงาน: [backward, forward] → **ขาซ้ายถอยหลัง** ✗

### หลังแก้ไข:
- Policy ส่ง: [+0.5, +0.5] สำหรับ [left, right]
- Custom class แก้: [**-0.5**, +0.5]
- URDF ได้: [-0.5, +0.5]
- แกนทำงาน: [forward, forward] → **ทั้งสองขาเดินหน้า** ✓

## การทดสอบ

เมื่อเทรน คุณจะเห็น log message:
```
[InvertedLeftHipAction] Found left_hip_pitch_joint at index X
[InvertedLeftHipAction] Will INVERT action for this joint to fix axis bug
```

## คำสั่งเทรน

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Humanoid-SNC-R5-v0 \
    --num_envs 4096 \
    --headless
```

## ผลที่คาดหวัง

✓ **ขาซ้ายและขาขวาเดินหน้าพร้อมกัน**
✓ Episode length เพิ่มขึ้น (หุ่นยนต์ไม่ล้ม)
✓ Reward เพิ่มขึ้น (เดินได้ถูกทิศทาง)
✓ Robot เรียนรู้การเดินได้

---

**สถานะ:** ✅ แก้ไขเสร็จสมบูรณ์ - พร้อมเทรน
