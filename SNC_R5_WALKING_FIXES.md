# SNC R5 Walking Fixes - แก้ปัญหาลำตัวเอียงหลัง & ก้าวยาวเกิน

## 🚨 ปัญหาที่พบ

1. **ลำตัวเอียงไปด้านหลัง** - ทำให้เสียสมดุลและดูไม่เป็นธรรมชาติ
2. **ก้าวขายาวเกินไป** - hip และ knee เคลื่อนไหวมากเกิน ทำให้การเดินไม่เป็นธรรมชาติ

## ✅ การแก้ไขที่ทำ

### 1. Improved Torso Orientation Control

```python
# Extract pitch angle from quaternion (detect backward lean)
pitch = torch.atan2(2.0 * (base_quat[:, 3] * base_quat[:, 1] - base_quat[:, 2] * base_quat[:, 0]),
                   1.0 - 2.0 * (base_quat[:, 1]**2 + base_quat[:, 2]**2))

# Strong penalty for backward lean (negative pitch)
torso_upright_reward = torch.exp(-torch.abs(pitch) * 3.0) * torch.exp(-torch.abs(roll) * 2.0)
backward_lean_penalty = torch.clamp(-pitch * 5.0, min=0.0, max=2.0)
```

**ผลลัพธ์**: 
- ✅ ตรวจจับและลงโทษการเอียงหลังโดยตรง
- ✅ รักษาท่าทางตั้งตรง (upright posture)

### 2. Step Length Control

```python
# Limit joint ranges for natural step length
# Hip pitch should be moderate (-0.3 to 0.3 radians)
step_length_penalty += torch.clamp(torch.abs(left_hip_pitch) - 0.3, min=0.0)**2
step_length_penalty += torch.clamp(torch.abs(right_hip_pitch) - 0.3, min=0.0)**2

# Knee should bend moderately (0 to 0.8 radians)  
step_length_penalty += torch.clamp(torch.abs(left_knee) - 0.8, min=0.0)**2
step_length_penalty += torch.clamp(torch.abs(right_knee) - 0.8, min=0.0)**2
```

**ผลลัพธ์**:
- ✅ จำกัดการเคลื่อนไหวของ hip pitch ไม่เกิน ±0.3 radians (~17 degrees)
- ✅ จำกัดการงอ knee ไม่เกิน 0.8 radians (~46 degrees)
- ✅ ป้องกันก้าวขายาวเกินปกติ

### 3. Natural Gait Timing

```python
# Slower gait phase progression for more natural walking
self.gait_phase += 0.05  # Reduced from 0.1

# Bounded gait reward to prevent extreme movements
gait_reward = torch.tanh(leg_alternation * 0.5)
```

**ผลลัพธ์**:
- ✅ จังหวะการเดินช้าลงและเป็นธรรมชาติมากขึ้น
- ✅ ป้องกันการเคลื่อนไหวขาที่รุนแรง

### 4. Controlled Arm Movement

```python
# Moderate arm swing with limits
natural_arm_swing = (left_shoulder_pitch - right_shoulder_pitch) * torch.sin(self.gait_phase) * 0.1
arm_swing_reward = torch.tanh(natural_arm_swing)

# Penalty for excessive arm movement
arm_movement_penalty = torch.sum(torch.abs(dof_pos[:, 0:10]), dim=-1) * 0.1
```

**ผลลัพธ์**:
- ✅ แขนแกว่งแบบธรรมชาติแต่ไม่มากเกิน
- ✅ ลงโทษการเคลื่อนไหวแขนที่ผิดปกติ

### 5. Conservative Walking Parameters

```python
# Adjusted target speed and rewards
target_forward_velocity: float = 0.7   # Reduced from 1.0 m/s
forward_reward_scale: float = 1.2      # Reduced from 1.5
torso_stability_reward: float = 0.6    # Increased from 0.4
```

**ผลลัพธ์**:
- ✅ ความเร็วเป้าหมายลดลงเพื่อความมั่นคง
- ✅ เน้นการทรงตัวมากกว่าความเร็ว

### 6. Improved Initial Pose

```python
# Natural walking-ready stance
joint_pos={
    "left_hip_pitch_joint": -0.02,      # Minimal forward lean
    "right_hip_pitch_joint": 0.02,      # Minimal opposite
    "left_knee_joint": 0.05,            # Slight knee bend
    "right_knee_joint": 0.03,           # Less bend
    "left_elbow_1_joint": -0.3,         # Natural arm bend
    "right_elbow_1_joint": -0.4,        # Opposite arm
}
```

**ผลลัพธ์**:
- ✅ เริ่มต้นในท่าที่พร้อมเดิน
- ✅ แขนอยู่ในท่าธรรมชาติ

### 7. Enhanced Learning Parameters

```python
# More stable learning configuration
init_noise_std=0.2,           # Lower exploration noise
clip_param=0.15,              # Smaller policy updates
learning_rate=1.5e-4,         # Lower learning rate
num_steps_per_env=48,         # Longer episodes
```

**ผลลัพธ์**:
- ✅ การเรียนรู้เสถียรมากขึ้น
- ✅ ลดความผันผวนในการฝึกฝน

## 🎯 Expected Results

### Before (ปัญหา)
- ❌ ลำตัวเอียงหลัง
- ❌ ก้าวยาวผิดปกติ  
- ❌ แขนเคลื่อนไหวแปลก
- ❌ การเดินไม่เสถียร

### After (หลังแก้ไข)
- ✅ ลำตัวตั้งตรง/เอียงหน้าเล็กน้อย
- ✅ ก้าวขาปกติ (ไม่เกิน 17° hip, 46° knee)
- ✅ แขนแกว่งประสานกับขา
- ✅ การเดินเสถียรและธรรมชาติ

## 🚀 Next Steps

1. **Test Environment**: ทดสอบ environment ที่แก้ไขแล้ว
2. **Monitor Training**: ติดตามการเทรนว่า rewards ปรับปรุงไหว
3. **Compare Models**: เปรียบเทียบกับ model เดิม
4. **Fine-tune**: ปรับแต่ง weights ตามผลลัพธ์

## 🛠️ Commands to Use

```bash
# Train improved walking model
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Humanoid-SNC-R5-Walking-Direct-v0 \
  --headless --num_envs 512

# Compare with standing model
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Humanoid-SNC-R5-Direct-v0 \
  --load_run 2025-09-22_13-18-43

# Monitor progress
python walking_training_monitor.py
```

---

## 📊 Summary

การปรับปรุงครั้งนี้เน้นการแก้ปัญหาเฉพาะ:

1. **Posture Control** - ป้องกันการเอียงหลัง
2. **Step Limitation** - จำกัดขนาดก้าว  
3. **Natural Motion** - ส่งเสริมการเคลื่อนไหวที่เป็นธรรมชาติ
4. **Stable Learning** - พารามิเตอร์ที่เรียนรู้ได้เสถียร

**Expected Outcome**: SNC R5 ที่เดินได้เป็นธรรมชาติ มีท่าทางดี และการเคลื่อนไหวที่สมเหตุสมผล! 🚶‍♂️