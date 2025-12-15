# Enhanced Lateral Stability Fixes

## 🚨 ปัญหา: ยังคงล้มไปด้านขวาตลอด

แม้ว่าจะทำ symmetric joint initialization แล้ว robot ยังคงล้มไปด้านขวา

## ✅ การแก้ไขเพิ่มเติม: Enhanced Lateral Stability

### 1. SUPER STRONG Lateral Velocity Penalty

```python
# Before: 2x penalty
lateral_penalty = torch.abs(base_lin_vel[:, 1]) * 2.0

# After: 5x penalty  
lateral_penalty = torch.abs(base_lin_vel[:, 1]) * 5.0  # PREVENT ANY sideways drift
```

### 2. NEW: Sideways Lean Penalty

```python
# NEW: Strong penalty for leaning to either side
sideways_lean_penalty = torch.abs(roll) * 8.0  # 8x penalty for side lean

# Enhanced roll penalty in upright reward
torso_upright_reward = torch.exp(-torch.abs(pitch) * 3.0) * torch.exp(-torch.abs(roll) * 5.0)
```

### 3. Enhanced Hip Roll Stiffness for Lateral Stability

```python
# Before: Reduced for flexibility
".*_hip_roll.*": 15.0,   # Too soft for lateral stability
".*_hip_roll.*": 4.0,    # Too low damping

# After: INCREASED for lateral stability
".*_hip_roll.*": 25.0,   # INCREASED back to 25.0 for lateral stability  
".*_hip_roll.*": 8.0,    # INCREASED back to 8.0 for lateral stability
```

**Rationale**: Hip roll joints ควบคุม lateral stability โดยตรง - ต้องแข็งพอที่จะป้องกันการล้มไปข้าง

### 4. Enhanced Reward Weight Balance

```python
total_reward = (
    # INCREASED torso stability
    0.6 * torso_upright_reward +  # INCREASED from 0.5
    
    # NEW: Strong sideways penalties
    0.8 * sideways_lean_penalty -      # NEW: 8x roll penalty
    1.0 * lateral_penalty -            # DOUBLED weight (from 0.5)
    
    # Maintain flexibility for forward movement
    0.3 * leg_smoothness_reward +      # Keep leg flexibility
    # ... other rewards
)
```

## 🎯 Strategy: Selective Stiffness

### Forward Movement: FLEXIBLE
- Hip pitch: 18.0 (reduced from 30.0)
- Knee: 15.0 (reduced from 25.0)  
- Ankle pitch: 6.0 (reduced from 10.0)

### Lateral Stability: STIFF
- Hip roll: 25.0 (INCREASED back from 15.0)
- Hip roll damping: 8.0 (INCREASED back from 4.0)
- Ankle roll: 5.0 (moderate)

## 📊 Expected Results

### Physics Benefits:
- ✅ **Lateral Stability**: Hip roll stiffness ป้องกันการล้มข้าง
- ✅ **Forward Flexibility**: Hip pitch ยังคงยืดหยุ่นสำหรับการเดิน
- ✅ **Balance Recovery**: แรงกลับคืนเมื่อเริ่มเอียงข้าง

### Reward Benefits:
- ✅ **5x lateral velocity penalty**: บังคับให้เดินตรง
- ✅ **8x sideways lean penalty**: ป้องกันการเอียงข้าง
- ✅ **Enhanced roll detection**: ตรวจจับการเอียงข้างได้ดีขึ้น

## 🚀 Testing Status

```bash
# Currently training with enhanced settings
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Humanoid-SNC-R5-Walking-Direct-v0 \
  --headless --num_envs 512

# Training progress: Learning iteration 37/2500
# Reward improving: -152.99 (was -1135.42 at start)
```

## 🔍 Key Innovation: **Selective Joint Stiffness**

แทนที่จะลด stiffness ทั้งหมด เราใช้:

- **Forward joints (pitch)**: SOFT สำหรับความยืดหยุ่น
- **Lateral joints (roll)**: STIFF สำหรับความมั่นคง

**Result Expected**: SNC R5 ที่เดินได้ตรง ไม่ล้มข้าง แต่ยังคงยืดหยุ่นในการเดินไปข้างหน้า! 🚶‍♂️⚖️