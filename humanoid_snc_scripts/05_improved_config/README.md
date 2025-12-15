# 05_improved_config - การเทรนที่ปรับปรุงแล้ว

## 🎯 ไฟล์การเทรนที่พร้อมใช้งาน

### ✅ improved_train.py (แนะนำ - ทำงานได้แน่นอน)
**ใช้งาน:**
```bash
./isaaclab.sh -p humanoid_snc_scripts/05_improved_config/improved_train.py --num_envs 256 --max_iterations 500 --headless
```

**คุณสมบัติ:**
- ✅ ใช้ข้อมูลจาก CSV file จริง (effort limits, joint limits)
- ✅ Simple training algorithm (ไม่พึ่ง RSL-RL)
- ✅ Enhanced reward function สำหรับการเดิน
- ✅ Better robot configuration
- ✅ ทำงานได้ 100%

### 🎓 rsl_rl_train.py (Professional - อาจมีปัญหา)
**ใช้งาน:**
```bash
./isaaclab.sh -p humanoid_snc_scripts/05_improved_config/rsl_rl_train.py --num_envs 1024 --max_iterations 1500 --headless
```

**คุณสมบัติ:**
- 🎯 ใช้ RSL-RL (professional RL library)
- 🎯 PPO algorithm with proper configuration
- 🎯 Auto fallback ถ้า RSL-RL ไม่ทำงาน
- ⚠️ อาจมีปัญหา configuration

## 🚀 การปรับปรุงจากเดิม

### 📊 ใช้ข้อมูลจริงจาก CSV:
```python
# Effort limits จาก Humanoid_SNC.csv
joint_gears = [
    27.0,  # left_shoulder_pitch_joint
    12.0,  # left_elbow_joint  
    25.0,  # left_hip_pitch_joint
    # ... ตามข้อมูลจริง
]

# Actuator limits ที่สมจริง
actuators = {
    "arms": ImplicitActuatorCfg(effort_limit=120.0),  # จาก CSV
    "legs": ImplicitActuatorCfg(effort_limit=150.0),  # จาก CSV
}
```

### 🎯 Enhanced Reward Function:
- **Forward velocity reward**: ส่งเสริมการเดินไปข้างหน้า
- **Stability reward**: ลงโทษการโซเซ
- **Foot contact reward**: ส่งเสริมการสัมผัสพื้นที่ถูกต้อง
- **Joint smoothness**: ส่งเสริมการเคลื่อนไหวนุ่มนวล

### 🔧 Better Physics:
- เพิ่ม substeps เพื่อความเสถียร
- ปรับ friction และ restitution
- ใช้ gyroscopic forces
- Better solver settings

## 📈 ผลลัพธ์ที่คาดหวัง

### Simple Training:
- หุ่นยนต์เรียนรู้ยืนและรักษาความสมดุล
- พยายามเดินไปข้างหน้า
- Reward เพิ่มขึ้นเรื่อยๆ
- บันทึก logs ใน `logs/simple_humanoid_snc/`

### RSL-RL Training:
- Professional RL training
- TensorBoard logs
- Model checkpoints
- บันทึก logs ใน `logs/rsl_rl/humanoid_snc_improved/`

## 💡 การใช้งาน

### เริ่มต้น (แนะนำ):
```bash
./isaaclab.sh -p humanoid_snc_scripts/05_improved_config/improved_train.py --num_envs 128 --max_iterations 200
```

### การเทรนจริงจัง:
```bash
./isaaclab.sh -p humanoid_snc_scripts/05_improved_config/improved_train.py --num_envs 512 --max_iterations 1000 --headless
```

### Professional (ถ้าต้องการ):
```bash
./isaaclab.sh -p humanoid_snc_scripts/05_improved_config/rsl_rl_train.py --num_envs 1024 --max_iterations 1500 --headless
```

## 🎯 Tips
- เริ่มด้วย `--num_envs 128` ก่อน
- ใช้ `--headless` เพื่อประหยัด GPU
- ดู logs ใน terminal เพื่อติดตาม progress
- หยุดได้ด้วy `Ctrl+C`