# 04_configs - Configuration Files

## 📋 ไฟล์ในโฟลเดอร์นี้

### 📄 humanoid_snc_locomotion.py
- Environment configuration class
- ใช้สำหรับสร้าง RL environment
- มี reward parameters ครบ
- รองรับ RSL-RL และ other frameworks

## 🔧 การใช้งาน
ไฟล์นี้เป็น library สำหรับไฟล์อื่น import:
```python
from humanoid_snc_scripts.04_configs.humanoid_snc_locomotion import HumanoidSNCEnvCfg, HumanoidSNCEnv
```

## 📊 ข้อมูล Configuration
- **Action space**: 21 joints
- **Observation space**: 75 dimensions  
- **Episode length**: 15 seconds
- **Physics timestep**: 1/120 sec
- **Control frequency**: 1/60 sec (decimation=2)