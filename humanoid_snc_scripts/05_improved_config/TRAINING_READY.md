# 🚀 Humanoid SNC Training Scripts - Ready to Use!

## ✅ Working Training Scripts

### 1. **working_train.py** (Main Script - Recommended)
```bash
./isaaclab.sh -p humanoid_snc_scripts/05_improved_config/working_train.py --num_envs 32 --max_iterations 500 --headless
```

**Features:**
- ✅ Proper Isaac Lab DirectRLEnv structure
- ✅ Real Humanoid SNC robot from your USD file
- ✅ Simple but effective training algorithm
- ✅ Upright balance and stability rewards
- ✅ Progress tracking and logging
- ✅ **WORKING AND TESTED**

### 2. **improved_train.py** (Advanced - With CSV Data)
```bash
./isaaclab.sh -p humanoid_snc_scripts/05_improved_config/improved_train.py --num_envs 256 --max_iterations 1000 --headless
```

**Features:**
- ✅ Uses real robot specifications from CSV file
- ✅ Enhanced reward function for locomotion
- ✅ Proper joint limits and effort constraints
- ✅ Fallback to simple training if needed

### 3. **rsl_rl_train.py** (Professional RL)
```bash
./isaaclab.sh -p humanoid_snc_scripts/05_improved_config/rsl_rl_train.py --num_envs 512 --max_iterations 1500 --headless
```

**Features:**
- 🎯 Professional RSL-RL integration
- 🎯 PPO algorithm with proper configuration
- 🎯 Automatic fallback to simple training
- ⚠️ May require RSL-RL library

## 🎯 Quick Start (Recommended)

### For Testing:
```bash
./isaaclab.sh -p humanoid_snc_scripts/05_improved_config/working_train.py --num_envs 8 --max_iterations 20
```

### For Training:
```bash
./isaaclab.sh -p humanoid_snc_scripts/05_improved_config/working_train.py --num_envs 64 --max_iterations 500 --headless
```

### For Serious Training:
```bash
./isaaclab.sh -p humanoid_snc_scripts/05_improved_config/improved_train.py --num_envs 256 --max_iterations 1000 --headless
```

## 📊 What the Robot Will Learn

1. **Balance**: Stay upright and maintain stability
2. **Joint Control**: Use joints smoothly and efficiently  
3. **Stability**: Minimize excessive movements
4. **Height Maintenance**: Keep target height around 1.0m

## 💡 Training Tips

- Start with `--num_envs 32` to test
- Use `--headless` to save GPU memory
- Monitor terminal output for progress
- Stop anytime with `Ctrl+C`
- Results saved automatically to `logs/` directory

## 🔧 Script Features

### Environment Setup:
- Real Humanoid SNC robot (21 joints)
- Proper physics simulation 
- Ground plane interaction
- Multiple parallel environments

### Reward Function:
- **Height reward**: Encourages staying upright
- **Joint penalty**: Prevents extreme joint angles
- **Stability reward**: Rewards smooth movements
- **Forward progress**: (in improved version) encourages walking

### Training Algorithm:
- Simple policy gradient approach
- Adaptive exploration noise
- Progress tracking and logging
- Automatic best model saving

## 📈 Expected Results

**Early Training (0-100 iterations):**
- Robot learns to balance
- Reduces falling episodes
- Improves joint control

**Mid Training (100-500 iterations):**
- Better stability and posture
- Smoother movements
- Higher average rewards

**Advanced Training (500+ iterations):**
- Consistent upright posture
- Potential walking behavior
- Stable long episodes

## 🚀 Ready to Train!

Your Humanoid SNC robot is ready for reinforcement learning training. All scripts have been tested and work with your robot configuration.

**Start with:**
```bash
./isaaclab.sh -p humanoid_snc_scripts/05_improved_config/working_train.py --num_envs 32 --max_iterations 100 --headless
```