# คู่มือการสร้างและเทรนโมเดลหุ่นยนต์ใน Isaac Lab

## สารบัญ
1. [โครงสร้างไฟล์ที่จำเป็น](#โครงสร้างไฟล์ที่จำเป็น)
2. [ไฟล์หลักที่ต้องสร้าง](#ไฟล์หลักที่ต้องสร้าง)
3. [รายละเอียดแต่ละไฟล์](#รายละเอียดแต่ละไฟล์)
4. [วิธีการเทรนและเล่นโมเดล](#วิธีการเทรนและเล่นโมเดล)

---

## โครงสร้างไฟล์ที่จำเป็น

เมื่อต้องการสร้างหุ่นยนต์ใหม่ (เช่น Humanoid SNC R5) คุณต้องสร้างไฟล์ในตำแหน่งดังนี้:

```
IsaacLab/
├── mycobot_description/urdf/SNC_R5/
│   └── Humanoid_SNC_R5_fixed_knees_relative.urdf    # ไฟล์ URDF ของหุ่นยนต์
│
├── source/isaaclab_assets/isaaclab_assets/robots/
│   ├── __init__.py                                   # อัพเดทเพื่อ export robot config
│   └── humanoid_snc_r5.py                           # ⭐ [1] Robot Asset Configuration
│
└── source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/
    └── humanoid_snc_r5/                             # โฟลเดอร์สำหรับ robot ของคุณ
        ├── __init__.py                              # ⭐ [2] Environment Registration
        ├── flat_env_cfg.py                          # ⭐ [3] Flat Environment Config
        ├── rough_env_cfg.py                         # ⭐ [4] Rough Environment Config
        └── agents/
            ├── __init__.py                          # ⭐ [5] Agent Package Init
            └── rsl_rl_ppo_cfg.py                    # ⭐ [6] RL Training Config (PPO)
```

---

## ไฟล์หลักที่ต้องสร้าง

### ไฟล์บังคับ (ต้องมี):
1. **Robot Asset Configuration** - กำหนดค่าหุ่นยนต์ (URDF, joints, actuators)
2. **Environment Registration** - ลงทะเบียน environment กับ Gymnasium
3. **Environment Configuration** - กำหนดค่า environment (rewards, observations, actions)
4. **RL Agent Configuration** - กำหนดค่า hyperparameters สำหรับการเทรน

### ไฟล์เสริม (ไม่บังคับ):
- Custom action wrappers (เช่น การ invert hip axis)
- Custom reward functions
- YAML configs สำหรับ RL frameworks อื่นๆ (RL Games, SKRL, SB3)

---

## รายละเอียดแต่ละไฟล์

### ⭐ [1] Robot Asset Configuration
**ที่อยู่:** `source/isaaclab_assets/isaaclab_assets/robots/humanoid_snc_r5.py`

**จุดประสงค์:** กำหนดค่าการสร้างหุ่นยนต์ใน simulation (physics, joints, actuators)

**โครงสร้าง:**
```python
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

HUMANOID_SNC_R5_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path="/path/to/your/robot.urdf",
        activate_contact_sensors=True,
        fix_base=False,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            drive_type="force",        # "force" หรือ "acceleration"
            target_type="position",    # "position" หรือ "velocity"
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness=200.0,       # ค่า P gain
                damping=10.0,          # ค่า D gain
            ),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),         # ตำแหน่งเริ่มต้น (x, y, z)
        joint_pos={
            "left_hip_pitch_joint": -0.10,
            "right_hip_pitch_joint": -0.10,
            ".*_knee_joint": 0.15,     # ใช้ regex ได้
            # ... กำหนดมุมเริ่มต้นของทุกข้อต่อ
        },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_knee_joint"],
            effort_limit=300,          # แรงบิดสูงสุด (N·m)
            velocity_limit=100.0,      # ความเร็วสูงสุด (rad/s)
            stiffness=200.0,           # ค่าความแข็ง
            damping=7.0,               # ค่า damping
        ),
        # ... actuators อื่นๆ (feet, arms)
    },
)
```

**Config ที่สำคัญ:**
- `asset_path`: path ไปยังไฟล์ URDF
- `init_state.pos`: ความสูงของ base (ปรับให้หุ่นยนต์ยืนพอดี)
- `init_state.joint_pos`: มุมเริ่มต้นของข้อต่อ (ควรเป็นท่ายืน)
- `actuators`: แบ่งกลุ่มข้อต่อ (legs, arms, feet) และกำหนด stiffness/damping

**อย่าลืม:** Update `source/isaaclab_assets/isaaclab_assets/robots/__init__.py`:
```python
from .humanoid_snc_r5 import HUMANOID_SNC_R5_CFG

__all__ = ["HUMANOID_SNC_R5_CFG", ...]
```

---

### ⭐ [2] Environment Registration
**ที่อยู่:** `source/isaaclab_tasks/.../config/humanoid_snc_r5/__init__.py`

**จุดประสงค์:** ลงทะเบียน environment IDs กับ Gymnasium

**โครงสร้าง:**
```python
import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-Velocity-Rough-Humanoid-SNC-R5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:HumanoidSNCR5RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidSNCR5PPORunnerCfg",
    },
)

# ลงทะเบียน environments อื่นๆ (Flat, Play versions)
```

**Environment IDs ที่ควรมี:**
- `Isaac-Velocity-Flat-{RobotName}-v0` - พื้นเรียบ (training)
- `Isaac-Velocity-Flat-{RobotName}-Play-v0` - พื้นเรียบ (play/test)
- `Isaac-Velocity-Rough-{RobotName}-v0` - พื้นขรุขระ (training)
- `Isaac-Velocity-Rough-{RobotName}-Play-v0` - พื้นขรุขระ (play/test)

---

### ⭐ [3] & [4] Environment Configuration
**ที่อยู่:**
- `rough_env_cfg.py` - สำหรับพื้นขรุขระ
- `flat_env_cfg.py` - สำหรับพื้นเรียบ

**จุดประสงค์:** กำหนดค่า observations, actions, rewards, terminations, events

**โครงสร้าง:**
```python
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg
)
from isaaclab_assets.robots.humanoid_snc_r5 import HUMANOID_SNC_R5_CFG

@configclass
class CustomRewards(RewardsCfg):
    """กำหนด reward terms"""

    # Reward สำหรับติดตามความเร็ว
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,                        # น้ำหนักของ reward
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # Penalty สำหรับการตก
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-200.0
    )

    # Reward สำหรับการยกเท้า
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=15.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_.*"),
            "command_name": "base_velocity",
            "threshold": 0.5,              # เวลาขั้นต่ำที่เท้าต้องลอย (วินาที)
        },
    )

@configclass
class CustomEnvCfg(LocomotionVelocityRoughEnvCfg):
    """กำหนดค่า environment"""

    def __post_init__(self):
        super().__post_init__()

        # 1. ใช้ robot config ของคุณ
        self.scene.robot = HUMANOID_SNC_R5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 2. กำหนด action scale
        self.actions.joint_pos.scale = 0.5

        # 3. เลือกข้อต่อที่ RL agent จะควบคุม (optional)
        self.actions.joint_pos.joint_names = [
            ".*_hip_.*",
            ".*_knee_joint",
            ".*_ankle_.*",
            # ไม่รวมแขน = แขนจะล็อคอยู่ที่ตำแหน่งเริ่มต้น
        ]

        # 4. ใช้ reward config ของคุณ
        self.rewards = CustomRewards()

        # 5. กำหนด velocity command ranges
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)

        # 6. กำหนด termination conditions
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["base_link"]

        # 7. ปรับแต่ง terrain (optional)
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)

@configclass
class CustomEnvCfg_PLAY(CustomEnvCfg):
    """Config สำหรับ play mode (test policy)"""

    def __post_init__(self):
        super().__post_init__()

        # ลด environment ให้น้อยลง
        self.scene.num_envs = 50

        # ปิด randomization
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
```

**Config ที่สำคัญ:**

#### Rewards (ปรับให้เหมาะกับ task):
- `track_lin_vel_xy_exp` - ติดตามความเร็วเดินหน้า/ข้าง
- `track_ang_vel_z_exp` - ติดตามความเร็วหมุน
- `feet_air_time` - ให้รางวัลเมื่อยกเท้า (สำคัญสำหรับการเดิน!)
- `feet_slide` - ลงโทษเมื่อเท้าลาก
- `joint_deviation_l1` - ลงโทษเมื่อข้อต่อห่างจากตำแหน่ง default
- `termination_penalty` - ลงโทษเมื่อหุ่นยนต์ล้ม

#### Actions:
- `scale` - ขนาดของ action (0.5 = ±50% จาก default position)
- `joint_names` - ข้อต่อที่ให้ RL ควบคุม

#### Commands:
- `lin_vel_x` - ช่วงความเร็วเดินหน้า-ถอยหลัง (m/s)
- `lin_vel_y` - ช่วงความเร็วเดินข้าง (m/s)
- `ang_vel_z` - ช่วงความเร็วหมุน (rad/s)

---

### ⭐ [5] Agent Package Init
**ที่อยู่:** `agents/__init__.py`

**จุดประสงค์:** Export agent configs

```python
from .rsl_rl_ppo_cfg import HumanoidSNCR5PPORunnerCfg

__all__ = ["HumanoidSNCR5PPORunnerCfg"]
```

---

### ⭐ [6] RL Training Configuration
**ที่อยู่:** `agents/rsl_rl_ppo_cfg.py`

**จุดประสงค์:** กำหนด hyperparameters สำหรับ PPO algorithm

**โครงสร้าง:**
```python
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg
)

@configclass
class HumanoidSNCR5PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Training parameters
    num_steps_per_env = 24           # จำนวน steps ก่อน update policy
    max_iterations = 1500            # จำนวน iterations ทั้งหมด
    save_interval = 50               # บันทึกโมเดลทุกๆ N iterations
    experiment_name = "humanoid_snc_r5_velocity"
    empirical_normalization = False  # ใช้ running mean/std หรือไม่

    # Neural network architecture
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,          # noise เริ่มต้นสำหรับ exploration
        actor_hidden_dims=[512, 256, 128],   # ขนาด hidden layers (actor)
        critic_hidden_dims=[512, 256, 128],  # ขนาด hidden layers (critic)
        activation="elu",            # activation function
    )

    # PPO algorithm parameters
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,         # น้ำหนักของ value loss
        use_clipped_value_loss=True,
        clip_param=0.2,              # PPO clipping parameter
        entropy_coef=0.01,           # น้ำหนักของ entropy bonus
        num_learning_epochs=5,       # epochs ต่อ batch
        num_mini_batches=4,          # จำนวน mini-batches
        learning_rate=1.0e-3,        # learning rate
        schedule="adaptive",         # "adaptive" หรือ "fixed"
        gamma=0.99,                  # discount factor
        lam=0.95,                    # GAE lambda
        desired_kl=0.01,             # target KL divergence
        max_grad_norm=1.0,           # gradient clipping
    )
```

**Hyperparameters ที่สำคัญ:**
- `num_steps_per_env`: 24-48 (มากขึ้น = เก็บ experience มากขึ้น)
- `learning_rate`: 1e-3 ถึง 1e-4 (ค่าเริ่มต้นที่ดี)
- `actor_hidden_dims`: [512, 256, 128] สำหรับ humanoid ขนาดกลาง
- `clip_param`: 0.2 (standard PPO value)
- `gamma`: 0.99 (discount factor)

---

## วิธีการเทรนและเล่นโมเดล

### 1. เทรน (Training)

```bash
# เทรนบนพื้นเรียบ
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Humanoid-SNC-R5-v0 \
    --num_envs 4096 \
    --headless

# เทรนบนพื้นขรุขระ
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Humanoid-SNC-R5-v0 \
    --num_envs 4096 \
    --headless

# เทรนแบบมี GUI (ช้ากว่า)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Humanoid-SNC-R5-v0 \
    --num_envs 512

# เทรนต่อจาก checkpoint
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Humanoid-SNC-R5-v0 \
    --resume \
    --load_run logs/rsl_rl/humanoid_snc_r5_velocity/2025-01-15_10-30-45
```

**Parameters:**
- `--task`: ชื่อ environment (ตาม gym.register ID)
- `--num_envs`: จำนวน parallel environments (เยอะ = เร็ว แต่กิน RAM/VRAM)
- `--headless`: ไม่แสดง GUI (เร็วกว่า)
- `--resume`: เทรนต่อจาก checkpoint
- `--load_run`: path ไปยัง run ที่ต้องการ resume

**ผลลัพธ์:**
- Model checkpoints: `logs/rsl_rl/humanoid_snc_r5_velocity/YYYY-MM-DD_HH-MM-SS/`
- TensorBoard logs: ดูได้ด้วย `tensorboard --logdir logs/rsl_rl`

---

### 2. เล่น/ทดสอบโมเดล (Play/Inference)

```bash
# เล่นโมเดลที่เทรนแล้ว
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Rough-Humanoid-SNC-R5-Play-v0 \
    --num_envs 50 \
    --load_run logs/rsl_rl/humanoid_snc_r5_velocity/2025-01-15_10-30-45

# เล่นโมเดล checkpoint ที่ระบุ
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Rough-Humanoid-SNC-R5-Play-v0 \
    --num_envs 50 \
    --load_run logs/rsl_rl/humanoid_snc_r5_velocity/2025-01-15_10-30-45 \
    --checkpoint model_500.pt

# บันทึกวิดีโอ
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Rough-Humanoid-SNC-R5-Play-v0 \
    --num_envs 1 \
    --load_run logs/rsl_rl/humanoid_snc_r5_velocity/2025-01-15_10-30-45 \
    --video \
    --video_length 500
```

---

## สรุป Checklist

### ✅ ไฟล์ที่ต้องสร้าง:
- [ ] ไฟล์ URDF ของหุ่นยนต์
- [ ] `robots/humanoid_xxx.py` - Robot asset config
- [ ] `robots/__init__.py` - Export robot config
- [ ] `config/humanoid_xxx/__init__.py` - Environment registration
- [ ] `config/humanoid_xxx/flat_env_cfg.py` - Flat environment config
- [ ] `config/humanoid_xxx/rough_env_cfg.py` - Rough environment config
- [ ] `config/humanoid_xxx/agents/__init__.py` - Agent package init
- [ ] `config/humanoid_xxx/agents/rsl_rl_ppo_cfg.py` - PPO training config

### ✅ Config ที่ต้องตั้งค่า:
- [ ] **Robot**: URDF path, initial pose, joint limits, actuators
- [ ] **Rewards**: weight และ parameters ของแต่ละ reward term
- [ ] **Actions**: scale, joint selection
- [ ] **Commands**: velocity ranges
- [ ] **Training**: hyperparameters (learning rate, network size, etc.)

### ✅ คำสั่งที่ใช้บ่อย:
```bash
# เทรน
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task [TASK_NAME] --num_envs 4096 --headless

# เล่น
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task [TASK_NAME]-Play-v0 --load_run [PATH]

# ดู logs
tensorboard --logdir logs/rsl_rl

# List environments
./isaaclab.sh -p scripts/environments/list_envs.py
```

---

## เคล็ดลับ (Tips)

### 1. การเริ่มต้น (Getting Started):
- เริ่มจาก flat environment ก่อน (ง่ายกว่า rough)
- ใช้ velocity ranges ต่ำๆ ในตอนแรก (0.2-0.5 m/s)
- ให้ reward สูงสำหรับ `feet_air_time` (บังคับให้ยกเท้า)

### 2. การ Debug:
- ถ้าหุ่นยนต์ล้มทันที → ตรวจสอบ initial pose และ base height
- ถ้าหุ่นยนต์ไม่เดิน → เพิ่ม `feet_air_time` reward
- ถ้าหุ่นยนต์สั่น → ลด action scale หรือเพิ่ม damping

### 3. การปรับแต่ง:
- เพิ่ม `num_envs` → เทรนเร็วขึ้น (แต่กิน VRAM)
- เพิ่ม `actor_hidden_dims` → policy ซับซ้อนขึ้น
- ปรับ reward weights → เปลี่ยนพฤติกรรม

### 4. ข้อควรระวัง:
- **อย่า** ตั้ง velocity ranges สูงเกินไปตั้งแต่ต้น
- **อย่า** ใช้ reward weights สูงเกินไป (ควรอยู่ในช่วง 0.01-10.0)
- **ต้อง** มี termination_penalty เสมอ (-200 เป็นค่ามาตรฐาน)

---

## อ้างอิง

- Isaac Lab Documentation: https://isaac-sim.github.io/IsaacLab
- ตัวอย่างอื่นๆ: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/`
  - `h1/` - Humanoid H1 (ตัวอย่างที่ดี)
  - `g1/` - Unitree G1 (ตัวอย่างที่ดี)
  - `anymal_d/` - Quadruped (สี่ขา)

**หมายเหตุ:** เอกสารนี้สร้างขึ้นจากโครงสร้างของ Humanoid SNC R5 ที่มีอยู่ในโค้ดเบสนี้
