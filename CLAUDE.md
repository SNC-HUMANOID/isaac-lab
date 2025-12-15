# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Isaac Lab is a GPU-accelerated robotics framework built on NVIDIA Isaac Sim for reinforcement learning, imitation learning, and motion planning. The framework is structured as multiple Python extensions with a modular architecture.

## Key Commands

### Development Commands
- **Install extensions**: `./isaaclab.sh -i` or `./isaaclab.sh -i [framework]` (framework options: rsl_rl, rl_games, sb3, skrl, robomimic, all, none)
- **Format code**: `./isaaclab.sh -f` (runs pre-commit formatting and linting)
- **Run tests**: `./isaaclab.sh -t` (runs pytest on tools directory)
- **Build documentation**: `./isaaclab.sh -d` (builds Sphinx docs, or use `cd docs && make current-docs` for current version only)
- **Run Python with Isaac Sim**: `./isaaclab.sh -p [script.py]`
- **Run Isaac Sim**: `./isaaclab.sh -s`
- **Setup conda environment**: `./isaaclab.sh -c [env_name]` (default: env_isaaclab)
- **Create new project/task**: `./isaaclab.sh -n` (generates project template from interactive prompts)
- **Generate VSCode settings**: `./isaaclab.sh -v` (creates VSCode configuration from template)
- **Docker helper**: `./isaaclab.sh -o` (runs docker/container.sh for container management)

### Training Commands
Examples for training RL environments:
```bash
# RSL-RL training
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Anymal-C-v0

# Play trained policy
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-Anymal-C-v0 --load_run logs/rsl_rl/anymal_c_flat/YYYY-MM-DD_HH-MM-SS/

# Other frameworks (RL Games, SB3, SKRL)
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task Isaac-Velocity-Flat-Anymal-C-v0
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Cartpole-v0
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Cartpole-v0
```

### Testing and Quality
- **Code formatting**: `./isaaclab.sh -f` (runs black, flake8, isort via pre-commit hooks)
- **Type checking**: Uses pyright with configuration in `pyproject.toml`
- **Testing**: `pytest` for unit tests in each extension's `test/` directory
- **Run single test**: `./isaaclab.sh -p -m pytest path/to/test_file.py::test_function`
- **Run tests for specific extension**: `./isaaclab.sh -p -m pytest source/[extension_name]/test/`
- **Linting**: Pre-commit hooks include flake8 with additional plugins (flake8-simplify, flake8-return)
- **License header insertion**: Automatically handled by pre-commit (BSD-3 for most files, Apache 2.0 for mimic extension)

## Architecture

### Extension Structure
The repository contains 5 main Python extensions in `source/`:

1. **isaaclab**: Core framework with simulation, assets, environments, sensors (BSD-3 license)
2. **isaaclab_assets**: Pre-configured robot and sensor assets (BSD-3 license)
3. **isaaclab_rl**: RL framework wrappers (RSL-RL, RL Games, SB3, SKRL) (BSD-3 license)
4. **isaaclab_mimic**: Imitation learning tools and environments (Apache 2.0 license)
5. **isaaclab_tasks**: Pre-built RL/IL environments (BSD-3 license)

Note: The `isaaclab_mimic` extension and its standalone scripts use Apache 2.0 license, while all other code uses BSD-3.

Each extension follows the same structure:
- `config/extension.toml`: Extension metadata and configuration
- `pyproject.toml`: Build configuration and development tools
- `setup.py`: Installation script
- `test/`: Unit tests and integration tests
- Main module directory with `__init__.py` for registration

### Key Directories
- `scripts/`: Example scripts for demos, training, tools
- `logs/`: Training outputs and model checkpoints
- `outputs/`: Hydra configuration outputs
- `docs/`: Sphinx documentation source
- `docker/`: Docker deployment configurations

### Environment Types
- **Direct environments**: Inherit from `DirectRLEnv`, lower-level control
- **Manager-based environments**: Use composition pattern with managers for actions, observations, rewards, etc.
- **MARL environments**: Multi-agent reinforcement learning support

### Configuration System
- Uses `@configclass` decorator for type-safe configurations
- Environment configs typically end with `_cfg.py`
- Agent configurations in `agents/` subdirectories
- Supports Hydra for configuration management
- Configuration validation ensures all required fields are present

### Core Framework Components (isaaclab/)
- **sim/**: Isaac Sim integration, USD/URDF converters, simulation context
- **assets/**: Articulation, rigid objects, deformable objects with data classes
- **envs/**: Environment base classes (Direct RL/MARL, Manager-based)
- **managers/**: Action, observation, reward, termination, event managers
- **sensors/**: Camera, contact, IMU, ray-caster sensors with data classes
- **terrains/**: Height field and mesh terrain generation
- **actuators/**: DC motor, implicit, neural network actuator models
- **controllers/**: Differential IK, operational space, impedance controllers

### Asset and Scene Management
- **Interactive Scene**: Central scene management with physics simulation
- **Assets**: Articulations, rigid objects, deformable objects, sensors
- **Spawners**: Programmatic creation of simulation primitives
- **Terrain Generation**: Procedural terrain creation tools

## Development Patterns

### Adding New Environments
1. Create environment class inheriting from appropriate base (DirectRLEnv or ManagerBasedRLEnv)
2. Create corresponding configuration class with `@configclass` decorator
3. Register environment in `__init__.py` using `gym.register()`
4. Add agent configurations in `agents/` subdirectory
5. Follow existing naming conventions: `Isaac-[Task]-[Robot]-[Terrain]-v0`

### Robot Asset Integration
1. **URDF/USD files**: Place in appropriate directories (often in external asset paths)
2. **Robot configurations**: Create in `isaaclab_assets/robots/` with ArticulationCfg
3. **Asset spawning**: Use `UsdFileCfg` or `UrdfFileCfg` in spawn configurations
4. **Joint configuration**: Define actuators, initial states, physics properties
5. **Validation**: Test with basic environment before full RL integration

### Environment Registration Pattern
```python
gym.register(
    id="Isaac-Task-Robot-v0", 
    entry_point=f"{__name__}.env_module:EnvClass",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_module:EnvCfg",
        "rl_games_cfg_entry_point": f"{_AGENTS_PKG}:rl_games_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_PKG}.rsl_rl_cfg:RunnerCfg",
        "skrl_cfg_entry_point": f"{_AGENTS_PKG}:skrl_cfg.yaml",
    },
)
```

### Configuration Classes
- Use `@configclass` decorator
- Inherit from appropriate base config classes
- Use type hints for all attributes
- Group related settings in nested config classes

### Testing
- Add tests in `test/` directory of relevant extension
- Use pytest fixtures and parametrization
- Test both functionality and configuration validity

### Asset Integration
- Place robot/sensor assets in `isaaclab_assets`
- Use Isaac Sim USD format or convert from URDF/MJCF
- Configure physics properties, visual materials, etc.

## URDF to USD Conversion Workflow

Converting URDF models to USD format for Isaac Lab:

### Basic Conversion Command
```bash
./isaaclab.sh -p scripts/tools/convert_urdf.py \
    /path/to/robot.urdf \
    /path/to/output.usd \
    --merge-joints \
    --joint-stiffness 100.0 \
    --joint-damping 1.0
```

### Common Conversion Options
- `--merge-joints`: Consolidate links connected by fixed joints (recommended for most robots)
- `--fix-base`: Fix the base link in place (useful for fixed-base manipulators)
- `--joint-stiffness`: Default stiffness for joint drives (default: 100.0)
- `--joint-damping`: Default damping for joint drives (default: 1.0)
- `--joint-target-type`: Control type - "position", "velocity", or "none" (default: "position")

### Pre-Conversion Checklist
Before converting a URDF to USD, ensure:
1. **Joint limits are realistic**: Check that all revolute joints have appropriate lower/upper bounds, effort, and velocity limits
2. **Initial joint positions**: Verify default joint positions in URDF fall within the specified limits
3. **Mass properties**: Ensure links have reasonable mass and inertia values
4. **Collision geometry**: Verify collision meshes exist and are properly defined
5. **Contact sensors**: Consider which links need contact sensing enabled

### Post-Conversion Validation
After conversion, verify the USD file:
1. Check that all joints are present and properly named
2. Verify joint limits are preserved from URDF
3. Test robot configuration with a simple visualization script
4. Validate initial pose doesn't violate joint limits

### Common Conversion Issues
- **Joint limits validation errors**: If you get "joint position out of limits" errors, update either:
  - The joint limits in the URDF before conversion, or
  - The `init_state.joint_pos` in the ArticulationCfg after conversion
- **Fixed joints**: Use `--merge-joints` to avoid unnecessary computation for fixed joints
- **Base link orientation**: Some URDFs may need rotation adjustments in ArticulationCfg spawn configuration

## Custom Robot Integration Pattern

Complete workflow for integrating a new robot (e.g., custom humanoid):

### Step 1: Prepare Robot Assets
```bash
# Directory structure for custom robot
/path/to/robot/
├── urdf/
│   └── robot.urdf           # Original URDF with joint limits
└── usd/
    └── robot.usd            # Will be created by conversion
```

### Step 2: Convert URDF to USD
```bash
./isaaclab.sh -p scripts/tools/convert_urdf.py \
    /path/to/robot/urdf/robot.urdf \
    /path/to/robot/usd/robot.usd \
    --merge-joints
```

### Step 3: Create Robot Configuration
Create file in `source/isaaclab_assets/isaaclab_assets/robots/robot_name.py`:

```python
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

ROBOT_NAME_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/path/to/robot/usd/robot.usd",
        activate_contact_sensors=True,  # Enable contact sensing
        rigid_props=sim_utils.RigidBodyPropertiesCfg(...),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(...),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),  # Spawn height
        joint_pos={
            # Use regex patterns for symmetric joints
            ".*_joint_name": 0.0,
            "left_specific_joint": 0.1,
            "right_specific_joint": -0.1,
        },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip.*", ".*_knee.*", ".*_ankle.*"],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={".*_hip.*": 150.0, ".*_knee.*": 200.0},
            damping={".*_hip.*": 5.0, ".*_knee.*": 5.0},
        ),
        # Define other actuator groups (arms, torso, etc.)
    },
)
```

### Step 4: Register in Assets Package
Add to `source/isaaclab_assets/isaaclab_assets/robots/__init__.py`:
```python
from .robot_name import ROBOT_NAME_CFG

__all__ = ["ROBOT_NAME_CFG", ...]
```

### Step 5: Create Environment Configuration
Create `source/isaaclab_tasks/isaaclab_tasks/direct/robot_task/robot_task_env.py`:

```python
import torch
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab_assets.robots import ROBOT_NAME_CFG

@configclass
class RobotTaskEnvCfg(DirectRLEnvCfg):
    # Simulation settings
    episode_length_s = 15.0
    decimation = 2
    num_actions = 21  # Match your robot's DOF
    num_observations = 75

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0)

    # Robot
    robot: ArticulationCfg = ROBOT_NAME_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Rewards, terminations, etc.
    # ...

class RobotTaskEnv(DirectRLEnv):
    cfg: RobotTaskEnvCfg

    def __init__(self, cfg: RobotTaskEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Custom initialization
```

### Step 6: Create Agent Configuration
Create `source/isaaclab_tasks/isaaclab_tasks/direct/robot_task/agents/rsl_rl_ppo_cfg.py`:

```python
from isaaclab_rl.rsl_rl.ppo import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class RunnerCfg:
    seed = 42
    run_name = "robot_task"
    max_iterations = 2000

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

### Step 7: Register Environment
Add to `source/isaaclab_tasks/isaaclab_tasks/direct/robot_task/__init__.py`:

```python
import gymnasium as gym

gym.register(
    id="Isaac-Robot-Task-v0",
    entry_point=f"{__name__}.robot_task_env:RobotTaskEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robot_task_env:RobotTaskEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS_PKG}.rsl_rl_ppo_cfg:RunnerCfg",
    },
)
```

### Step 8: Test and Train
```bash
# Quick test
./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Robot-Task-v0 --num_envs 64

# Train
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Robot-Task-v0 --num_envs 4096

# Play trained policy
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Robot-Task-v0 --num_envs 32 --load_run logs/rsl_rl/robot_task/YYYY-MM-DD_HH-MM-SS
```

## Joint Limits and Validation

### Critical Validation Points
Isaac Lab performs strict joint limit validation at environment initialization. Common errors:

**Error**: "The following joints have default positions out of the limits"

**Solutions**:
1. **Update joint limits in URDF** (before conversion):
   - Ensure all `<limit>` tags have appropriate lower/upper bounds
   - Set realistic effort and velocity limits
   - Example: `<limit lower="-2.23" upper="0.09" effort="45" velocity="30" />`

2. **Update initial positions in ArticulationCfg** (after conversion):
   ```python
   init_state=ArticulationCfg.InitialStateCfg(
       joint_pos={
           "elbow_joint": -0.5,  # Must be within [-2.23, 0.09]
       }
   )
   ```

### Best Practices for Joint Configuration
- **Reference existing robots**: Use joint limits from similar real robots (e.g., H1, G1 humanoids)
- **Test initial pose**: Visualize robot with `scripts/environments/random_agent.py` before training
- **Symmetric joints**: Use regex patterns for left/right symmetric joints
- **Safety margins**: Keep `soft_joint_pos_limit_factor` at 0.9 (default) for training safety

## Important Notes

- Always activate conda environment before development: `conda activate env_isaaclab`
- Use `./isaaclab.sh -f` before committing to ensure code formatting
- The framework requires Isaac Sim 4.5 for the main branch (version compatibility in README)
  - The `feature/isaacsim_5_0` branch requires Isaac Sim 5.0 built from source
- GPU acceleration requires NVIDIA RTX cards for optimal performance
- Multi-GPU training supported via RSL-RL framework
- Python 3.10 is required (as specified in pyproject.toml)
- Line length limit is 120 characters for code formatting
- Training logs are saved to `logs/` directory by framework name (e.g., `logs/rsl_rl/`)
- Hydra configuration outputs are saved to `outputs/` directory

## Debugging and Utilities

### Common Debug Scripts
- **List environments**: `./isaaclab.sh -p scripts/environments/list_envs.py`
- **Random agent**: `./isaaclab.sh -p scripts/environments/random_agent.py --task [env_name]`
- **Check instanceable assets**: `./isaaclab.sh -p scripts/tools/check_instanceable.py`
- **Convert assets**: Convert URDF/MJCF to USD with `scripts/tools/convert_urdf.py` or `scripts/tools/convert_mjcf.py`

### Script Categories
- `scripts/demos/`: Demonstration scripts for various features
- `scripts/environments/`: Environment testing and debugging
- `scripts/reinforcement_learning/`: Training scripts for RSL-RL, RL Games, SB3, SKRL
- `scripts/imitation_learning/`: Imitation learning with Robomimic (isaaclab_mimic extension)
- `scripts/tools/`: Asset conversion and utility scripts
- `scripts/tutorials/`: Step-by-step tutorial scripts
- `scripts/benchmarks/`: Performance benchmarking scripts

### Useful isaaclab.sh Options
- **Headless mode**: Add `--headless` to any script for headless simulation
- **Enable recording**: Add `--record` to capture videos during training/playing
- **Set number of environments**: Use `--num_envs [N]` to specify environment count

## Development Tools and Quality

### Code Quality Configuration
- **Python version**: 3.10 (specified in pyproject.toml)
- **Line length**: 120 characters
- **Type checking**: Pyright with basic type checking mode
- **Import sorting**: isort with custom sections for Isaac Lab modules
- **Spell checking**: codespell with robotics-specific ignore words

### Import Organization (isort)
Imports are organized in specific sections:
1. Future imports
2. Standard library 
3. Third-party packages
4. Isaac Lab assets (`isaaclab_assets`)
5. Core Isaac Lab (`isaaclab`)
6. Extra first-party (`isaaclab_rl`, `isaaclab_mimic`)
7. Tasks (`isaaclab_tasks`)
8. Local folder (`config`)

### Extension Development
- Each extension has its own `pyproject.toml` and `setup.py`
- Extensions are installed in development mode with `./isaaclab.sh -i`
- Test files follow naming pattern `test_*.py`
- Use `@configclass` for all configuration classes
- Inherit from appropriate base classes for consistency