#!/usr/bin/env python3
"""2️⃣ ทดสอบ physics กับ random actions"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test Humanoid SNC Physics")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

# Robot config
HUMANOID_SNC_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/sncbot/IsaacLab/humanoid_snc_assets/humanoid_snc.usd",
        copy_from_source=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=300.0,
            velocity_limit=10.0,
            stiffness=80.0,
            damping=4.0,
        ),
    },
)

@configclass
class TestEnvCfg(DirectRLEnvCfg):
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 21
    observation_space = 75
    state_space = 0
    
    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average", 
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=4.0,
        replicate_physics=True
    )
    robot: ArticulationCfg = HUMANOID_SNC_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # Required for locomotion env
    joint_gears: list = [50.0] * 21
    heading_weight: float = 0.5
    up_weight: float = 0.1
    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    joints_at_limit_cost_scale: float = 0.1
    death_cost: float = -1.0
    termination_height: float = 0.31
    alive_reward_scale: float = 0.5
    dof_vel_scale: float = 0.1
    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01

class TestEnv(LocomotionEnv):
    cfg: TestEnvCfg
    def __init__(self, cfg: TestEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

def main():
    env_cfg = TestEnvCfg()
    env = TestEnv(cfg=env_cfg)
    
    print(f"🤖 Environment created: {env.num_envs} robots")
    print(f"🎮 Action space: {env.action_space}")
    print(f"👁️  Observation space: {env.observation_space}")
    
    count = 0
    try:
        while simulation_app.is_running() and count < 1000:
            with torch.inference_mode():
                actions = torch.rand((env.num_envs, env.action_space.shape[-1]), device=env.device) * 0.1
                obs, rew, terminated, truncated, info = env.step(actions)
                
                if count % 100 == 0:
                    print(f"⏱️  Step: {count}, Mean reward: {rew.mean():.3f}")
                    
                if count % 500 == 0:
                    env.reset()
                    print("🔄 Reset environments")
                
                count += 1
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()