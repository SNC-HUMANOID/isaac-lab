#!/usr/bin/env python3
"""1️⃣ แสดง Humanoid SNC model - ง่ายที่สุด"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Show Humanoid SNC Model")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg

@configclass
class ShowSceneCfg(InteractiveSceneCfg):
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/sncbot/IsaacLab/humanoid_snc_assets/humanoid_snc.usd",
            copy_from_source=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=100.0,
                velocity_limit=5.0,
                stiffness=40.0,
                damping=2.0,
            ),
        },
    )

def main():
    # Ground + Light
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    cfg_light = sim_utils.DomeLightCfg(intensity=2000.0)
    cfg_light.func("/World/Light", cfg_light)
    
    # Simulation
    sim_cfg = SimulationCfg(dt=1/60)
    sim = SimulationContext(sim_cfg)
    scene_cfg = ShowSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    print("🤖 Humanoid SNC Model Loaded!")
    robot: Articulation = scene["robot"]
    print(f"📊 Joints: {robot.num_joints}")
    print(f"🔗 Names: {robot.joint_names}")
    print("💡 Use mouse to rotate view, Ctrl+C to exit")
    
    # Run
    try:
        while simulation_app.is_running():
            scene.update(dt=sim_cfg.dt)
    except KeyboardInterrupt:
        print("\n✨ Done!")
    
    simulation_app.close()

if __name__ == "__main__":
    main()