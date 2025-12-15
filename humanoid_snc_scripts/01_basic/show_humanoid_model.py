#!/usr/bin/env python3

"""แสดง Humanoid SNC model ใน Isaac Sim แบบง่ายๆ"""

import argparse
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Show Humanoid SNC Model")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
rror executing job with overrides: []
Traceback (most recent call last):
  File "/home/sncbot/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py", line 111, in hydra_main
    func(env_cfg, agent_cfg, *args, **kwargs)
  File "/home/sncbot/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py", line 183, in main
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
  File "/home/sncbot/env_isaaclab/lib/python3.10/site-packages/rsl_rl/runners/on_policy_runner.py", line 262, in learn
    loss_dict = self.alg.update()
  File "/home/sncbot/env_isaaclab/lib/python3.10/site-packages/rsl_rl/algorithms/ppo.py", line 260, in update
    self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
  File "/home/sncbot/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_snc/agents/rsl_rl_ppo_cfg.py", line 52, in patched_act
    return self.distribution.sample()
  File "/home/sncbot/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/torch/distributions/normal.py", line 73, in sample
    return torch.normal(self.loc.expand(shape), self.scale.expand(shape))
RuntimeError: normal expects all elements of std >= 0.0

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
2025-09-05 08:50:01 [38,996ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
2025-09-05 08:50:01 [38,996ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
2025-09-05 08:50:01 [38,996ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
2025-09-05 08:50:01 [38,996ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
2025-09-05 08:50:01 [38,996ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
2025-09-05 08:50:01 [38,996ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
2025-09-05 08:50:01 [38,996ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
2025-09-05 08:50:01 [38,996ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
2025-09-05 08:50:01 [38,996ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
2025-09-05 08:50:01 [38,996ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
2025-09-05 08:50:01 [38,996ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
2025-09-05 08:50:01 [38,996ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
2025-09-05 08:50:01 [38,996ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Core' for removal
2025-09-05 08:50:01 [39,001ms] [Warning] [omni.graph.core.plugin] Could not find category 'animation' for removal
2025-09-05 08:50:01 [39,258ms] [Warning] [omni.fabric.plugin] gFabricState->gUsdStageToSimStageWithHistoryMap had 1 outstanding SimStageWithHistory(s) at shutdown
2025-09-05 08:50:01 [39,260ms] [Warning] [carb.audio.context] 1 contexts were leaked
2025-09-05 08:50:01 [39,317ms] [Warning] [carb] Recursive unloadAllPlugins() detected!
(env_isaaclab) sncbot@sncbot-ThinkStation-P720:~/IsaacLab$ ./isaaclab.sh -p
      scripts/reinforcement_learning/rsl_rl/train.py --task
      Isaac-Humanoid-SNC-Direct-v0 --num_envs 4 --headless |
       head -100


import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg


@configclass
class SimpleSceneCfg(InteractiveSceneCfg):
    """Simple scene config to show robot model."""
    
    # Robot - แค่โหลด model มาแสดง
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/sncbot/IsaacLab/humanoid_snc_assets/humanoid_snc.usd",
            copy_from_source=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={".*": 0.0},  # ทุกข้อต่อ = 0
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
    """แสดง model หุ่นยนต์"""
    
    # สร้างพื้น
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    
    # เพิ่มแสง
    cfg_light = sim_utils.DomeLightCfg(intensity=2000.0)
    cfg_light.func("/World/Light", cfg_light)
    
    # ตั้งค่า simulation
    sim_cfg = SimulationCfg(dt=1/60)
    sim = SimulationContext(sim_cfg)
    
    # สร้าง scene
    scene_cfg = SimpleSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)
    scene = InteractiveScene(scene_cfg)
    
    # เริ่ม simulation
    sim.reset()
    print("=" * 50)
    print("🤖 แสดง Humanoid SNC Model")
    print("=" * 50)
    
    # ข้อมูล robot
    robot: Articulation = scene["robot"]
    
    print(f"✅ โหลด model สำเร็จ!")
    print(f"📊 จำนวน joints: {robot.num_joints}")
    print(f"🔗 ชื่อ joints: {robot.joint_names}")
    print(f"📏 ตำแหน่ง: {robot.data.root_pos_w[0].cpu().numpy()}")
    print("=" * 50)
    print("💡 Tips:")
    print("- ใช้เมาส์หมุนดูรอบๆ")
    print("- กด Ctrl+C เพื่อออก")
    print("=" * 50)
    
    # วน loop แสดงผล
    count = 0
    try:
        while simulation_app.is_running():
            # ไม่ทำอะไร แค่แสดง model
            scene.update(dt=sim_cfg.dt)
            
            if count % 300 == 0:  # ทุก 5 วินาที
                height = robot.data.root_pos_w[0, 2].item()
                print(f"⏱️  เวลา: {count//60:.1f}s, ความสูง: {height:.2f}m")
            
            count += 1
            
    except KeyboardInterrupt:
        print("\n🛑 หยุดการแสดงผล")
    
    print("✨ เสร็จสิ้น!")
    simulation_app.close()


if __name__ == "__main__":
    main()