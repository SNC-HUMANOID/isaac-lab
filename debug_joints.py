#!/usr/bin/env python3

"""Debug script เพื่อตรวจสอบ joints ของ Humanoid SNC"""

import argparse
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug Humanoid SNC joints")
parser.add_argument("--headless", action="store_true", default=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.direct.humanoid_snc.humanoid_snc_env import (
    HumanoidSNCEnv, 
    HumanoidSNCEnvCfg
)

def main():
    print("🔍 ตรวจสอบ joints ของ Humanoid SNC")
    
    try:
        # สร้าง environment
        env_cfg = HumanoidSNCEnvCfg()
        env_cfg.scene.num_envs = 1
        env = HumanoidSNCEnv(cfg=env_cfg)
        
        # ตรวจสอบ robot
        robot = env.scene["robot"]
        
        print(f"\n📊 ข้อมูล Robot:")
        print(f"   - Joint names: {robot.joint_names}")
        print(f"   - Num joints: {robot.num_joints}")
        print(f"   - Action space shape: {env.action_space.shape}")
        print(f"   - Config action_space: {env.cfg.action_space}")
        
        # ดูข้อมูลจาก data
        robot_data = robot.data
        print(f"   - Robot data joint_pos shape: {robot_data.joint_pos.shape}")
        print(f"   - Robot data joint_vel shape: {robot_data.joint_vel.shape}")
        
        # ตรวจสอบ joint names
        print(f"\n🔧 Joint names:")
        for i, name in enumerate(robot.joint_names):
            print(f"   {i:2d}: {name}")
            
        env.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()