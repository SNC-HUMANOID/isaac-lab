#!/usr/bin/env python3

"""
สคริปต์ทดสอบเร็วสำหรับ Humanoid SNC 
ตรวจสอบว่า environment ทำงานได้และไม่มี error
"""

import argparse
import torch
from isaaclab.app import AppLauncher

# แทรก args
parser = argparse.ArgumentParser(description="Quick test Humanoid SNC")
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments for testing")
args_cli = parser.parse_args()

# เริ่มต้น app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import หลังจาก app launcher
from isaaclab_tasks.direct.humanoid_snc.humanoid_snc_env import (
    HumanoidSNCEnv, 
    HumanoidSNCEnvCfg
)

def main():
    """ทดสอบ environment แบบเร็ว"""
    
    print("🧪 ทดสอบ Humanoid SNC Environment")
    print("=" * 50)
    
    # สร้าง config สำหรับทดสอบ
    env_cfg = HumanoidSNCEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = 5.0  # ทดสอบแป๊ปเดียว
    
    print(f"🌍 สร้าง {args_cli.num_envs} environments")
    
    try:
        # สร้าง environment
        env = HumanoidSNCEnv(cfg=env_cfg)
        print("✅ Environment สร้างสำเร็จ")
        
        # Reset environment
        obs, _ = env.reset()
        print(f"✅ Reset สำเร็จ - obs shape: {obs['policy'].shape}")
        
        # ทดสอบ step ไม่กี่ครั้ง
        print("\n🎮 ทดสอบ actions...")
        action_dim = env.action_space.shape[0]
        print(f"Action dimension: {action_dim}")
        print(f"Environment num_actions: {env.num_actions}")
        print(f"Joint gears length: {len(env.cfg.joint_gears)}")
        
        for i in range(10):
            # สุ่ม actions ตามจำนวนที่ถูกต้อง
            actions = torch.randn(args_cli.num_envs, action_dim, device=env.device) * 0.1
            
            # Step
            obs, rewards, dones, truncated, info = env.step(actions)
            
            print(f"Step {i+1:2d}: reward_mean={rewards.mean():.3f}, done_count={dones.sum()}")
            
        print("\n✅ การทดสอบเสร็จสิ้น - Environment ทำงานปกติ")
        
        # ปิด environment
        env.close()
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        simulation_app.close()
        print("🏁 ปิดโปรแกรมเรียบร้อย")

if __name__ == "__main__":
    main()