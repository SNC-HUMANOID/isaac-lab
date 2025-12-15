#!/usr/bin/env python3
"""🦵 ตรวจสอบการทำงานของ joints ขา"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check Leg Joints Activity")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np

from source.isaaclab_tasks.isaaclab_tasks.direct.humanoid_snc.humanoid_snc_env import HumanoidSNCEnv, HumanoidSNCEnvCfg

def main():
    """ตรวจสอบการทำงานของ joints ขา"""
    
    # สร้าง environment
    env_cfg = HumanoidSNCEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = HumanoidSNCEnv(cfg=env_cfg, render_mode="rgb_array")
    
    print("🦵 ตรวจสอบ Leg Joints...")
    print("=" * 60)
    
    # รายชื่อ leg joints
    leg_joints = [
        "left_hip_pitch_joint",   # ขาซ้าย - สะโพก หน้า/หลัง
        "left_hip_roll_joint",    # ขาซ้าย - สะโพก ซ้าย/ขวา
        "left_hip_yaw_joint",     # ขาซ้าย - สะโพก หมุน
        "left_ankle_joint",       # ขาซ้าย - ข้อเท้า
        "left_ankle_pitch_joint", # ขาซ้าย - ข้อเท้า หน้า/หลัง
        "left_ankle_yaw_joint",   # ขาซ้าย - ข้อเท้า หมุน
        "right_hip_pitch_joint",  # ขาขวา - สะโพก หน้า/หลัง
        "right_hip_roll_joint",   # ขาขวา - สะโพก ซ้าย/ขวา  
        "right_hip_yaw_joint",    # ขาขวา - สะโพก หมุน
        "right_ankle",            # ขาขวา - ข้อเท้า
        "right_ankle_roll_joint", # ขาขวา - ข้อเท้า ซ้าย/ขวา
        "right_ankle_yaw_joint",  # ขาขวา - ข้อเท้า หมุน
    ]
    
    # Reset environment
    env.reset()
    
    # เก็บข้อมูลเริ่มต้น
    joint_data = {}
    
    for step in range(200):  # รัน 200 steps
        # สุ่ม actions เบาๆ
        actions = torch.randn(args_cli.num_envs, env.cfg.action_space, device=env.device) * 0.2
        
        # Step simulation
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # เก็บข้อมูลทุก 20 steps
        if step % 20 == 0:
            joint_positions = env.robot.data.joint_pos[0]  # env แรก
            joint_efforts = env.robot.data.applied_torque[0]
            joint_velocities = env.robot.data.joint_vel[0]
            
            for joint_name in leg_joints:
                if joint_name not in joint_data:
                    joint_data[joint_name] = {
                        'positions': [],
                        'efforts': [],
                        'velocities': [],
                        'index': -1
                    }
                
                # หา index ของ joint
                try:
                    if joint_data[joint_name]['index'] == -1:
                        joint_data[joint_name]['index'] = env.robot.joint_names.index(joint_name)
                    
                    idx = joint_data[joint_name]['index']
                    joint_data[joint_name]['positions'].append(joint_positions[idx].item())
                    joint_data[joint_name]['efforts'].append(joint_efforts[idx].item())
                    joint_data[joint_name]['velocities'].append(joint_velocities[idx].item())
                    
                except ValueError:
                    # ถ้าหา joint ไม่เจอ
                    joint_data[joint_name]['positions'].append(0.0)
                    joint_data[joint_name]['efforts'].append(0.0)
                    joint_data[joint_name]['velocities'].append(0.0)
    
    # ปิด environment
    env.close()
    
    # วิเคราะห์ผล
    print("\n📊 สรุปการทำงานของ Leg Joints:")
    print("=" * 80)
    print(f"{'Joint Name':<25} {'Range (rad)':<12} {'Max Effort':<12} {'Max Vel':<12} {'Status'}")
    print("-" * 80)
    
    for joint_name in leg_joints:
        if joint_name in joint_data and joint_data[joint_name]['positions']:
            positions = np.array(joint_data[joint_name]['positions'])
            efforts = np.array(joint_data[joint_name]['efforts'])
            velocities = np.array(joint_data[joint_name]['velocities'])
            
            pos_range = np.max(positions) - np.min(positions)
            effort_max = np.max(np.abs(efforts))
            vel_max = np.max(np.abs(velocities))
            
            # กำหนดเกณฑ์ว่า joint ทำงานหรือไม่
            is_active = pos_range > 0.05 or effort_max > 3.0 or vel_max > 0.05
            status = "✅ ทำงาน" if is_active else "❌ ไม่ทำงาน"
            
            print(f"{joint_name:<25} {pos_range:<12.3f} {effort_max:<12.1f} {vel_max:<12.2f} {status}")
        else:
            print(f"{joint_name:<25} {'N/A':<12} {'N/A':<12} {'N/A':<12} ❓ ไม่พบ")
    
    print("\n📋 คำอธิบาย:")
    print("- Range: ช่วงการเคลื่อนไหวของ joint (มากกว่า 0.05 rad = ทำงาน)")
    print("- Max Effort: แรงบิด maximum ที่ใช้ (มากกว่า 3.0 N⋅m = ทำงาน)")
    print("- Max Vel: ความเร็วสูงสุด (มากกว่า 0.05 rad/s = ทำงาน)")
    
    print("\n🏃 Joint Groups:")
    left_hip = [j for j in leg_joints if 'left_hip' in j]
    left_ankle = [j for j in leg_joints if 'left_ankle' in j] 
    right_hip = [j for j in leg_joints if 'right_hip' in j]
    right_ankle = [j for j in leg_joints if 'right_ankle' in j]
    
    print(f"ซ้าย - สะโพก: {len([j for j in left_hip if j in joint_data])}/{len(left_hip)} joints")
    print(f"ซ้าย - ข้อเท้า: {len([j for j in left_ankle if j in joint_data])}/{len(left_ankle)} joints") 
    print(f"ขวา - สะโพก: {len([j for j in right_hip if j in joint_data])}/{len(right_hip)} joints")
    print(f"ขวา - ข้อเท้า: {len([j for j in right_ankle if j in joint_data])}/{len(right_ankle)} joints")

if __name__ == "__main__":
    main()
    simulation_app.close()