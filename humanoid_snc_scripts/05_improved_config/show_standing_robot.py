#!/usr/bin/env python3
"""🤖 แสดงหุ่นยนต์ Humanoid SNC ที่ยืนได้ใน Isaac Sim"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Show Standing Humanoid Robot")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import time

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.actuators import ImplicitActuatorCfg

# การตั้งค่าหุ่นยนต์ที่ยืนได้
HUMANOID_STANDING_CFG = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/sncbot/IsaacLab/humanoid_snc_assets/humanoid_snc.usd",
        copy_from_source=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.5),  # ยกสูงให้ยืนได้
        rot=(1.0, 0.0, 0.0, 0.0),  # ยืนตรง
        joint_pos={".*": 0.0},  # เริ่มที่ท่า neutral
        joint_vel={".*": 0.0},
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=200.0,    # แรงสูงเพื่อยืน
            velocity_limit=10.0,
            stiffness=100.0,       # แข็งมากเพื่อยืน
            damping=10.0,          # damping สูงเพื่อความเสถียร
        ),
    },
)

def show_standing_robot():
    """แสดงหุ่นยนต์ที่ยืนได้"""
    
    print("🤖 กำลังแสดงหุ่นยนต์ Humanoid SNC ที่ยืนได้")
    print("=" * 50)
    
    # เริ่ม simulation context ก่อน
    from isaaclab.sim import SimulationContext
    sim_cfg = SimulationCfg(dt=1/60, render_interval=1)
    sim_context = SimulationContext(sim_cfg)
    
    # สร้าง scene
    scene_cfg = InteractiveSceneCfg(
        num_envs=1, 
        env_spacing=2.0,
        replicate_physics=False
    )
    scene = InteractiveScene(scene_cfg)
    
    # เพิ่ม ground plane ที่มี friction ดี
    ground_cfg = GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.5,  # แรงเสียดทานสูง
            dynamic_friction=1.2,
            restitution=0.0,      # ไม่เด้ง
        )
    )
    spawn_ground_plane(prim_path="/World/ground", cfg=ground_cfg)
    
    # สร้างหุ่นยนต์
    robot = Articulation(HUMANOID_STANDING_CFG)
    scene.articulations["robot"] = robot
    
    # เพิ่มแสงที่ดี
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
    light_cfg.func("/World/Light", light_cfg)
    
    print("✅ สร้างหุ่นยนต์และ scene เรียบร้อย")
    
    # เริ่มการจำลอง
    scene.reset()
    
    print("🏃‍♂️ เริ่มการจำลอง - หุ่นยนต์กำลังยืน...")
    print("📍 ตำแหน่งเริ่มต้น:", robot.data.root_pos_w.cpu().numpy())
    print("📏 ความสูงเริ่มต้น:", robot.data.root_pos_w[0, 2].item(), "เมตร")
    
    # PD control parameters สำหรับการยืน
    kp = 80.0   # Proportional gain สูง
    kd = 8.0    # Derivative gain
    
    # เป้าหมาย joint positions สำหรับการยืน (ดัดแปลงจาก YAML)
    standing_pose = torch.zeros_like(robot.data.joint_pos)
    
    # ปรับ joint specific positions เพื่อการยืนที่ดีขึ้น
    try:
        joint_names = robot.joint_names
        print("🦴 Joint Names:", joint_names)
        
        for i, joint_name in enumerate(joint_names):
            # ปรับท่าทางให้เหมาะกับการยืน
            if "hip_pitch" in joint_name.lower():
                standing_pose[0, i] = -0.1  # เอียงสะโพกไปข้างหลังเล็กน้อย
            elif "ankle_pitch" in joint_name.lower():
                standing_pose[0, i] = 0.1   # เอียงข้อเท้าไปข้างหน้า
            elif "knee" in joint_name.lower():
                standing_pose[0, i] = 0.15  # งอเข่าเล็กน้อย
                
    except Exception as e:
        print(f"⚠️ ไม่สามารถปรับ joint positions: {e}")
        print("💡 ใช้ท่า neutral ทั่วไป")
    
    print("\n🎮 เริ่มการควบคุม PD เพื่อให้หุ่นยนต์ยืน...")
    print("📊 กด Ctrl+C เพื่อหยุด")
    print("=" * 50)
    
    try:
        step = 0
        while True:
            # คำนวณ PD control
            pos_error = standing_pose - robot.data.joint_pos
            vel_error = -robot.data.joint_vel  # เป้าหมาย velocity = 0
            
            # แรง torque สำหรับการยืน
            control_torque = kp * pos_error + kd * vel_error
            control_torque = torch.clamp(control_torque, -200.0, 200.0)
            
            # ส่งแรงไปยังหุ่นยนต์
            robot.set_joint_effort_target(control_torque)
            
            # อัปเดตการจำลอง
            scene.write_data_to_sim()
            simulation_app.update()
            scene.update(dt=1/60.0)  # 60 FPS
            
            step += 1
            
            # รายงานสถานะทุก 60 steps (1 วินาที)
            if step % 60 == 0:
                current_height = robot.data.root_pos_w[0, 2].item()
                current_pos = robot.data.root_pos_w[0].cpu().numpy()
                current_quat = robot.data.root_quat_w[0].cpu().numpy()
                
                print(f"⏱️  เวลา: {step/60:.1f}s")
                print(f"📏 ความสูง: {current_height:.3f}m")
                print(f"📍 ตำแหน่ง: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.3f}]")
                print(f"🔄 การหมุน (quat): [{current_quat[0]:.3f}, {current_quat[1]:.3f}, {current_quat[2]:.3f}, {current_quat[3]:.3f}]")
                
                # ประเมินสถานะการยืน
                if current_height > 1.0:
                    print("✅ สถานะ: ยืนได้ดีมาก!")
                elif current_height > 0.8:
                    print("👍 สถานะ: ยืนได้!")
                elif current_height > 0.5:
                    print("📈 สถานะ: กำลังปรับสมดุล...")
                else:
                    print("❌ สถานะ: ต้องปรับแต่งเพิ่มเติม")
                
                print("-" * 30)
                
                # หยุดถ้าเสถียรแล้ว (ยืนได้ดี)
                if current_height > 0.9 and step > 300:  # ยืนได้ดีเป็นเวลา 5+ วินาที
                    print("🎉 หุ่นยนต์ยืนเสถียรแล้ว!")
                    print("💡 คุณสามารถปิดหน้าต่างหรือกด Ctrl+C เพื่อหยุด")
            
            # Prevent infinite loop - หยุดหลัง 30 นาที
            if step > 108000:  # 30 minutes at 60 FPS
                print("⏰ หยุดการแสดงหลัง 30 นาที")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 การแสดงหยุดโดยผู้ใช้")
    
    # สรุปผล
    final_height = robot.data.root_pos_w[0, 2].item()
    final_pos = robot.data.root_pos_w[0].cpu().numpy()
    
    print("\n" + "=" * 50)
    print("📊 สรุปผลการแสดงหุ่นยนต์")
    print("=" * 50)
    print(f"📏 ความสูงสุดท้าย: {final_height:.3f} เมตร")
    print(f"📍 ตำแหน่งสุดท้าย: [{final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.3f}]")
    
    if final_height > 1.0:
        print("🏆 ผลลัพธ์: ยอดเยี่ยม! หุ่นยนต์ยืนได้ดีมาก")
        print("✅ พร้อมสำหรับการเทรน Reinforcement Learning")
    elif final_height > 0.8:
        print("✅ ผลลัพธ์: ดีมาก! หุ่นยนต์ยืนได้")
        print("👍 สามารถเริ่มการเทรนได้")
    elif final_height > 0.5:
        print("📈 ผลลัพธ์: ใช้ได้! หุ่นยนต์กำลังปรับสมดุล")
        print("💡 อาจต้องปรับแต่ง parameters เพิ่มเติม")
    else:
        print("⚠️ ผลลัพธ์: ต้องปรับปรุง")
        print("🔧 ลองเพิ่ม stiffness หรือ kp gain")
    
    print("\n🎯 การใช้งานต่อไป:")
    if final_height > 0.8:
        print("   - ใช้ working_train.py เพื่อเทรน RL")
        print("   - ปรับ reward function ตามต้องการ")
    else:
        print("   - ปรับ stiffness และ damping")
        print("   - ทดสอบ show_standing_robot.py อีกครั้ง")

def main():
    """Main function"""
    show_standing_robot()
    print("\n✅ การแสดงหุ่นยนต์เสร็จสิ้น")

if __name__ == "__main__":
    main()
    simulation_app.close()