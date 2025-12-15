#!/usr/bin/env python3
"""🚀 แก้ไขปัญหาหุ่นยนต์นอนอย่างง่ายๆ"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Simple Standing Fix")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.actuators import ImplicitActuatorCfg

# แก้ไขการตั้งค่าหุ่นยนต์ให้ยืนได้
HUMANOID_STANDING_CFG = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/sncbot/IsaacLab/humanoid_snc_assets/humanoid_snc.usd",
        copy_from_source=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # สำคัญมาก: ตั้งความสูงและท่าทาง
        pos=(0.0, 0.0, 1.5),  # ยกให้สูงขึ้นจากพื้น
        rot=(1.0, 0.0, 0.0, 0.0),  # ตรง ไม่เอียง
        # ตั้งค่า joint ให้อยู่ในท่าที่เหมาะสำหรับการยืน
        joint_pos={
            ".*hip_pitch.*": -0.1,     # เอียงสะโพกไปข้างหลังเล็กน้อย
            ".*ankle_pitch.*": 0.1,    # เอียงข้อเท้าไปข้างหน้า
            ".*": 0.0,                 # ส่วนอื่นๆ เป็น 0
        },
        joint_vel={".*": 0.0},  # ไม่มีความเร็วเริ่มต้น
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=150.0,    # แรงพอที่จะรองรับน้ำหนัก
            velocity_limit=10.0,
            stiffness=80.0,        # แข็งพอที่จะยืน
            damping=8.0,           # damping สูงเพื่อความเสถียร
        ),
    },
)

def test_simple_standing():
    """ทดสอบการยืนแบบง่ายๆ"""
    
    print("🚀 ทดสอบแก้ไขหุ่นยนต์นอน")
    
    # สร้าง scene
    scene_cfg = InteractiveSceneCfg(
        num_envs=1, 
        env_spacing=2.0,
        replicate_physics=False
    )
    scene = InteractiveScene(scene_cfg)
    
    # เพิ่ม ground
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    
    # สร้างหุ่นยนต์
    robot = Articulation(HUMANOID_STANDING_CFG)
    
    # เพิ่มหุ่นยนต์เข้า scene
    scene.articulations["robot"] = robot
    
    # เพิ่มแสง
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)
    
    print("🤖 สร้างหุ่นยนต์เรียบร้อย")
    
    # เริ่มการจำลอง
    scene.reset()
    
    print("🏃‍♂️ เริ่มการจำลอง...")
    print("Robot position:", robot.data.root_pos_w)
    print("Robot height:", robot.data.root_pos_w[0, 2].item())
    
    # ส่งแรงเล็กๆ เพื่อช่วยในการยืน
    for step in range(500):
        
        # สร้าง target joint positions สำหรับการยืน
        target_pos = torch.zeros_like(robot.data.joint_pos)
        target_pos[0, robot.find_joints(".*hip_pitch.*")[0]] = -0.1
        target_pos[0, robot.find_joints(".*ankle_pitch.*")[0]] = 0.1
        
        # PD control เพื่อช่วยยืน
        pos_error = target_pos - robot.data.joint_pos
        vel_error = -robot.data.joint_vel
        
        control_torque = 50.0 * pos_error + 5.0 * vel_error
        control_torque = torch.clamp(control_torque, -50.0, 50.0)
        
        # ส่งแรงไปยังหุ่นยนต์
        robot.set_joint_effort_target(control_torque)
        
        # อัปเดต scene
        scene.write_data_to_sim()
        simulation_app.update()
        scene.update(dt=1/60.0)
        
        if step % 100 == 0:
            height = robot.data.root_pos_w[0, 2].item()
            print(f"Step {step}: Robot height = {height:.3f}m")
            
            if height > 0.8:
                print("✅ หุ่นยนต์ยืนได้แล้ว!")
            elif height < 0.5:
                print("❌ หุ่นยนต์ยังนอนอยู่")
    
    final_height = robot.data.root_pos_w[0, 2].item()
    print(f"\n📊 ความสูงสุดท้าย: {final_height:.3f}m")
    
    if final_height > 0.8:
        print("🎉 สำเร็จ! หุ่นยนต์ยืนได้แล้ว")
        print("💡 ตอนนี้คุณสามารถใช้การตั้งค่านี้ในการเทรน RL ได้แล้ว")
    else:
        print("⚠️ หุ่นยนต์ยังไม่ค่อยยืน อาจต้องปรับการตั้งค่าเพิ่มเติม")
        print("💡 ลองเพิ่ม stiffness หรือปรับ initial position")

def main():
    print("🚀 เริ่มแก้ไขปัญหาหุ่นยนต์นอน...")
    test_simple_standing()
    print("✅ เสร็จสิ้น!")

if __name__ == "__main__":
    main()
    simulation_app.close()