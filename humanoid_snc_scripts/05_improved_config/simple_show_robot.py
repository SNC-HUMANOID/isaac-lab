#!/usr/bin/env python3
"""🤖 แสดงหุ่นยนต์ Humanoid SNC แบบง่ายๆ"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Simple Show Robot")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import time

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import spawn_ground_plane, GroundPlaneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

def simple_show_robot():
    """แสดงหุ่นยนต์แบบง่ายๆ"""
    
    print("🤖 แสดงหุ่นยนต์ Humanoid SNC")
    print("=" * 40)
    
    # สร้าง ground plane
    ground_cfg = GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        )
    )
    spawn_ground_plane("/World/ground", cfg=ground_cfg)
    
    # สร้างหุ่นยนต์แบบง่าย
    robot_usd_path = "/home/sncbot/IsaacLab/humanoid_snc_assets/humanoid_snc.usd"
    
    robot_cfg = sim_utils.UsdFileCfg(
        usd_path=robot_usd_path,
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
    )
    
    # Spawn หุ่นยนต์
    robot_prim = robot_cfg.func("/World/Robot", robot_cfg)
    
    # เพิ่มแสง
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
    light_cfg.func("/World/Light", light_cfg)
    
    print("✅ สร้างหุ่นยนต์เรียบร้อย")
    
    # ตั้งตำแหน่งหุ่นยนต์ให้อยู่สูงจากพื้น
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    robot_prim_path = "/World/Robot"
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    
    if robot_prim.IsValid():
        # ตั้งตำแหน่งเริ่มต้น (x, y, z)
        from pxr import Gf, UsdGeom
        xform = UsdGeom.Xformable(robot_prim)
        transform_op = xform.AddTransformOp()
        
        # สร้าง transformation matrix
        translation = Gf.Vec3d(0.0, 0.0, 1.5)  # ยกสูง 1.5 เมตร
        rotation = Gf.Rotation(Gf.Vec3d(1, 0, 0), 0)  # ไม่หมุน
        scale = Gf.Vec3d(1.0, 1.0, 1.0)  # ขนาดปกติ
        
        matrix = Gf.Matrix4d().SetTransform(rotation, translation, scale)
        transform_op.Set(matrix)
        
        print("📍 ตั้งตำแหน่งหุ่นยนต์ที่ความสูง 1.5 เมตร")
    else:
        print("⚠️ ไม่พบหุ่นยนต์ในฉาก")
    
    # เริ่มการจำลอง
    from isaaclab.sim import SimulationContext
    sim_context = SimulationContext.instance()
    if sim_context is None:
        sim_context = SimulationContext()
    
    sim_context.reset()
    
    print("🏃‍♂️ เริ่มการจำลอง...")
    print("💡 หุ่นยนต์จะอยู่ในท่ายืน")
    print("🎮 คุณสามารถดูได้ใน Isaac Sim window")
    print("📝 กด Ctrl+C ใน terminal เพื่อหยุด")
    print("=" * 40)
    
    try:
        step = 0
        while True:
            # อัปเดตการจำลอง
            simulation_app.update()
            sim_context.step()
            
            step += 1
            
            # รายงานสถานะทุก 600 steps (10 วินาที)
            if step % 600 == 0:
                print(f"⏱️  เวลา: {step/60:.1f} วินาที - หุ่นยนต์กำลังแสดงอยู่")
                print("🤖 ตรวจสอบ Isaac Sim window เพื่อดูหุ่นยนต์")
                
            # หยุดหลัง 30 นาที
            if step > 108000:  # 30 minutes at 60 FPS
                print("⏰ หยุดการแสดงหลัง 30 นาที")
                break
                
            # หน่วงเวลาเล็กน้อย
            time.sleep(1/60.0)  # 60 FPS
                
    except KeyboardInterrupt:
        print("\n🛑 การแสดงหยุดโดยผู้ใช้")
    
    print("\n" + "=" * 40)
    print("📊 สรุป:")
    print("✅ หุ่นยนต์ Humanoid SNC แสดงเสร็จแล้ว")
    print("💡 หุ่นยนต์อยู่ในท่าพื้นฐาน")
    print("🎯 พร้อมสำหรับการพัฒนาและเทรน RL ต่อไป")
    
    # การใช้งานต่อไป
    print("\n🚀 ขั้นตอนต่อไป:")
    print("1. ใช้ working_train.py เพื่อเทรนให้หุ่นยนต์ยืนได้")
    print("2. ปรับแต่ง reward function ตามต้องการ")
    print("3. ทดลองเทรน locomotion behaviors")

def main():
    """Main function"""
    print("🚀 เริ่มแสดงหุ่นยนต์...")
    simple_show_robot()
    print("✅ การแสดงหุ่นยนต์เสร็จสิ้น")

if __name__ == "__main__":
    main()
    simulation_app.close()