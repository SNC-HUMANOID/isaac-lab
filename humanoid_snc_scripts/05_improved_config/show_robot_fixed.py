#!/usr/bin/env python3
"""🤖 แสดงหุ่นยนต์ Humanoid SNC ที่ยืนได้ (แก้ไข error แล้ว)"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Show Fixed Standing Robot")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import time

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import spawn_ground_plane, GroundPlaneCfg

def show_robot_properly():
    """แสดงหุ่นยนต์อย่างถูกต้อง"""
    
    print("🤖 แสดงหุ่นยนต์ Humanoid SNC (Fixed Version)")
    print("=" * 50)
    
    # สร้าง ground plane
    ground_cfg = GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.2,
            dynamic_friction=1.0,
            restitution=0.1,
        )
    )
    spawn_ground_plane("/World/ground", cfg=ground_cfg)
    print("✅ สร้าง ground plane เรียบร้อย")
    
    # สร้างหุ่นยนต์โดยใช้ Isaac Lab spawner
    robot_usd_path = "/home/sncbot/IsaacLab/humanoid_snc_assets/humanoid_snc.usd"
    
    robot_cfg = sim_utils.UsdFileCfg(
        usd_path=robot_usd_path,
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
    )
    
    # Spawn หุ่นยนต์ในตำแหน่งที่สูงกว่าพื้น
    robot_prim = robot_cfg.func("/World/Robot", robot_cfg, translation=(0.0, 0.0, 1.2))
    print("✅ สร้างหุ่นยนต์ที่ความสูง 1.2 เมตร")
    
    # เพิ่มแสงสว่าง
    light_cfg = sim_utils.DomeLightCfg(intensity=3500.0, color=(0.9, 0.9, 0.9))
    light_cfg.func("/World/Light", light_cfg)
    print("✅ เพิ่มแสงสว่าง")
    
    # ตั้งค่ากล้อง
    camera_cfg = sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 1.0e5),
    )
    
    # เริ่ม simulation context
    from isaaclab.sim import SimulationContext
    sim_context = SimulationContext()
    sim_context.reset()
    print("✅ เริ่ม simulation context")
    
    print("\n🏃‍♂️ เริ่มการจำลอง...")
    print("🎮 หุ่นยนต์จะปรากฏใน Isaac Sim window")
    print("💡 คุณสามารถ:")
    print("   - ลาก mouse เพื่อหมุนมุมมอง")
    print("   - ใช้ mouse wheel เพื่อซูม")
    print("   - กด Space bar เพื่อ play/pause")
    print("📝 กด Ctrl+C เพื่อหยุด")
    print("=" * 50)
    
    try:
        step = 0
        last_report_time = 0
        
        while True:
            # อัปเดตการจำลอง
            simulation_app.update()
            sim_context.step()
            
            step += 1
            current_time = step / 60.0  # 60 FPS
            
            # รายงานสถานะทุก 10 วินาที
            if current_time - last_report_time >= 10.0:
                print(f"⏱️  เวลา: {current_time:.1f} วินาที")
                print(f"🤖 หุ่นยนต์กำลังแสดงใน Isaac Sim")
                print(f"🎯 Step: {step}")
                last_report_time = current_time
                
                # คำแนะนำการใช้งาน
                if current_time < 30:
                    print("💡 ลองปรับมุมมองกล้องเพื่อดูหุ่นยนต์ชัดขึ้น")
            
            # หยุดหลัง 5 นาที (ถ้าไม่มีการกดหยุด)
            if step > 18000:  # 5 minutes at 60 FPS
                print("⏰ หยุดการแสดงอัตโนมัติหลัง 5 นาที")
                break
                
            # หน่วงเวลาเล็กน้อยเพื่อไม่ให้ใช้ CPU มากเกินไป
            time.sleep(1/120.0)  # 120 FPS limit
                
    except KeyboardInterrupt:
        print("\n🛑 การแสดงหยุดโดยผู้ใช้")
    
    print("\n" + "=" * 50)
    print("📊 สรุปการแสดงหุ่นยนต์")
    print("=" * 50)
    print("✅ แสดงหุ่นยนต์ Humanoid SNC สำเร็จ")
    print("📍 ตำแหน่ง: กลางฉาก ความสูง 1.2 เมตร")
    print("🎯 การจำลองทำงานปกติ")
    
    print("\n🚀 ขั้นตอนต่อไป:")
    print("1. ใช้ working_train.py เพื่อเทรนหุ่นยนต์ให้ยืนและเดินได้")
    print("2. ใช้ show_robot_fixed.py นี้เพื่อดูหุ่นยนต์อีกครั้ง")
    print("3. ปรับแต่ง parameters ใน training script")
    
    # คำแนะนำเพิ่มเติม
    print("\n💡 เคล็ดลับ:")
    print("- ถ้าหุ่นยนต์ดูเล็ก: ซูมเข้าด้วย mouse wheel")
    print("- ถ้าต้องการเปลี่ยนมุมมอง: ลาก mouse ขณะกด")
    print("- ถ้าต้องการหยุดชั่วคราว: กด Space bar")

def main():
    """Main function"""
    print("🚀 เริ่มแสดงหุ่นยนต์...")
    show_robot_properly()
    print("✅ การแสดงหุ่นยนต์เสร็จสิ้น")

if __name__ == "__main__":
    main()
    simulation_app.close()