#!/usr/bin/env python3

"""Convert URDF to upright orientation and regenerate USD"""

import argparse
from isaaclab.app import AppLauncher
import xml.etree.ElementTree as ET
import shutil
import os

def fix_urdf_orientation():
    """Fix the URDF file to have upright orientation"""
    
    urdf_path = "/home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/urdf/Humanoid_SNC.urdf"
    fixed_urdf_path = "/home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/urdf/Humanoid_SNC_upright.urdf"
    
    print("📖 Reading original URDF...")
    
    # Read and parse URDF
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # Find base_link and add rotation to its visual and collision
    base_link = None
    for link in root.findall('link'):
        if link.get('name') == 'base_link':
            base_link = link
            break
    
    if base_link is None:
        print("❌ Could not find base_link in URDF")
        return False
    
    print("✅ Found base_link")
    
    # Add rotation to visual origin (90 degrees around Y-axis)
    visual = base_link.find('visual')
    if visual is not None:
        origin = visual.find('origin')
        if origin is not None:
            # Modify existing origin
            current_rpy = origin.get('rpy', '0 0 0').split()
            # Add 90° rotation around Y (1.5708 radians)
            new_rpy = f"{current_rpy[0]} 1.5708 {current_rpy[2]}"
            origin.set('rpy', new_rpy)
        else:
            # Create new origin with rotation
            origin = ET.SubElement(visual, 'origin')
            origin.set('xyz', '0 0 0')
            origin.set('rpy', '0 1.5708 0')  # 90° around Y
        
        print("✅ Added rotation to visual element")
    
    # Add rotation to collision origin
    collision = base_link.find('collision')
    if collision is not None:
        origin = collision.find('origin')
        if origin is not None:
            current_rpy = origin.get('rpy', '0 0 0').split()
            new_rpy = f"{current_rpy[0]} 1.5708 {current_rpy[2]}"
            origin.set('rpy', new_rpy)
        else:
            origin = ET.SubElement(collision, 'origin')
            origin.set('xyz', '0 0 0')
            origin.set('rpy', '0 1.5708 0')
            
        print("✅ Added rotation to collision element")
    
    # Save fixed URDF
    tree.write(fixed_urdf_path, encoding='utf-8', xml_declaration=True)
    print(f"✅ Saved upright URDF: {fixed_urdf_path}")
    
    return fixed_urdf_path

def main():
    parser = argparse.ArgumentParser(description="Convert URDF to upright and regenerate USD")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    print("\n" + "="*60)
    print("🔄 FIXING URDF ORIENTATION AND REGENERATING USD")
    print("="*60)
    
    # Fix URDF orientation
    fixed_urdf_path = fix_urdf_orientation()
    if not fixed_urdf_path:
        print("❌ Failed to fix URDF orientation")
        return
    
    # Launch Isaac Sim for USD conversion
    app = AppLauncher(args)
    simulation_app = app.app
    
    try:
        print("🔄 Converting fixed URDF to USD...")
        
        # Output path for new USD
        usd_output_path = "/home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/usd/Humanoid_SNC_upright.usd"
        
        # Import conversion modules
        import omni.kit.commands
        from pxr import Usd, UsdGeom
        import omni.usd
        
        # Convert URDF to USD using Isaac Sim converter
        success, stage = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=fixed_urdf_path,
            import_config=omni.isaac.urdf.ImportConfig(),
            dest_path="/World/Robot"
        )
        
        if success:
            print("✅ URDF conversion successful")
            
            # Save the stage as USD
            usd_context = omni.usd.get_context()
            current_stage = usd_context.get_stage()
            if current_stage:
                current_stage.Export(usd_output_path)
                print(f"✅ Upright USD saved: {usd_output_path}")
            
        else:
            print("❌ URDF conversion failed")
            
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        simulation_app.close()
        
    print("="*60)
    print("🎉 UPRIGHT ROBOT CONVERSION COMPLETED!")
    print("="*60)
    print("Files created:")
    print(f"📄 Fixed URDF: {fixed_urdf_path}")
    print(f"📄 Upright USD: {usd_output_path}")
    print("")
    print("Now use the upright USD file for visualization!")

if __name__ == "__main__":
    main()