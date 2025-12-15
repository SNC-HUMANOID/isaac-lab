#!/usr/bin/env python3

"""Fix SNC bot orientation permanently in URDF file"""

import xml.etree.ElementTree as ET
import shutil
import os

def fix_urdf_orientation():
    """Fix URDF base_link orientation to make robot stand upright"""
    
    urdf_original = "/home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/urdf/Humanoid_SNC.urdf"
    urdf_fixed = "/home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/urdf/Humanoid_SNC_fixed.urdf"
    
    print("🔧 Fixing SNC bot URDF orientation...")
    
    # Backup original
    backup_path = urdf_original.replace('.urdf', '_backup.urdf')
    if not os.path.exists(backup_path):
        shutil.copy2(urdf_original, backup_path)
        print(f"✅ Backup created: {backup_path}")
    
    # Parse URDF
    tree = ET.parse(urdf_original)
    root = tree.getroot()
    
    # Find base_link
    base_link = None
    for link in root.findall('link'):
        if link.get('name') == 'base_link':
            base_link = link
            break
    
    if base_link is None:
        print("❌ base_link not found!")
        return None
        
    print("✅ Found base_link")
    
    # Fix visual orientation
    visual = base_link.find('visual')
    if visual is not None:
        origin = visual.find('origin')
        if origin is not None:
            # Get current rpy
            current_rpy = origin.get('rpy', '0 0 0')
            rpy_values = current_rpy.split()
            # Add 90° rotation around Y (1.5708 radians)
            new_rpy = f"{rpy_values[0]} 1.5708 {rpy_values[2]}"
            origin.set('rpy', new_rpy)
            print(f"✅ Updated visual rpy: {current_rpy} → {new_rpy}")
        else:
            # Create origin with rotation
            origin = ET.SubElement(visual, 'origin')
            origin.set('xyz', '0 0 0')
            origin.set('rpy', '0 1.5708 0')
            print("✅ Added visual origin with Y rotation")
    
    # Fix collision orientation  
    collision = base_link.find('collision')
    if collision is not None:
        origin = collision.find('origin')
        if origin is not None:
            current_rpy = origin.get('rpy', '0 0 0')
            rpy_values = current_rpy.split()
            new_rpy = f"{rpy_values[0]} 1.5708 {rpy_values[2]}"
            origin.set('rpy', new_rpy)
            print(f"✅ Updated collision rpy: {current_rpy} → {new_rpy}")
        else:
            origin = ET.SubElement(collision, 'origin')
            origin.set('xyz', '0 0 0')  
            origin.set('rpy', '0 1.5708 0')
            print("✅ Added collision origin with Y rotation")
    
    # Save fixed URDF
    tree.write(urdf_fixed, encoding='utf-8', xml_declaration=True)
    print(f"💾 Fixed URDF saved: {urdf_fixed}")
    
    return urdf_fixed

def convert_fixed_urdf_to_usd():
    """Convert fixed URDF to USD using Isaac Lab"""
    
    from isaaclab.app import AppLauncher
    import argparse
    
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(['--headless'])
    
    app = AppLauncher(args)
    simulation_app = app.app
    
    try:
        print("🔄 Converting fixed URDF to USD...")
        
        urdf_path = "/home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/urdf/Humanoid_SNC_fixed.urdf"
        usd_output = "/home/sncbot/IsaacLab/Humaniod_SNC_ROS_URDF/Humanoid_SNC/usd/Humanoid_SNC_fixed.usd"
        
        # Convert using Isaac Sim
        import omni.kit.commands
        
        success, stage = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=urdf_path,
            import_config=omni.isaac.urdf.ImportConfig(),
            dest_path="/World/SNC_Robot"
        )
        
        if success:
            # Save USD
            import omni.usd
            usd_context = omni.usd.get_context()
            stage = usd_context.get_stage()
            if stage:
                stage.Export(usd_output)
                print(f"✅ Fixed USD saved: {usd_output}")
                return usd_output
        else:
            print("❌ URDF to USD conversion failed")
            
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        
    finally:
        simulation_app.close()
        
    return None

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔧 SNC BOT URDF ORIENTATION FIX")
    print("="*60)
    
    # Step 1: Fix URDF
    fixed_urdf = fix_urdf_orientation()
    
    if fixed_urdf:
        print(f"✅ Step 1: URDF fixed - {fixed_urdf}")
        
        # Step 2: Convert to USD
        print("\n🔄 Step 2: Converting to USD...")
        fixed_usd = convert_fixed_urdf_to_usd()
        
        if fixed_usd:
            print(f"✅ Step 2: USD created - {fixed_usd}")
            print("\n" + "="*60)
            print("🎉 SUCCESS! SNC bot orientation fixed permanently!")
            print("="*60)
            print("Files created:")
            print(f"📄 Fixed URDF: {fixed_urdf}")
            print(f"📄 Fixed USD: {fixed_usd}")
            print("\nNow SNC bot will stand upright by default!")
        else:
            print("❌ Step 2 failed")
    else:
        print("❌ Step 1 failed")