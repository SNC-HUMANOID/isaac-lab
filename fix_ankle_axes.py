#!/usr/bin/env python3

"""Fix asymmetric ankle joint axes in SNC R5 URDF."""

def fix_ankle_axes():
    """Fix the asymmetric ankle joint axes in the URDF."""
    
    urdf_path = "/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5.urdf"
    output_path = "/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_fixed_ankles.urdf"
    
    print("Reading URDF...")
    with open(urdf_path, 'r') as f:
        content = f.read()
    
    print("Fixing asymmetric ankle joint axes...")
    
    # Fix ankle roll joints - use standard X-axis for roll motion
    # Left ankle roll axis (original: -0.999132743950063 0.0102494724906537 0.0403572580892359)
    # Right ankle roll axis (original: -0.997786917534177 0.0655109378514733 -0.0113835064661871)
    # Fix to standard: 1 0 0 for left, -1 0 0 for right (mirror symmetry)
    
    content = content.replace(
        'xyz="-0.999132743950063 0.0102494724906537 0.0403572580892359"',
        'xyz="1 0 0"'
    )
    
    content = content.replace(
        'xyz="-0.997786917534177 0.0655109378514733 -0.0113835064661871"',
        'xyz="-1 0 0"'
    )
    
    # Fix ankle pitch joints - use standard Y-axis for pitch motion  
    # Left ankle pitch axis (original: -0.0073467 -0.99742 0.071429)
    # Right ankle pitch axis (original: -0.0661595904397678 -0.995245267655799 0.071482625872153)
    # Fix to standard: 0 1 0 for both (symmetric)
    
    content = content.replace(
        'xyz="-0.0073467 -0.99742 0.071429"',
        'xyz="0 1 0"'
    )
    
    content = content.replace(
        'xyz="-0.0661595904397678 -0.995245267655799 0.071482625872153"',
        'xyz="0 1 0"'
    )
    
    print("Writing fixed URDF...")
    with open(output_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed URDF saved to: {output_path}")
    print("\nFixed ankle joint axes:")
    print("- Left ankle roll:  1 0 0")
    print("- Right ankle roll: -1 0 0") 
    print("- Left ankle pitch: 0 1 0")
    print("- Right ankle pitch: 0 1 0")

if __name__ == "__main__":
    fix_ankle_axes()