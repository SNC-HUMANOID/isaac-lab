#!/usr/bin/env python3

"""Fix URDF asymmetries for SNC R5."""

def fix_urdf_symmetry():
    """Fix asymmetric joint positions in URDF."""
    
    urdf_path = "/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5.urdf"
    output_path = "/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_symmetric.urdf"
    
    print("Reading URDF...")
    with open(urdf_path, 'r') as f:
        content = f.read()
    
    print("Fixing asymmetric joint origins...")
    
    # 1. Fix Hip Pitch Origins (make them symmetric)
    # Left: xyz="-0.000452534747040726 0.0729985973310563 -0.0564999999999999"
    # Right: xyz="0.000452534747029725 -0.072998597331071 -0.0565000000000009"
    # Fix to: Left = (0, 0.073, -0.0565), Right = (0, -0.073, -0.0565)
    
    content = content.replace(
        'xyz="-0.000452534747040726 0.0729985973310563 -0.0564999999999999"',
        'xyz="0 0.073 -0.0565"'
    )
    
    content = content.replace(
        'xyz="0.000452534747029725 -0.072998597331071 -0.0565000000000009"',
        'xyz="0 -0.073 -0.0565"'
    )
    
    # 2. Fix Knee Joint Origins (make them symmetric)
    # Left: xyz="0 -0.000499973693548289 -0.172500000000001"
    # Right: xyz="3.27588224962126E-05 0.000498927126663967 -0.172500000000001"
    # Fix to: Left = (0, 0, -0.1725), Right = (0, 0, -0.1725)
    
    content = content.replace(
        'xyz="0 -0.000499973693548289 -0.172500000000001"',
        'xyz="0 0 -0.1725"'
    )
    
    content = content.replace(
        'xyz="3.27588224962126E-05 0.000498927126663967 -0.172500000000001"',
        'xyz="0 0 -0.1725"'
    )
    
    # 3. Fix Ankle Roll Origins (make them symmetric)
    # Left: xyz="0.0269450236686518 -0.000276411998827469 -0.317014020661395"
    # Right: xyz="0.0149854793516257 -0.000983891657905356 -0.317802560297351"
    # Fix to: Use average values for both
    # Average X: (0.0269450 + 0.0149854) / 2 = 0.0209652
    # Average Z: (-0.317014 + -0.317802) / 2 = -0.317408
    
    content = content.replace(
        'xyz="0.0269450236686518 -0.000276411998827469 -0.317014020661395"',
        'xyz="0.021 0 -0.317"'
    )
    
    content = content.replace(
        'xyz="0.0149854793516257 -0.000983891657905356 -0.317802560297351"',
        'xyz="0.021 0 -0.317"'
    )
    
    # 4. Fix Ankle Pitch Origins (make them symmetric)
    # Left: xyz="0.0413 -0.00070964 -0.0056613"  
    # Right: xyz="0.0414347430259809 -0.00300701414895965 -0.0035170920201526"
    # Fix to: Use average values
    
    content = content.replace(
        'xyz="0.0413 -0.00070964 -0.0056613"',
        'xyz="0.041 0 -0.005"'
    )
    
    content = content.replace(
        'xyz="0.0414347430259809 -0.00300701414895965 -0.0035170920201526"',
        'xyz="0.041 0 -0.005"'
    )
    
    # 5. Fix Hip Roll Origins (make them symmetric)
    # Check if there are other asymmetries
    
    print("Writing fixed URDF...")
    with open(output_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed URDF saved to: {output_path}")
    print("\nFixed joint origins:")
    print("- Hip Pitch: (0, ±0.073, -0.0565)")
    print("- Knee: (0, 0, -0.1725)")  
    print("- Ankle Roll: (0.021, 0, -0.317)")
    print("- Ankle Pitch: (0.041, 0, -0.005)")
    print("\nAll legs should now be symmetric!")

if __name__ == "__main__":
    fix_urdf_symmetry()