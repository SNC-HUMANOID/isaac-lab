#!/usr/bin/env python3
"""Convert Humanoid_10102025 URDF to USD format for Isaac Lab."""

import argparse
from pathlib import Path

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


def main():
    """Convert the Humanoid_10102025 URDF to USD."""

    # Set up paths
    urdf_path = "/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/urdf/Humanoid_10102025.urdf"
    output_dir = "/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd"

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Configure the URDF converter
    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=output_dir,
        usd_file_name="Humanoid_10102025.usd",
        # Force sensor settings
        force_usd_conversion=True,
        make_instanceable=False,
        # Physics settings
        fix_base=False,  # Humanoid should not be fixed to ground
        merge_fixed_joints=False,
        # Default drive settings for joints
        default_drive_type="none",  # We'll configure actuators separately
        default_drive_stiffness=0.0,
        default_drive_damping=0.0,
        # Joint settings
        override_joint_dynamics=True,
        # Visual/Collision settings
        import_sites=True,
    )

    # Create converter and run conversion
    print(f"Converting URDF: {urdf_path}")
    print(f"Output directory: {output_dir}")

    urdf_converter = UrdfConverter(urdf_converter_cfg)

    # Convert the URDF to USD
    usd_path = urdf_converter.convert()

    print(f"\n✓ Conversion complete!")
    print(f"USD file created at: {usd_path}")
    print(f"\nYou can now use this USD file in your Isaac Lab robot configuration.")


if __name__ == "__main__":
    main()
