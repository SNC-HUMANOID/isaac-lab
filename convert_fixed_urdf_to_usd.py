#!/usr/bin/env python3
"""Convert fixed URDF to USD using Isaac Sim."""

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
import isaaclab.sim as sim_utils

# Create simulation context
sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
sim = sim_utils.SimulationContext(sim_cfg)

# Configure URDF converter
urdf_path = "/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_fixed_hip_axis.urdf"
usd_path = "/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5/Humanoid_SNC_R5_fixed_hip_axis.usd"

converter_cfg = UrdfConverterCfg(
    asset_path=urdf_path,
    usd_dir="/home/sncbot/IsaacLab/mycobot_description/urdf/SNC_R5",
    usd_file_name="Humanoid_SNC_R5_fixed_hip_axis.usd",
    force_usd_conversion=True,  # Force reconversion
    make_instanceable=False,
    fix_base=False,
)

print(f"Converting URDF to USD...")
print(f"  Input:  {urdf_path}")
print(f"  Output: {usd_path}")

converter = UrdfConverter(converter_cfg)
usd_path_result = converter.convert()

print(f"\n✓ Conversion complete!")
print(f"  USD file: {usd_path_result}")

sim.close()
