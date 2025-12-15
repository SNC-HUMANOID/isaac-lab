#!/usr/bin/env python3
"""Check H1 configuration for left-right symmetry issues."""

import re

def check_h1_config():
    print("="*80)
    print("H1 CONFIGURATION SYMMETRY CHECK")
    print("="*80)

    # Check unitree.py for joint configuration
    with open("/home/sncbot/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py", "r") as f:
        content = f.read()

    # Find H1_CFG init_state section
    h1_section = re.search(r'H1_CFG = ArticulationCfg\((.*?)^\)', content, re.DOTALL | re.MULTILINE)

    if h1_section:
        init_state = re.search(r'init_state=ArticulationCfg\.InitialStateCfg\((.*?)\)', h1_section.group(1), re.DOTALL)
        if init_state:
            joint_pos = re.search(r'joint_pos=\{(.*?)\}', init_state.group(1), re.DOTALL)
            if joint_pos:
                print("\n📋 H1 Initial Joint Positions:")
                print("-" * 80)
                for line in joint_pos.group(1).strip().split('\n'):
                    line = line.strip().rstrip(',')
                    if line and ':' in line:
                        print(f"  {line}")

    # Check for left/right specific configurations
    print("\n\n🔍 Searching for LEFT-specific configurations:")
    print("-" * 80)
    left_matches = re.findall(r'.*left.*', content, re.IGNORECASE)
    if left_matches:
        for match in left_matches[:10]:
            print(f"  {match.strip()}")
    else:
        print("  ❌ No left-specific configs found")

    print("\n\n🔍 Searching for RIGHT-specific configurations:")
    print("-" * 80)
    right_matches = re.findall(r'.*right.*', content, re.IGNORECASE)
    if right_matches:
        for match in right_matches[:10]:
            print(f"  {match.strip()}")
    else:
        print("  ❌ No right-specific configs found")

    # Check reward configuration
    print("\n\n📊 Checking Reward Configuration:")
    print("-" * 80)

    with open("/home/sncbot/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/rough_env_cfg.py", "r") as f:
        reward_content = f.read()

    # Find feet_air_time configuration
    feet_air = re.search(r'feet_air_time.*?body_names.*?["\']([^"\']+)["\']', reward_content, re.DOTALL)
    if feet_air:
        print(f"  Feet air time body names: {feet_air.group(1)}")
        print(f"  ✅ Using pattern matching - should work for both feet")

    # Check if there are any asymmetric patterns
    print("\n\n⚠️  Checking for Asymmetric Patterns:")
    print("-" * 80)

    # Patterns that might cause asymmetry
    asymmetric_patterns = [
        (r'left.*\d+', "Left with specific value"),
        (r'right.*\d+', "Right with specific value"),
        (r'L_.*=', "L_ prefix assignment"),
        (r'R_.*=', "R_ prefix assignment"),
    ]

    full_content = content + reward_content
    issues_found = False

    for pattern, desc in asymmetric_patterns:
        matches = re.findall(pattern, full_content, re.IGNORECASE)
        if matches:
            print(f"  ⚠️  {desc}: {matches[:3]}")
            issues_found = True

    if not issues_found:
        print("  ✅ No obvious asymmetric patterns found")

    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print("""
1. The H1 config uses '.*' patterns which apply to BOTH legs equally
2. This should NOT cause left-leg-only issues

Possible causes of left leg not moving:

  a) ACTUATOR ISSUE: Check if left leg actuators are getting commands
     - Solution: Check actuator effort limits and stiffness

  b) CONTACT SENSOR: Left foot contact sensor might be stuck "ON"
     - Solution: Check contact force readings in training

  c) REWARD BIAS: feet_air_time might only reward right foot
     - Solution: Debug the feet_air_time_positive_biped function

  d) JOINT LIMITS: Left leg joints hitting limits
     - Solution: Check joint position limits in URDF/USD

Try this in training to debug:
  - Add print statements in feet_air_time_positive_biped
  - Check contact forces: print(env.scene["contact_forces"].data.net_forces_w)
  - Monitor joint positions for left vs right leg
""")

if __name__ == "__main__":
    check_h1_config()
