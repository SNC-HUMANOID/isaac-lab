#!/usr/bin/env python3
"""
แสดงค่า Output จากโมเดล AI และบอกว่าควบคุม Joint ไหน
"""

import torch
import numpy as np

def show_joint_mapping():
    """แสดงการแมพ output กับ joints"""
    
    print("🎯 G1 Model Outputs → Joint Control")
    print("=" * 60)
    
    # G1 joint names (37 joints)
    joint_names = [
        # Legs (9 joints)
        "left_hip_yaw_joint",       # 0
        "left_hip_roll_joint",      # 1  
        "left_hip_pitch_joint",     # 2 - สำคัญ!
        "left_knee_joint",          # 3 - สำคัญ!
        "torso_joint",              # 4
        "right_hip_yaw_joint",      # 5
        "right_hip_pitch_joint",    # 6 - สำคัญ!
        "right_knee_joint",         # 7 - สำคัญ!
        "right_hip_roll_joint",     # 8
        
        # Feet (4 joints)
        "left_ankle_pitch_joint",   # 9 - สำคัญ!
        "left_ankle_roll_joint",    # 10
        "right_ankle_pitch_joint",  # 11 - สำคัญ!
        "right_ankle_roll_joint",   # 12
        
        # Arms (6 main joints)
        "left_shoulder_pitch_joint",  # 13
        "left_shoulder_roll_joint",   # 14
        "left_elbow_pitch_joint",     # 15
        "right_shoulder_pitch_joint", # 16
        "right_shoulder_roll_joint",  # 17
        "right_elbow_pitch_joint",    # 18
    ] + [f"hand_joint_{i}" for i in range(19, 37)]  # Hand joints 19-36
    
    return joint_names

def analyze_walking_outputs():
    """วิเคราะห์ output สำหรับการเดิน"""
    
    print("\n🚶 Walking Output Analysis")
    print("-" * 40)
    
    # โหลดโมเดล
    model_path = 'logs/rsl_rl/g1_flat/2025-07-29_15-27-26/model_300.pt'
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # สร้างโมเดล
    state_dict = checkpoint['model_state_dict']
    actor_layers = [k for k in state_dict.keys() if 'actor' in k and 'weight' in k]
    
    layers = []
    for i, layer_name in enumerate(actor_layers):
        weight = state_dict[layer_name]
        in_size, out_size = weight.shape[1], weight.shape[0]
        layers.append(torch.nn.Linear(in_size, out_size))
        if i < len(actor_layers) - 1:
            layers.append(torch.nn.ELU())
    
    policy_net = torch.nn.Sequential(*layers)
    actor_state = {k.replace('actor.', ''): v for k, v in state_dict.items() if 'actor' in k}
    policy_net.load_state_dict(actor_state, strict=False)
    policy_net.eval()
    
    joint_names = show_joint_mapping()
    
    # ทดสอบการเดินไปข้างหน้า
    dummy_input = torch.zeros(1, 123)
    dummy_input[0, 0] = 0.5   # forward velocity
    dummy_input[0, 7] = -1.0  # gravity
    dummy_input[0, 9] = 0.5   # command forward
    
    with torch.no_grad():
        raw_output = policy_net(dummy_input)
    
    actions = raw_output[0].numpy()
    scaled_actions = actions * 0.5  # action scale
    
    # แสดงเฉพาะ joints ที่สำคัญสำหรับการเดิน
    walking_joints = [
        (2, "left_hip_pitch_joint"),
        (3, "left_knee_joint"), 
        (6, "right_hip_pitch_joint"),
        (7, "right_knee_joint"),
        (9, "left_ankle_pitch_joint"),
        (11, "right_ankle_pitch_joint"),
    ]
    
    print("\n🔥 PRIMARY WALKING JOINTS:")
    print("Index | Joint Name              | Raw Output | Scaled | Description")
    print("-" * 75)
    
    for idx, joint_name in walking_joints:
        desc = ""
        if "hip_pitch" in joint_name:
            desc = "ยกขา/วางขา"
        elif "knee" in joint_name:
            desc = "งอเข่า"
        elif "ankle_pitch" in joint_name:
            desc = "เท้าขึ้น/ลง"
            
        print(f"{idx:5d} | {joint_name:22s} | {actions[idx]:10.3f} | {scaled_actions[idx]:6.3f} | {desc}")
    
    # แสดง joints อื่นๆ
    print("\n⚖️ BALANCE & OTHER JOINTS:")
    print("Index | Joint Name              | Raw Output | Scaled | Description")
    print("-" * 75)
    
    other_joints = [
        (1, "left_hip_roll_joint", "สมดุลซ้าย"),
        (8, "right_hip_roll_joint", "สมดุลขวา"),
        (4, "torso_joint", "หมุนลำตัว"),
        (10, "left_ankle_roll_joint", "สมดุลเท้าซ้าย"),
        (12, "right_ankle_roll_joint", "สมดุลเท้าขวา"),
    ]
    
    for idx, joint_name, desc in other_joints:
        print(f"{idx:5d} | {joint_name:22s} | {actions[idx]:10.3f} | {scaled_actions[idx]:6.3f} | {desc}")
    
    # แสดงช่วงค่า
    print(f"\n📊 OUTPUT STATISTICS:")
    print(f"   Total joints: {len(actions)}")
    print(f"   Raw range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"   Scaled range: [{scaled_actions.min():.3f}, {scaled_actions.max():.3f}]")
    print(f"   Action scale: 0.5")
    
    print(f"\n💡 HOW IT WORKS:")
    print(f"   1. AI outputs RAW values [-1 to +1] for each joint")
    print(f"   2. Scale by 0.5 → reduces movement intensity")
    print(f"   3. Add to default joint positions → final target")
    print(f"   4. PD controller → converts to motor torques")
    
    print(f"\n🎯 EXAMPLE - Walking Forward:")
    print(f"   • left_hip_pitch = {scaled_actions[2]:.3f} rad")
    print(f"   • left_knee = {scaled_actions[3]:.3f} rad") 
    print(f"   • right_hip_pitch = {scaled_actions[6]:.3f} rad")
    print(f"   • right_knee = {scaled_actions[7]:.3f} rad")
    print(f"   → These values get added to default positions")
    print(f"   → Robot moves legs to create walking motion!")

if __name__ == "__main__":
    analyze_walking_outputs()