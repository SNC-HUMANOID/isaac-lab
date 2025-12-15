#!/usr/bin/env python3
"""
Model Debug Script - แสดงข้อมูล Input/Output ของโมเดล
"""

import torch
import numpy as np
import yaml

def analyze_model_io():
    """วิเคราะห์ Input/Output ของโมเดล"""
    
    print("🤖 G1 Model Input/Output Analysis")
    print("=" * 50)
    
    # 1. โหลดโมเดล
    model_path = "logs/rsl_rl/g1_flat/2025-07-29_15-27-26/model_300.pt"
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    print("✅ Model loaded successfully!")
    print(f"   Training iterations: {checkpoint.get('iter', 'Unknown')}")
    
    # 2. วิเคราะห์โครงสร้าง
    state_dict = checkpoint['model_state_dict']
    actor_layers = [k for k in state_dict.keys() if 'actor' in k and 'weight' in k]
    
    input_size = state_dict[actor_layers[0]].shape[1]   # 123
    output_size = state_dict[actor_layers[-1]].shape[0] # 37
    
    print(f"\n🧠 Neural Network Structure:")
    print(f"   Input size:  {input_size} values")
    print(f"   Output size: {output_size} values")
    
    # 3. แสดง Input breakdown
    print(f"\n📥 INPUT BREAKDOWN ({input_size} values):")
    print(f"   Values 1-3:   Base linear velocity [vx, vy, vz] m/s")
    print(f"   Values 4-6:   Base angular velocity [wx, wy, wz] rad/s") 
    print(f"   Values 7-9:   Projected gravity [gx, gy, gz] normalized")
    print(f"   Values 10-12: Velocity commands [cmd_vx, cmd_vy, cmd_wz]")
    print(f"   Values 13-49: Joint positions (37 joints) radians")
    print(f"   Values 50-86: Joint velocities (37 joints) rad/s")
    print(f"   Values 87-123: Previous actions (37 joints)")
    
    # 4. แสดง Output breakdown
    print(f"\n📤 OUTPUT BREAKDOWN ({output_size} values):")
    joint_groups = {
        "Legs (9 joints)": ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", 
                            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "torso"],
        "Feet (4 joints)": ["left_ankle_pitch", "left_ankle_roll", "right_ankle_pitch", "right_ankle_roll"],
        "Arms & Hands (24 joints)": ["left_shoulder_pitch", "left_shoulder_roll", "left_elbow_pitch", 
                                     "right_shoulder_pitch", "right_shoulder_roll", "right_elbow_pitch", "...hands"]
    }
    
    for group, joints in joint_groups.items():
        print(f"   {group}:")
        for joint in joints[:3]:
            print(f"     • {joint} target position (radians)")
        if len(joints) > 3:
            print(f"     • ... and {len(joints)-3} more")
    
    # 5. ทดสอบโมเดล
    print(f"\n🧪 TESTING MODEL:")
    
    # สร้างโมเดล
    try:
        layers = []
        for i, layer_name in enumerate(actor_layers):
            weight = state_dict[layer_name]
            in_size, out_size = weight.shape[1], weight.shape[0]
            layers.append(torch.nn.Linear(in_size, out_size))
            if i < len(actor_layers) - 1:
                layers.append(torch.nn.ELU())
        
        policy_net = torch.nn.Sequential(*layers)
        
        # โหลด weights
        actor_state = {k.replace('actor.', ''): v for k, v in state_dict.items() if 'actor' in k}
        policy_net.load_state_dict(actor_state, strict=False)
        policy_net.eval()
        
        # สร้าง dummy input (เดินไปข้างหน้า)
        dummy_input = torch.zeros(1, input_size)
        dummy_input[0, 0] = 0.5   # forward velocity
        dummy_input[0, 7] = -1.0  # gravity pointing down
        dummy_input[0, 9] = 0.5   # command: walk forward
        
        # ทดสอบ
        with torch.no_grad():
            output = policy_net(dummy_input)
        
        print(f"   ✅ Model works!")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # แสดงการแปลงค่า
        raw_actions = output[0].numpy()
        action_scale = 0.5  # จาก config
        scaled_actions = raw_actions * action_scale
        
        print(f"\n🎯 EXAMPLE - Walking Forward Command:")
        print(f"   Input: forward velocity = 0.5 m/s")
        print(f"   AI raw output (first 5 joints): {raw_actions[:5]}")
        print(f"   After scaling (×0.5): {scaled_actions[:5]}")
        print(f"   → These become joint position targets (radians)")
        
    except Exception as e:
        print(f"   ❌ Model test failed: {e}")
    
    # 6. แสดงการประมวลผล
    print(f"\n⚙️ PROCESSING PIPELINE:")
    print(f"   1. Collect sensor data → {input_size} values")
    print(f"   2. Neural network inference → {output_size} raw actions [-1,+1]")
    print(f"   3. Scale by 0.5 → reduce movement magnitude")
    print(f"   4. Add default joint positions → absolute targets")  
    print(f"   5. Send to PD controllers → convert to torques")
    print(f"   6. Apply torques to motors → robot moves")
    
    # 7. สรุป
    print(f"\n🎉 SUMMARY:")
    print(f"   • AI thinks in terms of JOINT POSITIONS, not torques")
    print(f"   • Input: {input_size} sensor readings (velocities, positions, etc.)")  
    print(f"   • Output: {output_size} joint angle targets in radians")
    print(f"   • PD controllers handle the low-level torque control")
    print(f"   • Control rate: 50 Hz (20ms per cycle)")
    
    print(f"\n💡 FOR REAL ROBOT DEPLOYMENT:")
    print(f"   while True:")
    print(f"       sensors = read_robot_sensors()      # {input_size} values")
    print(f"       targets = ai_model(sensors)         # {output_size} positions")  
    print(f"       send_position_commands(targets)     # to PD controllers")
    print(f"       time.sleep(0.02)                    # 50 Hz")

if __name__ == "__main__":
    analyze_model_io()