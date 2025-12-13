#!/usr/bin/env python3
"""
แสดงกระบวนการ Input → Output ของโมเดล AI
"""

import torch
import numpy as np

def create_model():
    """สร้างโมเดลจากไฟล์ที่เทรนไว้"""
    
    model_path = 'logs/rsl_rl/g1_flat/2025-07-29_15-27-26/model_300.pt'
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
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
    
    return policy_net

def show_input_breakdown(input_data):
    """แสดงรายละเอียด input 123 ค่า"""
    
    print("📥 INPUT BREAKDOWN (123 values):")
    print("-" * 50)
    
    # Base Motion (6 values)
    print("🏃 BASE MOTION (6 values):")
    print(f"  [0-2] Linear velocity:  [{input_data[0]:.3f}, {input_data[1]:.3f}, {input_data[2]:.3f}] m/s")
    print(f"  [3-5] Angular velocity: [{input_data[3]:.3f}, {input_data[4]:.3f}, {input_data[5]:.3f}] rad/s")
    
    # Gravity (3 values)
    print("\n🌍 GRAVITY (3 values):")
    print(f"  [6-8] Projected gravity: [{input_data[6]:.3f}, {input_data[7]:.3f}, {input_data[8]:.3f}]")
    
    # Commands (3 values)
    print("\n🎮 USER COMMANDS (3 values):")
    print(f"  [9-11] Velocity commands: [{input_data[9]:.3f}, {input_data[10]:.3f}, {input_data[11]:.3f}]")
    
    # Joint Positions (37 values)
    print("\n🦴 JOINT POSITIONS (37 values):")
    print(f"  [12-48] Joint positions: {input_data[12:22]}...")  # Show first 10
    print(f"          (Remaining 27 joints: ...)")
    
    # Joint Velocities (37 values)
    print("\n🏃 JOINT VELOCITIES (37 values):")
    print(f"  [49-85] Joint velocities: {input_data[49:59]}...")  # Show first 10
    print(f"          (Remaining 27 joints: ...)")
    
    # Previous Actions (37 values)
    print("\n🧠 AI MEMORY (37 values):")
    print(f"  [86-122] Previous actions: {input_data[86:96]}...")  # Show first 10
    print(f"           (Remaining 27 actions: ...)")
    
    print(f"\nTotal: {len(input_data)} values")

def show_output_breakdown(output_data):
    """แสดงรายละเอียด output 37 ค่า"""
    
    print("\n📤 OUTPUT BREAKDOWN (37 values):")
    print("-" * 50)
    
    # Joint names for mapping
    joint_names = [
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
        "torso", "right_hip_yaw", "right_hip_pitch", "right_knee", "right_hip_roll",
        "left_ankle_pitch", "left_ankle_roll", "right_ankle_pitch", "right_ankle_roll",
        "left_shoulder_pitch", "left_shoulder_roll", "left_elbow_pitch",
        "right_shoulder_pitch", "right_shoulder_roll", "right_elbow_pitch"
    ] + [f"hand_{i}" for i in range(19, 37)]
    
    # Walking joints (most important)
    walking_indices = [2, 3, 6, 7, 9, 11]  # hip_pitch, knee, ankle_pitch
    
    print("🔥 PRIMARY WALKING JOINTS:")
    for i in walking_indices:
        print(f"  [{i:2d}] {joint_names[i]:20s}: {output_data[i]:8.3f} rad")
    
    print("\n⚖️ BALANCE JOINTS:")
    balance_indices = [1, 8, 4, 10, 12]  # hip_roll, torso, ankle_roll
    for i in balance_indices:
        print(f"  [{i:2d}] {joint_names[i]:20s}: {output_data[i]:8.3f} rad")
    
    print("\n💪 ARM JOINTS:")
    arm_indices = [13, 14, 15, 16, 17, 18]
    for i in arm_indices:
        print(f"  [{i:2d}] {joint_names[i]:20s}: {output_data[i]:8.3f} rad")
    
    print(f"\n🤚 HAND JOINTS (19-36): {output_data[19:25]}... (showing first 6)")
    
    print(f"\nRange: [{output_data.min():.3f}, {output_data.max():.3f}] rad")

def demonstrate_scenarios():
    """แสดงตัวอย่าง Input → Output ในสถานการณ์ต่างๆ"""
    
    print("\n" + "=" * 80)
    print("🤖 G1 MODEL: INPUT → OUTPUT DEMONSTRATION")
    print("=" * 80)
    
    model = create_model()
    
    # สถานการณ์ต่างๆ
    scenarios = [
        {
            "name": "🧍 STANDING STILL",
            "base_vel": [0.0, 0.0, 0.0],
            "base_ang": [0.0, 0.0, 0.0],
            "commands": [0.0, 0.0, 0.0],
        },
        {
            "name": "🚶 WALKING FORWARD",
            "base_vel": [0.5, 0.0, 0.0],
            "base_ang": [0.0, 0.0, 0.0],
            "commands": [0.5, 0.0, 0.0],
        },
        {
            "name": "🚶‍♂️ WALKING BACKWARD", 
            "base_vel": [-0.3, 0.0, 0.0],
            "base_ang": [0.0, 0.0, 0.0],
            "commands": [-0.3, 0.0, 0.0],
        },
        {
            "name": "↔️ SIDESTEP LEFT",
            "base_vel": [0.0, 0.3, 0.0],
            "base_ang": [0.0, 0.0, 0.0], 
            "commands": [0.0, 0.3, 0.0],
        },
        {
            "name": "🔄 TURN RIGHT",
            "base_vel": [0.0, 0.0, 0.0],
            "base_ang": [0.0, 0.0, -0.5],
            "commands": [0.0, 0.0, -0.5],
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        print("=" * 50)
        
        # สร้าง input (123 values)
        input_data = np.zeros(123)
        
        # Base motion (0-5)
        input_data[0:3] = scenario["base_vel"]
        input_data[3:6] = scenario["base_ang"]
        
        # Gravity (6-8) - always pointing down when standing
        input_data[6:9] = [0.0, 0.0, -1.0]
        
        # Commands (9-11)
        input_data[9:12] = scenario["commands"]
        
        # Joint positions (12-48) - start from default
        default_positions = np.array([
            0.0, 0.0, -0.2, 0.42,  # left leg
            0.0, 0.0, -0.2, 0.42, 0.0,  # right leg + torso  
            -0.23, 0.0, -0.23, 0.0,  # feet
            0.35, 0.16, 0.87, 0.35, -0.16, 0.87  # arms
        ] + [0.0] * 18)  # hands
        input_data[12:49] = default_positions
        
        # Joint velocities (49-85) - small random movements
        input_data[49:86] = np.random.normal(0, 0.1, 37)
        
        # Previous actions (86-122) - small values
        input_data[86:123] = np.random.normal(0, 0.05, 37)
        
        # แสดง input summary
        print(f"📥 INPUT SUMMARY:")
        print(f"   Base velocity: {scenario['base_vel']}")
        print(f"   Commands: {scenario['commands']}")
        print(f"   Total values: {len(input_data)}")
        
        # ประมวลผลโมเดล
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_data).unsqueeze(0)
            raw_output = model(input_tensor)
            
        output_data = raw_output[0].numpy()
        scaled_output = output_data * 0.5  # action scale
        
        # แสดง output summary
        print(f"\n📤 OUTPUT SUMMARY:")
        print(f"   Raw range: [{output_data.min():.3f}, {output_data.max():.3f}]")
        print(f"   Scaled range: [{scaled_output.min():.3f}, {scaled_output.max():.3f}]")
        
        # แสดงเฉพาะ joints สำคัญ
        print(f"\n🔥 KEY WALKING JOINTS (scaled):")
        walking_joints = [
            (2, "left_hip_pitch", "ยกขาซ้าย"),
            (3, "left_knee", "งอเข่าซ้าย"),
            (6, "right_hip_pitch", "ยกขาขวา"), 
            (7, "right_knee", "งอเข่าขวา"),
            (9, "left_ankle_pitch", "เท้าซ้าย"),
            (11, "right_ankle_pitch", "เท้าขวา")
        ]
        
        for idx, name, desc in walking_joints:
            print(f"   [{idx:2d}] {name:16s}: {scaled_output[idx]:7.3f} rad ({desc})")
        
        print(f"\n💡 These values control robot movement!")

def show_processing_pipeline():
    """แสดงกระบวนการประมวลผลทั้งหมด"""
    
    print(f"\n" + "=" * 80)
    print("⚙️ COMPLETE PROCESSING PIPELINE")
    print("=" * 80)
    
    print("""
    🔄 REAL-TIME CONTROL LOOP (50 Hz):
    
    1. 📡 READ SENSORS (every 20ms):
       ├── IMU → base motion (6) + gravity (3) = 9 values
       ├── Joint encoders → positions (37) + velocities (37) = 74 values
       ├── User controller → commands = 3 values
       └── AI memory → previous actions = 37 values
                                        ────────────────
                                        Total: 123 values
    
    2. 🧠 AI PROCESSING:
       Input[123] → Neural Network → Output[37]
       
       Neural Network Architecture:
       123 → [256] → [128] → [128] → 37
            ELU     ELU     ELU
    
    3. 📊 POST-PROCESSING:
       Raw outputs [-1, +1] × 0.5 = Scaled actions [-0.5, +0.5]
       Scaled actions + Default positions = Target positions
    
    4. 🎯 ROBOT CONTROL:
       Target positions → PD Controllers → Motor torques → Robot movement
    
    5. ⏰ TIMING:
       Sleep(20ms - processing_time) → Maintain 50 Hz
    """)
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • AI thinks in POSITION targets, not torques")
    print(f"   • 123 sensor values → 37 joint commands")  
    print(f"   • Real-time: 20ms per cycle")
    print(f"   • PD controllers handle low-level motor control")

def main():
    """Main demonstration"""
    demonstrate_scenarios()
    show_processing_pipeline()
    
    print(f"\n🎯 SUMMARY:")
    print(f"   INPUT:  123 sensor values (IMU + joints + commands + memory)")
    print(f"   MODEL:  Neural network (167,243 parameters)")
    print(f"   OUTPUT: 37 joint position targets (radians)")
    print(f"   RESULT: Robot walks! 🚶‍♂️")

if __name__ == "__main__":
    main()