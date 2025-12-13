#!/usr/bin/env python3
"""
G1 Robot Sensor Example
แสดงตัวอย่างการอ่านข้อมูล sensor แต่ละประเภท
"""

import numpy as np
import time
import math

class MockG1Robot:
    """Mock G1 Robot สำหรับแสดงตัวอย่าง"""
    
    def __init__(self):
        # จำลองสถานะ robot
        self.position = [0.0, 0.0, 0.74]  # x, y, z
        self.orientation = [1.0, 0.0, 0.0, 0.0]  # quaternion w,x,y,z
        self.velocity = [0.0, 0.0, 0.0]
        self.angular_velocity = [0.0, 0.0, 0.0]
        
        # 37 joints (mock data)
        self.joint_names = [
            # Legs (9)
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", 
            "torso",
            # Feet (4) 
            "left_ankle_pitch", "left_ankle_roll",
            "right_ankle_pitch", "right_ankle_roll",
            # Arms (24) - simplified
            "left_shoulder_pitch", "left_shoulder_roll", "left_elbow_pitch",
            "right_shoulder_pitch", "right_shoulder_roll", "right_elbow_pitch",
        ] + [f"finger_{i}" for i in range(18)]  # 18 finger joints
        
        # Default joint positions (37 values)
        self.default_positions = np.array([
            # Legs
            0.0, 0.0, -0.2, 0.42,  # left leg
            0.0, 0.0, -0.2, 0.42,  # right leg  
            0.0,                   # torso
            # Feet
            -0.23, 0.0,           # left ankle
            -0.23, 0.0,           # right ankle
            # Arms
            0.35, 0.16, 0.87,     # left arm
            0.35, -0.16, 0.87,    # right arm
        ] + [0.0] * 18)           # fingers
        
        # Current joint states
        self.joint_positions = self.default_positions.copy()
        self.joint_velocities = np.zeros(37)
        
        # AI memory
        self.previous_actions = np.zeros(37)
        
        # User commands
        self.user_commands = [0.0, 0.0, 0.0]  # vx, vy, wz
        
        print("🤖 Mock G1 Robot initialized with 37 joints")

def demonstrate_sensor_reading():
    """แสดงตัวอย่างการอ่าน sensor ต่างๆ"""
    
    print("📡 G1 Robot Sensor Data Sources")
    print("=" * 50)
    
    robot = MockG1Robot()
    
    # จำลองการทำงานแบบ real-time
    for step in range(5):
        print(f"\n⏰ Time Step {step+1}/5 (every 0.02s)")
        print("-" * 30)
        
        # 1. IMU Data - Base Motion (6 values)
        print("🧭 IMU Sensors:")
        
        # Linear velocity (3 values)
        base_lin_vel = [
            0.5 * math.sin(step * 0.1),      # vx - walking forward/backward
            0.1 * math.cos(step * 0.2),      # vy - side stepping  
            0.0                              # vz - usually near zero
        ]
        
        # Angular velocity (3 values)  
        base_ang_vel = [
            0.05 * math.sin(step * 0.3),     # roll rate
            0.02 * math.cos(step * 0.4),     # pitch rate
            0.1 * math.sin(step * 0.15)      # yaw rate (turning)
        ]
        
        print(f"   Linear velocity:  [{base_lin_vel[0]:.3f}, {base_lin_vel[1]:.3f}, {base_lin_vel[2]:.3f}] m/s")
        print(f"   Angular velocity: [{base_ang_vel[0]:.3f}, {base_ang_vel[1]:.3f}, {base_ang_vel[2]:.3f}] rad/s")
        
        # 2. Gravity Vector (3 values)
        print("\n🌍 Gravity Sensor:")
        
        # Projected gravity (robot frame)
        tilt_x = 0.1 * math.sin(step * 0.1)  # slight tilt
        tilt_y = 0.05 * math.cos(step * 0.2) # slight tilt
        projected_gravity = [tilt_x, tilt_y, -1.0]  # mostly pointing down
        
        print(f"   Projected gravity: [{projected_gravity[0]:.3f}, {projected_gravity[1]:.3f}, {projected_gravity[2]:.3f}]")
        print("   (When standing straight: [0, 0, -1])")
        
        # 3. User Commands (3 values)
        print("\n🎮 User Input:")
        
        # Simulate user commands
        user_commands = [
            0.8 if step < 3 else 0.0,        # walk forward first 3 steps
            0.2 * math.sin(step * 0.5),      # side step
            0.3 * math.cos(step * 0.3)       # turning
        ]
        
        print(f"   Desired velocity: [{user_commands[0]:.3f}, {user_commands[1]:.3f}, {user_commands[2]:.3f}]")
        print("   (vx=forward/back, vy=left/right, wz=turn)")
        
        # 4. Joint Positions (37 values)  
        print("\n🦴 Joint Encoders - Positions:")
        
        # Simulate joint movement
        joint_positions = robot.default_positions.copy()
        for i in range(37):
            # Add some movement
            joint_positions[i] += 0.1 * math.sin(step * 0.2 + i * 0.1)
        
        # Show first 10 joints
        print(f"   First 10 joints: {joint_positions[:10]}")
        print("   (Relative to default positions, in radians)")
        
        # 5. Joint Velocities (37 values)
        print("\n🏃 Joint Encoders - Velocities:")
        
        # Simulate joint velocities
        joint_velocities = np.zeros(37)
        for i in range(37):
            joint_velocities[i] = 0.5 * math.cos(step * 0.3 + i * 0.2)
        
        print(f"   First 10 velocities: {joint_velocities[:10]}")
        print("   (Joint speeds in rad/s)")
        
        # 6. Previous Actions (37 values)
        print("\n🧠 AI Memory - Previous Actions:")
        
        # Previous AI commands
        previous_actions = robot.previous_actions
        if step > 0:
            # Update with some values
            previous_actions = 0.2 * np.random.randn(37)
        
        print(f"   First 10 prev actions: {previous_actions[:10]}")
        print("   (Last AI commands sent to joints)")
        
        # 7. Combine all data
        print("\n📊 Combined Sensor Data:")
        
        sensor_data = np.concatenate([
            base_lin_vel,          # 3 values
            base_ang_vel,          # 3 values
            projected_gravity,     # 3 values  
            user_commands,         # 3 values
            joint_positions,       # 37 values
            joint_velocities,      # 37 values
            previous_actions       # 37 values
        ])
        
        print(f"   Total data points: {len(sensor_data)}")
        print(f"   Expected: 123 values")
        print(f"   Breakdown: 3+3+3+3+37+37+37 = {3+3+3+3+37+37+37}")
        
        # Update robot state for next iteration
        robot.previous_actions = 0.1 * np.random.randn(37)
        
        # Simulate 50 Hz timing
        if step < 4:  # Don't sleep on last iteration
            time.sleep(0.02)  # 20ms = 50 Hz

def show_sensor_hardware():
    """แสดงข้อมูลฮาร์ดแวร์ที่ต้องใช้"""
    
    print("\n" + "=" * 50)
    print("🛠️  Required Hardware for Real Robot")
    print("=" * 50)
    
    hardware_list = {
        "IMU (Inertial Measurement Unit)": {
            "quantity": "1 unit",
            "location": "Robot torso/chest", 
            "data": "Linear velocity (3) + Angular velocity (3) + Gravity (3)",
            "frequency": "1000+ Hz",
            "example": "Xsens MTi-630, VectorNav VN-100"
        },
        "Joint Encoders": {
            "quantity": "37 units", 
            "location": "Each joint motor",
            "data": "Position (37) + Velocity (37)", 
            "frequency": "1000+ Hz",
            "example": "Magnetic encoders, Optical encoders"
        },
        "User Interface": {
            "quantity": "1 unit",
            "location": "Remote controller", 
            "data": "Velocity commands (3)",
            "frequency": "50+ Hz", 
            "example": "Xbox controller, Keyboard, ROS topics"
        },
        "Main Computer": {
            "quantity": "1 unit",
            "location": "Robot body",
            "data": "AI processing + Control loop",
            "frequency": "50 Hz",
            "example": "NVIDIA Jetson, Intel NUC"
        }
    }
    
    for name, specs in hardware_list.items():
        print(f"\n📡 {name}:")
        for key, value in specs.items():
            print(f"   {key.capitalize()}: {value}")

def main():
    """Main demonstration"""
    demonstrate_sensor_reading()
    show_sensor_hardware()
    
    print("\n" + "=" * 50)
    print("🎯 Summary: Sensor Data Sources")
    print("=" * 50)
    print("1. IMU → Base motion (6) + Gravity (3) = 9 values")
    print("2. Joint encoders → Positions (37) + Velocities (37) = 74 values") 
    print("3. User input → Commands = 3 values")
    print("4. AI memory → Previous actions = 37 values")
    print("Total: 9 + 74 + 3 + 37 = 123 values")
    print()
    print("💡 All sensors must be synchronized and read at 50 Hz!")

if __name__ == "__main__":
    main()