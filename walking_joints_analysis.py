#!/usr/bin/env python3
"""
G1 Walking Joints Analysis
วิเคราะห์ว่า Joint ไหนสำคัญสำหรับการเดิน และค่าที่ต้องดู
"""

import numpy as np
import matplotlib.pyplot as plt
import math

class G1WalkingAnalysis:
    """วิเคราะห์ joints ที่สำคัญสำหรับการเดิน"""
    
    def __init__(self):
        # กำหนด joint groups และความสำคัญ
        self.joint_groups = {
            "🦵 PRIMARY WALKING JOINTS (สำคัญที่สุด)": {
                "joints": [
                    ("left_hip_pitch_joint", "ยกขาซ้าย", "สำคัญมาก"),
                    ("left_knee_joint", "งอเข่าซ้าย", "สำคัญมาก"), 
                    ("left_ankle_pitch_joint", "เท้าซ้าย", "สำคัญมาก"),
                    ("right_hip_pitch_joint", "ยกขาขวา", "สำคัญมาก"),
                    ("right_knee_joint", "งอเข่าขวา", "สำคัญมาก"),
                    ("right_ankle_pitch_joint", "เท้าขวา", "สำคัญมาก"),
                ],
                "function": "ควบคุมการเดินไปข้างหน้า/หลัง",
                "pattern": "สลับกันระหว่างขาซ้าย-ขวา",
                "priority": 1
            },
            
            "⚖️ BALANCE & STABILITY JOINTS (สำคัญรอง)": {
                "joints": [
                    ("left_hip_roll_joint", "เอียงสะโพกซ้าย", "สำคัญ"),
                    ("left_ankle_roll_joint", "เอียงเท้าซ้าย", "สำคัญ"),
                    ("right_hip_roll_joint", "เอียงสะโพกขวา", "สำคัญ"),
                    ("right_ankle_roll_joint", "เอียงเท้าขวา", "สำคัญ"),
                    ("torso_joint", "หมุนลำตัว", "ปานกลาง"),
                ],
                "function": "รักษาสมดุลขณะเดิน",
                "pattern": "ปรับตามการเดิน",
                "priority": 2
            },
            
            "🔄 TURNING JOINTS (สำหรับเลี้ยว)": {
                "joints": [
                    ("left_hip_yaw_joint", "หมุนสะโพกซ้าย", "ปานกลาง"),
                    ("right_hip_yaw_joint", "หมุนสะโพกขวา", "ปานกลาง"),
                ],
                "function": "ควบคุมการเลี้ยวซ้าย/ขวา",
                "pattern": "หมุนตามทิศทางที่ต้องการ",
                "priority": 3
            },
            
            "💪 ARM JOINTS (ช่วยสมดุล)": {
                "joints": [
                    ("left_shoulder_pitch_joint", "แขนซ้าย", "น้อย"),
                    ("left_elbow_pitch_joint", "ข้อศอกซ้าย", "น้อย"),
                    ("right_shoulder_pitch_joint", "แขนขวา", "น้อย"),
                    ("right_elbow_pitch_joint", "ข้อศอกขวา", "น้อย"),
                ],
                "function": "แกว่งแขนช่วยสมดุล",
                "pattern": "แกว่งย้อนกับขา",
                "priority": 4
            }
        }
        
        # ค่าที่ต้องดูสำหรับการเดิน
        self.walking_parameters = {
            "position_ranges": {
                "hip_pitch": {"min": -0.5, "max": 0.3, "unit": "rad", "description": "ยกขา/วางขา"},
                "knee": {"min": 0.0, "max": 1.2, "unit": "rad", "description": "งอเข่า"},
                "ankle_pitch": {"min": -0.4, "max": 0.2, "unit": "rad", "description": "เท้า"},
            },
            "velocity_ranges": {
                "hip_pitch": {"min": -3.0, "max": 3.0, "unit": "rad/s", "description": "ความเร็วยกขา"},
                "knee": {"min": -4.0, "max": 4.0, "unit": "rad/s", "description": "ความเร็วงอเข่า"},
                "ankle_pitch": {"min": -2.0, "max": 2.0, "unit": "rad/s", "description": "ความเร็วเท้า"},
            }
        }

    def analyze_walking_joints(self):
        """วิเคราะห์ joints ที่สำคัญสำหรับการเดิน"""
        
        print("🤖 G1 Walking Joints Analysis")
        print("=" * 60)
        
        for group_name, group_info in self.joint_groups.items():
            print(f"\n{group_name}")
            print(f"Function: {group_info['function']}")
            print(f"Pattern: {group_info['pattern']}")
            print(f"Priority: {group_info['priority']}/4")
            print("-" * 40)
            
            for joint_name, description, importance in group_info['joints']:
                print(f"  📍 {joint_name}")
                print(f"     Role: {description}")
                print(f"     Importance: {importance}")
                print()

    def show_walking_cycle(self):
        """แสดงรูปแบบการเดินของ joints หลัก"""
        
        print("\n" + "=" * 60)
        print("🚶 Walking Cycle Pattern")
        print("=" * 60)
        
        # จำลองการเดิน 1 cycle (2 วินาที)
        time_steps = np.linspace(0, 2, 100)  # 2 วินาที, 100 จุด
        
        walking_data = {}
        
        # จำลอง joint patterns สำหรับการเดิน
        for t in time_steps:
            # Left leg pattern (เริ่มด้วยขาซ้าย)
            left_phase = 2 * np.pi * t / 2.0  # 2 วินาทีต่อ cycle
            
            # Right leg pattern (เลื่อนเฟส π)
            right_phase = left_phase + np.pi
            
            walking_data[t] = {
                # Left leg - PRIMARY JOINTS
                "left_hip_pitch": -0.2 + 0.3 * np.sin(left_phase),      # ยกขาซ้าย
                "left_knee": 0.4 + 0.4 * np.maximum(0, np.sin(left_phase)),  # งอเข่าซ้าย
                "left_ankle_pitch": -0.2 + 0.1 * np.sin(left_phase),   # เท้าซ้าย
                
                # Right leg - PRIMARY JOINTS  
                "right_hip_pitch": -0.2 + 0.3 * np.sin(right_phase),   # ยกขาขวา
                "right_knee": 0.4 + 0.4 * np.maximum(0, np.sin(right_phase)), # งอเข่าขวา
                "right_ankle_pitch": -0.2 + 0.1 * np.sin(right_phase), # เท้าขวา
                
                # Balance joints
                "left_hip_roll": 0.05 * np.sin(left_phase + np.pi/4),  # สมดุลซ้าย
                "right_hip_roll": 0.05 * np.sin(right_phase + np.pi/4), # สมดุลขวา
                
                # Arms (swing opposite to legs)
                "left_arm": 0.2 * np.sin(right_phase),   # แขนซ้ายแกว่งตรงข้ามขาซ้าย
                "right_arm": 0.2 * np.sin(left_phase),  # แขนขวาแกว่งตรงข้ามขาขวา
            }
        
        # แสดงตัวอย่างค่าที่สำคัญ
        print("📊 Key Joint Values During Walking:")
        print("Time | L_Hip | L_Knee | L_Ankle | R_Hip | R_Knee | R_Ankle")
        print("-" * 65)
        
        sample_times = [0.0, 0.5, 1.0, 1.5, 2.0]
        for t in sample_times:
            data = walking_data[t]
            print(f"{t:4.1f} | {data['left_hip_pitch']:5.2f} | {data['left_knee']:6.2f} | "
                  f"{data['left_ankle_pitch']:7.2f} | {data['right_hip_pitch']:5.2f} | "
                  f"{data['right_knee']:6.2f} | {data['right_ankle_pitch']:7.2f}")
        
        print("\n💡 Pattern Analysis:")
        print("   • Left/Right legs alternate (180° phase difference)")
        print("   • Hip pitch: negative = lift leg, positive = lower leg")
        print("   • Knee: higher values = more bent")
        print("   • Ankle: adjusts for ground contact")

    def show_critical_measurements(self):
        """แสดงค่าที่ต้องวัดและดูสำหรับการเดิน"""
        
        print("\n" + "=" * 60)
        print("📏 Critical Measurements for Walking")
        print("=" * 60)
        
        print("\n🎯 PRIMARY MONITORING JOINTS (ต้องดูทุกครั้ง):")
        critical_joints = [
            ("left_hip_pitch_joint", "ซ้าย ยกขา", [-0.5, 0.3], [-3.0, 3.0]),
            ("left_knee_joint", "ซ้าย งอเข่า", [0.0, 1.2], [-4.0, 4.0]), 
            ("left_ankle_pitch_joint", "ซ้าย เท้า", [-0.4, 0.2], [-2.0, 2.0]),
            ("right_hip_pitch_joint", "ขวา ยกขา", [-0.5, 0.3], [-3.0, 3.0]),
            ("right_knee_joint", "ขวา งอเข่า", [0.0, 1.2], [-4.0, 4.0]),
            ("right_ankle_pitch_joint", "ขวา เท้า", [-0.4, 0.2], [-2.0, 2.0]),
        ]
        
        for joint, desc, pos_range, vel_range in critical_joints:
            print(f"\n📍 {joint}")
            print(f"   Description: {desc}")
            print(f"   Position range: {pos_range[0]:.1f} to {pos_range[1]:.1f} rad")
            print(f"   Velocity range: {vel_range[0]:.1f} to {vel_range[1]:.1f} rad/s")
            print(f"   🚨 Alert if outside range!")
        
        print("\n⚖️ SECONDARY MONITORING JOINTS (สำหรับสมดุล):")
        balance_joints = [
            ("left_hip_roll_joint", "สมดุลซ้าย", [-0.2, 0.2]),
            ("right_hip_roll_joint", "สมดุลขวา", [-0.2, 0.2]),
            ("torso_joint", "หมุนลำตัว", [-0.3, 0.3]),
        ]
        
        for joint, desc, pos_range in balance_joints:
            print(f"   📍 {joint}: {desc} ({pos_range[0]:.1f} to {pos_range[1]:.1f} rad)")

    def detect_walking_problems(self):
        """แสดงวิธีตรวจหาปัญหาการเดิน"""
        
        print("\n" + "=" * 60)
        print("🚨 Walking Problem Detection")
        print("=" * 60)
        
        problems = {
            "🔴 CRITICAL PROBLEMS": [
                {
                    "symptom": "Hip pitch stuck at 0",
                    "cause": "ไม่ยกขา",
                    "check": "left/right_hip_pitch_joint position",
                    "normal_range": "[-0.3, 0.2] rad",
                    "action": "ตรวจสอบ hip motor"
                },
                {
                    "symptom": "Knee always straight (near 0)",
                    "cause": "ไม่งอเข่า",
                    "check": "left/right_knee_joint position", 
                    "normal_range": "[0.1, 0.8] rad while walking",
                    "action": "ตรวจสอบ knee motor"
                },
                {
                    "symptom": "No alternating pattern",
                    "cause": "ขาไม่สลับกัน",
                    "check": "Compare left vs right hip_pitch phases",
                    "normal_range": "180° phase difference",
                    "action": "ตรวจสอบ timing/coordination"
                }
            ],
            
            "🟡 WARNING SIGNS": [
                {
                    "symptom": "Excessive hip roll (>0.3 rad)",
                    "cause": "เสียสมดุล",
                    "check": "left/right_hip_roll_joint position",
                    "normal_range": "[-0.2, 0.2] rad",
                    "action": "ปรับ balance algorithm"
                },
                {
                    "symptom": "High joint velocities (>5 rad/s)",
                    "cause": "เคลื่อนไหวเร็วเกินไป",
                    "check": "All joint velocities",
                    "normal_range": "[-4, 4] rad/s",
                    "action": "ลด action scale หรือ control gains"
                }
            ]
        }
        
        for category, issues in problems.items():
            print(f"\n{category}:")
            for i, issue in enumerate(issues, 1):
                print(f"\n{i}. {issue['symptom']}")
                print(f"   Cause: {issue['cause']}")
                print(f"   Check: {issue['check']}")
                print(f"   Normal: {issue['normal_range']}")
                print(f"   Action: {issue['action']}")

    def generate_monitoring_code(self):
        """สร้างโค้ดตรวจสอบการเดิน"""
        
        print("\n" + "=" * 60)
        print("💻 Walking Monitoring Code")
        print("=" * 60)
        
        code = '''
def monitor_walking_joints(joint_positions, joint_velocities):
    """
    ตรวจสอบ joints ที่สำคัญสำหรับการเดิน
    
    Args:
        joint_positions: array of 37 joint positions (radians)
        joint_velocities: array of 37 joint velocities (rad/s)
    
    Returns:
        dict: walking status and alerts
    """
    
    # Joint indices (ต้องเช็คตาม joint order ของ robot)
    joints = {
        'left_hip_pitch': 2,    # Index ของ left_hip_pitch_joint
        'left_knee': 3,         # Index ของ left_knee_joint  
        'left_ankle_pitch': 9,  # Index ของ left_ankle_pitch_joint
        'right_hip_pitch': 6,   # Index ของ right_hip_pitch_joint
        'right_knee': 7,        # Index ของ right_knee_joint
        'right_ankle_pitch': 11 # Index ของ right_ankle_pitch_joint
    }
    
    # Normal ranges สำหรับการเดิน
    normal_ranges = {
        'hip_pitch': {'pos': [-0.5, 0.3], 'vel': [-3.0, 3.0]},
        'knee': {'pos': [0.0, 1.2], 'vel': [-4.0, 4.0]},
        'ankle_pitch': {'pos': [-0.4, 0.2], 'vel': [-2.0, 2.0]}
    }
    
    alerts = []
    status = {"walking": True, "balance": True}
    
    # 1. ตรวจสอบ PRIMARY WALKING JOINTS
    for side in ['left', 'right']:
        for joint_type in ['hip_pitch', 'knee', 'ankle_pitch']:
            joint_name = f"{side}_{joint_type}"
            if joint_name in joints:
                idx = joints[joint_name]
                pos = joint_positions[idx]
                vel = joint_velocities[idx]
                
                # ตรวจสอบ position range
                pos_range = normal_ranges[joint_type]['pos']
                if not (pos_range[0] <= pos <= pos_range[1]):
                    alerts.append(f"🔴 {joint_name} position out of range: {pos:.3f}")
                    status["walking"] = False
                
                # ตรวจสอบ velocity range  
                vel_range = normal_ranges[joint_type]['vel']
                if not (vel_range[0] <= vel <= vel_range[1]):
                    alerts.append(f"🟡 {joint_name} velocity too high: {vel:.3f}")
    
    # 2. ตรวจสอบ LEG COORDINATION
    left_hip = joint_positions[joints['left_hip_pitch']]
    right_hip = joint_positions[joints['right_hip_pitch']]
    
    # ขาควรเคลื่อนไหวแบบตรงข้าม
    if abs(left_hip - right_hip) < 0.1:  # ขาเคลื่อนไหวเหมือนกันเกินไป
        alerts.append("🔴 Legs not alternating properly")
        status["walking"] = False
    
    # 3. ตรวจสอบ STABILITY
    if len(alerts) > 5:  # มีปัญหาเยอะเกินไป
        status["balance"] = False
        alerts.append("🚨 CRITICAL: Multiple joint problems detected")
    
    return {
        "status": status,
        "alerts": alerts,
        "joint_health": len(alerts) == 0
    }

# ตัวอย่างการใช้งาน
joint_positions = get_current_joint_positions()  # 37 values
joint_velocities = get_current_joint_velocities()  # 37 values

walking_status = monitor_walking_joints(joint_positions, joint_velocities)

if not walking_status["joint_health"]:
    print("🚨 Walking problems detected:")
    for alert in walking_status["alerts"]:
        print(f"   {alert}")
        '''
        
        print(code)

def main():
    """Main analysis function"""
    analyzer = G1WalkingAnalysis()
    
    analyzer.analyze_walking_joints()
    analyzer.show_walking_cycle()
    analyzer.show_critical_measurements()
    analyzer.detect_walking_problems()
    analyzer.generate_monitoring_code()
    
    print("\n" + "=" * 60)
    print("🎯 SUMMARY: Joints เดิน Monitor ที่สำคัญ")
    print("=" * 60)
    print("1. 🔥 MUST MONITOR (6 joints):")
    print("   • left/right_hip_pitch_joint - ยกขา")
    print("   • left/right_knee_joint - งอเข่า")  
    print("   • left/right_ankle_pitch_joint - เท้า")
    print()
    print("2. ⚖️ BALANCE CHECK (3 joints):")
    print("   • left/right_hip_roll_joint - สมดุลข้าง")
    print("   • torso_joint - หมุนลำตัว")
    print()
    print("3. 📊 KEY METRICS:")
    print("   • Position ranges: ดูว่าอยู่ในช่วงปกติไหม")
    print("   • Velocity limits: ดูว่าเคลื่อนไหวเร็วเกินไปไหม")
    print("   • Phase coordination: ดูว่าขาสลับกันไหม")
    print()
    print("💡 TIP: Focus ที่ 6 joints หลักก่อน แล้วค่อยดู balance!")

if __name__ == "__main__":
    main()