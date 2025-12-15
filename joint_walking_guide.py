#!/usr/bin/env python3
"""
G1 Walking Joints Guide
คู่มือ Joint Encoders ที่สำคัญสำหรับการเดิน
"""

import numpy as np
import math

def show_walking_joints():
    """แสดง joints ที่สำคัญสำหรับการเดิน"""
    
    print("🤖 G1 Walking Joints Guide")
    print("=" * 50)
    
    print("\n🔥 PRIMARY WALKING JOINTS (6 ตัว - สำคัญที่สุด)")
    print("-" * 50)
    
    primary_joints = [
        {
            "name": "left_hip_pitch_joint",
            "index": 2,  # ตำแหน่งใน array 37 ตัว
            "function": "ยกขาซ้าย ขึ้น/ลง",
            "walking_range": "[-0.3, 0.2] rad",
            "standing_default": "-0.2 rad",
            "pattern": "สูงขึ้น เมื่อยกขา, ต่ำลง เมื่อวางขา"
        },
        {
            "name": "left_knee_joint", 
            "index": 3,
            "function": "งอเข่าซ้าย",
            "walking_range": "[0.1, 0.8] rad",
            "standing_default": "0.42 rad",
            "pattern": "งอมากขึ้น เมื่อยกขา, งอน้อยลง เมื่อวางขา"
        },
        {
            "name": "left_ankle_pitch_joint",
            "index": 9,
            "function": "เท้าซ้าย ขึ้น/ลง",
            "walking_range": "[-0.4, 0.1] rad", 
            "standing_default": "-0.23 rad",
            "pattern": "ปรับเพื่อสัมผัสพื้นให้นุ่มนวล"
        },
        {
            "name": "right_hip_pitch_joint",
            "index": 6,
            "function": "ยกขาขวา ขึ้น/ลง", 
            "walking_range": "[-0.3, 0.2] rad",
            "standing_default": "-0.2 rad",
            "pattern": "ตรงข้ามกับขาซ้าย (เลื่อนเฟส 180°)"
        },
        {
            "name": "right_knee_joint",
            "index": 7,
            "function": "งอเข่าขวา",
            "walking_range": "[0.1, 0.8] rad",
            "standing_default": "0.42 rad", 
            "pattern": "ตรงข้ามกับเข่าซ้าย (เลื่อนเฟส 180°)"
        },
        {
            "name": "right_ankle_pitch_joint",
            "index": 11,
            "function": "เท้าขวา ขึ้น/ลง",
            "walking_range": "[-0.4, 0.1] rad",
            "standing_default": "-0.23 rad",
            "pattern": "ตรงข้ามกับเท้าซ้าย (เลื่อนเฟส 180°)"
        }
    ]
    
    for i, joint in enumerate(primary_joints, 1):
        print(f"{i}. 📍 {joint['name']}")
        print(f"   Array Index: {joint['index']}")
        print(f"   Function: {joint['function']}")
        print(f"   Walking Range: {joint['walking_range']}") 
        print(f"   Default: {joint['standing_default']}")
        print(f"   Pattern: {joint['pattern']}")
        print()

def show_balance_joints():
    """แสดง joints ที่สำคัญสำหรับสมดุล"""
    
    print("⚖️ BALANCE JOINTS (5 ตัว - สำคัญรอง)")
    print("-" * 50)
    
    balance_joints = [
        {
            "name": "left_hip_roll_joint",
            "index": 1,
            "function": "เอียงสะโพกซ้าย (ซ้าย/ขวา)",
            "normal_range": "[-0.15, 0.15] rad",
            "alert_if": "> 0.3 rad (เสียสมดุล)"
        },
        {
            "name": "right_hip_roll_joint", 
            "index": 5,
            "function": "เอียงสะโพกขวา (ซ้าย/ขวา)",
            "normal_range": "[-0.15, 0.15] rad",
            "alert_if": "> 0.3 rad (เสียสมดุล)"
        },
        {
            "name": "left_ankle_roll_joint",
            "index": 10, 
            "function": "เอียงเท้าซ้าย (ซ้าย/ขวา)",
            "normal_range": "[-0.1, 0.1] rad",
            "alert_if": "> 0.2 rad (เสียสมดุล)"
        },
        {
            "name": "right_ankle_roll_joint",
            "index": 12,
            "function": "เอียงเท้าขวา (ซ้าย/ขวา)", 
            "normal_range": "[-0.1, 0.1] rad",
            "alert_if": "> 0.2 rad (เสียสมดุล)"
        },
        {
            "name": "torso_joint",
            "index": 8,
            "function": "หมุนลำตัว (ซ้าย/ขวา)",
            "normal_range": "[-0.2, 0.2] rad", 
            "alert_if": "> 0.4 rad (หมุนมากเกินไป)"
        }
    ]
    
    for i, joint in enumerate(balance_joints, 1):
        print(f"{i}. 📍 {joint['name']}")
        print(f"   Array Index: {joint['index']}")
        print(f"   Function: {joint['function']}")
        print(f"   Normal Range: {joint['normal_range']}")
        print(f"   Alert If: {joint['alert_if']}")
        print()

def show_walking_example():
    """แสดงตัวอย่างค่า joints ขณะเดิน"""
    
    print("🚶 Walking Example - Joint Values")
    print("-" * 50)
    
    # จำลองการเดิน 4 เฟส
    walking_phases = [
        {
            "phase": "Left Swing (ขาซ้ายยกขึ้น)",
            "time": "0.0s",
            "left_hip_pitch": 0.1,   # ยกขาซ้าย
            "left_knee": 0.8,        # งอเข่าซ้าย
            "left_ankle": -0.1,      # เท้าซ้ายขึ้น
            "right_hip_pitch": -0.3, # ขาขวาพยุง
            "right_knee": 0.2,       # เข่าขวาเกือบตรง
            "right_ankle": -0.3,     # เท้าขวาแนบพื้น
        },
        {
            "phase": "Left Contact (ขาซ้ายลง)",
            "time": "0.5s", 
            "left_hip_pitch": -0.1,  # ขาซ้ายลง
            "left_knee": 0.3,        # เข่าซ้ายเกือบตรง
            "left_ankle": -0.25,     # เท้าซ้ายแตะพื้น
            "right_hip_pitch": -0.2, # ขาขวาเริ่มยก
            "right_knee": 0.5,       # เข่าขวาเริ่มงอ
            "right_ankle": -0.2,     # เท้าขวาเริ่มขึ้น
        },
        {
            "phase": "Right Swing (ขาขวายกขึ้น)",
            "time": "1.0s",
            "left_hip_pitch": -0.3,  # ขาซ้ายพยุง
            "left_knee": 0.2,        # เข่าซ้ายเกือบตรง
            "left_ankle": -0.3,      # เท้าซ้ายแนบพื้น
            "right_hip_pitch": 0.1,  # ยกขาขวา
            "right_knee": 0.8,       # งอเข่าขวา
            "right_ankle": -0.1,     # เท้าขวาขึ้น
        },
        {
            "phase": "Right Contact (ขาขวาลง)",
            "time": "1.5s",
            "left_hip_pitch": -0.2,  # ขาซ้ายเริ่มยก
            "left_knee": 0.5,        # เข่าซ้ายเริ่มงอ
            "left_ankle": -0.2,      # เท้าซ้ายเริ่มขึ้น
            "right_hip_pitch": -0.1, # ขาขวาลง
            "right_knee": 0.3,       # เข่าขวาเกือบตรง
            "right_ankle": -0.25,    # เท้าขวาแตะพื้น
        }
    ]
    
    print("Phase | Time | L_Hip | L_Knee | L_Ankle | R_Hip | R_Knee | R_Ankle")
    print("-" * 70)
    
    for phase_data in walking_phases:
        phase = phase_data["phase"][:15] + "..." if len(phase_data["phase"]) > 15 else phase_data["phase"]
        print(f"{phase:<18} | {phase_data['time']} | "
              f"{phase_data['left_hip_pitch']:5.1f} | {phase_data['left_knee']:6.1f} | "
              f"{phase_data['left_ankle']:7.1f} | {phase_data['right_hip_pitch']:5.1f} | "
              f"{phase_data['right_knee']:6.1f} | {phase_data['right_ankle']:7.1f}")
    
    print("\n💡 Pattern Observations:")
    print("   • ขาซ้าย/ขวา เคลื่อนไหวตรงข้ามกัน (alternating)")
    print("   • Hip pitch: บวก = ยกขา, ลบ = วางขา/พยุง")  
    print("   • Knee: มากขึ้น = งอมาก (ขณะยกขา)")
    print("   • Ankle: ปรับเพื่อการสัมผัสพื้นที่นุ่มนวล")

def show_monitoring_code():
    """แสดงโค้ดตรวจสอบการเดิน"""
    
    print("\n💻 Monitoring Code Template")
    print("-" * 50)
    
    code = '''
def monitor_walking_joints(joint_positions, joint_velocities):
    """
    ตรวจสอบ joints ที่สำคัญสำหรับการเดิน
    
    Args:
        joint_positions: numpy array (37,) - ตำแหน่ง joints (radians)
        joint_velocities: numpy array (37,) - ความเร็ว joints (rad/s)
    
    Returns:
        dict: สถานะการเดินและ alerts
    """
    
    # 🔥 PRIMARY WALKING JOINTS (indices in 37-joint array)
    walking_joints = {
        'left_hip_pitch': 2,    'left_knee': 3,      'left_ankle_pitch': 9,
        'right_hip_pitch': 6,   'right_knee': 7,     'right_ankle_pitch': 11
    }
    
    # ⚖️ BALANCE JOINTS  
    balance_joints = {
        'left_hip_roll': 1,     'right_hip_roll': 5, 'torso': 8,
        'left_ankle_roll': 10,  'right_ankle_roll': 12
    }
    
    # Normal ranges for walking
    ranges = {
        'hip_pitch': {'pos': [-0.5, 0.3], 'vel': [-3.0, 3.0]},
        'knee': {'pos': [0.0, 1.2], 'vel': [-4.0, 4.0]}, 
        'ankle_pitch': {'pos': [-0.4, 0.2], 'vel': [-2.0, 2.0]},
        'hip_roll': {'pos': [-0.3, 0.3], 'vel': [-2.0, 2.0]},
        'ankle_roll': {'pos': [-0.2, 0.2], 'vel': [-1.5, 1.5]},
        'torso': {'pos': [-0.4, 0.4], 'vel': [-2.0, 2.0]}
    }
    
    alerts = []
    
    # 1. Check PRIMARY JOINTS
    for joint_name, idx in walking_joints.items():
        joint_type = joint_name.split('_')[-1] + ('_pitch' if 'pitch' in joint_name else '')
        pos = joint_positions[idx]
        vel = joint_velocities[idx]
        
        # Position check
        pos_range = ranges[joint_type]['pos']
        if not (pos_range[0] <= pos <= pos_range[1]):
            alerts.append(f"🔴 {joint_name} pos out of range: {pos:.3f}")
        
        # Velocity check 
        vel_range = ranges[joint_type]['vel']
        if not (vel_range[0] <= vel <= vel_range[1]):
            alerts.append(f"🟡 {joint_name} vel too high: {vel:.3f}")
    
    # 2. Check COORDINATION (ขาสลับกันไหม)
    left_hip = joint_positions[walking_joints['left_hip_pitch']]
    right_hip = joint_positions[walking_joints['right_hip_pitch']]
    
    if abs(left_hip - right_hip) < 0.05:  # เคลื่อนไหวเหมือนกันเกินไป
        alerts.append("🔴 Legs not alternating - check coordination")
    
    # 3. Check BALANCE
    for joint_name, idx in balance_joints.items():
        joint_type = 'hip_roll' if 'hip_roll' in joint_name else 'ankle_roll' if 'ankle_roll' in joint_name else 'torso'
        pos = joint_positions[idx]
        
        pos_range = ranges[joint_type]['pos'] 
        if not (pos_range[0] <= pos <= pos_range[1]):
            alerts.append(f"⚖️ {joint_name} balance issue: {pos:.3f}")
    
    # Status summary
    critical_alerts = [a for a in alerts if '🔴' in a]
    balance_alerts = [a for a in alerts if '⚖️' in a]
    
    return {
        'walking_ok': len(critical_alerts) == 0,
        'balance_ok': len(balance_alerts) == 0, 
        'total_alerts': len(alerts),
        'alerts': alerts
    }

# Example usage:
status = monitor_walking_joints(joint_positions, joint_velocities)
if not status['walking_ok']:
    print("🚨 Walking problems detected!")
    for alert in status['alerts']:
        print(f"   {alert}")
'''
    print(code)

def main():
    """Main function"""
    show_walking_joints()
    show_balance_joints() 
    show_walking_example()
    show_monitoring_code()
    
    print("\n" + "=" * 50)
    print("🎯 QUICK REFERENCE")
    print("=" * 50)
    
    print("🔥 ต้องดู 6 joints นี้เป็นหลัก:")
    print("   [2] left_hip_pitch   [3] left_knee        [9] left_ankle_pitch")
    print("   [6] right_hip_pitch  [7] right_knee      [11] right_ankle_pitch")
    
    print("\n⚖️ ตรวจสมดุล 5 joints นี้:")
    print("   [1] left_hip_roll    [5] right_hip_roll   [8] torso")
    print("  [10] left_ankle_roll [12] right_ankle_roll")
    
    print("\n📊 สิ่งที่ต้องเช็ค:")
    print("   1. Position values อยู่ในช่วงปกติไหม")
    print("   2. Velocity values ไม่เร็วเกินไปไหม") 
    print("   3. ขาซ้าย/ขวา เคลื่อนไหวสลับกันไหม")
    print("   4. Balance joints ไม่เอียงมากเกินไปไหม")
    
    print("\n💡 หาก joints เหล่านี้ทำงานปกติ = robot เดินได้!")

if __name__ == "__main__":
    main()