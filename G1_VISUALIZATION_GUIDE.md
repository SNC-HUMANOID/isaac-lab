# 🤖 G1 Model Visualization Guide

แนวทางการใช้งานโค้ดสำหรับดูการทำงานของโมเดล AI ที่เทรนแล้วใน Isaac Sim

## 📁 ไฟล์ที่สร้างขึ้น

1. **`visualize_g1_model.py`** - โค้ดหลักสำหรับแสดงผล
2. **`run_g1_viz.sh`** - สคริปต์เรียกใช้งานแบบง่าย
3. **`G1_VISUALIZATION_GUIDE.md`** - คู่มือนี้

## 🚀 วิธีใช้งาน

### วิธีที่ 1: ใช้สคริปต์ง่าย (แนะนำ)
```bash
# รันแบบธรรมดา (1 robot)
./run_g1_viz.sh

# รันกับ 4 robots
./run_g1_viz.sh --num_envs 4

# รันโดยไม่เปิด GUI
./run_g1_viz.sh --headless

# ใช้โมเดลอื่น
./run_g1_viz.sh --model logs/rsl_rl/g1_flat/2025-07-29_15-27-26/model_200.pt

# ดูตัวเลือกทั้งหมด
./run_g1_viz.sh --help
```

### วิธีที่ 2: เรียกโดยตรง
```bash
# รันผ่าน Isaac Lab
./isaaclab.sh -p visualize_g1_model.py \
  --model_path logs/rsl_rl/g1_flat/2025-07-29_15-27-26/model_300.pt \
  --num_envs 1
```

## 🎮 การควบคุมในโปรแกรม

โปรแกรมจะทำงานอัตโนมัติและแสดง:

1. **0-5s**: เดินไปข้างหน้า (vx=0.5)
2. **5-10s**: เลี้ยวซ้าย (vx=0.3, wz=0.5) 
3. **10-15s**: เดินถอยหลัง (vx=-0.3)
4. **15-20s**: เดินไปข้าง (vy=-0.3)
5. **20s+**: หยุด

## 📊 ข้อมูลที่แสดง

### Console Output
```
🤖 G1 Model Visualization Ready!
   • Model: logs/rsl_rl/g1_flat/2025-07-29_15-27-26/model_300.pt
   • Environments: 1
   • Joints: 37
   • Action dim: 29

✅ Model loaded successfully!
   • Training iterations: 300
   • Model parameters: 165,000
   • Network: 235 → [256,128,128] → 29

Step    0 | Time:   0.00s | Cmd: [0.50, 0.00, 0.00]
Step   10 | Time:   0.20s | Cmd: [0.50, 0.00, 0.00]
...
```

### ใน Isaac Sim
- **หุ่นยนต์ G1** เดินตามคำสั่งความเร็ว
- **Joint positions** ที่โมเดล AI สั่งแต่ละ frame
- **Contact forces** ที่เท้าของหุ่นยนต์

## 🔧 การปรับแต่ง

### เปลี่ยนคำสั่งความเร็ว
แก้ไขฟังก์ชัน `update_commands()` ใน `visualize_g1_model.py`:

```python
def update_commands(self):
    time = self.step_count * self.dt
    
    # ตัวอย่าง: วิ่งวนรอบ
    self.velocity_command = torch.tensor([
        0.8 * math.sin(time * 0.5),  # vx
        0.0,                         # vy  
        0.3 * math.cos(time * 0.5)   # wz
    ], device=self.robot.device)
```

### เปลี่ยนพารามิเตอร์การควบคุม
```python
# ในคลาส __init__()
self.action_scale = 0.3  # ลดการเคลื่อนไหว
self.control_freq = 100  # เพิ่มความถี่ควบคุม
```

## 🔍 สิ่งที่คุณจะเห็น

### 1. **Model Output → Joint Positions**
โมเดลจะส่งออก array ของตำแหน่ง joint:
```python
# ตัวอย่าง output จากโมเดล
raw_actions = [-0.1, 0.3, -0.2, 0.5, ...]  # 29 values

# หลังจาก scaling + offset  
final_positions = [
    0.42,   # left_knee_joint
    -0.25,  # left_hip_pitch_joint
    0.18,   # left_ankle_pitch_joint
    ...
]
```

### 2. **Real-time Control Loop**
```
Observation → Neural Network → Actions → Motor Commands → Robot Movement
     ↑                                                            ↓
     ←←←←←←←←←←←←← Sensor Feedback ←←←←←←←←←←←←←←←←←←←←←←←←←←←
```

### 3. **Joint Groups แยกตามการใช้งาน**
- **Legs** (9 joints): ขับเคลื่อนการเดิน
- **Feet** (4 joints): ปรับสมดุลและการเหยียบ  
- **Arms** (14 joints): ช่วยสมดุลและท่าทาง
- **Torso** (1 joint): ควบคุมลำตัว

## ⚠️ หมายเหตุ

1. **ต้องมี Isaac Sim**: โค้ดต้องการ Isaac Sim และ Isaac Lab
2. **Model path**: ตรวจสอบ path ของโมเดลให้ถูกต้อง
3. **GPU Memory**: การรันหลาย environments ใช้ memory เยอะ
4. **Performance**: ใช้ `--headless` ถ้าต้องการความเร็วสูง

## 🎯 จุดประสงค์

โค้ดนี้แสดงให้เห็น:
- **วิธีการโหลดโมเดล** ที่เทรนแล้ว
- **การแปลง observations** เป็น input สำหรับ neural network
- **การประมวลผล actions** และส่งไปยัง motors
- **การควบคุม real-time** ของหุ่นยนต์

นี่คือพื้นฐานสำหรับ**นำไปใช้กับหุ่นยนต์จริง**! 🤖