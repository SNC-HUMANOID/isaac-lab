# 📡 Sensor Data มาจากไหน?

## 🤖 G1 Robot Sensors และแหล่งข้อมูล

### **📊 ภาพรวม: ข้อมูล 123 ค่า มาจากไหน**

```python
sensor_data = [
    # 🏃‍♂️ Base Motion (6 ค่า) - จาก IMU
    velocity_x, velocity_y, velocity_z,      # Linear velocity
    rotation_x, rotation_y, rotation_z,     # Angular velocity
    
    # 🌍 Gravity (3 ค่า) - จาก IMU  
    gravity_x, gravity_y, gravity_z,        # Projected gravity
    
    # 🎮 Commands (3 ค่า) - จาก User/Controller
    want_velocity_x,                        # User command
    want_velocity_y,                        # User command  
    want_rotation_z,                        # User command
    
    # 🦴 Joint States (74 ค่า) - จาก Joint Encoders
    joint_positions × 37,                   # Current joint angles
    joint_velocities × 37,                  # Current joint speeds
    
    # 🧠 Memory (37 ค่า) - จาก AI System
    previous_actions × 37,                  # Last AI commands
]
# รวม: 6 + 3 + 3 + 37 + 37 + 37 = 123 ค่า
```

---

## 📡 แหล่งข้อมูลแต่ละประเภท

### **1. 🏃‍♂️ Base Motion Data (6 ค่า) - จาก IMU**

#### **🔌 Hardware: IMU (Inertial Measurement Unit)**
```python
# IMU ติดตั้งที่ลำตัวหุ่นยนต์
imu_data = robot.get_imu_data()

# Linear velocity (3 ค่า) - ความเร็วการเคลื่อนที่
base_lin_vel = [
    imu_data.linear_velocity.x,    # ไปข้างหน้า/หลัง (m/s)
    imu_data.linear_velocity.y,    # ไปข้างซ้าย/ขวา (m/s)  
    imu_data.linear_velocity.z,    # ขึ้น/ลง (m/s)
]

# Angular velocity (3 ค่า) - ความเร็วการหมุน
base_ang_vel = [
    imu_data.angular_velocity.x,   # Roll - เอียงซ้าย/ขวา (rad/s)
    imu_data.angular_velocity.y,   # Pitch - ก้มหงาย (rad/s)
    imu_data.angular_velocity.z,   # Yaw - หมุนซ้าย/ขวา (rad/s)
]
```

#### **📍 ตำแหน่ง IMU บน G1:**
- ติดตั้งที่ **torso (ลำตัว)** ของ robot
- วัดการเคลื่อนไหวของลำตัวเทียบกับพื้นโลก
- อัพเดตที่ความถี่สูง (1000+ Hz)

---

### **2. 🌍 Gravity Data (3 ค่า) - จาก IMU**

#### **🔌 Hardware: IMU Accelerometer**
```python
# Gravity vector ในระบบพิกัดของ robot
imu_data = robot.get_imu_data()
robot_orientation = robot.get_orientation()  # quaternion

# แปลง gravity จากพิกัดโลก → พิกัด robot
world_gravity = [0, 0, -9.81]  # แรงโน้มถ่วงโลก
projected_gravity = transform_to_robot_frame(world_gravity, robot_orientation)

gravity_data = [
    projected_gravity[0],    # แรงโน้มถ่วงในแกน X ของ robot
    projected_gravity[1],    # แรงโน้มถ่วงในแกน Y ของ robot  
    projected_gravity[2],    # แรงโน้มถ่วงในแกน Z ของ robot
]
# เมื่อ robot ยืนตรง: [0, 0, -1]
# เมื่อ robot เอียง: มีค่า X, Y ≠ 0
```

#### **💡 ทำไมต้องมี Projected Gravity?**
- บอก AI ว่า robot เอียงหรือเปล่า
- ช่วย AI รักษาสมดุล
- เป็นข้อมูล reference ที่สำคัญ

---

### **3. 🎮 Command Data (3 ค่า) - จาก User/Controller**

#### **🔌 Input Sources:**
```python
# วิธีที่ 1: Joystick/Gamepad
gamepad = get_gamepad_input()
commands = [
    gamepad.left_stick_y * max_velocity,    # เดินไปข้างหน้า/หลัง
    gamepad.left_stick_x * max_velocity,    # เดินไปข้าง
    gamepad.right_stick_x * max_angular_vel # หมุนซ้าย/ขวา
]

# วิธีที่ 2: Keyboard
keyboard = get_keyboard_input()
commands = [
    (W_key - S_key) * 1.0,     # W=เดินหน้า, S=ถอยหลัง
    (A_key - D_key) * 0.5,     # A=เดินซ้าย, D=เดินขวา  
    (Q_key - E_key) * 1.0,     # Q=หมุนซ้าย, E=หมุนขวา
]

# วิธีที่ 3: ROS Topics (สำหรับ autonomous)
commands = rospy.get_cmd_vel()  # จาก navigation stack

# วิธีที่ 4: Pre-programmed Pattern
time_now = get_current_time()
commands = [
    0.5 * sin(time_now),       # เดินเป็นคลื่น
    0.0,                       # ไม่เดินข้าง
    0.3 * cos(time_now)        # หมุนเป็นคลื่น
]
```

---

### **4. 🦴 Joint State Data (74 ค่า) - จาก Joint Encoders**

#### **🔌 Hardware: Joint Encoders**
```python
# Joint Positions (37 ค่า) - มุมปัจจุบันของแต่ละข้อต่อ
joint_positions = []
for joint_name in joint_names:  # 37 joints
    encoder_value = robot.get_joint_encoder(joint_name)
    angle_radians = encoder_to_radians(encoder_value)
    
    # เทียบกับตำแหน่งเริ่มต้น (relative position)
    default_angle = default_joint_positions[joint_name]
    relative_angle = angle_radians - default_angle
    
    joint_positions.append(relative_angle)

# Joint Velocities (37 ค่า) - ความเร็วปัจจุบันของแต่ละข้อต่อ  
joint_velocities = []
for joint_name in joint_names:  # 37 joints
    velocity_rad_per_sec = robot.get_joint_velocity(joint_name)
    joint_velocities.append(velocity_rad_per_sec)
```

#### **📍 Joint Encoder Locations:**
```python
joint_locations = {
    # ขา (9 joints)
    'left_hip_yaw_joint': 'ข้อต่อสะโพกซ้าย (หมุน)',
    'left_hip_roll_joint': 'ข้อต่อสะโพกซ้าย (เอียง)', 
    'left_hip_pitch_joint': 'ข้อต่อสะโพกซ้าย (ก้มเงย)',
    'left_knee_joint': 'ข้อเข่าซ้าย',
    # ... และอื่นๆ รวม 37 joints
    
    # เท้า (4 joints)  
    'left_ankle_pitch_joint': 'ข้อเท้าซ้าย (ก้มเงย)',
    'left_ankle_roll_joint': 'ข้อเท้าซ้าย (เอียง)',
    
    # แขน (24 joints)
    'left_shoulder_pitch_joint': 'ข้อไหล่ซ้าย (ก้มเงย)',
    'left_elbow_pitch_joint': 'ข้อศอกซ้าย',
    # ... fingers และอื่นๆ
}
```

---

### **5. 🧠 Memory Data (37 ค่า) - จาก AI System**

#### **🔌 Source: AI Control Loop**
```python
# Previous Actions - คำสั่งที่ AI ส่งไปรอบก่อน
class AIController:
    def __init__(self):
        self.previous_actions = np.zeros(37)  # เริ่มต้น = 0
    
    def get_actions(self, observations):
        # ประมวลผล AI
        current_actions = self.policy_net(observations)
        
        # เก็บสำหรับรอบถัดไป
        self.previous_actions = current_actions.copy()
        
        return current_actions
    
    def get_observations(self):
        # รวมข้อมูลทั้งหมด
        obs = np.concatenate([
            base_motion,        # 6 ค่า
            gravity,           # 3 ค่า  
            commands,          # 3 ค่า
            joint_positions,   # 37 ค่า
            joint_velocities,  # 37 ค่า
            self.previous_actions  # 37 ค่า ← จาก memory
        ])
        return obs
```

---

## 🔄 Data Flow Architecture

### **📊 Real-time Data Pipeline:**

```python
def control_loop():
    """Control loop ที่ 50 Hz"""
    
    while True:
        start_time = time.time()
        G1 Model เป็น POSITION CONTROL ไม่ใช่ Kinematic/Reward

  ❌ ไม่ใช่ Kinematic/Reward เพราะ:

  # ❌ ไม่ใช่ Kinematic (การคำนวณเรขาคณิต)
  # Kinematic = คำนวณตำแหน่งจากมุม หรือ มุมจากตำแหน่ง
  forward_kinematics(joint_angles) → end_effector_position
  inverse_kinematics(target_position) → joint_angles

  # ❌ ไม่ใช่ Reward (ค่าคะแนน)  
  # Reward = ค่าบอกว่า robot ทำได้ดีหรือไม่
  reward = calculate_reward(robot_state)  # ได้คะแนนเท่าไหร่

  ✅ เป็น POSITION CONTROL:

  # ✅ AI ส่งคำสั่ง POSITION TARGET
  target_positions = ai_model(sensor_data)  # 37 มุมเป้าหมาย

  # ส่งไปยัง PD Controller
  for i in range(37):
      motor[i].set_target_position(target_positions[i])
      # PD Controller จะหาวิธีขับมอเตอร์ให้ไปถึงมุมนั้น

  🔧 กระบวนการควบคุมจริง:

  1. AI → Position Targets

  outputs = [
      -0.185,  # left_hip_pitch target
       0.104,  # left_knee target  
       0.156,  # right_hip_pitch target
      # ... 37 targets total
  ]

  2. PD Controller → Motor Torques

  for joint in range(37):
      current_angle = robot.get_joint_angle(joint)
      target_angle = outputs[joint]

      # PD Control
      error = target_angle - current_angle
      torque = Kp * error + Kd * velocity_error

      robot.set_joint_torque(joint, torque)

  🎯 สรุป:

  - Input: 123 sensor values
  - AI Output: 37 position targets (radians)
  - Control Type: Position Control with PD controllers
  - Not: Kinematic calculation or Reward signal

  AI บอกว่า "ขยับข้อต่อไปมุมนี้", PD Controller หาวิธีขยับให้ถึง! 🎯🤖

        # 1. อ่านข้อมูลจากฮาร์ดแวร์
        imu_data = robot.imu.read()                    # IMU sensors
        joint_states = robot.joint_encoders.read_all() # Joint encoders  
        user_commands = gamepad.read()                 # User input
        
        # 2. ประมวลผลข้อมูล
        base_motion = process_imu_data(imu_data)       # 6 ค่า
        gravity = calculate_projected_gravity(imu_data) # 3 ค่า
        commands = process_user_input(user_commands)   # 3 ค่า  
        joint_pos = process_joint_positions(joint_states) # 37 ค่า
        joint_vel = process_joint_velocities(joint_states) # 37 ค่า
        prev_actions = ai_controller.get_previous_actions() # 37 ค่า
        
        # 3. รวมข้อมูลทั้งหมด
        observations = np.concatenate([
            base_motion, gravity, commands,
            joint_pos, joint_vel, prev_actions
        ])  # 123 ค่า
        
        # 4. ส่งไป AI
        actions = ai_controller.predict(observations)  # 37 ค่า
        G1 Model เป็น POSITION CONTROL ไม่ใช่ Kinematic/Reward

  ❌ ไม่ใช่ Kinematic/Reward เพราะ:

  # ❌ ไม่ใช่ Kinematic (การคำนวณเรขาคณิต)
  # Kinematic = คำนวณตำแหน่งจากมุม หรือ มุมจากตำแหน่ง
  forward_kinematics(joint_angles) → end_effector_position
  inverse_kinematics(target_position) → joint_angles

  # ❌ ไม่ใช่ Reward (ค่าคะแนน)  
  # Reward = ค่าบอกว่า robot ทำได้ดีหรือไม่
  reward = calculate_reward(robot_state)  # ได้คะแนนเท่าไหร่

  ✅ เป็น POSITION CONTROL:

  # ✅ AI ส่งคำสั่ง POSITION TARGET
  target_positions = ai_model(sensor_data)  # 37 มุมเป้าหมาย

  # ส่งไปยัง PD Controller
  for i in range(37):
      motor[i].set_target_position(target_positions[i])
      # PD Controller จะหาวิธีขับมอเตอร์ให้ไปถึงมุมนั้น

  🔧 กระบวนการควบคุมจริง:

  1. AI → Position Targets

  outputs = [
      -0.185,  # left_hip_pitch target
       0.104,  # left_knee target  
       0.156,  # right_hip_pitch target
      # ... 37 targets total
  ]

  2. PD Controller → Motor Torques

  for joint in range(37):
      current_angle = robot.get_joint_angle(joint)
      target_angle = outputs[joint]

      # PD Control
      error = target_angle - current_angle
      torque = Kp * error + Kd * velocity_error

      robot.set_joint_torque(joint, torque)

  🎯 สรุป:

  - Input: 123 sensor values
  - AI Output: 37 position targets (radians)
  - Control Type: Position Control with PD controllers
  - Not: Kinematic calculation or Reward signal

  AI บอกว่า "ขยับข้อต่อไปมุมนี้", PD Controller หาวิธีขยับให้ถึง! 🎯🤖

        # 5. ส่งไป robot
        robot.set_joint_targets(actions)
        
        # 6. รอให้ครบ 20ms (50 Hz)
        elapsed = time.time() - start_time
        time.sleep(max(0, 0.02 - elapsed))
```

---

## 🛠️ Hardware Requirements

### **📡 ฮาร์ดแวร์ที่ต้องมี:**

| Sensor Type | จำนวน | Purpose | Update Rate |
|-------------|-------|---------|-------------|
| **IMU** | 1 ตัว | Base motion + gravity | 1000+ Hz |
| **Joint Encoders** | 37 ตัว | Joint positions/velocities | 1000+ Hz |
| **User Interface** | 1 ตัว | Commands (gamepad/keyboard) | 50+ Hz |
| **Main Computer** | 1 ตัว | AI processing + control | 50 Hz |

### **💻 Software Stack:**
```python
# ตัวอย่าง software architecture
class G1RobotInterface:
    def __init__(self):
        self.imu = IMU_Driver("/dev/imu0")
        self.joint_encoders = [JointEncoder(i) for i in range(37)]
        self.gamepad = GamepadDriver("/dev/input/js0")
        self.ai_model = load_trained_model("model_300.pt")
    
    def get_sensor_data(self):
        """รวบรวมข้อมูลจาก sensors ทั้งหมด"""
        return {
            'imu': self.imu.read(),
            'joints': [enc.read() for enc in self.joint_encoders],
            'commands': self.gamepad.read()
        }
```

---

## 🎯 สรุป: Sensor Data มาจากไหน

### **📡 แหล่งข้อมูลหลัก:**

1. **IMU (1 ตัว)** → Base motion (6) + Gravity (3) = **9 ค่า**
2. **Joint Encoders (37 ตัว)** → Positions (37) + Velocities (37) = **74 ค่า**  
3. **User Input** → Commands = **3 ค่า**
4. **AI Memory** → Previous actions = **37 ค่า**

**รวม: 9 + 74 + 3 + 37 = 123 ค่า**

### **⚡ Key Points:**
- **Real-time**: ข้อมูลต้องอัพเดตทุก 20ms
- **Synchronized**: sensors ทั้งหมดต้อง sync กัน
- **Processed**: ข้อมูลต้องผ่านการประมวลผลก่อนส่งให้ AI
- **Hardware dependent**: ต้องมี IMU + Joint encoders + User interface

**ตอนนี้คุณรู้แล้วว่า sensor_data 123 ค่า มาจากไหนทุกค่า!** 🤖📡