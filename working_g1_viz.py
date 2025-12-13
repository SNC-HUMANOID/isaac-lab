import torch
import numpy as np
import csv
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import open_stage, get_current_stage
from omni.isaac.core.utils.prims import create_prim
import torch.nn as nn

# สร้างโมเดล PyTorch (สมมติว่าเป็น MLP จาก rsl_rl)
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim=34, act_dim=12):  # ปรับตาม config
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# ตั้งค่าโลกจำลอง
world = World(stage_units_in_meters=1.0)

# โหลดไฟล์ USD
robot_path = "/home/sncbot/IsaacLab/assets/g1.usd"  # ปรับพาธให้ถูกต้อง
try:
    open_stage(robot_path)
except Exception as e:
    print(f"ข้อผิดพลาดในการโหลด USD: {e}")
    robot_path = "/Isaac/Robots/Unitree/Go1/unitree_go1.usd"  # ใช้ไฟล์ตัวอย่าง
    open_stage(robot_path)

# ตรวจสอบและสร้าง prim
stage = get_current_stage()
prim_path = "/World/G1"
if not stage.GetPrimAtPath(prim_path):
    create_prim(prim_path, "Xform")
    print(f"สร้าง prim ที่ {prim_path}")

# สร้างหุ่นยนต์
try:
    robot = world.scene.add(Robot(prim_path=prim_path, name="g1_robot"))
    world.reset()
    print("สร้างหุ่นยนต์สำเร็จ")
except Exception as e:
    print(f"ข้อผิดพลาดในการสร้างหุ่นยนต์: {e}")
    robot = None

# โหลดโมเดล RL
model_path = "/home/sncbot/IsaacLab/logs/rsl_rl/g1_flat/2025-07-29_15-27-26/model_300.pt"
try:
    model_data = torch.load(model_path, map_location=torch.device('cpu'))
    model = PolicyNetwork(obs_dim=34, act_dim=12)  # ปรับ obs_dim, act_dim
    if isinstance(model_data, dict):
        state_dict = model_data.get('state_dict', model_data)
        model.load_state_dict(state_dict)
        print("โหลด state dict สำเร็จ")
    else:
        model = model_data
        print("โหลดโมเดลทั้งตัวสำเร็จ")
    model.eval()
except Exception as e:
    print(f"ข้อผิดพลาดในการโหลดโมเดล: {e}")
    model = None

# คำนวณ kinematic reward
def calculate_kinematic_reward(obs, action):
    lin_vel_xy = obs[0:2]  # ความเร็ว XY
    target_vel_xy = np.array([0.5, 0.0])  # ปรับตาม config
    track_reward = np.exp(-np.sum((lin_vel_xy - target_vel_xy) ** 2))
    joint_deviation = np.mean(action ** 2)
    return track_reward - 0.1 * joint_deviation

# รันการจำลองและบันทึกมุม joint
joint_data = []
if model and robot:
    for step in range(100):  # จำลอง 100 สเต็ป
        obs = world.get_observations()
        if obs is None or len(obs) == 0:
            print("ไม่พบ observation")
            break
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action = model(obs_tensor).numpy()

        robot.apply_action(action)
        world.step(render=True)

        joint_positions = robot.get_joint_positions()
        if joint_positions is not None:
            reward = calculate_kinematic_reward(obs, action)
            joint_data.append([step, reward] + joint_positions.tolist())
            print(f"Step {step}: Reward = {reward:.4f}, Joints = {joint_positions}")

    # บันทึกมุม joint และ reward
    with open("/home/sncbot/IsaacLab/joint_positions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Reward", "Hip", "Arm", "Finger", "Torso"])  # ปรับชื่อ
        writer.writerows(joint_data)
    
    print("บันทึกข้อมูล joint และ reward เสร็จแล้ว!")
else:
    print("ไม่สามารถรันได้ เนื่องจากโมเดลหรือหุ่นยนต์โหลดไม่สำเร็จ")

# ปรับตั้งค่าแสง (จัดการ /rtx-defaults-transient/meshlights/forceDisable)
try:
    from omni.isaac.core.utils import set_lighting_mode
    set_lighting_mode("forceDisable", False)  # เปิดหรือปิดแสง mesh
except ImportError:
    print("ไม่สามารถปรับตั้งค่าแสงได้")