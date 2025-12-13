from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

HUMANOID_10102025_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/sncbot/IsaacLab/humanoid_snc_8112025/humanoid_snc/usd/Humanoid_10102025.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.90),  # Starting position adjusted for proper ground contact
        joint_pos={
            # Leg joints - bent knee stance
            ".*_hip_pitch_joint": -0.25,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_joint": 0.5,  # More bent knees
            ".*_ankle_pitch_joint": -0.25,
            ".*_ankle_roll_joint": 0.0,
            # Torso
            "pelvis_joint": 0.0,
            # Arms
            ".*_shoulder_pitch_joint": 0.35,
            "left_shoulder_roll_joint": 0.16,
            "right_shoulder_roll_joint": -0.16,
            ".*_elbow.*": 0.87,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,  # Same as G1
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow.*"],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=5.0,
            armature=0.01,
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["pelvis_joint"],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=200.0,
            damping=5.0,
            armature=0.01,
        ),
    },
)