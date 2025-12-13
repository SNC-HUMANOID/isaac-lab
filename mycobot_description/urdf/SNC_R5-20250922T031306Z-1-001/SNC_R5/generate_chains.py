from ikpy import chain
from pathlib import Path

# use a robust path to the URDF (same folder as this script)
_URDF_FILE = Path(__file__).resolve().parent / "Humanoid_SNC.urdf"
_URDF_STR = str(_URDF_FILE)


###### Left Arm ######

robotis_op3_left_arm_links = ["base_link",
                              "left_shoulder_pitch_link",
                              "left_shoulder_roll_link",
                              "left_shoulder_yaw_link",
                              "left_elbow_1_link",
                              "left_elbow_2_link"]

robotis_op3_left_arm_joints = ["left_shoulder_pitch_joint",
                               "left_shoulder_roll_joint",
                               "left_shoulder_yaw_joint",
                               "left_elbow_1_joint",
                               "left_elbow_2_joint"]

robotis_op3_left_arm_elements = [x for pair in zip(robotis_op3_left_arm_links, robotis_op3_left_arm_joints) for x in pair] + ["left_elbow_2_link"]

# Note: 'Links' in IKPY correspond to 'Joints' in URDF terminology. 'Links' in URDF are stripped by IKPY. 
robotis_op3_left_arm_chain = chain.Chain.from_urdf_file(
    _URDF_STR,
    base_elements=robotis_op3_left_arm_elements,
    last_link_vector=[0, 0, 0],
    symbolic=False,
    name="robotis_op3_left_arm")

# ensure mask matches chain length
robotis_op3_left_arm_chain.active_links_mask = [False] + [True] * (len(robotis_op3_left_arm_chain.links) - 2) + [False]
robotis_op3_left_arm_chain.to_json_file(force=True)


###### Right Arm ######

robotis_op3_right_arm_links = ["base_link",
                              "right_shoulder_pitch_link",
                              "right_shoulder_roll_link",
                              "right_shoulder_yaw_link",
                              "right_elbow_1_link",
                              "right_elbow_2_link"]

robotis_op3_right_arm_joints = ["right_shoulder_pitch_joint",
                               "right_shoulder_roll_joint",
                               "right_shoulder_yaw_joint",
                               "right_elbow_1_joint",
                               "right_elbow_2_joint"]

robotis_op3_right_arm_elements = [x for pair in zip(robotis_op3_right_arm_links, robotis_op3_right_arm_joints) for x in pair] + ["right_elbow_2_link"]

robotis_op3_right_arm_chain = chain.Chain.from_urdf_file(
    _URDF_STR,
    base_elements=robotis_op3_right_arm_elements,
    last_link_vector=[0, 0, 0],
    symbolic=False,
    name="robotis_op3_right_arm"
)

robotis_op3_right_arm_chain.active_links_mask = [False] + [True] * (len(robotis_op3_right_arm_chain.links) - 2) + [False]
robotis_op3_right_arm_chain.to_json_file(force=True)


###### Left Leg ######

robotis_op3_left_leg_links = ["base_link",
                              "Pelvis_link",
                              "left_hip_pitch_link",
                              "left_hip_roll_link",
                              "left_hip_yaw_link",
                              "left_ankle_link",
                              "left_ankle_roll_link",
                              "left_ankle_pitch_link"]

robotis_op3_left_leg_joints = ["Pelvis_joint",
                               "left_hip_yaw_joint",
                               "left_hip_roll_joint",
                               "left_hip_pitch_joint",
                               "left_ankle_joint",
                               "left_ankle_roll_joint",
                               "left_ankle_pitch_joint"]

robotis_op3_left_leg_elements = [x for pair in zip(robotis_op3_left_leg_links, robotis_op3_left_leg_joints) for x in pair] + ["left_ankle_pitch_link"]

robotis_op3_left_leg_chain = chain.Chain.from_urdf_file(
    _URDF_STR,
    base_elements=robotis_op3_left_leg_elements,
    last_link_vector=[0, 0, -0],
    symbolic=False,
    name="robotis_op3_left_leg"
)

robotis_op3_left_leg_chain.active_links_mask = [False] + [True] * (len(robotis_op3_left_leg_chain.links) - 2) + [False]
robotis_op3_left_leg_chain.to_json_file(force=True)


###### Right Leg ######

robotis_op3_right_leg_links = ["base_link",
                              "Pelvis_link",
                              "right_hip_pitch_link",
                              "right_hip_roll_link",
                              "right_hip_yaw_link",
                              "right_ankle",
                              "right_ankle_roll_link",
                              "right_ankle_pitch_link"]

robotis_op3_right_leg_joints = ["Pelvis_joint",
                               "right_hip_yaw_joint",
                               "right_hip_roll_joint",
                               "right_hip_pitch_joint",
                               "right_ankle",   # match URDF joint name
                               "right_ankle_roll_joint",
                               "right_ankle_pitch_joint"]

robotis_op3_right_leg_elements = [x for pair in zip(robotis_op3_right_leg_links, robotis_op3_right_leg_joints) for x in pair] + ["right_ankle_pitch_link"]

robotis_op3_right_leg_chain = chain.Chain.from_urdf_file(
    _URDF_STR,
    base_elements=robotis_op3_right_leg_elements,
    last_link_vector=[0, 0, 0],
    symbolic=False,
    name="robotis_op3_right_leg"
)

robotis_op3_right_leg_chain.active_links_mask = [False] + [True] * (len(robotis_op3_right_leg_chain.links) - 2) + [False]
robotis_op3_right_leg_chain.to_json_file(force=True)
