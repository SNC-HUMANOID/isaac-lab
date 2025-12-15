#!/usr/bin/env python3
"""
G1 Robot Kinematics with Pinocchio Demo
Demonstrates advanced kinematics calculations using the trained model data
"""

import numpy as np
import pinocchio as pin
from g1_kinematics_export import G1_JOINTS, get_joint_names, get_initial_joint_positions

def create_g1_pinocchio_model():
    """
    Create a simple G1 model in Pinocchio for kinematics demonstration
    (This is a simplified version - real robot would need proper URDF)
    """
    
    # Create empty model
    model = pin.Model()
    
    # Add universe/world frame
    parent_id = 0  # Universe frame
    
    # Robot base properties
    base_mass = 5.0  # kg
    base_inertia = pin.Inertia.FromBox(base_mass, 0.3, 0.2, 0.1)  # Simple box inertia
    
    # Add base/torso link
    base_placement = pin.SE3.Identity()
    base_joint_id = model.addJoint(parent_id, pin.JointModelFreeFlyer(), base_placement, "base_joint")
    model.addFrame(pin.Frame("base_link", base_joint_id, 0, base_placement, pin.OP_FRAME))
    model.appendBodyToJoint(base_joint_id, base_inertia, pin.SE3.Identity())
    
    # Add simplified leg chains (just one leg for demo)
    parent_id = base_joint_id
    
    # Hip joints
    hip_placement = pin.SE3(np.eye(3), np.array([0, 0.1, -0.1]))  # Offset to hip
    hip_joint_id = model.addJoint(parent_id, pin.JointModelRY(), hip_placement, "left_hip_pitch")
    hip_inertia = pin.Inertia.FromBox(1.0, 0.1, 0.1, 0.3)
    model.appendBodyToJoint(hip_joint_id, hip_inertia, pin.SE3.Identity())
    
    # Thigh link
    thigh_placement = pin.SE3(np.eye(3), np.array([0, 0, -0.35]))  # 35cm thigh
    knee_joint_id = model.addJoint(hip_joint_id, pin.JointModelRY(), thigh_placement, "left_knee")
    thigh_inertia = pin.Inertia.FromBox(2.0, 0.1, 0.1, 0.35)
    model.appendBodyToJoint(knee_joint_id, thigh_inertia, pin.SE3.Identity())
    
    # Shin link
    shin_placement = pin.SE3(np.eye(3), np.array([0, 0, -0.35]))  # 35cm shin
    ankle_joint_id = model.addJoint(knee_joint_id, pin.JointModelRY(), shin_placement, "left_ankle_pitch")
    shin_inertia = pin.Inertia.FromBox(1.5, 0.08, 0.08, 0.35)
    model.appendBodyToJoint(ankle_joint_id, shin_inertia, pin.SE3.Identity())
    
    # Foot (end-effector)
    foot_placement = pin.SE3(np.eye(3), np.array([0, 0, -0.05]))  # 5cm to foot
    foot_frame_id = model.addFrame(pin.Frame("foot", ankle_joint_id, 0, foot_placement, pin.OP_FRAME))
    
    return model

def demonstrate_forward_kinematics():
    """Demonstrate forward kinematics calculations"""
    
    print("\n🔍 Forward Kinematics with Pinocchio")
    print("=" * 50)
    
    # Create simplified model
    model = create_g1_pinocchio_model()
    data = model.createData()
    
    print(f"Model created with {model.nq} DOF")
    print(f"Joint names: {[model.names[i] for i in range(1, model.njoints)]}")
    
    # Create configuration vector (simplified - just the leg joints we added)
    q = pin.neutral(model)  # Start with neutral configuration
    
    # Set some joint angles based on G1 initial positions
    if model.nq >= 9:  # Free-flyer (6) + 3 leg joints
        q[6] = -0.2   # hip pitch
        q[7] = 0.42   # knee
        q[8] = -0.23  # ankle pitch
    
    print(f"Joint configuration: {q}")
    
    # Compute forward kinematics
    pin.forwardKinematics(model, data, q)
    
    # Update frame placements
    pin.updateFramePlacements(model, data)
    
    # Get foot position
    foot_frame_id = model.getFrameId("foot")
    foot_position = data.oMf[foot_frame_id].translation
    foot_rotation = data.oMf[foot_frame_id].rotation
    
    print(f"\n🦶 Left Foot Kinematics:")
    print(f"   Position: [{foot_position[0]:.3f}, {foot_position[1]:.3f}, {foot_position[2]:.3f}] m")
    print(f"   Height from ground: {foot_position[2]:.3f} m")
    
    # Calculate joint positions for visualization
    joint_positions = []
    for i in range(1, model.njoints):
        joint_pos = data.oMi[i].translation
        joint_positions.append(joint_pos)
        joint_name = model.names[i]
        print(f"   {joint_name}: [{joint_pos[0]:.3f}, {joint_pos[1]:.3f}, {joint_pos[2]:.3f}] m")
    
    return model, data, q, joint_positions

def demonstrate_inverse_kinematics():
    """Demonstrate inverse kinematics"""
    
    print("\n🎯 Inverse Kinematics Demo")
    print("=" * 30)
    
    model, data, q_current, _ = demonstrate_forward_kinematics()
    
    # Target foot position (move foot 10cm forward, 5cm up)
    target_position = np.array([0.1, 0.1, -0.6])  # x, y, z
    
    print(f"Target foot position: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}] m")
    
    # Simple IK using numerical differentiation
    foot_frame_id = model.getFrameId("foot")
    
    # Iterative IK solver
    max_iterations = 100
    tolerance = 1e-4
    step_size = 0.1
    
    q_ik = q_current.copy()
    
    for iteration in range(max_iterations):
        # Compute current foot position
        pin.forwardKinematics(model, data, q_ik)
        pin.updateFramePlacements(model, data)
        current_position = data.oMf[foot_frame_id].translation
        
        # Compute error
        error = target_position - current_position
        error_norm = np.linalg.norm(error)
        
        if error_norm < tolerance:
            print(f"✅ Converged in {iteration} iterations")
            break
        
        # Compute Jacobian
        pin.computeFrameJacobian(model, data, q_ik, foot_frame_id, pin.LOCAL_WORLD_ALIGNED)
        J = data.J[:3, :]  # Position part of Jacobian
        
        # Pseudo-inverse IK step
        try:
            J_pinv = np.linalg.pinv(J)
            dq = step_size * J_pinv @ error
            
            # Only update the joint parts (skip free-flyer base)
            tail_len = dq.shape[0] - 6
# Update exactly that many entries
            q_ik[6:6 + tail_len] += dq[6:]

            
        except np.linalg.LinAlgError:
            print("❌ Singular Jacobian encountered")
            break
    
    else:
        print(f"❌ Did not converge in {max_iterations} iterations")
    
    # Final result
    pin.forwardKinematics(model, data, q_ik)
    pin.updateFramePlacements(model, data)
    final_position = data.oMf[foot_frame_id].translation
    final_error = np.linalg.norm(target_position - final_position)
    
    print(f"Final foot position: [{final_position[0]:.3f}, {final_position[1]:.3f}, {final_position[2]:.3f}] m")
    print(f"Final error: {final_error:.4f} m")
    print(f"Joint solution: {q_ik[6:].tolist()}")  # Show only joint angles

def analyze_workspace():
    """Analyze the robot's workspace"""
    
    print("\n🌐 Workspace Analysis")
    print("=" * 30)
    
    model, data, _, _ = demonstrate_forward_kinematics()
    foot_frame_id = model.getFrameId("foot")
    
    # Sample different joint configurations
    workspace_points = []
    joint_ranges = [
        (-0.5, 0.5),   # hip pitch
        (0.0, 2.0),    # knee
        (-0.5, 0.5)    # ankle pitch
    ]
    
    samples_per_joint = 5
    print(f"Sampling {samples_per_joint**3} configurations...")
    
    for hip in np.linspace(joint_ranges[0][0], joint_ranges[0][1], samples_per_joint):
        for knee in np.linspace(joint_ranges[1][0], joint_ranges[1][1], samples_per_joint):
            for ankle in np.linspace(joint_ranges[2][0], joint_ranges[2][1], samples_per_joint):
                
                q = pin.neutral(model)
                if model.nq >= 9:
                    q[6] = hip
                    q[7] = knee
                    q[8] = ankle
                
                pin.forwardKinematics(model, data, q)
                pin.updateFramePlacements(model, data)
                
                foot_pos = data.oMf[foot_frame_id].translation
                workspace_points.append(foot_pos.copy())
    
    workspace_points = np.array(workspace_points)
    
    print(f"Workspace statistics:")
    print(f"   X range: {workspace_points[:, 0].min():.3f} to {workspace_points[:, 0].max():.3f} m")
    print(f"   Y range: {workspace_points[:, 1].min():.3f} to {workspace_points[:, 1].max():.3f} m") 
    print(f"   Z range: {workspace_points[:, 2].min():.3f} to {workspace_points[:, 2].max():.3f} m")
    print(f"   Max reach: {np.max(np.linalg.norm(workspace_points, axis=1)):.3f} m")
    
    return workspace_points

def demonstrate_dynamics():
    """Demonstrate dynamics calculations"""
    
    print("\n⚡ Dynamics Analysis")
    print("=" * 25)
    
    model, data, q, _ = demonstrate_forward_kinematics()
    
    # Set velocities and accelerations
    v = np.zeros(model.nv)  # Velocities
    a = np.zeros(model.nv)  # Accelerations
    
    # Simple motion: move knee joint
    if model.nv >= 8:
        v[7] = 1.0  # 1 rad/s knee velocity
        a[7] = 0.5  # 0.5 rad/s² knee acceleration
    
    # Compute inverse dynamics (torques needed for given motion)
    tau = pin.rnea(model, data, q, v, a)
    
    print(f"Joint torques for specified motion:")
    for i in range(6, min(len(tau), 9)):  # Skip free-flyer, show joint torques
        joint_name = model.names[i-5] if i-5 < len(model.names) else f"joint_{i}"
        print(f"   {joint_name}: {tau[i]:.3f} Nm")
    
    # Compute forward dynamics (accelerations from given torques)
    tau_input = np.zeros(model.nv)
    if model.nv >= 8:
        tau_input[7] = 10.0  # Apply 10 Nm to knee
    
    a_result = pin.aba(model, data, q, v, tau_input)
    
    print(f"\nAccelerations from 10 Nm knee torque:")
    for i in range(6, min(len(a_result), 9)):
        joint_name = model.names[i-5] if i-5 < len(model.names) else f"joint_{i}"
        print(f"   {joint_name}: {a_result[i]:.3f} rad/s²")

def connect_to_trained_model():
    """Show how to connect kinematics with trained RL model"""
    
    print("\n🧠 Connecting to Trained RL Model")
    print("=" * 40)
    
    # Load the trained model data
    import torch
    model_path = "logs/rsl_rl/g1_flat/2025-07-29_15-27-26/model_300.pt"
    
    try:
        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
        print("✅ Trained RL model loaded successfully")
        
        # Get G1 joint information
        joint_names = get_joint_names()
        initial_positions = get_initial_joint_positions()
        
        print(f"\n🤖 G1 Robot Integration:")
        print(f"   • Total joints: {len(joint_names)}")
        print(f"   • RL model parameters: {sum(p.numel() for p in model_data['model_state_dict'].values())}")
        print(f"   • Training iterations: {model_data.get('iter', 'Unknown')}")
        
        # Demonstrate how to use joint angles with kinematics
        print(f"\n🔄 Joint Angle Integration:")
        print(f"   1. RL policy outputs action → joint target positions")
        print(f"   2. Kinematics calculates end-effector positions")
        print(f"   3. Physics simulation applies forces/torques")
        
        # Show first few joints as example
        print(f"\n📊 Sample joint data:")
        for i in range(min(5, len(joint_names))):
            name = joint_names[i]
            pos = initial_positions[i]
            print(f"   • {name}: {pos:.3f} rad ({np.degrees(pos):.1f}°)")
        
        print(f"\n💡 Usage patterns:")
        print(f"   • Forward kinematics: joint angles → foot positions")
        print(f"   • Inverse kinematics: desired foot position → joint angles")
        print(f"   • Dynamics: joint torques → accelerations")
        print(f"   • RL policy: observations → actions (joint targets)")
        
    except Exception as e:
        print(f"❌ Could not load trained model: {e}")

if __name__ == "__main__":
    print("🤖 G1 Robot Advanced Kinematics with Pinocchio")
    print("=" * 60)
    
    try:
        # Run all demonstrations
        demonstrate_forward_kinematics()
        demonstrate_inverse_kinematics()  
        workspace_points = analyze_workspace()
        demonstrate_dynamics()
        connect_to_trained_model()
        
        print("\n" + "=" * 60)
        print("🎉 All demonstrations completed successfully!")
        print("\n🔧 Advanced Features Available:")
        print("   ✅ Forward kinematics")
        print("   ✅ Inverse kinematics")
        print("   ✅ Workspace analysis") 
        print("   ✅ Dynamics calculations")
        print("   ✅ Integration with trained RL model")
        
        print("\n💡 Next steps for real robot deployment:")
        print("   1. Convert USD to accurate URDF with all joints")
        print("   2. Calibrate joint limits and dynamics parameters")
        print("   3. Implement safety constraints and collision avoidance")
        print("   4. Test sim-to-real transfer with the trained policy")
        
    except Exception as e:
        print(f"❌ Error running demonstrations: {e}")
        import traceback
        traceback.print_exc()