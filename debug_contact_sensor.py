#!/usr/bin/env python3
"""
Add this code to your training script to debug contact sensors.
Insert this after env.step() in your training loop.
"""

debug_code = """
# DEBUG: Check contact forces for left vs right foot
if step_count % 100 == 0:  # Print every 100 steps
    contact_sensor = env.scene.sensors["contact_forces"]

    # Get body names
    body_names = contact_sensor.cfg.body_names
    print(f"\\nStep {step_count} - Contact Sensor Debug:")
    print(f"  Body names pattern: {body_names}")
    print(f"  Number of bodies: {len(contact_sensor.data.body_names)}")

    # Get contact forces
    contact_forces = contact_sensor.data.net_forces_w[:, :, :]  # [num_envs, num_bodies, 3]
    contact_magnitude = contact_forces.norm(dim=-1)  # [num_envs, num_bodies]

    # Get air time
    air_time = contact_sensor.data.current_air_time  # [num_envs, num_bodies]
    contact_time = contact_sensor.data.current_contact_time  # [num_envs, num_bodies]

    # Print first environment
    env_id = 0
    print(f"\\n  Environment {env_id}:")
    for body_id in range(contact_magnitude.shape[1]):
        body_name = contact_sensor.data.body_names[body_id] if body_id < len(contact_sensor.data.body_names) else f"body_{body_id}"
        force = contact_magnitude[env_id, body_id].item()
        air = air_time[env_id, body_id].item()
        contact = contact_time[env_id, body_id].item()
        in_contact = contact > 0.0

        status = "CONTACT" if in_contact else "AIR"
        print(f"    {body_name:30s}: force={force:6.1f}N  air_time={air:5.2f}s  contact_time={contact:5.2f}s  [{status}]")

    # Check for stuck contacts
    stuck_threshold = 2.0  # If in contact for > 2 seconds continuously
    for env_id in range(min(5, contact_time.shape[0])):
        for body_id in range(contact_time.shape[1]):
            if contact_time[env_id, body_id] > stuck_threshold:
                body_name = contact_sensor.data.body_names[body_id] if body_id < len(contact_sensor.data.body_names) else f"body_{body_id}"
                print(f"  ⚠️  ENV {env_id}: {body_name} stuck in contact for {contact_time[env_id, body_id].item():.2f}s")

    # Check feet_air_time reward calculation
    body_ids = contact_sensor.cfg.body_ids
    air_time_bodies = contact_sensor.data.current_air_time[:, body_ids]
    contact_time_bodies = contact_sensor.data.current_contact_time[:, body_ids]
    in_contact = contact_time_bodies > 0.0
    single_stance = torch.sum(in_contact.int(), dim=1) == 1

    print(f"\\n  Feet Air Time Reward Analysis (first 5 envs):")
    for env_id in range(min(5, in_contact.shape[0])):
        num_feet_in_contact = torch.sum(in_contact[env_id].int()).item()
        is_single_stance = single_stance[env_id].item()
        print(f"    ENV {env_id}: {num_feet_in_contact} feet in contact, single_stance={is_single_stance}")
"""

print("="*80)
print("DEBUG CODE FOR CONTACT SENSOR")
print("="*80)
print("\nAdd this code to your training script:\n")
print(debug_code)
print("\n" + "="*80)
print("\nOr add this simple check in your training loop:")
print("="*80)

simple_check = """
# Quick check: Print contact status every 100 steps
if step_count % 100 == 0:
    import torch
    cs = env.scene.sensors["contact_forces"]
    air = cs.data.current_air_time[0, :]  # First env
    contact = cs.data.current_contact_time[0, :]

    print(f"Step {step_count}:")
    for i, name in enumerate(cs.data.body_names):
        if 'ankle' in name or 'foot' in name:
            status = "GROUND" if contact[i] > 0 else "AIR"
            print(f"  {name}: {status} (air={air[i]:.2f}s, contact={contact[i]:.2f}s)")
"""

print(simple_check)
print("\n" + "="*80)
