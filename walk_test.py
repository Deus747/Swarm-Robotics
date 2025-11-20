import mujoco
import mujoco.viewer
import numpy as np
import time

# --- Configuration ---
XML_PATH = "scene.xml"
SIM_DURATION = 600.0

# --- Control Parameters ---
KP = 60.0
KD = 3.0
VELOCITY_X = 0.5
STEERING = 0.7  # [-1.0, 1.0] Turn left/right

# --- Gait Parameters ---
STAND_BASE = np.array([0.0, 0.9, -1.8])

# --- Dimensions per Robot ---
n_qpos_per_robot = 19
n_qvel_per_robot = 18
n_ctrl_per_robot = 12

def pd_controller(target_q, current_q, current_v, kp, kd):
    return kp * (target_q - current_q) - kd * current_v

def get_gait_target(sim_time, actuator_idx, robot_index=0, steering=0.0, velocity=0.5):
    """
    Generates target angles with realistic steering dynamics.
    steering: float in range [-1, 1], controls turn sharpness
    velocity: forward speed in m/s
    """
    # Core gait parameters
    WALK_FREQ = 2.0
    SWING_AMP_THIGH = 0.3
    SWING_AMP_CALF = 0.4
    
    # --- Realistic steering dynamics ---
    turn_factor = np.clip(abs(steering), 0.0, 1.0)
    
    # Speed reduction during sharp turns (prevents slip, more realistic)
    effective_vel = velocity * (1.0 - turn_factor * 0.25)
    vel_scale = np.clip(effective_vel / 0.5, 0.5, 2.0)
    
    # --- Phase calculation (unchanged) ---
    phase_offset = robot_index * 0.5
    phase = (sim_time * WALK_FREQ * 2 * np.pi + phase_offset) % (2 * np.pi)
    
    # --- Leg identification (unchanged) ---
    is_pair_A = (actuator_idx < 3) or (actuator_idx >= 9)
    is_right_leg = (actuator_idx < 3) or (actuator_idx >= 6 and actuator_idx < 9)
    
    if is_pair_A:
        leg_phase = phase
    else:
        leg_phase = phase + np.pi
    
    # --- Stride scaling (unchanged logic, better tuning) ---
    if steering > 0:  # Right turn
        stride_scale = 1.0 + (0.4 * turn_factor if not is_right_leg else -0.25 * turn_factor)
    else:  # Left turn
        stride_scale = 1.0 + (0.4 * turn_factor if is_right_leg else -0.25 * turn_factor)
    
    stride_scale *= vel_scale
    
    # --- Joint calculations (enhanced) ---
    joint_type = actuator_idx % 3
    base_angle = STAND_BASE[joint_type]
    adjustment = 0.0
    
    # Hip: Enhanced with body lean and dynamic abduction
    if joint_type == 0:
        # Base lean angle (roll into turn)
        lean_angle = steering * turn_factor * 0.15
        adjustment = lean_angle
        
        # Dynamic abduction during swing phase (more natural foot placement)
        swing_intensity = np.clip(np.sin(leg_phase), 0, 1)
        adjustment += steering * swing_intensity * turn_factor * 0.2
    
    # Thigh: Standard swing (unchanged)
    elif joint_type == 1:
        adjustment = -SWING_AMP_THIGH * stride_scale * np.sin(leg_phase)
    
    # Calf: Standard extension (unchanged)
    elif joint_type == 2:
        swing_lift = np.cos(leg_phase)
        if swing_lift > 0:
            adjustment = -SWING_AMP_CALF * stride_scale * swing_lift
        else:
            adjustment = 0.0
    
    return base_angle + adjustment

def main():
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
    except ValueError:
        print(f"Error: Could not find {XML_PATH}. Run generate_scene.py first.")
        return

    # Determine number of robots based on total actuators
    num_robots = model.nu // n_ctrl_per_robot
    print(f"Loaded Scene with {num_robots} robots. Realistic Steering Enabled...")

    # --- MAPPING (Actuator -> Joint) ---
    joint_map = [
        3, 4, 5,    # FR Actuator -> FR Joint
        0, 1, 2,    # FL Actuator -> FL Joint
        9, 10, 11,  # RR Actuator -> RR Joint
        6, 7, 8     # RL Actuator -> RL Joint
    ]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Reset to home position
        mujoco.mj_resetDataKeyframe(model, data, 0)
        start_time = time.time()

        while viewer.is_running() and time.time() - start_time < SIM_DURATION:
            step_start = time.time()
            sim_time = data.time

            # Wait 1 second before walking
            walking_active = sim_time > 1.0

            # --- MAIN CONTROL LOOP ---
            for r in range(num_robots):
                # Calculate offsets for this specific robot
                qpos_start_idx = r * n_qpos_per_robot
                qvel_start_idx = r * n_qvel_per_robot
                ctrl_start_idx = r * n_ctrl_per_robot

                # Extract this robot's joint states (skip root)
                robot_joints_q = data.qpos[qpos_start_idx + 7: qpos_start_idx + 19]
                robot_joints_v = data.qvel[qvel_start_idx + 6: qvel_start_idx + 18]

                # Calculate controls for this robot's 12 actuators
                for i in range(12):
                    # Target angle with realistic steering
                    if walking_active:
                        target_angle = get_gait_target(
                            sim_time - 1.0, i, 
                            robot_index=r, 
                            steering=STEERING, 
                            velocity=VELOCITY_X
                        )
                    else:
                        target_angle = STAND_BASE[i % 3]

                    # Current state (mapped)
                    joint_idx = joint_map[i]
                    curr_q = robot_joints_q[joint_idx]
                    curr_v = robot_joints_v[joint_idx]

                    # PD Control
                    torque = pd_controller(target_angle, curr_q, curr_v, KP, KD)
                    
                    # Apply to global data array
                    data.ctrl[ctrl_start_idx + i] = torque

            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()

            # Sync time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()