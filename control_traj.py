import mujoco
import mujoco.viewer
import numpy as np
import time

# --- Configuration ---
XML_PATH = "scene.xml"
SIM_DURATION = 600.0

# --- Trajectory Generation ---
# Generates a Clockwise Circle starting at (0, 0)
def generate_circular_path(radius=3.0, steps=100):
    theta = np.linspace(0, 2*np.pi, steps)
    x = radius * np.sin(theta)
    y = radius * (np.cos(theta) - 1.0)
    return np.column_stack((x, y))

# Pre-generate path (Global for all bots)
PATH_POINTS = generate_circular_path(radius=3.0, steps=1000)

# --- Control Parameters ---
KP = 70.0
KD = 12.0
STAND_BASE = np.array([0.0, 0.9, -1.8])

# Gait Params
WALK_FREQ = 2.5
SWING_AMP_THIGH = 0.3
SWING_AMP_CALF  = 0.55

# --- Physics & Stability Parameters ---
LOOKAHEAD_DIST = 0.9    
VELOCITY_BASE   = 0.6   
GRAVITY = 9.81

# Gains
STEER_GAIN = 0.8        
BANK_GAIN   = 0.9       
WIDE_STANCE_GAIN = 0.15 
HEIGHT_COMP_GAIN = 0.25

# --- Robot Dimensions ---
N_QPOS = 19 # 7 (freejoint) + 12 (joints)
N_QVEL = 18 # 6 (freejoint) + 12 (joints)
N_CTRL = 12

# --- Helper Functions ---

def quaternion_to_euler(q):
    w, x, y, z = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def pd_controller(target_q, current_q, current_v, kp, kd):
    return kp * (target_q - current_q) - kd * current_v

def get_pure_pursuit_target(robot_pos, path_points, lookahead, last_idx):
    """Finds lookahead point and estimates curvature."""
    num_points = len(path_points)
    best_idx = last_idx
    found = False

    # Search forward window
    for i in range(last_idx, last_idx + 100):
        curr_idx = i % num_points
        dist = np.linalg.norm(path_points[curr_idx] - robot_pos)
        if dist > lookahead:
            best_idx = curr_idx
            found = True
            break
    if not found: best_idx = (last_idx + 1) % num_points
    target_pt = path_points[best_idx]

    # Estimate Curvature
    p1 = path_points[(best_idx - 10) % num_points]
    p2 = target_pt
    p3 = path_points[(best_idx + 10) % num_points]
    v1 = p2 - p1
    v2 = p3 - p2
    angle_change = wrap_angle(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0]))
    dist_segment = np.linalg.norm(v1) + np.linalg.norm(v2)

    curvature = angle_change / dist_segment if dist_segment > 0.01 else 0.0
    return target_pt, curvature, best_idx

def get_banking_gait_target(phase, actuator_idx, steer_cmd, curvature, velocity, pitch_err):
    """
    Generates gait with Corrected Steering and Banking.
    """
    is_pair_A = (actuator_idx < 3) or (actuator_idx >= 9)
    leg_phase = phase if is_pair_A else (phase + np.pi)

    joint_type = actuator_idx % 3
    base_angle = STAND_BASE[joint_type]
    adjustment = 0.0

    # Identity
    is_left = (actuator_idx >= 3 and actuator_idx <= 5) or (actuator_idx >= 9)
    is_front = (actuator_idx < 6)

    # --- BANKING PHYSICS ---
    # Ideal Bank Angle (Theta)
    ideal_bank = np.arctan((velocity**2 * abs(curvature)) / GRAVITY)
    ideal_bank = np.clip(ideal_bank, 0.0, 0.4)

    # Direction: curvature < 0 for Clockwise (Right Turn) -> Bank Right
    bank_direction = np.sign(curvature)

    # Leg Length Diff for Banking
    bank_offset = ideal_bank * BANK_GAIN
    leg_len_mod = 0.0

    if bank_direction > 0: # Left Turn
        if is_left: leg_len_mod = bank_offset
        else:       leg_len_mod = -bank_offset
    elif bank_direction < 0: # Right Turn
        if is_left: leg_len_mod = -bank_offset
        else:       leg_len_mod = bank_offset

    # Wide Stance (Abduction)
    stance_width = abs(curvature) * WIDE_STANCE_GAIN
    stance_width = np.clip(stance_width, 0.0, 0.3)

    # --- Joint Targets ---
    if joint_type == 1: # THIGH
        adjustment += -SWING_AMP_THIGH * np.sin(leg_phase)
        adjustment += leg_len_mod # Banking Action

        # Pitch Compensation
        if is_front: adjustment -= pitch_err * 0.3
        else:        adjustment += pitch_err * 0.3

        # Height Comp
        adjustment -= abs(bank_offset) * HEIGHT_COMP_GAIN

    elif joint_type == 2: # CALF
        swing_lift = np.cos(leg_phase)
        if swing_lift > 0:
            adjustment += -SWING_AMP_CALF * swing_lift

        adjustment += leg_len_mod * 0.5

    elif joint_type == 0: # HIP (Steering + Spacing)
        adjustment -= steer_cmd
        if is_left:
            adjustment += stance_width
        else:
            adjustment -= stance_width

    return base_angle + adjustment

def draw_path_once(viewer):
    """Draws the static global path"""
    if viewer is None: return
    # Draw Path points
    for i in range(0, len(PATH_POINTS), 10):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom: break
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE, [0.02, 0, 0],
            np.append(PATH_POINTS[i], 0.05), np.eye(3).flatten(), np.array([0, 1, 1, 0.3])
        )
        viewer.user_scn.ngeom += 1

def draw_target_marker(viewer, target_pt):
    """Draws the target sphere for a robot"""
    if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE, [0.05, 0, 0],
            np.append(target_pt, 0.2), np.eye(3).flatten(), np.array([1, 0, 0, 1])
        )
        viewer.user_scn.ngeom += 1

def main():
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
    except ValueError:
        print(f"Error: Could not find {XML_PATH}.")
        return

    # Detect number of robots
    num_robots = model.nu // N_CTRL
    print(f"Detected {num_robots} robots. Starting Multi-Bot Banked Controller.")

    joint_map = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    
    # --- Per-Robot State Initialization ---
    robot_phases = [0.0] * num_robots
    robot_path_idxs = [0] * num_robots
    
    # Stagger start phases slightly so they don't move in perfect unison
    for r in range(num_robots):
        robot_phases[r] = r * 0.5

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        start_time = time.time()

        while viewer.is_running() and time.time() - start_time < SIM_DURATION:
            step_start = time.time()
            
            # Reset debug lines for this frame
            viewer.user_scn.ngeom = 0
            draw_path_once(viewer)

            dt = model.opt.timestep
            
            # --- LOOP OVER EVERY ROBOT ---
            for r in range(num_robots):
                
                # 1. Calculate Indices
                q_idx = r * N_QPOS
                v_idx = r * N_QVEL
                c_idx = r * N_CTRL

                # 2. Extract State
                # Position is in the first 3 elements of the freejoint
                pos = data.qpos[q_idx : q_idx + 2] # X, Y only needed for path
                
                # Orientation (Quat) is in elements 3-7 of freejoint
                # NOTE: qpos for freejoint is [x,y,z, w,x,y,z]
                quat = data.qpos[q_idx+3 : q_idx+7]
                rpy = quaternion_to_euler(quat)
                
                # Joints data
                joint_q = data.qpos[q_idx+7 : q_idx+19]
                joint_v = data.qvel[v_idx+6 : v_idx+18]

                # 3. Pure Pursuit Calculation
                target_pt, curvature, new_idx = get_pure_pursuit_target(
                    pos, PATH_POINTS, LOOKAHEAD_DIST, robot_path_idxs[r]
                )
                robot_path_idxs[r] = new_idx

                dx = target_pt[0] - pos[0]
                dy = target_pt[1] - pos[1]
                desired_yaw = np.arctan2(dy, dx)
                yaw_error = wrap_angle(desired_yaw - rpy[2])

                # 4. Control Calculation
                steer_cmd = np.clip(STEER_GAIN * yaw_error, -0.35, 0.35)
                current_target_vel = VELOCITY_BASE

                # 5. Gait Update
                robot_phases[r] += (WALK_FREQ * (current_target_vel/VELOCITY_BASE)) * 2 * np.pi * dt
                robot_phases[r] = robot_phases[r] % (2 * np.pi)

                # 6. Actuator Loop
                for i in range(12):
                    # Calculate target angle
                    target = get_banking_gait_target(
                        robot_phases[r], i, steer_cmd, curvature, current_target_vel, rpy[1]
                    )
                    
                    # Map to actual joint
                    j_map_idx = joint_map[i]
                    
                    # PD Control
                    torque = pd_controller(target, joint_q[j_map_idx], joint_v[j_map_idx], KP, KD)
                    
                    # Apply Torque (Global Index)
                    data.ctrl[c_idx + i] = torque

                # Debug: Draw target for this robot
                draw_target_marker(viewer, target_pt)

            # Step physics
            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0: time.sleep(time_until_next)

if __name__ == "__main__":
    main()