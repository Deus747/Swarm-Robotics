import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import os

# =========================================================
# 1. CONFIGURATION
# =========================================================
XML_PATH = "scene.xml"
SIM_DURATION = 200.0

# --- FORMATION SELECTOR ---
FORMATION_TYPE = "TRIANGLE" 

# --- Formation Geometry ---
LINE_DIST = 1.0       
CIRCLE_RADIUS = 2.5   
TRIANGLE_SPACING = 1.5 

# --- Virtual Structure Gains ---
VS_GAIN_P = 4.0      # Increased Position Gain for tighter tracking
VS_GAIN_PSI = 2.0    # Increased Heading Gain

# --- Physics & Gait ---
STAND_BASE = np.array([0.0, 0.9, -1.8])
CALIB_VELOCITY = 0.5  
CALIB_FREQ = 2.0          
BASE_SWING_AMP_THIGH = 0.3
BASE_SWING_AMP_CALF  = 0.4

# --- CIRCULAR PATH PARAMETERS ---
PATH_RADIUS = 8.0 
PATH_DIRECTION = 1.0 # 1.0 = Left turn, -1.0 = Right turn

# --- Dimensions ---
n_qpos_per_robot = 19
n_qvel_per_robot = 18
n_ctrl_per_robot = 12

# =========================================================
# 2. MATH HELPERS
# =========================================================

def pd_controller(target_q, current_q, current_v, kp, kd):
    return kp * (target_q - current_q) - kd * current_v

def quat_to_yaw(quat):
    w, x, y, z = quat
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(t3, t4)

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def get_gait_target(phase, actuator_idx, amp_scale=1.0, steer_cmd=0.0):
    is_left_leg = (actuator_idx >= 3 and actuator_idx <= 5) or (actuator_idx >= 9)
    side_amp_mod = 1.0 - steer_cmd if is_left_leg else 1.0 + steer_cmd
    side_amp_mod = np.clip(side_amp_mod, 0.4, 1.6)

    thigh_amp = BASE_SWING_AMP_THIGH * amp_scale * side_amp_mod
    calf_amp  = BASE_SWING_AMP_CALF  * amp_scale * side_amp_mod

    is_pair_A = (actuator_idx < 3) or (actuator_idx >= 9) 
    leg_phase = phase if is_pair_A else phase + np.pi

    joint_type = actuator_idx % 3 
    base_angle = STAND_BASE[joint_type]
    adjustment = 0.0

    if joint_type == 1: # Thigh
        adjustment = -thigh_amp * np.sin(leg_phase)
    elif joint_type == 2: # Calf
        swing_lift = np.cos(leg_phase)
        adjustment = -calf_amp * swing_lift if swing_lift > 0 else 0.0

    return base_angle + adjustment

# =========================================================
# 3. VIRTUAL STRUCTURE LOGIC
# =========================================================

def get_formation_nodes(formation_type, num_robots):
    nodes = []
    
    if formation_type == "LINE":
        length = (num_robots - 1) * LINE_DIST
        start_x = length / 2.0
        for i in range(num_robots):
            nodes.append([start_x - i * LINE_DIST, 0.0])
            
    elif formation_type == "TRIANGLE":
        # Tip at Front (+X), Base at Back (-X)
        row_height = (np.sqrt(3)/2) * TRIANGLE_SPACING
        count = 0
        row = 0
        temp_nodes = []
        
        while count < num_robots:
            slots_in_row = row + 1
            row_width = (slots_in_row - 1) * TRIANGLE_SPACING
            x_pos = -row * row_height 
            start_y = -row_width / 2.0
            
            for i in range(slots_in_row):
                if count >= num_robots: break
                y_pos = start_y + i * TRIANGLE_SPACING
                temp_nodes.append([x_pos, y_pos])
                count += 1
            row += 1
            
        temp_nodes = np.array(temp_nodes)
        centroid = np.mean(temp_nodes, axis=0)
        nodes = temp_nodes - centroid
        
    elif formation_type == "CIRCLE":
        angle_step = 2 * np.pi / num_robots
        for i in range(num_robots):
            theta = i * angle_step
            nodes.append([CIRCLE_RADIUS * np.cos(theta), CIRCLE_RADIUS * np.sin(theta)])
            
    return np.array(nodes)

def fit_virtual_structure(robot_positions, local_nodes):
    """ Fits VS to robots (Procrustes) - Used for Logging/Initialization only """
    centroid_robots = np.mean(robot_positions, axis=0)
    centroid_nodes = np.mean(local_nodes, axis=0)
    
    robots_centered = robot_positions - centroid_robots
    nodes_centered = local_nodes - centroid_nodes
    
    num = 0.0
    den = 0.0
    for i in range(len(local_nodes)):
        xn, yn = nodes_centered[i]
        xr, yr = robots_centered[i]
        num += (xn * yr - yn * xr)
        den += (xn * xr + yn * yr)
        
    best_yaw = np.arctan2(num, den)
    return centroid_robots, best_yaw

# =========================================================
# 4. DASHBOARD
# =========================================================
def print_vs_dashboard(sim_time, formation_type, robot_data, ref_yaw, fit_yaw, omega):
    sys.stdout.write("\033[H\033[J")
    print(f"============================================================")
    print(f"   VIRTUAL STRUCTURE DASHBOARD  |  Type: {formation_type}")
    print(f"============================================================")
    print(f" Time: {sim_time:.2f} s  |  Robots: {len(robot_data)}")
    print(f" Path: CIRCLE (Radius: {PATH_RADIUS}m)")
    print(f" Reference Yaw: {ref_yaw:.2f} rad  |  Actual Yaw: {fit_yaw:.2f} rad")
    print(f" Turn Rate: {omega:.3f} rad/s")
    print(f"------------------------------------------------------------")
    print(f" {'ID':<4} | {'CMD_VEL':<8} | {'CMD_STR':<8} | {'ERR_LONG':<8} | {'ERR_LAT'}")
    print(f"------------------------------------------------------------")
    
    ids = sorted(robot_data.keys(), reverse=True)
    for r in ids:
        d = robot_data[r]
        role = "(L)" if r == ids[0] else ""
        print(f" R{r:<3}{role:<2}| {d['vel']:<8.3f} | {d['steer']:<8.3f} | {d['elong']:<8.3f} | {d['elat']:.3f}")
        
    print(f"------------------------------------------------------------")

# =========================================================
# 5. MAIN LOOP
# =========================================================

def main():
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
    except ValueError:
        print(f"Error: Could not find {XML_PATH}.")
        return

    num_robots = model.nu // n_ctrl_per_robot
    print(f"Detected {num_robots} robots.")

    # --- VS SETUP ---
    vs_nodes_raw = get_formation_nodes(FORMATION_TYPE, num_robots)
    # Sort for Leader assignment (Robot N-1 = Front)
    vs_nodes_target = vs_nodes_raw[::-1]
    
    joint_map = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    robot_phases = np.array([r * 0.5 for r in range(num_robots)]) 
    robot_desired_velocities = np.zeros(num_robots)
    robot_steering_cmds = np.zeros(num_robots)
    dash_data = {r: {'vel':0, 'steer':0, 'elong':0, 'elat':0} for r in range(num_robots)}

    # --- REFERENCE TRAJECTORY STATE ---
    # These variables hold the "Perfect" position of the virtual structure
    path_initialized = False
    ref_vs_pos = np.zeros(2)
    ref_vs_yaw = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        start_time = time.time()
        dt = model.opt.timestep
        step_counter = 0

        while viewer.is_running() and time.time() - start_time < SIM_DURATION:
            step_start = time.time()
            sim_time = data.time
            walking_active = sim_time > 1.0
            step_counter += 1

            # --- A. SENSE ---
            robot_states = []
            robot_positions = []
            for r in range(num_robots):
                q_start = r * n_qpos_per_robot
                pos = data.qpos[q_start : q_start + 3]
                quat = data.qpos[q_start + 3 : q_start + 7]
                yaw = quat_to_yaw(quat)
                robot_states.append({'pos': pos[:2], 'yaw': yaw})
                robot_positions.append(pos[:2])
            robot_positions = np.array(robot_positions)

            # Calculate fitted state just for logging/debug
            fit_pos, fit_yaw = fit_virtual_structure(robot_positions, vs_nodes_target)

            # --- B. VIRTUAL STRUCTURE UPDATE ---
            if walking_active:
                # 1. INITIALIZE PATH (Once)
                if not path_initialized:
                    # Spawn the "Ghost" structure exactly where the robots are now
                    ref_vs_pos = fit_pos.copy()
                    ref_vs_yaw = fit_yaw
                    path_initialized = True
                    print(f"Path Initialized at {ref_vs_pos}")

                # 2. UPDATE REFERENCE STATE (Kinematics)
                # This creates the perfect circular path independent of robots
                mission_vel = CALIB_VELOCITY
                mission_omega = (mission_vel / PATH_RADIUS) * PATH_DIRECTION
                
                ref_vs_yaw += mission_omega * dt
                ref_vs_pos += np.array([
                    np.cos(ref_vs_yaw) * mission_vel * dt,
                    np.sin(ref_vs_yaw) * mission_vel * dt
                ])

                # 3. ROBOT CONTROL (Track the Reference)
                for r in range(num_robots):
                    node = vs_nodes_target[r]
                    
                    # Transform Node to World Frame using REFERENCE (Ghost)
                    c_ref, s_ref = np.cos(ref_vs_yaw), np.sin(ref_vs_yaw)
                    target_x = (node[0] * c_ref - node[1] * s_ref) + ref_vs_pos[0]
                    target_y = (node[0] * s_ref + node[1] * c_ref) + ref_vs_pos[1]
                    
                    rx, ry = robot_states[r]['pos']
                    ryaw = robot_states[r]['yaw']
                    
                    # Global Error
                    ex_global = target_x - rx
                    ey_global = target_y - ry
                    
                    # Body Error
                    e_long = ex_global * np.cos(ryaw) + ey_global * np.sin(ryaw)
                    e_lat  = -ex_global * np.sin(ryaw) + ey_global * np.cos(ryaw)
                    
                    # Heading Error (Track Ref Yaw)
                    e_head = wrap_angle(ref_vs_yaw - ryaw)
                    
                    v_cmd = mission_vel + VS_GAIN_P * e_long
                    steer_cmd = VS_GAIN_PSI * e_head + VS_GAIN_P * e_lat
                    
                    v_cmd = np.clip(v_cmd, 0.0, 1.5)
                    steer_cmd = np.clip(steer_cmd, -1.0, 1.0)
                    
                    robot_desired_velocities[r] = v_cmd
                    robot_steering_cmds[r] = steer_cmd
                    dash_data[r] = {'vel': v_cmd, 'steer': steer_cmd, 'elong': e_long, 'elat': e_lat}
            else:
                robot_desired_velocities[:] = 0.0
                robot_steering_cmds[:] = 0.0
                mission_omega = 0.0

            # --- C. LOW LEVEL ---
            KP_LO, KD_LO = 60.0, 3.0
            for r in range(num_robots):
                v_cmd = robot_desired_velocities[r]
                s_cmd = robot_steering_cmds[r]
                
                freq_cmd = max(0.0, CALIB_FREQ * (v_cmd / CALIB_VELOCITY))
                amp_scale = np.clip(0.5 + 0.5 * (v_cmd / CALIB_VELOCITY), 0.2, 1.5) 

                if walking_active and v_cmd > 0.01:
                    robot_phases[r] += freq_cmd * 2 * np.pi * dt
                    robot_phases[r] %= (2 * np.pi)

                qpos_idx = r * n_qpos_per_robot
                qvel_idx = r * n_qvel_per_robot
                ctrl_idx = r * n_ctrl_per_robot

                robot_q = data.qpos[qpos_idx + 7 : qpos_idx + 19]
                robot_v = data.qvel[qvel_idx + 6 : qvel_idx + 18]

                for i in range(12):
                    if walking_active:
                        target_angle = get_gait_target(robot_phases[r], i, amp_scale, s_cmd)
                    else:
                        target_angle = STAND_BASE[i % 3]

                    joint_idx = joint_map[i]
                    torque = pd_controller(target_angle, robot_q[joint_idx], robot_v[joint_idx], KP_LO, KD_LO)
                    data.ctrl[ctrl_idx + i] = torque

            if step_counter % 50 == 0 and walking_active:
                print_vs_dashboard(sim_time, FORMATION_TYPE, dash_data, ref_vs_yaw, fit_yaw, mission_omega)

            mujoco.mj_step(model, data)
            viewer.sync()
            
            time_until_next = dt - (time.time() - step_start)
            if time_until_next > 0: time.sleep(time_until_next)

if __name__ == "__main__":
    main()