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
# Options: "LINE", "CIRCLE", "TRIANGLE"
FORMATION_TYPE = "TRIANGLE" 

# --- Formation Geometry ---
# LINE Parameters
LINE_DIST = 1.0       

# CIRCLE Parameters
CIRCLE_RADIUS = 2.0   

# TRIANGLE Parameters
TRIANGLE_SPACING = 1.5 # Distance between rows/cols

# Control Gains
GAIN_L1 = 2.0   # Neighbor Spring Strength
GAIN_L2 = 2.0   # Second Neighbor Spring Strength
GAIN_PSI = 1.5  # Bearing Control Strength

# --- Physics & Gait ---
STAND_BASE = np.array([0.0, 0.9, -1.8])
CALIB_VELOCITY = 0.5      
CALIB_FREQ = 2.0          
BASE_SWING_AMP_THIGH = 0.3
BASE_SWING_AMP_CALF  = 0.4

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
# 3. CONTROL LAWS (l-l and l-psi)
# =========================================================

def calculate_ll_control(my_pos, my_yaw, l1_pos, l2_pos, d1_target, d2_target):
    """ Leader-Leader: Triangulates position from two leaders """
    v1 = l1_pos - my_pos
    v2 = l2_pos - my_pos
    d1 = np.linalg.norm(v1)
    d2 = np.linalg.norm(v2)
    
    u1 = v1 / (d1 + 1e-6)
    u2 = v2 / (d2 + 1e-6)
    
    e1 = d1 - d1_target
    e2 = d2 - d2_target
    
    # Spring Force
    vel_global = (GAIN_L1 * e1 * u1) + (GAIN_L2 * e2 * u2)
    
    # Body Frame Projection
    c, s = np.cos(my_yaw), np.sin(my_yaw)
    target_speed = vel_global[0] * c + vel_global[1] * s
    target_lat   = -vel_global[0] * s + vel_global[1] * c
    
    steer_cmd = 2.0 * np.arctan2(target_lat, 1.0) 
    return target_speed, steer_cmd, d1, d2

def calculate_lpsi_control(my_pos, my_yaw, leader_pos, leader_yaw, dist_target, bearing_target):
    """ Leader-Follower: Targets a specific offset point """
    global_bearing = leader_yaw + bearing_target
    target_global_x = leader_pos[0] + dist_target * np.cos(global_bearing)
    target_global_y = leader_pos[1] + dist_target * np.sin(global_bearing)
    
    ex = target_global_x - my_pos[0]
    ey = target_global_y - my_pos[1]
    
    dist_error = np.sqrt(ex**2 + ey**2)
    
    c, s = np.cos(my_yaw), np.sin(my_yaw)
    v_surge = ex * c + ey * s
    v_sway  = -ex * s + ey * c
    
    cmd_speed = GAIN_PSI * v_surge 
    cmd_steer = 2.0 * np.arctan2(v_sway, 1.0)
    
    return cmd_speed, cmd_steer, dist_error

def build_triangle_topology(num_robots, spacing):
    """ dynamically builds the formation graph """
    topology = {}
    current_idx = num_robots - 1
    rows = [] 
    r_idx = 0
    
    # 1. Layout rows (N-1 at front)
    while current_idx >= 0:
        count_in_row = r_idx + 1
        current_row = []
        for k in range(count_in_row):
            if current_idx < 0: break
            current_row.append(current_idx)
            current_idx -= 1
        rows.append(current_row)
        r_idx += 1
        
    leader_id = rows[0][0]
    topology[leader_id] = {'type': 'LEADER'}
    
    # 2. Assign Strategies
    for r_i in range(1, len(rows)):
        curr_row = rows[r_i]
        prev_row = rows[r_i - 1]
        
        for k, bot_id in enumerate(curr_row):
            # Edges use l-psi
            if k == 0:
                target_id = prev_row[0]
                dx, dy = -spacing, -spacing/2.0
                dist = np.sqrt(dx**2 + dy**2)
                bearing = np.arctan2(dy, dx)
                topology[bot_id] = {'type': 'LPSI', 'leader': target_id, 'dist': dist, 'bearing': bearing}
            elif k == len(curr_row) - 1:
                target_id = prev_row[-1]
                dx, dy = -spacing, spacing/2.0
                dist = np.sqrt(dx**2 + dy**2)
                bearing = np.arctan2(dy, dx)
                topology[bot_id] = {'type': 'LPSI', 'leader': target_id, 'dist': dist, 'bearing': bearing}
            # Internals use l-l
            else:
                l1 = prev_row[k-1] 
                l2 = prev_row[k]   
                dist_common = np.sqrt(spacing**2 + (spacing/2.0)**2)
                topology[bot_id] = {'type': 'LL', 'l1': l1, 'd1': dist_common, 'l2': l2, 'd2': dist_common}
                
    return topology, leader_id

# =========================================================
# 4. TERMINAL DASHBOARD
# =========================================================
def print_dashboard(sim_time, formation_type, robot_data, leader_id):
    # ANSI Clear Screen
    sys.stdout.write("\033[H\033[J")
    
    print(f"============================================================")
    print(f"   FORMATION CONTROL DASHBOARD  |  Type: {formation_type}")
    print(f"============================================================")
    print(f" Time: {sim_time:.2f} s  |  Robots: {len(robot_data)}")
    print(f"------------------------------------------------------------")
    print(f" {'ID':<4} | {'ROLE':<18} | {'VEL':<6} | {'STR':<6} | {'STATUS / ERROR'}")
    print(f"------------------------------------------------------------")
    
    # Sort by ID descending (Leader first usually)
    sorted_ids = sorted(robot_data.keys(), reverse=True)
    
    for r in sorted_ids:
        d = robot_data[r]
        role = d['role']
        vel = d['vel']
        steer = d['steer']
        err_str = d['error']
        
        # Colorize
        vel_str = f"{vel:.2f}"
        if r == leader_id:
            row_color = "\033[92m" # Green for Leader
        else:
            row_color = "\033[0m"  # Default
            
        print(f"{row_color} R{r:<3} | {role:<18} | {vel_str:<6} | {steer:>5.2f}  | {err_str} \033[0m")
        
    print(f"------------------------------------------------------------")
    print(f" Global Leader Velocity: {robot_data[leader_id]['vel']:.2f}")

# =========================================================
# 5. MAIN LOOP
# =========================================================

def main():
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
    except ValueError:
        print(f"Error: Could not find {XML_PATH}. Run generate_xml.py first.")
        return

    num_robots = model.nu // n_ctrl_per_robot
    
    # --- GEOMETRY & TOPOLOGY SETUP ---
    formation_map = {}
    global_leader_id = num_robots - 1
    circle_d1, circle_d2, circle_bearing = 0,0,0
    
    if FORMATION_TYPE == "TRIANGLE":
        formation_map, global_leader_id = build_triangle_topology(num_robots, TRIANGLE_SPACING)
    elif FORMATION_TYPE == "CIRCLE" and num_robots >= 3:
        theta_step = 2 * np.pi / num_robots
        circle_d1 = 2 * CIRCLE_RADIUS * np.sin(theta_step / 2.0)
        circle_d2 = 2 * CIRCLE_RADIUS * np.sin(theta_step)
        vec_dx = CIRCLE_RADIUS * np.cos(-theta_step) - CIRCLE_RADIUS
        vec_dy = CIRCLE_RADIUS * np.sin(-theta_step)
        circle_bearing = np.arctan2(vec_dy, vec_dx)

    joint_map = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    robot_phases = np.array([r * 0.5 for r in range(num_robots)]) 
    robot_desired_velocities = np.zeros(num_robots)
    robot_steering_cmds = np.zeros(num_robots)
    
    # Dashboard Data Container
    dash_data = {r: {'role':'-', 'vel':0, 'steer':0, 'error':'-'} for r in range(num_robots)}

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

            # --- A. STATE ESTIMATION ---
            robot_states = []
            for r in range(num_robots):
                q_start = r * n_qpos_per_robot
                pos = data.qpos[q_start : q_start + 3]
                quat = data.qpos[q_start + 3 : q_start + 7]
                robot_states.append({
                    'pos': pos[:2], 'yaw': quat_to_yaw(quat)
                })

            # --- B. HIGH LEVEL CONTROLLER ---
            if walking_active:
                # Leader moves forward
                robot_desired_velocities[global_leader_id] = 0.6    
                robot_steering_cmds[global_leader_id] = 0.2
                
                # Update Dashboard for Leader
                dash_data[global_leader_id]['role'] = "GLOBAL LEADER"
                dash_data[global_leader_id]['vel'] = 0.6
                dash_data[global_leader_id]['error'] = "N/A"
            
            for r in range(num_robots - 1, -1, -1):
                if r == global_leader_id: continue
                
                target_vel = 0.0
                steer_cmd = 0.0
                curr_st = robot_states[r]
                
                # --- FORMATION LOGIC ---
                if FORMATION_TYPE == "TRIANGLE":
                    spec = formation_map.get(r, {})
                    mode = spec.get('type', 'NONE')
                    
                    if mode == 'LPSI':
                        lid = spec['leader']
                        tv, sc, err = calculate_lpsi_control(
                            curr_st['pos'], curr_st['yaw'], 
                            robot_states[lid]['pos'], robot_states[lid]['yaw'], 
                            spec['dist'], spec['bearing']
                        )
                        tv += robot_desired_velocities[lid] # Daisy Chain
                        target_vel, steer_cmd = tv, sc
                        
                        dash_data[r]['role'] = f"L-Psi (R{lid})"
                        dash_data[r]['error'] = f"DstErr: {err:+.2f}m"

                    elif mode == 'LL':
                        l1, l2 = spec['l1'], spec['l2']
                        tv, sc, d1, d2 = calculate_ll_control(
                            curr_st['pos'], curr_st['yaw'], 
                            robot_states[l1]['pos'], robot_states[l2]['pos'], 
                            spec['d1'], spec['d2']
                        )
                        tv += (robot_desired_velocities[l1] + robot_desired_velocities[l2]) / 2.0
                        target_vel, steer_cmd = tv, sc

                        e1 = d1 - spec['d1']
                        e2 = d2 - spec['d2']
                        dash_data[r]['role'] = f"L-L (R{l1}, R{l2})"
                        dash_data[r]['error'] = f"E1:{e1:+.1f} E2:{e2:+.1f}"

                elif FORMATION_TYPE == "CIRCLE":
                    if r == global_leader_id - 1:
                        lid = r + 1
                        tv, sc, err = calculate_lpsi_control(
                            curr_st['pos'], curr_st['yaw'], 
                            robot_states[lid]['pos'], robot_states[lid]['yaw'], 
                            circle_d1, circle_bearing
                        )
                        tv += robot_desired_velocities[lid]
                        target_vel, steer_cmd = tv, sc
                        dash_data[r]['role'] = f"L-Psi (R{lid})"
                        dash_data[r]['error'] = f"DstErr: {err:+.2f}m"
                    else:
                        l1, l2 = r+1, r+2
                        tv, sc, d1, d2 = calculate_ll_control(
                            curr_st['pos'], curr_st['yaw'], 
                            robot_states[l1]['pos'], robot_states[l2]['pos'], 
                            circle_d1, circle_d2
                        )
                        tv += robot_desired_velocities[l1]
                        target_vel, steer_cmd = tv, sc
                        e1 = d1 - circle_d1
                        e2 = d2 - circle_d2
                        dash_data[r]['role'] = f"L-L (R{l1}, R{l2})"
                        dash_data[r]['error'] = f"E1:{e1:+.1f} E2:{e2:+.1f}"

                elif FORMATION_TYPE == "LINE":
                    lid = r + 1
                    tv, sc, err = calculate_lpsi_control(
                        curr_st['pos'], curr_st['yaw'], 
                        robot_states[lid]['pos'], robot_states[lid]['yaw'], 
                        LINE_DIST, np.pi
                    )
                    tv += robot_desired_velocities[lid]
                    target_vel, steer_cmd = tv, sc
                    dash_data[r]['role'] = f"Follow R{lid}"
                    dash_data[r]['error'] = f"Err: {err:+.2f}m"

                # --- SAFETY CLAMPS ---
                ACCEL_LIMIT = 0.005
                prev_vel = robot_desired_velocities[r]
                vel_diff = np.clip(target_vel - prev_vel, -ACCEL_LIMIT, ACCEL_LIMIT)
                target_vel = np.clip(prev_vel + vel_diff, 0.0, 1.0)
                steer_cmd = np.clip(steer_cmd, -0.6, 0.6)
                
                robot_desired_velocities[r] = target_vel
                robot_steering_cmds[r] = steer_cmd
                
                dash_data[r]['vel'] = target_vel
                dash_data[r]['steer'] = steer_cmd

            # --- C. LOW LEVEL CONTROLLER ---
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

            # --- D. TERMINAL DASHBOARD ---
            # Update at 10Hz (every ~50 steps)
            if step_counter % 50 == 0 and walking_active:
                print_dashboard(sim_time, FORMATION_TYPE, dash_data, global_leader_id)

            mujoco.mj_step(model, data)
            viewer.sync()
            
            time_until_next = dt - (time.time() - step_start)
            if time_until_next > 0: time.sleep(time_until_next)

if __name__ == "__main__":
    main()