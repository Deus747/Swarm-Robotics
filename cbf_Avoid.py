import mujoco
import mujoco.viewer
import numpy as np
import time
import math

# --- Configuration ---
XML_PATH = "scene.xml"
SIM_DURATION = 60.0

# --- Obstacle Config ---
OBSTACLE_POS = np.array([3.0, 0.0]) 
# We increase the radius logic so the robot reacts SOONER
OBSTACLE_RADIUS = 0.6
SAFE_BUFFER = 1.5     # Robot starts reacting 1.5m away
AVOIDANCE_GAIN = 3.0  # How hard it turns

# --- Control Gains ---
KP = 60.0  
KD = 3.0   

# --- Gait Settings ---
# [Abduction, Thigh, Calf]
STAND_BASE = np.array([0.0, 0.9, -1.8])
WALK_FREQ = 2.0
BASE_THIGH_AMP = 0.3
BASE_CALF_AMP = 0.4

# --- Dimensions ---
n_qpos_per_robot = 19 
n_qvel_per_robot = 18 
n_ctrl_per_robot = 12

def pd_controller(target_q, current_q, current_v, kp, kd):
    return kp * (target_q - current_q) - kd * current_v

def get_steering_command(robot_pos, robot_yaw):
    """
    Calculates a turning magnitude (-1.0 to 1.0).
    Positive = Turn Left
    Negative = Turn Right
    """
    # Vector from Robot -> Obstacle
    to_obs = OBSTACLE_POS - robot_pos
    dist = np.linalg.norm(to_obs)
    
    # If we are safe, do nothing
    if dist > (OBSTACLE_RADIUS + SAFE_BUFFER):
        return 0.0

    # Robot Heading Vector
    heading = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
    
    # Cross Product: (u_x * v_y - u_y * v_x)
    # This tells us if the obstacle is to our Left or Right
    cross = heading[0] * to_obs[1] - heading[1] * to_obs[0]
    
    # --- CRITICAL FIX: HEAD-ON SINGULARITY ---
    # If cross is almost 0, we are walking straight into it.
    # We force a decision to turn LEFT (positive bias).
    if abs(cross) < 0.1:
        cross = -0.1 # Pretend obstacle is slightly right, so we turn Left

    # Calculate severity based on distance (Closer = Stronger Turn)
    # We map distance 0..SAFE_BUFFER to Strength 1..0
    urgency = 1.0 - ((dist - OBSTACLE_RADIUS) / SAFE_BUFFER)
    urgency = np.clip(urgency, 0.0, 1.0)
    
    # If obstacle is to the Left (cross > 0), we want to turn Right (Negative)
    # If obstacle is to the Right (cross < 0), we want to turn Left (Positive)
    direction = -1.0 if cross > 0 else 1.0
    
    return direction * urgency * AVOIDANCE_GAIN

def get_gait_target(sim_time, actuator_idx, robot_index=0, turn_cmd=0.0):
    """
    Enhanced Gait Generator with Differential Steering.
    turn_cmd: Positive (Turn Left), Negative (Turn Right)
    """
    phase_offset = robot_index * 0.5
    phase = (sim_time * WALK_FREQ * 2 * np.pi + phase_offset) % (2 * np.pi)
    
    # Identify Leg ID and Joint Type
    # 0-2: FR, 3-5: FL, 6-8: RR, 9-11: RL
    leg_id = actuator_idx // 3 
    joint_type = actuator_idx % 3 # 0:Abd, 1:Thigh, 2:Calf
    
    is_right_leg = (leg_id == 0 or leg_id == 2) # FR or RR
    is_left_leg  = (leg_id == 1 or leg_id == 3) # FL or RL
    
    # --- DIFFERENTIAL STEERING LOGIC ---
    # If Turning Left (turn_cmd > 0):
    #   Right legs (Outer) -> Longer Steps (Higher Amplitude)
    #   Left legs (Inner)  -> Shorter Steps (Lower Amplitude)
    
    thigh_amp = BASE_THIGH_AMP
    
    if abs(turn_cmd) > 0.1:
        if turn_cmd > 0: # Turn Left
            if is_right_leg: thigh_amp *= (1.0 + abs(turn_cmd)) # Outer leg strides more
            if is_left_leg:  thigh_amp *= (1.0 - abs(turn_cmd)*0.5) # Inner leg strides less
        else: # Turn Right
            if is_left_leg:  thigh_amp *= (1.0 + abs(turn_cmd))
            if is_right_leg: thigh_amp *= (1.0 - abs(turn_cmd)*0.5)

    # Trot Phase Logic
    is_pair_A = (leg_id == 0 or leg_id == 3) # FR + RL
    leg_phase = phase if is_pair_A else phase + np.pi
    
    base = STAND_BASE[joint_type]
    adjustment = 0.0

    # 1. Thigh (Walking Motion)
    if joint_type == 1: 
        adjustment = -thigh_amp * np.sin(leg_phase)

    # 2. Calf (Lift Motion)
    elif joint_type == 2: 
        swing_lift = np.cos(leg_phase)
        if swing_lift > 0:
            adjustment = -BASE_CALF_AMP * swing_lift
            
    # 3. Abduction (Steering Assist)
    # We also slightly angle the hips into the turn
    elif joint_type == 0:
        adjustment = turn_cmd * 0.2 # Add a small constant yaw bias
        
    return base + adjustment

def main():
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
    except ValueError:
        print(f"Error: Could not find {XML_PATH}.")
        return

    num_robots = model.nu // n_ctrl_per_robot
    print(f"Loaded {num_robots} robots.")

    # Mapping based on standard Unitree Go2 XML structure
    joint_map = [
        3, 4, 5,    # FR (Actuators 0-2 mapped to Joints 3-5)
        0, 1, 2,    # FL
        9, 10, 11,  # RR
        6, 7, 8     # RL
    ]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        start_time = time.time()

        while viewer.is_running() and time.time() - start_time < SIM_DURATION:
            step_start = time.time()
            sim_time = data.time
            walking_active = sim_time > 1.0

            for r in range(num_robots):
                # Indexes
                qpos_start = r * n_qpos_per_robot
                qvel_start = r * n_qvel_per_robot
                ctrl_start = r * n_ctrl_per_robot
                
                # 1. Get Robot State
                root_pos = data.qpos[qpos_start:qpos_start+3]
                quat = data.qpos[qpos_start+3:qpos_start+7]
                
                # Calculate Yaw
                w, x, y, z = quat
                yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
                
                # 2. Calculate Steering (Avoidance)
                turn_signal = get_steering_command(root_pos[:2], yaw)
                
                # Debug print for first robot if turning hard
                if r == 0 and abs(turn_signal) > 0.5:
                    pass # print(f"Avoiding! Signal: {turn_signal:.2f}")

                # 3. Control Loop
                robot_q = data.qpos[qpos_start+7 : qpos_start+19]
                robot_v = data.qvel[qvel_start+6 : qvel_start+18]

                for i in range(12):
                    if walking_active:
                        target = get_gait_target(sim_time-1.0, i, r, turn_signal)
                    else:
                        target = STAND_BASE[i % 3]
                    
                    # Actuate
                    j_idx = joint_map[i]
                    torque = pd_controller(target, robot_q[j_idx], robot_v[j_idx], KP, KD)
                    data.ctrl[ctrl_start + i] = torque

            mujoco.mj_step(model, data)
            viewer.sync()

            # Timing
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

if __name__ == "__main__":
    main()