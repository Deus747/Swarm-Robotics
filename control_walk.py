import mujoco
import mujoco.viewer
import numpy as np
import time

# --- Configuration ---
XML_PATH = "scene.xml" # Changed to the generated file
SIM_DURATION = 60.0

# --- Control Parameters ---
KP = 60.0  
KD = 3.0   
VELOCITY_X = 0.5 

# --- Gait Parameters ---
STAND_BASE = np.array([0.0, 0.9, -1.8])

# --- Dimensions per Robot ---
# 1 Freejoint (7 pos, 6 vel) + 12 Motor Joints (12 pos, 12 vel)
n_qpos_per_robot = 19 # 7 + 12
n_qvel_per_robot = 18 # 6 + 12
n_ctrl_per_robot = 12

def pd_controller(target_q, current_q, current_v, kp, kd):
    return kp * (target_q - current_q) - kd * current_v

def get_gait_target(sim_time, actuator_idx, robot_index=0):
    """
    Generates target angles.
    robot_index can be used to offset phase so they don't step in perfect unison (optional)
    """
    WALK_FREQ = 2.0
    SWING_AMP_THIGH = 0.3
    SWING_AMP_CALF  = 0.4
    
    # Optional: Add phase offset per robot so they look more natural
    phase_offset = robot_index * 0.5 
    phase = (sim_time * WALK_FREQ * 2 * np.pi + phase_offset) % (2 * np.pi)

    # Actuator Index (0-11) determines leg
    is_pair_A = (actuator_idx < 3) or (actuator_idx >= 9) # FR or RL

    if is_pair_A:
        leg_phase = phase
    else:
        leg_phase = phase + np.pi

    joint_type = actuator_idx % 3 # 0=Hip, 1=Thigh, 2=Calf
    base_angle = STAND_BASE[joint_type]
    adjustment = 0.0

    if joint_type == 1: # Thigh
        adjustment = -SWING_AMP_THIGH * np.sin(leg_phase)

    elif joint_type == 2: # Calf
        swing_lift = np.cos(leg_phase)
        if swing_lift > 0:
            adjustment = -SWING_AMP_CALF * swing_lift
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
    print(f"Loaded Scene with {num_robots} robots. Walking Trot Gait initiated...")

    # --- MAPPING (Actuator -> Joint) ---
    # This map applies to the LOCAL index (0-11) of each robot
    joint_map = [
        3, 4, 5,    # FR Actuator -> FR Joint
        0, 1, 2,    # FL Actuator -> FL Joint
        9, 10, 11,  # RR Actuator -> RR Joint
        6, 7, 8     # RL Actuator -> RL Joint
    ]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Reset keyframe 0 (home position)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        start_time = time.time()

        while viewer.is_running() and time.time() - start_time < SIM_DURATION:
            step_start = time.time()
            sim_time = data.time

            # Wait 1 second before walking
            walking_active = sim_time > 1.0

            # --- MAIN CONTROL LOOP ---
            # We loop through every robot
            for r in range(num_robots):
                
                # 1. Calculate Offsets for this specific robot
                # qpos: [Root(7) + Joints(12)] * r
                qpos_start_idx = r * n_qpos_per_robot
                # qvel: [Root(6) + Joints(12)] * r
                qvel_start_idx = r * n_qvel_per_robot
                # ctrl: [Actuators(12)] * r
                ctrl_start_idx = r * n_ctrl_per_robot

                # 2. Extract this robot's state
                # Skip the first 7 qpos (root) and first 6 qvel (root) to get just joints
                robot_joints_q = data.qpos[qpos_start_idx + 7 : qpos_start_idx + 19]
                robot_joints_v = data.qvel[qvel_start_idx + 6 : qvel_start_idx + 18]

                # 3. Calculate controls for this robot's 12 actuators
                for i in range(12):
                    # i is the local actuator index (0-11)
                    
                    # Target
                    if walking_active:
                        target_angle = get_gait_target(sim_time - 1.0, i, robot_index=r)
                    else:
                        target_angle = STAND_BASE[i % 3]

                    # Actual (mapped)
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