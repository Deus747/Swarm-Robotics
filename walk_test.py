import mujoco
import mujoco.viewer
import numpy as np
import time

# --- Configuration ---
XML_PATH = "scene.xml"
SIM_DURATION = 600.0

# --- Conservative PD Parameters ---
KP = np.array([70.0, 60.0, 55.0])
KD = np.array([3.5, 3.0, 2.8])
TORQUE_LIMITS = np.array([23.7, 23.7, 45.43])
SMOOTHING = 0.95

# --- Original Gait Parameters ---
STAND_BASE = np.array([0.0, 0.9, -1.8])
WALK_FREQ = 2.0
SWING_AMP_THIGH = 0.3
SWING_AMP_CALF = 0.4

# --- Steering Parameters ---
STEERING_COMMAND = 0.0  # -1.0 (left turn) to 1.0 (right turn), 0.0 = straight
TURN_RATE = 0.5         # Turn aggressiveness (0.0 to 1.0)
LEAN_ANGLE = 0.12       # Lean into turn [rad] for stability

# Robot dimensions
n_qpos_per_robot = 19
n_qvel_per_robot = 18
n_ctrl_per_robot = 12

class LowPassFilter:
    """Simple exponential smoothing filter"""
    def __init__(self, alpha):
        self.alpha = alpha
        self.value = None
    
    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * new_value
        return self.value

def pd_controller(target_q, current_q, current_v, kp, kd, torque_limit):
    """PD controller with torque saturation only"""
    torque = kp * (target_q - current_q) - kd * current_v
    return np.clip(torque, -torque_limit, torque_limit)

def get_gait_target(sim_time, actuator_idx, robot_index=0, steering=0.0):
    """
    Original gait logic with steering capability
    steering: -1.0 (left) to 1.0 (right), 0.0 = straight
    """
    WALK_FREQ_LOCAL = 2.0
    SWING_AMP_THIGH_LOCAL = 0.4
    SWING_AMP_CALF_LOCAL = 0.4
    
    # Leg identification
    # Actuator index to leg: 0-2=FR, 3-5=FL, 6-8=RR, 9-11=RL
    is_left_leg = (actuator_idx >= 3 and actuator_idx < 6) or (actuator_idx >= 9)
    is_pair_A = (actuator_idx < 3) or (actuator_idx >= 9)  # FR or RL
    
    # Apply steering by modulating gait frequency per side
    # Right turn: left legs move faster, right legs slower
    freq_mod = 1.0
    if steering != 0.0:
        # Left legs get +freq, right legs get -freq based on steering direction
        side_factor = 1.0 if is_left_leg else -1.0
        freq_mod = 1.0 + (TURN_RATE * steering * side_factor)
    
    # Phase calculation with optional robot offset
    phase_offset = robot_index * 0.5
    phase = (sim_time * WALK_FREQ_LOCAL * freq_mod * 2 * np.pi + phase_offset) % (2 * np.pi)

    if is_pair_A:
        leg_phase = phase
    else:
        leg_phase = phase + np.pi

    joint_type = actuator_idx % 3
    base_angle = STAND_BASE[joint_type]
    adjustment = 0.0

    if joint_type == 1:  # Thigh
        adjustment = -SWING_AMP_THIGH_LOCAL * np.sin(leg_phase)

    elif joint_type == 2:  # Calf
        swing_lift = np.cos(leg_phase)
        if swing_lift > 0:
            adjustment = -SWING_AMP_CALF_LOCAL * swing_lift
        else:
            adjustment = 0.0

    # Add lean for steering (abduction joints only)
    elif joint_type == 0:  # Abduction
        # Lean opposite to turn direction for centripetal stability
        adjustment = -steering * LEAN_ANGLE

    return base_angle + adjustment

def main():
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
    except ValueError:
        print(f"Error: Could not find {XML_PATH}.")
        return

    num_robots = model.nu // n_ctrl_per_robot
    print(f"Loaded scene with {num_robots} robots. Steering active: {STEERING_COMMAND}")

    # Original actuator→joint mapping (CRITICAL - DO NOT CHANGE)
    joint_map = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    
    # Pre-compute robot data ranges
    robot_ranges = []
    for r in range(num_robots):
        robot_ranges.append({
            'qpos_start': r * n_qpos_per_robot + 7,  # Skip root pos/quat
            'qvel_start': r * n_qvel_per_robot + 6,  # Skip root vel
            'ctrl_start': r * n_ctrl_per_robot,
            # One filter per actuator for smooth transitions
            'filters': [LowPassFilter(SMOOTHING) for _ in range(12)]
        })

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        start_time = time.time()

        while viewer.is_running() and time.time() - start_time < SIM_DURATION:
            step_start = time.time()
            sim_time = data.time
            
            # Wait 1 second before walking
            walking_active = sim_time > 1.0

            # --- Control Loop ---
            for r in robot_ranges:
                # Get joint states
                qpos = data.qpos[r['qpos_start'] : r['qpos_start'] + 12]
                qvel = data.qvel[r['qvel_start'] : r['qvel_start'] + 12]
                
                for i in range(12):
                    joint_type = i % 3
                    
                    # Get smoothed target with steering
                    raw_target = get_gait_target(sim_time - 1.0, i, 
                                                 robot_index=robot_ranges.index(r),
                                                 steering=STEERING_COMMAND) \
                                 if walking_active else STAND_BASE[joint_type]
                    
                    target = r['filters'][i].update(raw_target)
                    
                    # Get current state
                    joint_idx = joint_map[i]
                    curr_q = qpos[joint_idx]
                    curr_v = qvel[joint_idx]
                    
                    # Apply PD with torque limiting
                    torque = pd_controller(
                        target, curr_q, curr_v,
                        kp=KP[joint_type],
                        kd=KD[joint_type],
                        torque_limit=TORQUE_LIMITS[joint_type]
                    )
                    
                    data.ctrl[r['ctrl_start'] + i] = torque
            
            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()

            # Timing
            elapsed = time.time() - step_start
            if (wait_time := model.opt.timestep - elapsed) > 0:
                time.sleep(wait_time)

if __name__ == "__main__":
    main()