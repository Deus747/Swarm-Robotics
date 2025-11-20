import os
import math
import random

# ==============================================================================
# 1. FORMATION CONFIGURATION
# ==============================================================================

NUM_ROBOTS = 3

# Options: "LINE", "GRID", "CIRCLE", "POLYGON", "RANDOM", "CUSTOM", "TRIANGLE"
FORMATION_TYPE = "TRIANGLE" 

# Options: "CENTER", "OUTWARD", "FIXED", "TANGENT"
# TANGENT = Facing the direction of travel (Clockwise/Counter-Clockwise)
FACING_MODE = "FIXED"

# --- Parameters ---

# CIRCLE / POLYGON
# Note: For a 6-robot hexagon, Side Length = Radius.
CIRCLE_RADIUS = 2.1   

# LINE
LINE_SPACING_X = 1.0
LINE_SPACING_Y = 0.0

# GRID / TRIANGLE
GRID_COLS = 3         
GRID_SPACING = 1.5    

# RANDOM
RANDOM_AREA_X = 5.0   
RANDOM_AREA_Y = 5.0   

# CUSTOM
CUSTOM_POSITIONS = [
    (0, 0, 0),
    (1, 1, 45),
    (2, 0, 90),
]

# --- Visual Settings ---
HIGHLIGHT_LEADER = True
LEADER_COLOR_RGBA = "0.9 0.1 0.1 1.0"  # Red
OBSTACLE_COLOR = "0.2 0.3 0.4 1.0"

# --- Obstacle ---
ADD_OBSTACLE = False # Disabled for circle formation to prevent initial collision
OBSTACLE_POS = [5.0, 0.0, 0.6] 
OBSTACLE_SIZE = [0.6, 1.2]     

# ==============================================================================
# 2. SHARED DEFINITIONS
# ==============================================================================

MATERIALS = {
    "body_mat": "black", "hip_mat": "metal", "thigh_mat": "metal", 
    "calf_mat": "gray", "foot_mat": "black"
}

SHARED_HEADER = """
  <compiler angle="radian" meshdir="assets" autolimits="true" />
  <option cone="elliptic" impratio="100" />
  <default>
    <default class="go2">
      <geom friction="0.4" margin="0.001" condim="1"/>
      <joint axis="0 1 0" damping="0.1" armature="0.01" frictionloss="0.2"/>
      <motor ctrlrange="-23.7 23.7"/>
      <default class="abduction"><joint axis="1 0 0" range="-1.0472 1.0472"/></default>
      <default class="hip">
        <default class="front_hip"><joint range="-1.5708 3.4907"/></default>
        <default class="back_hip"><joint range="-0.5236 4.5379"/></default>
      </default>
      <default class="knee"><joint range="-2.7227 -0.83776"/><motor ctrlrange="-45.43 45.43"/></default>
      <default class="visual"><geom type="mesh" contype="0" conaffinity="0" group="2"/></default>
      <default class="collision"><geom group="3"/>
        <default class="foot"><geom size="0.022" pos="-0.002 0 -0.213" priority="1" condim="6" friction="0.4 0.02 0.01"/></default>
      </default>
    </default>
  </default>
"""

SHARED_ASSETS = f"""
  <asset>
    <material name="metal" rgba=".9 .95 .95 1" />
    <material name="black" rgba="0 0 0 1" />
    <material name="white" rgba="1 1 1 1" />
    <material name="gray" rgba="0.671705 0.692426 0.774270 1" />
    
    <material name="mat_leader"   rgba="{LEADER_COLOR_RGBA}"   shininess="0.8" reflectance="0.3"/>
    <material name="mat_obstacle" rgba="{OBSTACLE_COLOR}"      shininess="0.1"/>

    <mesh file="base_0.obj" /><mesh file="base_1.obj" /><mesh file="base_2.obj" /><mesh file="base_3.obj" /><mesh file="base_4.obj" />
    <mesh file="hip_0.obj" /><mesh file="hip_1.obj" />
    <mesh file="thigh_0.obj" /><mesh file="thigh_1.obj" /><mesh file="thigh_mirror_0.obj" /><mesh file="thigh_mirror_1.obj" />
    <mesh file="calf_0.obj" /><mesh file="calf_1.obj" /><mesh file="calf_mirror_0.obj" /><mesh file="calf_mirror_1.obj" />
    <mesh file="foot.obj" />
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texrepeat="5 5" reflectance="0.2"/>
  </asset>
"""

ROBOT_BODY_TEMPLATE = """
    <body name="go2_{i}" pos="{pos_str}" euler="{euler_str}" childclass="go2">
      <freejoint name="root_{i}"/>
      <inertial pos="0.021112 0 -0.005366" quat="-0.000543471 0.713435 -0.00173769 0.700719" mass="6.921" diaginertia="0.107027 0.0980771 0.0244531" />
      
      <geom mesh="base_0" material="{mat[body_mat]}" class="visual" />
      <geom mesh="base_1" material="{mat[body_mat]}" class="visual" />
      <geom mesh="base_2" material="{mat[body_mat]}" class="visual" />
      <geom mesh="base_3" material="white" class="visual" />
      <geom mesh="base_4" material="{mat[calf_mat]}" class="visual" />
      <geom size="0.1881 0.04675 0.057" type="box" class="collision" />
      <geom size="0.05 0.045" pos="0.285 0 0.01" type="cylinder" class="collision" />
      <geom size="0.047" pos="0.293 0 -0.06" class="collision" />
      <site name="imu_{i}" pos="-0.02557 0 0.04232" />
      
      <body name="FL_hip_{i}" pos="0.1934 0.0465 0">
        <joint name="FL_hip_joint_{i}" class="abduction" />
        <geom mesh="hip_0" material="{mat[hip_mat]}" class="visual" />
        <geom mesh="hip_1" material="{mat[calf_mat]}" class="visual" />
        <geom size="0.046 0.02" pos="0 0.08 0" quat="1 1 0 0" type="cylinder" class="collision" />
        <body name="FL_thigh_{i}" pos="0 0.0955 0">
          <joint name="FL_thigh_joint_{i}" class="front_hip" />
          <geom mesh="thigh_0" material="{mat[thigh_mat]}" class="visual" />
          <geom mesh="thigh_1" material="{mat[calf_mat]}" class="visual" />
          <geom size="0.1065 0.01225 0.017" pos="0 0 -0.1065" quat="0.707107 0 0.707107 0" type="box" class="collision" />
          <body name="FL_calf_{i}" pos="0 0 -0.213">
            <joint name="FL_calf_joint_{i}" class="knee" />
            <geom mesh="calf_0" material="{mat[calf_mat]}" class="visual" />
            <geom mesh="calf_1" material="{mat[body_mat]}" class="visual" />
            <geom size="0.012 0.06" pos="0.008 0 -0.06" quat="0.994493 0 -0.104807 0" type="cylinder" class="collision" />
            <geom size="0.011 0.0325" pos="0.02 0 -0.148" quat="0.999688 0 0.0249974 0" type="cylinder" class="collision" />
            <geom pos="0 0 -0.213" mesh="foot" class="visual" material="{mat[foot_mat]}" />
            <geom name="FL_{i}" class="foot" />
          </body>
        </body>
      </body>

      <body name="FR_hip_{i}" pos="0.1934 -0.0465 0">
        <joint name="FR_hip_joint_{i}" class="abduction" />
        <geom mesh="hip_0" material="{mat[hip_mat]}" class="visual" quat="4.63268e-05 1 0 0" />
        <geom mesh="hip_1" material="{mat[calf_mat]}" class="visual" quat="4.63268e-05 1 0 0" />
        <geom size="0.046 0.02" pos="0 -0.08 0" quat="0.707107 0.707107 0 0" type="cylinder" class="collision" />
        <body name="FR_thigh_{i}" pos="0 -0.0955 0">
          <joint name="FR_thigh_joint_{i}" class="front_hip" />
          <geom mesh="thigh_mirror_0" material="{mat[thigh_mat]}" class="visual" />
          <geom mesh="thigh_mirror_1" material="{mat[calf_mat]}" class="visual" />
          <geom size="0.1065 0.01225 0.017" pos="0 0 -0.1065" quat="0.707107 0 0.707107 0" type="box" class="collision" />
          <body name="FR_calf_{i}" pos="0 0 -0.213">
            <joint name="FR_calf_joint_{i}" class="knee" />
            <geom mesh="calf_mirror_0" material="{mat[calf_mat]}" class="visual" />
            <geom mesh="calf_mirror_1" material="{mat[body_mat]}" class="visual" />
            <geom size="0.013 0.06" pos="0.01 0 -0.06" quat="0.995004 0 -0.0998334 0" type="cylinder" class="collision" />
            <geom size="0.011 0.0325" pos="0.02 0 -0.148" quat="0.999688 0 0.0249974 0" type="cylinder" class="collision" />
            <geom pos="0 0 -0.213" mesh="foot" class="visual" material="{mat[foot_mat]}" />
            <geom name="FR_{i}" class="foot" />
          </body>
        </body>
      </body>

      <body name="RL_hip_{i}" pos="-0.1934 0.0465 0">
        <joint name="RL_hip_joint_{i}" class="abduction" />
        <geom mesh="hip_0" material="{mat[hip_mat]}" class="visual" quat="4.63268e-05 0 1 0" />
        <geom mesh="hip_1" material="{mat[calf_mat]}" class="visual" quat="4.63268e-05 0 1 0" />
        <geom size="0.046 0.02" pos="0 0.08 0" quat="0.707107 0.707107 0 0" type="cylinder" class="collision" />
        <body name="RL_thigh_{i}" pos="0 0.0955 0">
          <joint name="RL_thigh_joint_{i}" class="back_hip" />
          <geom mesh="thigh_0" material="{mat[thigh_mat]}" class="visual" />
          <geom mesh="thigh_1" material="{mat[calf_mat]}" class="visual" />
          <geom size="0.1065 0.01225 0.017" pos="0 0 -0.1065" quat="0.707107 0 0.707107 0" type="box" class="collision" />
          <body name="RL_calf_{i}" pos="0 0 -0.213">
            <joint name="RL_calf_joint_{i}" class="knee" />
            <geom mesh="calf_0" material="{mat[calf_mat]}" class="visual" />
            <geom mesh="calf_1" material="{mat[body_mat]}" class="visual" />
            <geom size="0.013 0.06" pos="0.01 0 -0.06" quat="0.995004 0 -0.0998334 0" type="cylinder" class="collision" />
            <geom size="0.011 0.0325" pos="0.02 0 -0.148" quat="0.999688 0 0.0249974 0" type="cylinder" class="collision" />
            <geom pos="0 0 -0.213" mesh="foot" class="visual" material="{mat[foot_mat]}" />
            <geom name="RL_{i}" class="foot" />
          </body>
        </body>
      </body>

      <body name="RR_hip_{i}" pos="-0.1934 -0.0465 0">
        <joint name="RR_hip_joint_{i}" class="abduction" />
        <geom mesh="hip_0" material="{mat[hip_mat]}" class="visual" quat="2.14617e-09 4.63268e-05 4.63268e-05 -1" />
        <geom mesh="hip_1" material="{mat[calf_mat]}" class="visual" quat="2.14617e-09 4.63268e-05 4.63268e-05 -1" />
        <geom size="0.046 0.02" pos="0 -0.08 0" quat="0.707107 0.707107 0 0" type="cylinder" class="collision" />
        <body name="RR_thigh_{i}" pos="0 -0.0955 0">
          <joint name="RR_thigh_joint_{i}" class="back_hip" />
          <geom mesh="thigh_mirror_0" material="{mat[thigh_mat]}" class="visual" />
          <geom mesh="thigh_mirror_1" material="{mat[calf_mat]}" class="visual" />
          <geom size="0.1065 0.01225 0.017" pos="0 0 -0.1065" quat="0.707107 0 0.707107 0" type="box" class="collision" />
          <body name="RR_calf_{i}" pos="0 0 -0.213">
            <joint name="RR_calf_joint_{i}" class="knee" />
            <geom mesh="calf_mirror_0" material="{mat[calf_mat]}" class="visual" />
            <geom mesh="calf_mirror_1" material="{mat[body_mat]}" class="visual" />
            <geom size="0.013 0.06" pos="0.01 0 -0.06" quat="0.995004 0 -0.0998334 0" type="cylinder" class="collision" />
            <geom size="0.011 0.0325" pos="0.02 0 -0.148" quat="0.999688 0 0.0249974 0" type="cylinder" class="collision" />
            <geom pos="0 0 -0.213" mesh="foot" class="visual" material="{mat[foot_mat]}" />
            <geom name="RR_{i}" class="foot" />
          </body>
        </body>
      </body>
    </body>
"""

ROBOT_ACTUATOR_TEMPLATE = """
    <motor class="abduction" name="FR_hip_{i}" joint="FR_hip_joint_{i}" />
    <motor class="hip" name="FR_thigh_{i}" joint="FR_thigh_joint_{i}" />
    <motor class="knee" name="FR_calf_{i}" joint="FR_calf_joint_{i}" />
    <motor class="abduction" name="FL_hip_{i}" joint="FL_hip_joint_{i}" />
    <motor class="hip" name="FL_thigh_{i}" joint="FL_thigh_joint_{i}" />
    <motor class="knee" name="FL_calf_{i}" joint="FL_calf_joint_{i}" />
    <motor class="abduction" name="RR_hip_{i}" joint="RR_hip_joint_{i}" />
    <motor class="hip" name="RR_thigh_{i}" joint="RR_thigh_joint_{i}" />
    <motor class="knee" name="RR_calf_{i}" joint="RR_calf_joint_{i}" />
    <motor class="abduction" name="RL_hip_{i}" joint="RL_hip_joint_{i}" />
    <motor class="hip" name="RL_thigh_{i}" joint="RL_thigh_joint_{i}" />
    <motor class="knee" name="RL_calf_{i}" joint="RL_calf_joint_{i}" />
"""

ROBOT_SENSOR_TEMPLATE = """
    <framequat name="imu_quat_{i}" objtype="site" objname="imu_{i}" />
    <gyro name="imu_gyro_{i}" site="imu_{i}" />
    <accelerometer name="imu_acc_{i}" site="imu_{i}" />
"""

OBSTACLE_TEMPLATE = """
    <body name="obstacle" pos="{x} {y} {z}">
        <geom type="cylinder" size="{radius} {height}" material="mat_obstacle" density="1000" condim="3"/>
    </body>
"""

def get_positions():
    """ Returns a list of (x, y, z, yaw) tuples based on formation type """
    positions = []
    
    if FORMATION_TYPE == "LINE":
        for i in range(NUM_ROBOTS):
            x = i * LINE_SPACING_X
            y = i * LINE_SPACING_Y
            positions.append((x, y, 0.445, 0)) 

    elif FORMATION_TYPE == "CIRCLE":
        # Distribute robots evenly on the circle
        for i in range(NUM_ROBOTS):
            # SHIFT: Add (i + 1) so Leader (Index 5) ends up at 360 (0) deg.
            # i=0 -> 60 deg, i=1 -> 120, ..., i=5 -> 360 (0)
            angle = (2 * math.pi / NUM_ROBOTS) * (i + 1)
            x = CIRCLE_RADIUS * math.cos(angle)
            y = CIRCLE_RADIUS * math.sin(angle)
            
            # Calculate Yaw
            yaw = 0
            angle_deg = math.degrees(math.atan2(y, x))
            
            if FACING_MODE == "CENTER":
                yaw = angle_deg + 180
            elif FACING_MODE == "OUTWARD":
                yaw = angle_deg
            elif FACING_MODE == "TANGENT":
                # Face the direction of movement (Counter-Clockwise)
                yaw = angle_deg + 90
                
            positions.append((x, y, 0.445, yaw))

    elif FORMATION_TYPE == "GRID":
        for i in range(NUM_ROBOTS):
            row = i // GRID_COLS
            col = i % GRID_COLS
            x = col * GRID_SPACING
            y = row * GRID_SPACING
            positions.append((x, y, 0.445, 0))
            
    elif FORMATION_TYPE == "TRIANGLE":
        # Triangle V-shape logic
        # Rows: 1, 2, 3, 4...
        # Filled from Front to Back using Decreasing Indices
        tri_spacing = GRID_SPACING
        
        pos_map = {}
        current_idx = NUM_ROBOTS - 1 # Leader (Highest Index)
        row = 0
        
        while current_idx >= 0:
            count_in_row = row + 1
            row_width = (count_in_row - 1) * tri_spacing
            y_start = -row_width / 2.0
            x_pos = -row * tri_spacing
            
            for k in range(count_in_row):
                if current_idx < 0: break
                y_pos = y_start + k * tri_spacing
                
                # Standard forward facing
                pos_map[current_idx] = (x_pos, y_pos, 0.445, 0)
                current_idx -= 1
            row += 1
            
        for i in range(NUM_ROBOTS):
            positions.append(pos_map[i])

    elif FORMATION_TYPE == "CUSTOM":
        for pos in CUSTOM_POSITIONS:
            if len(pos) == 3:
                positions.append((pos[0], pos[1], 0.445, pos[2]))
            elif len(pos) == 4:
                positions.append(pos)
                
    # Default Fallback
    if not positions:
        positions = [(0,0,0.45,0)]

    return positions

def euler_to_str(yaw_deg):
    yaw_rad = math.radians(yaw_deg)
    return f"0 0 {yaw_rad:.4f}"

def generate_files():
    positions = get_positions()
    actual_num_bots = len(positions)
    
    print(f"Generating {actual_num_bots} robots in '{FORMATION_TYPE}' formation...")

    bodies_xml = ""
    actuators_xml = ""
    sensors_xml = ""
    
    leader_index = actual_num_bots - 1

    for i, (x, y, z, yaw) in enumerate(positions):
        pos_str = f"{x:.4f} {y:.4f} {z:.4f}"
        euler_str = euler_to_str(yaw)
        
        current_materials = MATERIALS.copy()
        
        if HIGHLIGHT_LEADER and i == leader_index:
            current_materials["body_mat"] = "mat_leader"
        
        bodies_xml += ROBOT_BODY_TEMPLATE.format(
            i=i, 
            pos_str=pos_str, 
            euler_str=euler_str,
            mat=current_materials
        )
        actuators_xml += ROBOT_ACTUATOR_TEMPLATE.format(i=i)
        sensors_xml += ROBOT_SENSOR_TEMPLATE.format(i=i)

    obstacle_xml = ""
    if ADD_OBSTACLE:
        obstacle_xml = OBSTACLE_TEMPLATE.format(
            x=OBSTACLE_POS[0], 
            y=OBSTACLE_POS[1], 
            z=OBSTACLE_POS[2], 
            radius=OBSTACLE_SIZE[0],
            height=OBSTACLE_SIZE[1]
        )

    scene_content = f"""<mujoco model="go2 scene">
  {SHARED_HEADER}
  {SHARED_ASSETS}

  <statistic center="0 0 0.1" extent="0.8"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>
  
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    
    {obstacle_xml}
    {bodies_xml}
  </worldbody>

  <actuator>
    {actuators_xml}
  </actuator>

  <sensor>
    {sensors_xml}
  </sensor>
</mujoco>
"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "scene.xml")

    with open(output_path, "w") as f:
        f.write(scene_content)
    
    print(f"- scene.xml created at {output_path}")

if __name__ == "__main__":
    generate_files()