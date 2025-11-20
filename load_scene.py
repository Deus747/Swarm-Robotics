import mujoco
import mujoco.viewer
import time

# 1. Load the model and data
try:
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    print("Successfully loaded scene.xml")
except ValueError:
    print("Error: Could not find or parse 'scene.xml'. Make sure you ran generate_scene.py first.")
    exit()

# 2. Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    
    # 3. Simulation Loop
    while viewer.is_running():
        step_start = time.time()

        # Step the physics
        mujoco.mj_step(model, data)

        # Sync data to viewer
        viewer.sync()

        # Try to maintain real-time speed
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)