import sys
import os
# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(project_root)

# Rest of your imports
import mujoco
import mujoco.viewer
import numpy as np
import time
from LQR_controller import create_astrobee_lqr, Obstacle

# Load the model
model_path = r"C:\Users\jack\Documents\GitHub\Project-Libra\neural_clbf\libra\Other_controller\astrobee_deux.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Create the LQR controller
lqr = create_astrobee_lqr()

# Simulation parameters
max_duration = 240
dt = model.opt.timestep
slowdown_factor = 1.0  # Increased for smoother visualization

# Initial position (positions followed by orientations)
start_pos = np.array([-6, -2, 1, 0, 0, 0])
data.qpos[:] = start_pos
data.qvel[:] = np.zeros(6)

# Print initial state for debugging
print(f"Initial position: {data.qpos[:3]}")
print(f"Initial velocity: {data.qvel[:3]}")

# Goal state (positions followed by orientations and zero velocities)
goal = np.array([5, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Create obstacles
obstacles = [
    Obstacle(position=[0, 1, 0], size=[1.0, 1.0, 1.0], shape='box'),
]

# Threshold to determine if the goal is reached
goal_threshold = 0.5

# Position and velocity limits
pos_limit = 10.0
vel_limit = 2.0

# Create a viewer object for visualization
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Render the initial state
    viewer.sync()
    time.sleep(1)  # Pause to visualize the initial state

    # Main simulation loop
    for t in range(int(max_duration / dt)):
        # Current state (12-dimensional: pos, orientation, vel, angular vel)
        pos = data.qpos[:3]
        orientation = data.qpos[3:6]
        vel = data.qvel[:3]
        ang_vel = data.qvel[3:6]
        state = np.concatenate([pos, orientation, vel, ang_vel])

        # Compute control input using LQR with obstacle avoidance
        u = lqr.get_control(state, obstacles, goal)

        # Apply control to the actuators
        data.ctrl[:] = u

        # Step the simulation
        mujoco.mj_step(model, data)

        # Apply position and velocity limits
        data.qpos[:3] = np.clip(data.qpos[:3], -pos_limit, pos_limit)
        data.qvel[:3] = np.clip(data.qvel[:3], -vel_limit, vel_limit)
        data.qvel[3:6] = np.clip(data.qvel[3:6], -vel_limit, vel_limit)

        # Update the viewer
        viewer.sync()
        if t % 100 == 0:  # Update camera less frequently
            viewer.cam.azimuth += 1
            viewer.cam.distance = 15  # Increased camera distance

        # Slow down the simulation for better visualization
        time.sleep(dt * slowdown_factor)

        # Calculate the distance to the goal and obstacle
        distance_to_goal = np.linalg.norm(pos - goal[:3])
        distance_to_obstacle = obstacles[0].distance(pos)

        # Periodic logging for debugging
        if t % 100 == 0:  # Reduced logging frequency
            print(f"Time: {t*dt:.2f}s")
            print(f"Position: {pos}, Velocity: {vel}")
            print(f"Distance to goal: {distance_to_goal:.2f}, Distance to obstacle: {distance_to_obstacle:.2f}")
            print(f"Control input: {u}")
            if lqr.path is not None and lqr.waypoint_index < len(lqr.path):
                print(f"Current waypoint: {lqr.path[lqr.waypoint_index]}")
            print("---")

        # Check if the goal is reached
        if distance_to_goal < goal_threshold:
            print(f"Goal reached at time {t*dt:.2f}s!")
            break

    # Final state
    print(f"Final position: {data.qpos[:3]}")
    print(f"Final velocity: {data.qvel[:3]}")

    # Keep the viewer open until user closes it
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.1)
