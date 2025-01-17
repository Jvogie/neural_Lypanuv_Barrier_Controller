import mujoco
import mujoco.viewer
import numpy as np
import time
from mpc_controller import MPCController
from global_planner import Obstacle

# Load the model
model_path = r"C:\Users\jack\Documents\GitHub\Project-Libra\neural_clbf\libra\Other_controller\astrobee_deux.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Create the MPC controller
A = np.eye(12)
dt = model.opt.timestep
A[0:3, 6:9] = np.eye(3) * dt
A[3:6, 9:12] = np.eye(3) * dt

B = np.zeros((12, 6))
B[6:9, 0:3] = np.eye(3) * dt
B[9:12, 3:6] = np.eye(3) * dt

Q = np.eye(12)
Q[0:3, 0:3] *= 10
Q[2, 2] = 20    # Reduced weight for z-position (index 2) from 50 to 30
Q[3:6, 3:6] *= 1
Q[6:9, 6:9] *= 1
Q[9:12, 9:12] *= 1

R = np.eye(6) * 1.5    # Increased control effort penalties

mpc = MPCController(A, B, Q, R, N=10)

# Simulation parameters
max_duration = 240
dt = model.opt.timestep
slowdown_factor = 1.0

# Initial position and velocity
start_pos = np.array([-6, -2, 1, 0, 0, 0])
data.qpos[:] = start_pos
data.qvel[:] = np.zeros(6)

# Goal state
goal = np.array([5, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Create obstacles
obstacles = [
    Obstacle(position=[0, 1, 0], size=[1.0, 1.0, 1.0], shape='box'),
]

goal_threshold = 0.5
pos_limit = 10.0
vel_limit = 2.0

# Create a viewer object for visualization
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.sync()
    time.sleep(1)

    for t in range(int(max_duration / dt)):
        state = np.concatenate([data.qpos, data.qvel])

        # Compute control input using MPC
        u = mpc.compute_mpc_control(state, goal, obstacles)

        # Apply control to actuators
        data.ctrl[:] = u

        # Step the simulation
        mujoco.mj_step(model, data)

        # Apply position and velocity limits
        data.qpos[:3] = np.clip(data.qpos[:3], -pos_limit, pos_limit)
        data.qvel[:] = np.clip(data.qvel, -vel_limit, vel_limit)

        # Update the viewer
        viewer.sync()
        if t % 100 == 0:
            viewer.cam.azimuth += 1
            viewer.cam.distance = 15

        time.sleep(dt * slowdown_factor)

        # Calculate distances
        pos = data.qpos[:3]
        velocity = data.qvel[:3]
        distance_to_goal = np.linalg.norm(pos - goal[:3])
        distance_to_obstacle = obstacles[0].distance(pos)

        # Logging
        if t % 100 == 0:
            print(f"Time: {t*dt:.2f}s")
            print(f"Position: {pos}, Velocity: {velocity}")
            print(f"Distance to goal: {distance_to_goal:.2f}, Distance to obstacle: {distance_to_obstacle:.2f}")
            print(f"Control input: {u}")
            print("---")

        # Check if goal is reached
        if distance_to_goal < goal_threshold:
            print(f"Goal reached at time {t*dt:.2f}s!")
            break

    # Final state
    print(f"Final position: {data.qpos[:3]}")
    print(f"Final velocity: {data.qvel[:3]}")

    # Keep the viewer open
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.1)
