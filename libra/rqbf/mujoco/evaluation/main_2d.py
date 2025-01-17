## main_2d.py
## not really used
## does not do obstacles or anything useful, just inital training for 2d
## does not use the finetuned model

import mujoco
import numpy as np
import glfw
import torch
import os
from typing import List, Tuple
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R

from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.systems.utils import ScenarioList
from libra.rqbf.system.rqbf import SixDOFVehicle

xml_path = os.path.join(os.path.dirname(__file__), "astrobee.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

if not glfw.init():
    raise Exception("Could not initialize GLFW")

window = glfw.create_window(800, 600, "3D MuJoCo Simulation", None, None)
if not window:
    glfw.terminate()
    raise Exception("Could not create GLFW window")

glfw.make_context_current(window)

context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
scene = mujoco.MjvScene(model, maxgeom=1000)

cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.trackbodyid = 0
cam.distance = 10.0
cam.azimuth = 0
cam.elevation = -90  ## Top-down view for 2D visualization

opt = mujoco.MjvOption()

SIMULATION_STEPS = 6000
DT = 0.01
roller
log_file = r"C:\Users\Tetra\Documents\Repostories\OSCorp\Project-Libra\checkpoints\six_dof_vehicle\epoch=0-val_loss=6.20.ckpt"
neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)
neural_controller.clf_lambda = 1.0
neural_controller.controller_period = 0.05

goal_position = torch.tensor([1.0, -3.0, 0.0])
neural_controller.dynamics_model.set_goal(goal_position)

# Set up the dynamics model
nominal_params = {
    "mass": torch.tensor(1.0),
    "inertia_matrix": torch.eye(3),
    "gravity": torch.tensor([0.0, 0.0, 0.0]),
}
scenarios: ScenarioList = [nominal_params]
dynamics_model = SixDOFVehicle(
    nominal_params,
    dt=DT,
    controller_dt=neural_controller.controller_period,
    scenarios=scenarios,
)

def get_state(data: mujoco.MjData) -> np.ndarray:
    position = data.qpos[:3]  # x, y, z positions
    roll_pitch_yaw = data.qpos[3:6]  # roll, pitch, yaw angles

    # Convert roll, pitch, yaw to quaternion
    r = R.from_euler('xyz', roll_pitch_yaw)
    quaternion = r.as_quat()  # Returns [x, y, z, w]
    # Reorder to [w, x, y, z]
    quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])

    linear_velocity = data.qvel[:3]
    angular_velocity = data.qvel[3:6]
    
    ## input state
    state = np.zeros(19)
    state[0:3] = position
    state[3:7] = quaternion
    state[7:10] = linear_velocity
    state[10:13] = angular_velocity
    state[13:16] = 0.0 
    state[16:19] = 0.0  
    return state

def run_simulation(start_positions: List[Tuple[float, float, float]], goal_position: Tuple[float, float, float]) -> List[List[np.ndarray]]:
    trajectories = []

    for start_pos in start_positions:
        trajectory = []

        ## Reset the simulation
        mujoco.mj_resetData(model, data)

        ## Set initial position
        data.qpos[:3] = start_pos

        ## Set initial orientation (zero roll, pitch, yaw)
        data.qpos[3:6] = [0, 0, 0]

        ## Explicitly set initial velocity to zero, otherwise it'll use the last velocity
        data.qvel[:] = 0 

        goal_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        if goal_body_id >= 0:
            model.body_pos[goal_body_id] = goal_position
        else:
            print("Warning: 'goal' body not found in the Mujoco model.")

        for step in range(SIMULATION_STEPS):
            state = get_state(data)
            trajectory.append(state)

            ## Convert state to tensor and get control from Neural CLBF
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            control = neural_controller.u(state_tensor).squeeze().detach().numpy()

            data.ctrl[:] = control

            mujoco.mj_step(model, data)

            mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
            viewport = mujoco.MjrRect(0, 0, 800, 600)
            mujoco.mjr_render(viewport, scene, context)
            glfw.swap_buffers(window)
            glfw.poll_events()

            if glfw.window_should_close(window):
                break

        trajectories.append(trajectory)

    return trajectories

def plot_trajectories_2d(trajectories: List[List[np.ndarray]], goal_position: Tuple[float, float]):
    fig = go.Figure()

    colors = ['blue', 'red', 'green', 'purple']
    
    for i, trajectory in enumerate(trajectories):
        trajectory = np.array(trajectory)
        fig.add_trace(go.Scatter(
            x=trajectory[:, 0], y=trajectory[:, 1],
            mode='lines',
            name=f'Trajectory {i+1}',
            line=dict(color=colors[i], width=4)
        ))
        
        fig.add_trace(go.Scatter(
            x=[trajectory[0, 0]], y=[trajectory[0, 1]],
            mode='markers',
            name=f'Start {i+1}',
            marker=dict(size=8, color=colors[i], symbol='circle')
        ))
        
        fig.add_trace(go.Scatter(
            x=[trajectory[-1, 0]], y=[trajectory[-1, 1]],
            mode='markers',
            name=f'End {i+1}',
            marker=dict(size=8, color=colors[i], symbol='square')
        ))

    fig.add_trace(go.Scatter(
        x=[goal_position[0]], y=[goal_position[1]],
        mode='markers',
        name='Goal',
        marker=dict(size=10, color='gold', symbol='diamond')
    ))

    threshold = 0.5 
    for i, trajectory in enumerate(trajectories):
        trajectory = np.array(trajectory)
        distances = np.linalg.norm(trajectory[:, 0:2] - np.array(goal_position), axis=1)
        within_threshold = distances <= threshold
        fig.add_trace(go.Scatter(
            x=trajectory[within_threshold, 0], y=trajectory[within_threshold, 1],
            mode='markers',
            name=f'Within {threshold}m Goal {i+1}',
            marker=dict(size=6, color=colors[i], symbol='star')
        ))

    fig.update_layout(
        xaxis_title="X",
        yaxis_title="Y",
        title="2D Visualization of Trajectories under Neural CLBF Controller",
        legend=dict(x=0.7, y=0.9),
    )

    fig.show()

if __name__ == "__main__":
    start_positions = [
        (5.0, 0.0, 0.0),
        (-5.0, 0.0, 0.0),
        (0.0, 5.0, 0.0),
        (0.0, -5.0, 0.0)
    ]
    goal_position_list = goal_position.tolist()

    trajectories = run_simulation(start_positions, goal_position_list)

    for i, trajectory in enumerate(trajectories):
        final_position = trajectory[-1][0:2]  # Get x, y
        error = np.linalg.norm(np.array(goal_position_list)[:2] - final_position)  # 2D error
        print(f"Agent {i+1} final position: {final_position}, Error: {error:.2f} meters")

    plot_trajectories_2d(trajectories, goal_position_list[:2])

    glfw.terminate()