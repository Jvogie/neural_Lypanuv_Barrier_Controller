## Evaluation Script for Rattle Quaternion Barrier Function
## arguably useless now, because we have to finetune the model in MuJoCo to get it to work a better
## eval_rqbf.py

from neural_clbf.monkeystrap import monkey_patch_distutils

monkey_patch_distutils()

from pytorch_lightning import Callback
import torch
import traceback
import plotly.graph_objects as go
import os
import numpy as np
from typing import Tuple, List, Optional
import random
from neural_clbf.controllers import NeuralCLBFController
from libra.rqbf.experiments.rqbf_experiment import SixDOFVehicleRolloutExperiment
from libra.rqbf.system.rqbf import SixDOFVehicle
from neural_clbf.controllers.neural_clbf_controller import NeuralCLBFController


log_file = r"C:\Users\Tetra\Documents\Repostories\OSCorp\Project-Libra\checkpoints\six_dof_vehicle\epoch=0-val_loss=23.26.ckpt"

## Goal position for all plots
goal_position = torch.tensor([0.0, -5.0, -10.0])

## Define initial conditions for simulation
def get_start_conditions(n_dims, num_start_positions):
    """
    Generate random start conditions.
    
    Args:
        n_dims (int): Number of dimensions for each start position
        num_start_positions (int or torch.Tensor): Number of start positions to generate
    
    Returns:
        torch.Tensor: Tensor of start positions
    """
    if isinstance(num_start_positions, torch.Tensor):
        num_start_positions = num_start_positions.shape[0]
    start_x = torch.zeros(num_start_positions, n_dims)
    start_x[:, :3] = torch.rand(num_start_positions, 3) * 20 - 10  # Random positions between -10 and 10
    return start_x

def setup_experiment(neural_controller):
    num_start_positions = neural_controller.dynamics_model.start_positions.shape[0] if isinstance(neural_controller.dynamics_model.start_positions, torch.Tensor) else 4
    start_x = get_start_conditions(neural_controller.dynamics_model.n_dims, num_start_positions)
    
    return SixDOFVehicleRolloutExperiment(
        "6-DoF Vehicle Rollout",
        start_x,
        scenarios=neural_controller.scenarios,
        n_sims_per_start=1,
        t_sim=120.0, 
        goal_position=neural_controller.dynamics_model.goal_position,
    )

def generate_random_obstacle(
    goal_position: torch.Tensor,
    start_positions: torch.Tensor,
    existing_obstacles: List[Tuple[torch.Tensor, float]],
    x_range: float,
    y_range: float,
    z_range: float,
    min_radius: float = 1.0,
    max_radius: float = 2.0
) -> Optional[Tuple[torch.Tensor, float]]:
    for _ in range(10000):

        start_position = start_positions[random.randint(0, start_positions.shape[0] - 1)]
        
        t = random.uniform(0.3, 0.7)  ## Place obstacle between 30% and 70% of the way
        position = start_position + t * (goal_position - start_position)
        
        ## Add some randomness to the position
        position += torch.tensor([
            random.uniform(-x_range/8, x_range/8),
            random.uniform(-y_range/8, y_range/8),
            random.uniform(-z_range/8, z_range/8)
        ])
        
        radius = random.uniform(min_radius, max_radius)
        
        ## Check if the obstacle overlaps with the goal
        if torch.norm(position - goal_position) <= radius + 1.0:
            continue
        
        ## Check if the obstacle overlaps with start positions
        if torch.any(torch.norm(start_positions - position, dim=1) <= radius + 1.0):
            continue
        
        ## Check if the obstacle overlaps with existing obstacles
        if any(torch.norm(position - obs[0]) <= radius + obs[1] for obs in existing_obstacles):
            continue
        
        return position, radius
    
    raise ValueError("Failed to generate a valid obstacle")

def generate_random_obstacles(
    goal_position: torch.Tensor,
    start_positions: torch.Tensor,
    num_obstacles: int,
    x_range: float = 20.0,
    y_range: float = 20.0,
    z_range: float = 20.0,
    min_radius: float = 1.0,
    max_radius: float = 2.0
) -> List[Tuple[torch.Tensor, float]]:
    obstacles = []
    
    for start_position in start_positions:
        for _ in range(2):
            obstacle = generate_random_obstacle(
                goal_position, start_position.unsqueeze(0), obstacles,
                x_range, y_range, z_range, min_radius, max_radius
            )
            if obstacle is not None:
                obstacles.append(obstacle)
    
    ## Generate additional random obstacles if needed
    max_attempts = 100 * (num_obstacles - len(obstacles))
    for _ in range(max_attempts):
        if len(obstacles) >= num_obstacles:
            break
        obstacle = generate_random_obstacle(
            goal_position, start_positions, obstacles,
            x_range, y_range, z_range, min_radius, max_radius
        )
        if obstacle is not None:
            obstacles.append(obstacle)
    
    return obstacles[:num_obstacles]

class RandomizeEnvironmentCallback(Callback):
    """
    PyTorch Lightning Callback to randomize the goal position, obstacles, and starting points at the start of each epoch.
    """
    def __init__(self, dynamics_model: SixDOFVehicle, position_range: Tuple[float, float, float], num_obstacles: int, num_start_points: int):
        super().__init__()
        self.dynamics_model = dynamics_model
        self.position_range = position_range
        self.num_obstacles = num_obstacles
        self.num_start_points = num_start_points

    def on_train_epoch_start(self, trainer, pl_module):
        new_goal = self.generate_random_position()
        new_start_positions = self.generate_random_start_positions()
        new_obstacles = generate_random_obstacles(
            new_goal, new_start_positions, self.num_obstacles,
            *self.position_range
        )
        
        self.dynamics_model.set_goal(new_goal)
        self.dynamics_model.obstacles = new_obstacles
        self.dynamics_model.start_positions = new_start_positions ## type: ignore
        
        trainer.logger.log_metrics({
            "current_goal_x": new_goal[0], ## type: ignore
            "current_goal_y": new_goal[1], ## type: ignore
            "current_goal_z": new_goal[2], ## type: ignore
            "num_obstacles": len(new_obstacles),
            "num_start_points": len(new_start_positions)
        }, step=trainer.current_epoch) ## type: ignore
        
        print(f"Epoch {trainer.current_epoch}: New goal position set to {new_goal.tolist()}") ## type: ignore
        print(f"New obstacles: {new_obstacles}")
        print(f"New start positions: {new_start_positions}")

    def generate_random_position(self):
        return torch.tensor([
            random.uniform(-self.position_range[0], self.position_range[0]),
            random.uniform(-self.position_range[1], self.position_range[1]),
            random.uniform(-self.position_range[2], self.position_range[2]),
        ], dtype=torch.float32)

    def generate_random_start_positions(self):
        return torch.stack([self.generate_random_position() for _ in range(self.num_start_points)])

def setup_neural_controller():
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)
    neural_controller.clf_lambda = 1.0
    neural_controller.controller_period = 0.05
    neural_controller.dynamics_model.obstacle_buffer = 1.0 ## type: ignore
    neural_controller.dynamics_model.set_goal(goal_position) ## type: ignore
    
    ## Generate 4 random start positions
    start_positions = get_start_conditions(neural_controller.dynamics_model.n_dims, 4)[:, :3]
    neural_controller.dynamics_model.start_positions = start_positions ## type: ignore
    
    ## Generate random obstacles
    obstacles = generate_random_obstacles(
        goal_position,
        start_positions,
        num_obstacles=max(8, len(start_positions)),  ## Ensure at least two obstacles per start position
        x_range=20.0,
        y_range=20.0,
        z_range=20.0,
        min_radius=1.0,
        max_radius=2.0
    )
    neural_controller.dynamics_model.obstacles = obstacles ## type: ignore
    
    return neural_controller

def run_experiment(neural_controller):
    rollout_experiment = setup_experiment(neural_controller)
    return rollout_experiment.run(neural_controller)

def save_trajectory_data(traj_df):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    csv_file_path = os.path.join(output_dir, "trajectory_data.csv")
    traj_df.to_csv(csv_file_path, index=False)
    print(f"Trajectory data saved to {csv_file_path}")

def plot_trajectories(fig, traj_df, colors):
    for sim_index, color in zip(traj_df['sim'].unique(), colors):
        sim_data = traj_df[traj_df['sim'] == sim_index]
        
        fig.add_trace(go.Scatter3d(
            x=sim_data["$x$"], y=sim_data["$y$"], z=sim_data["$z$"],
            mode='lines',
            name=f'Trajectory {sim_index+1}',
            line=dict(color=color, width=4)
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[sim_data["$x$"].iloc[0]], y=[sim_data["$y$"].iloc[0]], z=[sim_data["$z$"].iloc[0]],
            mode='markers',
            name=f'Start {sim_index+1}',
            marker=dict(size=8, color=color, symbol='circle')
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[sim_data["$x$"].iloc[-1]], y=[sim_data["$y$"].iloc[-1]], z=[sim_data["$z$"].iloc[-1]],
            mode='markers',
            name=f'End {sim_index+1}',
            marker=dict(size=8, color=color, symbol='square')
        ))

def add_goal_to_plot(fig):
    fig.add_trace(go.Scatter3d(
        x=[goal_position[0]], y=[goal_position[1]], z=[goal_position[2]],
        mode='markers',
        name='Goal',
        marker=dict(size=15, color='gold', symbol='diamond', line=dict(color='black', width=2))
    ))

def add_obstacles_to_plot(fig, neural_controller):
    for obstacle_center, obstacle_radius in neural_controller.dynamics_model.obstacles:
        x, y, z = obstacle_center
        
        ## Create a mesh sphere
        phi, theta = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x_sphere = x + obstacle_radius * np.sin(theta) * np.cos(phi)
        y_sphere = y + obstacle_radius * np.sin(theta) * np.sin(phi)
        z_sphere = z + obstacle_radius * np.cos(theta)
        
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            colorscale=[[0, 'red'], [1, 'red']],
            opacity=0.3,
            showscale=False,
            name=f'Obstacle at ({x}, {y}, {z})'
        ))

def update_layout(fig, title):
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="z (m)",
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        legend=dict(x=0.7, y=0.9),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.update_scenes(
        xaxis=dict(range=[-100, 100]),
        yaxis=dict(range=[-100, 100]),
        zaxis=dict(range=[-100, 100])
    )

def save_and_show_plot(fig, filename):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    plot_file_path = os.path.join(output_dir, filename)
    fig.write_html(plot_file_path)
    print(f"Interactive plot saved to {plot_file_path}")
    fig.show()

def plot_six_dof_vehicle():
    neural_controller = setup_neural_controller()
    traj_df = run_experiment(neural_controller)
    save_trajectory_data(traj_df)

    fig = go.Figure()
    colors = ['blue', 'orange', 'green', 'red']
    
    plot_trajectories(fig, traj_df, colors)
    add_goal_to_plot(fig)
    add_obstacles_to_plot(fig, neural_controller)
    update_layout(fig, "6-DoF Vehicle Trajectories under Neural CLBF Controller")
    save_and_show_plot(fig, "trajectory_plot.html")

def plot_six_dof_vehicle_with_barrier():
    neural_controller = setup_neural_controller()
    traj_df = run_experiment(neural_controller)

    fig = go.Figure()
    all_points = torch.cat((get_start_conditions(neural_controller.dynamics_model.n_dims, neural_controller.dynamics_model.start_positions)[:, :3], goal_position.unsqueeze(0)), dim=0) # type: ignore
    min_x, min_y, min_z = all_points.min(dim=0).values
    max_x, max_y, max_z = all_points.max(dim=0).values

    padding = 50.0
    min_x, min_y, min_z = min_x - padding, min_y - padding, min_z - padding
    max_x, max_y, max_z = max_x + padding, max_y + padding, max_z + padding

    x = torch.linspace(min_x, max_x, 50) ## type: ignore
    y = torch.linspace(min_y, max_y, 50) ## type: ignore
    z = torch.linspace(min_z, max_z, 50) ## type: ignore
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
    grid_points = torch.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), dim=-1)

    padded_grid_points = torch.zeros(grid_points.shape[0], neural_controller.dynamics_model.n_dims)
    padded_grid_points[:, :3] = grid_points
    
    with torch.no_grad():
        barrier_values = neural_controller.V(padded_grid_points).reshape(grid_x.shape)

    barrier_min = float(barrier_values.min().item())
    barrier_max = float(barrier_values.max().item())

    fig.add_trace(go.Isosurface(
        x=grid_x.flatten().numpy(),
        y=grid_y.flatten().numpy(),
        z=grid_z.flatten().numpy(),
        value=barrier_values.flatten().numpy(),
        isomin=barrier_min,
        isomax=barrier_max,
        surface_count=5,
        colorscale='Viridis',
        opacity=0.3,
        name='Barrier Function'
    ))

    colors = ['blue', 'orange', 'green', 'red']
    plot_trajectories(fig, traj_df, colors)
    add_goal_to_plot(fig)
    add_obstacles_to_plot(fig, neural_controller)
    update_layout(fig, "6-DoF Vehicle Trajectories with Neural Barrier Function (Zoomed Out)")
    save_and_show_plot(fig, "barrier_trajectory_plot.html")

def plot_six_dof_vehicle_fake_paraboloid():
    neural_controller = setup_neural_controller()
    traj_df = run_experiment(neural_controller)
    save_trajectory_data(traj_df)

    fig = go.Figure()

    start_x = get_start_conditions(neural_controller.dynamics_model.n_dims, neural_controller.dynamics_model.start_positions) # type: ignore
    avg_start_x = start_x[:, 0].mean().item()
    avg_start_y = start_x[:, 1].mean().item()
    avg_start_z = start_x[:, 2].mean().item()

    radius = 18
    k = 0.04
    vertical_shift = -3

    r = np.linspace(0, radius, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    R, THETA = np.meshgrid(r, theta)

    X = R * np.cos(THETA) + avg_start_x
    Y = R * np.sin(THETA) + avg_start_y
    Z = k * (R**2) + vertical_shift + min(avg_start_z, goal_position[2].item())

    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis', opacity=0.3, name='Paraboloid',
        showscale=False
    ))

    colors = ['blue', 'orange', 'green', 'red']
    plot_trajectories(fig, traj_df, colors)
    add_goal_to_plot(fig)
    add_obstacles_to_plot(fig, neural_controller)
    update_layout(fig, "6-DoF Vehicle Trajectories with Lowered Paraboloid Barrier")
    save_and_show_plot(fig, "paraboloid_trajectory_plot.html")

def plot_six_dof_vehicle_with_lyapunov():
    neural_controller = setup_neural_controller()
    traj_df = run_experiment(neural_controller)

    fig = go.Figure()

    all_points = torch.cat((get_start_conditions(neural_controller.dynamics_model.n_dims, neural_controller.dynamics_model.start_positions)[:, :3], goal_position.unsqueeze(0)), dim=0) # type: ignore
    min_x, min_y, min_z = all_points.min(dim=0).values
    max_x, max_y, max_z = all_points.max(dim=0).values

    padding = 20.0
    min_x, min_y, min_z = min_x - padding, min_y - padding, min_z - padding
    max_x, max_y, max_z = max_x + padding, max_y + padding, max_z + padding

    x = torch.linspace(min_x, max_x, 50) ## type: ignore
    y = torch.linspace(min_y, max_y, 50) ## type: ignore
    z = torch.linspace(min_z, max_z, 50) ## type: ignore
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
    grid_points = torch.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), dim=-1)

    padded_grid_points = torch.zeros(grid_points.shape[0], neural_controller.dynamics_model.n_dims)
    padded_grid_points[:, :3] = grid_points
    
    with torch.no_grad():
        lyapunov_values, _ = neural_controller.V_with_jacobian(padded_grid_points)
        lyapunov_values = lyapunov_values.reshape(grid_x.shape)

    lyapunov_min, lyapunov_max = float(lyapunov_values.min().item()), float(lyapunov_values.max().item())

    fig.add_trace(go.Isosurface(
        x=grid_x.flatten().numpy(),
        y=grid_y.flatten().numpy(),
        z=grid_z.flatten().numpy(),
        value=lyapunov_values.flatten().numpy(),
        isomin=lyapunov_min,
        isomax=lyapunov_max,
        surface_count=10,
        colorscale='Viridis',
        opacity=0.7,
        name='Lyapunov Function'
    ))

    colors = ['blue', 'orange', 'green', 'red']
    plot_trajectories(fig, traj_df, colors)
    add_goal_to_plot(fig)
    add_obstacles_to_plot(fig, neural_controller)
    fig.update_layout(
        title="6-DoF Vehicle Trajectories with Lyapunov Function",
        scene=dict(
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="z (m)",
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1),
                up=dict(x=0, y=0, z=1)
            ),
            xaxis=dict(range=[min_x.item(), max_x.item()]),
            yaxis=dict(range=[min_y.item(), max_y.item()]),
            zaxis=dict(range=[min_z.item(), max_z.item()])
        ),
        legend=dict(x=0.7, y=0.9),
        width=1200,
        height=1000,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    save_and_show_plot(fig, "lyapunov_trajectory_plot.html")

if (__name__ == "__main__"):
    try:
        plot_functions = [
            plot_six_dof_vehicle,
            plot_six_dof_vehicle_with_lyapunov,
            plot_six_dof_vehicle_with_barrier,
            plot_six_dof_vehicle_fake_paraboloid
        ]
        
        for plot_func in plot_functions:
            try:
                plot_func()
            except KeyboardInterrupt:
                print(f"User interrupted {plot_func.__name__}. Exiting...")
                break
            except Exception as error:
                print(f"Error in {plot_func.__name__}:")
                print(traceback.format_exc())
    except KeyboardInterrupt:
        print("User interrupted the script. Exiting...")


## Not sure when this was added, but I suspect it was added by Brendan during his work on the MuJoCo finetuning
class RandomGoalCallback:
    """Placeholder for backward compatibility"""
    pass



