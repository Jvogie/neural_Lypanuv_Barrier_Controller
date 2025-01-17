import mujoco
import numpy as np
import glfw
import torch
import os
from typing import List, Tuple, Dict, Any, Type
import logging
from math import cos, sin, pi
import random
import imageio
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(project_root)

# Import controllers and utilities
from neural_clbf.controllers import NeuralCLBFController
from libra.rqbf.system.rqbf import SixDOFVehicle
from libra.rqbf.system.rrt import rrt_plan
from libra.Other_controller.LQR.LQR_wrapper import LQRControllerWrapper
from libra.Other_controller.MPC.mpc_wrapper import MPCControllerWrapper

# Import all the necessary functions from main_3d
from libra.rqbf.mujoco.evaluation.main_3d import (
    calculate_safety_score, plot_trajectories_3d, setup_mujoco, setup_glfw, setup_mujoco_visualization, get_state,
    calculate_fuel_cost, generate_random_obstacles, setup_neural_controller,
    setup_dynamics_model, configure_dynamics, add_obstacles_to_astrobee_xml
)



@dataclass
class SimulationConfig:
    """Configuration class for simulation parameters"""
    cam_pos: List[float] = None
    cam_rot: List[float] = None
    zoom_level: float = 40.0
    astrobee_radius: float = 0.26
    obstacle_buffer_distance: float = 0.2
    goal_threshold: float = 1.5
    translation_fuel_cost: float = 1.0
    rotation_fuel_cost: float = 0.5
    velocity_limit: float = 0.1
    force_limit: float = 0.849
    torque_limit: float = 0.1
    model_timestep: float = 0.1
    max_simulation_steps: int = 6000
    debug_frequency: int = 50
    render_frequency: int = 4
    recording: bool = False
    
    def __post_init__(self):
        if self.cam_pos is None:
            self.cam_pos = [0, -40, 20]
        if self.cam_rot is None:
            self.cam_rot = [pi/4, 0]


class SimulationEnvironment:
    """Simulation environment setup and management"""
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.model = None
        self.data = None
        self.window = None
        self.context = None
        self.scene = None
        self.cam = None
        self.opt = None
        self.viewport = None
        self.video_writer = None
        self.frame_count = 0
        self.render_frame = 0
        
    def setup(self):
        self.model, self.data = setup_mujoco()
        self.window = setup_glfw()
        self.context, self.scene, self.cam, self.opt = setup_mujoco_visualization(self.model)
        self.viewport = mujoco.MjrRect(0, 0, 1920, 1080)
        
    def cleanup(self):
        if self.video_writer:
            self.video_writer.close()
        glfw.terminate()


def check_collision(state: np.ndarray, static_obstacles: List[Tuple[np.ndarray, float]], config: SimulationConfig) -> bool:
    """
    Extract the collision check logic from main_3d.py's run_simulation function
    """
    position = state[:3]
    for obs_pos, obs_radius in static_obstacles:
        distance_to_obstacle = np.linalg.norm(position - obs_pos)
        if distance_to_obstacle <= obs_radius + config.astrobee_radius:
            return True
    return False

def check_goal_reached(position: np.ndarray, goal_position: Tuple[float, float, float], config: SimulationConfig) -> bool:
    """
    Extract the goal check logic from main_3d.py's run_simulation function
    """
    distance_to_goal = np.linalg.norm(position - np.array(goal_position))
    return distance_to_goal <= config.goal_threshold

def render_frame(env: SimulationEnvironment) -> None:
    """
    Extract the rendering logic from main_3d.py's run_simulation function
    """
    try:
        mujoco.mjv_updateScene(env.model, env.data, env.opt, None, env.cam, mujoco.mjtCatBit.mjCAT_ALL, env.scene)
        mujoco.mjr_render(env.viewport, env.scene, env.context)

        if env.config.recording:
            pixels = np.zeros((env.viewport.height, env.viewport.width, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(pixels, None, env.viewport, env.context)
            pixels = np.flipud(pixels)
            env.video_writer.append_data(pixels)
            env.frame_count += 1

        glfw.swap_buffers(env.window)
    except Exception as e:
        if env.config.recording:
            logging.error(f"Error during rendering: {str(e)}")
            # continue
        else:
            raise

def run_simulation_controllers(
    controller_type: str,
    start_positions: List[Tuple[float, float, float]],
    goal_position: Tuple[float, float, float],
    static_obstacles: List[Tuple[np.ndarray, float]],
    config: SimulationConfig,
    neural_controller: NeuralCLBFController = None,
    lqr_controller: LQRControllerWrapper = None,
    mpc_controller: MPCControllerWrapper = None,
) -> List[Dict]:
    """Run simulation with specified controller"""
    env = SimulationEnvironment(config)
    env.setup()
    simulation_results = []

    try:
        for idx, start_pos in enumerate(start_positions):
            agent_id = idx + 1
            logging.info(f"Starting simulation for Agent {agent_id} with {controller_type}")

            trajectory = []
            collision_occurred = False
            total_fuel_cost = 0.0
            control_history = []
            collision_details = {
                'collision_time': None,
                'collision_position': None,
                'collision_obstacle_id': None
            }

            # Reset simulation state
            mujoco.mj_resetData(env.model, env.data)
            mujoco.mj_forward(env.model, env.data)
            
            env.data.qpos[:3] = start_pos
            env.data.qpos[3:6] = [0, 0, 0]
            env.data.qvel[:] = 0

            simulation_time = 0.0
            
            for step in range(config.max_simulation_steps):
                if glfw.window_should_close(env.window):
                    break

                state = get_state(env.data)
                trajectory.append(state.copy())

                # Get control based on controller type
                if controller_type == "neural_clbf":
                    with torch.no_grad():
                        x_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                        control = neural_controller.u(x_tensor).squeeze().detach().numpy()
                elif controller_type == "lqr":
                    control = lqr_controller.get_control(state, np.array(goal_position), static_obstacles)
                elif controller_type == "mpc":
                    control = mpc_controller.get_control(state, np.array(goal_position), static_obstacles)

                # Apply control limits
                control = np.clip(control, -config.force_limit, config.force_limit)
                control[3:] = np.clip(control[3:], -config.torque_limit, config.torque_limit)

                # Calculate fuel cost
                fuel_cost_config = {
                    'translation_fuel_cost': config.translation_fuel_cost,
                    'rotation_fuel_cost': config.rotation_fuel_cost
                }
                step_fuel_cost = calculate_fuel_cost(control, fuel_cost_config)
                total_fuel_cost += step_fuel_cost
                control_history.append(control)

                # Apply control and step physics
                env.data.ctrl[:6] = control
                env.data.ctrl[6:] = 0

                # Double physics step like in main_3d
                if controller_type == "mpc":
                    for _ in range(10):
                        mujoco.mj_step(env.model, env.data)
                else:
                    for _ in range(2):
                        mujoco.mj_step(env.model, env.data)
                simulation_time += env.model.opt.timestep

                # Check for collision and goal
                if check_collision(state, static_obstacles, config):
                    collision_occurred = True
                    collision_details.update({
                        'collision_time': simulation_time,
                        'collision_position': state[:3].tolist(),
                        'collision_obstacle_id': len(static_obstacles)
                    })
                    break

                if check_goal_reached(state[:3], goal_position, config):
                    logging.info(f"Agent {agent_id} reached goal")
                    break

                # Render
                if env.render_frame % config.render_frequency == 0:
                    render_frame(env)

                env.render_frame += 1
                glfw.poll_events()

            simulation_results.append({
                'agent_id': agent_id,
                'controller_type': controller_type,
                'start_position': start_pos,
                'trajectory': trajectory,
                'collision': collision_occurred,
                'collision_details': collision_details,
                'total_fuel_cost': total_fuel_cost,
                'control_history': control_history,
                'simulation_time': simulation_time,
                'final_position': trajectory[-1][:3].tolist() if trajectory else None
            })

    finally:
        env.cleanup()

    return simulation_results

def plot_all_trajectories(
    trajectories: List[List[List[np.ndarray]]], 
    goal_position: Tuple[float, float, float], 
    obstacles: List[Tuple[np.ndarray, float]], 
    object_radius: float, 
    efficiencies: List[List[float]], 
    fuel_costs: List[List[float]], 
    safety_scores: List[List[Tuple[float, Dict[str, float]]]],
    config: dict,
    controller_names: List[str] = ['neural_clbf', 'lqr', 'mpc']
):
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scene", "rowspan": 2}, {"type": "table"}],
               [None, {"type": "table"}]],
        subplot_titles=(f'3D Trajectories of All Controllers', 'Performance Metrics', 'Safety Metrics'),
        vertical_spacing=0.02,
        horizontal_spacing=0.02
    )

    ## Calculate plot bounds
    all_x, all_y, all_z = [], [], []
    
    ## 3D Trajectory Plot
    # colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']
    colors = ['green', 'blue', 'purple', 'orange', 'cyan', 'red']
    for k in range(len(trajectories)):
        for i, trajectory in enumerate(trajectories[k]):
            if not trajectory:
                continue
            points = np.array(trajectory)[:, :3]
            agent_color = colors[k % len(colors)]
            
            all_x.extend(points[:, 0])
            all_y.extend(points[:, 1])
            all_z.extend(points[:, 2])
            
            fig.add_trace(
                go.Scatter3d(
                    x=points[:, 0], y=points[:, 1], z=points[:, 2],
                    mode='lines',
                    name=f'{controller_names[k].upper()} (Agent {i+1} Path)',
                    line=dict(width=5, color=agent_color),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            ## Add start point
            fig.add_trace(
                go.Scatter3d(
                    x=[points[0, 0]], y=[points[0, 1]], z=[points[0, 2]],
                    mode='markers',
                    name=f'{controller_names[k].upper()} (Agent {i+1} Start)',
                    marker=dict(
                        size=10,
                        color=agent_color,
                        symbol='circle',
                        line=dict(color='white', width=2)
                    ),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            ## Add end point
            fig.add_trace(
                go.Scatter3d(
                    x=[points[-1, 0]], y=[points[-1, 1]], z=[points[-1, 2]],
                    mode='markers',
                    name=f'{controller_names[k].upper()} (Agent {i+1} End)',
                    marker=dict(
                        size=10,
                        color=agent_color,
                        symbol='square',
                        line=dict(color='white', width=2)
                    ),
                    showlegend=True
                ),
                row=1, col=1
            )

    ## Add obstacles and collect their coordinates for axis scaling
    for obs_pos, obs_radius in obstacles:
        x, y, z = obs_pos
        all_x.extend([x - obs_radius, x + obs_radius])
        all_y.extend([y - obs_radius, y + obs_radius])
        all_z.extend([z - obs_radius, z + obs_radius])
        
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_sphere = x + obs_radius * np.outer(np.cos(u), np.sin(v))
        y_sphere = y + obs_radius * np.outer(np.sin(u), np.sin(v))
        z_sphere = z + obs_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(
            go.Surface(
                x=x_sphere, y=y_sphere, z=z_sphere,
                colorscale=[[0, 'red'], [1, 'red']],
                opacity=0.5,
                showscale=False,
                name='Obstacle'
            ),
            row=1, col=1
        )

    ## Add goal marker and its coordinates
    all_x.append(goal_position[0])
    all_y.append(goal_position[1])
    all_z.append(goal_position[2])
    
    fig.add_trace(
        go.Scatter3d(
            x=[goal_position[0]], y=[goal_position[1]], z=[goal_position[2]],
            mode='markers',
            name='Goal',
            marker=dict(size=10, color='gold', symbol='diamond'),
            showlegend=True
        ),
        row=1, col=1
    )
    
    ## Calculate axis ranges to be equal
    for k in range(len(trajectories)):
        max_range = max(
            max(all_x) - min(all_x),
            max(all_y) - min(all_y),
            max(all_z) - min(all_z)
        ) / 2.0
        
        mid_x = (max(all_x) + min(all_x)) / 2
        mid_y = (max(all_y) + min(all_y)) / 2
        mid_z = (max(all_z) + min(all_z)) / 2

    ## Performance Metrics Table
    perf_headers = ['Controller', 'Path Efficiency', 'Fuel Cost', 'Distance Traveled']
    perf_data = []
    # Initialize accumulators for averages
    
    num_agents = len(trajectories)
    for k in range(len(trajectories)):
        total_efficiency = 0.0
        total_fuel_cost = 0.0
        total_distance = 0.0
        num_agents = len(trajectories[k])
        for i in range(len(trajectories[k])):
            if i < len(trajectories[k]) and trajectories[k][i]:
                points = np.array(trajectories[k][i])[:, :3]
                distance = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
                total_efficiency += efficiencies[k][i]
                total_fuel_cost += fuel_costs[k][i]
                total_distance += distance

        # Calculate averages
        avg_efficiency = total_efficiency / num_agents if num_agents > 0 else 0
        avg_fuel_cost = total_fuel_cost / num_agents if num_agents > 0 else 0
        avg_distance = total_distance / num_agents if num_agents > 0 else 0

        perf_data.append([
            f'{controller_names[k]}',
            f'{avg_efficiency:.2%}',
            f'{avg_fuel_cost:.2f}',
            f'{avg_distance:.2f}m'
        ])


    fig.add_trace(
        go.Table(
            header=dict(values=perf_headers, align='center'),
            cells=dict(values=list(zip(*perf_data)), align='center')
        ),
        row=1, col=2
    )

    # ## Safety Metrics Table
    # safety_headers = ['Agent', 'Overall Safety', 'Velocity Safety', 'Obstacle Safety']
    # safety_data = [
    #     [f'{controller_names[i]}', f'{scores.mean():.2%}', f'{metrics["velocity_score"].mean():.2%}', f'{metrics["obstacle_score"].mean():.2%}']
    #     for i, (scores, metrics) in enumerate(safety_scores)
    # ]

    safety_headers = ['Controller', 'Average Safety', 'Velocity Safety', 'Obstacle Safety']
    safety_data = []
    
    for k in range(len(safety_scores)):
        # Initialize accumulators for this controller
        total_safety = 0.0
        total_velocity_safety = 0.0
        total_obstacle_safety = 0.0
        num_agents = len(safety_scores[k])
        
        # Sum up scores for all agents for this controller
        for scores, metrics in safety_scores[k]:
            total_safety += scores
            total_velocity_safety += metrics["velocity_score"]
            total_obstacle_safety += metrics["obstacle_score"]
        
        # Calculate averages
        avg_safety = total_safety / num_agents if num_agents > 0 else 0
        avg_velocity_safety = total_velocity_safety / num_agents if num_agents > 0 else 0
        avg_obstacle_safety = total_obstacle_safety / num_agents if num_agents > 0 else 0
        
        safety_data.append([
            f'{controller_names[k]}',
            f'{avg_safety:.2%}',
            f'{avg_velocity_safety:.2%}',
            f'{avg_obstacle_safety:.2%}'
        ])

    fig.add_trace(
        go.Table(
            header=dict(values=safety_headers, align='center'),
            cells=dict(values=list(zip(*safety_data)), align='center')
        ),
        row=2, col=2
    )

    ## Update layout with controller name
    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            xaxis=dict(range=[mid_x - max_range, mid_x + max_range], title="X"),
            yaxis=dict(range=[mid_y - max_range, mid_y + max_range], title="Y"),
            zaxis=dict(range=[mid_z - max_range, mid_z + max_range], title="Z")
        ),
        width=1500,
        height=800,
        showlegend=True,
        title=f"Multi-Agent Navigation Performance Analysis - All Controller Avgs"
    )

    ## Save and show the figure with controller name in filename
    fig.write_html(f"trajectories_and_metrics_all_controller_avgs.html")
    fig.show()



def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create configuration
    config = SimulationConfig(
        max_simulation_steps=10000,
        model_timestep=0.1,
        recording=False
    )
    
    # Define simulation scenario
    goal_position = (1.0, -3.0, 1.0)
    start_positions = [
        # (15.0, 0.0, 0.0),
        # (-15.0, 0.0, 0.0),
        # (0.0, 15.0, 0.0),
        (0.0, -15.0, 0.0)
    ]

    # Generate obstacles
    obstacles = generate_random_obstacles(
        torch.tensor(goal_position),
        start_positions,
        num_obstacles=6
    )
    obstacles_list = [(obs[0].numpy(), obs[1]) for obs in obstacles]
    add_obstacles_to_astrobee_xml(obstacles_list, goal_position)

    # Setup Neural CLBF Controller
    base_checkpoint = r"C:\Users\brend\Documents\Project-Libra\neural_clbf\checkpoints\six_dof_vehicle\epoch=0-val_loss=6.38.ckpt"
    fine_tuned_checkpoint = r"C:\Users\brend\Documents\Project-Libra\neural_clbf\checkpoints\mujoco_six_dof_vehicle\epoch=0-val_loss=5.10.ckpt"
    
    neural_controller = setup_neural_controller(base_checkpoint, fine_tuned_checkpoint)
    dynamics_model = setup_dynamics_model(neural_controller)
    configure_dynamics(neural_controller, dynamics_model, obstacles, torch.tensor(goal_position))

    # Setup LQR controller
    lqr_controller = LQRControllerWrapper(use_rrt=False)
    lqr_controller.setup()
    
    # Setup MPC controller
    mpc_controller = MPCControllerWrapper(use_rrt=False)  # Add any necessary parameters
    mpc_controller.setup()  # Make sure MPCControllerWrapper has a setup method
    
    # Define which controllers to run
    controllers_to_run = [
        ("neural_clbf", True),
        ("lqr", True),
        ("mpc", True)
    ]

    all_results = {}
    
    # Run simulations for each controller
    for controller_type, should_run in controllers_to_run:
        if should_run:
            logging.info(f"\nRunning simulation with {controller_type}")
            results = run_simulation_controllers(
                controller_type,
                start_positions,
                goal_position,
                obstacles_list,
                config,
                neural_controller=neural_controller if controller_type == "neural_clbf" else None,
                lqr_controller=lqr_controller if controller_type == "lqr" else None,
                mpc_controller=mpc_controller if controller_type == "mpc" else None  # Pass the MPC controller
            )
            all_results[controller_type] = results
            # print(results)

    # Print results
    print("\nSimulation Results:")
    print("=" * 80)
    
    for controller_type, results in all_results.items():
        successful_runs = sum(1 for r in results if not r['collision'] and 
                            np.linalg.norm(np.array(r['final_position']) - np.array(goal_position)) < config.goal_threshold)
        
        print(f"\nController: {controller_type}")
        print(f"Success Rate: {successful_runs/len(results)*100:.1f}%")
        print(f"Average Fuel Cost: {np.mean([r['total_fuel_cost'] for r in results]):.2f}")
        print(f"Collisions: {sum(1 for r in results if r['collision'])}/{len(results)}")
        
        successful_times = [r['simulation_time'] for r in results if not r['collision'] and 
                          np.linalg.norm(np.array(r['final_position']) - np.array(goal_position)) < config.goal_threshold]
        if successful_times:
            print(f"Average Time to Goal: {np.mean(successful_times):.2f} seconds")

    # Process results and plot like main_3d.py
    all_trajectories = []
    all_efficiencies = []
    all_fuel_costs = []
    all_safety_scores = []
    for controller_type, controller_results in all_results.items():
        trajectories = []
        efficiencies = []
        fuel_costs = []
        safety_scores = []

        print(f"\nProcessing results for {controller_type}")
        
        for result in controller_results:  # Each result is a dict for one agent
            if not isinstance(result, dict):
                logging.error(f"Unexpected result type: {type(result)}")
                continue
                
            try:
                trajectories.append(result['trajectory'])
                fuel_costs.append(result['total_fuel_cost'])

                # Calculate safety score
                safety_score, safety_metrics = calculate_safety_score(
                    result['trajectory'],
                    obstacles_list,
                    config.astrobee_radius
                )
                safety_scores.append((safety_score, safety_metrics))

                # Calculate path efficiency (commented out for now)
                # start_position = tuple(result['trajectory'][0][:3])
                # actual_path = np.array([state[:3] for state in result['trajectory']])
                efficiencies.append(1.0)  # Placeholder efficiency

            except KeyError as e:
                logging.error(f"Missing key in result: {e}")
                logging.error(f"Available keys: {result.keys()}")
                continue
            except Exception as e:
                logging.error(f"Error processing result: {e}")
                continue

        # Only plot if we have valid results
        if trajectories:
            try:
                plot_trajectories_3d(
                    trajectories,
                    goal_position,
                    obstacles_list,
                    config.astrobee_radius,
                    efficiencies,
                    fuel_costs,
                    safety_scores,
                    config.__dict__,  # Convert config to dict for plotting
                    controller_name=controller_type.upper()  # Add controller name
                )
            except Exception as e:
                logging.error(f"Error plotting trajectories: {e}")

        all_trajectories.append(trajectories)
        all_efficiencies.append(efficiencies)
        all_fuel_costs.append(fuel_costs)
        all_safety_scores.append(safety_scores)

    plot_all_trajectories(
        all_trajectories,
        goal_position,
        obstacles_list,
        config.astrobee_radius,
        all_efficiencies,
        all_fuel_costs,
        all_safety_scores,
        config.__dict__,
        [controller_type for controller_type, should_run in controllers_to_run if should_run]
    )

if __name__ == "__main__":
    main()