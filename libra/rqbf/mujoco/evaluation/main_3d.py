## main_3d.py

import mujoco
import numpy as np
import glfw
import torch
import os
from typing import List, Tuple, Dict, Set, Any, Optional
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
from math import cos, sin, pi
import random
import imageio
import heapq
from plotly.subplots import make_subplots
import pandas as pd
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(project_root)

from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.systems.utils import ScenarioList
from libra.rqbf.system.rqbf import SixDOFVehicle
from libra.rqbf.experiments.rqbf_clf_contour_experiment import CLFContourExperiment
from libra.rqbf.experiments.rqbf_bf_contour_experiment import BFContourExperiment
from libra.rqbf.mujoco.evaluation import plot_saved_results

import logging

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

_config = None  # Global config storage

def setup_simulation_config(
    # Camera and view settings
    cam_pos: List[float] = [0, -40, 20],
    cam_rot: List[float] = [pi/4, 0],
    last_mouse_x: float = 0,
    last_mouse_y: float = 0,
    mouse_button_pressed: bool = False,
    zoom_level: float = 40,
    
    # Simulation state
    recording: bool = False,
    video_writer: Any = None,
    frame_count: int = 0,
    render_frame: int = 0,
    last_rrt_update: int = 0,
    
    # Vehicle parameters
    astrobee_radius: float = 0.26,
    obstacle_buffer_distance: float = 0.2,
    goal_threshold: float = 1.5,
    
    # Cost parameters
    translation_fuel_cost: float = 1.0,
    rotation_fuel_cost: float = 0.5,
    
    # Viewport
    viewport: Any = None,
    
    # Visualization
    plot_primary_visualization: bool = True,
    plot_barrier_function: bool = True,
    plot_clf_contour: bool = True,
    
    # Reset config
    reset: bool = False,
    
    ## Vehicle control parameters
    velocity_limit: float = 0.1,
    force_limit: float = 0.849,
    torque_limit: float = 0.1,
    
    ## Controller parameters
    clf_lambda: float = 2,
    controller_period: float = 0.05,
    
    ## Simulation parameters
    model_timestep: float = 0.1,
    max_simulation_steps: int = 6000,
    
    ## Debug and update frequencies
    debug_frequency: int = 50,
    rrt_update_frequency: int = 25,
    render_frequency: int = 4,
    terminate_requested: bool = False,
) -> dict:
    """Initialize and return all simulation configuration parameters."""
    global _config
    if _config is not None and not reset:
        return _config
        
    _config = {
        ## Camera and view settings
        'cam_pos': cam_pos,
        'cam_rot': cam_rot,
        'last_mouse_x': last_mouse_x,
        'last_mouse_y': last_mouse_y,
        'mouse_button_pressed': mouse_button_pressed,
        'zoom_level': zoom_level,
        
        ## Simulation state
        'recording': recording,
        'video_writer': video_writer,
        'frame_count': frame_count,
        'render_frame': render_frame,
        'last_rrt_update': last_rrt_update,
        
        ## Vehicle parameters
        'astrobee_radius': astrobee_radius,
        'obstacle_buffer_distance': obstacle_buffer_distance,
        'goal_threshold': goal_threshold,
        
        ## Cost parameters
        'translation_fuel_cost': translation_fuel_cost,
        'rotation_fuel_cost': rotation_fuel_cost,
        
        ## Viewport
        'viewport': viewport,
        
        ## Plotly visualizations
        'plot_primary_visualization': plot_primary_visualization,
        'plot_barrier_function': plot_barrier_function,
        'plot_clf_contour': plot_clf_contour,
        
        ## Vehicle control parameters
        'velocity_limit': velocity_limit,
        'force_limit': force_limit,
        'torque_limit': torque_limit,
        
        ## Controller parameters
        'clf_lambda': clf_lambda,
        'controller_period': controller_period,
        
        ## Simulation parameters
        'model_timestep': model_timestep,
        'max_simulation_steps': max_simulation_steps,
        
        ## Debug and update frequencies
        'debug_frequency': debug_frequency,
        'rrt_update_frequency': rrt_update_frequency,
        'render_frequency': render_frequency,
        'terminate_requested': terminate_requested,
    }
    
    return _config

def update_config(key: str, value):
    """Update a specific config value."""
    global _config
    if _config is None:
        _config = setup_simulation_config()
    _config[key] = value

def setup_mujoco():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outer_dir = os.path.dirname(script_dir)
    xml_path = os.path.join(outer_dir, "astrobee.xml")
    if not os.path.exists(xml_path):
        logging.error(f"XML file not found at path: {xml_path}")
        raise FileNotFoundError(f"XML file not found at path: {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path) ## type: ignore
    data = mujoco.MjData(model) ## type: ignore
    
    config = setup_simulation_config()
    
    ## Set velocity limits
    model.actuator_gainprm[:, 0] = 1
    model.actuator_biasprm[:, 1] = -config['velocity_limit']
    model.actuator_biasprm[:, 2] = config['velocity_limit']

    return model, data

def setup_controller(log_file, fine_tuned_weights_path):
    neural_controller = NeuralCLBFController.load_from_checkpoint(log_file)
    
    checkpoint = torch.load(fine_tuned_weights_path, map_location=torch.device('cpu'))
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_state_dict = {k: v for k, v in state_dict.items() if k in neural_controller.state_dict()}
    
    neural_controller.load_state_dict(model_state_dict, strict=False) ## type: ignore
    
    config = setup_simulation_config()
    neural_controller.clf_lambda = config['clf_lambda']
    neural_controller.controller_period = config['controller_period']
    neural_controller.eval()

    neural_controller.dynamics_model.rrt_path = None ## type: ignore

    return neural_controller

def setup_glfw():
    if not glfw.init():
        logging.error("Could not initialize GLFW")
        raise Exception("Could not initialize GLFW")

    window = glfw.create_window(1920, 1080, "3D MuJoCo Simulation", None, None)
    if not window:
        glfw.terminate()
        logging.error("Could not create GLFW window")
        raise Exception("Could not create GLFW window")

    glfw.make_context_current(window)
    logging.info("GLFW window created and context made current.")
    
    glfw.set_key_callback(window, key_callback)
    glfw.set_cursor_pos_callback(window, mouse_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    
    return window

def setup_mujoco_visualization(model):
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150) ## type: ignore
    scene = mujoco.MjvScene(model, maxgeom=1000) ## type: ignore

    cam = mujoco.MjvCamera() ## type: ignore
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE ## type: ignore
    cam.trackbodyid = 0
    cam.distance = 40
    cam.azimuth = 45
    cam.elevation = -30

    opt = mujoco.MjvOption() ## type: ignore

    return context, scene, cam, opt

def update_camera(cam, config):
    """Update camera position without printing settings."""
    cam.lookat[0] = config['cam_pos'][0] + cos(config['cam_rot'][0]) * sin(config['cam_rot'][1])
    cam.lookat[1] = config['cam_pos'][1] + cos(config['cam_rot'][0]) * cos(config['cam_rot'][1])
    cam.lookat[2] = config['cam_pos'][2] + sin(config['cam_rot'][0])
    cam.distance = config['zoom_level']
    cam.azimuth = config['cam_rot'][1] * 180 / pi
    cam.elevation = -config['cam_rot'][0] * 180 / pi

def print_camera_settings(cam, config):
    """Print current camera settings."""
    print("\nCurrent Camera Settings:")
    print(f"Position: {config['cam_pos']}")
    print(f"Rotation: {config['cam_rot']}")
    print(f"Zoom Level: {config['zoom_level']}")
    print(f"Look At: [{cam.lookat[0]:.2f}, {cam.lookat[1]:.2f}, {cam.lookat[2]:.2f}]")
    print(f"Azimuth: {cam.azimuth:.2f}")
    print(f"Elevation: {cam.elevation:.2f}")

def key_callback(window, key, scancode, action, mods):
    config = setup_simulation_config()
    if action == glfw.PRESS:
        if key == glfw.KEY_ESCAPE:
            update_config('terminate_requested', True)
            print("\nTermination requested. Cleaning up...")
        elif key == glfw.KEY_R:
            update_config('cam_pos', [0, -40, 20])
            update_config('cam_rot', [pi/4, 0])
            update_config('zoom_level', 40)
            print("\nCamera Reset to Default Position")
        elif key == glfw.KEY_P:
            cam = mujoco.MjvCamera() ## type: ignore
            update_camera(cam, config)
            print_camera_settings(cam, config)
            print("\nCurrent Camera Configuration for setup_simulation_config():")
            print(f"cam_pos={config['cam_pos']},")
            print(f"cam_rot={config['cam_rot']},")
            print(f"zoom_level={config['zoom_level']},")
        elif key == glfw.KEY_V:
            if not config['recording']:
                video_writer = imageio.get_writer('simulation_video.mp4', fps=30)
                update_config('video_writer', video_writer)
                update_config('recording', True)
                update_config('frame_count', 0)
                print("Started recording")
            else:
                config['video_writer'].close()
                update_config('recording', False)
                print(f"Stopped recording. Saved {config['frame_count']} frames.")

def scroll_callback(window, xoffset, yoffset):
    config = setup_simulation_config()
    update_config('zoom_level', config['zoom_level'] - yoffset)

def mouse_callback(window, xpos, ypos):
    config = setup_simulation_config()
    if config['mouse_button_pressed']:
        dx = xpos - config['last_mouse_x']
        dy = ypos - config['last_mouse_y']
        update_config('cam_rot', [config['cam_rot'][0], config['cam_rot'][1] - dx * 0.01])
        update_config('cam_rot', [max(min(config['cam_rot'][0] + dy * 0.01, pi/2), -pi/2), config['cam_rot'][1]])
    update_config('last_mouse_x', xpos)
    update_config('last_mouse_y', ypos)

def mouse_button_callback(window, button, action, mods):
    config = setup_simulation_config()
    if button == glfw.MOUSE_BUTTON_LEFT:
        update_config('mouse_button_pressed', action == glfw.PRESS)

def get_state(data) -> np.ndarray:
    position = data.qpos[:3]
    roll_pitch_yaw = data.qpos[3:6]
    r = R.from_euler('xyz', roll_pitch_yaw)
    quaternion = r.as_quat()
    quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    linear_velocity = data.qvel[:3]
    angular_velocity = data.qvel[3:6]
    
    state = np.zeros(19)
    state[0:3] = position
    state[3:7] = quaternion
    state[7:10] = linear_velocity
    state[10:13] = angular_velocity
    state[13:16] = 0.0  ## Set accelerometer bias to zero
    state[16:19] = 0.0  ## Set gyroscope bias to zero
    return state

def calculate_fuel_cost(trajectory: List[np.ndarray], config: dict) -> float:
    """
    Calculate fuel cost based on distance traveled, time taken, and speed profile
    
    Args:
        trajectory (List[np.ndarray]): List of state vectors
        config (dict): Configuration dictionary containing cost parameters
        
    Returns:
        float: Total fuel cost
    """
    if not trajectory:
        return 0.0
        
    # Extract positions and velocities
    positions = np.array([state[:3] for state in trajectory])
    velocities = np.array([state[7:10] for state in trajectory])
    
    # Calculate distance and speed metrics
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    speeds = np.linalg.norm(velocities, axis=1)
    
    total_distance = np.sum(distances)
    avg_speed = np.mean(speeds)
    max_speed = np.max(speeds)
    
    # Time taken (based on trajectory length and timestep)
    time_taken = len(trajectory) * config['model_timestep']
    
    # Calculate base cost from distance
    base_cost = config['translation_fuel_cost'] * total_distance
    
    # Add penalties for high speeds and rapid changes
    speed_penalty = config['translation_fuel_cost'] * (max_speed / config['velocity_limit'])
    acceleration_penalty = config['rotation_fuel_cost'] * np.sum(np.abs(np.diff(speeds)))
    
    # Combine components with weights
    total_cost = (base_cost + 
                 0.3 * speed_penalty + 
                 0.2 * acceleration_penalty)
    
    return float(total_cost)

def get_control_from_controller(
    controller, 
    x_tensor: torch.Tensor, 
    controller_type: str, 
    goal_position: Optional[np.ndarray] = None  # Change type hint to Optional
) -> np.ndarray:
    """Get control from different types of controllers.
    
    Args:
        controller: The controller object (Neural CLBF, LQR, or MPC)
        x_tensor: Current state as tensor
        controller_type: Type of controller ('neural_clbf', 'lqr', or 'mpc')
        goal_position: Goal position for LQR/MPC controllers
    """
    if controller_type == 'neural_clbf':
        with torch.no_grad():
            return controller.u(x_tensor).squeeze().detach().numpy()
    else:  # LQR or MPC
        state = x_tensor.squeeze().numpy()
        # Get goal position from parameter or try to get from dynamics model
        if goal_position is None and hasattr(controller.dynamics_model, 'goal_state'):
            goal_position = controller.dynamics_model.goal_state[:3]
        elif goal_position is None:
            raise ValueError("Goal position must be provided for LQR/MPC controllers")
            
        obstacles = []
        if hasattr(controller.dynamics_model, 'obstacles'):
            obstacles = [(obs[0].numpy(), obs[1]) for obs in controller.dynamics_model.obstacles]
        return controller.get_control(state, goal_position, obstacles)

def run_simulation(
    start_positions: List[Tuple[float, float, float]],
    goal_position: Tuple[float, float, float],
    static_obstacles: List[Tuple[np.ndarray, float]],
    controller,
    controller_type: str = 'neural_clbf'
) -> List[Dict]:
    """Run simulation with specified controller"""
    config = setup_simulation_config()
    model, data = setup_mujoco()
    window = setup_glfw()
    context, scene, cam, opt = setup_mujoco_visualization(model)

    ## Force window to be fullscreen and prevent minimization if recording
    if config['recording']:
        glfw.set_window_attrib(window, glfw.DECORATED, False)
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        glfw.set_window_monitor(window, monitor, 0, 0, mode.size.width, mode.size.height, mode.refresh_rate)
        glfw.set_window_attrib(window, glfw.RESIZABLE, False)
        glfw.set_window_attrib(window, glfw.AUTO_ICONIFY, False)

    simulation_results = []
    model.opt.timestep = config['model_timestep']

    font = mujoco.mjtFont.mjFONT_BIG.value # type: ignore
    gridpos = mujoco.mjtGridPos.mjGRID_TOPLEFT.value # type: ignore
    simulation_time = 0.0

    config['viewport'] = mujoco.MjrRect(0, 0, 1920, 1080) # type: ignore

    try:
        for idx, start_pos in enumerate(start_positions):
            agent_id = idx + 1
            logging.info(f"Starting simulation for Agent {agent_id} with start position: {start_pos}")

            controller.dynamics_model.rrt_path = None if hasattr(controller, 'dynamics_model') else None

            trajectory = []
            barrier_trajectory = []
            collision_occurred = False
            control_history = []
            collision_details = {
                'collision_time': None,
                'collision_position': None,
                'collision_obstacle_id': None
            }

            mujoco.mj_resetData(model, data) # type: ignore
            mujoco.mj_forward(model, data) # type: ignore

            data.qpos[:3] = start_pos
            data.qpos[3:6] = [0, 0, 0]
            data.qvel[:] = 0

            simulation_step = 0
            config['last_rrt_update'] = 0
            
            while simulation_step < config['max_simulation_steps']:
                if config['terminate_requested']:
                    logging.info("Termination requested - stopping simulation")
                    break

                if glfw.window_should_close(window):
                    if config['recording']:
                        logging.warning("Window close attempted during recording - ignoring")
                        glfw.set_window_should_close(window, False)
                        continue
                    else:
                        break

                # Ensure window stays focused if recording
                if config['recording'] and not glfw.get_window_attrib(window, glfw.FOCUSED):
                    glfw.focus_window(window)

                update_camera(cam, config)

                simulation_step += 1
                simulation_time += model.opt.timestep

                state = get_state(data)
                trajectory.append(state.copy())

                if simulation_step % config['debug_frequency'] == 0 or simulation_step < 10:
                    position = state[0:3]
                    velocity = state[7:10]
                    speed = np.linalg.norm(velocity)
                    logging.debug(f"Agent {agent_id} Step {simulation_step}: Position [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}], Speed: {speed:.3f}")

                with torch.no_grad():
                    x_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    
                    if hasattr(controller, 'dynamics_model') and hasattr(controller.dynamics_model, 'rrt_path'):
                        if (controller.dynamics_model.rrt_path is None or 
                            simulation_step - config['last_rrt_update'] > config['rrt_update_frequency']):
                            start = x_tensor[0, :3]
                            controller.dynamics_model.plan_rrt_path(start)
                            config['last_rrt_update'] = simulation_step

                    control = get_control_from_controller(
                        controller, 
                        x_tensor, 
                        controller_type,
                        np.array(goal_position)
                    )
                    
                    if controller_type == 'neural_clbf':
                        barrier = controller.V(x_tensor).squeeze().item() - 1.0
                        barrier_trajectory.append(barrier)
                    else:
                        barrier_trajectory.append(0.0)

                    step_fuel_cost = calculate_fuel_cost(trajectory, config)
                    total_fuel_cost = step_fuel_cost
                    control_history.append(control)

                control = np.clip(control, -config['force_limit'], config['force_limit'])
                control[3:] = np.clip(control[3:], -config['torque_limit'], config['torque_limit'])

                data.ctrl[:6] = control
                data.ctrl[6:] = 0 

                for _ in range(2): 
                    mujoco.mj_step(model, data) # type: ignore

                ## Check for collision and goal reaching
                distance_to_goal = np.linalg.norm(state[:3] - np.array(goal_position))
                if distance_to_goal <= config['goal_threshold']:
                    logging.info(f"Agent {agent_id} reached the goal at step {simulation_step}.")
                    break

                for obs_idx, (obs_pos, obs_radius) in enumerate(static_obstacles):
                    obs_pos_tensor = torch.tensor(obs_pos, dtype=torch.float32)
                    distance_to_obstacle = torch.norm(torch.tensor(state[:3], dtype=torch.float32) - obs_pos_tensor)
                    if distance_to_obstacle <= obs_radius + config['astrobee_radius']:
                        collision_occurred = True
                        collision_details = {
                            'collision_time': simulation_time,
                            'collision_position': state[:3].tolist(),
                            'collision_obstacle_id': obs_idx + 1
                        }
                        logging.warning(f"Agent {agent_id} collided with obstacle {obs_idx + 1} at time {simulation_time:.2f}s")
                        break

                if collision_occurred:
                    break

                ## Render every 4 frames
                if config['render_frame'] % config['render_frequency'] == 0:
                    try:
                        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene) # type: ignore
                        mujoco.mjr_render(config['viewport'], scene, context) # type: ignore

                        # Add text overlay for speed only
                        velocity = state[7:10]
                        speed = np.linalg.norm(velocity)
                        speed_text = f"Speed: {speed:.2f} m/s"

                        # Render speed text
                        mujoco.mjr_overlay( # type: ignore
                            mujoco.mjtFont.mjFONT_NORMAL.value, # type: ignore
                            mujoco.mjtGridPos.mjGRID_TOPLEFT.value, # type: ignore
                            config['viewport'],
                            speed_text,
                            "",
                            context
                        )

                        if config['recording']:
                            pixels = np.zeros((config['viewport'].height, config['viewport'].width, 3), dtype=np.uint8)
                            mujoco.mjr_readPixels(pixels, None, config['viewport'], context) # type: ignore
                            pixels = np.flipud(pixels)
                            config['video_writer'].append_data(pixels)
                            config['frame_count'] += 1

                        glfw.swap_buffers(window)
                    except Exception as e:
                        if config['recording']:
                            logging.error(f"Error during rendering: {str(e)}")
                            continue
                        else:
                            raise

                config['render_frame'] += 1
                glfw.poll_events()

                if config['terminate_requested']:
                    break

            simulation_results.append({
                'agent_id': agent_id,
                'start_position': start_pos,
                'trajectory': trajectory,
                'barrier_trajectory': barrier_trajectory,
                'collision': collision_occurred,
                'collision_time': collision_details['collision_time'] if collision_occurred else None,
                'collision_position': collision_details['collision_position'] if collision_occurred else None,
                'collision_obstacle_id': collision_details['collision_obstacle_id'] if collision_occurred else None,
                'total_fuel_cost': total_fuel_cost,
                'control_history': control_history
            })

            if config['terminate_requested']:
                break

    except Exception as e:
        logging.error(f"Error during simulation: {str(e)}")
        raise
    finally:
        if config['recording'] and 'video_writer' in config:
            try:
                config['video_writer'].close()
                logging.info(f"Recording finished. Saved {config['frame_count']} frames.")
            except Exception as e:
                logging.error(f"Error closing video writer: {str(e)}")

        try:
            glfw.terminate()
        except Exception as e:
            logging.error(f"Error terminating GLFW: {str(e)}")

    return simulation_results

def calculate_path_efficiency(actual_path, a_star_path, collision, config):
    """Calculate path efficiency based on deviation from optimal A* path points."""

    if collision:
        logging.info("Path efficiency is 0 due to collision")
        return 0.0

    if not a_star_path:
        logging.info("Path efficiency is 0 due to no A* path found")
        return 0.0

    actual_path = np.array(actual_path)
    a_star_path = np.array(a_star_path)
    
    # Interpolate A* path to get more reference points
    num_points = 50
    t_optimal = np.linspace(0, 1, len(a_star_path))
    t_interp = np.linspace(0, 1, num_points)
    optimal_path_interp = np.array([
        np.interp(t_interp, t_optimal, a_star_path[:, i]) 
        for i in range(3)
    ]).T

    # Interpolate actual path to same number of points
    t_actual = np.linspace(0, 1, len(actual_path))
    actual_path_interp = np.array([
        np.interp(t_interp, t_actual, actual_path[:, i]) 
        for i in range(3)
    ]).T

    # Calculate path lengths
    actual_length = np.sum(np.linalg.norm(np.diff(actual_path, axis=0), axis=1))
    optimal_length = np.sum(np.linalg.norm(np.diff(a_star_path, axis=0), axis=1))
    
    # Calculate average deviation from optimal path
    deviations = np.linalg.norm(actual_path_interp - optimal_path_interp, axis=1)
    avg_deviation = np.mean(deviations)
    max_deviation = np.max(deviations)
    
    # Calculate normalized deviation score (0 to 1, where 1 is best)
    # Much more lenient deviation scale - allow up to 50% of path lengt
    deviation_scale = optimal_length * 0.5
    deviation_score = 1.0 / (1.0 + (avg_deviation / deviation_scale) ** 0.5)  # Square root makes penalty more gradual
    
    length_ratio = min(actual_length, optimal_length) / max(actual_length, optimal_length)

    # Apply square root to make length penalty more gradual
    length_score = np.sqrt(length_ratio)
    
    final_efficiency = (0.8 * deviation_score + 0.2 * length_score)
    
    final_efficiency = 0.3 + (0.7 * final_efficiency)  # Minimum score of 0.3 unless collision
    final_efficiency = min(max(final_efficiency, 0.3), 1.0)  # Clamp between 0.3 and 1

    logging.info("\nPath Analysis:")
    logging.info(f"Actual path length: {actual_length:.3f}")
    logging.info(f"Optimal path length: {optimal_length:.3f}")
    logging.info(f"Length ratio: {length_ratio:.3f}")
    logging.info(f"Length score: {length_score:.3f}")
    logging.info(f"Average deviation from optimal: {avg_deviation:.3f}")
    logging.info(f"Maximum deviation from optimal: {max_deviation:.3f}")
    logging.info(f"Deviation score: {deviation_score:.3f}")
    logging.info(f"Final efficiency score: {final_efficiency:.3f}")

    # Final assessment with adjusted thresholds
    logging.info("\nFinal Efficiency Assessment:")
    if final_efficiency < 0.4:
        logging.info("Basic efficiency - completed path with significant deviations")
    elif final_efficiency < 0.6:
        logging.info("Fair efficiency - completed path with moderate deviations")
    elif final_efficiency < 0.8:
        logging.info("Good efficiency - reasonable path to goal")
    else:
        logging.info("Excellent efficiency - close to optimal path")

    return final_efficiency

def calculate_path_angles(path):
    """Calculate angles between consecutive segments in a path.
    
    Args:
        path (np.ndarray): Array of points representing path
        
    Returns:
        np.ndarray: Array of angles in degrees
    """
    if len(path) < 3:
        return np.array([])
        
    vectors = np.diff(path, axis=0)
    vectors_normalized = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    
    dot_products = np.sum(vectors_normalized[:-1] * vectors_normalized[1:], axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angles = np.arccos(dot_products)
    
    return np.degrees(angles)

def compute_performance_metrics(trajectory: List[np.ndarray], goal_position: List[float], collision: bool) -> Dict[str, float]:
    """Compute performance metrics for a single agent's trajectory.
    
    Args:
        trajectory (List[np.ndarray]): List of state vectors representing the agent's trajectory
        goal_position (List[float]): The goal position coordinates
        collision (bool): Whether the agent collided with an obstacle
        
    Returns:
        Dict[str, float]: Dictionary containing performance metrics
    """
    if not trajectory:
        return {
            'final_distance_to_goal': float('inf'),
            'total_distance_traveled': 0.0,
            'max_speed': 0.0,
            'avg_speed': 0.0
        }
    
    trajectory_np = np.array([state[:3] for state in trajectory])  # Extract position components
    velocities = np.array([state[7:10] for state in trajectory])  # Extract velocity components
    speeds = np.linalg.norm(velocities, axis=1)
    
    final_position = trajectory_np[-1]
    final_distance = float(np.linalg.norm(np.array(goal_position[:3]) - final_position))
    
    distances = np.linalg.norm(np.diff(trajectory_np, axis=0), axis=1)
    total_distance = float(np.sum(distances))
    
    return {
        'final_distance_to_goal': final_distance,
        'total_distance_traveled': total_distance,
        'max_speed': float(np.max(speeds)),
        'avg_speed': float(np.mean(speeds))
    }

def calculate_safety_score(
    trajectory: List[np.ndarray],
    obstacles: List[Tuple[np.ndarray, float]],
    object_radius: float,
    velocity_threshold: float = 0.1,
    buffer_distance: float = 0.2
) -> Tuple[float, Dict[str, float]]:
    """Calculate safety score based on velocity and obstacle proximity."""
    if not trajectory:
        return 0.0, {"velocity_score": 0.0, "obstacle_score": 0.0}
    
    trajectory_array = np.stack(trajectory)  # Convert list to numpy array
    velocities = trajectory_array[:, 7:10]  # Linear velocity indices 
    positions = trajectory_array[:, 0:3]    # Position indices
    
    # Velocity safety check
    velocity_norms = np.linalg.norm(velocities, axis=1)
    velocity_score = float(np.mean(velocity_norms <= velocity_threshold))
    
    # Obstacle safety check - now binary (0 if ever hits obstacle, 1 if always safe)
    obstacle_score = 1.0
    for position in positions:
        for obstacle_center, obstacle_radius in obstacles:
            distance = float(np.linalg.norm(position - obstacle_center))
            if distance <= (obstacle_radius + object_radius + buffer_distance):
                obstacle_score = 0.0
                break
        if obstacle_score == 0.0:
            break
    
    # Combined safety score (equal weighting)
    overall_safety = 0.5 * velocity_score + 0.5 * obstacle_score
    
    metrics = {
        "velocity_score": velocity_score,
        "obstacle_score": obstacle_score
    }
    
    return overall_safety, metrics

def heuristic(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))  # Explicitly cast to float

def a_star(
    start: Tuple[float, float, float], 
    goal: Tuple[float, float, float], 
    obstacles: List[Tuple[np.ndarray, float]], 
    object_radius: float, 
    config: dict,
    step_size: Optional[float] = None
) -> List[Tuple[float, float, float]]:
    if step_size is None:
        step_size = max(object_radius / 2, 0.25)
    
    start = (float(start[0]), float(start[1]), float(start[2]))
    goal = (float(goal[0]), float(goal[1]), float(goal[2]))
    
    open_set: List[Tuple[float, Tuple[float, float, float]]] = [(0, start)]
    came_from: Dict[Tuple[float, float, float], Tuple[float, float, float]] = {}
    g_score: Dict[Tuple[float, float, float], float] = {start: 0}
    f_score: Dict[Tuple[float, float, float], float] = {start: heuristic(start, goal)}
    
    closed_set: Set[Tuple[float, float, float]] = set()
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if np.linalg.norm(np.array(current) - np.array(goal)) < step_size:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            raw_path = list(reversed(path))
            smoothed_path = smooth_path(raw_path, obstacles, object_radius, config)
            return smoothed_path
        
        closed_set.add(current)
        
        for neighbor in get_neighbors(current, step_size):
            if neighbor in closed_set or not is_valid(neighbor, obstacles, object_radius, config):
                continue
            
            tentative_g_score = g_score[current] + step_size
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return []  ## No path found

def plot_trajectories_3d(
    trajectories: List[List[np.ndarray]], 
    goal_position: Tuple[float, float, float], 
    obstacles: List[Tuple[np.ndarray, float]], 
    object_radius: float, 
    efficiencies: List[float], 
    fuel_costs: List[float], 
    safety_scores: List[Tuple[float, Dict[str, float]]],
    config: dict,
    simulation_results: List[Dict],
    controller_name: str = "Unknown"  # Move default argument to end
):
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "scene", "rowspan": 3}, {"type": "table"}],
            [None, {"type": "table"}],
            [None, {"type": "table"}] 
        ],
        subplot_titles=(
            '3D Trajectories', 
            'Performance Metrics', 
            'Safety Metrics',
            'Goal Status'
        ),
        vertical_spacing=0.02,
        horizontal_spacing=0.02
    )

    ## Calculate plot bounds
    all_x, all_y, all_z = [], [], []
    
    ## 3D Trajectory Plot
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']
    for i, trajectory in enumerate(trajectories):
        if not trajectory:
            continue
        points = np.array(trajectory)[:, :3]
        agent_color = colors[i % len(colors)]
        
        all_x.extend(points[:, 0])
        all_y.extend(points[:, 1])
        all_z.extend(points[:, 2])
        
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='lines',
                name=f'Agent {i+1} Path',
                line=dict(width=15, color=agent_color),
                showlegend=True
            ),
            row=1, col=1
        )
        
        ## Add start point
        fig.add_trace(
            go.Scatter3d(
                x=[points[0, 0]], y=[points[0, 1]], z=[points[0, 2]],
                mode='markers',
                name=f'Agent {i+1} Start',
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
                name=f'Agent {i+1} End',
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
        
        ## Calculate and add A* path
        start_position = tuple(points[0])
        a_star_path = a_star(start_position, goal_position, obstacles, object_radius, config)
        if a_star_path:
            a_star_points = np.array(a_star_path)
            all_x.extend(a_star_points[:, 0])
            all_y.extend(a_star_points[:, 1])
            all_z.extend(a_star_points[:, 2])
            
            fig.add_trace(
                go.Scatter3d(
                    x=a_star_points[:, 0],
                    y=a_star_points[:, 1],
                    z=a_star_points[:, 2],
                    mode='lines',
                    name=f'Agent {i+1} A* Path',
                    line=dict(
                        color=agent_color,
                        width=3,
                        dash='dash'
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
    max_range = max(
        max(all_x) - min(all_x),
        max(all_y) - min(all_y),
        max(all_z) - min(all_z)) / 2.0
    mid_x = (max(all_x) + min(all_x)) / 2
    mid_y = (max(all_y) + min(all_y)) / 2
    mid_z = (max(all_z) + min(all_z)) / 2

    ## Performance Metrics Table (row 1)
    perf_headers = ['Agent', 'Path Efficiency', 'Fuel Cost', 'Distance', 'Max Speed', 'Avg Speed']
    perf_data = []
    for i in range(len(trajectories)):
        if i < len(trajectories) and trajectories[i]:
            points = np.array(trajectories[i])[:, :3]
            distance = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
            velocities = np.array([state[7:10] for state in trajectories[i]])
            speeds = np.linalg.norm(velocities, axis=1)
            max_speed = np.max(speeds)
            avg_speed = np.mean(speeds)
            perf_data.append([
                f'Agent {i+1}',
                f'{efficiencies[i]:.2%}',
                f'{fuel_costs[i]:.2f}',
                f'{distance:.2f}m',
                f'{max_speed:.2f}m/s',
                f'{avg_speed:.2f}m/s'
            ])

    fig.add_trace(
        go.Table(header=dict(values=perf_headers), cells=dict(values=list(zip(*perf_data)))),
        row=1, col=2
    )

    ## Safety Metrics Table (row 2)
    safety_headers = ['Agent', 'Overall Safety', 'Velocity Safety', 'Obstacle Safety']
    safety_data = [
        [f'Agent {i+1}', f'{score:.2%}', f'{metrics["velocity_score"]:.2%}', f'{metrics["obstacle_score"]:.2%}']
        for i, (score, metrics) in enumerate(safety_scores)
    ]

    fig.add_trace(
        go.Table(header=dict(values=safety_headers), cells=dict(values=list(zip(*safety_data)))),
        row=2, col=2
    )

    ## Goal Status Table (row 3)
    goal_headers = ['Agent', 'Goal Status', 'Final Distance', 'Reason']
    goal_data = []
    for i, trajectory in enumerate(trajectories):
        if trajectory:
            final_distance = np.linalg.norm(np.array(trajectory[-1][:3]) - np.array(goal_position))
            goal_reached = final_distance <= config['goal_threshold']
            
            # Determine reason
            if goal_reached:
                reason = "N/A"
            elif len(trajectory) >= config['max_simulation_steps']:
                reason = "Timeout"
            elif any(result['collision'] for result in simulation_results if result['agent_id'] == i + 1):
                reason = "Collision"
            else:
                reason = "Failed"
            
            goal_data.append([
                f'Agent {i+1}',
                'REACHED' if goal_reached else 'NOT REACHED',
                f'{final_distance:.2f}m',
                reason
            ])

    fig.add_trace(
        go.Table(
            header=dict(
                values=goal_headers,
                align='center',
                font=dict(size=12, color='white'),
                fill_color='darkblue'
            ),
            cells=dict(
                values=list(zip(*goal_data)),
                align='center',
                font=dict(size=11),
                fill_color=[['lightgray', 'white'] * (len(goal_data)//2 + 1)]
            )
        ),
        row=3, col=2 
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
        title=f"Multi-Agent Navigation Performance Analysis - {controller_name}"
    )

    ## Save and show the figure with controller name in filename
    fig.write_html(f"trajectories_and_metrics_{controller_name}.html")
    fig.show()

def add_obstacles_to_astrobee_xml(obstacles, goal_position, goal_radius, filepath='astrobee.xml'):
    """
    Adds obstacles to the Astrobee XML file.
    Basically a way to dynamically add obstacles to the simulation without having to manually edit the XML file each time
    """
    import xml.etree.ElementTree as ET

    script_dir = os.path.dirname(os.path.abspath(__file__))
    outer_dir = os.path.dirname(script_dir)
    filepath = os.path.join(outer_dir, filepath)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")

    tree = ET.parse(filepath)
    root = tree.getroot()

    worldbody = root.find('.//worldbody')

    if worldbody is None:
        raise ValueError("Could not find <worldbody> element in the XML file.")

    existing_elements = worldbody.findall('body') + worldbody.findall('site')
    for element in existing_elements:
        if element.get('name') in ['goal'] or element.get('name', '').startswith(('obstacle', 'moving_')):
            worldbody.remove(element)
    
    print(f"Removed existing obstacles and goal elements.")

    ## Add static obstacles
    for i, (position, radius) in enumerate(obstacles, start=1):
        x, y, z = position.tolist()
        obstacle_body = ET.SubElement(worldbody, 'body', {
            'name': f'obstacle{i}',
            'pos': f'{x} {y} {z}'
        })
        ET.SubElement(obstacle_body, 'geom', {
            'name': f'obstacle{i}',
            'type': 'sphere',
            'size': str(radius),
            'rgba': '1 0 0 0.5'
        })

    ## Add visual marker for goal
    ET.SubElement(worldbody, 'site', {
        'name': 'goal',
        'pos': f'{goal_position[0]} {goal_position[1]} {goal_position[2]}',
        'size': str(goal_radius),
        'type': 'sphere',
        'rgba': '0 1 0 0.7'
    })

    print(f"Successfully added {len(obstacles)} static obstacle(s) and goal visual marker to {filepath}.")

    tree.write(filepath, encoding='utf-8', xml_declaration=True)

def generate_random_obstacle(
    goal_position: torch.Tensor,
    start_positions: List[Tuple[float, float, float]],
    existing_obstacles: List[Tuple[torch.Tensor, float]],
    x_range: float,
    y_range: float,
    z_range: float,
    min_radius: float = 0.5,
    max_radius: float = 1.5
) -> Tuple[torch.Tensor, float]:
    max_attempts = 100
    for _ in range(max_attempts):
        start = torch.tensor(random.choice(start_positions))
        
        ## Generate a position along the path from start to goal
        t = random.uniform(0.3, 0.7)  ## Generate obstacle between 30% and 70% of the path
        position = start + t * (goal_position - start)
        
        ## Add some randomness to the position
        position += torch.tensor([
            random.uniform(-x_range/4, x_range/4),
            random.uniform(-y_range/4, y_range/4),
            random.uniform(-z_range/4, z_range/4)
        ])
        
        radius = random.uniform(min_radius, max_radius)
        
        ## Check if the obstacle overlaps with the goal or start positions
        if torch.norm(position - goal_position) <= radius + 1.0:
            continue
        if any(torch.norm(torch.tensor(start) - position) <= radius + 1.0 for start in start_positions):
            continue
        
        ## Check if the obstacle overlaps with existing obstacles
        if any(torch.norm(position - obs[0]) <= radius + obs[1] for obs in existing_obstacles):
            continue
        
        return position, radius

    raise ValueError("Could not generate a valid obstacle.")

def generate_random_obstacles(
    goal_position: torch.Tensor,
    start_positions: List[Tuple[float, float, float]],
    num_obstacles: int,
    x_range: float = 20.0,
    y_range: float = 20.0,
    z_range: float = 5.0,
    min_radius: float = 0.5,
    max_radius: float = 1.5
) -> List[Tuple[torch.Tensor, float]]:
    obstacles = []
    
    ## Generate at least one obstacle for each start position
    for start_pos in start_positions:
        start = torch.tensor(start_pos)
        
        ## Generate obstacle between 40% and 60% of the path
        t = random.uniform(0.4, 0.6)
        position = start + t * (goal_position - start)
        
        ## Add less randomness to keep obstacles closer to the direct path
        position += torch.tensor([
            random.uniform(-x_range/8, x_range/8),
            random.uniform(-y_range/8, y_range/8),
            random.uniform(-z_range/8, z_range/8)
        ])
        
        radius = random.uniform(min_radius, max_radius)
        
        obstacles.append((position, radius))
        
        logging.info(f"Generated obstacle for start position {start_pos}: position {position.tolist()}, radius {radius}")
    
    ## Generate additional random obstacles if needed
    while len(obstacles) < num_obstacles:
        obstacle = generate_random_obstacle(
            goal_position,
            start_positions,
            obstacles,
            x_range,
            y_range,
            z_range,
            min_radius,
            max_radius
        )
        if obstacle is not None:
            obstacles.append(obstacle)
            logging.info(f"Generated additional obstacle: position {obstacle[0].tolist()}, radius {obstacle[1]}")
    
    return obstacles

def framebuffer_size_callback(window, width, height):
    config = setup_simulation_config()
    config['viewport'] = mujoco.MjrRect(0, 0, width, height) ## type: ignore

def get_neighbors(current: Tuple[float, float, float], step_size: float) -> List[Tuple[float, float, float]]:
    x, y, z = current
    return [
        (x + step_size, y, z), (x - step_size, y, z),
        (x, y + step_size, z), (x, y - step_size, z),
        (x, y, z + step_size), (x, y, z - step_size)
    ]

def is_valid(point: Tuple[float, float, float], obstacles: List[Tuple[np.ndarray, float]], object_radius: float, config) -> bool:
    safety_margin = object_radius + config['obstacle_buffer_distance']
    for obstacle_center, obstacle_radius in obstacles:
        combined_radius = obstacle_radius + safety_margin
        if np.linalg.norm(np.array(point) - obstacle_center) <= combined_radius:
            return False
    return True

def is_path_clear(start: np.ndarray, end: np.ndarray, obstacles: List[Tuple[np.ndarray, float]], object_radius: float, config) -> bool:
    direction = end - start
    distance = np.linalg.norm(direction)
    if distance < 1e-6:
        return True
    direction /= distance
    step_size = min(0.1, distance / 10)  # Adaptive step size
    steps = int(distance / step_size)
    for i in range(steps + 1):
        point = start + i * step_size * direction
        if not is_valid(tuple(point), obstacles, object_radius, config):
            return False
    return True

def smooth_path(path: List[Tuple[float, float, float]], obstacles: List[Tuple[np.ndarray, float]], object_radius: float, config, iterations: int = 5) -> List[Tuple[float, float, float]]:
    smoothed_path = path.copy()
    for _ in range(iterations):
        i = 0
        while i < len(smoothed_path) - 2:
            start = np.array(smoothed_path[i])
            end = np.array(smoothed_path[i + 2])
            
            ## Check if the direct path between start and end is collision-free
            if is_path_clear(start, end, obstacles, object_radius, config):
                ## Remove the intermediate point
                smoothed_path.pop(i + 1)
            else:
                ## If we can't remove the point, try to smooth it
                mid = np.array(smoothed_path[i + 1])
                new_mid = 0.5 * (start + end)
                if is_path_clear(start, new_mid, obstacles, object_radius, config) and is_path_clear(new_mid, end, obstacles, object_radius, config):
                    smoothed_path[i + 1] = tuple(new_mid)
                i += 1
    
    return smoothed_path

def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_neural_controller(base_checkpoint: str, fine_tuned_checkpoint: str) -> NeuralCLBFController:
    """Initialize and configure the neural controller."""
    neural_controller = setup_controller(base_checkpoint, fine_tuned_checkpoint)
    logging.info("Neural CLBF controller loaded from checkpoint.")
    return neural_controller

def setup_environment(goal_position: torch.Tensor, start_positions: List[Tuple[float, float, float]], num_obstacles: int, config: dict) -> Tuple[List[Tuple[torch.Tensor, float]], List[Tuple[np.ndarray, float]]]:
    """Setup the simulation environment including obstacles."""
    obstacles = generate_random_obstacles(
        goal_position,
        start_positions,
        num_obstacles,
        x_range=20.0,
        y_range=20.0,
        z_range=5.0,
        min_radius=0.5,
        max_radius=1.5
    )

    logging.info(f"Generated {len(obstacles)} random obstacles:")
    for i, (pos, radius) in enumerate(obstacles):
        logging.info(f"Obstacle {i+1}: position {pos.tolist()}, radius {radius}")

    obstacles_list = [(obs[0].numpy(), obs[1]) for obs in obstacles]
    add_obstacles_to_astrobee_xml(obstacles, goal_position.tolist(), config['goal_threshold'])
    
    return obstacles, obstacles_list

def configure_dynamics(neural_controller: NeuralCLBFController, dynamics_model: SixDOFVehicle, obstacles: List[Tuple[torch.Tensor, float]], goal_position: torch.Tensor):
    """Configure the dynamics model with obstacles and goal."""
    # Set the dynamics model in the controller
    neural_controller.dynamics_model = dynamics_model
    
    # Configure obstacles and goal
    neural_controller.dynamics_model.obstacles = [ ## type: ignore
        (pos.clone().detach(), radius) for pos, radius in obstacles
    ]
    neural_controller.dynamics_model.obstacle_buffer = 4.0 ## type: ignore
    neural_controller.dynamics_model.set_goal(goal_position.clone().detach()) ## type: ignore
    logging.info(f"Goal position set to: {goal_position.tolist()}")

def process_simulation_results(
    simulation_results,
    goal_position_list,
    goal_position_tuple,
    obstacles_list,
    config
) -> Tuple[List[List[np.ndarray]], List[float], List[float], List[Tuple[float, Dict[str, float]]]]:
    """Process simulation results and calculate metrics."""
    trajectories = []
    efficiencies = []
    fuel_costs = []
    safety_scores = []

    for result in simulation_results:
        agent_id = result['agent_id']
        trajectory = result['trajectory']
        collision = result['collision']
        total_fuel_cost = result['total_fuel_cost']
        
        logging.info(f"\nProcessing results for Agent {agent_id}:")
        
        if not trajectory:
            logging.warning(f"Agent {agent_id} has no trajectory data - skipping")
            continue

        trajectories.append(trajectory)
        fuel_costs.append(total_fuel_cost)

        final_position = np.array(trajectory[-1][:3])
        goal_position = np.array(goal_position_list[:3])
        final_distance = np.linalg.norm(goal_position - final_position)
        
        # Detailed status determination
        goal_reached = final_distance <= config['goal_threshold']
        timeout = len(trajectory) >= config['max_simulation_steps']
        
        logging.info(f"Status determination for Agent {agent_id}:")
        logging.info(f"  - Final distance to goal: {final_distance:.2f}")
        logging.info(f"  - Goal threshold: {config['goal_threshold']}")
        logging.info(f"  - Trajectory length: {len(trajectory)}/{config['max_simulation_steps']}")
        logging.info(f"  - Collision occurred: {collision}")
        logging.info(f"  - Goal reached: {goal_reached}")
        logging.info(f"  - Timeout occurred: {timeout}")

        metrics = compute_performance_metrics(trajectory, goal_position_list, collision)
        logging.info(f"Performance metrics for Agent {agent_id}: {metrics}")

        start_position = tuple(trajectory[0][:3])
        logging.info(f"Calculating A* path from {start_position} to {goal_position_tuple}")
        a_star_path = a_star(start_position, goal_position_tuple, obstacles_list, config['astrobee_radius'], config)
        
        if not a_star_path:
            logging.warning(f"A* path calculation failed for Agent {agent_id}")
        
        actual_path = np.array([state[:3] for state in trajectory])
        efficiency = calculate_path_efficiency(actual_path, a_star_path, collision, config)
        efficiencies.append(efficiency)
        
        # Update result with detailed status
        if collision:
            status_reason = 'Collision'
            logging.info(f"Agent {agent_id} failed due to collision")
            if result.get('collision_time'):
                logging.info(f"  - Collision occurred at time: {result['collision_time']:.2f}s")
                logging.info(f"  - Collision position: {result['collision_position']}")
                logging.info(f"  - Collided with obstacle: {result['collision_obstacle_id']}")
        elif timeout:
            status_reason = 'Timeout'
            logging.info(f"Agent {agent_id} failed due to timeout after {len(trajectory)} steps")
        elif goal_reached:
            status_reason = 'N/A'
            logging.info(f"Agent {agent_id} successfully reached goal")
        else:
            status_reason = 'Unknown'
            logging.info(f"Agent {agent_id} failed to reach goal (unknown reason)")
            
        result['reason'] = status_reason        
        safety_score, safety_metrics = calculate_safety_score(
            trajectory, 
            obstacles_list, 
            config['astrobee_radius']
        )
        safety_scores.append((safety_score, safety_metrics))

        result['performance_metrics'] = metrics
        result['efficiency'] = efficiency
        result['safety_score'] = safety_score
        result['safety_metrics'] = safety_metrics
        result['goal_reached'] = goal_reached
        result['timeout'] = timeout
        result['final_distance'] = final_distance

        log_agent_metrics(agent_id, efficiency, total_fuel_cost, metrics, result['control_history'])
        log_safety_metrics(agent_id, safety_score, safety_metrics)

    logging.info("\nOverall simulation results summary:")
    logging.info(f"Total agents processed: {len(trajectories)}")
    if trajectories:
        logging.info(f"Average efficiency: {np.mean(efficiencies):.2%}")
        logging.info(f"Average fuel cost: {np.mean(fuel_costs):.2f}")
        logging.info(f"Average safety score: {np.mean([score for score, _ in safety_scores]):.2%}")

    return trajectories, efficiencies, fuel_costs, safety_scores

def log_agent_metrics(agent_id: int, efficiency: float, total_fuel_cost: float, metrics: Dict, control_history: List[np.ndarray]):
    """Log metrics for a single agent."""
    logging.info(f"Agent {agent_id} path efficiency: {efficiency:.2f} (1.00 is optimal)")
    logging.info(f"Agent {agent_id} total fuel cost: {total_fuel_cost:.2f}")
    
    if metrics['total_distance_traveled'] > 0:
        fuel_efficiency = total_fuel_cost / metrics['total_distance_traveled']
        logging.info(f"Agent {agent_id} fuel efficiency: {fuel_efficiency:.2f} units/meter")
        
    logging.info(f"Agent {agent_id} maximum speed: {metrics['max_speed']:.2f} m/s")
    logging.info(f"Agent {agent_id} average speed: {metrics['avg_speed']:.2f} m/s")

    controls = np.stack(control_history)  # Stack list of arrays properly
    translation_controls = controls[:, 0:3]  # Access as separate steps
    rotation_controls = controls[:, 3:6]
    
    avg_translation_usage = np.mean(np.abs(translation_controls))
    avg_rotation_usage = np.mean(np.abs(rotation_controls))
    
    logging.info(f"Agent {agent_id} average translation usage: {avg_translation_usage:.3f}")
    logging.info(f"Agent {agent_id} average rotation usage: {avg_rotation_usage:.3f}")

def log_safety_metrics(agent_id: int, safety_score: float, safety_metrics: Dict[str, float]):
    """Log safety metrics for a single agent."""
    logging.info(f"Agent {agent_id} safety metrics:")
    logging.info(f"  Overall Safety Score: {safety_score:.2%}")
    logging.info(f"  Velocity Safety: {safety_metrics['velocity_score']:2%}")
    logging.info(f"  Obstacle Safety: {safety_metrics['obstacle_score']:2%}")

def plot_secondary_visualizations(neural_controller: NeuralCLBFController):
    """Generate and display secondary plots."""
    config = setup_simulation_config()
    try:
        if config['plot_barrier_function']:
            bf_experiment = BFContourExperiment(
                name="BF_Contour",
                domain=[(-20, 20), (-20, 20)],
                n_grid=100,
                x_axis_index=0,
                y_axis_index=1,
                x_axis_label="$x$",
                y_axis_label="$y$",
                default_state=torch.tensor([0.0] * neural_controller.dynamics_model.n_dims)
            )
            results_df_bf = bf_experiment.run(neural_controller)
            results_df_bf.to_csv("bf_contour_data.csv", index=False)
            logging.info("Saved BF contour data")
            
            try:
                bf_data = pd.read_csv("bf_contour_data.csv")
                plot_saved_results.plot_bf_contour(bf_data)
            except Exception as e:
                logging.error(f"Error plotting barrier function contour: {str(e)}")
        
        if config['plot_clf_contour']:
            clf_experiment = CLFContourExperiment(
                name="CLF_Contour",
                domain=[(-20, 20), (-20, 20)],
                n_grid=100,
                x_axis_index=0,
                y_axis_index=1,
                x_axis_label="$x$",
                y_axis_label="$y$",
                default_state=torch.tensor([0.0] * neural_controller.dynamics_model.n_dims)
            )
            results_df_clf = clf_experiment.run(neural_controller)
            results_df_clf.to_csv("clf_contour_data.csv", index=False)
            logging.info("Saved CLF contour data")
            
            try:
                clf_data = pd.read_csv("clf_contour_data.csv")
                plot_saved_results.plot_clf_contour(clf_data)
            except Exception as e:
                logging.error(f"Error plotting CLF contour: {str(e)}")
                import traceback
                logging.error(traceback.format_exc())
            
    except Exception as e:
        logging.error(f"Error in plot_secondary_visualizations: {str(e)}")

def setup_dynamics_model(controller, controller_period: float = 0.05) -> SixDOFVehicle:
    """Setup the dynamics model for the simulation."""
    from typing import Dict, Union, List
    from torch import Tensor
    
    # Define Scenario type properly
    Scenario = Dict[str, Union[float, Tensor]]
    ScenarioList = List[Scenario]

    nominal_params: Scenario = { # type: ignore
        "mass": torch.tensor(9.58).float(),
        "inertia_matrix": torch.eye(3).float(),
        "gravity": torch.tensor([0.0, 0.0, 0.0]).float(),
    }

    scenarios: ScenarioList = [nominal_params] # type: ignore
    dynamics_model = SixDOFVehicle(
        nominal_params,
        dt=0.01,
        controller_dt=controller_period,
        scenarios=scenarios,
    )
    
    return dynamics_model

def plot_single_agent_trajectory(
    trajectory: List[np.ndarray],
    goal_position: Tuple[float, float, float],
    obstacles: List[Tuple[np.ndarray, float]],
    agent_id: int,
    object_radius: float,
    efficiency: float,
    fuel_cost: float,
    safety_score: Tuple[float, Dict[str, float]],
    run_number: int,
    goal_reached: bool,
    collision_occurred: bool,
    output_path: Optional[str] = None
) -> None:
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "scene", "rowspan": 3}, {"type": "table"}],
            [None, {"type": "table"}],
            [None, {"type": "table"}]
        ],
        subplot_titles=(
            f'Agent {agent_id} Trajectory (Run {run_number})', 
            'Performance Metrics', 
            'Safety Metrics',
            'Goal Status'
        ),
        vertical_spacing=0.05,
        horizontal_spacing=0.05
    )

    ## Calculate plot bounds
    all_x, all_y, all_z = [], [], []
    
    ## Plot trajectory
    points = np.array(trajectory)[:, :3]
    agent_color = 'blue'
    
    all_x.extend(points[:, 0])
    all_y.extend(points[:, 1])
    all_z.extend(points[:,2])
    
    ## Main trajectory
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='lines',
            name='Actual Path',
            line=dict(width=5, color=agent_color),
            showlegend=True
        ),
        row=1, col=1
    )
    
    ## Start point
    fig.add_trace(
        go.Scatter3d(
            x=[points[0, 0]], y=[points[0, 1]], z=[points[0, 2]],
            mode='markers',
            name='Start',
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
    
    ## End point
    fig.add_trace(
        go.Scatter3d(
            x=[points[-1, 0]], y=[points[-1, 1]], z=[points[-1, 2]],
            mode='markers',
            name='End',
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
    
    ## Calculate and add A* path
    start_position = tuple(points[0])
    config = setup_simulation_config()
    a_star_path = a_star(start_position, goal_position, obstacles, object_radius, config)
    if a_star_path:
        a_star_points = np.array(a_star_path)
        all_x.extend(a_star_points[:, 0])
        all_y.extend(a_star_points[:, 1])
        all_z.extend(a_star_points[:, 2])
        
        fig.add_trace(
            go.Scatter3d(
                x=a_star_points[:, 0],
                y=a_star_points[:, 1],
                z=a_star_points[:, 2],
                mode='lines',
                name='Optimal Path',
                line=dict(
                    color=agent_color,
                    width=3,
                    dash='dash'
                ),
                showlegend=True
            ),
            row=1, col=1
        )

    ## Add obstacles
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

    ## Add goal marker
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

    ## Calculate axis ranges
    max_range = max(
        max(all_x) - min(all_x),
        max(all_y) - min(all_y),
        max(all_z) - min(all_z)) / 2.0
    
    mid_x = (max(all_x) + min(all_x)) / 2
    mid_y = (max(all_y) + min(all_y)) / 2
    mid_z = (max(all_z) + min(all_z)) / 2

    ## Performance metrics table
    metrics_data = [
        ['Path Efficiency', f'{efficiency:.2%}'],
        ['Fuel Cost', f'{fuel_cost:.2f}']
    ]

    fig.add_trace(
        go.Table(
            header=dict(
                values=['Metric', 'Value'],
                align='center',
                font=dict(size=12, color='white'),
                fill_color='darkblue'
            ),
            cells=dict(
                values=list(zip(*metrics_data)),
                align='center',
                font=dict(size=11),
                fill_color=[['lightgray', 'white'] * (len(metrics_data)//2 + 1)]
            )
        ),
        row=1, col=2
    )

    ## Safety metrics table
    safety_data = [
        ['Overall Safety', f'{safety_score[0]:.2%}'],
        ['Velocity Safety', f'{safety_score[1]["velocity_score"]:.2%}'],
        ['Obstacle Safety', f'{safety_score[1]["obstacle_score"]:.2%}']
    ]

    fig.add_trace(
        go.Table(
            header=dict(
                values=['Metric', 'Value'],
                align='center',
                font=dict(size=12, color='white'),
                fill_color='darkblue'
            ),
            cells=dict(
                values=list(zip(*safety_data)),
                align='center',
                font=dict(size=11),
                fill_color=[['lightgray', 'white'] * (len(safety_data)//2 + 1)]
            )
        ),
        row=2, col=2
    )

    ## Goal status table
    goal_data = [
        ['Goal Status', 'REACHED' if goal_reached else 'NOT REACHED'],
        ['Final Distance', f'{np.linalg.norm(np.array(trajectory[-1][:3]) - np.array(goal_position)):.2f}m'],
        ['Time Steps', f'{len(trajectory)}'],
        ['Reason', 'N/A' if goal_reached else 
                  'Timeout' if len(trajectory) >= config['max_simulation_steps'] else
                  'Collision' if collision_occurred else 'Unknown']
    ]

    fig.add_trace(
        go.Table(
            header=dict(
                values=['Metric', 'Value'],
                align='center',
                font=dict(size=12, color='white'),
                fill_color='darkgreen' if goal_reached else 'darkred'
            ),
            cells=dict(
                values=list(zip(*goal_data)),
                align='center',
                font=dict(size=11),
                fill_color=[['lightgray', 'white'] * (len(goal_data)//2 + 1)]
            )
        ),
        row=3, col=2
    )

    ## Update layout
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
        width=1800,
        height=900,  # Increased height to better accommodate tables
        showlegend=True,
        title=f"Agent {agent_id} Navigation Analysis - Run {run_number}"
    )

    if output_path:
        fig.write_html(output_path)
    else:
        fig.show()

def main():
    """Main execution function."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-saved', action='store_true', 
                      help='Load saved contour data instead of rerunning experiments')
    args = parser.parse_args()
    
    ## if you change any settings, do it here.
    config = setup_simulation_config(
        ## Camera and view settings

        ### change these to change the camera view
        cam_pos=[0, -40, 20],
        cam_rot=[0.525398163397448, 1.5500000000000005],
        zoom_level=-21.0,

        ## do not change these
        last_mouse_x=0,
        last_mouse_y=0,
        mouse_button_pressed=False,
        
        ## Simulation state (should be left alone)
        recording=False,
        video_writer=None,
        frame_count=0,
        render_frame=0,
        last_rrt_update=0,
        
        ## Vehicle parameters
        astrobee_radius=0.26,
        obstacle_buffer_distance=0.2,
        goal_threshold=1.5,
        
        ## Cost parameters
        translation_fuel_cost=1.0,
        rotation_fuel_cost=0.5,
        
        ## Viewport (do not change)
        viewport=None,
        
        ## Plotly visualizations (primary is the main 3d agent/obstacle visualization, secondary is the barrier function and CLF contour)
        plot_primary_visualization=True,
        plot_barrier_function=False,
        plot_clf_contour=False,
        
        ## Reset config (do not change)
        reset=False,
        
        ## Vehicle control parameters
        velocity_limit=0.1,  ## Default velocity limit
        force_limit=0.849,   ## Default force limit
        torque_limit=0.1,    ## Default torque limit
        
        ## Controller parameters
        clf_lambda=2,        ## Default lambda value
        controller_period=0.05,  ## Default controller period
        
        ## Simulation parameters
        model_timestep=0.1,  ## Default model timestep (matches trained one and in rqbf.py)
        max_simulation_steps=6000,  ## Default max steps (how long the sim will go roughly (rtt is variable))
        
        ## Debug and update frequencies
        debug_frequency=50,   ## Default debug frequency
        rrt_update_frequency=25,  ## Default RRT update frequency
        render_frequency=4,   ## Default render frequency
        terminate_requested=False,
    )
    
    setup_logging()

    # Setup checkpoints and controller

    base_checkpoint = r"C:\Users\Tetra\Documents\Repostories\OSCorp\Project-Libra\checkpoints\six_dof_vehicle\epoch=0-val_loss=3.89.ckpt"
    fine_tuned_checkpoint = r"C:\Users\Tetra\Documents\Repostories\OSCorp\Project-Libra\checkpoints\mujoco_six_dof_vehicle\epoch=0-val_loss=4.91.ckpt"
    neural_controller = setup_neural_controller(base_checkpoint, fine_tuned_checkpoint)

    # Setup dynamics model
    dynamics_model = setup_dynamics_model(neural_controller)

    # Setup environment
    goal_position = torch.tensor([1.0, -3.0, 1.0])
    start_positions = [
        (15.0, 0.0, 0.0),
        (-15.0, 0.0, 0.0),
        (0.0, 15.0, 0.0),
        (0.0, -15.0, 0.0)
    ]
    obstacles, obstacles_list = setup_environment(goal_position, start_positions, num_obstacles=6, config=config)
    
    # Configure dynamics with the dynamics model
    configure_dynamics(neural_controller, dynamics_model, obstacles, goal_position)

    # Run simulation
    goal_position_list = goal_position.tolist()
    goal_position_tuple = tuple(goal_position_list)
    
    logging.info("Starting simulations for all agents.")
    simulation_results = run_simulation(
        start_positions, 
        goal_position_tuple, 
        obstacles_list,
        neural_controller
    )
    logging.info("All simulations completed.")

    # Process results
    trajectories, efficiencies, fuel_costs, safety_scores = process_simulation_results(
        simulation_results,
        goal_position_list,
        goal_position_tuple,
        obstacles_list,
        config
    )

    if config['plot_primary_visualization']:
        plot_trajectories_3d(
            trajectories, 
            goal_position_tuple, 
            obstacles_list, 
            config['astrobee_radius'], 
            efficiencies,
            fuel_costs,
            safety_scores,
            config,
            simulation_results=simulation_results,
            controller_name="Neural CLBF Controller"
        )
    
    plot_secondary_visualizations(neural_controller)
    
    logging.info("Simulation complete.")

if __name__ == "__main__":
    main()