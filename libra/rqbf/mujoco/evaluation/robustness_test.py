## Known issues:
## 1. Mujoco window has to be full screened and kept open for it to record videos (will crash if not, either need to fix this crashing and allow it to record with it minimized, or not allow the window to be minimized)
## 2. Starting position/goal position is not yet randomized

import imageio
import torch
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional, Tuple
from pathlib import Path
import json
from statistics import mean, median
import time
import glfw
from libra.rqbf.mujoco.evaluation.main_3d import (
    add_obstacles_to_astrobee_xml,
    setup_simulation_config,
    setup_neural_controller,
    setup_dynamics_model,
    setup_environment,
    configure_dynamics,
    run_simulation,
    process_simulation_results,
    plot_single_agent_trajectory,
    setup_glfw,
    setup_mujoco_visualization,
    setup_mujoco
)
from libra.Other_controller.LQR.LQR_wrapper import LQRControllerWrapper
from libra.Other_controller.MPC.mpc_wrapper import MPCControllerWrapper
from enum import Enum
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class TestMode(Enum):
    """Enum for different test modes"""
    NEURAL_CLBF_ONLY = "neural_clbf"
    ALL_CONTROLLERS = "all"

def format_time(seconds: float) -> str:
    """Convert seconds to a human-readable format."""
    return str(timedelta(seconds=int(seconds)))

class RobustnessTest:
    def __init__(
        self,
        num_runs: int = 10,
        base_checkpoint: Union[str, None] = None,
        fine_tuned_checkpoint: Union[str, None] = None,
        output_dir: str = "robustness_test_results",
        num_obstacles: Union[int, Tuple[int, int]] = 6,
        record_videos: bool = True,
        test_mode: TestMode = TestMode.NEURAL_CLBF_ONLY
    ):
        self.num_runs = num_runs
        self.base_checkpoint = base_checkpoint
        self.fine_tuned_checkpoint = fine_tuned_checkpoint
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / self.timestamp
        self.num_obstacles = num_obstacles
        self.record_videos = record_videos
        self.start_time = None
        self.test_mode = test_mode
        self.window = None
        
        self.setup_directories()
        self.setup_logging()
        self.setup_config()
        self.setup_mujoco()

    def setup_mujoco(self):
        """Setup MuJoCo and GLFW window."""
        try:
            self.model, self.data = setup_mujoco()
            self.window = setup_glfw()
            self.context, self.scene, self.cam, self.opt = setup_mujoco_visualization(self.model)
            logging.info("MuJoCo and GLFW initialized successfully")
        except Exception as e:
            logging.error(f"Error setting up MuJoCo/GLFW: {str(e)}")
            raise

    def cleanup(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'window') and self.window is not None:
                try:
                    # Check if GLFW is still initialized
                    if glfw.get_current_context() is not None:
                        glfw.make_context_current(self.window)
                        glfw.destroy_window(self.window)
                        self.window = None
                except Exception as e:
                    logging.debug(f"Window cleanup error: {str(e)}")
                
                try:
                    # Only terminate if GLFW is still initialized
                    if glfw.get_current_context() is not None:
                        glfw.terminate()
                        logging.info("GLFW terminated")
                except Exception as e:
                    logging.debug(f"GLFW termination error: {str(e)}")
        except Exception as e:
            logging.debug(f"Final cleanup error: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()

    def get_active_controllers(self) -> Dict:
        """Get dictionary of controllers based on test mode"""
        if self.test_mode == TestMode.NEURAL_CLBF_ONLY:
            if not self.base_checkpoint or not self.fine_tuned_checkpoint:
                raise ValueError("Neural CLBF controller requires both base and fine-tuned checkpoints")
            return {
                'neural_clbf': setup_neural_controller(
                    str(self.base_checkpoint),
                    str(self.fine_tuned_checkpoint)
                )
            }
        else:
            controllers = {}
            if self.base_checkpoint and self.fine_tuned_checkpoint:
                controllers['neural_clbf'] = setup_neural_controller(
                    str(self.base_checkpoint),
                    str(self.fine_tuned_checkpoint)
                )
            controllers['lqr'] = self.setup_lqr_controller()
            controllers['mpc'] = self.setup_mpc_controller()
            return controllers

    def setup_directories(self):
        """Create output directories for results and videos."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.record_videos:
            self.video_dir = self.output_dir / "videos"
            self.video_dir.mkdir(exist_ok=True)
            
        self.metrics_dir = self.output_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
    def setup_logging(self):
        """Configure logging for the robustness test."""
        log_file = self.output_dir / "robustness_test.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def setup_config(self):
        """Setup simulation configuration with specific robustness test settings."""
        self.config = setup_simulation_config(
            cam_pos=[0, -40, 20],
            cam_rot=[0.525398163397448, 1.5500000000000005],
            zoom_level=-21.0,
            plot_primary_visualization=False,
            plot_barrier_function=False,
            plot_clf_contour=False,
            recording=self.record_videos,
            max_simulation_steps=6000
        )
        
    def setup_lqr_controller(self) -> Optional[LQRControllerWrapper]:
        """Setup the LQR controller."""
        try:
            lqr_controller = LQRControllerWrapper(use_rrt=False)
            lqr_controller.setup()
            logging.info("LQR controller initialized successfully")
            return lqr_controller
        except Exception as e:
            logging.error(f"Error setting up LQR controller: {str(e)}")
            return None

    def setup_mpc_controller(self) -> Optional[MPCControllerWrapper]:
        """Setup the MPC controller."""
        try:
            mpc_controller = MPCControllerWrapper(use_rrt=False)
            mpc_controller.setup()
            logging.info("MPC controller initialized successfully")
            return mpc_controller
        except Exception as e:
            logging.error(f"Error setting up MPC controller: {str(e)}")
            return None

    def get_random_obstacle_count(self) -> int:
        """Generate a random number of obstacles based on the configuration."""
        if isinstance(self.num_obstacles, tuple):
            min_obstacles, max_obstacles = self.num_obstacles
            return random.randint(min_obstacles, max_obstacles)
        return self.num_obstacles
        
    def get_controller_period(self, controller) -> float:
        """Get the controller period, with fallback for controllers that don't have it."""
        if hasattr(controller, 'controller_period'):
            return controller.controller_period
        return 0.05
        
    def run_single_test(self, run_number: int) -> Dict:
        """Execute a single robustness test run."""
        run_start_time = time.time()
        logging.info(f"Starting robustness test run {run_number}/{self.num_runs}")
        
        run_results = {}
        
        controllers = self.get_active_controllers()
        
        for controller_name, controller in controllers.items():
            if controller is None:
                logging.info(f"Skipping {controller_name} controller - not configured")
                continue
                
            logging.info(f"Running simulation with {controller_name} controller")
            
            controller_period = self.get_controller_period(controller)
            dynamics_model = setup_dynamics_model(controller, controller_period)
            
            goal_position = torch.tensor([1.0, -3.0, 1.0])
            start_positions = [
                (15.0, 0.0, 0.0),
                (-15.0, 0.0, 0.0),
                (0.0, 15.0, 0.0),
                (0.0, -15.0, 0.0)
            ]
            
            obstacle_count = self.get_random_obstacle_count()
            logging.info(f"Run {run_number}: Using {obstacle_count} obstacles")
            
            obstacles, obstacles_list = setup_environment(
                goal_position,
                start_positions,
                obstacle_count,
                self.config
            )
            
            configure_dynamics(controller, dynamics_model, obstacles, goal_position)
            
            agent_times = []
            agent_results = []
            trajectories = []
            
            for agent_idx, start_pos in enumerate(start_positions):
                agent_start_time = time.time()
                
                if self.record_videos:
                    video_path = self.video_dir / f"{controller_name}_run_{run_number:03d}_agent_{agent_idx+1:02d}.mp4"
                    video_writer = imageio.get_writer(str(video_path), fps=30)
                    self.config['video_writer'] = video_writer
                
                single_agent_results = run_simulation(
                    [start_pos],
                    tuple(goal_position.tolist()),
                    obstacles_list,
                    controller,
                    controller_type=controller_name
                )
                
                if self.record_videos and 'video_writer' in self.config:
                    self.config['video_writer'].close()
                
                agent_end_time = time.time()
                agent_duration = agent_end_time - agent_start_time
                agent_times.append(agent_duration)
                agent_results.extend(single_agent_results)
                
                logging.info(f"Run {run_number}, {controller_name}, Agent {agent_idx + 1} completed in {format_time(agent_duration)}")
            
            trajectories, efficiencies, fuel_costs, safety_scores = process_simulation_results(
                agent_results,
                goal_position.tolist(),
                tuple(goal_position.tolist()),
                obstacles_list,
                self.config
            )
            
            for i, (trajectory, efficiency, fuel_cost, safety_score) in enumerate(zip(trajectories, efficiencies, fuel_costs, safety_scores)):
                plot_path = self.plots_dir / f"{controller_name}_run_{run_number:03d}_agent_{i+1:02d}.html"
                
                goal_reached = efficiency > 0
                
                plot_single_agent_trajectory(
                    trajectory=trajectory,
                    goal_position=tuple(goal_position.tolist()),
                    obstacles=obstacles_list,
                    agent_id=i + 1,
                    object_radius=self.config['astrobee_radius'],
                    efficiency=efficiency,
                    fuel_cost=fuel_cost,
                    safety_score=safety_score,
                    run_number=run_number,
                    goal_reached=goal_reached,
                    collision_occurred=agent_results[i]['collision'],
                    output_path=str(plot_path)
                )
                logging.info(f"Generated trajectory plot for Run {run_number}, {controller_name}, Agent {i + 1}")
            
            controller_metrics = {
                'run_number': run_number,
                'controller': controller_name,
                'run_time': {
                    'total': time.time() - run_start_time,
                    'formatted': format_time(time.time() - run_start_time)
                },
                'agent_times': {
                    f'agent_{i+1}': {
                        'time': t,
                        'formatted': format_time(t)
                    } for i, t in enumerate(agent_times)
                },
                'agent_metrics': [
                    {
                        'agent_id': i + 1,
                        'efficiency': eff,
                        'fuel_cost': fuel,
                        'safety_score': safety[0],
                        'safety_metrics': safety[1],
                        'time': agent_times[i],
                        'time_formatted': format_time(agent_times[i]),
                        'collision': result['collision'],
                        'collision_details': {
                            'had_collision': result['collision'],
                            'collision_time': result.get('collision_time', None),
                            'collision_position': result.get('collision_position', None),
                            'collision_obstacle_id': result.get('collision_obstacle_id', None)
                        } if result['collision'] else None,
                        'performance_metrics': result['performance_metrics']
                    }
                    for i, (eff, fuel, safety, result) in enumerate(zip(efficiencies, fuel_costs, safety_scores, agent_results))
                ]
            }
            
            run_results[controller_name] = controller_metrics
            
            metrics_file = self.metrics_dir / f"run_{run_number:03d}_{controller_name}_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(controller_metrics, f, indent=4)
        
        return run_results
        
    def calculate_aggregate_metrics(self, obstacle_results: List[Dict]) -> Dict:
        """Calculate aggregate metrics for all controllers at a specific obstacle count."""
        controller_metrics = {}
        
        # Group results by controller
        for run_result in obstacle_results:
            for controller_name, run_data in run_result.items():
                if controller_name not in controller_metrics:
                    controller_metrics[controller_name] = {
                        'total_runs': 0,
                        'successful_runs': 0,
                        'collisions': 0,
                        'efficiencies': [],
                        'fuel_costs': [],
                        'safety_scores': []
                    }
                
                metrics = run_data['metrics']
                controller_metrics[controller_name]['total_runs'] += 1
                controller_metrics[controller_name]['successful_runs'] += (
                    1 if not metrics['collision'] and metrics['efficiency'] > 0 else 0
                )
                controller_metrics[controller_name]['collisions'] += (
                    1 if metrics['collision'] else 0
                )
                controller_metrics[controller_name]['efficiencies'].append(metrics['efficiency'])
                controller_metrics[controller_name]['fuel_costs'].append(metrics['fuel_cost'])
                controller_metrics[controller_name]['safety_scores'].append(metrics['safety_score'])
        
        # Calculate averages and rates for each controller
        aggregate_metrics = {}
        for controller_name, data in controller_metrics.items():
            total_runs = data['total_runs']
            aggregate_metrics[controller_name] = {
                'success_rate': data['successful_runs'] / total_runs,
                'collision_rate': data['collisions'] / total_runs,
                'average_efficiency': sum(data['efficiencies']) / total_runs,
                'average_fuel_cost': sum(data['fuel_costs']) / total_runs,
                'average_safety_score': sum(data['safety_scores']) / total_runs,
                'total_runs': total_runs,
                'successful_runs': data['successful_runs'],
                'collisions': data['collisions']
            }
        
        return aggregate_metrics
        
    def run(self):
        """Execute the complete robustness test suite."""
        self.start_time = time.time()
        logging.info(f"Starting robustness test suite with {self.num_runs} runs in {self.test_mode.value} mode")
        
        all_run_metrics = []
        try:
            for run_number in range(1, self.num_runs + 1):
                run_metrics = self.run_single_test(run_number)
                all_run_metrics.append(run_metrics)
                
                config = setup_simulation_config()
                if config['terminate_requested']:
                    logging.info("Termination requested - stopping robustness test")
                    break
                
        except KeyboardInterrupt:
            logging.info("Test suite interrupted by user")
        finally:
            total_time = time.time() - self.start_time
            logging.info(f"Total test suite time: {format_time(total_time)}")
                
            if all_run_metrics:
                aggregate_metrics = self.calculate_aggregate_metrics(all_run_metrics)
                logging.info(f"Robustness test suite completed. Results saved to {self.output_dir}")
                return aggregate_metrics
            else:
                logging.warning("No metrics collected - test suite may have been interrupted")
                return None

    def systematic_evaluation(
        self,
        n_runs: int,
        max_obstacles: int,
        starting_points: List[Tuple[float, float, float]],
        output_subdir: str = "systematic_evaluation"
    ) -> Dict:
        """
        Run systematic evaluation of controllers across different starting points and obstacle counts.
        """
        evaluation_start_time = time.time()
        logging.info("Starting systematic evaluation")
        
        # Setup directories
        eval_output_dir = self.output_dir / output_subdir
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        
        obstacle_dirs = {
            n_obstacles: eval_output_dir / f"obstacles_{n_obstacles}"
            for n_obstacles in range(1, max_obstacles + 1)
        }
        for dir_path in obstacle_dirs.values():
            dir_path.mkdir(exist_ok=True)
            if self.record_videos:
                (dir_path / "videos").mkdir(exist_ok=True)
            (dir_path / "metrics").mkdir(exist_ok=True)
        
        # Initialize results structure
        final_results = {
            'neural_clbf': {},
            'lqr': {},
            'mpc': {}
        }
        
        try:
            if self.window:
                glfw.make_context_current(self.window)
            
            # For each obstacle count
            for n_obstacles in range(1, max_obstacles + 1):
                logging.info(f"\nStarting evaluation with {n_obstacles} obstacle(s)")
                self.num_obstacles = n_obstacles
                current_output_dir = obstacle_dirs[n_obstacles]
                obstacle_results = []
                
                # For each run
                for run_number in range(1, n_runs + 1):
                    # Choose starting point and setup obstacles once per run
                    start_pos = starting_points[run_number % len(starting_points)]
                    logging.info(f"\nRun {run_number}/{n_runs} with {n_obstacles} obstacle(s)")
                    logging.info(f"Run number {n_obstacles*run_number} of {max_obstacles*n_runs}")
                    # logging.info(f"Using starting position: {start_pos}")
                    
                    # Setup environment once for all controllers
                    goal_position = torch.tensor([1.0, -3.0, 1.0])
                    obstacles, obstacles_list = setup_environment(
                        goal_position,
                        [start_pos],
                        n_obstacles,
                        self.config
                    )
                    
                    run_results = {}
                    
                    # Get all controllers
                    controllers = self.get_active_controllers()
                    logging.info(f"Testing controllers: {list(controllers.keys())}")
                    
                    # Run each controller with the same obstacles and starting point
                    for controller_name, controller in controllers.items():
                        if controller is None:
                            logging.info(f"******************Skipping {controller_name} controller - not configured*************************")
                            continue
                        
                        try:
                            logging.info(f"Running {controller_name} controller")
                            
                            # Setup controller-specific components
                            controller_period = self.get_controller_period(controller)
                            dynamics_model = setup_dynamics_model(controller, controller_period)
                            configure_dynamics(controller, dynamics_model, obstacles, goal_position)
                            
                            # Setup video recording
                            if self.record_videos:
                                video_path = current_output_dir / "videos" / f"{controller_name}_run_{run_number:03d}.mp4"
                                video_writer = imageio.get_writer(str(video_path), fps=30)
                                self.config['video_writer'] = video_writer
                            
                            # Run simulation
                            results = run_simulation(
                                [start_pos],
                                tuple(goal_position.tolist()),
                                obstacles_list,
                                controller,
                                controller_type=controller_name
                            )
                            
                            # Close video writer
                            if self.record_videos and 'video_writer' in self.config:
                                self.config['video_writer'].close()
                            
                            # Process results
                            trajectories, efficiencies, fuel_costs, safety_scores = process_simulation_results(
                                results,
                                goal_position.tolist(),
                                tuple(goal_position.tolist()),
                                obstacles_list,
                                self.config
                            )
                            
                            # Store metrics
                            run_metrics = {
                                'run_number': run_number,
                                'n_obstacles': n_obstacles,
                                'controller': controller_name,
                                'metrics': {
                                    'efficiency': efficiencies[0],
                                    'fuel_cost': fuel_costs[0],
                                    'safety_score': safety_scores[0][0],
                                    'safety_metrics': safety_scores[0][1],
                                    'collision': results[0]['collision'],
                                    'collision_details': {
                                        'had_collision': results[0]['collision'],
                                        'collision_time': results[0].get('collision_time', None),
                                        'collision_position': results[0].get('collision_position', None),
                                        'collision_obstacle_id': results[0].get('collision_obstacle_id', None)
                                    } if results[0]['collision'] else None
                                }
                            }
                            
                            run_results[controller_name] = run_metrics
                            logging.info(f"Completed {controller_name} controller run")
                            
                        except Exception as e:
                            logging.error(f"Error during {controller_name} controller run: {str(e)}")
                            continue
                        
                        # Process GLFW events after each controller
                        if self.window:
                            glfw.poll_events()
                    
                    # After all controllers complete for this run
                    if run_results:
                        obstacle_results.append(run_results)
                        # logging.info(f"Completed run {run_number} with {len(run_results)} controller results ({n_obstacles} obstacle(s))")
                
                # After all runs complete for this obstacle count
                if obstacle_results:
                    logging.info(f"Calculating aggregate metrics for {n_obstacles} obstacles")
                    aggregate_metrics = self.calculate_aggregate_metrics(obstacle_results)
                    
                    # Store results by controller and obstacle count
                    for controller_name, metrics in aggregate_metrics.items():
                        final_results[controller_name][f'{n_obstacles}_obstacles'] = metrics
        
        except Exception as e:
            logging.error(f"Error during systematic evaluation: {str(e)}")
            raise
        finally:
            # Save all results to a single file
            metrics_file = eval_output_dir / "metrics" / "aggregate_metrics.json"
            metrics_file.parent.mkdir(exist_ok=True)
            
            with open(metrics_file, 'w') as f:
                json.dump(final_results, f, indent=4)
            
            # Plot success rates
            plot_success_rates(str(metrics_file))
            
            logging.info(f"\nSystematic evaluation completed in {format_time(time.time() - evaluation_start_time)}")
            return final_results
        


def plot_aggregate_metrics(aggregate_metrics: Dict, output_path: str) -> None:
    """Create a detailed plotly table visualization of aggregate metrics for all controllers."""
    tables_data = []
    
    for controller_type, metrics in aggregate_metrics.items():
        timing_data = [
            ['Total Time Elapsed', metrics['timing_stats']['total_time_elapsed']],
            ['Average Run Time', metrics['timing_stats']['run_stats']['average_run_time']],
            ['Fastest Run', metrics['timing_stats']['run_stats']['fastest_run']],
            ['Slowest Run', metrics['timing_stats']['run_stats']['slowest_run']],
            ['Average Agent Time', metrics['timing_stats']['agent_stats']['average_agent_time']],
            ['Median Agent Time', metrics['timing_stats']['agent_stats']['median_agent_time']],
            ['Fastest Agent', metrics['timing_stats']['agent_stats']['fastest_agent']],
            ['Slowest Agent', metrics['timing_stats']['agent_stats']['slowest_agent']]
        ]

        distance_data = [
            ['Average Distance', metrics['distance_stats']['average_distance']],
            ['Median Distance', metrics['distance_stats']['median_distance']],
            ['Min Distance', metrics['distance_stats']['min_distance']],
            ['Max Distance', metrics['distance_stats']['max_distance']]
        ]

        performance_data = [
            ['Total Runs', str(metrics['total_runs'])],
            ['Total Agents', str(metrics['total_agents'])],
            ['Total Collisions', str(metrics['collision_stats']['total_collisions'])],
            ['Collision Rate', metrics['collision_stats']['collision_rate']],
            ['Total Goal Reaches', str(metrics['goal_reaching_stats']['total_goal_reaches'])],
            ['Success Rate', metrics['goal_reaching_stats']['success_rate']]
        ]

        performance_data.extend(distance_data)

        efficiency_data = [
            ['Average Efficiency', f"{metrics['performance_stats']['average_efficiency']:.2%}"],
            ['Average Fuel Cost', f"{metrics['performance_stats']['average_fuel_cost']:.2f}"],
            ['Average Safety Score', f"{metrics['performance_stats']['average_safety_score']:.2%}"]
        ]
        
        tables_data.append((controller_type, timing_data, performance_data, efficiency_data))

    num_controllers = len(tables_data)
    fig = make_subplots(
        rows=num_controllers, cols=3,
        subplot_titles=[
            f"{controller.upper()} - {title}" 
            for controller, _, _, _ in tables_data 
            for title in ['Timing', 'Performance', 'Efficiency']
        ],
        vertical_spacing=0.05,
        specs=[[{"type": "table"} for _ in range(3)] for _ in range(num_controllers)]
    )

    colors = {
        'neural_clbf': 'darkblue',
        'lqr': 'darkgreen',
        'mpc': 'darkred'
    }

    for idx, (controller, timing, performance, efficiency) in enumerate(tables_data, 1):
        color = colors.get(controller, 'darkgray')
        
        # Timing table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Metric', 'Value'],
                    align='center',
                    font=dict(size=12, color='white'),
                    fill_color=color
                ),
                cells=dict(
                    values=list(zip(*timing)),
                    align='center',
                    font=dict(size=11),
                    fill_color=[['lightgray', 'white'] * (len(timing)//2 + 1)]
                )
            ),
            row=idx, col=1
        )

        # Performance table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Metric', 'Value'],
                    align='center',
                    font=dict(size=12, color='white'),
                    fill_color=color
                ),
                cells=dict(
                    values=list(zip(*performance)),
                    align='center',
                    font=dict(size=11),
                    fill_color=[['lightgray', 'white'] * (len(performance)//2 + 1)]
                )
            ),
            row=idx, col=2
        )

        # Efficiency table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Metric', 'Value'],
                    align='center',
                    font=dict(size=12, color='white'),
                    fill_color=color
                ),
                cells=dict(
                    values=list(zip(*efficiency)),
                    align='center',
                    font=dict(size=11),
                    fill_color=[['lightgray', 'white'] * (len(efficiency)//2 + 1)]
                )
            ),
            row=idx, col=3
        )

    fig.update_layout(
        height=400 * num_controllers,
        width=1200,
        title_text="Aggregate Metrics Summary",
        showlegend=False
    )

    fig.write_html(output_path)
    fig.show()


def setup_environment_systematic(goal_position: torch.Tensor, start_positions: List[Tuple[float, float, float]], num_obstacles: int, config: dict) -> Tuple[List[Tuple[torch.Tensor, float]], List[Tuple[np.ndarray, float]]]:
    """Setup the simulation environment including obstacles."""
    obstacles = []
    
    # Place first obstacle between start and goal
    start_pos = torch.tensor(start_positions[0])
    direction = goal_position - start_pos
    
    # Calculate a point roughly 1/3 to 2/3 of the way from start to goal
    random_fraction = torch.rand(1).item() * 0.3 + 0.3  # Random number between 0.3 and 0.6
    obstacle_pos = start_pos + direction * random_fraction
    
    # Add some random offset to avoid being exactly on the line
    offset = torch.randn(3) * 0.5  # Random offset with standard deviation of 0.5
    offset[2] *= 0.2  # Reduce vertical offset
    obstacle_pos += offset
    
    # Add the first obstacle
    radius = torch.rand(1).item() * (1.5 - 0.5) + 0.5  # Random radius between 0.5 and 1.5
    
    # Check if the first obstacle is too close to the goal
    min_dist_from_goal = 1.0 + 1.5  # 1.0 buffer + max possible radius
    while torch.norm(obstacle_pos - goal_position) <= min_dist_from_goal + radius:
        # Recalculate position if too close
        random_fraction = torch.rand(1).item() * 0.3 + 0.3
        obstacle_pos = start_pos + direction * random_fraction
        offset = torch.randn(3) * 0.5
        offset[2] *= 0.2
        obstacle_pos += offset
    
    obstacles.append((obstacle_pos, radius))
    
    # Generate remaining obstacles completely randomly
    if num_obstacles > 1:
        goal_pos_np = np.array(goal_position.tolist())
        
        for _ in range(num_obstacles - 1):
            valid_position = False
            while not valid_position:
                # Generate random position within specified ranges
                pos = torch.tensor([
                    torch.rand(1).item() * 20.0 - 10.0,  # x: [-10, 10]
                    torch.rand(1).item() * 20.0 - 10.0,  # y: [-10, 10]
                    torch.rand(1).item() * 2.0 - 1.0     # z: [-1, 1]
                ])
                
                # Random radius
                radius = torch.rand(1).item() * (1.5 - 0.5) + 0.5  # Random radius between 0.5 and 1.5
                
                # Check distance from goal
                dist_from_goal = np.linalg.norm(np.array(pos.tolist()) - goal_pos_np)
                if dist_from_goal > min_dist_from_goal + radius:
                    valid_position = True
                    
                    # Also check if it overlaps with existing obstacles
                    for existing_pos, existing_radius in obstacles:
                        if torch.norm(pos - existing_pos) <= radius + existing_radius:
                            valid_position = False
                            break
            
            obstacles.append((pos, radius))

    obstacles_list = [(obs[0].numpy(), obs[1]) for obs in obstacles]
    add_obstacles_to_astrobee_xml(obstacles, goal_position.tolist(), config['goal_threshold'])
    
    return obstacles, obstacles_list





def plot_success_rates(metrics_file: str) -> None:
    """
    Plot success rates for each controller across different obstacle counts with trend lines.
    
    Args:
        metrics_file: Path to the aggregate_metrics.json file
    """
    # Read the metrics file
    with open(metrics_file, 'r') as f:
        results = json.load(f)
    
    # Extract data for plotting
    controllers = list(results.keys())
    obstacle_counts = []
    success_rates = {controller: [] for controller in controllers}
    
    # Get data for each controller
    for controller in controllers:
        for obstacle_key, metrics in results[controller].items():
            count = int(obstacle_key.split('_')[0])
            if count not in obstacle_counts:
                obstacle_counts.append(count)
            success_rates[controller].append(metrics['success_rate'])
    
    # Sort obstacle counts and reorder data accordingly
    obstacle_counts.sort()
    for controller in controllers:
        ordered_rates = []
        for count in obstacle_counts:
            for obs_key, metrics in results[controller].items():
                if int(obs_key.split('_')[0]) == count:
                    ordered_rates.append(metrics['success_rate'])
        success_rates[controller] = ordered_rates
    
    # Create the line plot
    fig = go.Figure()
    
    colors = {
        'neural_clbf': 'blue',
        'lqr': 'green',
        'mpc': 'red'
    }
    
    names = {
        'neural_clbf': 'Neural CLBF',
        'lqr': 'LQR',
        'mpc': 'MPC'
    }
    
    for controller in controllers:
        # Add line plot connecting actual data points
        fig.add_trace(go.Scatter(
            x=obstacle_counts,
            y=success_rates[controller],
            mode='lines+markers',  # Changed to lines+markers to connect dots
            name=names.get(controller, controller),
            line=dict(
                color=colors.get(controller, 'gray'),
                width=2
            ),
            marker=dict(
                color=colors.get(controller, 'gray'),
                size=10
            ),
            showlegend=True
        ))
        
        # Add trend line
        coefficients = np.polyfit(obstacle_counts, success_rates[controller], 3)
        poly = np.poly1d(coefficients)
        x_trend = np.linspace(min(obstacle_counts), max(obstacle_counts), 100)
        y_trend = poly(x_trend)
        
        fig.add_trace(go.Scatter(
            x=x_trend,
            y=y_trend,
            mode='lines',
            name=f'{names.get(controller, controller)} Trend',
            line=dict(
                color=colors.get(controller, 'gray'),
                width=2,
                dash='dash'
            ),
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title='Controller Success Rates vs Number of Obstacles',
        xaxis_title='Number of Obstacles',
        yaxis_title='Success Rate',
        yaxis_tickformat='.0%',
        xaxis=dict(
            tickmode='linear',
            tick0=min(obstacle_counts),
            dtick=1
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        hovermode='x unified'
    )
    
    # Save the plot
    output_dir = Path(metrics_file).parent
    fig.write_html(str(output_dir / 'success_rates.html'))
    fig.show()



def plot_collision_rates(metrics_file: str) -> None:
    """
    Plot success rates for each controller across different obstacle counts with trend lines.
    
    Args:
        metrics_file: Path to the aggregate_metrics.json file
    """
    # Read the metrics file
    with open(metrics_file, 'r') as f:
        results = json.load(f)
    
    # Extract data for plotting
    controllers = list(results.keys())
    obstacle_counts = []
    success_rates = {controller: [] for controller in controllers}
    
    # Get data for each controller
    for controller in controllers:
        for obstacle_key, metrics in results[controller].items():
            count = int(obstacle_key.split('_')[0])
            if count not in obstacle_counts:
                obstacle_counts.append(count)
            success_rates[controller].append(metrics['collision_rate'])
    
    # Sort obstacle counts and reorder data accordingly
    obstacle_counts.sort()
    for controller in controllers:
        ordered_rates = []
        for count in obstacle_counts:
            for obs_key, metrics in results[controller].items():
                if int(obs_key.split('_')[0]) == count:
                    ordered_rates.append(metrics['collision_rate'])
        success_rates[controller] = ordered_rates
    
    # Create the line plot
    fig = go.Figure()
    
    colors = {
        'neural_clbf': 'blue',
        'lqr': 'green',
        'mpc': 'red'
    }
    
    names = {
        'neural_clbf': 'Neural CLBF',
        'lqr': 'LQR',
        'mpc': 'MPC'
    }
    
    for controller in controllers:
        # Add line plot connecting actual data points
        fig.add_trace(go.Scatter(
            x=obstacle_counts,
            y=success_rates[controller],
            mode='lines+markers',  # Changed to lines+markers to connect dots
            name=names.get(controller, controller),
            line=dict(
                color=colors.get(controller, 'gray'),
                width=2
            ),
            marker=dict(
                color=colors.get(controller, 'gray'),
                size=10
            ),
            showlegend=True
        ))
        
        # Add trend line
        coefficients = np.polyfit(obstacle_counts, success_rates[controller], 3)
        poly = np.poly1d(coefficients)
        x_trend = np.linspace(min(obstacle_counts), max(obstacle_counts), 100)
        y_trend = poly(x_trend)
        
        fig.add_trace(go.Scatter(
            x=x_trend,
            y=y_trend,
            mode='lines',
            name=f'{names.get(controller, controller)} Trend',
            line=dict(
                color=colors.get(controller, 'gray'),
                width=2,
                dash='dash'
            ),
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title='Controller Collision Rates vs Number of Obstacles',
        xaxis_title='Number of Obstacles',
        yaxis_title='Collision Rate',
        yaxis_tickformat='.0%',
        xaxis=dict(
            tickmode='linear',
            tick0=min(obstacle_counts),
            dtick=1
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        hovermode='x unified'
    )
    
    # Save the plot
    output_dir = Path(metrics_file).parent
    fig.write_html(str(output_dir / 'collision_rates.html'))
    fig.show()


def main():
    # base_checkpoint = r"C:\Users\Tetra\Documents\Repostories\OSCorp\Project-Libra\checkpoints\six_dof_vehicle\epoch=0-val_loss=3.89.ckpt"
    # fine_tuned_checkpoint = r"C:\Users\Tetra\Documents\Repostories\OSCorp\Project-Libra\checkpoints\mujoco_six_dof_vehicle\epoch=0-val_loss=4.91.ckpt"
    base_checkpoint = r"C:\Users\brend\Documents\Project-Libra\neural_clbf\checkpoints\six_dof_vehicle\epoch=0-val_loss=6.38.ckpt"
    fine_tuned_checkpoint = r"C:\Users\brend\Documents\Project-Libra\neural_clbf\checkpoints\mujoco_six_dof_vehicle\epoch=0-val_loss=5.10.ckpt"
    plot_systematic_eval = False
    robustness_run = False
    systematic_eval = False

    # if plot_systematic_eval or robustness_test or systematic_eval:
    robustness_test = RobustnessTest(
        num_runs=1,
        base_checkpoint=base_checkpoint,
        fine_tuned_checkpoint=fine_tuned_checkpoint,
        output_dir="robustness_test_results",
        num_obstacles=(4, 8),
        record_videos=True,
        test_mode=TestMode.ALL_CONTROLLERS,
    )

    if robustness_run:
        aggregate_metrics = robustness_test.run()

        if aggregate_metrics:
            print("\nAggregate Metrics for All Controllers:")
            for controller, metrics in aggregate_metrics.items():
                print(f"\n{controller.upper()} Results:")
                print(json.dumps(metrics, indent=2))
    
    if systematic_eval:
        try:
            base_checkpoint = r"C:\Users\brend\Documents\Project-Libra\neural_clbf\checkpoints\six_dof_vehicle\epoch=0-val_loss=6.38.ckpt"
            fine_tuned_checkpoint = r"C:\Users\brend\Documents\Project-Libra\neural_clbf\checkpoints\mujoco_six_dof_vehicle\epoch=0-val_loss=5.10.ckpt"
            
            starting_points = [
                (15.0, 0.0, 0.0),
                (-15.0, 0.0, 0.0),
                (0.0, 15.0, 0.0),
                (0.0, -15.0, 0.0)
            ]
            
            robustness_test = None
            try:
                robustness_test = RobustnessTest(
                    num_runs=1,
                    base_checkpoint=base_checkpoint,
                    fine_tuned_checkpoint=fine_tuned_checkpoint,
                    output_dir="systematic_evaluation_results",
                    num_obstacles=6,
                    record_videos=True,
                    test_mode=TestMode.ALL_CONTROLLERS
                )
                
                # Run a smaller test first
                results = robustness_test.systematic_evaluation(
                    n_runs=4, 
                    max_obstacles=3,
                    starting_points=starting_points
                )
                results = robustness_test.run()
                
                if results:
                    print("\nResults Summary:")
                    for controller, obstacle_results in results.items():
                        print(f"\n{controller.upper()}:")
                        for obstacle_count, metrics in obstacle_results.items():
                            print(f"\n{obstacle_count}:")
                            print(f"Success Rate: {metrics['success_rate']:.2%}")
                            print(f"Collision Rate: {metrics['collision_rate']:.2%}")
                            print(f"Average Efficiency: {metrics['average_efficiency']:.2f}")
                            print(f"Average Fuel Cost: {metrics['average_fuel_cost']:.2f}")
                            print(f"Average Safety Score: {metrics['average_safety_score']:.2f}")
                else:
                    print("No results were generated during the evaluation")
                    
            finally:
                if robustness_test:
                    robustness_test.cleanup()
                    
        except Exception as e:
            logging.error(f"Error in main: {str(e)}")
            raise
    
    if plot_systematic_eval:
        # Plot existing results
        try:
            # Use absolute path
            project_root = Path(__file__).parent.parent.parent.parent.parent
            metrics_file = project_root / "systematic_evaluation_results/20241112_134415/systematic_evaluation/metrics/aggregate_metrics.json"
            
            if not metrics_file.exists():
                print(f"Error: Metrics file not found at {metrics_file}")
                print("\nAvailable results directories:")
                results_dir = project_root / "systematic_evaluation_results"
                if results_dir.exists():
                    for d in results_dir.iterdir():
                        if d.is_dir():
                            print(f"- {d.name}")
                return
                
            print(f"Plotting results from: {metrics_file}")
            plot_success_rates(str(metrics_file))
            plot_collision_rates(str(metrics_file))
            
        except Exception as e:
            print(f"Error plotting results: {str(e)}")
            print("Please check the path to your metrics file")

    



if __name__ == "__main__":
    main()
