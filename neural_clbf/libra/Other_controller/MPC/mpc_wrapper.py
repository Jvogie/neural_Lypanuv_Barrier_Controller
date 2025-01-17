import numpy as np
import torch
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass

from libra.Other_controller.base_controller import BaseController
from libra.rqbf.system.rrt import rrt_plan
from libra.Other_controller.MPC.mpc_controller import MPCController
from libra.Other_controller.LQR.LQR_wrapper import Obstacle

class MPCControllerWrapper(BaseController):
    def __init__(self, prediction_horizon: int = 10, use_rrt: bool = True):
        """
        Initialize MPC controller wrapper.
        Args:
            prediction_horizon (int): Number of steps to predict ahead
            use_rrt (bool): Whether to use RRT for path planning
        """
        self.controller: Optional[MPCController] = None
        self.prediction_horizon = prediction_horizon
        self.use_rrt = use_rrt
        self.current_path = None
        self.path_index = 0
        self.path_threshold = 0.5
        self.rrt_replan_threshold = 2.0
        self.last_rrt_update = 0
        self.rrt_update_frequency = 25

    def setup(self) -> None:
        """Initialize the MPC controller"""
        # Define system matrices
        A = np.eye(12)
        dt = 0.01
        
        # Position to velocity coupling
        A[0:3, 6:9] = np.eye(3) * dt
        # Orientation to angular velocity coupling
        A[3:6, 9:12] = np.eye(3) * dt
        
        # Input matrix
        B = np.zeros((12, 6))
        B[6:9, 0:3] = np.eye(3) * dt  # Forces affect velocities
        B[9:12, 3:6] = np.eye(3) * dt  # Torques affect angular velocities
        
        # Cost matrices
        Q = np.eye(12)  # State cost
        Q[0:3, 0:3] *= 10.0  # Position error cost
        Q[3:6, 3:6] *= 5.0   # Orientation error cost
        Q[6:9, 6:9] *= 1.0   # Linear velocity cost
        Q[9:12, 9:12] *= 1.0 # Angular velocity cost
        
        R = np.eye(6) * 0.1  # Control cost
        
        self.controller = MPCController(A, B, Q, R, N=self.prediction_horizon)

    def get_control(self, state: np.ndarray, goal: np.ndarray, obstacles: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """
        Get control input using MPC with optional RRT path following.
        
        Args:
            state (np.ndarray): Current state [pos, quat, vel, ang_vel]
            goal (np.ndarray): Goal position
            obstacles (List[Tuple[np.ndarray, float]]): List of (position, radius) tuples for obstacles
            
        Returns:
            np.ndarray: Control input [force_x, force_y, force_z, torque_x, torque_y, torque_z]
        """
        if self.controller is None:
            raise RuntimeError("Controller not initialized. Call setup() first.")

        # Convert quaternion to euler angles for MPC
        quat = state[3:7]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        euler = r.as_euler('xyz')

        # Construct MPC state
        mpc_state = np.concatenate([
            state[0:3],    # position
            euler,         # euler angles
            state[7:10],   # linear velocity
            state[10:13]   # angular velocity
        ])

        if self.use_rrt:
            current_goal = self._handle_rrt_path_following(state[:3], goal, obstacles)
        else:
            current_goal = goal

        # Create full goal state including orientation and velocities
        full_goal = np.concatenate([current_goal, np.zeros(9)])
        
        # Convert obstacles to Obstacle objects (same as LQR)
        obstacle_objects = [Obstacle(obs[0], obs[1]) for obs in obstacles]
        
        # Get MPC control
        control = self.controller.compute_mpc_control(mpc_state, full_goal, obstacle_objects)
        
        return control

    def _handle_rrt_path_following(self, current_pos: np.ndarray, goal: np.ndarray, obstacles: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """Handle RRT path following logic"""
        # Initialize path if none exists
        if self.current_path is None:
            self.current_path = self._plan_rrt_path(current_pos, goal, obstacles)
            self.path_index = 0
            return self._get_current_waypoint(current_pos)

        # Check if we need to replan
        distance_to_goal = np.linalg.norm(current_pos - goal)
        if distance_to_goal > self.rrt_replan_threshold:
            self.last_rrt_update += 1
            if self.last_rrt_update >= self.rrt_update_frequency:
                new_path = self._plan_rrt_path(current_pos, goal, obstacles)
                if new_path is not None:
                    self.current_path = new_path
                    self.path_index = 0
                    self.last_rrt_update = 0

        return self._get_current_waypoint(current_pos)

    def _plan_rrt_path(self, start: np.ndarray, goal: np.ndarray, obstacles: List[Tuple[np.ndarray, float]]) -> Optional[np.ndarray]:
        """
        Plan path using RRT.
        
        Args:
            start (np.ndarray): Start position
            goal (np.ndarray): Goal position
            obstacles (List[Tuple[np.ndarray, float]]): List of obstacles
            
        Returns:
            Optional[np.ndarray]: Planned path if successful, None otherwise
        """
        # Convert to torch tensors for RRT
        start_tensor = torch.tensor(start, dtype=torch.float32)
        goal_tensor = torch.tensor(goal, dtype=torch.float32)
        obstacles_tensor = [(torch.tensor(obs, dtype=torch.float32), rad) for obs, rad in obstacles]
        
        # Define bounds for RRT
        bounds = (
            torch.tensor([-20.0, -20.0, -5.0]),
            torch.tensor([20.0, 20.0, 5.0])
        )
        
        try:
            path = rrt_plan(start_tensor, goal_tensor, obstacles_tensor, bounds)
            return path.numpy() if path is not None else None
        except Exception as e:
            print(f"RRT planning failed: {str(e)}")
            return None

    def _get_current_waypoint(self, current_pos: np.ndarray) -> np.ndarray:
        """
        Get current waypoint from RRT path.
        
        Args:
            current_pos (np.ndarray): Current position
            
        Returns:
            np.ndarray: Current waypoint to follow
        """
        if self.current_path is None or self.path_index >= len(self.current_path):
            return current_pos
        
        current_waypoint = self.current_path[self.path_index]
        distance = np.linalg.norm(current_pos - current_waypoint)
        
        # Move to next waypoint if close enough
        if distance < self.path_threshold and self.path_index < len(self.current_path) - 1:
            self.path_index += 1
            current_waypoint = self.current_path[self.path_index]
        
        return current_waypoint

    @property
    def name(self) -> str:
        return f"MPC{'_RRT' if self.use_rrt else ''}"

    def reset(self) -> None:
        """Reset the controller's internal state"""
        self.current_path = None
        self.path_index = 0
        self.last_rrt_update = 0