import numpy as np
import torch
from typing import List, Tuple, Optional, Any
from scipy.spatial.transform import Rotation as R

from libra.Other_controller.base_controller import BaseController
from libra.rqbf.system.rrt import rrt_plan
from libra.Other_controller.LQR.LQR_controller import create_astrobee_lqr, LQRController

class Obstacle:
    """Simple obstacle class that matches the interface expected by LQR controller"""
    def __init__(self, position: np.ndarray, radius: float):
        self.position = position
        self.radius = radius
        
    def distance(self, pos: np.ndarray) -> float:
        """Calculate distance from a point to this obstacle's surface"""
        return np.linalg.norm(pos - self.position) - self.radius

class LQRControllerWrapper(BaseController):
    def __init__(self, use_rrt: bool = False):
        """
        Initialize LQR controller wrapper.
        Args:
            use_rrt (bool): Whether to use RRT for path planning
        """
        self.controller: Optional[LQRController] = None
        self.use_rrt = use_rrt
        self.rrt_path = None
        self.path_index = 0
        self.path_threshold = 0.5  # Distance threshold to move to next waypoint
        self.rrt_replan_threshold = 2.0  # Distance threshold for replanning RRT
        self.last_rrt_update = 0
        self.rrt_update_frequency = 25  # Update RRT every N steps
        # Define workspace bounds
        self.bounds = (
            torch.tensor([-20.0, -20.0, -5.0]),
            torch.tensor([20.0, 20.0, 5.0])
        )

    def setup(self) -> None:
        """Initialize the LQR controller"""
        self.controller = create_astrobee_lqr()

    def get_control(self, state: np.ndarray, goal: np.ndarray, obstacles: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """
        Get control input using LQR with optional RRT path following.
        
        Args:
            state (np.ndarray): Current state [pos, quat, vel, ang_vel]
            goal (np.ndarray): Goal position
            obstacles (List[Tuple[np.ndarray, float]]): List of (position, radius) tuples for obstacles
            
        Returns:
            np.ndarray: Control input [force_x, force_y, force_z, torque_x, torque_y, torque_z]
        """
        if self.controller is None:
            raise RuntimeError("Controller not initialized. Call setup() first.")

        # Convert quaternion to euler angles for LQR
        quat = state[3:7]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Convert to scipy quaternion format
        euler = r.as_euler('xyz')

        # Construct LQR state (12-dimensional: [pos, euler, vel, ang_vel])
        lqr_state = np.concatenate([
            state[0:3],    # position
            euler,         # euler angles
            state[7:10],   # linear velocity
            state[10:13]   # angular velocity
        ])

        # Convert obstacles to Obstacle objects
        obstacle_objects = [Obstacle(obs[0], obs[1]) for obs in obstacles]

        if self.use_rrt:
            current_goal = self._handle_rrt_path_following(state[:3], goal, obstacles)
        else:
            current_goal = goal

        # Append zeros for orientation and velocity targets
        full_goal = np.concatenate([current_goal, np.zeros(9)])
        
        # Get LQR control using obstacle objects
        control = self.controller.get_control(lqr_state, obstacle_objects, full_goal)
        
        return control

    def _handle_rrt_path_following(self, current_pos: np.ndarray, goal: np.ndarray, obstacles: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """Handle RRT path following logic"""
        # Initialize path if none exists
        if self.rrt_path is None:
            self.rrt_path = rrt_plan(
                torch.tensor(current_pos, dtype=torch.float32),
                torch.tensor(goal, dtype=torch.float32),
                obstacles,
                self.bounds
            )
            self.path_index = 0
            return self._get_current_waypoint(current_pos)

        # Check if we need to replan
        distance_to_goal = np.linalg.norm(current_pos - goal)
        if distance_to_goal > self.rrt_replan_threshold:
            self.last_rrt_update += 1
            if self.last_rrt_update >= self.rrt_update_frequency:
                new_path = rrt_plan(
                    torch.tensor(current_pos, dtype=torch.float32),
                    torch.tensor(goal, dtype=torch.float32),
                    obstacles,
                    self.bounds
                )
                if new_path is not None:
                    self.rrt_path = new_path
                    self.path_index = 0
                    self.last_rrt_update = 0

        return self._get_current_waypoint(current_pos)

    def _get_current_waypoint(self, current_pos: np.ndarray) -> np.ndarray:
        """
        Get current waypoint from RRT path.
        
        Args:
            current_pos (np.ndarray): Current position
            
        Returns:
            np.ndarray: Current waypoint to follow
        """
        if self.rrt_path is None or self.path_index >= len(self.rrt_path):
            return current_pos
        
        current_waypoint = self.rrt_path[self.path_index]
        distance = np.linalg.norm(current_pos - current_waypoint)
        
        # Move to next waypoint if close enough
        if distance < self.path_threshold and self.path_index < len(self.rrt_path) - 1:
            self.path_index += 1
            current_waypoint = self.rrt_path[self.path_index]
        
        return current_waypoint

    @property
    def name(self) -> str:
        return f"LQR{'_RRT' if self.use_rrt else ''}"

    def reset(self) -> None:
        """Reset the controller's internal state"""
        self.rrt_path = None
        self.path_index = 0
        self.last_rrt_update = 0