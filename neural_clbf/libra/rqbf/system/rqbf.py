## Rattle Quaternion Barrier Function System (RQBF)
## rqbf.py

import torch
import numpy as np
from typing import Tuple, Optional, List

from neural_clbf.systems.control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList, lqr, continuous_lyap, robust_continuous_lyap
from torch.autograd.functional import jacobian
from libra.rqbf.system.rrt import rrt_plan


class SixDOFVehicle(ControlAffineSystem):
    """
    Represents a 6-DoF vehicle with position, orientation, velocities, and biases.

    State variables (19 dimensions):
    - r: position (3D)
    - q: quaternion orientation (4D)
    - v: linear velocity (3D)
    - omega: angular velocity (3D)
    - b_a: accelerometer bias (3D)
    - b_g: gyroscope bias (3D)

    Control inputs (6 dimensions):
    - F: force (3D) 
    - tau: torque (3D)
    """
    ## State indices
    R = slice(0, 3)        ## position indices
    Q = slice(3, 7)        ## quaternion indices
    V = slice(7, 10)       ## linear velocity indices
    OMEGA = slice(10, 13)  ## angular velocity indices
    B_A = slice(13, 16)    ## accelerometer bias indices
    B_G = slice(16, 19)    ## gyroscope bias indices

    ## Control indices
    F = slice(0, 3)        ## force indices
    TAU = slice(3, 6)      ## torque indices

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
        obstacles: Optional[List[Tuple[torch.Tensor, float]]] = None,
        obstacle_buffer: float = 2.5,
    ):
        """
        Initialize the 6-DoF vehicle.

        Args:
            nominal_params: A dictionary giving the parameter values for the system.
                            Requires keys ["mass", "inertia_matrix"]
            dt: The timestep to use for the simulation
            controller_dt: The timestep for the LQR discretization. Defaults to dt
            scenarios: Optional list of scenarios for robust control
            obstacles: Optional list of tuples containing obstacle positions and radii
            obstacle_buffer: Additional buffer distance to maintain from obstacles
        """

        self.mass = torch.tensor(nominal_params["mass"], dtype=torch.float32)
        self.I = torch.tensor(nominal_params["inertia_matrix"], dtype=torch.float32)
        self.I_inv = torch.inverse(self.I)

        ## Gravity vector
        self.g = nominal_params.get("gravity", torch.tensor([0.0, 0.0, -9.81]))
        
        ## This will get overridden later
        self.goal_position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        self.scenarios = scenarios

        self.obstacles = obstacles if obstacles is not None else []  
        self.obstacle_buffer = obstacle_buffer

        self.rrt_path = None

        ## Call this after adding custom vars
        super().__init__(
            nominal_params, dt=dt, controller_dt=controller_dt, scenarios=scenarios
        )

    def validate_params(self, params: Scenario) -> bool:
        """
        Validate the given parameters for the 6-DoF vehicle.

        Args:
            params: A dictionary of parameters to validate

        Returns:
            bool: True if parameters are valid, False otherwise
        """
        required_keys = ["mass", "inertia_matrix"]
        for key in required_keys:
            if key not in params:
                return False
        if params["mass"] <= 0:
            return False
        inertia_matrix = torch.tensor(params["inertia_matrix"], dtype=torch.float32)
        if inertia_matrix.shape != (3, 3):
            return False
        ## Check that inertia matrix is symmetric
        if not torch.allclose(inertia_matrix, inertia_matrix.T):
            return False

        ## Check that inertia matrix is positive definite
        eigenvalues = torch.linalg.eigvalsh(inertia_matrix)
        
        if not torch.all(eigenvalues > 0):
            return False  ## Inertia matrix should be positive definite
        return True

    @property
    def n_dims(self) -> int:
        return 19

    @property
    def n_controls(self) -> int:
        return 6

    @property
    def angle_dims(self) -> List[int]:
        ## Indices corresponding to quaternion components
        return [3, 4, 5, 6]

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the upper and lower limits of the state space.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Upper and lower limits
        """
        upper_limit = torch.ones(self.n_dims)
        lower_limit = -torch.ones(self.n_dims)

        ## Position limits (e.g., within a 100m cube)
        upper_limit[self.R] = 100.0
        lower_limit[self.R] = -100.0

        ## Quaternion components are between -1 and 1
        upper_limit[self.Q] = 1.0
        lower_limit[self.Q] = -1.0

        ## Velocity limits (e.g., -50 to 50 m/s)
        upper_limit[self.V] = 50.0
        lower_limit[self.V] = -50.0

        ## Angular velocity limits (e.g., -10 to 10 rad/s)
        upper_limit[self.OMEGA] = 10.0
        lower_limit[self.OMEGA] = -10.0

        ## Bias limits (e.g., small biases)
        upper_limit[self.B_A] = 0.1
        lower_limit[self.B_A] = -0.1

        upper_limit[self.B_G] = 0.1
        lower_limit[self.B_G] = -0.1

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the upper and lower limits of the control inputs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Upper and lower control limits
        """
        ## Define the maximum force and torque the actuators can produce
        upper_limit = torch.tensor([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
        lower_limit = -upper_limit
        return (upper_limit, lower_limit)



    # This function is irrelevant as collision is a fail state
    ## Was responsible for appearing to work, but was inherently flawed
    # def handle_collision(self, position: torch.Tensor, velocity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Handle collisions with obstacles.
    #     
    #     Args:
    #         position (torch.Tensor): Current position (batch_size, 3)
    #         velocity (torch.Tensor): Current velocity (batch_size, 3)
    #     
    #     Returns:
    #         Tuple[torch.Tensor, torch.Tensor]: Adjusted position and velocity
    #     """
    #     for obstacle_pos, obstacle_radius in self.obstacles:
    #         # Include buffer in the collision radius
    #         effective_radius = obstacle_radius + self.obstacle_buffer
    #         to_obstacle = position - obstacle_pos.to(position.device)
    #         distance = torch.norm(to_obstacle, dim=1, keepdim=True)
    #         
    #         # Check for collision
    #         collision_mask = (distance <= effective_radius).squeeze()
    #         
    #         if collision_mask.any():
    #             # Move the position to the surface of the obstacle
    #             collision_normal = to_obstacle[collision_mask] / distance[collision_mask]
    #             position[collision_mask] = obstacle_pos.to(position.device) + collision_normal * effective_radius
    #             
    #             # Reflect the velocity off the obstacle surface
    #             dot_product = torch.sum(velocity[collision_mask] * collision_normal, dim=1, keepdim=True)
    #             velocity[collision_mask] = velocity[collision_mask] - 1.2 * dot_product * collision_normal
    #             
    #             # Add some damping to the reflected velocity
    #             damping_factor = 0.6
    #             velocity[collision_mask] *= damping_factor
    #     
    #     return position, velocity
    

    # def closed_loop_dynamics(self, x: torch.Tensor, u: torch.Tensor, params: Optional[Scenario] = None) -> torch.Tensor:
    #     """
    #     Compute the closed-loop dynamics of the system, preventing obstacle penetration.
    # 
    #     Args:
    #         x (torch.Tensor): Current state
    #         u (torch.Tensor): Control input
    #         params (Optional[Scenario]): Scenario parameters (unused in this implementation)
    # 
    #     Returns:
    #         torch.Tensor: State derivative
    #     """
    #     # Get the original dynamics
    #     xdot = super().closed_loop_dynamics(x, u, params)
    #     
    #     # Extract position and velocity
    #     position = x[:, self.R]
    #     velocity = x[:, self.V]
    #     
    #     # Predict the next position and velocity
    #     next_position = position + self.dt * xdot[:, self.R]
    #     next_velocity = velocity + self.dt * xdot[:, self.V]
    #     
    #     # Handle collisions
    #     adjusted_position, adjusted_velocity = self.handle_collision(next_position, next_velocity)
    #     
    #     # Update xdot with the adjusted values
    #     xdot[:, self.R] = (adjusted_position - position) / self.dt
    #     xdot[:, self.V] = (adjusted_velocity - velocity) / self.dt
    #     
    #     return xdot

    def compute_A_matrix(self, scenario: Optional[Scenario] = None) -> np.ndarray:
        """
        Compute the linearized continuous-time state-space dynamics matrix A.
        """
        if scenario is None:
            scenario = self.nominal_params

        x0 = self.goal_point
        u0 = self.u_eq

        def dynamics(x):
            return self.closed_loop_dynamics(x, u0, scenario).squeeze()

        with torch.no_grad():
            A = jacobian(dynamics, x0).squeeze().cpu().numpy() ## type: ignore

        return A
    
    def safe_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Determine which states are considered safe.

        Args:
            x: Batch of state vectors

        Returns:
            torch.Tensor: Boolean mask indicating safe states
        """
        position = x[:, self.R]
        
        operational_range = 100.0
        within_range = (position.norm(dim=1) <= operational_range)
        
        ## Check if not too close to any obstacle
        safe_from_obstacles = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        for obstacle_pos, obstacle_radius in self.obstacles:
            distance_to_obstacle = torch.norm(position - obstacle_pos.to(position.device), dim=1)
            safe_distance = obstacle_radius + self.obstacle_buffer
            safe_from_obstacles = torch.logical_and(safe_from_obstacles, distance_to_obstacle > safe_distance)
        
        return torch.logical_and(within_range, safe_from_obstacles)

    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Determine which states are considered unsafe.

        Args:
            x: Batch of state vectors

        Returns:
            torch.Tensor: Boolean mask indicating unsafe states
        """
        position = x[:, self.R]
        
        ## Check if outside operational range (this was mostly arbitrary)
        operational_range = 100.0
        outside_range = (position.norm(dim=1) > operational_range)
        
        ## Check if too close to any obstacle
        too_close_to_obstacle = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        for obstacle_pos, obstacle_radius in self.obstacles:
            distance_to_obstacle = torch.norm(position - obstacle_pos.to(position.device), dim=1)
            unsafe_distance = obstacle_radius + self.obstacle_buffer
            too_close_to_obstacle = torch.logical_or(too_close_to_obstacle, distance_to_obstacle <= unsafe_distance)
        
        return torch.logical_or(outside_range, too_close_to_obstacle)

    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Determine which states are considered to have reached the goal.

        Args:
            x: Batch of state vectors

        Returns:
            torch.Tensor: Boolean mask indicating goal states
        """
        ## Goal is to reach the goal position with zero velocity
        position = x[:, self.R]
        velocity = x[:, self.V]
        position_tolerance = 1.0
        velocity_tolerance = 0.5
        in_position = (position - self.goal_position.type_as(x)).norm(dim=1) <= position_tolerance
        in_velocity = (velocity.norm(dim=1) <= velocity_tolerance)
        return torch.logical_and(in_position, in_velocity)

    @property
    def goal_point(self) -> torch.Tensor:
        """
        Get the goal point in the state space.

        Returns:
            torch.Tensor: Goal point
        """
        goal = torch.zeros((1, self.n_dims), dtype=torch.float32)
        # Set position components to the goal position
        goal[0, self.R] = self.goal_position
        # Set quaternion to [1, 0, 0, 0] for no rotation
        goal[0, self.Q] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        return goal

    @property
    def u_eq(self) -> torch.Tensor:
        """
        Get the equilibrium control input.

        Returns:
            torch.Tensor: Equilibrium control input
        """
        ## For example, to hover and compensate gravity
        m = self.mass
        g = self.g
        F_eq = -m * g  ## Force to cancel gravity
        ## Assuming torque is zero at equilibrium
        tau_eq = torch.zeros(3)
        u_eq = torch.cat((F_eq, tau_eq)).unsqueeze(0)
        return u_eq

    def _f(self, x: torch.Tensor, params: Scenario):
        """
        Compute the uncontrolled dynamics of the system.

        Args:
            x: Batch of state vectors
            params: Scenario parameters

        Returns:
            torch.Tensor: Uncontrolled dynamics
        """
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1)).type_as(x)

        ## Extract state variables
        r = x[:, self.R]           ## Position
        q = x[:, self.Q]           ## Quaternion
        v = x[:, self.V]           ## Linear velocity
        omega = x[:, self.OMEGA]   ## Angular velocity
        b_a = x[:, self.B_A]       ## Accelerometer bias
        b_g = x[:, self.B_G]       ## Gyroscope bias

        ## System parameters
        m = torch.tensor(params["mass"], dtype=torch.float32)
        I = torch.tensor(params["inertia_matrix"], dtype=torch.float32)
        I_inv = torch.inverse(I)
        g_vec = params.get("gravity", torch.tensor([0.0, 0.0, -9.81]).type_as(x))

        ## 1. Position derivative
        f[:, self.R, 0] = v

        ## 2. Quaternion derivative
        ## Compute quaternion multiplication with omega
        omega_quat = torch.cat((torch.zeros((batch_size, 1)).type_as(x), omega), dim=1)
        q_dot = 0.5 * self.quaternion_multiply(q, omega_quat)
        f[:, self.Q, 0] = q_dot

        ## 3. Linear velocity derivative
        f[:, self.V, 0] = g_vec - b_a

        ## 4. Angular velocity derivative
        omega_cross_Iomega = torch.cross(omega, (I @ omega.unsqueeze(-1)).squeeze(-1), dim=1)
        f[:, self.OMEGA, 0] = - (I_inv @ (omega_cross_Iomega + b_g).unsqueeze(-1)).squeeze(-1)

        ## 5. Bias derivatives (assumed constant)
        ## Accelerometer bias derivatives
        f[:, self.B_A, 0] = 0.0
        ## Gyroscope bias derivatives
        f[:, self.B_G, 0] = 0.0

        return f

    def _g(self, x: torch.Tensor, params: Scenario):
        """
        Compute the control-dependent dynamics of the system.

        Args:
            x: Batch of state vectors
            params: Scenario parameters

        Returns:
            torch.Tensor: Control-dependent dynamics
        """
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls)).type_as(x)

        ## Extract state variables
        q = x[:, self.Q]  ## Quaternion

        ## System parameters
        m = torch.tensor(params["mass"], dtype=torch.float32)
        I = torch.tensor(params["inertia_matrix"], dtype=torch.float32)
        I_inv = torch.inverse(I)

        ## 1. Linear velocity control influence
        R = self.quaternion_to_rotation_matrix(q)  ## Convert quaternion to rotation matrix
        ## Force influence
        g[:, self.V, 0:3] = (1.0 / m) * R

        ## 2. Angular velocity control influence
        ## Torque influence
        g[:, self.OMEGA, 3:6] = I_inv.unsqueeze(0)

        ## The rest of g is zero
        return g

    def quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Perform quaternion multiplication q1 * q2.

        Args:
            q1: First quaternion (batch_size, 4)
            q2: Second quaternion (batch_size, 4)

        Returns:
            torch.Tensor: Result of quaternion multiplication (batch_size, 4)
        """
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return torch.stack((w, x, y, z), dim=1)

    def quaternion_to_rotation_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion to rotation matrix.

        Args:
            q: Quaternion (batch_size, 4)

        Returns:
            torch.Tensor: Rotation matrix (batch_size, 3, 3)
        """
        batch_size = q.shape[0]
        q_norm = q.norm(dim=1, keepdim=True)
        ## Avoid division by zero
        q_norm = torch.where(q_norm == 0, torch.ones_like(q_norm), q_norm)
        q = q / q_norm  ## Normalize quaternion
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        ## Compute rotation matrix components
        R = torch.zeros((batch_size, 3, 3)).type_as(q)
        R[:, 0, 0] = 1 - 2*(y**2 + z**2)
        R[:, 0, 1] = 2*(x*y - z*w)
        R[:, 0, 2] = 2*(x*z + y*w)
        R[:, 1, 0] = 2*(x*y + z*w)
        R[:, 1, 1] = 1 - 2*(x**2 + z**2)
        R[:, 1, 2] = 2*(y*z - x*w)
        R[:, 2, 0] = 2*(x*z - y*w)
        R[:, 2, 1] = 2*(y*z + x*w)
        R[:, 2, 2] = 1 - 2*(x**2 + y**2)
        return R

    def compute_linearized_controller(self, scenarios: Optional[ScenarioList] = None):
        """
        Computes the linearized controller K and Lyapunov matrix P with respect to the current goal.
        """
        if scenarios is None:
            scenarios = [self.nominal_params]

        u0 = self.u_eq
        x0 = self.goal_point.type_as(u0)
        n_states = self.n_dims
        n_controls = self.n_controls

        ## Define cost matrices as identity
        Q_np = np.eye(n_states)
        R_np = np.eye(n_controls)

        ## Adjust weights for bias states
        for idx in range(13, 19):  ## Bias states
            Q_np[idx, idx] = 1e-4  ## Small weight

        Acl_list = []

        for s in scenarios:
            ## Compute the linearized continuous-time dynamics matrices at the goal
            Act, Bct = self.linearized_ct_dynamics_matrices(s)

            Act_np = Act  ## Already a NumPy array
            Bct_np = Bct  ## Already a NumPy array

            ## Define controllable indices (excluding bias states)
            controllable_indices = list(range(3)) + list(range(7, 13))  # Indices 0-2, 7-12

            A_controllable = Act_np[np.ix_(controllable_indices, controllable_indices)]
            B_controllable = Bct_np[controllable_indices, :]

            Q_controllable = Q_np[np.ix_(controllable_indices, controllable_indices)]

            ## Get feedback matrix using LQR
            K_controllable = lqr(A_controllable, B_controllable, Q_controllable, R_np)

            ## Expand K to full size by adding zeros for excluded states
            K_full = np.zeros((n_controls, n_states))
            K_full[:, controllable_indices] = K_controllable

            self.K = torch.tensor(K_full).type_as(u0)

            Acl_np = Act_np - Bct_np @ K_full
            Acl_list.append(Acl_np)

        ## Compute Lyapunov matrix
        if len(scenarios) > 1:
            P_np = robust_continuous_lyap(Acl_list, Q_np)
        else:
            P_np = continuous_lyap(Acl_list[0], Q_np)

        self.P = torch.tensor(P_np).type_as(u0)


    def is_controllable(self, A: np.ndarray, B: np.ndarray) -> bool:
        """
        Checks if the system is controllable.

        Args:
            A: System matrix
            B: Input matrix

        Returns:
            bool: True if the system is controllable, False otherwise
        """
        n = A.shape[0]
        controllability_matrix = B
        for i in range(1, n):
            controllability_matrix = np.hstack((controllability_matrix, np.linalg.matrix_power(A, i) @ B))
        rank = np.linalg.matrix_rank(controllability_matrix)
        print(f"Controllability matrix rank: {rank} (expected: {n})")
        return rank == n
    
    def set_goal(self, goal_position: torch.Tensor):
        """
        Set a new goal position for the vehicle.

        Args:
            goal_position (torch.Tensor): The desired goal position.
        """

        self.goal_position = goal_position

        ## This used to be used, but it frankly will fall apart as it'll start saying things are unfeasible no matter what
        ## This could be a root issue, but we'll discuss this more in Documentation
        ##self.compute_linearized_controller(self.scenarios)

    def plan_rrt_path(self, start: torch.Tensor):
        """
        Plan a path using RRT from the current position to the goal.
        """
        goal = self.goal_position
        obstacles = self.obstacles
        bounds = self.state_limits

        # Only pass the position components to the RRT planner
        start_position = start[:3]
        self.rrt_path = rrt_plan(start_position, goal, obstacles, bounds)
        return self.rrt_path

    def u_nominal(self, x: torch.Tensor, params: Optional[Scenario] = None) -> torch.Tensor:
        position = x[:, self.R]
        velocity = x[:, self.V]
        quaternion = x[:, self.Q]
        angular_velocity = x[:, self.OMEGA]

        distance_to_goal = torch.norm(self.goal_position - position)
        direct_path_threshold = 5.0
        max_speed = 3.0
        

        speed_gain = torch.ones_like(distance_to_goal)
        
        if distance_to_goal < direct_path_threshold and self.is_path_clear(position, self.goal_position):
            position_error = self.goal_position - position
            velocity_error = -velocity

            kp_direct = torch.tensor([15.0, 30.0, 60.0], dtype=x.dtype, device=x.device)
            kd_direct = torch.tensor([10.0, 20.0, 40.0], dtype=x.dtype, device=x.device)

            desired_acceleration = kp_direct * position_error + kd_direct * velocity_error
            desired_force = self.mass * desired_acceleration

            desired_up = torch.tensor([0.0, 0.0, 1.0], dtype=x.dtype, device=x.device).expand_as(desired_acceleration)
            desired_forward = desired_acceleration / (torch.norm(desired_acceleration, dim=1, keepdim=True) + 1e-6)
            desired_right = torch.cross(desired_forward, desired_up, dim=1)
            desired_up = torch.cross(desired_right, desired_forward, dim=1)

            desired_rotation_matrix = torch.stack([desired_right, desired_forward, desired_up], dim=2)
            current_rotation_matrix = self.quaternion_to_rotation_matrix(quaternion)

            orientation_error = torch.matmul(desired_rotation_matrix.transpose(1, 2), current_rotation_matrix) - torch.eye(3, dtype=x.dtype, device=x.device).unsqueeze(0)
            orientation_error_vector = torch.stack([orientation_error[:, 2, 1], orientation_error[:, 0, 2], orientation_error[:, 1, 0]], dim=1)

            kp_orientation = torch.tensor([6.0, 6.0, 6.0], dtype=x.dtype, device=x.device)
            kd_orientation = torch.tensor([4.0, 4.0, 4.0], dtype=x.dtype, device=x.device)

            desired_torque = kp_orientation * orientation_error_vector - kd_orientation * angular_velocity

        else:
            if self.rrt_path is None or torch.norm(position[0] - self.rrt_path[0]) > 1.0:
                self.plan_rrt_path(position[0])

            if self.rrt_path is not None:
                distances = torch.norm(self.rrt_path - position[0].unsqueeze(0), dim=1)
                closest_idx = torch.argmin(distances)
                
                if closest_idx + 1 < len(self.rrt_path):
                    intermediate_goal = self.rrt_path[closest_idx + 1]
                else:
                    intermediate_goal = self.rrt_path[-1]
            else:
                intermediate_goal = self.goal_position

            position_error = intermediate_goal.to(x.device) - position
            velocity_error = -velocity

            distance_to_intermediate = torch.norm(position_error, dim=1, keepdim=True)
            unit_direction = position_error / (distance_to_intermediate + 1e-6)

            min_speed = 0.05
            speed_gain = torch.clamp((distance_to_intermediate / 10.0) ** 2, min_speed/max_speed, 1.0)

            kp = torch.tensor([9.0, 18.0, 36.0], dtype=x.dtype, device=x.device)
            kd = torch.tensor([7.0, 13.0, 26.0], dtype=x.dtype, device=x.device)

            sigmoid_scale = torch.tensor([0.3, 0.4, 0.5], dtype=x.dtype, device=x.device)
            position_gain = 2 / (1 + torch.exp(-sigmoid_scale * torch.abs(position_error))) - 1

            base_acceleration = (
                kp * position_gain * unit_direction * speed_gain +
                kd * velocity_error * speed_gain
            )

            close_distance_threshold = 2.0
            close_distance_gain = torch.tensor([3.0, 4.0, 5.0], dtype=x.dtype, device=x.device)
            close_distance_term = torch.where(
                distance_to_intermediate < close_distance_threshold,
                close_distance_gain * position_error,
                torch.zeros_like(position_error)
            )

            obstacle_force = torch.zeros_like(position)
            for obstacle_pos, obstacle_radius in self.obstacles:
                to_obstacle = position - obstacle_pos.to(position.device)
                distance_to_obstacle = torch.norm(to_obstacle, dim=1, keepdim=True)
                
                influence_distance = 6.0
                repulsive_gain = 50.0
                
                mask = (distance_to_obstacle < influence_distance).float()
                repulsive_force = mask * repulsive_gain * (1/distance_to_obstacle - 1/influence_distance) * (1/distance_to_obstacle**2) * (to_obstacle / distance_to_obstacle)
                
                obstacle_force += repulsive_force

            attractive_gain = 1.0
            attractive_force = attractive_gain * position_error

            total_force = attractive_force + obstacle_force

            desired_acceleration = total_force + base_acceleration + close_distance_term

            desired_force = self.mass * desired_acceleration

            desired_up = torch.tensor([0.0, 0.0, 1.0], dtype=x.dtype, device=x.device).expand_as(desired_acceleration)
            desired_forward = desired_acceleration / (torch.norm(desired_acceleration, dim=1, keepdim=True) + 1e-6)
            desired_right = torch.cross(desired_forward, desired_up, dim=1)
            desired_up = torch.cross(desired_right, desired_forward, dim=1)

            desired_rotation_matrix = torch.stack([desired_right, desired_forward, desired_up], dim=2)
            current_rotation_matrix = self.quaternion_to_rotation_matrix(quaternion)

            orientation_error = torch.matmul(desired_rotation_matrix.transpose(1, 2), current_rotation_matrix) - torch.eye(3, dtype=x.dtype, device=x.device).unsqueeze(0)
            orientation_error_vector = torch.stack([orientation_error[:, 2, 1], orientation_error[:, 0, 2], orientation_error[:, 1, 0]], dim=1)

            kp_orientation = torch.tensor([4.0, 4.0, 4.0], dtype=x.dtype, device=x.device)
            kd_orientation = torch.tensor([3.0, 3.0, 3.0], dtype=x.dtype, device=x.device)

            desired_torque = kp_orientation * orientation_error_vector - kd_orientation * angular_velocity

        linear_velocity = x[:, self.V]
        angular_velocity = x[:, self.OMEGA]

        linear_velocity_penalty_weight = 0.1
        angular_velocity_penalty_weight = 0.05

        linear_velocity_penalty = linear_velocity_penalty_weight * torch.mean(linear_velocity ** 2)
        angular_velocity_penalty = angular_velocity_penalty_weight * torch.mean(angular_velocity ** 2)

        desired_force -= linear_velocity_penalty * linear_velocity
        desired_torque -= angular_velocity_penalty * angular_velocity

        damping_force = -0.5 * linear_velocity
        damping_torque = -0.3 * angular_velocity

        desired_force += damping_force
        desired_torque += damping_torque

        current_speed = torch.norm(velocity, dim=1, keepdim=True)
        max_allowed_speed = torch.min(max_speed * speed_gain, torch.tensor(0.1, device=x.device))
        speed_scale = torch.clamp(max_allowed_speed / (current_speed + 1e-6), max=1.0)
        desired_force = desired_force * speed_scale

        u = torch.cat((desired_force, desired_torque), dim=1)

        upper_u_lim, lower_u_lim = self.control_limits
        for dim_idx in range(self.n_controls):
            u[:, dim_idx] = torch.clamp(
                u[:, dim_idx],
                min=lower_u_lim[dim_idx].item(),
                max=upper_u_lim[dim_idx].item(),
            )

        return u

    def is_path_clear(self, start: torch.Tensor, end: torch.Tensor) -> bool:
        """
        Check if there's a clear path between start and end positions.

        Args:
            start (torch.Tensor): Start position (3D)
            end (torch.Tensor): End position (3D)

        Returns:
            bool: True if the path is clear, False otherwise
        """
        direction = end - start
        distance = torch.norm(direction)
        unit_direction = direction / distance

        step_size = 0.1  # Adjust this value based on your needs
        num_steps = int(distance / step_size)

        for i in range(num_steps + 1):
            point = start + i * step_size * unit_direction
            for obstacle_pos, obstacle_radius in self.obstacles:
                if torch.norm(point - obstacle_pos) <= obstacle_radius + self.obstacle_buffer:
                    return False
        return True